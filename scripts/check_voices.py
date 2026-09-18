# ============================================================
# check_voices.py —— 部署後確認 TTS 的男女聲真的不一樣。
#
# /health 只會說「有註冊 male」，不會說它聽起來是不是男的：zero-shot 參考音檔
# 放錯、逐字稿對不上、voice 參數沒傳到，都會讓兩個聲音其實一模一樣，而且完全
# 不會報錯。這支直接量基頻來判斷（成年男聲約 85-180Hz、女聲約 165-255Hz）。
#
# 順便量首塊延遲 —— 那是使用者講完話到聽見回覆的其中一段。
#
# 用法：
#   python scripts/check_voices.py [--url http://127.0.0.1:8001]
# ============================================================

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_male_prompt import SAMPLE_RATE, median_f0  # noqa: E402

TEXT = "今天的天气看起来很不错，要不要出去走走？"


def synth_timed(url: str, text: str, voice: str):
    pcm = bytearray()
    first_ms = None
    t0 = time.perf_counter()
    with httpx.Client(timeout=httpx.Timeout(connect=5.0, read=180.0,
                                            write=10.0, pool=10.0)) as client:
        with client.stream("POST", f"{url}/synthesize",
                           json={"text": text, "voice": voice}) as r:
            r.raise_for_status()
            for chunk in r.iter_bytes():
                if chunk and first_ms is None:
                    first_ms = (time.perf_counter() - t0) * 1000.0
                pcm.extend(chunk)
    total_ms = (time.perf_counter() - t0) * 1000.0
    return bytes(pcm), first_ms, total_ms


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="verify the two TTS voices differ")
    p.add_argument("--url", default="http://127.0.0.1:8001")
    p.add_argument("--text", default=TEXT)
    p.add_argument("--save_dir", default=None)
    args = p.parse_args(argv)

    with httpx.Client(timeout=10.0) as c:
        health = c.get(f"{args.url}/health").json()
    print("[check] backend = {}  voices = {}".format(
        health.get("backend"), health.get("voices")))

    rows = {}
    for voice in ("female", "male"):
        pcm, first_ms, total_ms = synth_timed(args.url, args.text, voice)
        if not pcm:
            sys.exit("[check] {} 回了空音訊".format(voice))
        dur = len(pcm) / 2 / SAMPLE_RATE
        f0 = median_f0(pcm)
        rows[voice] = f0
        rtf = (total_ms / 1000.0) / dur if dur else float("nan")
        print("  {:<7} 基頻 {:6.1f} Hz   {:.2f}s 音訊   首塊 {:6.1f}ms   RTF {:.2f}x"
              .format(voice, f0, dur, first_ms or 0, rtf))
        if args.save_dir:
            import wave
            out = Path(args.save_dir) / "voice_{}.wav".format(voice)
            out.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(out), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(pcm)
            print("           -> {}".format(out))

    gap = rows["female"] - rows["male"]
    print("")
    if not np.isfinite(gap):
        print("[check] 失敗：基頻算不出來")
        return 1
    if gap < 30:
        print("[check] 失敗：兩個聲音的基頻只差 {:.1f} Hz，男聲很可能沒生效"
              "（參考音檔沒掛上？voice 參數沒傳到？）".format(gap))
        return 1
    print("[check] 通過：女聲比男聲高 {:.1f} Hz，兩個聲音確實不同".format(gap))
    return 0


if __name__ == "__main__":
    sys.exit(main())
