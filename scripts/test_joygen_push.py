# ============================================================
# test_joygen_push.py —— 驗證 joygen_client.py 跟 JoyGen 服務對得上。
#
# 協定有兩份實作（這邊 Windows / Python 3.12，JoyGen 那邊 WSL / Python 3.8），
# 所以要有一個測試真的把兩份接起來跑，不能只各自單測。
#
# 先在 WSL 起服務：
#   conda activate joygen
#   bash ~/imood_project/joygen-deployment-notes/scripts/run_input.sh serve \
#       demo/example_5s.mp4 results/smoke_test/edit_exp xinwen_5s \
#       ~/imood_project/joygen-deployment-notes/results/push_test.mp4 --no_idle
#
# 再跑這支：
#   venv\Scripts\python.exe scripts\test_joygen_push.py
#
# 不需要 TTS：這裡用合成的正弦波當 PCM，切成 320ms 一塊，跟 tts_service 的
# 輸出格式一模一樣（docs/tts-streaming-spec.md）。
# ============================================================

from __future__ import annotations

import argparse
import asyncio
import math
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from joygen_client import JoyGenClient, SAMPLE_RATE  # noqa: E402

CHUNK_MS = 320
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2


def tone_pcm(duration_ms: int) -> bytes:
    n = int(SAMPLE_RATE * duration_ms / 1000)
    return struct.pack(
        "<%dh" % n,
        *[int(8000 * math.sin(i * 0.05)) for i in range(n)],
    )


async def run(host: str, port: int, seconds: float, utterances: int) -> int:
    client = JoyGenClient(host, port)
    failures = []

    def check(name, cond, detail=""):
        if cond:
            print("  PASS  %s" % name)
        else:
            print("  FAIL  %s  %s" % (name, detail))
            failures.append(name)

    try:
        for i in range(utterances):
            voice = "male" if i % 2 else "female"
            pcm = tone_pcm(int(seconds * 1000))
            tag = "t%d" % (i + 1)

            t0 = time.perf_counter()
            ok = await client.begin("pytest", tag, voice=voice,
                                    emotion="joy", text="測試")
            check("%s begin 被接受" % tag, ok)
            if not ok:
                print("  （JoyGen 服務沒開？先照檔頭的指令啟動）")
                return 1

            for start in range(0, len(pcm), CHUNK_BYTES):
                await client.audio(pcm[start:start + CHUNK_BYTES])
            result = await client.end()
            wall = (time.perf_counter() - t0) * 1000.0

            check("%s 有回結果" % tag, result is not None, result)
            if not result:
                continue
            expected = int(seconds * 25) + 2
            check("%s 畫面張數合理 (%s ~ %d)" % (tag, result.get("frames"), expected),
                  abs(result.get("frames", 0) - expected) <= 2, result.get("frames"))
            check("%s 音訊沒被丟" % tag,
                  result.get("ring", {}).get("dropped_ms") == 0.0,
                  result.get("ring"))
            print("        voice=%s frames=%s first_frame=%sms wall=%.0fms"
                  % (voice, result.get("frames"),
                     result.get("ingest_to_first_frame_ms"), wall))
    finally:
        await client.close()

    print("")
    if failures:
        print("FAILED: %s" % ", ".join(failures))
        return 1
    print("joygen_client 與 JoyGen 服務協定一致")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--utterances", type=int, default=2)
    args = p.parse_args()
    return asyncio.run(run(args.host, args.port, args.seconds, args.utterances))


if __name__ == "__main__":
    sys.exit(main())
