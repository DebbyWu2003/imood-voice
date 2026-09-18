# ============================================================
# make_male_prompt.py —— 產生 CosyVoice2 zero-shot 要用的男聲參考音檔。
#
# 前端要讓使用者選一男一女，但正式後端 CosyVoice2 走 zero-shot，需要一段
# 參考錄音；我們手上沒有男聲素材。
#
# 這支做的事：用**舊後端 CosyVoice-300M-SFT 內建的「中文男」**合成一段話，
# 存成 16kHz wav，之後拿它當 CosyVoice2 的 zero-shot prompt。等於用舊模型的
# 音色去 bootstrap 新模型，不必等人錄音。
#
# 音色會是「CosyVoice2 模仿 CosyVoice1 的男聲」，不如真人自然。拿到真人素材
# 之後直接換掉 TTS_PROMPT_WAV_MALE 指到的檔案即可，程式不用動。
#
# 用法（tts_service 要先用 300M-SFT 後端起來）：
#   COSYVOICE_MODEL_DIR=/root/CosyVoice/pretrained_models/CosyVoice-300M-SFT \
#     python -m uvicorn tts_service:app --port 8001
#   python scripts/make_male_prompt.py --out assets/male_prompt.wav
#
# 輸出的逐字稿要原樣填進 TTS_PROMPT_TEXT_MALE：zero-shot 靠「音檔 + 對應文字」
# 抽特徵，文字對不上音檔的話抽出來的音色會偏。
# ============================================================

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import httpx
import numpy as np

REPO = Path(__file__).resolve().parent.parent

# 簡體：tts_service 合成前會把文字經 OpenCC 轉簡體，所以直接用簡體寫，
# 逐字稿才會跟音檔實際念出來的內容一致。
DEFAULT_TEXT = (
    "大家好，我是 imood 的语音助理。"
    "今天天气不错，希望你有个愉快的一天，"
    "有什么需要帮忙的地方都可以告诉我。"
)
SAMPLE_RATE = 16000


def synth(url: str, text: str, voice: str) -> bytes:
    pcm = bytearray()
    with httpx.Client(timeout=httpx.Timeout(connect=5.0, read=180.0,
                                            write=10.0, pool=10.0)) as client:
        with client.stream("POST", f"{url}/synthesize",
                           json={"text": text, "voice": voice}) as r:
            r.raise_for_status()
            for chunk in r.iter_bytes():
                pcm.extend(chunk)
    return bytes(pcm)


def write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


def median_f0(pcm: bytes) -> float:
    """有聲段的基頻中位數。

    沒辦法用聽的確認「這真的是男聲」，所以量它：成年男聲大致 85-180Hz、
    女聲 165-255Hz。用自相關估，夠分辨兩者就好。
    """
    x = np.frombuffer(pcm, np.int16).astype(np.float64) / 32768.0
    win, hop = 1024, 256
    f0s = []
    for i in range(0, len(x) - win, hop):
        frame = x[i:i + win]
        if np.sqrt(np.mean(frame ** 2)) < 0.02:      # 靜音略過
            continue
        frame = frame - frame.mean()
        corr = np.correlate(frame, frame, mode="full")[win - 1:]
        # 只看 70-400Hz 對應的 lag，避開倍頻與直流
        lo, hi = SAMPLE_RATE // 400, SAMPLE_RATE // 70
        if hi >= len(corr):
            continue
        lag = int(np.argmax(corr[lo:hi])) + lo
        if corr[lag] > 0.3 * corr[0]:
            f0s.append(SAMPLE_RATE / lag)
    return float(np.median(f0s)) if f0s else float("nan")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="bootstrap a male zero-shot prompt")
    p.add_argument("--url", default="http://127.0.0.1:8001")
    p.add_argument("--out", default=str(REPO / "assets" / "male_prompt.wav"))
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--also_female", default=None,
                   help="同時存一份女聲，用來對照基頻")
    args = p.parse_args(argv)

    male = synth(args.url, args.text, "male")
    if not male:
        sys.exit("[male-prompt] 服務回了空的音訊 —— 後端是 300M-SFT 嗎？")
    out = Path(args.out)
    write_wav(out, male)
    dur = len(male) / 2 / SAMPLE_RATE
    f0_male = median_f0(male)
    print(f"[male-prompt] {dur:.2f}s -> {out}")
    print(f"[male-prompt] 基頻中位數 {f0_male:.1f} Hz")

    if args.also_female:
        female = synth(args.url, args.text, "female")
        write_wav(Path(args.also_female), female)
        f0_female = median_f0(female)
        print(f"[male-prompt] 對照女聲 {f0_female:.1f} Hz -> {args.also_female}")
        if not (f0_male < f0_female):
            print("[male-prompt] 警告：男聲基頻沒有低於女聲，voice 參數可能沒生效")

    print("")
    print("接著把這兩行放進 tts_service 的環境變數（逐字稿要原樣照抄）：")
    print(f'  TTS_PROMPT_WAV_MALE={out}')
    print(f'  TTS_PROMPT_TEXT_MALE="{args.text}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
