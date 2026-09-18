# ============================================================
# test_full_chain.py —— 麥克風以外的完整鏈路測試。
#
#   語音檔 → /ws/audio → VAD 斷句 → faster-whisper → Qwen → CosyVoice2
#          → 瀏覽器音訊 + JoyGen ingest → avatar 畫面
#
# 走的路徑跟真人對麥克風講話**完全一樣**：前端也是把麥克風重取樣成 16kHz
# 單聲道 PCM、每 320ms 一個 binary frame 送進 /ws/audio。差別只在音訊來源是
# 檔案而不是麥克風，所以不需要人在場也能驗證整條鏈。
#
# 要先起三個服務：
#   1) tts_service    （wsl -d Ubuntu-24.04，port 8001）
#      bash scripts/start_tts_cosyvoice2.sh
#   2) JoyGen service （wsl -d Ubuntu-22.04，port 8100）
#      bash joygen-deployment-notes/scripts/run_input.sh serve ...
#   3) server.py      （Windows venv，port 8000）
#      venv\Scripts\python -m uvicorn server:app --port 8000
#
# 然後：
#   venv\Scripts\python scripts\test_full_chain.py <speech.wav|mp3> --voice male
#
# 印出來的是每一段的時間點，加起來就是使用者「講完 → 看到 avatar 開口」
# 的延遲拆解。
# ============================================================

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import wave
import time
from pathlib import Path

import websockets

SAMPLE_RATE = 16000
CHUNK_MS = 320


def decode_pcm16(path: str) -> bytes:
    """16kHz mono PCM16。.wav 直接用標準庫讀，其餘才叫 ffmpeg —— Windows 這側
    通常沒有 ffmpeg 在 PATH 上，而測試音檔轉成 wav 是很便宜的事。"""
    p = Path(path)
    if p.suffix.lower() == ".wav":
        with wave.open(str(p), "rb") as w:
            fmt = (w.getnchannels(), w.getsampwidth(), w.getframerate())
            if fmt != (1, 2, SAMPLE_RATE):
                sys.exit("[chain] {} 必須是 16kHz / 單聲道 / 16-bit，"
                         "目前是 {}".format(p, fmt))
            return w.readframes(w.getnframes())
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(p),
           "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-"]
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    except FileNotFoundError:
        sys.exit("[chain] 找不到 ffmpeg —— 改傳 16kHz 單聲道的 .wav")


async def run(args) -> int:
    pcm = decode_pcm16(args.audio)
    chunk = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2
    url = "{}?voice={}&session=fullchain".format(args.url, args.voice)

    marks = {}
    reply_parts = []
    audio_chunks = 0
    audio_bytes = 0
    t0 = None

    def mark(name):
        if name not in marks:
            marks[name] = (time.perf_counter() - t0) * 1000.0

    # 關掉 client 端的 keepalive：JoyGen 算一句要數十秒，期間沒有訊息往來，
    # 預設 20 秒的 ping timeout 會誤判成斷線。瀏覽器不會有這問題（它不主動
    # 送 ping），所以這只是測試腳本要處理的事。
    async with websockets.connect(url, max_size=None, ping_interval=None) as ws:
        print("[chain] connected {}".format(url))
        t0 = time.perf_counter()

        async def sender():
            # 開頭先送一段靜音。Endpointer 用連線後的前 calibration_ms 估環境
            # 底噪，真人用麥克風時那段本來就是安靜的；一連上就直接灌語音的話
            # 底噪會被估成「說話的音量」，之後真正的語音反而過不了能量門檻，
            # 結果就是永遠不斷句。
            lead = b"\x00" * chunk
            for _ in range(args.lead_chunks):
                await ws.send(lead)
                if args.pace == "realtime":
                    await asyncio.sleep(CHUNK_MS / 1000.0)

            for i in range(0, len(pcm), chunk):
                await ws.send(pcm[i:i + chunk])
                if args.pace == "realtime":
                    await asyncio.sleep(CHUNK_MS / 1000.0)
            mark("audio_sent")
            # 說完之後要繼續送靜音：VAD 靠「連續靜音」判斷句子結束，
            # 直接停止送資料的話它永遠等不到句尾。
            silence = b"\x00" * chunk
            for _ in range(args.tail_chunks):
                await ws.send(silence)
                if args.pace == "realtime":
                    await asyncio.sleep(CHUNK_MS / 1000.0)

        send_task = asyncio.create_task(sender())

        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=args.timeout)
                msg = json.loads(raw)
                kind = msg.get("type")

                if kind == "asr_start":
                    mark("asr_start")
                elif kind == "transcript":
                    mark("transcript")
                    print("  [ASR] {!r}  (音訊 {}ms, 辨識 {}ms)".format(
                        msg.get("text"), msg.get("audio_ms"),
                        msg.get("asr_latency_ms")))
                elif kind == "reply_delta":
                    mark("reply_first")
                    reply_parts.append(msg.get("delta", ""))
                elif kind == "reply_done":
                    mark("reply_done")
                    print("  [LLM] {!r}".format("".join(reply_parts)))
                elif kind == "audio_delta":
                    mark("tts_first")
                    audio_chunks += 1
                    audio_bytes += len(msg.get("audio", "")) * 3 // 4
                elif kind == "audio_done":
                    mark("tts_done")
                    print("  [TTS] {} 塊 / 約 {:.2f}s 音訊".format(
                        audio_chunks, audio_bytes / 2 / SAMPLE_RATE))
                    # 不在這裡收工：avatar_video 是 JoyGen 算完才送的，
                    # 那才是整條鏈的終點。
                elif kind == "avatar_video":
                    mark("avatar_video")
                    print("  [JoyGen] {} ({} 張, 生成 {:.0f}s)".format(
                        msg.get("url"), msg.get("frames"),
                        (msg.get("generate_ms") or 0) / 1000))
                    break
                elif kind == "error":
                    print("  [ERR] {}".format(msg.get("error")))
        except asyncio.TimeoutError:
            print("  [chain] 等超過 {}s 沒有下一個訊息".format(args.timeout))
        finally:
            send_task.cancel()

    print("")
    print("[ 時間軸（從開始送音訊算起）]")
    order = ["audio_sent", "asr_start", "transcript", "reply_first",
             "reply_done", "tts_first", "tts_done", "avatar_video"]
    label = {"audio_sent": "音訊送完", "asr_start": "偵測到句尾、開始辨識",
             "transcript": "辨識結果", "reply_first": "LLM 第一個字",
             "reply_done": "LLM 回覆完成", "tts_first": "第一塊語音",
             "tts_done": "語音送完", "avatar_video": "avatar 影片就緒"}
    prev = 0.0
    for k in order:
        if k in marks:
            print("  {:<22}{:>9.0f}ms   (+{:.0f}ms)".format(
                label[k], marks[k], marks[k] - prev))
            prev = marks[k]

    if "transcript" in marks and "tts_first" in marks:
        print("")
        print("  使用者講完 → 聽見第一聲：{:.0f}ms".format(
            marks["tts_first"] - marks.get("audio_sent", 0)))
        print("  （avatar 的第一張畫面再加上 JoyGen 的 ingest→首幀，"
              "見 JoyGen 服務的 log）")

    missing = [k for k in ("transcript", "reply_done", "tts_done",
                           "avatar_video") if k not in marks]
    if missing:
        print("\n[chain] 失敗：沒有走到 {}".format(", ".join(missing)))
        return 1
    print("\n[chain] 完整鏈路通過")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="full chain: audio file -> avatar")
    p.add_argument("audio")
    p.add_argument("--url", default="ws://localhost:8000/ws/audio")
    p.add_argument("--voice", default="female", choices=["female", "male"])
    p.add_argument("--pace", choices=["fast", "realtime"], default="realtime")
    p.add_argument("--lead_chunks", type=int, default=5,
                   help="開頭補幾塊靜音給 VAD 估底噪（320ms 一塊）")
    p.add_argument("--tail_chunks", type=int, default=12,
                   help="講完之後補幾塊靜音讓 VAD 判斷句尾（320ms 一塊）")
    p.add_argument("--timeout", type=float, default=180.0)
    args = p.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
