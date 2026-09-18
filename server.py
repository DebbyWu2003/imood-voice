# ============================================================
# imood.ai 後端 — 用 llama.cpp 跑 Qwen2.5-1.5B-Instruct
#
# 用法:
#   1. pip install -r requirements.txt
#   2. 到 Hugging Face 下載 GGUF 量化版模型:
#        搜尋 "Qwen2.5-1.5B-Instruct-GGUF"，抓 q4_k_m 這個等級
#        放到 ./models/qwen2.5-1.5b-instruct-q4_k_m.gguf
#      (若想先用最輕量版本驗證串接，也可以先抓 0.5B 版本，
#       改下面 MODEL_PATH 即可，介面完全不用動)
#   3. uvicorn server:app --host 0.0.0.0 --port 8000 --reload
#
# 之後拿到 RTX 4090，若想換更大的模型(如 MiniCPM-2B、ChatGLM3-6B)，
# 只要換 MODEL_PATH，或改用 transformers 版本的載入方式即可，
# /api/chat 這個介面不需要變。
#
# GPU：載入時會自動偵測（見 _resolve_n_gpu_layers()）——llama-cpp-python 若是
# CUDA build 就整包 offload 到 GPU，否則純 CPU。2026-09-09 起主 venv 裝的是
# cu124 prebuilt wheel（abetlen 的 whl/cu124 index）+ nvidia-cuda-runtime-cu12，
# 所以在這台 4090 上預設就會上 GPU。LLM_DEVICE=cpu 可強制關掉。
# ============================================================

from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import json
import os
import time
from pathlib import Path

import base64

# voice_asr 在 import 時就把 pip 版 nvidia-*-cu12 的 bin 目錄加進 DLL 搜尋路徑
# （見 voice_asr._register_nvidia_dll_dirs）。llama-cpp-python 的 CUDA build 同樣
# 要靠這些 DLL（cudart64_12 / cublas64_12），Windows 上又不會自己找 pip 裝的版本，
# 所以一定要在 import llama_cpp 之前先 import voice_asr。
from voice_asr import EndpointConfig, Endpointer, Transcriber
from llama_cpp import Llama
from tts_client import stream_tts
import pipeline_config
from joygen_client import JoyGenClient

MODEL_PATH = "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
N_CTX = 2048
N_THREADS = 8  # 依機器 CPU 核心數調整

# voice-only 輸入路徑：麥克風送 16kHz / 單聲道 / 16-bit PCM 進來，後端做 VAD
# 斷句 + ASR（見 docs/asr-42-vad-plan.md）。
# 註：JoyGen/audio2motion 吃的是「LLM 回覆的 TTS 音訊」不是這條使用者輸入
# （品靜 2026-08-29 確認），所以這條 PCM 只給 ASR 用。
PCM_SAMPLE_RATE = 16000
PCM_SAMPLE_WIDTH_BYTES = 2  # 16-bit

ASR_MODEL_SIZE = "small"  # 選型見 docs/asr-41-results.md

# transcript 之後暫緩丟 LLM，再等這麼久的靜音看使用者有沒有續句（念頭中間
# 停頓 > Endpointer 的 end_silence_ms 會被切成兩段，這裡把兩段 transcript
# 併起來一起丟 LLM）。0 = 關閉（辨識完立刻回覆）。見 docs/asr-42-vad-plan.md。
PIPELINE_CFG = pipeline_config.load()
ENDPOINT_KWARGS = pipeline_config.endpoint_kwargs(PIPELINE_CFG)

VOICE_COALESCE_MS = PIPELINE_CFG["vad"].get("coalesce_ms", 1000)
VOICE_COALESCE_MAX_SEGMENTS = 6  # 安全上限：最多併這麼多段就強制送出

# JoyGen input streaming：TTS 的每一塊 PCM 同時送給 JoyGen 生成嘴型畫面。
# 連不上就只是沒有畫面，語音與文字流程不受影響（見 joygen_client.py）。
_JOYGEN_CFG = PIPELINE_CFG["joygen_input"]
JOYGEN_ENABLED = bool(_JOYGEN_CFG.get("enabled", True))
JOYGEN_HOST = _JOYGEN_CFG.get("ingest_host", "127.0.0.1")
JOYGEN_PORT = int(_JOYGEN_CFG.get("ingest_port", 8100))
# JoyGen 送出 END 之後要把整句畫完才回報，所以這個逾時要蓋過「一句話的生成
# 時間」而不是網路往返。pose-driven 在慢的 GPU 上一句可能要好幾十秒。
JOYGEN_STATUS_TIMEOUT = float(_JOYGEN_CFG.get("ingest_status_timeout_s", 300))

# JoyGen 每句話輸出一支 mp4（joygen_output.mode = utterance_file），瀏覽器
# 播不了 RTP，所以由這裡把那個目錄靜態送出去。utterance_dir 是 JoyGen 那端
# 看到的路徑（相對於 notes 根目錄），clip_dir() 換算成這台機器走得到的路徑。
JOYGEN_CLIP_DIR = pipeline_config.clip_dir(PIPELINE_CFG)
JOYGEN_CLIP_ROUTE = "/avatar-clips"
# 0.0.0.0 是監聽位址，不能拿來連線
if JOYGEN_HOST in ("0.0.0.0", "::"):
    JOYGEN_HOST = "127.0.0.1"

DEFAULT_VOICE = PIPELINE_CFG["tts"].get("voice", "female")
VALID_VOICES = ("female", "male")

# 回覆語音（TTS）——獨立 process（tts_service.py，跑在另一個 conda env），
# 見 docs/tts-prototype-notes.md。這裡連不上就優雅降級成純文字，不影響
# 既有的 /ws/audio 文字回覆流程。
TTS_ENABLED = True
# 用 127.0.0.1 而非 localhost：WSL 的 port relay 只聽 IPv4，localhost 會先試
# ::1、等它 timeout 才 fallback，每次新連線平白多 ~2s（tts_client 每次呼叫
# 都開新的 AsyncClient，所以每句話都付一次，而且已頂到 connect=2.0 的上限）。
TTS_SERVICE_URL = PIPELINE_CFG["tts"].get(
    "endpoint", "http://127.0.0.1:8001/synthesize").rsplit("/synthesize", 1)[0]

SYSTEM_PROMPT = (
    "你是 imood，一個溫暖、有同理心的陪伴型虛擬人。"
    "請一律使用繁體中文回覆，語氣自然、簡短，避免長篇說教。"
)

app = FastAPI(title="imood.ai chat backend")

# 前端頁面由這個服務一起送出，不要用 file:// 直接開。
# demo-imood-dashboard.html 的「播放範例語音」是 fetch('demo-assets/sample-zh.wav')，
# 而瀏覽器會擋 file:// 來源的 fetch（file:// 是 opaque origin），
# 按下去只會得到「範例語音載入失敗」。從 http://localhost:8000/ 開就同源了。
REPO_DIR = Path(__file__).resolve().parent
if (REPO_DIR / "demo-assets").is_dir():
    app.mount("/demo-assets",
              StaticFiles(directory=str(REPO_DIR / "demo-assets")),
              name="demo-assets")


if JOYGEN_CLIP_DIR:
    # 先建再掛：JoyGen 是第一句話進來時才建這個目錄的，而這邊通常比它早啟動。
    # 之前用 isdir() 判斷的版本會在目錄還不存在時靜靜地跳過掛載，之後每支
    # 影片都 404，而且完全沒有錯誤訊息。
    try:
        os.makedirs(JOYGEN_CLIP_DIR, exist_ok=True)
        app.mount(JOYGEN_CLIP_ROUTE,
                  StaticFiles(directory=JOYGEN_CLIP_DIR), name="avatar-clips")
        print(f"[joygen] avatar 影片目錄: {JOYGEN_CLIP_DIR}", flush=True)
    except OSError as exc:
        print(f"[joygen] 影片目錄掛不上（{exc}），前端只會看到靜態照片",
              flush=True)


@app.get("/")
def dashboard():
    # 開發期不要快取：改完 HTML/JS 之後重新整理就該拿到新的，不然會對著
    # 舊的程式碼除錯。
    return FileResponse(str(REPO_DIR / "demo-imood-dashboard.html"),
                        headers={"Cache-Control": "no-store"})

# 開發階段先全開，正式上線後應該改成白名單網域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm: Optional[Llama] = None
transcriber: Optional[Transcriber] = None
# faster-whisper / ctranslate2 與 llama.cpp 對同一個 model 併發呼叫都不保證
# 安全，各用一把 lock 序列化（主要是保護 /ws/audio 這條路）
_asr_lock = asyncio.Lock()
_llm_lock = asyncio.Lock()


def _resolve_n_gpu_layers() -> int:
    """決定 llama.cpp 要 offload 幾層到 GPU，跟 voice_asr.Transcriber._resolve_device()
    同一套思路：預設 "auto" —— 這顆 llama-cpp-python wheel 若編了 GPU backend
    （llama_supports_gpu_offload() 為真，通常是 CUDA build）就整包 offload（-1），
    否則純 CPU（0）。可用環境變數強制：LLM_DEVICE=cpu 關掉、LLM_DEVICE=cuda 強制開，
    或 LLM_N_GPU_LAYERS 直接指定層數（-1 = 全部）。
    註：CPU 版的 llama-cpp-python 吃不到 GPU，要先換成 CUDA build。"""
    override = os.environ.get("LLM_N_GPU_LAYERS")
    if override is not None:
        try:
            return int(override)
        except ValueError:
            print(f"[llm] LLM_N_GPU_LAYERS={override!r} 不是整數，忽略", flush=True)

    device = os.environ.get("LLM_DEVICE", "auto").lower()
    if device == "cpu":
        return 0
    if device == "cuda":
        return -1
    try:
        from llama_cpp import llama_supports_gpu_offload

        return -1 if llama_supports_gpu_offload() else 0
    except Exception as exc:  # noqa: BLE001 — 偵測失敗一律當沒有 GPU
        print(f"[llm] GPU 偵測失敗，改用 CPU: {exc}", flush=True)
        return 0


@app.on_event("startup")
def load_model():
    global llm, transcriber
    n_gpu_layers = _resolve_n_gpu_layers()
    t0 = time.perf_counter()
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 — GPU 載入失敗（缺 CUDA runtime 等）→ 退回 CPU
        if n_gpu_layers != 0:
            print(f"[llm] GPU 載入失敗，退回 CPU: {exc}", flush=True)
            n_gpu_layers = 0
            llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=N_CTX,
                n_threads=N_THREADS,
                n_gpu_layers=0,
                verbose=False,
            )
        else:
            raise
    where = f"GPU (n_gpu_layers={n_gpu_layers})" if n_gpu_layers != 0 else "CPU"
    print(f"[llm] Qwen2.5-1.5B on {where} loaded in {time.perf_counter() - t0:.1f}s",
          flush=True)
    transcriber = Transcriber(model_size=ASR_MODEL_SIZE)
    transcriber.load()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    latency_ms: int


# ------------------------------------------------------------
# LLM helper —— /api/chat、/api/chat/stream、/ws/audio 共用同一套
# prompt 組裝與生成參數，避免三個地方各寫一份。
# ------------------------------------------------------------

LLM_MAX_TOKENS = 200
LLM_TEMPERATURE = 0.7


def _build_messages(user_text: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def llm_complete(user_text: str) -> str:
    """非 streaming：回完整回覆文字。"""
    result = llm.create_chat_completion(
        messages=_build_messages(user_text),
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
    )
    return result["choices"][0]["message"]["content"].strip()


def llm_stream(user_text: str):
    """streaming：逐段 yield 回覆文字 delta（同步 generator）。"""
    stream = llm.create_chat_completion(
        messages=_build_messages(user_text),
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        stream=True,
    )
    for chunk in stream:
        piece = chunk["choices"][0].get("delta", {}).get("content")
        if piece:
            yield piece


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="模型尚未載入完成")

    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="訊息不可為空")

    start = time.time()
    reply = llm_complete(text)
    latency_ms = int((time.time() - start) * 1000)
    return ChatResponse(reply=reply, latency_ms=latency_ms)


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """
    SSE streaming 版本的 /api/chat。前端不能用 EventSource（那個只能發 GET），
    改用 fetch + ReadableStream 自己解析 "data: {...}\n\n" 這種格式。

    wire 格式（跟 4.3 前就一樣，前端不用改）：
      data: {"delta": "..."}          逐字
      data: {"done": true, "latency_ms": N}
    """
    if llm is None:
        raise HTTPException(status_code=503, detail="模型尚未載入完成")

    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="訊息不可為空")

    def event_generator():
        start = time.time()
        for piece in llm_stream(text):
            yield f"data: {json.dumps({'delta': piece}, ensure_ascii=False)}\n\n"
        latency_ms = int((time.time() - start) * 1000)
        yield f"data: {json.dumps({'done': True, 'latency_ms': latency_ms}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def _stream_reply_to_ws(websocket: WebSocket, loop, user_text: str,
                              voice: str = "female",
                              joygen: Optional[JoyGenClient] = None,
                              session: str = "ws") -> None:
    """
    把同步的 llm_stream() generator 橋接成 async，逐段送 reply_delta，
    最後送 reply_done。生成在 thread pool 跑，透過 queue 把 delta 丟回
    event loop。
    """
    queue: asyncio.Queue = asyncio.Queue()
    DONE = object()

    def produce():
        try:
            for piece in llm_stream(user_text):
                loop.call_soon_threadsafe(queue.put_nowait, piece)
        except Exception as exc:  # noqa: BLE001 — 丟回主 coroutine 統一處理
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, DONE)

    start = time.time()
    fut = loop.run_in_executor(None, produce)
    full_text_parts: list = []
    failed = False
    try:
        while True:
            item = await queue.get()
            if item is DONE:
                break
            if isinstance(item, Exception):
                await websocket.send_json({"type": "error", "error": f"LLM 生成失敗: {item}"})
                failed = True
                break
            full_text_parts.append(item)
            await websocket.send_json({"type": "reply_delta", "delta": item})
    finally:
        await fut
    await websocket.send_json({
        "type": "reply_done",
        "latency_ms": int((time.time() - start) * 1000),
    })

    if not failed and TTS_ENABLED:
        await _stream_tts_to_ws(websocket, "".join(full_text_parts),
                                voice=voice, joygen=joygen, session=session)


async def _stream_tts_to_ws(websocket: WebSocket, text: str,
                            voice: str = "female",
                            joygen: Optional[JoyGenClient] = None,
                            session: str = "ws") -> None:
    """
    LLM 回覆全部生成完之後才合成語音（先求簡單能動，見
    docs/tts-prototype-notes.md 的取捨說明；之後要更即時可以改成逐句合成）。
    tts_service 連不上或出錯都不當作致命錯誤——文字回覆已經送完了，這裡
    失敗只送一個 error 訊息，不能讓 /ws/audio 的主迴圈掛掉。

    同一份 PCM 分兩路：base64 送瀏覽器播放，raw bytes 推 JoyGen 生成嘴型。
    兩邊拿到同一份音訊，畫面與聲音才會對得上；JoyGen 那路失敗只是沒有畫面。

    注意 joygen.end() 會等 JoyGen 把整句畫完才回，所以它排在 audio_done
    之後——不能讓畫面的生成時間卡住瀏覽器的播放。
    """
    text = text.strip()
    if not text:
        return

    utterance = f"u{int(time.time() * 1000)}"
    pushing = False
    if joygen is not None:
        pushing = await joygen.begin(session, utterance, voice=voice,
                                     emotion=None,  # 之後接 BERT 填這裡
                                     text=text)

    seq = 0
    try:
        async for chunk in stream_tts(text, TTS_SERVICE_URL, voice=voice):
            seq += 1
            await websocket.send_json({
                "type": "audio_delta",
                "audio": base64.b64encode(chunk).decode("ascii"),
                "sample_rate": PCM_SAMPLE_RATE,
                "seq": seq,
            })
            if pushing:
                await joygen.audio(chunk)
        await websocket.send_json({"type": "audio_done"})
    except Exception as exc:  # noqa: BLE001 — TTS 失敗不影響已完成的文字回覆
        print(f"[tts] 合成失敗，改為純文字回覆: {exc}", flush=True)
        await websocket.send_json({"type": "error", "error": "語音合成暫時無法使用"})
    finally:
        if pushing:
            # 不要在這裡 await JoyGen —— 它要畫幾十秒，而這段期間 handler
            # 沒有回到 receive_bytes()，uvicorn 會停止讀取這條 WebSocket
            # （流量控制），連 ping/pong 都不處理，最後整條連線被 keepalive
            # 判定斷線。丟到背景，畫好了再把 avatar_video 送出去。
            async def _on_joygen_done(result):
                print(f"[joygen] {utterance} voice={voice} "
                      f"frames={result.get('frames')} "
                      f"first_frame={result.get('ingest_to_first_frame_ms')}ms "
                      f"video={result.get('video')}", flush=True)
                clip = result.get("video")
                if not clip:
                    return
                try:
                    await websocket.send_json({
                        "type": "avatar_video",
                        "url": f"{JOYGEN_CLIP_ROUTE}/{clip}",
                        "avatar": result.get("avatar"),
                        "frames": result.get("frames"),
                        "generate_ms": result.get("utterance_ms"),
                    })
                except Exception:  # noqa: BLE001 — 連線可能已經關了
                    pass

            await joygen.end_in_background(_on_joygen_done)


@app.websocket("/ws/audio")
async def audio_stream(websocket: WebSocket):
    """
    Voice-only 輸入路徑的接收端（見 docs/asr-42-vad-plan.md）。

    前端麥克風經 AudioWorklet 即時 resample 成 16kHz / mono / 16-bit PCM，
    每個 chunk（預設 320ms）以 binary frame 送過來。後端回傳的 JSON 訊息
    都有 "type" 欄位：

      {"type":"ack",       "ack":N, "chunk_ms":.., "total_bytes":..}   每個 chunk
      {"type":"asr_start", "audio_ms":..}                              偵測到句尾、開始辨識
      {"type":"asr_empty"}                                             有聲音但辨識不出內容
      {"type":"transcript","text":.., "audio_ms":.., "asr_latency_ms":..} 一段語音辨識完
      {"type":"reply_delta","delta":".."}                              LLM 回覆逐段
      {"type":"reply_done", "latency_ms":N}                            LLM 回覆結束
      {"type":"error",     "error":".."}

    流程：chunk → Endpointer 累積 → 句尾靜音 → faster-whisper 辨識 → transcript。
    辨識後**不馬上**丟 LLM，先放進 pending、再等 VOICE_COALESCE_MS 的靜音——
    使用者若在念頭中間停頓（>end_silence_ms 會被切成兩段），這段等待會把後面
    的續句 transcript 併進來，一起丟 LLM，避免「一個念頭 → 回兩次」。
    真的停下來（沒有續句）才 flush pending → llm_stream() → reply_delta → reply_done。
    """
    await websocket.accept()

    # 男/女聲用 query 參數帶進來（/ws/audio?voice=male）。主迴圈是純 binary
    # 的 receive_bytes()，插 text frame 進來會讓它拋例外，所以不走訊息協定。
    # 換聲音 = 前端重連一次，一個連線一種聲音。
    voice = (websocket.query_params.get("voice") or DEFAULT_VOICE).lower()
    if voice not in VALID_VOICES:
        voice = DEFAULT_VOICE
    session = websocket.query_params.get("session") or f"ws{int(time.time())}"

    # VAD 參數來自共用的 pipeline.yaml（斷句秒數等在那裡調，不要改這裡）
    endpointer = Endpointer(EndpointConfig(**ENDPOINT_KWARGS))
    joygen = JoyGenClient(JOYGEN_HOST, JOYGEN_PORT, enabled=JOYGEN_ENABLED,
                          status_timeout=JOYGEN_STATUS_TIMEOUT)

    loop = asyncio.get_running_loop()
    chunk_count = 0
    byte_count = 0
    start = time.time()
    print(f"[voice] 連線 session={session} voice={voice} "
          f"end_silence_ms={ENDPOINT_KWARGS.get('end_silence_ms')} "
          f"joygen={'on' if JOYGEN_ENABLED else 'off'}", flush=True)

    pending: list = []  # 已辨識、還在等可能續句、還沒丟 LLM 的 transcript 片段

    async def flush_pending() -> None:
        nonlocal pending
        if not pending:
            return
        text = "".join(pending).strip()
        pending = []
        if not text:
            return
        if llm is None:
            await websocket.send_json({"type": "error", "error": "LLM 模型尚未載入完成"})
            return
        print(f"[voice] -> LLM ({len(text)} 字): {text!r}", flush=True)
        async with _llm_lock:
            await _stream_reply_to_ws(websocket, loop, text, voice=voice,
                                      joygen=joygen, session=session)

    try:
        while True:
            # pending 時用短 timeout 收，好讓「續句等待窗到期」能定期檢查
            if pending:
                try:
                    data = await asyncio.wait_for(websocket.receive_bytes(), timeout=0.25)
                except asyncio.TimeoutError:
                    data = None
            else:
                data = await websocket.receive_bytes()

            if data is not None:
                if len(data) % PCM_SAMPLE_WIDTH_BYTES != 0:
                    await websocket.send_json({
                        "type": "error",
                        "error": f"chunk 長度 {len(data)} bytes 不是 16-bit PCM 的整數倍",
                    })
                else:
                    chunk_count += 1
                    byte_count += len(data)
                    n_samples = len(data) // PCM_SAMPLE_WIDTH_BYTES
                    chunk_ms = n_samples / PCM_SAMPLE_RATE * 1000
                    await websocket.send_json({
                        "type": "ack",
                        "ack": chunk_count,
                        "chunk_ms": round(chunk_ms, 1),
                        "total_bytes": byte_count,
                    })

                    utterance = endpointer.feed(data)
                    if utterance is not None:
                        if transcriber is None or not transcriber.ready:
                            await websocket.send_json({"type": "error", "error": "ASR 模型尚未載入完成"})
                        else:
                            await websocket.send_json(
                                {"type": "asr_start", "audio_ms": round(utterance.duration_ms)}
                            )
                            async with _asr_lock:
                                result = await loop.run_in_executor(
                                    None, transcriber.transcribe, utterance.pcm
                                )
                            print(
                                f"[voice] utterance {utterance.duration_ms:.0f}ms "
                                f"(voiced {utterance.voiced_ms:.0f}ms) -> ASR {result.latency_ms}ms "
                                f"RTF {result.latency_ms / max(result.audio_ms, 1):.2f}x: {result.text!r}",
                                flush=True,
                            )
                            if not result.text:
                                await websocket.send_json({"type": "asr_empty"})
                            else:
                                await websocket.send_json({
                                    "type": "transcript",
                                    "text": result.text,
                                    "audio_ms": round(utterance.duration_ms),
                                    "asr_latency_ms": result.latency_ms,
                                })
                                if VOICE_COALESCE_MS <= 0:
                                    if llm is None:
                                        await websocket.send_json(
                                            {"type": "error", "error": "LLM 模型尚未載入完成"}
                                        )
                                    else:
                                        async with _llm_lock:
                                            await _stream_reply_to_ws(websocket, loop, result.text)
                                else:
                                    pending.append(result.text)
                                    if len(pending) >= VOICE_COALESCE_MAX_SEGMENTS:
                                        await flush_pending()

            # 續句等待窗：上一段之後累積了夠久的靜音、且現在沒在講話 → 併起來丟 LLM
            if (pending
                    and not endpointer.triggered
                    and endpointer.silence_since_last_ms >= VOICE_COALESCE_MS):
                await flush_pending()
    except WebSocketDisconnect:
        elapsed = time.time() - start
        print(
            f"[voice] client disconnected: {chunk_count} chunks, "
            f"{byte_count} bytes, {elapsed:.1f}s",
            flush=True,
        )
    finally:
        # 一個連線一條 JoyGen 通道，斷線就收掉，不要留著佔住服務端的 session
        await joygen.close()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": llm is not None,
        "voice_default": DEFAULT_VOICE,
        "vad_end_silence_ms": ENDPOINT_KWARGS.get("end_silence_ms"),
        "coalesce_ms": VOICE_COALESCE_MS,
        "joygen": {"enabled": JOYGEN_ENABLED,
                   "host": JOYGEN_HOST, "port": JOYGEN_PORT,
                   "status_timeout_s": JOYGEN_STATUS_TIMEOUT},
    }
