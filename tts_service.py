# ============================================================
# tts_service.py —— CosyVoice 中文語音合成服務（獨立 process）
#
# 這個檔案**不**跑在 imood-voice 主 venv（Python 3.14）裡，因為 CosyVoice
# 需要 PyTorch + pynini，跟主 venv 的 llama-cpp-python 環境不相容。
#
# 兩種後端，用 COSYVOICE_MODEL_DIR 指到哪個模型就跑哪個：
#
#   1) CosyVoice2-0.5B（預設，跑在 WSL2 的 cosyvoice_vllm conda env）
#      - LLM backbone 換成 Qwen2 → vLLM（load_vllm=True）吃得到，AR 語音
#        token decoder 大幅加速，這是壓 `reply_done -> audio_first` 那 ~4s 的
#        唯一有效路徑（見 docs/gpu-notes.md）。vLLM 沒有 Windows CUDA 版，
#        所以這條路一定在 WSL2 跑；server.py 仍在 Windows，打 localhost:8001。
#      - 沒有內建 SFT 語者（中文女/男），改用 zero-shot：啟動時餵一段 3~10s
#        參考音檔註冊成一個 speaker id，之後每個請求引用該 id（不重抽特徵）。
#      啟動：
#        COSYVOICE_REPO=/root/CosyVoice MODELSCOPE_OFFLINE=1 \
#        /root/miniconda3/envs/cosyvoice_vllm/bin/python -m uvicorn \
#        tts_service:app --host 0.0.0.0 --port 8001
#
#   2) CosyVoice-300M-SFT（舊路徑，Windows cosyvoice conda env）
#      - 內建「中文女」SFT 語者，inference_sft。vLLM 對 v1 無效。
#      啟動：
#        C:\imood_project\imood-voice\miniconda3\envs\cosyvoice\python.exe -m uvicorn \
#        tts_service:app --host 0.0.0.0 --port 8001
#
# server.py 透過 tts_client.py 呼叫 /synthesize，HTTP 通訊、兩個 process
# 互相獨立——這個服務掛掉不影響 /ws/audio 的文字回覆流程。
# ============================================================

import os
import sys
from typing import Iterator

import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# CosyVoice 是外部 checkout，不進這個 repo（跟 models/*.gguf 一樣太大）。
COSYVOICE_REPO = os.environ.get("COSYVOICE_REPO", r"C:\imood_project\imood-voice\CosyVoice")
sys.path.insert(0, COSYVOICE_REPO)
sys.path.insert(0, os.path.join(COSYVOICE_REPO, "third_party", "Matcha-TTS"))

# --- 讓 wetext 從本地快取載入，不要連 modelscope ------------------------------
# CosyVoice 的文字正規化（把「50%」「3:20」「23.5」變成口語中文）走 wetext。
# wetext.Normalizer 沒拿到明確的 tagger/verbalizer 路徑時，會呼叫 modelscope 的
# snapshot_download() 做版本檢查——即使 FST 快取早就在 ~/.cache/modelscope 底下。
# 匿名請求前幾次會過，很快就被 modelscope 限速擋成 403，而 CosyVoice 的
# cli/frontend.py 用裸 except: 把失敗吞掉，靜默降級成 text_frontend=''。結果是
# 正規化整個不做、只留一行 "no frontend is avaliable" 的 log，數字和時間就原樣
# 送進聲學模型念出來。
#
# modelscope 1.20.0 沒有任何環境變數能開 local_files_only（MODELSCOPE_OFFLINE
# 在它的原始碼裡根本不存在，設了也沒用），所以只能在這裡補上預設值。必須在
# CosyVoice 被 import 之前跑：wetext 是 `from modelscope import snapshot_download`。
#
# 先試本地、失敗才連網，所以全新環境（還沒有快取）仍然抓得到，不會倒退。
try:
    import modelscope as _modelscope

    _ms_snapshot_download = _modelscope.snapshot_download

    def _snapshot_download_local_first(model_id, *args, **kwargs):
        if "local_files_only" not in kwargs:
            try:
                return _ms_snapshot_download(
                    model_id, *args, local_files_only=True, **kwargs
                )
            except Exception:
                pass  # 沒有快取 → 照原本的方式連網抓一次
        return _ms_snapshot_download(model_id, *args, **kwargs)

    _modelscope.snapshot_download = _snapshot_download_local_first
except ImportError:
    pass  # 沒有 modelscope 的話 CosyVoice 本來就跑不起來，留給它自己報錯

# 指到哪個模型就跑哪個後端。預設 CosyVoice2-0.5B。
_DEFAULT_MODEL = os.path.join(COSYVOICE_REPO, "pretrained_models", "CosyVoice2-0.5B")
MODEL_DIR = os.environ.get("COSYVOICE_MODEL_DIR", _DEFAULT_MODEL)

# 是否為 CosyVoice2/3（有 cosyvoice2.yaml / cosyvoice3.yaml）→ 走 zero-shot。
_IS_V2 = os.path.exists(os.path.join(MODEL_DIR, "cosyvoice2.yaml")) or \
         os.path.exists(os.path.join(MODEL_DIR, "cosyvoice3.yaml"))

# --- CosyVoice2 zero-shot 參考語者 -------------------------------------------
# 預設用 CosyVoice repo 自帶的 asset/zero_shot_prompt.wav（一段中文女聲）。
# 換音色只要換這兩個環境變數（prompt wav + 對應逐字稿），不用改程式。
PROMPT_WAV = os.environ.get(
    "TTS_PROMPT_WAV", os.path.join(COSYVOICE_REPO, "asset", "zero_shot_prompt.wav")
)
PROMPT_TEXT = os.environ.get("TTS_PROMPT_TEXT", "希望你以后能够做的比我还好呦。")
ZERO_SHOT_SPK_ID = "imood_default"

# --- 男聲 -------------------------------------------------------------------
# 前端讓使用者選一男一女，所以 startup 時註冊兩個 zero-shot 語者。
# 男聲要一段 ~5-10 秒乾淨人聲 + 對應逐字稿（逐字稿要跟音檔內容一致，不然抽出來
# 的特徵會偏）。沒設定就只註冊女聲，收到 voice="male" 時退回女聲並印一行警告，
# 不讓服務因為缺素材而起不來。
PROMPT_WAV_MALE = os.environ.get("TTS_PROMPT_WAV_MALE", "")
PROMPT_TEXT_MALE = os.environ.get("TTS_PROMPT_TEXT_MALE", "")
ZERO_SHOT_SPK_ID_MALE = "imood_male"

# voice -> zero_shot_spk_id，startup 時依實際註冊成功的填
VOICE_SPK_IDS = {}

# --- CosyVoice-300M-SFT 語者 -----------------------------------------------
# 舊後端有內建語者，直接對應，不需要參考音檔
SPEAKER = os.environ.get("TTS_SFT_SPEAKER", "中文女")
SFT_SPEAKERS = {"female": "中文女", "male": "中文男"}

# --- 加速開關 --------------------------------------------------------------
# CosyVoice2：照 repo 的 vllm_example，vllm + trt + fp16 全開；jit 預設關
#   （缺對應 zip 會直接炸，要開再開）。首次啟動會 export vllm 權重 + build
#   TensorRT engine（一次性，之後快取成 flow.decoder.estimator.*.plan）。
# CosyVoice-300M-SFT：jit/fp16 實測更慢（見 docs/gpu-notes.md），一律不開，
#   只有 TRT 有效果——設 TTS_LOAD_TRT=1 啟用。
def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) == "1"


LOAD_VLLM = _flag("TTS_LOAD_VLLM", "1" if _IS_V2 else "0")
LOAD_TRT = _flag("TTS_LOAD_TRT", "1" if _IS_V2 else "0")
LOAD_JIT = _flag("TTS_LOAD_JIT", "0")
FP16 = _flag("TTS_FP16", "1" if _IS_V2 else "0")


def _f(name: str, default: str) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


# --- 音量 -----------------------------------------------------------------
# CosyVoice2 zero-shot 原始輸出偏小聲（active RMS ~-22 dBFS）。_level() 對每個
# model chunk（~2s）做 RMS 正規化到 TTS_RMS_DBFS，峰值壓在 TTS_PEAK_DBFS 以下，
# 超過的部分用 tanh soft knee 收（不硬削）。TTS_GAIN 是最後再乘的手動微調。
# 想更大聲：TTS_RMS_DBFS=-12（再吵設 -10）。想關掉正規化：TTS_NORMALIZE=0。
NORMALIZE = _flag("TTS_NORMALIZE", "1")
_TARGET_RMS = 10 ** (_f("TTS_RMS_DBFS", "-14.0") / 20)
_CEIL = 10 ** (_f("TTS_PEAK_DBFS", "-1.0") / 20)
GAIN = _f("TTS_GAIN", "1.0")
# TTS_DRIVE > 1 先做一層 tanh 軟飽和壓峰值（降 crest factor），normalize 之後
# RMS 就能推更高＝聽起來更大聲。1.0 = 關；1.5~2.5 之間微調（太高會鼻音/破）。
_DRIVE = max(_f("TTS_DRIVE", "1.0"), 1.0)

TARGET_SAMPLE_RATE = 16000
CHUNK_MS = 320  # 對齊 JoyGen diffusion decoder 8-frame batch @25fps
CHUNK_BYTES = int(TARGET_SAMPLE_RATE * (CHUNK_MS / 1000) * 2)  # 16-bit = 2 bytes/sample

app = FastAPI(title="imood-voice TTS service (CosyVoice)")

cosyvoice = None  # startup 時載入一次，避免每個請求都要重載模型
_t2s = None       # 繁體轉簡體，見下方 load_model() 的說明


@app.on_event("startup")
def load_model():
    global cosyvoice, _t2s
    from cosyvoice.cli.cosyvoice import AutoModel
    from opencc import OpenCC
    import torch

    cuda = torch.cuda.is_available()
    load_vllm = LOAD_VLLM and cuda
    load_trt = LOAD_TRT and cuda
    load_jit = LOAD_JIT and cuda
    fp16 = FP16 and cuda
    print(
        f"[tts] model={MODEL_DIR}\n"
        f"[tts] v2={_IS_V2} cuda={cuda} vllm={load_vllm} trt={load_trt} "
        f"jit={load_jit} fp16={fp16}",
        flush=True,
    )

    kwargs = dict(model_dir=MODEL_DIR, load_trt=load_trt, load_jit=load_jit, fp16=fp16)
    if _IS_V2:
        kwargs["load_vllm"] = load_vllm
        if load_vllm:
            # vLLM 需要在 import cosyvoice 前把 CosyVoice2ForCausalLM 註冊進去
            from vllm import ModelRegistry
            from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM

            ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

    cosyvoice = AutoModel(**kwargs)

    if _IS_V2:
        # 註冊 zero-shot 參考語者：抽一次 prompt 特徵存進 spk2info，之後每個
        # 請求用 zero_shot_spk_id 引用，不重抽（省首塊延遲）。
        print(f"[tts] registering zero-shot speaker (female) from {PROMPT_WAV}",
              flush=True)
        cosyvoice.add_zero_shot_spk(PROMPT_TEXT, PROMPT_WAV, ZERO_SHOT_SPK_ID)
        VOICE_SPK_IDS["female"] = ZERO_SHOT_SPK_ID

        if PROMPT_WAV_MALE and os.path.isfile(PROMPT_WAV_MALE) and PROMPT_TEXT_MALE:
            print(f"[tts] registering zero-shot speaker (male) from "
                  f"{PROMPT_WAV_MALE}", flush=True)
            cosyvoice.add_zero_shot_spk(PROMPT_TEXT_MALE, PROMPT_WAV_MALE,
                                        ZERO_SHOT_SPK_ID_MALE)
            VOICE_SPK_IDS["male"] = ZERO_SHOT_SPK_ID_MALE
        else:
            print("[tts] WARNING: 沒有男聲參考音檔（TTS_PROMPT_WAV_MALE / "
                  "TTS_PROMPT_TEXT_MALE），voice=male 會退回女聲", flush=True)

    # imood 的 LLM 一律回覆繁體中文，但 CosyVoice 的文字前處理偏簡體，餵繁體
    # 進去時字典沒有的字會念出不像中文的音。這裡只轉「要合成的文字」，前端
    # 顯示的文字不受影響，使用者看到的還是繁體。
    _t2s = OpenCC("t2s")

    # 這個服務最會「安靜地壞掉」的地方：文字正規化載不到時 CosyVoice 只印一行
    # info 就繼續跑，數字/時間/百分比會原樣送進聲學模型，音訊聽起來也還「正常」，
    # 所以很難發現。這裡把狀態講清楚。
    _tn = getattr(getattr(cosyvoice, "frontend", None), "text_frontend", "")
    if _tn:
        print(f"[tts] text frontend = {_tn}", flush=True)
    else:
        print(
            "[tts] WARNING: no text frontend — 數字/時間/百分比不會被正規化，"
            "會被逐字念出",
            flush=True,
        )

    print("[tts] ready", flush=True)


class SynthesizeRequest(BaseModel):
    text: str
    voice: str = "female"     # female | male；認不得的值一律退回 female


def _level(x: "np.ndarray") -> "np.ndarray":
    """單一 model chunk（~2s）的音量處理：RMS 正規化到目標、峰值不超過天花板，
    再用 tanh soft knee 收尾。純 numpy / CPU。"""
    peak = float(np.abs(x).max())
    if peak < 1e-4:
        return x  # 整段近乎靜音，別放大噪音底
    if _DRIVE > 1.0:
        x = np.tanh(x * _DRIVE) / np.tanh(_DRIVE)
        peak = float(np.abs(x).max())
    if NORMALIZE:
        voiced = x[np.abs(x) > 0.02]
        if voiced.size > x.size * 0.05:
            rms = float(np.sqrt(np.mean(voiced ** 2)))
            g = min(_TARGET_RMS / max(rms, 1e-6), _CEIL / peak)
            x = x * float(np.clip(g, 0.25, 12.0))
    if GAIN != 1.0:
        x = x * GAIN
    a = np.abs(x)
    hot = a > _CEIL
    if hot.any():
        x = x.copy()
        x[hot] = np.sign(x[hot]) * (
            _CEIL + (1.0 - _CEIL) * np.tanh((a[hot] - _CEIL) / (1.0 - _CEIL))
        )
    return x


def _pcm16_bytes(speech_tensor, resampler) -> bytes:
    """CosyVoice tensor（22050/24000Hz float, shape [1, N]）-> 16kHz PCM16 bytes（套 _level）。"""
    x = resampler(speech_tensor).squeeze(0).numpy().astype(np.float32)
    x = _level(x)
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def _model_chunks(text: str, voice: str = "female"):
    """依後端選 inference 方式，逐段 yield CosyVoice 原生輸出（dict）。"""
    if _IS_V2:
        spk_id = VOICE_SPK_IDS.get(voice) or VOICE_SPK_IDS.get("female")             or ZERO_SHOT_SPK_ID
        yield from cosyvoice.inference_zero_shot(
            text, "", "", zero_shot_spk_id=spk_id, stream=True
        )
    else:
        yield from cosyvoice.inference_sft(
            text, SFT_SPEAKERS.get(voice, SPEAKER), stream=True)


def _synthesize_chunks(text: str, voice: str = "female") -> Iterator[bytes]:
    """
    逐段呼叫 CosyVoice stream=True，把每段輸出 resample 成 16kHz，再切成固定
    320ms 的 PCM16 區塊依序 yield。CosyVoice 原生一段 ~1.7-2 秒，比 JoyGen 要
    的 320ms 粗很多，所以切塊這一步是必要的，不能直接轉發原生分段。
    """
    import torchaudio

    resampler = torchaudio.transforms.Resample(
        orig_freq=cosyvoice.sample_rate, new_freq=TARGET_SAMPLE_RATE
    )

    text = _t2s.convert(text)
    carry = b""  # 上一個 model chunk 切剩、不足 320ms 的尾巴
    for out in _model_chunks(text, voice):
        pcm = carry + _pcm16_bytes(out["tts_speech"], resampler)
        n_full = len(pcm) // CHUNK_BYTES
        for i in range(n_full):
            yield pcm[i * CHUNK_BYTES:(i + 1) * CHUNK_BYTES]
        carry = pcm[n_full * CHUNK_BYTES:]

    if carry:
        yield carry


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    text = req.text.strip()
    if not text or cosyvoice is None:
        return StreamingResponse(iter(()), media_type="application/octet-stream")
    voice = (req.voice or "female").lower()
    if voice not in ("female", "male"):
        voice = "female"
    return StreamingResponse(
        _synthesize_chunks(text, voice), media_type="application/octet-stream"
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": cosyvoice is not None,
        "model_dir": MODEL_DIR,
        "backend": "cosyvoice2-zeroshot" if _IS_V2 else "cosyvoice-300m-sft",
        # 部署時用這個確認男聲素材到底有沒有掛上
        "voices": sorted(VOICE_SPK_IDS) if _IS_V2 else sorted(SFT_SPEAKERS),
    }
