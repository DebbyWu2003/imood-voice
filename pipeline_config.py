# ============================================================
# pipeline_config.py —— 讀 joygen-deployment-notes/configs/pipeline.yaml
#
# 那個檔案是 imood voice 與 JoyGen 共用的唯一調參入口（VAD 斷句秒數、TTS
# 語者、JoyGen ingest 位址等）。兩邊讀同一份，才不會出現「這邊改了那邊沒改」。
#
# 找檔案的順序：
#   1. 環境變數 IMOOD_PIPELINE_CONFIG
#   2. <repo>/../joygen-deployment-notes/configs/pipeline.yaml
#   3. <repo>/configs/pipeline.yaml（本機副本，兩台機器分開跑時用）
# 都找不到就用下面的內建預設值並印一行警告——設定檔不在不該讓服務起不來。
#
# 注意：正式部署時兩個服務會在同一台機器上，第 2 條就會命中。開發期間
# imood voice 在 Windows、JoyGen 在 WSL，要嘛設 IMOOD_PIPELINE_CONFIG 指到
# \\wsl.localhost\... 的路徑，要嘛各留一份副本。
# ============================================================

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent

# 開發機上 JoyGen 所在的 WSL distro（只在 Windows 上當退路用，見 _candidates）
_WSL_DISTROS = ("Ubuntu-22.04",)

# 內建預設值：跟 pipeline.yaml 裡的值一致，找不到設定檔時用這組。
DEFAULTS: Dict[str, Any] = {
    "vad": {
        "frame_ms": 20,
        "vad_aggressiveness": 2,
        "end_silence_ms": 900,
        "min_utterance_ms": 400,
        "max_utterance_ms": 15000,
        "pre_pad_ms": 300,
        "calibration_ms": 500,
        "energy_gate_factor": 3.0,
        "min_noise_floor": 60.0,
        "coalesce_ms": 1000,
    },
    "tts": {
        "endpoint": "http://127.0.0.1:8001/synthesize",
        "voice": "female",
        "chunk_ms": 320,
        "sample_rate": 16000,
    },
    "joygen_input": {
        "ingest_host": "127.0.0.1",
        "ingest_port": 8100,
        "enabled": True,
    },
}

# Endpointer 認得的欄位。coalesce_ms 不是它的參數（那是 server.py 的續句
# 合併窗），所以要濾掉，否則 EndpointConfig(**vad) 會 TypeError。
_ENDPOINT_FIELDS = {
    "frame_ms", "vad_aggressiveness", "end_silence_ms", "min_utterance_ms",
    "max_utterance_ms", "pre_pad_ms", "calibration_ms", "energy_gate_factor",
    "min_noise_floor",
}


def _candidates():
    env = os.environ.get("IMOOD_PIPELINE_CONFIG")
    if env:
        yield Path(env)
    yield REPO_ROOT.parent / "joygen-deployment-notes" / "configs" / "pipeline.yaml"
    yield REPO_ROOT / "configs" / "pipeline.yaml"
    # 開發機專用的退路：這個 repo 在 Windows、JoyGen 在 WSL，上面第 2 條永遠
    # 不會命中（C:\imood_project\ 底下沒有 joygen-deployment-notes）。同機部署
    # 之後這條會自然失效（UNC 路徑不存在），第 2 條才是正式環境要走的。
    if os.name == "nt":
        for distro in _WSL_DISTROS:
            yield Path(rf"\\wsl.localhost\{distro}\home\cgmhaha\imood_project"
                       r"\joygen-deployment-notes\configs\pipeline.yaml")


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load() -> Dict[str, Any]:
    for path in _candidates():
        try:
            if not path.is_file():
                continue
        except OSError:
            continue  # UNC 路徑不通時 is_file() 會丟，不要讓它炸掉啟動
        try:
            import yaml
            with path.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            print(f"[config] 讀取 {path}", flush=True)
            cfg = _deep_merge(DEFAULTS, raw)
            # 記住讀到哪一份：yaml 裡的相對路徑是相對於 notes 根目錄
            # （= 這個檔案的上上層），clip_dir() 靠這個推導。
            cfg["_config_path"] = str(path)
            return cfg
        except Exception as exc:  # noqa: BLE001
            print(f"[config] {path} 讀取失敗（{exc}），改用內建預設值", flush=True)
            break
    # 這裡要吵一點。內建預設值裡沒有 joygen_output，所以掉到這條路的後果是
    # 前端的 avatar 影片目錄變成空字串、影片全部 404，而且不會有別的錯誤訊息。
    # VAD 那些值因為跟 yaml 一致所以看起來正常，只有調過參之後才會發現不同步。
    print("[config] !! 找不到 pipeline.yaml，使用內建預設值。", flush=True)
    print("[config] !! 後果：avatar 影片會 404，且 yaml 裡調的參數不會生效。",
          flush=True)
    print("[config] !! 修法：設 IMOOD_PIPELINE_CONFIG 指到共用的那一份。",
          flush=True)
    return dict(DEFAULTS)


def clip_dir(cfg: Dict[str, Any]) -> str:
    """JoyGen 每句輸出的 mp4 目錄，**從 imood-voice 這台機器看到的路徑**。

    三種情況，依序：

    1. `utterance_dir_host` 有填 -> 直接用（自動推導不出來時的逃生口）
    2. `utterance_dir` 是相對路徑 -> 接在 notes 根目錄後面。notes 根目錄是從
       「pipeline.yaml 讀到的位置」反推的，所以不管 yaml 是在本機還是在
       `\\\\wsl.localhost\\...`，算出來都是這台機器走得到的路徑。
    3. `utterance_dir` 是 POSIX 絕對路徑而我們在 Windows -> JoyGen 在 WSL 裡，
       把 /home/... 翻成 UNC。同機部署時這條不會觸發（絕對路徑本來就走得到）。

    推不出來就回空字串，呼叫端會跳過掛載。
    """
    jout = cfg.get("joygen_output", {}) or {}
    host = (jout.get("utterance_dir_host") or "").strip()
    if host:
        return host

    raw = (jout.get("utterance_dir") or "").strip()
    if not raw:
        return ""

    cfg_path = cfg.get("_config_path")
    if not Path(raw).is_absolute():
        if not cfg_path:
            return ""                      # 沒有基準點就別亂猜
        notes_root = Path(cfg_path).resolve().parent.parent
        return str(notes_root / raw)

    p = Path(raw)
    if os.name == "nt" and raw.startswith("/"):
        # POSIX 絕對路徑 + 我們在 Windows = JoyGen 在 WSL。優先沿用
        # pipeline.yaml 自己所在的那個 UNC 前綴，才不會猜錯 distro。
        if cfg_path and str(cfg_path).startswith("\\\\wsl"):
            parts = Path(cfg_path).parts          # ('\\\\wsl.localhost\\<distro>', 'home', ...)
            return str(Path(parts[0]) / raw.lstrip("/").replace("/", os.sep))
        for distro in _WSL_DISTROS:
            cand = Path(rf"\\wsl.localhost\{distro}") / raw.lstrip("/").replace("/", os.sep)
            if cand.parent.exists():
                return str(cand)
        return ""
    return str(p)


def endpoint_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """只取 EndpointConfig 認得的欄位。"""
    vad = cfg.get("vad", {})
    return {k: v for k, v in vad.items() if k in _ENDPOINT_FIELDS}
