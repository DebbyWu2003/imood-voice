# ============================================================
# tts_client.py —— 呼叫獨立的 tts_service.py（見該檔開頭說明）。
#
# 純邏輯、不依賴 FastAPI，跟 voice_asr.py 一樣可以獨立測試。
# ============================================================

from __future__ import annotations

from typing import AsyncIterator

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(connect=2.0, read=30.0, write=5.0, pool=5.0)


async def stream_tts(text: str, service_url: str,
                     voice: str = "female") -> AsyncIterator[bytes]:
    """
    呼叫 tts_service 的 /synthesize，逐塊 yield 16kHz/mono/16-bit PCM bytes
    （每塊固定 320ms，見 tts_service.py 的切塊邏輯）。

    voice 選 "female" / "male"。舊版 tts_service 不認得這個欄位也沒關係，
    FastAPI 會忽略多餘的 key，行為退回預設語者。

    連線失敗、timeout、非 200 都直接 raise（httpx 的例外或 RuntimeError），
    由呼叫端（server.py）決定要不要優雅降級。
    """
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        async with client.stream(
            "POST", f"{service_url}/synthesize",
            json={"text": text, "voice": voice},
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk
