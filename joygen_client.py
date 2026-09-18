# ============================================================
# joygen_client.py —— 把 TTS 的 PCM 推給 JoyGen 的 input streaming 服務。
#
# 協定定義在 joygen-deployment-notes/streaming_input/ingest.py，那邊是規格的
# 來源；這裡是 imood-voice 這側的實作（不同 repo、不同 venv，所以各自帶一份，
# 不做跨 repo import）。
#
#   | type: 1 byte | length: 4 bytes big-endian | payload |
#   1 BEGIN  JSON：{"session","utterance","voice","sample_rate","emotion","text"}
#   2 AUDIO  raw PCM16 LE mono 16kHz
#   3 END    空
#   4 STATUS JSON（server -> client）
#
# 走 TCP 不走 WebSocket：這是 server 對 server，沒有瀏覽器，而且 JoyGen 那端的
# conda env 釘在 Python 3.8 + CUDA 11.7，不值得為了握手協定多塞一個依賴。
# 音訊不能丟包（少一個 sample 整條 mel 時間軸就位移），所以用 TCP；影像才走 RTP。
#
# 這裡所有失敗都只記 log 不往外丟：JoyGen 沒開機、連不上、中途斷線都不該影響
# /ws/audio 的文字與語音回覆，跟 tts_service 連不上時的處理方式一致。
# ============================================================

from __future__ import annotations

import asyncio
import json
import logging
import struct
from typing import Optional

logger = logging.getLogger("joygen_client")

BEGIN, AUDIO, END, STATUS = 1, 2, 3, 4
HEADER = struct.Struct(">BI")
SAMPLE_RATE = 16000


class JoyGenClient:
    """一個 WebSocket 連線（一位使用者）配一個。每句回覆呼叫一次
    begin() -> audio() xN -> end()。"""

    def __init__(self, host: str, port: int, enabled: bool = True,
                 connect_timeout: float = 2.0, status_timeout: float = 120.0):
        self.host = host
        self.port = port
        self.enabled = enabled
        self.connect_timeout = connect_timeout
        self.status_timeout = status_timeout
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._broken = False          # 這次連線已壞，不要每塊都再試一次
        self._pending = None          # 還在等 JoyGen 回報的那一句

    # -- 連線 ---------------------------------------------------

    async def _ensure(self) -> bool:
        if not self.enabled or self._broken:
            return False
        if self._writer is not None:
            return True
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self.connect_timeout)
            sock = self._writer.get_extra_info("socket")
            if sock is not None:
                import socket as _socket
                sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
            logger.info("connected to joygen %s:%s", self.host, self.port)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("joygen 連不上 %s:%s（%s），這次不送畫面",
                           self.host, self.port, exc)
            self._broken = True
            return False

    async def _send(self, ftype: int, payload: bytes = b"") -> bool:
        if self._writer is None:
            return False
        try:
            self._writer.write(HEADER.pack(ftype, len(payload)) + payload)
            await self._writer.drain()     # drain 就是 backpressure 的來源
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("joygen 寫入失敗，停止這次推送: %s", exc)
            self._broken = True
            return False

    async def _status(self) -> Optional[dict]:
        if self._reader is None:
            return None
        try:
            head = await asyncio.wait_for(self._reader.readexactly(HEADER.size),
                                          timeout=self.status_timeout)
            ftype, length = HEADER.unpack(head)
            payload = await asyncio.wait_for(self._reader.readexactly(length),
                                             timeout=self.status_timeout) \
                if length else b""
            if ftype != STATUS:
                return None
            return json.loads(payload.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("joygen 沒有回 status: %s", exc)
            self._broken = True
            return None

    # -- 一句話 -------------------------------------------------

    async def begin(self, session: str, utterance: str, voice: str = "female",
                    emotion: Optional[str] = None,
                    text: Optional[str] = None) -> bool:
        # 上一句的結果還沒收完就不能開下一句：協定是一問一答，STATUS 沒讀走
        # 的話下一次讀到的會是錯的那個 frame。
        if self._pending is not None:
            try:
                await self._pending
            except Exception:  # noqa: BLE001
                pass
            self._pending = None
        if not await self._ensure():
            return False
        meta = {"session": session, "utterance": utterance, "voice": voice,
                "sample_rate": SAMPLE_RATE, "emotion": emotion, "text": text}
        if not await self._send(BEGIN,
                                json.dumps(meta, ensure_ascii=False).encode("utf-8")):
            return False
        status = await self._status()
        if not status or not status.get("ok"):
            logger.warning("joygen 拒絕這句: %s", status)
            return False
        return True

    async def audio(self, pcm: bytes) -> bool:
        if not pcm or self._broken or self._writer is None:
            return False
        return await self._send(AUDIO, pcm)

    async def end(self) -> Optional[dict]:
        """送出 END 並等 JoyGen 回報結果。它要把整句畫完才會回，所以
        status_timeout 要比一句話的生成時間長。

        注意：**不要在 WebSocket handler 裡直接 await 這個**。JoyGen 畫一句
        要數十秒，那段期間 handler 沒有回到 receive()，uvicorn 就會停止讀取
        那條連線（流量控制），連 ping/pong 都不處理，最後被 keepalive 判定
        斷線。要在 handler 裡用就走 end_in_background()。
        """
        if self._broken or self._writer is None:
            return None
        if not await self._send(END):
            return None
        return await self._status()

    async def end_in_background(self, on_done) -> bool:
        """送出 END 之後把「等結果」丟到背景，讓呼叫端可以馬上回去收訊息。

        on_done(status) 是個 coroutine function，JoyGen 回報時才會被呼叫。
        """
        if self._broken or self._writer is None:
            return False
        if not await self._send(END):
            return False

        async def waiter():
            status = await self._status()
            if status is not None and on_done is not None:
                try:
                    await on_done(status)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("joygen 結果回呼失敗: %s", exc)

        self._pending = asyncio.create_task(waiter())
        return True

    async def close(self) -> None:
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
        if self._writer is None:
            return
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._reader = None
            self._writer = None
            self._broken = False
