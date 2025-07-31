# ws.py
import asyncio, websockets
from typing import Optional

class WebSocketManager:
    def __init__(self, url: str, headers: Optional[dict]=None):
        self.url = url
        self.headers = headers or {}
        self._ws = None

    async def __aenter__(self):
        self._ws = await websockets.connect(self.url, extra_headers=self.headers)
        return self._ws

    async def __aexit__(self, exc_type, exc, tb):
        if self._ws:
            await self._ws.close()
