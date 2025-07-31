from typing import Callable
import asyncio
from ..core import StreamingProvider

class StreamingSession:
    def __init__(self, provider: StreamingProvider, on_text: Callable[[str], None]):
        self.provider = provider
        self.on_text  = on_text
        self._running = False

    async def start(self):
        await self.provider.connect()
        self._running = True
        # launch receive loop
        asyncio.create_task(self._recv_loop())

    async def send_audio(self, chunk: bytes):
        if self._running:
            await self.provider.send_audio_chunk(chunk)

    async def _recv_loop(self):
        async for msg in self.provider.receive_messages():
            # dispatch “transcript” messages to callback
            if msg.get("type","").endswith(".completed"):
                text = msg.get("transcript","").strip()
                if text:
                    self.on_text(text)
        await self.stop()

    async def stop(self):
        self._running = False
        await self.provider.close()
