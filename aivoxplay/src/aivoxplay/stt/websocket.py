# realtime_stt_client.py
import asyncio
import sounddevice as sd
import numpy as np
import websockets
import json

class RealtimeSTTWebSocketClient:
    def __init__(self, server_url="ws://localhost:8000/v1/realtime", sample_rate=16000):
        self.server_url = server_url
        self.sample_rate = sample_rate
        self.running = False
        self.transcript = []

    async def _stream_audio(self, websocket):
        def callback(indata, frames, time_info, status):
            if self.running:
                asyncio.run_coroutine_threadsafe(
                    websocket.send(indata.tobytes()), asyncio.get_event_loop()
                )

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16', callback=callback):
            print("🎤 Microphone streaming started")
            while self.running:
                await asyncio.sleep(0.1)

    async def _receive_transcripts(self, websocket):
        async for message in websocket:
            try:
                data = json.loads(message)
                text = data.get("text", "")
                print("📝", text)
                self.transcript.append(text)
            except Exception as e:
                print("Error parsing response:", e)

    async def run(self):
        self.running = True
        async with websockets.connect(self.server_url) as websocket:
            send_task = asyncio.create_task(self._stream_audio(websocket))
            recv_task = asyncio.create_task(self._receive_transcripts(websocket))
            await asyncio.gather(send_task, recv_task)

    def stop(self):
        self.running = False

    def get_transcript(self):
        return " ".join(self.transcript)

    def clear_transcript(self):
        self.transcript.clear()
