# ../providers/openai.py
import json, base64, struct
from typing import AsyncIterator
from websockets.exceptions import ConnectionClosedOK
from ..core import StreamingProvider, StreamError
from ..helper.ws import WebSocketManager

class OpenAIProvider(StreamingProvider):
    def __init__(self, api_key: str, model: str="gpt-4o-mini-transcribe", language: str="en"):
        self.url = "wss://api.openai.com/v1/realtime?intent=transcription"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Beta":   "realtime=v1",
        }
        self.model = model
        self.language = language
        self._ws = None

    async def connect(self):
        try:
            manager = WebSocketManager(self.url, self.headers)
            self._ws = await manager.__aenter__()
        except Exception as e:
            print("[OpenAIProvider] WebSocket connect failed:", e)
            raise
        # receive session.created
        created = json.loads(await self._ws.recv())
        sess_id = created["session"]["id"]
        print("OpenAI Session ID:",sess_id)
        # send update
        await self._ws.send(json.dumps({
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",

                "input_audio_transcription": {
                "model": self.model,   
                "language": "en",               
                "prompt": ""                    
                },

                "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500
                },

                "input_audio_noise_reduction": { "type": "near_field" },

                "include": ["item.input_audio_transcription.logprobs"]
            }
        }))

    async def send_audio_chunk(self, chunk: bytes):
        """
        Expects raw PCM16LE. Base64-encode into the JSON envelope.
        """
        b64 = base64.b64encode(chunk).decode()
        msg = {"type":"input_audio_buffer.append", "audio": b64}
        await self._ws.send(json.dumps(msg))

    async def receive_messages(self) -> AsyncIterator[dict]:
        try:
            while True:
                raw = await self._ws.recv()
                print(raw)
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError:
                    raise StreamError("Received non-JSON frame")
        except ConnectionClosedOK:
            print("[OpenAIProvider] WebSocket closed cleanly (1000)")
            return
            

    async def close(self):
        await self._ws.close()
