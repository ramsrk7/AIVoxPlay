# proxy_ws.py
from dotenv import load_dotenv
load_dotenv()

import os, json, re, asyncio, struct
from collections import deque
from typing import Deque

import websockets
import uvicorn
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect

from aivoxplay.tts.orpheus import OrpheusTTS
from aivoxplay.utils.tts import QueueSession

app = FastAPI()

# STT helper (unchanged)
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
def process_transcription_message(raw: str, buffer: str, queue: Deque[str]) -> str:
    msg = json.loads(raw); t = msg.get("type","")
    if t.endswith(".delta"):
        return buffer + msg.get("delta","")
    if t.endswith(".completed"):
        txt = msg.get("transcript","").strip()
        if txt:
            for s in _SENTENCE_SPLIT.split(txt):
                s = s.strip()
                if s: queue.append(s)
        return ""
    return buffer

# TTS setup (unchanged)
TTS_ENDPOINT = os.getenv("TTS_ENDPOINT")
tts_engine = OrpheusTTS(endpoint=TTS_ENDPOINT)
qsession   = QueueSession(engine=tts_engine)

def build_wav_header(sr=24000, bits=16, ch=1):
    byte_rate   = sr * ch * bits // 8
    block_align = ch * bits // 8
    return struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36, b'WAVE', b'fmt ',16,1,ch,
        sr, byte_rate, block_align, bits, b'data',0
    )

@app.websocket("/ws/realtime")
async def proxy_realtime(ws: WebSocket):
    await ws.accept()
    buffer = ""
    sentence_queue: Deque[str] = deque()
    cancel_tts = False

    # connect to OpenAI STT
    headers = {
      "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
      "OpenAI-Beta": "realtime=v1",
    }
    async with websockets.connect(
        "wss://api.openai.com/v1/realtime?intent=transcription",
        extra_headers=headers
    ) as openai_ws:

        # handle **both** audio→STT and control messages
        async def client_reader():
            nonlocal cancel_tts
            try:
                while True:
                    msg = await ws.receive()  # can be text or bytes
                    if 'text' in msg:
                        # control channel
                        try:
                            j = json.loads(msg['text'])
                            if j.get("type") == "cancel_audio":
                                cancel_tts = True
                                continue
                        except:
                            # forward any other text to OpenAI
                            await openai_ws.send(msg['text'])
                    elif 'bytes' in msg:
                        # raw mic PCM → STT 
                        await openai_ws.send(msg['bytes'])
            except:
                await openai_ws.close()

        async def openai_reader():
            nonlocal buffer, cancel_tts
            try:
                while True:
                    raw = await openai_ws.recv()
                    buffer = process_transcription_message(raw, buffer, sentence_queue)

                    # for each completed sentence, run TTS and stream it
                    while sentence_queue:
                        sent = sentence_queue.popleft()
                        qsession.add(sent)
                        ctl = qsession.play_and_rotate()
                        ctl.close()

                        # header
                        await ws.send_bytes(build_wav_header())
                        # chunks
                        for chunk in ctl.stream():
                            if cancel_tts: break
                            await ws.send_bytes(chunk)

                        # signal end or cancellation
                        if cancel_tts:
                            await ws.send_text(json.dumps({"type":"audio.cancelled"}))
                            cancel_tts = False
                        else:
                            await ws.send_text(json.dumps({"type":"audio.complete"}))
            except:
                await ws.close()

        await asyncio.gather(client_reader(), openai_reader())


if __name__ == "__main__":
    uvicorn.run("proxy_ws:app", host="0.0.0.0", port=8001)
