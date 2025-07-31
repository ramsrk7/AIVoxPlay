from dotenv import load_dotenv
load_dotenv()
import os, json, struct, asyncio, base64, re
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import websockets
from websockets.exceptions import ConnectionClosedOK
from aivoxplay.tts.orpheus import OrpheusTTS
from aivoxplay.utils.tts import QueueSession
import uvicorn

app = FastAPI()
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

def process_transcription_message(raw, buffer, queue):
    msg = json.loads(raw)
    t = msg.get("type","")
    if t.endswith(".delta"):
        return buffer + msg["delta"]
    if t.endswith(".completed"):
        txt = msg["transcript"].strip()
        for sent in _SENTENCE_SPLIT.split(txt):
            if sent.strip(): queue.append(sent.strip())
        return ""
    return buffer

def build_wav_header(sr=24000, bits=16, ch=1):
    byte_rate   = sr*ch*bits//8
    block_align = ch*bits//8
    return struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF',36,b'WAVE',b'fmt ',16,1,ch, sr,byte_rate,block_align,bits,b'data',0
    )

tts = OrpheusTTS(endpoint="https://o1qhu00kch42ga-8000.proxy.runpod.net/v1")
qsession = QueueSession(engine=tts)

@app.websocket("/ws/realtime")
async def proxy_realtime(ws: WebSocket):
    await ws.accept()
    buffer = ""
    sentence_q = deque()
    cancel_tts = False

    # 1) Dial out to OpenAI STT
    async with websockets.connect(
        "wss://api.openai.com/v1/realtime?intent=transcription",
        extra_headers={
          "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
          "OpenAI-Beta":   "realtime=v1"
        }
    ) as stt_ws:
        # receive session.created + send update…
        qsession.add("Processing!")  
        created = json.loads(await stt_ws.recv())
        sess_id = created["session"]["id"]
        await stt_ws.send(json.dumps({
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",

                "input_audio_transcription": {
                "model": "gpt-4o-transcribe",   
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

        async def client_to_stt():
            nonlocal cancel_tts
            try:
                while True:
                    raw = await ws.receive_text()
                    msg = json.loads(raw)
                    if msg.get("type") == "cancel_audio":
                        cancel_tts = True
                    elif msg.get("type") == "input_audio_buffer.append":
                        # forward STT audio to OpenAI
                        await stt_ws.send(raw)
            except (WebSocketDisconnect, asyncio.CancelledError, ConnectionClosedOK):
                pass
            finally:
                await stt_ws.close()

        async def stt_to_client():
            nonlocal buffer, cancel_tts
            try:
                while True:
                    raw = await stt_ws.recv()
                    print(raw)
                    buffer = process_transcription_message(raw, buffer, sentence_q)

                    # when full sentences arrive, run TTS immediately
                    while sentence_q:
                        sent = sentence_q.popleft()
                        print(sent)
                        # send transcript back
                        await ws.send_text(json.dumps({
                          "type": "transcript",
                          "text": sent
                        }))

                        # TTS → chunks
                        
                        qsession.add(sent)        # or your process_text()
                        ctl = qsession.play_and_rotate()
                        ctl.close()

                        # header (binary)
                        await ws.send_bytes(build_wav_header())

                        for chunk in ctl.stream():
                            if cancel_tts:
                                break
                            await ws.send_bytes(chunk)

                        # notify end/cancel
                        await ws.send_text(json.dumps({
                          "type": "audio.cancelled" if cancel_tts else "audio.complete"
                        }))
                        cancel_tts = False
            except (asyncio.CancelledError, ConnectionClosedOK):
                pass
            finally:
                await ws.close()

        # drive both loops and tear down on first exit
        t1 = asyncio.create_task(client_to_stt())
        t2 = asyncio.create_task(stt_to_client())
        done, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending: t.cancel()

if __name__ == "__main__":
    uvicorn.run("sts_server:app", host="0.0.0.0", port=8001)
