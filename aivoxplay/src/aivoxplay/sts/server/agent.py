# src/my_voice_server/server.py
import os
import re
import json
import base64
import asyncio
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosedOK
from ...stt.helper.realtime import StreamingSession
from ...chat.naive import AIVoxChat, consumer

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

    # --- anywhere in your code, you can now push messages: ---
#await q.put("Hello!")

def get_app(stt, tts, external_chat_fn) -> FastAPI:
    app = FastAPI()

    @app.websocket("/voice")
    async def voice_endpoint(ws: WebSocket):
        await ws.accept()
        sentence_q = deque()
        agent_output_q = deque()
        q: asyncio.Queue[str] = asyncio.Queue()
        chat = AIVoxChat(external_chat_fn=external_chat_fn, agent_output=agent_output_q)
        consumer_task = asyncio.create_task(consumer(chat, q, stop_word="Bye"))
        cancel_tts = False
        loop = asyncio.get_running_loop()
        def on_transcript(text: str):
            sentence_q.append(text)
            # option 1: no backpressure, safe in same process
            loop.call_soon_threadsafe(q.put_nowait, text)

        session = StreamingSession(provider=stt, on_text=on_transcript)
        await session.start()

        async def client_reader():
            nonlocal cancel_tts
            try:
                while True:
                    frame = await ws.receive_text()
                    msg = json.loads(frame)
                    if msg.get("type") == "cancel_audio":
                        cancel_tts = True
                    elif msg.get("type") == "input_audio_buffer.append":
                        b64_payload = msg["audio"]
                        pcm_chunk = base64.b64decode(b64_payload)
                        await session.provider.send_audio_chunk(pcm_chunk)
            except (WebSocketDisconnect, asyncio.CancelledError, ConnectionClosedOK):
                pass
            finally:
                await session.stop()

        async def server_writer():
            nonlocal cancel_tts
            try:
                while True:
                    if agent_output_q:
                        sentence = agent_output_q.popleft()
                        print("Sentence: ", sentence)
                        await ws.send_text(json.dumps({
                            "type": "transcript",
                            "text": sentence
                        }))
                        async for chunk in tts.synth_stream(sentence):
                            if cancel_tts:
                                tts.cancel()
                                break
                            print("Sending Bytes....")
                            await ws.send_bytes(chunk)
                        await ws.send_text(json.dumps({"type": "audio.complete"}))
                        cancel_tts = False
                    else:
                        await asyncio.sleep(0.01)
            except (WebSocketDisconnect, asyncio.CancelledError):
                pass
            finally:
                await session.stop()

        task_in = asyncio.create_task(client_reader())
        task_out = asyncio.create_task(server_writer())
        done, pending = await asyncio.wait({task_in, task_out}, return_when=asyncio.FIRST_COMPLETED)
        await consumer_task

        for t in pending:
            t.cancel()


    return app

