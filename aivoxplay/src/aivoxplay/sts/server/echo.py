import os
import re
import json
import base64
import asyncio
import time
from collections import defaultdict, deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosedOK
from ...stt.helper.realtime import StreamingSession

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

client_state = defaultdict(lambda: {
    "stt_session": None,
    "sentence_q": deque(),
    "cancel_tts": False,
    "interaction_log": {}
})


def process_transcription_message(raw, buffer, queue):
    msg = json.loads(raw)
    t = msg.get("type", "")
    if t.endswith(".delta"):
        return buffer + msg["delta"]
    if t.endswith(".completed"):
        txt = msg["transcript"].strip()
        for sent in _SENTENCE_SPLIT.split(txt):
            if sent.strip():
                queue.append(sent.strip())
        return ""
    return buffer


def get_app(stt, tts) -> FastAPI:
    app = FastAPI()

    @app.websocket("/audio/in/{client_id}")
    async def audio_in(ws: WebSocket, client_id: str):
        await ws.accept()
        state = client_state[client_id]
        log = state["interaction_log"]

        def on_transcript(text: str):
            print(f"[STT] {text}")
            log["stt_end"] = time.time()
            state["sentence_q"].append(text)

        session = StreamingSession(provider=stt, on_text=on_transcript)
        state["stt_session"] = session
        log["stt_start"] = time.time()
        await session.start()

        try:
            while True:
                frame = await ws.receive_text()
                msg = json.loads(frame)
                if msg.get("type") == "cancel_audio":
                    state["cancel_tts"] = True
                elif msg.get("type") == "input_audio_buffer.append":
                    if "stt_start" not in log:
                        log["stt_start"] = time.time()
                    pcm_chunk = base64.b64decode(msg["audio"])
                    await session.provider.send_audio_chunk(pcm_chunk)
        except (WebSocketDisconnect, asyncio.CancelledError, ConnectionClosedOK):
            print(f"[DISCONNECT] {client_id} /audio/in")
        finally:
            await session.stop()

    @app.websocket("/audio/out/{client_id}")
    async def audio_out(ws: WebSocket, client_id: str):
        await ws.accept()
        state = client_state[client_id]
        log = state["interaction_log"]

        try:
            while True:
                if state["sentence_q"]:
                    sentence = state["sentence_q"].popleft()
                    print(f"[OUT] Transmitting: {sentence}")
                    await ws.send_text(json.dumps({
                        "type": "transcript",
                        "text": sentence
                    }))

                    t_agent_end = time.time()
                    first_token_time = None

                    async for chunk in tts.synth_stream(sentence):
                        if state["cancel_tts"]:
                            tts.cancel()
                            break
                        if not first_token_time:
                            first_token_time = time.time()
                            print(f"[TTS] First chunk after {first_token_time - t_agent_end:.2f}s")
                            log["tts_first_token"] = first_token_time
                        await ws.send_bytes(chunk)

                    t_tts_end = time.time()
                    await ws.send_text(json.dumps({"type": "audio.complete"}))
                    state["cancel_tts"] = False

                    # Log durations
                    stt_dur = log.get("stt_end", 0) - log.get("stt_start", 0)
                    tts_first = log.get("tts_first_token", t_tts_end) - t_agent_end
                    tts_total = t_tts_end - t_agent_end
                    total = t_tts_end - log.get("stt_start", t_tts_end)

                    print(f"--- Interaction Timing ({client_id}) ---")
                    print(f"STT Duration:        {stt_dur:.2f}s")
                    print(f"TTS First Token:     {tts_first:.2f}s")
                    print(f"TTS Total Duration:  {tts_total:.2f}s")
                    print(f"Total Time:          {total:.2f}s")
                    print("----------------------------------------")

                    # Reset for next round
                    client_state[client_id]["interaction_log"] = {}
                else:
                    await asyncio.sleep(0.01)
        except (WebSocketDisconnect, asyncio.CancelledError, ConnectionClosedOK):
            print(f"[DISCONNECT] {client_id} /audio/out")
        finally:
            if state["stt_session"]:
                await state["stt_session"].stop()

    return app
