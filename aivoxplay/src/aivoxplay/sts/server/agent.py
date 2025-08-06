import os
import json
import base64
import asyncio
import time
from collections import defaultdict, deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosedOK
from ...stt.helper.realtime import StreamingSession
from ...chat.naive import AIVoxChat, consumer

client_state = defaultdict(lambda: {
    "stt_session": None,
    "agent_input_q": asyncio.Queue(),
    "agent_output_q": deque(),
    "consumer_task": None,
    "cancel_tts": False,
    "interaction_log": {}  # track timestamps
})


def get_app(stt, tts, external_chat_fn) -> FastAPI:
    app = FastAPI()

    @app.websocket("/audio/in/{client_id}")
    async def audio_in(ws: WebSocket, client_id: str):
        await ws.accept()
        state = client_state[client_id]

        def on_transcript(text: str):
            now = time.time()
            print(f"[STT] {text}")
            state["interaction_log"]["transcript_text"] = text
            state["interaction_log"]["stt_end"] = now
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(state["agent_input_q"].put_nowait, text)

        # Start STT session
        session = StreamingSession(provider=stt, on_text=on_transcript)
        state["stt_session"] = session
        state["interaction_log"]["stt_start"] = time.time()
        await session.start()

        # Launch LLM agent consumer
        chat = AIVoxChat(
            external_chat_fn=external_chat_fn,
            agent_output=state["agent_output_q"],
            on_agent_response=lambda: state["interaction_log"].update({"agent_end": time.time()})
        )
        state["consumer_task"] = asyncio.create_task(
            consumer(chat, state["agent_input_q"], stop_word="Bye")
        )

        try:
            while True:
                frame = await ws.receive_text()
                msg = json.loads(frame)
                if msg.get("type") == "cancel_audio":
                    state["cancel_tts"] = True
                elif msg.get("type") == "input_audio_buffer.append":
                    if "stt_start" not in state["interaction_log"]:
                        state["interaction_log"]["stt_start"] = time.time()
                    pcm_chunk = base64.b64decode(msg["audio"])
                    await session.provider.send_audio_chunk(pcm_chunk)
        except (WebSocketDisconnect, asyncio.CancelledError, ConnectionClosedOK):
            print(f"[DISCONNECT] {client_id} /audio/in")
        finally:
            await session.stop()
            if state["consumer_task"]:
                state["consumer_task"].cancel()

    @app.websocket("/audio/out/{client_id}")
    async def audio_out(ws: WebSocket, client_id: str):
        await ws.accept()
        state = client_state[client_id]

        try:
            while True:
                if state["agent_output_q"]:
                    sentence = state["agent_output_q"].popleft()
                    t_agent_end = state["interaction_log"].get("agent_end", time.time())
                    print(f"[AGENT RESPONSE] {sentence}")

                    await ws.send_text(json.dumps({
                        "type": "transcript",
                        "text": sentence
                    }))

                    t_tts_start = time.time()
                    first_token_time = None

                    async for chunk in tts.synth_stream(sentence):
                        if state["cancel_tts"]:
                            tts.cancel()
                            break
                        if not first_token_time:
                            first_token_time = time.time()
                            print(f"[TTS] First audio chunk after {first_token_time - t_agent_end:.2f}s")
                            state["interaction_log"]["tts_first_token"] = first_token_time
                        await ws.send_bytes(chunk)

                    t_tts_end = time.time()
                    await ws.send_text(json.dumps({"type": "audio.complete"}))
                    state["cancel_tts"] = False

                    # Final logging
                    stt_dur = state["interaction_log"].get("stt_end", 0) - state["interaction_log"].get("stt_start", 0)
                    agent_dur = t_agent_end - state["interaction_log"].get("stt_end", t_agent_end)
                    tts_first_token_dur = state["interaction_log"].get("tts_first_token", t_tts_start) - t_agent_end
                    tts_total_dur = t_tts_end - t_agent_end
                    total_dur = t_tts_end - state["interaction_log"].get("stt_start", t_tts_end)

                    print(f"--- Interaction Latency Breakdown ({client_id}) ---")
                    print(f"• STT duration:             {stt_dur:.2f}s")
                    print(f"• Agent response time:     {agent_dur:.2f}s")
                    print(f"• TTS first token latency: {tts_first_token_dur:.2f}s")
                    print(f"• TTS total duration:      {tts_total_dur:.2f}s")
                    print(f"• Total interaction time:  {total_dur:.2f}s")
                    print("--------------------------------------------------\n")

                    # Reset
                    state["interaction_log"] = {}

                else:
                    await asyncio.sleep(0.01)
        except (WebSocketDisconnect, asyncio.CancelledError, ConnectionClosedOK):
            print(f"[DISCONNECT] {client_id} /audio/out")
        finally:
            if state["stt_session"]:
                await state["stt_session"].stop()
            if state["consumer_task"]:
                state["consumer_task"].cancel()

    return app
