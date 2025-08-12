import os
import json
import base64
import asyncio
import time
import contextlib
from collections import defaultdict, deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosedOK
from ...stt.helper.realtime import StreamingSession
from ...chat.naive import AIVoxChat, consumer


CLEAR_QUEUE_ON_CANCEL = True

client_state = defaultdict(lambda: {
    "stt_session": None,
    "agent_input_q": asyncio.Queue(),
    "agent_output_q": deque(),
    "consumer_task": None,
    "cancel_event": asyncio.Event(),
    "ws_out": None,                 # <— track the client’s /audio/out socket
    "interaction_log": {}
})

def get_app(stt, tts, external_chat_fn) -> FastAPI:
    app = FastAPI()

    @app.websocket("/audio/in/{client_id}")
    async def audio_in(ws: WebSocket, client_id: str):
        await ws.accept()
        state = client_state[client_id]

        def on_transcript(text: str):
            now = time.time()
            state["interaction_log"]["transcript_text"] = text
            state["interaction_log"]["stt_end"] = now
            asyncio.get_event_loop().call_soon_threadsafe(
                state["agent_input_q"].put_nowait, text
            )

        session = StreamingSession(provider=stt, on_text=on_transcript)
        state["stt_session"] = session
        state["interaction_log"]["stt_start"] = time.time()
        await session.start()

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
                    # 1) signal cancel
                    state["cancel_event"].set()

                    # 2) stop current provider if it supports cancel
                    with contextlib.suppress(Exception):
                        await tts.cancel()

                    # 3) drop queued sentences (stale)
                    if CLEAR_QUEUE_ON_CANCEL:
                        state["agent_output_q"].clear()

                    # 4) tell client to flush and hard-close /audio/out
                    if state.get("ws_out"):
                        with contextlib.suppress(Exception):
                            await state["ws_out"].send_text(json.dumps({"type": "audio.flush"}))
                            await state["ws_out"].close(code=4000, reason="flush")

                    # 5) reset for next turn
                    state["interaction_log"].pop("tts_first_token", None)

                elif msg.get("type") == "input_audio_buffer.append":
                    if "stt_start" not in state["interaction_log"]:
                        state["interaction_log"]["stt_start"] = time.time()
                    pcm_chunk = base64.b64decode(msg["audio"])
                    await session.provider.send_audio_chunk(pcm_chunk)

        except (WebSocketDisconnect, asyncio.CancelledError, ConnectionClosedOK):
            pass
        finally:
            with contextlib.suppress(Exception):
                await session.stop()
            if state["consumer_task"]:
                state["consumer_task"].cancel()

    @app.websocket("/audio/out/{client_id}")
    async def audio_out(ws: WebSocket, client_id: str):
        await ws.accept()
        state = client_state[client_id]
        state["ws_out"] = ws   # <— store reference

        async def stream_sentence(sentence: str):
            t_agent_end = state["interaction_log"].get("agent_end", time.time())
            await ws.send_text(json.dumps({"type": "transcript", "text": sentence}))
            first_token_time = None

            agen = tts.synth_stream(sentence).__aiter__()
            try:
                while True:
                    next_chunk_task = asyncio.create_task(agen.__anext__())
                    cancel_wait_task = asyncio.create_task(state["cancel_event"].wait())
                    done, pending = await asyncio.wait(
                        {next_chunk_task, cancel_wait_task},
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    if cancel_wait_task in done:
                        next_chunk_task.cancel()
                        with contextlib.suppress(Exception):
                            await tts.cancel()
                        # Do NOT send anything else; /audio/out will be closed by /audio/in
                        state["cancel_event"].clear()
                        return "cancelled"

                    try:
                        chunk = next_chunk_task.result()
                    except StopAsyncIteration:
                        break

                    if not first_token_time:
                        first_token_time = time.time()
                        state["interaction_log"]["tts_first_token"] = first_token_time

                    await ws.send_bytes(chunk)

            finally:
                for t in (locals().get("next_chunk_task"), locals().get("cancel_wait_task")):
                    if isinstance(t, asyncio.Task): t.cancel()

            await ws.send_text(json.dumps({"type": "audio.complete"}))
            state["interaction_log"] = {}
            return "done"

        try:
            while True:
                # If cancel happened and queue should be purged, it's already cleared
                if state["agent_output_q"]:
                    sentence = state["agent_output_q"].popleft()
                    res = await stream_sentence(sentence)
                    if res == "cancelled":
                        # Immediately stop loop; socket will close via /audio/in
                        return
                else:
                    await asyncio.sleep(0.01)
        except (WebSocketDisconnect, asyncio.CancelledError, ConnectionClosedOK):
            pass
        finally:
            if state.get("ws_out") is ws:
                state["ws_out"] = None
            with contextlib.suppress(Exception):
                if state["stt_session"]:
                    await state["stt_session"].stop()
            if state["consumer_task"]:
                state["consumer_task"].cancel()

    return app

