# proxy_ws.py
from dotenv import load_dotenv
load_dotenv()
import os
import asyncio
import websockets
import uvicorn
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect
import json
import re
from collections import deque
from typing import Deque, Tuple

# A queue to hold all finished sentences
sentence_queue: Deque[str] = deque()

# Regex to split on end‐of‐sentence punctuation (., !, ?), keeping the punctuation
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')


def process_transcription_message(
    raw: str,
    buffer: str,
    queue: Deque[str]
) -> str:
    """
    Parse one JSON line `raw` from the OpenAI realtime STT websocket.
    - Append any 'delta' text to `buffer`.
    - On a 'completed' event, take its full transcript, split into sentences,
      enqueue them, and clear buffer.
    Returns the updated buffer.
    """
    msg = json.loads(raw)
    t = msg.get("type", "")
    
    # 1) partial chunk
    if t.endswith(".delta"):
        chunk = msg.get("delta", "")
        return buffer + chunk
    
    # 2) final transcript for this item
    if t.endswith(".completed"):
        transcript = msg.get("transcript", "").strip()
        if transcript:
            # split into sentences, e.g. "Hello world. How are you?"
            parts = _SENTENCE_SPLIT.split(transcript)
            for sent in parts:
                sent = sent.strip()
                if sent:
                    queue.append(sent)
        # reset buffer for next utterance
        return ""
    
    # other message types (e.g. buffer‐commit) we ignore
    return buffer


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # or hardcode your key for local testing
print(f"API KEY: {OPENAI_API_KEY}")

app = FastAPI()

@app.websocket("/ws/realtime")
async def proxy_realtime(websocket: WebSocket):
    await websocket.accept()
    buffer = ""
    sentence_queue = deque()
    # Use correct OpenAI WebSocket endpoint
    openai_url = "wss://api.openai.com/v1/realtime?intent=transcription"

    try:
        # Establish outbound connection to OpenAI with Authorization header
        # stt_proxy_ws.py  (only the connect() block changed)
        async with websockets.connect(
                "wss://api.openai.com/v1/realtime?intent=transcription",
                extra_headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "OpenAI-Beta": "realtime=v1",          # <-- NEW
                }
        ) as openai_ws:

            async def client_to_openai():
                try:
                    while True:
                        data = await websocket.receive_text()
                        #print(data)
                        await openai_ws.send(data)
                except WebSocketDisconnect:
                    await openai_ws.close()
                except Exception as e:
                    print("Client -> OpenAI Error:", e)

            async def openai_to_client():

                nonlocal buffer

                try:
                    while True:
                        data = await openai_ws.recv()
                        # 1) feed it into your helper, get updated buffer
                        buffer = process_transcription_message(data, buffer, sentence_queue)

                        # 2) send out any complete sentences immediately
                        while sentence_queue:
                            sentence = sentence_queue.popleft()
                            # await websocket.send_text(json.dumps({
                            #     "type": "transcript.sentence",
                            #     "sentence": sentence
                            # }))
                            print(sentence)
                        await websocket.send_text(data)
                except Exception as e:
                    print("OpenAI -> Client Error:", e)
                    await websocket.close()

            await asyncio.gather(client_to_openai(), openai_to_client())

    except Exception as e:
        print("Connection setup failed:", e)
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run("stt_proxy_ws:app", host="0.0.0.0", port=8001)