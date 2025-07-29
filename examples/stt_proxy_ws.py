# proxy_ws.py
from dotenv import load_dotenv
load_dotenv()
import os
import asyncio
import websockets
import uvicorn
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # or hardcode your key for local testing
print(f"API KEY: {OPENAI_API_KEY}")

app = FastAPI()

@app.websocket("/ws/realtime")
async def proxy_realtime(websocket: WebSocket):
    await websocket.accept()

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
                        await openai_ws.send(data)
                except WebSocketDisconnect:
                    await openai_ws.close()
                except Exception as e:
                    print("Client -> OpenAI Error:", e)

            async def openai_to_client():
                try:
                    while True:
                        data = await openai_ws.recv()
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