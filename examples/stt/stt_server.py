from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import threading
import asyncio
import uvicorn
from aivoxplay.stt.websocket import RealtimeSTTWebSocketClient
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
import aiohttp

app = FastAPI()
stt_client = RealtimeSTTWebSocketClient()
thread = None

# CORS for client.html
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/start")
async def start_streaming():
    global thread
    if stt_client.running:
        return JSONResponse({"status": "already running"})
    
    def run_client():
        asyncio.run(stt_client.run())

    thread = threading.Thread(target=run_client)
    thread.start()
    return {"status": "started"}

@app.post("/stop")
async def stop_streaming():
    stt_client.stop()
    return {"status": "stopped"}

@app.get("/transcript")
async def get_transcript():
    return {"transcript": stt_client.get_transcript()}

@app.post("/clear")
async def clear_transcript():
    stt_client.clear_transcript()
    return {"status": "cleared"}


if __name__ == "__main__":
    uvicorn.run("stt_server:app", host="0.0.0.0", port=8001)
