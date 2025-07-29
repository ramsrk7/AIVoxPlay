from fastapi import FastAPI, Request, Body
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import struct
from typing import Generator
import uvicorn

from aivoxplay.tts.orpheus import OrpheusTTS
from aivoxplay.utils.tts import QueueSession

# Initialize TTS model with streaming enabled
tts_engine = OrpheusTTS(endpoint="https://6isojpjg84ev41-8000.proxy.runpod.net/v1")
app = FastAPI()
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create global queue stream controller
#stream = tts_engine.queue(max_workers=2)
qsession = QueueSession(engine=tts_engine)

def build_wav_header(sample_rate: int = 24000, bits: int = 16, channels: int = 1) -> bytes:
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    data_length = 0  # Placeholder for streaming
    return struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_length,
        b'WAVE',
        b'fmt ',
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
        b'data',
        data_length
    )

def stream_audio_from_text_pipeline() -> Generator[bytes, None, None]:
    yield build_wav_header()
    for audio_chunk in stream.stream():
        yield audio_chunk

@app.post("/add")
async def add_prompt(text: str = Body(..., embed=True)):
    qsession.add(text)
    return JSONResponse({"status": "added", "text": text})

@app.post("/clear")
async def clear_queue():
    qsession.clear()
    return JSONResponse({"status": "cleared"})

@app.post("/cancel")
async def cancel_jobs():
    qsession.cancel()
    return JSONResponse({"status": "cancelled"})

def gen_from_controller(ctl):
    yield build_wav_header()
    for chunk in ctl.stream():   # finishes because we called close() above
        yield chunk

@app.get("/play")
async def play_audio():
    ctl = qsession.play_and_rotate()
    return StreamingResponse(gen_from_controller(ctl), media_type="audio/wav")

@app.get("/tts")
def tts(prompt: str = "Hello"):
    ctl = tts_engine.queue(max_workers=2)  # new controller for this request
    ctl.add(prompt)
    ctl.add("The End!")
    ctl.close()                            # end after draining

    def gen():
        yield build_wav_header()
        for chunk in ctl.stream():         # finishes when queue drains
            yield chunk

    return StreamingResponse(gen(), media_type="audio/wav")

if __name__ == "__main__":
    uvicorn.run("stream_server_queue:app", host="0.0.0.0", port=8080)
