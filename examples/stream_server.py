from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import struct
from typing import Generator
import uvicorn

from aivoxplay.tts.orpheus import OrpheusTTS


# Initialize TTS model with streaming enabled
tts_engine = OrpheusTTS(endpoint="https://b9yzo99l0z7cgu-8000.proxy.runpod.net/v1")

app = FastAPI()

def build_wav_header(sample_rate: int = 24000, bits: int = 16, channels: int = 1) -> bytes:
    """Generate a WAV file header with given audio parameters."""
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    data_length = 0  # Placeholder for streaming use

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

def stream_audio_from_text(prompt_text: str) -> Generator[bytes, None, None]:
    """Yield WAV header followed by audio chunks from the TTS model."""
    yield build_wav_header()

    stream = tts_engine.speak(
        text=prompt_text,
        voice="tara",
        stream=True
    )

    for audio_chunk in stream:
        yield audio_chunk

@app.get("/tts")
async def text_to_speech(request: Request):
    """Convert the prompt query parameter into audio and stream it back as WAV."""
    prompt = request.query_params.get(
        "prompt",
        "Hey there! You forgot to include a prompt for speech synthesis."
    )
    return StreamingResponse(stream_audio_from_text(prompt), media_type="audio/wav")

if __name__ == "__main__":
    uvicorn.run("stream_server:app", host="0.0.0.0", port=8080)
