from dotenv import load_dotenv
load_dotenv()
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from aivoxplay.sts.server.echo import get_app
from aivoxplay.tts.orpheus import OrpheusTTS
from aivoxplay.tts.engine import TTSEngine
from aivoxplay.stt.providers.openai import OpenAIProvider

tts = TTSEngine(tts=OrpheusTTS(endpoint="https://m6gwtl16xaun5i-8000.proxy.runpod.net/v1"))
stt = OpenAIProvider(api_key=os.getenv('OPENAI_API_KEY'))

app = get_app(tts=tts, stt=stt)


# allow your client’s origin (or "*" to allow all)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # or ["http://localhost:8002"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run("echo_server:app", host="0.0.0.0", port=8000)
