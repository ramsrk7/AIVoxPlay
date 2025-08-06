import asyncio
import os
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer
from dotenv import load_dotenv
from .pipeline import ParagraphStreamController

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai = AsyncOpenAI()

class OpenAITTS:
    def __init__(self):
        self.openai = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.decode_tokens_fn = None

    async def speak(self, text: str):
        async with self.openai.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=text,
            instructions="Speak in a cheerful and positive tone.",
            response_format="pcm",
        ) as response:
            await LocalAudioPlayer().play(response)

    async def stream_audio(self, text: str, voice: str = "coral"):
        response_format: str = "pcm"
        async with self.openai.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
            response_format=response_format,
        ) as response:
            print("Streaming audio...")
            async for chunk in response.iter_bytes():
                yield chunk

    def stream_audio_sync(self, text: str, voice: str = "coral"):
        """
        Synchronous generator wrapper over OpenAI's async streaming TTS.
        Yields raw PCM chunks so callers can `for b in stream_audio(...): ...`.
        """
        import asyncio, threading, queue

        _END = object()
        q: "queue.Queue[object]" = queue.Queue(maxsize=32)

        async def produce():
            try:
                async with self.openai.audio.speech.with_streaming_response.create(
                    model="gpt-4o-mini-tts",
                    voice=voice,
                    input=text,
                    response_format="pcm",
                ) as response:
                    async for chunk in response.iter_bytes():
                        q.put(chunk)
            except Exception as e:
                q.put(e)
            finally:
                q.put(_END)

        def runner():
            asyncio.run(produce())

        t = threading.Thread(target=runner, daemon=True)
        t.start()

        while True:
            item = q.get()
            if item is _END:
                break
            if isinstance(item, Exception):
                raise item
            yield item

        t.join()

    
    def queue(self, voice="coral", max_workers=1):
        controller = ParagraphStreamController(
            speak_tokens_fn=self.stream_audio_sync,                    # your token stream
            decode_tokens_fn=self.decode_tokens_fn,  # your byte decoder
            voice=voice,
            max_workers=max_workers,
        )
        return controller


async def main() -> None:
    async with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input="Today is a wonderful day to build something people love!",
        instructions="Speak in a cheerful and positive tone.",
        response_format="pcm",
    ) as response:
        await LocalAudioPlayer().play(response)

if __name__ == "__main__":
    asyncio.run(main())
