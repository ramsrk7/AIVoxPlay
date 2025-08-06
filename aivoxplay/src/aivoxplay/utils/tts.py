import soundfile as sf
import numpy as np
import os
from itertools import chain
import threading

def save_audio(tokens, file_path, sr=24000):
    """
    tokens: audio tokens generated
    file_path: output file path
    sr: sample rate
    """
    sf.write(file_path, tokens, sr)


def save_audio_stream(audio_chunks, file_path, sr=24000):
    """
    Save chunks of PCM audio data as a valid WAV file.
    audio_chunks: Iterable of numpy arrays or raw PCM data (must be int16 format)
    """
    with sf.SoundFile(file_path, mode='w', samplerate=sr, channels=1, subtype='PCM_16') as f:
        for chunk in audio_chunks:
            if isinstance(chunk, bytes):
                chunk = np.frombuffer(chunk, dtype=np.int16)
            elif isinstance(chunk, str):
                chunk = np.frombuffer(chunk.encode('utf-8'), dtype=np.int16)
            f.write(chunk)


class QueueSession:
    def __init__(self, engine, max_workers=2, voice="tara"):
        self.engine = engine
        self.max_workers = max_workers
        self.voice = self.engine.voice
        self._lock = threading.Lock()
        self._controller = self.engine.queue(voice=self.voice, max_workers=self.max_workers)

    def add(self, text: str):
        with self._lock:
            print("Adding text to queue")
            self._controller.add(text)

    def clear(self):
        with self._lock:
            self._controller.clear()

    def cancel(self):
        # optional: cancel current job if one is active
        with self._lock:
            try:
                self._controller.cancel()   # if you don’t have cancel(), replace with clear()
            except AttributeError:
                self._controller.clear()

    def play_and_rotate(self):
        """
        Close the CURRENT controller (so its stream finishes after draining),
        return it for streaming, and immediately create a NEW controller for
        any future /add calls.
        """
        with self._lock:
            current = self._controller
            current.close()  # stream() will finish after queued items are drained

            # New session for subsequent /add calls (arrive while current is streaming)
            print("Creating new controller")
            print(self.engine)
            self._controller = self.engine.queue(voice=self.voice, max_workers=self.max_workers)

            return current
