import soundfile as sf
import numpy as np
import os
from itertools import chain

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
