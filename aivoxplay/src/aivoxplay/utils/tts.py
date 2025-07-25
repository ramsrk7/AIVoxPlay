import soundfile as sf
import sounddevice as sd
import numpy as np
import os

def save_audio(tokens, file, sr=24000):
    """
    tokens: audio tokens generated
    file: output file path
    sr: sample rate
    """
    sf.write(file, tokens, sr)

def save_audio_stream(audio_chunks, file, sr=24000):
    """
    Save audio chunks to a file as they are generated.
    audio_chunks: an iterable/generator yielding numpy arrays (audio data)
    file: output file path
    sr: sample rate
    """
    with sf.SoundFile(file, mode='w', samplerate=sr, channels=1, subtype='PCM_16') as f:
        for chunk in audio_chunks:
            print("Writing....")
            f.write(chunk)

def save_audio_stream_2(audio_chunks, file: str, sr: int = 24000, stream_dir: str = "stream", save_full_file: bool = False):
    """
    Save audio chunks:
    - Each chunk is saved separately to `stream/chunk_i.wav`
    - Optionally saves the full stream to `file`
    
    Args:
        audio_chunks: generator yielding np.ndarray chunks
        file: output file path for full stream (optional)
        sr: sample rate
        stream_dir: folder to save individual chunks
        save_full_file: whether to also write the full stream to `file`
    """
    os.makedirs(stream_dir, exist_ok=True)

    full_audio = []  # Optional accumulation
    chunk_count = 0

    for chunk in audio_chunks:
        chunk_file = os.path.join(stream_dir, f"chunk_{chunk_count:03d}.wav")
        print(f"Writing chunk {chunk_count} to {chunk_file}")
        sf.write(chunk_file, chunk, sr)
        chunk_count += 1

        if save_full_file:
            full_audio.append(chunk)

    if save_full_file and full_audio:
        print(f"Writing full audio to {file}")
        full_audio_concat = np.concatenate(full_audio)
        sf.write(file, full_audio_concat, sr)
