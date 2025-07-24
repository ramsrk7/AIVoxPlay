import soundfile as sf

def save_audio(tokens, file, sr=24000):
    sf.write(file, tokens, sr)