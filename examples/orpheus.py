from aivoxplay.tts.orpheus import OrpheusTTS
from aivoxplay.utils.tts import save_audio

vox_tts = OrpheusTTS(endpoint="https://yq5n8uhw0n4c87-8000.proxy.runpod.net/v1")

audio_tokens, sr = vox_tts.speak(text="The day is gloomy. I just feel like napping. What should I do?", voice="dan")

save_audio(tokens=audio_tokens, file="test_audio.wav")
