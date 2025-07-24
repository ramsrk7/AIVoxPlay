from .orpheus import OrpheusTTS

def get_tts(model_name: str):
    if "orpheus" in model_name.lower():
        return OrpheusTTS(model_name)
    raise ValueError(f"No TTS class for model: {model_name}")


"""
Usage:

from aivoxplay.tts.factory import get_tts

tts = get_tts("unsloth/orpheus-3b-tts")
tts.speak("Hello, this is Orpheus speaking!", stream=False)

"""

