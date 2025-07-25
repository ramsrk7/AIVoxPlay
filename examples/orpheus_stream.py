from aivoxplay.tts.orpheus import OrpheusTTS
from aivoxplay.utils.tts import save_audio_stream, save_and_play_stream

vox_tts = OrpheusTTS(endpoint="https://s2p6vdnxqnupj9-8000.proxy.runpod.net/v1")

# Get a generator that yields (audio_chunk, sample_rate)
audio_stream = vox_tts._streaming_speak(
    text="Hey people! The world of artificial intelligence has seen remarkable advancements in recent years, transforming the way we interact with technology on a daily basis. From virtual assistants that can schedule our appointments and answer complex questions, to sophisticated language models capable of generating human-like text, the possibilities seem endless. As researchers continue to push the boundaries of what machines can achieve, we are witnessing the emergence of systems that can not only understand and process natural language, but also generate speech that is nearly indistinguishable from that of a real person. This progress has profound implications for accessibility, education, entertainment, and countless other fields. Imagine a future where language barriers are effortlessly overcome, where personalized learning experiences are available to everyone, and where creative collaboration between humans and machines leads to innovations we have yet to imagine. Of course, with these advancements come important ethical considerations, such as ensuring privacy, preventing misuse, and promoting fairness. As we embrace the benefits of AI-driven technologies, it is crucial that we also remain vigilant and thoughtful about the challenges they present. By fostering open dialogue and responsible development, we can harness the power of artificial intelligence to create a more inclusive, efficient, and imaginative world for generations to come.",
    voice="dan"
)

# Save the audio as it streams in
save_and_play_stream((chunk for chunk, _ in audio_stream), file="test_audio_stream_3.wav", sr=24000)