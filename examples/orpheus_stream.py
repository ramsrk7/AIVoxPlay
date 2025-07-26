from aivoxplay.tts.orpheus import OrpheusTTS
from aivoxplay.utils.tts import save_audio_stream
import time

vox_tts = OrpheusTTS(endpoint="https://b9yzo99l0z7cgu-8000.proxy.runpod.net/v1")

start_time = time.time()
print(f"Start time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
# Get a generator that yields (audio_chunk, sample_rate)
audio_stream = vox_tts.speak(
    text="No worries — it happens to all of us. Here's how you can reset your Gmail password: Go to the Gmail login page at https://accounts.google.com/ — Enter your Gmail address and click Next. On the password screen click Forgot password. Then, follow the on-screen instructions. Google may ask you to verify your identity using a recovery email, your phone number, or by answering some security questions. If you’ve set up two-step verification, it may prompt you to use that as well. Once you're verified, you’ll be able to create a new password. Would you like me to walk you through any of these steps in more detail?",
    voice="tara",
    stream=True
)

# Save the audio as it streams in
#save_and_play_stream((chunk for chunk, _ in audio_stream), file="test_audio_stream_3.wav", sr=24000)
save_audio_stream(audio_chunks=audio_stream, file_path="tara_it_agent.wav")

end_time = time.time()
print(f"End time: {time.strftime('%H:%M:%S', time.localtime(end_time))}")

duration = end_time - start_time
print(f"Duration: {duration:.3f} seconds")