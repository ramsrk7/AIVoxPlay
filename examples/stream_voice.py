from flask import Flask, request, Response
import io
import soundfile as sf
import numpy as np

app = Flask(__name__)

@app.route('/tts', methods=['GET'])
def tts():
    prompt = request.args.get('prompt')
    if not prompt:
        return "Prompt is required", 400

    # Stream the audio data
    def generate():
        # Use the existing code to generate audio
        # For simplicity, let's assume `generate_audio` is a function that yields audio data
        for audio_chunk in generate_audio(prompt):
            # Convert the audio chunk to bytes
            audio_io = io.BytesIO()
            sf.write(audio_io, audio_chunk, 24000, format='WAV')
            audio_io.seek(0)
            yield audio_io.read()

    return Response(generate(), mimetype='audio/wav')

def generate_audio(prompt):
    # This function should implement the logic from your existing script
    # to generate audio from the prompt
    # For now, let's yield dummy audio chunks
    for _ in range(10):  # Simulate 10 chunks
        yield np.zeros(2400)  # 0.1 second of silence per chunk

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)