
## 🖥️ Example: Setting Up the Server

To run the AIVoxPlay server and enable real-time voice interaction, follow these steps:

### 1. Install Requirements

Make sure you have installed all dependencies:

```bash
cd AIVoxPlay/aivoxplay
pip install -r requirements.txt
```

### 2. Set Your Environment Variables

Set your OpenAI API key in a `.env` file or as an environment variable:

```env
OPENAI_API_KEY=sk-...
```

### 3. Start the Example Agent Server

Navigate to the `examples` directory and run the server:

```bash
cd ../examples
python agent_server.py
```

This will start a FastAPI server on `localhost:8000` exposing the following WebSocket endpoints:

- `/audio/in/{client_id}`: Receives audio from the client for transcription and agent processing.
- `/audio/out/{client_id}`: Sends back transcripts and synthesized audio to the client.

### 4. (Optional) CORS Configuration

The server is configured to allow all origins by default for development:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your client’s origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

You can restrict this to your client’s URL for better security.

### 5. Try the Example Client

Open `examples/sts_client.html` in your browser to test the full pipeline.

---

**Note:**  
You can customize the agent’s behavior by editing the `external_chat_fn` in `agent_server.py` to connect to any LLM or business logic you prefer.

---

Would you like this inserted at a specific place in your README, or as a new section after "Getting Started"?


## 🧑‍💻 Building Your Own Web Client (`client.html`)

To build a custom web client that interacts with the AIVoxPlay server, you’ll need to connect to two WebSocket endpoints:

- `/audio/in/{client_id}`: For sending microphone audio to the server.
- `/audio/out/{client_id}`: For receiving transcripts and synthesized audio from the server.

Here are some recommendations and a high-level outline:

### 1. Generate a Unique Client ID

Each session should use a unique `client_id` (e.g., via `crypto.randomUUID()` in JavaScript) to pair the input and output streams.

```js
const CLIENT_ID = crypto.randomUUID();
```

### 2. Open Two WebSocket Connections

- One for sending audio (`/audio/in/{client_id}`)
- One for receiving responses (`/audio/out/{client_id}`)

```js
const wsIn = new WebSocket(`ws://localhost:8000/audio/in/${CLIENT_ID}`);
const wsOut = new WebSocket(`ws://localhost:8000/audio/out/${CLIENT_ID}`);
```

### 3. Capture and Stream Microphone Audio

- Use the Web Audio API (`getUserMedia`, `AudioContext`) to capture audio.
- Convert audio to 16-bit PCM, encode as base64, and send via `wsIn` using the message type `input_audio_buffer.append`.

```js
wsIn.send(JSON.stringify({
  type: "input_audio_buffer.append",
  audio: btoa(String.fromCharCode(...new Uint8Array(pcm16.buffer)))
}));
```

### 4. Handle Barge-In (Interrupt TTS)

- If the user starts speaking while TTS is playing, send a `cancel_audio` message to interrupt playback.

```js
wsIn.send(JSON.stringify({ type: "cancel_audio" }));
```

### 5. Receive and Play Audio Responses

- Listen for binary audio data on `wsOut` and play it using the Web Audio API.
- Handle transcript messages (`type: "transcript"`) to update the UI.
- Handle playback status messages (`audio.complete`, `audio.cancelled`) to manage UI state.

### 6. UI Recommendations

- Show a visual indicator for listening/playing states.
- Display the transcript history.
- Provide Start/Stop controls for the session.

### 7. See Example

Refer to [`examples/sts_client.html`](examples/sts_client.html) for a complete, ready-to-use implementation.

---

**Sample Workflow:**

1. User clicks "Start" → open both WebSockets, start capturing audio.
2. Audio is streamed to `/audio/in/{client_id}`.
3. Server transcribes and responds via `/audio/out/{client_id}`:
    - Text transcript (JSON)
    - Audio (binary PCM)
4. Client plays audio and updates transcript.
5. User can interrupt TTS by speaking (barge-in), sending `cancel_audio`.
6. User clicks "Stop" → close connections and clean up.

---

**Tip:**  
You can adapt the example client to your needs—add authentication, change the UI, or integrate with other services as required.
