# AIVoxPlay


**AI-powered real-time voice interaction framework for building conversational audio agents.**

**aivoxplay** is a modular Python framework for creating real-time, voice-enabled AI applications. It supports streaming Speech-to-Text (STT), and responsive Text-to-Speech (TTS), and Speech to Speech(STS) connected via WebSockets for low-latency performance.

TTS: Only Orpheus 3B is currenly supported. 

---

## 🚀 Getting Started (Development)

**AIVoxPlay** is currently intended for experimentation and rapid prototyping. It started as a side project and remains one—a way to keep up with the fast-moving voice AI ecosystem. Expect breaking changes and evolving APIs.

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/AIVoxPlay.git
cd AIVoxPlay/aivoxplay
pip install -r requirements.txt
```

### 2. Requirements

- **OpenAI Realtime API** for Speech-to-Text (STT)
- **Orpheus 3B** endpoint for Text-to-Speech (TTS) - OpenAI-style VLLM endpoint is required.
- Python 3.8+

### 3. Environment Setup

Set your OpenAI API key in a `.env` file or your environment:

```env
OPENAI_API_KEY=sk-...
```

### 4. Running the Example Agent Server

```bash
cd examples
python agent_server.py
```

This will start a FastAPI server on `localhost:8001` exposing a `/voice` WebSocket endpoint.

### 5. Try the Web Client

Open `examples/sts_client.html` in your browser. This client:

- Streams microphone audio to the server via WebSocket.
- Receives real-time transcripts and synthesized audio responses.
- Handles barge-in (interrupting TTS with new speech).
- Shows a transcript and playback indicator.

### 6. How It Works

- **/voice WebSocket API**:  
  - Client sends audio chunks (PCM, base64) to the server.
  - Server transcribes audio using OpenAI STT, sends text to a chat function (can be any LLM/agent), then streams TTS audio back.
  - Supports "cancel_audio" for barge-in.

- **External Chat Function**:  
  - You can plug in any function (OpenAI, VLLM, LangChain, etc.) to generate responses.
  - See `examples/agent_server.py` for a minimal example.

### 7. Extensibility & Roadmap

- **Current**: Only OpenAI Realtime STT and Orpheus 3B TTS are supported.
- **Planned**:  
  - More TTS and STT providers.
  - Better chat orchestration (e.g., [LangGraph](https://github.com/langchain-ai/langgraph)).
  - Improved client UX and orchestration patterns.

---

## ⚠️ Disclaimer

This package is for experimentation and learning. It is not production-ready. The API and features will change rapidly as the voice AI landscape evolves.

---

## 🛠️ API Overview

- **/voice** (WebSocket):  
  - `input_audio_buffer.append`: Send base64-encoded PCM audio.
  - `cancel_audio`: Interrupt current TTS playback.
  - Receives:
    - `transcript`: Recognized text.
    - Binary audio chunks (PCM) for playback.
    - `audio.complete`/`audio.cancelled`: Playback status.

See `examples/sts_client.html` for a reference client implementation.

---

## 🤝 Contributing

PRs and issues are welcome! This project is a playground for new ideas in real-time voice AI.
