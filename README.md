# AIVoxPlay

```bash
docker run --rm --detach --publish 8000:8000 --name speaches \
  -e SPEACHES_TRANSCRIPTION_HOST=localhost \
  -e SPEACHES_TRANSCRIPTION_PORT=8000 \
  -e OPENAI_API_KEY=dummy_key \
  -e TRANSCRIPTION__HOST=localhost \
  -e TRANSCRIPTION__PORT=8000 \
  -e LOOPBACK_HOST_URL=http://127.0.0.1:8000 \
  -e RESPONSE_ENABLED=false \
  -e RESPONSE__ENABLED=false \
  -e TRANSCRIPTION__SCHEME=https \
  --volume hf-hub-cache:/home/ubuntu/.cache/huggingface/hub \
  --volume /Users/ram/Documents/AIVoxPlay/aivoxplay/src/aivoxplay/stt/faster_whisper_model_aliases.json:/home/ubuntu/speaches/model_aliases.json \
  ghcr.io/speaches-ai/speaches:latest-cpu


```