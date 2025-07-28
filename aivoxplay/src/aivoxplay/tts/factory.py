# Adapted from: https://github.com/canopyai/Orpheus-TTS
# Originally licensed under the Apache License, Version 2.0
# Modifications may have been made by Ramkumar. See Git history for details.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from collections import deque
from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import queue
import os
import queue as _q
import inspect

_decode_lock = threading.Lock()

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
# Respect the environment variable if set; otherwise fallback to mps -> cuda -> cpu
snac_device = torch.device(
    os.environ.get("SNAC_DEVICE") or
    ("mps" if torch.backends.mps.is_available() else
     "cuda" if torch.cuda.is_available() else
     "cpu")
)

model = model.to(snac_device)

# ---- one shared event loop for all async decoding ----
class _AsyncLoopRunner:
    _loop = None
    _thread = None

    @classmethod
    def start(cls):
        if cls._loop is None:
            import asyncio, threading
            cls._loop = asyncio.new_event_loop()
            cls._thread = threading.Thread(
                target=cls._loop.run_forever, name="audio-decode-loop", daemon=True
            )
            cls._thread.start()

    @classmethod
    def submit(cls, coro):
        cls.start()
        import asyncio
        # schedule coroutine on the shared loop from any thread
        return asyncio.run_coroutine_threadsafe(coro, cls._loop)

# Optional: lock to serialize non-threadsafe model calls if needed
_decode_lock = threading.Lock()

def _ensure_pcm16le_bytes(chunk, expect_sr=24000):
    import numpy as np
    if isinstance(chunk, (bytes, bytearray)):
        return bytes(chunk)
    if isinstance(chunk, np.ndarray):
        # Accept float32/float64 or int16 arrays and convert to little-endian int16 bytes
        if chunk.dtype.kind == 'f':
            chunk = np.clip(chunk, -1.0, 1.0)
            chunk = (chunk * 32767.0).astype('<i2', copy=False)
        elif chunk.dtype != np.dtype('<i2'):
            chunk = chunk.astype('<i2', copy=False)
        return chunk.tobytes()
    raise TypeError(f"Unsupported audio chunk type: {type(chunk)}")


class TokenStreamParser:
    def __init__(self):
        self.buffer = ""  # For holding leftover strings
        self.token_pattern = re.compile(r"<custom_token_\d+>")
        self.token_queue = deque()

    def feed(self, chunk: str):
        """Feed a new chunk and extract complete tokens."""
        self.buffer += chunk  # Append new chunk to the buffer
        matches = list(self.token_pattern.finditer(self.buffer))

        last_index = 0
        for match in matches:
            self.token_queue.append(match.group())
            last_index = match.end()

        # Keep the unprocessed part (partial token) in buffer
        self.buffer = self.buffer[last_index:]

    def get_tokens(self):
        """Retrieve all tokens parsed so far."""
        tokens = list(self.token_queue)
        self.token_queue.clear()
        return tokens

class OrpheusAudioProcessor:

    @staticmethod
    def convert_to_audio(multiframe, count):
        frames = []
        if len(multiframe) < 7:
            return
        
        codes_0 = torch.tensor([], device=snac_device, dtype=torch.int32)
        codes_1 = torch.tensor([], device=snac_device, dtype=torch.int32)
        codes_2 = torch.tensor([], device=snac_device, dtype=torch.int32)

        num_frames = len(multiframe) // 7
        frame = multiframe[:num_frames*7]

        for j in range(num_frames):
            i = 7*j
            if codes_0.shape[0] == 0:
                codes_0 = torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)
            else:
                codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)])

            if codes_1.shape[0] == 0:
            
                codes_1 = torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])
            else:
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)])
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])
            
            if codes_2.shape[0] == 0:
                codes_2 = torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])
            else:
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])

        codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
        # check that all tokens are between 0 and 4096 otherwise return *
        if torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or torch.any(codes[2] < 0) or torch.any(codes[2] > 4096):
            return

        with torch.inference_mode():
            audio_hat = model.decode(codes)
        
        audio_slice = audio_hat[:, :, 2048:4096]
        detached_audio = audio_slice.detach().cpu()
        audio_np = detached_audio.numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        return audio_bytes

    @staticmethod
    def turn_token_into_id(token_string, index):
        # Strip whitespace
        token_string = token_string.strip()
        
        # Find the last token in the string
        last_token_start = token_string.rfind("<custom_token_")
        
        if last_token_start == -1:
            return None
        
        # Extract the last token
        last_token = token_string[last_token_start:]
        
        # Process the last token
        if last_token.startswith("<custom_token_") and last_token.endswith(">"):
            try:
                number_str = last_token[14:-1]
                return int(number_str) - 10 - ((index % 7) * 4096)
            except ValueError:
                return None
        else:
            return None
    
    @staticmethod    
    async def tokens_decoder(token_gen):
        buffer = []
        count = 0
        async for token_sim in token_gen:       
            token = OrpheusAudioProcessor.turn_token_into_id(token_sim, count)
            if token is None:
                pass
            else:
                if token > 0:
                    buffer.append(token)
                    count += 1

                    if count % 7 == 0 and count > 27:
                        buffer_to_proc = buffer[-28:]
                        audio_samples = OrpheusAudioProcessor.convert_to_audio(buffer_to_proc, count)
                        if audio_samples is not None:
                            yield audio_samples


    # ------------------ Synchronous Tokens Decoder Wrapper ------------------ #
    @staticmethod
    def tokens_decoder_sync(speak_result):
        """
        Accepts:
        - a SYNC token generator
        - an ASYNC token generator
        - a coroutine that returns either of the above
        and yields PCM16LE bytes.
        """
        audio_queue = _q.Queue()

        async def _to_async_gen(sr):
            # Normalize anything -> async generator of tokens
            if inspect.isasyncgen(sr):
                async for t in sr:
                    yield t
            elif inspect.iscoroutine(sr):
                r = await sr
                async for t in _to_async_gen(r):
                    yield t
            else:
                # assume sync iterable
                for t in sr:
                    yield t

        async def async_producer():
            async for audio_chunk in OrpheusAudioProcessor.tokens_decoder(_to_async_gen(speak_result)):
                # If decode isn’t thread-safe, uncomment:
                # with _decode_lock:
                pcm = _ensure_pcm16le_bytes(audio_chunk)
                audio_queue.put(pcm)
            audio_queue.put(None)  # sentinel

        fut = _AsyncLoopRunner.submit(async_producer())

        while True:
            try:
                audio = audio_queue.get(timeout=0.2)  # periodically check fut
            except _q.Empty:
                if fut.done():
                    # Raises if producer crashed; prevents silent hang
                    fut.result()
                continue

            if audio is None:
                break
            yield audio

        # Propagate any late exceptions
        fut.result()



def dummy_stream_custom_tokens_without_parser(full_token_string: str):
    buffer = ""
    pos = 0

    while pos < len(full_token_string):
        # Simulate random chunk size
        chunk_size = random.randint(5, 30)
        chunk = full_token_string[pos:pos + chunk_size]
        buffer += chunk
        pos += chunk_size

        tokens = []
        while True:
            start = buffer.find("<custom_token_")
            end = buffer.find(">", start)

            # No complete token found yet
            if start == -1 or end == -1:
                break

            # Extract the token
            token = buffer[start:end + 1]
            tokens.append(token)

            # Remove the processed part
            buffer = buffer[end + 1:]

        # If no complete token found, yield partial chunk as list of characters
        if not tokens and chunk:
            yield ''.join(chunk)   # Yield raw characters for partial token accumulation
        elif tokens:
            yield ''.join(chunk)   # Yield parsed tokens as list

        time.sleep(0.05)  # Simulate delay
def pcm16le_normalizer(chunk):
    if isinstance(chunk, (bytes, bytearray)):
        return bytes(chunk)
    if isinstance(chunk, np.ndarray):
        if chunk.dtype.kind == 'f':
            chunk = np.clip(chunk, -1.0, 1.0)
            chunk = (chunk * 32767.0).astype('<i2', copy=False)
        elif chunk.dtype != np.dtype('<i2'):
            chunk = chunk.astype('<i2', copy=False)
        return chunk.tobytes()
    return bytes(chunk)