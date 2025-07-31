# src/my_voice_server/tts.py
import struct
from aivoxplay.tts.orpheus import OrpheusTTS
from aivoxplay.utils.tts import QueueSession

class TTSEngine:
    def __init__(self, tts):
        self._tts = tts
        self._qsession = QueueSession(self._tts)

    def build_wav_header(self, sr=24000, bits=16, ch=1):
        byte_rate   = sr * ch * bits // 8
        block_align = ch * bits // 8
        return struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF', 36, b'WAVE', b'fmt ',16,1,ch,
            sr, byte_rate, block_align, bits, b'data', 0
        )
    def build_full_wav(self, chunks: list[bytes], sr=24000, bits=16, ch=1):
        pcm_data = b''.join(chunks)
        data_len = len(pcm_data)
        header = build_wav_header(sr=sr, bits=bits, ch=ch, data_len=data_len)
        return header + pcm_data

    async def synth_stream(self, text: str):
        """
        Returns an async generator of binary chunks (including header).
        """
        # queue & play
        self._qsession.add(text)
        ctl = self._qsession.play_and_rotate()
        ctl.close()
        # header
        yield self.build_wav_header()
        # then chunks
        for chunk in ctl.stream():
            yield chunk
    
    def cancel(self):
        """
        Returns an async generator of binary chunks (including header).
        """
        # queue & play
        return self._qsession.cancel()