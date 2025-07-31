# src/streaming_client/core.py
import abc
from typing import AsyncIterator, Any

class StreamError(Exception):
    """Base exception for streaming client errors."""

class StreamingProvider(abc.ABC):
    """
    Abstract interface for any streaming STT-like provider.
    """

    @abc.abstractmethod
    async def connect(self) -> None:
        """Establish websocket (or other) connection."""

    @abc.abstractmethod
    async def send_audio_chunk(self, chunk: bytes) -> None:
        """Send one PCM audio chunk to the provider."""

    @abc.abstractmethod
    async def receive_messages(self) -> AsyncIterator[Any]:
        """
        Async generator yielding provider messages (dicts or raw bytes).
        """
        yield

    @abc.abstractmethod
    async def close(self) -> None:
        """Cleanly tear down connection."""
