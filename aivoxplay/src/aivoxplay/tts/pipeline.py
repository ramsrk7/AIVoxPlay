import threading
import queue
from collections import deque
from typing import Callable, Iterable, Optional

# Unique sentinels
_END = object()
_CANCEL = object()

class ParagraphStreamController:
    """
    Generic ordered streaming controller.

    speak_tokens_fn: Callable[[str, str], Iterable]  -> returns a token generator for given text+voice
    decode_tokens_fn: Callable[[Iterable], Iterable[bytes]] -> yields audio bytes from token gen
    """
    def __init__(
        self,
        speak_tokens_fn: Callable[[str, str], Iterable],
        decode_tokens_fn: Callable[[Iterable], Iterable[bytes]],
        voice: str,
        max_workers: int = 2,
        per_item_queue_maxsize: int = 0,
    ):
        self._speak_tokens_fn = speak_tokens_fn
        self._decode_tokens_fn = decode_tokens_fn
        self._voice = voice

        self._sem = threading.BoundedSemaphore(max_workers)
        self._jobs = deque()                       # pending jobs in order
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

        self._generation = 0                      # increases on clear()
        self._closed = False
        self._next_id = 0
        self._current_job = None                  # job currently being consumed by stream()

        self._per_item_queue_maxsize = per_item_queue_maxsize

    class _Job:
        __slots__ = ("id", "gen", "text", "out_q", "done", "cancel", "thread")
        def __init__(self, jid, gen, text, out_q):
            self.id = jid
            self.gen = gen
            self.text = text
            self.out_q: queue.Queue = out_q
            self.done = threading.Event()
            self.cancel = threading.Event()
            self.thread: Optional[threading.Thread] = None

    # ------------------------------ Public API ------------------------------

    def add(self, text: str) -> int:
        """Add a paragraph/sentence to be processed. Returns job id."""
        with self._cv:
            if self._closed:
                raise RuntimeError("Stream is closed")

            jid = self._next_id
            self._next_id += 1
            job = self._Job(
                jid,
                self._generation,
                text,
                queue.Queue(maxsize=self._per_item_queue_maxsize),
            )
            self._jobs.append(job)

            # Start worker immediately (non-blocking) with concurrency guard
            job.thread = threading.Thread(target=self._run_job, args=(job,), daemon=True)
            job.thread.start()

            self._cv.notify_all()
            return jid

    def clear(self) -> None:
        """
        Cancel everything (including the job currently streaming).
        Any in-flight workers will be signaled to stop ASAP.
        """
        with self._cv:
            self._generation += 1
            # cancel pending jobs
            for job in self._jobs:
                job.cancel.set()
                # unblock stream() if it happens to be waiting on this queue
                try:
                    job.out_q.put_nowait(_CANCEL)
                except queue.Full:
                    pass
            self._jobs.clear()

            # cancel current job (if stream() is consuming one)
            if self._current_job is not None:
                self._current_job.cancel.set()
                try:
                    self._current_job.out_q.put_nowait(_CANCEL)
                except queue.Full:
                    pass

            self._cv.notify_all()

    def close(self) -> None:
        """Stop accepting new items; stream() will finish after current items (if not cleared)."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def stream(self):
        """
        Generator that yields audio bytes in FIFO order of added items.
        Safe to call once; add() can be called concurrently from other threads.
        """
        while True:
            with self._cv:
                # Wait for a job or closure
                while not self._jobs and not self._closed:
                    self._cv.wait()

                if not self._jobs and self._closed:
                    break

                job = self._jobs.popleft()
                self._current_job = job

            # Consume this job's output queue in order
            try:
                while True:
                    chunk = job.out_q.get()
                    if chunk is _END or chunk is _CANCEL or job.cancel.is_set():
                        break
                    yield chunk
            finally:
                # Mark job done; allow GC
                job.done.set()
                with self._cv:
                    # If a clear happened during consumption, _current_job may already be canceled.
                    self._current_job = None

    # ------------------------------ Internals ------------------------------

    def _run_job(self, job: "_Job") -> None:
        try:
            with self._sem:
                token_gen = self._speak_tokens_fn(job.text, self._voice)
                for b in self._decode_tokens_fn(token_gen):
                    if job.cancel.is_set():
                        break
                    job.out_q.put(b)
        except Exception as e:
            # <-- DO NOT swallow. At least print/log it.
            import traceback, sys
            print(f"[PSC] job {job.id} crashed: {e}", file=sys.stderr)
            traceback.print_exc()
        finally:
            try:
                job.out_q.put(_END)
            except Exception:
                pass
            job.done.set()

