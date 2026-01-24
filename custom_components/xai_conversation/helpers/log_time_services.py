"""LogTimeServices - A centralized context manager for timing and logging service calls.

Provide automatic start/end logging and time tracking (API vs Local) for
consistency across services.
"""

from __future__ import annotations

import time
import logging
from typing import AsyncGenerator, Any
import contextlib


class LogTimeServices:
    """A context manager to uniformly log and time service calls."""

    def __init__(
        self,
        logger: logging.Logger,
        service_name: str,
        context_info: dict[str, Any] | None = None,
    ):
        """
        Initialize the timer.

        Args:
            logger: The logger instance to use for output.
            service_name: The name of the service being timed (e.g., "image_generation").
            context_info: A dictionary of extra context to include in log messages.
        """
        self.logger = logger
        self.service_name = service_name
        self.context_info = context_info or {}
        self.start_time: float | None = None
        self.api_time: float = 0.0
        self.total_time: float = 0.0
        self.local_process_time: float = 0.0
        self.latency: float = 0.0  # Time to First Token (TTFT) or First Response
        self.ttfb: float = 0.0  # Time to First Byte (Metadata)
        self._first_chunk_received: bool = False
        self._response_started: bool = False

    async def __aenter__(self) -> LogTimeServices:
        """Enter the context, log the start time and initial context."""
        self.start_time = time.time()
        log_context = " ".join(f"{k}={v}" for k, v in self.context_info.items())
        self.logger.info(
            f"chat_start: service={self.service_name} {log_context}".strip()
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context, calculate final times, and log the end result."""
        if self.start_time is None:
            return  # Should not happen in normal use

        self.total_time = time.time() - self.start_time
        self.local_process_time = self.total_time - self.api_time

        # Calculate non-overlapping components for "human math"
        # wait (TTFB) + think (TTFT-TTFB) + gen (TotalAPI-TTFT) + local (Total-TotalAPI) = Total
        wait = self.ttfb
        think = max(0.0, self.latency - self.ttfb)
        gen = max(0.0, self.api_time - self.latency)
        local = self.local_process_time

        # Prepare context for the final log, excluding any sensitive or overly verbose data
        log_context = self.context_info.copy()

        log_context_str = " ".join(
            f"{k}={v}"
            for k, v in {
                **log_context,
                "total_time": f"{self.total_time:.2f}s",
                "generation_time": f"{gen:.2f}s",
                "think_time": f"{think:.2f}s",
                "wait_time": f"{wait:.2f}s",
                "local_time": f"{local:.2f}s",
            }.items()
        )

        if exc_type:
            self.logger.error(
                f"chat_error: service={self.service_name} error_type={exc_type.__name__} {log_context_str}"
            )
        else:
            self.logger.info(f"chat_end: service={self.service_name} {log_context_str}")

    def record_api_time(self, duration: float):
        """Add a specific duration to the total API time."""
        if not self._response_started and duration > 0:
            # TTFB is the wait for the very first response from API
            self.ttfb = self.api_time + duration
            self._response_started = True
            # For non-streaming calls, latency (response time) is default if not set by content logic
            if not self._first_chunk_received:
                self.latency = self.ttfb
        self.api_time += duration

    @contextlib.asynccontextmanager
    async def record_api_call(self) -> AsyncGenerator[None, None]:
        """A nested context manager to time a block of code as an API call."""
        start = time.time()
        try:
            yield
        finally:
            end = time.time()
            duration = end - start
            if not self._response_started and duration > 0:
                self.ttfb = self.api_time + duration
                self._response_started = True
                if not self._first_chunk_received:
                    self.latency = self.ttfb
            self.api_time += duration


async def timed_stream_generator(
    stream_iterator: AsyncGenerator[Any, None], timer: LogTimeServices
) -> AsyncGenerator[Any, None]:
    """
    Asynchronously iterates a stream and records ONLY the execution time of the iterator
    (API latency) to a LogTimeServices instance.

    This implementation strictly separates API waiting time from consumer processing time.

    Args:
        stream_iterator: The async generator (API stream) to iterate over.
        timer: The LogTimeServices instance to report the API time to.
    """
    iterator = stream_iterator.__aiter__()
    while True:
        start_wait = time.time()
        try:
            chunk = await iterator.__anext__()
        except StopAsyncIteration:
            # Record time spent waiting for the final signal that stream is done
            timer.record_api_time(time.time() - start_wait)
            break
        except Exception:
            # Record time spent waiting before the error occurred
            timer.record_api_time(time.time() - start_wait)
            raise

        # Record time spent waiting for this chunk
        chunk_wait = time.time() - start_wait
        timer.record_api_time(chunk_wait)

        # First contentful chunk marks the latency (TTFT)
        if not timer._first_chunk_received:
            # We check if the chunk actually contains content to be precise about 'Time to First TOKEN'
            # (ignoring purely metadata chunks if they arrive earlier)
            has_content = False
            if hasattr(chunk, "content") and chunk.content:
                has_content = True
            elif isinstance(chunk, dict) and chunk.get("content"):
                has_content = True

            if has_content:
                # Latency is the total api_time accumulated until the first contentful chunk
                timer.latency = timer.api_time
                timer._first_chunk_received = True

        yield chunk
