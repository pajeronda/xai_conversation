"""LogTimeServices - A centralized context manager for timing and logging service calls.

This module provides the LogTimeServices class, which is designed to be used
as an asynchronous context manager (`async with`). It automatically handles
the start and end logging for any service call, and provides mechanisms to
record the time spent in API calls vs. local processing.
"""

from __future__ import annotations

import time
import logging
from typing import TYPE_CHECKING, AsyncGenerator, Any
import contextlib

if TYPE_CHECKING:
    from ..const import LOGGER


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

        # Prepare context for the final log, excluding any sensitive or overly verbose data
        log_context = self.context_info.copy()

        log_context_str = " ".join(
            f"{k}={v}"
            for k, v in {
                **log_context,
                "duration": f"{self.total_time:.2f}s",
                "api_time": f"{self.api_time:.2f}s",
                "local_process_time": f"{self.local_process_time:.2f}s",
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
        self.api_time += duration

    @contextlib.asynccontextmanager
    async def record_api_call(self) -> AsyncGenerator[None, None]:
        """A nested context manager to time a block of code as an API call."""
        start = time.time()
        try:
            yield
        finally:
            end = time.time()
            self.api_time += end - start


async def timed_stream_generator(
    stream_iterator: AsyncGenerator[Any, None], timer: LogTimeServices
) -> AsyncGenerator[Any, None]:
    """
    Asynchronously iterates a stream and records the execution time to a LogTimeServices instance.

    Args:
        stream_iterator: The async generator (API stream) to iterate over.
        timer: The LogTimeServices instance to report the API time to.
    """
    start_time = time.time()
    try:
        async for chunk in stream_iterator:
            yield chunk
    finally:
        end_time = time.time()
        timer.record_api_time(end_time - start_time)
