"""Sliding-window rate limiter to stay under EDGAR's 10 req/s fair-access limit."""

from __future__ import annotations

import threading
import time
from collections import deque


class SlidingWindowRateLimiter:
    """Thread-safe sliding-window rate limiter.

    Callers invoke ``acquire()`` (or call the instance) before each request.
    If the configured rate would be exceeded the call blocks until the window
    clears.

    Args:
        max_calls: Maximum requests allowed in the window (default 8 to give
            a safety margin below EDGAR's 10 req/s limit).
        period: Window duration in seconds (default 1.0).
    """

    def __init__(self, max_calls: int = 8, period: float = 1.0) -> None:
        self.max_calls = max_calls
        self.period = period
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a request slot is available, then claim it."""
        with self._lock:
            now = time.monotonic()

            # Evict timestamps that have fallen outside the window.
            while self._timestamps and self._timestamps[0] <= now - self.period:
                self._timestamps.popleft()

            if len(self._timestamps) >= self.max_calls:
                # Sleep until the oldest timestamp exits the window.
                sleep_for = self.period - (now - self._timestamps[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)
                # Re-evict after sleeping.
                now = time.monotonic()
                while self._timestamps and self._timestamps[0] <= now - self.period:
                    self._timestamps.popleft()

            self._timestamps.append(time.monotonic())

    # Allow instances to be used as plain callables: rate_limiter()
    def __call__(self) -> None:
        self.acquire()
