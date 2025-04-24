"""
Async Rate Limiter Utility

This module provides an async-compatible rate limiter to help respect API rate limits.
"""

import asyncio
import time
from collections import deque
import logging
from typing import Deque, Optional

log = logging.getLogger(__name__)

class AsyncRateLimiter:
    def __init__(self, calls_per_minute: int):
        """Initialize rate limiter with calls per minute limit."""
        self.calls_per_minute = max(1, calls_per_minute)
        self.time_between_calls = 60.0 / self.calls_per_minute
        self.call_times: Deque[float] = deque(maxlen=calls_per_minute)
        self._lock = asyncio.Lock()

    async def wait_if_needed(self) -> None:
        """Wait if necessary to respect the rate limit."""
        async with self._lock:
            current_time = time.monotonic()
            
            # Remove old timestamps
            while self.call_times and current_time - self.call_times[0] > 60:
                self.call_times.popleft()
            
            if len(self.call_times) >= self.calls_per_minute:
                # Calculate required wait time
                wait_time = 60 - (current_time - self.call_times[0])
                if wait_time > 0:
                    log.debug(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
            
            # Add current call time
            self.call_times.append(time.monotonic())

    async def __aenter__(self):
        """Context manager support."""
        await self.wait_if_needed()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        pass