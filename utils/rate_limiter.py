"""
Utilities for API rate limiting and asynchronous operation.
"""

import time
import asyncio
import logging

log = logging.getLogger(__name__)

class RateLimiter:
    """Synchronous rate limiter to control API call frequency."""
    def __init__(self, calls_per_minute: int):
        if calls_per_minute <= 0:
            raise ValueError("calls_per_minute must be positive")
        self.calls_per_minute = calls_per_minute
        self.interval = 60.0 / calls_per_minute
        self.last_call_time = 0

    def wait_if_needed(self):
        """Waits if the time since the last call is less than the allowed interval."""
        now = time.monotonic() # Use monotonic clock for interval measurement
        time_since_last_call = now - self.last_call_time

        if time_since_last_call < self.interval:
            wait_time = self.interval - time_since_last_call
            log.debug(f"Rate limiting: waiting for {wait_time:.2f} seconds")
            time.sleep(wait_time)
            self.last_call_time = time.monotonic() # Update after waiting
        else:
            self.last_call_time = now # Update immediately if no wait needed

class AsyncRateLimiter:
    """Asynchronous rate limiter to control API call frequency without blocking the event loop."""
    def __init__(self, calls_per_minute: int):
        if calls_per_minute <= 0:
            raise ValueError("calls_per_minute must be positive")
        self.calls_per_minute = calls_per_minute
        self.interval = 60.0 / calls_per_minute
        self.last_call_time = 0

    async def wait_if_needed(self):
        """Asynchronously waits if the time since the last call is less than the allowed interval."""
        now = time.monotonic() # Use monotonic clock for interval measurement
        time_since_last_call = now - self.last_call_time

        if time_since_last_call < self.interval:
            wait_time = self.interval - time_since_last_call
            log.debug(f"Async rate limiting: waiting for {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
            self.last_call_time = time.monotonic() # Update after waiting
        else:
            self.last_call_time = now # Update immediately if no wait needed