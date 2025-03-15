"""Decorator for stochastic testing."""
from collections.abc import Callable
import pytest

def stochastic(
    samples: int = 10,
    threshold: float = 0.5,
    batch_size: int | None = None,
    retry_on: list[type[Exception]] | None = None,
    max_retries: int = 3,
    timeout: float | None = None,
) -> Callable:
    """
    Mark a test to be executed stochastically multiple times.

    Args:
        samples: Number of times to run the test
        threshold: Minimum required success rate (0.0 to 1.0)
        batch_size: Number of tests to run concurrently (None = all at once)
        retry_on: List of exception types that should trigger retries
        max_retries: Maximum number of retry attempts per sample
        timeout: Maximum seconds per sample (None = no timeout)
    """
    return pytest.mark.stochastic(
        samples=samples,
        threshold=threshold,
        batch_size=batch_size,
        retry_on=retry_on,
        max_retries=max_retries,
        timeout=timeout,
    )
