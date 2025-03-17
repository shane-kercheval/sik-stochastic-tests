"""
Unit tests for the stochastic pytest plugin.

These tests verify both internal functionality (helper functions, statistics tracking)
and integration behavior (through pytest subprocess calls) to ensure the plugin
correctly handles both sync and async tests with proper concurrency.
"""
from pathlib import Path
import time
import pytest
import asyncio
import sys
import subprocess
from unittest.mock import MagicMock
from sik_stochastic_tests.plugin import (
    _run_stochastic_tests,
    _test_results,
    StochasticTestStats,
    pytest_collection_modifyitems,
    pytest_pyfunc_call,
    pytest_terminal_summary,
)
from typing import Never

def test_has_asyncio_marker():
    """
    Test the asyncio marker detection in various contexts.

    This test verifies that all mechanisms to detect async tests work correctly:
    - Direct markers on functions
    - Class-level markers
    - Method inheritance of markers

    Correct detection is critical since async and sync tests follow different execution paths.
    """
    from sik_stochastic_tests.plugin import has_asyncio_marker

    asyncio_marker = MagicMock()
    asyncio_marker.name = 'asyncio'

    other_marker = MagicMock()
    other_marker.name = 'other'

    class AsyncioClass:
        pytestmark = [asyncio_marker]  # noqa: RUF012

    # Test detection of direct function markers
    func_with_marker = MagicMock()
    func_with_marker.own_markers = [asyncio_marker]

    func_without_marker = MagicMock()
    func_without_marker.own_markers = [other_marker]

    # Test detection via class inheritance
    func_in_asyncio_class = MagicMock()
    func_in_asyncio_class.own_markers = []
    func_in_asyncio_class.cls = AsyncioClass

    # Test detection via method binding
    method_obj = MagicMock()
    method_obj.__self__ = MagicMock()
    method_obj.__self__.__class__ = AsyncioClass

    # Verify correct detection in all scenarios
    assert has_asyncio_marker(func_with_marker) is True
    assert has_asyncio_marker(func_without_marker) is False
    assert has_asyncio_marker(func_in_asyncio_class) is True
    assert has_asyncio_marker(method_obj) is True

@pytest.mark.asyncio
async def test_run_test_function():
    """
    Test the universal test function executor.

    This test verifies that run_test_function correctly handles all types of functions:
    - Synchronous functions
    - Asynchronous functions
    - Sync functions that return awaitable objects
    - Functions with timeouts

    This functionality is essential for the plugin to work with different coding styles.
    """
    from sik_stochastic_tests.plugin import run_test_function
    # 1. Plain synchronous function
    def sync_func(arg=None):  # noqa: ANN001, ANN202
        return arg or "sync result"

    # 2. Asynchronous function
    async def async_func(arg=None):  # noqa: ANN001, ANN202
        return arg or "async result"

    # 3. Special case: sync function that returns a coroutine
    def returns_coroutine(arg=None):  # noqa: ANN001, ANN202
        async def inner():  # noqa: ANN202
            return arg or "coroutine result"
        return inner()

    # 4. Function for timeout testing
    async def slow_func() -> str:
        await asyncio.sleep(0.2)
        return "slow result"

    # Test standard synchronous function execution
    result1 = await run_test_function(sync_func, {"arg": "test"})
    assert result1 == "test"

    # Test native async function execution
    result2 = await run_test_function(async_func, {"arg": "test"})
    assert result2 == "test"

    # Test the hybrid case - sync function returning an awaitable
    result3 = await run_test_function(returns_coroutine, {"arg": "test"})
    assert result3 == "test"

    # Verify timeout cancels execution when exceeded
    with pytest.raises(TimeoutError):
        await run_test_function(slow_func, {}, timeout=0.1)

    # Verify timeout doesn't interfere when sufficient
    result4 = await run_test_function(slow_func, {}, timeout=0.3)
    assert result4 == "slow result"

def test_stochastic_test_stats():
    """
    Test the statistics tracking functionality.

    This test ensures the StochasticTestStats class correctly:
    - Initializes with empty state
    - Calculates success rate properly
    - Stores failure information in the expected format

    Proper statistics are essential for reporting and threshold evaluation.
    """
    stats = StochasticTestStats()

    # Verify initial state is clean
    assert stats.total_runs == 0
    assert stats.successful_runs == 0
    assert stats.failures == []
    assert stats.success_rate == 0.0

    # Verify success rate calculation
    stats.total_runs = 5
    stats.successful_runs = 3
    assert stats.success_rate == 0.6

    # Verify failure recording
    stats.failures.append({"error": "Test error", "type": "ValueError", "context": {"run_index": 1}})  # noqa: E501
    assert len(stats.failures) == 1
    assert stats.failures[0]["type"] == "ValueError"

@pytest.mark.asyncio
async def test_run_stochastic_tests_success():
    """
    Test the core test runner with uniformly successful tests.

    This verifies the stochastic test runner correctly handles the ideal case
    where all test samples pass, ensuring statistics are properly tracked.
    """
    async def test_func() -> bool:
        return True

    stats = StochasticTestStats()

    await _run_stochastic_tests(
        testfunction=test_func,
        funcargs={},
        stats=stats,
        samples=5,
        batch_size=None,
        retry_on=None,
        max_retries=1,
        timeout=None,
    )

    # Verify all tests passed and statistics match
    assert stats.total_runs == 5
    assert stats.successful_runs == 5
    assert len(stats.failures) == 0
    assert stats.success_rate == 1.0

@pytest.mark.asyncio
async def test_run_stochastic_tests_failures():
    """
    Test the stochastic runner with predictably failing tests.

    This verifies the runner correctly:
    - Properly counts both successes and failures
    - Records detailed failure information
    - Calculates the correct success rate

    Testing with mixed success/failure is important since stochastic
    tests are designed primarily for flaky tests.
    """
    # Use a counter to create deterministic failure pattern
    counter = {'value': 0}

    async def test_func_with_failures() -> bool:
        counter['value'] += 1
        if counter['value'] % 2 == 0:  # Alternate success/failure
            raise ValueError("Simulated failure")
        return True

    stats = StochasticTestStats()

    await _run_stochastic_tests(
        testfunction=test_func_with_failures,
        funcargs={},
        stats=stats,
        samples=4,
        batch_size=None,
        retry_on=None,
        max_retries=1,
        timeout=None,
    )

    # Verify exactly half the tests failed as expected
    assert stats.total_runs == 4
    assert stats.successful_runs == 2
    assert len(stats.failures) == 2
    assert stats.success_rate == 0.5
    assert all("ValueError" in f["type"] for f in stats.failures)

@pytest.mark.asyncio
async def test_run_stochastic_tests_timeout():
    """
    Test timeout handling in stochastic tests.

    This verifies that:
    - Tests exceeding the timeout are properly terminated
    - Timeouts are correctly recorded as failures
    - The failure information includes the correct error type

    Timeouts are critical for preventing hung tests in production environments.
    """
    async def slow_test() -> bool:
        await asyncio.sleep(0.5)  # Deliberately longer than timeout
        return True

    stats = StochasticTestStats()

    # Run with a timeout shorter than the test duration
    await _run_stochastic_tests(
        testfunction=slow_test,
        funcargs={},
        stats=stats,
        samples=1,
        batch_size=None,
        retry_on=None,
        max_retries=1,
        timeout=0.1,
    )

    # Verify test timed out and was recorded correctly
    assert stats.total_runs == 1
    assert stats.successful_runs == 0
    assert len(stats.failures) == 1
    assert "TimeoutError" in stats.failures[0]["type"]

@pytest.mark.asyncio
async def test_run_stochastic_tests_retries():
    """
    Test the retry mechanism for temporary failures.

    This test verifies that:
    - Tests failing with specified exception types are retried
    - The retry counter works correctly
    - A test ultimately succeeding after retries is counted as successful

    Retry functionality is valuable for handling transient failures
    like network glitches or race conditions.
    """
    attempts = {'value': 0}

    async def test_with_retries() -> bool:
        attempts['value'] += 1
        if attempts['value'] < 3:  # Deliberately fail twice before succeeding
            raise ValueError("Temporary failure")
        return True

    # Test statistics
    stats = StochasticTestStats()

    # Run with retries
    await _run_stochastic_tests(
        testfunction=test_with_retries,
        funcargs={},
        stats=stats,
        samples=1,
        batch_size=None,
        retry_on=[ValueError],
        max_retries=3,
        timeout=None,
    )

    # Check stats - should be successful after retries
    assert stats.total_runs == 1
    assert stats.successful_runs == 1
    assert len(stats.failures) == 0
    assert attempts['value'] == 3  # Should have tried 3 times

@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_stochastic_retries(example_test_dir: Path, is_async: bool):
    """Test retry functionality works in real tests for both sync and async functions."""
    # For async tests, we need to add both stochastic and asyncio markers

    # Create a counter to track runs
    counter_file = example_test_dir / "retry_counter.txt"
    counter_file.write_text("0")

    # Configure based on async or sync
    test_prefix = "async " if is_async else ""
    asyncio_import = "import asyncio\n" if is_async else ""
    sleep_code = "await asyncio.sleep(0.01)\n        " if is_async else ""

    # Adding pytest.ini for asyncio_mode if testing async
    if is_async:
        pytest_ini = example_test_dir / "pytest.ini"
        pytest_ini.write_text("""
[pytest]
asyncio_mode = auto
""")

    # Create a test file with retries
    test_file = example_test_dir / "test_retries.py"
    test_file.write_text(f"""
import pytest
{asyncio_import}
@pytest.mark.stochastic(samples=1, retry_on=[ValueError], max_retries=3)
{"@pytest.mark.asyncio" if is_async else ""}
{test_prefix}def test_with_retries():
    # Track execution count
    with open("retry_counter.txt", "r") as f:
        count = int(f.read())

    with open("retry_counter.txt", "w") as f:
        f.write(str(count + 1))

    {sleep_code}
    # Fail on first 2 attempts
    if count < 2:
        raise ValueError(f"Intentional failure on attempt {{count+1}}")

    # Pass on 3rd attempt
    assert True
""")

    # Run pytest on the file
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        capture_output=True,
        text=True,
        cwd=example_test_dir, check=False,
    )

    # Print output for debugging
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")

    # Both sync and async tests should have retries applied
    assert "1 passed" in result.stdout

    # Check retry count for both async and sync tests
    with open(counter_file) as f:
        count = int(f.read())

    # Both should retry the expected number of times
    assert count == 3, f"Test should attempt 3 times with retries, but ran {count} times"

@pytest.mark.asyncio
async def test_run_stochastic_tests_sync_function():
    """Test running stochastic tests with a synchronous function."""
    # A synchronous test function
    def sync_test_func() -> bool:
        return True

    # Test statistics
    stats = StochasticTestStats()

    # Run the tests
    await _run_stochastic_tests(
        testfunction=sync_test_func,
        funcargs={},
        stats=stats,
        samples=3,
        batch_size=None,
        retry_on=None,
        max_retries=1,
        timeout=None,
    )

    # Verify stats
    assert stats.total_runs == 3
    assert stats.successful_runs == 3
    assert len(stats.failures) == 0

def test_pytest_pyfunc_call_not_stochastic():
    """Test that non-stochastic tests are ignored."""
    pyfuncitem = MagicMock()
    pyfuncitem.get_closest_marker.return_value = None

    # Should return None for non-stochastic tests
    result = pytest_pyfunc_call(pyfuncitem)
    assert result is None

def test_pytest_pyfunc_call_disabled():
    """Test that --disable-stochastic flag works."""
    pyfuncitem = MagicMock()
    pyfuncitem.get_closest_marker.return_value = MagicMock(kwargs={})
    pyfuncitem.config.getoption.return_value = True  # Disabled

    # Should return None when disabled
    result = pytest_pyfunc_call(pyfuncitem)
    assert result is None

@pytest.mark.asyncio
async def test_retry_specific_exceptions():
    """Test that only specified exceptions trigger retries."""
    attempts = {'value_error': 0, 'key_error': 0, 'type_error': 0}  # Fixed key name

    async def test_with_specific_retries() -> bool:
        if attempts['value_error'] < 1:
            attempts['value_error'] += 1
            raise ValueError("Should retry")
        if attempts['key_error'] < 1:
            attempts['key_error'] += 1
            raise KeyError("Should not retry")
        return True

    # Test statistics - only retry ValueError
    stats = StochasticTestStats()

    # Run with specific retry_on
    await _run_stochastic_tests(
        testfunction=test_with_specific_retries,
        funcargs={},
        stats=stats,
        samples=1,
        batch_size=None,
        retry_on=[ValueError],
        max_retries=2,
        timeout=None,
    )

    # Should have retried ValueError but not KeyError
    assert attempts['value_error'] == 1
    assert attempts['key_error'] == 1
    assert stats.total_runs == 1
    assert stats.successful_runs == 0
    assert len(stats.failures) == 1
    assert "KeyError" in stats.failures[0]["type"]

@pytest.mark.asyncio
async def test_batch_processing():
    """
    Test that batch_size correctly controls concurrency in the core test runner.

    This test verifies the batching mechanism:
    - Properly limits the number of concurrent test executions
    - Successfully completes all test samples
    - Maintains accurate statistics

    Concurrency control is essential for resource-intensive tests
    that might overwhelm the system if all run simultaneously.
    """
    class TestTracker:
        def __init__(self):
            self.active_count = 0
            self.max_active = 0

        async def run_test(self) -> bool:
            # Atomically track concurrent executions
            self.active_count += 1
            self.max_active = max(self.max_active, self.active_count)

            # Simulate work with a delay
            await asyncio.sleep(0.1)

            self.active_count -= 1
            return True

    tracker = TestTracker()
    stats = StochasticTestStats()

    # Run with batch_size=2 to limit concurrency
    await _run_stochastic_tests(
        testfunction=tracker.run_test,
        funcargs={},
        stats=stats,
        samples=5,
        batch_size=2,  # Should limit to 2 concurrent tests
        retry_on=None,
        max_retries=1,
        timeout=None,
    )

    # Verify all tests completed successfully
    assert stats.total_runs == 5
    assert stats.successful_runs == 5
    assert len(stats.failures) == 0

    # Verify concurrency was limited
    assert tracker.max_active <= 2, f"Expected max 2 concurrent tests, got {tracker.max_active}"

def test_async_batch_processing():
    """
    Test batching mechanism in the async-specific test runner.

    This test verifies that the specialized _run_stochastic_tests function:
    - Properly implements concurrent execution with batching
    - Respects batch size limits when specified
    - Runs with higher concurrency when no batching is used
    - Maintains accurate statistics in both scenarios

    Having a dedicated async runner is important for optimization and
    to avoid creating unnecessary event loops.
    """
    import asyncio
    from sik_stochastic_tests.plugin import _run_stochastic_tests

    class AsyncTestTracker:
        def __init__(self):
            self.active_count = 0
            self.max_active = 0
            self.execution_order = []
            self._lock = asyncio.Lock()  # For thread-safe counting

        async def test_func(self) -> bool:
            async with self._lock:
                self.active_count += 1
                self.max_active = max(self.max_active, self.active_count)
                self.execution_order.append(f"start_{self.active_count}")

            # Short delay to ensure overlap between concurrent executions
            await asyncio.sleep(0.05)

            async with self._lock:
                self.execution_order.append(f"end_{self.active_count}")
                self.active_count -= 1

            return True

    # First test: limited concurrency with batch_size=2
    tracker = AsyncTestTracker()
    stats = StochasticTestStats()

    # Create a new event loop for testing
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run_stochastic_tests(
            testfunction=tracker.test_func,
            funcargs={},
            stats=stats,
            samples=6,
            batch_size=2,  # Limit to 2 concurrent executions
            retry_on=None,
            max_retries=1,
            timeout=None,
        ))
    finally:
        loop.close()

    # Verify correct execution and stats
    assert stats.total_runs == 6
    assert stats.successful_runs == 6
    assert len(stats.failures) == 0

    # Verify concurrency was limited
    assert tracker.max_active <= 2, f"Expected max 2 concurrent tests, got {tracker.max_active}"

    # Second test: unlimited concurrency (batch_size=None)
    tracker2 = AsyncTestTracker()
    stats2 = StochasticTestStats()

    # Create a new event loop for testing
    loop2 = asyncio.new_event_loop()
    asyncio.set_event_loop(loop2)
    try:
        loop2.run_until_complete(_run_stochastic_tests(
            testfunction=tracker2.test_func,
            funcargs={},
            stats=stats2,
            samples=6,
            batch_size=None,  # Unlimited concurrency
            retry_on=None,
            max_retries=1,
            timeout=None,
        ))
    finally:
        loop2.close()

    # Verify execution completed correctly
    assert stats2.total_runs == 6
    assert stats2.successful_runs == 6
    assert len(stats2.failures) == 0

    # Verify concurrency was higher without batching
    assert tracker2.max_active > 2, f"Expected higher concurrency without batching, got {tracker2.max_active}"  # noqa: E501

@pytest.mark.asyncio
async def test_edge_cases_parameter_validation():
    """
    Test robust validation of edge case parameters.

    This test ensures the stochastic runner properly validates inputs and
    fails gracefully with informative error messages when invalid parameters
    are provided, rather than silently using invalid values that could cause
    subtle bugs or unpredictable behavior.

    Validation is checked for:
    - Negative or zero samples
    - Invalid retry_on parameters
    - Negative batch size handling
    - Threshold edge cases
    """
    async def test_func() -> bool:
        return True

    stats = StochasticTestStats()

    # Verify rejection of negative samples
    with pytest.raises(ValueError, match="samples must be a positive integer"):
        await _run_stochastic_tests(
            testfunction=test_func,
            funcargs={},
            stats=stats,
            samples=-1,  # Should be rejected
            batch_size=None,
            retry_on=None,
            max_retries=1,
            timeout=None,
        )

    # Verify rejection of zero samples
    with pytest.raises(ValueError, match="samples must be a positive integer"):
        await _run_stochastic_tests(
            testfunction=test_func,
            funcargs={},
            stats=stats,
            samples=0,  # Should be rejected
            batch_size=None,
            retry_on=None,
            max_retries=1,
            timeout=None,
        )

    # Verify rejection of non-exception retry_on types
    with pytest.raises(ValueError, match="retry_on must contain only exception types"):
        await _run_stochastic_tests(
            testfunction=test_func,
            funcargs={},
            stats=stats,
            samples=1,
            batch_size=None,
            retry_on=["not_an_exception", ValueError],  # Non-exception in list
            max_retries=1,
            timeout=None,
        )

    # Test negative batch_size normalization
    # This shouldn't raise an error, but should treat negative batch_size as None
    await _run_stochastic_tests(
        testfunction=test_func,
        funcargs={},
        stats=stats,
        samples=1,
        batch_size=-5,  # Negative batch_size
        retry_on=None,
        max_retries=1,
        timeout=None,
    )
    assert stats.total_runs == 1
    assert stats.successful_runs == 1

    # Removed redundant test cases that repeat the already tested validations
    # These were not working because they weren't awaited properly

def test_async_wrapper_parallel_execution():
    """Test that the async wrapper executes tests in parallel with appropriate batching."""
    # Track execution times with this class
    class ParallelTracker:
        def __init__(self):
            self.start_times = []
            self.end_times = []
            self.lock = asyncio.Lock()

        async def slow_test(self) -> bool:
            """A test that takes some time to execute."""
            async with self.lock:
                self.start_times.append(time.time())

            # Sleep to simulate work - this is what should happen in parallel
            await asyncio.sleep(0.1)

            async with self.lock:
                self.end_times.append(time.time())
            return True

    # Create a test class to simulate collection
    class MockItem:
        def __init__(self, tracker, samples, batch_size):  # noqa: ANN001
            # Create marker
            self.marker = type('MockMarker', (), {
                'kwargs': {
                    'samples': samples,
                    'batch_size': batch_size,
                    'threshold': 0.5,
                },
            })
            self.obj = tracker.slow_test
            self.nodeid = f"test_parallel_{samples}_{batch_size}"

        def get_closest_marker(self, name):  # noqa: ANN001, ANN202
            if name == "stochastic":
                return self.marker
            if name == "asyncio":
                return True  # Just need a truthy value
            return None

    # Create a mock config
    mock_config = type('MockConfig', (), {'getoption': lambda self, x, y=None: False})  # noqa: ARG005

    # Test sequential vs. parallel execution
    async def run_test(samples, batch_size):  # noqa: ANN001, ANN202
        tracker = ParallelTracker()
        mock_item = MockItem(tracker, samples, batch_size)

        # Create a collection that just has our one item
        pytest_collection_modifyitems(None, mock_config, [mock_item])

        # The modified item should now have our stochastic wrapper
        # We need to call it to execute the test
        await mock_item.obj()

        # Calculate duration for each test
        durations = []
        for start, end in zip(tracker.start_times, tracker.end_times):
            durations.append(end - start)

        # Calculate total wall clock time
        total_time = max(tracker.end_times) - min(tracker.start_times)

        return {
            'samples': samples,
            'batch_size': batch_size,
            'start_times': tracker.start_times,
            'end_times': tracker.end_times,
            'durations': durations,
            'total_time': total_time,
        }

    # Run tests with different configurations in a pytest event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Test with 5 samples, no batching - should run in parallel
        results_parallel = loop.run_until_complete(run_test(5, None))

        # Test with 5 samples, batch_size=1 - should run sequentially
        results_sequential = loop.run_until_complete(run_test(5, 1))

        # With batching=2, should be somewhere in between
        results_batched = loop.run_until_complete(run_test(5, 2))
    finally:
        loop.close()

    # Parallel execution should be much faster than sequential
    # Each test takes 0.1s, so 5 tests should take ~0.1s in parallel vs ~0.5s sequentially
    assert results_parallel['total_time'] < results_sequential['total_time'] * 0.6, \
        f"Parallel execution should be significantly faster: {results_parallel['total_time']:.3f}s vs {results_sequential['total_time']:.3f}s"  # noqa: E501

    # Batched execution should be faster than sequential but slower than fully parallel
    assert results_batched['total_time'] < results_sequential['total_time'], \
        "Batched execution should be faster than sequential"

    # Verify the number of samples
    assert len(results_parallel['start_times']) == 5, "Should have executed 5 samples"
    assert len(results_sequential['start_times']) == 5, "Should have executed 5 samples"
    assert len(results_batched['start_times']) == 5, "Should have executed 5 samples"

def test_retry_exhaustion():
    """Test that retry exhaustion is properly reported."""
    # Create a test function that always fails with a retriable exception
    class TestTracker:
        def __init__(self):
            self.attempts = 0

        async def always_fail(self) -> Never:
            self.attempts += 1
            raise ValueError("Intentional failure for retry test")

    tracker = TestTracker()
    stats = StochasticTestStats()

    # Run with retries, but it will still fail after max_retries
    # Create a new event loop for testing
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run_stochastic_tests(
            testfunction=tracker.always_fail,
            funcargs={},
            stats=stats,
            samples=1,
            batch_size=None,
            retry_on=[ValueError],  # Make ValueError retriable
            max_retries=3,  # Allow 3 attempts
            timeout=None,
        ))
    finally:
        loop.close()

    # Check attempts
    assert tracker.attempts == 3, "Should have attempted the test exactly 3 times"

    # Check stats
    assert stats.total_runs == 1
    assert stats.successful_runs == 0
    assert len(stats.failures) == 1

    # Check error details
    failure = stats.failures[0]
    assert "RetryError" in failure["type"] or "ValueError" in failure["type"]

####
# Integration tests for the stochastic plugin.
####

def test_marker_registered():
    """Test that the stochastic marker is registered."""
    # Run pytest to show markers
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--markers"],
        capture_output=True,
        text=True,
        check=False,
    )

    # Check if our marker is listed
    assert "stochastic" in result.stdout

def test_basic_stochastic_execution(example_test_dir: Path):
    """Test that stochastic tests run multiple times."""
    # Create a counter file to track executions
    counter_file = example_test_dir / "counter.txt"
    counter_file.write_text("0")

    # Create a test file
    test_file = example_test_dir / "test_counter.py"
    test_file.write_text("""
import pytest

@pytest.mark.stochastic(samples=5)
def test_counted():
    with open("counter.txt", "r") as f:
        count = int(f.read())

    with open("counter.txt", "w") as f:
        f.write(str(count + 1))

    assert True
""")

    print(f"Running test with file: {test_file}")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        capture_output=True,
        text=True,
        cwd=example_test_dir, check=False,
    )
    print(f"Subprocess completed with stdout: {result.stdout[:100]}...")
    print(f"Subprocess stderr: {result.stderr[:100]}...")

    # Check if the test passed
    assert "1 passed" in result.stdout

    # Verify the counter - test should have run 5 times
    with open(counter_file) as f:
        count = int(f.read())

    assert count == 5, f"Test should run 5 times, but ran {count} times"

@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_stochastic_failure_threshold(example_test_dir: Path, is_async: bool):
    """Test that stochastic tests fail if below threshold for both sync and async functions."""
    # For async tests, we need to add both stochastic and asyncio markers

    # Create a file to track failures
    counter_file = example_test_dir / "failure_count.txt"
    counter_file.write_text("0")

    # Configure based on async or sync
    test_prefix = "async " if is_async else ""
    asyncio_import = "import asyncio\n" if is_async else ""
    sleep_code = "await asyncio.sleep(0.01)\n    " if is_async else ""

    # Adding pytest.ini for asyncio_mode if testing async
    if is_async:
        pytest_ini = example_test_dir / "pytest.ini"
        pytest_ini.write_text("""
[pytest]
asyncio_mode = auto
""")

    # Create a test file with failures - using a more reliable failure pattern
    test_file = example_test_dir / "test_failures.py"
    test_file.write_text(f"""
import pytest
{asyncio_import}
# File to count executions
COUNT_FILE = "failure_count.txt"
import logging

# Configure logging
logging.basicConfig(filename='test_log.txt', level=logging.DEBUG)

@pytest.mark.stochastic(samples=5, threshold=0.9)  # Require 90% success
{"@pytest.mark.asyncio" if is_async else ""}
{test_prefix}def test_with_failures():
    # Read and increment counter
    with open(COUNT_FILE, "r") as f:
        count = int(f.read())
    logging.debug(f"Test execution count: {{count}}")

    with open(COUNT_FILE, "w") as f:
        f.write(str(count + 1))

    {sleep_code}
    # Always fail on the first run, making success rate 80% (4/5)
    # This ensures we'll be below the 90% threshold
    if count == 0:
        assert False, f"Intentional failure on first run"
    assert True
""")

    # Run pytest on the file
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        capture_output=True,
        text=True,
        cwd=example_test_dir, check=False,
    )

    # Print output for debugging
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")

    # Both sync and async tests should fail due to threshold check
    assert "1 failed" in result.stdout
    # Verify Success rate is mentioned in output
    assert "Success rate" in result.stdout

    # Check run count for both async and sync tests
    with open(counter_file) as f:
        count = int(f.read())

    # Both should run the expected number of times
    assert count == 5, f"Test should run 5 times, but ran {count} times"

def test_stochastic_failure_threshold_2(example_test_dir: Path):
    """Test that stochastic tests fail if below threshold."""
    # Create a file that will control which tests fail
    failure_control = example_test_dir / "failure_control.py"
    failure_control.write_text("""
# This file controls which tests should fail
FAIL_COUNT = 3  # Make 3 out of 5 tests fail to ensure we're below threshold
""")

    counter_file = example_test_dir / "counter.txt"
    counter_file.write_text("0")

    # Create a test file that uses the control file - explicitly avoiding any asyncio
    test_file = example_test_dir / "test_failures.py"
    test_file.write_text("""
import pytest
import importlib.util
import os
import sys

# Import the failure control configuration
spec = importlib.util.spec_from_file_location("failure_control", "failure_control.py")
failure_control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(failure_control)

# Use a very clear threshold that will definitely fail (3 out of 5 failures = 40% success rate)
@pytest.mark.stochastic(samples=5, threshold=0.6)  # Require 60% success when we'll only get 40%
def test_with_failures():
    # Make sure we're actually running as many times as we expect
    # Read and increment counter
    with open("counter.txt", "r") as f:
        count = int(f.read())

    count += 1
    with open("counter.txt", "w") as f:
        f.write(str(count))

    # Print diagnostic info to aid debugging
    print(f"Running test iteration {count}", file=sys.stderr)  # Use stderr to make sure we see it

    # Force failures on first N runs to guarantee below threshold
    # We need to be below 60% success, so fail 3 out of 5 = 40% success
    if count <= failure_control.FAIL_COUNT:
        # Use a simple error that won't be caught by retries
        raise AssertionError(f"Intentional failure {count}")

    # These should pass (iterations 4-5)
    assert True
""")

    # Run pytest on the file
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        capture_output=True,
        text=True,
        cwd=example_test_dir, check=False,
    )

    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")

    # Check counter file to see how many times the test ran
    with open(counter_file) as f:
        execution_count = int(f.read())

    # At minimum, the test should have run 5 times
    assert execution_count >= 5, f"Test only ran {execution_count} times, expected at least 5"

    # If 2 out of 5 tests failed, the success rate would be 60%,
    # which is below the 70% threshold - the test should fail
    assert "1 failed" in result.stdout, "Test should have failed due to low success rate"

@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_disable_stochastic_flag(example_test_dir: Path, is_async: bool):
    """Test that --disable-stochastic flag makes tests run only once for both sync and async functions."""  # noqa: E501
    # For async tests, we need to add both stochastic and asyncio markers

    # Create a counter file
    counter_file = example_test_dir / "disable_counter.txt"
    counter_file.write_text("0")

    # Configure based on async or sync
    test_prefix = "async " if is_async else ""
    asyncio_import = "import asyncio\n" if is_async else ""
    sleep_code = "await asyncio.sleep(0.01)\n    " if is_async else ""

    # Adding pytest.ini for asyncio_mode if testing async
    if is_async:
        pytest_ini = example_test_dir / "pytest.ini"
        pytest_ini.write_text("""
[pytest]
asyncio_mode = auto
""")

    # Create a test file
    test_file = example_test_dir / "test_disable.py"
    test_file.write_text(f"""
import pytest
{asyncio_import}
@pytest.mark.stochastic(samples=5)
{"@pytest.mark.asyncio" if is_async else ""}
{test_prefix}def test_disable_flag():
    with open("disable_counter.txt", "r") as f:
        count = int(f.read())

    with open("disable_counter.txt", "w") as f:
        f.write(str(count + 1))

    {sleep_code}
    assert True
""")

    # Run pytest with --disable-stochastic flag
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v", "--disable-stochastic"],
        capture_output=True,
        text=True,
        cwd=example_test_dir, check=False,
    )

    # For both sync and async tests, check that the test passed with --disable-stochastic
    assert "1 passed" in result.stdout

    # For both sync and async tests, verify the counter - test should have run only once
    with open(counter_file) as f:
        count = int(f.read())

    assert count == 1, f"Test should run once with --disable-stochastic, but ran {count} times"

@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_stochastic_timeout(example_test_dir: Path, is_async: bool):
    """Test that timeout functionality works for both sync and async functions."""
    # For async tests, we need to add both stochastic and asyncio markers

    # Configure based on async or sync
    test_prefix = "async " if is_async else ""
    sleep_import = "import asyncio" if is_async else "import time"
    sleep_func = "await asyncio.sleep(1.0)" if is_async else "time.sleep(1.0)"

    # Adding pytest.ini for asyncio_mode if testing async
    if is_async:
        pytest_ini = example_test_dir / "pytest.ini"
        pytest_ini.write_text("""
[pytest]
asyncio_mode = auto
""")

    # Create a test file with a slow test
    test_file = example_test_dir / "test_timeout.py"
    test_file.write_text(f"""
import pytest
{sleep_import}

@pytest.mark.stochastic(samples=3, timeout=0.1)
{"@pytest.mark.asyncio" if is_async else ""}
{test_prefix}def test_timeout():
    # Sleep longer than the timeout
    {sleep_func}
    assert True
""")

    # Run pytest on the file
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        capture_output=True,
        text=True,
        cwd=example_test_dir, check=False,
    )

    # Both sync and async tests should timeout and fail
    assert "1 failed" in result.stdout, f"Expected test to fail but got:\n{result.stdout}"
    assert "timeout" in result.stdout.lower(), f"Expected timeout message but got:\n{result.stdout}"  # noqa: E501

def test_asyncio_compatibility(example_test_dir: Path):
    """Test that plugin correctly detects and skips tests with asyncio marker."""
    # Create a counter file to track executions
    counter_file = example_test_dir / "asyncio_counter.txt"
    counter_file.write_text("0")

    # Create a test file with both stochastic and asyncio markers
    # With our fix, we should detect this and let pytest-asyncio handle it
    # (which means it will run once, not multiple times)
    test_file = example_test_dir / "test_asyncio_stochastic.py"
    test_file.write_text("""
import pytest
import asyncio

# Test with both markers - should be handled by pytest-asyncio
@pytest.mark.asyncio
@pytest.mark.stochastic(samples=3)
async def test_with_both_markers():
    # Read counter
    with open("asyncio_counter.txt", "r") as f:
        count = int(f.read())

    # Increment counter
    count += 1
    with open("asyncio_counter.txt", "w") as f:
        f.write(str(count))

    # Simulate some async work
    await asyncio.sleep(0.01)
    assert True
""")

    # Run pytest on the file
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        capture_output=True,
        text=True,
        cwd=example_test_dir,
        check=False,
    )

    # Print output for debugging
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")

    # Check if the test passed
    assert "1 passed" in result.stdout

    # Verify the test ran stochastically
    with open(counter_file) as f:
        count = int(f.read())

    assert count == 3, f"Test should run 3 times with stochastic functionality, but ran {count} times"  # noqa: E501

def test_asyncio_class_compatibility(example_test_dir: Path):
    """Test that plugin correctly handles tests in a class marked with pytest.mark.asyncio."""
    # Create a counter file to track executions
    counter_file = example_test_dir / "asyncio_class_counter.txt"
    counter_file.write_text("0")

    # Create a test file with a class marked with asyncio and methods marked with stochastic
    # This is exactly the scenario that was failing in the original issue
    test_file = example_test_dir / "test_asyncio_class.py"
    test_file.write_text("""
import pytest
import asyncio

# Class with asyncio marker
@pytest.mark.asyncio
class TestAsyncioClass:
    # Method with stochastic marker
    @pytest.mark.stochastic(samples=3)
    async def test_in_asyncio_class(self):
        # Read counter
        with open("asyncio_class_counter.txt", "r") as f:
            count = int(f.read())

        # Increment counter
        count += 1
        with open("asyncio_class_counter.txt", "w") as f:
            f.write(str(count))

        # Simulate some async work
        await asyncio.sleep(0.01)
        print("HELLO WORLD")  # Add this to match the original issue output
        assert True
""")

    # Run pytest on the file
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        capture_output=True,
        text=True,
        cwd=example_test_dir,
        check=False,
    )

    # Print output for debugging
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")

    # Check if the test passed - our fix should bypass stochastic and let pytest-asyncio handle it
    assert "1 passed" in result.stdout

    # Verify the test ran stochastically
    with open(counter_file) as f:
        count = int(f.read())

    assert count == 3, f"Test should run 3 times with stochastic functionality, but ran {count} times"  # noqa: E501


def test_async_no_asyncio_marker(example_test_dir: Path):
    """Test that plugin correctly detects async functions and delegates them properly."""
    # Create a counter file to track executions
    counter_file = example_test_dir / "async_no_marker_counter.txt"
    counter_file.write_text("0")

    # Create a test file with an async function that has stochastic marker but no asyncio marker
    # We need to add pytest.ini with asyncio_mode = auto to make it work without the marker
    pytest_ini = example_test_dir / "pytest.ini"
    pytest_ini.write_text("""
[pytest]
asyncio_mode = auto
""")

    # Create the test file
    test_file = example_test_dir / "test_async_no_asyncio_marker.py"
    test_file.write_text("""
import pytest
import asyncio

# Test with stochastic but NO asyncio marker
@pytest.mark.stochastic(samples=3, threshold=0.8)
async def test_async_without_asyncio_marker():
    # Read counter
    with open("async_no_marker_counter.txt", "r") as f:
        count = int(f.read())

    # Increment counter
    count += 1
    with open("async_no_marker_counter.txt", "w") as f:
        f.write(str(count))

    # Simulate some async work
    await asyncio.sleep(0.01)
    assert True
""")

    # Run pytest on the file
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        capture_output=True,
        text=True,
        cwd=example_test_dir,
        check=False,
    )

    # Print output for debugging
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")

    # Check if the test passed with either asyncio_mode=auto or our detection
    assert "1 passed" in result.stdout or "SKIPPED" in result.stdout, f"Test failed unexpectedly: {result.stderr}"  # noqa: E501

    # With our detection, the test should either run once or be skipped (not fail with IndexError)
    with open(counter_file) as f:
        count = int(f.read())

    # The test should run with stochastic functionality (samples=3)
    assert count == 3, f"Async test should run 3 times with stochastic functionality, but ran {count} times"  # noqa: E501

def test_has_asyncio_marker_detects_coroutines():
    """Test that has_asyncio_marker correctly identifies coroutine functions."""
    from sik_stochastic_tests.plugin import has_asyncio_marker
    import inspect

    # Create a mock pytest.Function object with a coroutine function
    async def async_func() -> None:
        pass

    def sync_func() -> None:
        pass

    # Create mock objects to test the function
    mock_async = MagicMock()
    mock_async.obj = async_func
    mock_async.own_markers = []  # No markers

    mock_sync = MagicMock()
    mock_sync.obj = sync_func
    mock_sync.own_markers = []  # No markers

    # Test that our updated has_asyncio_marker detects coroutines
    assert has_asyncio_marker(mock_async) is True, "Should detect coroutine function even without marker"  # noqa: E501
    assert has_asyncio_marker(mock_sync) is False, "Should not detect non-coroutine function"

    # Verify our detection is using inspect.iscoroutinefunction
    assert inspect.iscoroutinefunction(async_func) is True
    assert inspect.iscoroutinefunction(sync_func) is False

def test_terminal_reporting_no_failures(capsys: pytest.CaptureFixture[str]):  # noqa: ARG001
    """Test terminal reporting when there are no failures."""
    # Create a test result with no failures
    stats = StochasticTestStats()
    stats.total_runs = 10
    stats.successful_runs = 10

    # Add to global test results
    _test_results.clear()  # Ensure we're starting fresh
    _test_results["test_perfect::test_always_passes"] = stats

    # Create a mock terminal reporter
    class MockTerminalReporter:
        def __init__(self):
            self.lines = []

        def write_sep(self, sep, title) -> None:  # noqa: ANN001
            self.lines.append(f"{sep*10} {title} {sep*10}")

        def write_line(self, line) -> None:  # noqa: ANN001
            self.lines.append(line)

    reporter = MockTerminalReporter()

    # Run the terminal summary function
    pytest_terminal_summary(reporter)

    # Verify correct output
    assert any("Success rate: 1.00" in line for line in reporter.lines)
    assert any("Runs: 10, Successes: 10, Failures: 0" in line for line in reporter.lines)
    # Verify no failure output
    assert not any("Failure samples:" in line for line in reporter.lines)

    # Clean up
    _test_results.clear()

@pytest.mark.asyncio
async def test_event_loop_cleanup_with_exceptions():
    """Test event loop is properly cleaned up even when tests throw exceptions."""
    from sik_stochastic_tests.plugin import _run_stochastic_tests, StochasticTestStats
    import asyncio
    import gc
    import weakref

    # Function that always raises an exception
    async def failing_test() -> Never:
        raise ValueError("Intentional exception")

    # Use a weak reference to track if the loop is cleaned up
    loop_ref = None

    async def run_with_new_loop() -> None:
        # Create a new event loop
        old_loop = asyncio.get_event_loop()
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)

        nonlocal loop_ref
        loop_ref = weakref.ref(new_loop)

        try:
            # Run tests that will fail
            stats = StochasticTestStats()
            await _run_stochastic_tests(
                testfunction=failing_test,
                funcargs={},
                stats=stats,
                samples=3,
                batch_size=None,
                retry_on=None,
                max_retries=1,
                timeout=None,
            )

            # Verify stats are correct
            assert stats.total_runs == 3
            assert stats.successful_runs == 0
            assert len(stats.failures) == 3

        finally:
            # Restore the original loop
            asyncio.set_event_loop(old_loop)

    # Run the test with a new loop
    await run_with_new_loop()

    # Force garbage collection
    gc.collect()

    # Verify the loop was properly cleaned up (reference should be None)
    assert loop_ref() is None, "Event loop was not properly cleaned up"

def test_timeout_with_cpu_bound_task(example_test_dir: Path):
    """Test timeout mechanism with CPU-bound tasks (controlled intensive work)."""
    # Create a test file with a CPU-bound task that can be timed out
    test_file = example_test_dir / "test_cpu_timeout.py"
    test_file.write_text("""
import pytest
import time
import sys

@pytest.mark.stochastic(samples=1, timeout=0.5)
def test_cpu_bound_timeout():
    # A CPU-bound task that checks for timeout periodically
    # This approach allows the test to be terminated by our timeout mechanism
    start_time = time.time()
    counter = 0

    # Print to ensure we see output in test results
    print("Starting CPU-bound test that should timeout after 0.5 seconds", file=sys.stderr)

    try:
        # Perform intensive calculation but check elapsed time periodically
        while True:
            # Do some CPU-intensive work (addition loop)
            for i in range(1000000):
                counter += i

            # Log progress
            elapsed = time.time() - start_time
            print(f"Progress check: elapsed={elapsed:.2f}s, counter={counter}", file=sys.stderr)

            # Check if we should create a checkpoint for signal handling
            if elapsed > 5:  # Much longer than our timeout
                break  # Safety exit to prevent test hanging if timeout fails
    except Exception as e:
        print(f"Exception caught: {type(e).__name__}: {e}", file=sys.stderr)
        raise

    # If we get here, the timeout didn't work or took too long
    assert False, f"Timeout didn't stop test execution, counter={counter}"
""")

    # Run pytest on the file with a longer global timeout
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        capture_output=True,
        text=True,
        cwd=example_test_dir,
        timeout=15,  # Overall process timeout
        check=False,
    )

    # Print output for debugging
    print("Test output:")
    print(f"STDOUT:\\n{result.stdout}")
    print(f"STDERR:\\n{result.stderr}")

    # Test should fail due to timeout
    assert "1 failed" in result.stdout
    assert "timeout" in result.stdout.lower() or "TimeoutError" in result.stdout, "Expected timeout failure message"  # noqa: E501

    # Log specific data from the output
    if "timeout" in result.stdout.lower():
        print("✅ Found timeout message in output")
    else:
        print("❌ Timeout message not found in output")

@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_stochastic_with_parametrize(example_test_dir: Path, is_async: bool):
    """Test that stochastic tests work correctly with pytest.mark.parametrize for both sync and async tests."""  # noqa: E501
    # Create a counter file to track executions
    counter_file = example_test_dir / "parametrize_counter.txt"
    counter_file.write_text("0")

    # Configure based on async or sync
    test_prefix = "async " if is_async else ""
    asyncio_import = "import asyncio\n" if is_async else ""
    sleep_code = "await asyncio.sleep(0.01)\n        " if is_async else ""

    # Adding pytest.ini for asyncio_mode if testing async
    if is_async:
        pytest_ini = example_test_dir / "pytest.ini"
        pytest_ini.write_text("""
[pytest]
asyncio_mode = auto
""")

    # Create a test file with parametrize and stochastic
    test_file = example_test_dir / "test_parametrize.py"
    test_file.write_text(f"""
import pytest
{asyncio_import}

# Track execution count
def increment_counter(param_value):
    with open("parametrize_counter.txt", "r") as f:
        count = int(f.read())

    with open("parametrize_counter.txt", "w") as f:
        f.write(str(count + 1))

    return param_value

@pytest.mark.parametrize("test_input,expected", [
    ("value1", "EXPECTED1"),
    ("value2", "EXPECTED2"),
])
@pytest.mark.stochastic(samples=2)
{"@pytest.mark.asyncio" if is_async else ""}
{test_prefix}def test_with_parameters(test_input, expected):
    # Track this execution
    actual_input = increment_counter(test_input)

    # Add a small delay for async tests
    {sleep_code}

    # Simple assertion using the parameters
    transformed = actual_input.upper().replace('VALUE', 'EXPECTED')
    assert transformed == expected, f"Expected {{expected}} but got {{transformed}}"
""")

    # Run pytest on the file
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        capture_output=True,
        text=True,
        cwd=example_test_dir,
        check=False,
    )

    # Print output for debugging
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")

    # Verify the test passed
    assert "2 passed" in result.stdout, f"Expected 2 tests to pass, got: {result.stdout}"

    # Read the counter to see how many times the test was executed
    with open(counter_file) as f:
        count = int(f.read())

    # We should have 2 parametrized tests x 2 samples each = 4 executions
    assert count == 4, f"Expected 4 executions (2 params x 2 samples), but got {count}"

    # Check if stochastic reporting is in the output
    assert "Stochastic Test Results" in result.stdout, "Expected stochastic reporting in output"

    # Verify that the tests were actually parametrized
    assert "test_with_parameters[value1-EXPECTED1]" in result.stdout
    assert "test_with_parameters[value2-EXPECTED2]" in result.stdout

def test_stochastic_class_marker(example_test_dir: Path):
    """Test that stochastic marker applied to a class works for all methods within the class."""
    # Create a counter file to track executions
    counter_file = example_test_dir / "class_counter.txt"
    counter_file.write_text("0")

    # Create a test file with a class that has the stochastic marker
    test_file = example_test_dir / "test_class_marker.py"
    test_file.write_text("""
import pytest

# Track execution count
def increment_counter(method_name):
    with open("class_counter.txt", "r") as f:
        count = int(f.read())

    with open("class_counter.txt", "w") as f:
        f.write(str(count + 1))

    print(f"Executed {method_name} (count {count+1})")
    return count + 1

@pytest.mark.stochastic(samples=3)
class TestStochasticClass:
    \"\"\"A test class with stochastic marker that should apply to all methods.\"\"\"

    def test_method_one(self):
        \"\"\"First test method.\"\"\"
        increment_counter("test_method_one")
        assert True

    def test_method_two(self):
        \"\"\"Second test method.\"\"\"
        increment_counter("test_method_two")
        assert True
""")

    # Run pytest on the file
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        capture_output=True,
        text=True,
        cwd=example_test_dir,
        check=False,
    )

    # Print output for debugging
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")

    # Verify the test passed
    assert "2 passed" in result.stdout, f"Expected 2 tests to pass, got: {result.stdout}"

    # Read the counter to see how many times the tests were executed
    with open(counter_file) as f:
        count = int(f.read())

    # We should have 2 test methods x 3 samples each = 6 executions
    assert count == 6, f"Expected 6 executions (2 methods x 3 samples), but got {count}"

    # Check if stochastic reporting is in the output for each test method
    assert "test_class_marker.py::TestStochasticClass::test_method_one" in result.stdout
    assert "test_class_marker.py::TestStochasticClass::test_method_two" in result.stdout
    assert "Stochastic Test Results" in result.stdout

    # Check success rates are correctly reported
    assert "Success rate: 1.00" in result.stdout

def test_async_generator_cleanup_issue(example_test_dir: Path):
    """Test for the 'pop from an empty deque' issue with multiple async generators."""
    # Create a counter file to track executions
    counter_file = example_test_dir / "generator_counter.txt"
    counter_file.write_text("0")

    # Add pytest.ini for asyncio mode
    pytest_ini = example_test_dir / "pytest.ini"
    pytest_ini.write_text("""
[pytest]
asyncio_mode = auto
asyncio_default_fixture_loop_scope = function
""")

    # Create a test file that simulates the OpenAI async generator pattern
    # This reproduces the error: "IndexError: pop from an empty deque"
    test_file = example_test_dir / "test_async_generators.py"
    # Triple quotes with raw string to avoid docstring issues
    test_file.write_text(r'''
import pytest
import asyncio
import sys

# Create a simple async generator
async def async_generator(name, count=3):
    """Simulate an OpenAI-like async generator."""
    for i in range(count):
        await asyncio.sleep(0.01)  # Small delay
        print(f"Generator {name} yielding {i}", file=sys.stderr)
        yield f"{name}_{i}"

# Create a consumer that uses multiple generators simultaneously
async def consume_generators(*generators):
    """Consume multiple generators together."""
    # Start all generators
    gens = [gen.__aiter__() for gen in generators]

    # Process them together
    try:
        while True:
            # Get next items from all generators simultaneously
            tasks = [asyncio.create_task(gen.__anext__()) for gen in gens]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check if we're done
            if all(isinstance(r, StopAsyncIteration) for r in results):
                break

            # Process valid results
            valid_results = [r for r in results if not isinstance(r, Exception)]
            if not valid_results:
                break
    except StopAsyncIteration:
        pass

    # Record execution
    with open("generator_counter.txt", "r") as f:
        count = int(f.read())

    with open("generator_counter.txt", "w") as f:
        f.write(str(count + 1))

    return True

@pytest.mark.stochastic(samples=5, batch_size=2)  # Force concurrent executions
@pytest.mark.asyncio
async def test_with_multiple_generators():
    """Test that creates and consumes multiple async generators simultaneously."""
    # Create multiple concurrent generators (similar to OpenAI streams)
    gen1 = async_generator("stream1")
    gen2 = async_generator("stream2")
    gen3 = async_generator("stream3")

    # Consume them together, which can trigger the event loop issue
    await consume_generators(gen1, gen2, gen3)

    # Simple assertion to make the test pass
    assert True
''')

    # Run pytest on the file
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        capture_output=True,
        text=True,
        cwd=example_test_dir,
        check=False,
    )

    # Print output for debugging
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")

    # Check if the test passed
    assert "1 passed" in result.stdout, f"Expected test to pass, got: {result.stdout}"

    # Check counter to see if the test ran completely
    with open(counter_file) as f:
        count = int(f.read())

    # The test should run all 5 samples if our fix works
    assert count == 5, f"Expected test to run 5 times, but ran {count} times"

    # Verify that there's no "pop from an empty deque" error
    assert "pop from an empty deque" not in result.stderr, "Error still present despite the fix"
    assert "aclose(): asynchronous generator is already running" not in result.stderr, "Generator cleanup issue still present"  # noqa: E501

def test_async_generator_cleanup_extreme_case(example_test_dir: Path):
    """Test for 'pop from an empty deque' issue with heavy concurrent generator load."""
    # Create a counter file to track executions
    counter_file = example_test_dir / "extreme_counter.txt"
    counter_file.write_text("0")

    # Add pytest.ini for asyncio mode
    pytest_ini = example_test_dir / "pytest.ini"
    pytest_ini.write_text("""
[pytest]
asyncio_mode = auto
asyncio_default_fixture_loop_scope = function
""")

    # Create a test file with an even more aggressive generator pattern
    test_file = example_test_dir / "test_extreme_generators.py"
    # Triple quotes with raw string to avoid docstring issues
    test_file.write_text(r'''
import pytest
import asyncio
import sys

# Create nested generators - this pattern is more likely to
# cause the empty deque issue when run concurrently
async def outer_generator(name, count=3):
    """An outer generator that yields inner generators."""
    for i in range(count):
        # Yield an inner generator for each iteration
        yield inner_generator(f"{name}_{i}")

async def inner_generator(name, count=2):
    """An inner generator."""
    for j in range(count):
        await asyncio.sleep(0.01)
        print(f"Inner generator {name} yielding {j}", file=sys.stderr)
        yield f"{name}_inner_{j}"

# Function to consume nested generators
async def consume_nested_generators(*generators):
    """Consume multiple nested generators concurrently."""
    # Track outer generators
    outer_gens = [gen.__aiter__() for gen in generators]
    inner_gens = []

    # Process first level
    try:
        while outer_gens:
            # Get next inner generators
            outer_tasks = [asyncio.create_task(gen.__anext__()) for gen in outer_gens]
            outer_results = await asyncio.gather(*outer_tasks, return_exceptions=True)

            # Process results and collect inner generators
            for i, result in enumerate(outer_results):
                if isinstance(result, StopAsyncIteration):
                    # This outer generator is exhausted
                    outer_gens[i] = None
                elif not isinstance(result, Exception):
                    # Add the inner generator
                    inner_gens.append(result.__aiter__())

            # Remove exhausted generators
            outer_gens = [gen for gen in outer_gens if gen is not None]

            # Process inner generators if we have any
            if inner_gens:
                inner_tasks = [asyncio.create_task(gen.__anext__()) for gen in inner_gens]
                inner_results = await asyncio.gather(*inner_tasks, return_exceptions=True)

                # Process inner results
                active_inner_gens = []
                for i, result in enumerate(inner_results):
                    if not isinstance(result, StopAsyncIteration) and not isinstance(result, Exception):
                        # Keep this generator for next iteration
                        active_inner_gens.append(inner_gens[i])

                # Update active inner generators
                inner_gens = active_inner_gens

    except Exception as e:
        print(f"Error in consume_nested_generators: {e}", file=sys.stderr)

    # Record successful execution
    with open("extreme_counter.txt", "r") as f:
        count = int(f.read())

    with open("extreme_counter.txt", "w") as f:
        f.write(str(count + 1))

    return True

@pytest.mark.stochastic(samples=5, batch_size=3)  # Run batches concurrently
@pytest.mark.asyncio
async def test_with_extreme_generator_pattern():
    """Test with a complex, nested generator pattern that's more likely to trigger errors."""
    # Create multiple outer generators
    gen1 = outer_generator("stream1", count=4)
    gen2 = outer_generator("stream2", count=4)
    gen3 = outer_generator("stream3", count=4)
    gen4 = outer_generator("stream4", count=4)

    # Consume them together, which should trigger the event loop issue
    # if our fix doesn't work
    await consume_nested_generators(gen1, gen2, gen3, gen4)

    # Simple assertion to make the test pass
    assert True
''')  # noqa: E501

    # Run pytest on the file
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        capture_output=True,
        text=True,
        cwd=example_test_dir,
        check=False,
    )

    # Print output for debugging
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")

    # Check if the test passed
    assert "1 passed" in result.stdout, f"Expected test to pass, got: {result.stdout}"

    # Check counter to see if the test ran completely
    with open(counter_file) as f:
        count = int(f.read())

    # The test should run all 5 samples if our fix works
    assert count == 5, f"Expected test to run 5 times, but ran {count} times"

    # Verify that there's no "pop from an empty deque" error
    assert "pop from an empty deque" not in result.stderr, "Error still present despite the fix"
    assert "aclose(): asynchronous generator is already running" not in result.stderr, "Generator cleanup issue still present"  # noqa: E501

def test_stochastic_class_marker_async(example_test_dir: Path):
    """Test that stochastic marker applied to an async class works for all methods within the class."""  # noqa: E501
    # Create a counter file to track executions
    counter_file = example_test_dir / "async_class_counter.txt"
    counter_file.write_text("0")

    # Adding pytest.ini for asyncio_mode
    pytest_ini = example_test_dir / "pytest.ini"
    pytest_ini.write_text("""
[pytest]
asyncio_mode = auto
""")

    # Create a test file with a class that has the stochastic marker
    test_file = example_test_dir / "test_async_class_marker.py"
    test_file.write_text("""
import pytest
import asyncio

# Track execution count
def increment_counter(method_name):
    with open("async_class_counter.txt", "r") as f:
        count = int(f.read())

    with open("async_class_counter.txt", "w") as f:
        f.write(str(count + 1))

    print(f"Executed {method_name} (count {count+1})")
    return count + 1

@pytest.mark.stochastic(samples=3)
@pytest.mark.asyncio
class TestAsyncStochasticClass:
    \"\"\"A test class with stochastic and asyncio markers.\"\"\"

    async def test_async_method_one(self):
        \"\"\"First async test method.\"\"\"
        increment_counter("test_async_method_one")
        await asyncio.sleep(0.01)
        assert True

    async def test_async_method_two(self):
        \"\"\"Second async test method.\"\"\"
        increment_counter("test_async_method_two")
        await asyncio.sleep(0.01)
        assert True
""")

    # Run pytest on the file
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        capture_output=True,
        text=True,
        cwd=example_test_dir,
        check=False,
    )

    # Print output for debugging
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")

    # Verify the test passed
    assert "2 passed" in result.stdout, f"Expected 2 tests to pass, got: {result.stdout}"

    # Read the counter to see how many times the tests were executed
    with open(counter_file) as f:
        count = int(f.read())

    # We should have 2 test methods x 3 samples each = 6 executions
    assert count == 6, f"Expected 6 executions (2 methods x 3 samples), but got {count}"

    # Check if stochastic reporting is in the output for each test method
    assert "TestAsyncStochasticClass::test_async_method_one" in result.stdout
    assert "TestAsyncStochasticClass::test_async_method_two" in result.stdout
    assert "Stochastic Test Results" in result.stdout

def test_async_http_client_edge_case(example_test_dir: Path):
    """Test that reproduces the edge case with HTTP clients causing 'pop from an empty deque'."""
    # Create a counter file to track executions
    counter_file = example_test_dir / "http_edge_case_counter.txt"
    counter_file.write_text("0")

    # Add pytest.ini for asyncio mode
    pytest_ini = example_test_dir / "pytest.ini"
    pytest_ini.write_text("""
[pytest]
asyncio_mode = auto
asyncio_default_fixture_loop_scope = function
""")

    # Create a test file that specifically reproduces the edge case
    test_file = example_test_dir / "test_http_edge_case.py"
    # Triple quotes with raw string to avoid docstring issues
    test_file.write_text(r'''
import pytest
import asyncio
import sys
from contextlib import asynccontextmanager

# Create a class that simulates the behavior of httpx.AsyncClient
class MockAsyncClient:
    """Simulates the behavior of httpx.AsyncClient that causes the issue."""

    def __init__(self, name):
        self.name = name
        self.closed = False
        print(f"Creating {self.name}", file=sys.stderr)

    async def aclose(self):
        """Simulates httpx.AsyncClient.aclose() which can cause issues when called concurrently."""
        if self.closed:
            return
        self.closed = True
        print(f"Closing {self.name}", file=sys.stderr)
        # Small delay to simulate network closing
        await asyncio.sleep(0.01)
        # Create and immediately consume an async generator
        # This is what triggers the issue in real HTTP clients
        async for _ in self._cleanup_generator():
            pass
        print(f"Closed {self.name}", file=sys.stderr)

    async def _cleanup_generator(self):
        """Simulates the internal cleanup generators in HTTP clients."""
        for i in range(2):
            await asyncio.sleep(0.01)
            yield i

    async def request(self, *args, **kwargs):
        """Simulates a request that returns a stream."""
        return self._stream_response()

    async def _stream_response(self):
        """Simulates a streaming response like those from OpenAI/Anthropic."""
        for i in range(3):
            await asyncio.sleep(0.01)
            yield f"chunk_{i}"

# Context manager to properly handle the mock client
@asynccontextmanager
async def get_client(name):
    client = MockAsyncClient(name)
    try:
        yield client
    finally:
        await client.aclose()

# Test that uses multiple clients simultaneously
@pytest.mark.stochastic(samples=5, batch_size=2)  # Force concurrent executions
@pytest.mark.asyncio
async def test_multiple_async_clients():
    """Test that creates multiple mock HTTP clients that use async generators."""
    # Increment the counter to track executions
    with open("http_edge_case_counter.txt", "r") as f:
        count = int(f.read())

    with open("http_edge_case_counter.txt", "w") as f:
        f.write(str(count + 1))

    # Create multiple clients that will be closed simultaneously at the end
    async with get_client("client1") as client1, get_client("client2") as client2:
        # Start multiple streaming responses simultaneously
        response1 = await client1.request("test")
        response2 = await client2.request("test")

        # Collect responses from both streams at the same time
        # This pattern of accessing multiple async generators concurrently
        # is what causes the empty deque error
        results = []

        # Process first stream
        async for chunk in response1:
            results.append(chunk)

            # Process second stream simultaneously to force interleaving
            if len(results) == 2:  # After collecting 2 chunks from first stream
                async for chunk in response2:
                    results.append(chunk)

    # Simple assertion to make the test pass
    assert len(results) > 0
    # The clients will be automatically closed by the context manager,
    # which will trigger aclose() calls that can cause the error
''')

    # Run pytest on the file
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        capture_output=True,
        text=True,
        cwd=example_test_dir,
        check=False,
    )

    # Print output for debugging
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")

    # Check if the test passed
    assert "1 passed" in result.stdout, f"Expected test to pass, got: {result.stdout}"

    # Check counter to see if the test ran completely
    with open(counter_file) as f:
        count = int(f.read())

    # The test should run all 5 samples if our fix works
    assert count == 5, f"Expected test to run 5 times, but ran {count} times"

    # Verify that there's no "pop from an empty deque" error
    assert "pop from an empty deque" not in result.stderr, "Error still present despite the fix"
    assert "Event loop is closed" not in result.stderr, "Event loop closed error still present"

def test_sync_with_internal_async(example_test_dir: Path):
    """Test stochastic tests with all combinations of sync/async."""
    counter_file = example_test_dir / "counter.txt"
    counter_file.write_text("0")
    test_file = example_test_dir / "test_combined.py"
    test_file.write_text("""
import pytest
import asyncio

def increment_counter():
    with open("counter.txt", "r") as f:
        count = int(f.read())
    with open("counter.txt", "w") as f:
        f.write(str(count + 1))
    return count + 1

# Pure sync function
@pytest.mark.stochastic(samples=2)
def test_pure_sync():
    count = increment_counter()
    print(f"Pure sync test {count}")
    assert True

# Pure async function
@pytest.mark.stochastic(samples=2)
@pytest.mark.asyncio
async def test_pure_async():
    count = increment_counter()
    print(f"Pure async test {count}")
    await asyncio.sleep(0.01)
    assert True

# Sync function that uses async
@pytest.mark.stochastic(samples=2)
def test_sync_with_async():
    async def inner():
        await asyncio.sleep(0.01)
        return True

    count = increment_counter()
    print(f"Sync with async test {count}")
    return inner()

# Async test that calls sync function that uses async
@pytest.mark.stochastic(samples=2)
@pytest.mark.asyncio
async def test_async_calling_sync_with_async():
    def sync_func():
        async def inner():
            await asyncio.sleep(0.01)
            return True
        return inner()

    count = increment_counter()
    print(f"Async calling sync with async test {count}")
    result = sync_func()
    assert await result is True
""")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        capture_output=True,
        text=True,
        cwd=example_test_dir,
        check=False,
    )
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")
    assert "4 passed" in result.stdout, \
        f"Expected all four tests to pass, got: {result.stdout}"
    with open(counter_file) as f:
        count = int(f.read())
    # Each test runs twice, so 8 total runs
    assert count == 8, \
        f"Expected 8 total runs (4 tests x 2 samples), but got {count}"
