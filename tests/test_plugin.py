"""Unit tests for the stochastic plugin components."""
from pathlib import Path
import time
import pytest
import asyncio
import sys
import subprocess
from unittest.mock import MagicMock
from sik_stochastic_tests.plugin import (
    StochasticTestStats,
    _run_stochastic_tests,
    pytest_collection_modifyitems,
    pytest_pyfunc_call,
    run_stochastic_tests_for_async,
)
from typing import Never

def test_has_asyncio_marker():
    """Test the has_asyncio_marker helper function."""
    from sik_stochastic_tests.plugin import has_asyncio_marker

    # Create asyncio marker mock with correct name property
    asyncio_marker = MagicMock()
    asyncio_marker.name = 'asyncio'

    # Create non-asyncio marker mock
    other_marker = MagicMock()
    other_marker.name = 'other'

    # Create mock objects to test the function
    class AsyncioClass:
        pytestmark = [asyncio_marker]  # noqa: RUF012

    class NonAsyncioClass:
        pytestmark = [other_marker]  # noqa: RUF012

    # Mock a pytest function with asyncio marker
    func_with_marker = MagicMock()
    func_with_marker.own_markers = [asyncio_marker]

    # Mock a pytest function without asyncio marker
    func_without_marker = MagicMock()
    func_without_marker.own_markers = [other_marker]

    # Mock a pytest function in asyncio class
    func_in_asyncio_class = MagicMock()
    func_in_asyncio_class.own_markers = []
    func_in_asyncio_class.cls = AsyncioClass

    # Mock a method with asyncio class
    method_obj = MagicMock()
    method_obj.__self__ = MagicMock()
    method_obj.__self__.__class__ = AsyncioClass

    # Test the function
    assert has_asyncio_marker(func_with_marker) is True
    assert has_asyncio_marker(func_without_marker) is False
    assert has_asyncio_marker(func_in_asyncio_class) is True
    assert has_asyncio_marker(method_obj) is True

@pytest.mark.asyncio
async def test_run_test_function():
    """Test the run_test_function helper for different types of functions."""
    from sik_stochastic_tests.plugin import run_test_function

    # Test cases
    # 1. Synchronous function
    def sync_func(arg=None):  # noqa: ANN001, ANN202
        return arg or "sync result"

    # 2. Asynchronous function
    async def async_func(arg=None):  # noqa: ANN001, ANN202
        return arg or "async result"

    # 3. Sync function that returns a coroutine
    def returns_coroutine(arg=None):  # noqa: ANN001, ANN202
        async def inner():  # noqa: ANN202
            return arg or "coroutine result"
        return inner()

    # 4. Function with timeout
    async def slow_func() -> str:
        await asyncio.sleep(0.2)
        return "slow result"

    # Run the tests
    # Sync functions
    result1 = await run_test_function(sync_func, {"arg": "test"})
    assert result1 == "test"

    # Async functions
    result2 = await run_test_function(async_func, {"arg": "test"})
    assert result2 == "test"

    # Sync functions returning coroutines
    result3 = await run_test_function(returns_coroutine, {"arg": "test"})
    assert result3 == "test"

    # Test timeout handling
    with pytest.raises(TimeoutError):
        await run_test_function(slow_func, {}, timeout=0.1)

    # Test successful execution with sufficient timeout
    result4 = await run_test_function(slow_func, {}, timeout=0.3)
    assert result4 == "slow result"

def test_stochastic_test_stats():
    """Test the StochasticTestStats class."""
    stats = StochasticTestStats()

    # Initial state
    assert stats.total_runs == 0
    assert stats.successful_runs == 0
    assert stats.failures == []
    assert stats.success_rate == 0.0

    # After successful runs
    stats.total_runs = 5
    stats.successful_runs = 3
    assert stats.success_rate == 0.6

    # With failures
    stats.failures.append({"error": "Test error", "type": "ValueError", "context": {"run_index": 1}})  # noqa: E501
    assert len(stats.failures) == 1
    assert stats.failures[0]["type"] == "ValueError"

@pytest.mark.asyncio
async def test_run_stochastic_tests_success():
    """Test running stochastic tests with all successes."""
    # Mock test function
    async def test_func() -> bool:
        return True

    # Test statistics
    stats = StochasticTestStats()

    # Run the tests
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

    # Verify stats
    assert stats.total_runs == 5
    assert stats.successful_runs == 5
    assert len(stats.failures) == 0
    assert stats.success_rate == 1.0

@pytest.mark.asyncio
async def test_run_stochastic_tests_failures():
    """Test running stochastic tests with some failures."""
    # Create a counter to control failures
    counter = {'value': 0}

    async def test_func_with_failures() -> bool:
        counter['value'] += 1
        if counter['value'] % 2 == 0:  # Every second call will fail
            raise ValueError("Simulated failure")
        return True

    # Test statistics
    stats = StochasticTestStats()

    # Run the tests
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

    # Verify stats
    assert stats.total_runs == 4
    assert stats.successful_runs == 2
    assert len(stats.failures) == 2
    assert stats.success_rate == 0.5
    assert all("ValueError" in f["type"] for f in stats.failures)

@pytest.mark.asyncio
async def test_run_stochastic_tests_timeout():
    """Test running stochastic tests with timeout."""
    async def slow_test() -> bool:
        await asyncio.sleep(0.5)
        return True

    # Test statistics
    stats = StochasticTestStats()

    # Run the tests with a short timeout
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

    # Verify stats
    assert stats.total_runs == 1
    assert stats.successful_runs == 0
    assert len(stats.failures) == 1
    assert "TimeoutError" in stats.failures[0]["type"]

@pytest.mark.asyncio
async def test_run_stochastic_tests_retries():
    """Test retry functionality with async function."""
    # Create a counter to track retries
    attempts = {'value': 0}

    async def test_with_retries() -> bool:
        attempts['value'] += 1
        if attempts['value'] < 3:  # Fail twice then succeed
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
    """Test that batch_size correctly limits concurrency."""
    # Simplify the test to just check that the batching works correctly
    # using a single test run with 5 samples and batch_size=2

    # Create an array of results to track when tests start and end
    # Using a class to ensure we have proper synchronization
    class TestTracker:
        def __init__(self):
            self.active_count = 0
            self.max_active = 0

        async def run_test(self) -> bool:
            # Track the number of active tests
            self.active_count += 1
            self.max_active = max(self.max_active, self.active_count)

            # Sleep to simulate work
            await asyncio.sleep(0.1)

            # Decrement active count
            self.active_count -= 1
            return True

    tracker = TestTracker()
    stats = StochasticTestStats()

    # Run a batch of tests with batch_size=2
    await _run_stochastic_tests(
        testfunction=tracker.run_test,
        funcargs={},
        stats=stats,
        samples=5,  # Run 5 samples
        batch_size=2,  # Max 2 concurrent
        retry_on=None,
        max_retries=1,
        timeout=None,
    )

    # Verify the statistics
    assert stats.total_runs == 5
    assert stats.successful_runs == 5
    assert len(stats.failures) == 0

    # Check that we never had more than 2 tests running at once
    assert tracker.max_active <= 2, f"Expected max 2 concurrent tests, got {tracker.max_active}"

def test_async_batch_processing():
    """Test that batch_size correctly limits concurrency in run_stochastic_tests_for_async."""
    import asyncio
    from sik_stochastic_tests.plugin import run_stochastic_tests_for_async

    # Create a class to track concurrent test execution
    class AsyncTestTracker:
        def __init__(self):
            self.active_count = 0
            self.max_active = 0
            self.execution_order = []
            self._lock = asyncio.Lock()

        async def test_func(self) -> bool:
            async with self._lock:
                self.active_count += 1
                self.max_active = max(self.max_active, self.active_count)
                self.execution_order.append(f"start_{self.active_count}")

            # Simulate work with a small delay
            await asyncio.sleep(0.05)

            async with self._lock:
                self.execution_order.append(f"end_{self.active_count}")
                self.active_count -= 1

            return True

    # Test with batch_size=2
    tracker = AsyncTestTracker()
    stats = StochasticTestStats()

    # Run the tests with batch_size=2
    run_stochastic_tests_for_async(
        testfunction=tracker.test_func,
        funcargs={},
        stats=stats,
        samples=6,  # Run 6 samples
        batch_size=2,  # Max 2 concurrent
        retry_on=None,
        max_retries=1,
        timeout=None,
    )

    # Verify the statistics
    assert stats.total_runs == 6
    assert stats.successful_runs == 6
    assert len(stats.failures) == 0

    # Check that we never had more than 2 tests running at once
    assert tracker.max_active <= 2, f"Expected max 2 concurrent tests, got {tracker.max_active}"

    # Test with no batch_size (should run all concurrently)
    tracker2 = AsyncTestTracker()
    stats2 = StochasticTestStats()

    # Run the tests with no batch_size
    run_stochastic_tests_for_async(
        testfunction=tracker2.test_func,
        funcargs={},
        stats=stats2,
        samples=6,  # Run 6 samples
        batch_size=None,  # No batching
        retry_on=None,
        max_retries=1,
        timeout=None,
    )

    # Verify the statistics
    assert stats2.total_runs == 6
    assert stats2.successful_runs == 6
    assert len(stats2.failures) == 0

    # With no batching, we should see more concurrency
    # Note: This may not always be true depending on exact timing, so we use a higher number
    # to account for possible event loop scheduling variations
    assert tracker2.max_active > 2, f"Expected higher concurrency without batching, got {tracker2.max_active}"  # noqa: E501

@pytest.mark.asyncio
async def test_edge_cases_parameter_validation():
    """Test parameter validation for edge cases in stochastic functions."""
    # Simple test function that always succeeds
    async def test_func() -> bool:
        return True

    stats = StochasticTestStats()

    # Test negative samples validation
    with pytest.raises(ValueError, match="samples must be a positive integer"):
        await _run_stochastic_tests(
            testfunction=test_func,
            funcargs={},
            stats=stats,
            samples=-1,  # Invalid negative samples
            batch_size=None,
            retry_on=None,
            max_retries=1,
            timeout=None,
        )

    # Test zero samples validation
    with pytest.raises(ValueError, match="samples must be a positive integer"):
        await _run_stochastic_tests(
            testfunction=test_func,
            funcargs={},
            stats=stats,
            samples=0,  # Invalid zero samples
            batch_size=None,
            retry_on=None,
            max_retries=1,
            timeout=None,
        )

    # Test invalid retry_on parameter
    with pytest.raises(ValueError, match="retry_on must contain only exception types"):
        await _run_stochastic_tests(
            testfunction=test_func,
            funcargs={},
            stats=stats,
            samples=1,
            batch_size=None,
            retry_on=["not_an_exception", ValueError],  # Invalid retry_on with a string
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

    # Test the same validations for run_stochastic_tests_for_async
    with pytest.raises(ValueError, match="samples must be a positive integer"):
        run_stochastic_tests_for_async(
            testfunction=test_func,
            funcargs={},
            stats=stats,
            samples=-1,
            batch_size=None,
            retry_on=None,
            max_retries=1,
            timeout=None,
        )

    with pytest.raises(ValueError, match="retry_on must contain only exception types"):
        run_stochastic_tests_for_async(
            testfunction=test_func,
            funcargs={},
            stats=stats,
            samples=1,
            batch_size=None,
            retry_on=["not_an_exception"],
            max_retries=1,
            timeout=None,
        )

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
    run_stochastic_tests_for_async(
        testfunction=tracker.always_fail,
        funcargs={},
        stats=stats,
        samples=1,
        batch_size=None,
        retry_on=[ValueError],  # Make ValueError retriable
        max_retries=3,  # Allow 3 attempts
        timeout=None,
    )

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
