"""Unit tests for the stochastic plugin components."""
from pathlib import Path
import pytest
import asyncio
import sys
import subprocess
from unittest.mock import MagicMock
from sik_stochastic_tests.plugin import (
    StochasticTestStats,
    _run_stochastic_tests,
    pytest_pyfunc_call,
)

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
    """Test retry functionality."""
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
    # Use a list to track when tests are running
    execution_order = []

    async def test_with_delay(index) -> bool:  # noqa: ANN001
        execution_order.append(f"start_{index}")
        await asyncio.sleep(0.1)  # Small delay
        execution_order.append(f"end_{index}")
        return True

    # Run tests with different indices
    tasks = []
    # Create different test functions with different indices
    for i in range(5):
        tasks.append(
            _run_stochastic_tests(
                testfunction=lambda index=i: test_with_delay(index),
                funcargs={},
                stats=StochasticTestStats(),
                samples=1,
                batch_size=2,  # Run max 2 at a time
                retry_on=None,
                max_retries=1,
                timeout=None,
            ),
        )

    await asyncio.gather(*tasks)
    # Check execution pattern - with batch_size=2, we should see at most 2 "start" before an "end"
    starts_before_end = 0
    for event in execution_order:
        if event.startswith("start_"):
            starts_before_end += 1
        else:
            starts_before_end -= 1
        # Should never have more than 2 concurrent starts
        assert starts_before_end <= 2

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

def test_stochastic_failure_threshold(example_test_dir: Path):
    """Test that stochastic tests fail if below threshold."""
    # Create a file to track failures
    counter_file = example_test_dir / "failure_count.txt"
    counter_file.write_text("0")

    # Create a test file with failures - using a more reliable failure pattern
    test_file = example_test_dir / "test_failures.py"
    test_file.write_text("""
import pytest

# File to count executions
COUNT_FILE = "failure_count.txt"
import logging

# Configure logging
logging.basicConfig(filename='test_log.txt', level=logging.DEBUG)

@pytest.mark.stochastic(samples=5, threshold=0.9)  # Require 90% success
def test_with_failures():
    # Read and increment counter

    with open(COUNT_FILE, "r") as f:
        count = int(f.read())
    logging.debug(f"Test execution count: {count}")

    with open(COUNT_FILE, "w") as f:
        f.write(str(count + 1))

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

    # Check if the test failed
    assert "1 failed" in result.stdout
    # Verify Success rate is mentioned in output
    assert "Success rate" in result.stdout

    # Verify the test ran the expected number of times
    with open(counter_file) as f:
        count = int(f.read())

    assert count == 5, f"Test should run 5 times, but ran {count} times"

def test_stochastic_failure_threshold_2(example_test_dir: Path):
    """Test that stochastic tests fail if below threshold."""
    # Create a file that will control which tests fail
    failure_control = example_test_dir / "failure_control.py"
    failure_control.write_text("""
# This file controls which tests should fail
FAIL_COUNT = 2  # Make 2 out of 5 tests fail
""")

    counter_file = example_test_dir / "counter.txt"
    counter_file.write_text("0")

    # Create a test file that uses the control file
    test_file = example_test_dir / "test_failures.py"
    test_file.write_text("""
import pytest
import importlib.util

# Import the failure control configuration
spec = importlib.util.spec_from_file_location("failure_control", "failure_control.py")
failure_control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(failure_control)

@pytest.mark.stochastic(samples=5, threshold=0.7)  # Require 70% success
def test_with_failures():
    # Read and increment counter
    with open("counter.txt", "r") as f:
        count = int(f.read())

    count += 1
    with open("counter.txt", "w") as f:
        f.write(str(count))

    # Fail for the specified number of tests
    if count <= failure_control.FAIL_COUNT:
        pytest.fail(f"Intentional failure {count}")

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

def test_disable_stochastic_flag(example_test_dir: Path):
    """Test that --disable-stochastic flag makes tests run only once."""
    # Create a counter file
    counter_file = example_test_dir / "disable_counter.txt"
    counter_file.write_text("0")

    # Create a test file
    test_file = example_test_dir / "test_disable.py"
    test_file.write_text("""
import pytest

@pytest.mark.stochastic(samples=5)
def test_disable_flag():
    with open("disable_counter.txt", "r") as f:
        count = int(f.read())

    with open("disable_counter.txt", "w") as f:
        f.write(str(count + 1))

    assert True
""")

    # Run pytest with --disable-stochastic flag
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v", "--disable-stochastic"],
        capture_output=True,
        text=True,
        cwd=example_test_dir, check=False,
    )

    # Check if the test passed
    assert "1 passed" in result.stdout

    # Verify the counter - test should have run only once
    with open(counter_file) as f:
        count = int(f.read())

    assert count == 1, f"Test should run once with --disable-stochastic, but ran {count} times"

def test_stochastic_timeout(example_test_dir: Path):
    """Test that timeout functionality works."""
    # Create a test file with a slow test
    test_file = example_test_dir / "test_timeout.py"
    test_file.write_text("""
import pytest
import time

@pytest.mark.stochastic(samples=3, timeout=0.1)
def test_timeout():
    # Sleep longer than the timeout
    time.sleep(0.5)
    assert True
""")

    # Run pytest on the file
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        capture_output=True,
        text=True,
        cwd=example_test_dir, check=False,
    )

    # Test should fail due to timeouts
    assert "1 failed" in result.stdout
    assert "timeout" in result.stdout.lower()


# def test_asyncio_compatibility(example_test_dir: Path):
#     """Test that stochastic works with pytest-asyncio."""
#     # Create a counter file to track executions
#     counter_file = example_test_dir / "asyncio_counter.txt"
#     counter_file.write_text("0")

#     # Create a test file that uses both asyncio and stochastic
#     test_file = example_test_dir / "test_asyncio_stochastic.py"
#     test_file.write_text("""
# import pytest
# import asyncio

# @pytest.mark.asyncio
# class TestAsyncioStochastic:
#     @pytest.mark.stochastic(samples=3)
#     async def test_with_both_decorators(self):
#         # Read counter
#         with open("asyncio_counter.txt", "r") as f:
#             count = int(f.read())
        
#         # Increment counter
#         count += 1
#         with open("asyncio_counter.txt", "w") as f:
#             f.write(str(count))
        
#         # Simulate some async work
#         await asyncio.sleep(0.01)
#         assert True
# """)

#     # Run pytest on the file
#     result = subprocess.run(
#         [sys.executable, "-m", "pytest", str(test_file), "-v"],
#         capture_output=True,
#         text=True,
#         cwd=example_test_dir,
#         check=False,
#     )

#     # Print output for debugging
#     print(f"STDOUT: {result.stdout}")
#     print(f"STDERR: {result.stderr}")

#     # Check if the test passed
#     assert "1 passed" in result.stdout

#     # Verify the counter - test should have run 3 times
#     with open(counter_file) as f:
#         count = int(f.read())

#     assert count == 3, f"Test should run 3 times, but ran {count} times"
