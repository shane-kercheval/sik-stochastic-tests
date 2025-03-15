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
    # Simplify the test to just check that the batching works correctly
    # using a single test run with 5 samples and batch_size=2
    
    # Create an array of results to track when tests start and end
    # Using a class to ensure we have proper synchronization
    class TestTracker:
        def __init__(self):
            self.active_count = 0
            self.max_active = 0
            
        async def run_test(self):
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

    # Verify the counter - test should have run exactly once
    # because with our fix, we're letting pytest-asyncio handle it directly
    with open(counter_file) as f:
        count = int(f.read())

    assert count == 1, f"Test should run 1 time when delegated to pytest-asyncio, but ran {count} times"
    
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
    
    # Verify the counter - test should run once since we're delegating to pytest-asyncio
    with open(counter_file) as f:
        count = int(f.read())

    assert count == 1, f"Test in asyncio class should run 1 time (not {count}) with our fix that detects and avoids conflict"
