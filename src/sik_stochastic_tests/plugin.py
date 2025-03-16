"""Pytest plugin for running stochastic tests."""
import pytest
import asyncio
import inspect
from dataclasses import dataclass, field

# Store test results for reporting
_test_results = {}

@dataclass
class StochasticTestStats:
    """Statistics for a stochastic test."""

    total_runs: int = 0
    successful_runs: int = 0
    failures: list[dict[str, object]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of the test."""
        return self.successful_runs / self.total_runs if self.total_runs > 0 else 0.0

def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers with pytest."""
    config.addinivalue_line(
        "markers",
        "stochastic(samples, threshold, batch_size, retry_on, max_retries, timeout): "
        "mark test to run stochastically multiple times",
    )

@pytest.hookimpl(hookwrapper=True)
def pytest_generate_tests(metafunc):  # noqa
    """Modify test collection to intercept async stochastic tests."""
    yield
    # This hook runs after test generation, allowing us to modify how tests will be executed

def pytest_make_collect_report(collector):  # noqa
    """Hook that runs after collection is complete but before tests are executed."""
    # We can use this to intercept collected tests and modify them
    return

@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items) -> None:  # noqa
    """Modify collected items to handle async stochastic tests."""
    # Skip modification if stochastic mode is disabled
    if config.getoption("--disable-stochastic", False):
        return

    # This runs after all tests are collected
    for item in items:
        # Check if this is a stochastic test and if it's async
        if (hasattr(item, 'get_closest_marker') and  # noqa: SIM102
                item.get_closest_marker('stochastic') and
                hasattr(item, 'obj')):

            # Only modify async tests
            if has_asyncio_marker(item):
                # Get stochastic parameters
                marker = item.get_closest_marker("stochastic")
                samples = marker.kwargs.get('samples', 10)
                threshold = marker.kwargs.get('threshold', 0.5)
                batch_size = marker.kwargs.get('batch_size', None)
                retry_on = marker.kwargs.get('retry_on', None)
                max_retries = marker.kwargs.get('max_retries', 3)
                timeout = marker.kwargs.get('timeout', None)

                # Get the original test function
                original_func = item.obj

                # Create a wrapper that will implement stochastic behavior
                @pytest.mark.asyncio  # Keep the asyncio marker
                async def stochastic_async_wrapper(*args, **kwargs) -> None:  # noqa: ANN002, ANN003, PLR0912
                    # Create stats tracker
                    stats = StochasticTestStats()

                    # Validate inputs
                    if samples <= 0:
                        raise ValueError("samples must be a positive integer")

                    # Create a helper to run a single test sample with retries
                    async def run_single_test(i: int) -> tuple[bool, dict[str, object]]:
                        for attempt in range(max_retries):
                            try:
                                # Run with timeout if specified
                                if timeout is not None:
                                    try:
                                        await asyncio.wait_for(original_func(*args, **kwargs), timeout)  # noqa: E501
                                    except TimeoutError:
                                        raise TimeoutError(f"Test timed out after {timeout} seconds")  # noqa: E501
                                else:
                                    await original_func(*args, **kwargs)

                                # If we get here, test passed
                                return True
                            except Exception as e:
                                # Check if we should retry
                                if retry_on and isinstance(e, tuple(retry_on)) and attempt < max_retries - 1:  # noqa: E501
                                    continue

                                # Final failure
                                failure_info = {
                                    "error": str(e),
                                    "type": type(e).__name__,
                                    "context": {"run_index": i},
                                }
                                return False, failure_info

                        # If we get here, all retries were exhausted
                        return False, {
                            "error": f"Test failed after {max_retries} attempts",
                            "type": "RetryError",
                            "context": {"run_index": i},
                        }

                    # Create tasks for all samples
                    tasks = [run_single_test(i) for i in range(samples)]

                    # Normalize batch size (None or negative becomes "run all at once")
                    effective_batch_size = None
                    if batch_size and batch_size > 0:
                        effective_batch_size = batch_size

                    # Run tests with batching if specified, otherwise all at once
                    if effective_batch_size:
                        # Run in batches
                        for i in range(0, len(tasks), effective_batch_size):
                            batch = tasks[i:i + effective_batch_size]
                            batch_results = await asyncio.gather(*batch, return_exceptions=True)

                            # Process this batch's results
                            for result in batch_results:
                                stats.total_runs += 1
                                if isinstance(result, Exception):
                                    # Unexpected error
                                    stats.failures.append({
                                        "error": str(result),
                                        "type": type(result).__name__,
                                        "context": {"unexpected_error": True},
                                    })
                                elif result is True:
                                    # Success
                                    stats.successful_runs += 1
                                else:
                                    # Expected failure format (False, failure_info)
                                    stats.failures.append(result[1])
                    else:
                        # Run all tests at once
                        all_results = await asyncio.gather(*tasks, return_exceptions=True)

                        # Process all results
                        for result in all_results:
                            stats.total_runs += 1
                            if isinstance(result, Exception):
                                # Unexpected error
                                stats.failures.append({
                                    "error": str(result),
                                    "type": type(result).__name__,
                                    "context": {"unexpected_error": True},
                                })
                            elif result is True:
                                # Success
                                stats.successful_runs += 1
                            else:
                                # Expected failure format (False, failure_info)
                                stats.failures.append(result[1])

                    # Store results for reporting
                    _test_results[item.nodeid] = stats

                    # Check if we met the threshold
                    if stats.success_rate < threshold:
                        message = (
                            f"Stochastic test failed: success rate {stats.success_rate:.2f} below threshold {threshold}\n"  # noqa: E501
                            f"Ran {stats.total_runs} times, {stats.successful_runs} successes, {len(stats.failures)} failures\n"  # noqa: E501
                            f"Failure details: {stats.failures[:5]}" +
                            ("..." if len(stats.failures) > 5 else "")
                        )
                        raise AssertionError(message)

                # Copy needed metadata from original function
                stochastic_async_wrapper.__name__ = original_func.__name__
                stochastic_async_wrapper.__module__ = original_func.__module__
                if hasattr(original_func, '__qualname__'):
                    stochastic_async_wrapper.__qualname__ = original_func.__qualname__

                # Replace the function with our wrapper
                item.obj = stochastic_async_wrapper

def has_asyncio_marker(obj: pytest.Function | object) -> bool:
    """
    Check if an object has pytest.mark.asyncio marker.

    Works for both test functions and test classes.

    Args:
        obj: A pytest function item or any object that might have markers

    Returns:
        True if the object has asyncio marker, False otherwise
    """
    # Check if it's a pytest function with own markers
    if hasattr(obj, 'own_markers'):
        for marker in obj.own_markers:
            if marker.name == 'asyncio':
                return True

    # Check if it's in a class with asyncio marker
    if hasattr(obj, 'cls') and obj.cls is not None:  # noqa: SIM102
        if hasattr(obj.cls, 'pytestmark'):
            for marker in obj.cls.pytestmark:
                if marker.name == 'asyncio':
                    return True

    # Check if the object itself is a class with pytestmark
    if hasattr(obj, '__self__') and obj.__self__ is not None:  # noqa: SIM102
        if hasattr(obj.__self__.__class__, 'pytestmark'):
            for marker in obj.__self__.__class__.pytestmark:
                if marker.name == 'asyncio':
                    return True

    # If none of the above checks detected an asyncio marker but
    # the function is a coroutine function, we'll need to treat it
    # as an asyncio test to avoid errors
    if hasattr(obj, 'obj') and inspect.iscoroutinefunction(obj.obj):  # noqa: SIM103
        return True

    return False

def is_async_function(func: callable) -> bool:
    """Check if a function is asynchronous."""
    return inspect.iscoroutinefunction(func)

def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command-line options."""
    parser.addoption(
        "--disable-stochastic",
        action="store_true",
        help="Disable stochastic mode for tests marked with @stochastic",
    )

@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """Execute stochastic tests."""
    marker = pyfuncitem.get_closest_marker("stochastic")
    if not marker:
        return None  # Not a stochastic test, let pytest handle it

    # Check if stochastic mode is disabled
    if pyfuncitem.config.getoption("--disable-stochastic", False):
        return None  # Run normally

    # For async tests, our collection modification hook already handled it
    if has_asyncio_marker(pyfuncitem):
        return None  # Let pytest-asyncio handle the test with our wrapper

    # Get stochastic parameters from marker
    samples = marker.kwargs.get('samples', 10)
    threshold = marker.kwargs.get('threshold', 0.5)
    batch_size = marker.kwargs.get('batch_size', None)
    retry_on = marker.kwargs.get('retry_on', None)
    max_retries = marker.kwargs.get('max_retries', 3)
    timeout = marker.kwargs.get('timeout', None)

    # Initialize stats
    stats = StochasticTestStats()

    # Get the function to test
    testfunction = pyfuncitem.obj

    # Get only the function arguments that it actually needs
    sig = inspect.signature(testfunction)
    actual_argnames = set(sig.parameters.keys())
    funcargs = {name: arg for name, arg in pyfuncitem.funcargs.items() if name in actual_argnames}

    # Create a new event loop for the test
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Run the stochastic tests
        loop.run_until_complete(
            _run_stochastic_tests(
                testfunction=testfunction,
                funcargs=funcargs,
                stats=stats,
                samples=samples,
                batch_size=batch_size,
                retry_on=retry_on,
                max_retries=max_retries,
                timeout=timeout,
            ),
        )
    finally:
        # Always clean up
        loop.close()
        # Reset the event loop policy to default for pytest-asyncio compatibility
        asyncio.set_event_loop(None)

    # Store the results for reporting
    _test_results[pyfuncitem.nodeid] = stats

    # If we didn't meet the threshold, raise an AssertionError
    if stats.success_rate < threshold:
        message = (
            f"Stochastic test failed: success rate {stats.success_rate:.2f} below threshold {threshold}\n"  # noqa: E501
            f"Ran {stats.total_runs} times, {stats.successful_runs} successes, {len(stats.failures)} failures\n"  # noqa: E501
            f"Failure details: {stats.failures[:5]}" +
            ("..." if len(stats.failures) > 5 else "")
        )
        raise AssertionError(message)

    return True  # We handled this test

async def run_test_function(
    testfunction: callable,
    funcargs: dict[str, object],
    timeout: int | None = None,  # noqa: ASYNC109
) -> object:
    """
    Execute a test function with optional timeout.

    Handles both sync and async functions, and properly awaits coroutines.

    Args:
        testfunction: The test function to execute
        funcargs: Arguments to pass to the test function
        timeout: Optional timeout in seconds

    Returns:
        The result of the function call

    Raises:
        TimeoutError: If the function execution exceeds the timeout
        Any exceptions raised by the test function
    """
    # Determine if the function is asynchronous
    is_async = inspect.iscoroutinefunction(testfunction) or (
        hasattr(testfunction, '__code__') and testfunction.__code__.co_flags & 0x80
    )

    # Create a wrapper to execute the function
    if is_async:
        # For async functions
        if timeout is not None:
            try:
                return await asyncio.wait_for(testfunction(**funcargs), timeout=timeout)
            except TimeoutError:
                raise TimeoutError(f"Test timed out after {timeout} seconds")
        else:
            return await testfunction(**funcargs)
    else:  # noqa: PLR5501
        # For sync functions
        if timeout is not None:
            # Run in executor with timeout
            def run_sync():  # noqa: ANN202
                return testfunction(**funcargs)

            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(None, run_sync)
            try:
                return await asyncio.wait_for(future, timeout=timeout)
            except TimeoutError:
                raise TimeoutError(f"Test timed out after {timeout} seconds")
        else:
            # Direct call for sync functions
            result = testfunction(**funcargs)
            # Handle the case where a sync function returns a coroutine
            if inspect.isawaitable(result):
                return await result
            return result

def run_stochastic_tests_for_async(  # noqa: PLR0915
        testfunction: callable,
        funcargs: dict[str, object],
        stats: StochasticTestStats,
        samples: int,
        batch_size: int | None,
        retry_on: tuple[Exception] | list[type[Exception]] | None,
        max_retries: int = 3,
        timeout: int | None = None,
    ) -> None:
    """
    Run stochastic tests for async tests without creating a new event loop.

    Args:
        testfunction: The async test function to execute
        funcargs: Arguments to pass to the test function
        stats: Statistics object to track test results
        samples: Number of times to run the test (must be positive)
        batch_size: Number of tests to run concurrently (None or <= 0 runs all at once)
        retry_on: Exception types that should trigger a retry
        max_retries: Maximum number of retry attempts per test
        timeout: Maximum time in seconds for each test execution

    Raises:
        ValueError: If samples is not positive or if retry_on contains non-exception types
    """
    # Validate inputs
    if samples <= 0:
        raise ValueError("samples must be a positive integer")

    # Validate retry_on contains only exception types
    if retry_on is not None:
        for exc in retry_on:
            if not isinstance(exc, type) or not issubclass(exc, Exception):
                raise ValueError(f"retry_on must contain only exception types, got {exc}")

    # Normalize batch_size
    if batch_size is not None and batch_size <= 0:
        batch_size = None  # Treat negative or zero batch_size as None

    # Create a coroutine that will apply timeout logic and capture exceptions
    async def run_test_with_timeout(run_index: int, attempt: int) -> tuple[bool, object]:
        """Helper to run a single test with timeout handling."""
        context = {"run_index": run_index}

        try:
            # Create a copy of the function args for this run
            run_args = funcargs.copy()

            # Run the test function with proper timeout
            if timeout is not None:
                try:
                    # Apply timeout using asyncio.wait_for
                    await asyncio.wait_for(testfunction(**run_args), timeout=timeout)
                except TimeoutError:
                    # Convert asyncio's TimeoutError to our TimeoutError
                    raise TimeoutError(f"Test timed out after {timeout} seconds")
            else:
                # No timeout, just run the test
                await testfunction(**run_args)

            # If we get here, the test passed
            return True, None
        except Exception as e:
            # Check if we should retry this exception
            if retry_on and isinstance(e, tuple(retry_on)) and attempt < max_retries - 1:
                # Signal that we should retry
                return False, "retry"
            else:  # noqa: RET505
                # Return the exception for recording in stats
                return False, (str(e), type(e).__name__, context)

    # Get the current loop, or create a new one to handle deprecation
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop, create a new one
        loop = asyncio.new_event_loop()
        # Set it as the current event loop
        asyncio.set_event_loop(loop)

    # Create tasks to run in parallel
    async def run_single_sample(i: int) -> tuple[bool, tuple | None]:
        # For each sample, try up to max_retries times
        for attempt in range(max_retries):
            success, result = await run_test_with_timeout(i, attempt)

            if success:
                # Test passed
                return True, None
            if result == "retry":
                # Retry the test if we haven't exhausted all attempts
                if attempt < max_retries - 1:
                    continue
                # Otherwise, mark as a retry exhaustion
                return False, (f"Test failed after {max_retries} attempts", "RetryError", {"run_index": i})  # noqa: E501
            # Test failed, return the failure details
            return False, result
        # If we exhausted all retries (defensive programming - this line should never be reached)
        return False, (f"Test failed after {max_retries} attempts", "RetryError", {"run_index": i})

    # Run tasks with optional batching
    async def run_full_test() -> None:  # noqa: PLR0912
        tasks = [run_single_sample(i) for i in range(samples)]

        # Run in batches if batch_size is specified
        if batch_size and batch_size > 0:
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)

                # Process results for this batch
                for result in batch_results:
                    if isinstance(result, Exception):
                        # Handle unexpected exceptions
                        stats.total_runs += 1
                        stats.failures.append({
                            "error": str(result),
                            "type": type(result).__name__,
                            "context": {"unexpected_error": True},
                        })
                    else:
                        # Unpack the success flag and result
                        success, error_info = result
                        stats.total_runs += 1

                        if success:
                            stats.successful_runs += 1
                        else:
                            # Add the failure details
                            error_msg, error_type, context = error_info
                            stats.failures.append({
                                "error": error_msg,
                                "type": error_type,
                                "context": context,
                            })
        else:
            # Run all tests at once
            all_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process all results
            for result in all_results:
                if isinstance(result, Exception):
                    # Handle unexpected exceptions
                    stats.total_runs += 1
                    stats.failures.append({
                        "error": str(result),
                        "type": type(result).__name__,
                        "context": {"unexpected_error": True},
                    })
                else:
                    # Unpack the success flag and result
                    success, error_info = result
                    stats.total_runs += 1

                    if success:
                        stats.successful_runs += 1
                    else:
                        # Add the failure details
                        error_msg, error_type, context = error_info
                        stats.failures.append({
                            "error": error_msg,
                            "type": error_type,
                            "context": context,
                        })

    # Run the full test synchronously
    loop.run_until_complete(run_full_test())

async def _run_stochastic_tests(
        testfunction: callable,
        funcargs: dict[str, object],
        stats: StochasticTestStats,
        samples: int,
        batch_size: int | None,
        retry_on: tuple[Exception] | list[type[Exception]] | None,
        max_retries: int = 3,
        timeout: int | None = None,  # noqa: ASYNC109
    ) -> list[bool]:
    """
    Run tests multiple times and collect statistics.

    Args:
        testfunction: The test function to execute (can be sync or async)
        funcargs: Arguments to pass to the test function
        stats: Statistics object to track test results
        samples: Number of times to run the test (must be positive)
        batch_size: Number of tests to run concurrently (None or <= 0 runs all at once)
        retry_on: Exception types that should trigger a retry
        max_retries: Maximum number of retry attempts per test
        timeout: Maximum time in seconds for each test execution

    Returns:
        A list of boolean results indicating success/failure for each sample

    Raises:
        ValueError: If samples is not positive or if retry_on contains non-exception types
    """
    # Validate inputs
    if samples <= 0:
        raise ValueError("samples must be a positive integer")

    # Validate retry_on contains only exception types
    if retry_on is not None:
        for exc in retry_on:
            if not isinstance(exc, type) or not issubclass(exc, Exception):
                raise ValueError(f"retry_on must contain only exception types, got {exc}")

    # Normalize batch_size
    if batch_size is not None and batch_size <= 0:
        batch_size = None  # Treat negative or zero batch_size as None
    async def run_single_test(run_index: int) -> bool:
        context = {"run_index": run_index}
        for attempt in range(max_retries):
            try:
                # Helper function to execute a test function with optional timeout
                await run_test_function(testfunction, funcargs, timeout)

                stats.total_runs += 1
                stats.successful_runs += 1
                return True
            except Exception as e:
                if retry_on and isinstance(e, tuple(retry_on)) and attempt < max_retries - 1:
                    continue

                # Always increment total_runs and record failure on the final attempt
                stats.total_runs += 1
                stats.failures.append({
                    "error": str(e),
                    "type": type(e).__name__,
                    "context": context,
                })
                return False
        raise AssertionError(f"Test failed after {max_retries} attempts")

    # Run tests in batches with error handling
    tasks = [run_single_test(i) for i in range(samples)]

    async def safe_gather(coroutines: list[asyncio.Task], batch_size: int | None) -> list[bool]:
        """Safely gather coroutines with optional batching and error handling."""
        if not coroutines:
            return []

        if batch_size and batch_size > 0:  # Ensure batch_size is positive
            results = []
            for i in range(0, len(coroutines), batch_size):
                batch = coroutines[i:i + batch_size]
                try:
                    batch_results = await asyncio.gather(*batch, return_exceptions=True)
                    # Handle exceptions
                    for result in batch_results:
                        if isinstance(result, Exception):
                            # Log the exception but treat as a failed test
                            print(f"Exception during test execution: {result}")
                            results.append(False)
                        else:
                            results.append(result)
                except Exception as e:
                    # This should rarely happen, but just in case
                    print(f"Unexpected error during batch execution: {e}")
                    results.extend([False] * len(batch))
            return results
        else:  # noqa: RET505
            # If no batch size or invalid batch size, run all at once with error handling
            results = await asyncio.gather(*coroutines, return_exceptions=True)
            return [r if not isinstance(r, Exception) else False for r in results]

    return await safe_gather(tasks, batch_size)


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter) -> None:  # noqa: ANN001
    """Add stochastic test results to terminal summary."""
    if _test_results:
        terminalreporter.write_sep("=", "Stochastic Test Results")
        for nodeid, stats in _test_results.items():
            terminalreporter.write_line(f"\n{nodeid}:")
            terminalreporter.write_line(f"  Success rate: {stats.success_rate:.2f}")
            terminalreporter.write_line(f"  Runs: {stats.total_runs}, Successes: {stats.successful_runs}, Failures: {len(stats.failures)}")  # noqa: E501
            if stats.failures:
                terminalreporter.write_line("  Failure samples:")
                for i, failure in enumerate(stats.failures[:3]):
                    terminalreporter.write_line(f"    {i+1}. {failure['type']}: {failure['error']}")  # noqa: E501
                if len(stats.failures) > 3:
                    terminalreporter.write_line(f"    ... and {len(stats.failures) - 3} more")
