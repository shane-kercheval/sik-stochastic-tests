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

    # Check if test has asyncio marker - if so, let pytest-asyncio handle it
    # This avoids event loop conflicts with pytest-asyncio
    
    # First check if the function itself has the asyncio marker
    has_asyncio_marker = False
    for marker in pyfuncitem.own_markers:
        if marker.name == 'asyncio':
            has_asyncio_marker = True
            break
            
    # Also check if it's in an asyncio class
    if not has_asyncio_marker and hasattr(pyfuncitem, 'cls') and pyfuncitem.cls is not None:
        if hasattr(pyfuncitem.cls, 'pytestmark'):
            for marker in pyfuncitem.cls.pytestmark:
                if marker.name == 'asyncio':
                    has_asyncio_marker = True
                    break
                    
    # If it has asyncio marker, let pytest-asyncio handle it
    if has_asyncio_marker:
        return None

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

async def _run_stochastic_tests(  # noqa: PLR0915
        testfunction: callable,
        funcargs: dict[str, object],
        stats: StochasticTestStats,
        samples: int,
        batch_size: int,
        retry_on: tuple[Exception] | None,
        max_retries: int = 3,
        timeout: int | None = None,  # noqa: ASYNC109
    ) -> list[bool]:
    """Run tests multiple times and collect statistics."""
    async def run_single_test(run_index: int) -> bool:
        context = {"run_index": run_index}
        for attempt in range(max_retries):
            try:
                # Special handling for pytest-asyncio's TestCase class methods
                is_pytest_asyncio_method = False
                if hasattr(testfunction, '__self__') and testfunction.__self__ is not None:
                    if hasattr(testfunction.__self__.__class__, 'pytestmark'):
                        for marker in testfunction.__self__.__class__.pytestmark:
                            if marker.name == 'asyncio':
                                is_pytest_asyncio_method = True
                                break
                
                # Check if the function is a coroutine function
                if inspect.iscoroutinefunction(testfunction):
                    # For async tests
                    if timeout is not None:
                        try:
                            # Simple timeout for async functions
                            await asyncio.wait_for(
                                testfunction(**funcargs), 
                                timeout=timeout
                            )
                        except asyncio.TimeoutError:  # noqa: UP041
                            raise TimeoutError(f"Test timed out after {timeout} seconds")
                    else:
                        # For pytest-asyncio class methods, we need to be extra careful
                        if is_pytest_asyncio_method:
                            # Create a new task instead of directly awaiting
                            task = asyncio.create_task(testfunction(**funcargs))
                            await task
                        else:
                            await testfunction(**funcargs)
                else:  # noqa: PLR5501
                    # For sync tests
                    if timeout is not None:
                        def run_sync() -> None:
                            return testfunction(**funcargs)
                        try:
                            # Use an executor for sync functions
                            loop = asyncio.get_running_loop()
                            future = loop.run_in_executor(None, run_sync)
                            await asyncio.wait_for(future, timeout=timeout)
                        except asyncio.TimeoutError:  # noqa: UP041
                            raise TimeoutError(f"Test timed out after {timeout} seconds")
                    else:
                        # Check for special case - function might be using async def but not awaited
                        if hasattr(testfunction, '__code__') and testfunction.__code__.co_flags & 0x80:
                            # This is likely an async function that wasn't detected - try to await it
                            result = testfunction(**funcargs)
                            if inspect.isawaitable(result):
                                await result  # Await the coroutine if it returns one
                            # Otherwise it's already been handled
                        else:
                            # Direct call for normal sync functions
                            testfunction(**funcargs)

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

        if batch_size:
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
            try:
                return await asyncio.gather(*coroutines, return_exceptions=True)
            except Exception as e:
                print(f"Unexpected error during test execution: {e}")
                return [False] * len(coroutines)

    results = await safe_gather(tasks, batch_size)

    # Handle any exception objects in results
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            print(f"Exception during test execution: {result}")
            processed_results.append(False)
        else:
            processed_results.append(result)

    return processed_results

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
