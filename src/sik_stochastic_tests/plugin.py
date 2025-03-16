"""
Pytest plugin for running stochastic tests.

This plugin enables tests to be run multiple times to verify their statistical properties.
It's useful for testing code with inherent randomness or non-deterministic behavior
(e.g., ML models, API calls with variable responses, race conditions).

Key features:
- Run tests multiple times and report aggregate statistics
- Set success thresholds for flaky tests
- Concurrent execution for speed with configurable batch sizes
- Automatic retry on specified exceptions
- Support for both synchronous and asynchronous tests
- Configurable timeouts per test
"""
import pytest
import asyncio
import inspect
from dataclasses import dataclass, field

# Global dictionary to store test results for reporting at the end of the test session
# Key: test nodeid, Value: statistics for that test
_test_results = {}

@dataclass
class StochasticTestStats:
    """
    Statistics tracker for stochastic test runs.

    This class collects metrics about test runs including total attempts,
    successful runs, and detailed failure information to help diagnose
    issues with non-deterministic tests.
    """

    total_runs: int = 0
    successful_runs: int = 0
    failures: list[dict[str, object]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """
        Calculate the success rate of the test.

        Returns:
            The proportion of successful runs (0.0 to 1.0).
            Returns 0.0 if no tests have been run yet.
        """
        return self.successful_runs / self.total_runs if self.total_runs > 0 else 0.0

def pytest_configure(config: pytest.Config) -> None:
    """
    Register the stochastic marker with pytest.

    This hook is called during pytest startup to register the custom marker
    so it can be used in test files without generating warnings.
    """
    config.addinivalue_line(
        "markers",
        "stochastic(samples, threshold, batch_size, retry_on, max_retries, timeout): "
        "mark test to run stochastically multiple times",
    )

@pytest.hookimpl(hookwrapper=True)
def pytest_generate_tests(metafunc):  # noqa
    """
    Hook for test collection phase.

    This is a placeholder hook that could be used for future parametrization needs.
    Currently, we only use the yield to ensure execution order.
    """
    yield
    # This hook runs after test generation, allowing us to modify how tests will be executed

def pytest_make_collect_report(collector):  # noqa
    """
    Hook that runs after collection is complete but before tests are executed.

    This is a placeholder for potential collection modifications.
    """
    # We can use this to intercept collected tests and modify them
    return

@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items) -> None:  # noqa
    """
    Modify collected test items to handle async stochastic tests.

    This hook intercepts async test functions marked with @stochastic and
    wraps them with our stochastic execution logic. We now use the same
    thread-based approach for both sync and async tests, but the preparation
    is different because async tests need special handling with pytest-asyncio.

    Args:
        session: The pytest session
        config: The pytest config object
        items: List of collected test items to be modified
    """
    # Skip modification if stochastic mode is disabled via the command line
    if config.getoption("--disable-stochastic", False):
        return

    # Process each collected test item
    for item in items:
        # Check if this is a stochastic test with the required attributes
        if (hasattr(item, 'get_closest_marker') and  # noqa: SIM102
                item.get_closest_marker('stochastic') and
                hasattr(item, 'obj')):

            # We only need to pre-process async tests because they need special execution within the event loop  # noqa: E501
            if has_asyncio_marker(item):
                # Get stochastic parameters from the marker - use sensible defaults if not specified  # noqa: E501
                marker = item.get_closest_marker("stochastic")
                samples = marker.kwargs.get('samples', 10)
                threshold = marker.kwargs.get('threshold', 0.5)
                batch_size = marker.kwargs.get('batch_size', None)
                retry_on = marker.kwargs.get('retry_on', None)
                max_retries = marker.kwargs.get('max_retries', 3)
                timeout = marker.kwargs.get('timeout', None)

                # Store the original test function that we'll wrap
                original_func = item.obj

                # Create a wrapper that will implement stochastic behavior for async tests
                #
                # IMPORTANT: The interaction between pytest-asyncio and our plugin requires
                # special handling of function signatures.
                #
                # When pytest-asyncio processes an async test:
                # 1. It wraps the original function with pytest_asyncio.plugin.wrap_in_sync()
                # 2. It calls this wrapped function with the exact fixture values it collected
                # 3. If the test has **kwargs parameters, pytest-asyncio needs to pass them correctly  # noqa: E501
                #
                # Our stochastic wrapper must therefore:
                # 1. Preserve the EXACT parameter signature of the original test function
                # 2. Forward parameters to the original function with the exact same structure
                # 3. Handle special cases like *args and **kwargs correctly
                sig = inspect.signature(original_func)

                # Create a parameter string that exactly matches the original function's signature.
                # This is CRITICAL for proper fixture injection and interaction with pytest-asyncio.  # noqa: E501
                param_list = []
                for name, param in sig.parameters.items():
                    if param.kind == inspect.Parameter.POSITIONAL_ONLY:  # noqa: SIM114
                        param_list.append(f"{name}")
                    elif param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                        param_list.append(f"{name}")
                    elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                        param_list.append(f"*, {name}")
                    elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                        param_list.append(f"*{name}")
                    elif param.kind == inspect.Parameter.VAR_KEYWORD:
                        param_list.append(f"**{name}")

                param_str = ", ".join(param_list)

                # When we call the original function from inside our wrapper, we need to
                # forward all parameters exactly as they were received. This is especially
                # important for *args and **kwargs parameters.
                param_forwarding = []
                for name, param in sig.parameters.items():
                    if param.kind == inspect.Parameter.VAR_POSITIONAL:
                        # For *args parameters, we need to unpack them with *
                        param_forwarding.append(f"*{name}")
                    elif param.kind == inspect.Parameter.VAR_KEYWORD:
                        # For **kwargs parameters, we need a special handling to avoid duplicates
                        # This ensures we don't get "got multiple values for keyword argument" errors  # noqa: E501
                        param_forwarding.append(f"**{{k: v for k, v in {name}.items() if k not in funcargs}}")  # noqa: E501
                    else:
                        # For all other parameters, we pass them as keyword arguments
                        # to ensure they're correctly passed regardless of order
                        param_forwarding.append(f"{name}={name}")

                param_passing = ", ".join(param_forwarding)

                # The exec_context provides all the variables that will be accessible
                # inside our dynamically created wrapper function.
                exec_context = {'pytest': pytest,
                               'original_func': original_func,
                               'samples': samples,
                               'threshold': threshold,
                               'batch_size': batch_size,
                               'retry_on': retry_on,
                               'max_retries': max_retries,
                               'timeout': timeout,
                               'asyncio': asyncio,
                               'StochasticTestStats': StochasticTestStats,
                               'TimeoutError': TimeoutError,
                               'Exception': Exception,
                               '_test_results': _test_results,
                               'item_nodeid': item.nodeid,
                               '_run_stochastic_tests': _run_stochastic_tests,
                               '_run_stochastic_tests_in_thread': _run_stochastic_tests_in_thread,
                               'ValueError': ValueError,  # Common exception types for retry_on
                               'TypeError': TypeError,
                               'AssertionError': AssertionError}

                # Define a more robust wrapper function with a single string
                # to avoid indentation issues in the dynamically generated code
                wrapper_code = f"""
@pytest.mark.asyncio
async def stochastic_async_wrapper({param_str}):
    # Create a dictionary of function arguments
    funcargs = {{}}
"""

                # Add parameter collection code with careful indentation
                for name, param in sig.parameters.items():
                    if param.kind == inspect.Parameter.VAR_POSITIONAL:
                        # We skip *args parameters as they'll be passed directly
                        pass
                    elif param.kind == inspect.Parameter.VAR_KEYWORD:
                        # For **kwargs, we update the funcargs dictionary
                        wrapper_code += f"    funcargs.update({name})\n"
                    else:
                        # For normal parameters, add them to funcargs
                        wrapper_code += f"    funcargs['{name}'] = {name}\n"

                # Add the rest of the function implementation
                wrapper_code += f"""
    # Initialize statistics tracker for this test
    stats = StochasticTestStats()

    # Create a specialized proxy function that handles different parameter kinds correctly
    async def proxy_func(**kwargs):
        # We need to reconstruct the function arguments based on the original signature
        # The key is to make sure we're handling the params consistently
        return await original_func({param_passing})

    # Use the thread-based approach for consistent execution
    import threading
    import queue

    # Use a queue to get results from the thread
    result_queue = queue.Queue()

    def thread_worker():
        try:
            # Create a new event loop for this thread
            thread_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(thread_loop)

            # Add custom exception handler for empty deque errors
            def custom_exception_handler(loop, context):
                exc = context.get('exception')
                if isinstance(exc, IndexError) and str(exc) == "pop from an empty deque":
                    return
                thread_loop.default_exception_handler(context)

            thread_loop.set_exception_handler(custom_exception_handler)

            try:
                # Run stochastic tests in this thread's event loop
                thread_loop.run_until_complete(
                    _run_stochastic_tests(
                        testfunction=proxy_func,
                        funcargs=funcargs,
                        stats=stats,
                        samples={samples},
                        batch_size={batch_size},
                        retry_on=retry_on,  # Pass the retry_on value directly without string formatting
                        max_retries={max_retries},
                        timeout={timeout},
                    )
                )
                result_queue.put(("success", None))
            except Exception as e:
                result_queue.put(("error", e))
            finally:
                thread_loop.close()
                asyncio.set_event_loop(None)
        except Exception as e:
            result_queue.put(("error", e))

    # Run the tests in a dedicated thread
    worker_thread = threading.Thread(target=thread_worker, daemon=True)
    worker_thread.start()
    worker_thread.join()

    # Get results from the thread
    if not result_queue.empty():
        status, exception = result_queue.get()
        if status == "error" and exception is not None:
            raise exception

    # Store test results in the global dictionary for reporting at the end of the session
    _test_results[item_nodeid] = stats

    # Enforce the success threshold
    if stats.success_rate < {threshold}:
        message = (
            f"Stochastic test failed: success rate {{stats.success_rate:.2f}} below threshold {threshold}\\n"
            f"Ran {{stats.total_runs}} times, {{stats.successful_runs}} successes, {{len(stats.failures)}} failures\\n"
            f"Failure details: {{stats.failures[:5]}}" +
            ("..." if len(stats.failures) > 5 else "")
        )
        raise AssertionError(message)
"""  # noqa: E501
                # Execute our dynamically created wrapper function code in a controlled namespace
                local_ns = {}
                exec(wrapper_code, exec_context, local_ns)
                stochastic_async_wrapper = local_ns['stochastic_async_wrapper']

                # Copy metadata from original function to wrapper
                stochastic_async_wrapper.__name__ = original_func.__name__
                stochastic_async_wrapper.__module__ = original_func.__module__
                if hasattr(original_func, '__qualname__'):
                    stochastic_async_wrapper.__qualname__ = original_func.__qualname__

                # The signature is vitally important for fixture resolution
                stochastic_async_wrapper.__signature__ = sig

                # Replace the original test function with our wrapper
                item.obj = stochastic_async_wrapper

def has_asyncio_marker(obj: pytest.Function | object) -> bool:
    """
    Check if an object has pytest.mark.asyncio marker or is asynchronous.

    This function performs comprehensive detection of async tests, checking:
    1. Direct pytest.mark.asyncio markers on the function
    2. Inherited markers from test classes
    3. Native coroutine functions (even without explicit markers)

    This comprehensive detection is essential because we need different
    execution paths for async vs sync stochastic tests.

    Args:
        obj: A pytest function item or any object that might have markers

    Returns:
        True if the object has asyncio marker or is a coroutine, False otherwise
    """
    # Check if it's a pytest function with its own markers
    if hasattr(obj, 'own_markers'):
        for marker in obj.own_markers:
            if marker.name == 'asyncio':
                return True

    # Check if it's in a class with asyncio marker (inheritance case)
    if hasattr(obj, 'cls') and obj.cls is not None:  # noqa: SIM102
        if hasattr(obj.cls, 'pytestmark'):
            for marker in obj.cls.pytestmark:
                if marker.name == 'asyncio':
                    return True

    # Check if the object itself is a class method with pytestmark
    if hasattr(obj, '__self__') and obj.__self__ is not None:  # noqa: SIM102
        if hasattr(obj.__self__.__class__, 'pytestmark'):
            for marker in obj.__self__.__class__.pytestmark:
                if marker.name == 'asyncio':
                    return True

    # Detect coroutine functions even without explicit markers
    # This is needed to handle async functions in pytest-asyncio's auto mode
    if hasattr(obj, 'obj') and inspect.iscoroutinefunction(obj.obj):  # noqa: SIM103
        return True

    return False

def is_async_function(func: callable) -> bool:
    """
    Check if a function is asynchronous (defined with async def).

    This is a simple utility wrapper around inspect.iscoroutinefunction
    that makes the code more readable. Used to determine execution path
    for different function types.

    Args:
        func: The function to check

    Returns:
        True if the function is a coroutine function, False otherwise
    """
    return inspect.iscoroutinefunction(func)

def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Add plugin-specific command-line options to pytest.

    This adds the --disable-stochastic flag that allows users to
    temporarily disable all stochastic test features without removing
    markers from test code.

    Args:
        parser: The pytest command line parser
    """
    parser.addoption(
        "--disable-stochastic",
        action="store_true",
        help="Disable stochastic mode for tests marked with @stochastic",
    )

# Helper function to run stochastic tests in a separate thread
def _run_stochastic_tests_in_thread(
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
    Run stochastic tests in a dedicated thread with its own event loop.

    RACE CONDITION EXPLANATION:
    We encountered a race condition in asyncio where multiple async generators
    being cleaned up simultaneously across different contexts would sometimes
    cause the event loop's _ready deque to become empty unexpectedly, resulting
    in an "IndexError: pop from an empty deque" error. This occurred primarily
    with async tests that use external libraries like OpenAI, httpx, or httpcore,
    which rely heavily on async generators for streaming responses.

    WHY THREAD ISOLATION HELPS:
    In Python, each thread can have its own event loop. By running each stochastic
    test session in a dedicated thread with its own isolated event loop, we prevent
    interference between pytest-asyncio's event loop management and our concurrent
    test executions. This isolation eliminates the race condition by ensuring that
    async generator cleanup operations don't interfere with each other across
    different execution contexts.

    IMPLEMENTATION DETAILS:
    1. We create a dedicated thread for running the test samples
    2. This thread gets its own event loop with a custom exception handler
    3. Test results are safely passed back to the main thread through a queue
    4. The thread's event loop is properly closed and dereferenced when done

    Args:
        testfunction: The test function to execute
        funcargs: Arguments to pass to the test function
        stats: Statistics object to track test results
        samples: Number of times to run the test
        batch_size: Number of tests to run concurrently
        retry_on: Exception types that should trigger a retry
        max_retries: Maximum retry attempts per test
        timeout: Maximum time in seconds per test execution
    """
    import threading
    import queue

    # Use a queue to get results back from the thread
    result_queue = queue.Queue()

    def thread_worker() -> None:
        """
        Run all stochastic test samples in an isolated thread with its own event loop.

        This function is the core of our thread isolation strategy:

        1. It creates a new, isolated event loop for this thread only
        2. Sets up a custom exception handler that specifically ignores the
           "pop from an empty deque" IndexError that can occur during async cleanup
        3. Runs all test samples in this isolated environment
        4. Properly cleans up the event loop to prevent resource leaks
        5. Returns results safely to the main thread via a queue

        This approach resolves the race condition that occurs when async generators
        from packages like httpx/httpcore (used by OpenAI, Anthropic clients) are
        being cleaned up concurrently in the same event loop.
        """
        try:
            # Create a new event loop specific to this thread
            thread_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(thread_loop)

            # Set custom error handler to catch empty deque errors
            def custom_exception_handler(loop, context) -> None:  # noqa: ANN001, ARG001
                exc = context.get('exception')
                if isinstance(exc, IndexError) and str(exc) == "pop from an empty deque":
                    # Silently ignore this specific error
                    return
                # Use default handler for all other errors
                thread_loop.default_exception_handler(context)

            thread_loop.set_exception_handler(custom_exception_handler)


            try:
                # Run all test samples in this thread's event loop
                thread_loop.run_until_complete(
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
                result_queue.put(("success", None))
            except Exception as e:
                # Pass any exceptions back to the main thread
                result_queue.put(("error", e))
            finally:
                # Always clean up the loop
                thread_loop.close()
                # Reset this thread's event loop
                asyncio.set_event_loop(None)
        except Exception as e:
            # Catch any unexpected errors in thread setup
            result_queue.put(("error", e))

    # Create and start the worker thread
    worker_thread = threading.Thread(target=thread_worker, daemon=True)
    worker_thread.start()
    worker_thread.join()  # Wait for thread to complete

    # Get results from the thread
    if not result_queue.empty():
        status, exception = result_queue.get()
        if status == "error" and exception is not None:
            # Re-raise any exceptions that occurred in the thread
            raise exception

@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """
    Custom test execution hook for synchronous stochastic tests.

    This function intercepts the test execution phase for synchronous tests
    marked with @stochastic and runs them multiple times with the specified
    parameters. We use tryfirst to ensure we get priority before other hooks.

    Note: Async tests are handled separately in pytest_collection_modifyitems
    because they need to be wrapped before execution.

    Args:
        pyfuncitem: The pytest function item being executed

    Returns:
        True if we handled the test, None to let pytest handle it normally
    """
    # Check if this test has the stochastic marker
    marker = pyfuncitem.get_closest_marker("stochastic")
    if not marker:
        return None  # Not a stochastic test, let pytest handle it normally

    # Skip stochastic behavior if disabled via command line
    if pyfuncitem.config.getoption("--disable-stochastic", False):
        return None  # Run the test normally (just once)

    # For async tests, we already modified them during collection
    # Let pytest-asyncio handle them with our wrapper in place
    if has_asyncio_marker(pyfuncitem):
        return None

    # Extract stochastic parameters with sensible defaults
    samples = marker.kwargs.get('samples', 10)
    threshold = marker.kwargs.get('threshold', 0.5)
    batch_size = marker.kwargs.get('batch_size', None)
    retry_on = marker.kwargs.get('retry_on', None)
    max_retries = marker.kwargs.get('max_retries', 3)
    timeout = marker.kwargs.get('timeout', None)

    # Initialize statistics tracker for this test
    stats = StochasticTestStats()

    # Get the original test function
    testfunction = pyfuncitem.obj

    # Filter arguments to only pass those the function accepts
    sig = inspect.signature(testfunction)
    actual_argnames = set(sig.parameters.keys())
    funcargs = {name: arg for name, arg in pyfuncitem.funcargs.items() if name in actual_argnames}

    # Run stochastic tests in a separate thread with its own event loop
    # This isolates our tests from pytest-asyncio's event loop, fixing the "pop from empty deque" error  # noqa: E501
    #
    # IMPORTANT: This thread-based approach is critical for preventing race conditions
    # during async generator cleanup. By isolating each stochastic test run in its own
    # thread with a dedicated event loop, we prevent interference with pytest-asyncio's
    # event loop and avoid the "pop from empty deque" error that can occur when multiple
    # async generators are being cleaned up simultaneously in shared execution contexts.
    #
    # This is particularly important for tests that use the OpenAI API, httpx, or other
    # libraries that rely heavily on async generators for streaming responses.
    _run_stochastic_tests_in_thread(
        testfunction=testfunction,
        funcargs=funcargs,
        stats=stats,
        samples=samples,
        batch_size=batch_size,
        retry_on=retry_on,
        max_retries=max_retries,
        timeout=timeout,
    )

    # Store test results for terminal reporting
    _test_results[pyfuncitem.nodeid] = stats

    # Fail if success rate is below threshold
    if stats.success_rate < threshold:
        message = (
            f"Stochastic test failed: success rate {stats.success_rate:.2f} below threshold {threshold}\n"  # noqa: E501
            f"Ran {stats.total_runs} times, {stats.successful_runs} successes, {len(stats.failures)} failures\n"  # noqa: E501
            f"Failure details: {stats.failures[:5]}" +
            ("..." if len(stats.failures) > 5 else "")
        )
        raise AssertionError(message)

    return True  # Signal to pytest that we handled the test execution


async def run_test_function(
    testfunction: callable,
    funcargs: dict[str, object],
    timeout: int | None = None,  # noqa: ASYNC109
) -> object:
    """
    Execute a test function with optional timeout, handling both sync and async functions.

    This is a universal test function executor that correctly handles:
    1. Native async functions (defined with async def)
    2. Synchronous functions (regular functions)
    3. Synchronous functions that return awaitables (coroutines/futures)

    It applies timeouts consistently across all function types and ensures
    proper exception propagation. This is critical for running both sync
    and async tests in a unified stochastic framework.

    Args:
        testfunction: The test function to execute (can be sync or async)
        funcargs: Dictionary of arguments to pass to the test function
        timeout: Optional timeout in seconds after which the test will be cancelled

    Returns:
        The result of the function call

    Raises:
        TimeoutError: If the function execution exceeds the specified timeout
        Any exceptions raised by the test function itself
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
    Internal helper to run tests multiple times with concurrency and statistics collection.

    This is the core implementation shared by both synchronous and asynchronous test paths.
    It handles batched execution, retries, and timeout control in a unified way for both
    sync and async functions.

    Key features:
    - Runs multiple samples concurrently in configurable batch sizes
    - Supports retrying specific exceptions a configurable number of times
    - Automatically handles both sync and async test functions
    - Applies timeout control to prevent hanging tests
    - Tracks detailed statistics on test runs

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
                    for result in batch_results:
                        if isinstance(result, Exception):
                            print(f"Exception during test execution: {result}")
                            results.append(False)
                        else:
                            results.append(result)
                except Exception as e:
                    print(f"Unexpected error during batch execution: {e}")
                    results.extend([False] * len(batch))
            return results
        else:  # noqa: RET505
            # Run all tests at once
            results = await asyncio.gather(*coroutines, return_exceptions=True)
            return [r if not isinstance(r, Exception) else False for r in results]

    return await safe_gather(tasks, batch_size)


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter) -> None:  # noqa: ANN001
    """
    Add stochastic test results to the pytest terminal output.

    This hook runs at the end of the test session and provides a detailed
    summary of all stochastic tests that were executed, including:
    - Success rates for each test
    - Number of total runs, successes, and failures
    - Sample error information for failures

    This summary is valuable for understanding the statistical properties
    of non-deterministic tests and diagnosing intermittent failures.

    We use trylast to ensure this summary appears after other test results.

    Args:
        terminalreporter: The pytest terminal reporter object
    """
    # Only add summary if we have stochastic test results to report
    if _test_results:
        terminalreporter.write_sep("=", "Stochastic Test Results")
        for nodeid, stats in _test_results.items():
            # Write test identifier
            terminalreporter.write_line(f"\n{nodeid}:")
            # Show success rate with 2 decimal precision
            terminalreporter.write_line(f"  Success rate: {stats.success_rate:.2f}")
            # Show overall statistics
            terminalreporter.write_line(f"  Runs: {stats.total_runs}, Successes: {stats.successful_runs}, Failures: {len(stats.failures)}")  # noqa: E501

            # Show sample failures if any exist (up to 3 to avoid overwhelming output)
            if stats.failures:
                terminalreporter.write_line("  Failure samples:")
                for i, failure in enumerate(stats.failures[:3]):
                    terminalreporter.write_line(f"    {i+1}. {failure['type']}: {failure['error']}")  # noqa: E501
                if len(stats.failures) > 3:
                    terminalreporter.write_line(f"    ... and {len(stats.failures) - 3} more")
