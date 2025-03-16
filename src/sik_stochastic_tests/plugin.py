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
    wraps them with our concurrent execution logic. We use trylast to ensure
    we run after other hooks that might modify the test collection.

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

            # We handle async tests differently because they need special execution within the event loop  # noqa: E501
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
                # 1. Maintain the original function signature
                sig = inspect.signature(original_func)
                
                # Create a wrapper using the exact same parameter specification 
                # This is key to ensuring fixtures are properly injected
                print(f"DEBUG - Creating wrapper for {original_func.__name__} with signature {sig}")
                print(f"DEBUG - Original function: {original_func}")
                print(f"DEBUG - Parameters: {list(sig.parameters.keys())}")
                print(f"DEBUG - Parameter kinds: {[(name, param.kind) for name, param in sig.parameters.items()]}")
                print(f"DEBUG - Item.funcargs: {getattr(item, 'funcargs', 'No funcargs')}")
                
                # We need to preserve the EXACT parameter structure of the original function
                # including VAR_POSITIONAL (*args) and VAR_KEYWORD (**kwargs) parameters
                param_list = []
                for name, param in sig.parameters.items():
                    if param.kind == inspect.Parameter.POSITIONAL_ONLY:
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
                print(f"DEBUG - Generated param string: {param_str}")
                
                # Build code to reconstruct the params dict for passing to the original function
                param_reconstruction = []
                has_var_positional = False
                var_positional_name = ""
                has_var_keyword = False
                var_keyword_name = ""
                
                for name, param in sig.parameters.items():
                    if param.kind == inspect.Parameter.VAR_POSITIONAL:
                        has_var_positional = True
                        var_positional_name = name
                    elif param.kind == inspect.Parameter.VAR_KEYWORD:
                        has_var_keyword = True
                        var_keyword_name = name
                
                # This is the key part that preserves the original function signature
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
                               'inspect': inspect,
                               '_test_results': _test_results,  # Add this here
                               'item_nodeid': item.nodeid}

                # Construct param_passing code for calling the original function
                param_passing = ""
                if has_var_positional and has_var_keyword:
                    # Function has both *args and **kwargs
                    regular_params = [name for name, param in sig.parameters.items() 
                                     if param.kind not in (inspect.Parameter.VAR_POSITIONAL, 
                                                          inspect.Parameter.VAR_KEYWORD)]
                    param_passing = ", ".join([f"{name}={name}" for name in regular_params])
                    if param_passing:
                        param_passing += f", *{var_positional_name}, **{var_keyword_name}"
                    else:
                        param_passing = f"*{var_positional_name}, **{var_keyword_name}"
                elif has_var_keyword:
                    # Function has **kwargs only
                    regular_params = [name for name, param in sig.parameters.items() 
                                     if param.kind != inspect.Parameter.VAR_KEYWORD]
                    param_passing = ", ".join([f"{name}={name}" for name in regular_params])
                    if param_passing:
                        param_passing += f", **{var_keyword_name}"
                    else:
                        param_passing = f"**{var_keyword_name}"
                elif has_var_positional:
                    # Function has *args only
                    regular_params = [name for name, param in sig.parameters.items() 
                                     if param.kind != inspect.Parameter.VAR_POSITIONAL]
                    param_passing = ", ".join([f"{name}={name}" for name in regular_params])
                    if param_passing:
                        param_passing += f", *{var_positional_name}"
                    else:
                        param_passing = f"*{var_positional_name}"
                else:
                    # Function has only regular parameters
                    param_passing = ", ".join([f"{name}={name}" for name in sig.parameters.keys()])
                
                print(f"DEBUG - Parameter passing code: {param_passing}")

                # Define the wrapper function preserving parameter names
                wrapper_code = f"""
@pytest.mark.asyncio
async def stochastic_async_wrapper({param_str}):
    print(f"DEBUG - stochastic_async_wrapper called with params types: {{[type(val).__name__ for key, val in locals().items() if key in {list(sig.parameters.keys())}]}}")
    # Create stats tracker for this test run
    stats = StochasticTestStats()

    # Validate inputs to prevent common issues
    if samples <= 0:
        raise ValueError("samples must be a positive integer")

    # Helper function to run a single test sample with retry logic
    async def run_single_test(i: int):
        for attempt in range(max_retries):
            try:
                # Apply timeout if specified to prevent tests from hanging
                if timeout is not None:
                    try:
                        # asyncio.wait_for cancels the task if it exceeds the timeout
                        # Pass parameters directly with the exact same structure as received
                        test_coro = original_func({param_passing})
                        await asyncio.wait_for(test_coro, timeout)
                    except TimeoutError:
                        raise TimeoutError(f"Test timed out after {{timeout}} seconds")
                else:
                    # Run the test with the exact parameters we received
                    print(f"DEBUG - Running original_func with {param_passing}")
                    await original_func({param_passing})

                # If we get here without an exception, the test passed
                return True
            except Exception as e:
                # Check if this exception should trigger a retry
                if retry_on and isinstance(e, tuple(retry_on)) and attempt < max_retries - 1:
                    # Continue to the next retry attempt
                    continue

                # If we're not retrying, record the failure details
                failure_info = {{
                    "error": str(e),
                    "type": type(e).__name__,
                    "context": {{"run_index": i}},
                }}
                return False, failure_info

        # This executes if all retry attempts were exhausted
        return False, {{
            "error": f"Test failed after {{max_retries}} attempts",
            "type": "RetryError",
            "context": {{"run_index": i}},
        }}

    # Create tasks for all samples
    tasks = [run_single_test(i) for i in range(samples)]

    # Normalize batch size
    effective_batch_size = None
    if batch_size and batch_size > 0:
        effective_batch_size = batch_size

    # Run tests with batching if specified, otherwise all at once
    if effective_batch_size:
        # Run in batches to limit concurrency
        for i in range(0, len(tasks), effective_batch_size):
            batch = tasks[i:i + effective_batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)

            # Process results
            for result in batch_results:
                stats.total_runs += 1
                if isinstance(result, Exception):
                    stats.failures.append({{
                        "error": str(result),
                        "type": type(result).__name__,
                        "context": {{"unexpected_error": True}},
                    }})
                elif result is True:
                    stats.successful_runs += 1
                else:
                    stats.failures.append(result[1])
    else:
        # Run all tests at once
        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in all_results:
            stats.total_runs += 1
            if isinstance(result, Exception):
                stats.failures.append({{
                    "error": str(result),
                    "type": type(result).__name__,
                    "context": {{"unexpected_error": True}},
                }})
            elif result is True:
                stats.successful_runs += 1
            else:
                stats.failures.append(result[1])

    # Store test results for final reporting
    _test_results[item_nodeid] = stats

    # Fail the test if success rate is below threshold
    if stats.success_rate < threshold:
        message = (
            f"Stochastic test failed: success rate {{stats.success_rate:.2f}} below threshold {{threshold}}\\n"
            f"Ran {{stats.total_runs}} times, {{stats.successful_runs}} successes, {{len(stats.failures)}} failures\\n"
            f"Failure details: {{stats.failures[:5]}}" +
            ("..." if len(stats.failures) > 5 else "")
        )
        raise AssertionError(message)
"""
                # Create the wrapper function in a controlled namespace
                local_ns = {}
                exec(wrapper_code, exec_context, local_ns)
                stochastic_async_wrapper = local_ns['stochastic_async_wrapper']
                
                # Copy metadata from original function to wrapper to preserve
                # important attributes like name and module for pytest reporting
                stochastic_async_wrapper.__name__ = original_func.__name__
                stochastic_async_wrapper.__module__ = original_func.__module__
                if hasattr(original_func, '__qualname__'):
                    stochastic_async_wrapper.__qualname__ = original_func.__qualname__
                
                # Make sure we carry the required fixtures information
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

    # Create a new event loop for running sync tests via our async helper
    # This is needed because even sync tests use async execution for concurrency
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Execute the sync test using our async test runner
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
        # Always clean up resources even if there are exceptions
        loop.close()
        # Reset the event loop to avoid interference with pytest-asyncio
        asyncio.set_event_loop(None)

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
    Run stochastic tests for async functions using the existing event loop.

    This function handles the parallel execution of async test functions,
    with batching and retry capabilities. It uses the current event loop
    rather than creating a new one, which is more efficient for async tests.

    The key advantage of this function is its ability to control concurrency
    through batching, which helps prevent resource exhaustion when running
    many samples of resource-intensive tests (like network calls or database operations).

    It's used primarily for:
    1. Direct calls from non-pytest code
    2. Tests run through the pytest test execution hook

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

    async def run_test_with_timeout(run_index: int, attempt: int) -> tuple[bool, object]:
        """Helper to run a single test with timeout handling."""
        context = {"run_index": run_index}

        try:
            run_args = funcargs.copy()

            if timeout is not None:
                try:
                    # Filter arguments to only pass those the function accepts
                    sig = inspect.signature(testfunction)
                    actual_argnames = set(sig.parameters.keys())
                    filtered_args = {name: arg for name, arg in run_args.items() if name in actual_argnames}
                    
                    await asyncio.wait_for(testfunction(**filtered_args), timeout=timeout)
                except TimeoutError:
                    raise TimeoutError(f"Test timed out after {timeout} seconds")
            else:
                # Filter arguments to only pass those the function accepts
                sig = inspect.signature(testfunction)
                actual_argnames = set(sig.parameters.keys()) 
                filtered_args = {name: arg for name, arg in run_args.items() if name in actual_argnames}
                
                await testfunction(**filtered_args)

            return True, None
        except Exception as e:
            if retry_on and isinstance(e, tuple(retry_on)) and attempt < max_retries - 1:
                return False, "retry"
            else:  # noqa: RET505
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