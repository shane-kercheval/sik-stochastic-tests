import pytest
import asyncio
import inspect
from typing import Dict, Any, List, Type, Optional
from dataclasses import dataclass, field
from functools import wraps

# Store test results for reporting
_test_results = {}

@dataclass
class StochasticTestStats:
    total_runs: int = 0
    successful_runs: int = 0
    failures: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        return self.successful_runs / self.total_runs if self.total_runs > 0 else 0.0

def pytest_configure(config):
    """Register custom markers with pytest."""
    config.addinivalue_line(
        "markers", 
        "stochastic(samples, threshold, batch_size, retry_on, max_retries, timeout): "
        "mark test to run stochastically multiple times"
    )

def pytest_addoption(parser):
    """Add command-line options."""
    parser.addoption("--disable-stochastic", action="store_true", 
                    help="Disable stochastic mode for tests marked with @stochastic")

@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Execute stochastic tests."""
    marker = pyfuncitem.get_closest_marker("stochastic")
    if not marker:
        return None  # Not a stochastic test, let pytest handle it
    
    # Check if stochastic mode is disabled
    if pyfuncitem.config.getoption("--disable-stochastic", False):
        return None  # Run normally
    
    # Get stochastic parameters from marker
    samples = marker.kwargs.get("samples", 10)
    threshold = marker.kwargs.get("threshold", 0.5)
    batch_size = marker.kwargs.get("batch_size", None)
    retry_on = marker.kwargs.get("retry_on", None)
    max_retries = marker.kwargs.get("max_retries", 3)
    timeout = marker.kwargs.get("timeout", None)
    
    # Initialize stats
    stats = StochasticTestStats()
    
    # Get the function to test
    testfunction = pyfuncitem.obj
    
    # Get only the function arguments that it actually needs
    sig = inspect.signature(testfunction)
    actual_argnames = set(sig.parameters.keys())
    funcargs = {name: arg for name, arg in pyfuncitem.funcargs.items() if name in actual_argnames}
    
    # Run the test stochastically
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            _run_stochastic_tests(
                testfunction=testfunction,
                funcargs=funcargs,
                stats=stats,
                samples=samples,
                batch_size=batch_size,
                retry_on=retry_on,
                max_retries=max_retries,
                timeout=timeout
            )
        )
    finally:
        loop.close()
    
    # Store the results for reporting
    _test_results[pyfuncitem.nodeid] = stats
    
    # If we didn't meet the threshold, raise an AssertionError
    if stats.success_rate < threshold:
        message = (
            f"Stochastic test failed: success rate {stats.success_rate:.2f} below threshold {threshold}\n"
            f"Ran {stats.total_runs} times, {stats.successful_runs} successes, {len(stats.failures)} failures\n"
            f"Failure details: {stats.failures[:5]}" + 
            ("..." if len(stats.failures) > 5 else "")
        )
        raise AssertionError(message)
    
    return True  # We handled this test

async def _run_stochastic_tests(
    testfunction,
    funcargs,
    stats,
    samples,
    batch_size,
    retry_on,
    max_retries,
    timeout
):
    """Run tests multiple times and collect statistics."""
    async def run_single_test(run_index):
        context = {"run_index": run_index}
        
        for attempt in range(max_retries):
            try:
                if inspect.iscoroutinefunction(testfunction):
                    # For async tests
                    if timeout is not None:
                        async def run_test():
                            return await testfunction(**funcargs)
                        try:
                            await asyncio.wait_for(run_test(), timeout=timeout)
                        except asyncio.TimeoutError:
                            raise TimeoutError(f"Test timed out after {timeout} seconds")
                    else:
                        await testfunction(**funcargs)
                else:
                    # For sync tests
                    if timeout is not None:
                        def run_sync():
                            return testfunction(**funcargs)
                        try:
                            future = asyncio.get_event_loop().run_in_executor(None, run_sync)
                            await asyncio.wait_for(future, timeout=timeout)
                        except asyncio.TimeoutError:
                            raise TimeoutError(f"Test timed out after {timeout} seconds")
                    else:
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
                    "context": context
                })
                return False
    
    # Run tests in batches
    tasks = [run_single_test(i) for i in range(samples)]
    if batch_size:
        results = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
    else:
        results = await asyncio.gather(*tasks)
    
    return results

@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter):
    """Add stochastic test results to terminal summary."""
    if _test_results:
        terminalreporter.write_sep("=", "Stochastic Test Results")
        for nodeid, stats in _test_results.items():
            terminalreporter.write_line(f"\n{nodeid}:")
            terminalreporter.write_line(f"  Success rate: {stats.success_rate:.2f}")
            terminalreporter.write_line(f"  Runs: {stats.total_runs}, Successes: {stats.successful_runs}, Failures: {len(stats.failures)}")
            if stats.failures:
                terminalreporter.write_line("  Failure samples:")
                for i, failure in enumerate(stats.failures[:3]):
                    terminalreporter.write_line(f"    {i+1}. {failure['type']}: {failure['error']}")
                if len(stats.failures) > 3:
                    terminalreporter.write_line(f"    ... and {len(stats.failures) - 3} more")