# sik-stochastic-tests

[This package was written mostly by Claude Code via Sonnet 3.7]

A pytest plugin for testing non-deterministic systems such as LLMs, running tests multiple times with configurable sample sizes and success thresholds to establish reliable results despite occasional random failures.

## Overview

When testing non-deterministic systems such as large language models, traditional pass/fail testing is problematic because of sporadic errors or response inconsistencies. This plugin allows you to run tests multiple times and determine success based on a threshold, ensuring your tests are reliable even with occasional random failures.

## Features

- Run tests multiple times with a single decorator
- Set success thresholds to allow for occasional failures
- Batch test execution for performance optimization
- Retry capability for flaky tests
- Timeout control for long-running tests
- Detailed reporting of success rates and failure patterns
- Full support for both synchronous and asynchronous tests

## Installation

```bash
pip install sik-stochastic-tests
```

Or with uv:

```bash
uv add sik-stochastic-tests
```

## Usage

### Basic Usage

Mark any test with the `stochastic` decorator to run it multiple times with a success threshold:

```python
import pytest

@pytest.mark.stochastic(samples=5)  # Run 5 times with default 50% success threshold
def test_llm_response():
    response = my_llm.generate("What is the capital of France?")
    assert 'paris' in response.lower()
```

By default, the test will pass if at least 50% of the runs succeed (3 out of 5 in this example). This is ideal for testing non-deterministic systems where occasional failures are expected and acceptable.

### Async Tests

For asynchronous tests, you must use **both** the `@pytest.mark.asyncio` and `@pytest.mark.stochastic` decorators (you will need `pytest-asyncio` installed):

```python
import pytest

@pytest.mark.asyncio  # Required for async tests
@pytest.mark.stochastic(samples=5)  # Uses default 50% threshold
async def test_async_llm_response():
    response = await my_async_llm.generate("What is the capital of France?")
    assert 'paris' in response.lower()
```

### Setting a Success Threshold

You can specify what percentage of runs must succeed for the test to pass:

```python
@pytest.mark.stochastic(samples=10, threshold=0.8)  # 80% must pass
def test_with_threshold():
    # This test will pass if at least 8 out of 10 runs succeed
    result = random_function()
    assert result > 0
```

### Retrying Flaky Tests

Specify which exceptions should trigger retries:

```python
@pytest.mark.stochastic(
    samples=5,
    retry_on=[ConnectionError, TimeoutError],
    max_retries=3
)
def test_with_retry():
    response = api_call()  # Might occasionally fail with connection issues
    assert response.status_code == 200
```

### Handling Timeouts

Set a timeout for long-running tests:

```python
# For synchronous tests
@pytest.mark.stochastic(samples=3, timeout=5.0)  # 5 second timeout
def test_with_timeout():
    result = long_running_operation()
    assert result.is_valid

# For asynchronous tests
@pytest.mark.asyncio
@pytest.mark.stochastic(samples=3, timeout=5.0)  # 5 second timeout
async def test_async_with_timeout():
    result = await async_long_running_operation()
    assert result.is_valid
```

### Batch Processing

Control concurrency with batch processing:

```python
# Works for both sync and async tests
@pytest.mark.stochastic(samples=20, batch_size=5)  # Run 5 at a time
def test_with_batching():
    result = my_operation()
    assert result.success

@pytest.mark.asyncio
@pytest.mark.stochastic(samples=20, batch_size=5)  # Run 5 at a time
async def test_async_with_batching():
    result = await async_operation()
    assert result.success
```

### Disabling Stochastic Mode

Temporarily disable stochastic behavior with a command-line flag:

```bash
pytest --disable-stochastic
```

This will run each test only once, ignoring the stochastic parameters for both sync and async tests.

## Parameter Defaults

The stochastic marker accepts the following parameters with these defaults:

```python
@pytest.mark.stochastic(
    samples=10,       # Run 10 times by default
    threshold=0.5,    # 50% success rate required by default
    batch_size=None,  # Run all samples concurrently by default
    retry_on=None,    # No automatic retries by default
    max_retries=3,    # Maximum 3 retries if retry_on is specified
    timeout=None      # No timeout by default
)
```

You only need to specify the parameters you want to change from these defaults.

## Advanced Examples

### Testing LLM Outputs

```python
@pytest.mark.stochastic(samples=10, threshold=0.7)
def test_llm_instruction_following():
    prompt = "Write a haiku about programming"
    response = llm.generate(prompt)
    
    # Test passes if at least 70% of responses contain all these criteria
    assert len(response.split("\n")) == 3, "Should have 3 lines"
    assert "program" in response.lower(), "Should mention programming"
    
    # Count syllables (simplified)
    lines = response.split("\n")
    syllable_counts = [count_syllables(line) for line in lines]
    assert syllable_counts == [5, 7, 5], f"Should follow 5-7-5 pattern, got {syllable_counts}"
```

### Testing with External APIs (Async)

```python
@pytest.mark.asyncio  # Required for async tests
@pytest.mark.stochastic(
    samples=5, 
    threshold=0.8,
    retry_on=[requests.exceptions.RequestException],
    max_retries=3,
    timeout=10.0
)
async def test_weather_api():
    response = await fetch_weather("New York")
    
    # Basic schema validation
    assert "temperature" in response
    assert "humidity" in response
    assert "wind_speed" in response
    
    # Reasonable values check
    assert -50 <= response["temperature"] <= 50  # Celsius
    assert 0 <= response["humidity"] <= 100      # Percentage
```

### Combining Multiple Features

```python
# Synchronous example with multiple features
@pytest.mark.stochastic(
    samples=20,
    threshold=0.9,
    batch_size=5,
    retry_on=[ConnectionError, TimeoutError],
    max_retries=2,
    timeout=3.0
)
def test_complex_scenario():
    # This test will:
    # - Run 20 times total
    # - Run 5 at a time (batched)
    # - Pass if at least 18 runs succeed (90% threshold)
    # - Retry on connection or timeout errors (up to 2 retries)
    # - Timeout after 3 seconds for each run
    result = complex_operation()
    assert result.is_successful

# Asynchronous equivalent
@pytest.mark.asyncio
@pytest.mark.stochastic(
    samples=20,
    threshold=0.9,
    batch_size=5,
    retry_on=[ConnectionError, TimeoutError],
    max_retries=2,
    timeout=3.0
)
async def test_async_complex_scenario():
    result = await async_complex_operation()
    assert result.is_successful
```

## Test Result Output

The plugin provides detailed test results in the console for both synchronous and asynchronous tests:

```
=========================== Stochastic Test Results ===========================

test_llm.py::test_llm_instruction_following:
  Success rate: 0.80
  Runs: 10, Successes: 8, Failures: 2
  Failure samples:
    1. AssertionError: Should follow 5-7-5 pattern, got [4, 6, 5]
    2. AssertionError: Should mention programming

test_async.py::test_async_weather_api:
  Success rate: 0.60
  Runs: 5, Successes: 3, Failures: 2
  Failure samples:
    1. TimeoutError: Test timed out after 10.0 seconds
    2. AssertionError: 'temperature' not in response
```

## Requirements for Async Tests

1. `pytest-asyncio` must be installed:
   ```bash
   pip install pytest-asyncio
   ```

2. Each async test must have both decorators:
   ```python
   @pytest.mark.asyncio
   @pytest.mark.stochastic(...)
   async def test_something():
       ...
   ```

3. For pytest.ini configuration, you might want to set:
   ```ini
   [pytest]
   asyncio_mode = auto
   ```

## Configuration

You can configure default behavior in your `pyproject.toml` or `pytest.ini` file:

```toml
[tool.pytest.ini_options]
# Set option to exclude stochastic tests from certain environments
addopts = "--disable-stochastic"
```

## Compatibility

- Python 3.11+
- pytest 8.0+
- pytest-asyncio 0.21+ (for async tests)

## Troubleshooting

### Async Tests Not Running Multiple Times

If your async tests are only running once instead of the specified number of samples:

1. Check that you have **both** `@pytest.mark.asyncio` and `@pytest.mark.stochastic` decorators
2. Ensure pytest-asyncio is installed
3. Check that `--disable-stochastic` flag is not present

### Timeout Not Working for Async Tests

Timeouts for async tests are implemented using `asyncio.wait_for()`. If your timeouts aren't working:

1. Ensure you're using a recent version of this plugin
2. Make sure your test is actually running asynchronously

### Known Limitations

#### Synchronous Code Using Async Internally

When testing synchronous code that internally manages its own asyncio event loops (like libraries with synchronous API wrappers over async implementations), you may encounter event loop conflicts with the stochastic plugin:

```python
# This kind of code may cause issues with the stochastic plugin
@pytest.mark.stochastic(samples=3)
def test_problematic():
    # This function looks synchronous but internally creates/manages its own event loop
    result = client(messages=[{"role": "user", "content": "test"}])
    assert result is not None
```

**Error symptoms:**
- `IndexError: pop from an empty deque` 
- `RuntimeError: Event loop is closed`

These errors occur because of conflicts between the event loop management in the stochastic plugin and the event loop created by the synchronous wrapper.

**Workaround:**
Use the async interface directly whenever possible:
```python
@pytest.mark.asyncio
@pytest.mark.stochastic(samples=3)
async def test_better():
    # Using the async API directly avoids the event loop conflict
    result = await client.run_async(messages=[{"role": "user", "content": "test"}])
    assert result is not None
```

This approach avoids the conflict by using a single event loop managed by pytest-asyncio.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---
