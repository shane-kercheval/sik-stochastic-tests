# sik-stochastic-tests

A pytest plugin for testing non-deterministic systems such as LLMs, running tests multiple times with configurable sample sizes and success thresholds to establish reliable results despite occasional random failures.

## Overview

When testing non-deterministic systems such as large language models, traditional pass/fail testing is problematic because of sporatic errors or response inconsistencies. This plugin allows you to run tests multiple times and determine success based on a threshold, ensuring your tests are reliable even with occasional random failures.

## Features

- Run tests multiple times with a single decorator
- Set success thresholds to allow for occasional failures
- Batch test execution for performance optimization
- Retry capability for flaky tests
- Timeout control for long-running tests
- Detailed reporting of success rates and failure patterns
- Support for both synchronous and asynchronous tests

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

Mark any test with the `stochastic` decorator to run it multiple times:

```python
import pytest

@pytest.mark.stochastic(samples=5)  # Run 5 times
def test_llm_response():
    response = my_llm.generate("What is the capital of France?")
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
@pytest.mark.stochastic(samples=3, timeout=5.0)  # 5 second timeout
async def test_with_timeout():
    result = await long_running_operation()
    assert result.is_valid
```

### Batch Processing

Control concurrency with batch processing:

```python
@pytest.mark.stochastic(samples=20, batch_size=5)  # Run 5 at a time
async def test_with_batching():
    result = await async_operation()
    assert result.success
```

### Disabling Stochastic Mode

Temporarily disable stochastic behavior with a command-line flag:

```bash
pytest --disable-stochastic
```

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

### Testing with External APIs

```python
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

## Test Result Output

The plugin provides detailed test results in the console:

```
=========================== Stochastic Test Results ===========================

test_llm.py::test_llm_instruction_following:
  Success rate: 0.80
  Runs: 10, Successes: 8, Failures: 2
  Failure samples:
    1. AssertionError: Should follow 5-7-5 pattern, got [4, 6, 5]
    2. AssertionError: Should mention programming
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
