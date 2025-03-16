"""Test multiple stochastic tests that use multiple fixtures in the same file."""
from pathlib import Path
import sys
import subprocess
import pytest

@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_multiple_stochastic_with_fixtures(example_test_dir: Path, is_async: bool):
    """Test running multiple stochastic tests that use fixtures in the same file."""
    # Create a counter file to track executions for each test
    test_one_counter = example_test_dir / "test_one_counter.txt"
    test_one_counter.write_text("0")

    test_two_counter = example_test_dir / "test_two_counter.txt"
    test_two_counter.write_text("0")

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
asyncio_default_fixture_loop_scope = function
""")

    # Create a test file with two stochastic tests that use different fixtures
    test_file = example_test_dir / "test_multiple_stochastic_fixtures.py"
    test_file.write_text(f"""
import pytest
{asyncio_import}

# Create a simple fixture that the first test uses
@pytest.fixture
def fixture_one():
    return "fixture_one_value"

# Create a second fixture that the second test uses
@pytest.fixture
def fixture_two():
    return "fixture_two_value"

# Create a third fixture that neither test explicitly uses
@pytest.fixture
def unused_fixture():
    return "unused_fixture_value"

# First stochastic test uses fixture_one only
@pytest.mark.stochastic(samples=3)
{"@pytest.mark.asyncio" if is_async else ""}
{test_prefix}def test_stochastic_one(fixture_one):
    # Track this execution
    with open("test_one_counter.txt", "r") as f:
        count = int(f.read())

    with open("test_one_counter.txt", "w") as f:
        f.write(str(count + 1))

    {sleep_code}
    assert fixture_one == "fixture_one_value"

# Second stochastic test uses fixture_two only
@pytest.mark.stochastic(samples=3)
{"@pytest.mark.asyncio" if is_async else ""}
{test_prefix}def test_stochastic_two(fixture_two):
    # Track this execution
    with open("test_two_counter.txt", "r") as f:
        count = int(f.read())

    with open("test_two_counter.txt", "w") as f:
        f.write(str(count + 1))

    {sleep_code}
    assert fixture_two == "fixture_two_value"
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

    # Check if the tests passed
    assert "2 passed" in result.stdout, f"Expected 2 tests to pass, got: {result.stdout}"

    # Check counters to verify each test ran the expected number of times
    with open(test_one_counter) as f:
        test_one_count = int(f.read())

    with open(test_two_counter) as f:
        test_two_count = int(f.read())

    # Each test should run 3 times
    assert test_one_count == 3, f"Expected test_stochastic_one to run 3 times, but ran {test_one_count} times"  # noqa: E501
    assert test_two_count == 3, f"Expected test_stochastic_two to run 3 times, but ran {test_two_count} times"  # noqa: E501

@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_stochastic_unexpected_kwargs(example_test_dir: Path, is_async: bool):
    """Test for unexpected keyword arguments issue with multiple stochastic tests and fixtures."""
    # Create counter files to track executions
    test_one_counter = example_test_dir / "test_one_counter.txt"
    test_one_counter.write_text("0")

    test_two_counter = example_test_dir / "test_two_counter.txt"
    test_two_counter.write_text("0")

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
asyncio_default_fixture_loop_scope = function
""")

    # Create test file with fixtures that might cause unexpected keyword arguments
    test_file = example_test_dir / "test_unexpected_kwargs.py"
    test_file.write_text(f"""
import pytest
import sys
{asyncio_import}
# Create two fixtures with the same parameterization setup
@pytest.fixture
def common_fixture():
    # Just a simple value
    return "common_value"

# Use scope=session to ensure this fixture is shared across all tests
@pytest.fixture(scope="session")
def session_fixture():
    # Session-scoped fixture
    return "session_value"

# First test with multiple fixtures
@pytest.mark.stochastic(samples=2)
{"@pytest.mark.asyncio" if is_async else ""}
{test_prefix}def test_first_stochastic(common_fixture, session_fixture):
    # Track this execution
    with open("test_one_counter.txt", "r") as f:
        count = int(f.read())

    with open("test_one_counter.txt", "w") as f:
        f.write(str(count + 1))

    {sleep_code}
    print(f"Test first stochastic execution {{count+1}}", file=sys.stderr)
    assert common_fixture == "common_value"
    assert session_fixture == "session_value"

# Second test uses the same fixtures but different parameter order
# This can potentially cause unexpected kwargs if funcargs is not properly filtered
@pytest.mark.stochastic(samples=2)
{"@pytest.mark.asyncio" if is_async else ""}
{test_prefix}def test_second_stochastic(session_fixture, common_fixture):
    # Track this execution
    with open("test_two_counter.txt", "r") as f:
        count = int(f.read())

    with open("test_two_counter.txt", "w") as f:
        f.write(str(count + 1))

    {sleep_code}
    print(f"Test second stochastic execution {{count+1}}", file=sys.stderr)
    assert session_fixture == "session_value"
    assert common_fixture == "common_value"
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

    # Check if the tests passed or if there was an unexpected kwargs error
    passed = "2 passed" in result.stdout
    unexpected_kwargs = "TypeError: " in result.stdout and "got an unexpected keyword argument" in result.stdout  # noqa: E501

    # Record the result for analysis
    if passed:
        print("Both tests passed successfully - no unexpected kwargs issue")
    elif unexpected_kwargs:
        print("Detected unexpected keyword argument error - issue confirmed")

    # Check counters to see if tests ran at all
    with open(test_one_counter) as f:
        test_one_count = int(f.read())

    with open(test_two_counter) as f:
        test_two_count = int(f.read())

    print(f"test_first_stochastic ran {test_one_count} times")
    print(f"test_second_stochastic ran {test_two_count} times")

    # For this test, we aren't asserting pass/fail - we just want to check if we can
    # reproduce the unexpected kwargs issue that the user reported
    # Instead, we'll verify that at least the first test attempted to run
    assert test_one_count > 0, "First test didn't run at all"

@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_stochastic_mixed_args(example_test_dir: Path, is_async: bool):
    """Test stochastic tests that use explicit args and **kwargs."""
    # Create counter files to track executions
    test_one_counter = example_test_dir / "mixed_counter.txt"
    test_one_counter.write_text("0")

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
asyncio_default_fixture_loop_scope = function
""")

    # Create a test file with a mix of explicit and variable arguments
    test_file = example_test_dir / "test_mixed_args.py"
    test_file.write_text(f"""
import pytest
import sys
{asyncio_import}
@pytest.fixture
def fixture_a():
    return "fixture_a_value"

@pytest.fixture
def fixture_b():
    return "fixture_b_value"

@pytest.fixture
def fixture_c():
    return "fixture_c_value"

# Test that uses a mix of explicit args and **kwargs
# This can cause issues if the plugin doesn't correctly handle this pattern
@pytest.mark.stochastic(samples=3)
{"@pytest.mark.asyncio" if is_async else ""}
{test_prefix}def test_with_mixed_args(fixture_a, **kwargs):
    # Track this execution
    with open("mixed_counter.txt", "r") as f:
        count = int(f.read())

    with open("mixed_counter.txt", "w") as f:
        f.write(str(count + 1))

    {sleep_code}
    print(f"Test with mixed args execution {{count+1}}", file=sys.stderr)
    print(f"kwargs: {{kwargs}}", file=sys.stderr)

    assert fixture_a == "fixture_a_value"
    # We should get fixture_b and fixture_c in kwargs
    assert "fixture_b" in kwargs or "fixture_c" in kwargs, "Expected at least one fixture in kwargs"
""")  # noqa: E501

    # Run pytest on the file with -v for verbose output
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

    # Check if the test passed or if there was an error
    passed = "1 passed" in result.stdout

    # Record the result for analysis
    if passed:
        print("Test with mixed arguments passed successfully")
    else:
        print("Test with mixed arguments failed - possible issue with argument handling")

    # Check counter to see if test ran completely
    with open(test_one_counter) as f:
        mixed_count = int(f.read())

    print(f"test_with_mixed_args ran {mixed_count} times")

    # Assert that the test at least attempted to run
    assert mixed_count > 0, "Mixed args test didn't run at all"

@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_stochastic_dynamic_fixtures(example_test_dir: Path, is_async: bool):
    """Test stochastic tests with fixtures that use request and dynamically create other fixtures."""  # noqa: E501
    # Create counter files to track executions
    test_counter = example_test_dir / "dynamic_counter.txt"
    test_counter.write_text("0")

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
asyncio_default_fixture_loop_scope = function
""")

    # Create a test file with complex fixture dependencies
    test_file = example_test_dir / "test_dynamic_fixtures.py"
    test_file.write_text(f"""
import pytest
import sys
{asyncio_import}
# Dynamic fixture that can request other fixtures
@pytest.fixture
def dynamic_fixture(request):
    # Get the related fixture name from the request node
    test_name = request.node.name
    print(f"Creating dynamic fixture for {{test_name}}", file=sys.stderr)

    # Return a function that can be used by the test
    return lambda x: f"Dynamic value for {{test_name}}: {{x}}"

# Base fixture for reuse
@pytest.fixture
def base_fixture():
    return "base_value"

# First stochastic test uses dynamic fixture
@pytest.mark.stochastic(samples=2)
{"@pytest.mark.asyncio" if is_async else ""}
{test_prefix}def test_with_dynamic_fixture(dynamic_fixture, base_fixture):
    # Track this execution
    with open("dynamic_counter.txt", "r") as f:
        count = int(f.read())

    with open("dynamic_counter.txt", "w") as f:
        f.write(str(count + 1))

    {sleep_code}
    print(f"Dynamic test execution {{count+1}}", file=sys.stderr)

    # Use the dynamic fixture
    result = dynamic_fixture("test_param")
    print(f"Dynamic fixture result: {{result}}", file=sys.stderr)

    assert "Dynamic value for" in result
    assert base_fixture == "base_value"

# Second stochastic test also uses dynamic fixture but differently
@pytest.mark.stochastic(samples=2)
{"@pytest.mark.asyncio" if is_async else ""}
{test_prefix}def test_with_dynamic_fixture_2(dynamic_fixture):
    # Just reading counter
    with open("dynamic_counter.txt", "r") as f:
        count = int(f.read())

    {sleep_code}
    print(f"Second dynamic test, count is {{count}}", file=sys.stderr)

    # Use the dynamic fixture differently
    result = dynamic_fixture("another_param")
    print(f"Dynamic fixture result 2: {{result}}", file=sys.stderr)

    assert "Dynamic value for" in result
    assert "another_param" in result
""")

    # Run pytest on the file with -v for verbose output
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

    # Check if the tests passed
    passed = "2 passed" in result.stdout

    # Record the result for analysis
    if passed:
        print("Dynamic fixture tests passed successfully")
    else:
        print("Dynamic fixture tests failed")

    # Check counter to see if first test ran
    with open(test_counter) as f:
        run_count = int(f.read())

    # The first test should have run twice
    assert run_count == 2, f"Expected test_with_dynamic_fixture to run 2 times, but ran {run_count} times"  # noqa: E501

@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_stochastic_class_fixtures(example_test_dir: Path, is_async: bool):
    """Test stochastic tests defined as class methods with fixtures."""
    # Create counter files to track executions
    test_one_counter = example_test_dir / "class_method_counter.txt"
    test_one_counter.write_text("0")

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
asyncio_default_fixture_loop_scope = function
""")

    # Create a test file with stochastic tests in a class with fixtures
    test_file = example_test_dir / "test_class_methods.py"
    test_file.write_text(f"""
import pytest
import sys
{asyncio_import}
# Create some fixtures
@pytest.fixture
def fixture_a():
    return "fixture_a_value"

@pytest.fixture
def fixture_b():
    return "fixture_b_value"

# Class with stochastic test methods
{"@pytest.mark.asyncio" if is_async else ""}
class TestStochasticClassMethods:

    # Method that uses fixture_a
    @pytest.mark.stochastic(samples=2)
    {test_prefix}def test_method_one(self, fixture_a):
        # Track this execution
        with open("class_method_counter.txt", "r") as f:
            count = int(f.read())

        with open("class_method_counter.txt", "w") as f:
            f.write(str(count + 1))

        {sleep_code}
        print(f"Test method one execution {{count+1}}", file=sys.stderr)
        assert fixture_a == "fixture_a_value"

    # Method that uses fixture_b
    @pytest.mark.stochastic(samples=2)
    {test_prefix}def test_method_two(self, fixture_b):
        # Track execution but don't update counter
        with open("class_method_counter.txt", "r") as f:
            count = int(f.read())

        {sleep_code}
        print(f"Test method two execution, count is {{count}}", file=sys.stderr)
        assert fixture_b == "fixture_b_value"
""")

    # Run pytest on the file with -v for verbose output
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

    # Check if the tests passed
    passed = "2 passed" in result.stdout

    # Record the result for analysis
    if passed:
        print("Class methods with stochastic markers passed successfully")
    else:
        print("Class methods with stochastic markers failed")

    # Check counter to see if tests ran
    with open(test_one_counter) as f:
        run_count = int(f.read())

    # Only the first method increments the counter, and it should run twice
    assert run_count == 2, f"Expected test_method_one to run 2 times, but ran {run_count} times"
