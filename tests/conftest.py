"""Test fixtures and configuration for pytest plugin tests."""
import pytest
import tempfile
import shutil
from pathlib import Path

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def example_test_dir(temp_dir: Path):
    """Create a directory with example test files."""
    # Copy our example test files to the temp directory
    example_dir = Path(__file__).parent / "test_examples"
    if example_dir.exists():
        for file in example_dir.glob("*.py"):
            shutil.copy(file, temp_dir / file.name)
    return temp_dir
