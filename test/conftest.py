"""
Pytest configuration file to set up the test environment.
"""

import sys
from pathlib import Path
import pytest

# Add the source directory to Python path so we can import pacmap
source_dir = Path(__file__).parent.parent / "source"
sys.path.insert(0, str(source_dir))

# Ensure test output directory exists
test_output_dir = Path(__file__).parent / "output"
test_output_dir.mkdir(exist_ok=True)

# Import data loader from the same directory
from data_loader import load_datasets_from_fixture


@pytest.fixture(scope="session")
def openml_datasets():
    """
    Load OpenML datasets from fixture file.

    This fixture is scoped to the session level to avoid reloading
    datasets for each test, improving test performance.

    Returns:
        dict: Dictionary mapping dataset names to loaded OpenML datasets
    """
    return load_datasets_from_fixture()
