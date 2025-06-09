"""
Pytest configuration file to set up the test environment.
"""

import sys
from pathlib import Path

# Add the source directory to Python path so we can import pacmap
source_dir = Path(__file__).parent.parent / "source"
sys.path.insert(0, str(source_dir))
