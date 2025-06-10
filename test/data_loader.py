"""
Data loader utilities for loading datasets from fixtures.
"""

import json
from sklearn.datasets import fetch_openml


def load_datasets_from_fixture(path=None):
    """
    Load datasets from a JSON fixture file.

    Args:
        path (str): Path to the JSON fixture file containing dataset specifications.
                   If None, uses the default path relative to this file.

    Returns:
        dict: Dictionary mapping dataset names to loaded OpenML datasets

    Note:
        Downloaded datasets are cached by sklearn in ~/scikit_learn_data/ by default.
    """
    if path is None:
        # Get path relative to this file
        from pathlib import Path
        current_dir = Path(__file__).parent
        path = current_dir / "fixtures" / "datasets.json"

    with open(path) as f:
        datasets = json.load(f)

    return {
        d["name"]: fetch_openml(d["name"], version=d["version"], return_X_y=True, parser="pandas", as_frame=False)
        for d in datasets
    }