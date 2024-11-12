from .pacmap import PaCMAP, sample_neighbors_pair

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version('pacmap')
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["PaCMAP", "sample_neighbors_pair"]
