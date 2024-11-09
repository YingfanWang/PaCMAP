from .pacmap import PaCMAP, sample_neighbors_pair

import pkg_resources
__version__ = pkg_resources.get_distribution('pacmap').version
__all__ = ["pacmap"]
