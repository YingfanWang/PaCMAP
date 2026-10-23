"""scikit-learn-conformant port of PaCMAP (LocalMAP and Rashomon refinement are switches).

Staging package for a ``sklearn.manifold.PaCMAP`` proposal. See README.md.
"""

from ._pacmap import PaCMAP

__all__ = ["PaCMAP"]
