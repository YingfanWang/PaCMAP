"""scikit-learn-conformant port of PaCMAP and LocalMAP.

Staging package for a ``sklearn.manifold.PaCMAP`` proposal. See README.md.
"""

from ._pacmap import LocalMAP, PaCMAP

__all__ = ["LocalMAP", "PaCMAP"]
