"""Build script for the scikit-learn port of PaCMAP.

scikit-learn itself builds with meson; ``meson.build.sklearn`` shows the
equivalent stanza for ``sklearn/manifold``. setuptools is used here only so
the port can be built and tested standalone.
"""

import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

if sys.platform == "win32":
    openmp_compile, openmp_link = ["/openmp"], []
elif sys.platform == "darwin":
    # Apple clang needs libomp from Homebrew; fall back to serial if absent.
    openmp_compile, openmp_link = ["-Xpreprocessor", "-fopenmp"], ["-lomp"]
else:
    openmp_compile, openmp_link = ["-fopenmp"], ["-fopenmp"]

extensions = [
    Extension(
        "pacmap_sklearn._pacmap_utils",
        ["pacmap_sklearn/_pacmap_utils.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"] + openmp_compile
        if sys.platform != "win32"
        else openmp_compile,
        extra_link_args=openmp_link,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_9_API_VERSION")],
    )
]

setup(ext_modules=cythonize(extensions, language_level=3))
