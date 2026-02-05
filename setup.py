from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pacmap",
    version="0.8.0",
    author="Yingfan Wang, Haiyang Huang, Cynthia Rudin, Yaron Shaposhnik",
    author_email="yingfan.wang@duke.edu",
    description="The official implementation for PaCMAP: Pairwise Controlled Manifold Approximation Projection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YingfanWang/PaCMAP",
    project_urls={
        "Bug Tracker": "https://github.com/YingfanWang/PaCMAP/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    package_dir={"": "source"},
    packages=find_packages(where="source"),
    python_requires=">=3.8",
    install_requires=[
        "scikit-learn>=0.20",
        "numba>=0.57",
        "annoy>=1.11",
        "numpy>=1.20",
        "faiss-cpu",
        "voyager @ git+https://github.com/spotify/voyager.git#subdirectory=python",
    ],
)
