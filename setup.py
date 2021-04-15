import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pacmap",
    version="0.3",
    author="Yingfan Wang, Haiyang Huang, Cynthia Rudin, Yaron Shaposhnik",
    description="the official implementation for PaCMAP: Pairwise Controlled" + \
                " Manifold Approximation Projection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://https://github.com/YingfanWang/PaCMAP",
    packages=['pacmap'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires='>=3.6',
    install_requires=['scikit-learn >= 0.20',
                        'numba >= 0.34',
                        'annoy >= 1.11',
                        'numpy >= 1.18']

)
