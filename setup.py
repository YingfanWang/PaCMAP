import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pacmap",
    version="1.0",
    author="Yingfan Wang, Haiyang Huang, Cynthia Rudin, Yaron Shaposhnik",
    author_email="yingfan.wang@duke.edu",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://https://github.com/YingfanWang/PaCMAP",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires='>=3.6',
)
