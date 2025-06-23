# PaCMAP [![Tests](https://github.com/YingfanWang/PaCMAP/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/YingfanWang/PaCMAP/actions/workflows/test.yml)

## <a name='News'></a>News

### NeurIPS 2024: New Parametric DR method ParamRepulsor accepted

We're excited to announce that our latest work has been published at the Thirty-Eighth Conference on Neural Information Processing Systems (**NeurIPS 2024**)!🎉🎉

Traditional dimensionality reduction (DR) algorithms struggle with online-learning scenarios, while existing parametric DR approaches often fail to preserve local structure in visualizations.
Our latest algorithm, *ParamRepulsor*, builds on Parametric PaCMAP to address these challenges, achieving state-of-the-art performance in both local and global structure preservation. With GPU support using PyTorch, ParamRepulsor delivers exceptional speed and scalability, making it suitable for large-scale and dynamic datasets.

Check out the [NeurIPS paper](https://openreview.net/pdf?id=eYNYnYle41) and the [code](https://github.com/hyhuang00/ParamRepulsor) for detailed insights into the new approach.

### AAAI 2025: New DR method LocalMAP for Local Adjusted Graph accepted

We're excited to announce that our latest work has been published at the The 39th Annual AAAI Conference on Artificial Intelligence (**AAAI 2025**)!🎉🎉

General dimension reduction (DR) algorithm often involves converting the original high-dimensional data into a graph. Each edge within the graph represents the similarity or dissimilarity between pairs of data points. However, this graph is frequently suboptimal due to unreliable high-dimensional distances and the limited information extracted from the high-dimensional data.

Our latest algorithm, Pairwise Controlled Manifold Approximation with Local Adjusted Graph(*LocalMAP*), from a nonparametric perspective, address this problem by dynamically and locally adjusting the graph during the final stage, making the real clusters within the dataset to be easier to identify and more separable comparing to other DR methods that may overlook or combine.

Check out our [AAAI Paper](https://arxiv.org/abs/2412.15426) and the [Code](https://github.com/williamsyy/LocalMAP) for detailed insights into the new approach. This method will be embedded into PaCMAP package soon.

## <a name='Introduction'></a>Introduction

Our work has been published in the **Journal of Machine Learning Research (JMLR)** 📚 and has earned the prestigious **John M. Chambers Statistical Software Award** 🥇 and the **Award for Innovation in Statistical Programming and Analytics** 💡 presented by the Statistical Computing Section (SCS) and the Statistical Programming and Analytics Section (SSPA) of the American Statistical Association (ASA).

PaCMAP (Pairwise Controlled Manifold Approximation) is a dimensionality reduction method that can be used for visualization, preserving both local and global structure of the data in original space. PaCMAP optimizes the low dimensional embedding using three kinds of pairs of points: neighbor pairs (pair_neighbors), mid-near pair (pair_MN), and further pairs (pair_FP).

Previous dimensionality reduction techniques focus on either local structure (e.g. t-SNE, LargeVis and UMAP) or global structure (e.g. TriMAP), but not both, although with carefully tuning the parameter in their algorithms that controls the balance between global and local structure, which mainly adjusts the number of considered neighbors. Instead of considering more neighbors to attract for preserving global structure, PaCMAP dynamically uses a special group of pairs -- mid-near pairs, to first capture global structure and then refine local structure, which both preserve global and local structure. For a thorough background and discussion on this work, please read [our paper](https://jmlr.org/papers/v22/20-1061.html).

## <a name='ReleaseNotes'></a>Release Notes

Please see the [release notes](release_notes.md).

## <a name='Installation'></a>Installation

### <a name='Installfromconda-forgeviacondaormamba'></a>Install from conda-forge via conda or mamba

You can use [conda](https://docs.conda.io/en/latest/) or [mamba](https://mamba.readthedocs.io/en/latest/index.html)
to install PaCMAP from the conda-forge channel.

conda:

```bash
conda install pacmap -c conda-forge
```

mamba:

```bash
mamba install pacmap -c conda-forge
```

### <a name='InstallfromPyPIviapip'></a>Install from PyPI via pip

You can use [pip](https://pip.pypa.io/en/stable/) to install pacmap from PyPI.
It will automatically install the dependencies for you:

```bash
pip install pacmap
```

If you have any problems during the installation of dependencies, such as
`Failed building wheel for annoy`, you can try to install these dependencies
with `conda` or `mamba`. Users have also reported that in some cases, you may
wish to use `numba >= 0.57`.

```bash
conda install -c conda-forge python-annoy
pip install pacmap
```

## <a name='Usage'></a>Usage

### <a name='UsingPaCMAPinPython'></a>Using PaCMAP in Python

The `pacmap` package is designed to be compatible with `scikit-learn`, meaning that it has a similar interface with functions in the `sklearn.manifold` module. To run `pacmap` on your own dataset, you should install the package following the instructions in [installation](#installation), and then import the module. The following code clip includes a use case about how to use PaCMAP on the [COIL-20](https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php) dataset:

```python
import pacmap
import numpy as np
import matplotlib.pyplot as plt

# loading preprocessed coil_20 dataset
# you can change it with any dataset that is in the ndarray format, with the shape (N, D)
# where N is the number of samples and D is the dimension of each sample
X = np.load("./data/coil_20.npy", allow_pickle=True)
X = X.reshape(X.shape[0], -1)
y = np.load("./data/coil_20_labels.npy", allow_pickle=True)

# initializing the pacmap instance
# Setting n_neighbors to "None" leads to an automatic choice shown below in "parameter" section
embedding = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0) 

# fit the data (The index of transformed data corresponds to the index of the original data)
X_transformed = embedding.fit_transform(X, init="pca")

# visualize the embedding
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(X_transformed[:, 0], X_transformed[:, 1], cmap="Spectral", c=y, s=0.6)
```

### <a name='UsingPaCMAPinR'></a>Using PaCMAP in R

You can also use PaCMAP in R with the [reticulate package](https://rstudio.github.io/reticulate/).
We provide a sample [R notebook](./demo/pacmap_Rnotebook_example.Rmd) that demonstrates
how PaCMAP can be called in R for visualization. We also provide a [Seurat Intergation](https://github.com/williamsyy/gdc-frontend-framework) that allows seamless integration with
[Seurat](https://github.com/satijalab/seurat) Objects for single cell genomics.

### <a name='UsingPaCMAPinRust'></a>Using PaCMAP in Rust

A [Rust implementation](https://github.com/beamform/pacmap-rs.git) of PaCMAP has
recently be released by @hadronzoo. This implementation is Python free, meaning that
it does not depend on a Python runtime or Python environment.

## <a name='Benchmarks'></a>Benchmarks

The following images are visualizations of two datasets: [MNIST](http://yann.lecun.com/exdb/mnist/) (n=70,000, d=784) and [Mammoth](https://github.com/PAIR-code/understanding-umap/tree/master/raw_data) (n=10,000, d=3), generated by PaCMAP. The two visualizations demonstrate the local and global structure's preservation ability of PaCMAP respectively.

![MNIST](/images/MNIST.jpg?raw=true "PaCMAP's result on MNIST")

![Mammoth](/images/Mammoth.jpg?raw=true "PaCMAP's result on Mammoth")

## <a name='Parameters'></a>Parameters

The list of the most important parameters is given below. Changing these values will affect the result of dimension reduction significantly, as specified in section 8.3 in our paper.

- `n_components`: the number of dimension of the output. Default to 2.

- `n_neighbors`: the number of neighbors considered in the k-Nearest Neighbor graph. Default to 10. We also allow this parameter to be set to `None` to enable the auto-selection of number of neighbors: the number of neighbors will be set to 10 for dataset whose sample size is smaller than 10000. For large dataset whose sample size (n) is larger than 10000, the value is: 10 + 15 * (log10(n) - 4).

- `MN_ratio`: the ratio of the number of mid-near pairs to the number of neighbors, `n_MN` = <img src="https://latex.codecogs.com/gif.latex?\lfloor" title="\lfloor" /> `n_neighbors * MN_ratio` <img src="https://latex.codecogs.com/gif.latex?\rfloor" title="\rfloor" /> . Default to 0.5.

- `FP_ratio`: the ratio of the number of further pairs to the number of neighbors, `n_FP` = <img src="https://latex.codecogs.com/gif.latex?\lfloor" title="\lfloor" /> `n_neighbors * FP_ratio` <img src="https://latex.codecogs.com/gif.latex?\rfloor" title="\rfloor" />  Default to 2.

The initialization is also important to the result, but it's a parameter of the `fit` and `fit_transform` function.

- `init`: the initialization of the lower dimensional embedding. One of `"pca"` or `"random"`, or a user-provided numpy ndarray with the shape (N, 2). Default to `"pca"`.

Other parameters include:

- `num_iters`: number of iterations. Default to 450. 450 iterations is enough for most dataset to converge.
- `pair_neighbors`, `pair_MN` and `pair_FP`: pre-specified neighbor pairs, mid-near points, and further pairs. Allows user to use their own graphs. Default to `None`.
- `verbose`: print the progress of pacmap. Default to `False`
- `lr`: learning rate of the AdaGrad optimizer. Default to 1.
- `apply_pca`: whether pacmap should apply PCA to the data before constructing the k-Nearest Neighbor graph. Using PCA to preprocess the data can largely accelerate the DR process without losing too much accuracy. Notice that this option does not affect the initialization of the optimization process.
- `intermediate`: whether pacmap should also output the intermediate stages of the optimization process of the lower dimension embedding. If `True`, then the output will be a numpy array of the size (n, `n_components`, 13), where each slice is a "screenshot" of the output embedding at a particular number of steps, from [0, 10, 30, 60, 100, 120, 140, 170, 200, 250, 300, 350, 450].

## <a name='Methods'></a>Methods

Similar to the scikit-learn API, the PaCMAP instance can generate embedding for a dataset via `fit`, `fit_transform` and `transform` method. We currently support numpy.ndarray format as our input. Specifically, to convert pandas DataFrame to ndarray format, please refer to the [pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_numpy.html). For a more detailed walkthrough, please see the [demo](./demo/) directory.

## <a name='Howtouseuser-specifiednearestneighbor'></a>How to use user-specified nearest neighbor

In version 0.4, we have provided a new option to allow users to use their own nearest neighbors when mapping large-scale datasets. Please see the [demo](./demo/specify_nn_demo.py) for a detailed walkthrough about how to use PaCMAP with the user-specified nearest neighbors.

## <a name='Reproducingourexperiments'></a>Reproducing our experiments

We have provided the code we use to run experiment for better reproducibility. The code are separated into three parts, in three folders, respectively:

- `data`, which includes all the datasets we used, preprocessed into the file format each DR method use. Notice that since the Mouse single cell RNA sequence dataset is too big (~4GB), you may need to download from the [link](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE93374) here. MNIST and FMNIST dataset is compressed, and you need to unzip them before using. COIL-100 dataset is still too large after compressed, please preprocess it using the file Preprocessing.ipynb on your own.
- `experiments`, which includes all the scripts we use to produce DR results.
- `evaluation`, which includes all the scripts we use to evaluate DR results, stated in Section 8 in our paper.

After downloading the code, you may need to specify some of the paths in the script to make them fully functional.

## <a name='Testing'></a>Testing

We have a suite of tests. We run these tests on all pull requests, and every commit to `master` branch.

We test against several version of Python. (See [`.github/workflows/test.yml`][workflow-python-versions] for our currently tested versions.)

If you are contributing code, please confirm our tests pass, and consider adding your own for any new functionality.

You may run the test suite like so:

```sh
# Clean up past outputs (plots will be DELETED),
# and uninstall the public package version of pacmap.
make clean

# Install the development dependencies.
make install-dev

# Run the test suite against your local version of the codebase.
make test
```

## <a name='Citation'></a>Citation

If you used PaCMAP in your publication, or you used the implementation in this repository, please cite our paper using the following bibtex:

```bibtex
@article{JMLR:v22:20-1061,
  author  = {Yingfan Wang and Haiyang Huang and Cynthia Rudin and Yaron Shaposhnik},
  title   = {Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {201},
  pages   = {1-73},
  url     = {http://jmlr.org/papers/v22/20-1061.html}
}
```

For PaCMAP's performance on biological dataset, please check the following paper:

```bibtex
@article{huang2022towards,
  title={Towards a comprehensive evaluation of dimension reduction methods for transcriptomic data visualization},
  author={Huang, Haiyang and Wang, Yingfan and Rudin, Cynthia and Browne, Edward P},
  journal={Communications biology},
  volume={5},
  number={1},
  pages={719},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
```

## <a name='License'></a>License

Please see the license file.

<!-- Link -->
   [workflow-python-versions]: .github/workflows/test.yml#L15