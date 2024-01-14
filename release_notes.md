# Release Notes

- 0.7.2

  Fix the problem in using user-provided initialization, as well as discrepancy
  between documentation and code in default `n_neighbors` parameter.

- 0.7.1

  Update the required `numba` version to `numba >= 0.57`.

- 0.7.0

  Now officially supports the `save` and `load` methods.
  `pacmap.save(reducer, common_prefix)` will save the PaCMAP instance (and the AnnoyIndex if `save_tree=True`) to the location specified by the `common_prefix`. The PaCMAP instance will be named as `{common_prefix}.pkl` and the Annoy Index will be named as `{common_fix}.ann`. Similarly, `pacmap.load(common_prefix)` loads the saved PaCMAP instance.

- 0.6.0

  Now officially supports the `transform` feature. The transform operation is useful for projecting a new dataset into an existing embedded space. **In the current version of implementation, the `transform` method will treat the input as an additional dataset, which means the same point could be mapped into a different place.**

- 0.5.0
  
  Now support setting `random_state` when creating `pacmap.PaCMAP` instances for better reproducibility.

  Fix the default initialization to `PCA` to resolve inconsistency between code and description.

  **Setting the `random_state` will affect the numpy random seed in your local environment. However, you may still get different results even if the `random_state` parameter is set to be the same. This is because numba parallelization makes some of the functions undeterministic.** That being said, fixing the random state will always give you the same set of pairs and initialization, which ensure the difference is minimal.
- 0.4.1

  Now the default value for `n_neighbors` is 10. To enable automatic parameter selection, please set it to `None`.
- 0.4
  
  Now supports user-specified nearest neighbor pairs. See section `How to use user-specified nearest neighbor` below.

  The `fit` function and the `fit_transform` function now has an extra parameter `save_pairs` that decides whether the pairs sampled in this run will be saved to save time for reproducing experiments with other hyperparameters (default to `True`).
- 0.3
  
  Now supports user-specified matrix as initialization through `init` parameter. The matrix must be an numpy ndarray with the shape (N, 2).
- 0.2
  
  Adding adaptive default value for `n_neighbors`: for large datasets with sample size N > 10000, the default value will be set to 10 + 15 * (log10(N) - 4), rounding to the nearest integer.
- 0.1

  Initial Release
