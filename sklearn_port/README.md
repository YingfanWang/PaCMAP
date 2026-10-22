# PaCMAP → scikit-learn port

Staging area for a `sklearn.manifold.PaCMAP` proposal. The estimator here
depends only on numpy, scipy and scikit-learn at runtime, implements its
kernels in Cython + OpenMP, and passes `sklearn.utils.estimator_checks`
for the euclidean, angular, manhattan and precomputed configurations.

Files are laid out as they would be inside scikit-learn:

| here | in scikit-learn |
|---|---|
| `pacmap_sklearn/_pacmap.py` | `sklearn/manifold/_pacmap.py` |
| `pacmap_sklearn/_pacmap_utils.pyx` | `sklearn/manifold/_pacmap_utils.pyx` |
| `pacmap_sklearn/tests/test_pacmap.py` | `sklearn/manifold/tests/test_pacmap.py` |
| `meson.build.sklearn` | stanza for `sklearn/manifold/meson.build` |
| `benchmarks/bench_pacmap.py` | `benchmarks/bench_pacmap.py` |

`test_parity.py` and the benchmark compare against the numba reference in
`../source/pacmap` and are not meant to move upstream.

## Build and test

```bash
cd sklearn_port
pip install numpy scipy scikit-learn Cython pytest threadpoolctl
python setup.py build_ext --inplace
python -m pytest            # parity tests run if `pacmap` (numba) is installed
cd .. && PYTHONPATH=sklearn_port python sklearn_port/benchmarks/bench_pacmap.py
```

## What changed relative to `source/pacmap`

**Kernels (numba → Cython).**

* Pair sampling uses a counter-based generator (splitmix64) seeded per row
  from one seed drawn from `random_state`, so the sampled pairs depend only
  on the seed and not on thread scheduling.
* The gradient is computed as a per-node *gather* over a CSR incidence list
  that holds every pair once per endpoint. The reference *scatters* into
  `grad[j]`. The gather needs no atomics, and the floating-point
  accumulation order is fixed.
* Consequence of both: output is bitwise identical for any number of OpenMP
  threads (`test_output_independent_of_thread_count`).
* Each pair is evaluated twice, once per endpoint. In return the loop
  parallelises with no atomics.
* The gradient matches the numba kernel to float32 precision
  (`test_gradient_matches_reference`) and matches central finite
  differences of the loss.

**Nearest neighbors.**

* Search is exact, via `NearestNeighbors`, replacing the faiss HNSW / annoy /
  voyager backends.
* `metric="precomputed"` accepts a dense square distance matrix.
* Everything after the search is unchanged:
  * locally scaled re-ranking of the top `n_neighbors + 50` candidates;
  * MN rule ("2nd closest of 6");
  * FP exclusion of NN partners;
  * three-phase weight schedule;
  * Adam constants.

**Estimator API.**

| reference | port | why |
|---|---|---|
| `MN_ratio`, `FP_ratio`, `distance`, `lr` | `mn_ratio`, `fp_ratio`, `metric`, `learning_rate` | sklearn naming |
| `fit(X, init=...)` | `PaCMAP(init=...)` | fit params must be data; matches `TSNE` |
| validation + mutation in `__init__` | `_parameter_constraints`, nothing in `__init__` | required for `clone` / `set_params` |
| `n_neighbors` overwritten in `fit` | stored as `n_neighbors_` | params must not change during fit |
| `random_state` in module global `_RANDOM_STATE` | per-instance, `check_random_state` | see bug below |
| float32 | float64 | sklearn default; float32 input is upcast |
| `transform`, `intermediate`, `save`/`load`, `pair_*` inputs, `knn_backend`, `save_tree`, `LocalMAP` | dropped | not needed for a first upstream PR (`TSNE` has no `transform` either) |

**Bug found in the reference implementation.** `random_state` is stored in a
module-level global, and every `PaCMAP.__init__` overwrites it.

* Constructing an unseeded instance after a seeded one silently makes the
  seeded one non-reproducible:

  ```python
  b = pacmap.PaCMAP(random_state=0)
  pacmap.PaCMAP()
  b.fit_transform(X)   # no longer reproducible
  ```

  Verified on the current `master`.
* `random_state=None` is also rewritten to `0` on the instance.
* `test_instances_do_not_share_random_state` covers this in the port. The
  reference should get a separate fix.

**Small-n edge cases.** The port works for any `n_samples >= 3`.

* σᵢ (mean distance to neighbors 4–6) falls back to the available
  neighbor columns. With fewer than 4 neighbors the reference takes the mean
  of an empty slice (NaN).
* The MN and FP counts are clamped to what the samplers can draw without
  looping forever.

Sampled pairs are not bitwise equal to the reference's, because the RNG is
different. They follow the same distribution.

## Results

### Quality

Quality numbers are from `benchmarks/bench_pacmap.py`, run on 2 CPU cores and
averaged over 3 seeds.

Metrics:

* **trust**: trustworthiness@10 on a 3000-point subsample.
* **10-NN**: 5-fold kNN accuracy in the embedding.
* **triplet**: random-triplet order agreement, a global-structure measure.

| dataset | method | trust | 10-NN | triplet |
|---|---|---|---|---|
| USPS (9298×256) | port | 0.9527 | 0.9593 | 0.6373 |
| | reference | 0.9537 | 0.9626 | 0.6400 |
| | sklearn TSNE | 0.9753 | 0.9622 | 0.6602 |
| 20NG (18846×100) | port | 0.8632 | 0.5350 | 0.6575 |
| | reference | 0.8668 | 0.5336 | 0.6578 |
| | sklearn TSNE | 0.9035 | 0.6284 | 0.6436 |
| synthetic (70000×784, 1 seed) | port | 0.9635 | 1.0000 | 0.7712 |
| | reference | 0.9629 | 1.0000 | 0.7718 |

The port is within seed-to-seed noise of the reference on all three
metrics.

### Wall time

Times were measured in a separate run with no other load; the seconds
shown are per fit.

| dataset | port | reference (faiss HNSW) | sklearn TSNE |
|---|---|---|---|
| USPS | 2.7–3.0 | 2.8–3.0 | 59 |
| 20NG | 5.1–5.6 | 5.6–5.8 | 144 |
| synthetic 70k | 45.8 | 28.0 | — |

Up to ~2×10⁴ points the port is as fast as the reference. At 7×10⁴ the
exact kNN search makes it 1.6× slower. That gap grows as O(n²), which is
open question 2 below. Both are 20–27× faster than `TSNE`.

A profile on USPS splits the port's 2.1 s fit as follows:

| phase | time |
|---|---|
| Optimizer | 0.86 s |
| Randomized SVD to 100 dims | 0.78 s |
| Exact kNN | 0.40 s |

## Open questions to settle before proposing upstream

1. **Sparse precomputed kNN graphs.** This is the sklearn idiom for plugging in
   ANN search: `KNeighborsTransformer` → `metric="precomputed"`, which `TSNE`
   supports. PaCMAP cannot consume a sparse kNN graph as-is, because the MN rule
   needs distances from *i* to 6 uniformly random points, which a kNN graph
   doesn't contain. Options:
   * (a) dense precomputed only, the current state;
   * (b) accept the kNN graph for NN pairs plus features for MN distances;
   * (c) change the MN rule for graph input.

   This is an algorithmic decision for the authors, not an engineering one.
2. **Exact kNN cost.** Brute-force search on the 100-dim PCA projection is
   O(n²·100). That is fine to ~10⁵ points and dominates beyond. Upstream
   reviewers will compare against `TSNE`, which has the same exact-search
   default.
3. **Naming.** Options to settle:
   * `"angular"` vs sklearn's `"cosine"`; the distance differs,
     √(2 − 2cos) vs 1 − cos, and the difference matters for the scaled
     re-ranking.
   * `num_iters` (3-tuple) vs a `max_iter` + schedule convention.
4. **Stored pairs.** `pair_*_` attributes cost about 560 B/sample at
   defaults. Useful for analysis, but reviewers may ask to drop them.
5. **`transform`.** The reference's out-of-sample method optimizes new points
   against a frozen embedding with NN attraction only. Omitted here until
   its semantics are pinned down.
