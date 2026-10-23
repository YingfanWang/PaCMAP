# PaCMAP → scikit-learn port

Staging area for a `sklearn.manifold.PaCMAP` proposal. The package exposes a
single estimator, `PaCMAP`.

* It depends only on numpy, scipy and scikit-learn at runtime.
* Its kernels are written in Cython + OpenMP.
* It passes `sklearn.utils.estimator_checks` (euclidean, angular,
  manhattan, precomputed, with `localmap=True`, and with `n_rashomon` set).

Two extensions are hidden behind switches, both off by default:

| switch | method | effect |
|---|---|---|
| `localmap=True` (+ `low_dist_thres`) | LocalMAP, Wang et al. (AAAI 2025) | local graph adjustment in phase 3 |
| `n_rashomon=k` (+ `rashomon_penalty`) | Sun et al. (AISTATS 2026) | trustworthy-neighbor refinement |

The two switches cannot be combined; `fit` raises `ValueError`. See
"Not supported with LocalMAP" below.

```python
PaCMAP()                     # plain PaCMAP
PaCMAP(localmap=True)        # LocalMAP
PaCMAP(n_rashomon=5)         # Rashomon-refined PaCMAP (experimental, warns)
```

**Scope for upstream.** The scikit-learn PR would ship `PaCMAP` with both
switches removed. LocalMAP and the Rashomon refinement are too recent for
the
[inclusion criteria](https://scikit-learn.org/stable/faq.html#what-are-the-inclusion-criteria-for-new-algorithms).
As switches, they are easy to strip: two parameters each, plus one
`if self.localmap` branch in `_optimize` and one `if self.n_rashomon`
branch in `_fit`.

Files are laid out as they would be inside scikit-learn:

| here | in scikit-learn |
|---|---|
| `pacmap_sklearn/_pacmap.py` | `sklearn/manifold/_pacmap.py` |
| `pacmap_sklearn/_pacmap_utils.pyx` | `sklearn/manifold/_pacmap_utils.pyx` |
| `pacmap_sklearn/tests/test_pacmap.py` | `sklearn/manifold/tests/test_pacmap.py` |
| `pacmap_sklearn/tests/test_localmap_rashomon.py` | contrib only |
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
| `transform`, `intermediate`, `save`/`load`, `pair_*` inputs, `knn_backend`, `save_tree` | dropped | not needed for a first upstream PR (`TSNE` has no `transform` either) |

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

## LocalMAP (`localmap=True`)

`PaCMAP(localmap=True)` reproduces the reference LocalMAP, including its
iteration indexing. With `localmap=False`, `low_dist_thres` is ignored. Phase 3 has two changes relative to PaCMAP.

1. **NN gradient.** The NN gradient coefficient is multiplied by
   $(\text{low\_dist\_thres}/2)/\sqrt{\tilde d_{ij}}$. This is an extra
   argument to the same gradient kernel; the reported loss is unchanged, as
   in the reference.
2. **FP resampling.** Every 10 iterations, FP pairs are resampled among
   points within `low_dist_thres` of each point in the current embedding.
   `resample_fp_nearby` mirrors the reference sampler's control flow
   exactly:
   * the per-slot draw counter is checked only after a duplicate or NN
     rejection, or after an acceptance;
   * after more than 100 draws, the slot keeps its old partner.

   After each resample, the incidence list is rebuilt with an O(n + P)
   counting sort (`build_incidence`), which now also serves `PaCMAP`.

**Iteration indexing kept as in the reference.**

* Phase 3 begins at `n1 + n2`, but both changes start only at
  `itr > n1 + n2`, so the first phase-3 step is a plain PaCMAP step.
* Resampling happens when `itr % 10 == 0`.
* `test_localmap_equals_pacmap_before_phase3` pins this behaviour.

**Unbounded rejection loop in the reference sampler.** A too-far rejection
never reaches the counter check. For an isolated point (no other point
within `low_dist_thres`), a slot therefore runs until it happens to draw an
NN partner or a duplicate. That takes about n / n_NN draws instead of 100.
The port reproduces this. A bounded version would change the algorithm,
so that is your call.

**Parity.**

* The gradient matches `pacmap_grad_nearby_recip_sqrt` to float32
  precision.
* Resampling statistics match `sample_FP_pair_nearby`: the fraction of
  slots that keep their old partner, the median new-pair distance, and the
  duplicate rate.
* Embedding quality matches the reference LocalMAP (`test_parity.py`).

## Rashomon trustworthy-neighbor refinement (`n_rashomon`)

This is Algorithm 2 (mean score) of Sun, Huang, Parikh & Rudin, *The
Rashomon Effect for Visualizing High-Dimensional Data* (AISTATS 2026). It is
available on `PaCMAP` only and is off by default.

1. **Candidates.** Take the $m = 50$ nearest input-space neighbors of each
   point (the raw distance order, from the same exact search).
2. **Rashomon runs.** Fit `n_rashomon` embeddings with independent seeds
   drawn from `random_state`, using the default NN pairs. The
   preprocessing and neighbor search are shared.
3. **Statistics.** For each candidate pair, over the runs compute:
   * the mean $\mu_{ij}$;
   * the population standard deviation $\sigma_{ij}$;
   * the max $m_{ij}$ of $\|y_i - y_j\|_2$;
   * the threshold $M = \operatorname{mean}_{i,j}\mu_{ij}$.
4. **Score.**
   $$\bar\Psi_{ij} = \begin{cases}\mu_{ij} + \lambda\sigma_{ij} & \mu_{ij} < M\\ m_{ij} + \lambda\sigma_{ij} & \text{otherwise}\end{cases}$$
   The `n_neighbors` candidates with the lowest $\bar\Psi_{ij}$ become the
   NN partners of $i$.
5. **Final fit.** Refit with the refined NN pairs. MN pairs are sampled
   afresh, and FP pairs are sampled afresh excluding the refined NN
   partners. The candidates and scores are exposed as
   `rashomon_candidates_` and `rashomon_scores_`.

**Where the paper is unspecific and the port had to choose.** Please
confirm or correct each of these.

| point | choice made | alternative |
|---|---|---|
| calibration set vs Rashomon set | the same `n_rashomon` runs serve as both | separate runs (paper: "separate calibration set") |
| μ and threshold | mean μ and M = mean μ (Appendix B) | median μ\*, and comparing $d^y_{ij}$ to M\* (Section 5, Eq. 5) |
| embedding standardization | none (pseudocode uses raw distances) | standardize each run to zero mean, unit mean squared norm (Sec. 3) |
| λ (`rashomon_penalty`) | 1.0 | not given in the paper |
| MN / FP pairs in the refit | resampled | not stated |
| m | 50, as in the experiments | — |

If the calibration and scoring runs were separate,
$\bar\Psi$ would average $d^{(t)}_{ij}$ over the scoring runs rather than
reusing μ. With a shared set the two coincide.

**Results.** Mean over 2 seeds, `n_rashomon=5`, λ = 1.

Metrics:

* **purity**: fraction of NN pairs whose endpoints share a label.
* **trust**: trustworthiness@10 on a 3000-point subsample.
* **10-NN**: 5-fold kNN accuracy in the embedding.

| dataset | method | n_rashomon | purity | trust | 10-NN |
|---|---|---|---|---|---|
| USPS | PaCMAP | – | 0.9335 | 0.9526 | 0.9587 |
| | PaCMAP | 5 | **0.9583** | **0.9596** | **0.9662** |
| | LocalMAP | – | 0.9335 | 0.9657 | 0.9672 |
| | LocalMAP | 5 | **0.9578** | **0.9673** | 0.9661 |
| 20NG | PaCMAP | – | 0.5903 | 0.8610 | 0.5334 |
| | PaCMAP | 5 | 0.5625 | **0.8771** | **0.5556** |
| | LocalMAP | – | 0.5903 | 0.8957 | 0.5687 |
| | LocalMAP | 5 | 0.5610 | 0.8895 | **0.5835** |

Cost is about (n_rashomon + 1)× a plain fit. The kNN search is shared;
everything else is repeated.

What the numbers show:

* **USPS.** Refinement raises NN-graph purity by about 2.5 points. It also
  improves PaCMAP's trust and 10-NN accuracy.
* **20NG.** Refinement *lowers* purity by about 3 points, yet 10-NN
  accuracy still rises by about 2 points for both estimators.
* **Likely cause on 20NG.** Here the 50 input-space candidates of a point
  are mostly cross-topic. Choosing the most stable of them is not the same
  as choosing same-label ones.

**Not supported with LocalMAP.** The LocalMAP rows above come from an
earlier version of the branch. Although its metrics hold up, the LocalMAP +
refinement embedding fragments every USPS digit into filaments and rings
(`benchmarks/plot_variants.py`).

* The refined graph stays connected.
* Its reciprocity (j is i's neighbor and i is j's) drops from 0.81 to 0.54.
* LocalMAP's phase-3 local repulsion then tears clusters apart. PaCMAP has
  no local repulsion and does not fragment.

`PaCMAP(localmap=True, n_rashomon=...)` therefore raises `ValueError` at
`fit`, including when the combination is set via `set_params`. A NOTE in
`PaCMAP._check_switches` and `test_localmap_rejects_rashomon` keep it that
way. `benchmarks/plot_variants.py` bypasses the check only to reproduce the
evidence.

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

## Approximate nearest neighbors (`neighbors_transformer`)

Both estimators take a `neighbors_transformer`. This is the scikit-learn
pattern from the gallery example [Approximate nearest neighbors in
TSNE](https://scikit-learn.org/stable/auto_examples/neighbors/approximate_nearest_neighbors.html),
adapted to PaCMAP.

```python
from pynndescent import PyNNDescentTransformer
PaCMAP(neighbors_transformer=PyNNDescentTransformer(n_neighbors=61)).fit(X)
```

**How it works.**

* A clone of the transformer is fitted on the *preprocessed* data, i.e.
  after the 100-dim PCA projection and scaling.
* It must return a sparse distance graph following the
  `KNeighborsTransformer` convention. Self-loops are dropped, and rows are
  sorted by (distance, index).
* Only the NN candidates come from the graph. Mid-near pairs still use the
  features, so the algorithm is unchanged.

**Why not `metric="precomputed"` with a sparse graph, as `TSNE` does?** The
MN rule needs distances from *i* to 6 uniformly random points, which a kNN
graph doesn't contain. Supporting that would mean changing the MN rule.

**Checks.**

* `KNeighborsTransformer(n_neighbors=61)` reproduces the internal search
  bitwise, for both estimators.
* A graph with fewer than `n_neighbors + 51` neighbors per point warns. One
  with fewer than `n_neighbors + 1` raises.
* `test_neighbors_transformer.py` covers both of the above, plus graphs
  without self-loops, unsorted rows and invalid graphs.

**70k × 784 synthetic, seed 0** (`benchmarks/bench_ann.py`, 2 cores):

| neighbors | fit [s] | kNN [s] | NN-pair recall vs exact | trust | 10-NN | triplet |
|---|---|---|---|---|---|---|
| exact (default) | 41.0 | 18.0 | 1.0000 | 0.9635 | 1.0000 | 0.7712 |
| `KNeighborsTransformer` | 47.8 | 21.1 | 1.0000 | 0.9635 | 1.0000 | 0.7712 |
| `PyNNDescentTransformer` | 37.9 | 16.9 | 1.0000 | 0.9635 | 1.0000 | 0.7712 |
| faiss HNSW, M=32 (gallery-style wrapper) | **29.2** | **8.2** | 0.9798 | 0.9634 | 1.0000 | 0.7706 |

* **faiss HNSW** brings the port to the reference's 28.0 s, which uses the
  same index, with no change in quality. The remaining ~21 s is the SVD
  and the optimizer.
* **pynndescent** is exact on this easy synthetic data and barely faster
  at this n. That matches the gallery's own conclusion: approximate
  search only pays off once its query-time gain exceeds the cost of
  building the index.
* **Variance in the exact search.** It took 18 s here and 40 s in the
  earlier run. The earlier run had other load on the machine.

## Corner cases checked against t-SNE's bug history

`TSNE` has had about 50 bug reports in scikit-learn over the years. Each one
that could apply to PaCMAP was turned into a probe:

* `benchmarks/probe_edge_cases.py` runs 53 edge-case inputs, each in its own
  subprocess with a timeout, and reports crash / hang / NaN / non-determinism.
* Every failure mode below has a regression test in
  `tests/test_corner_cases.py`. Issue numbers refer to scikit-learn.

| t-SNE failure (issue) | in the port before | now |
|---|---|---|
| Constant / identical data → NaN → crash (#28368, #32788) | finite, but numpy `RuntimeWarning`s leaked | collapsed embedding plus a clear `UserWarning`, no numpy noise. Follows the resolution proposed in #32789 |
| `set_config(transform_output="pandas")` breaks the internal PCA (#25365) | **crashed** ("output array is read-only") | internal PCA / SVD / scaler forced to numpy; output is a DataFrame with `pacmap0, pacmap1` columns |
| `init` array not validated (#7204) | **NaN `init` silently gave an all-NaN embedding** | `check_array` (finite, 2D, shape); warns on a constant column, which never moves |
| Stale or NaN reported loss (#6507, #10593) | `loss_` was NaN with 0 iterations and lagged one step otherwise | `loss_` is the loss of the final embedding, always finite |
| perplexity ≥ n silently accepted (#10805, #23471) | pair counts silently clamped | still clamped as in the reference, but now with a `UserWarning` stating requested vs used |
| Overflow in the distance transform (#19471 class) | `d²` overflowed for precomputed distances ~1e200 | scaled distance computed as `(d/σᵢ)(d/σⱼ)`; scale-invariant, no warnings |
| NaN / divergence surfacing late (#6665, #9026) | not observed | final finiteness check raises a clear "diverged, reduce learning_rate" error |
| O(N²) memory (#7089, #18966) | Rashomon refinement built an (n, 50, dim) temporary, 800 MB at n = 10⁶ | computed in row chunks |
| OpenMP silently off (#29665) | — | verified linked (`libgomp`) and used (3.6 s → 2.1 s on 1 → 2 threads) |
| Ties / thread nondeterminism (#15821, #15082) | — | tied-distance data is bitwise identical across thread counts (test) |
| `n_components=1` / large `n_components` (#4154, #9941) | fine | tested 1, 3, 10, 25 |
| float32 internals, underflow (#4124, #19471) | fine (float64 throughout) | test: float32 and float64 input give identical output; inputs at 1e-30 / 1e30 scale give the same embedding |
| Sparse, NaN, inf, zero-feature, 1-sample input | clean errors | unchanged |

**Unbounded LocalMAP resampling.** The reference checks its 100-draw cutoff
only after duplicate / NN rejections. A point with no other point within
`low_dist_thres` therefore loops about n / n_NN times, and a resampling
round costs O(n²).

* A round on isolated points took 5.6 s at 20k and 37 s at 50k.
* A 20k fit with `low_dist_thres=1e-3` took 149 s, against 9 s at the default.
* Outliers and clusters of ≤ n_neighbors points trigger it, not only tiny
  thresholds.

The port now counts every draw:

* a 50k round takes 0.7 s, and the 20k fit takes 16 s;
* on normal data, the fraction of kept-old pairs and the median new-pair
  distance still match the reference at thresholds 2, 5 and 10.

**Not changed, by design.**

* **Default kNN algorithm.** At 100k × 10, sklearn's `algorithm="auto"`
  picks a kd-tree. On uniform data that takes 70 s vs 18 s brute force. On
  data with intrinsic dimension 3 the tree wins, 4.7 s vs 18 s. `TSNE` has
  the same default (#10044). Users can pass
  `neighbors_transformer=KNeighborsTransformer(algorithm="brute", ...)`.
* **Tiny negative precomputed distances.** These are rejected, as `TSNE`
  does (#11376, #5772).

## Runtime warnings

The code warns at fit time wherever results come from territory that has not
been verified. Every warning is conditional, so default fits on digits,
USPS and 20NG emit none.

| warning | when it fires | why |
|---|---|---|
| `n_rashomon is experimental …` | `n_rashomon` is set | lists the five implementation choices the paper leaves open |
| `… local graph adjustment is largely inactive …` | LocalMAP, more than 50% of FP slots in the last resampling round kept their old partner | `low_dist_thres` is too small for the embedding's scale, so LocalMAP is effectively PaCMAP |
| `Exact nearest-neighbor search on n_samples=… costs O(n_samples^2) …` | n_samples > 100k, no `neighbors_transformer`, not precomputed | points to an ANN transformer |
| `… compiled without OpenMP …` | the extension was built without OpenMP | the macOS / Windows builds are untested; results are identical, only single-threaded |
| `All samples are identical …` | zero-variance input | the embedding is degenerate |
| `… too small for the requested pairs …` | pair counts reduced for small n | states requested vs used |
| `init has a constant column …` | array `init` with a constant column | that coordinate never moves |
| `neighbors_transformer returned k neighbors …` | the graph is smaller than `n_neighbors + 50` | the candidate pool differs from the exact search |

`pacmap_sklearn._pacmap_utils.OPENMP_ENABLED` reports whether the build has
OpenMP. `test_this_build_has_openmp` fails on a build without it.

## Open questions to settle before proposing upstream

1. **ANN plug-in API.** `neighbors_transformer` (an estimator as a parameter)
   is the chosen design. Reviewers may still ask for `TSNE`'s sparse
   `metric="precomputed"` route. That would need a different MN rule for
   graph-only input, an algorithmic decision for the authors.
2. **Exact kNN cost.** The default brute-force search on the 100-dim PCA
   projection is O(n²·100). It dominates beyond ~10⁵ points; there,
   `neighbors_transformer` is the escape hatch.
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
