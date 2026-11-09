# Covariates and fixed effects

Two designs with 8 periods and 7 cohorts and extra regressors. Every library
fits the same task from the same Parquet file (see [the overview](index.md)).

- **G: two covariates.** Callaway-Sant'Anna by outcome regression, inverse
  probability weighting and doubly robust estimation, and the other
  estimators with the covariates in the regression.
- **H: region-by-period fixed effects (240 levels) and a covariate.**
  Sun-Abraham, BJS and the two-stage estimator, with the fixed effects
  estimated iteratively (the default at this many levels) and, for
  Sun-Abraham and BJS, with the dense solver.

## G: two covariates

```{image} ../figures/covariates_G_time.light.svg
:class: only-light benchmark
:alt: Wall time against the number of units with two covariates, one line per implementation, log-log
```
```{image} ../figures/covariates_G_time.dark.svg
:class: only-dark benchmark
:alt: Wall time against the number of units with two covariates, one line per implementation, log-log
```

- **diff-stream** fits a million units in 2.5 to 4.0 seconds,
  depending on the estimator, and ten million in 48.6 to 58.6
  seconds. The three Callaway-Sant'Anna variants take 2.6 (regression),
  3.0 (inverse probability weighting) and 3.0 (doubly
  robust) seconds at a million units.
- **Callaway-Sant'Anna in other implementations:** at a million units diff-diff
  takes 9.0 to 12.8 seconds and R's `did`
  24.8 to 34.2. At ten million units diff-diff takes
  68.5 (regression), 95.0 (inverse probability weighting) and
  132 (doubly robust) seconds, with a peak of 24.4 GB, against
  3.8 GB for diff-stream. R's `did` runs out of memory.
- **Sun-Abraham:** fixest's `sunab` takes 26.7 seconds at a million
  units and diff-diff 41.1; both run out of memory at ten million.
- **BJS, two-stage, extended TWFE:** diff-diff's imputation estimator takes
  31.5 seconds at a million units and runs out of memory at
  ten million. R's `didimputation`, `did2s` and `etwfe` and diff-diff's two-stage estimator
  run out of memory at a million units.
- diff-stream's peak at ten million units is 3.5 to 3.8 GB.

| Estimator | Implementation | 10,000 | 100,000 | 1,000,000 | 10,000,000 |
|---|---|---:|---:|---:|---:|
| Callaway-Sant'Anna | diff-stream, regression | 0.05 | 0.24 | 2.6 | 49.5 |
|  | diff-stream, IPW | 0.05 | 0.28 | 3.0 | 58.1 |
|  | diff-stream, doubly robust | 0.05 | 0.29 | 3.0 | 58.4 |
|  | diff-stream, doubly robust, 1 thread | 0.09 | 0.74 | 7.7 | 103 |
|  | did (R, regression) | 2.8 | 3.8 | 24.8 | out of memory |
|  | diff-diff CallawaySantAnna (Python, regression) | 0.34 | 0.57 | 9.0 | 68.5 |
|  | did (R, IPW) | 1.7 | 4.4 | 29.6 | out of memory |
|  | diff-diff CallawaySantAnna (Python, IPW) | 0.13 | 0.68 | 7.4 | 95.0 |
|  | did (R, doubly robust) | 1.6 | 4.7 | 34.2 | out of memory |
|  | diff-diff CallawaySantAnna (Python, doubly robust) | 0.16 | 0.96 | 12.8 | 132 |
| Sun-Abraham | diff-stream | 0.05 | 0.28 | 3.0 | 48.6 |
|  | diff-stream, 1 thread | 0.09 | 0.67 | 6.7 | 91.0 |
|  | fixest sunab (R) | – | – | 26.7 | out of memory |
|  | diff-diff SunAbraham (Python) | – | – | 41.1 | out of memory |
| BJS | diff-stream | 0.06 | 0.33 | 3.4 | 53.0 |
|  | diff-stream, 1 thread | 0.10 | 0.73 | 7.3 | 98.8 |
|  | didimputation (R) | – | – | out of memory | – |
|  | diff-diff ImputationDiD (Python) | – | – | 31.5 | out of memory |
| Two-stage | diff-stream | 0.04 | 0.24 | 2.5 | 52.0 |
|  | diff-stream, 1 thread | 0.07 | 0.49 | 5.2 | 80.9 |
|  | did2s (R) | – | – | out of memory | – |
|  | diff-diff TwoStageDiD (Python) | – | – | out of memory | – |
| Extended TWFE | diff-stream | 0.12 | 0.44 | 4.0 | 58.6 |
|  | diff-stream, 1 thread | 0.09 | 0.66 | 6.6 | 92.8 |
|  | etwfe (R) | – | – | out of memory | – |

"out of memory" and a dash mean the same as on the [previous page](scale.md).

## H: many fixed-effect levels

```{image} ../figures/covariates_H_time.light.svg
:class: only-light benchmark
:alt: Wall time against the number of units with 240 region-by-period effects, one line per implementation, log-log
```
```{image} ../figures/covariates_H_time.dark.svg
:class: only-dark benchmark
:alt: Wall time against the number of units with 240 region-by-period effects, one line per implementation, log-log
```

- **BJS** fits a million units in 16.5 seconds and ten million in
  226. The dense solver takes 29.5 and 302.
- **Two-stage** takes 11.7 seconds at a million units and 146 at ten million.
- **Sun-Abraham** with the iterative solver takes 16.2 seconds at a million units, against
  50.9 for the dense solver, but at ten million units the iterative solver takes
  170 seconds and the dense solver 537. The dense solver also peaks at
  4.1 GB, where the iterative solver reaches 5.7 GB, a figure that
  counts pages of a memory-mapped file (see [the overview](index.md#peak-memory)). For which
  solver `"auto"` picks, see [Sun-Abraham](../estimators/sun_abraham.md#fixed-effects-with-many-levels).
- **Other implementations:** fixest's `sunab` takes 27.9 seconds at a million units and runs out of
  memory at ten million. R's `didimputation` takes 196 seconds at
  100,000 units and runs out of memory at a million. R's `did2s` was not installed for this design.
- One thread takes 432 seconds for Sun-Abraham,
  364 for BJS and
  282 for two-stage at ten million units.

| Estimator | Implementation | 10,000 | 100,000 | 1,000,000 | 10,000,000 |
|---|---|---:|---:|---:|---:|
| Sun-Abraham | diff-stream | 0.27 | 2.4 | 16.2 | 170 |
|  | diff-stream, 1 thread | 0.45 | 4.1 | 40.7 | 432 |
|  | diff-stream, dense solver | 0.68 | 5.5 | 50.9 | 537 |
|  | fixest sunab (R) | 0.98 | 3.6 | 27.9 | out of memory |
| BJS | diff-stream | 0.59 | 2.0 | 16.5 | 226 |
|  | diff-stream, 1 thread | 0.38 | 2.8 | 29.2 | 364 |
|  | diff-stream, dense solver | 0.50 | 3.3 | 29.5 | 302 |
|  | didimputation (R) | 3.7 | 196 | out of memory | – |
| Two-stage | diff-stream | 0.42 | 1.3 | 11.7 | 146 |
|  | diff-stream, 1 thread | 0.28 | 2.1 | 21.8 | 282 |

Every comparator that finished agrees with diff-stream's fit of the same
estimator to 2e-8 or better, except diff-diff's doubly robust
Callaway-Sant'Anna, which agrees to 2.1e-6.

Figures for both designs (time, memory and disk) are in `docs/figures/`, and
the numbers are in `benchmarks/results/covariates.csv`.
