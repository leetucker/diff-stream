# Fixed effects with millions of levels

Two designs with 8 periods and 7 cohorts, in which units belong to counties
(one county for every 40 units, so a million units have 25,000 counties and ten
million have 250,000) and the fixed effects are absorbed out of core by
[hdfe-stream](https://github.com/leetucker/hdfe-stream):

- **I: county-by-period effects.** At ten million units, two million levels.
- **J: county-by-period and industry-by-period effects, and a covariate.**

The benchmark fits Sun-Abraham and BJS, the estimators that take fixed effects
with any number of levels. fixest's `sunab` is the only other implementation of a
comparable regression that ran.

```{image} ../figures/absorbed_I_time.light.svg
:class: only-light benchmark
:alt: Wall time against the number of units with county-by-period effects, one line per implementation, log-log
```
```{image} ../figures/absorbed_I_time.dark.svg
:class: only-dark benchmark
:alt: Wall time against the number of units with county-by-period effects, one line per implementation, log-log
```

- **BJS** fits a million units in 17.5 seconds in design I and
  25.2 in design J, and ten million in 260 and
  399.
- **Sun-Abraham** takes 14.7 and 21.0 seconds at a million units, but at ten million
  units it takes 173 seconds in design I and 252 in design J,
  less than BJS takes.
- **fixest's `sunab`** takes 29.1 and 32.6 seconds at a million units,
  with peaks of 19.8 and 20.5 GB, and runs
  out of memory at ten million.
- **R's `didimputation`** takes 229 and 239 seconds at
  100,000 units. At a million and ten million units the benchmark's call fails
  (`predict()` needs the fixed-effect values that fixest drops with this many
  levels), so there is no result.
- With one thread, the fits take 1.7 to 2.8 times as long at a million units and
  1.9 to 2.6 times as long at ten million.

## Design I

| Estimator | Implementation | 10,000 | 100,000 | 1,000,000 | 10,000,000 |
|---|---|---:|---:|---:|---:|
| Sun-Abraham | diff-stream | 0.25 | 1.8 | 14.7 | 173 |
|  | diff-stream, 1 thread | 0.45 | 4.1 | 41.5 | 442 |
|  | diff-stream, 4 GB limit | 0.27 | 4.0 | 18.8 | 326 |
|  | fixest sunab (R) | 1.0 | 3.3 | 29.1 | out of memory |
| BJS | diff-stream | 0.47 | 1.9 | 17.5 | 260 |
|  | diff-stream, 1 thread | 0.35 | 2.6 | 29.9 | 492 |
|  | diff-stream, 4 GB limit | 0.49 | 2.0 | 20.5 | 487 |
|  | didimputation (R) | 3.9 | 229 | – | – |

## Design J

| Estimator | Implementation | 10,000 | 100,000 | 1,000,000 | 10,000,000 |
|---|---|---:|---:|---:|---:|
| Sun-Abraham | diff-stream | 0.33 | 2.4 | 21.0 | 252 |
|  | diff-stream, 1 thread | 0.59 | 5.4 | 56.3 | 620 |
|  | fixest sunab (R) | 0.89 | 3.6 | 32.6 | out of memory |
| BJS | diff-stream | 0.65 | 2.7 | 25.2 | 399 |
|  | diff-stream, 1 thread | 0.52 | 4.3 | 49.0 | 777 |
|  | didimputation (R) | 4.0 | 239 | – | – |

Every comparator that finished agrees with diff-stream's fit to 6e-9 or
better.

With `max_memory_gb=4` (and Polars held to 4 threads, as a small job would be), Sun-Abraham in
design I takes 326 seconds at ten million
units, against 173 without a limit, with a peak of
6.0 GB (which counts memory-mapped pages; see
[the overview](index.md#peak-memory)) and 5.1 GB of scratch
disk. BJS takes 487 seconds under the same limit, against
260.

Figures (time, memory and disk) are in `docs/figures/`, and the numbers are in
`benchmarks/results/absorbed.csv`.
