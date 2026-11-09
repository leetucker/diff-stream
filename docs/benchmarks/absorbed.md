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

- **BJS** fits a million units in 17.4 seconds in design I and
  24.8 in design J, and ten million in 262 and
  396.
- **Sun-Abraham** takes 18.3 and 24.1 seconds at a million units, but at ten million
  units it takes 1,684 seconds in design I and times out at 1,800 seconds in design J.
  BJS fits these designs much faster than Sun-Abraham at ten million units.
- **fixest's `sunab`** takes 29.1 and 32.6 seconds at a million units,
  with peaks of 19.8 and 20.5 GB, and runs
  out of memory at ten million.
- **R's `didimputation`** takes 229 and 239 seconds at
  100,000 units. At a million and ten million units the benchmark's call fails
  (`predict()` needs the fixed-effect values that fixest drops with this many
  levels), so there is no result.
- With one thread, the fits take 1.7 to 2.5 times as long at a million units and
  1.9 to 2.0 times as long at ten million, where Sun-Abraham times out in both designs.

## Design I

| Estimator | Implementation | 10,000 | 100,000 | 1,000,000 | 10,000,000 |
|---|---|---:|---:|---:|---:|
| Sun-Abraham | diff-stream | 0.29 | 2.1 | 18.3 | 1,684 |
|  | diff-stream, 1 thread | 0.46 | 4.3 | 44.0 | timeout |
|  | diff-stream, 4 GB limit | 0.29 | 2.3 | 19.3 | 1,656 |
|  | fixest sunab (R) | 1.0 | 3.3 | 29.1 | out of memory |
| BJS | diff-stream | 0.61 | 2.0 | 17.4 | 262 |
|  | diff-stream, 1 thread | 0.33 | 2.7 | 29.7 | 491 |
|  | diff-stream, 4 GB limit | 0.59 | 2.1 | 16.5 | 617 |
|  | didimputation (R) | 3.9 | 229 | – | – |

## Design J

| Estimator | Implementation | 10,000 | 100,000 | 1,000,000 | 10,000,000 |
|---|---|---:|---:|---:|---:|
| Sun-Abraham | diff-stream | 0.37 | 2.9 | 24.1 | timeout |
|  | diff-stream, 1 thread | 0.61 | 5.8 | 59.5 | timeout |
|  | fixest sunab (R) | 0.89 | 3.6 | 32.6 | out of memory |
| BJS | diff-stream | 0.71 | 2.7 | 24.8 | 396 |
|  | diff-stream, 1 thread | 0.52 | 4.3 | 49.1 | 782 |
|  | didimputation (R) | 4.0 | 239 | – | – |

Every comparator that finished agrees with diff-stream's fit to 6e-9 or
better.

With `max_memory_gb=4`, Sun-Abraham in design I still reports a peak of
29.8 GB at ten million units; the peak counts
memory-mapped pages (see [the overview](index.md#peak-memory)).

Figures (time, memory and disk) are in `docs/figures/`, and the numbers are in
`benchmarks/results/absorbed.csv`.
