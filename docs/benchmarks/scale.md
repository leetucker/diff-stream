# Panels without covariates

Time and peak memory as the number of units grows, for the event study (with
standard errors) of each estimator, on five simulated designs:

| Design | Periods | Cohorts | Rows at 1,000,000 units |
|---|---:|---:|---:|
| A | 8 | 3 | 8 million |
| B | 8 | 7 | 8 million |
| C | 30 | 3 | 30 million |
| D | 30 | 29 | 30 million |
| E | 8 | 7, with 10% of the rows missing | 7.3 million |

Every library fits the same task from the same Parquet file. Times are the
whole fit, including reading the file for the libraries that read it first. The
first fit in a new installation spends several seconds compiling diff-stream's
kernels, once; the benchmarks fit 2,000 units first, so the times are for fits
with the compiled kernels loaded. See [the overview](index.md) for the machine
and the limits.

## Design B: 8 periods, 7 cohorts

```{image} ../figures/scale_B_time.light.svg
:class: only-light benchmark
:alt: Wall time against the number of units for each estimator, one line per implementation, log-log
```
```{image} ../figures/scale_B_time.dark.svg
:class: only-dark benchmark
:alt: Wall time against the number of units for each estimator, one line per implementation, log-log
```

- **diff-stream** fits a million units (8 million rows) in 0.87 to 1.6 seconds,
  depending on the estimator, and ten million (80 million rows) in
  18.7 to 25.7 seconds. Time grows a little faster than the number of units: ten
  times the units takes 15 to 23 times as long.
- **Callaway-Sant'Anna:** at a million units R's `did` takes 16.4
  seconds and diff-diff 2.9, against 1.0 for
  diff-stream. At ten million units `did` takes 161 seconds and
  diff-diff 34.5 (with a peak of 17.0 GB),
  against 20.1 for diff-stream.
- **Sun-Abraham:** at a million units fixest's `sunab` takes 26.2
  seconds, diff-diff 39.1
  and pyfixest 105.
- **Extended TWFE:** at a million units R's `etwfe` takes 867
  seconds, diff-diff 48.0.
- **BJS, two-stage:** diff-diff's imputation estimator takes
  25.2 seconds at a million
  units. R's `didimputation` takes 223 seconds at
  100,000 units and runs out of memory at a million. diff-diff's two-stage
  estimator takes 7.8 seconds at 10,000
  units and runs out of memory at 100,000.
- **Stacked DiD:** fixest on the stacked data takes
  45.3 seconds at ten million units, against
  22.5 for diff-stream, and diff-diff's `StackedDiD` runs out of memory.

| Estimator | Implementation | 10,000 | 100,000 | 1,000,000 | 10,000,000 |
|---|---|---:|---:|---:|---:|
| Callaway-Sant'Anna | diff-stream | 0.04 | 0.16 | 1.0 | 20.1 |
|  | diff-stream, 1 thread | 0.04 | 0.26 | 2.1 | 32.0 |
|  | did (R) | 1.4 | 3.2 | 16.4 | 161 |
|  | diff-diff CallawaySantAnna (Python) | 0.05 | 0.28 | 2.9 | 34.5 |
| Sun-Abraham | diff-stream | 0.04 | 0.16 | 1.0 | 20.1 |
|  | diff-stream, 1 thread | 0.10 | 0.26 | 2.1 | 31.7 |
|  | fixest sunab (R) | 0.86 | 3.3 | 26.2 | out of memory |
|  | pyfixest saturated (Python) | 2.0 | 11.0 | 105 | out of memory |
|  | diff-diff SunAbraham (Python) | 0.44 | 3.5 | 39.1 | out of memory |
| BJS | diff-stream | 0.04 | 0.15 | 0.87 | 18.7 |
|  | diff-stream, 1 thread | 0.04 | 0.23 | 2.0 | 30.6 |
|  | didimputation (R) | 3.9 | 223 | out of memory | – |
|  | diff-diff ImputationDiD (Python) | 0.30 | 2.3 | 25.2 | out of memory |
| Two-stage | diff-stream | 0.04 | 0.16 | 1.2 | 23.7 |
|  | diff-stream, 1 thread | 0.05 | 0.34 | 3.3 | 43.7 |
|  | diff-diff TwoStageDiD (Python) | 7.8 | out of memory | – | – |
| Extended TWFE | diff-stream | 0.05 | 0.22 | 1.6 | 25.7 |
|  | diff-stream, 1 thread | 0.05 | 0.33 | 3.0 | 40.3 |
|  | etwfe (R) | 13.6 | 96.7 | 867 | out of memory |
|  | diff-diff ETWFE (Python) | 0.47 | 4.0 | 48.0 | out of memory |
| Stacked DiD | diff-stream | 0.04 | 0.18 | 1.2 | 22.5 |
|  | diff-stream, 1 thread | 0.04 | 0.26 | 2.4 | 33.9 |
|  | fixest on stacked data (R) | 0.48 | 1.1 | 5.4 | 45.3 |
|  | diff-diff StackedDiD (Python) | 0.10 | 0.66 | 7.4 | out of memory |

"out of memory" is a process killed by the operating system or a memory error
in the library. A dash means the configuration was not run at that size; a
configuration that fails, times out or runs out of memory is not run at larger
sizes.

```{image} ../figures/scale_B_memory.light.svg
:class: only-light benchmark
:alt: Peak memory against the number of units for each estimator, one line per implementation, log-log
```
```{image} ../figures/scale_B_memory.dark.svg
:class: only-dark benchmark
:alt: Peak memory against the number of units for each estimator, one line per implementation, log-log
```

diff-stream's memory grows slowly: at ten million units on 8 periods its peak
is 3.4 to 3.6 GB with all threads and
2.3 to 2.4 GB with one (design A), with
1.4 GB of scratch disk for the sorted panel. The libraries that hold the
panel in memory need much more: at the same size R's `did` peaks at
21.2 GB, and diff-diff's Callaway-Sant'Anna and fixest on the
stacked data at 17.0 GB.

With few cohorts the gap in time is smaller. In design A (3 cohorts) at ten
million units diff-diff's Callaway-Sant'Anna takes 26.8 seconds and
fixest on the stacked data 28.8, against 19.0 and
21.1 for diff-stream, which is a modest difference in time and a
large one in memory (3.6 GB against 17.0 GB). Below about 100,000 units
the faster Python and R implementations are within a small factor of
diff-stream or ahead of it.

## All designs

Seconds to fit, all threads, at 1,000,000 units:

| Estimator | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| Callaway-Sant'Anna | 0.95 | 1.0 | 3.9 | 8.3 | 1.8 |
| Sun-Abraham | 0.94 | 1.0 | 3.9 | 8.2 | 1.4 |
| BJS | 0.91 | 0.87 | 3.8 | 3.8 | 2.3 |
| Two-stage | 1.1 | 1.2 | 7.0 | 7.2 | 1.2 |
| Extended TWFE | 1.4 | 1.6 | 5.8 | 15.8 | 1.6 |
| Stacked DiD | 1.2 | 1.2 | 4.3 | 8.2 | 1.4 |

and at 10,000,000 units:

| Estimator | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| Callaway-Sant'Anna | 19.0 | 20.1 | 82.7 | 126 | 27.7 |
| Sun-Abraham | 19.1 | 20.1 | 82.6 | 127 | 24.4 |
| BJS | 18.5 | 18.7 | 83.0 | 82.7 | 32.6 |
| Two-stage | 23.6 | 23.7 | 116 | 118 | 26.3 |
| Extended TWFE | 23.7 | 25.7 | 103 | 207 | 26.1 |
| Stacked DiD | 21.1 | 22.5 | 89.9 | 129 | 24.4 |

- **More periods and more cohorts cost more.** Design C (30 periods, 3 cohorts)
  takes 3.8 to 7.0 seconds at a million units and 82.6 to 116 at ten million.
  Design D (29 cohorts) takes 3.8 to 15.8 and 82.7 to 207. Extended TWFE is the
  slowest in D because its regression has a column for every cohort-period
  cell.
- **Unbalanced panels (E)** take 1.2 to 2.3 seconds at a million units and
  24.4 to 32.6 at ten million. BJS is the slowest, because its standard errors
  need extra passes over the rows when the panel is unbalanced.
- **Other implementations stop earlier in the larger designs.** At a million
  units in design D, R's `did` takes 590 seconds and diff-diff's
  Callaway-Sant'Anna 26.4, against 8.3 for
  diff-stream; of the other comparators only diff-diff's imputation estimator
  (213 seconds) finished at that size. At 10,000 units in design D, R's `etwfe` does not finish in
  1,800 seconds and diff-diff's ETWFE takes 48.7, against 0.62.
  At ten million units in designs C and D (300 million rows) no comparator
  finished.

```{image} ../figures/scale_D_time.light.svg
:class: only-light benchmark
:alt: Wall time against the number of units in design D (30 periods, 29 cohorts), log-log
```
```{image} ../figures/scale_D_time.dark.svg
:class: only-dark benchmark
:alt: Wall time against the number of units in design D (30 periods, 29 cohorts), log-log
```

## Threads

The times above use all 16 vCPUs. At ten million units one thread takes 1.3 to 2.6 times
as long, and at a million units 1.2 to 3.5 times (the large ratios are in the
larger designs and the two-stage estimator): in design B at ten million units Callaway-Sant'Anna takes
32.0 seconds on one thread against 20.1, and in design C the
two-stage estimator takes 310 against 116. For
unbalanced panels the difference is larger:
Callaway-Sant'Anna in design E takes 78.1 seconds on one thread and
27.7 on all. The one-thread fits and the other variants (memory limits, the
dense solver) agree with the default fit to 5e-14 or better.

## Memory limits

With `max_memory_gb=2`, design B at a million units peaks at about
0.8 GB, in 1.8 to 2.5 seconds, against a peak of about
1.8 GB and 0.87 to 1.6 seconds
without a limit. At ten million units (80 million rows) it still fits, at a peak of 1.0 GB in 27.0 to 33.3 seconds, with Polars held to 4 threads, as a job with a 2 GB allocation would be. Polars' read buffers grow with its thread pool (about 130 MB a thread), so with every thread of the benchmark machine they alone can exceed 2 GB at this size.
See [Memory and threads](../resources.md) for what a limit does and how to choose one.

## Diagnostics (design F)

Goodman-Bacon decomposition and TWFE weights, on the panel of design B, with
the time to the whole diagnostic:

| Diagnostic | Implementation | 10,000 | 100,000 | 1,000,000 | 10,000,000 |
|---|---|---:|---:|---:|---:|
| Goodman-Bacon decomposition | diff-stream | 0.03 | 0.12 | 0.69 | 17.5 |
|  | diff-stream, 1 thread | 0.03 | 0.20 | 1.7 | 29.2 |
|  | bacondecomp (R) | 854 | out of memory | – | – |
|  | diff-diff BaconDecomposition (Python) | 0.21 | 1.5 | 18.0 | 217 |
| TWFE weights | diff-stream | 0.11 | 0.92 | 8.4 | 94.4 |
|  | diff-stream, 1 thread | 0.12 | 0.97 | 8.9 | 103 |
|  | TwoWayFEWeights (R) | 1.1 | 7.4 | 58.8 | out of memory |

## Agreement

Every comparator that finished agrees with diff-stream's fit of the same
estimator to 5e-8 or better in designs A to D (most to 1e-9 or better), as the
largest absolute difference of any coefficient or standard error. In design E,
with 10% of the rows missing, R's `etwfe` and diff-diff's ETWFE differ from
diff-stream's extended TWFE by 0.002, diff-diff's `StackedDiD` by 1.5e-4, and
pyfixest's saturated regression by 0.007 to 0.018; the other comparators agree
to 2.2e-8 or better. The conventions for unbalanced panels are listed under
[differences between implementations](../implementations.md).

R's `did2s` was not installed for these designs, so there is no R two-stage
comparison here.

Figures for every design are in `docs/figures/` (time, memory and disk), and
the numbers are in `benchmarks/results/scale.csv`.
