# Benchmarks

The comparisons with the R, Stata and Python implementations are written
(`benchmarks/`) and wait for a run on a large machine; their results will
replace this page.

## This machine

A laptop (8 cores, 26 GB, WSL2), Python 3.12, the simulated panel of
`simulate_panel(1_000_000, 8)`: one million units over 8 periods (8 million
rows, 7 cohorts), read from Parquet, no covariates. Wall time of a whole fit in
a fresh process after the first fit has compiled the kernels, the median of
three runs:

| | 1 thread | 4 threads |
|---|---:|---:|
| `did_callaway_santanna` | 1.8 s | 1.4 s |
| `did_sun_abraham` | 1.7 s | 1.4 s |
| `did_bjs` | 1.6 s | 1.0 s |
| `did_two_stage` | 4.9 s | 4.2 s |
| `did_extended_twfe` | 6.3 s | 5.4 s |

A wider panel, 200,000 units over 30 periods (29 cohorts, 841 cohort-period
effects): `did_callaway_santanna` in 1.5 s on 4 threads.

With `max_memory_gb=1`, every estimator completes on the 8-million-row panel,
with a peak resident memory of about 0.8 GB (the panel is then split into 14
pieces on disk); without a limit the peak is about 1.6 GB. See
[Memory and threads](../resources.md).
