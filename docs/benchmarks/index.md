# Benchmarks

The benchmarks compare diff-stream with the R and Python implementations of the
same estimators (R's `did`, `fixest`, `didimputation`, `did2s`, `etwfe`,
`bacondecomp` and `TwoWayFEWeights`; Python's `diff-diff` and `pyfixest`) on
simulated panels of 10,000 to 10 million units, as time and peak memory against
the number of units. Stata's commands are not benchmarked. See
`benchmarks/README.md` in the repository to run them.

- **diff-stream scales to panels the others cannot hold.** At ten million units
  over 8 periods (80 million rows) all six estimators finish in
  19.2 to 27.6 seconds with a peak of 3.4 to 3.7 GB, and at ten million
  units over 30 periods (300 million rows) in
  72.8 to 115 seconds. At that size most other implementations run out of memory
  on this 30 GB machine; the ones that finish (R's `did`, fixest on stacked
  data, diff-diff's Callaway-Sant'Anna with few cohorts) take as long or longer and
  need 17.0 to 21.2 GB.
- **It is faster at a million units.** The fits take 0.83 to 1.8 seconds, against
  2.9 to 16.4 for Callaway-Sant'Anna in other
  implementations, and from a few seconds to over 800 seconds for the other estimators. Below
  about 100,000 units the faster Python and R implementations are comparable.
- **Covariates and fixed effects cost more** but scale the same way
  ([covariates](covariates.md)). With millions of fixed-effect levels BJS fits ten
  million units in 262 to 396 seconds; Sun-Abraham needs
  1,684 seconds or more ([absorbed fixed effects](absorbed.md)).
- **Estimates agree** with the other implementations to 5e-8 or better where
  they use the same conventions; the exceptions are listed on each page.

```{toctree}
:maxdepth: 1

scale
covariates
absorbed
```

## How the benchmarks were run

All the results come from one EC2 spot instance (c8id.4xlarge: Intel Xeon 6975P-C,
16 vCPUs, 30.8 GB of memory, local NVMe disk), with Python 3.13, NumPy 2.5,
numba 0.68 and Polars 2.1. Each fit runs in its own process, with a limit of
1,800 seconds, and the memory of the benchmarks capped at 96% of the machine. A
fit that exceeds the limit or runs out of memory is a result, and the
configuration is not run at larger sizes. diff-stream's fits are timed after a
warm-up fit of 2,000 units that loads its cached compiled kernels; the first fit in a
new installation spends several seconds compiling. Every library fits the same task, the event
study with its standard errors, from the same Parquet file, and the estimates
of each are compared with diff-stream's fit of the same estimator.

(peak-memory)=
## Peak memory

Peak memory in the benchmarks is the process's resident high-water mark, for
every library alike. It counts resident pages of memory-mapped files as well as
ordinary memory. With high-dimensional fixed effects, diff-stream's Sun-Abraham
and BJS (through hdfe-stream) map a table from disk, so most of their reported
peak can be file-backed memory that the operating system drops under pressure
rather than memory a limit has to hold. A reported peak at the benchmark's cap
(about 29.6 GB) means the resident pages, file-backed ones included, reached the
cap; it does not measure the memory the fit needs.
