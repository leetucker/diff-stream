# Memory, disk and threads

## Memory

`max_memory_gb` is the memory a fit may use, and `memory_reserve` the share
of it held back for allocations the estimates do not model (stream-limits'
`MemoryBudget`). By default both come from the environment: the job's
allocation on a scheduler (PBS, Slurm, a container's cgroup) or the machine's
memory, and a 15% reserve.

Two things follow from the memory left:

- **How the panel is staged.** With the default `preprocess_mb=None`, a
  quarter of the memory left is the working memory for sorting and checking
  the panel. A panel that fits is sorted in one piece; a larger one is split
  into whole-unit pieces on disk that fit. Pass `preprocess_mb` to cap it
  explicitly.
- **The numerical batch.** With the default `batch_units=None`,
  Callaway-Sant'Anna, Sun-Abraham, BJS and ETWFE process as many units at a
  time as is fast (16,384 or 4,096) and halve that until the batch's buffers fit.
  The next batch is read in a background thread while the current one is
  worked on, so up to one more batch of decoded units is in memory.
- **Passes over the staged panel.** Callaway-Sant'Anna with covariates makes
  about ten passes over the units. When their outcomes, covariates and weights,
  decoded, take at most a tenth of the memory left, they are kept after the first
  pass instead of being read from disk each time; otherwise every pass reads
  them again.

Before each large allocation the fit checks the memory left and stops with
`MemoryError` if the allocation would not fit, before making it. These are
checks, not a hard limit: the source query, Polars and Arrow's pools and the
allocator can hold more than the estimates. Enforce a hard limit with the
job's own limit or stream-limits' launcher.

What does not shrink with the batch: the joint covariance of the effects
(cohort-period cells squared), and the design of a model with many covariates
or fixed effects (its columns squared). A panel with 50 periods and 50 cohorts
has about 2,500 Callaway-Sant'Anna effects, a 50 MB covariance.

On the 8-million-row simulated panel of the [benchmarks](benchmarks/scale.md),
every estimator fits with `max_memory_gb=2` at a peak of about 0.8 GB, and
peaks at about 1.8 GB without a limit. On the 80-million-row panel the limit
holds only if Polars has few threads: the memory it holds while reading grows with
its thread pool, about 130 MB for each thread, and is not covered by
`max_memory_gb`. In a test on that panel the fit peaked at about 1.1 GB with 4
Polars threads, 1.5 GB with 8 and 2.6 GB with 16, so with 16 threads
`max_memory_gb=2` stops with `MemoryError` before allocating. Set the thread
limit before Python starts (see [Threads](#threads)).

Sun-Abraham with fixed effects absorbed iteratively keeps to the memory limit
only approximately; see
[Sun-Abraham](estimators/sun_abraham.md#fixed-effects-with-many-levels).

## Disk

`workdir` (default: the system temporary directory) holds a sorted copy of
the columns the fit uses, and the pieces when the panel is split: about 64
to 128 bytes per row. It is removed when the fit returns, except for BJS
results with covariates, additional fixed effects, weights or an unbalanced
panel, which keep it for aggregation until `cleanup()`. With fixed effects
absorbed iteratively, BJS also keeps 8 bytes per unit and cohort-period cell
(the untreated scores), and hdfe-stream's own files for the untreated model
while it fits.

## Threads

`n_threads` holds numba and BLAS to that many threads during the fit
(stream-limits' `thread_scope`); the default is every core the job has.
Polars' thread pool is fixed when polars is first imported, so on a shared
machine set it in the job script, before Python starts:

```bash
export STREAM_LIMITS_MAX_THREADS=8
```

or call `stream_limits.limit_threads(8)` before importing polars. Importing
`diff_stream` first also applies the scheduler's allocation to Polars.

Importing `diff_stream` also sets `OMP_WAIT_POLICY=passive`, so that idle threads
sleep between the compiled passes instead of spinning. In the benchmarks this was
2% to 15% faster at a million units and more at small sizes, with the same memory,
and on a laptop with performance and efficiency cores it is the difference between
8 threads being slower than 1 and faster. A value you have set yourself is kept
(`OMP_WAIT_POLICY=active` restores spinning). It applies to the process, so other
OpenMP libraries in it wait the same way. On Windows, numba's OpenMP runtime does not
read the variable, so it changes nothing there.

`refit_bootstrap(n_workers=k)` divides `n_threads` and the memory left among
its k worker processes.

## Profiling: `profile=True`

Every function takes `profile=True` and returns hdfe-stream's `Profile`
instead of fitting: it stages and checks the panel, then estimates the memory
and disk the fit needs, by component.

```python
p = did_callaway_santanna("y ~ x", "panel/*.parquet", panel=("id", "year"),
                          cohort="first_treat", profile=True)
print(p)              # memory and disk, expected and upper bound, by component
p.recommend()         # a job request (e.g. for PBS) that fits it
p.notes               # units, periods, cells, pieces, batch size
```

The memory estimate is the larger of what staging and the fit need. Staging
takes the least of the preprocessing allowance (a quarter of the memory left,
or what `preprocess_mb` says) and about eight times the staged rows' size. The
upper bound always uses the whole allowance. In measurements on 300,000 to
3 million units the expected figure stayed above the measured peak, by about
a half on small panels and by more on panels with covariates. Set
`max_memory_gb` to the job's limit before profiling, so the figures are for the
job you will run.

The estimates are heuristics, not guarantees.
