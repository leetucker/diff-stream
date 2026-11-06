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

Before each large allocation the fit checks the memory left and stops with
`MemoryError` if the allocation would not fit, before making it. These are
checks, not a hard limit: the source query, Polars and Arrow's pools and the
allocator can hold more than the estimates. Enforce a hard limit with the
job's own limit or stream-limits' launcher.

What does not shrink with the batch: the joint covariance of the effects
(cohort-period cells squared), and the design of a model with many covariates
or fixed effects (its columns squared). A panel with 50 periods and 50 cohorts
has about 2,500 Callaway-Sant'Anna effects, a 50 MB covariance.

On the 8-million-row simulated panel of the [benchmarks](benchmarks/index.md),
every estimator fits within a 1 GB limit, and peaks at about 1.6 GB without
one.

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

The estimates are heuristics, not guarantees.
