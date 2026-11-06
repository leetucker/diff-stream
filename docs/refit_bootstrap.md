# Unit-cluster refit bootstrap

`refit_bootstrap` supplies ordinary unit-cluster resampling with full model
refits for CS, Sun–Abraham, BJS, Gardner and linear ETWFE. It is a separate
wrapper: ordinary fits do not retain source data or change their defaults.

**Release-scope decision:** simultaneous refit-bootstrap bands are deferred by
the user pending an agreed, fully specified end-to-end reference. They are not
part of initial-release cleanup. See the [release handoff](release_handoff.md)
for the rationale, distinctions from existing joint/multiplier inference and
the requirements for revisiting that decision.

```python
from diff_stream import refit_bootstrap

boot = refit_bootstrap(
    "y ~ x + z",
    "panel.parquet",
    panel=("id", "time"),
    cohort="first_treat",
    estimator="cs",
    est_method="dr",
    aggregate="dynamic",
    reps=999,
    seed=42,
    threads=2,
    preprocess_mb=64,
)
boot.tidy()  # percentile pointwise intervals
boot.se  # bootstrap standard errors
boot.confint(method="basic")
boot.confint(method="normal")
boot.draws()  # every attempted vector
boot.failures()  # failed attempts and reasons
boot.refit_warnings()  # warnings from individual refits
boot.summary_json("bootstrap.json")  # sample, RNG and failure metadata
```

The wrapper fits the original statistic once, then samples exactly N units
uniformly with replacement for each attempt. Each selected copy contains that
unit's entire observed history and receives a distinct panel ID. Original
columns, covariates and sampling weights travel with the rows. First stages,
propensity scores, trimming and aggregation shares are recomputed in each fit.
This is ordinary pairs resampling, with sampling weights carried as estimator
weights; it is not a complex-survey replicate-weight procedure or PPS sampling.

The fitted coefficients are the default target (`aggregate=None`). Set
`aggregate="dynamic"`, `"simple"`, `"cohort"` or `"calendar"` to refit a
group-time aggregation, with `aggregate_options` for its existing options.
Gardner returns direct coefficients: use `aggregate=None` with
`specification="static"` or `"dynamic"`. An optional `coefficients` sequence
selects exact names in the chosen target. BJS clean-placebo coefficients and
Goodman–Bacon/TWFE diagnostics are outside this wrapper's scope.

## Fixed sample and unsuccessful draws

The original fit determines the analysis sample, complete-panel deletion and
reference window before sampling. Last-treated references use the original
cutoff throughout. A draw that loses the required reference, a requested
coefficient, an analysis date or identification fails; the wrapper does not
choose an earlier reference or replace that draw. All coefficients in a failed
vector are missing together. A changing analysis population is also rejected.
Unbalanced draws preserve each selected unit's missing observations.

Failed statistical refits generate `BootstrapFailureWarning` and remain in
`failures()`. SEs and intervals then raise by default. After investigating,
`boot.tidy(allow_incomplete=True)` or
`boot.standard_errors(allow_incomplete=True)` explicitly requests inference
conditional on successful fits. This selection can affect validity; neither
silently discarding failures nor repeating draws repairs identification.
Every inference method requires at least two successful vectors. Memory, I/O
and unexpected programming errors stop the run and clean its scratch files.

## Pointwise inference

SEs are the sample standard deviation of successful estimates, with denominator
B−1. Percentile intervals use empirical quantiles with linear interpolation
(R `quantile(type=7)`). Basic intervals subtract the opposite percentile
endpoint from twice the original estimate. Normal intervals use the original
estimate ± the normal critical value times bootstrap SE; they have no bootstrap
bias correction. `tidy()` reports the bootstrap mean and estimated bias for
inspection, without adjusting the point estimate or producing p-values.

These definitions do not promise identical endpoints to R `boot::boot.ci`,
whose interpolation and normal bias correction differ. This milestone does
not supply joint bands, bootstrap-t, BCa or stratified resampling
or a coverage guarantee. Choose a sufficiently large number of draws for the
desired tail precision; the small validation fixtures test numerical agreement
and cannot assess coverage. Existing CS [multiplier intervals](inference.md)
remain available separately.

## Resources and reproducibility

Refits default to serial execution and can cost roughly one original fit plus `reps` refits,
including nuisance estimation and staging. Required input columns are
snapshotted once. Whole-unit source partitions and draw plans live on disk;
the resident draw matrix scales with `reps × selected coefficients`, with
bounded staging buffers and memory guards. The caller must budget scratch
space for the snapshot, partitions and each active resample/refit. Scratch is
removed before returning; results contain no live source handle.

`seed` reproduces a fixed input/partition layout and NumPy RNG version.
Changing preprocessing layout can change the mapping of random indices to
units. Metadata records the actual seed entropy (also when `seed=None`), RNG,
unit-order hash and per-attempt draw-plan hashes. Original ID-valued covariates
retain their original values; fresh IDs are used only as panel identifiers.
An explicit standalone original unit FE in the formula is replaced by the
automatic synthetic unit FE in each resample.

## Parallel refits

```python
boot = refit_bootstrap(
    "y ~ x + z",
    "panel.parquet",
    panel=("id", "time"),
    cohort="first_treat",
    estimator="cs",
    aggregate="dynamic",
    reps=999,
    seed=42,
    workers=2,
    threads=4,
    memory_gb=4,
)
```

`workers=1` retains serial execution in the current process. Larger values run
independent refits in fresh Python processes, in batches of at most `workers`
draws. This avoids reusing fork-inherited numerical state and works from scripts
or interactive sessions without a multiprocessing `__main__` guard. The
process mechanism supports Linux, macOS and Windows; local execution is tested
on Linux, with the portable resource fixture included in all three OS CI jobs.
No parallel framework dependency is added.

`threads` is the total fit-thread budget, not the threads per worker. When it
is omitted, stream-limits detects the job's CPU allocation. The effective
worker count is `min(workers, reps)` and cannot exceed that budget. Each worker
receives `floor(threads / effective_workers)` threads, applied before numerical
imports, including to Polars. The original fit and resample staging run in the
parent; staging pauses while a batch fits. This avoids overlapping parent
preprocessing with the full worker allocation. Completed batches are removed
before staging the next batch, so queued resamples do not grow with `reps`.

`memory_gb` continues to mean the whole-job budget. Before each parallel batch,
the coordinator subtracts its current RSS and the configured job reserve,
then divides remaining memory among the effective workers. Each child uses
that share for its estimator's cooperative memory guards and Polars spill
budget, retaining its own reserve. An initial guard allows 350 MB per worker
for its runtime, plus staging buffers; this is a conservative screening rule,
not an exact prediction of import or fit memory. Insufficient headroom raises
`MemoryError` rather than silently using fewer workers. Reduce `workers` or
`preprocess_mb`, or increase the job budget, to retry.

These guards are not OS enforcement. To enforce a process-tree memory limit,
launch the entire bootstrap call through stream-limits' launcher, using a
delegated kernel controller where available or its portable watchdog. Parallel
fits need extra memory for each numerical runtime and additional scratch for
each active draw. Fresh-process startup can make small fits slower; increasing
workers is useful only when enough refit work and memory are available.

All draw plans are generated in the parent in the same order as serial
execution. Changing `workers` preserves the unit-order hash, per-draw plan
hashes and selected draws for the same input/layout/seed. Results and ledgers
are indexed by draw, regardless of completion order. Changing numerical thread
counts can introduce small floating-point differences. Metadata records the
actual worker count, per-worker threads, per-batch memory shares and worker
PIDs/Polars thread counts. Statistical failures keep the serial failure policy;
fatal errors or interruption stop and reap workers before scratch cleanup.
No simultaneous bands are introduced by parallel execution.

## Validation

Tests compare every draw with independent explicit whole-unit duplication for
all five estimators, including unbalanced, date-weighted and last-treated
designs. Frozen native R `did`/`DRDID` references compare nine identical draws
plus the original fit for regression, IPW and DR CS, using both balanced and
unbalanced last-reference panels. Base R `sd`, `quantile(type=7)` and `qnorm`
independently validate all three interval definitions. Maximum native point
difference is below 1.8e-8. Reproduce with:

```bash
python benchmarks/refit_bootstrap_validation.py
# Regenerate using a local R installation with did and DRDID:
python benchmarks/refit_bootstrap_validation.py --generate --rscript /path/to/Rscript
```

The native comparisons validate CS full refits; other estimators combine
explicit-draw tests here with their existing native estimator references.
They are not additional authors' Stata bootstrap comparisons.

Six local fresh-process scaling probes, with three refits each, completed under
a verified 1.5 GB kernel cap. For CS/BJS/Gardner on five-date panels, peak RSS
was 402–404 MiB at 10,000 units and 531–537 MiB at 100,000 units. All scratch
was removed and no OOM occurred. These probes use small `reps` to exercise
resource growth, not to estimate inferential precision. Raw reports are in
`benchmarks/results/refit_bootstrap_resources`; reproduce with
`python benchmarks/refit_bootstrap_resources.py` (portable RSS watchdog) or
add `--cgroup-parent /path/to/delegated/parent` for a Linux kernel cap.

Matched serial/parallel probes use four refits, two total fit threads and a
2.5 GB whole-job kernel cap. Draw-plan hashes and bootstrap SEs match exactly
for CS, BJS and Gardner at both scales. Peak process-tree RSS was:

| Units (five dates) | Serial, MiB | Two workers, MiB |
|---|---:|---:|
| 10,000 | 403–417 | 1,071–1,088 |
| 100,000 | 515–528 | 1,266–1,349 |

All runs removed scratch and completed without OOMs. Timings were collected
alongside other tests and do not establish an isolated speedup. Raw runs and
the numerical/resource comparison live in
`benchmarks/results/parallel_bootstrap_resources`,
`benchmarks/results/parallel_bootstrap_serial_resources` and
`benchmarks/results/parallel_bootstrap_comparison.json`. Run the parallel probe
with:

```bash
python benchmarks/refit_bootstrap_resources.py --workers 2 --reps 4 \
  --memory-gb 2.5 --results benchmarks/results/parallel_bootstrap_resources
```
