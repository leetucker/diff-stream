# Benchmarks

The local subprocess benchmark records end-to-end fit time including source
reading, process peak resident memory, baseline resident memory after imports,
and sampled peak allocated scratch disk. Each estimator and implementation runs
in a fresh subprocess with a timeout. Repeats remain in CSVs; figures use medians.
Failures and timeouts are retained. Local results are development measurements,
not publication benchmarks or evidence of universal speed advantages.

See `benchmarks/README.md` in the source checkout for commands, reference-package
setup, interpretation and the later EC2 benchmark matrix. R references read CSV;
diff-stream reads Parquet, so these end-to-end figures include format differences.
The initial BJS timings precede the explicit inference validation; its default
standard errors have since been checked against R didimputation.

## Local development observations

The November 3, 2026 run used simulated panels with 1,000, 10,000, and 100,000
units, eight periods, heterogeneous effects, and serially correlated errors.
At 100,000 units, the medians were:

| Estimator | diff-stream, 1 thread | diff-stream, 4 threads | R reference, 4 threads |
|---|---:|---:|---:|
| CS | 0.75 s | 0.67 s | 2.69 s |
| SA | 0.76 s | 0.63 s | 4.44 s |
| BJS | 0.84 s | 0.71 s | Deferred |

There were two repeats per configuration except BJS at the largest size, which
had one Python repeat. The slow large R BJS fit was interrupted and recorded as
deferred; no performance estimate is inferred for it. These measurements use
Python 3.14, WSL2 Linux, a populated Numba cache, and the R package versions
recorded with the reference fixtures. Import time is excluded; JIT cache loading
is included. No claim is made about general or EC2 performance.

Peak RSS at the largest size illustrates the tradeoff: four-thread diff-stream
used about 771 MB for CS versus 497 MB for R did, and 797 MB for SA versus
2,033 MB for R fixest. Numerical batching alone does not guarantee lower total
memory; validation grouping, Polars pools and the JIT runtime matter. The modest
thread speedups also suggest that these runs are too small to establish strong
multi-core scaling. Scratch disk peaked near 6.7 MB for diff-stream.

```{image} ../figures/local_scale_time.light.svg
:class: only-light
:alt: Local fit wall time by estimator, units and implementation
```

```{image} ../figures/local_scale_time.dark.svg
:class: only-dark
:alt: Local fit wall time by estimator, units and implementation
```

```{image} ../figures/local_scale_memory.light.svg
:class: only-light
:alt: Local process peak RSS by estimator, units and implementation
```

```{image} ../figures/local_scale_memory.dark.svg
:class: only-dark
:alt: Local process peak RSS by estimator, units and implementation
```

```{image} ../figures/local_scale_disk.light.svg
:class: only-light
:alt: Local sampled peak scratch disk by estimator and units
```

```{image} ../figures/local_scale_disk.dark.svg
:class: only-dark
:alt: Local sampled peak scratch disk by estimator and units
```

## Larger CS memory check

A targeted follow-up held periods/cohorts fixed and increased to one million
units (eight million rows), with one repeat and a 60-second fit timeout:

| Units | diff-stream, 1 thread | diff-stream, 4 threads | R did, 4 threads |
|---|---:|---:|---:|
| 10,000 | 0.39 GiB | 0.45 GiB | 0.27 GiB |
| 100,000 | 0.58 GiB | 0.75 GiB | 0.49 GiB |
| 1,000,000 | 1.62 GiB | 1.79 GiB | 2.23 GiB |

Thus the small-sample memory ordering reversed at the larger size. The Python
fits took 5.49 s and 2.48 s versus 28.87 s for R did. These are observations
for one unconditional balanced design, not asymptotic guarantees or broad
package comparisons. Input formats remain Parquet versus CSV.

A separate 5 ms RSS sampler on the four-thread Python fit observed phase peaks
of 1.40 GiB during validation, 1.72 GiB during sorting/staging, and 1.06 GiB
during score accumulation. RSS includes allocator-retained pages, so these
are process phase peaks, not isolated live-object sizes. See
`benchmarks/results/local_cs_phases.json` and `benchmarks/memory_probe.py`.

Import-only baselines missed fixed first-fit LLVM/JIT initialization. That
helps explain the small-sample gap, but the preprocessing buffers also grow
with data size. Reducing validation state and controlling sort/spill memory
are priorities before promising tight cluster-memory bounds. The numerical
score batches remain bounded at fixed periods and effect count. More periods
or cohorts increase covariance storage independently of row count.

```{image} ../figures/local_cs_memory_memory.light.svg
:class: only-light
:alt: CS peak RSS at ten thousand through one million units
```

```{image} ../figures/local_cs_memory_memory.dark.svg
:class: only-dark
:alt: CS peak RSS at ten thousand through one million units
```

## Bounded-preprocessing revision

The curves above describe the initial global-grouping/global-sort prototype.
The revised pipeline uses recursive whole-unit disk partitions and adds
`preprocess_mb` and pre-execution/per-phase allocation guards. A parent watchdog
samples child RSS every 5 ms and terminates the child if its ceiling is crossed;
this is an external sampled check, not an OS cgroup or proof about unsampled peaks.

With eight million rows, four threads, a 16 MB preprocessing allowance and a
1 GB (decimal) process budget, the revised Polars 2.1 path completed with a
sampled peak around 766 MB; Polars 1.37.0 completed around 795 MB. The
800,000-row check peaked around 506 MB. Partition I/O increased: the larger run used 134 leaves,
with the largest decoded leaf about 2.1 MB. An earlier isolated development run took about
8.5 seconds; partition I/O trades speed for memory. Final version probes were
memory smoke checks that overlapped, so their wall times are not controlled
benchmarks.
The revised probe uses Python 3.12, so it is not a controlled timing comparison
with the earlier Python 3.14 curves.

The version-specific JSON records in `benchmarks/results/bounded_cs_*.json`
preserve the current probes and guard outcomes. Polars 1.37 initially stopped
at the inference-allocation guard below the RSS ceiling because unused Arrow
pool pages remained resident after preparation. The pipeline now releases
unused Arrow pool pages once closed preparation readers/writers are no longer
needed. This releases cached free pages, not live model data. Functional tests
run on both Polars 1.37.0 and 2.1.0, including recursive partitioning.

See `benchmarks/limited_memory.py` for the parent watchdog and
`benchmarks/memory_probe.py` for phase instrumentation. Local measurements
remain development checks and should be replaced before publishing performance
claims.

```{image} ../figures/bounded_cs_memory_memory.light.svg
:class: only-light
:alt: Bounded CS preprocessing under a monitored 1 GB process budget
```

```{image} ../figures/bounded_cs_memory_memory.dark.svg
:class: only-dark
:alt: Bounded CS preprocessing under a monitored 1 GB process budget
```

## Conditional CS development check

A separate serial subprocess run compares numeric covariate adjustment on
1,000 and 10,000 units, five periods, with time-varying x and unit-level z.
Two repeats per configuration are retained in `conditional.csv`; the earlier
exploratory run that overlapped tests is saved separately. At 10,000 units,
median fit times were:

| Method | diff-stream, 1 thread | diff-stream, 4 threads | R did |
|---|---:|---:|---:|
| Regression | 1.10 s | 0.90 s | 0.66 s |
| IPW | 0.89 s | 0.80 s | 0.60 s |
| DR | 1.04 s | 0.93 s | 1.20 s |

At this size, R is faster for regression/IPW, and thread gains are modest.
Peak Python RSS was about 317–321 MiB with one thread and 375–377 MiB with four,
versus 277–279 MiB for R. These small fits do not establish large-scale memory
or speed advantages. The current conditional implementation prioritizes bounded
scores, nuisance-fit reuse and shared source passes; larger conditional designs
need further profiling and eventual EC2 measurement.

For IPW/DR, nine distinct time-varying-covariate nuisance systems were fitted in
seven shared numerical source passes, including moment and score passes. CSV
versus Parquet reading and first-use formula initialization remain part of the
end-to-end fit differences; package imports are excluded. Numeric designs avoid
full pyfixest model-matrix imports. This is a local development check only.

```{image} ../figures/conditional_time.light.svg
:class: only-light
:alt: Local conditional CS wall times for regression, IPW and DR
```

```{image} ../figures/conditional_time.dark.svg
:class: only-dark
:alt: Local conditional CS wall times for regression, IPW and DR
```

```{image} ../figures/conditional_memory.light.svg
:class: only-light
:alt: Local conditional CS process peak memory
```

```{image} ../figures/conditional_memory.dark.svg
:class: only-dark
:alt: Local conditional CS process peak memory
```

## Unbalanced-panel development check

`benchmarks/unbalanced_benchmark.py` measures all three native observed-row
paths, including one simple aggregation and result cleanup, in fresh subprocesses.
A local two-thread run with five periods and about 18% of rows removed retained
40,771 observations at 10,000 units and 407,693 at 100,000 units. No R timing
comparison is claimed by this check.

| Estimator | 10,000 units, seconds | 100,000 units, seconds | Peak RSS at 100,000 units, MiB |
|---|---:|---:|---:|
| CS, intercept-only DR | 0.50 | 4.91 | 405 |
| Sun–Abraham | 0.20 | 1.12 | 344 |
| BJS | 0.32 | 2.35 | 424 |

These are single development runs, including first-use initialization during
fitting. Process import baseline was about 195 MiB. CSV measurements are retained
in `benchmarks/results/unbalanced.csv` in the source checkout. Peak scratch disk
is sampled; BJS aggregation files are removed before the child exits.

The initial SA adapter used the general HDFE backend. A separate fresh-process
probe separated package imports, a fixed 500-row warm-up, and the subsequent fit:

| Units | General backend peak RSS, MiB | Specialized SA peak RSS, MiB |
|---|---:|---:|
| 10,000 | 595 | 306 |
| 100,000 | 733 | 364 |
| 1,000,000 | 1,434 | 502 |

The warmed fixed baseline was about 490 MiB for the general backend and 270 MiB
for specialized SA. Thus the original cost included both fixed imports/JIT and
data-dependent growth; it was not just Numba/Polars imports. The general backend
staged wide interaction rows and mapped observation-by-variable cell arrays.
The replacement is an exact reparameterization into small within-cohort time
systems, avoiding those representations and extra formula initialization.
At one million units, its numerical SA pass peaked at 407 MiB; the overall
502 MiB peak occurred during preparation.

The million-unit input has 4,076,930 rows and tiles the 100,000-unit file ten
times with distinct IDs. It tests resource scaling, not another independent
statistical replication. Sampling includes resident mapped pages and allocator
retention; 5 ms sampling can miss brief peaks. Measurements remain development
checks, not hard-memory guarantees. RSS still changes with input size and
runtime buffering, and numerical memory grows with periods, cohorts and joint
covariance dimensions.

`benchmarks/unbalanced_memory_probe.py input.parquet` reproduces the specialized
probe; add `--hdfe-reference` to temporarily use the former adapter without
editing either library. Set thread environment variables before running.
Raw phase JSON files are in `benchmarks/results/unbalanced_*_memory_*.json`.
The earlier general-adapter timing run is retained as `unbalanced_hdfe.csv`.
No OS memory cap was enforced for these runs.

## Larger conditional CS and unbalanced-panel resource matrix

The [expanded resource checks](../resource_checks.md) include narrow and expanded
conditional designs, all three CS methods, one-million-unit panels, 12 periods
and six treated cohorts, varied input layouts and constrained buffers. The
report provides light/dark figures, observed RSS and disk use, source-pass counts,
R covariance comparisons and reproducible offline commands. It also documents
the logit objective-roundoff issue found at large N and the shared-solver fix.
Raw per-case results, progress, earlier measurements and versions are retained
under `benchmarks/results/resource_checks` in the source checkout.
