# Conditional CS and unbalanced-panel resource checks

Retained results: **39 of 39 planned Python cases**, 39 with their expected outcome.
Actual local R comparisons: **9**, all matched estimates and full covariance.

These are sequential local development checks, not public speed claims. Each fit runs in a fresh process,
including first-use initialization and Parquet reads. R reads CSV. Input generation is excluded.
Peak RSS includes imported libraries, allocator retention and resident mapped pages.

## Larger fits

| Case | Units | Observed rows | Periods / cohorts | Columns | Fit / aggregation seconds | Peak RSS MiB | Peak scratch MB |
|---|---:|---:|---|---:|---:|---:|---:|
| cs_n100000_balanced_k3 | 100,000 | 500,000 | 5 / 3 | 3 | 7.39 / 0.00 | 477 | 15.8 |
| cs_n1000000_balanced_k3 | 1,000,000 | 5,000,000 | 5 / 3 | 3 | 77.44 / 0.00 | 625 | 164.6 |
| cs_n1000000_entry_exit_k3 | 1,000,000 | 2,842,639 | 5 / 3 | 3 | 98.50 / 0.01 | 560 | 99.6 |
| cs_million_1gb | 1,000,000 | 5,000,000 | 5 / 3 | 3 | 23.99 / 0.00 | 628 | 168.4 |
| cs_n100000_balanced_k17_dr | 100,000 | 500,000 | 5 / 3 | 17 | 18.16 / 0.00 | 656 | 74.4 |
| cs_n100000_entry_exit_k17_dr | 100,000 | 284,063 | 5 / 3 | 17 | 20.14 / 0.02 | 613 | 45.1 |
| cs_k33 | 100,000 | 284,036 | 5 / 3 | 33 | 30.99 / 0.02 | 695 | 115.1 |
| cs_more_cohorts_k17 | 50,000 | 284,972 | 12 / 6 | 17 | 9.43 / 0.00 | 641 | 32.8 |
| sunab_n1000000_t5 | 1,000,000 | 2,842,639 | 5 / 3 | 1 | 12.54 / 0.00 | 486 | 54.6 |
| bjs_n1000000_t5 | 1,000,000 | 2,842,639 | 5 / 3 | 1 | 13.94 / 3.04 | 487 | 52.7 |
| sunab_n250000_t12 | 250,000 | 1,429,347 | 12 / 6 | 1 | 5.80 / 0.00 | 430 | 26.4 |
| bjs_n250000_t12 | 250,000 | 1,429,347 | 12 / 6 | 1 | 3.36 / 1.16 | 480 | 22.2 |
| bjs_coarse_leaveout_aggregations | 100,000 | 924,670 | 12 / 6 | 1 | 1.20 / 0.93 | 460 | 14.9 |

Most cases use a **1.5 GB decimal** process budget/watchdog, two threads, `preprocess_mb=16`,
and `batch_units=512`. The tighter-buffer and million-unit 1 GB cases use 1.0 GB;
the deliberate refusal uses 0.35 GB. The largest R comparison is retried with a 3 GB watchdog.
The watchdog samples the process tree every 5 ms and checks reported process high-water RSS too.
It is not an OS/cgroup memory guarantee. Disk is sampled every 200 ms and can miss short-lived files.
Process-tree RSS can count shared pages more than once; it is not unique/cgroup-charged memory.

```{image} figures/conditional_resources.light.svg
:class: only-light
:alt: Conditional CS process RSS and fit time across unit counts
```

```{image} figures/conditional_resources.dark.svg
:class: only-dark
:alt: Conditional CS process RSS and fit time across unit counts
```

## Wider designs, support and nuisance reuse

The generated data include time-varying covariates, four categorical levels and numeric/category
interactions (17 or 33 compiled columns), serial dependence and heterogeneous effects.
Entry/exit panels combine late entry, early exit and interior gaps, retaining an untreated observation
for every unit. Each input is generated in blocks of at most 4,096 units; its manifest records row
count, schema configuration, Parquet layout, seed and checksum. These are resource/reference checks,
not a claim that observed-panel selection always identifies the simulated causal effect.

| Case | Cells | Propensity models | Numerical source passes | Maximum logit iterations |
|---|---:|---:|---:|---:|
| cs_n100000_balanced_k3 | 12 | 7 | 8 | 4 |
| cs_n100000_entry_exit_k3 | 12 | 12 | 9 | 4 |
| cs_invariant | 12 | 3 | 8 | 4 |
| cs_invariant_entry_exit | 12 | 12 | 9 | 4 |
| cs_more_periods | 28 | 28 | 9 | 4 |
| cs_more_cohorts_k17 | 66 | 66 | 10 | 5 |

Raw diagnostics retain overlap ranges, trimmed/capped contributions, effective controls,
partition counts and largest decoded leaf. Trimming counts are comparison contributions,
not unique units. Time-invariant covariates permit balanced-panel nuisance reuse; observed-row
CS still needs date-pair populations because observation frequencies vary.

## Buffer, layout and initialization checks

| Same-sample variant | Peak RSS MiB | Fit seconds | Decoded numerical batches |
|---|---:|---:|---:|
| cs_n100000_entry_exit_k17_dr | 613 | 20.14 | 2475 |
| cs_tighter_buffers | 585 | 37.83 | 9009 |
| cs_four_threads | 637 | 24.60 | 2475 |
| cs_strings_large_groups | 565 | 19.98 | 2511 |
| cs_preloaded_formula | 634 | 15.75 | 2475 |
| compat137_entry_exit | 532 | 6.61 | 2484 |

The tighter case changes batch units from 512 to 128 and preprocessing from 16 to 4 MB.
Its extra batch/I/O work buys a modest memory reduction in this sample. Four threads did not
improve this particular run. The combined string-ID/large-row-group case has actual 131,072-row
input groups; it preserves the underlying sample and compiled design.

The formula-preload variant moves fixed import work outside the fit timer, not outside total
process RSS or process time. Its measured baseline rose from about 196 to 336 MiB after loading
formula machinery. This distinguishes fixed initialization from data-dependent working memory.

## Numerical stability finding

The original million-unit balanced run stayed below about 604 MiB but timed out after 240 seconds.
A nuisance-only probe then took 55 source passes without convergence. An absolute-only likelihood
line-search allowance rejected tiny Newton improvements that were below roundoff in the summed
objective. The shared temporary nuisance solver now uses compensated objective accumulation and
a roundoff allowance scaled to the objective. Its coefficient-step convergence tolerance,
unpenalized MLE, separation checks and singular-design checks are unchanged.

The first corrected million-unit nuisance probe finished in six passes. A 600,000-row known-solution
test checks both small and large batches. All measured fits below use the corrected solver;
earlier results and the failing probe are retained in `history/` and `initial_logit_failure.txt`.

## Equivalence and limits

| Check | Status | Max cell estimate difference | Max covariance difference |
|---|---|---:|---:|
| compat137_balanced (invariance) | True | 4.4e-16 | 1.8e-19 |
| compat137_entry_exit (invariance) | True | 2e-15 | 7.3e-19 |
| cs_four_threads (invariance) | True | 0 | 0 |
| cs_million_1gb (invariance) | True | 0 | 0 |
| cs_more_cohorts_k17 (r_comparison) | True | 3.2e-10 | 1.4e-11 |
| cs_n100000_balanced_k3 (r_comparison) | True | 6e-12 | 9.6e-15 |
| cs_n100000_entry_exit_k3 (r_comparison) | True | 1.5e-14 | 1.1e-16 |
| cs_preloaded_formula (invariance) | True | 0 | 0 |
| cs_strings_large_groups (invariance) | True | 2.9e-15 | 9.8e-19 |
| cs_tighter_buffers (invariance) | True | 4.2e-15 | 1.5e-18 |
| smoke_balanced_dr (r_comparison) | True | 3.2e-14 | 1.5e-15 |
| smoke_balanced_ipw (r_comparison) | True | 1.7e-13 | 3.7e-15 |
| smoke_balanced_reg (r_comparison) | True | 4.4e-16 | 1.7e-17 |
| smoke_entry_wide_dr (r_comparison) | True | 2.2e-12 | 1.3e-14 |
| smoke_entry_wide_ipw (r_comparison) | True | 1.6e-12 | 7.3e-13 |
| smoke_entry_wide_reg (r_comparison) | True | 1.2e-15 | 3.5e-17 |

The high-dimensional R reference initially exceeded the 1.5 GB process-tree watchdog.
A separate retry with a 3 GB ceiling completed at 855 MiB peak RSS
and passed the full numerical comparison. Both outcomes are retained. This variation is not
evidence of a minimum required R memory budget or a stable R/Python memory ratio.

Timing also varied materially across fresh-process reruns of identical data/algorithms:
the million-unit balanced cases have identical estimates, covariance and eight source passes,
but different observed wall times. File-cache state, initialization, allocators and competing
machine load are not isolated here. Use these as resource-validation observations, not stable speed estimates.

The 1.37 compatibility cases use the locally cached Polars 1.37.0/runtime overlay; no download is needed.
String labels, large row groups, tighter batches/partitions, four threads and preloaded formula imports
are compared on the same underlying sample. Fits and aggregation values/covariance must agree.
The intentional budget refusal is an expected outcome, not an out-of-memory success.
The watchdog self-check separately verifies memory and timeout termination.

These finite checks do not establish flat RSS or a universal cluster-scale guarantee. Numerical state
still grows with design columns, cohorts, periods and joint covariance size. Source scans, query
operators, runtime prefetch and allocator retention are outside the decoded-partition allowance.
An external kill can leave scratch files; the resumable runner removes only its own abandoned case
directory. Scheduler-enforced limits and larger/richer designs remain separate release checks.

## Reproduce offline

```bash
python benchmarks/resource_checks.py --suite smoke
python benchmarks/resource_checks.py --suite scale
# Resume and retry failures after an interruption or fix:
python benchmarks/resource_checks.py --suite scale --retry-failed
python benchmarks/resource_report.py
```

Use `--only CASE...` for selected cases and `--rscript PATH` for another local R installation.
`--no-r` explicitly skips R comparisons. The local cached R library and Python environment must
already be installed. The scripts never install packages or access a network.
One driver holds a filesystem lock to prevent concurrent timing runs in the same result directory.
Each case writes an atomic JSON result and stdout/stderr logs. Long fits write phase/pass progress.
Library source hashes and package versions are recorded, and reruns archive previous artifacts.
Input data live under ignored `benchmarks/data/resource_checks`; retained measurements, case
configurations, history and `summary.csv` live under `benchmarks/results/resource_checks`.

The later [numerical stress and complexity review](numerical_stress.md) adds
period/cohort/column sweeps through 256 columns and 1,953 effects, a corrected
memory-admission estimate, observed-stratum support diagnostics and Linux
address-space probes. It preserves the limitations of sampled RSS ceilings.

The [stronger memory review](memory_checks.md) extends source-layout and unit-count
checks through 10 million rows, retains clean boundary refusals, and provides an
optional delegated-cgroup runner for a host with a writable memory controller.
