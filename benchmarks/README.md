# Development validation and benchmarks

Install `.[test,benchmark]`. All local measurements are development results;
do not include them in public benchmark claims. No EC2 runs are started here.

## Numerical references

```bash
python benchmarks/make_reference_data.py
# In R, in a separate library: install.packages(c('did','fixest','didimputation'))
Rscript benchmarks/external/reference.R tests/data/reference_panel.csv tests/data/r_reference
pytest -q
```

Use `R_LIBS_USER` for an isolated reference library. An existing R executable
can be selected with `DIFF_RSCRIPT` for benchmarks. No test requires writing
to another estimation package's environment. On a conda R installation, activate
its compiler toolchain or supply `R_MAKEVARS_USER`; missing conda compiler
executables are an environment problem, not an estimator failure.

The deterministic reference panel has heterogeneous cohort/duration effects and
serially correlated errors. R exports record versions. Tests compare CS cells,
the full analytic covariance, dynamic and simple aggregation (both bases and
both controls), SA dynamic point estimates/SE with explicit SSC conventions,
and BJS dynamic/simple point estimates and default standard errors. The
original authors' Stata baseline has been returned and all 16 fits pass. Normal CIs are distinct from R did's default bootstrap bands.

If checked-in outputs are absent, external tests skip with a reason; the dense
independent regression and score oracles still run. To refresh a fixture,
rerun the exporter and review numerical/metadata changes together.

Reference outputs are committed gzipped (`name.csv.gz`); input panels in
`tests/data/` stay plain CSV because the exporters read them by path, and
`tests/data/refit_bootstrap_reference/` stays plain because its provenance pins
the script that reads it. Code names references by their `.csv` path and
`benchmarks/_files.py` falls back to the compressed file; recorded hashes are
of the CSV content, so they hold either way. A regenerated plain `.csv` takes
precedence, so compress after reviewing a refresh:

```bash
python benchmarks/_files.py gzip tests/data/<reference folder>
```

On a machine with Stata, run `do benchmarks/external/validate.do` from the repo
root, after installing its listed commands. The bundle produces a text log and
saved estimates; no local Stata run is claimed. The commands use their own
variance corrections, so inspect those conventions before comparing SEs.

## Local small/medium runs

```bash
python benchmarks/scale_benchmark.py --sizes 1000 10000 --threads 1 4 --repeats 3
python benchmarks/plot_benchmarks.py
```

Each fit runs in a fresh subprocess. CSVs retain all repeats, timeout/failure
statuses, total wall time (including source reads), baseline RSS, total peak
RSS and sampled peak allocated scratch disk. Figure scripts adapted from
dpd-stream use its accessible palette, distinct markers, and light/dark SVGs;
repeated successful measurements are summarized by their median. Python uses
Parquet; R uses CSV. Compare these as end-to-end pipelines and record that input
format difference. R wall timing excludes package loading, as Python timing
excludes imports. Python's first fit includes JIT cache load/compilation; warm
the cache first if measuring repeated-use performance and record that choice.

The standard configuration labels support 1 and 4 thread curves. The CLI accepts
other thread counts, retained in CSVs; add them to plot SERIES to display them.
Disk excludes source files, Python/R installations and pre-existing JIT caches.
R reports zero scratch disk for these in-memory reference fits. Sampled disk can
miss very short-lived files; process peak memory includes import/runtime memory.
The harness is adapted from dpd-stream, preserving resource measurement methods.

## Later EC2 matrix

Use CPU-optimized instances and identical data and resource limits per library.
Vary units (10k–10m), periods (8, 20, 50), cohort count (sparse/dense adoption),
thread count (1 through allocated cores), and treatment-effect heterogeneity.
Include R did/fixest/didimputation and investigate csa-py/diff-diff as additional
Python references. Separate cold/warm runs, retain all failures, publish machine
and package versions, and include a real example such as mpdta after irregular
time grids are supported. Add scheduler/EC2 orchestration before publishing;
this milestone provides the local reproducible subprocess harness only.

## Targeted memory follow-up

`results/local_cs_memory.csv` extends the CS curves to one million units/eight
million rows, one repeat at the largest size. `results/local_cs_phases.json`
records a separate sampled phase profile. Reproduce with:

```bash
python benchmarks/scale_benchmark.py --models cs --sizes 1000000 --threads 1 4 --repeats 1 --timeout 60 --output benchmarks/results/local_cs_memory.csv
STREAM_LIMITS_MAX_THREADS=4 python benchmarks/memory_probe.py benchmarks/data/sim_1000000.parquet --threads 4
```

The committed memory curve also includes the CS rows from `local_scale.csv`.
The larger check reverses the small-sample memory ordering; preprocessing is
the largest Python phase. The Sphinx benchmark page explains the measured
limits. Local result sections/figures should be replaced by EC2 measurements
before publishing performance claims.

## BJS inference hardening

The standard R package fixtures now check BJS dynamic and simple standard errors.
For joint covariance and averaging/leave-out variants, regenerate the independent
full-unit-dummy R oracle:

```bash
Rscript benchmarks/external/bjs_reference.R tests/data/reference_panel.csv tests/data/r_reference
```

This independently checks the published score equations; it does not execute
Stata. With Stata available, run `do benchmarks/external/validate_bjs.do` to
export the original authors' dynamic/simple covariance matrices for all four
averaging schemes and both leave-out settings. The optional pytest check reads
`benchmarks/results/stata_bjs`, or the `DIFF_STATA_BJS` directory. Comparisons
use `minn(0)` to avoid automatic small-effective-sample suppression.

The returned baseline passes estimates, standard errors and full covariance.
Run `python benchmarks/stata_bjs_review.py` for the hashed audit report.
`python benchmarks/build_stata_bjs_bundle.py` preflights and builds the extended
transfer ZIP for weights, unbalanced panels, anticipation and clean placebos.
The returned extended results complete all 176 fits and pass inference checks.
The original run retains 18 strict point-tolerance findings; all are resolved
by the returned 48-fit double-storage confirmation without relaxing tolerance.
The reviewer can combine both runs into a passing validation summary, verifying
identical data, specifications and installed command source hashes.
`python benchmarks/build_stata_bjs_precision_bundle.py` builds a focused
48-fit double-storage confirmation ZIP from the preserved original bundle.
See [Stata validation](../docs/stata_validation.md) for transfer and review steps.

## Bounded preprocessing and a monitored memory ceiling

The earlier scale and phase CSVs describe the first prototype. The revised
preprocessing path snapshots rows and recursively hash-partitions whole units
before sorting/validation. It accepts `preprocess_mb` and checks the process
budget before row execution, partition allocations and numerical work.
`profile=True` now uses temporary disk for validation.

```bash
python benchmarks/limited_memory.py benchmarks/data/sim_1000000.parquet --memory-gb 1 --preprocess-mb 16 --threads 4
```

The parent samples its child's RSS every 5 ms and terminates it if it exceeds
the chosen decimal-GB ceiling or timeout. This is a watchdog, not an OS cgroup;
it can miss shorter peaks. The child also receives stream-limits memory/thread
environment settings and the explicit fit budget. This gives both an allocation-
guard check and an external sampled-ceiling check without changing system limits.
The saved `bounded_cs_*.json` files record outcomes. Polars/version differences,
reader metadata and retained allocator pages still affect peak memory.

## Conditional CS

`external/cs_covariates.R` exports actual R did group-time estimates, full
analytic covariance and aggregates for the three methods, both control/base
choices, and numeric versus expanded categorical/interaction designs.
`external/cs_trim.R` adds actual DRDID 2×2 checks for control-only trimming.
Fixtures live in `tests/data/cs_reference`; source data include time-varying
covariates to distinguish earlier-date selection from a global fixed baseline.

```bash
Rscript benchmarks/external/cs_covariates.R tests/data/conditional_panel.csv tests/data/cs_reference
Rscript benchmarks/external/cs_trim.R tests/data/trim_panel.csv tests/data/cs_reference
python benchmarks/conditional_benchmark.py --sizes 1000 10000 --threads 1 4 --repeats 2
python benchmarks/plot_benchmarks.py
```

The conditional benchmark follows the same subprocess RSS/disk measurements and
light/dark figure conventions. It uses numeric x/z, five periods, time-varying x,
never-treated controls, varying pre-treatment bases, and reg/ipw/dr. Shared-pass
counts are recorded for diff-stream. R reads CSV and Python Parquet. Imports are
excluded from fit time, while first-use formula initialization remains included.
`conditional_initial.csv` retains the exploratory run that overlapped tests;
`conditional.csv` is the subsequent serial development run. Local measurements
remain unsuitable for public performance claims.

Shared-core extraction and limitations are documented in
`../SHARED_INFRASTRUCTURE.md`. No external package source is modified by these
scripts or tests.

## Resumable offline resource matrix

Run `python benchmarks/resource_checks.py --suite smoke`, followed by
`--suite scale`. Existing local Python and R packages are used; no network or
install step occurs. `--retry-failed` resumes failures, `--only CASE...` selects
cases, `--no-r` explicitly skips R, and `--rscript PATH` selects an installed R.
Inputs are generated in bounded unit blocks (large requested Parquet row groups
buffer one row group plus working copies). Source data/checksums are cached in
`data/resource_checks`; per-case JSON/log/progress artifacts and versions live in
`results/resource_checks`. Reruns archive prior measurements. A filesystem lock
prevents overlapping drivers in the same result directory.

Fits have a process-tree RSS watchdog, high-water RSS check, timeout and sampled
scratch-disk monitor. The watchdog self-test verifies memory/timeout kills.
These are not cgroup guarantees: sampling can miss short spikes and an external
kill can leave scratch; resuming removes only the runner's own old directory.
Only package-owned files are cleaned. R reads CSV; Python reads Parquet. Imports
and first-use initialization belong to measured process RSS, while generation
occurs in a separate process and is excluded from fit timing.

`python benchmarks/resource_report.py` renders the retained measurements to
`docs/resource_checks.md` and light/dark figures. The report records the numerical
logit-stability fix found by the matrix. Larger designs, scheduler-enforced
limits and EC2 remain separate release checks; no paid resources are launched.

## Numerical stress and model complexity

`numerical_stress.py` runs balanced/native unbalanced regression/IPW/DR probes
and actual R did/DRDID comparisons, retaining expected refusals and warnings.
`complexity_checks.py` measures fresh-process period/cohort/column sweeps with
source/configuration hashes, timeouts, an RSS watchdog and selected Linux
address-space limits. The resource sweep retains two explained boundary failures
and exits nonzero; see the report before interpreting this as a regression.
`stress_report.py` rebuilds the compact JSON summary and complete resource CSV.
See [the numerical stress review](../docs/numerical_stress.md) for findings,
reproduction commands, fixes, limits and the source of all 47 added tests.

## Castle Doctrine Sun–Abraham reanalysis

`castle_replication.py` reruns the frozen public-data checks offline, including
full group-time/aggregate covariance, independent R/Python dummy regressions,
timing and sampling-weight sensitivities, and pre-trend tests. Add `--plot` for
standalone light/dark SVGs and PDF/PNG exports; `--regenerate --rscript ...`
refreshes references using installed fixest and the pinned source R data object.
No network or package installation occurs during regeneration.
See [the empirical worked example](../docs/castle_replication.md) for explicit
adoption-year/following-year treatment coding, sample provenance and limits.

## Stronger local memory checks

`python benchmarks/memory_checks.py` runs a fresh-process source-layout and
unit-count matrix through 10 million rows, reusing the resource harness. It
checks long string IDs, shuffled rows, large/small physical Parquet groups,
unused wide inputs, richer conditional-CS designs and native unbalanced SA/BJS.
It retains exact estimate/full-covariance comparisons and checks decoded-leaf
allowances and scratch cleanup. Boundary probes report clean headroom refusals
separately from completed fits. No R installation or network is needed.

`python benchmarks/memory_report.py` rebuilds
[the report](../docs/memory_checks.md). Add `--only CASE...` to select cases,
including their baseline first. Changed source/configuration hashes archive
earlier results. Logs, measurements and initial failures live in
`results/memory_checks`; generated inputs live in ignored `data/memory_checks`.

For actual kernel enforcement, supply `--cgroup-parent` with an existing writable
delegated Linux cgroup v1/v2 memory parent and use a separate `--results` directory.
The runner requires a successful control/cap-detection test and a kernel OOM
self-check, then records per-fit kernel events and peak charged memory.
It never changes system-wide controller settings. The v1 route has now passed
selected fits on this WSL host outside the sandbox after administrator delegation;
see [the kernel report](../docs/kernel_memory_checks.md) and the reviewable
`setup_memory_cgroup.py` command. `STREAM_LIMITS_TEST_CGROUP_PARENT` enables the
real-kernel integration test; ordinary tests skip it explicitly. V2 remains
available for separate host validation. `kernel_memory_report.py` regenerates
the combined report; `host_memory_route.py` retains the user/systemd route probes.

## Observation weights and richer regressions

`python benchmarks/richer_validation.py` audits frozen R did/fixest and independent
dense R regression references: 168 fits, 532 estimate/full-covariance comparisons,
balanced/unbalanced panels, anticipation 0/1, richer SA/BJS controls and region-by-year
effects, all BJS residual groups/leave-out settings and separate clean placebos.
Add `--regenerate --rscript PATH` to regenerate using already installed R packages.
Input/exporter/reference hashes are checked; no downloads or installations occur.
New specifications have not yet been run against the authors' Stata command.

`python benchmarks/richer_resources.py` runs ten serial fresh-process resource
cases, including million-unit SA/BJS and wider region-by-year systems. It uses
the existing RSS watchdog, high-water mark and timeouts, checks aggregation and
cleanup, retains input manifests/source hashes, and archives changed runs.
Use `--only CASE...` to select cases. Numerical/source passes and phase peaks
live in `results/richer_resources`, and the numerical audit in
`results/richer_validation`. Generated Parquet is under ignored `data/richer_resources`.
See [the guide](../docs/richer_models.md) for the explicit weighting opt-in,
256-column added-design limit and remaining unrestricted-FE backend scope.

### Shared-core/reader follow-up

The November shared-core migration audit and six fresh kernel-constrained CS
measurements are in `results/shared_core_migration`. Definition hashes and an
AST-level equivalence review are in `ast_review.json`; before/after resource and
numerical comparisons are in `resource_comparison.json`. Full fits, controller
counters and successful-control/deliberate-OOM proof are in `kernel/`. These
results supplement rather than overwrite the earlier kernel matrix. See
`docs/shared_core_review.md` for the migration and dependency-release caveats.

### Gardner two-stage DiD

`python benchmarks/gardner_validation.py` audits 48 frozen authors' R-command
comparisons and independent dense stacked-regression checks, including four
public Castle Doctrine fits. The pinned source/hash manifest and offline R
outputs live in `tests/data/gardner_reference`. The authors' unmodified R command
is sourced directly from a pinned cache during `--regenerate`; this is not a
claim of testing an installed did2s release. R's demeaning tolerance is tightened
to `1e-10`, without relaxing comparison tolerances.

`python benchmarks/gardner_resources.py` runs six fresh-process resource probes
through the shared launcher, using existing richer-model datasets. Its default
is a portable process-tree RSS watchdog; `--cgroup-parent` explicitly selects a
verified Linux hard cap. Outputs include source/data hashes, kernel counters,
cleanup assertions and a tight-batch covariance-invariance comparison. See
`docs/gardner.md` for the model and public-data example.

### Linear ETWFE

`python benchmarks/etwfe_validation.py` checks frozen authors' R etwfe/emfx
outputs, cell effects and every aggregate's full covariance. The reference
exports retain native R covariance; comparisons explicitly handle the native
unconditional cluster correction. `--regenerate` uses pinned unmodified ETWFE
source functions and matching marginaleffects development code in the ignored
local R cache. Source/input/exporter/output hashes and R session details live
in `tests/data/etwfe_reference`. Ordinary tests require neither R nor downloads.

`python benchmarks/etwfe_resources.py` runs six fresh-process scaling cases,
including one million units, 448 expanded columns, both intercept/control
specifications, and tight-buffer invariance. The shared launcher's portable
watchdog is the default; an explicit `--cgroup-parent` selects verified Linux
kernel enforcement. Raw data/source hashes, kernel counters and cleanup checks
are retained in `results/etwfe_resources`. Nonlinear implementations are linked
from `docs/etwfe.md` and remain outside this implementation.

### Last-treated references and CS balance

`python benchmarks/reference_groups_validation.py` audits 108 CS/SA R reference
specifications; `--regenerate` runs native R settings and a precision confirmation
with only the in-process propensity stopping tolerance/iteration limit changed.
Both output sets and their hashes are retained. The explicit clean last-treated
window and anticipation-label conventions are described in
`docs/reference_groups.md`. No installed R package file is modified.

`python benchmarks/reference_balance_resources.py` probes all-treated CS/SA
samples with varying weights, including two opt-in balance cases. Like the other
new probes, it uses the shared launcher, a portable watchdog by default and
verified Linux enforcement with `--cgroup-parent`. It reuses cached richer-model
inputs. Its `source_generated_units` count includes units removed by the
all-treated filter; actual estimation counts are reported separately.
`docs/covariate_balance.md` explains before/after weights, fixed SMD scales,
observed-date strata and eCDF costs. Normal numerical tests remain in process.
