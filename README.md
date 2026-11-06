# diff-stream

All five estimators support an optional serial or parallel
[unit-cluster refit bootstrap](docs/refit_bootstrap.md), with pointwise bootstrap
SEs and percentile, basic or normal intervals. Ordinary fit defaults are unchanged.

For release preparation, start with the
[scope and handoff checklist](docs/release_handoff.md). Simultaneous
refit-bootstrap bands are explicitly deferred; remaining estimator-extension
decisions and release checks are listed there.

Staggered difference-in-differences with disk-backed panel passes and a common
group-time results interface. **Development prototype; not a public release.**

The development package implements Borusyak–Jaravel–Spiess (BJS) imputation,
Sun–Abraham saturated interactions, and covariate-adjusted Callaway–Sant'Anna (CS)
for balanced or explicitly unbalanced absorbing-treatment panels. All estimators
support a restricted last-treated reference window when no never-treated group exists.

```python
from diff_stream import cs_stream

fit = cs_stream(
    "earnings ~ 1",
    "panels/*.parquet",
    panel=("person", "year"),
    cohort="first_treat",
    control_group="notyettreated",
    threads=4,
)
fit.group_time()
fit.aggregate().summary()  # event study
fit.aggregate("simple").summary()  # post-treatment ATT
fit.aggregate("cohort").summary()
fit.aggregate("calendar").summary()
fit.aggregate(balance_e=2).composition()
fit.aggregate().confint(joint=True, seed=1)
```

`bjs_stream` and `sunab_stream` take the same formula, data, panel and cohort
arguments. `diff_stream(..., estimator="cs"|"sunab"|"bjs"|"gardner"|"etwfe")` is the common entry
point. Results inherit hdfe-stream's `tidy`, `confint`, `wald_test`, `summary`,
`summary_dict`, `summary_json`, and optional pyfixest `etable` integration.

`gardner_stream` (or `diff_stream(..., estimator="gardner")`) adds Gardner's
first-stage-adjusted two-stage DiD: direct static ATT or dynamic event-study
coefficients, with covariates, additional fixed effects and sampling weights.
Use `specification="static"` for the ATT; dynamic fits are the default and
`fit.select(horizons)` selects already fitted coefficients. See
[Gardner two-stage DiD](docs/gardner.md) for inference, reference conventions and
the public Castle Doctrine example.

`etwfe_stream` (or `diff_stream(..., estimator="etwfe")`) implements linear
extended TWFE with saturated covariate interactions, both control conventions,
cohort/unit fixed effects and conditional or unconditional effect inference.
It uses the common group-time and aggregation interface. Nonlinear ETWFE remains
available in existing R/Stata/Python packages; see the
[linear ETWFE guide and alternatives](docs/etwfe.md).

`bacon_decomp("y ~ 1", data, panel=("id", "time"), cohort="first_treat")`
provides a Goodman–Bacon decomposition of conventional TWFE into comparison
estimates, weights and contributions. This diagnostic supports balanced
panels, with optional `weights="column"`, covariate adjustment and always-treated
groups, including designs without a never-treated group. Adjusted fits report
within-timing and full-period between-timing components separately. See
[Goodman–Bacon](docs/bacon.md) and [extensions](docs/bacon_extensions.md).
`twfe_weights("y ~ x", data, panel=("id", "time"), treatment="treated")`
reports the published `feTR` treatment-effect weights and heterogeneity
sensitivity measures, with native unbalanced panels and reversing binary
treatment available. See [TWFE weights](docs/twfe_weights.md) for cell export,
assumptions and validation against the authors' R TwoWayFEWeights package.
BJS clean-placebo estimation is available with `pretrends=L`:
`fit.placebos().tidy()` reports the separate clean-lead regression and
`fit.pretrend_test()` tests it jointly. See [BJS placebos](docs/bjs_placebos.md).

Time must be consecutive positive integers. Cohort is the first treated time;
0 denotes never treated. Future cohorts are treated as never treated only when
their anticipation window also starts after the observed sample. Missing values,
duplicate keys and genuine repeated cross sections are unsupported. Always-treated
units are accepted by the separate Bacon and TWFE-weight diagnostics; estimation entry points
require clean pre-treatment observations.
The default `panel_mode="balanced"` rejects incomplete units. Select
`panel_mode="unbalanced"` to use observed rows or `panel_mode="complete"` to
explicitly drop incomplete units. See [unbalanced panels](docs/unbalanced_panels.md)
for estimator-specific conventions and sample diagnostics. Native unbalanced BJS
and weighted BJS retain temporary Parquet for aggregation; call `fit.cleanup()` when finished.

CS, SA, BJS, Gardner and ETWFE accept `reference_group="auto"` (default), `"nevertreated"`
(strict), or `"lasttreated"`. Without a never-treated cohort, auto uses the last
cohort only as a clean reference and excludes dates from its anticipation-adjusted
onset onward. See [reference populations](docs/reference_groups.md).
CS `balance=True` saves comparison-specific before/after propensity-weight
balance tables for `fit.covariate_balance()`; see
[covariate balance](docs/covariate_balance.md).

CS formulas also accept covariates, e.g. `"y ~ x + C(region)"`, with
`est_method="reg"`, `"ipw"`, or `"dr"` (default). Earlier-comparison-date
covariates on balanced panels (observed-date covariates on native unbalanced panels), control-only propensity trimming and nuisance-adjusted joint
inference follow R did/DRDID. BJS/SA also accept covariates and additional fixed
effects, e.g. `"y ~ x | region^time"`, with unit/time intercepts included automatically.
The additional compiled design is limited to 256 columns; see
[richer SA/BJS models](docs/richer_models.md) for backend scope and identification rules.

All three estimators accept `weights="sampling_weight"` for positive weights
constant within unit by default. Set `allow_time_varying_weights=True` for
date-specific weights, with documented estimator-specific conventions.
Weights enter fitting, aggregation and clustered inference;
actual unit counts remain separate from weight totals. See [sampling weights](docs/weights.md).
CS warns below 30 effective observations in any required comparison stratum.
Inspect `fit.effective_sample_size()` and optionally use `fit.suppress(30)`
to mask coefficients while preserving the original fit and aggregation weights.
See [effective sample size](docs/effective_sample.md).

All three estimators accept `anticipation=A` (default 0). Clean observations
precede `g-A`; event times remain relative to actual treatment. Anticipation
cells enter event studies, while simple/cohort/calendar summaries start at
actual treatment. See [anticipation](docs/anticipation.md).

CS/SA provide `fit.pretrend_test()` for explicitly selected clean pre-treatment
cells or event-time averages. For CS unit-multiplier bands, request
`bootstrap_reps=999, bootstrap_seed=42` at fit time, then call
`fit.aggregate().multiplier_confint()`. See [inference](docs/inference.md) for
anticipation-aware selection, shared infrastructure and memory conventions.

[Aggregation and sample composition](docs/aggregation.md) documents event-window
balancing, contributor weights and observed/cohort sample counts. Cohort support
balancing does not hold individual-unit identities fixed in unbalanced panels.

All three methods return group-time estimates and a joint unit-clustered
covariance. CS supports varying/universal pre-treatment bases and
never/not-yet-treated controls. SA includes every supported cohort interaction
except the clean g-anticipation-1 reference; selecting event-study horizons afterward does not
truncate that specification. BJS returns affected-period effects (including requested anticipation cells), with smart
residual averaging and optional cluster leave-out adjustment, including the
first-stage contribution. See the inference conventions in the Sphinx guide.

## Install and develop

```bash
uv venv --python 3.12
uv pip install -e '.[test,docs,benchmark]'
pytest -q
sphinx-build -W -b html docs docs/_build/html
python examples/quickstart.py
```

Set thread limits before importing Polars, for example
`STREAM_LIMITS_MAX_THREADS=4 python examples/quickstart.py`. `threads=` controls
Numba/BLAS at fit time; Polars sizes its pool when imported. `stream-limits`
detects scheduler allocations. `profile=True` returns the family `Profile`
object after disk-backed validation; profiling uses temporary disk and its
estimates are heuristic, not hard limits.

The [mpdta empirical replication](docs/mpdta_replication.md) checks the documented
minimum-wage/teen-employment example against R did: 24 CS specifications,
full group-time/aggregate covariance and common-draw multiplier inference.
The frozen-data report reruns offline without R or Stata.

The [math_reform replication](docs/math_reform_replication.md) reproduces the
public Goodman–Bacon worked example: all 16 comparisons, decomposition weights,
type summaries and independent TWFE reconstruction, with no Stata dependency.

The [Castle Doctrine Sun–Abraham example](docs/castle_replication.md) checks
the public Cheng–Hoekstra state panel against R fixest and independent dense
regressions, with full covariance, an event-study plot and explicit adoption-year
versus following-year timing sensitivities. Frozen checks need Python only.

## Memory and performance

The source is projected to a streaming Parquet snapshot, then recursively
hash-partitioned by unit. Validation and sorting operate on bounded leaves,
without a global unit-ID table or global sort. Arrow reads whole-unit batches;
Numba/BLAS accumulate CS/SA scores, while BJS retains cohort-sized second moments.
Every lazy sink explicitly requests the streaming engine, including on Polars 1.x.

`preprocess_mb` controls decoded partition/sort buffers; smaller values trade I/O
for memory. `memory_gb` guards allocations before staging and throughout preparation.
Readers, writers and the source query have separate runtime overhead. The joint
covariance still requires O(number of effects squared) memory. A local eight-
million-row CS check completed below a monitored 1 GB RSS ceiling with a 16 MB
preprocessing allowance. See [design notes](DESIGN_NOTES.md).

Tests include independent dense regression/score oracles. R reference exporters
and runnable Stata scripts live in `benchmarks/external/`; reference tests
distinguish matched estimates/variance conventions from pending adjustments.
The [authors’ Stata BJS baseline](docs/stata_validation.md) passes all 16 fits,
including full covariance. The 176-fit extended matrix also passes, with a
48-fit double-storage confirmation resolving the original precision findings.
See [validation instructions](benchmarks/README.md) and the Sphinx guides.

Local benchmark runs are development measurements. The harness records wall
time, total peak RSS, baseline RSS, peak allocated scratch disk, failures and
repeats in subprocesses, and adapts the sibling packages' light/dark figures.
EC2 publication benchmarks are deferred.

IPW and influence-function infrastructure now come from `hdfe_stream.core`;
their temporary copies have been removed. The remaining `diff_stream._core`
contains distinct bounded nuisance/contrast, partition, design and inference
adapters. See [shared infrastructure](SHARED_INFRASTRUCTURE.md) for the reviewed
contracts and the matching hdfe-stream checkout requirement while published
dependency floors and clean-install compatibility are being resolved.
No match-stream dependency is introduced. MIT license.

The [expanded resource checks](docs/resource_checks.md) provide offline,
resumable conditional-CS/unbalanced-panel measurements, RSS ceilings, R
comparisons and retained failure/progress artifacts. They distinguish measured
resource behavior from hard-memory or asymptotic guarantees.

The [stronger memory review](docs/memory_checks.md) extends these checks through
10 million rows and difficult source layouts, with clean headroom refusals and
an optional delegated-cgroup runner. Selected [kernel-enforced checks](docs/kernel_memory_checks.md)
now pass on this WSL host using a delegated v1 memory controller.

The [numerical stress and complexity review](docs/numerical_stress.md) records
102 numerical cases, 80 actual R comparisons and 31 local resource probes.
It includes fixes for covariate-unit-dependent logit convergence and an
overestimated memory admission bound, plus retained support/address-space limits.
