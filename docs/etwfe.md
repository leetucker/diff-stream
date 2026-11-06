# Linear extended two-way fixed effects

`etwfe_stream` implements linear ETWFE with fully interacted covariates,
cohort/time effects and average treatment effects recovered from the regression.
It returns the common `DIDResult`: group-time margins, their joint covariance,
and dynamic, simple, cohort and calendar aggregations.

```python
from diff_stream import etwfe_stream

fit = etwfe_stream(
    "y ~ x + z",
    "panel.parquet",
    panel=("id", "time"),
    cohort="first_treat",
    control_group="notyettreated",
    fixed_effects="cohort",
    inference="unconditional",
    weights="sampling_weight",
    allow_time_varying_weights=True,
    panel_mode="unbalanced",
    threads=2,
)
fit.group_time()
fit.aggregate().tidy()  # event study
fit.aggregate("simple").confint()  # post-treatment ATT
fit.aggregate("cohort").summary()
fit.aggregate("calendar").contributions()
```

`diff_stream(..., estimator="etwfe")` provides the same estimator. Common
reporting, Wald tests, covariance switching, sample-composition diagnostics and
post-fit event windows are available. All temporary partitions are removed before
returning; results do not need the staged data for aggregation.

## Scope and nonlinear alternatives

**This implementation is linear only. Users who need nonlinear ETWFE can use
existing packages:** R's [`etwfe`](https://grantmcdermott.com/etwfe/reference/etwfe.html)
supports nonlinear families including logit and Poisson, with treatment-effect
recovery through `emfx`; Stata's
[`jwdid`](https://friosavila.github.io/app_metrics/app_metrics11.html) also supports
nonlinear ETWFE; Python's
[`diff-diff`](https://github.com/igerber/diff-diff/blob/main/docs/api/wooldridge_etwfe.rst)
documents Poisson and logit implementations. They are alternatives rather than
runtime dependencies or automatic backends. Nonlinear ETWFE and subgroup-specific
`xvar` effects are outside this milestone.

The supported design is a balanced, explicitly completed or observed-row
unbalanced absorbing-treatment panel. `reference_group="auto"` uses clean
full-window controls when available, otherwise a last-treated reference window.
Explicit `"nevertreated"` is strict and `"lasttreated"` selects that window even
with existing never-treated units. The reference is not estimated. This sample
policy is independent of both control conventions; see
[reference populations](reference_groups.md). Unit/time values, missing-value policy,
positive weights and future-cohort normalization follow the existing panel
store. Genuine repeated cross sections, always-treated groups, arbitrary extra
fixed-effect sections, cohort trends and nonbinary treatment are unsupported.

## Regression and reference conventions

The linear regression includes:

- Cohort and time intercepts, or unit and time intercepts with
  `fixed_effects="unit"`.
- Covariate slopes interacted with cohort and calendar time.
- Treatment-cell intercepts and covariate slopes.

Covariates may use the existing numeric, categorical and interaction compiler.
The effect for a cohort/date is the weighted average of that cell's fitted
linear treatment contrast. It generally differs from its treatment-interaction
intercept when covariates are present. Raw-covariate coding spans the same
regression as the R implementation's cohort-demeaned interaction coding;
reported margins and their inference do not depend on that choice of basis.

| Option | Convention |
|---|---|
| `control_group="nevertreated"` | Estimate supported leads and affected cells; omit `g-anticipation-1` separately in each cohort. |
| `control_group="notyettreated"` | Estimate affected cells only; unaffected observations identify the common time structure. There are no estimated pre-treatment coefficients. |
| `fixed_effects="cohort"` | Cohort intercepts; the reference R package's default intercept specification. |
| `fixed_effects="unit"` | Eliminate unit intercepts within bounded batches; corresponds to supplying `ivar` in R. |

Diff-stream defaults to **never-treated** controls, consistently with its other
estimators. R `etwfe` defaults to not-yet-treated controls; select the convention
explicitly when comparing outputs. With no covariates, never-treated controls
and unit intercepts, the supported saturated linear model coincides with the
unit-FE interaction regression already used for SA.

`anticipation=A` moves the affected window to `t >= g-A` while retaining actual
adoption labels and event times. Dynamic effects include requested anticipation
cells; simple/cohort/calendar summaries start at actual adoption, as elsewhere
in the library. R validation shifts adoption dates and applies an explicit
post-estimation window for those summaries. Never-treated dynamic reference
cells are omitted, rather than returned as estimated zeros.

## Conditional and unconditional inference

`inference="unconditional"` is the default. Unit influence functions include
both regression estimation and the empirical covariate distribution over which
cell effects are averaged. Aggregation also accounts for estimated cell-weight
shares and their covariance with the effects. No all-unit score table is saved.

`inference="conditional"` treats the observed covariates and aggregation shares
as fixed, matching an explicit conditional covariance request in R `emfx`.
Both modes use analytic gradients and unit clustering. CRV0 is the raw sum of
unit influence outer products; CRV1 adds `N/(N-1)` only. Normal intervals and
clean pre-treatment Wald tests follow the family conventions. A not-yet-treated
fit cannot supply a pretrend test because it estimates no clean leads.

The pinned R unconditional implementation applies the cluster finite-sample
factor even with HC0; its raw outputs are compared with diff-stream's CRV1
scaling. Conditional R model covariance explicitly disables both K and G
adjustments and is compared with CRV0. Frozen files retain native R covariances;
the comparison code makes this convention visible.

The R exporter aligns any retained placeholder covariance groups with the
filtered `emfx` table using the original group keys, and verifies the resulting
diagonal against the reported standard errors. Windowed anticipation summaries
can trigger R's numerical-Jacobian fallback; those references use centered
differences with step `1e-4`. Python always uses analytic gradients.

## Bounded execution and identification

Cohort/time/cell interactions can multiply design width rapidly. The formula
compiler permits 256 input design columns, while `max_design_columns=2048`
separately limits the fully expanded regression. This is a column limit, not
a guarantee that every such model fits within the process budget. Memory guards
check the dense Gram, covariance and batch-design allowances before allocation.
Reduce `batch_units` or narrow the specification when headroom is insufficient.
`profile=True` returns the heuristic resource profile without fitting.

Working numerical memory depends on expanded columns squared, joint effects
squared and batch size. Unit intercepts are locally eliminated rather than
materialized as global dummy columns. Source execution, readers and allocator
retention still have their own costs; `memory_gb` is a guard rather than an
OS-enforced limit. Resource tests use the separate shared launcher.

Redundant regression columns may be dropped only if every reported treatment
margin remains identified under that normalization. An unidentified margin
raises an error; the estimator does not silently assign it a reference value.
Diagnostics expose the expanded/retained columns and dropped aliases.

## Validation and public example

The reference matrix covers both control groups, both fixed-effect choices,
conditional/unconditional inference, balanced/unbalanced samples, equal/unit/
time-varying weights, covariates, anticipation and the documented public
`mpdta` application. Each reference fit checks group-time margins and the full
covariance of every aggregation. An independent dense all-unit-dummy SVD and
stacked-influence calculation also verifies points and full covariance.

References execute unmodified authors' `etwfe` and `emfx` functions from pinned
commit `d36673d3f07783aa297bbec31d6d8aabfc66243b` (DESCRIPTION: 0.6.2.99), using
matching pinned `marginaleffects` development code. They do not claim testing
an installed ETWFE release. The reference parameterization uses explicit cohort
coefficients (`fe="none"`) or unit FE with explicit slopes (`fe="feo"`), which
span the supported linear models. The matching marginaleffects source is pinned at
`a4ac200424ec397862b434b67c8cd8fff131e972` (1.0.0.10007). Source hashes, exporter/input/output hashes
and the R session are in `tests/data/etwfe_reference`.

```python
import polars as pl
from diff_stream import etwfe_stream

# Bundled public data; run from the source checkout's root.
d = pl.read_csv("tests/data/mpdta.csv")
fit = etwfe_stream(
    "lemp ~ lpop",
    d,
    panel=("countyreal", "year"),
    cohort="first.treat",
    control_group="notyettreated",
    threads=2,
)
fit.aggregate().tidy()
```

This reproduces a documented R-package application/specification, with the
explicit covariance conventions above. It is separate from the existing CS
[mpdta replication](mpdta_replication.md).

Run `python benchmarks/etwfe_validation.py` for the offline numerical audit.
`--regenerate --rscript /path/to/Rscript` uses the pinned ignored source cache
under `benchmarks/external/.r` and installed local validation libraries. Normal
tests neither download sources nor require R.
`python benchmarks/etwfe_resources.py` runs fresh-process scaling probes with a
portable watchdog; pass `--cgroup-parent /path/to/delegated/group` for verified
Linux kernel enforcement. The constrained pytest fit uses the common portable
resource fixture; ordinary numerical tests stay in process.

The [release handoff](release_handoff.md) consolidates completed BJS/SA/CS/
Goodman–Bacon scope, open extension decisions and explicit deferrals. Adding
linear ETWFE does not close those decisions; `DESIGN_NOTES.md` retains the history.

## Local resource observations

Six fresh-process weighted, native-unbalanced fits completed under a 1.5 GB
Linux/WSL cgroup v1 cap, with no OOM kills in estimation and successful scratch
cleanup. The wider case expands to 448 design columns (336 retained), exercising
complexity beyond the small reference examples.

| Specification | Units | Observed rows | Expanded columns | Peak RSS (MiB) |
|---|---:|---:|---:|---:|
| Not-yet-treated, cohort FE | 10,000 | 40,449 | 112 | 374 |
| Not-yet-treated, cohort FE | 100,000 | 403,224 | 112 | 494 |
| Not-yet-treated, cohort FE | 1,000,000 | 4,031,952 | 112 | 507 |
| Never-treated, unit FE | 100,000 | 403,224 | 155 | 481 |
| Wider never-treated, cohort FE | 20,000 | 145,074 | 448 | 480 |
| Not-yet-treated, tighter buffers | 100,000 | 403,224 | 112 | 449 |

Reducing batch units from 256 to 64 and preprocessing allowance from 8 MB to
4 MB changed cell coefficients by at most `2.3e-14` and covariance entries by
`1.1e-17`; every aggregation also matched. These are individual local runs, not
a universal memory bound or a public speed comparison. Kernel charge differs
from process RSS. Raw metrics, kernel counters and the deliberate-OOM control
are in `benchmarks/results/etwfe_resources`.

The completed audit covers **60 reference configurations**. Maximum absolute
errors against R were `1.11e-12` for cell effects and `1.65e-12` across cell and
aggregate covariance entries. The complete test run passed 1,291 tests with one
kernel-only skip; all 145 ETWFE tests also passed on Polars 1.37. The local
kernel-constrained fixture passed separately. Other operating systems are
covered by the configured portable CI, not by these local observations.
