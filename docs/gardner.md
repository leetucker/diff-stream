# Gardner two-stage difference-in-differences

`gardner_stream` implements Gardner's two-stage DiD for absorbing-treatment
panels. It returns a `GardnerResult` with direct static or event-study
coefficients and their joint covariance.

```python
from diff_stream import gardner_stream

fit = gardner_stream(
    "y ~ x | region^time",
    "panel.parquet",
    panel=("id", "time"),
    cohort="first_treat",
    specification="dynamic",  # default; or "static"
    weights="sampling_weight",  # optional
    allow_time_varying_weights=True,
    panel_mode="unbalanced",
    threads=2,
)
fit.tidy()
fit.select([-3, -2, 0, 1, 2]).confint()
fit.pretrend_test(horizons=[-3, -2])
```

The common entry point also accepts `diff_stream(..., estimator="gardner")`.
Use `specification="static"` for one `ATT` coefficient. Standard `summary`,
`summary_json`, `confint`, `wald_test`, `with_vcov` and `etable` methods are
available. Static/dynamic specifications are separate fits. There is no
`aggregate()` or cohort-time table: the second-stage coefficients are the
estimands. `support()` reports each coefficient's observed cohort/date support
and original-scale sampling-weight totals. These are descriptive support totals,
not a decomposition of its first-stage-adjusted influence function.

## Specification and inference

The first stage estimates outcome covariates and unit/time intercepts using
clean observations, `t < g-anticipation`, plus all never-treated observations.
Additional crossed fixed effects use the same bounded compiler as richer BJS/SA
models. Its additional design is limited to 256 columns; unit-interacted fixed
effects and varying slopes remain unsupported. Every retained unit must have a
clean observation, and the untreated nuisance system must be identified.

The second stage regresses the outcome minus its first-stage prediction on:

- **Static:** an indicator for affected observations, `t >= g-anticipation`.
- **Dynamic:** mutually exclusive event-time indicators for treated cohorts,
  omitting a clean reference period, by default `-anticipation-1`.

There is no second-stage intercept. Never-treated and omitted-reference rows
have zero second-stage design. All supported event times are fitted. `select()`
selects already fitted coefficients and the corresponding covariance submatrix;
it does not impose zero effects at other event times or refit the regression.
An explicit `reference_period` must be clean and observed in the retained sample.
Event times remain relative to actual adoption, including when anticipation is
positive. With anticipation, the static ATT pools affected pre-adoption rows too.

Sampling weights enter both WLS stages and unit scores. They must be positive,
finite and constant within unit unless time variation is explicitly enabled.
The balanced, complete and native unbalanced sample policies match the other
estimators; native unbalanced Gardner uses the observed-row regressions.
Future cohorts whose clean window extends beyond the sample are normalized to
never-treated under the library's existing convention. Always-treated units and
genuine repeated cross sections are unsupported. `reference_group="auto"`
uses an available clean full-window reference; otherwise it restricts dates to
before the last cohort is affected. Explicit `"nevertreated"` remains strict,
and `"lasttreated"` selects that clean window explicitly. The last reference
cohort has zero second-stage design, including leads; dynamic estimates refer
to the earlier target cohorts. See [reference populations](reference_groups.md).

Inference uses the second-stage unit score **minus its first-stage estimation
correction**. It does not treat the first-stage predictions as known. CRV0 sums
unit influence outer products; CRV1 multiplies by `N/(N-1)` only. Intervals use
normal inference; pre-treatment tests use the shared clean-horizon Wald helper.
The fitted first-stage model conditions on observed covariates/timing. No
resampling/refit bootstrap or other clustering dimension is implemented.

The implementation eliminates unit intercepts within each bounded batch and
accumulates small first-stage Grams, cross derivatives and cluster influence
cross-products. It retains no all-unit score matrix. Temporary partitions are
removed before returning; no result cleanup is required. `profile=True` returns
a heuristic resource profile. Memory still grows with model columns, periods,
event-time covariance and batch size; source execution is not hard capped by
`memory_gb`. See [resource controls](resources.md).

## Validation and public example

The frozen references cover 48 fits: balanced/unbalanced panels, static/dynamic
specifications, anticipation 0/1, equal/unit/date-specific weights, numeric and
crossed-FE first stages, custom references, and four Castle Doctrine fits.
Every coefficient and full covariance matrix is compared with both the authors'
R command and an independent dense all-unit-dummy stacked-equation calculation.

The authors' unmodified `did2s.R` is sourced from pinned commit
`3adaf985980d4b46205de2e1856f93ba075baefe`, whose DESCRIPTION labels it 1.3.0;
validation does not claim an installed `did2s` package release. `fixest`'s
fixed-effect tolerance is set to `1e-10` for accurate comparisons. Source URLs,
hashes, exporter/input/output hashes and R session details are retained in
`tests/data/gardner_reference`. Production code has no R dependency.

The public example reuses the frozen Cheng–Hoekstra Castle Doctrine data
[described here](castle_replication.md), with treatment beginning in the year
after `effyear`, consistent with the source `post` indicator. Both unweighted
and population-weighted static/dynamic Gardner fits are validated. This is a
reanalysis of that outcome/treatment specification, not a replication of every
original-paper specification.

```python
import polars as pl
from diff_stream import gardner_stream

# From the repository root; data are bundled for offline reproduction.
d = pl.read_csv("tests/data/castle.csv", null_values="NA").with_columns(
    pl.when(pl.col("effyear").is_null())
    .then(0)
    .otherwise(pl.col("effyear") + 1)
    .cast(pl.Int64)
    .alias("first_treat")
)
static = gardner_stream(
    "l_homicide ~ 1",
    d,
    panel=("state", "year"),
    cohort="first_treat",
    specification="static",
    threads=2,
)
static.tidy()
```

Run the offline numerical audit with `python benchmarks/gardner_validation.py`.
`--regenerate --rscript /path/to/Rscript` reruns the authors' command using the
pinned source cache under `benchmarks/external/.r/did2s-source/<commit>/`.
Populate that ignored cache from the URLs/hashes in the frozen provenance
manifest; no source downloads or R installation occur during tests.

`python benchmarks/gardner_resources.py` runs fresh-process scaling checks using
the shared portable watchdog. Pass `--cgroup-parent /path/to/delegated/group`
for explicitly verified Linux kernel enforcement. The normal test suite also
includes a constrained richer unbalanced fit; its shared resource fixture runs
on Linux, macOS and Windows. These are distinct from ordinary numerical tests.

Method references: [Gardner's paper](https://arxiv.org/abs/2207.05943),
[authors' software and explanation](https://www.kylebutts.com/papers/did2s/),
and [pinned R implementation](https://github.com/kylebutts/did2s/blob/3adaf985980d4b46205de2e1856f93ba075baefe/R/did2s.R).

## Local resource results

Six fresh-process checks completed under the delegated WSL/Linux cgroup v1
controller with a 1.5 GB cap, no kernel OOM kills and clean temporary-file
removal. They use time-varying weights, crossed region/time effects and native
unbalanced panels; the unit count describes generated units, while the row count
reflects actual observed rows.

| Specification | Units | Observed rows | First-stage columns after unit elimination | Second-stage coefficients | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| Dynamic | 10,000 | 40,449 | 30 | 13 | 483 |
| Dynamic | 100,000 | 403,224 | 30 | 13 | 572 |
| Dynamic | 1,000,000 | 4,031,952 | 30 | 13 | 591 |
| Wider dynamic | 20,000 | 145,074 | 122 | 29 | 540 |
| Dynamic, tighter buffers | 100,000 | 403,224 | 30 | 13 | 536 |
| Static | 100,000 | 403,224 | 30 | 1 | 583 |

Reducing batch units from 256 to 64 and preprocessing allowance from 8 MB to
4 MB changed coefficients by at most `1.7e-14` and covariance entries by at most
`3.3e-18`. All kernel counters and the positive/deliberate-OOM self-check are
retained under `benchmarks/results/gardner_resources`. RSS differs from total
kernel-charged memory. These individual runs establish local feasibility, not
a general constant-memory guarantee or a public performance comparison.

The numerical audit's maximum absolute differences from the authors' command
were `2.54e-11` for coefficients and `2.32e-13` for covariance entries. The
independent dense calculation differed by at most `8.49e-14` and `1.39e-15`,
respectively. Ordinary tests also check covariate/weight rescaling, identification
failures, complete-panel dropping, shuffled long IDs, CRV switching, profile
output, post-fit selection and portable constrained fits.

Before release, review the [current scope and handoff](release_handoff.md).
It distinguishes completed BJS/SA/CS/Goodman–Bacon extensions from the open
BJS/SA/CS decisions and agreed deferrals. Adding Gardner does not close those
decisions; `DESIGN_NOTES.md` retains the milestone history.
