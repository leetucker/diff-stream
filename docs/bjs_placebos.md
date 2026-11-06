# BJS clean-placebo estimates and pre-trend tests

Request `pretrends=L` when fitting BJS to estimate its last `L` clean leads in
a separate regression. The treatment-effect fit still uses the original
imputation specification. The default `pretrends=0` skips the placebo regression.
With [richer BJS models](richer_models.md), this separate clean regression also
includes the specified covariates/additional fixed effects and uses actual
date-specific weights when enabled. It augments the same nuisance specification
with clean leads; it does not revert to an unconditional unit/time model.

```python
from diff_stream import bjs_stream

fit = bjs_stream(
    "y ~ 1",
    "panel.parquet",
    panel=("id", "time"),
    cohort="first_treat",
    panel_mode="unbalanced",
    weights="sampling_weight",
    anticipation=2,
    pretrends=3,
)
try:
    print(fit.placebos().tidy())
    print(fit.placebos().support())
    print(fit.pretrend_test())
    print(fit.aggregate("simple").tidy())
finally:
    fit.cleanup()

# Placebo results and tests remain available after cleanup:
fit.placebos().confint(joint=True, seed=42)
fit.pretrend_test(horizons=[-5, -3])
```

## The separate clean regression

For cohort $g_i$ and anticipation $A$, clean observations have $t<g_i-A$;
never-treated observations are always clean. On these rows only, the placebo
regression fits

$$
Y_{it}=\alpha_i+\lambda_t+
\sum_{h=-A-L}^{-A-1}\gamma_h 1\{g_i>0,t-g_i=h\}+\epsilon_{it}.
$$

This is the separate untreated-sample lead regression described by the
[authors' pre-trend option](https://github.com/borusyak/did_imputation/blob/main/did_imputation.sthlp).
It does not average residuals from the treatment-effect first stage. It includes
the lead indicators jointly with unit and time effects; ordinary first-stage
residual averages generally do not recover these coefficients.

With `anticipation=2, pretrends=3`, the actual event labels are `-5`, `-4`, `-3`.
Potential anticipation periods `-2`, `-1` and actual post-treatment observations
never enter this regression. The reference consists of the remaining clean
observations, including older untreated observations and never-treated units.
There is no single normalized reference event: the last clean lead `-A-1` is
estimated, and the placebo test reports `reference_event_time=None`.

These are **pooled lead-regression coefficients**, rather than cohort-specific
group-time ATTs or averages of the main imputation effects. Changing
`pretrends=L` changes the fitted placebo regression. Selecting `horizons` in a
subsequent test selects restrictions from that existing regression and does
not refit it or change the other fitted lead indicators.

## Panels, weights and future treatment

Balanced, complete and native unbalanced modes support placebos. Native
unbalanced fitting uses actual observed clean rows; no absent outcome is
imputed for the placebo regression. Complete-panel deletion applies to both
the treatment-effect and placebo samples. Every retained unit still needs at
least one clean observation, as required by the BJS imputation fit.

Optional positive, time-invariant sampling weights enter the regression's
weighted normal equations and unit-clustered scores. Multiplying every weight
by a common positive constant does not change coefficients or covariance.
`support()` reports physical observed-unit counts and input-scale weight totals
for each selected lead's cohort/date cells. These totals describe sample support;
they are not regression contrast weights.

The placebo regression retains original future-cohort labels. A cohort treated
just beyond the observed sample can therefore supply clean lead observations,
even when its realized treatment indicator is zero throughout the sample and
the main effect fit pools it with never-treated controls. Anticipation and event
labels remain relative to actual treatment in both results. Original timing is
staged in one extra numeric column only when placebos are requested.

Every requested clean horizon must have observed support, and the jointly
augmented regression must be identified. Missing requested leads, a disconnected
design, collinearity, or too many lead indicators raise explicit errors rather
than dropping columns or choosing a reference silently. Reduce `pretrends` or
provide more clean history when appropriate. Never-treated observations alone
do not resolve the collinearity that occurs when the selected indicators cover
every clean period for every treated unit.

## Inference and result ownership

`fit.placebos()` returns a `BJSPlaceboResult` with `coef`, `tidy`, `confint`,
`wald_test`, `summary`, `summary_json`, `with_vcov`, `support` and
`pretrend_test`. The covariance uses ordinary clean-regression residual scores
clustered by original unit. It does not apply the post-treatment smart-residual
averaging or leave-out correction. Changing `avgeffectsby` or `leaveout` affects
the main treatment inference, while leaving the placebo regression unchanged.

`fit.pretrend_test()` jointly tests all fitted clean lead coefficients by
default; `horizons=[...]` selects a subset. Both `level="auto"` and
`level="dynamic"` use these pooled leads. `level="cell"` is unavailable because
the regression does not estimate cohort-time placebo coefficients. The returned
dictionary lists selected coefficients, contributing cohort/date cells, tested
horizons, and the full `fitted_horizons` used in the regression.

The family conventions apply: CRV0, optional CRV1 multiplying covariance by
$N/(N-1)$, normal coefficient inference, and a chi-squared joint Wald test.
Singular/zero-variance test covariance raises. These conventions differ from
the authors' Stata command's finite-sample degrees-of-freedom correction and
F-test. No exact Stata finite-sample agreement is claimed.

Enabling placebos does not change the main group-time estimates, aggregation
weights or smart/leave-out covariance. Placebos are kept in a separate result
and carried through treatment-effect aggregation and `with_vcov()`. A pre-trend
test called on a main aggregate still tests the stored placebo regression;
its horizons are independent of that aggregate's displayed treatment horizons.
Cleanup removes optional main BJS aggregation files, while placebo tables,
covariance and tests remain available without source rereads.

The package does **not** estimate a joint covariance between placebo and
treatment-effect coefficients. Their separate Gaussian joint intervals should
not be interpreted as one simultaneous band covering a combined pre/post
vector. CS multiplier bands also remain specific to CS.

## Computation and validation

Unit intercepts are eliminated within bounded whole-unit batches. Cohort/date
design matrices and weighted sums form a $(T-1+L)$-dimensional time/lead system;
a second pass accumulates unit-clustered covariance. The numerical state grows
with periods and leads, not the number of units, apart from bounded batches and
existing cohort/time metadata. No full unit-dummy matrix or unit-score table is
stored. Fit/profile guards include the extra system and support-table buffers.
The generic checked linear solver is reused from `_core`; timing and lead
regression conventions remain DiD-specific.

`benchmarks/external/bjs_placebo_reference.R` exports independent dense
full-unit-dummy WLS coefficients and joint covariance. It also checks actual
`fixest` and R `didimputation` lead estimates and SEs after disabling their
finite-sample K/G corrections. The 48 cases cover balanced, unbalanced, complete
and future-cohort samples, sampling weights on/off, anticipation 0/2 and lead
counts 1/2/3. Version notes and fixtures live in
`tests/data/bjs_placebo_reference/`. Regenerate with:

```bash
Rscript benchmarks/external/bjs_placebo_reference.R \
  tests/data/anticipation_panel.csv tests/data/bjs_placebo_reference
```

Python checks also cover all post-treatment averaging/leave-out settings,
unchanged main effects/covariance, known deterministic lead effects, exclusion
of arbitrary affected-period outcomes, input-weight scaling, small shuffled
batches/partitions, CRV1, horizon subsets, shifted/far-future labels, explicit
identification errors, profiling and cleanup.

Run the offline resource checks with:

```bash
STREAM_LIMITS_MAX_THREADS=2 POLARS_MAX_THREADS=2 \
  .venv/bin/python benchmarks/bjs_placebo_resources.py
```

The weighted three-lead checks use eight periods, two threads, an 8 MB
preprocessing allowance and a 1 GB sampled RSS watchdog:

| Input units | Anticipation | Panel mode | Peak RSS |
|---:|---:|---|---:|
| 10,000 | 2 | Unbalanced | 385.7 MiB |
| 100,000 | 2 | Unbalanced | 437.0 MiB |
| 100,000 | 0 | Unbalanced | 446.4 MiB |
| 100,000 | 2 | Complete | 477.4 MiB |

The unbalanced 100,000-unit sample has 750,000 rows; complete mode retains
75,000 units and 600,000 rows. Fits, main-effect aggregation, placebo tests and
scratch cleanup completed successfully. Results and logs live under
`benchmarks/results/bjs_placebo_resources/`. These are finite checks, not a
universal hard-memory guarantee at larger period/lead dimensions.
