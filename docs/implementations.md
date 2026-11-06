# Estimators and validation conventions

| Estimator | Numerical reference | Inference |
|---|---|---|
| CS | R did::att_gt/DRDID, regression/IPW/DR with covariates | Nuisance-adjusted unit scores and estimated-share aggregation |
| SA | R fixest::sunab, saturated unit/time FE regression | Unit-clustered regression covariance, conditional shares |
| Linear ETWFE | Pinned R etwfe/emfx and independent dense stacked WLS | Unit clustering; conditional or unconditional margin/share inference |
| Gardner | Pinned authors’ R did2s command and independent dense stacked WLS | First-stage-adjusted unit-clustered scores; direct static/dynamic coefficients |
| BJS | R didimputation points and SEs; independent dense R joint-covariance oracle | Smart residual averaging and optional cluster leave-out adjustment |
| TWFE weights (`feTR`) | Authors' R TwoWayFEWeights; independent dummy WLS and constrained sensitivity optimization | Descriptive weights/sensitivity thresholds; no ATT inference |

CRV0 sums unit score outer products. CRV1 adds N/(N-1) only. Analytic pointwise intervals
are normal. The separate [refit bootstrap](refit_bootstrap.md) offers percentile,
basic and normal pointwise intervals; simultaneous refit bands are explicitly
deferred in the [release scope](release_handoff.md).
R fixest checks disable K.adj/G.adj; R point/covariance checks disable bootstrap; additional common-draw references
validate the actual R mboot inference calculations.
Inherited joint intervals simulate a Gaussian vector from the estimated
covariance. CS also offers explicit unit influence-function multiplier bands
through `multiplier_confint()`; see [inference](inference.md).

See [unbalanced panels](unbalanced_panels.md) for observed-row sample conventions
and validation.

The [richer-model validation](richer_models.md) extends these references to
explicit time-varying observation weights, SA/BJS covariates and crossed fixed
effects. Its new BJS/clean-placebo covariance references are independent dense
R regressions, with separate Python dummy-regression checks; they are not
additional authors' Stata runs. The added compiled design is bounded at 256 columns.

## BJS residual averaging and small cohorts

```python
from diff_stream import bjs_stream

fit = bjs_stream(
    "y ~ 1",
    "panel.parquet",
    panel=("id", "time"),
    cohort="first_treat",
    avgeffectsby="event_time",
    leaveout=True,
)
fit.aggregate().summary()
```

`avgeffectsby` accepts `cohort_time` (default), `event_time`, `cohort`, or `all`.
Smart averages use weights specific to each output coefficient. `leaveout=True`
applies the authors' cluster leave-out adjustment to treated residuals while
retaining the untreated first-stage contribution. It does not refit the entire
untreated model after deleting each unit. With default cohort-time groups,
the treated residual multiplier is n_g/(n_g-1).

The default cohort-time smart average already reduces to the ordinary cell mean
in the supported equal-weight, unit-clustered design. Its dynamic and simple SEs
are checked against R didimputation. Joint covariance for all four averaging
schemes, with/without leave-out, is checked against an independent R oracle that
builds the full unit/time dummy design. The oracle is independent validation of
the equations; it is not execution of the authors' Stata command.

A contributing singleton makes default cohort-time leave-out undefined and
raises an error. For coarser groups, singleton cell variances are reported as
NaN with a diagnostic; aggregation may produce an identified variance by pooling
other contributing units. An aggregate whose leave-out group still has only one
contributing unit raises an error. Never-treated units cannot supply treatment-
effect variation for this purpose. Broader averaging groups impose no equality
of point effects; residual heterogeneity can make their variance conservative.

The package does not silently suppress small effective samples. Stata's `minn`
default is therefore a separate convention: external comparisons explicitly use
`minn(0)`. Optional CRV1 is also separate from the leave-out adjustment.

`benchmarks/external/validate_bjs.do` exports dynamic/simple estimates and full
covariances from the original authors' command for every averaging/leave-out
combination. The returned external Stata run passes all 16 baseline comparisons
including full covariance. See [Stata validation](stata_validation.md) for the
audit: all 176 extended fits completed and inference checks pass. The 48-fit
double-storage confirmation resolves all 18 original point-tolerance findings.
The optional test reads its outputs from `benchmarks/results/stata_bjs`, or
`DIFF_STATA_BJS`. R exporters and versioned fixtures are in the source checkout.

Method references:

- [Borusyak, Jaravel and Spiess, Revisiting Event Study Designs](https://arxiv.org/abs/2108.12419).
- [Authors' did_imputation implementation](https://github.com/borusyak/did_imputation).
- [Sun and Abraham interactions in fixest](https://lrberge.github.io/fixest/reference/sunab.html).
- [Callaway and Sant'Anna group-time effects](https://bcallaway11.github.io/did/reference/att_gt.html).

Additional comparison projects for future work are
[csa-py](https://github.com/jsr-p/csa-py) and
[diff-diff](https://github.com/igerber/diff-diff). Neither is a runtime dependency.

See [conditional CS](conditional_cs.md) for covariate dates and overlap conventions.

The [mpdta replication](mpdta_replication.md) checks the documented empirical
CS example and 24 specification variants against R did, including full
aggregate covariance and common-draw multiplier bands. It is available offline
and does not require Stata.

The [math_reform worked example](math_reform_replication.md) supplies a public
empirical Goodman–Bacon replication against actual package outputs, with
independent full-sample and pairwise OLS checks in R and Python.

The [Castle Doctrine SA reanalysis](castle_replication.md) adds a public empirical
example: four timing/weighting specifications match actual fixest coefficients,
full covariance and aggregation, with independent R/Python dummy-regression
oracles. The source post indicator starts after the adoption year, so both
timing conventions are retained; this is not a replication of every original
Cheng–Hoekstra specification.

See [Gardner](gardner.md) for its direct coefficient interface and validation.

See [linear ETWFE](etwfe.md) for reference conventions and existing nonlinear alternatives.
