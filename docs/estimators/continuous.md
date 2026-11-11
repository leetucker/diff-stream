# Continuous treatment

`did_continuous` estimates the effects of a treatment that comes in different
amounts (a dose), as in Callaway, Goodman-Bacon and Sant'Anna (2024). Each treated
unit receives its dose when it is first treated and keeps it; untreated units receive
none. The estimates are the average effect of each dose on the units that received it,
ATT(d), and its derivative, the average causal response ACRT(d): how much the effect
changes with a little more of the dose.

```python
from diff_stream import did_continuous

fit = did_continuous(
    "y ~ 1",
    "panel.parquet",
    panel=("id", "time"),
    cohort="first_treat",
    dose="dose",
)
fit.summary()                          # overall ATT and ACRT
fit.dose_response()                    # ATT(d) over a grid of doses
fit.dose_response("slope")             # ACRT(d)
fit.aggregate(target="slope").summary()  # event study of the ACRT
```

## The data

- **`dose`**: a column with each treated unit's dose, positive and constant within
  the unit. The doses of never-treated units are not used. A dose that changes over
  time within a unit, or a treated unit with no dose, is an error.
- **`cohort`** or **`treatment`**: the timing, as for the other estimators (see
  [Your data](../data.md)). Units may be treated at different times; the panel needs
  never-treated units.
- The panel must be balanced: `incomplete_units="drop"` drops units with missing
  periods. Sampling weights are not supported. Covariates and a fixed effect are
  experimental (see [Covariates](#covariates-experimental)).

## The estimates

For each cohort and period, the outcome change of the cohort's units since the base
period is regressed on a B-spline in the dose, and the comparison units' mean change
is subtracted. The comparisons are those of
[Callaway-Sant'Anna](callaway_santanna.md): `control_group="not_yet_treated"`
(default) or `"never_treated"`, and `base_period="varying"` (default) or
`"universal"`.

- **The spline**: `degree` (default 3, cubic) and `interior_knots` (default 0), placed
  at quantiles of the treated doses. With no interior knots the dose-response curve
  is a polynomial of that degree. The spline's boundary is the range of the treated
  doses, and doses outside it cannot be evaluated.
- **A discrete dose**: with `dose_type="discrete"`, a dose that takes a few values
  gets a separate mean for each value in place of the spline (the paper's saturated
  regression). ATT(d) is then the mean outcome change of the units at dose d less the
  comparison units', and ACRT(d) is the change in ATT(d) from the next lower dose, or
  from no dose, divided by the difference in doses. The curves are evaluated at the
  dose values only. Every cohort needs units at every dose value, and the dose can
  take at most 1,000 values; `degree` and `interior_knots` do not apply.
- **`dose_response(target, dose)`**: ATT(d) (`target="level"`, default) or ACRT(d)
  (`"slope"`), averaged over the post-treatment cohort-period cells. Each cohort's
  weight is its share of the treated units, split evenly over its post-treatment
  periods. `dose` defaults to `dose_grid`, which by default is the 10th to 99th
  percentiles of the treated doses, or a discrete dose's values.
- **The coefficients**: the overall ATT, the effect of being treated at the doses
  received, and the overall ACRT, the average of ACRT(d) over the treated units'
  doses, both with the same weights as the curves. The overall ATT equals R did's
  `aggte(type = "group")` of Callaway-Sant'Anna on being treated.
- **`aggregate(kind, target)`**: aggregations of the cohort-period effects, with the
  options of `DIDResult.aggregate`: an event study (default), by cohort, by calendar
  period or overall. With `target="level"` it aggregates the effect of being treated,
  with `"slope"` the ACRT. Its `"overall"` weights cells by their units, as for
  Callaway-Sant'Anna, so it can differ from the coefficients, which weight cohorts.
- **`cohort_time(target)`**: the cohort-period effects.

## Covariates (experimental)

```python
fit = did_continuous(
    "y ~ income | region",
    "panel.parquet",
    panel=("id", "time"),
    cohort="first_treat",
    dose="dose",
)
```

Callaway, Goodman-Bacon and Sant'Anna do not develop an estimator with covariates.
Their paper shows which effects given the covariates are identified (Supplementary
Appendix SE.3), and their R package contdid takes no covariates. Covariates here are
an extension, and every fit with them issues an `ExperimentalWarning` that links to
this section.

**How they enter.** In each cohort and period, the comparison units' outcome change is
regressed on the covariates and the fixed effect. Each treated unit's change less its
predicted change is then fitted on the dose, as without covariates. This is Callaway
and Sant'Anna's outcome-regression adjustment, and the effect of being treated equals
`did_callaway_santanna(..., adjustment="regression")` with the same covariates.

- **The formula**: covariates before `|`, and one fixed-effect term after it, which
  may be an interaction (`region^industry`). The fixed effect is absorbed exactly,
  whatever its number of levels. Because each cohort and period has its own
  regression, `| region` gives region-by-period effects. The unit and the period
  cannot be fixed effects: unit effects are already differenced out, and the
  regressions already differ by period. Several fixed-effect terms are not supported
  yet; a categorical variable with few levels can be given as a covariate,
  `C(industry)`.
- **Covariate values**: each unit's covariates and fixed-effect level are taken in the
  base period, or in the period itself when it comes first (pre-treatment periods with
  `base_period="universal"`), as in R's did. Values after treatment, which treatment
  may have changed, are not used.
- **Units left out**: a treated unit whose fixed-effect level has no comparison units
  in one of its cohort's periods cannot be adjusted. It is left out of its cohort's
  estimates, with a warning, and stays a comparison unit for other cohorts. The
  numbers are in `diagnostics["treated_units_left_out"]`.

**What the estimates mean, and when.** ATT(d) is the average effect of dose d on the
units that received it, averaged over their covariates. It needs:

- *Parallel trends given the covariates*: among units with the same covariates (and
  fixed-effect level), the outcome without treatment would have changed in the same
  way whatever the cohort and the dose.
- *Overlap*: comparison units with covariates like the treated units'. For the fixed
  effect, comparison units in each level, each period.
- *A linear comparison regression*: the untreated change is linear in the
  covariates. This holds by construction for the fixed effect and for categorical
  covariates; for continuous covariates it is an assumption.

ACRT(d), the overall ACRT and any comparison of the effects of different doses also
need *strong parallel trends given the covariates*: units at each dose would have had
the same effect of any dose as units at other doses with the same covariates. Even
then, ACRT(d) is the slope of the covariate-averaged ATT(d). If the covariates of the
units change with the dose, and the effects vary with the covariates, that slope mixes
the effect of a little more dose with the change in which units receive it. It is an
average causal response only if the effects do not vary with the covariates or the
dose is unrelated to them.

The standard errors include the estimation of the comparison regressions. Once you
have weighed these conditions, you can silence the warning with
`warnings.filterwarnings("ignore", category=diff_stream.ExperimentalWarning)`;
`hdfe_stream.ExperimentalWarning` silences it for every package built on hdfe-stream.

## Inference

Standard errors come from the influence functions of the estimates, clustered by
unit. For the overall ATT and for `aggregate()`, they include the uncertainty of the
estimated cohort shares that weight them, as R's did does. The curves and the overall
ACRT treat those weights as fixed, as R's contdid does. The overall ACRT includes the
uncertainty of the treated units' doses it averages over, which for a discrete dose
are each cohort's shares of units at each dose. With
`multiplier_bootstrap_reps`, `dose_response(uniform=True)` gives confidence bands that
hold over all doses at once, studentized by the draws' interquartile range.

The fit makes three passes over the staged panel, after staging it. It keeps the
treated units' doses in memory to place the knots, 8 bytes per treated unit; a
discrete dose keeps only its values.

## Validated against

The authors' R package contdid, and R did for the effect of being treated: the
overall ATT and ACRT, both dose-response curves, and the event studies, with a cubic
spline and with a quadratic spline with interior knots, with not-yet-treated
comparisons and a varying base period (the only settings contdid 0.1.1 applies).
The estimates match contdid's where its curves are evaluated with the basis they
were fitted with; the standard errors match once three problems in contdid 0.1.1
and ptetools 1.0.1 are corrected. The other settings are checked against R did and
against a regression of each cell's outcome change on the dose.

contdid has no discrete dose. That is checked against Python's diff-diff 4.0.0, with
both comparison groups: the estimates, both curves and their standard errors, the
cohort-period effects and the event study. diff-diff holds the estimated shares
fixed in its overall and event-study standard errors; those are checked against a
direct calculation from the units' influence on each estimate, which gives
diff-diff's numbers when the shares are held fixed.

Covariates are checked against diff-diff's regression adjustment
(`estimation_method="reg"`), with a covariate that changes over time and the region
entered as dummies, for both dose types and comparison groups: the estimates, both
curves and their standard errors, the cohort-period effects and the event study. The
effect of being treated equals Callaway-Sant'Anna's regression adjustment, an absorbed
fixed effect equals its dummies, and every standard error equals a direct calculation.
See
[Differences from other packages](../implementations.md#conventions-that-differ) and
[Validation](../validation.md).

Callaway, B., Goodman-Bacon, A. and Sant'Anna, P. H. C. (2024).
Difference-in-differences with a continuous treatment. NBER Working Paper 32117.
