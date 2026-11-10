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
  periods. Covariates and sampling weights are not supported.

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
- **`dose_response(target, dose)`**: ATT(d) (`target="level"`, default) or ACRT(d)
  (`"slope"`), averaged over the post-treatment cohort-period cells. Each cohort's
  weight is its share of the treated units, split evenly over its post-treatment
  periods. `dose` defaults to `dose_grid`, which by default is the 10th to 99th
  percentiles of the treated doses.
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

## Inference

Standard errors come from the influence functions of the estimates, clustered by
unit. For the overall ATT and for `aggregate()`, they include the uncertainty of the
estimated cohort shares that weight them, as R's did does. The curves and the overall
ACRT treat those weights as fixed, as R's contdid does. With
`multiplier_bootstrap_reps`, `dose_response(uniform=True)` gives confidence bands that
hold over all doses at once, studentized by the draws' interquartile range.

The fit makes three passes over the staged panel, after staging it. It keeps the
treated units' doses in memory to place the knots, 8 bytes per treated unit.

## Validated against

The authors' R package contdid, and R did for the effect of being treated: the
overall ATT and ACRT, both dose-response curves, and the event studies, with a cubic
spline and with a quadratic spline with interior knots, with not-yet-treated
comparisons and a varying base period (the only settings contdid 0.1.1 applies).
The estimates match contdid's where its curves are evaluated with the basis they
were fitted with; the standard errors match once three problems in contdid 0.1.1
and ptetools 1.0.1 are corrected. The other settings are checked against R did and
against a regression of each cell's outcome change on the dose. See
[Differences from other packages](../implementations.md#conventions-that-differ) and
[Validation](../validation.md).

Callaway, B., Goodman-Bacon, A. and Sant'Anna, P. H. C. (2024).
Difference-in-differences with a continuous treatment. NBER Working Paper 32117.
