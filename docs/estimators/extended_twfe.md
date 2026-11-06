# Extended TWFE

`did_extended_twfe` is Wooldridge's extended two-way fixed effects estimator
(linear): one regression with a separate treatment effect for every cohort and
period, and covariates interacted with cohort, time and treatment, so that the
effects can vary with the covariates. Each cohort-period effect is the
average, over that cell's units, of its fitted treatment effect, as R's etwfe
reports them with `emfx`.

```python
from diff_stream import did_extended_twfe

fit = did_extended_twfe(
    "lemp ~ lpop",
    "mpdta.parquet",
    panel=("countyreal", "year"),
    cohort="first.treat",
    control_group="not_yet_treated",
)
fit.aggregate().summary()            # event study
fit.aggregate("overall").summary()
```

## Options

- **`control_group`**: `"never_treated"` (default, as for the other
  estimators) also estimates pre-treatment effects, omitting the last
  untreated period; `"not_yet_treated"` (R etwfe's default) estimates
  post-treatment effects only, and without covariates equals BJS imputation.
- **`fixed_effects`**: `"cohort"` intercepts (R etwfe's default) or `"unit"`
  fixed effects (R's `ivar`).
- **`treat_sample_as_fixed`**: False (default) gives standard errors for the
  population the units were drawn from, including the variation from
  averaging over the sample's covariates and cohort shares. True treats those
  as fixed, giving the effect for these units (smaller standard errors; R
  emfx's conditional option).
- **`max_design_columns`** (default 2048) limits the expanded design, after
  the interactions.

Only the linear model is implemented. For logit or Poisson ETWFE use R's
[etwfe](https://grantmcdermott.com/etwfe/) or Stata's `jwdid`.

## Validated against

The authors' R `etwfe` and `emfx` (pinned source) and an independent dense
regression: both control groups, both fixed-effect choices, conditional and
unconditional inference, balanced and unbalanced, weights, covariates and
anticipation, each with cell effects and the full covariance of every
aggregation, and the `mpdta` example. See [Validation](../validation.md).

Wooldridge, J. M. (2021). Two-way fixed effects, the two-way Mundlak
regression, and difference-in-differences estimators. SSRN 3906345.
