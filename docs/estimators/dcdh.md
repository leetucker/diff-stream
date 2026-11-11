# Switching treatment (de Chaisemartin and D'Haultfoeuille)

`did_dcdh` estimates the effects of a treatment that may take several values and
switch on and off, as de Chaisemartin and D'Haultfoeuille's `did_multiplegt_dyn`
does. Each group's baseline is its treatment in its first period, and its first
switch the first period its treatment differs from that. Effect l compares the
outcome change of the groups that first switched l − 1 periods earlier, since the
period before their switch, with that of groups with the same baseline that have not
switched yet.

```python
from diff_stream import did_dcdh

fit = did_dcdh(
    "y ~ 1",
    "panel.parquet",
    panel=("county", "year"),
    treatment="policy",
    effects=5,
    placebos=3,
)
fit.summary()               # effect::1 ... effect::5, placebo::1 ... placebo::3,
                            # average_total_effect
fit.pretrend_test()         # placebos jointly zero
fit.effects_equal_test()    # all effects equal
fit.switchers()             # switching groups behind each estimate
```

## The data

- **`treatment`**: a numeric column, binary or with several values, which may change
  in any period, up or down, and more than once.
- **Groups are compared only with groups of the same first-period treatment.** A
  first-period treatment with very many values (a continuous one) leaves few groups to
  compare, and more than 10,000 distinct values is an error.
- **`incomplete_units`**: `"error"` (default), `"drop"` (groups missing some periods),
  or `"keep"`, which uses each group's observed periods. A change is then used only
  where both of its periods are observed, and a group not observed in the period
  before its switch has an uncertain switch date: it is used only as a control, until
  its last observed period before the switch. Rows with an outcome but no treatment
  are not supported; drop them or fill the treatment in first.

## The estimates

- **`effect::l`**, for l = 1 to `effects`: the outcome change from the period before
  the first switch to l − 1 periods after it, among the switchers, less the change of
  the groups with the same baseline that have not switched by then. Groups whose
  treatment rose (switchers in) and fell (switchers out) are estimated separately and
  combined, weighted by their numbers, the latter with the sign reversed, so that each
  effect is that of more treatment. `switchers="in"` or `"out"` uses only one of them.
- **`placebo::l`**: the same comparison, for the same groups, of the change from the
  period before the switch back l periods, before the switchers switched. Placebos near zero
  support parallel trends.
- **`average_total_effect`**: the effects averaged over horizons and switchers, per
  unit of treatment received.
- **`normalized=True`** divides each effect by the switchers' average change in
  treatment up to it, so that effects are per unit of treatment.
- **`only_never_switchers=True`** compares switchers only with groups whose treatment
  never changes, rather than with any group not yet switched.
- **Covariates**: `"y ~ x + z"`. Within each baseline, the outcome's one-period
  changes among the groups not yet switched are regressed on the covariates' changes,
  with period effects, and each comparison uses the outcome changes net of the
  covariates' fitted part. The standard errors account for estimating that
  regression.

Some groups are not used, as in `did_multiplegt_dyn`: groups whose treatment goes both
above and below their baseline, baselines in which every group first switches in the
same period (there is no one to compare with), and switchers whose average treatment
after the switch equals their baseline. A switcher's effects reach only as far as some
group with its baseline is still unswitched; `switchers()` and
`diagnostics["last_period_with_controls"]` show what is left.

## Inference

The standard errors are the authors' conservative ones, clustered by group: each
switcher's change is demeaned within the groups that share its baseline, first switch
and new treatment, and each control's within its baseline and period. The joint
covariance of all the estimates is in `fit.vcov`; `pretrend_test()` and
`effects_equal_test()` are Wald tests that use it.

The fit stages the panel once and reads it three times: for the baselines and switch
periods, for sums by baseline and period, and for each group's scores.

## Validated against

The authors' R package DIDmultiplegtDYN: estimates, standard errors, the numbers of
switchers and the p-values of its joint placebo and equal-effects tests, which use the
covariances between the estimates, with each option above, covariates, a panel in which
every group switches, and an unbalanced panel with and without covariates. See
[Differences from other packages](../implementations.md#conventions-that-differ) and
[Validation](../validation.md).

de Chaisemartin, C. and D'Haultfoeuille, X. (2024). Difference-in-differences
estimators of intertemporal treatment effects. NBER Working Paper 29873.
