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
  first-period treatment with very many values leaves few groups to compare, and more
  than 10,000 distinct values is an error. For a continuous one, see
  [A continuous first-period treatment](#a-continuous-first-period-treatment).
- **`incomplete_units`**: `"error"` (default), `"drop"` (groups missing some periods),
  or `"keep"`, which uses each group's observed periods. A change is then used only
  where both of its periods are observed, and a group not observed in the period
  before its switch has an uncertain switch date: it is used only as a control, until
  its last observed period before the switch. Rows with an outcome but no treatment
  are not supported; drop them or fill the treatment in first.
- **`weights`**: sampling weights, which may change over time with
  `allow_time_varying_weights=True`. A group enters each comparison with its weight in
  the period the effect is measured at (for a placebo, that of the matching effect),
  as in `did_multiplegt_dyn`.

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
- **`trends_by="region"`** (or a list of columns, constant within each group) lets
  trends differ across regions: switchers are compared only with groups of the same
  region as well as the same first-period treatment. With covariates, their
  coefficients are still estimated per first-period treatment, with period effects by
  region.
- **`linear_trends=True`** allows each group its own linear trend. The comparisons
  use the first differences of the outcome and covariates, and effect l is the sum of
  the first l of them, for the switchers that have all l (so that the trend is
  removed from the same groups at every horizon); placebo l likewise. Groups that
  first switch in the second period have no trend to remove and are not used, and
  the average total effect is not estimated. With missing periods, a first difference
  is used only where both of its periods are observed.

Some groups are not used, as in `did_multiplegt_dyn`: groups whose treatment goes both
above and below their baseline, baselines in which every group first switches in the
same period (there is no one to compare with), and switchers whose average treatment
after the switch equals their baseline. A switcher's effects reach only as far as some
group with its baseline is still unswitched; `switchers()` and
`diagnostics["last_period_with_controls"]` show what is left.

## A continuous first-period treatment

When the first-period treatment is continuous (a tax rate, a price), no two groups
share one, so no group has a comparison. `continuous_baseline=k` compares all groups
together instead, assuming that untreated outcome trends may differ by a polynomial
of degree k in the first-period treatment, period by period. This is the authors'
`continuous` option (de Chaisemartin et al., Web Appendix, Section 1.10):

```python
fit = did_dcdh("y ~ 1", data, panel=("county", "year"), treatment="tax_rate",
               effects=4, placebos=2, continuous_baseline=2)
```

- The polynomial enters as covariates: for each period from the second, the
  first-period treatment's powers 1 to k times being at or after that period. Their
  changes are each period's trend as a polynomial in the first-period treatment, and
  they are estimated and removed as other covariates are (see Covariates above), so
  other covariates can be added.
- Switchers in and out are still defined by whether their treatment rose or fell, and
  `normalized=True` and the average total effect use their actual changes in
  treatment. The switching cohorts of the variance are defined by the period and
  direction of the switch.
- **The standard errors are not backed by a proven asymptotic result**, as the
  authors note: in their simulations they are conservative with a correctly
  specified polynomial and can be too small with a misspecified one. Every fit issues
  a `ContinuousBaselineWarning` saying so. Checking how the estimates change with k
  is a useful guard. The authors also suggest a bootstrap, which `did_dcdh` does not
  provide.

## Inference

The standard errors are the authors' conservative ones, clustered by group: each
switcher's change is demeaned within the groups that share its baseline, first switch
and new treatment, and each control's within its baseline and period. The joint
covariance of all the estimates is in `fit.vcov`; `pretrend_test()` and
`effects_equal_test()` are Wald tests that use it.

**Clustering above the group.** `cluster="state"` clusters the standard errors by a
column constant within each group, as `did_multiplegt_dyn`'s `cluster` option does:
the groups' terms are summed by cluster, and the demeaning's degrees-of-freedom
correction counts distinct clusters rather than groups. The variance has no
small-sample factor, and the degrees of freedom are the clusters less one. Each
group must be in one cluster.

The fit stages the panel once and reads it three times: for the baselines and switch
periods, for sums by baseline and period, and for each group's scores. With
`linear_trends`, the switchers' sums take a read of their own, since which switchers
each effect uses depends on the controls' sums.

## Validated against

The authors' R package DIDmultiplegtDYN: estimates, standard errors, the numbers of
switchers and the p-values of its joint placebo and equal-effects tests, which use the
covariances between the estimates, with each option above, covariates, a panel in which
every group switches, an unbalanced panel with and without covariates, weights that
change over time, trends by region and group-specific linear trends, alone and
combined; standard errors clustered above the group, with each of those options; and
a continuous first-period treatment with polynomials of degree 1 and 2, with
normalized effects, covariates, weights, switchers in only and clustering. With
missing periods in switchers' paths, `did_multiplegt_dyn`'s `continuous` option
departs from the estimator it describes; there `did_dcdh` is checked against that
estimator built by hand and run through `did_multiplegt_dyn` without the option. See
[Differences from other packages](../implementations.md#conventions-that-differ) and
[Validation](../validation.md).

de Chaisemartin, C. and D'Haultfoeuille, X. (2024). Difference-in-differences
estimators of intertemporal treatment effects. NBER Working Paper 29873.
