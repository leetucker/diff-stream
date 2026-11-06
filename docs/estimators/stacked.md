# Stacked DiD

`did_stacked` builds one sub-experiment per adoption cohort: the cohort's
units, and the units not yet treated by the end of its event window
(never-treated units and later cohorts), over that window. Stacking the
sub-experiments and fitting an event study on the stack compares each cohort
only with clean controls. With the corrective weights of Wing, Freedman and
Hollingsworth (2024), each event-time coefficient is the average of the
sub-experiments' effects, weighted by their treated units.

```python
from diff_stream import did_stacked

fit = did_stacked(
    "y ~ 1", "panel.parquet", panel=("id", "year"), cohort="first_treat",
    event_window=(-4, 4),
)
fit.tidy()                      # event::-4 ... event::4, without the reference -1
fit.overall().summary()         # the average of the post-treatment effects
fit.pretrend_test()             # the leads jointly zero
fit.sub_experiments()           # the cohorts stacked, their units and weights
```

## The stack

- **`event_window=(first, last)`** sets each sub-experiment's event times; it
  must include the reference period (`-anticipation - 1`) and 0. A cohort is
  included only if its whole window lies inside the sample (the "trimmed"
  aggregate), and if it has at least one clean control; `diagnostics
  ["excluded_cohorts"]` lists the others and why.
- **Controls** are units with no treatment through the end of the window
  (before `cohort - anticipation`): a later cohort can be a control for an
  earlier one.
- **Weights**: treated rows weigh 1, a sub-experiment's controls
  (N^D_a / N^D) / (N^C_a / N^C), its share of the stack's treated units over its
  share of the controls, times any sampling weights.
- **Standard errors** are clustered by unit: a control unit that enters several
  sub-experiments is one cluster.

The stacked data repeat a control unit once for each sub-experiment it enters,
so they can be many times the size of the panel. They are never built: the
regression is saturated, so its coefficients are differences of weighted means,
which two passes over the panel compute, and each unit's contribution to the
covariance is summed over its sub-experiments.

Covariates are not supported. The result is a `StackedResult`, with the
methods of the two-stage result (`select`, `support`, `pretrend_test`) and
`overall()` and `sub_experiments()`.

## Validated against

The stacked data built explicitly as in the authors' R code and fitted with
fixest (`feols` with the corrective weights, clustered by unit), balanced,
unbalanced, weighted and with anticipation; and a dense stacked regression in
the test suite. See [Validation](../validation.md).

Cengiz, D., Dube, A., Lindner, A. and Zipperer, B. (2019). The effect of
minimum wages on low-wage jobs. *Quarterly Journal of Economics* 134(3),
1405-1454.
Wing, C., Freedman, S. M. and Hollingsworth, A. (2024). Stacked
difference-in-differences. NBER Working Paper 32054.
