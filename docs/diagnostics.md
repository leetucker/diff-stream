# Diagnostics of a two-way fixed effects estimate

A conventional difference-in-differences regression, the outcome on a
treatment dummy with unit and time fixed effects, is a weighted average of
comparisons. With staggered timing some of them use already-treated units as
controls, and with effects that change over time those comparisons, or
negative weights on some treated cells, can move the estimate far from any
average effect. The two diagnostics show what a given estimate averages. Neither
estimates a causal effect or has standard errors.

## Goodman-Bacon decomposition

`bacon_decomp` writes the coefficient as a weighted average of every
two-group, two-period comparison: treated against never-treated, earlier- against
later-treated, and later- against earlier-treated (the comparisons with an
already-treated control).

```python
from diff_stream import bacon_decomp

bacon = bacon_decomp("y ~ 1", "panel.parquet", panel=("id", "time"), cohort="first_treat")
bacon.twfe_estimate          # the coefficient
bacon.summary()              # estimate, weight and contribution by comparison type
bacon.comparisons()          # every pair of timing groups
bacon.reconstruction_error   # weighted average minus the coefficient (rounding only)
```

- Treatment is built from `cohort` (treated from the cohort's period on).
  Units treated before the first period form an always-treated group; a
  never-treated group is not required.
- Covariates, `"y ~ x + z"`, give the covariate-adjusted decomposition: a
  within-timing component and comparisons between timing groups, which
  together recover the adjusted coefficient (`bacon.within_timing()`).
- `weights=` (constant within unit) decomposes the weighted regression.
- The panel must be balanced (`incomplete_units="drop"` drops incomplete units).

Validated against R's bacondecomp, with and without covariates and weights,
including its documented `math_reform` example (all 16 comparisons) and the
Castle Doctrine panel. Goodman-Bacon, A. (2021). *Journal of Econometrics* 225(2),
254-277.

## Treatment-effect weights

`twfe_weights` writes the coefficient as a weighted sum of the effects in each
treated cohort-period (or unit-period) cell, and reports the weights,
some possibly negative, and two sensitivity measures: how much the cell effects
would have to vary for the estimate to have the opposite sign from every one
of them.

```python
from diff_stream import twfe_weights

with twfe_weights(
    "y ~ 1", "panel.parquet", panel=("id", "time"), treatment="treated", keep_cells=True
) as diagnostic:
    diagnostic.twfe_estimate
    diagnostic.summary()       # positive and negative weights, their sums
    diagnostic.sensitivity()   # the two thresholds
    diagnostic.cells()         # every treated cell's weight (kept on disk)
```

- Give exactly one of `treatment=` (a 0/1 column, which may switch on and off)
  or `cohort=`.
- Covariates are allowed; unbalanced panels (`incomplete_units="keep"`) use the
  observed cells; times may have gaps.
- `keep_cells=True` keeps the cell weights on disk for `cells()`,
  `iter_cells()` and `write_cells(path)`, until `cleanup()`.

This is the authors' `feTR` diagnostic, validated against their R package
TwoWayFEWeights (`type="feTR"`) and the wage-panel example.
de Chaisemartin, C. and D'Haultfoeuille, X. (2020). *American Economic Review*
110(9), 2964-2996.
