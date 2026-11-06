# Estimators

| Function | Estimator | Result |
|---|---|---|
| [`did_callaway_santanna`](callaway_santanna.md) | Callaway and Sant'Anna (2021): group-time effects, with regression, IPW or doubly robust covariate adjustment | `DIDResult` |
| [`did_sun_abraham`](sun_abraham.md) | Sun and Abraham (2021): interaction-weighted event study | `DIDResult` |
| [`did_bjs`](bjs.md) | Borusyak, Jaravel and Spiess (2024): imputation | `DIDResult` |
| [`did_two_stage`](two_stage.md) | Gardner (2022): two-stage DiD | `TwoStageResult` |
| [`did_extended_twfe`](extended_twfe.md) | Wooldridge (2021): extended two-way fixed effects (linear) | `DIDResult` |
| [`did_stacked`](stacked.md) | Cengiz et al. (2019), Wing, Freedman and Hollingsworth (2024): stacked DiD with corrective weights | `StackedResult` |

All six share the data arguments (`panel`, `cohort`, `incomplete_units`,
`weights`, `anticipation`, `last_treated_controls`; see [Your data](../data.md)),
the standard errors (clustered by unit, `vcov="CRV0"` or `"CRV1"`) and the
memory and thread options (see [Memory and threads](../resources.md)). The
four that return a `DIDResult` estimate one effect per cohort and period and
share its [aggregations](../results.md); the two-stage and stacked estimators
return their event-time coefficients directly.

```{toctree}
:hidden:

callaway_santanna
sun_abraham
bjs
two_stage
extended_twfe
stacked
```
