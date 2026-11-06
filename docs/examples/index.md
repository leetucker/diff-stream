# Examples

Each page shows a script from the repository's `examples/` directory and its
output. Run the scripts from the repository's root:

```bash
python examples/quickstart.py
```

`python docs/examples/refresh.py` reruns them all and updates these pages.

| Page | |
|---|---|
| [Quickstart](quickstart.md) | Callaway-Sant'Anna, Sun-Abraham and BJS on a simulated panel of 2,000 units over 8 periods, whose effects grow with time since treatment and differ by cohort. |
| [Covariates in Callaway-Sant'Anna](covariates.md) | Outcome regression, inverse probability weighting and the doubly robust estimator, with two covariates. |
| [Aggregation and its composition](aggregation.md) | Cohort and calendar aggregates of a Callaway-Sant'Anna fit on an unbalanced panel, with the cells and weights behind each estimate, then an event study restricted to the cohorts observed at event times 0 and 1. |
| [Anticipation](anticipation.md) | Units may respond two periods before treatment (`anticipation=2`): only periods before `cohort - 2` count as untreated. |
| [Sampling weights](weights.md) | The three estimators with a sampling weight per unit. |
| [Covariates and fixed effects in Sun-Abraham and BJS](fixed_effects.md) | Time-varying weights on an unbalanced panel; covariates in Callaway-Sant'Anna; covariates and region-by-time fixed effects in Sun-Abraham and BJS, with BJS's pre-trend test. |
| [When every unit is eventually treated](all_treated.md) | No unit is never treated, so the last-treated cohort serves as the comparison until it is treated (`last_treated_controls="if_needed"`, with a warning). |
| [BJS pre-trend test](bjs_pretrends.md) | `pretrend_leads=1` fits a separate regression on the untreated observations with an indicator for the period before treatment, and tests it. |
| [Simultaneous bands](inference.md) | Callaway-Sant'Anna's multiplier bootstrap: a simultaneous 95% band over the event study, then a pre-trend test of the event-time averages. |
| [Two-stage DiD on the Castle Doctrine panel](two_stage.md) | Gardner's estimator on Cheng and Hoekstra's state panel of homicide rates, with treatment starting the year after a state's castle doctrine law: the overall ATT, an event-study window and a test of two leads. |
| [Extended TWFE on mpdta](extended_twfe.md) | Wooldridge's estimator with not-yet-treated controls and a covariate (log county population) on R did's minimum-wage example. |
| [Sun-Abraham on the Castle Doctrine panel](castle.md) | The event study, the overall ATT, a pre-trend test of three event-time averages, and the composition of an event study balanced over event times 0 to 2. |
| [Goodman-Bacon decomposition](bacon.md) | The decomposition of a two-way fixed effects estimate on a simulated panel, the same with sampling weights, and the covariate-adjusted decomposition on the Castle Doctrine panel. |
| [TWFE treatment-effect weights](twfe_weights.md) | The weights behind the union wage premium in the authors' wage panel, where union membership switches on and off: their sums, the sensitivity measures, and the cells with negative weights. |

The simulated panels come from `diff_stream.simulate`: `simulate_panel` (no
covariates) and `simulate_conditional_panel` (two covariates that shift both
treatment timing and trends). The public data sets are frozen in
`tests/data/`: the Castle Doctrine panel (Cheng and Hoekstra 2013, as
distributed with R's did2s), `mpdta` (a subset of Callaway and Sant'Anna's
2021 county data, built from the Census Bureau's Quarterly Workforce
Indicators, as distributed with R's did), and the Vella and Verbeek (1998)
wage panel (as distributed with TwoWayFEWeights). Full citations are in
`tests/data/DATA_SOURCES.md`.

```{toctree}
:hidden:

quickstart
covariates
aggregation
anticipation
weights
fixed_effects
all_treated
bjs_pretrends
inference
two_stage
extended_twfe
castle
bacon
twfe_weights
```
