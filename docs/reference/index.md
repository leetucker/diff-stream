# Function reference

Everything here is importable from `diff_stream`.

| Name | What it is |
|---|---|
| **[Estimators](estimation.md)** | |
| `did_callaway_santanna` | Callaway and Sant'Anna's group-time effects |
| `did_sun_abraham` | Sun and Abraham's interaction-weighted event study |
| `did_bjs` | Borusyak, Jaravel and Spiess's imputation estimator |
| `did_two_stage` | Gardner's two-stage DiD |
| `did_extended_twfe` | Wooldridge's extended two-way fixed effects (linear) |
| `did_stacked` | stacked DiD with corrective weights |
| **[Diagnostics and bootstrap](diagnostics.md)** | |
| `bacon_decomp` | the Goodman-Bacon decomposition of a TWFE estimate |
| `twfe_weights` | de Chaisemartin and D'Haultfoeuille's weights behind a TWFE estimate |
| `refit_bootstrap` | standard errors and intervals from refits on unit resamples |
| **[Results](results.md)** | |
| `DIDResult` | cohort-period effects, their aggregates and tests |
| `TwoStageResult` | the two-stage estimator's coefficients |
| `StackedResult` | the stacked estimator's coefficients |
| `BJSPlaceboResult` | the BJS pre-trend leads |
| `BaconResult`, `TWFEWeightResult`, `RefitBootstrapResult` | the diagnostics' and the bootstrap's results |
| `LowEffectiveSampleWarning`, `ReferenceWindowWarning`, `UnimputedObservationsWarning`, `BootstrapFailureWarning` | warnings |
| **[Utilities](utilities.md)** | |
| `simulate_panel`, `simulate_conditional_panel` | simulated panels with known effects (`diff_stream.simulate`) |
| `etable`, `limit_threads`, `Profile` | from hdfe-stream and stream-limits |

```{toctree}
:hidden:

estimation
diagnostics
results
utilities
```
