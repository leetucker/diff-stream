# diff-stream

Disk-backed staggered difference-in-differences: Borusyak–Jaravel–Spiess
imputation, Sun–Abraham saturated interactions, and covariate-adjusted
Callaway–Sant'Anna, with unit-clustered inference and a common group-time result. Gardner two-stage DiD adds direct
static/event-study coefficients with first-stage-adjusted inference. Linear
ETWFE supplies saturated covariate interactions and conditional/unconditional
margin inference; see [ETWFE](etwfe.md) for nonlinear alternatives.

**Development milestone, not a public release.** The supported design is a
balanced or explicitly unbalanced absorbing-treatment panel, with optional positive
sampling weights and consecutive positive integer times. All five estimators can use a
[clean last-treated reference window](reference_groups.md) when no never-treated
cohort is available, with estimator-specific lead conventions. CS supports outcome regression,
IPW and doubly robust covariate adjustment; SA/BJS support covariates and bounded
additional fixed-effect designs. Explicit opt-in enables time-varying weights.
Unsupported designs raise errors; see [richer models](richer_models.md) for scope.

The [release-preparation handoff](release_handoff.md) records completed scope,
remaining extension decisions and explicit deferrals, including simultaneous
refit-bootstrap bands.

```python
from diff_stream import cs_stream

fit = cs_stream("y ~ 1", "data/*.parquet", panel=("id", "time"), cohort="first_treat")
fit.aggregate().summary()
```

A separate [Goodman–Bacon diagnostic](bacon.md) decomposes conventional TWFE
on balanced panels with optional covariates, always-treated groups and unit sampling weights. BJS also supports a separate
[clean-placebo regression and pre-trend test](bjs_placebos.md).
The separate [TWFE treatment-effect-weight diagnostic](twfe_weights.md) reports
negative weights and heterogeneity sensitivity measures for binary treatment,
including reversing treatment and observed-cell unbalanced panels.

The implementation shares hdfe-stream's result and resource-profile interfaces
and stream-limits' scheduler-aware thread and memory machinery. It snapshots rows to Parquet, validates and sorts bounded whole-unit
hash partitions, and reads whole-unit batches for inference. Polars operations explicitly request the streaming engine on Polars 1.x.

```{toctree}
:hidden:
:caption: Guide

quickstart
conditional_cs
anticipation
weights
richer_models
effective_sample
inference
refit_bootstrap
bjs_placebos
mpdta_replication
math_reform_replication
castle_replication
bacon
bacon_extensions
twfe_weights
aggregation
unbalanced_panels
reference_groups
reference_population_review
release_handoff
covariate_balance
examples
how_it_works
resources
resource_checks
memory_checks
kernel_memory_checks
shared_core_review
numerical_stress
implementations
gardner
etwfe
stata_validation
```

```{toctree}
:hidden:
:caption: Benchmarks

benchmarks/index
```

```{toctree}
:hidden:
:caption: Reference

reference/index
```
