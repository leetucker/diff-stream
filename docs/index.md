# diff-stream

Difference-in-differences with staggered treatment timing, on panels larger
than memory. Six estimators share one interface: Callaway and Sant'Anna's
group-time effects, Sun and Abraham's interaction-weighted event study,
Borusyak, Jaravel and Spiess's imputation estimator, Gardner's two-stage DiD,
Wooldridge's extended two-way fixed effects, and stacked DiD with Wing,
Freedman and Hollingsworth's corrective weights. Two diagnostics show what a
conventional two-way fixed effects regression averages: the Goodman-Bacon
decomposition and de Chaisemartin and D'Haultfoeuille's weights. Results are
tested against the authors' R and Stata packages.

```python
from diff_stream import did_callaway_santanna

fit = did_callaway_santanna(
    "earnings ~ age + C(region)",
    "panel/*.parquet",
    panel=("person", "year"),
    cohort="first_treated_year",
)
fit.aggregate().summary()           # event study
fit.aggregate("overall").summary()  # overall ATT
```

## When to use this, and when not to

Use diff-stream when the panel is large (tens of millions of rows and up, or
more than fits in memory) and treatment is staggered and absorbing: units
start treatment at different times and stay treated. In particular:

- an event study or overall ATT that is robust to effects varying across
  cohorts and over time, by any of six estimators, with the same results
  interface and aggregations;
- covariates, by outcome regression, inverse probability weighting or doubly
  robust estimation (Callaway-Sant'Anna), or additional fixed effects such as
  region-by-year (Sun-Abraham, BJS, two-stage), with any number of levels
  for Sun-Abraham and BJS (county-by-year, absorbed by hdfe-stream);
- sampling weights, unbalanced panels, anticipation, and panels where every
  unit is eventually treated;
- standard errors clustered by unit that include the uncertainty in the cohort
  shares used to aggregate, simultaneous confidence bands, and a bootstrap
  that refits the whole estimator;
- jobs on a cluster scheduler, with memory and thread limits taken from the
  allocation.

Use something else when:

- **The panel is small.** Below a few million rows, R's
  [did](https://bcallaway11.github.io/did/),
  [fixest](https://lrberge.github.io/fixest/) (`sunab`),
  [didimputation](https://github.com/kylebutts/didimputation),
  [did2s](https://github.com/kylebutts/did2s) and
  [etwfe](https://grantmcdermott.com/etwfe/), and Stata's `csdid`,
  `did_imputation`, `did2s` and `jwdid`, are just as fast, and their
  ecosystems (plots, sensitivity analysis) are larger.
- **Treatment switches on and off, or is continuous.** These estimators need
  absorbing binary treatment. de Chaisemartin and D'Haultfoeuille's
  `did_multiplegt_dyn` (R, Stata) covers those designs. `twfe_weights()` here
  accepts switching treatment, as a diagnostic.
- **The outcome needs a nonlinear model** (logit, Poisson ETWFE): use R's
  etwfe or Stata's `jwdid`.
- **You need sensitivity analysis for parallel trends.** Pass the event-study
  coefficients and covariance (`fit.aggregate().beta`, `.vcov`) to R's
  [HonestDiD](https://github.com/asheshrambachan/HonestDiD).

See [Choosing an estimator](choosing.md) for how the five differ, and
[Differences from other packages](implementations.md) for their conventions.

## How it is fast

- **Sorted once.** The panel is sorted by unit in memory when it fits, or in
  whole-unit pieces on disk when it does not. Every later pass reads whole
  units in large batches.
- **Small sufficient statistics.** Without covariates, the estimators reduce
  to cohort-by-period means plus unit scores, computed in parallel.
- **Covariance by cohort.** A unit's scores touch only the cells its cohort
  enters, so the joint covariance of hundreds of effects is accumulated
  cohort by cohort.
- **Shared passes.** The propensity and outcome models of all comparisons are
  fitted together, one pass over the data per Newton step.

See [How it works](how_it_works.md), and [Memory and threads](resources.md)
for the limits.

## Install

```bash
pip install diff-stream
```

(Not on PyPI yet; install from the repository with `pip install -e .`.)

```{toctree}
:hidden:
:caption: Getting started

quickstart
choosing
```

```{toctree}
:hidden:
:caption: Guide

data
estimators/index
results
inference
diagnostics
examples/index
```

```{toctree}
:hidden:
:caption: Background

how_it_works
resources
implementations
validation
benchmarks/index
```

```{toctree}
:hidden:
:caption: Reference

reference/index
changelog
```
