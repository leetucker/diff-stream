# diff-stream

Difference-in-differences with staggered treatment timing, on panels larger than
memory. Includes Callaway and Sant'Anna, Sun and Abraham, Borusyak, Jaravel and
Spiess imputation, Gardner's two-stage DiD, Wooldridge's extended two-way fixed
effects, and stacked DiD. Two diagnostics show what a conventional two-way fixed
effects estimate averages. Results are tested against the authors' R and Stata
packages.

It reads Parquet in a stream, sorts the panel once (in memory when it fits,
in whole-unit pieces on disk when it does not), and computes standard errors
clustered by unit without holding all units' scores.

```python
from diff_stream import did_callaway_santanna

fit = did_callaway_santanna(
    "earnings ~ age + C(region)",
    "panel/*.parquet",
    panel=("person", "year"),
    cohort="first_treated_year",
    control_group="not_yet_treated",
)
fit.aggregate().summary()           # event study
fit.aggregate("overall").summary()  # overall ATT
```

**Status: in development** (version 0.1.0.dev0, not yet on PyPI). The
estimators, inference and documentation are complete and tested; the
benchmark suites wait for a full run.

## What it does

**Estimators** (each cites its paper in its docstring)
- `did_callaway_santanna`: group-time effects, with outcome regression, IPW
  or doubly robust covariate adjustment; never-treated or not-yet-treated
  controls; varying or universal base periods; panels or repeated cross
  sections.
- `did_sun_abraham`: the interaction-weighted event study, with covariates
  and additional fixed effects.
- `did_bjs`: imputation, with the authors' standard errors (residual
  averaging groups, leave-out) and their pre-trend test.
- `did_two_stage`: Gardner's two-stage DiD, event study or overall ATT.
- `did_extended_twfe`: Wooldridge's extended TWFE (linear), with
  covariate-interacted effects.
- `did_stacked`: stacked DiD with corrective weights, without building the
  stack.

**Data**
- Unbalanced panels, sampling weights (also time-varying), anticipation, and
  panels where every unit is eventually treated (the last-treated cohort as
  comparison).

**Results and inference**
- Aggregation by event time, cohort and calendar period, or overall, with
  balanced event windows and the cells and cohorts behind every estimate.
- Standard errors clustered by unit, including the uncertainty in the cohort
  shares used to aggregate; pre-trend tests.
- Callaway-Sant'Anna's multiplier bootstrap for simultaneous bands, and a
  bootstrap that refits any estimator on unit resamples, in parallel.

**Diagnostics**
- `bacon_decomp`: the Goodman-Bacon decomposition (with covariates and
  weights).
- `twfe_weights`: de Chaisemartin and D'Haultfoeuille's weights and
  sensitivity measures.

**Resources**
- `n_threads=` and `max_memory_gb=` respect a job's limits (stream-limits);
  `profile=True` reports the memory and disk a fit needs, without fitting.
- Results with hdfe-stream's reporting: `summary()`, `tidy()`, `confint()`,
  `wald_test()`, `etable`.

## Why it is fast

- **Sorted once.** The panel is sorted by unit in one piece when it fits in
  memory, in whole-unit pieces on disk otherwise; later passes read whole units
  in large batches.
- **Sufficient statistics.** Without covariates, the estimators reduce to
  cohort-by-period means plus unit scores computed in parallel.
- **Covariance by cohort.** A unit's scores touch only its cohort's cells, so
  the joint covariance of hundreds of effects is accumulated cohort by cohort.
- **Shared passes.** All comparisons' propensity and outcome models are
  fitted together.

On a simulated panel of one million units over eight periods (8 million
rows), each estimator fits in one to six seconds on four threads of a laptop, and within
1 GB of memory when limited to it.

## Validation against other implementations

| Implementation | Compared on |
|---|---|
| R did, DRDID | Callaway-Sant'Anna: effects, full covariance, aggregations, multiplier bootstrap; the `mpdta` example |
| R fixest (`sunab`) | Sun-Abraham: coefficients, covariance, aggregation; the Castle Doctrine panel |
| Stata `did_imputation` (authors), R didimputation | BJS: estimates and full covariance, every averaging group, leave-out, pre-trend leads |
| R did2s (authors) | two-stage: coefficients and covariance |
| R etwfe, emfx (authors) | extended TWFE: effects and the covariance of every aggregation |
| fixest on the stacked data | stacked DiD: coefficients and covariance |
| R bacondecomp | Goodman-Bacon: every comparison; the `math_reform` example |
| R TwoWayFEWeights (authors) | TWFE weights (`feTR`) and sensitivity measures |

Agreement is within 2e-7 absolute, usually 1e-10. The frozen references and
the R and Stata scripts that make them are in `tests/data/` and
`benchmarks/external/`; `docs/implementations.md` lists the conventions on
which these packages differ.

## Install

diff-stream is not on PyPI yet. Install it from the repository:

```bash
pip install git+https://github.com/leetucker/diff-stream
```

Python 3.10 or later. It depends on hdfe-stream (with its formula extra),
stream-limits, Polars, NumPy 2, numba, SciPy and pyarrow.

## Documentation

Build the documentation locally with:

```bash
pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs docs/_build/html
```

It covers choosing an estimator, your data, each estimator, results and
inference, the diagnostics, worked examples with their output, how it works,
memory and threads, the conventions that differ from other packages, and the
validation. `examples/` has runnable scripts, and `benchmarks/` the
performance comparisons (see `benchmarks/README.md`).

A sibling of [hdfe-stream](https://github.com/leetucker/hdfe-stream),
[logit-stream](https://github.com/leetucker/logit-stream),
[dpd-stream](https://github.com/leetucker/dpd-stream) and
[match-stream](https://github.com/leetucker/match-stream).

## License

MIT.
