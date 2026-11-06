# Quickstart

Install from the source checkout:

```bash
uv venv --python 3.12
uv pip install -e '.[test,docs,benchmark]'
```

Simulate a panel and estimate all three methods:

```python
from diff_stream import bjs_stream, cs_stream, sunab_stream
from diff_stream.simulate import simulate_panel

data = simulate_panel(2000, 8)
for estimator in [bjs_stream, sunab_stream, cs_stream]:
    fit = estimator("y ~ 1", data, panel=("id", "time"), cohort="first_treat")
    fit.aggregate("simple").summary()
```

For data on disk, pass a Parquet path/glob or a Polars LazyFrame. Time must be
consecutive positive integers; first treatment is constant per unit, with 0 for
never treatment. Used values must be finite and non-missing. Exclude
always-treated units explicitly. The default requires one row per unit in every
period. Use `panel_mode="unbalanced"` for observed rows, or `panel_mode="complete"`
to explicitly drop incomplete units. Duplicate keys remain errors.
See [unbalanced panels](unbalanced_panels.md) for identification rules,
R-compatible CS conventions, and BJS temporary-file cleanup.

CS accepts covariates, interactions and categorical coding, for example
`cs_stream("y ~ x + C(region)", data, ..., est_method="dr")`. Choose `reg`,
`ipw`, or `dr`. See [conditional CS](conditional_cs.md) for baseline covariate
dates, overlap diagnostics and trimming. BJS/SA still use `y ~ 1`; richer
fixed-effect specifications remain later work.

All three estimators accept `anticipation=2` for two potentially affected
periods before actual treatment. Their clean sample/reference shifts while
cohort/event-time labels remain unchanged. See [anticipation](anticipation.md)
for eligibility, future cohorts and aggregation conventions.

For BJS small-cohort inference, use `leaveout=True`. The default residual groups
are cohort-time cells; `avgeffectsby="event_time"`, `"cohort"`, or `"all"` pools
residual averaging more broadly. See [inference conventions](implementations.md)
for singleton groups and how aggregation recomputes smart weights.

Use `fit.aggregate("cohort")` or `fit.aggregate("calendar")` for post-treatment
cohort/calendar effects. `fit.aggregate(balance_e=2)` balances post-treatment
cohort support; `balance_window=(0, 2)` selects an explicit event window.
`result.contributions()` shows cell weights and observed/cohort counts;
`result.composition()` summarizes membership and weight changes.
See [aggregation and sample composition](aggregation.md).

For a constrained job, set the scheduler memory/thread environment before import
and choose `preprocess_mb` explicitly, for example `preprocess_mb=16, memory_gb=1.0`.
Smaller partitions require more disk I/O; see [resources](resources.md).

For sampling weights, pass `weights="sampling_weight"` to any estimator.
Weights must be finite, strictly positive and constant within unit by default.
Set `allow_time_varying_weights=True` to enable date-specific weights. They affect
fitting, aggregation and inference. Weighted BJS also retains temporary
aggregation data; call `fit.cleanup()` when finished. See [sampling weights](weights.md).

SA/BJS accept covariates and additional fixed effects, for example
`"y ~ x | region^time"`; unit/time intercepts are always included. See
[richer models](richer_models.md) for the bounded design limit, weighting rules,
clean-placebo support and examples.

For clean pre-trend diagnostics, call `fit.pretrend_test()` on CS/SA results or
`fit.pretrend_test(level="dynamic")` to test event-time averages. Enable CS
multiplier bands with `bootstrap_reps=999, bootstrap_seed=42` at fit time and
call `fit.aggregate().multiplier_confint()`. See [inference](inference.md) for
exact selection, bootstrap scales and reproducibility.

For a diagnostic of conventional TWFE, use `bacon_decomp("y ~ 1", data, ...)`.
It reports Goodman–Bacon comparison estimates, weights and contributions on
balanced panels, including always-treated groups. Pass `weights="column"` for
positive time-invariant unit sampling weights. A formula such as `"y ~ x + z"`
adds covariate adjustment with separate within/between timing components. See
[the decomposition guide](bacon.md) and [extensions](bacon_extensions.md).

Use `twfe_weights("y ~ x", data, panel=("id", "time"), treatment="treated")`
for the published `feTR` weights on underlying treatment effects and sensitivity
thresholds. Set `panel_mode="unbalanced"` for observed-cell panels; binary
treatment can reverse. See [TWFE weights](twfe_weights.md).

BJS supports a separate clean-placebo regression with `pretrends=L`. For example,
`bjs_stream("y ~ 1", data, ..., anticipation=2, pretrends=3)` estimates clean
leads -5, -4 and -3. Inspect `fit.placebos().tidy()` and call
`fit.pretrend_test()`. Main treatment effects are unchanged. See
[BJS clean placebos](bjs_placebos.md).
