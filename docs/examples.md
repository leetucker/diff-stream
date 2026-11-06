# Examples

## Changing the CS comparison population

```python
from diff_stream import cs_stream

fit = cs_stream(
    "y ~ 1",
    "panel.parquet",
    panel=("id", "time"),
    cohort="first_treat",
    control_group="notyettreated",
    base_period="universal",
)
fit.group_time().write_csv("att_gt.csv")
```

Never-treated controls are the default. Not-yet-treated controls must be
untreated at both dates in the comparison, and cannot belong to the target
cohort. Both choices currently require a never-treated cohort in the sample.

## Event windows and post-estimation

```python
dynamic = fit.aggregate(horizons=[-3, -2, 0, 1, 2])
dynamic.tidy()
dynamic.confint(joint=True, seed=42)
dynamic.wald_test(keep="event::-", distribution="chi2")
dynamic.summary_json("event_study.json", coefficients=True)
overall = fit.aggregate("simple")
```

Selecting horizons aggregates existing cells; it does not drop interactions
from the fitted SA specification. Cohort composition can change across event
times. BJS returns affected-period effects, including requested anticipation periods;
its clean placebos use a separate regression requested with `pretrends=L`. CS/SA provide
`pretrend_test()` with explicit clean-period selection.
CS varying-base pre-treatment effects measure changes from t-1, while
universal-base pre-treatment effects measure changes from g-1.

For reporting with pyfixest, install `hdfe-stream[formula]` and use the family's
`etable` adapter. The core coefficient tables and JSON export need no pyfixest.
`examples/quickstart.py` is a standalone executable example.
`examples/inference.py` demonstrates CS multiplier bands and clean pre-trend tests.

`examples/bacon.py` demonstrates the Goodman–Bacon decomposition and summaries
by comparison type. See [Goodman–Bacon](bacon.md) for supported samples and how
its weighted contributions reconstruct conventional TWFE.

`examples/bacon_covariates.py` decomposes the public bacondecomp Castle example
with log population/log income controls, displaying the within-timing and
full-period dyad components. See [Bacon extensions](bacon_extensions.md).

`examples/twfe_weights.py` reproduces the authors' public wage-panel `feTR`
example with union membership, weight signs and both heterogeneity sensitivity
thresholds. See [TWFE weights](twfe_weights.md).

`examples/reference_populations.py` demonstrates BJS, Gardner and ETWFE on an
all-treated panel, including original-onset BJS placebos and the explicit clean
last-treated window. See [reference populations](reference_groups.md).

`examples/bjs_placebos.py` demonstrates the separate BJS clean-lead result,
pre-trend testing and result availability after cleanup. See
[BJS placebos](bjs_placebos.md).

The [Castle Doctrine worked example](castle_replication.md) supplies a real-data
Sun–Abraham reanalysis, with `python examples/castle.py` as the runnable entry
point and a complete frozen numerical audit in `benchmarks/castle_replication.py`.

`python examples/richer_models.py` demonstrates date-specific weights, richer
SA/BJS formulas and clean placebos using the frozen numerical-validation sample.
See [richer models](richer_models.md) for conventions and backend limits.
