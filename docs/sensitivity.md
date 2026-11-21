# Sensitivity to parallel trends

Pre-treatment coefficients that are close to zero do not show that trends
would have stayed parallel after treatment, and a pre-trend test that does not
reject is weak evidence. Rambachan and Roth (2023, *A More Credible Approach
to Parallel Trends*) ask instead how large a departure from parallel trends
the conclusion survives, measuring it against the pre-treatment coefficients.
`fit.trend_sensitivity()` gives confidence intervals for an effect that are
valid when the violation of parallel trends after treatment is at most M times
as large as before it, for the values of M you choose, and the largest M at
which the effect stays significant.

It is the method of the authors' R package
[HonestDiD](https://github.com/asheshrambachan/HonestDiD), with the same
restrictions, intervals and defaults, and gives the same intervals, about a hundred
times faster (the computation is
[hdfe-stream's](https://leetucker.github.io/hdfe-stream/reference/sensitivity.html),
which documents it in full).

```python
from diff_stream import did_callaway_santanna

fit = did_callaway_santanna("lemp ~ 1", "mpdta.parquet", panel=("countyreal", "year"),
                            cohort="first.treat", base_period="universal")
s = fit.trend_sensitivity(M=[0.5, 1, 1.5, 2])
s.table       # M, lower, upper: the robust interval at each M
s.original    # the usual interval, under exact parallel trends
s.breakdown   # the largest M in the grid at which the interval excludes zero
s.plot()      # needs matplotlib
```

## Which fits

The analysis uses the event study: the event-time aggregate of a
Callaway-Sant'Anna, Sun-Abraham or extended TWFE fit (computed for you from
the fit, or pass `fit.aggregate()` yourself), the event-time coefficients of a
two-stage or stacked fit, or the placebos and effects of `did_dcdh`. It needs
at least one pre-treatment coefficient and their joint covariance with the
effects.

- **Callaway-Sant'Anna needs `base_period="universal"`.** With the default
  varying base period, the pre-treatment coefficients are changes from one
  period to the next, not differences from one reference period, which is
  what the restrictions describe (R's `honest_did` refuses them too).
- **BJS is not available.** Its pre-treatment leads (`pretrend_leads=`) come
  from a separate regression with no covariance with the treatment effects.
- **Extended TWFE needs never-treated controls** (its default), with which it
  estimates pre-treatment effects; with `control_group="not_yet_treated"` it
  has none.
- **`did_dcdh`** needs `placebos >= 1`. Placebo l is the event time l periods
  before the reference period (the period before the switch), and effect l is
  l - 1 periods after it, so the default target, event time 0, is effect 1.
- **`did_continuous`** results are not supported.

The reference period is the omitted one (`-anticipation - 1`). With
anticipation, the anticipation periods come after it and count as
post-treatment periods. The event times on each side of the reference must be
consecutive; `horizons=` selects them.

## Options

- **`target`**: the effect bounded. An event time (default 0, the first
  treated period), `"average"` (the mean over every event time after the
  reference), or weights `{event time: weight}`.
- **`restriction`**: `"relative_magnitudes"` (default): no period-to-period
  change in the violation after treatment larger than M times the largest one
  before it. `"smoothness"`: the violation's slope changes by at most M per
  period, so M = 0 allows a linear trend. `"smoothness_relative_magnitudes"`:
  changes in slope after treatment at most M times the largest before it.
- **`bias`** (`"positive"`, `"negative"`) or **`monotonicity`**
  (`"increasing"`, `"decreasing"`): add a sign or shape restriction.
- **`ci_method`**: `"fixed_length"` (R's FLCI), `"conditional"`,
  `"hybrid_fixed_length"` (C-F) or `"hybrid_least_favorable"` (C-LF); R's
  default for each restriction.
- **`M`**: the values of M; by default R's grid for the restriction.
- `grid_points` (1,000), `grid_bounds`, `simulations` (1,000), `seed` (0),
  `n_threads`.

See [hdfe-stream's page](https://leetucker.github.io/hdfe-stream/reference/sensitivity.html)
for what each restriction and interval does, the correspondence with R's
names, and the few places the results differ from R's on purpose.

## Choosing M

The restriction states what you are willing to assume. Relative magnitudes
with M = 1 says the violation after treatment changes no faster than the
fastest change before it; M = 2, twice as fast. Report a range, and the
breakdown value: "the effect is significant unless the post-treatment
violation is more than 1.5 times the largest pre-treatment one" is a statement
a reader can weigh. The R package's
[vignette](https://github.com/asheshrambachan/HonestDiD#readme) discusses the
choice for each restriction.
