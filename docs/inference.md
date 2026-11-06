# Pre-trend tests and CS multiplier bands

All five estimators also support a separate
[unit-cluster full-refit bootstrap](refit_bootstrap.md) with pointwise SEs and
percentile, basic or normal intervals. It resamples unit histories and refits
nuisance models; the CS multiplier machinery below remains unchanged.

`pretrend_test()` adds explicit clean-period selection to the family's analytic
Wald test. CS also supports unit-level influence-function multiplier intervals
when requested at fit time. Gaussian joint intervals through `confint(joint=True)`
remain available for all estimators.

## Clean pre-treatment tests

```python
test = fit.pretrend_test()
print(test["statistic"], test["df1"], test["pvalue"])
print(test["coefficients"], test["cells"], test["horizons"])

# Aggregate clean leads before testing them jointly:
fit.pretrend_test(level="dynamic", horizons=[-5, -4])

# Or test clean leads of an existing event study:
fit.aggregate(balance_window=(-5, 1)).pretrend_test()
```

On original CS/SA results, the default jointly tests supported group-time cells.
`level="dynamic"` instead tests event-time averages, including CS estimated-share
uncertainty. On an existing dynamic result, the default tests its clean horizons.
Post-treatment simple/cohort/calendar summaries cannot supply pre-trend tests.

For anticipation $A$, only estimated events $t-g < -A$ qualify. Anticipation
periods $-A,\ldots,-1$ and actual post-treatment periods are excluded. Omitted
normalized references are not zero-valued restrictions. In particular,
CS varying-base can estimate an effect at $-A-1$ (a successive-period change),
so that event remains eligible. Universal CS and SA omit that reference.
Requested `horizons` must be clean integers with estimated support; invalid or
unavailable requests raise rather than silently changing the hypothesis.

The test uses the analytic covariance and a chi-squared reference with one
degree of freedom per selected restriction. Singular/ill-conditioned covariance,
zero variance and absent clean leads raise explicit errors. A smaller horizon
selection or dynamic aggregation may yield an estimable test. The returned
dictionary identifies the exact restrictions, contributing cells, horizons,
anticipation and base convention. A failure to reject does not establish
parallel trends. For BJS, request `pretrends=L` at fit time and use
`fit.placebos()` or `fit.pretrend_test()` for the separate pooled clean-lead
regression. Its anticipation cells are not clean placebos; see
[BJS placebos](bjs_placebos.md) for reference and inference conventions.

## CS unit multiplier intervals

```python
from diff_stream import cs_stream

fit = cs_stream(
    "y ~ x + z",
    "panel.parquet",
    panel=("id", "time"),
    cohort="first_treat",
    panel_mode="unbalanced",
    weights="sampling_weight",
    anticipation=2,
    bootstrap_reps=999,
    bootstrap_seed=42,
)
dynamic = fit.aggregate()
dynamic.multiplier_confint()  # simultaneous over all returned horizons
dynamic.multiplier_confint(keep=["event::-5", "event::-4"], exact_match=True)
dynamic.multiplier_confint(joint=False)  # pointwise multiplier critical values
```

The fit computes nuisance-adjusted unit scores as usual and accumulates
$\Delta_b=\sum_i \xi_{ib}\phi_i$. Observed-row CS folds scores by original unit
first, so every observation of the same unit receives the same multiplier.
Sampling weights enter the scores; multiplier probabilities do not depend on
those sampling weights. Units with only one comparison-date observation still
contribute under the native unbalanced convention.

The same unit multiplier is applied to cohort-share scores. For aggregation
maps $A$ and share derivatives $B$, the saved perturbation is
$\Delta_b A' + \Delta^{share}_b B'$. Dynamic, simple, cohort, calendar and balanced
event-window aggregates therefore include the same share uncertainty as the
analytic covariance. Aggregation and band calculations need no source reread,
refit or retained temporary files.

The default `bootstrap_weights="rademacher"` uses independent signs, matching the
distribution used by R `did`/BMisc. `"mammen"`, `"webb"` and `"norm"` are also
available through hdfe-stream's existing samplers. Every requested repetition is
drawn; small cluster counts do not trigger sign-pattern enumeration.
Seeds use NumPy, so the same numeric seed does not reproduce R's RNG stream.

`multiplier_confint()` defaults to bootstrap studentization: each scale is its
draw IQR divided by the standard-normal IQR. Critical values are empirical
type-1 quantiles of the largest absolute standardized perturbation, as in
[`did::mboot`](https://bcallaway11.github.io/did/reference/mboot.html).
The `keep`/`drop` selection defines the exact simultaneous family. Pointwise
intervals use each coefficient's own critical value. Zero/non-finite scales
raise; increase repetitions or select supported coefficients.

Use `studentization="analytic"` to divide by stored analytic SEs instead.
Neither option replaces `fit.vcov`, `fit.se`, `tidy()` or inherited `confint()`.
The multiplier table reports its own `Std. Error`, `critical_value`, and bounds.
CRV1 scales perturbations by $\sqrt{N/(N-1)}$, matching the analytic correction.
`multiplier_draws()` returns a copy of the centered estimate-scale perturbations
(repetitions by coefficients), with the selected CRV scaling.

`bootstrap_reps=0` (default) skips this work and retains no multiplier state.
At least two repetitions are required when enabled; use substantially more for
tail inference. The feature currently applies to CS only. These are
influence-function multiplier bands, without resampling/refitting the data or
using restricted-OLS wild-bootstrap t statistics.

## Reused infrastructure, memory and reproducibility

The samplers are imported from `hdfe_stream.core.wildboot.WEIGHTS`. Its restricted
OLS test engine uses per-cluster regression Hessians and re-studentizes bootstrap
fits, which differ from CS influence-function bands. The added
`diff_stream._core.multiplier` reducer supplies only generic bounded score sums
and interval scaling; it is a candidate for future `hdfe_stream.core` promotion.
No sibling source or immutable copied core utility is modified.

The reducer stores repetitions-by-effects and repetitions-by-share-score arrays.
Multiplier buffers are limited to 262,144 entries, or one unit's repetitions
when that is larger. It retains no full units-by-effects, units-by-repetitions
or global ID table. Fit/profile guards include draw state and working products.
Memory grows with repetitions and effects; bootstrap computation still grows
with units, repetitions and effects.

Draws are assigned in staged whole-unit traversal order. A fixed seed and
partition layout reproduce results, including across numerical batch sizes.
Changing preprocessing partitions, input representation/order, or Polars hashing
can change assignments to units and the finite simulation result. It preserves
the multiplier distribution; exact equality across arbitrary layouts is not
promised. Record these settings with the seed and repetition count.

## Validation and resource checks

Common Rademacher draws validate every saved perturbation and interval against
R `did` influence functions and its actual `mboot` scaling/critical-value code.
The 56 scenarios cover balanced/native unbalanced panels, weights on/off,
anticipation 0/2, intercept-only and conditional CS, regression/IPW/DR, both
control populations and both base conventions. All aggregation kinds are
checked. R pre-test references explicitly select clean cells and use its analytic
covariance. The anticipation varying-base reference adjustment is documented in
the [anticipation guide](anticipation.md).

Only the reference R process's draw generator is replaced to supply common
multipliers; installed package files and the inference calculations are
unchanged. Regenerate from the repository root with:

```bash
.venv/bin/python benchmarks/external/inference_multipliers.py tests/data/inference_reference
Rscript benchmarks/external/inference_reference.R \
  tests/data/anticipation_panel.csv tests/data/inference_reference
```

Core checks compare all four samplers directly with hdfe-stream, exercise buffer
splitting and batch invariance, and verify quantiles, family selection, CRV1,
complete-panel deletion, disabled inference and guarded allocations.

Run the offline resource checks with:

```bash
STREAM_LIMITS_MAX_THREADS=2 POLARS_MAX_THREADS=2 \
  .venv/bin/python benchmarks/inference_resources.py
```

The weighted conditional-CS checks use eight periods, anticipation 2, native
unbalanced rows, two threads and a 1 GB RSS watchdog. Peaks were 393.7 MiB at
10,000 units/999 draws, 502.7 MiB at 100,000 units/999 draws, and 509.5 MiB at
100,000 units/1,999 draws. Saved score/share draw state was about 0.32/0.64 MiB
for 999/1,999 repetitions, independent of unit count at this effect dimension.
Fits, balanced-window aggregation, bands, pre-tests and scratch cleanup all
completed. Raw results live in `benchmarks/results/inference_resources/`.
