# Unbalanced panels

All three estimators accept missing **rows** through an explicit sample policy:

| `panel_mode` | Sample and computation |
|---|---|
| `"balanced"` (default) | Require exactly one row per unit in every sample period; raise on incomplete units. |
| `"complete"` | Drop every incomplete unit, then fit the balanced estimator. |
| `"unbalanced"` | Retain observed rows and use the estimator-specific methods below. |

```python
fit = cs_stream(
    "y ~ x + z",
    data,
    panel=("id", "time"),
    cohort="first_treat",
    panel_mode="unbalanced",
    est_method="dr",
)
fit.aggregate("simple").summary()
print(fit.diagnostics["dropped_units"])
```

Missing/null/non-finite values in used columns, duplicate unit-time keys and
changing cohort labels remain errors. The global time grid must still consist
of consecutive positive integers. Units may have gaps, enter late or leave
early. Treatment remains absorbing; zero denotes never treated. Always-treated
cohorts are still unsupported. Balanced input uses the existing balanced fast
path, including when `panel_mode="unbalanced"` is requested.

Diagnostics include input units/rows, dropped units/rows, whether the retained
sample is balanced, and omitted unsupported cells. `n_obs` reports retained
observations, rather than units multiplied by periods. Complete-panel selection
can change the target population; it is never performed implicitly.

## Callaway–Sant'Anna

Native unbalanced CS matches R `did::att_gt(allow_unbalanced_panel=TRUE)`:
for each comparison it uses all eligible observed rows at the target and base
dates, with covariates measured on each row's actual date. A unit observed at
only one date can contribute. This is the repeated-cross-section functional
used by R for unbalanced panels, with row influences **summed within original
unit** before computing covariance. It is not a paired-complete-unit estimator.
See the [R did reference](https://bcallaway11.github.io/did/reference/att_gt.html).

Regression, normalized IPW and traditional DR remain available. Logit fits
share passes; identical date-pair populations share propensity fits. Outcome
regressions reuse the same population/date strata across comparisons. Attrition
can change those populations, so time-invariant covariates alone do not justify
reusing a propensity model across different date pairs.

Each required treated/control date stratum must have observations. Cells
without such support are omitted and listed in `diagnostics["omitted_cells"]`;
no supported cells is an error. Singular nuisance designs, separation and
all-controls-trimmed comparisons raise errors. Cohort-share aggregation uses
unique units in the retained panel, including units missing its first period,
and includes the uncertainty from estimated shares.

Using this convention requires the identifying assumptions for the observed
populations at each date. Treatment-propensity weighting does not itself correct
selective attrition. A change in panel composition can change the interpretation
of an estimate even when the software comparison matches R exactly.

## Sun–Abraham

The balanced change-of-means identity does not generally apply with missing
periods. Native SA instead fits the exact observed-row saturated regression
through independent within-cohort calendar-effect systems. Unit intercepts
absorb each cohort's normalization; with a separate interaction for every
observed non-reference date, the calendar effects can be fitted separately in
each cohort. Subtract the never-treated calendar change from the treated
cohort's change relative to `g-anticipation-1` to recover each SA interaction.

This reparameterization needs only small time-effect systems and bounded unit
score batches. It avoids staging the full row-by-interaction design and does
not load the general backend's extra formula machinery. Coefficients and full
unit-clustered covariance match fixest and dense unit/time FE regression checks.
Empty cells are listed in diagnostics; missing cohort references, missing never-
treated dates and disconnected cohort observation graphs raise identification
errors. CRV0/CRV1 conventions are unchanged.

SA aggregation weights cells by their observed treated-unit counts, matching
`fixest::sunab`'s observation shares. Single-observation units are retained in
the declared sample; their within-regression contribution can be zero.

## BJS imputation

Each unit's intercept is eliminated using its **actual** untreated observations.
The time-effects system remains only `T-1` dimensional. Each observed treated
cell uses its own contributing-unit count and its units' untreated histories
in the first-stage derivative. No imputation is attempted for missing outcomes.

A unit with no untreated observation raises an explicit identification error;
exclude such units deliberately before fitting. A disconnected or rank-deficient
untreated time design also raises. Native unbalanced BJS supports every existing
`avgeffectsby` group and `leaveout` setting, with smart averages and leverage
recomputed for each requested contrast.

To preserve bounded memory with arbitrary observation patterns, the returned
native unbalanced BJS result owns temporary sorted Parquet partitions. Each
aggregation makes two passes over those files. They remain under `workdir`
until the result is garbage collected or explicitly cleaned:

```python
fit = bjs_stream("y ~ 1", data, panel=("id", "time"), cohort="first_treat", panel_mode="unbalanced")
try:
    event_study = fit.aggregate()
    overall = fit.aggregate("simple")
finally:
    fit.cleanup()
```

After cleanup, stored coefficient tables, covariance and already-created
aggregates still work; another BJS aggregation requires refitting. Copies from
`with_vcov()` share these files. Balanced BJS retains its in-memory sufficient
moments and needs no aggregation files. CS and SA remove fit scratch files on
return.

## Validation

Offline fixtures check actual R did/DRDID estimates, full covariance and
aggregation for 24 unbalanced comparisons; fixest checks cell covariance and
Sun–Abraham event-time aggregation. BJS checks all four residual groupings,
with and without leave-out, against independent dense full-unit-dummy R score
calculations. `didimputation` outputs provide an additional package comparison.
Regenerate fixtures with `benchmarks/external/unbalanced_reference.R` and
`benchmarks/external/bjs_reference.R`, using `tests/data/unbalanced_panel.csv`.

[Anticipation](anticipation.md) is supported in every panel mode. Clean BJS
observations precede `g-A`; SA/CS references move to `g-A-1`. Missing clean
observations can create identification failures even when a fit with A=0 works.

All panel modes also support [sampling weights](weights.md). Positive weights
constant within unit enter fitting and inference; explicit opt-in also permits
[time-varying observation weights and richer SA/BJS models](richer_models.md).
Weighted BJS uses the same
temporary-file lifecycle described above, including on balanced input.
