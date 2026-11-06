# Time-varying weights and richer SA/BJS models

All three estimators support positive observation weights through the explicit
`allow_time_varying_weights=True` option. Sun–Abraham and BJS also support
covariates and additional fixed effects. Unit and calendar-time intercepts remain
part of both regression models.

```python
from diff_stream import bjs_stream, sunab_stream

fit = bjs_stream(
    "earnings ~ age + I(age**2) | region^year",
    "panels/*.parquet",
    panel=("person", "year"),
    cohort="first_treat",
    weights="population_weight",
    allow_time_varying_weights=True,
    panel_mode="unbalanced",
    pretrends=2,
)
try:
    fit.aggregate().summary()
    fit.aggregate("simple").summary()
    fit.pretrend_test()
    print(fit.diagnostics["absorbed_or_redundant_columns"])
finally:
    fit.cleanup()

# The same formula and weights work with sunab_stream; pretrends is BJS-only.
```

Unit/time effects are included automatically. Thus `| region^year` adds crossed
region-by-year effects; `| person + year + region^year` specifies the same model.
Additional additive effects use `+`, and crossings use `^`. Covariates use the
existing family expression and categorical compiler, including `C()`, numeric
transformations and interactions.

## Weight conventions

The default remains `allow_time_varying_weights=False`: a weight column must be
constant within each retained unit, preserving the earlier validation contract.
Enabling the option permits finite, strictly positive date-specific weights.
Zero, negative, missing and non-finite source weights remain errors. A common
positive rescaling leaves point estimates and inference unchanged. Physical
retained units determine the cluster count and CRV1 factor.

**SA:** each observed row's weight enters WLS and the unit-clustered regression
score. Aggregations use each target cohort/date cell's observed weight total,
conditional on treatment timing and observed weights.

**BJS:** each clean row's weight enters the untreated WLS first stage. Each
affected cell averages imputation residuals with that cell's actual row weights.
Unit intercepts use weighted clean-period means, not unweighted means. Smart
residual averaging and leave-out leverage use contrast-specific sums of target
weights within unit/residual group. All four `avgeffectsby` settings, `leaveout`,
anticipation, window balancing, and the separate clean-placebo regression remain
available. Aggregation conditions on treatment timing and target weight mass.

**Balanced CS:** each comparison freezes weights at its earlier date,
`min(target, base)`, just as it already does for covariates. Nuisance-fit reuse
also respects the weight date: time-invariant covariates alone do not justify
sharing fits across changing weight populations. In anticipation windows the
clean base is `g - anticipation - 1`.

**Native unbalanced CS:** each observed row carries its own date's covariates
and weight in the repeated-cross-section functional, and row influences fold to
original units. No complete-date-pair selection is introduced. As elsewhere in
this library, actually balanced input retains balanced comparison conventions.

CS cohort aggregation uses one unit weight: the first period for balanced
panels, and the mean of observed-period weights for unbalanced panels. Its
estimated-share influences include this weighted denominator. These comparison
and aggregation conventions match the installed R `did` 2.5.1 reference. They
also agree with the documented [R default comparison conventions](https://bcallaway11.github.io/did/reference/att_gt.html)
and [aggregation implementation](https://github.com/bcallaway11/did/blob/master/R/compute.aggte.R).
The optional R `fix_weights` overrides are not exposed in this implementation.

Time-varying weights define changing weighted populations. Numerical agreement
with another implementation does not itself establish the identifying assumptions
for that choice of weights.

`group_time()` and `contributions()` report input-scale target weight totals.
For CS those are anchored cohort totals; for SA/BJS they are observed target-cell
totals. Diagnostics record `weights_time_invariant`, `unit_weight_anchor`,
`weight_convention`, and the normalization scale. The reported
`total_unit_weight` and `cohort_weight_totals` use the unit anchor; they do not sum
all unit-period weights.

## Regression backend and limits

The richer regression path eliminates each unit intercept in a bounded whole-unit
batch, using the actual estimation-row weights. It then accumulates small normal
systems and unit-score cross-products. It does not retain all unit effects,
all-unit scores, or a global row-by-treatment-interaction design. SA assembles
cohort-specific active blocks into the joint system; BJS fits only clean rows
and streams the subsequent imputation/inference passes.

Unit absorption and low-dimensional Gram reduction live in
`diff_stream._core.within` for eventual promotion into `hdfe_stream.core`.
The DiD design, clean-row rules, target contrasts and smart residual grouping
remain in `diff_stream.linear` and the existing BJS inference context. The
existing hdfe-stream formula compiler is reused; no sibling package changes or
new dependency are required.

The **additional compiled design is limited to 256 columns**, including dummy
coding for extra fixed effects and covariates. This limit applies before
redundant columns are removed. Unit intercepts are eliminated and do not count
toward it; calendar-time columns and SA treatment cells add separate numerical
state. Extra effects use bounded-batch dummy coding and a dense reduced system;
this is not an unrestricted high-dimensional FE backend. Unit-interacted FEs,
varying slopes, IV formulas, replacement/removal of the compulsory unit/time
intercepts, and alternative clustering dimensions are unsupported.

Numerically redundant nuisance columns are listed in
`absorbed_or_redundant_columns`, including time-invariant covariates absorbed by
unit intercepts. Treatment cells are protected: unidentified joint SA treatment
effects raise rather than being silently removed. BJS raises when its richer
untreated design is unidentified, including nuisance levels that cannot be
learned from clean observations. No automatic target-sample deletion or
regularization resolves those failures.

Numerical state grows with design columns, periods, and treatment cells; the SA
joint solve and reported covariance remain quadratic in those dimensions.
Profiling and admission guards include batch-by-time-by-design working arrays
and joint systems. The original specialized paths remain in use when additional
terms and varying row weights are absent. See [resources](resources.md) and the
[stronger memory review](memory_checks.md) for source-reader overhead and
[kernel checks](kernel_memory_checks.md) for completed v1 host validation.
Live v2 and cross-platform execution evidence remain release-review items in
the [handoff](release_handoff.md).

Richer or time-varying-weight BJS retains sorted Parquet for contrast-specific
inference. Call `cleanup()` after the last aggregation, as for the existing
weighted/unbalanced BJS path. Coefficients and stored covariance remain available;
new BJS aggregations require the retained files.

## Validation and reproduction

Frozen inputs have 360 units, seven periods, positive nonseparable time-varying
weights, covariates, three regions, and balanced or observed-row samples with
missing initial/interior/final dates. The offline audit checks **168 fits and
532 estimate/full-covariance comparisons**:

- Actual R `did`: regression/IPW/DR, both control populations and base conventions,
  covariates or intercept only, anticipation 0/1, both panel types, and aggregate
  covariance including estimated cohort shares.
- Actual R `fixest`: fully joint weighted Sun–Abraham regression with numeric
  covariates and region-by-year effects, alongside simpler and invariant/equal-weight
  specifications; full unit-clustered covariance without degrees-of-freedom adjustments.
- Independent dense R full-unit-dummy BJS and clean-placebo regressions; all smart
  grouping/leave-out combinations for the richest varying-weight specification.
  Independent Python full-dummy regressions additionally check SA and BJS.

Maximum differences in the retained audit are:

| Reference | Estimate | Covariance |
|---|---:|---:|
| CS / actual did | 2.86e-8 | 4.11e-9 |
| SA / actual fixest | 1.20e-13 | 3.52e-12 |
| BJS / dense R | 3.58e-14 | 6.93e-15 |
| Clean placebos / dense R | 2.19e-14 | 3.54e-16 |

The declared CS tolerance is 3e-8; regression references use 1e-9. No Stata
execution for these new model specifications is claimed; the earlier Stata
validation matrix covers its documented simpler specifications.

Further regression checks cover rescaling weights by 1e-150/1e150, rescaling
covariates by 1e12, string IDs, shuffled Parquet with small row groups, small
numerical batches/partitions, CRV switching, complete-panel deletion, nuisance
aliases, unidentified target effects, explicit opt-in, ESS and multiplier bands,
and temporary-file cleanup. Frozen input/exporter/reference hashes live in
`tests/data/richer_reference/PROVENANCE.json`.

```bash
# Python-only audit; no package installation, downloads or R required:
python benchmarks/richer_validation.py
# Regenerate with existing locally installed R did/DRDID/fixest:
python benchmarks/richer_validation.py --regenerate --rscript /path/to/Rscript
# Sequential fresh-process resource matrix, generated offline:
python benchmarks/richer_resources.py
```

The resource runner uses the existing process-tree RSS watchdog and timeouts,
with raw results under `benchmarks/results/richer_resources`. Generation occurs
in separate processes and is excluded from fit measurements. These sampled RSS
checks are not OS-enforced physical-memory guarantees.

## Local resource outcomes

All ten cases passed, including fit, finite full covariance, requested aggregates,
clean placebos where enabled, and scratch cleanup. They use two threads,
`preprocess_mb=8`, `batch_units=256`, and native entry/exit panels. Smaller fits
have a 1 GB budget; million-unit and wide fits use 1.5 GB. Generation and
competing fits are excluded from the sequential fit measurements.

| Case | Observed rows | Additional compiled columns | Peak RSS MiB | Whole-process seconds |
|---|---:|---:|---:|---:|
| cs_10000 | 40,449 | 3 | 358.0 | 1.9 |
| sunab_10000 | 40,449 | 34 | 475.9 | 2.4 |
| bjs_10000 | 40,449 | 34 | 464.9 | 3.5 |
| cs_100000 | 403,224 | 3 | 513.5 | 11.5 |
| sunab_100000 | 403,224 | 34 | 591.8 | 8.1 |
| bjs_100000 | 403,224 | 34 | 578.1 | 19.6 |
| sunab_1000000 | 4,031,952 | 34 | 774.5 | 75.1 |
| sunab_wide | 145,074 | 130 | 552.5 | 11.7 |
| bjs_1000000 | 4,031,952 | 34 | 769.6 | 276.2 |
| bjs_wide | 145,074 | 130 | 499.0 | 47.7 |

The million-unit models have 4,031,952 observed rows. SA uses three numerical
source passes; BJS uses fourteen when the separate placebo regression and all
three requested aggregates are included. Their regression phases peak around
545/530 MiB, with preparation setting the larger whole-process peaks. BJS
aggregation recomputes contrast-specific smart weights through additional disk
passes, explaining a substantial time cost without retaining all-unit scores.
These finite probes do not establish a universal memory bound or a public speed
comparison. Subsequent [kernel-enforced host validation](kernel_memory_checks.md)
now includes these million-unit richer models under actual 1.5 GB cgroup v1 caps.
