# Goodman–Bacon decomposition

`bacon_decomp` decomposes the OLS coefficient on actual treatment in a
unit/time fixed-effects regression into weighted two-group comparisons. Optional
positive time-invariant sampling weights define a WLS decomposition. It
reports which comparisons contribute to TWFE, including those using an
already-treated control cohort. It is a descriptive decomposition, with no
separate causal ATT, standard errors or parallel-trends p-value.

```python
from diff_stream import bacon_decomp

bacon = bacon_decomp(
    "earnings ~ 1",
    "panels/*.parquet",
    panel=("person", "year"),
    cohort="first_treat",
    weights="sampling_weight",  # omit for equal unit weights
    preprocess_mb=8,
    memory_gb=1.0,
)
print(bacon.twfe_estimate)
print(bacon.decomposed_estimate)
print(bacon.reconstruction_error)
bacon.comparisons()
bacon.summary()
bacon.summary_json("bacon.json", comparisons=True)
```

As with the other entry points, `"earnings ~ 1"` supplies the outcome. Treatment
is constructed from the cohort column as $D_{it}=1\{g_i>0,t\geq g_i\}$. The
coefficient being decomposed is from earnings on **this treatment indicator**
with unit and time effects. No treatment column is required.

## Supported sample

The implementation supports balanced absorbing-treatment panels with optional
covariates, with equal or positive time-invariant unit sampling weights.
Time must be consecutive positive integers;
cohort must be constant within unit, with 0 for never treated. Future cohorts
whose treatment starts after the sample normalize to 0. Positive cohorts on or
before the first sample date form an always-treated group, labelled by that
first date. Duplicate keys, missing/non-finite values, negative cohorts and
non-integer timing raise errors.

Strict `panel_mode="balanced"` is the default. `panel_mode="complete"` explicitly
drops incomplete units and reports dropped rows/units. Native unbalanced panels
remain unsupported and deferred. Always-treated units are retained. For
`"y ~ x + z"`, the [covariate-adjusted decomposition](bacon_extensions.md) uses
within-timing and full-period between-timing components. The two-block formulas
below describe fits **without covariates**.

A never-treated group is optional for this diagnostic. At least two distinct
realized timing groups must remain, with residual treatment variation after
unit/time effects. A single treated timing plus never-treated units supplies
one comparison. Multiple treated timings without never-treated units supply
the earlier/later comparisons. Always-treated plus never-treated units alone
do not identify treatment after unit effects. With covariates, treatment must
remain identified after the full projection. Other estimators retain their own
[reference-population policies](reference_groups.md).

The diagnostic describes the regression using **actual** treatment timing. It
does not shift treatment or reference periods through an anticipation option.
Anticipatory outcomes can influence its comparison estimates, just as they can
influence the conventional TWFE regression being decomposed.

## Comparisons and outputs

For first/last sample dates $t_{min},t_{max}$ and cohorts $e<\ell$:

| Type | Treated cohort | Control cohort | Pre block | Post block |
|---|---|---|---|---|
| `treated_vs_never` | $g$ | 0 | $t_{min},\ldots,g-1$ | $g,\ldots,t_{max}$ |
| `earlier_vs_later` | $e$ | $\ell$ | $t_{min},\ldots,e-1$ | $e,\ldots,\ell-1$ |
| `later_vs_earlier` | $\ell$ | $e$ | $e,\ldots,\ell-1$ | $\ell,\ldots,t_{max}$ |
| `later_vs_always` | $g$ | $t_{min}$ | $t_{min},\ldots,g-1$ | $g,\ldots,t_{max}$ |

Each estimate is the treated cohort's change between block means minus the
control cohort's change. The last two types use an already-treated control in both
blocks. Changing effects in that earlier cohort can therefore affect the
comparison. The [Goodman–Bacon paper](https://www.nber.org/papers/w25018)
derives the decomposition and its interpretation under treatment-effect
heterogeneity.

`comparisons()` (also `tidy()`) returns:

- `treated_cohort`, `control_cohort`, `comparison_type`, and
  `control_already_treated`;
- exact pre/post block endpoints and period counts;
- physical treated/control unit counts, alongside input-scale
  `treated_weight_total` and `control_weight_total`;
- `estimate`, `weight`, and `contribution = weight * estimate`.

The weights are nonnegative and sum to one. A negative comparison estimate is
distinct from a negative decomposition weight. `summary()` returns totals by
comparison type: comparison count, total weight, within-type weighted estimate,
and contribution to the overall TWFE coefficient. Those type contributions sum
to `decomposed_estimate`.

Diagnostics include sample counts, cohort sizes, partition information,
residualized treatment variance, the weight sum, and
`already_treated_control_weight`. This last number reports the decomposition
weight on `later_vs_earlier` and `later_vs_always` comparisons; it does not measure the size of bias
or supply a formal test.

With covariates, `comparisons()` instead reports full-period dyads and
`within_timing()` reports the additional within-timing component. Their total
weights sum to one; `summary()` includes both. Dyad rows are not separate
two-block DiDs. See [extensions](bacon_extensions.md) for the weight conventions
and interpretation.

## Sampling weights

Pass `weights="column"` to decompose weighted TWFE. Weights must be numeric,
finite, strictly positive and constant within each unit. The same unit weights
enter cohort-time outcome means, cohort population shares, every two-group
comparison, and the independent grouped WLS calculation. There is no change to
the comparison time blocks or type definitions. A common positive rescaling of
weights leaves estimates, normalized decomposition weights and contributions
unchanged.

`n_units`, `n_obs`, `n_treated_units` and `n_control_units` remain physical counts.
The two `*_weight_total` columns report each comparison cohort's sampling-weight
mass on the input scale. They differ from `weight`, which remains the normalized
Goodman–Bacon contribution weight. Diagnostics include the weight column, total
retained unit weight, and cohort weight totals. Complete-panel deletion
recomputes these quantities on retained units. Weight validation applies to
source rows before deletion; zero/missing or time-varying weights are errors.

This is a descriptive decomposition of a sampling-weighted regression. It
does not add standard errors or replicate observations in the fitted data.
Never-treated units remain optional, and future cohorts retain the existing
within-sample control normalization.

## Computation and resources

The implementation reuses the existing streaming projection, disk-backed
whole-unit validation and bounded partition reductions. Without covariates, only cohort-time means
and unit-weight masses enter the decomposition. Prefix sums yield block means without
refitting a regression for every pair. For cohort population shares $p_a,p_b$,
block lengths $L_0,L_1$ and sample length $T$, an unnormalized comparison weight
is $p_a p_b L_0 L_1/T^2$. With sampling weights, $p_g$ is the cohort's retained
weight total divided by total retained unit weight. Their sum equals the
weighted variance of treatment after removing unit/time effects; normalization
gives the decomposition weights.

A separate grouped within-transformation computes the TWFE coefficient. Both
the treatment-variance identity and coefficient reconstruction are checked.
R and dense regression validation additionally check the calculation without
using these grouped formulas.

The existing panel-store validation and weight normalization are reused.
Weights are scaled by the input maximum and normalized to retained-unit mean
one for numerical reductions. No new shared-core weighting kernel is needed.
Only the extra numeric weight column is staged; no global unit-weight table is
retained.

Without covariates numerical/output state is $O(GT+G^2)$ for $G$ timing groups.
With $K$ changing treated cohorts and indicators $N,A$ for never/always-treated
groups, the table has $K(K-1)+K(N+A)$ rows. Controlled fits add a dense projection
and bounded batches, with $G(G-1)/2$ dyads. Neither path retains a global unit-score
or unit-regression table.
Partition readers, source execution, allocator retention and the directory of
disk leaves still contribute to process memory. `profile=True` validates the
source and estimates resource needs before constructing the comparison table;
the numerical headroom guard accounts for the quadratic output. The heuristic
does not replace an operating-system memory limit.

Scratch files are removed before the result is returned, and on Python errors.
The result is self-contained and needs no `cleanup()` call. `threads`,
`memory_gb`, `workdir` and `preprocess_mb` follow the other entry points.
`batch_units` bounds whole-unit numerical batches for covariate adjustment.

## Validation

The R exporter `benchmarks/external/bacon_reference.R` runs actual
`bacondecomp::bacon` and independently fits TWFE with `lm` and full unit/time
dummies. Versioned fixtures cover unequal cohort sizes, timing gaps, no never
treated, one treated timing, future cohorts, shifted time origin, explicit
complete-panel deletion, constant/null effects and a two-period design.
Python dense regressions additionally check each comparison's restricted sample,
random panels, input paths/globs/lazy frames, string IDs, partition changes,
unsupported requests, resource guards and cleanup.

Regenerate the reference files with R and `bacondecomp` installed:

```bash
Rscript benchmarks/external/bacon_reference.R \
  tests/data/bacon_panel.csv tests/data/bacon_reference
```

Weighted references use `benchmarks/external/bacon_weights_reference.R`.
It independently fits original-row WLS with full unit/time dummies for general
positive weights. Because R `bacondecomp` has no sampling-weight argument,
rational weights are additionally checked through exact integer unit replication:
for $w_i=(1+5(i\bmod 9))/5$, the reference repeats unit $i$ exactly $5w_i$ times
under distinct replicated IDs. The resulting package weights and estimates are
actual `bacondecomp` outputs. This replication is a reference check; diff-stream
retains the original rows and physical counts. Irrational-weight comparison
estimates are separately checked with restricted-sample WLS.

The weighted tests cover all nine sample/timing cases, equal-weight equivalence,
rescaling by $10^{-200}$/$10^{200}$, complete-panel deletion, shuffled/string
IDs, Parquet globs/lazy input, small partitions, invalid weights, profiling and
cleanup. Generate these additional references with:

```bash
Rscript benchmarks/external/bacon_weights_reference.R \
  tests/data/bacon_panel.csv tests/data/bacon_weights_reference
```

Reproduce the offline resource suite with:

```bash
STREAM_LIMITS_MAX_THREADS=2 POLARS_MAX_THREADS=2 \
  .venv/bin/python benchmarks/bacon_resources.py
```

Generation runs separately in 1,000-unit blocks. Fresh-process decompositions
use two threads, an 8 MB preprocessing allowance and a 1 GB sampled RSS watchdog:

| Units | Periods | Comparisons | Peak RSS | Result |
|---:|---:|---:|---:|---|
| 10,000 | 8 | 49 | 329.7 MiB | Reconstructed TWFE |
| 100,000 | 8 | 49 | 443.6 MiB | Reconstructed TWFE |
| 1,000,000 | 8 | 49 | 551.7 MiB | Reconstructed TWFE |
| 10,000 | 64 | 3,969 | 447.0 MiB | Reconstructed TWFE |
| 1,024 | 512 | 261,121 planned | 449.1 MiB | Comparison-table guard refused allocation |

All cases cleaned up successfully. Raw results, logs and configuration files
are retained in `benchmarks/results/bacon_resources/`. These finite checks show
the supported sample sizes and the distinct cohort-dimension limit; they are
not a universal hard-memory guarantee.

Weighted resource checks use the same panel sizes, thread/preprocessing settings
and watchdog, with positive unit weights `w=0.2+id%9`:

```bash
STREAM_LIMITS_MAX_THREADS=2 POLARS_MAX_THREADS=2 \
  .venv/bin/python benchmarks/bacon_weights_resources.py
```

| Units | Periods | Comparisons | Peak RSS | Result |
|---:|---:|---:|---:|---|
| 10,000 | 8 | 49 | 320.3 MiB | Reconstructed weighted TWFE |
| 100,000 | 8 | 49 | 456.4 MiB | Reconstructed weighted TWFE |
| 1,000,000 | 8 | 49 | 580.7 MiB | Reconstructed weighted TWFE |
| 10,000 | 64 | 3,969 | 448.6 MiB | Reconstructed weighted TWFE |
| 1,024 | 512 | 261,121 planned | 454.6 MiB | Comparison-table guard refused allocation |

All weighted cases cleaned up successfully. Raw configurations, results and
logs live in `benchmarks/results/bacon_weights_resources/`. The same
cohort/period/output-size limits apply as for the equal-weight diagnostic.

## Public worked-example replication

The [math_reform replication](math_reform_replication.md) checks the exact
public example in the R package documentation: 26 states, 20 graduating classes,
all 16 comparisons, type summaries and independent TWFE reconstruction.
The frozen data and references rerun offline without R or Stata.

## BJS follow-up

The BJS clean-placebo work that followed this decomposition is now implemented.
Request `pretrends=L` for separate clean lead estimates and joint testing; see
[BJS clean placebos](bjs_placebos.md). Anticipation cells in the main BJS effect
result remain affected-period estimates, separate from those placebos.
