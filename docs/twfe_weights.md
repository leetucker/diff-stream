# TWFE treatment-effect weights

`twfe_weights` implements the published de Chaisemartin–D'Haultfoeuille `feTR`
diagnostic for a conventional binary-treatment TWFE regression. It reports
weights on group/date treatment effects, including negative weights, and two
heterogeneity sensitivity measures. The target is the regression with unit/date
effects and the supplied covariates, using the retained observed sample.

The method is described in
[Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects](https://www.aeaweb.org/articles?id=10.1257/aer.20181169).
The numerical reference is the authors' R **TwoWayFEWeights** package,
`twowayfeweights(..., type="feTR")`; their Stata command is **twowayfeweights**,
`type(feTR)`. Both are maintained in the
[authors' repository](https://github.com/Credible-Answers/twowayfeweights).
Neither is a runtime dependency.

```python
from diff_stream import twfe_weights

diagnostic = twfe_weights(
    "y ~ x + z",
    "panel.parquet",
    panel=("id", "time"),
    treatment="treated",
    panel_mode="unbalanced",
    weights="sampling_weight",
    allow_time_varying_weights=True,
    batch_units=256,
    preprocess_mb=8,
    memory_gb=1.5,
)
print(diagnostic.twfe_estimate)
print(diagnostic.summary())
print(diagnostic.sensitivity())
diagnostic.summary_json("twfe_weights.json")
```

## Supported design and sample

Supply exactly one of `treatment="binary_column"` or `cohort="first_treat"`.
Direct treatment may reverse. With cohorts, zero denotes never treated and
positive cohorts give actual first-treatment dates; pre-sample onsets remain
always treated and future onsets remain untreated within the sample. Cohorts
must be nonnegative integers and constant within each unit. No anticipation
adjustment is applied to this diagnostic of actual treatment.

Each input row must be one observed unit/date cell. Duplicate keys raise rather
than being silently collapsed into a different regression. Time must consist
of positive integers; gaps in the calendar or in individual histories are
permitted. Unit IDs can be numeric or strings. Missing/nonfinite required values
raise; unrelated columns are not used. Treatment must be 0/1 (Boolean allowed).

`panel_mode="balanced"` is the default. `"unbalanced"` uses observed cells,
without filling missing outcomes or requiring complete two-by-two rectangles.
`"complete"` explicitly removes units missing any date in the input's observed
date grid, reporting dropped units and rows. Source validity is checked before
deletion. Never-treated and always-treated units are retained. Treatment must
remain identified after all nuisance effects; otherwise the function raises.

Unit and date effects are automatic. The existing formula compiler accepts
bounded numeric/categorical covariate expansions (up to 256 compiled input
columns). Additional absorbed FE specifications are outside this diagnostic's
scope. Redundant nuisance columns and disconnected date components are reported
in diagnostics; the target treatment column must remain identified. Singleton
units remain in the sample with zero residual treatment, rather than changing
the treated population through automatic deletion.

Sampling weights must be finite and strictly positive. They are constant within
unit by default; explicitly set `allow_time_varying_weights=True` for observed-date
weights. These are regression/cell weights; zero computational weights used
internally for absent cells are not accepted as input weights. A common positive
rescaling of sampling weights leaves the diagnostic unchanged.

## What the weights mean

Let $m_{it}$ be the cell's sampling weight and $r_{it}$ treatment residualized
on unit/date effects and covariates. Define

$$
S=\sum_{it}m_{it}r_{it}D_{it}=\sum_{it}m_{it}r_{it}^2,\qquad
\ell_{it}=\frac{m_{it}r_{it}}{S},\qquad q_{it}=\ell_{it}D_{it}.
$$

The observed coefficient satisfies
$\hat\beta_{TWFE}=\sum\ell_{it}y_{it}$ and the treatment-effect weights satisfy
$\sum q_{it}=1$. For binary treatment, untreated cells have $q_{it}=0$.
The coefficient is generally **not** $\sum q_{it}y_{it}$.

Under the paper's common-trends and treatment-effect assumptions, the $q$'s
weight group/date treatment effects. Effects may vary across groups and dates;
this `feTR` calculation does not impose the stable-effect restriction of `feS`.
Negative weights can make the coefficient's sign differ from the signs of the
underlying effects. Computing the weights does not establish parallel trends
or identify an ATT. For reversing treatment, the contemporaneous-effect
interpretation additionally requires that omitted treatment history or persistent
effects after exit do not invalidate that representation.

`summary()` reports positive/negative/zero counts among **treated** cells, weight
sums, and baseline treated-population shares. The latter use
$p_{it}=m_{it}D_{it}/\sum m_{it}D_{it}$. The relative multiplier $W=q/p$ on treated
cells is distinct from the normalized weight $q$. Untreated cells' multipliers
are null. A negative residual in an untreated cell is not a negative weight on
a treated-cell effect.

`summary_dict()["weights"]` additionally reports raw weight sum, extrema,
treated-cell count and negative mass (the positive magnitude of the classified
negative sum). Counts/sums use `zero_tol=1e-10` by default, matching R's cutoff:
`abs(q) < zero_tol` is classified as zero. Raw cell weights are preserved and
still sum to one. Displayed classified sums may differ slightly from one.
Set `zero_tol=0` to inspect raw signs, which can expose rounding-sized weights.

This diagnostic differs from [Goodman–Bacon](bacon.md): Bacon weights two-group
regression comparisons; this method weights underlying treated-cell effects.
It is a separate published formulation, not a native-unbalanced Bacon extension.

## Sensitivity summaries

`sensitivity()` returns a value and status for each measure:

- `zero_average_sd`: the minimum heterogeneity SD compatible with a zero overall
  treated average and the observed TWFE coefficient, using the authors' R
  finite-cell convention. Specifically, it is $|\hat\beta|/s_W$, with
  $s_W^2=\{M/(M-1)\}\sum p(W-1)^2$ and $M$ the treated-cell count. The population-SD
  version omits that finite-cell correction. The convention is included in JSON.
- `all_opposite_sign_sd`: the published threshold compatible with all treated-cell
  effects having the opposite sign from the coefficient, computed when there
  are negative weights under the chosen classification tolerance.

These are sensitivity calculations in outcome units, not estimated effect
dispersion, standard errors, p-values or estimates of bias. Constant multipliers
give an infinite first threshold for a nonzero coefficient. Without classified
negative weights, the second measure is unavailable, with status
`infeasible_no_negative_weights` under that classification. A zero coefficient
has no positive/negative sign, so the second measure has status
`undefined_zero_coefficient`; the first threshold is zero when its denominator
is positive. A zero coefficient and zero denominator are explicitly undefined.
Infinite/undefined values are represented by null plus status, rather than
invalid JSON infinities or a substituted zero.

The second measure uses bounded repeated tail reductions to find the published
cutoff. It does not collect or globally sort all cell weights in RAM. The number
of cutoff iterations is reported in the sensitivity metadata.

## Inspecting and exporting cell weights

Ordinary fits return reduced summaries and remove scratch files before returning.
Opt into retained disk-backed cells when needed:

```python
with twfe_weights(
    "y ~ x",
    "panel.parquet",
    panel=("id", "time"),
    treatment="treated",
    panel_mode="unbalanced",
    keep_cells=True,
) as diagnostic:
    negative = diagnostic.cells().filter(pl.col("classified_weight") < 0)
    # negative is a LazyFrame; collect only when its size is manageable.
    diagnostic.write_cells("results/cell_weights.parquet")
    for batch in diagnostic.iter_cells(batch_rows=32768):
        pass  # bounded Polars frames
```

Import `polars as pl` to use the filter expression. Export paths are caller owned
and must be outside the fit's temporary directory; existing files are not
overwritten. Exported Parquet survives context-manager exit or `cleanup()`.
Reduced summaries also survive cleanup, while retained lazy frames do not.
Without a context manager, call `diagnostic.cleanup()` when done.

Cells contain `unit`, `time`, `treatment`, input-scale `sampling_weight`,
`treatment_residual`, `outcome_loading`, raw `weight`, `treated_share`,
`relative_weight` and `classified_weight`. Ordering across partitions is
unspecified; there is no automatic global sort or in-memory cell table.

## Public wage-panel example and validation

```bash
.venv/bin/python examples/twfe_weights.py
.venv/bin/python benchmarks/twfe_weights_validation.py
```

The worked example uses the authors' public wage panel: log wage (`lwage`) on
union membership (`union`), worker (`nr`) and year effects. There are 545 workers
and 4,360 observations. Union membership can reverse. The unweighted coefficient
is approximately **0.10662747**, with negative weight mass **0.01052899**.
The first sensitivity threshold is **0.09686910** and the second **3.17585890**.
This reproduces the package's `feTR` example; it does not reproduce the separate
optional random-weight correlation tests.

The 78-configuration matrix covers absorbing/reversing treatment, covariates,
equal/unit/date-varying weights, always-treated and no-never-treated designs,
unbalanced/complete samples, gaps, disconnected panels, singleton units, and
constant-multiplier/zero-outcome cases. References use native R
TwoWayFEWeights **2.1.0**, with `fixest` singleton removal disabled to preserve
the stated observed-cell sample and FE tolerance set to 1e-10. That version's
implementation takes `weights` as a numeric vector despite its column-name
documentation; the exporter supplies the vector without replacing package code.

There are 69 identified native-package comparisons. Six all-zero-outcome cases
pass independent checks where the package refuses its constant outcome
regression. Three covariate-saturated six-cell designs fully explain treatment:
diff-stream refuses them based on identification checks, while the reference
returns numerical output that is not used as a valid benchmark. Raw reference
outputs/refusals and session metadata are preserved, not silently discarded.

Independent original-row dummy projections check every identified target and
cell weight. Known-effect synthetic panels verify $\hat\beta=\sum q\tau$ under
an exactly spanned untreated outcome. Separate constrained quadratic optimization
checks both sensitivity calculations, including the finite-cell correction.
Additional tests cover scaling, aliases, batching, paths, export/cleanup,
identification and resource guards. The final full regression run passed 1,597
tests with one kernel-only skip; the 103 focused checks pass on current Polars
and Polars 1.37. Frozen data, outputs and hashes are in
`tests/data/twfe_weights_reference/`, with the public input in
`tests/data/twfe_wagepan.csv`. R is only needed to regenerate:

```bash
.venv/bin/python benchmarks/twfe_weights_validation.py \
  --regenerate --rscript /path/to/Rscript
```

## Resources and remaining scope

Projection, normalization and cell export use bounded whole-unit passes. With
$P=T-1+K$ for $K$ compiled covariates and batch size $B$, dense state includes
$O(P^2+BTP)$. Source execution, partitions, Parquet metadata and allocators have
additional overhead. `max_design_columns=2048` is an explicit projection-width
guard; the memory headroom check may refuse smaller designs. `profile=True`
validates/stages the source and reports a heuristic before projection. Full
cell output scales with observed cells on disk, even when summaries stay small.
Cell writes coalesce small unit batches into bounded 32,768-row groups, avoiding
excessive Parquet metadata in the repeated sensitivity passes.

The resource driver `benchmarks/twfe_weights_resources.py` uses the shared
stream-limits launcher, with portable watchdog mode or an explicitly delegated
Linux kernel cap. The driver requests RSS measurement explicitly; production
and launcher defaults are unchanged. The constrained test also uses common
fixtures and is included in Linux/macOS/Windows CI. Execution on this WSL host
does not substitute for those other OS jobs.

Six fresh-process kernel-capped probes completed without OOM kills or limit
failures, with scratch cleanup verified. Every probe used reversing treatment,
two or more covariates, date-varying weights, both sensitivity measures and full
cell export. Results under the verified 1.5 GB kernel cap on this WSL host:

| Units | Periods | Covariates | Panel | Observations | Peak RSS MiB |
|---:|---:|---:|---|---:|---:|
| 10,000 | 8 | 2 | Balanced | 80,000 | 440 |
| 100,000 | 8 | 2 | Balanced | 800,000 | 516 |
| 1,000,000 | 8 | 2 | Balanced | 8,000,000 | 496 |
| 1,000,000 | 8 | 2 | Unbalanced | 7,800,000 | 507 |
| 10,000 | 64 | 2 | Unbalanced | 638,000 | 515 |
| 5,000 | 16 | 32 | Unbalanced | 79,000 | 430 |

The eight-million-row export occupied about 249 MiB on disk. These are finite
local checks rather than a universal memory bound or speed comparison. Raw
results, kernel counters, environment and source hashes are in
`benchmarks/results/twfe_weights_resources/`. Reproduce with:

```bash
.venv/bin/python benchmarks/twfe_weights_resources.py \
  --cgroup-parent /path/to/delegated/test-parent
```

Omit the cgroup argument for portable watchdog enforcement.

Deferred: stable-effect `feS`, first-difference variants, continuous/multivalued
treatments, multiple-treatment contamination weights, microdata aggregation,
random-weight correlation tests, and extra high-dimensional absorbed effects.
The authors' R/Stata packages cover broader functionality; use those while it
remains outside diff-stream's scope. Existing-estimator pre-release extensions
in `DESIGN_NOTES.md` remain separate work.
