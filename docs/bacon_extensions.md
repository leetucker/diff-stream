# Covariate-adjusted Bacon and always-treated groups

`bacon_decomp` supports balanced panels with covariates and always-treated
groups, including positive time-invariant unit sampling weights. The target is
the conventional OLS/WLS coefficient on actual treatment after unit/time effects
and the supplied covariates. This remains a descriptive regression decomposition.

```python
from diff_stream import bacon_decomp

r = bacon_decomp(
    "y ~ x + z",
    "panel.parquet",
    panel=("id", "time"),
    cohort="first_treat",
    weights="sampling_weight",
    batch_units=256,
    preprocess_mb=8,
    memory_gb=1.5,
)
print(r.twfe_estimate)
print(r.summary())  # includes within-timing and between-timing pieces
print(r.within_timing())
print(r.comparisons())
```

## Covariate-adjusted identity and weight conventions

This implements the covariate-adjusted decomposition documented by
[Goodman–Bacon](https://www.nber.org/papers/w25018) and
[R bacondecomp](https://search.r-project.org/CRAN/refmans/bacondecomp/html/bacon.html).
Covariates change the decomposition: separate early/late two-block DiDs no longer
describe every term. Let $r_{it}$ be treatment residualized on unit/time effects
and covariates, and let $\bar r_{gt}$ be its weighted timing-group/date mean.
With unit sampling weights $w_i$, define

$$
S = \sum_{it}w_i r_{it}^2,\qquad
r^W_{it}=r_{it}-\bar r_{g_i t},\qquad
\Omega=\frac{\sum_{it}w_i(r^W_{it})^2}{S}.
$$

The within-timing estimate is
$\beta_W=\sum w_i r^W_{it}y_{it}/\sum w_i(r^W_{it})^2$.
For timing groups $g<h$, write $m_g=\sum_{i:g_i=g}w_i$,
$M=\sum_g m_g$ and $\Delta r_{gh,t}=\bar r_{gt}-\bar r_{ht}$.
The full-period dyad has

$$
W_{gh}=\frac{m_gm_h}{MS}\sum_t(\Delta r_{gh,t})^2,\qquad
\beta_{gh}=\frac{\sum_t\Delta r_{gh,t}(\bar y_{gt}-\bar y_{ht})}
{\sum_t(\Delta r_{gh,t})^2}.
$$

Thus $\hat\beta_{TWFE}=\Omega\beta_W+\sum_{g<h}W_{gh}\beta_{gh}$,
and $\Omega+\sum W_{gh}=1$. Unit centering of outcomes leaves this identity
unchanged and reduces cancellation from large unit intercepts.

- `within_timing()` reports `weight=Omega`, `estimate=beta_W` and their
  contribution. A zero-weight component has a null estimate and zero contribution.
- `comparisons()` reports `weight=W_gh`: weights on the **total** TWFE coefficient.
  Those weights alone sum to `1-Omega`. Its additional `between_weight` column
  normalizes dyads to sum to one, matching the conditional weights in R's
  `two_by_twos`. Use `weight * estimate` for contributions to overall TWFE.
- `summary()` includes the within-timing row. All its weights sum to one and all
  contributions sum to `decomposed_estimate`; JSON includes both components.
- `both_treated` is a full-period composite. Pre/post block endpoints are null;
  `control_already_treated` is null for this composite. The adjusted diagnostic's
  `already_treated_control_weight` is also null: a single clean/forbidden weight
  would misdescribe these terms.

These nonnegative decomposition weights are distinct from weights on underlying
treatment effects, which can be negative. Neither a dyad coefficient nor the
within-timing coefficient is automatically a causal ATT. Extra absorbed FE
terms are not supported by this diagnostic. The existing formula compiler's
bounded covariate expansion applies; absorbed or redundant columns are reported
in diagnostics. If covariates fully explain treatment, the fit raises an
identification error. Covariates absorbed entirely by unit/time effects produce
a zero within component and the original TWFE coefficient.

## Always-treated groups

Positive cohorts on or before the first retained date collapse to one
always-treated timing group, labelled by that date. The diagnostic reports the
number of retained always-treated units. This opt-in behavior belongs to Bacon;
the estimation entry points still require clean pre-treatment observations.
Negative cohort values remain invalid; zero retains its never-treated meaning.

Without covariates, `later_vs_always` compares a switching cohort against this
already-treated group over its pre/post blocks. It contributes to
`already_treated_control_weight`. Always-treated versus never-treated alone
provides no treatment variation after unit effects and raises an error.

With covariates, comparisons are full-period residualized dyads. An
`always_vs_never` dyad can receive weight even though neither group's raw
treatment changes: global covariate residualization can induce a contrast in
their residualized treatment. It is part of the algebraic decomposition, not a
new treatment-switching DiD. No always-treated units are silently dropped.

## Public example and external validation

Run the repository example:

```bash
.venv/bin/python examples/bacon_covariates.py
```

It uses the public `castle` dataset distributed by R bacondecomp, with log
homicide as outcome and log population/log income controls. Treatment begins
at the first observed `post=1` date. The frozen dataset has 50 states and
11 annual observations per state. It is a decomposition of that package's
documented regression example, rather than a replication of every specification
in the empirical paper. It is a separate package-supplied snapshot from the
did2s Castle data used in the Gardner example. The unweighted controlled
coefficient is 0.0905853; the within-timing weight is approximately 0.013605.
The general-weight validation case uses each state's mean `popwt` over the
sample, so its sampling weights are time invariant.

The extended reference matrix contains 48 fits: eight samples, controlled and
uncontrolled specifications, and equal/integer/general positive unit weights.
It covers always-treated groups, no never-treated units, several pre-sample
onsets, shifted time origins, future treatment and explicit complete-panel
deletion, alongside the public data.

All 48 target coefficients match independent original-row R `lm` WLS with full
unit/time dummies, and independent Python row/dummy projections validate the
components. Native R bacondecomp supplies usable decompositions in 28 cases;
integer weights use exact distinct-ID unit replication because R has no
sampling-weight argument. Sixteen general-weight cases use independent WLS
rather than native Bacon references. Four controlled always-treated cases
trigger an internal R constant-dependent-variable regression error. Their
errors are retained in `models.csv`; they are validated against independent
WLS/projection equations, not presented as native R matches.

The hashes, package versions, native errors and raw outputs are frozen in
`tests/data/bacon_extended_reference/`; no R runtime dependency is introduced.
The numerical tests also cover redundant/absorbed covariates, treatment
collinearity, covariate/weight rescaling, string IDs, input paths, batch changes,
resource admission and cleanup. To regenerate with R installed:

```bash
.venv/bin/python benchmarks/bacon_extended_validation.py \
  --regenerate --rscript /path/to/Rscript
```

Without `--regenerate`, validation uses the frozen references offline.

## Computation and constrained-resource checks

The adjusted path makes three bounded whole-unit passes: the treatment
projection, cohort/date reductions, and the within-timing component. There is
no global unit-score matrix or separate source regression per dyad. With
$P=T-1+K$ for $K$ compiled non-intercept covariates and batch size $B$, numerical
state includes $O(P^2+BTP+GT+G^2)$. The profile/admission estimate includes
projection buffers and the comparison table; it is a heuristic, not an OS cap.

Five fresh-process checks used the shared stream-limits launcher under a
verified 1.5 GB kernel cap on this WSL host. Sampled RSS is requested explicitly
(`memory_measure="rss"`); ordinary fitting and launcher defaults are unchanged.
All fits completed, reconstructed TWFE within 1.3e-13, cleaned their scratch
files and recorded no estimation OOM kills.

| Units | Periods | Covariates | Observations | Peak RSS MiB |
|---:|---:|---:|---:|---:|
| 10,000 | 8 | 2 | 80,000 | 389 |
| 100,000 | 8 | 2 | 800,000 | 501 |
| 1,000,000 | 8 | 2 | 8,000,000 | 525 |
| 10,000 | 64 | 2 | 640,000 | 504 |
| 5,000 | 16 | 32 | 80,000 | 445 |

These are finite local checks of combined covariates, sampling weights and
always-treated groups. They do not establish a universal memory bound or a
cross-package performance comparison. Raw launcher reports and kernel counters
are in `benchmarks/results/bacon_extended_resources/`. Reproduce with:

```bash
.venv/bin/python benchmarks/bacon_extended_resources.py \
  --cgroup-parent /path/to/delegated/test-parent
```

Omit the cgroup option to use the portable watchdog. The ordinary test suite's
constrained-resource case uses common fixtures and is included in the
Linux/macOS/Windows CI matrix; this host has not executed those other OS jobs.

## Deferred work

Native unbalanced Goodman–Bacon decomposition remains deferred. Explicit
`panel_mode="complete"` is available, with reported sample deletion. A
regression can be identified on an incomplete panel even when there are no
complete two-by-two rectangles, so this implementation does not generalize the
classic comparison identity by silently discarding observations or renormalizing
whatever comparisons happen to remain.

The published de Chaisemartin–D'Haultfœuille TWFE treatment-effect-weight
diagnostic is now available separately as [twfe_weights](twfe_weights.md)
([paper](https://www.aeaweb.org/articles?id=10.1257/aer.20181169),
[authors' implementation](https://github.com/Credible-Answers/twowayfeweights)).
It is a distinct diagnostic with its own assumptions and decomposition target,
including observed-cell unbalanced panels. Its implementation does not change
the deferred status of native unbalanced Bacon decomposition.
