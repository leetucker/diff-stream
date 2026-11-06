# Published Goodman–Bacon example: math_reform

The public `math_reform` worked example from the R `bacondecomp` documentation
is replicated in diff-stream, including all 16 comparisons and their weights.
The replication also verifies reconstruction against independent full-sample
and pairwise OLS regressions. **No Stata run is required.** Frozen data and
references support subsequent Python-only offline checks.

## Source and sample

The [documented example](https://github.com/evanjflack/bacondecomp#example)
uses aggregated data from Goodman (2019, *Journal of Labor Economics*) on
compulsory high-school mathematics coursework and subsequent income. The
[package dataset description](https://raw.githubusercontent.com/evanjflack/bacondecomp/master/R/data.R)
identifies the variables and aggregation. The installed source package is
`bacondecomp` 0.1.3, with its declared `MIT + file LICENSE` package license;
source/version metadata is retained alongside the external data snapshot.
The dataset's original provenance is retained without changing its licensing.

The snapshot contains **520 observations: 26 states × 20 graduating classes,
1975–1994**. Every state/class key is unique and the panel is balanced.
Outcomes and treatment indicators are complete. There are 11 never-treated
states and 15 adopting states:

| First exposed graduating class | States |
|---:|---:|
| Never treated | 11 |
| 1984 | 1 |
| 1985 | 4 |
| 1986 | 1 |
| 1987 | 9 |

The R formula is `incearn_ln ~ reform_math`, using `state` as unit and `class`
as time. The diff-stream outcome formula is `incearn_ln ~ 1`, with treatment
constructed from `reformyr_math`. Missing adoption years encode never treatment;
an explicit analysis column maps these values to cohort 0. Original source
columns and graduating-class values are preserved. No rows or states are
excluded, and no covariates or sampling weights are added.

This reproduces the documented aggregated-data example. It does not reproduce
every microdata sample, specification or table in the underlying paper.

## Results

Both the direct TWFE coefficient and the sum of weighted comparison estimates
are **0.002459972920**. The reconstruction difference is approximately
$1.65\times10^{-17}$.

| Comparison type | Comparisons | Total weight | Within-type weighted estimate | Contribution to TWFE |
|---|---:|---:|---:|---:|
| Treated vs never | 4 | 0.883820620 | -0.001288487 | -0.001138791 |
| Earlier vs later | 6 | 0.063530446 | 0.028675932 | 0.001821795 |
| Later vs earlier | 6 | 0.052648934 | 0.033751285 | 0.001776969 |

The three contributions sum to the TWFE coefficient. The last row uses an
already-treated comparison group. These are descriptive decomposition results;
no separate causal ATT or standard errors are supplied by this diagnostic.

All 16 comparison estimates, normalized weights, and contributions match actual
R `bacondecomp` output. The maximum estimate difference was
$1.73\times10^{-15}$ and maximum weight difference $5.56\times10^{-16}$.
The type summaries also reproduce the documentation's five-decimal display.

Independent full-unit/time-dummy OLS in R and Python checks the full regression
and each comparison's restricted time window. The largest difference from the
independent Python pair regressions was $2.01\times10^{-14}$.

## Run the example

```python
import polars as pl
from diff_stream import bacon_decomp

data = pl.read_csv("tests/data/math_reform.csv", null_values="NA").with_columns(
    pl.col("reformyr_math").fill_null(0).cast(pl.Int64).alias("first_treat")
)
fit = bacon_decomp(
    "incearn_ln ~ 1",
    data,
    panel=("state", "class"),
    cohort="first_treat",
)
print(fit.twfe_estimate)
print(fit.summary())
fit.comparisons()
```

To regenerate the complete report using only frozen Python-side inputs:

```bash
STREAM_LIMITS_MAX_THREADS=2 POLARS_MAX_THREADS=2 \
  .venv/bin/python benchmarks/math_reform_replication.py
```

Outputs under `benchmarks/results/math_reform_replication/` are:

- `report.json`: sample/specification details, software versions, checks and
  SHA-256 hashes of data, scripts and reference files.
- `comparisons.csv`: every comparison, block endpoint, estimate, weight and
  contribution.
- `type_summary.csv`: the unrounded decomposition summary above.

The input snapshot is `tests/data/math_reform.csv`. Versioned R references live
in `tests/data/math_reform_reference/`. `tests/test_math_reform.py` checks the
snapshot, explicit treatment mapping, reference agreement, shuffled/lazy input,
ID relabeling, Parquet input and scratch cleanup. Run these checks with:

```bash
STREAM_LIMITS_MAX_THREADS=2 .venv/bin/python -m pytest -q tests/test_math_reform.py
```

## Refresh the R reference

With `bacondecomp` installed in R, regenerate the snapshot and references using:

```bash
STREAM_LIMITS_MAX_THREADS=2 POLARS_MAX_THREADS=2 \
  .venv/bin/python benchmarks/math_reform_replication.py \
  --regenerate --rscript /path/to/Rscript
```

The exporter calls actual `bacondecomp::bacon` and `bacon_summary` without
replacing package functions. Full and pairwise OLS use R `lm` with explicit
unit/time factors. The runner respects `R_LIBS_USER`; its local default points
to the reference package library used by earlier validations. Version metadata
records the installation actually used.

This is an empirical correctness check. Large-unit/cohort memory scaling is
covered separately by the [Goodman–Bacon resource checks](bacon.md).
