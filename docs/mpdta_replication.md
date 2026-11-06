# Published-data replication: mpdta

The documented empirical example from R `did` now has an offline replication
in diff-stream. **Stata is not required for this check.** It compares CS with
the installed R `did`/DRDID implementation on a frozen copy of its `mpdta` data.
The separate Stata BJS validation bundle remains an independent check.

The source is the [official did empirical example](https://bcallaway11.github.io/did/articles/did-basics.html),
using a subset of the minimum-wage/teen-employment data from Callaway and
Sant'Anna (2021). The snapshot has 500 counties and 2,500 rows, with consecutive
annual observations for 2003–2007. Cohort sizes are 309 never-treated counties,
20 first treated in 2004, 40 in 2006, and 131 in 2007. There are no missing
observations and no sample exclusions. Original column names and calendar
years are preserved; `first.treat=0` still denotes never treated. The installed
`did` package declares GPL-3; source/version information is retained with the
external data references, without changing the source data's licensing.

This reproduces the documented empirical specifications and extends their
software checks. It is not a replication of every sample, specification or
table in the full paper. Inference matches the documented county-unit
clustering; no additional state-level clustering is introduced.

## Results

The baseline specification uses DR, never-treated controls, varying bases,
no anticipation and no sampling weights. Overall **simple** aggregation gives:

| Covariates | Simple ATT | Analytic SE |
|---|---:|---:|
| Intercept only | -0.0399512752 | 0.0120340128 |
| `lpop` | -0.0417517721 | 0.0115028382 |

These results match R `did`. The largest absolute estimate difference over
the specification matrix was $4.44\times10^{-11}$, and the largest covariance
entry difference was $4.16\times10^{-13}$. Analytic SEs are reported here;
the reference's ordinary bootstrap is disabled for deterministic covariance
comparisons. Separate common-draw checks validate multiplier inference.

```python
from diff_stream import cs_stream

fit = cs_stream(
    "lemp ~ lpop",
    "tests/data/mpdta.csv",
    panel=("countyreal", "year"),
    cohort="first.treat",
    est_method="dr",
    control_group="nevertreated",
    base_period="varying",
    bootstrap_reps=999,
    bootstrap_seed=815,
    preprocess_mb=64,
    batch_units=73,
    threads=2,
)
fit.aggregate("simple").tidy()
fit.aggregate().multiplier_confint()
fit.pretrend_test()
```

## What is checked

The 24 specifications cover intercept-only/`lpop`, regression/IPW/DR,
never-treated/not-yet-treated controls, and varying/universal bases. Each checks:

- Group-time estimates, analytic SEs and full joint covariance.
- Dynamic, simple, cohort and calendar aggregations, with their full covariance.
- Dynamic aggregation balanced through exposure 1 (`balance_e=1`).
- Clean group-time joint pre-trend Wald statistics and p-values.

For the two baseline DR/never/varying models (without and with `lpop`), another
check compares every saved multiplier perturbation, IQR standard error,
simultaneous critical value and interval endpoint for all five aggregation
outputs. It uses 999 common Rademacher repetitions drawn in sorted county-ID
order. The largest difference in multiplier interval endpoints was
$8.95\times10^{-11}$ and in critical values $4.25\times10^{-10}$.

Common draws are passed to the reference R process by replacing only its
draw-generator function in memory. Estimation, aggregation, and actual
`did::mboot` IQR/type-1 quantile calculations are unchanged. Installed package
files are not edited. This isolates inference agreement from R/NumPy RNG
differences; the same numeric seed is not claimed to generate the same native
R RNG stream. A single sorted county-ID leaf is asserted for this common-draw
test; arbitrary repartitioning can change unit-to-draw assignments.

## Reproduce without R or Stata

The frozen input is `tests/data/mpdta.csv`; versioned references live in
`tests/data/mpdta_reference/`. Run the complete report using Python only:

```bash
STREAM_LIMITS_MAX_THREADS=2 POLARS_MAX_THREADS=2 \
  .venv/bin/python benchmarks/mpdta_replication.py
```

This writes `benchmarks/results/mpdta_replication/report.json` and
`estimates.csv`. The report contains per-case maximum differences, data and
script hashes, software versions, cohort counts and bootstrap settings. The
CSV includes all aggregation estimates and the matched baseline multiplier
bands. The same numerical comparisons are checked in `tests/test_mpdta.py`:

```bash
STREAM_LIMITS_MAX_THREADS=2 .venv/bin/python -m pytest -q tests/test_mpdta.py
```

## Refresh the R reference

With R `did`/DRDID installed, regenerate the dataset snapshot, analytic
references and common multiplier results using:

```bash
STREAM_LIMITS_MAX_THREADS=2 POLARS_MAX_THREADS=2 \
  .venv/bin/python benchmarks/mpdta_replication.py \
  --regenerate --rscript /path/to/Rscript
```

The bundled environment defaults point to the local R installation and library
used for earlier validations. The runner respects an explicit `R_LIBS_USER`; set it when an alternate R
installation should use a different package library. The script records the actual package versions instead of
assuming equality to results from another release.

This numerical replication checks the supported design and inference
conventions. Runtime/memory scaling remains covered by the separate large
resource suites; the small empirical dataset supplies a real-data correctness
check rather than a performance claim.
