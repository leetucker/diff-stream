# Castle Doctrine: Sun–Abraham empirical reanalysis

The public Cheng and Hoekstra (2013) Castle Doctrine panel supplies an empirical
worked example for Sun–Abraham estimation. The baseline uses log homicide rates,
state/year effects, never-treated controls and state-clustered CRV0 inference.
All group-time estimates and their full covariance match actual R
`fixest::sunab`; independent full-dummy regressions in R and Python also agree.

This is **a Sun–Abraham reanalysis of a published application**. The original
paper predates SA. Its richer controls, trend specifications, exposure coding
and substantive conclusions are not reproduced in full here. The empirical
timing sensitivity below is part of the result, not a numerical discrepancy.
No estimator changes or Stata execution were required.

## Source, sample and treatment timing

Source: the public `data/castle.rda` in the
[did2s repository](https://github.com/kylebutts/did2s/tree/3adaf985980d4b46205de2e1856f93ba075baefe),
pinned to commit `3adaf985980d4b46205de2e1856f93ba075baefe`. The archived
`DESCRIPTION` identifies source version 1.3.0. Its declared software license is
`MIT + file LICENSE`; original data provenance is retained without relicensing
the data. The small original R object, source notices, hashes and CSV snapshot
are included in the source checkout. Running the frozen checks needs Python
only; regenerating references needs installed R fixest, without did2s installation.

The [dataset documentation](https://kylebutts.github.io/did2s/reference/castle.html)
attributes the data to Cheng and Hoekstra's
[study of expanded self-defense laws](https://business.baylor.edu/scott_cunningham/teaching/cheng-and-hoekstra-2013.pdf).
The frozen snapshot has **550 observations: 50 states × 11 years, 2000–2010**.
Keys are unique, every state has every year and required estimation values are
complete. No rows or states are excluded. The raw seven columns are retained:
`year`, `state`, `popwt`, `l_homicide`, `post`, `effyear`, `time_til`.

| Source `effyear` | States |
| --- | ---: |
| Never treated (missing) | 29 |
| 2005 | 1 |
| 2006 | 13 |
| 2007 | 4 |
| 2008 | 2 |
| 2009 | 1 |

The snapshot therefore has **21 adopting states**, although the online dataset
description says 20. This is an explicit source-version discrepancy, not a
reason to delete or recode a state.

There is also a treatment-timing distinction:

- The baseline sets `first_treat = effyear`, with missing adoption years mapped
  to zero. Event time zero is the recorded law-adoption year, and year −1 is
  the reference. The supplied `time_til` agrees with that timing for adopters.
- The raw `post` instead equals `year > effyear` for adopters. It excludes the
  adoption year, producing 21 state-year differences from the baseline's
  `year >= effyear` treatment indicator.
- A following-year sensitivity sets `first_treat = effyear + 1`, matching the
  supplied `post` exactly. This also changes the reference year, so its ATT
  is a different timing/normalization specification.

The baseline was selected from the proposed adoption-year specification before
examining estimates. Both conventions are retained and clearly labeled. The
raw `post`, `effyear` and `time_til` fields are never overwritten. The raw
`time_til` uses negative infinity for never-treated observations; that unused
source field is retained, while analysis event times are constructed explicitly.

## Specification and inference

The model saturates all supported cohort/event-time interactions, omitting
event time −1, with state and calendar-year fixed effects. There are 50
estimated cohort-time cells in each specification. No leads/lags are binned or
excluded from the regression. Selecting an event window afterward only selects
reported aggregates.

The R reference uses:

```r
feols(
  l_homicide ~ sunab(gref, year, ref.p = -1) | state + year,
  data = data, vcov = ~state, fixef.rm = "none", fixef.tol = 1e-10,
  ssc = ssc(K.adj = FALSE, G.adj = FALSE)
)
```

`gref` maps never-treated states to a date beyond the sample for fixest. The
Python analysis maps them to zero. The reference period is explicit; defaults
are not relied on. Cohort/observation aggregation shares are held fixed, matching
fixest. No estimated-share variance term is added for SA.

CRV0 uses unadjusted state-cluster score outer products. Family confidence
intervals use normal critical values; fixest's default t(49) p-values and
intervals are not asserted to match. CRV1 switching is tested separately as
the documented 50/49 covariance multiplier. All 50 state clusters remain in use.

The baseline is unweighted. Two supplementary variants use the source `popwt`,
which is positive and constant within each state in this snapshot. Those are
sampling-weight sensitivities, not a full reconstruction of the original
paper's population-weighted regressions with additional controls.

## Results and timing sensitivity

| Timing | Sampling weights | Post-treatment ATT, log points | State-cluster SE |
| --- | --- | ---: | ---: |
| Adoption year (`effyear`), baseline | None | 0.1103830355 | 0.0386019208 |
| Adoption year | Source `popwt` | 0.1081768429 | 0.0240661021 |
| Following year (matches raw `post`) | None | 0.0194028080 | 0.0341850722 |
| Following year | Source `popwt` | 0.0785429936 | 0.0261900358 |

Timing materially changes the unweighted estimate. Agreement between software
implementations does not resolve that substantive choice. These results verify
an explicitly specified estimator on the empirical data; they do not establish
that the assumptions of a causal analysis hold.

```{image} figures/castle_sunab.light.svg
:class: only-light
:alt: Sun–Abraham event-study estimates compared with R fixest and contributing state counts
```

```{image} figures/castle_sunab.dark.svg
:class: only-dark
:alt: Sun–Abraham event-study estimates compared with R fixest and contributing state counts
```

The R markers overlap the Python estimates. The hollow marker at −1 is a
normalized reference, not an estimated zero with a confidence interval. The
bar panel exposes changing support: both extreme horizons have just one
contributing treated state. Several cohorts are small. Apparent significance
at thinly supported horizons is not evidence of reliable small-sample inference.

The selected baseline pre-trend test for events −4, −3 and −2 has
chi-squared statistic 5.1783 (3 restrictions), p=0.1592. The corresponding
following-year specification has p=0.03269; population-weighted versions have
p=0.2397 and p=0.002189 respectively. These are specified joint tests, not a
certificate of parallel trends. Other pre-periods remain visible in the plot.

## Validation coverage

All four specifications pass comparisons of:

- Group-time coefficients, standard errors and the full 50×50 covariance.
- Dynamic, simple ATT and cohort estimates/SEs against actual fixest
  aggregation. Their full covariance is propagated from fixest's cell covariance
  using its observed model-matrix shares, independently of Python's map.
- Calendar aggregates and a balanced 0..2 exposure window against explicit R
  linear maps; no native fixest calendar/balancing shortcut is claimed.
- Independent full-unit/year-dummy OLS or WLS with cluster-score covariance,
  calculated separately in both R and Python.
- The selected dynamic pre-trend chi-squared statistic and p-value.

Across four specifications, the largest fixest cell-estimate difference is
2.96e-14 and largest full-cell covariance difference 1.34e-15. The comparison
tolerance is absolute 1e-9 with no relative tolerance; the pre-trend statistic
uses absolute 1e-8. Python dense-oracle differences are below 8.3e-15 for
estimates and 3.6e-16 for covariance. Reference versions are R 4.5.3 and
fixest 0.14.2.

Eight regression checks cover source hashes, source columns and sample counts,
both explicit timing mappings, four full reference/dense-oracle comparisons,
shuffled lazy/Parquet sources, small row groups/batches, CRV switching,
saturated-model preservation under reporting windows, and thin-support counts.
No Stata run is required or claimed for this example.

## Run and reproduce

```bash
STREAM_LIMITS_MAX_THREADS=2 POLARS_MAX_THREADS=2 \
  python examples/castle.py

# Frozen Python-only numerical report:
STREAM_LIMITS_MAX_THREADS=2 POLARS_MAX_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python benchmarks/castle_replication.py

# Also regenerate standalone SVGs, PDF and PNG (requires matplotlib):
MPLCONFIGDIR=/tmp/diff-stream-mpl STREAM_LIMITS_MAX_THREADS=2 \
  POLARS_MAX_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python benchmarks/castle_replication.py --plot

# Offline reference refresh from the frozen original R object:
python benchmarks/castle_replication.py --regenerate --rscript /path/to/Rscript
```

The refresh uses the isolated `benchmarks/external/.r/library`; the reference
packages must already be installed. It performs no download or install.
The normal offline runner verifies input/source/exporter/reference hashes.

Artifacts:

- `tests/data/castle.csv`: original source columns in stable state/year order.
- `tests/data/castle_source/`: pinned R object, version/license notices and hashes.
- `tests/data/castle_reference/`: R coefficients, full covariance, dense references,
  aggregates, pre-trend tests, versions and provenance manifest.
- `benchmarks/results/castle_replication/report.json`: all four specifications,
  samples, discrepancies, conventions and hashes of inputs, scripts and references.
- The same output directory contains coefficient/contribution CSVs and,
  with `--plot`, `event_study.pdf` and `event_study.png`.

The GitHub input is pinned and frozen. A network connection, R and Stata are
unnecessary for subsequent Python-only checks.
