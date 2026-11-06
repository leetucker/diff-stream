# Authors' Stata BJS validation

The returned Stata run passes all 16 baseline comparisons: dynamic/simple
aggregation, four smart averaging schemes, and plugin/leave-out covariance.
All 32 expected CSVs are present. The log closes normally without an estimation
error. Routine first-write “file not found” messages and exporter-generated
missing values are harmless; all returned estimates and covariance entries are finite.

The baseline contains 240 units and 1,200 observations, with balanced panels,
no sampling weights, no anticipation, and unconditional imputation. The reported
environment is Stata/MP 18.0 (15 May 2023), `did_imputation` November 22, 2023,
`reghdfe` 6.13.1 (10 January 2026), and `ftools` 2.50.0 (9 January 2026).

| Quantity | Largest absolute Python–Stata difference |
| --- | ---: |
| Estimate | 5.783e-8 |
| Standard error | 7.909e-9 |
| Full covariance entry | 2.780e-9 |

Comparisons use absolute tolerance 1e-7 with no relative tolerance. Every
covariance is symmetric and positive semidefinite within numerical tolerance;
exported squared standard errors match its diagonal. The original transfer
manifest matches the local input and exporter hashes. Raw returned files remain
unchanged. These are numerical correctness comparisons, not resource benchmarks.

Reproduce the audit from the repository root:

```bash
python benchmarks/stata_bjs_review.py
```

This writes `benchmarks/results/stata_bjs_review/report.json`, including
per-fit discrepancies, input provenance and hashes of all returned files.
The regression suite also checks these outputs when present.

## Extended results reviewed

The returned extended run completed all **176 fits** successfully, with all
**562 CSVs**, command source copies and a normally closed log. Covariance
matrices are finite, symmetric and positive semidefinite within numerical
tolerance; their diagonals agree with the exported standard errors.

| Quantity | Largest absolute difference |
| --- | ---: |
| Post estimate | 2.560e-7 |
| Post standard error | 2.216e-8 |
| Post covariance entry | 1.310e-8 |
| Placebo estimate | 1.008e-8 |
| Placebo standard error, SSC removed | 4.853e-9 |
| Placebo covariance entry, SSC removed | 2.191e-9 |
| Independent clean-regression estimate | 5.662e-15 |
| Independent clean-regression covariance entry, SSC removed | 3.539e-16 |

All post standard errors/covariances, placebo estimates/covariances and native
Stata F-test comparisons pass their specified tolerances. The largest F-statistic
difference is 1.497e-7 (tolerance 1e-6 plus relative 1e-7); its p-value difference
is 1.046e-8 (absolute tolerance 1e-7). Stata drops 20 singleton clean observations
in the `mpdta` placebo regression; its results agree after the stated corrections.

The original float-storage run has **18 post-estimate failures at the strict absolute 1e-7
tolerance**: all 16 `mpdta` specifications, plus the all-group leave-out dynamic
fits for weighted balanced anticipation-2 and weighted complete anticipation-2.
The other 158 pass. The focused double-storage run below resolves all 18 without
changing tolerance. The authors' returned command uses implicit storage types
for temporary weights, predictions and effects, which explains why importing
CSV inputs as double alone did not remove the differences. No estimator code
was changed during these reviews.

The initial reviewer also had a convention error: it used the authors'
`did_imputation` finite-sample factor for the separate native `reghdfe` clean
regression. With unit effects nested in unit clusters, the two factors differ:

```text
q_imputation = (N-1)/(N-df_m-df_a)   * G/(G-1)
q_reghdfe   = (N-1)/(N-df_m-df_a-1) * G/(G-1)
```

This follows the returned authors' source and the
[native reghdfe clustered covariance code](https://github.com/sergiocorreia/reghdfe/blob/master/src/reghdfe.mata).
Applying each factor to its own output removes the apparent clean-regression
SE mismatch to machine precision. Neither factor is applied to post effects.
The package continues to report its documented CRV0/CRV1 and chi-square
conventions; native Stata F inference is checked by an explicit conversion.

The hashed report is `benchmarks/results/stata_bjs_review/extended_report.json`.
Reviewing that original run alone still exits with status 1 and retains its
18 findings. The combined review links those findings to the passing precision
run, verifies identical data/specifications and installed command source hashes,
and closes the validation gate. The external regression check now passes using
both runs; it retains a check against new or worsening original-run differences.

## Focused precision confirmation

`benchmarks/results/stata_bundles/stata-bjs-precision-validation.zip` contains a focused rerun of the three
affected cases (48 fits). It derives inputs and options from the preserved
extended transfer ZIP. The only estimator-setting change is `set type double`
after `clear all`, so implicit temporary variables also use double storage.
It does not edit the installed commands or loosen estimation tolerances.

Extract it into a fresh folder, set Stata's working directory there and run
`do RUN_STATA.do`. Return its entire `results` folder to
`benchmarks/results/stata_bjs_precision/`; keep earlier results unchanged.
The returned precision run contains all **48 successful fits and 154 CSVs**,
with a normally closed log. All comparisons pass the original tolerances.

| Quantity | Largest absolute difference in precision run |
| --- | ---: |
| Post estimate | 3.738e-8 |
| Post standard error | 5.829e-11 |
| Post covariance entry | 1.227e-10 |
| Placebo estimate | 1.166e-15 |
| Placebo covariance entry, SSC removed | 6.731e-16 |
| Native F statistic | 6.240e-14 |

All 18 previously flagged specifications pass. The `did_imputation`, `reghdfe`
and `ftools` source copies are byte-identical between runs, and bundle hashes
verify that the only estimator-setting change is `set type double`. This
confirms default temporary-variable storage as the source of the previous
precision failures. The remaining differences are below the unchanged tolerance.
The independent native clean regression also agrees to machine precision.

`benchmarks/results/stata_bjs_review/precision_report.json` records the precision
audit; `validation_summary.json` records the resolved 176-fit matrix, with the
48 double-storage results replacing their original counterparts. All effective
post-estimate differences are below 7.805e-8. Raw outputs and the original
failed comparisons remain preserved. This completes the present BJS Stata
validation scope; no further Stata run is required for these findings. No local
Stata execution is claimed.

Rebuild it with `python benchmarks/build_stata_bjs_precision_bundle.py`.
Review its returned results with the same reviewer, passing its extracted
`diff-stream-stata-bjs-precision` folder as `--bundle` and a separate output path.
To reproduce the combined resolution using both extracted transfer bundles:

```bash
python benchmarks/stata_bjs_review.py \
  --directory benchmarks/results/stata_bjs_extended \
  --bundle /path/to/diff-stream-stata-bjs-extended \
  --precision-directory benchmarks/results/stata_bjs_precision \
  --precision-bundle /path/to/diff-stream-stata-bjs-precision \
  --output benchmarks/results/stata_bjs_review/validation_summary.json
```

## Preserved extended transfer bundle

`benchmarks/results/stata_bundles/stata-bjs-extended-validation.zip` contains everything needed for a second
Stata run. Extract it into a fresh folder, set Stata's working directory to
the extracted folder containing `RUN_STATA.do`, and run:

```stata
do RUN_STATA.do
```

Use the same installed commands if possible; no Python, R or repository checkout
is needed on that machine. Return the entire `results` folder as
`benchmarks/results/stata_bjs_extended/`, including logs, command source copies,
status CSV and any partial outputs from failed fits. Preserve the first run.

The matrix has 11 cases and 176 imputation fits:

- Balanced/native unbalanced panels, crossed with weights on/off and
  anticipation 0/2 (eight cases).
- Weighted complete-panel selection with anticipation 2.
- Weighted future-treated controls with anticipation 0.
- The public `mpdta` data with anticipation 0.

Each case crosses four averaging schemes, plugin/leave-out, and dynamic/simple
aggregation. Synthetic cases fit three clean leads; `mpdta` fits one.
All Python fits passed preflight. The external Stata results have now been
reviewed above; no local Stata execution is claimed. Successful execution
produces 562 CSVs including status and clean-regression diagnostics.

Inputs explicitly import as double and exports retain 17 significant digits.
For anticipation, dynamic labels are restored to actual treatment dates, and
simple ATT uses `wtr(actual_post)` to exclude anticipation rows. Complete-panel
selection is explicit in the Stata data and compared with Python's complete
mode applied to the incomplete input.

Stata's clean-placebo covariance includes a finite-sample factor that the
library's CRV0 convention omits. The bundle records the clean regression's
sample size, absorbed/model degrees of freedom and cluster count. The reviewer
divides each placebo covariance by its command-specific factor and checks the native Stata
F statistic/p-value separately from the library's chi-square test. Post-effect
covariance is compared directly. Post/placebo cross covariance is exported
but has no library counterpart and is outside the comparison.
These conventions follow the [authors' command source](https://github.com/borusyak/did_imputation/blob/main/did_imputation.ado).

Rebuild the bundle and repeat the Python preflight:

```bash
python benchmarks/build_stata_bjs_bundle.py
```

After results return, extract the preserved transfer ZIP and review against its
exact manifest and inputs:

```bash
python benchmarks/stata_bjs_review.py \
  --directory benchmarks/results/stata_bjs_extended \
  --bundle /path/to/diff-stream-stata-bjs-extended \
  --output benchmarks/results/stata_bjs_review/extended_report.json
```

Missing outputs, unsuccessful status rows, hash changes and numerical mismatches
fail the review. The strict point-estimate tolerance remains unchanged; a
focused double-storage rerun resolves the 18 original findings as recorded above.
