# Reference-population review: BJS, Gardner and linear ETWFE

The review supports adding a clean last-treated-window policy to all three
estimators. This is an extension of the existing CS/SA sample-selection policy,
with estimator-specific treatment of lead diagnostics. This document records
the historical review and implementation requirements. The extension is now
implemented; see [the reference-population guide](reference_groups.md) for the
current public options, lead conventions and validation.

## Findings

A permanent never-treated population is not intrinsically necessary. What is
needed is an identified model for untreated outcomes and a well-defined target
sample. Future cohorts already provide full-window clean controls under the
library's current normalization: if adoption minus anticipation is beyond the
sample, they normalize to cohort 0. Six review fits confirm that BJS, Gardner
and ETWFE already accept that case without discarding dates.

The remaining gap is a panel where **every cohort becomes affected within the
observed window**. With compulsory unrestricted date effects, the three models
cannot recover unrestricted effects after the last source of clean observations
disappears. An earlier-adopting population cannot supply untreated outcomes
merely because its adoption date differs.

| Estimator | Supported route without a permanent never-treated cohort | Important qualification |
|---|---|---|
| BJS | Fit the untreated-outcome model on clean observations and impute only where the model is identified. | All-treated dates lack identified date effects. Extra crossed FEs can create additional unsupported strata. |
| Gardner | Fit its clean first stage, then estimate the second stage only on observations with identified untreated predictions. | Specify which cohorts enter dynamic leads; the last reference cohort's leads need an explicit convention. |
| Linear ETWFE | Use a last-treated reference and stop before it becomes affected. | The authors' `notyet` option already selects the last cohort when no never-treated sentinel exists; the `never` option does not automatically do so. |

These conclusions agree with the authors' [BJS imputation-support documentation](https://github.com/borusyak/did_imputation/blob/main/did_imputation.sthlp),
the [Gardner implementation's no-never-treated warning](https://github.com/kylebutts/did2s_stata/blob/main/ado/did2s.sthlp),
and the [ETWFE reference implementation](https://github.com/grantmcdermott/etwfe/blob/d36673d3f07783aa297bbec31d6d8aabfc66243b/R/etwfe.R).
The common-window proposal does not impose stable treatment effects or
extrapolate untreated date effects into unsupported periods.

## Numerical audit

The audit explicitly restricts no-never-treated samples to
$t < g_{last}-A$, then supplies a within-window reference using the existing
entry points. Eight data/specification combinations cover balanced/unbalanced
panels, anticipation 0/1, equal weights, date-varying sampling weights,
covariates and crossed fixed effects where supported.

Forty restricted comparisons pass:

- Eight BJS dynamic point comparisons against R `didimputation` 0.5.1. These do
  not constitute new native Stata inference validation.
- Sixteen Gardner static/dynamic coefficient and full-covariance comparisons
  against pinned, unmodified authors' R `did2s` source, also checked with
  independent dense stacked equations.
- Sixteen linear ETWFE conditional/unconditional cell-effect and full-covariance
  comparisons against pinned, unmodified authors' R `etwfe`/`marginaleffects`,
  also checked with independent dense equations. The existing documented
  marginaleffects finite-cluster convention is aligned explicitly.

Maximum native point discrepancy is approximately **3.29e-11**; maximum aligned
joint-covariance discrepancy is **3.89e-13**. Two extra crossed-FE cases confirm
that BJS/Gardner correctly refuse a clean-stage design that remains unidentified
despite the global date restriction. This guards against treating a global last
cohort as sufficient support for every subgroup/date effect.

A complete-panel selection probe confirms the required order of operations:
dropping incomplete units first changes the last cohort from 7 to 6 and the
last clean date from 6 to 5. Truncating first would incorrectly rescue the
incomplete last cohort and select a different reference/target sample.

The 123 existing CS/SA reference-window checks also pass. No estimator source,
production defaults, resource policies or sibling packages changed in this
review; only audit code, retained evidence and documentation were added.

## Why reference-package output needs sample inspection

The audit also records forty unrestricted native R probes, with source pins,
warnings, errors and raw outputs preserved:

- R `didimputation` removes rows with missing untreated predictions. All eight
  full-window runs return one or two nonfinite unsupported horizon estimates.
  Valid earlier effects remain available, but a returned object is not proof
  that every originally requested horizon is identified.
- The pinned R Gardner implementation returns both static and dynamic fits in
  these examples after unavailable predictions are excluded. Its dynamic
  specification still includes the last cohort's clean leads when supplied
  their actual event times. That differs from assigning the last cohort zero
  second-stage design as a designated reference. Common lead estimates differ
  by as much as **0.1611** in the review; affected-period estimates agree to
  numerical precision. This is a change in the lead population, not evidence
  of biased post-treatment calculations.
- Pinned ETWFE `notyet` automatically selects the last cohort and excludes its
  affected dates. Four unweighted examples return successfully. Four weighted
  automatic-window examples fail during `emfx` recovery with a weight-length
  mismatch in the tested source/dependency combination. All explicitly
  restricted weighted reference fits succeed. The `never` option refuses all
  eight full-window samples without a never-treated reference, as expected.

These observations describe the tested source versions, not a claim that all
versions of the external packages behave identically. No external package code
was patched, and no messages or bug reports were sent to its maintainers.

## Lead conventions that implementation must preserve

For **BJS clean placebos**, preserve the original known adoption dates even when
a cohort is normalized to zero for post-effect estimation. The current BJS
placebo API deliberately uses original future-cohort timing. Relabeling the last
cohort permanently to zero before constructing placebo indicators changes the
pooled lead regression: the review observes differences up to **0.3440**. The
existing reference-window restager must therefore carry original cohort data
and its support metadata through the second staging pass. Simply reusing the
CS/SA restager would lose that information.

For **Gardner dynamic results**, the recommended explicit last-reference policy
assigns the designated last cohort zero second-stage design, consistently with
the reference role in CS/SA. Its lead coefficients then describe the earlier
target cohorts; that convention must be documented and used explicitly in the
R comparison. It should not be presented as identical to an unrestricted
native event study that pools the last cohort's clean leads. The reference
event time and the reference population remain separate concepts.

## Recommended implementation

1. Extend the existing `reference_group="auto"`, `"nevertreated"`,
   `"lasttreated"` interface to BJS, Gardner and linear ETWFE. Auto retains an
   available full-window clean reference; otherwise it selects the last cohort
   and reports a restricted window. Explicit `nevertreated` retains strict
   full-window-reference behavior. ETWFE's separate `control_group` continues
   to choose its regression specification; this extension does not redefine
   that option.
2. Apply the original balanced/complete/unbalanced policy first, then retain
   only dates strictly before `g_last-anticipation`. Reuse the current warning
   and dropped-row/unit/cutoff diagnostics. Recompute sampling-weight anchors,
   nuisance moments, cohort shares, margins and aggregations on the retained
   sample. Do not reuse full-window covariance or aggregation weights.
3. Preserve original BJS timing/support for clean placebos, and explicitly
   exclude the designated reference cohort from Gardner second-stage effects.
   Do not report post-treatment effects for the last reference cohort.
4. Keep first-stage rank/support checks after restriction. Require clean
   observations in each needed date/stratum, sufficient pre-treatment unit
   support and identified target contrasts. A clean observation somewhere in
   each date is necessary but does not guarantee connectivity in an unbalanced
   panel or identification with crossed FEs.
5. Add public-API tests for reference selection, complete-panel ordering,
   anticipation, weights, future controls, leads, aggregations and cleanup.
   Promote the relevant audit cases into frozen reference tests. Add selected
   shared-launcher resource probes for the additional staging pass. Existing
   guarded bounded kernels suffice; no new hdfe_stream.core primitive is
   required for this extension.

The recommendation is a conservative common-window extension. General
stratum-specific automatic imputation-support trimming, always-treated unit
effects, or extrapolation after universal treatment would be separate designs.
At review time, refit bootstrap was the next extension after reference-policy
validation. Serial and parallel pointwise refit bootstrap have since been
implemented; see the [current release handoff](release_handoff.md) for remaining
decisions and the explicit deferral of simultaneous refit bands.

## Reproduction

```bash
# Offline audit using retained native references
.venv/bin/python benchmarks/reference_population_review.py

# Regenerate native results using the existing local R/cache setup
.venv/bin/python benchmarks/reference_population_review.py --regenerate \
  --rscript /path/to/Rscript
```

Raw outputs and hashes are in
`benchmarks/results/reference_population_review/`. Gardner uses source commit
`3adaf985980d4b46205de2e1856f93ba075baefe`; ETWFE uses
`d36673d3f07783aa297bbec31d6d8aabfc66243b` (development version 0.6.2.99).
The cache is verified against its source hashes before native regeneration.
Installed dependency versions are captured in `R_session.txt`; no R runtime
dependency is added to diff-stream.
