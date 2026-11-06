# Changelog

## 0.1.0.dev0 — unreleased

- Consolidate initial-release scope, outstanding extension decisions, dependency
  provenance and release checks in `docs/release_handoff.md`. Simultaneous
  refit-bootstrap bands are explicitly deferred pending an agreed end-to-end
  reference/convention; existing Gaussian joint and CS multiplier bands remain
  available. This records scope without changing estimator behavior.

- Add optional parallel refits (`workers>1`) to the unit-cluster bootstrap.
  Preserve serial defaults and draw plans, use fresh processes with thread caps
  applied before imports, and split the whole-job memory budget after reserving
  coordinator memory. Bound staged draws by worker count and stop/reap workers
  on fatal errors or interruption. Inference remains pointwise.

- Add ordinary unit-cluster full-refit bootstrap for CS, SA, BJS, Gardner and
  linear ETWFE, with pointwise SEs and percentile/basic/normal intervals.
  Preserve the original sample/reference window, give duplicated histories new
  panel IDs, recompute nuisance fits and shares, and record unsuccessful draws
  without replacement. Incomplete-run inference requires explicit acknowledgment.
  Serial disk-backed staging leaves ordinary estimator defaults unchanged.

- Extend `reference_group=auto/nevertreated/lasttreated` to BJS, Gardner and
  linear ETWFE. Auto falls back to a warned clean last-treated window when no
  full-window reference is available. Preserve original BJS adoption dates for
  clean placebos; assign Gardner's reference zero second-stage design including
  leads. Reference selection follows original panel policy, and weighting,
  inference and aggregation use the retained sample. Added 48 native reference
  configurations, sample-policy/identification tests and shared resource probes.

- Published `feTR` TWFE treatment-effect-weight diagnostic for single binary
  treatment, including reversals, native unbalanced panels, covariates and
  explicit date-varying sampling weights. Added both heterogeneity sensitivity
  thresholds with explicit degenerate statuses, bounded-pass computation and
  optional disk-backed cell export. Frozen authors' R TwoWayFEWeights 2.1.0
  references, independent projections/optimization and a public wage-panel
  example validate the implementation. Existing estimator defaults are unchanged.

- Covariate-adjusted balanced-panel Goodman–Bacon decomposition with separate
  within-timing and full-period between-timing components, plus always-treated
  groups and positive time-invariant sampling weights. Added public Castle data,
  native R and independent WLS validation, bounded projection batches and shared-
  launcher resource checks. Native unbalanced decomposition remains deferred;
  published TWFE-weight diagnostics are listed as a potential next task.

- Last-treated clean reference windows for CS/SA, with explicit/automatic policy,
  anticipation-aware date restrictions, sample diagnostics and strict never-
  treated opt-in. Added 108 native/precision R specification references.
- Optional CS covariate balance via shared core kernels: before/after propensity
  weights, sampling weights, observed-date unbalanced strata, fixed SMD scales
  and optional eCDF gaps. Reduced tables survive post-estimation operations.

- Linear ETWFE with saturated covariate interactions, never/not-yet-treated
  controls, cohort/unit fixed effects, weights, unbalanced panels and explicit
  conditional/unconditional inference. Reuses common group-time aggregation
  and bounded score accumulation. Added reference/dense checks, mpdta example
  and resource probes. Nonlinear users are directed to existing packages.

- Gardner two-stage DiD with direct static/dynamic results, corrected clustered
  inference, balanced/unbalanced panels, anticipation, sampling/time-varying
  weights and bounded covariate/crossed-FE first stages. Post-fit selection,
  reporting and pretrend tests retain the full fitted model. Added 48 frozen
  authors' R command references, independent dense covariance checks, a public
  Castle Doctrine reanalysis and shared-launcher resource/scaling probes.
  Existing-estimator pre-release extensions remain explicit in DESIGN_NOTES.md.

- Reuse the opt-in stream-limits launcher and resource fixtures. Constrained
  fits run with portable RSS watchdogs or an explicitly verified Linux kernel
  cap; ordinary tests/fits keep their defaults. Add macOS/Windows/Linux resource CI.
- Import promoted IPW and cluster-score primitives from hdfe_stream.core;
  remove three temporary copies and their manifest. Preserve CS's existing
  covariance correction. Disable Arrow pre-buffering in all ParquetFile readers.
  The new upstream APIs currently require the matching hdfe-stream source checkout;
  see SHARED_INFRASTRUCTURE.md for remaining migration candidates.

- Verified kernel-enforced memory limits on this WSL host through delegated
  cgroup v1: 12 cases, 11 completed fits, one clean headroom refusal, zero OOM
  kills in estimation. Added v1 support, kernel readback, control/cap-detection
  and deliberate-OOM proof, a reviewable setup command and host-route reporting.
  User-systemd MemoryMax was accepted without enforcement; the report records
  that finding and the shared stream-limits launcher/testing boundary.

- Explicit time-varying sampling weights for CS/SA/BJS, with R-compatible CS
  comparison dates and cohort-share anchors. SA/BJS now accept covariates and
  additional crossed fixed effects through bounded weighted unit elimination;
  the additional compiled design retains its 256-column limit. Protected target
  identification, BJS smart/leave-out and richer clean-placebo inference, frozen
  R did/fixest/dense-dummy references and independent Python regression checks.

- Expanded local memory matrix: 23 cases through 10 million rows, 20 completed
  fits, three clean headroom refusals and 11 estimate/full-covariance invariance
  comparisons. Long IDs, shuffled rows, source row-group extremes, unused wide
  inputs and tighter buffers receive nine additional regression checks. An
  optional delegated-cgroup runner and real-kernel integration test are provided;
  kernel-enforced validation remains pending on a suitable host. No estimator changes.

- Public Castle Doctrine Sun–Abraham reanalysis: pinned source data, four
  timing/weighting specifications, full fixest covariance/aggregation references
  and independent R/Python dummy-regression oracles. Eight regression checks,
  a Python-only audit and standalone event-study figures document the material
  adoption-year versus source-post timing sensitivity; no estimator changes.

- CS Kish ESS diagnostics for treated/control strata, with a default warning
  below 30, optional fit-time coefficient suppression and a copy-returning
  post-estimation `suppress()` method. Aggregation preserves original weights
  and propagates suppressed components; joint inference requires supported selections.

- Propensity-logit convergence now uses weighted RMS covariate units, preventing
  premature convergence after large changes of units; weighted/batched and
  end-to-end joint-inference regression coverage added.
- Corrected CS numerical-memory admission estimates to exclude unallocated
  covariate-indexed effect-covariance tensors; richer nuisance-stratum errors.
- Offline numerical stress matrix (102 cases, 80 actual R comparisons), 18 frozen
  external-reference cases and 31 local complexity probes through 256 columns
  and 1,953 effects, with raw failures, histories and a morning review report.

- Returned authors’ Stata BJS baseline: all 16 fits pass estimate, SE and full
  covariance checks, with input/output hashes and a repeatable audit.
- Extended authors’ Stata run completed all 176 fits; post covariance, clean
  placebos and converted native F inference checks pass. Corrected the reviewer’s
  distinct did_imputation/reghdfe nested-FE finite-sample adjustments.
- All 48 focused double-storage Stata fits pass, resolving the 18 original
  estimate-tolerance findings; unchanged data, specifications and command sources
  confirm temporary-storage rounding. Combined 176-fit validation passes with
  raw findings preserved, no estimator code changes and no tolerance relaxation.

- Frozen public math_reform Goodman–Bacon worked-example replication: all 16
  comparisons, package weights/contributions and type summaries, plus independent
  full-sample/pairwise OLS checks in R and Python.
- Offline runner/report with data/reference hashes, explicit treatment-label
  mapping and regression fixtures; no Stata dependency or estimator changes.

- Frozen mpdta documented-example replication against R did/DRDID across 24 CS
  specifications, full group-time/aggregate covariance, balanced exposure
  windows and pre-trend tests, plus matched-draw multiplier checks for baseline
  unconditional/conditional DR models.
- Offline report generation with data/script hashes, reference versions and
  regression fixtures; no Stata dependency for this empirical replication.
- Source distributions include compressed CSV reference fixtures.

- Positive time-invariant sampling weights for balanced/complete Goodman–Bacon
  decomposition, including weighted cohort means, comparison weights and WLS
  reconstruction; physical unit counts stay separate from source-scale masses.
- Exact-replication R bacondecomp and independent full-dummy WLS references,
  rescaling/equal-weight/layout checks and eight-million-row resource checks.

- BJS clean-placebo lead regression via `pretrends=L`, with unit/time effects
  fitted jointly on clean observations, including anticipation, sampling weights
  and balanced/complete/native unbalanced panels.
- Separate `BJSPlaceboResult`, observed support and joint pre-trend tests,
  available through aggregation, CRV switching and cleanup; post effects and
  their smart/leave-out inference remain unchanged.
- Original future-cohort timing retained for clean leads; explicit missing-lead,
  unidentified-design and memory errors, with no silently omitted columns.
- 48 dense R/fixest/didimputation lead references and 100,000-unit resource checks.

- Balanced equal-weight Goodman–Bacon decomposition with comparison estimates,
  nonnegative weights, contributions, exact time blocks and type summaries.
- Independent grouped TWFE/treatment-variance reconstruction checks; supports
  no-never-treated designs and explicit complete-panel deletion.
- Actual R bacondecomp/full-dummy OLS fixtures and independent dense Python
  checks; one-million-unit resource check and many-cohort output guard.
- BJS clean-placebo follow-up subsequently implemented with `pretrends=L` (above).

- Explicit CS/SA clean pre-trend Wald tests with cell/dynamic selection,
  anticipation-aware eligibility, reported restrictions and singularity checks.
- Optional CS unit-multiplier bands across all aggregations, including weighted
  nuisance scores, observed-row unit clustering and cohort-share uncertainty.
- Reuses hdfe-stream wild-bootstrap samplers through a bounded streaming score
  reducer; no units-by-repetitions or global score table is retained.
- Common-draw R mboot references and 10k/100k-unit bootstrap resource checks.

- Positive time-invariant sampling weights across CS, SA and BJS, including
  conditional nuisances, clustered covariance, cohort-share uncertainty,
  aggregation, anticipation and all panel modes.
- Source-scale weight totals reported separately from physical counts; weighted
  BJS retains bounded disk-backed aggregation data with explicit cleanup.
- Weighted R/dense references, scale/batch/partition checks and 100,000-unit
  resource checks; reusable validation and weighted numerical utilities in `_core`.

- Common scalar anticipation support for CS, SA and BJS, with clean reference/
  control/first-stage samples, retained actual cohort labels and anticipatory
  event-study cells. Actual post-treatment aggregation weights are preserved.
- Future cohorts retained when their anticipation window enters the sample;
  explicit no-clean-period errors, shifted event-window normalization and
  period-type diagnostics.
- Anticipation R/dense covariance fixtures for balanced/unbalanced panels,
  conditional CS and BJS smart/leave-out settings; reference metadata documents
  the installed R varying-base corner case and effective-onset validation.

- Offline, resumable conditional-CS/unbalanced resource matrix with bounded data
  generation, process-tree RSS/time watchdogs, phase telemetry, R comparisons,
  layout/thread/Polars invariance checks and retained per-case results.
- Shared logit nuisance solver now compensates objective sums and scales its
  line-search roundoff allowance, preventing excessive source scans at large N
  while retaining the existing coefficient convergence and estimator conventions.

- Cohort/group and calendar-time aggregation, preserving CS share uncertainty
  and BJS contrast-specific smart/leave-out inference.
- R-style post-exposure balancing and explicit balanced pre/post event windows,
  with interior-support checks and cohort-exclusion diagnostics.
- Contributor and composition tables showing weights, observed-cell and cohort
  counts, and changes in cohort membership/weights; inspection survives cleanup.
- Actual R aggte joint-covariance fixtures on balanced/unbalanced panels.

- Explicit balanced/complete/unbalanced sample policies across all estimators,
  with observed-row counts, dropped-unit diagnostics and unsupported-cell reports.
- Native unbalanced CS matches R did's observed-date row functionals and
  unit-clustered analytic inference; shared stratified OLS and normalized means
  live temporarily in `_core` for eventual promotion to hdfe_stream.core.
- Unbalanced SA uses exact within-cohort time-effect systems, avoiding wide
  row-by-interaction staging; BJS uses actual untreated
  histories, smart averaging and leave-out with bounded disk-backed aggregation.
- Actual R unbalanced CS/fixest fixtures and dense BJS joint-covariance checks.

- Covariate-adjusted CS regression, normalized IPW and traditional DR, with
  earlier-comparison-date covariates, nuisance-adjusted joint inference and
  explicit control-only propensity trimming/overlap diagnostics.
- Shared-pass nuisance fits, safe reuse for identical comparisons, bounded
  formula designs and numeric/categorical/interaction support for CS.
- Temporary `_core` copies of match-stream IPW/influence/M-estimation utilities,
  plus reusable nuisance/contrast, whole-group partition and design adapters;
  provenance and migration boundaries in SHARED_INFRASTRUCTURE.md.
- Actual R did/DRDID fixtures for 24 method/design/control/base combinations and
  additional explicit-trimming cases; local conditional benchmark harness.

- BJS smart residual averaging across cohort-time, event-time, cohort or all
  treated groups, recomputed per aggregate; optional cluster leave-out adjustment.
- BJS R package SE matching and full joint-covariance fixtures from an independent
  dense R oracle, plus an external authors' Stata comparison/export bundle.
- Bounded recursive unit-hash preprocessing replaces global unit grouping/sorting;
  pre-execution and per-phase memory guards, `preprocess_mb`, and disk-backed profiles.
- Polars 1.37 compatibility and a parent-monitored 1 GB RSS check on eight million rows.

Initial foundation:

- Balanced-panel BJS, saturated Sun–Abraham, and unconditional CS prototypes.
- Group-time results, dynamic/simple aggregation and analytic unit clustering.
- Streaming staging, parallel bounded score batches, job-budget guard and profiles.
- Dense numerical oracles, R exporters, Stata validation bundle and local benchmarks.
