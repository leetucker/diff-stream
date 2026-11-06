# Initial-release scope and handoff

This is the current release-preparation checklist, consolidated on 2026-11-05.
It supersedes historical “next step” and “remains open” statements in
`DESIGN_NOTES.md`. That file retains the implementation history; this page
separates completed work, decisions still requiring review, explicit deferrals
and release verification. Preparing an initial release does not require adding
another estimator or implementing every possible extension.

## Explicit decision: defer simultaneous refit-bootstrap bands

The user has deferred simultaneous **refit-bootstrap** bands for now, including
the initial-release preparation scope. Do not implement them during cleanup
unless the user reopens the feature. Pointwise refit-bootstrap inference and
parallel refits are complete and remain available.

The reason is the absence of an agreed, fully specified end-to-end reference
for the proposed combination of ordinary unit-cluster resampling, full refits,
bootstrap-SD scaling and simultaneous critical values. This does not mean
simultaneous bootstrap inference is a novel or unsupported statistical idea.
R `did` supplies a close methodological reference: its
[`mboot()` source](https://github.com/bcallaway11/did/blob/master/R/mboot.R)
uses a maximum absolute standardized bootstrap deviation, but its draws are
influence-function multipliers, its fixed scale is IQR-based and its critical
quantile uses `type=1`. That is not an exact reference for the proposed refit
procedure. Independent base-R calculations were suggested as a numerical oracle,
but the full procedure/reference was not agreed before this deferral.

Reopening the feature requires specifying the coefficient family, centering,
scale and quantile conventions, zero/nonfinite-scale handling, failed-draw
policy, reference fixtures and coverage-validation plan. Fully studentized
bootstrap-t, BCa and other bootstrap designs remain separate future work.

This deferral does **not** remove the existing Gaussian joint intervals based
on fitted covariance, or CS influence-function multiplier simultaneous bands.
Those are different procedures; keep the documentation and APIs distinct.

## Completed implementation scope

| Area | Implemented scope and evidence |
|---|---|
| Callaway–Sant’Anna | Regression, IPW and traditional DR covariate adjustment; never/not-yet-treated controls; base-period options; balanced, explicit-complete and native-unbalanced panel policies; unit-clustered inference and estimated-share aggregation. R `did`/`DRDID` references and public `mpdta` replication. See [conditional CS](conditional_cs.md), [unbalanced panels](unbalanced_panels.md), [aggregation](aggregation.md), [mpdta](mpdta_replication.md). |
| Sun–Abraham | Saturated interaction regression; covariates and bounded additional/crossed FE designs; unit-clustered joint covariance and common aggregations. Native `fixest::sunab` references and the public Castle Doctrine application. See [richer models](richer_models.md), [Castle replication](castle_replication.md). |
| Borusyak–Jaravel–Spiess | Imputation, smart residual averaging/leave-out options, covariates and bounded additional FE designs; separate clean-placebo regression and pretrend tests. Authors’ Stata baseline, extended and double-precision validation are complete for their documented specifications. Richer specifications additionally use independent dense oracles; do not describe every richer model as Stata-validated. See [BJS placebos](bjs_placebos.md), [Stata validation](stata_validation.md), [richer models](richer_models.md). |
| Gardner | Direct static/event-study coefficients with first-stage-adjusted inference and bounded richer clean first stages. Pinned authors’ R command and independent dense stacked-WLS references. See [Gardner](gardner.md). |
| Linear ETWFE | The two authorized linear milestones: interacted regression, group-time margins and four common aggregations; cohort/unit FE choices, never/not-yet controls and conditional/unconditional margin/share inference. Pinned authors’ R `etwfe`/`emfx` and dense equation references. See [ETWFE](etwfe.md). |
| Goodman–Bacon | Balanced/explicit-complete decomposition; positive time-invariant unit weights; covariate adjustment and always-treated groups; TWFE reconstruction and component summaries. Native R references, public math-reform/Castle applications and independent WLS checks. See [Bacon](bacon.md), [extensions](bacon_extensions.md). |
| TWFE treatment-effect weights | Published binary-treatment `feTR` diagnostic, including reversals, observed-cell unbalanced panels, covariates, explicit date-varying weights and both heterogeneity-sensitivity thresholds. Authors’ `TwoWayFEWeights` references, independent projections/optimization and a public example. See [TWFE weights](twfe_weights.md) and `TWFE_WEIGHTS_PLAN.md`. |
| Shared panel/diagnostic work | Scalar anticipation; positive sampling weights and explicit time-varying opt-in; sample-composition/window diagnostics; last-treated reference policy across all five estimators; CS covariate balance; CS/BJS ESS warnings below 30 and optional coefficient suppression. See [reference groups](reference_groups.md), [weights](weights.md), [effective samples](effective_sample.md), [balance](covariate_balance.md). |
| Refit bootstrap | Ordinary unit-cluster pairs resampling, full nuisance/aggregation refits, pointwise sample-SD SEs and percentile/basic/uncorrected-normal intervals; optional parallel fresh-process refits. Fixed analysis/reference policy, fresh IDs, reproducible draw plans, no retries and explicit acknowledgment before incomplete-run inference. Serial remains the default. See [bootstrap](refit_bootstrap.md). |
| Resources/shared infrastructure | Bounded whole-unit staging, explicit Arrow `pre_buffer=False`, numerical guards, local complexity/layout stress tests and selected real Linux/WSL cgroup-v1 runs. Constrained tests use stream-limits’ portable fixtures; ordinary tests stay in process. Shared IPW/influence/balance APIs are consumed directly from hdfe-stream. See [shared-core review](shared_core_review.md), [kernel checks](kernel_memory_checks.md), `SHARED_INFRASTRUCTURE.md`. |

The five DiD estimators target absorbing-treatment **panels** with unit
clustering and consecutive positive integer dates. Supported panel policies,
clean-period requirements and treatment/reference conventions remain
estimator-specific. Do not infer a genuine repeated-cross-section API from
the native-unbalanced CS comparison functional. The separate TWFE-weight
diagnostic has its own broader observed-cell/time-support contract.

## Open pre-release extension decisions

The user requested revisiting these extensions before release and indicated
that they may want them included. They are **not implemented**, and they must
not silently disappear from the release discussion. Review them explicitly;
either agree and implement a bounded milestone, or record an explicit deferral.
They are not blanket authorization to develop every extension during cleanup.

| Decision | Current boundary | Next design work if selected |
|---|---|---|
| Richer BJS/SA absorbed-FE backend | Additional covariate/FE design is limited to 256 compiled columns, with local unit elimination and dense reduced systems. Unit-interacted FEs, varying slopes and an unrestricted high-dimensional absorbed backend are excluded. Time-varying weights and the bounded richer milestone are already complete. | Choose specific supported formulas and identification rules; review shared hdfe-stream infrastructure and resource contracts; establish native/reference covariance checks and model-complexity probes. This is architectural work, not a cleanup-sized promise. |
| CS genuine repeated cross sections | Current APIs require panel unit identities; observed-row unbalanced estimation is not a standalone repeated-cross-section interface. | Define sampling units, clustering, period-specific populations, weighting/aggregation and validation against R `did`/`DRDID`. |
| CS alternative nuisance estimators/options | Current regression/IPW/traditional-DR nuisance systems and documented trimming/weight rules are implemented. Alternative nuisance learners, DR variants and R `fix_weights` overrides are not implied by existing compatibility claims. | Identify the particular options wanted, their estimating equations/inference and a reference implementation before expanding the API. |

Keep these decisions separate from the explicitly deferred simultaneous bands.
No additional unfinished Gardner or linear ETWFE milestone is identified in
the authorized initial scope. Further extensions to those estimators need their
own scope rather than being inferred from a generic “finish the package.”

## Explicit deferrals and exclusions

- **Native-unbalanced Goodman–Bacon decomposition:** deferred by agreement.
  Explicit-complete mode remains available. An identified unbalanced TWFE need
  not contain complete 2×2 comparisons; do not invent a Bacon identity during
  cleanup. Date-varying Bacon weights are also outside the current scope;
  supported sampling weights are time-invariant within unit. The TWFE-weight
  diagnostic is a separate published diagnostic.
- **Nonlinear ETWFE and subgroup `xvar` effects:** outside the two authorized
  linear milestones. Keep the documented R `etwfe`, Stata `jwdid` and Python
  `diff-diff` alternatives for nonlinear users; they are not runtime backends.
- **Broader TWFE-weight methods:** `feS`, first-difference variants,
  continuous/multivalued treatment, multi-treatment contamination, generic
  microdata-to-cell aggregation, random-weight correlation tests and unrestricted
  absorbed effects remain deferred. Do not relabel `feTR` as full package parity.
- **Other bootstrap designs:** simultaneous refit bands as above, bootstrap-t,
  BCa, stratification, complex-survey replication and BJS-placebo/diagnostic
  targets beyond the current wrapper are not part of the completed milestone.
- **Additional estimators and publication-scale EC2 benchmarks:** no automatic
  next estimator and no EC2 launch is required for release cleanup. Existing
  local resource observations do not support universal memory or speed claims.
- **Generic-core promotion:** remaining bounded adapters, including resampling
  and process batches, stay in `diff_stream._core` for now. Their potential
  shared home is hdfe_stream.core; promotion is a separate API review, not a
  prerequisite to packaging this implementation.

## Release-preparation work

### Resolve dependency and environment provenance

- [ ] Verify which **published** hdfe-stream and stream-limits versions contain
  every imported API and resource behavior. Then update production/test/benchmark
  dependency floors and installation instructions to those real versions.
  Current `pyproject.toml` still permits hdfe-stream 0.8.0 and stream-limits
  0.1.1 in production, despite development using newer shared functionality.
  Do not choose floors from local version strings alone or restore fallback
  copies of removed kernels.
- [ ] Reinstall matching editable dependencies when continuing development:
  at handoff, local source metadata declares hdfe-stream **0.9.0** and
  stream-limits **0.2.1**, but this `.venv`'s installed distribution metadata
  reports **0.8.0** and **0.2.0.dev0**. Imports resolve to the sibling checkouts.
  These source labels do not establish package-index availability. Refresh and
  record metadata before claiming a released-version validation run.
- [ ] Test a clean environment installing the built diff-stream wheel and the
  chosen published dependencies, without editable checkouts/PYTHONPATH/source
  working-directory imports. Verify shared core imports, representative fits,
  parallel worker startup from outside the repository and resource extras.
- [ ] CI currently checks out sibling repositories rather than testing only
  released distributions. Make the release-path CI exercise supported published
  dependencies; retain a separate source-integration job if useful.

No match-stream runtime dependency is needed. Generic launcher/fixtures belong
to stream-limits; numerical shared kernels belong to hdfe-stream. Consult
`SHARED_INFRASTRUCTURE.md` before proposing any sibling changes; release cleanup
alone does not require new shared kernels or migrating the remaining adapters.

### Close validation and portability evidence

- [ ] Review actual CI results for Python 3.10/3.12/3.14 and Polars 1.37/current,
  plus the configured macOS/Windows portable resource jobs. Configuration is not
  evidence of execution. Latest local current Polars was 2.1.0; the complete
  suite was not run locally on every Python/OS combination.
- [ ] Review/add targeted irregular-time and sparse-support fixtures where
  coverage is missing. Preserve consecutive-date validation for the five DiD
  estimators; distinguishing global calendar gaps from within-unit missing rows
  is essential. Do not silently compress dates or expand the supported time
  contract. TWFE-weight gap support is documented separately.
- [ ] Obtain or explicitly record the status of real **cgroup-v2** integration
  for diff-stream. This WSL host validated cgroup v1; mocked/shared-backend tests
  are not a substitute for live diff-stream v2 evidence. Check existing
  stream-limits validation before duplicating infrastructure. Keep portable
  watchdog tests passing and kernel-only tests opt-in/skippable.
- [ ] Review whether any newly selected richer BJS specification needs an
  additional authors’ Stata bundle. Baseline/extended/precision Stata validation
  is complete; do not request all those runs again by default. Existing richer
  model oracles and their limitations are documented. Numerical changes need
  relevant reference checks without weakening tolerances to conceal differences.
- [ ] Preserve frozen reference inputs, provenance hashes, source-pinned
  exporters and raw reports. Routine tests need neither R/Stata nor downloads;
  regenerating references is a separate, explicit reproducibility operation.
- [ ] After actual fixes, run affected checks and the final regression/docs/build
  checks. Resource tests alone use the launcher. Re-run larger resource matrices
  only when resource-path changes or unresolved findings justify doing so.

### Finish the release artifact and documentation

- [ ] Review public defaults, reporting/cleanup semantics, error messages,
  scope limitations and examples. Distinguish Gaussian joint, CS multiplier
  and pointwise refit-bootstrap inference consistently.
- [ ] Resolve the open extension decisions above, then freeze the initial
  supported scope. Update README, changelog, this page and development-version
  notices together; do not imply deferred methods are implemented.
- [ ] Select the initial release version and synchronize package metadata,
  `diff_stream.__version__`, changelog and any release/citation metadata adopted
  by the maintainer. Finish repository/package metadata and review included
  upstream code/data provenance and distribution notices.
- [ ] Build and inspect wheel/sdist contents, verify a clean wheel installation,
  exclude caches/generated build/scratch files, and include docs/tests/reference
  assets intended for the source release. `MANIFEST.in` currently includes this
  guide through `recursive-include docs`.
- [ ] Review the VCS baseline before staging/committing. At this handoff all
  project source appears untracked in `git status`; do not mistake it for an
  empty implementation or discard it as generated output. Check ignores and
  intended benchmark/reference files deliberately.

## Validation baseline and commands for the receiving agent

Latest completed implementation checks, before this documentation handoff:

- Full suite: **1,745 passed, one kernel-only skip**.
- Bootstrap checks: **52 passed on Polars 1.37**; 17 parallel-specific checks
  also passed locally. Native CS/R refit comparisons, independent draw-copy
  checks across all five estimators, worker errors/interrupts and cleanup pass.
- Warnings-as-errors docs, focused Ruff checks and wheel/sdist content checks
  pass. This handoff edit is documentation-only; those historical results are
  not a claim of a newly rerun complete suite.
- Six matched serial/parallel kernel-capped probes (10k/100k units, five dates,
  four refits, two total threads, 2.5 GB cap) have identical draw hashes/SEs and
  no OOMs or scratch leaks. Parallel tree RSS is 1,071–1,349 MiB. Timings overlap
  other tests and do not establish isolated speedups. See
  `benchmarks/results/parallel_bootstrap_comparison.json` and the linked
  [bootstrap resource discussion](refit_bootstrap.md).

Continue with matching checkouts until published dependency floors are resolved.
Typical development commands from the repository root:

```bash
python -m pip install -e '../stream-limits[testing,launcher]'
python -m pip install -e '../hdfe-stream[formula]'
python -m pip install -e '.[test,docs,benchmark]'

STREAM_LIMITS_MAX_THREADS=2 POLARS_MAX_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m pytest -q
sphinx-build -W -b html docs docs/_build/html
python -m build
```

`python -m build` requires the development build frontend; it is not a runtime
dependency. The recent local artifact checks instead called setuptools’ build
backend directly because that frontend was absent. The `/tmp` logs/artifacts
from this session are conveniences, not durable release evidence.

Frozen-reference audits include `benchmarks/richer_validation.py`,
`gardner_validation.py`, `etwfe_validation.py`, `bacon_extended_validation.py`,
`twfe_weights_validation.py`, `reference_groups_validation.py`,
`reference_population_validation.py` and `refit_bootstrap_validation.py`.
Check each script’s `--help` before regeneration; the refit exporter uses
`--generate`, while several other exporters use `--regenerate`.
`benchmarks/README.md` and the estimator guides map additional public-data and
Stata checks. Avoid launching all expensive probes merely because they exist.
