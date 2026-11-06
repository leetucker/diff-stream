# Numerical stress and model-complexity review

Local review, 4 November 2026. **Two implementation issues were found and fixed.**
The numerical matrix has 102 cases with their intended outcomes: 80 successful
fits matched actual R did/DRDID estimates and full covariance, and 22 explicitly
refused singular, ill-conditioned or separated designs. The resource matrix
has 31 cases: 28 successful fits, the expected 257-column refusal, and two
explained boundary failures retained without relabeling them as successes.

At completion of the stress review, the full regression suite passed **858 tests**,
including 47 new checks. The
99 targeted numerical/conditional-CS checks also pass on Polars 1.37.0; the
full suite uses Polars 2.1.0. The documentation builds, and the source archive
includes the frozen inputs, reference JSON and provenance. No
sibling package or immutable copied utility was changed. These results are
local development observations, not public speed comparisons or memory guarantees.

## Fixes

### Propensity convergence depended on covariate units

An exact-solution logit example has empirical success probabilities .2/.5/.8
at x=-1/0/1, giving intercept zero and slope log(4). Before the fix, multiplying
x by 1e12 made the raw coefficient step tiny enough to accept the starting
slope of zero. Multiplying x by 1e8 also stopped prematurely. This could affect
conditional CS IPW/DR estimates and nuisance-adjusted covariance.

The shared temporary nuisance solver now checks each coefficient step in
weighted RMS design units. Its unpenalized likelihood, coefficient tolerance,
line search, separation checks and maximum iteration count remain in place.
The scale is fixed from the first source pass, so this adds no source pass or
observation-sized retained state. It does not silently center or drop columns.

Regression tests cover the known MLE across covariate scales 1e-12 through
1e12, two batch sizes, unequal weights, and opposing-label boundary ties that
produce quasi-separation. End-to-end balanced/native unbalanced CS tests check
both estimates and full joint covariance under the unit changes.

`benchmarks/results/numerical_stress/initial_convergence_failure.json` preserves
the original failure and source hash. The improvement lives in
`diff_stream._core.nuisance`, remaining a migration candidate for
`hdfe_stream.core`.

### Memory admission counted tensors that are never allocated

The numerical-buffer estimate included effect-by-effect-by-covariate and
batch-by-effect-by-covariate terms. CS stores joint effect covariance without
a covariate axis; covariate state is in nuisance Gram systems and bounded
row designs. The estimate could therefore reject feasible complex fits.

The estimate now reflects those actual structures while retaining generous
allowances for Newton information, unbalanced outcome strata and contrast
derivatives. Two 64-period/31-cohort cases previously refused by the 2 GB
admission guard now complete at about 662 and 735 MiB peak RSS. The model
profile and admission check use the same corrected estimate. This remains a
heuristic allowance; the RSS watchdog is still necessary in these checks.

The count of effects and covariance geometry are DiD-specific, so this change
stays in `diff_stream.api`. It does not add a generic shared kernel.

## Numerical comparisons

The matrix crosses balanced and observed-row panels with regression/IPW/DR.
Cases include covariate scales 1e-12/1e12, outcome scales 1e-6/1e6, nearly
collinear columns, weak overlap, control trimming, rare cohorts, sampling
weights spanning approximately twelve orders of magnitude, weights aligned
with covariates, exact singularity, large offsets and separation.

| Comparison with R, 80 fits | Largest absolute difference |
| --- | ---: |
| Group-time estimate | 4.911e-8 |
| Full covariance entry | 1.292e-8 |
| Simple ATT | 1.135e-8 |
| Simple standard error | 1.956e-9 |

The comparison threshold is 2e-7 in canonical outcome units. Intentional unit
changes are compared with R on the equivalent canonical design; Python outcome
estimates and covariance are converted back by the known scale and its square.
This isolates estimator invariance from the reference solver's sensitivity to
units. Other comparisons use the actual stress design. Control trimming .8
uses custom wrappers calling the actual DRDID panel or repeated-cross-section
functions inside did's group-time/influence machinery; other cases use .995.

R versions are 4.5.3, did 2.5.1 and DRDID 1.4.0. Frozen compressed inputs and
18 external-reference regression cases cover very weak overlap, wide sampling
weights and borderline collinearity in `tests/data/numerical_stress/`, with
input/exporter hashes and versions. No installed R is needed for those tests.

Large offsets such as x+1e8 are explicitly refused as ill-conditioned, while
using the existing `I(x-1e8)` formula succeeds. Automatic centering is outside
this change. Outcome regression can remain identified in a quasi-separated
propensity design; the matrix expects it to succeed while IPW/DR refuse.

## Scaling with model complexity

Fresh Python processes use two threads and a 2 GB decimal library/RSS-watchdog
budget, `batch_units=512` and `preprocess_mb=16`. Inputs are generated separately
in bounded blocks and excluded from fit timing. Peak RSS includes imports and
allocator retention; scratch disk excludes the generated source data.

| CS design | Units | Periods | Cohorts | Columns | Effects | Fit seconds | Peak RSS MiB | Peak scratch MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 10,000 | 8 | 3 | 3 | 21 | 0.58 | 401 | 1.3 |
| Longer panel | 10,000 | 64 | 3 | 3 | 189 | 3.12 | 494 | 15.3 |
| Expanded design | 5,000 | 8 | 3 | 65 | 21 | 2.35 | 612 | 18.5 |
| Expanded design | 5,000 | 8 | 3 | 129 | 21 | 4.91 | 743 | 66.7 |
| Expanded design | 5,000 | 8 | 3 | 256 | 21 | 12.62 | 923 | 142.0 |
| Many comparisons | 6,000 | 64 | 31 | 3 | 1,953 | 9.23 | 662 | 9.4 |
| Many comparisons, wider design | 6,000 | 64 | 31 | 33 | 1,953 | 61.14 | 735 | 133.9 |
| Observed-row design | 16,000 | 24 | 8 | 33 | 184 | 11.55 | 642 | 46.8 |

The largest comparison design fits 978 propensity systems in nine shared source
passes. The 256-column fit uses 12 systems and nine passes. Source passes do
not grow one-for-one with the number of nuisance systems; arithmetic and
small-system solves still become more expensive with design width and system
count. Observed memory growth reflects real design/buffer/covariance costs,
while the earlier admission-growth issue was an estimate rather than an
allocated tensor.

The one-factor width sweep also includes 17, 33 and 193 columns. Designs of
17 columns and above share categorical formula machinery; the jump from the
three-column numeric model includes fixed formula-import costs. The previous
[resource checks](resource_checks.md) separately measured that import component.
SA and BJS also complete the 16-period/3-cohort and 32-period/15-cohort probes.

These are single-run measurements. File-cache state and concurrent local work
are not isolated; timings and memory differences are not stable speed ratios.
Disk is sampled every 200 ms and can miss short-lived files. Large covariance
matrices are summarized by shape/trace/hash in resource outputs so converting
millions of entries to JSON does not dominate the measured process peak.

## Retained boundaries and decisions

- **Sparse observed-row regressions:** the 4,000-unit, 24-period, 33-column
  entry/exit case has a nuisance stratum with only 17 observations for 33
  columns. Refusal is correct. The error now reports system index, observation
  count and width. The paired 16,000-unit case succeeds; no columns are dropped
  and no regularization is substituted.
- **Address-space cap:** the numeric baseline succeeds under Linux's 3 GiB
  `RLIMIT_AS`, but the expanded 33-column case fails. A diagnostic probe records
  a PyArrow allocation failure as virtual memory approaches that cap with only
  about 504 MiB resident. The same case succeeds under a 6 GiB address-space
  limit at about 574 MiB RSS. Address space and physical RAM are different
  resources. The failed result and probe are retained.
- **Physical-memory enforcement:** the local cgroup filesystem is read-only
  and the user systemd service is unavailable. The 2 GB RSS bound is a sampled
  watchdog, not an OS-enforced physical-memory limit. No such guarantee is
  inferred from the address-space checks.
- **Statistical support:** one deliberately weight-concentrated case has only
  about one effective control despite hundreds of physical controls. It agrees
  numerically with R, but that does not establish useful confidence-interval
  coverage. The diagnostics expose effective sample size. The subsequent [effective-sample policy](effective_sample.md) now warns below
  30 in a required CS stratum and offers explicit coefficient suppression.
  This changes reporting policy, not the stress-run estimates or reference comparisons.

No further correctness failure remains in the tested numerical matrix. Wider
model support or a different nuisance estimator is not needed for the fixes.
Quadratic effect covariance and increasing nuisance-system dimensions remain
real limits; these finite runs do not prove resource behavior for arbitrary
numbers of periods, cohorts, columns or source layouts.

## Reproduction and retained artifacts

```bash
STREAM_LIMITS_MAX_THREADS=2 POLARS_MAX_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python benchmarks/numerical_stress.py
python benchmarks/complexity_checks.py
python benchmarks/stress_report.py
pytest -q
```

The numerical script uses the existing isolated R installation by default;
`--rscript` selects another installation. No network access or installs are
performed. The complexity driver is resumable and validates configuration,
library and driver hashes before reusing a result. Earlier versions are
archived under each results directory's `history/`.

- `benchmarks/results/stress_summary.json`: compact numerical/resource summary.
- `benchmarks/results/numerical_stress/`: all Python results, actual R outputs,
  warnings, versions, configurations and source/input hashes.
- `benchmarks/results/complexity_checks/summary.csv`: all 31 resource cases.
- `benchmarks/results/complexity_checks/`: per-case configuration/result/logs,
  progress, phase peaks, passes, cleanup status and retained failures.
- `benchmarks/results/complexity_checks/as3g_probe.log`: allocation traceback
  with the OS address-space limit and resident/virtual memory at failure.

The resource driver exits nonzero for the sparse-stratum and 3 GiB boundary
failures; those expected-to-fit probes were not silently reclassified after
execution. The 257-column refusal is the explicit planned guard test.

Environment clarification: the read-only cgroup mounts and inaccessible user
systemd bus reported above were observations inside the Codex sandbox. The host
has running systemd and a writable v1 memory controller. After administrator
delegation of a capped test parent, [actual kernel validation](kernel_memory_checks.md)
passed 12 selected cases (11 completed fits and one clean refusal), including the
richer million-unit models. WSL itself did not require a different machine.
