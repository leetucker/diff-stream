# Shared infrastructure review

The November 2026 review moved generic subprocess enforcement and resource-test
fixtures into stream-limits and removed three duplicate treatment-effect modules
from diff-stream. No other estimation package was migrated.

## Launcher and test policy

The stream-limits base installation remains standard-library only. Its explicit
`launcher` and `testing` extras add psutil; neither requires Polars, NumPy, Numba
or an estimator. Normal imports, fits and thread/budget defaults are unchanged.
No launcher infrastructure belongs in `hdfe_stream.core`.

Only constrained-resource tests use the launcher. Common fixtures use a sampled
process-tree RSS watchdog on Linux, macOS and Windows; this can overshoot between
samples and is not a kernel hard cap. An explicitly configured, verified delegated
Linux memory controller selects kernel enforcement instead. Kernel-only proof
skips without delegation; the estimator resource tests still run using watchdogs.
Native hard-limit backends for macOS/Windows remain unimplemented. CI jobs cover
all three operating systems; local results below are from Linux/WSL only.

The reusable code owns child attachment, kernel limit/counter readback, cleanup,
a successful allocation, inherited-cap detection and a deliberate kernel OOM.
Diff-stream retains datasets, numerical assertions and model-specific memory
admission. See [kernel memory checks](kernel_memory_checks.md) for host setup.
The environment variable is now `STREAM_LIMITS_TEST_CGROUP_PARENT`.

## Shared numerical core

The IPW, influence and M-estimation copies were compared with hdfe-stream after
stripping docstrings. Every original function/class matches. Diff-stream imports
`ipw_weight` directly and uses `ClusterScoreAccumulator.add_clusters` for its
already per-unit CS scores. Raw `sums` preserve diff-stream's existing CRV0/CRV1
correction. The three local modules and their empty provenance manifest are gone.

The shared core also contains balance helpers, now used by the subsequent CS
balance milestone. At the migration milestone eight distinct bounded helpers
remained local, including nuisance/contrast systems,
whole-group byte-budget partitioning and multiplier inference. Their contracts
and potential consumers are listed in the repository's
[shared infrastructure notes](https://github.com/leetucker/diff-stream/blob/main/SHARED_INFRASTRUCTURE.md).

At the migration milestone these APIs were unreleased in a checkout labelled
0.8.0. Source version labels have since changed; see the
[release handoff](release_handoff.md) for current dependency provenance and
published-version verification work. Development and CI still use matching
checkouts. No fallback copies remain.

## Validation

- Stream-limits full suite: 93 passed, four kernel-only skips without delegation.
- Shared launcher plus diff-stream resource checks with the delegated host
  controller: 39 passed, including positive allocation and deliberate OOM proof.
- Diff-stream full suite after migration: 1,079 passed, one kernel-only skip.
- Upstream core API/influence/IPW/balance checks: 125 passed.
- Conditional CS and sampling-weight checks on Polars 1.37: 186 passed.

All five diff-stream `ParquetFile` readers explicitly use `pre_buffer=False`.
Batch iteration alone did not disable Arrow's selected-column read-ahead buffer;
whole-leaf reads remain bounded by the partition budget. Follow-up measurements
are retained separately from the earlier reports under
`benchmarks/results/shared_core_migration/kernel`. The definition audit is in
`benchmarks/results/shared_core_migration/ast_review.json`.

Six follow-up CS fits used the same delegated host controller. Each fit ran in a
fresh process. These are individual observations rather than repeated statistical
benchmarks; the review includes both the reader change and shared score reducer.

| Case | Earlier peak RSS (MiB) | Current peak RSS (MiB) | Current result |
|---|---:|---:|---|
| 100,000 units / 500,000 rows | 485 | 471 | Completed |
| 1 million units / 5 million rows | 675 | 511 | Completed |
| 2 million units / 10 million rows | 862 | 537 | Completed |
| Same 10 million rows, 1 GB limit | 781, then guard refusal | 530 | Completed |
| Wide conditional design | 711 | 650 | Completed |
| Wide design, shuffled long IDs and large row groups | 721 | 701 | Completed |

All six completed with no kernel OOM kills and passed partition-budget/cleanup
checks. Every available earlier successful estimate and full covariance matrix
matched exactly. The previously refused 1 GB case matches the current 1.5 GB
baseline exactly. Its kernel-charged peak was about 675 MiB, distinct from RSS;
the corresponding 1.5 GB run charged about 902 MiB. Kernel accounting includes
more than process RSS and varies with caching/reclaim. The raw before/after
comparison is `benchmarks/results/shared_core_migration/resource_comparison.json`.

The selected cases demonstrate improvement without establishing a universal
constant-memory guarantee. Other source operators and wider model designs still
need their own resource checks. Historical reports retain their original results.
