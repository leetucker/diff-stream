# Stronger local memory checks

Retained matrix: **23 cases, 20 completed fits, 3 clean headroom refusals**, 0 unexpected failures.
Same-sample comparisons: **11**, checking cell estimates, full covariance, simple ATT and its standard error; maximum absolute difference **1.29e-14** against the declared 2e-7 resource-check tolerance.

These are sequential fresh-process local development checks with fixed five-period/three-cohort designs. Generation runs separately and is excluded from fit measurements. They extend the [conditional/unbalanced resource matrix](resource_checks.md) and [model-complexity review](numerical_stress.md); they do not establish a public speed claim or flat memory at arbitrary scale. No estimator changes were made for this matrix.

## Measured outcomes

GB and MB are decimal; MiB is binary. Each case uses two threads and a 1.0 GB process budget/watchdog, except `cs_n2000000`, which uses 1.5 GB. Normal buffers are `preprocess_mb=16`, `batch_units=512`; `tight` cases use 4 MB and 128 units.

| Case | Observed rows | Peak RSS MiB | Whole-process seconds | Leaf files | Largest decoded leaf MB | Outcome |
|---|---:|---:|---:|---:|---:|---|
| cs_n10000 | 50,000 | 365.2 | 1.1 | 1 | 3.637506 | ok |
| cs_n100000 | 500,000 | 487.9 | 3.2 | 16 | 2.29824 | ok |
| cs_n1000000 | 5,000,000 | 638.4 | 27.0 | 192 | 1.95552 | ok |
| cs_n2000000 | 10,000,000 | 818.7 | 53.4 | 256 | 2.91456 | ok |
| cs_n2000000_1gb | 10,000,000 | 781.8 | 6.0 | — | — | guard_refused |
| cs_n2000000_large_groups | 10,000,000 | 773.7 | 7.3 | — | — | guard_refused |
| cs_n2000000_tight | 10,000,000 | 785.9 | 11.3 | — | — | guard_refused |
| cs_tiny_groups | 500,000 | 500.3 | 3.5 | 16 | 2.29824 | ok |
| cs_large_groups | 500,000 | 409.6 | 3.4 | 16 | 2.29824 | ok |
| cs_strings | 500,000 | 597.0 | 4.2 | 48 | 1.8684 | ok |
| cs_strings_shuffled_large | 500,000 | 597.4 | 4.4 | 48 | 1.8684 | ok |
| cs_unused_wide | 500,000 | 423.1 | 3.4 | 16 | 2.29824 | ok |
| cs_wide | 284,036 | 693.1 | 11.3 | 80 | 1.945809 | ok |
| cs_wide_strings_shuffled_large | 284,036 | 724.1 | 12.3 | 96 | 1.978922 | ok |
| cs_wide_tight | 284,036 | 678.1 | 28.4 | 256 | 0.657153 | ok |
| sunab_n100000 | 284,413 | 443.8 | 1.1 | 5 | 1.843872 | ok |
| sunab_strings_n100000 | 284,413 | 450.9 | 1.3 | 16 | 2.447333 | ok |
| sunab_n1000000 | 2,842,639 | 484.4 | 6.2 | 48 | 1.93088 | ok |
| sunab_strings_n1000000 | 2,842,639 | 690.6 | 9.9 | 192 | 2.053121 | ok |
| bjs_n100000 | 284,413 | 432.0 | 1.6 | 5 | 1.843872 | ok |
| bjs_strings_n100000 | 284,413 | 435.4 | 2.5 | 16 | 2.447333 | ok |
| bjs_n1000000 | 2,842,639 | 500.1 | 12.4 | 48 | 1.93088 | ok |
| bjs_strings_n1000000 | 2,842,639 | 699.9 | 15.9 | 192 | 2.053121 | ok |

## What the checks establish

Source-layout variants preserve the generated sample and cohort assignment. They exercise 257-row and up-to-524,288-row input groups, 101-character unit strings (a prefix plus 96 digits), shuffled rows within generation blocks, 64 unused numeric columns, 33 compiled conditional-CS columns, and native unbalanced SA/BJS. Actual physical row groups, input checksums, configurations, source hashes, phase peaks and complete inference results are retained per case. All completed fits must respect the decoded-leaf allowance and release their scratch files. Every successful layout/buffer variant must match its baseline's estimates and full covariance.

Nine additional regression checks cover all three estimators on balanced and unbalanced panels with long identifiers, shuffled rows, one large source row group, unused non-finite columns, recursive partitioning, bounded decoded leaves, inference equivalence and cleanup. The generator check verifies that layout-only variants preserve every estimation column. These are deterministic algorithm invariants; machine-specific RSS thresholds remain in the benchmark runner rather than ordinary unit tests.

## The 10-million-row boundary

The initial 20-case matrix completed 19 fits. The 2-million-unit conditional-CS case stopped during preparation at approximately 793 MiB under a 1 GB budget, with a clean headroom refusal and no numerical fit. Its raw result and the original nonzero summary are retained under `history/initial_matrix/`. A diagnostic experiment coalescing hash-writer row groups also refused near 789 MiB; it did not justify a production change.

The expanded matrix reruns that sample with 1.5 GB and independently probes the original 1 GB settings, larger source groups and tighter preprocessing. Those three explicitly designated boundary probes may complete or refuse cleanly; a refusal is counted separately and never presented as a completed fit. Watchdog kills, timeouts, missing comparisons and failed cleanup remain failures. A successful boundary probe must match the 1.5 GB fit.

The completed 1.5 GB run peaks at **818.7 MiB**, with nuisance-phase RSS at **453.4 MiB**. Observed growth is concentrated in preparation, rather than a retained all-unit score matrix. These measurements do not isolate reader metadata, runtime buffering and allocator retention from each other. Input layout and unit-label width matter, and reducing the decoded-leaf allowance alone need not reduce total RSS.

`stream-limits` normally reserves 15% of the declared budget in addition to the upcoming allocation. A refusal below 1 GB is therefore expected behavior when the remaining headroom is insufficient. The measured growth should still be budgeted: import overhead does not explain all large-sample RSS.

## Kernel enforcement

This original matrix used **sampled RSS watchdog plus process high-water RSS**. Its 5 ms watchdog and process high-water RSS do not impose a kernel cap. Subsequent [host validation](kernel_memory_checks.md) now verifies the selected larger and richer models under actual cgroup v1 limits on this WSL host: 12 cases, 11 completed fits, one clean headroom refusal and no OOM kills in fits.

The earlier read-only mounts and inaccessible systemd bus were sandbox observations. Outside it, systemd runs and cgroup mounts are writable, but memory uses v1. An unprivileged user service accepted MemoryMax without enforcement; administrator delegation of a capped test parent enabled the ordinary-user v1 runner. WSL itself was not a blocker.

The runner supports explicit delegated v1/v2 memory parents, kernel limit readback, a successful control allocation, stream-limits cap detection, and an oversized allocation with a recorded kernel OOM kill. Each fit uses a fresh child group and records kernel peak/counters; an OOM kill is a failed fit. The v1 route has been validated here; v2 still requires its own real-host integration run. STREAM_LIMITS_TEST_CGROUP_PARENT enables the real-kernel test.

Cgroup accounting includes charged file cache and kernel memory, so its peak and limit are not equivalent to process RSS. A scheduler-provided whole-job limit also includes the driver and generation if they execute inside that job. Both boundaries must be reported explicitly when repeating these checks elsewhere.

## Reproduce offline

```bash
python benchmarks/memory_checks.py
python benchmarks/memory_report.py
# With an explicitly delegated cgroup v1/v2 memory parent:
python benchmarks/memory_checks.py --results benchmarks/results/memory_checks_cgroup \
  --cgroup-parent /path/to/delegated/group
STREAM_LIMITS_TEST_CGROUP_PARENT=/path/to/delegated/group \
  python -m pytest tests/test_memory_layouts.py -q
```

Use `--only CASE...` to select cases; include or run their named baselines first. Results are atomic and resumable under a driver lock. Changed configurations/source hashes archive earlier measurements. Inputs live in ignored `benchmarks/data/memory_checks`; retained logs, measurements, input manifests, comparisons and histories live in `benchmarks/results/memory_checks`. The scripts use installed local packages and perform no downloads, package installations, cloud launches or sibling-package modifications.

After this memory review, time-varying weights and richer BJS/SA absorbed-FE models were prioritized before additional estimators. Their subsequent bounded-design implementation is documented in [richer models](richer_models.md).
