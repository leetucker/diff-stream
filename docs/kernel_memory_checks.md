# Kernel-enforced memory validation on WSL

The selected host matrix has **12 cases: 11 completed fits, 1 clean headroom refusal, 0 unexpected failures**. Every case ran outside the Codex sandbox, as the ordinary user, inside its own kernel-enforced cgroup v1 memory limit. The supervisor stayed outside the case group.

## Verifying that a limit exists

An unprivileged `systemd-run --user` service accepted `MemoryMax=64M` and then successfully allocated 128 MiB. Its memory-controller membership remained at the unrestricted user-manager parent. This was **not an enforced limit**. The failed assumption was treating an accepted service property as proof of enforcement. systemd documents that user-instance resource controls are disabled under the [legacy hierarchy](https://www.freedesktop.org/software/systemd/man/247/systemd.resource-control.html).

The system-manager route required interactive authentication; noninteractive sudo also required a password. The user therefore ran the reviewable setup script in their terminal, delegating an empty, fixed-purpose test parent. That parent has a root-owned 3 GiB hierarchical cap. The script changes neither controller mounts nor existing system/user-service limits, and it moves no processes.

The direct v1 runner then verified all of the following:

- Kernel readback of the requested 64 MiB case limit.
- A 16 MiB control allocation completes successfully.
- `stream_limits.detect_memory_gb(environ={})` detects **0.067108864 GB**, exactly 64 MiB, without an environment-variable budget masking detection.
- A 256 MiB allocation exits with **-9** and the kernel records **1 OOM kill**.
- Kernel peak charged memory is **67,108,864 bytes**.

The runner refuses to certify enforcement from a successful launch, a signal alone, or sampled RSS alone. Each real fit gets a fresh group; an OOM kill in a fit is a failure rather than an accepted headroom refusal. Kernel readback must match the requested cap within page rounding.

## Fit results

All **11 successful fits** also match their prior RSS-only runs in cell estimates, full covariance, ATT and SE. Full comparison errors are retained in `numerical_invariance.json`.

The original larger-fit/source-layout cases use 1.0 GB budgets except the 10-million-row successful CS case, which uses 1.5 GB. Both richer million-unit models use 1.5 GB. Caps are decimal bytes, rounded down by the kernel to whole pages. All fits use two threads; exact inputs/configurations/source hashes remain in the per-case JSON and logs.

| Case | Outcome | Peak process RSS MiB | Peak cgroup-charged MB | Kernel OOM kills |
|---|---|---:|---:|---:|
| kernel_memory_checks/cs_n100000 | ok | 485.0 | 629.2 | 0 |
| kernel_memory_checks/cs_n1000000 | ok | 674.9 | 803.6 | 0 |
| kernel_memory_checks/cs_n2000000 | ok | 862.2 | 1293.1 | 0 |
| kernel_memory_checks/cs_n2000000_1gb | guard_refused | 780.6 | 1000.0 | 0 |
| kernel_memory_checks/cs_wide | ok | 711.1 | 841.3 | 0 |
| kernel_memory_checks/cs_wide_strings_shuffled_large | ok | 721.4 | 765.1 | 0 |
| kernel_memory_checks/sunab_n1000000 | ok | 508.2 | 477.7 | 0 |
| kernel_memory_checks/sunab_strings_n1000000 | ok | 699.8 | 677.6 | 0 |
| kernel_memory_checks/bjs_n1000000 | ok | 524.7 | 467.1 | 0 |
| kernel_memory_checks/bjs_strings_n1000000 | ok | 697.7 | 638.2 | 0 |
| kernel_richer_resources/sunab_1000000 | ok | 791.5 | 954.0 | 0 |
| kernel_richer_resources/bjs_1000000 | ok | 769.9 | 835.6 | 0 |

The 10-million-row 1 GB probe refuses cleanly while kernel-charged memory reaches its cap. It is not a completed fit. The kernel can reclaim cache while the library independently checks RSS/headroom; reaching the charged cap does not by itself imply an OOM kill.

Source-layout counterparts check cell estimates, full covariance, ATT and SE against their named baselines. The richer models additionally require finite cell/aggregate covariance, clean-placebo inference where enabled, and scratch cleanup. The same generated samples were previously checked against numerical references; this matrix adds actual kernel enforcement.

Process RSS and cgroup charges are different quantities. Cgroups charge memory including file cache and kernel allocations, while shared pages can be charged to another group and still appear in process RSS. Therefore neither reported peak substitutes for the other. The v1 swap policy is `memory.memsw.limit_in_bytes` equal to the memory limit plus `memory.swappiness=0`; the v2 route uses `memory.swap.max=0`. See the [kernel v1 memory-controller documentation](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v1/memory.html).

These are finite selected tests, not a universal memory bound across model dimensions, source queries or runtimes. The host's v1 route is validated; the v2 branch still needs its own real-host integration run.

## What this means for stream-limits

The live control found no defect in v1 limit detection: stream-limits read the actual inherited 64 MiB cap correctly. Its current contract is detection, runtime thread/spill settings, RSS measurement and allocation headroom. `MemoryBudget` subtracts current process RSS and a reserve; it does not create cgroups or guarantee that arbitrary source operators stay below a hard cap.

The demonstrated gap is in whole-workload enforcement validation and in distinguishing requested budgets from verified kernel enforcement. RSS guard success and a service's accepted `MemoryMax` property are insufficient. Cgroup usage/pressure reporting would also make the difference between RSS headroom and charged job usage easier to inspect; adding that is a shared API/accounting design decision, not a DiD-specific numerical fix.

Controller discovery, explicit delegated v1/v2 child groups, readback, attachment before exec, kernel counters, cleanup and the positive/negative self-check now live in the opt-in stream-limits enforcement/launcher/testing modules. Its base installation remains standard-library only; launcher/testing extras add psutil without numerical dependencies. Ordinary imports and fits retain their defaults.

Only constrained-resource tests use the common subprocess fixtures. Without an explicit verified delegate they run under a sampled process-tree RSS watchdog on Linux, macOS and Windows; this is not a kernel hard cap. Native hard limits currently require Linux. Kernel-only proof skips without delegation. CI includes all three OSes; only Linux has been exercised locally. Estimator datasets and numerical assertions stay in diff-stream. No sibling estimator was migrated, and no launcher code is needed in hdfe_stream.core.

The measurements above predate the shared-core/Parquet-reader migration. See the [migration review](shared_core_review.md) for the separate follow-up runs.

## Reproduce on this host

Run outside the execution sandbox. First-time delegation requires the user's own administrator authentication; if the parent already exists, reuse it.

```bash
sudo /usr/bin/python3 benchmarks/setup_memory_cgroup.py --user lee
.venv/bin/python benchmarks/memory_checks.py --selfcheck-only \
  --cgroup-parent /sys/fs/cgroup/memory/diff-stream-tests-1000 \
  --results benchmarks/results/kernel_memory_checks
.venv/bin/python benchmarks/memory_checks.py \
  --cgroup-parent /sys/fs/cgroup/memory/diff-stream-tests-1000 \
  --results benchmarks/results/kernel_memory_checks \
  --only cs_n100000 cs_n1000000 cs_n2000000 cs_n2000000_1gb \
  cs_wide cs_wide_strings_shuffled_large sunab_n1000000 \
  sunab_strings_n1000000 bjs_n1000000 bjs_strings_n1000000
.venv/bin/python benchmarks/richer_resources.py \
  --cgroup-parent /sys/fs/cgroup/memory/diff-stream-tests-1000 \
  --results benchmarks/results/kernel_richer_resources \
  --only sunab_1000000 bjs_1000000
STREAM_LIMITS_TEST_CGROUP_PARENT=/sys/fs/cgroup/memory/diff-stream-tests-1000 \
  .venv/bin/python -m pytest tests/test_memory_layouts.py -q
.venv/bin/python benchmarks/host_memory_route.py
.venv/bin/python benchmarks/kernel_memory_report.py
# Optional administrator cleanup, after all children have finished:
sudo /usr/bin/python3 benchmarks/setup_memory_cgroup.py --user lee --remove
```

The setup script refuses to overwrite an existing group, and removal refuses a group that still has processes or children. Fits run as the ordinary user; the delegated parent remains available for reruns until explicitly removed. Raw host-route probes, self-checks and model runs are retained in `benchmarks/results/kernel_memory_checks` and `kernel_richer_resources`.
