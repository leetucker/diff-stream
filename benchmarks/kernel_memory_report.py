"""Summarize measured kernel-cap evidence and the shared-infrastructure boundary."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks import resource_checks as h  # noqa: E402

BASE = ROOT / "benchmarks/results/kernel_memory_checks"
RICH = ROOT / "benchmarks/results/kernel_richer_resources"


def main():
    check = json.loads((BASE / "cgroup_selfcheck.json").read_text())
    rows = []
    comparisons = []
    unexpected, selfchecks = [], []
    for directory in [BASE, RICH]:
        summary = json.loads((directory / "summary.json").read_text())
        unexpected.extend(summary["unexpected_failures"])
        selfchecks.append(json.loads((directory / "cgroup_selfcheck.json").read_text())["passed"])
        for row in summary["rows"]:
            result = json.loads((directory / (row["name"] + ".json")).read_text())
            result["suite"] = directory.name
            rows.append(result)
            if result["status"] == "ok":
                original = (
                    ROOT
                    / "benchmarks/results"
                    / ("memory_checks" if directory == BASE else "richer_resources")
                    / (row["name"] + ".json")
                )
                comparison = h.compare(result, json.loads(original.read_text()))
                comparisons.append(
                    dict(case=row["name"], reference=str(original.relative_to(ROOT)), **comparison)
                )
                if not comparison["passed"]:
                    unexpected.append(row["name"])
    completed = sum(r["status"] == "ok" for r in rows)
    refused = sum(bool(r.get("safe_guard_refusal")) for r in rows)
    unexpected.extend(
        r["config"]["name"]
        for r in rows
        if not r["cleanup_ok"] or r["cgroup"]["events"]["oom_kill"]
    )
    unexpected = sorted(set(unexpected))
    lines = [
        "# Kernel-enforced memory validation on WSL",
        "",
        f"The selected host matrix has **{len(rows)} cases: {completed} completed fits, "
        f"{refused} clean headroom refusal, {len(unexpected)} unexpected failures**. "
        "Every case ran outside the Codex sandbox, as the ordinary user, inside its own "
        "kernel-enforced cgroup v1 memory limit. The supervisor stayed outside the case group.",
        "",
        "## Verifying that a limit exists",
        "",
        "An unprivileged `systemd-run --user` service accepted `MemoryMax=64M` and "
        "then successfully allocated 128 MiB. Its memory-controller membership remained "
        "at the unrestricted user-manager parent. This was **not an enforced limit**. "
        "The failed assumption was treating an accepted service property as proof of enforcement. "
        "systemd documents that user-instance resource controls are disabled under the "
        "[legacy hierarchy](https://www.freedesktop.org/software/systemd/man/247/systemd.resource-control.html).",
        "",
        "The system-manager route required interactive authentication; noninteractive sudo "
        "also required a password. The user therefore ran the reviewable setup script in "
        "their terminal, delegating an empty, fixed-purpose test parent. That parent has a "
        "root-owned 3 GiB hierarchical cap. The script changes neither controller mounts nor "
        "existing system/user-service limits, and it moves no processes.",
        "",
        "The direct v1 runner then verified all of the following:",
        "",
        "- Kernel readback of the requested 64 MiB case limit.",
        "- A 16 MiB control allocation completes successfully.",
        f"- `stream_limits.detect_memory_gb(environ={{}})` detects "
        f"**{check['control']['fit']['detected_memory_gb']:.9f} GB**, exactly 64 MiB, "
        "without an environment-variable budget masking detection.",
        f"- A 256 MiB allocation exits with **{check['result']['returncode']}** "
        f"and the kernel records **{check['cgroup']['events']['oom_kill']} OOM kill**.",
        f"- Kernel peak charged memory is **{check['cgroup']['memory_peak_bytes']:,} bytes**.",
        "",
        "The runner refuses to certify enforcement from a successful launch, a signal alone, "
        "or sampled RSS alone. Each real fit gets a fresh group; an OOM kill in a fit is a "
        "failure rather than an accepted headroom refusal. Kernel readback must match the "
        "requested cap within page rounding.",
        "",
        "## Fit results",
        "",
        f"All **{len(comparisons)} successful fits** also match their prior RSS-only "
        "runs in cell estimates, full covariance, ATT and SE. Full comparison errors "
        "are retained in `numerical_invariance.json`.",
        "",
        "The original larger-fit/source-layout cases use 1.0 GB budgets except the "
        "10-million-row successful CS case, which uses 1.5 GB. Both richer million-unit "
        "models use 1.5 GB. Caps are decimal bytes, rounded down by the kernel to whole "
        "pages. All fits use two threads; exact inputs/configurations/source hashes remain "
        "in the per-case JSON and logs.",
        "",
        "| Case | Outcome | Peak process RSS MiB | Peak cgroup-charged MB | Kernel OOM kills |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        cg = r["cgroup"]
        lines.append(
            f"| {r['suite']}/{r['config']['name']} | {r['status']} | "
            f"{r['peak_rss_mib']:.1f} | {cg['memory_peak_bytes'] / 1e6:.1f} | "
            f"{cg['events']['oom_kill']} |"
        )
    lines += [
        "",
        "The 10-million-row 1 GB probe refuses cleanly while kernel-charged memory "
        "reaches its cap. It is not a completed fit. The kernel can reclaim cache "
        "while the library independently checks RSS/headroom; reaching the charged cap "
        "does not by itself imply an OOM kill.",
        "",
        "Source-layout counterparts check cell estimates, full covariance, ATT and SE "
        "against their named baselines. The richer models additionally require finite "
        "cell/aggregate covariance, clean-placebo inference where enabled, and scratch "
        "cleanup. The same generated samples were previously checked against numerical "
        "references; this matrix adds actual kernel enforcement.",
        "",
        "Process RSS and cgroup charges are different quantities. Cgroups charge memory "
        "including file cache and kernel allocations, while shared pages can be charged "
        "to another group and still appear in process RSS. Therefore neither reported "
        "peak substitutes for the other. The v1 swap policy is `memory.memsw.limit_in_bytes` "
        "equal to the memory limit plus `memory.swappiness=0`; the v2 route uses "
        "`memory.swap.max=0`. See the [kernel v1 memory-controller documentation](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v1/memory.html).",
        "",
        "These are finite selected tests, not a universal memory bound across model "
        "dimensions, source queries or runtimes. The host's v1 route is validated; the "
        "v2 branch still needs its own real-host integration run.",
        "",
        "## What this means for stream-limits",
        "",
        "The live control found no defect in v1 limit detection: stream-limits read "
        "the actual inherited 64 MiB cap correctly. Its current contract is detection, "
        "runtime thread/spill settings, RSS measurement and allocation headroom. "
        "`MemoryBudget` subtracts current process RSS and a reserve; it does not create "
        "cgroups or guarantee that arbitrary source operators stay below a hard cap.",
        "",
        "The demonstrated gap is in whole-workload enforcement validation and in "
        "distinguishing requested budgets from verified kernel enforcement. RSS guard "
        "success and a service's accepted `MemoryMax` property are insufficient. "
        "Cgroup usage/pressure reporting would also make the difference between RSS "
        "headroom and charged job usage easier to inspect; adding that is a shared "
        "API/accounting design decision, not a DiD-specific numerical fix.",
        "",
        "Controller discovery, explicit delegated v1/v2 child groups, readback, attachment "
        "before exec, kernel counters, cleanup and the positive/negative self-check now "
        "live in the opt-in stream-limits enforcement/launcher/testing modules. Its base "
        "installation remains standard-library only; launcher/testing extras add psutil "
        "without numerical dependencies. Ordinary imports and fits retain their defaults.",
        "",
        "Only constrained-resource tests use the common subprocess fixtures. Without an "
        "explicit verified delegate they run under a sampled process-tree RSS watchdog "
        "on Linux, macOS and Windows; this is not a kernel hard cap. Native hard limits "
        "currently require Linux. Kernel-only proof skips without delegation. CI includes "
        "all three OSes; only Linux has been exercised locally. Estimator datasets and "
        "numerical assertions stay in diff-stream. No sibling estimator was migrated, "
        "and no launcher code is needed in hdfe_stream.core.",
        "",
        "The measurements above predate the shared-core/Parquet-reader migration. See "
        "the [migration review](shared_core_review.md) for the separate follow-up runs.",
        "",
        "## Reproduce on this host",
        "",
        "Run outside the execution sandbox. First-time delegation requires the user's "
        "own administrator authentication; if the parent already exists, reuse it.",
        "",
        "```bash",
        "sudo /usr/bin/python3 benchmarks/setup_memory_cgroup.py --user lee",
        ".venv/bin/python benchmarks/memory_checks.py --selfcheck-only \\",
        "  --cgroup-parent /sys/fs/cgroup/memory/diff-stream-tests-1000 \\",
        "  --results benchmarks/results/kernel_memory_checks",
        ".venv/bin/python benchmarks/memory_checks.py \\",
        "  --cgroup-parent /sys/fs/cgroup/memory/diff-stream-tests-1000 \\",
        "  --results benchmarks/results/kernel_memory_checks \\",
        "  --only cs_n100000 cs_n1000000 cs_n2000000 cs_n2000000_1gb \\",
        "  cs_wide cs_wide_strings_shuffled_large sunab_n1000000 \\",
        "  sunab_strings_n1000000 bjs_n1000000 bjs_strings_n1000000",
        ".venv/bin/python benchmarks/richer_resources.py \\",
        "  --cgroup-parent /sys/fs/cgroup/memory/diff-stream-tests-1000 \\",
        "  --results benchmarks/results/kernel_richer_resources \\",
        "  --only sunab_1000000 bjs_1000000",
        "STREAM_LIMITS_TEST_CGROUP_PARENT=/sys/fs/cgroup/memory/diff-stream-tests-1000 \\",
        "  .venv/bin/python -m pytest tests/test_memory_layouts.py -q",
        ".venv/bin/python benchmarks/host_memory_route.py",
        ".venv/bin/python benchmarks/kernel_memory_report.py",
        "# Optional administrator cleanup, after all children have finished:",
        "sudo /usr/bin/python3 benchmarks/setup_memory_cgroup.py --user lee --remove",
        "```",
        "",
        "The setup script refuses to overwrite an existing group, and removal refuses "
        "a group that still has processes or children. Fits run as the ordinary user; "
        "the delegated parent remains available for reruns until explicitly removed. "
        "Raw host-route probes, self-checks and model runs are retained in "
        "`benchmarks/results/kernel_memory_checks` and `kernel_richer_resources`.",
        "",
    ]
    (ROOT / "docs/kernel_memory_checks.md").write_text("\n".join(lines))
    summary = dict(
        cases=len(rows),
        completed_fits=completed,
        safe_refusals=refused,
        unexpected_failures=unexpected,
        kernel_selfcheck=all(selfchecks),
        kernel_versions=sorted({r["cgroup"]["version"] for r in rows}),
        kernel_oom_kills_in_fits=sum(r["cgroup"]["events"]["oom_kill"] for r in rows),
        numerical_invariance=all(c["passed"] for c in comparisons),
    )
    (BASE / "combined_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (BASE / "numerical_invariance.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
