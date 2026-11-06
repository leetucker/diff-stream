"""Render the retained local memory matrix without running fits or accessing a network."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/results/memory_checks"


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    results = [json.loads((OUT / (r["name"] + ".json")).read_text()) for r in summary["rows"]]
    differences = [r["invariance"] for r in results if r.get("invariance")]
    maximum = max(
        (d[k] for d in differences for k in ("beta_max_abs", "vcov_max_abs", "att_abs", "se_abs")),
        default=0,
    )
    lines = [
        "# Stronger local memory checks",
        "",
        f"Retained matrix: **{summary['cases']} cases, {summary['completed_fits']} completed fits, "
        f"{summary['safe_guard_refusals']} clean headroom refusals**, "
        f"{len(summary['unexpected_failures'])} unexpected failures.",
        f"Same-sample comparisons: **{len(differences)}**, checking cell estimates, full covariance, "
        f"simple ATT and its standard error; maximum absolute difference **{maximum:.3g}** "
        "against the declared 2e-7 resource-check tolerance.",
        "",
        "These are sequential fresh-process local development checks with fixed five-period/three-cohort "
        "designs. Generation runs separately and is excluded from fit measurements. They extend the "
        "[conditional/unbalanced resource matrix](resource_checks.md) and "
        "[model-complexity review](numerical_stress.md); they do not establish a public speed claim "
        "or flat memory at arbitrary scale. No estimator changes were made for this matrix.",
        "",
        "## Measured outcomes",
        "",
        "GB and MB are decimal; MiB is binary. Each case uses two threads and a 1.0 GB process "
        "budget/watchdog, except `cs_n2000000`, which uses 1.5 GB. Normal buffers are "
        "`preprocess_mb=16`, `batch_units=512`; `tight` cases use 4 MB and 128 units.",
        "",
        "| Case | Observed rows | Peak RSS MiB | Whole-process seconds | Leaf files | Largest decoded leaf MB | Outcome |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        fit = r["fit"]
        d = fit.get("diagnostics", {})
        leaf = d.get("max_decoded_leaf_bytes")
        lines.append(
            f"| {r['config']['name']} | {r['input']['n_rows']:,} | {r['peak_rss_mib']:.1f} | "
            f"{r['process_s']:.1f} | {d.get('partitions', '—')} | "
            f"{leaf / 1e6 if leaf is not None else '—'} | {r['status']} |"
        )
    lines += [
        "",
        "## What the checks establish",
        "",
        "Source-layout variants preserve the generated sample and cohort assignment. They exercise "
        "257-row and up-to-524,288-row input groups, 101-character unit strings (a prefix plus 96 "
        "digits), shuffled rows within generation blocks, 64 unused numeric columns, 33 compiled "
        "conditional-CS columns, and native unbalanced SA/BJS. Actual physical row groups, input "
        "checksums, configurations, source hashes, phase peaks and complete inference results are "
        "retained per case. All completed fits must respect the decoded-leaf allowance and release "
        "their scratch files. Every successful layout/buffer variant must match its baseline's "
        "estimates and full covariance.",
        "",
        "Nine additional regression checks cover all three estimators on balanced and unbalanced "
        "panels with long identifiers, shuffled rows, one large source row group, unused non-finite "
        "columns, recursive partitioning, bounded decoded leaves, inference equivalence and cleanup. "
        "The generator check verifies that layout-only variants preserve every estimation column. "
        "These are deterministic algorithm invariants; machine-specific RSS thresholds remain in "
        "the benchmark runner rather than ordinary unit tests.",
        "",
        "## The 10-million-row boundary",
        "",
        "The initial 20-case matrix completed 19 fits. The 2-million-unit conditional-CS case "
        "stopped during preparation at approximately 793 MiB under a 1 GB budget, with a clean "
        "headroom refusal and no numerical fit. Its raw result and the original nonzero summary "
        "are retained under `history/initial_matrix/`. A diagnostic experiment coalescing hash-writer "
        "row groups also refused near 789 MiB; it did not justify a production change.",
        "",
        "The expanded matrix reruns that sample with 1.5 GB and independently probes the original "
        "1 GB settings, larger source groups and tighter preprocessing. Those three explicitly "
        "designated boundary probes may complete or refuse cleanly; a refusal is counted separately "
        "and never presented as a completed fit. Watchdog kills, timeouts, missing comparisons and "
        "failed cleanup remain failures. A successful boundary probe must match the 1.5 GB fit.",
        "",
    ]
    baseline = next(r for r in results if r["config"]["name"] == "cs_n2000000")
    peaks = baseline["fit"].get("phase_peaks_mib", {})
    lines += [
        f"The completed 1.5 GB run peaks at **{baseline['peak_rss_mib']:.1f} MiB**, "
        f"with nuisance-phase RSS at **{peaks.get('nuisance', float('nan')):.1f} MiB**. "
        "Observed growth is concentrated in preparation, rather than a retained all-unit score "
        "matrix. These measurements do not isolate reader metadata, runtime buffering and allocator "
        "retention from each other. Input layout and unit-label width matter, and reducing the "
        "decoded-leaf allowance alone need not reduce total RSS.",
        "",
        "`stream-limits` normally reserves 15% of the declared budget in addition to the upcoming "
        "allocation. A refusal below 1 GB is therefore expected behavior when the remaining "
        "headroom is insufficient. The measured growth should still be budgeted: import overhead "
        "does not explain all large-sample RSS.",
        "",
        "## Kernel enforcement",
        "",
        f"This original matrix used **{summary['enforcement']}**. Its 5 ms watchdog "
        "and process high-water RSS do not impose a kernel cap. Subsequent "
        "[host validation](kernel_memory_checks.md) now verifies the selected larger "
        "and richer models under actual cgroup v1 limits on this WSL host: 12 cases, "
        "11 completed fits, one clean headroom refusal and no OOM kills in fits.",
        "",
        "The earlier read-only mounts and inaccessible systemd bus were sandbox "
        "observations. Outside it, systemd runs and cgroup mounts are writable, but "
        "memory uses v1. An unprivileged user service accepted MemoryMax without "
        "enforcement; administrator delegation of a capped test parent enabled the "
        "ordinary-user v1 runner. WSL itself was not a blocker.",
        "",
        "The runner supports explicit delegated v1/v2 memory parents, kernel limit "
        "readback, a successful control allocation, stream-limits cap detection, "
        "and an oversized allocation with a recorded kernel OOM kill. Each fit uses "
        "a fresh child group and records kernel peak/counters; an OOM kill is a failed "
        "fit. The v1 route has been validated here; v2 still requires its own real-host "
        "integration run. STREAM_LIMITS_TEST_CGROUP_PARENT enables the real-kernel test.",
        "",
        "Cgroup accounting includes charged file cache and kernel memory, so its peak and limit "
        "are not equivalent to process RSS. A scheduler-provided whole-job limit also includes "
        "the driver and generation if they execute inside that job. Both boundaries must be "
        "reported explicitly when repeating these checks elsewhere.",
        "",
        "## Reproduce offline",
        "",
        "```bash",
        "python benchmarks/memory_checks.py",
        "python benchmarks/memory_report.py",
        "# With an explicitly delegated cgroup v1/v2 memory parent:",
        "python benchmarks/memory_checks.py --results benchmarks/results/memory_checks_cgroup \\",
        "  --cgroup-parent /path/to/delegated/group",
        "STREAM_LIMITS_TEST_CGROUP_PARENT=/path/to/delegated/group \\",
        "  python -m pytest tests/test_memory_layouts.py -q",
        "```",
        "",
        "Use `--only CASE...` to select cases; include or run their named baselines first. Results "
        "are atomic and resumable under a driver lock. Changed configurations/source hashes archive "
        "earlier measurements. Inputs live in ignored `benchmarks/data/memory_checks`; retained "
        "logs, measurements, input manifests, comparisons and histories live in "
        "`benchmarks/results/memory_checks`. The scripts use installed local packages and perform "
        "no downloads, package installations, cloud launches or sibling-package modifications.",
        "",
        "After this memory review, time-varying weights and richer BJS/SA absorbed-FE models "
        "were prioritized before additional estimators. Their subsequent bounded-design implementation "
        "is documented in [richer models](richer_models.md).",
        "",
    ]
    (ROOT / "docs/memory_checks.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
