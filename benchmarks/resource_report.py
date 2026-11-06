"""Render the retained offline resource-check results and light/dark figures."""

import json

from resource_checks import RESULTS, ROOT, accepted, cases


def load():
    rows = {}
    for path in RESULTS.glob("*.json"):
        value = json.loads(path.read_text())
        if "config" in value and "fit" in value and not path.name.endswith(".r.json"):
            rows[value["config"]["name"]] = value
    return rows


def plots(rows):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from plot_benchmarks import SLOTS, THEMES

    for theme, style in THEMES.items():
        with plt.rc_context(
            {
                "font.size": 10,
                "text.color": style["text"],
                "axes.labelcolor": style["text"],
                "xtick.color": style["muted"],
                "ytick.color": style["muted"],
                "axes.edgecolor": style["axis"],
            }
        ):
            fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), layout="constrained")
            fig.set_facecolor(style["surface"])
            for pattern, label, color, marker in [
                ("balanced", "Balanced", "blue", "o"),
                ("entry_exit", "Entry/exit", "orange", "s"),
            ]:
                selected = [rows.get(f"cs_n{n}_{pattern}_k3") for n in [10000, 100000, 1000000]]
                selected = [r for r in selected if r and r["status"] == "ok"]
                x = [r["config"]["data"]["n"] for r in selected]
                for ax, key in zip(axes, ["peak_rss_mib", "fit_s"], strict=True):
                    y = [r[key] if key == "peak_rss_mib" else r["fit"][key] for r in selected]
                    ax.plot(x, y, label=label, color=SLOTS[color][style["slot"]], marker=marker)
            for ax, ylabel in zip(
                axes, ["Peak process RSS (MiB)", "Fit wall time (seconds)"], strict=True
            ):
                ax.set_facecolor(style["surface"])
                ax.set_xscale("log")
                ax.set_ylim(bottom=0)
                ax.set_xticks([10000, 100000, 1000000])
                ax.xaxis.set_major_formatter(FuncFormatter(lambda n, _: f"{int(n):,}"))
                ax.set_xlabel("Units")
                ax.set_ylabel(ylabel)
                ax.grid(axis="y", color=style["grid"])
                ax.legend(
                    facecolor=style["surface"], edgecolor=style["axis"], labelcolor=style["text"]
                )
            fig.suptitle(
                "Conditional CS DR: 5 periods, 3 treated cohorts, 3 design columns",
                color=style["text"],
            )
            fig.savefig(
                ROOT / f"docs/figures/conditional_resources.{theme}.svg",
                facecolor=fig.get_facecolor(),
            )
            plt.close(fig)


def render(rows):
    total = len(cases("smoke")) + len(cases("scale"))
    passed = sum(accepted(r) for r in rows.values())
    refs = [r["r_comparison"] for r in rows.values() if r.get("r_comparison")]
    lines = [
        "# Conditional CS and unbalanced-panel resource checks",
        "",
        f"Retained results: **{len(rows)} of {total} planned Python cases**, {passed} with their expected outcome.",
        f"Actual local R comparisons: **{len(refs)}**, "
        + (
            "all matched estimates and full covariance."
            if refs and all(r["passed"] for r in refs)
            else "see validation details below."
        ),
        "",
        "These are sequential local development checks, not public speed claims. Each fit runs in a fresh process,",
        "including first-use initialization and Parquet reads. R reads CSV. Input generation is excluded.",
        "Peak RSS includes imported libraries, allocator retention and resident mapped pages.",
        "",
        "## Larger fits",
        "",
        "| Case | Units | Observed rows | Periods / cohorts | Columns | Fit / aggregation seconds | Peak RSS MiB | Peak scratch MB |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    selected = [
        "cs_n100000_balanced_k3",
        "cs_n1000000_balanced_k3",
        "cs_n1000000_entry_exit_k3",
        "cs_million_1gb",
        "cs_n100000_balanced_k17_dr",
        "cs_n100000_entry_exit_k17_dr",
        "cs_k33",
        "cs_more_cohorts_k17",
        "sunab_n1000000_t5",
        "bjs_n1000000_t5",
        "sunab_n250000_t12",
        "bjs_n250000_t12",
        "bjs_coarse_leaveout_aggregations",
    ]
    for name in selected:
        if name not in rows:
            continue
        r = rows[name]
        c = r["config"]
        s = c["data"]
        f = r["fit"]
        runtime = (
            f"{f['fit_s']:.2f} / {f['aggregation_s']:.2f}" if r["status"] == "ok" else r["status"]
        )
        lines.append(
            f"| {name} | {s['n']:,} | {f.get('n_obs', '—'):,}"
            if f.get("n_obs")
            else f"| {name} | {s['n']:,} | —"
        )
        lines[-1] += (
            f" | {s['periods']} / {s['cohorts']} | {s['width']} | {runtime} | {r['peak_rss_mib']:.0f} | {r['peak_disk_mb']:.1f} |"
        )
    lines += [
        "",
        "Most cases use a **1.5 GB decimal** process budget/watchdog, two threads, `preprocess_mb=16`,",
        "and `batch_units=512`. The tighter-buffer and million-unit 1 GB cases use 1.0 GB;",
        "the deliberate refusal uses 0.35 GB. The largest R comparison is retried with a 3 GB watchdog.",
        "The watchdog samples the process tree every 5 ms and checks reported process high-water RSS too.",
        "It is not an OS/cgroup memory guarantee. Disk is sampled every 200 ms and can miss short-lived files.",
        "Process-tree RSS can count shared pages more than once; it is not unique/cgroup-charged memory.",
        "",
        "```{image} figures/conditional_resources.light.svg",
        ":class: only-light",
        ":alt: Conditional CS process RSS and fit time across unit counts",
        "```",
        "",
        "```{image} figures/conditional_resources.dark.svg",
        ":class: only-dark",
        ":alt: Conditional CS process RSS and fit time across unit counts",
        "```",
        "",
        "## Wider designs, support and nuisance reuse",
        "",
        "The generated data include time-varying covariates, four categorical levels and numeric/category",
        "interactions (17 or 33 compiled columns), serial dependence and heterogeneous effects.",
        "Entry/exit panels combine late entry, early exit and interior gaps, retaining an untreated observation",
        "for every unit. Each input is generated in blocks of at most 4,096 units; its manifest records row",
        "count, schema configuration, Parquet layout, seed and checksum. These are resource/reference checks,",
        "not a claim that observed-panel selection always identifies the simulated causal effect.",
        "",
        "| Case | Cells | Propensity models | Numerical source passes | Maximum logit iterations |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in [
        "cs_n100000_balanced_k3",
        "cs_n100000_entry_exit_k3",
        "cs_invariant",
        "cs_invariant_entry_exit",
        "cs_more_periods",
        "cs_more_cohorts_k17",
    ]:
        if name not in rows or rows[name]["status"] != "ok":
            continue
        d = rows[name]["fit"]["diagnostics"]
        comp = d.get("comparisons", [])
        iterations = [v.get("iterations", v.get("logit_iterations", 0)) or 0 for v in comp]
        lines.append(
            f"| {name} | {len(rows[name]['fit']['cells'])} | {d.get('propensity_models')} | {d.get('nuisance_source_passes')} | {max(iterations, default=0)} |"
        )
    lines += [
        "",
        "Raw diagnostics retain overlap ranges, trimmed/capped contributions, effective controls,",
        "partition counts and largest decoded leaf. Trimming counts are comparison contributions,",
        "not unique units. Time-invariant covariates permit balanced-panel nuisance reuse; observed-row",
        "CS still needs date-pair populations because observation frequencies vary.",
        "",
        "## Buffer, layout and initialization checks",
        "",
        "| Same-sample variant | Peak RSS MiB | Fit seconds | Decoded numerical batches |",
        "|---|---:|---:|---:|",
    ]
    for name in [
        "cs_n100000_entry_exit_k17_dr",
        "cs_tighter_buffers",
        "cs_four_threads",
        "cs_strings_large_groups",
        "cs_preloaded_formula",
        "compat137_entry_exit",
    ]:
        if name in rows and rows[name]["status"] == "ok":
            r = rows[name]
            lines.append(
                f"| {name} | {r['peak_rss_mib']:.0f} | {r['fit']['fit_s']:.2f} | {r['fit']['numerical_work']['batches']} |"
            )
    lines += [
        "",
        "The tighter case changes batch units from 512 to 128 and preprocessing from 16 to 4 MB.",
        "Its extra batch/I/O work buys a modest memory reduction in this sample. Four threads did not",
        "improve this particular run. The combined string-ID/large-row-group case has actual 131,072-row",
        "input groups; it preserves the underlying sample and compiled design.",
        "",
        "The formula-preload variant moves fixed import work outside the fit timer, not outside total",
        "process RSS or process time. Its measured baseline rose from about 196 to 336 MiB after loading",
        "formula machinery. This distinguishes fixed initialization from data-dependent working memory.",
        "",
        "## Numerical stability finding",
        "",
        "The original million-unit balanced run stayed below about 604 MiB but timed out after 240 seconds.",
        "A nuisance-only probe then took 55 source passes without convergence. An absolute-only likelihood",
        "line-search allowance rejected tiny Newton improvements that were below roundoff in the summed",
        "objective. The shared temporary nuisance solver now uses compensated objective accumulation and",
        "a roundoff allowance scaled to the objective. Its coefficient-step convergence tolerance,",
        "unpenalized MLE, separation checks and singular-design checks are unchanged.",
        "",
        "The first corrected million-unit nuisance probe finished in six passes. A 600,000-row known-solution",
        "test checks both small and large batches. All measured fits below use the corrected solver;",
        "earlier results and the failing probe are retained in `history/` and `initial_logit_failure.txt`.",
        "",
        "## Equivalence and limits",
        "",
        "| Check | Status | Max cell estimate difference | Max covariance difference |",
        "|---|---|---:|---:|",
    ]
    for name, r in sorted(rows.items()):
        for key in ["r_comparison", "invariance"]:
            v = r.get(key)
            if v:
                lines.append(
                    f"| {name} ({key}) | {v['passed']} | {v['beta_max_abs']:.2g} | {v['vcov_max_abs']:.2g} |"
                )
    for name, r in rows.items():
        if not accepted(r):
            lines.append(
                f"\nUnresolved outcome: `{name}`: `{r['status']}`, {r.get('fit', {}).get('error', r.get('stderr', ''))}."
            )
    reference = RESULTS / "cs_more_cohorts_k17.r.json"
    if reference.exists():
        ref = json.loads(reference.read_text())
        lines += [
            "",
            "The high-dimensional R reference initially exceeded the 1.5 GB process-tree watchdog.",
            f"A separate retry with a {ref['config']['memory_gb']:g} GB ceiling completed at {ref['peak_rss_mib']:.0f} MiB peak RSS",
            "and passed the full numerical comparison. Both outcomes are retained. This variation is not",
            "evidence of a minimum required R memory budget or a stable R/Python memory ratio.",
        ]
    lines += [
        "",
        "Timing also varied materially across fresh-process reruns of identical data/algorithms:",
        "the million-unit balanced cases have identical estimates, covariance and eight source passes,",
        "but different observed wall times. File-cache state, initialization, allocators and competing",
        "machine load are not isolated here. Use these as resource-validation observations, not stable speed estimates.",
    ]
    lines += [
        "",
        "The 1.37 compatibility cases use the locally cached Polars 1.37.0/runtime overlay; no download is needed.",
        "String labels, large row groups, tighter batches/partitions, four threads and preloaded formula imports",
        "are compared on the same underlying sample. Fits and aggregation values/covariance must agree.",
        "The intentional budget refusal is an expected outcome, not an out-of-memory success.",
        "The watchdog self-check separately verifies memory and timeout termination.",
        "",
        "These finite checks do not establish flat RSS or a universal cluster-scale guarantee. Numerical state",
        "still grows with design columns, cohorts, periods and joint covariance size. Source scans, query",
        "operators, runtime prefetch and allocator retention are outside the decoded-partition allowance.",
        "An external kill can leave scratch files; the resumable runner removes only its own abandoned case",
        "directory. Scheduler-enforced limits and larger/richer designs remain separate release checks.",
        "",
        "## Reproduce offline",
        "",
        "```bash",
        "python benchmarks/resource_checks.py --suite smoke",
        "python benchmarks/resource_checks.py --suite scale",
        "# Resume and retry failures after an interruption or fix:",
        "python benchmarks/resource_checks.py --suite scale --retry-failed",
        "python benchmarks/resource_report.py",
        "```",
        "",
        "Use `--only CASE...` for selected cases and `--rscript PATH` for another local R installation.",
        "`--no-r` explicitly skips R comparisons. The local cached R library and Python environment must",
        "already be installed. The scripts never install packages or access a network.",
        "One driver holds a filesystem lock to prevent concurrent timing runs in the same result directory.",
        "Each case writes an atomic JSON result and stdout/stderr logs. Long fits write phase/pass progress.",
        "Library source hashes and package versions are recorded, and reruns archive previous artifacts.",
        "Input data live under ignored `benchmarks/data/resource_checks`; retained measurements, case",
        "configurations, history and `summary.csv` live under `benchmarks/results/resource_checks`.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    rows = load()
    (ROOT / "docs/resource_checks.md").write_text(render(rows))
    plots(rows)
