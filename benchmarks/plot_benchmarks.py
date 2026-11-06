"""Draw the benchmark figures from the CSVs the benchmarks write.

    python benchmarks/plot_benchmarks.py

Adapted from dpd-stream. Reads benchmarks/results/{scale,covariates,absorbed}.csv and
writes, for each scenario (benchmarks/_designs.py), figures of wall time, peak memory
and peak disk against units, a panel per estimator and a line per configuration, in a
light and a dark version:

    docs/figures/<suite>_<scenario>_{time,memory,disk}.{light,dark}.svg

The sibling packages' palette and distinct markers distinguish configurations
in both themes. Points that failed, ran out of memory or timed out are left out and
listed under the figure.

Only this file needs changing to restyle them; the numbers stay in the CSVs.
"""

from __future__ import annotations

import csv
import os
import textwrap
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

HERE = Path(__file__).resolve().parent
# DIFF_BENCH_RESULTS and DIFF_BENCH_FIGURES: elsewhere, for a trial run
RESULTS = Path(os.environ.get("DIFF_BENCH_RESULTS", HERE / "results"))
FIGURES = Path(os.environ.get("DIFF_BENCH_FIGURES", HERE.parent / "docs" / "figures"))

SLOTS = {
    "blue": ("#2a78d6", "#3987e5"),
    "orange": ("#eb6834", "#d95926"),
    "aqua": ("#1baf7a", "#199e70"),
    "yellow": ("#eda100", "#c98500"),
    "magenta": ("#e87ba4", "#d55181"),
    "green": ("#008300", "#008300"),
    "violet": ("#6250d6", "#9085e9"),
    "red": ("#e34948", "#e66767"),
}
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3de",
        "axis": "#b9b8b1",
        "slot": 0,
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#34342f",
        "axis": "#5b5a55",
        "slot": 1,
    },
}

# Each figure: its CSV, its series -- configuration -> (color slot, marker,
# line style) -- its panels (title, model) and x label.
SERIES = {
    "diff-stream (1 thread)": ("magenta", "v", "-"),
    "diff-stream (4 threads)": ("violet", "^", "-"),
    "R reference (4 threads)": ("blue", "o", "-"),
}
FIGS = {
    "local_cs_memory": {
        "csv": "local_cs_memory",
        "xlabel": "Units (8 periods each)",
        "panels": [("CS memory scaling", "cs")],
        "series": SERIES,
        "metrics": ["memory"],
        "note": "Local development only; one repeat at one million units.",
    },
}
FIGS["bounded_cs_memory"] = {
    "csv": "bounded_cs_memory",
    "xlabel": "Units (8 periods each)",
    "panels": [("CS, 1 GB process budget", "cs")],
    "metrics": ["memory"],
    "series": {"Polars 2.1.0": ("magenta", "v", "-"), "Polars 1.37.0": ("violet", "^", "-")},
    "note": "Local 5 ms RSS sampling; four threads, 16 MB preprocessing allowance. Single memory smoke checks.",
}
FIGS["conditional"] = {
    "csv": "conditional",
    "xlabel": "Units (5 periods each)",
    "panels": [("Outcome regression", "reg"), ("Normalized IPW", "ipw"), ("Doubly robust", "dr")],
    "series": SERIES,
    "note": "Local development only; numeric, time-varying covariates. Parquet vs CSV input; two repeats.",
}


def suite_figures():
    """One figure per scenario of the three suites: a panel per estimator (per
    adjustment in G), the lines the scenario's configurations, coloured by their role."""
    import sys

    sys.path.insert(0, str(HERE))
    from _designs import DESIGNS
    from _suites import DS, configs, parse

    suite_of = {
        **dict.fromkeys("ABCDEF", "scale"),
        **dict.fromkeys("GH", "covariates"),
        **dict.fromkeys("IJ", "absorbed"),
    }
    titles = {
        "cs": "Callaway–Sant'Anna",
        "sunab": "Sun–Abraham",
        "bjs": "BJS imputation",
        "gardner": "Two-stage (Gardner)",
        "etwfe": "Extended TWFE",
        "stacked": "Stacked DiD",
        "bacon": "Goodman-Bacon decomposition",
        "twfe_weights": "TWFE weights",
    }
    adjustments = {"reg": "regression", "ipw": "IPW", "dr": "doubly robust"}
    what = {
        "A": "8 periods, 3 cohorts",
        "B": "8 periods, 7 cohorts",
        "C": "30 periods, 3 cohorts",
        "D": "30 periods, 29 cohorts",
        "E": "8 periods, 7 cohorts, 10% of rows missing",
        "F": "8 periods, 7 cohorts",
        "G": "8 periods, two covariates",
        "H": "8 periods, region-by-period effects (240 levels) and a covariate",
        "I": "8 periods, county-by-period effects (n/40 counties)",
        "J": "8 periods, county- and industry-by-period effects, a covariate",
    }
    # a line's style by its role: diff-stream's variants, then the other libraries
    ours = [("violet", "^", "-"), ("magenta", "v", "-"), ("aqua", "s", "--"), ("green", "h", "--")]
    others = [("blue", "o", "-"), ("orange", "D", "-"), ("yellow", "P", "-"), ("red", "X", "-")]
    out = {}
    for scenario in DESIGNS:
        cfg = configs(scenario)
        groups = {}
        for label, kind in cfg.items():
            _s, library, fit, options = parse(kind)
            groups.setdefault(
                (fit, options.get("adjustment") if scenario == "G" else None), []
            ).append((label, library))
        panels, per_panel, series = [], [], {}
        for (fit, adjustment), members in groups.items():
            title = titles[fit] + (f", {adjustments[adjustment]}" if adjustment else "")
            panels.append((title, scenario))
            labels = [label for label, _ in members]
            per_panel.append(labels)
            n_ours = n_other = 0
            for label, library in members:
                if library == "ds":
                    series[label] = ours[min(n_ours, len(ours) - 1)]
                    n_ours += 1
                else:
                    series[label] = others[min(n_other, len(others) - 1)]
                    n_other += 1
        suite = suite_of[scenario]
        out[f"{suite}_{scenario}"] = {
            "csv": suite,
            "xlabel": f"Units ({DESIGNS[scenario].periods} periods each)",
            "panels": panels,
            "panel_series": per_panel,
            "series": series,
            "legend": "panel",
            "columns": min(3, len(panels)),
            "note": f"Scenario {scenario}: {what[scenario]}. The event study, read from Parquet; "
            f"{DS} on all cores unless marked.",
        }
    return out


METRICS = {
    "time": ("wall_s", "Wall time (seconds)", 1.0),
    "memory": ("peak_memory_mb", "Peak memory (GiB)", 1 / 1024),
    "disk": ("peak_disk_mb", "Peak disk (GB)", 1e-3),
}


def read(name):
    path = RESULTS / f"{name}.csv"
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = {}
    for row in rows:
        grouped.setdefault((row["model"], row["n"], row["configuration"]), []).append(row)
    out = []
    for batch in grouped.values():
        ok = [r for r in batch if r["status"] == "ok"]
        if not ok:
            out.append(batch[0])
            continue
        row = dict(ok[0])
        for metric, _, _ in METRICS.values():
            values = [float(r[metric]) for r in ok if r.get(metric) not in (None, "")]
            if values:
                row[metric] = median(values)
        out.append(row)
    return out


def _count(v, _pos=None):
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


def _value(v, _pos=None):
    if v >= 100:
        return f"{v:,.0f}"
    if v >= 1:
        return f"{v:g}"
    return f"{v:.2g}"


def figure(name, metric, theme):
    spec = FIGS[name]
    rows = read(spec["csv"])
    if not rows:
        return None
    col, ylabel, scale = METRICS[metric]
    t = THEMES[theme]
    panels, series = spec["panels"], spec["series"]
    ncols = spec.get("columns", len(panels))
    nrows = -(-len(panels) // ncols)
    width = max(4.2 * ncols + 0.6, 7.0)
    # panels of one model on one scale (the mixed logit's two)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width, 4.0 * nrows), squeeze=False, sharey=bool(spec.get("sharey"))
    )
    for ax in axes.ravel()[len(panels) :]:
        ax.set_visible(False)
    fig.patch.set_facecolor(t["surface"])
    failed, handles = [], {}
    per_panel = spec.get("panel_series") or [list(series)] * len(panels)
    for ax, (title, model), labels in zip(axes.ravel(), panels, per_panel, strict=False):
        ax.set_facecolor(t["surface"])
        mrows = [r for r in rows if r.get("model", model) == model]
        local = {}
        for label in labels:
            slot, marker, style = series[label]
            pts = []
            for r in mrows:
                if r["configuration"] != label:
                    continue
                if r["status"] != "ok":
                    where = f" ({model})" if len(panels) > 1 else ""
                    failed.append(
                        f"{label}{where} at {_count(float(r['n']))}: {r['status'].split(':')[0]}"
                    )
                    continue
                v = float(r.get(col) or 0) * scale
                if metric == "disk" and v <= 0:
                    continue
                pts.append((float(r["n"]), max(v, 1e-3)))
            if not pts:
                continue
            pts.sort()
            color = SLOTS[slot][t["slot"]]
            (h,) = ax.plot(
                *zip(*pts, strict=True),
                linestyle=style,
                marker=marker,
                color=color,
                linewidth=2,
                markersize=7,
                markeredgecolor=t["surface"],
                markeredgewidth=1.2,
                label=label,
            )
            handles.setdefault(label, h)
            local[label] = h
        if spec.get("legend") == "panel" and local:
            ax.legend(
                list(local.values()),
                list(local),
                loc="upper left",
                frameon=False,
                fontsize=7.5,
                labelcolor=t["text"],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, color=t["text"], fontsize=11, loc="left")
        ax.set_xlabel(spec["xlabel"], color=t["muted"], fontsize=9)
        if spec.get("xticks"):
            ax.set_xticks(spec["xticks"])
        ax.xaxis.set_major_formatter(FuncFormatter(_count))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(_value))
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", color=t["grid"], linewidth=0.8)
        ax.tick_params(colors=t["muted"], labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(t["axis"])
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, color=t["muted"], fontsize=9)
    order = [lb for lb in series if lb in handles]
    # the note of what is not shown at the bottom, the legend above it
    note = ""
    if failed and metric != "disk":  # only diff-stream writes to disk
        note = "Not shown: " + "; ".join(dict.fromkeys(failed)) + "."
    if spec.get("note") and metric != "disk":
        note = (note + " " + spec["note"]).strip()
    if note:
        note = "\n".join(textwrap.wrap(note, int(width * 19)))
        fig.text(0.01, 0.01, note, color=t["muted"], fontsize=7, va="bottom")
    note_h = (0.03 * (note.count("\n") + 1) + 0.01) if note else 0.0
    if spec.get("legend") == "panel":
        order = []
    ncol = min(len(order), 4 if len(panels) > 1 else 3)
    nrow = -(-len(order) // max(ncol, 1)) if order else 0
    if order:
        fig.legend(
            [handles[lb] for lb in order],
            order,
            loc="lower center",
            ncol=ncol,
            frameon=False,
            fontsize=8.5,
            labelcolor=t["text"],
            bbox_to_anchor=(0.5, note_h),
        )
    fig.tight_layout(rect=(0, note_h + 0.06 + 0.05 * nrow, 1, 1))
    return fig


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    FIGS.update(suite_figures())
    for name, spec in FIGS.items():
        for metric in spec.get("metrics", METRICS):
            for theme in THEMES:
                fig = figure(name, metric, theme)
                if fig is None:
                    continue
                path = FIGURES / f"{name}_{metric}.{theme}.svg"
                fig.savefig(path, facecolor=fig.get_facecolor())
                plt.close(fig)
                print(f"wrote {path}")


if __name__ == "__main__":
    main()
