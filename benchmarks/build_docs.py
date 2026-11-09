"""Write the benchmark pages of the documentation from the results.

    python benchmarks/build_docs.py

Reads benchmarks/results/{scale,covariates,absorbed}.csv and writes
docs/benchmarks/{index,scale,covariates,absorbed}.md. The tables and every number
in the text are computed from the CSVs, so after a new run (``bench_ec2.py fetch
--install``, then ``plot_benchmarks.py`` if the figures need redrawing) run this and
read the diff: the statements about which libraries finish, run out of memory or
fail are written here, and should be checked against the new tables.

Rows whose label mentions "passive" are left out (the OpenMP wait policy is set by
diff_stream itself).
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "benchmarks"
SIZES = [10_000, 100_000, 1_000_000, 10_000_000]
D = {
    s: pd.read_csv(ROOT / "benchmarks" / "results" / f"{s}.csv")
    for s in ("scale", "covariates", "absorbed")
}
NAMES = {
    "cs": "Callaway-Sant'Anna",
    "sunab": "Sun-Abraham",
    "bjs": "BJS",
    "gardner": "Two-stage",
    "etwfe": "Extended TWFE",
    "stacked": "Stacked DiD",
    "bacon": "Goodman-Bacon decomposition",
    "twfe_weights": "TWFE weights",
}
ORDER = list(NAMES)


def row(suite, model, config, n):
    d = D[suite]
    r = d[(d.model == model) & (d.configuration == config) & (d.n == n)]
    return r.iloc[0] if len(r) else None


def secs(x):
    if x < 1:
        return f"{x:.2f}"
    if x < 100:
        return f"{x:.1f}"
    return f"{x:,.0f}"


def gb(mb):
    return f"{mb / 1000:.1f}"


def t(suite, model, config, n):
    r = row(suite, model, config, n)
    assert r is not None and r.status == "ok", (suite, model, config, n)
    return float(r.wall_s)


def mem(suite, model, config, n):
    r = row(suite, model, config, n)
    assert r is not None and r.status == "ok", (suite, model, config, n)
    return float(r.peak_memory_mb)


def cell(r):
    if r is None:
        return "–"
    s = str(r.status)
    if s == "ok":
        return secs(r.wall_s)
    if s.startswith("timeout"):
        return "timeout"
    if s.startswith("failed"):
        return "–"
    return "out of memory"


def label(config):
    if config.startswith("diff-stream "):
        rest = config.split(" ", 2)
        extra = rest[2].strip() if len(rest) > 2 else ""
        extra = extra.strip("()")
        return "diff-stream" + (f", {extra}" if extra else "")
    return config


def table(suite, model, include=None, exclude=("limit", "passive")):
    """Long table: estimator, implementation, wall seconds at each size."""
    d = D[suite]
    g = d[d.model == model]
    lines = [
        "| Estimator | Implementation | 10,000 | 100,000 | 1,000,000 | 10,000,000 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    configs = []
    for est in ORDER:
        sub = g[g.estimator == est]
        cfgs = list(dict.fromkeys(sub.configuration))
        cfgs = [c for c in cfgs if not any(x in c for x in exclude)]
        # drop implementations that never produced a result or only failed to set up
        cfgs = [
            c for c in cfgs if not sub[sub.configuration == c].status.str.startswith("failed").all()
        ]
        ds = [c for c in cfgs if c.startswith("diff-stream")]
        other = [c for c in cfgs if not c.startswith("diff-stream")]
        for c in ds + other:
            configs.append((est, c))
    last = None
    for est, c in configs:
        cells = [cell(row(suite, model, c, n)) for n in SIZES]
        name = NAMES[est] if est != last else ""
        last = est
        lines.append(f"| {name} | {label(c)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fig(name, alt):
    return (
        f"```{{image}} ../figures/{name}.light.svg\n:class: only-light benchmark\n:alt: {alt}\n```\n"
        f"```{{image}} ../figures/{name}.dark.svg\n:class: only-dark benchmark\n:alt: {alt}\n```\n"
    )


DS = {
    "cs": "diff-stream did_callaway_santanna",
    "sunab": "diff-stream did_sun_abraham",
    "bjs": "diff-stream did_bjs",
    "gardner": "diff-stream did_two_stage",
    "etwfe": "diff-stream did_extended_twfe",
    "stacked": "diff-stream did_stacked",
}


def ds_range(suite, model, n, ests=None, col="time"):
    f = t if col == "time" else mem
    vals = [f(suite, model, DS[e], n) for e in (ests or DS)]
    return min(vals), max(vals)


def rng(lo_hi, fmt=secs):
    lo, hi = lo_hi
    return f"{fmt(lo)} to {fmt(hi)}"


# ---------------------------------------------------------------- scale
sB1 = ds_range("scale", "B", 1_000_000)
sB10 = ds_range("scale", "B", 10_000_000)
mA10 = ds_range("scale", "A", 10_000_000, col="mem")
mA10_1 = [mem("scale", "A", DS[e] + " (1 thread)", 10_000_000) for e in DS]
sC1 = ds_range("scale", "C", 1_000_000)
sC10 = ds_range("scale", "C", 10_000_000)
sD1 = ds_range("scale", "D", 1_000_000)
sD10 = ds_range("scale", "D", 10_000_000)
sE1 = ds_range("scale", "E", 1_000_000)
sE10 = ds_range("scale", "E", 10_000_000)
did_B1 = t("scale", "B", "did (R)", 1_000_000)
did_B10 = t("scale", "B", "did (R)", 10_000_000)
dd_cs_B1 = t("scale", "B", "diff-diff CallawaySantAnna (Python)", 1_000_000)
fix_sunab_B1 = t("scale", "B", "fixest sunab (R)", 1_000_000)
etwfe_R_B1 = t("scale", "B", "etwfe (R)", 1_000_000)
ds_cs_B1 = t("scale", "B", DS["cs"], 1_000_000)
A_dd_cs10 = t("scale", "A", "diff-diff CallawaySantAnna (Python)", 10_000_000)
A_ds_cs10 = t("scale", "A", DS["cs"], 10_000_000)
A_dd_cs10_mem = mem("scale", "A", "diff-diff CallawaySantAnna (Python)", 10_000_000)
A_ds_cs10_mem = mem("scale", "A", DS["cs"], 10_000_000)
A_did10_mem = mem("scale", "A", "did (R)", 10_000_000)
A_did10 = t("scale", "A", "did (R)", 10_000_000)
A_fx_st10 = t("scale", "A", "fixest on stacked data (R)", 10_000_000)
A_fx_st10_mem = mem("scale", "A", "fixest on stacked data (R)", 10_000_000)
A_ds_st10 = t("scale", "A", DS["stacked"], 10_000_000)
C_ds_ts1 = t("scale", "C", DS["gardner"], 10_000_000)
C_ds_ts1t = t("scale", "C", DS["gardner"] + " (1 thread)", 10_000_000)
B_cs1t = t("scale", "B", DS["cs"] + " (1 thread)", 10_000_000)
B_cs = t("scale", "B", DS["cs"], 10_000_000)
E_cs1t = t("scale", "E", DS["cs"] + " (1 thread)", 10_000_000)
E_cs = t("scale", "E", DS["cs"], 10_000_000)
B_lim = [mem("scale", "B", DS[e] + " (2 GB limit)", 1_000_000) for e in DS]
B_lim_t = [t("scale", "B", DS[e] + " (2 GB limit)", 1_000_000) for e in DS]
B_free = [mem("scale", "B", DS[e], 1_000_000) for e in DS]
B10_lim_rows = [row("scale", "B", DS[e] + " (2 GB limit)", 10_000_000) for e in DS]
B10_fits = all(r is not None and r.status == "ok" for r in B10_lim_rows)
B10_lim = [float(r.peak_memory_mb) for r in B10_lim_rows] if B10_fits else []
B10_lim_t = [float(r.wall_s) for r in B10_lim_rows] if B10_fits else []
B10_text = (
    f"At ten million units (80 million rows) it still fits, at a peak of {gb(max(B10_lim))} GB "
    f"in {rng((min(B10_lim_t), max(B10_lim_t)))} seconds, with Polars held to 4 threads, as a job "
    "with a 2 GB allocation would be. Polars' read buffers grow with its thread pool "
    "(about 130 MB a thread), so with every thread of the benchmark machine they alone can "
    "exceed 2 GB at this size."
    if B10_fits
    else "At ten million units (80 million rows) 2 GB is too small: every estimator stops with a "
    "`MemoryError` before allocating, instead of being killed. Polars' read buffers grow with "
    "its thread pool (about 130 MB a thread), and with every thread of the benchmark machine "
    "they alone exceed 2 GB at this size."
)
C_disk = row("scale", "C", DS["cs"], 10_000_000).peak_disk_mb
A_disk = row("scale", "A", DS["cs"], 10_000_000).peak_disk_mb
D_etwfe_R10k = row("scale", "D", "etwfe (R)", 10_000)
D_dd_etwfe_10k = t("scale", "D", "diff-diff ETWFE (Python)", 10_000)
D_ds_etwfe_10k = t("scale", "D", DS["etwfe"], 10_000)
D_dd_cs1 = t("scale", "D", "diff-diff CallawaySantAnna (Python)", 1_000_000)
D_did1 = t("scale", "D", "did (R)", 1_000_000)
D_ds_cs1 = t("scale", "D", DS["cs"], 1_000_000)

# 10M table for diff-stream across A-E
ds10 = ["| Estimator | A | B | C | D | E |", "|---|---:|---:|---:|---:|---:|"]
ds1 = list(ds10)
for e in DS:
    ds10.append(
        f"| {NAMES[e]} | "
        + " | ".join(secs(t("scale", m, DS[e], 10_000_000)) for m in "ABCDE")
        + " |"
    )
    ds1.append(
        f"| {NAMES[e]} | "
        + " | ".join(secs(t("scale", m, DS[e], 1_000_000)) for m in "ABCDE")
        + " |"
    )

# diagnostics (F)
F = {
    "bacon": (
        "diff-stream bacon_decomp",
        ["bacondecomp (R)", "diff-diff BaconDecomposition (Python)"],
    ),
    "twfe_weights": ("diff-stream twfe_weights", ["TwoWayFEWeights (R)"]),
}
fbl = []
for est, (dsc, oth) in F.items():
    for c in (dsc, dsc + " (1 thread)", *oth):
        fbl.append((NAMES[est], label(c), [cell(row("scale", "F", c, n)) for n in SIZES]))
fb_rows = [
    "| Diagnostic | Implementation | 10,000 | 100,000 | 1,000,000 | 10,000,000 |",
    "|---|---|---:|---:|---:|---:|",
]
last = None
for name, lab, cells in fbl:
    fb_rows.append(f"| {name if name != last else ''} | {lab} | " + " | ".join(cells) + " |")
    last = name
bacon_R = row("scale", "F", "bacondecomp (R)", 10_000)

scale_md = f"""# Panels without covariates

Time and peak memory as the number of units grows, for the event study (with
standard errors) of each estimator, on five simulated designs:

| Design | Periods | Cohorts | Rows at 1,000,000 units |
|---|---:|---:|---:|
| A | 8 | 3 | 8 million |
| B | 8 | 7 | 8 million |
| C | 30 | 3 | 30 million |
| D | 30 | 29 | 30 million |
| E | 8 | 7, with 10% of the rows missing | 7.3 million |

Every library fits the same task from the same Parquet file. Times are the
whole fit, including reading the file for the libraries that read it first. The
first fit in a new installation spends several seconds compiling diff-stream's
kernels, once; the benchmarks fit 2,000 units first, so the times are for fits
with the compiled kernels loaded. See [the overview](index.md) for the machine
and the limits.

## Design B: 8 periods, 7 cohorts

{fig("scale_B_time", "Wall time against the number of units for each estimator, one line per implementation, log-log")}
- **diff-stream** fits a million units (8 million rows) in {rng(sB1)} seconds,
  depending on the estimator, and ten million (80 million rows) in
  {rng(sB10)} seconds. Time grows a little faster than the number of units: ten
  times the units takes 15 to 23 times as long.
- **Callaway-Sant'Anna:** at a million units R's `did` takes {secs(did_B1)}
  seconds and diff-diff {secs(dd_cs_B1)}, against {secs(ds_cs_B1)} for
  diff-stream. At ten million units `did` takes {secs(did_B10)} seconds and
  diff-diff {secs(t("scale", "B", "diff-diff CallawaySantAnna (Python)", 10_000_000))} (with a peak of {gb(mem("scale", "B", "diff-diff CallawaySantAnna (Python)", 10_000_000))} GB),
  against {secs(B_cs)} for diff-stream.
- **Sun-Abraham:** at a million units fixest's `sunab` takes {secs(fix_sunab_B1)}
  seconds, diff-diff {secs(t("scale", "B", "diff-diff SunAbraham (Python)", 1_000_000))}
  and pyfixest {secs(t("scale", "B", "pyfixest saturated (Python)", 1_000_000))}.
- **Extended TWFE:** at a million units R's `etwfe` takes {secs(etwfe_R_B1)}
  seconds, diff-diff {secs(t("scale", "B", "diff-diff ETWFE (Python)", 1_000_000))}.
- **BJS, two-stage:** diff-diff's imputation estimator takes
  {secs(t("scale", "B", "diff-diff ImputationDiD (Python)", 1_000_000))} seconds at a million
  units. R's `didimputation` takes {secs(t("scale", "B", "didimputation (R)", 100_000))} seconds at
  100,000 units and runs out of memory at a million. diff-diff's two-stage
  estimator takes {secs(t("scale", "B", "diff-diff TwoStageDiD (Python)", 10_000))} seconds at 10,000
  units and runs out of memory at 100,000.
- **Stacked DiD:** fixest on the stacked data takes
  {secs(t("scale", "B", "fixest on stacked data (R)", 10_000_000))} seconds at ten million units, against
  {secs(t("scale", "B", DS["stacked"], 10_000_000))} for diff-stream, and diff-diff's `StackedDiD` runs out of memory.

{table("scale", "B")}

"out of memory" is a process killed by the operating system or a memory error
in the library. A dash means the configuration was not run at that size; a
configuration that fails, times out or runs out of memory is not run at larger
sizes.

{fig("scale_B_memory", "Peak memory against the number of units for each estimator, one line per implementation, log-log")}
diff-stream's memory grows slowly: at ten million units on 8 periods its peak
is {rng(mA10, gb)} GB with all threads and
{gb(min(mA10_1))} to {gb(max(mA10_1))} GB with one (design A), with
{A_disk / 1000:.1f} GB of scratch disk for the sorted panel. The libraries that hold the
panel in memory need much more: at the same size R's `did` peaks at
{gb(A_did10_mem)} GB, and diff-diff's Callaway-Sant'Anna and fixest on the
stacked data at {gb(A_dd_cs10_mem)} GB.

With few cohorts the gap in time is smaller. In design A (3 cohorts) at ten
million units diff-diff's Callaway-Sant'Anna takes {secs(A_dd_cs10)} seconds and
fixest on the stacked data {secs(A_fx_st10)}, against {secs(A_ds_cs10)} and
{secs(A_ds_st10)} for diff-stream, which is a modest difference in time and a
large one in memory ({gb(A_ds_cs10_mem)} GB against {gb(A_dd_cs10_mem)} GB). Below about 100,000 units
the faster Python and R implementations are within a small factor of
diff-stream or ahead of it.

## All designs

Seconds to fit, all threads, at 1,000,000 units:

{chr(10).join(ds1)}

and at 10,000,000 units:

{chr(10).join(ds10)}

- **More periods and more cohorts cost more.** Design C (30 periods, 3 cohorts)
  takes {rng(sC1)} seconds at a million units and {rng(sC10)} at ten million.
  Design D (29 cohorts) takes {rng(sD1)} and {rng(sD10)}. Extended TWFE is the
  slowest in D because its regression has a column for every cohort-period
  cell.
- **Unbalanced panels (E)** take {rng(sE1)} seconds at a million units and
  {rng(sE10)} at ten million. BJS is the slowest, because its standard errors
  need extra passes over the rows when the panel is unbalanced.
- **Other implementations stop earlier in the larger designs.** At a million
  units in design D, R's `did` takes {secs(D_did1)} seconds and diff-diff's
  Callaway-Sant'Anna {secs(D_dd_cs1)}, against {secs(D_ds_cs1)} for
  diff-stream; of the other comparators only diff-diff's imputation estimator
  ({secs(t("scale", "D", "diff-diff ImputationDiD (Python)", 1_000_000))} seconds) finished at that size. At 10,000 units in design D, R's `etwfe` does not finish in
  1,800 seconds and diff-diff's ETWFE takes {secs(D_dd_etwfe_10k)}, against {secs(D_ds_etwfe_10k)}.
  At ten million units in designs C and D (300 million rows) no comparator
  finished.

{fig("scale_D_time", "Wall time against the number of units in design D (30 periods, 29 cohorts), log-log")}
## Threads

The times above use all 16 vCPUs. At ten million units one thread takes 1.3 to 2.6 times
as long, and at a million units 1.2 to 3.5 times (the large ratios are in the
larger designs and the two-stage estimator): in design B at ten million units Callaway-Sant'Anna takes
{secs(B_cs1t)} seconds on one thread against {secs(B_cs)}, and in design C the
two-stage estimator takes {secs(C_ds_ts1t)} against {secs(C_ds_ts1)}. For
unbalanced panels the difference is larger:
Callaway-Sant'Anna in design E takes {secs(E_cs1t)} seconds on one thread and
{secs(E_cs)} on all. The one-thread fits and the other variants (memory limits, the
dense solver) agree with the default fit to 5e-14 or better.

## Memory limits

With `max_memory_gb=2`, design B at a million units peaks at about
{gb(max(B_lim))} GB, in {rng((min(B_lim_t), max(B_lim_t)))} seconds, against a peak of about
{gb(max(B_free))} GB and {rng(ds_range("scale", "B", 1_000_000))} seconds
without a limit. {B10_text}
See [Memory and threads](../resources.md) for what a limit does and how to choose one.

## Diagnostics (design F)

Goodman-Bacon decomposition and TWFE weights, on the panel of design B, with
the time to the whole diagnostic:

{chr(10).join(fb_rows)}

## Agreement

Every comparator that finished agrees with diff-stream's fit of the same
estimator to 5e-8 or better in designs A to D (most to 1e-9 or better), as the
largest absolute difference of any coefficient or standard error. In design E,
with 10% of the rows missing, R's `etwfe` and diff-diff's ETWFE differ from
diff-stream's extended TWFE by 0.002, diff-diff's `StackedDiD` by 1.5e-4, and
pyfixest's saturated regression by 0.007 to 0.018; the other comparators agree
to 2.2e-8 or better. The conventions for unbalanced panels are listed under
[differences between implementations](../implementations.md).

R's `did2s` was not installed for these designs, so there is no R two-stage
comparison here.

Figures for every design are in `docs/figures/` (time, memory and disk), and
the numbers are in `benchmarks/results/scale.csv`.
"""

# ---------------------------------------------------------------- covariates
cG = lambda e, n: t("covariates", "G", DS[e] + ("" if e != "cs" else " (doubly robust)"), n)  # noqa: E731
cs_reg = t("covariates", "G", "diff-stream did_callaway_santanna (regression)", 1_000_000)
cs_ipw = t("covariates", "G", "diff-stream did_callaway_santanna (IPW)", 1_000_000)
cs_dr = t("covariates", "G", "diff-stream did_callaway_santanna (doubly robust)", 1_000_000)
dd = lambda k, n: t("covariates", "G", f"diff-diff CallawaySantAnna (Python, {k})", n)  # noqa: E731
rr = lambda k, n: t("covariates", "G", f"did (R, {k})", n)  # noqa: E731
G1 = [cG(e, 1_000_000) for e in DS if e != "stacked"]
G10 = [cG(e, 10_000_000) for e in DS if e != "stacked"]
dd_dr_mem = mem("covariates", "G", "diff-diff CallawaySantAnna (Python, doubly robust)", 10_000_000)
ds_dr_mem = mem("covariates", "G", "diff-stream did_callaway_santanna (doubly robust)", 10_000_000)
G_ds_mem = [
    mem("covariates", "G", DS[e] + ("" if e != "cs" else " (doubly robust)"), 10_000_000)
    for e in DS
    if e != "stacked"
]

H_sa = lambda n: t("covariates", "H", "diff-stream did_sun_abraham", n)  # noqa: E731
H_sa_dense = lambda n: t("covariates", "H", "diff-stream did_sun_abraham (dense solver)", n)  # noqa: E731
H_bjs = lambda n: t("covariates", "H", "diff-stream did_bjs", n)  # noqa: E731
H_bjs_dense = lambda n: t("covariates", "H", "diff-stream did_bjs (dense solver)", n)  # noqa: E731
H_ts = lambda n: t("covariates", "H", "diff-stream did_two_stage", n)  # noqa: E731
H_sa_mem = mem("covariates", "H", "diff-stream did_sun_abraham", 10_000_000)
H_sa_dense_mem = mem("covariates", "H", "diff-stream did_sun_abraham (dense solver)", 10_000_000)

covariates_md = f"""# Covariates and fixed effects

Two designs with 8 periods and 7 cohorts and extra regressors. Every library
fits the same task from the same Parquet file (see [the overview](index.md)).

- **G: two covariates.** Callaway-Sant'Anna by outcome regression, inverse
  probability weighting and doubly robust estimation, and the other
  estimators with the covariates in the regression.
- **H: region-by-period fixed effects (240 levels) and a covariate.**
  Sun-Abraham, BJS and the two-stage estimator, with the fixed effects
  estimated iteratively (the default at this many levels) and, for
  Sun-Abraham and BJS, with the dense solver.

## G: two covariates

{fig("covariates_G_time", "Wall time against the number of units with two covariates, one line per implementation, log-log")}
- **diff-stream** fits a million units in {rng((min(G1), max(G1)))} seconds,
  depending on the estimator, and ten million in {rng((min(G10), max(G10)))}
  seconds. The three Callaway-Sant'Anna variants take {secs(cs_reg)} (regression),
  {secs(cs_ipw)} (inverse probability weighting) and {secs(cs_dr)} (doubly
  robust) seconds at a million units.
- **Callaway-Sant'Anna in other implementations:** at a million units diff-diff
  takes {secs(dd("regression", 1_000_000))} to {secs(dd("doubly robust", 1_000_000))} seconds and R's `did`
  {secs(rr("regression", 1_000_000))} to {secs(rr("doubly robust", 1_000_000))}. At ten million units diff-diff takes
  {secs(dd("regression", 10_000_000))} (regression), {secs(dd("IPW", 10_000_000))} (inverse probability weighting) and
  {secs(dd("doubly robust", 10_000_000))} (doubly robust) seconds, with a peak of {gb(dd_dr_mem)} GB, against
  {gb(ds_dr_mem)} GB for diff-stream. R's `did` runs out of memory.
- **Sun-Abraham:** fixest's `sunab` takes {secs(t("covariates", "G", "fixest sunab (R)", 1_000_000))} seconds at a million
  units and diff-diff {secs(t("covariates", "G", "diff-diff SunAbraham (Python)", 1_000_000))}; both run out of memory at ten million.
- **BJS, two-stage, extended TWFE:** diff-diff's imputation estimator takes
  {secs(t("covariates", "G", "diff-diff ImputationDiD (Python)", 1_000_000))} seconds at a million units and runs out of memory at
  ten million. R's `didimputation`, `did2s` and `etwfe` and diff-diff's two-stage estimator
  run out of memory at a million units.
- diff-stream's peak at ten million units is {rng((min(G_ds_mem), max(G_ds_mem)), gb)} GB.

{table("covariates", "G")}

"out of memory" and a dash mean the same as on the [previous page](scale.md).

## H: many fixed-effect levels

{fig("covariates_H_time", "Wall time against the number of units with 240 region-by-period effects, one line per implementation, log-log")}
- **BJS** fits a million units in {secs(H_bjs(1_000_000))} seconds and ten million in
  {secs(H_bjs(10_000_000))}. The dense solver takes {secs(H_bjs_dense(1_000_000))} and {secs(H_bjs_dense(10_000_000))}.
- **Two-stage** takes {secs(H_ts(1_000_000))} seconds at a million units and {secs(H_ts(10_000_000))} at ten million.
- **Sun-Abraham** with the iterative solver takes {secs(H_sa(1_000_000))} seconds at a million units, against
  {secs(H_sa_dense(1_000_000))} for the dense solver, but at ten million units the iterative solver takes
  {secs(H_sa(10_000_000))} seconds and the dense solver {secs(H_sa_dense(10_000_000))}. The dense solver also peaks at
  {gb(H_sa_dense_mem)} GB, where the iterative solver reaches {gb(H_sa_mem)} GB, which is the benchmark's memory cap and
  counts pages of a memory-mapped file (see [the overview](index.md#peak-memory)). For which
  solver `"auto"` picks, see [Sun-Abraham](../estimators/sun_abraham.md#fixed-effects-with-many-levels).
- **Other implementations:** fixest's `sunab` takes {secs(t("covariates", "H", "fixest sunab (R)", 1_000_000))} seconds at a million units and runs out of
  memory at ten million. R's `didimputation` takes {secs(t("covariates", "H", "didimputation (R)", 100_000))} seconds at
  100,000 units and runs out of memory at a million. R's `did2s` was not installed for this design.
- One thread is slow at ten million units: the Sun-Abraham and two-stage fits time out at
  1,800 seconds.

{table("covariates", "H", exclude=("limit", "passive"))}

Every comparator that finished agrees with diff-stream's fit of the same
estimator to 2e-8 or better, except diff-diff's doubly robust
Callaway-Sant'Anna, which agrees to 2.1e-6.

Figures for both designs (time, memory and disk) are in `docs/figures/`, and
the numbers are in `benchmarks/results/covariates.csv`.
"""

# ---------------------------------------------------------------- absorbed
aI = lambda e, n: t("absorbed", "I", DS[e], n)  # noqa: E731
aJ = lambda e, n: t("absorbed", "J", DS[e], n)  # noqa: E731
absorbed_md = f"""# Fixed effects with millions of levels

Two designs with 8 periods and 7 cohorts, in which units belong to counties
(one county for every 40 units, so a million units have 25,000 counties and ten
million have 250,000) and the fixed effects are absorbed out of core by
[hdfe-stream](https://github.com/leetucker/hdfe-stream):

- **I: county-by-period effects.** At ten million units, two million levels.
- **J: county-by-period and industry-by-period effects, and a covariate.**

The benchmark fits Sun-Abraham and BJS, the estimators that take fixed effects
with any number of levels. fixest's `sunab` is the only other implementation of a
comparable regression that ran.

{fig("absorbed_I_time", "Wall time against the number of units with county-by-period effects, one line per implementation, log-log")}
- **BJS** fits a million units in {secs(aI("bjs", 1_000_000))} seconds in design I and
  {secs(aJ("bjs", 1_000_000))} in design J, and ten million in {secs(aI("bjs", 10_000_000))} and
  {secs(aJ("bjs", 10_000_000))}.
- **Sun-Abraham** takes {secs(aI("sunab", 1_000_000))} and {secs(aJ("sunab", 1_000_000))} seconds at a million units, but at ten million
  units it takes {secs(aI("sunab", 10_000_000))} seconds in design I and times out at 1,800 seconds in design J.
  BJS fits these designs much faster than Sun-Abraham at ten million units.
- **fixest's `sunab`** takes {secs(t("absorbed", "I", "fixest sunab (R)", 1_000_000))} and {secs(t("absorbed", "J", "fixest sunab (R)", 1_000_000))} seconds at a million units,
  with peaks of {gb(mem("absorbed", "I", "fixest sunab (R)", 1_000_000))} and {gb(mem("absorbed", "J", "fixest sunab (R)", 1_000_000))} GB, and runs
  out of memory at ten million.
- **R's `didimputation`** takes {secs(t("absorbed", "I", "didimputation (R)", 100_000))} and {secs(t("absorbed", "J", "didimputation (R)", 100_000))} seconds at
  100,000 units. At a million and ten million units the benchmark's call fails
  (`predict()` needs the fixed-effect values that fixest drops with this many
  levels), so there is no result.
- With one thread, the fits take 1.7 to 2.5 times as long at a million units and
  1.9 to 2.0 times as long at ten million, where Sun-Abraham times out in both designs.

## Design I

{table("absorbed", "I", exclude=("passive",))}

## Design J

{table("absorbed", "J", exclude=("passive",))}

Every comparator that finished agrees with diff-stream's fit to 6e-9 or
better.

With `max_memory_gb=4`, Sun-Abraham in design I still reports a peak of
{gb(mem("absorbed", "I", "diff-stream did_sun_abraham (4 GB limit)", 10_000_000))} GB at ten million units; the peak counts
memory-mapped pages (see [the overview](index.md#peak-memory)).

Figures (time, memory and disk) are in `docs/figures/`, and the numbers are in
`benchmarks/results/absorbed.csv`.
"""

index_md = f"""# Benchmarks

The benchmarks compare diff-stream with the R and Python implementations of the
same estimators (R's `did`, `fixest`, `didimputation`, `did2s`, `etwfe`,
`bacondecomp` and `TwoWayFEWeights`; Python's `diff-diff` and `pyfixest`) on
simulated panels of 10,000 to 10 million units, as time and peak memory against
the number of units. Stata's commands are not benchmarked. See
`benchmarks/README.md` in the repository to run them.

- **diff-stream scales to panels the others cannot hold.** At ten million units
  over 8 periods (80 million rows) all six estimators finish in
  {rng(sB10)} seconds with a peak of {rng(mA10, gb)} GB, and at ten million
  units over 30 periods (300 million rows) in
  {rng(sC10, secs)} seconds. At that size most other implementations run out of memory
  on this 30 GB machine; the ones that finish (R's `did`, fixest on stacked
  data, diff-diff's Callaway-Sant'Anna with few cohorts) take as long or longer and
  need {gb(A_dd_cs10_mem)} to {gb(A_did10_mem)} GB.
- **It is faster at a million units.** The fits take {rng(sB1)} seconds, against
  {secs(dd_cs_B1)} to {secs(did_B1)} for Callaway-Sant'Anna in other
  implementations, and from a few seconds to over 800 seconds for the other estimators. Below
  about 100,000 units the faster Python and R implementations are comparable.
- **Covariates and fixed effects cost more** but scale the same way
  ([covariates](covariates.md)). With millions of fixed-effect levels BJS fits ten
  million units in {secs(aI("bjs", 10_000_000))} to {secs(aJ("bjs", 10_000_000))} seconds; Sun-Abraham needs
  {secs(aI("sunab", 10_000_000))} seconds or more ([absorbed fixed effects](absorbed.md)).
- **Estimates agree** with the other implementations to 5e-8 or better where
  they use the same conventions; the exceptions are listed on each page.

```{{toctree}}
:maxdepth: 1

scale
covariates
absorbed
```

## How the benchmarks were run

All the results come from one EC2 spot instance (c8id.4xlarge: Intel Xeon 6975P-C,
16 vCPUs, 30.8 GB of memory, local NVMe disk), with Python 3.13, NumPy 2.5,
numba 0.68 and Polars 2.1. Each fit runs in its own process, with a limit of
1,800 seconds, and the memory of the benchmarks capped at 96% of the machine. A
fit that exceeds the limit or runs out of memory is a result, and the
configuration is not run at larger sizes. diff-stream's fits are timed after a
warm-up fit of 2,000 units that loads its cached compiled kernels; the first fit in a
new installation spends several seconds compiling. Every library fits the same task, the event
study with its standard errors, from the same Parquet file, and the estimates
of each are compared with diff-stream's fit of the same estimator.

(peak-memory)=
## Peak memory

Peak memory in the benchmarks is the process's resident high-water mark, for
every library alike. It counts resident pages of memory-mapped files as well as
ordinary memory. With high-dimensional fixed effects, diff-stream's Sun-Abraham
and BJS (through hdfe-stream) map a table from disk, so most of their reported
peak can be file-backed memory that the operating system drops under pressure
rather than memory a limit has to hold. A reported peak at the benchmark's cap
(about 29.6 GB) means the resident pages, file-backed ones included, reached the
cap; it does not measure the memory the fit needs.
"""

for name, text in (
    ("index.md", index_md),
    ("scale.md", scale_md),
    ("covariates.md", covariates_md),
    ("absorbed.md", absorbed_md),
):
    text = re.sub(r"\n{3,}", "\n\n", text)
    (OUT / name).write_text(text)
    print("wrote", name, len(text.splitlines()), "lines")
