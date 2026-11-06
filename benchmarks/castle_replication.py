"""Offline Sun–Abraham reanalysis of the public Cheng–Hoekstra castle panel.

--regenerate uses installed R fixest and the pinned, frozen R data object.
--plot writes standalone light/dark SVGs and a PDF using matplotlib.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks._files import digest as file_digest  # noqa: E402
from benchmarks._files import read_csv  # noqa: E402
from diff_stream import sunab_stream

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data/castle.csv"
SOURCE = ROOT / "tests/data/castle_source"
REF = ROOT / "tests/data/castle_reference"
OUT = ROOT / "benchmarks/results/castle_replication"
EXPORTER = ROOT / "benchmarks/external/castle_reference.R"
SCENARIOS = [(shift, weighted) for shift in (0, 1) for weighted in (False, True)]


def digest(path):
    return file_digest(path)


def prepare_data(data=None, shift=0):
    if shift not in (0, 1):
        raise ValueError("shift must be 0 (effyear) or 1 (source post convention)")
    raw = read_csv(DATA, null_values="NA") if data is None else data
    return raw.with_columns(
        pl.when(pl.col("effyear").is_null())
        .then(0)
        .otherwise(pl.col("effyear") + shift)
        .cast(pl.Int64)
        .alias("first_treat")
    )


def verify_provenance():
    source = json.loads((SOURCE / "PROVENANCE.json").read_text())
    for name, checksum in source["sha256"].items():
        assert digest(SOURCE / name) == checksum, name
    manifest = json.loads((REF / "PROVENANCE.json").read_text())
    assert digest(DATA) == manifest["data_sha256"]
    assert digest(EXPORTER) == manifest["exporter_sha256"]
    assert manifest["source_commit"] == source["commit"]
    for name, checksum in manifest["reference_sha256"].items():
        assert digest(REF / name) == checksum, name


def dense_oracle(data, weighted=False):
    """Independent full-dummy WLS and unit-cluster CRV0, small-example only."""
    unit, time, cohort, y = (
        data[name].to_numpy() for name in ("state", "year", "first_treat", "l_homicide")
    )
    units, inverse = np.unique(unit, return_inverse=True)
    periods = np.unique(time)
    cells = [(int(g), int(t)) for g in np.unique(cohort) if g for t in periods if t != g - 1]
    X = np.column_stack(
        [
            *[unit == u for u in units],
            *[time == t for t in periods[1:]],
            *[(cohort == g) & (time == t) for g, t in cells],
        ]
    ).astype(float)
    w = data["popwt"].to_numpy() if weighted else np.ones(len(y))
    w = w / w.mean()
    b, _, rank, _ = np.linalg.lstsq(X * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None)
    assert rank == X.shape[1]
    residual = y - X @ b
    score = np.zeros((len(units), X.shape[1]))
    np.add.at(score, inverse, X * (w * residual)[:, None])
    H = np.linalg.solve(X.T @ (w[:, None] * X), np.eye(X.shape[1])[:, -len(cells) :])
    influence = score @ H
    return cells, b[-len(cells) :], influence.T @ influence


def compare_replication(shift=0, weighted=False, data=None, *, dense=True, **options):
    verify_provenance()
    d = prepare_data(data, shift)
    prefix = f"shift{shift}_{'popwt' if weighted else 'unweighted'}"
    fit = sunab_stream(
        "l_homicide ~ 1",
        d,
        panel=("state", "year"),
        cohort="first_treat",
        weights="popwt" if weighted else None,
        threads=2,
        **options,
    )
    differences = {}

    def check(key, a, b, atol=1e-9):
        a, b = np.asarray(a), np.asarray(b)
        assert_allclose(a, b, atol=atol, rtol=0, err_msg=prefix + ":" + key)
        differences[key] = float(np.max(np.abs(a - b)))

    try:
        assert fit.n_units == 50 and fit.n_obs == 550 and len(fit.cells) == 50
        table = read_csv(REF / (prefix + "_cells.csv"))
        keys = list(zip(table["g"], table["t"], strict=True))
        assert set(keys) == set(fit.cells)
        ix = [keys.index(c) for c in fit.cells]
        V = read_csv(REF / (prefix + "_cells_vcov.csv")).to_numpy()
        denseV = read_csv(REF / (prefix + "_dense_vcov.csv")).to_numpy()
        check("cells_beta", fit.beta, table["att"].to_numpy()[ix])
        check("cells_se", fit.se, table["se"].to_numpy()[ix])
        check("cells_vcov", fit.vcov, V[np.ix_(ix, ix)])
        check("R_dummy_beta", fit.beta, table["dense_att"].to_numpy()[ix])
        check("R_dummy_vcov", fit.vcov, denseV[np.ix_(ix, ix)])
        if dense:
            dk, db, dv = dense_oracle(d, weighted)
            di = [dk.index(c) for c in fit.cells]
            check("Python_dummy_beta", fit.beta, db[di])
            check("Python_dummy_vcov", fit.vcov, dv[np.ix_(di, di)])
        for kind in ("dynamic", "simple", "cohort", "calendar", "balanced_dynamic"):
            a = (
                fit.aggregate(balance_e=2, horizons=[0, 1, 2])
                if kind == "balanced_dynamic"
                else fit.aggregate(kind)
            )
            r = read_csv(REF / f"{prefix}_{kind}.csv")
            labels = (
                [0] if kind == "simple" else [int(name.split("::")[-1]) for name in a.coefnames]
            )
            order = [r["level"].to_list().index(level) for level in labels]
            rv = read_csv(REF / f"{prefix}_{kind}_vcov.csv").to_numpy()
            check(kind + "_beta", a.beta, r["att"].to_numpy()[order])
            check(kind + "_se", a.se, r["se"].to_numpy()[order])
            check(kind + "_vcov", a.vcov, rv[np.ix_(order, order)])
            assert np.linalg.eigvalsh(a.vcov).min() >= -1e-12
        ref_test = read_csv(REF / f"{prefix}_pretrend.csv").row(0, named=True)
        test = fit.pretrend_test(level="dynamic", horizons=[-4, -3, -2])
        check("pretrend_statistic", test["statistic"], ref_test["statistic"], atol=1e-8)
        check("pretrend_pvalue", test["pvalue"], ref_test["pvalue"])
        return fit, differences
    except BaseException:
        fit.cleanup()
        raise


def plot_example(fit):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from benchmarks.plot_benchmarks import SLOTS, THEMES

    a = fit.aggregate()
    x = np.array([int(name.split("::")[-1]) for name in a.coefnames])
    reference = read_csv(REF / "shift0_unweighted_dynamic.csv")
    contribution = a.contributions()
    support = np.array(
        [
            contribution.filter(pl.col("aggregate") == name)["weight_units"].sum()
            for name in a.coefnames
        ]
    )
    directory = ROOT / "docs/figures"
    directory.mkdir(parents=True, exist_ok=True)
    for theme, style in THEMES.items():
        color = SLOTS["violet"][style["slot"]]
        contrast = SLOTS["aqua"][style["slot"]]
        with plt.rc_context(
            {
                "text.color": style["text"],
                "axes.labelcolor": style["text"],
                "xtick.color": style["text"],
                "ytick.color": style["text"],
                "axes.edgecolor": style["axis"],
                "font.size": 10,
            }
        ):
            fig, (ax, counts) = plt.subplots(
                2, 1, figsize=(9, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
            )
            fig.patch.set_facecolor(style["surface"])
            for panel in (ax, counts):
                panel.set_facecolor(style["surface"])
                panel.grid(axis="y", color=style["grid"], zorder=0)
                panel.axvline(0, color=style["muted"], ls=":", lw=1)
            ax.axhline(0, color=style["muted"], lw=1)
            ax.errorbar(
                x,
                a.beta,
                yerr=1.959963984540054 * a.se,
                fmt="o",
                color=color,
                capsize=3,
                label="diff-stream; 95% normal intervals",
                zorder=3,
            )
            ax.scatter(
                reference["level"],
                reference["att"],
                marker="x",
                s=42,
                color=contrast,
                label="R fixest::sunab",
                zorder=4,
            )
            ax.scatter(
                [-1],
                [0],
                facecolors="none",
                edgecolors=style["muted"],
                s=44,
                label="Reference: year −1 (normalized)",
                zorder=4,
            )
            ax.set_ylabel("Effect on log homicide rate")
            ax.set_title(
                "Castle Doctrine: Sun–Abraham reanalysis\nUnweighted state/year effects; never-treated controls",
                color=style["text"],
            )
            legend = ax.legend(
                fontsize=9, loc="lower right", facecolor=style["surface"], edgecolor=style["axis"]
            )
            for label in legend.get_texts():
                label.set_color(style["text"])
            counts.bar(x, support, color=color, width=0.65, zorder=2)
            counts.set_ylabel("Treated states")
            counts.set_xlabel("Years relative to effyear (law-adoption year)")
            counts.set_xticks(np.arange(x.min(), x.max() + 1))
            fig.text(
                0.12,
                0.015,
                "50 states, 2000–2010; 21 adopters. Extreme horizons have limited support.\nSaturated specification; source post uses the following-year convention (see sensitivity results).",
                fontsize=8,
                color=style["muted"],
            )
            fig.tight_layout(rect=(0, 0.07, 1, 1))
            fig.savefig(directory / f"castle_sunab.{theme}.svg", facecolor=fig.get_facecolor())
            if theme == "light":
                fig.savefig(OUT / "event_study.pdf", facecolor=fig.get_facecolor())
                fig.savefig(OUT / "event_study.png", dpi=150, facecolor=fig.get_facecolor())
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--rscript", default=os.environ.get("DIFF_RSCRIPT") or shutil.which("Rscript") or "Rscript"
    )
    args = parser.parse_args()
    if args.regenerate:
        source = json.loads((SOURCE / "PROVENANCE.json").read_text())
        for name, checksum in source["sha256"].items():
            assert digest(SOURCE / name) == checksum, name
        env = dict(
            os.environ,
            R_LIBS_USER=str(ROOT / "benchmarks/external/.r/library"),
            R_USER_CONFIG_DIR="/tmp/diff-castle-r-config",
            OPENBLAS_NUM_THREADS="2",
            OMP_NUM_THREADS="2",
        )
        subprocess.run(
            [args.rscript, str(EXPORTER), str(SOURCE / "castle.rda"), str(DATA), str(REF)],
            env=env,
            check=True,
        )
        manifest = dict(
            source_commit=source["commit"],
            data_sha256=digest(DATA),
            exporter_sha256=digest(EXPORTER),
            reference_sha256={
                p.name: digest(p)
                for p in sorted(REF.iterdir())
                if p.is_file() and p.name != "PROVENANCE.json"
            },
        )
        (REF / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    OUT.mkdir(parents=True, exist_ok=True)
    reports = []
    for shift, weighted in SCENARIOS:
        fit, differences = compare_replication(shift, weighted)
        try:
            prefix = f"shift{shift}_{'popwt' if weighted else 'unweighted'}"
            fit.group_time().write_csv(OUT / f"{prefix}_cells.csv")
            for kind in ("dynamic", "simple", "cohort", "calendar"):
                a = fit.aggregate(kind)
                a.tidy().write_csv(OUT / f"{prefix}_{kind}.csv")
                a.contributions().write_csv(OUT / f"{prefix}_{kind}_contributions.csv")
            simple = fit.aggregate("simple")
            reports.append(
                dict(
                    case=prefix,
                    shift=shift,
                    sampling_weights="popwt" if weighted else None,
                    att=float(simple.beta[0]),
                    se=float(simple.se[0]),
                    pretrend_test=fit.pretrend_test(level="dynamic", horizons=[-4, -3, -2]),
                    max_abs_difference=differences,
                )
            )
            if args.plot and shift == 0 and not weighted:
                plot_example(fit)
        finally:
            fit.cleanup()
    d = prepare_data()
    cohorts = d.unique("state").group_by("first_treat").len().sort("first_treat").to_dicts()
    report = dict(
        status="passed",
        scope="Unconditional Sun–Abraham reanalysis of Cheng and Hoekstra (2013); not full original-paper replication",
        n_units=50,
        n_obs=550,
        n_cells=50,
        years=[2000, 2010],
        cohorts=cohorts,
        excluded_rows=0,
        adoption_year_post_discrepancies=21,
        source_post_convention="year > effyear; shift1 matches source post",
        specifications=reports,
        reference_versions=(REF / "versions.txt").read_text().splitlines(),
        files={
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                DATA,
                EXPORTER,
                Path(__file__),
                *sorted(SOURCE.iterdir()),
                *sorted(REF.iterdir()),
            ]
            if p.is_file()
        },
        tolerance_absolute=1e-9,
        pretrend_statistic_tolerance=1e-8,
        stata_executed=False,
        python_version=sys.version,
        numpy_version=np.__version__,
        polars_version=pl.__version__,
    )
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "specifications": [{k: r[k] for k in ("case", "att", "se")} for r in reports],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
