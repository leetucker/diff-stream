"""Frozen native R TwoWayFEWeights references and independent row-dummy WLS."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks._estimators import public  # noqa: E402
from benchmarks._files import csv_files, read_csv  # noqa: E402
from benchmarks._files import digest as file_digest  # noqa: E402
from diff_stream import twfe_weights  # noqa: E402

REF = ROOT / "tests/data/twfe_weights_reference"
EXPORTER = ROOT / "benchmarks/external/twfe_weights_reference.R"
CASES = [
    "absorbing",
    "always",
    "reversals",
    "unbalanced",
    "complete",
    "gaps",
    "no_never",
    "disconnected",
    "singleton",
    "cycle",
    "constant_multiplier",
    "zero_outcome",
    "wagepan",
]


def sample(case):
    if case == "wagepan":
        return (
            read_csv(ROOT / "tests/data/twfe_wagepan.csv", null_values="NA")
            .select(
                pl.col("nr").alias("id"),
                pl.col("year").alias("time"),
                pl.col("lwage").alias("y"),
                pl.col("union").cast(pl.Float64).alias("d"),
                (pl.col("hours") / 1000).alias("x"),
                pl.col("married").cast(pl.Float64).alias("z"),
            )
            .with_columns(
                (1 + pl.col("id") % 3).alias("unit_w"),
                ((1 + pl.col("id") % 5) * (1 + 0.03 * (pl.col("time") - 1980))).alias("w"),
            )
        )
    d = read_csv(ROOT / "tests/data/bacon_extended_panel.csv").with_columns(
        ((pl.col("first_treat") > 0) & (pl.col("time") >= pl.col("first_treat")))
        .cast(pl.Float64)
        .alias("d"),
        (pl.col("w") * (1 + 0.05 * pl.col("time"))).alias("w"),
    )
    if case == "always":
        d = d.with_columns(
            pl.when(pl.col("first_treat") == 2).then(1.0).otherwise(pl.col("d")).alias("d")
        )
    if case == "reversals":
        d = d.with_columns(
            (((pl.col("id") * 3 + pl.col("time")) % 5) < 2).cast(pl.Float64).alias("d")
        )
    if case in ["unbalanced", "complete"]:
        d = d.filter(~((pl.col("id") % 4 == 0) & (pl.col("time") == 3)))
    if case == "gaps":
        d = d.with_columns((17 + 2 * pl.col("time")).alias("time"))
    if case == "no_never":
        d = d.filter(pl.col("first_treat") != 0)
    if case == "disconnected":
        d = d.filter(
            ((pl.col("id") < 12) & (pl.col("time") <= 4))
            | ((pl.col("id") >= 12) & (pl.col("time") > 4))
        )
    if case == "singleton":
        d = pl.concat(
            [
                d,
                d.head(1).with_columns(
                    pl.lit(99).cast(d.schema["id"]).alias("id"), pl.lit(1.0).alias("d")
                ),
            ]
        )
    if case == "cycle":
        d = d.filter(
            ((pl.col("id") == 0) & pl.col("time").is_in([1, 2]))
            | ((pl.col("id") == 1) & pl.col("time").is_in([2, 3]))
            | ((pl.col("id") == 2) & pl.col("time").is_in([1, 3]))
        )
        d = d.with_columns(
            (
                ((pl.col("id") == 0) & (pl.col("time") == 2))
                | ((pl.col("id") == 1) & (pl.col("time") == 3))
            )
            .cast(pl.Float64)
            .alias("d")
        )
    if case == "constant_multiplier":
        d = d.with_columns(
            ((pl.col("id") % 2 == 0) & (pl.col("time") >= 5)).cast(pl.Float64).alias("d")
        )
    if case == "zero_outcome":
        d = d.with_columns(pl.lit(0.0).alias("y"))
    return d.select("id", "time", "y", "d", "x", "z", "unit_w", "w")


def configs():
    return [
        dict(
            name=f"{case}_{int(control)}_{kind}",
            case=case,
            controlled=control,
            weight=weight,
            mode="complete"
            if case == "complete"
            else "unbalanced"
            if case in ["unbalanced", "disconnected", "singleton", "cycle"]
            else "balanced",
        )
        for case in CASES
        for control in [False, True]
        for kind, weight in [("equal", None), ("unit", "unit_w"), ("changing", "w")]
    ]


def retained(c):
    d = sample(c["case"])
    return d.filter(pl.len().over("id") == d["time"].n_unique()) if c["mode"] == "complete" else d


def dense(d, controlled, weight):
    ids, t, y, D = [d[c].to_numpy() for c in ["id", "time", "y", "d"]]
    w = np.ones(len(d)) if weight is None else d[weight].to_numpy()
    w = w / w.max()
    X = np.column_stack(
        [
            *[ids == u for u in np.unique(ids)],
            *[t == v for v in np.unique(t)[1:]],
            *([d["x"].to_numpy(), d["z"].to_numpy()] if controlled else []),
        ]
    ).astype(float)
    sw = np.sqrt(w)
    r = D - X @ np.linalg.lstsq(X * sw[:, None], D * sw, rcond=None)[0]
    den = np.sum(w * r * r)
    if den < 1e-18:
        return None
    loading = w * r / den
    q = loading * D
    return dict(beta=float(loading @ y), r=r, q=q, p=w * D / (w @ D), loading=loading)


def fit(c, **kw):
    return twfe_weights(
        "y~x+z" if c["controlled"] else "y~1",
        sample(c["case"]),
        panel=("id", "time"),
        treatment="d",
        incomplete_units=public("incomplete_units", c["mode"]),
        weights=c["weight"],
        allow_time_varying_weights=True,
        n_threads=2,
        preprocess_mb=4,
        batch_units=17,
        **kw,
    )


def digest(p):
    return file_digest(p)


def provenance():
    m = json.loads((REF / "PROVENANCE.json").read_text())
    for n, h in m["files"].items():
        assert digest(ROOT / n) == h, n


def check(c):
    ref = (
        read_csv(
            REF / "models.csv",
            null_values=["NA", "NaN", "Inf", "-Inf"],
            schema_overrides={"sensibility": pl.Float64, "sensibility2": pl.Float64},
        )
        .filter(pl.col("name") == c["name"])
        .row(0, named=True)
    )
    if ref["status"] != "ok" and not (
        c["case"] == "zero_outcome" or (c["case"] == "cycle" and c["controlled"])
    ):
        raise AssertionError(f"unexpected native R refusal for {c['name']}: {ref['error']}")
    oracle = dense(retained(c), c["controlled"], c["weight"])
    if oracle is None:
        try:
            fit(c)
        except ValueError:
            return dict(name=c["name"], status="unidentified", native_status=ref["status"])
        raise AssertionError("unidentified treatment was not refused")
    with fit(c, keep_cells=True) as r:
        cells = r.cells().collect().sort(["unit", "time"])
        d = retained(c).sort(["id", "time"])
        oracle = dense(d, c["controlled"], c["weight"])
        assert_allclose(r.twfe_estimate, oracle["beta"], atol=2e-8, rtol=0)
        for col, target in [
            ("treatment_residual", "r"),
            ("weight", "q"),
            ("treated_share", "p"),
            ("outcome_loading", "loading"),
        ]:
            assert_allclose(cells[col], oracle[target], atol=2e-8, rtol=0)
        assert_allclose(
            cells["outcome_loading"].to_numpy() @ d["y"].to_numpy(),
            r.twfe_estimate,
            atol=2e-8,
            rtol=0,
        )
        if ref["status"] == "ok":
            native = read_csv(REF / (c["name"] + ".csv")).sort(["G", "T"])
            assert_allclose(cells["classified_weight"], native["weight"], atol=2e-8, rtol=0)
            assert_allclose(r.twfe_estimate, ref["beta"], atol=2e-8, rtol=0)
            stats = r.summary_dict()["weights"]
            assert stats["by_sign"][0]["n_cells"] == ref["nr_plus"]
            assert stats["by_sign"][1]["n_cells"] == ref["nr_minus"]
            measures = r.sensitivity().to_dicts()
            if measures[0]["status"] == "finite":
                assert_allclose(measures[0]["value"], ref["sensibility"], atol=2e-8, rtol=1e-8)
            if measures[1]["status"] == "finite":
                assert_allclose(measures[1]["value"], ref["sensibility2"], atol=2e-8, rtol=1e-8)
        return dict(
            name=c["name"],
            status="ok",
            native_status=ref["status"],
            beta=r.twfe_estimate,
            sensitivity=r.sensitivity().to_dicts(),
            negative_mass=r.summary_dict()["weights"]["negative_weight_mass"],
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regenerate", action="store_true")
    p.add_argument(
        "--rscript", default=os.environ.get("DIFF_RSCRIPT") or shutil.which("Rscript") or "Rscript"
    )
    a = p.parse_args()
    if a.regenerate:
        REF.mkdir(parents=True, exist_ok=True)
        for case in CASES:
            sample(case).write_csv(REF / (case + ".input.csv"))
        pl.DataFrame([{**c, "weight": c["weight"] or ""} for c in configs()]).write_csv(
            REF / "configs.csv"
        )
        env = dict(
            os.environ,
            R_LIBS_USER=str(ROOT / "benchmarks/external/.r/library"),
            OPENBLAS_NUM_THREADS="2",
            OMP_NUM_THREADS="2",
            TZ="UTC",
        )
        subprocess.run([a.rscript, str(EXPORTER), str(REF)], env=env, check=True)
        paths = [
            *csv_files(REF),
            REF / "R_session.txt",
            ROOT / "tests/data/twfe_wagepan.csv",
            ROOT / "tests/data/bacon_extended_panel.csv",
            EXPORTER,
        ]
        m = dict(
            reference='TwoWayFEWeights::twowayfeweights(type="feTR"); native package, no code replacement',
            settings='fixest fixef.rm="none", fixef.tol=1e-10; weights passed as numeric vector (2.1.0 implementation)',
            wagepan_source="https://raw.githubusercontent.com/Credible-Answers/twowayfeweights/main/wagepan_twfeweights.csv",
            files={str(x.relative_to(ROOT)): digest(x) for x in paths},
        )
        (REF / "PROVENANCE.json").write_text(json.dumps(m, indent=2) + "\n")
    provenance()
    rows = [check(c) for c in configs()]
    out = ROOT / "benchmarks/results/twfe_weights_validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(dict(cases=len(rows), passed=True, rows=rows), indent=2) + "\n"
    )
    print(json.dumps(dict(cases=len(rows), passed=True)))


if __name__ == "__main__":
    main()
