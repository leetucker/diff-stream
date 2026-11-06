"""Offline Gardner audit against frozen authors' R outputs and dense WLS.

--regenerate runs the pinned, unmodified R source from the ignored benchmark
cache. No downloads or package installation occur in this script.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks._estimators import public  # noqa: E402
from benchmarks._files import digest as file_digest  # noqa: E402
from benchmarks._files import read_csv  # noqa: E402
from diff_stream import did_two_stage  # noqa: E402

DATA = ROOT / "tests/data"
REF = DATA / "gardner_reference"
COMMIT = "3adaf985980d4b46205de2e1856f93ba075baefe"
SOURCE = ROOT / "benchmarks/external/.r/did2s-source" / COMMIT
EXPORTER = ROOT / "benchmarks/external/gardner_reference.R"
CASES = ["plain_varying", "numeric_varying", "extra_fe_varying", "extra_fe_unit", "extra_fe_equal"]


def configuration(case):
    formula = "y ~ 1" if case.startswith("plain") else "y ~ x + I(x**2)"
    if case.startswith("extra"):
        formula += " | region^time"
    weight = None if case.endswith("equal") else "unit_w" if case.endswith("unit") else "w"
    return formula, weight


def dense_oracle(d, case, specification, anticipation=0, reference=None):
    """Independent all-unit dummy regression and stacked sandwich (small data only)."""
    formula, weight = configuration(case)
    y, ids, time, g = [d[c].to_numpy() for c in ("y", "id", "time", "first_treat")]
    w = np.ones(len(d)) if weight is None else d[weight].to_numpy()
    F = np.column_stack(
        [(ids == u) for u in np.unique(ids)] + [(time == t) for t in np.unique(time)[1:]]
    )
    if not case.startswith("plain"):
        x = d["x"].to_numpy()
        F = np.column_stack([F, x, x * x])
    if case.startswith("extra"):
        r = d["region"].to_numpy()
        F = np.column_stack(
            [F, *[(r == region) & (time == t) for region in np.unique(r) for t in np.unique(time)]]
        )
    clean = (g == 0) | (time < g - anticipation)
    event = time - g
    reference = -anticipation - 1 if reference is None else reference
    horizons = sorted(set(event[g != 0]) - {reference})
    if specification == "static":
        Z = ((g != 0) & (event >= -anticipation)).astype(float)[:, None]
        names = ["ATT"]
    else:
        Z = np.column_stack([(g != 0) & (event == h) for h in horizons]).astype(float)
        names = [f"event::{h}" for h in horizons]
    root = np.sqrt(w[clean])
    A = np.linalg.pinv(F[clean] * root[:, None], rcond=1e-12)
    theta = A @ (y[clean] * root)
    e1 = y - F @ theta
    Q2 = Z.T @ (w[:, None] * Z)
    beta = np.linalg.solve(Q2, Z.T @ (w * e1))
    cross = Z.T @ (w[:, None] * F)
    correction = cross @ (A @ A.T)
    e2 = e1 - Z @ beta
    score = Z * (w * e2)[:, None] - (F * (w * clean * e1)[:, None]) @ correction.T
    clusters = np.vstack([score[ids == u].sum(axis=0) for u in np.unique(ids)])
    influence = np.linalg.solve(Q2, clusters.T).T
    return names, beta, influence.T @ influence


def digest(path):
    return file_digest(path)


def verify_provenance():
    manifest = json.loads((REF / "PROVENANCE.json").read_text())
    assert manifest["source"]["commit"] == COMMIT
    assert manifest["exporter_sha256"] == digest(EXPORTER)
    for name, value in manifest["inputs"].items():
        assert digest(DATA / name) == value, name
    for name, value in manifest["outputs"].items():
        assert digest(REF / name) == value, name
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument(
        "--rscript", default=os.environ.get("DIFF_RSCRIPT") or shutil.which("Rscript") or "Rscript"
    )
    args = parser.parse_args()
    if args.regenerate:
        source = json.loads((REF / "PROVENANCE.json").read_text())["source"]
        for name, entry in source["files"].items():
            assert digest(SOURCE / name) == entry["sha256"]
        env = dict(
            os.environ,
            R_LIBS_USER=str(ROOT / "benchmarks/external/.r/library"),
            TZ="UTC",
            OPENBLAS_NUM_THREADS="2",
        )
        subprocess.run(
            [args.rscript, str(EXPORTER), str(ROOT), str(REF), str(SOURCE / "R/did2s.R")],
            env=env,
            check=True,
        )
        manifest = dict(
            source=source,
            execution="Unmodified authors did2s.R sourced directly; no installed did2s namespace. Analytic bootstrap=FALSE.",
            exporter_sha256=digest(EXPORTER),
            inputs={
                name: digest(DATA / name)
                for name in ["richer_panel.csv", "richer_unbalanced_panel.csv", "castle.csv"]
            },
            outputs={
                p.name: digest(p)
                for p in sorted(REF.iterdir())
                if p.name != "PROVENANCE.json" and p.is_file()
            },
        )
        (REF / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_provenance()
    rows = []
    for panel in ["balanced", "unbalanced"]:
        d = read_csv(
            DATA / ("richer_panel.csv" if panel == "balanced" else "richer_unbalanced_panel.csv")
        )
        for A in [0, 1]:
            for case in CASES:
                for specification in ["static", "dynamic"]:
                    formula, weight = configuration(case)
                    fit = did_two_stage(
                        formula,
                        d,
                        panel=("id", "time"),
                        cohort="first_treat",
                        event_study=public("event_study", specification),
                        anticipation=A,
                        weights=weight,
                        allow_time_varying_weights=True,
                        incomplete_units=public("incomplete_units", panel),
                        n_threads=2,
                        preprocess_mb=4,
                        batch_units=37,
                    )
                    prefix = f"{panel}_{A}_{case}_{specification}"
                    ref = read_csv(REF / f"{prefix}.csv")
                    names = ref["term"].to_list()
                    ix = [names.index(n) for n in fit.coefnames]
                    V = read_csv(REF / f"{prefix}_vcov.csv", has_header=False).to_numpy()[
                        np.ix_(ix, ix)
                    ]
                    rb = ref["estimate"].to_numpy()[ix]
                    dn, db, dV = dense_oracle(d, case, specification, A)
                    assert dn == fit.coefnames
                    for actual, expected in [
                        (fit.beta, rb),
                        (fit.vcov, V),
                        (fit.beta, db),
                        (fit.vcov, dV),
                    ]:
                        assert_allclose(actual, expected, atol=2e-9, rtol=0)
                    rows.append(
                        dict(
                            case=prefix,
                            r_beta_max_abs=float(np.max(abs(fit.beta - rb))),
                            r_vcov_max_abs=float(np.max(abs(fit.vcov - V))),
                            dense_beta_max_abs=float(np.max(abs(fit.beta - db))),
                            dense_vcov_max_abs=float(np.max(abs(fit.vcov - dV))),
                        )
                    )
    from benchmarks.castle_replication import prepare_data

    additional = []
    for panel in ["balanced", "unbalanced"]:
        d = read_csv(
            DATA / ("richer_panel.csv" if panel == "balanced" else "richer_unbalanced_panel.csv")
        )
        for A in [0, 1]:
            additional.append(
                (
                    f"{panel}_{A}_custom_reference",
                    d,
                    "extra_fe_varying",
                    "dynamic",
                    A,
                    -A - 2,
                    panel,
                )
            )
    for weighted in [False, True]:
        d = (
            prepare_data(shift=1)
            .rename({"state": "id", "year": "time", "l_homicide": "y"})
            .with_columns(pl.col("popwt").alias("w"))
        )
        for spec in ["static", "dynamic"]:
            additional.append(
                (
                    f"castle_{str(weighted).upper()}_{spec}",
                    d,
                    "plain_varying" if weighted else "plain_equal",
                    spec,
                    0,
                    -1,
                    "balanced",
                )
            )
    for prefix, d, case, spec, A, reference, panel in additional:
        formula, weight = configuration(case)
        fit = did_two_stage(
            formula,
            d,
            panel=("id", "time"),
            cohort="first_treat",
            event_study=public("event_study", spec),
            anticipation=A,
            reference_period=reference if spec == "dynamic" else None,
            weights=weight,
            allow_time_varying_weights=True,
            incomplete_units=public("incomplete_units", panel),
            n_threads=2,
            preprocess_mb=4,
            batch_units=37,
        )
        ref = read_csv(REF / f"{prefix}.csv")
        ix = [ref["term"].to_list().index(n) for n in fit.coefnames]
        rb = ref["estimate"].to_numpy()[ix]
        V = read_csv(REF / f"{prefix}_vcov.csv", has_header=False).to_numpy()[np.ix_(ix, ix)]
        names, db, dV = dense_oracle(d, case, spec, A, reference)
        assert names == fit.coefnames
        for actual, expected in [(fit.beta, rb), (fit.vcov, V), (fit.beta, db), (fit.vcov, dV)]:
            assert_allclose(actual, expected, atol=2e-9, rtol=0)
        rows.append(
            dict(
                case=prefix,
                r_beta_max_abs=float(np.max(abs(fit.beta - rb))),
                r_vcov_max_abs=float(np.max(abs(fit.vcov - V))),
                dense_beta_max_abs=float(np.max(abs(fit.beta - db))),
                dense_vcov_max_abs=float(np.max(abs(fit.vcov - dV))),
            )
        )
    out = ROOT / "benchmarks/results/gardner_validation"
    out.mkdir(parents=True, exist_ok=True)
    report = dict(cases=len(rows), passed=True, comparisons=rows, reference_commit=COMMIT)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}))


if __name__ == "__main__":
    main()
