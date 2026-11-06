"""Offline extended Bacon validation against R and independent row dummy WLS."""

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
from benchmarks._files import digest as file_digest  # noqa: E402
from benchmarks._files import read_csv  # noqa: E402
from diff_stream import bacon_decomp  # noqa: E402

DATA = ROOT / "tests/data/bacon_extended_panel.csv"
REF = ROOT / "tests/data/bacon_extended_reference"
EXPORTER = ROOT / "benchmarks/external/bacon_extended_reference.R"
CASES = [
    "plain",
    "always",
    "always_no_never",
    "shifted_always",
    "complete",
    "future",
    "multiple_always",
    "castle",
]


def sample(case):
    d = read_csv(DATA.with_name("bacon_castle_covariates.csv") if case == "castle" else DATA)
    if case in ["always", "always_no_never", "shifted_always", "multiple_always"]:
        d = d.with_columns(
            pl.when(pl.col("first_treat") == 2)
            .then(1)
            .otherwise(pl.col("first_treat"))
            .alias("first_treat")
        )
    if case == "always_no_never":
        d = d.filter(pl.col("first_treat") != 0)
    if case in ["shifted_always", "multiple_always"]:
        d = d.with_columns(
            pl.col("time") + 20,
            pl.when(pl.col("first_treat") > 1)
            .then(pl.col("first_treat") + 20)
            .otherwise(pl.col("first_treat"))
            .alias("first_treat"),
        )
    if case == "multiple_always":
        d = d.with_columns(
            pl.when(pl.col("id") == 0).then(2).otherwise(pl.col("first_treat")).alias("first_treat")
        )
    if case == "complete":
        d = d.filter(~((pl.col("id") % 4 == 0) & (pl.col("time") == 3)))
    if case == "future":
        d = d.with_columns(
            pl.when(pl.col("first_treat") == 0)
            .then(100)
            .otherwise(pl.col("first_treat"))
            .alias("first_treat")
        )
    return d


def retained(d, case):
    return d.filter(pl.len().over("id") == d["time"].n_unique()) if case == "complete" else d


def dense_oracle(d, controlled, weight):
    ids, t, g, y = [d[c].to_numpy() for c in ["id", "time", "first_treat", "y"]]
    w = np.ones(len(d)) if weight is None else d[weight].to_numpy()
    w = w / w.max()
    nuisance = np.column_stack(
        [
            *[ids == u for u in np.unique(ids)],
            *[t == v for v in np.unique(t)[1:]],
            *([d["x"].to_numpy(), d["z"].to_numpy()] if controlled else []),
        ]
    ).astype(float)
    D = ((g > 0) & (t >= g)).astype(float)
    root = np.sqrt(w)
    theta = np.linalg.lstsq(nuisance * root[:, None], D * root, rcond=None)[0]
    r = D - nuisance @ theta
    twfe = (w * r) @ y / ((w * r) @ r)
    g = np.where(g > t.max(), 0, np.where((g > 0) & (g <= t.min()), t.min(), g))
    rm = np.array(
        [
            np.average(r[(g == cg) & (t == tt)], weights=w[(g == cg) & (t == tt)])
            for cg, tt in zip(g, t, strict=True)
        ]
    )
    rw = r - rm
    den = (w * r) @ r
    omega = (w * rw) @ rw / den
    within = (w * rw) @ y / den
    # Full dummy solve also checks the FWL target independently.
    beta = np.linalg.lstsq(np.column_stack([D, nuisance]) * root[:, None], y * root, rcond=None)[0][
        0
    ]
    assert_allclose(twfe, beta, atol=2e-9, rtol=0)
    pairs = {}
    total = w.sum() / len(np.unique(t))
    for i, cg in enumerate(np.unique(g)):
        for ch in np.unique(g)[i + 1 :]:
            left = [(g == cg) & (t == tt) for tt in np.unique(t)]
            right = [(g == ch) & (t == tt) for tt in np.unique(t)]
            dr = np.array(
                [
                    np.average(r[a], weights=w[a]) - np.average(r[b], weights=w[b])
                    for a, b in zip(left, right, strict=True)
                ]
            )
            dy = np.array(
                [
                    np.average(y[a], weights=w[a]) - np.average(y[b], weights=w[b])
                    for a, b in zip(left, right, strict=True)
                ]
            )
            mass_a = w[g == cg].sum() / len(np.unique(t))
            mass_b = w[g == ch].sum() / len(np.unique(t))
            weight_pair = mass_a * mass_b / total * (dr @ dr) / den
            pairs[(int(cg), int(ch))] = (
                float(weight_pair),
                float((dr @ dy) / (dr @ dr)) if dr @ dr > 1e-24 else None,
            )
    return float(twfe), float(omega), float(within), pairs


def digest(path):
    return file_digest(path)


def provenance():
    m = json.loads((REF / "PROVENANCE.json").read_text())
    assert m["public_data_sha256"] == digest(DATA.with_name("bacon_castle_covariates.csv"))
    assert m["input_sha256"] == digest(DATA)
    assert m["exporter_sha256"] == digest(EXPORTER)
    for n, v in m["outputs"].items():
        assert digest(REF / n) == v, n
    return m


def check(fit, case, controlled, kind):
    prefix = f"{case}_{int(controlled)}_{kind}"
    row = (
        read_csv(REF / "models.csv", null_values="NA")
        .filter(pl.col("name") == prefix)
        .row(0, named=True)
    )
    assert_allclose(fit.twfe_estimate, row["twfe"], atol=2e-9, rtol=0)
    if row["package_status"] == "ok":
        table = read_csv(REF / (prefix + ".csv"), null_values="NA")
        actual = fit.comparisons()
        if controlled:
            # R orients all dyads by adoption order; our always-treated rows
            # consistently put the always-treated group in the control column.
            mapping = {
                tuple(sorted((r["treated_cohort"], r["control_cohort"]))): r
                for r in actual.to_dicts()
            }
            for r in table.to_dicts():
                own = mapping[tuple(sorted((int(r["treated"]), int(r["untreated"]))))]
                assert_allclose(own["between_weight"], r["weight"], atol=2e-9, rtol=0)
                if own["weight"] > 1e-14:
                    assert_allclose(own["estimate"], r["estimate"], atol=2e-8, rtol=0)
            assert_allclose(fit.within_weight, row["Omega"], atol=2e-9, rtol=0)
            if fit.within_weight > 1e-12:
                assert_allclose(fit.within_estimate, row["beta_w"], atol=2e-8, rtol=0)
        else:
            own = actual.sort("treated_cohort", "control_cohort")
            table = table.sort("treated", "untreated")
            assert own["treated_cohort"].to_list() == table["treated"].to_list()
            assert own["control_cohort"].to_list() == table["untreated"].to_list()
            assert_allclose(own["weight"], table["weight"], atol=2e-9, rtol=0)
            assert_allclose(own["estimate"], table["estimate"], atol=2e-9, rtol=0)
    return row["package_status"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regenerate", action="store_true")
    p.add_argument(
        "--rscript", default=os.environ.get("DIFF_RSCRIPT") or shutil.which("Rscript") or "Rscript"
    )
    a = p.parse_args()
    if a.regenerate:
        env = dict(
            os.environ,
            R_LIBS_USER=str(ROOT / "benchmarks/external/.r/library"),
            OPENBLAS_NUM_THREADS="2",
            OMP_NUM_THREADS="2",
            TZ="UTC",
        )
        subprocess.run([a.rscript, str(EXPORTER), str(DATA), str(REF)], env=env, check=True)
        m = dict(
            input_sha256=digest(DATA),
            public_data_sha256=digest(DATA.with_name("bacon_castle_covariates.csv")),
            exporter_sha256=digest(EXPORTER),
            outputs={
                x.name: digest(x)
                for x in REF.iterdir()
                if x.is_file() and x.name != "PROVENANCE.json"
            },
            convention="Native bacondecomp with unweighted or exact integer unit-replicated weights; original-row lm WLS for all cases. Native errors retained, not used as numerical references.",
        )
        (REF / "PROVENANCE.json").write_text(json.dumps(m, indent=2) + "\n")
    provenance()
    rows = []
    for case in CASES:
        for controlled in [False, True]:
            for kind, weight in [("equal", None), ("integer", "unit_w"), ("irrational", "w")]:
                fit = bacon_decomp(
                    "y~x+z" if controlled else "y~1",
                    sample(case),
                    panel=("id", "time"),
                    cohort="first_treat",
                    panel_mode="complete" if case == "complete" else "balanced",
                    weights=weight,
                    threads=2,
                    preprocess_mb=4,
                    batch_units=7,
                )
                status = check(fit, case, controlled, kind)
                twfe, omega, within, pairs = dense_oracle(
                    retained(sample(case), case), controlled, weight
                )
                assert_allclose(fit.twfe_estimate, twfe, atol=2e-9, rtol=0)
                if controlled:
                    assert_allclose(fit.within_weight, omega, atol=2e-9, rtol=0)
                    assert_allclose(fit.within_contribution, within, atol=2e-9, rtol=0)
                rows.append(
                    dict(
                        case=case,
                        controlled=controlled,
                        weights=kind,
                        package_status=status,
                        reconstruction_error=fit.reconstruction_error,
                    )
                )
    out = ROOT / "benchmarks/results/bacon_extended_validation"
    out.mkdir(parents=True, exist_ok=True)
    report = dict(cases=len(rows), passed=True, rows=rows)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
