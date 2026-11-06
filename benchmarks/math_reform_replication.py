"""Offline replication of bacondecomp's public math_reform worked example.

--regenerate refreshes the frozen data and references using installed R packages.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import polars as pl
from numpy.testing import assert_allclose

from benchmarks._files import digest as file_digest  # noqa: E402
from benchmarks._files import read_csv  # noqa: E402
from diff_stream import bacon_decomp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data/math_reform.csv"
REF = ROOT / "tests/data/math_reform_reference"
OUT = ROOT / "benchmarks/results/math_reform_replication"
TYPES = {
    "Treated vs Untreated": "treated_vs_never",
    "Earlier vs Later Treated": "earlier_vs_later",
    "Later vs Earlier Treated": "later_vs_earlier",
}


def prepare_data(data=None):
    raw = read_csv(DATA, null_values="NA") if data is None else data
    # Keep original source columns intact; map missing adoption dates only in
    # this explicit analysis column. No sample deletion or date compression.
    return raw.with_columns(
        pl.col("reformyr_math").fill_null(0).cast(pl.Int64).alias("first_treat")
    )


def dense_twfe(data):
    """Independent full row/unit/time-dummy OLS, used only on this small fixture."""
    ids, t, y, D = (data[c].to_numpy() for c in ("state", "class", "incearn_ln", "reform_math"))
    X = np.column_stack(
        [D, *[ids == u for u in np.unique(ids)], *[t == v for v in np.unique(t)[1:]]]
    ).astype(float)
    return float(np.linalg.lstsq(X, y, rcond=None)[0][0])


def compare_replication(data=None):
    d = prepare_data(data)
    fit = bacon_decomp(
        "incearn_ln ~ 1", d, panel=("state", "class"), cohort="first_treat", n_threads=2
    )
    actual = fit.comparisons().sort(["treated_cohort", "control_cohort"])
    reference = read_csv(REF / "comparisons.csv").sort(["treated", "untreated"])
    differences = {}

    def check(key, a, b, atol=1e-10):
        a, b = np.asarray(a), np.asarray(b)
        assert_allclose(a, b, atol=atol, rtol=1e-10, err_msg=key)
        differences[key] = max(differences.get(key, 0.0), float(np.max(np.abs(a - b))))

    assert actual.height == reference.height == 16
    assert actual["treated_cohort"].to_list() == reference["treated"].to_list()
    assert actual["control_cohort"].to_list() == reference["untreated"].to_list()
    assert actual["comparison_type"].to_list() == [TYPES[s] for s in reference["type"]]
    check("comparison_estimates", actual["estimate"], reference["estimate"])
    check("comparison_weights", actual["weight"], reference["weight"], atol=1e-12)
    check(
        "comparison_contributions",
        actual["contribution"],
        reference["estimate"] * reference["weight"],
    )
    check("pair_R_dense_OLS", actual["estimate"], reference["dense_estimate"])
    check("weight_sum", actual["weight"].sum(), 1.0, atol=1e-12)
    check("reconstruction", fit.twfe_estimate, fit.decomposed_estimate)
    full = read_csv(REF / "twfe.csv").row(0, named=True)
    check("full_R_dense_OLS", fit.twfe_estimate, full["twfe"])
    check("full_Python_dense_OLS", fit.twfe_estimate, dense_twfe(d))
    assert fit.n_units == 26 and fit.n_obs == 520
    for row in actual.iter_rows(named=True):
        pair = d.filter(
            pl.col("first_treat").is_in([row["treated_cohort"], row["control_cohort"]])
            & pl.col("class").is_between(row["pre_start"], row["post_end"])
        )
        check("pair_Python_dense_OLS", row["estimate"], dense_twfe(pair))
        for role in ("treated", "control"):
            n = d.filter(pl.col("first_treat") == row[f"{role}_cohort"])["state"].n_unique()
            assert n == row[f"n_{role}_units"]
    summary = fit.summary().sort("comparison_type")
    rs = (
        read_csv(REF / "type_summary.csv")
        .with_columns(pl.col("type").replace_strict(TYPES).alias("comparison_type"))
        .sort("comparison_type")
    )
    for col in ("weight", "estimate", "contribution", "n_comparisons"):
        check("summary_" + col, summary[col], rs[col])
    printed = (
        read_csv(REF / "documented_summary.csv")
        .with_columns(pl.col("type").replace_strict(TYPES).alias("comparison_type"))
        .sort("comparison_type")
    )
    check(
        "documented_weight",
        np.round(summary["weight"].to_numpy(), 5),
        printed["weight"],
        atol=1e-14,
    )
    check(
        "documented_estimate",
        np.round(summary["estimate"].to_numpy(), 5),
        printed["avg_est"],
        atol=1e-14,
    )
    check(
        "already_treated_control_weight",
        fit.diagnostics["already_treated_control_weight"],
        summary.filter(pl.col("comparison_type") == "later_vs_earlier")["weight"].item(),
    )
    return fit, differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument(
        "--rscript", default=os.environ.get("DIFF_RSCRIPT") or shutil.which("Rscript") or "Rscript"
    )
    args = parser.parse_args()
    if args.regenerate:
        env = dict(os.environ)
        env.setdefault("R_LIBS_USER", str(ROOT / "benchmarks/external/.r/library"))
        env.setdefault("R_USER_CONFIG_DIR", "/tmp/diff-math-reform-r-config")
        Path(env["R_USER_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                args.rscript,
                str(ROOT / "benchmarks/external/math_reform_reference.R"),
                str(DATA),
                str(REF),
            ],
            env=env,
            check=True,
        )
    fit, differences = compare_replication()
    OUT.mkdir(parents=True, exist_ok=True)
    fit.comparisons().write_csv(OUT / "comparisons.csv")
    fit.summary().write_csv(OUT / "type_summary.csv")
    raw = read_csv(DATA, null_values="NA")
    cohort_sizes = (
        read_csv(REF / "cohorts.csv").rename({"Var1": "cohort", "Freq": "n_units"}).to_dicts()
    )
    paths = [
        DATA,
        ROOT / "benchmarks/external/math_reform_reference.R",
        Path(__file__),
        *sorted(REF.glob("*")),
    ]
    report = dict(
        status="passed",
        source="bacondecomp documented math_reform example; aggregated Goodman (2019) data",
        sample=dict(
            units=26,
            rows=520,
            first_class=1975,
            last_class=1994,
            cohort_sizes=cohort_sizes,
            excluded_units=0,
            excluded_rows=0,
            missing_adoption_rows=raw["reformyr_math"].null_count(),
        ),
        specification=dict(
            outcome="incearn_ln",
            treatment="reform_math",
            unit="state",
            time="class",
            covariates=None,
            sampling_weights=None,
            normalization="missing reformyr_math maps to first_treat=0; original columns unchanged",
        ),
        twfe_estimate=fit.twfe_estimate,
        decomposed_estimate=fit.decomposed_estimate,
        reconstruction_error=fit.reconstruction_error,
        n_comparisons=fit.comparisons().height,
        type_summary=fit.summary().to_dicts(),
        max_abs_difference=differences,
        reference_versions=(REF / "versions.txt").read_text().splitlines(),
        files={str(p.relative_to(ROOT)): file_digest(p) for p in paths},
        python_version=sys.version,
        numpy_version=np.__version__,
        polars_version=pl.__version__,
        stata_executed=False,
    )
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                status="passed",
                twfe=fit.twfe_estimate,
                reconstructed=fit.decomposed_estimate,
                max_abs_difference=differences,
            )
        ),
        flush=True,
    )
    print(fit.summary())


if __name__ == "__main__":
    main()
