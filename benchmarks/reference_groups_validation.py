"""Offline last-treated reference audit; --regenerate runs native and precision R."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks._files import digest as file_digest  # noqa: E402
from benchmarks._files import read_csv  # noqa: E402
from diff_stream import ReferenceWindowWarning, diff_stream  # noqa: E402

DATA = ROOT / "tests/data"
REF = DATA / "reference_groups"
NATIVE = DATA / "reference_groups_native"
EXPORTER = ROOT / "benchmarks/external/reference_groups.R"


def digest(path):
    return file_digest(path)


def write_provenance():
    manifest = dict(
        exporter_sha256=digest(EXPORTER),
        inputs={n: digest(DATA / n) for n in ["richer_panel.csv", "richer_unbalanced_panel.csv"]},
        outputs={
            str(p.relative_to(DATA)): digest(p)
            for folder in [REF, NATIVE]
            for p in sorted(folder.iterdir())
            if p.is_file() and p.name != "PROVENANCE.json"
        },
        convention="Explicit reference window t < g_last-A, last cohort recoded to clean control. CS onsets shifted for anticipation; actual labels restored. Non-event R aggregations retained for A=0; actual-adoption aggregation at A>0 checked by explicit sample invariance.",
        precision="Native DRDID fastglm tolerance 1e-8/maxit 100 vs confirmation 1e-12/maxit 1000; two in-process helper constants only. Installed package files unchanged.",
    )
    (REF / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n")


def verify_provenance():
    m = json.loads((REF / "PROVENANCE.json").read_text())
    assert m["exporter_sha256"] == digest(EXPORTER)
    for n, v in m["inputs"].items():
        assert digest(DATA / n) == v, n
    for n, v in m["outputs"].items():
        assert digest(DATA / n) == v, n
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument(
        "--rscript", default=os.environ.get("DIFF_RSCRIPT") or shutil.which("Rscript") or "Rscript"
    )
    args = parser.parse_args()
    if args.regenerate:
        env = dict(
            os.environ,
            R_LIBS_USER=str(ROOT / "benchmarks/external/.r/library"),
            OPENBLAS_NUM_THREADS="2",
            OMP_NUM_THREADS="2",
            TZ="UTC",
        )
        for folder, extra in [(NATIVE, []), (REF, ["precision"])]:
            subprocess.run(
                [args.rscript, str(EXPORTER), str(ROOT), str(folder), *extra], env=env, check=True
            )
        write_provenance()
    verify_provenance()
    rows = []
    warnings.filterwarnings("ignore", category=ReferenceWindowWarning)
    for panel in ["balanced", "unbalanced"]:
        d = read_csv(
            DATA / ("richer_panel.csv" if panel == "balanced" else "richer_unbalanced_panel.csv")
        ).filter(pl.col("first_treat") != 0)
        for A in [0, 1]:
            jobs = [
                ("cs", design, control, base, method)
                for design in ["constant", "numeric"]
                for control in ["nevertreated", "notyettreated"]
                for base in ["varying", "universal"]
                for method in ["reg", "ipw", "dr"]
            ]
            jobs += [
                ("sunab", design, "nevertreated", "varying", "dr")
                for design in ["constant", "numeric", "extra_fe"]
            ]
            for estimator, design, control, base, method in jobs:
                formula = "y~1" if design == "constant" else "y~x+z"
                if design == "extra_fe":
                    formula += "|region^time"
                fit = diff_stream(
                    formula,
                    d,
                    panel=("id", "time"),
                    cohort="first_treat",
                    estimator=estimator,
                    control_group=control,
                    base_period=base,
                    est_method=method,
                    anticipation=A,
                    panel_mode=panel,
                    weights="w",
                    allow_time_varying_weights=True,
                    ess_warn_threshold=None,
                    threads=2,
                    batch_units=37,
                    preprocess_mb=4,
                )
                prefix = (
                    f"cs_{panel}_{A}_{design}_{control}_{base}_{method}"
                    if estimator == "cs"
                    else f"sunab_{panel}_{A}_{design}"
                )
                errors = {}
                for label, folder, tol in [("native", NATIVE, 2e-7), ("precision", REF, 2e-9)]:
                    ref = read_csv(folder / (prefix + ".csv"))
                    ix = [ref["term"].to_list().index(n) for n in fit.coefnames]
                    b = ref["estimate"].to_numpy()[ix]
                    V = read_csv(folder / (prefix + "_vcov.csv")).to_numpy()[np.ix_(ix, ix)]
                    assert_allclose(fit.beta, b, atol=tol, rtol=0)
                    assert_allclose(fit.vcov, V, atol=tol, rtol=0)
                    errors[label + "_beta_max_abs"] = float(np.max(abs(fit.beta - b)))
                    errors[label + "_vcov_max_abs"] = float(np.max(abs(fit.vcov - V)))
                rows.append(dict(case=prefix, **errors))
    out = ROOT / "benchmarks/results/reference_groups"
    out.mkdir(parents=True, exist_ok=True)
    report = dict(
        cases=len(rows),
        passed=True,
        comparisons=rows,
        maxima={k: max(r[k] for r in rows) for k in rows[0] if k != "case"},
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}))


if __name__ == "__main__":
    main()
