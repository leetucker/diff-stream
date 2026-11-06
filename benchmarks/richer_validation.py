"""Audit frozen richer-model references offline; regenerate only with installed R."""

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks._files import csv_files, read_csv  # noqa: E402
from benchmarks._files import digest as file_digest  # noqa: E402

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diff-stream-mpl")
import stream_limits  # noqa: E402

stream_limits.limit_threads(2)
import numpy as np  # noqa: E402

from diff_stream import diff_stream  # noqa: E402

REF = ROOT / "tests/data/richer_reference"
EXPORTER = ROOT / "benchmarks/external/richer_reference.R"
KW = dict(panel=("id", "time"), cohort="first_treat", threads=2)
SPECS = ["plain_varying", "numeric_varying", "extra_fe_varying", "extra_fe_unit", "extra_fe_equal"]


def digest(path):
    return file_digest(path)


def provenance(write=False):
    files = [
        ROOT / "tests/data/richer_panel.csv",
        ROOT / "tests/data/richer_unbalanced_panel.csv",
        EXPORTER,
        *csv_files(REF),
        REF / "versions.txt",
    ]
    value = dict(
        files={str(p.relative_to(ROOT)): digest(p) for p in files},
        network_required=False,
        stata_executed=False,
    )
    path = REF / "PROVENANCE.json"
    if write:
        path.write_text(json.dumps(value, indent=2) + "\n")
    elif json.loads(path.read_text()) != value:
        raise ValueError("richer-model source/input/reference provenance mismatch")


def formula(spec):
    f = "y ~ 1" if spec == "plain_varying" else "y ~ x + I(x**2)"
    if spec.startswith("extra_fe"):
        f += " | id + time + region^time"
    w = None if spec == "extra_fe_equal" else "unit_w" if spec == "extra_fe_unit" else "w"
    return f, w


def audit():
    provenance()
    comparisons = []

    def compare(result, prefix, atol):
        ref = read_csv(REF / (prefix + ".csv"))
        keys = ref["term"].to_list()
        ix = [keys.index(k) for k in result.coefnames]
        beta_error = float(np.max(np.abs(result.beta - ref["estimate"].to_numpy()[ix])))
        reference = read_csv(REF / (prefix + "_vcov.csv")).to_numpy()[np.ix_(ix, ix)]
        covariance_error = float(np.max(np.abs(result.vcov - reference)))
        passed = max(beta_error, covariance_error) < atol
        comparisons.append(
            dict(
                reference=prefix,
                beta_max_abs=beta_error,
                vcov_max_abs=covariance_error,
                tolerance=atol,
                passed=passed,
            )
        )
        if not passed:
            raise ValueError(f"{prefix}: {comparisons[-1]}")

    fits = 0
    for label, A in itertools.product(["balanced", "unbalanced"], [0, 1]):
        name = "richer_panel.csv" if label == "balanced" else "richer_unbalanced_panel.csv"
        d = read_csv(ROOT / "tests/data" / name)
        for design, control, base, method in itertools.product(
            ["constant", "numeric"],
            ["nevertreated", "notyettreated"],
            ["varying", "universal"],
            ["reg", "ipw", "dr"],
        ):
            r = diff_stream(
                "y ~ 1" if design == "constant" else "y ~ x+z",
                d,
                weights="w",
                allow_time_varying_weights=True,
                panel_mode=label,
                anticipation=A,
                est_method=method,
                control_group=control,
                base_period=base,
                ess_warn_threshold=None,
                **KW,
            )
            prefix = f"cs_{label}_{A}_{design}_{control}_{base}_{method}"
            compare(r, prefix, 3e-8)
            fits += 1
            for kind in ["dynamic", "simple", "cohort", "calendar"]:
                if base == "varying" and kind != "dynamic":
                    continue
                compare(
                    r.aggregate(kind), prefix + "_" + ("group" if kind == "cohort" else kind), 3e-8
                )
        for spec, est in itertools.product(SPECS, ["sunab", "bjs"]):
            f, w = formula(spec)
            r = diff_stream(
                f,
                d,
                weights=w,
                allow_time_varying_weights=True,
                estimator=est,
                anticipation=A,
                panel_mode=label,
                pretrends=2 if est == "bjs" else 0,
                batch_units=37,
                **KW,
            )
            fits += 1
            try:
                prefix = f"{est}_{label}_{A}_{spec}"
                if est == "sunab":
                    compare(r, prefix, 1e-9)
                else:
                    for kind in ["cells", "dynamic", "simple"]:
                        compare(
                            r if kind == "cells" else r.aggregate(kind),
                            prefix + "_cohort_time_plugin_" + kind,
                            1e-9,
                        )
                    compare(r.placebos(), f"placebo_{label}_{A}_{spec}", 1e-9)
            finally:
                r.cleanup()
        for avg, lo in itertools.product(
            ["cohort_time", "event_time", "cohort", "all"], [False, True]
        ):
            r = diff_stream(
                formula("extra_fe_varying")[0],
                d,
                weights="w",
                allow_time_varying_weights=True,
                estimator="bjs",
                anticipation=A,
                panel_mode=label,
                avgeffectsby=avg,
                leaveout=lo,
                **KW,
            )
            fits += 1
            try:
                for kind in ["cells", "dynamic", "simple"]:
                    compare(
                        r if kind == "cells" else r.aggregate(kind),
                        f"bjs_{label}_{A}_extra_fe_varying_{avg}_{'leaveout' if lo else 'plugin'}_{kind}",
                        1e-9,
                    )
            finally:
                r.cleanup()
        print(f"{label} anticipation={A}: verified", flush=True)
    report = dict(
        fits=fits,
        comparisons=len(comparisons),
        passed=all(c["passed"] for c in comparisons),
        beta_max_abs=max(c["beta_max_abs"] for c in comparisons),
        vcov_max_abs=max(c["vcov_max_abs"] for c in comparisons),
        references=comparisons,
        versions=(REF / "versions.txt").read_text().splitlines(),
        stata_executed=False,
        network_required=False,
    )
    out = ROOT / "benchmarks/results/richer_validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k not in ["references", "versions"]}),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regenerate", action="store_true")
    p.add_argument(
        "--rscript", default=os.environ.get("DIFF_RSCRIPT") or shutil.which("Rscript") or "Rscript"
    )
    args = p.parse_args()
    if args.regenerate:
        env = dict(
            os.environ,
            R_LIBS_USER=str(ROOT / "benchmarks/external/.r/library"),
            R_USER_CONFIG_DIR="/tmp/diff-rich-r-config",
        )
        subprocess.run(
            [args.rscript, str(EXPORTER), str(ROOT / "tests/data"), str(REF)], env=env, check=True
        )
        provenance(write=True)
    audit()
