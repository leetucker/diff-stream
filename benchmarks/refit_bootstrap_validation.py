"""Frozen native did refits and explicit base-R pointwise interval formulas.

Run with --generate to export the actual unit-pairs samples and invoke R;
without it, verify provenance and audit the frozen reference outputs.
Nine draws exercise numerical equivalence, not Monte Carlo coverage.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks._estimators import public  # noqa: E402
from diff_stream import did_callaway_santanna, refit_bootstrap  # noqa: E402
from diff_stream._core.resample import DiskGroupResampler  # noqa: E402
from diff_stream.simulate import simulate_conditional_panel  # noqa: E402

REF = ROOT / "tests/data/refit_bootstrap_reference"
EXPORTER = ROOT / "benchmarks/external/refit_bootstrap_reference.R"


def sample(case):
    data = simulate_conditional_panel(120, 5, seed=17).with_columns(
        ((0.3 + pl.col("id") % 4) * (1 + 0.05 * pl.col("time"))).alias("w")
    )
    if case == "unbalanced_last":
        data = data.filter(pl.col("first_treat") != 0).filter(
            ~((pl.col("id") % 4 == 0) & (pl.col("time") == 2))
        )
    return data


def fit(case, method):
    return refit_bootstrap(
        "y~x",
        sample(case),
        panel=("id", "time"),
        cohort="first_treat",
        estimator=did_callaway_santanna,
        adjustment=public("adjustment", method),
        reps=9,
        seed=29,
        aggregate="event_time",
        weights="w",
        allow_time_varying_weights=True,
        incomplete_units="error" if case == "balanced" else "keep",
        effective_sample_warning=None,
        n_threads=2,
        preprocess_mb=4,
        batch_units=37,
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(case, method):
    r = fit(case, method)
    assert r.successful_reps == 9
    prefix = REF / f"{case}_{method}"
    draws = pl.read_csv(str(prefix) + "_draws.csv")
    summary = pl.read_csv(str(prefix) + "_summary.csv")
    errors = []
    for j, name in enumerate(r.coefnames):
        expected = draws.filter(pl.col("term") == name).sort("draw")["estimate"].to_numpy()
        actual = np.r_[r.beta[j], r._draws[:, j]]
        assert_allclose(actual, expected, atol=3e-7, rtol=0)
        errors.append(float(np.max(abs(actual - expected))))
    summary = summary[[summary["term"].to_list().index(n) for n in r.coefnames]]
    assert_allclose(r.se, summary["se"], atol=3e-7, rtol=0)
    for method_name in ["percentile", "basic", "normal"]:
        ci = r.confint(method=method_name)
        assert_allclose(ci["Lower"], summary[method_name + "_low"], atol=6e-7, rtol=0)
        assert_allclose(ci["Upper"], summary[method_name + "_high"], atol=6e-7, rtol=0)
    return dict(case=case, method=method, max_point_error=max(errors))


def provenance():
    p = json.loads((REF / "PROVENANCE.json").read_text())
    for name, expected in p["files"].items():
        assert digest(ROOT / name) == expected, name
    return p


def generate(rscript):
    REF.mkdir(parents=True, exist_ok=True)
    original = DiskGroupResampler.materialize
    for case in ["balanced", "unbalanced_last"]:
        folder = REF / case
        folder.mkdir(exist_ok=True)
        baseline = sample(case)
        if case == "unbalanced_last":
            baseline = baseline.filter(pl.col("time") < 5)
        baseline.write_csv(folder / "sample_0.csv")
        count = 0

        def observe(self, *args, folder=folder, **kwargs):
            nonlocal count
            value = original(self, *args, **kwargs)
            count += 1
            pl.read_parquet(value[0]).rename({"id": "source_id", self.draw_column: "id"}).write_csv(
                folder / f"sample_{count}.csv"
            )
            return value

        with patch.object(DiskGroupResampler, "materialize", observe):
            result = fit(case, "reg")
        assert result.successful_reps == count == 9
    env = dict(os.environ, R_LIBS_USER=str(ROOT / "benchmarks/external/.r/library"))
    subprocess.run([rscript, str(EXPORTER), str(REF)], env=env, check=True)
    files = [EXPORTER, Path(__file__), *REF.rglob("*.csv"), REF / "R_session.txt"]
    (REF / "PROVENANCE.json").write_text(
        json.dumps(
            dict(
                reference="Unmodified R did/DRDID on identical unit-pairs draws; base R sd/quantile(type=7)/qnorm",
                draws=9,
                seed=29,
                coverage_claim=False,
                files={str(p.relative_to(ROOT)): digest(p) for p in files},
            ),
            indent=2,
        )
        + "\n"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generate", action="store_true")
    p.add_argument(
        "--rscript", default=os.environ.get("DIFF_RSCRIPT") or shutil.which("Rscript") or "Rscript"
    )
    args = p.parse_args()
    if args.generate:
        generate(args.rscript)
    provenance()
    print(
        json.dumps(
            [audit(c, m) for c in ["balanced", "unbalanced_last"] for m in ["reg", "ipw", "dr"]],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
