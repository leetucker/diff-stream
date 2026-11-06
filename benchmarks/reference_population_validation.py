"""Public BJS/Gardner/ETWFE last-reference audit; frozen native R comparisons."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks._files import csv_files, read_csv  # noqa: E402
from benchmarks._files import digest as file_digest  # noqa: E402
from benchmarks._files import exists as csv_exists  # noqa: E402
from benchmarks.reference_population_review import (  # noqa: E402
    ETWFE_SOURCE,
    GARDNER_SOURCE,
    KW,
    cases,
    configurations,
    sample,
)
from diff_stream import ReferenceWindowWarning, bjs_stream  # noqa: E402

REF = ROOT / "tests/data/reference_population_extensions"
EXPORTER = ROOT / "benchmarks/external/reference_population_extensions.R"


def digest(p):
    return file_digest(p)


def provenance():
    value = json.loads((REF / "PROVENANCE.json").read_text())
    for name, h in value["files"].items():
        assert digest(ROOT / name) == h, name
    return value


def compare(result, prefix):
    ref = read_csv(REF / (prefix + ".csv"), null_values="NA")
    names = [f"g::{g}:t::{t}" for g, t in result.cells] if "_etwfe_" in prefix else result.coefnames
    ix = [ref["term"].to_list().index(n) for n in names]
    beta_error = float(np.max(abs(result.beta - ref["estimate"].to_numpy()[ix])))
    assert_allclose(result.beta, ref["estimate"].to_numpy()[ix], atol=3e-8, rtol=0)
    covariance_error = None
    if csv_exists(REF / (prefix + "_vcov.csv")):
        V = read_csv(REF / (prefix + "_vcov.csv"), has_header=False).to_numpy()[np.ix_(ix, ix)]
        factor = (
            result.n_units / (result.n_units - 1) if prefix.endswith("etwfe_unconditional") else 1
        )
        covariance_error = float(np.max(abs(result.vcov * factor - V)))
        assert_allclose(result.vcov * factor, V, atol=3e-8, rtol=0)
    return dict(beta_max_abs=beta_error, vcov_max_abs=covariance_error)


def options(c):
    return dict(
        **KW,
        panel_mode=c["panel"],
        anticipation=c["A"],
        weights="w" if c["rich"] else None,
        allow_time_varying_weights=True,
    )


def audit():
    import warnings

    provenance()
    rows = []
    for c in cases():
        for name, kwargs, entry in configurations(c):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ReferenceWindowWarning)
                with entry(
                    data=sample(c),
                    reference_group="auto",
                    **kwargs,
                    **options(c),
                    **({"ess_warn_threshold": None} if name == "bjs" else {}),
                ) as r:
                    assert any(isinstance(w.message, ReferenceWindowWarning) for w in caught)
                    target = r.aggregate("dynamic") if name == "bjs" else r
                    rows.append(
                        dict(
                            case=c["name"],
                            estimator=name,
                            **compare(target, c["name"] + "_" + name),
                        )
                    )
        fml = "y~x+I(x**2)|region^time" if c["rich"] else "y~1"
        with warnings.catch_warnings(record=True):
            with bjs_stream(
                fml, sample(c), pretrends=2, ess_warn_threshold=None, **options(c)
            ) as r:
                rows.append(
                    dict(
                        case=c["name"],
                        estimator="bjs_placebos",
                        **compare(r.placebos(), c["name"] + "_bjs_placebos"),
                    )
                )
    report = dict(cases=len(rows), passed=True, rows=rows)
    out = ROOT / "benchmarks/results/reference_population_extensions"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regenerate", action="store_true")
    p.add_argument(
        "--rscript", default=os.environ.get("DIFF_RSCRIPT") or shutil.which("Rscript") or "Rscript"
    )
    a = p.parse_args()
    if a.regenerate:
        REF.mkdir(parents=True, exist_ok=True)
        for source in [ETWFE_SOURCE, GARDNER_SOURCE]:
            for name, item in json.loads((source / "source.json").read_text())["files"].items():
                assert digest(source / name) == item["sha256"]
        for c in cases():
            sample(c).write_csv(REF / (c["name"] + ".input.csv"))
        env = dict(
            os.environ,
            R_LIBS_USER=str(ROOT / "benchmarks/external/.r/library"),
            OPENBLAS_NUM_THREADS="2",
            OMP_NUM_THREADS="2",
            TZ="UTC",
        )
        subprocess.run(
            [
                a.rscript,
                str(EXPORTER),
                str(REF),
                str(GARDNER_SOURCE / "R/did2s.R"),
                str(ETWFE_SOURCE / "R"),
                str(ROOT),
            ],
            env=env,
            check=True,
        )
        paths = [
            *csv_files(REF),
            REF / "R_session.txt",
            EXPORTER,
            ROOT / "benchmarks/external/reference_population_review.R",
            ROOT / "tests/data/richer_panel.csv",
            ROOT / "tests/data/richer_unbalanced_panel.csv",
        ]
        identity = dict(
            files={str(f.relative_to(ROOT)): digest(f) for f in paths},
            gardner_source=json.loads((GARDNER_SOURCE / "source.json").read_text()),
            etwfe_source=json.loads((ETWFE_SOURCE / "source.json").read_text()),
            bjs="R didimputation points and clean-lead estimates/SEs; fixest full clean-lead covariance; no new native Stata run",
        )
        (REF / "PROVENANCE.json").write_text(json.dumps(identity, indent=2) + "\n")
    report = audit()
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
