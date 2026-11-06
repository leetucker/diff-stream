"""Build a portable follow-up Stata BJS bundle; preflight every Python fit."""

import argparse
import json
import shutil
import sys
import zipfile
from itertools import product
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks._files import read_csv  # noqa: E402
from benchmarks.stata_bjs_review import AVERAGES, ROOT, sha256
from diff_stream import bjs_stream


def build(destination, *, preflight=True):
    destination.mkdir(parents=True, exist_ok=True)
    base = read_csv(ROOT / "tests/data/anticipation_panel.csv").select(
        "id", "time", "first_treat", "y", (0.2 + pl.col("id") % 9).alias("sampling_weight")
    )
    unbalanced = base.filter(~((pl.col("id") % 4 == 0) & pl.col("time").is_in([3, 7])))
    samples = dict(
        balanced=base,
        unbalanced=unbalanced,
        complete=base.filter(pl.col("id") % 4 != 0),
        future=base.with_columns(
            pl.when((pl.col("first_treat") == 0) & (pl.col("id") % 2 == 0))
            .then(9)
            .otherwise(pl.col("first_treat"))
            .alias("first_treat")
        ),
        mpdta=read_csv(ROOT / "tests/data/mpdta.csv").select(
            pl.col("countyreal").alias("id"),
            pl.col("year").alias("time"),
            pl.col("first.treat").alias("first_treat"),
            pl.col("lemp").alias("y"),
            pl.lit(1.0).alias("sampling_weight"),
        ),
    )
    for name, data in samples.items():
        data.write_csv(destination / f"{name}.csv")
    settings = list(product(("balanced", "unbalanced"), (False, True), (0, 2)))
    settings += [("complete", True, 2), ("future", True, 0), ("mpdta", False, 0)]
    cases, commands, expected = [], [], []
    for mode, weighted, anticipation in settings:
        name = f"{mode}_w{int(weighted)}_a{anticipation}"
        case = dict(
            name=name,
            stata_input=f"{mode}.csv",
            python_input=f"{'unbalanced' if mode == 'complete' else mode}.csv",
            panel_mode=mode if mode in ("unbalanced", "complete") else "balanced",
            anticipation=anticipation,
            pretrends=1 if mode == "mpdta" else 3,
            weighted=weighted,
        )
        cases.append(case)
        commands.append(
            f'run_case, name("{name}") input("{case["stata_input"]}") '
            f"anticipation({anticipation}) pretrends({case['pretrends']}) weighted({int(weighted)})"
        )
        if preflight:
            for avg, leaveout in product(AVERAGES, (False, True)):
                fit = bjs_stream(
                    "y ~ 1",
                    read_csv(destination / case["python_input"]),
                    panel=("id", "time"),
                    cohort="first_treat",
                    panel_mode=case["panel_mode"],
                    weights="sampling_weight" if weighted else None,
                    anticipation=anticipation,
                    pretrends=case["pretrends"],
                    avgeffectsby=avg,
                    leaveout=leaveout,
                    threads=2,
                )
                try:
                    for kind in ("dynamic", "simple"):
                        result = fit.aggregate(kind)
                        expected.append(
                            dict(
                                case=name,
                                average=avg,
                                leaveout=leaveout,
                                kind=kind,
                                coefficients=result.coefnames,
                                beta=result.beta.tolist(),
                                vcov=result.vcov.tolist(),
                                placebo_beta=fit.placebos().beta.tolist(),
                                placebo_vcov=fit.placebos().vcov.tolist(),
                            )
                        )
                finally:
                    fit.cleanup()
            print(f"Preflight passed: {name}", flush=True)
    (destination / "cases.do").write_text("\n".join(commands) + "\n")
    shutil.copyfile(
        ROOT / "benchmarks/external/validate_bjs_extended.do",
        destination / "validate_bjs_extended.do",
    )
    (destination / "RUN_STATA.do").write_text(
        "* Set Stata working directory to this extracted folder, then run this file.\n"
        "do validate_bjs_extended.do\n"
    )
    (destination / "README.txt").write_text(
        "BJS follow-up validation (11 cases, 176 did_imputation fits)\n\n"
        "1. Extract the whole ZIP to a new writable folder. Do not reuse old results.\n"
        "2. In Stata, cd to that folder (the one containing RUN_STATA.do).\n"
        "3. Ensure did_imputation, reghdfe and ftools are installed. Use the same versions\n"
        "   as the first bundle if possible; no package updates are required by this bundle.\n"
        "4. Run: do RUN_STATA.do\n"
        "5. Return the entire results folder, including status.csv and validation.log.\n"
        "   Place it at benchmarks/results/stata_bjs_extended in the Python checkout.\n\n"
        "No Python, R or original repository is needed on the Stata machine.\n"
        "The log should end with BJS EXTENDED VALIDATION FINISHED. Fits that fail are\n"
        "recorded in status.csv and the script continues; return failures too. Do not\n"
        "manually remove terms, change tolerances, or rerun individual outputs in place.\n\n"
        "Cases cover weights, balanced/native unbalanced/complete panels, anticipation,\n"
        "future-treated controls, and the public mpdta dataset. Each case crosses four\n"
        "smart averages, plugin/leaveout inference, and dynamic/simple aggregation.\n"
        "Simple ATT excludes anticipation rows via wtr(actual_post). Clean placebos\n"
        "retain actual future cohort labels. CSVs import as double; exports retain\n"
        "17 significant digits. Authors' placebo SSC inputs and F tests are recorded.\n"
        "Python comparison divides only placebo covariance by the recorded SSC;\n"
        "post covariance is CRV0. Post/placebo cross covariance has no Python counterpart.\n\n"
        "Input origins: anticipation_panel.csv is the checkout's deterministic synthetic\n"
        "fixture; mpdta.csv is frozen from R did::mpdta (see docs/mpdta_replication.md).\n"
        "MANIFEST.json records cases, source hashes, and transferred file hashes.\n"
        "This bundle has Python preflight checks; its Stata execution remains external.\n"
    )
    manifest = dict(
        cases=cases,
        expected_fits=len(cases) * 16,
        expected_csv_files=len(cases) * 51 + 1,
        source_sha256={
            str(p.relative_to(ROOT)): sha256(p)
            for p in [
                ROOT / "tests/data/anticipation_panel.csv",
                ROOT / "tests/data/mpdta.csv",
                ROOT / "benchmarks/external/validate_bjs_extended.do",
            ]
        },
        sha256={
            p.name: sha256(p)
            for p in sorted(destination.iterdir())
            if p.is_file() and p.name != "MANIFEST.json"
        },
    )
    (destination / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = ROOT / "benchmarks/results/stata_bjs_review/extended_python_preflight.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(dict(fits=len(expected), checks=expected), indent=2) + "\n")
    archive = ROOT / "benchmarks/results/stata_bundles/stata-bjs-extended-validation.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(destination.iterdir()):
            if p.is_file():
                z.write(p, f"diff-stream-stata-bjs-extended/{p.name}")
    print(archive)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=ROOT / "benchmarks/work/stata-bjs-extended"
    )
    args = parser.parse_args()
    build(args.directory)
