"""Audit returned authors' Stata BJS estimates, including full covariance.

Run from the checkout; does not require Stata. Raw returned files are read only.
"""

import argparse
import hashlib
import json
import re
import sys
import zipfile
from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
from numpy.testing import assert_allclose
from scipy.stats import f as f_distribution

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks._files import csv_files, read_csv  # noqa: E402
from benchmarks._files import digest as file_digest  # noqa: E402
from diff_stream import bjs_stream

ROOT = Path(__file__).resolve().parents[1]
AVERAGES = ("cohort_time", "event_time", "cohort", "all")


def sha256(path):
    return file_digest(path)


def check_log(directory, *markers):
    """Check a returned Stata log for a clean finish, when the log is present.

    Stata's logs carry the paths of the machine they ran on, so they are kept
    out of the repository (as in the sibling packages); a checkout without them
    still audits every returned estimate and covariance.
    """
    path = directory / "validation.log"
    if not path.exists():
        return
    log = path.read_text(errors="replace")
    for marker in (*markers, ". log close", "closed on:"):
        assert marker in log, marker
    assert not re.search(r"(?m)^r\([1-9][0-9]*\);", log)


def read_export(stem):
    table = read_csv(str(stem) + ".csv")
    names = table["term"].to_list()
    beta = table["estimate"].to_numpy()
    se = table["se"].to_numpy()
    cov = read_csv(str(stem) + "_vcov.csv").to_numpy()
    assert len(names) == len(set(names)), stem
    assert cov.shape == (len(names), len(names)), stem
    assert np.isfinite(beta).all() and np.isfinite(cov).all(), stem
    assert_allclose(cov, cov.T, atol=1e-12, rtol=0)
    assert_allclose(se**2, np.diag(cov), atol=1e-12, rtol=1e-12)
    assert np.linalg.eigvalsh(cov).min() >= -1e-10, stem
    return names, beta, se, cov


def compare(result, exported, labels, *, scale=1, atol=1e-7, check_beta=True):
    names, beta, se, cov = exported
    assert len(names) == len(labels)
    ix = [labels.index(c) for c in result.coefnames]
    beta, se, cov = beta[ix], se[ix] / np.sqrt(scale), cov[np.ix_(ix, ix)] / scale
    if check_beta:
        assert_allclose(result.beta, beta, atol=atol, rtol=0)
    assert_allclose(result.se, se, atol=atol, rtol=0)
    assert_allclose(result.vcov, cov, atol=atol, rtol=0)
    return dict(
        max_abs_beta=float(np.max(np.abs(result.beta - beta))),
        max_abs_se=float(np.max(np.abs(result.se - se))),
        max_abs_vcov=float(np.max(np.abs(result.vcov - cov))),
    )


def placebo_adjustments(meta):
    """Distinct SSC in did_imputation and native reghdfe for this bundle's model.

    All cases absorb(id time) and cluster(id), so the unit FE is nested in the
    cluster. Native reghdfe subtracts one extra degree of freedom; the authors'
    manual score covariance omits that nested adjustment. Do not infer a scale
    from observed variances or apply either correction to post effects.
    """
    residual_df = meta["N"] - meta["df_m"] - meta["df_a"]
    numerator = (meta["N"] - 1) * meta["clusters"] / (meta["clusters"] - 1)
    assert residual_df > 1 and meta["clusters"] > 1
    return dict(imputation=numerator / residual_df, clean_regression=numerator / (residual_df - 1))


def audit_baseline(directory):
    data = ROOT / "tests/data/reference_panel.csv"
    checks = []
    for avg, leaveout in product(AVERAGES, (False, True)):
        fit = bjs_stream(
            "y ~ 1",
            read_csv(data),
            panel=("id", "time"),
            cohort="first_treat",
            avgeffectsby=avg,
            leaveout=leaveout,
            threads=2,
        )
        try:
            for kind in ("dynamic", "simple"):
                stem = f"bjs_{avg}_{'leaveout' if leaveout else 'plugin'}_{kind}"
                exported = read_export(directory / stem)
                labels = ["ATT" if t == "tau" else "event::" + t[3:] for t in exported[0]]
                assert set(labels) == set(fit.aggregate(kind).coefnames)
                checks.append(dict(case=stem, **compare(fit.aggregate(kind), exported, labels)))
        finally:
            fit.cleanup()
    check_log(directory)
    assert len(list(csv_files(directory))) == 32
    archive = ROOT / "benchmarks/results/stata_bundles/stata-bjs-validation.zip"
    provenance = None
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            provenance = json.loads(z.read("diff-stream-stata-bjs/MANIFEST.json"))
        for filename, entry in provenance["files"].items():
            if filename.startswith(("tests/", "benchmarks/")):
                assert sha256(ROOT / filename) == entry["sha256"], filename
    return dict(
        scope="Balanced unweighted unconditional BJS; 4 smart averages x 2 leaveout x 2 aggregates",
        tolerance_absolute=1e-7,
        fits=len(checks),
        checks=checks,
        maxima={key: max(row[key] for row in checks) for key in checks[0] if key != "case"},
        input_sha256=sha256(data),
        original_bundle_manifest=provenance,
        returned_sha256={p.name: sha256(p) for p in sorted(directory.iterdir()) if p.is_file()},
    )


def audit_extended(directory, bundle):
    manifest = json.loads((bundle / "MANIFEST.json").read_text())
    for name, digest in manifest["sha256"].items():
        assert sha256(bundle / name) == digest, name
    statuses = read_csv(directory / "status.csv")
    expected = {
        f"{case['name']}_{avg}_{lo}_{kind}"
        for case in manifest["cases"]
        for avg, lo, kind in product(AVERAGES, ("plugin", "leaveout"), ("dynamic", "simple"))
    }
    assert statuses.height == len(expected)
    assert set(statuses["stem"]) == expected
    assert statuses["rc"].eq(0).all(), statuses.filter(pl.col("rc") != 0)
    assert statuses["clean_rc"].eq(0).all()
    assert len(list(csv_files(directory))) == manifest["expected_csv_files"]
    check_log(directory, "BJS EXTENDED VALIDATION FINISHED")
    checks = []
    for case in manifest["cases"]:
        clean = read_export(directory / (case["name"] + "_clean"))
        meta = read_csv(directory / (case["name"] + "_clean_meta.csv")).row(0, named=True)
        corrections = placebo_adjustments(meta)
        adjustment = corrections["imputation"]
        assert np.isfinite(adjustment) and adjustment > 0
        for avg, leaveout in product(AVERAGES, (False, True)):
            fit = bjs_stream(
                "y ~ 1",
                read_csv(bundle / case["python_input"]),
                panel=("id", "time"),
                cohort="first_treat",
                panel_mode=case["panel_mode"],
                anticipation=case["anticipation"],
                pretrends=case["pretrends"],
                weights="sampling_weight" if case["weighted"] else None,
                avgeffectsby=avg,
                leaveout=leaveout,
                threads=2,
            )
            try:
                placebo = fit.placebos()
                lead_labels = [f"event::{-case['anticipation'] - int(t[4:])}" for t in clean[0]]
                clean_check = compare(
                    placebo, clean, lead_labels, scale=corrections["clean_regression"]
                )
                for kind in ("dynamic", "simple"):
                    stem = f"{case['name']}_{avg}_{'leaveout' if leaveout else 'plugin'}_{kind}"
                    exported = read_export(directory / stem)
                    labels = [
                        f"event::{-case['anticipation'] - int(t[3:])}"
                        if t.startswith("pre")
                        else "ATT"
                        if t == "tau"
                        else f"event::{int(t[3:]) - case['anticipation']}"
                        for t in exported[0]
                    ]
                    assert set(labels) == set(fit.aggregate(kind).coefnames + placebo.coefnames)
                    # Keep strict point-estimate failures in the report and continue
                    # auditing every fit. Do not relax the original 1e-7 threshold.
                    post = compare(fit.aggregate(kind), exported, labels, check_beta=False)
                    pre = compare(placebo, exported, labels, scale=adjustment)
                    stats = read_csv(directory / (stem + "_meta.csv")).row(0, named=True)
                    L = case["pretrends"]
                    F = placebo.pretrend_test()["statistic"] / (L * adjustment)
                    assert_allclose(F, stats["pre_F"], atol=1e-6, rtol=1e-7)
                    assert_allclose(
                        f_distribution.sf(F, L, stats["pre_df"]), stats["pre_p"], atol=1e-7, rtol=0
                    )
                    assert stats["pre_df"] == meta["df_r"]
                    checks.append(
                        dict(
                            case=stem,
                            post=post,
                            placebo=pre,
                            clean_regression=clean_check,
                            adjustments=corrections,
                            clean_metadata=meta,
                            max_abs_F=float(abs(F - stats["pre_F"])),
                            max_abs_F_pvalue=float(
                                abs(f_distribution.sf(F, L, stats["pre_df"]) - stats["pre_p"])
                            ),
                        )
                    )
            finally:
                fit.cleanup()
    point_failures = [row["case"] for row in checks if row["post"]["max_abs_beta"] > 1e-7]
    return dict(
        fits=len(checks),
        passed=not point_failures,
        inference_passed=True,
        point_tolerance_failures=point_failures,
        maxima={
            group: {key: max(row[group][key] for row in checks) for key in checks[0][group]}
            for group in ("post", "placebo", "clean_regression")
        },
        max_abs_F=max(row["max_abs_F"] for row in checks),
        max_abs_F_pvalue=max(row["max_abs_F_pvalue"] for row in checks),
        checks=checks,
        tolerance_absolute=1e-7,
        convention="Post CRV0; did_imputation placebo SSC and native reghdfe SSC differ by nested-FE adjustment of one; native F checked separately",
        bundle_manifest=manifest,
        returned_sha256={p.name: sha256(p) for p in sorted(directory.iterdir()) if p.is_file()},
    )


def resolve_precision(extended, precision):
    """Close only findings covered by a passing rerun of identical specifications.

    Original float-run failures remain in its report. This records the evidence
    linking them to the focused double-storage run, without relaxing tolerance.
    """
    assert extended["inference_passed"] and precision["passed"]
    manifest = precision["bundle_manifest"]
    assert manifest["default_storage_type"] == "double"
    assert manifest["only_estimator_setting_change"] == "set type double"
    source = ROOT / "benchmarks/results/stata_bundles/stata-bjs-extended-validation.zip"
    assert sha256(source) == manifest["source_bundle_sha256"]
    with zipfile.ZipFile(source) as archive:
        assert (
            json.loads(archive.read("diff-stream-stata-bjs-extended/MANIFEST.json"))
            == extended["bundle_manifest"]
        )
        original_script = archive.read(
            "diff-stream-stata-bjs-extended/validate_bjs_extended.do"
        ).decode()
        expected_script = original_script.replace(
            "clear all\n",
            'clear all\nset type double\ndisplay "Default storage type set to double for precision validation"\n',
            1,
        )
        assert (
            hashlib.sha256(expected_script.encode()).hexdigest()
            == manifest["sha256"]["validate_bjs_extended.do"]
        )
    for case in manifest["cases"]:
        assert case in extended["bundle_manifest"]["cases"]
        for key in ("python_input", "stata_input"):
            filename = case[key]
            assert manifest["sha256"][filename] == extended["bundle_manifest"]["sha256"][filename]
    for filename in ("did_imputation.ado", "reghdfe.ado", "ftools.ado"):
        assert extended["returned_sha256"][filename] == precision["returned_sha256"][filename]
    rerun = {row["case"]: row for row in precision["checks"]}
    assert set(extended["point_tolerance_failures"]) <= rerun.keys()
    original = {row["case"]: row for row in extended["checks"]}
    assert rerun.keys() <= original.keys()
    effective = [rerun.get(name, row) for name, row in original.items()]
    assert all(row["post"]["max_abs_beta"] <= extended["tolerance_absolute"] for row in effective)
    return dict(
        passed=True,
        inference_passed=True,
        fits=len(effective),
        precision_fits=len(rerun),
        resolved_point_tolerance_failures=extended["point_tolerance_failures"],
        point_tolerance_failures=[],
        tolerance_absolute=extended["tolerance_absolute"],
        resolution="Identical inputs, specifications and installed command sources; focused rerun changes only default temporary storage to double",
        original_maxima=extended["maxima"],
        precision_maxima=precision["maxima"],
        effective_maxima={
            group: {key: max(row[group][key] for row in effective) for key in effective[0][group]}
            for group in ("post", "placebo", "clean_regression")
        },
        original_bundle_manifest=extended["bundle_manifest"],
        precision_bundle_manifest=manifest,
        original_returned_sha256=extended["returned_sha256"],
        precision_returned_sha256=precision["returned_sha256"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "benchmarks/results/stata_bjs")
    parser.add_argument(
        "--bundle", type=Path, help="Extracted extended bundle root, if reviewing extended results"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmarks/results/stata_bjs_review/report.json"
    )
    parser.add_argument(
        "--precision-directory",
        type=Path,
        help="Optional focused rerun results to resolve original findings",
    )
    parser.add_argument("--precision-bundle", type=Path, help="Extracted focused rerun bundle root")
    args = parser.parse_args()
    if bool(args.precision_directory) != bool(args.precision_bundle):
        parser.error("--precision-directory and --precision-bundle must be supplied together")
    if args.precision_directory and not args.bundle:
        parser.error("Precision resolution also requires the original extended --bundle")
    report = (
        audit_extended(args.directory, args.bundle)
        if args.bundle
        else audit_baseline(args.directory)
    )
    if args.precision_directory:
        precision = audit_extended(args.precision_directory, args.precision_bundle)
        report = resolve_precision(report, precision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k in ("fits", "maxima", "scope")}))
    print(args.output)
    if report.get("point_tolerance_failures"):
        print(
            f"REVIEW REQUIRED: {len(report['point_tolerance_failures'])} fits exceed point tolerance 1e-7; inference checks pass"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
