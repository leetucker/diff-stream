"""Returned-output integrity and explicit conversion of Stata SSC conventions."""

import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from benchmarks._files import exists as csv_exists
from benchmarks.stata_bjs_review import (
    audit_baseline,
    audit_extended,
    compare,
    placebo_adjustments,
    read_export,
    resolve_precision,
)

ROOT = Path(__file__).resolve().parents[1]


def test_stata_review_baseline_when_available():
    directory = ROOT / "benchmarks/results/stata_bjs"
    if not csv_exists(directory / "bjs_cohort_time_plugin_dynamic.csv"):
        pytest.skip("External Stata baseline not supplied")
    report = audit_baseline(directory)
    assert report["fits"] == 16
    assert report["maxima"]["max_abs_vcov"] < 1e-7


def test_distinct_nested_fe_placebo_corrections():
    correction = placebo_adjustments(dict(N=100, df_m=3, df_a=7, clusters=20))
    assert correction["imputation"] == pytest.approx(99 / 90 * 20 / 19)
    assert correction["clean_regression"] == pytest.approx(99 / 89 * 20 / 19)


def test_export_integrity_and_reordered_ssc_conversion(tmp_path):
    stem = tmp_path / "example"
    # External order reversed; exported covariance includes a 1.5 finite-dof factor.
    pl.DataFrame(dict(term=["pre1", "pre2"], estimate=[2.0, 1.0], se=[3.0, 6.0])).write_csv(
        str(stem) + ".csv"
    )
    V = np.array([[9.0, 3.0], [3.0, 36.0]])
    pl.DataFrame(V).write_csv(str(stem) + "_vcov.csv")
    result = SimpleNamespace(
        coefnames=["event::-2", "event::-1"],
        beta=np.array([1.0, 2.0]),
        se=np.sqrt(np.array([36.0, 9.0]) / 1.5),
        vcov=V[::-1, ::-1] / 1.5,
    )
    check = compare(result, read_export(stem), ["event::-1", "event::-2"], scale=1.5)
    assert check["max_abs_vcov"] == 0
    shifted = SimpleNamespace(**{**result.__dict__, "beta": result.beta + 2e-7})
    with pytest.raises(AssertionError):
        compare(shifted, read_export(stem), ["event::-1", "event::-2"], scale=1.5)
    finding = compare(
        shifted, read_export(stem), ["event::-1", "event::-2"], scale=1.5, check_beta=False
    )
    assert finding["max_abs_beta"] > 1e-7
    # Corrupted/mismatched standalone standard errors must fail even if V is correct.
    pl.DataFrame(dict(term=["pre1", "pre2"], estimate=[2.0, 1.0], se=[3.0, 5.0])).write_csv(
        str(stem) + ".csv"
    )
    with pytest.raises(AssertionError):
        read_export(stem)


def test_stata_review_extended_when_available(tmp_path):
    directory = ROOT / "benchmarks/results/stata_bjs_extended"
    if not csv_exists(directory / "status.csv"):
        pytest.skip("Extended Stata bundle has not yet been returned")
    with zipfile.ZipFile(
        ROOT / "benchmarks/results/stata_bundles/stata-bjs-extended-validation.zip"
    ) as archive:
        archive.extractall(tmp_path)
    report = audit_extended(directory, tmp_path / "diff-stream-stata-bjs-extended")
    assert report["fits"] == 176
    assert report["inference_passed"]
    if report["point_tolerance_failures"]:
        known = {row["case"] for row in report["checks"] if row["case"].startswith("mpdta_")}
        known.update({"balanced_w1_a2_all_leaveout_dynamic", "complete_w1_a2_all_leaveout_dynamic"})
        assert set(report["point_tolerance_failures"]) == known
        assert report["maxima"]["post"]["max_abs_beta"] < 2.6e-7
        precision_directory = ROOT / "benchmarks/results/stata_bjs_precision"
        if csv_exists(precision_directory / "status.csv"):
            with zipfile.ZipFile(
                ROOT / "benchmarks/results/stata_bundles/stata-bjs-precision-validation.zip"
            ) as archive:
                archive.extractall(tmp_path)
            precision = audit_extended(
                precision_directory, tmp_path / "diff-stream-stata-bjs-precision"
            )
            resolved = resolve_precision(report, precision)
            assert resolved["fits"] == 176 and resolved["precision_fits"] == 48
            assert len(resolved["resolved_point_tolerance_failures"]) == 18
            assert resolved["passed"] and not resolved["point_tolerance_failures"]
            return
        pytest.xfail(
            "Strict point tolerance exceeded; external double-precision confirmation pending"
        )
    assert report["passed"]


def test_stata_review_precision_when_available(tmp_path):
    directory = ROOT / "benchmarks/results/stata_bjs_precision"
    if not csv_exists(directory / "status.csv"):
        pytest.skip("Focused double-storage Stata run not supplied")
    with zipfile.ZipFile(
        ROOT / "benchmarks/results/stata_bundles/stata-bjs-precision-validation.zip"
    ) as archive:
        archive.extractall(tmp_path)
    report = audit_extended(directory, tmp_path / "diff-stream-stata-bjs-precision")
    assert report["fits"] == 48
    assert report["passed"] and not report["point_tolerance_failures"]
