"""Review no-never-treated support without changing estimator policies.

Fits explicit clean-window counterfactuals using existing entry points. Native
R probes distinguish valid restricted references from unrestricted failures or
automatic sample changes. --regenerate requires the cached validation sources.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stream_limits import MemoryBudget  # noqa: E402

from benchmarks._estimators import public  # noqa: E402
from benchmarks._files import csv_files, read_csv  # noqa: E402
from benchmarks._files import digest as file_digest  # noqa: E402
from benchmarks._files import exists as csv_exists  # noqa: E402
from benchmarks.etwfe_validation import SOURCE as ETWFE_SOURCE  # noqa: E402
from benchmarks.etwfe_validation import dense_oracle as etwfe_dense  # noqa: E402
from benchmarks.gardner_validation import SOURCE as GARDNER_SOURCE  # noqa: E402
from benchmarks.gardner_validation import dense_oracle as gardner_dense  # noqa: E402
from diff_stream import (  # noqa: E402
    did_bjs,  # noqa: E402
    did_extended_twfe,
    did_two_stage,
)
from diff_stream.reference import reference_store  # noqa: E402
from diff_stream.store import PanelStore, inspect  # noqa: E402

OUT = ROOT / "benchmarks/results/reference_population_review"
EXPORTER = ROOT / "benchmarks/external/reference_population_review.R"
KW = dict(panel=("id", "time"), cohort="first_treat", n_threads=2, preprocess_mb=4, batch_units=37)


def cases():
    for panel in ["balanced", "unbalanced"]:
        for A in [0, 1]:
            for rich in [False, True]:
                yield dict(name=f"{panel}_{A}_{int(rich)}", panel=panel, A=A, rich=rich)


def sample(c, restricted=False):
    d = read_csv(
        ROOT
        / "tests/data"
        / ("richer_panel.csv" if c["panel"] == "balanced" else "richer_unbalanced_panel.csv")
    )
    d = d.filter(pl.col("first_treat") != 0)
    if restricted:
        last = d["first_treat"].max()
        d = d.filter(pl.col("time") < last - c["A"]).with_columns(
            pl.when(pl.col("first_treat") == last)
            .then(0)
            .otherwise(pl.col("first_treat"))
            .alias("first_treat")
        )
    return d


def configurations(c):
    rich = c["rich"]
    yield "bjs", dict(fml="y~x+I(x**2)|region^time" if rich else "y~1"), did_bjs
    for spec in ["static", "dynamic"]:
        yield (
            "gardner_" + spec,
            dict(fml="y~x+I(x**2)|region^time" if rich else "y~1", event_study=spec == "dynamic"),
            did_two_stage,
        )
    for inf in ["conditional", "unconditional"]:
        yield (
            "etwfe_" + inf,
            dict(
                fml="y~x+z" if rich else "y~1",
                control_group="not_yet_treated",
                treat_sample_as_fixed=inf == "conditional",
            ),
            did_extended_twfe,
        )


def compare(r, name, depvar="term"):
    ref = read_csv(OUT / (name + ".csv"), null_values="NA")
    terms = [f"g::{g}:t::{t}" for g, t in r.cells] if "_etwfe_" in name else r.coefnames
    ix = [ref[depvar].to_list().index(t) for t in terms]
    be = float(np.max(abs(r.beta - ref["estimate"].to_numpy()[ix])))
    assert_allclose(r.beta, ref["estimate"].to_numpy()[ix], atol=3e-8, rtol=0)
    ve = None
    if csv_exists(OUT / (name + "_vcov.csv")):
        V = read_csv(OUT / (name + "_vcov.csv"), has_header=False).to_numpy()
        factor = r.n_units / (r.n_units - 1) if name.endswith("etwfe_unconditional") else 1
        ve = float(np.max(abs(r.vcov * factor - V[np.ix_(ix, ix)])))
        assert_allclose(r.vcov * factor, V[np.ix_(ix, ix)], atol=3e-8, rtol=0)
    return dict(beta_max_abs=be, vcov_max_abs=ve)


def audit():
    manifest = json.loads((OUT / "PROVENANCE.json").read_text())
    for name, h in manifest["files"].items():
        assert file_digest(ROOT / name) == h, name
    rows, dynamic_reference_differences, placebos = [], [], []
    for c in cases():
        full, restricted = sample(c), sample(c, True)
        weight = "w" if c["rich"] else None
        for name, kwargs, entry in configurations(c):
            options = dict(
                **KW,
                incomplete_units=public("incomplete_units", c["panel"]),
                anticipation=c["A"],
                weights=weight,
                allow_time_varying_weights=True,
            )
            try:
                unrestricted = entry(data=full, **kwargs, **options)
            except ValueError as e:
                current_full_status, current_full_message = "refused", str(e)
            else:
                current_full_status, current_full_message = "returned", ""
                unrestricted.cleanup()
            r = entry(data=restricted, **kwargs, **options)
            try:
                target = r.aggregate("event_time") if name == "bjs" else r
                errors = compare(target, c["name"] + "_" + name)
                if name.startswith("gardner"):
                    terms, b, V = gardner_dense(
                        restricted,
                        "extra_fe_varying" if c["rich"] else "plain_equal",
                        kwargs["specification"],
                        c["A"],
                    )
                    ix = [terms.index(t) for t in r.coefnames]
                    assert_allclose(r.beta, b[ix], atol=3e-8, rtol=0)
                    assert_allclose(r.vcov, V[np.ix_(ix, ix)], atol=3e-8, rtol=0)
                elif name.startswith("etwfe"):
                    _, b, V, _ = etwfe_dense(
                        restricted,
                        "numeric_varying" if c["rich"] else "plain_equal",
                        "notyet",
                        "cohort",
                        c["A"],
                        kwargs["inference"],
                    )
                    assert_allclose(r.beta, b, atol=3e-8, rtol=0)
                    assert_allclose(r.vcov, V, atol=3e-8, rtol=0)
                rows.append(
                    dict(
                        case=c["name"],
                        estimator=name,
                        rows=restricted.height,
                        units=restricted["id"].n_unique(),
                        last_clean_time=restricted["time"].max(),
                        current_full_status=current_full_status,
                        current_full_message=current_full_message,
                        **errors,
                    )
                )
                if name == "gardner_dynamic":
                    native = read_csv(
                        OUT / (c["name"] + "_gardner_dynamic_full_native.csv"), null_values="NA"
                    )
                    common = [n for n in r.coefnames if n in native["term"].to_list()]
                    if common:
                        errors = {
                            n: float(
                                abs(
                                    r.beta[r.coefnames.index(n)]
                                    - native.filter(pl.col("term") == n)["estimate"][0]
                                )
                            )
                            for n in common
                        }
                        dynamic_reference_differences.append(
                            dict(
                                case=c["name"],
                                max_common_difference=max(errors.values()),
                                by_term=errors,
                            )
                        )
            finally:
                r.cleanup()
        # Known future adoption dates still define BJS clean-placebo indicators.
        raw_window = full.filter(pl.col("time") < full["first_treat"].max() - c["A"])
        opts = dict(
            **KW,
            incomplete_units=public("incomplete_units", c["panel"]),
            anticipation=c["A"],
            pretrend_leads=2,
            weights=weight,
            allow_time_varying_weights=True,
        )
        raw_fit = did_bjs("y~1", raw_window, **opts)
        coded_fit = did_bjs("y~1", restricted, **opts)
        try:
            placebos.append(
                dict(
                    case=c["name"],
                    max_abs_difference=float(
                        np.max(abs(raw_fit.placebos().beta - coded_fit.placebos().beta))
                    ),
                )
            )
        finally:
            raw_fit.cleanup()
            coded_fit.cleanup()
    # A global last-cohort cutoff is insufficient for cohort-specific crossed FEs.
    d = sample(next(cases()), True).with_columns(
        (pl.col("first_treat") == 0).cast(pl.Int64).alias("region")
    )
    failures = []
    for name, entry in [("bjs", did_bjs), ("gardner", did_two_stage)]:
        try:
            entry(
                "y~1|region^time",
                d,
                **KW,
            )
        except ValueError as e:
            assert "unidentified" in str(e), str(e)
            failures.append(dict(estimator=name, reason=str(e)))
        else:
            raise AssertionError("unsupported crossed-FE imputation was not refused")
    futures = []
    for A in [0, 1]:
        d = sample(next(cases())).with_columns(
            pl.when(pl.col("first_treat") == 7)
            .then(9)
            .otherwise(pl.col("first_treat"))
            .alias("first_treat")
        )
        for name, entry in [
            ("bjs", did_bjs),
            ("gardner", did_two_stage),
            ("etwfe", did_extended_twfe),
        ]:
            result = entry(
                "y~1",
                d,
                anticipation=A,
                **KW,
            )
            assert result.n_obs == len(d)
            futures.append(dict(estimator=name, anticipation=A, retained_rows=result.n_obs))
            result.cleanup()
    d = sample(next(cases()))
    incomplete = d.filter(~((pl.col("first_treat") == 7) & (pl.col("time") == 7)))
    with TemporaryDirectory(prefix="diff-reference-review-") as work:
        lf, _, _ = inspect("y~1", incomplete, ("id", "time"), "first_treat")
        store = PanelStore(
            lf, work, 37, MemoryBudget(None), 4, panel_mode="complete", require_never_treated=False
        )
        with warnings.catch_warnings(record=True):
            restricted, selection = reference_store(store, "auto", 0)
        assert selection["reference_cohort"] == 6 and restricted.times[-1] == 5
        complete_policy = dict(
            **selection, original_last_cohort=7, naive_truncate_first_last_clean_time=6
        )
    probes = read_csv(OUT / "native_probes.csv", null_values="NA").to_dicts()
    for probe in probes:
        if probe["status"] == "returned":
            tab = read_csv(
                OUT / (probe["name"] + "_native.csv"), null_values=["NA", "NaN", "Inf", "-Inf"]
            )
            probe["returned_rows"] = len(tab)
            probe["nonfinite_estimates"] = tab["estimate"].null_count() + int(
                (~tab["estimate"].is_finite()).fill_null(False).sum()
            )
    report = dict(
        restricted_comparisons=len(rows),
        passed=True,
        rows=rows,
        unsupported_strata=failures,
        native_unrestricted_probes=probes,
        gardner_dynamic_reference_differences=dynamic_reference_differences,
        bjs_placebo_relabeling_differences=placebos,
        future_cohort_full_window_fits=futures,
        complete_panel_selection=complete_policy,
        review_only=True,
        production_reference_policies_changed=False,
    )
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regenerate", action="store_true")
    p.add_argument(
        "--rscript", default=os.environ.get("DIFF_RSCRIPT") or shutil.which("Rscript") or "Rscript"
    )
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.regenerate:
        for src in [ETWFE_SOURCE, GARDNER_SOURCE]:
            identity = json.loads((src / "source.json").read_text())
            for name, item in identity["files"].items():
                assert hashlib.sha256((src / name).read_bytes()).hexdigest() == item["sha256"]
        for c in cases():
            sample(c).write_csv(OUT / (c["name"] + ".input.csv"))
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
                str(OUT),
                str(GARDNER_SOURCE / "R/did2s.R"),
                str(ETWFE_SOURCE / "R"),
            ],
            env=env,
            check=True,
        )
        files = [
            *csv_files(OUT),
            OUT / "R_session.txt",
            EXPORTER,
            ROOT / "tests/data/richer_panel.csv",
            ROOT / "tests/data/richer_unbalanced_panel.csv",
        ]
        manifest = dict(
            files={str(f.relative_to(ROOT)): file_digest(f) for f in files},
            gardner_source=json.loads((GARDNER_SOURCE / "source.json").read_text()),
            etwfe_source=json.loads((ETWFE_SOURCE / "source.json").read_text()),
            bjs_reference="R didimputation 0.5.1, points only; no new native Stata inference claim",
        )
        (OUT / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    r = audit()
    print(
        json.dumps(
            {
                k: r[k]
                for k in [
                    "restricted_comparisons",
                    "passed",
                    "review_only",
                    "production_reference_policies_changed",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
