"""Reproduce and audit the documented mpdta CS example against installed R did.

Default operation uses committed data/reference fixtures and needs Python only.
--regenerate refreshes those fixtures with R, including common bootstrap draws.
"""

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import polars as pl

from benchmarks._estimators import public
from benchmarks._files import digest as file_digest  # noqa: E402
from benchmarks._files import read_csv  # noqa: E402
from diff_stream import did_callaway_santanna

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data/mpdta.csv"
REF = ROOT / "tests/data/mpdta_reference"
OUT = ROOT / "benchmarks/results/mpdta_replication"
SCENARIOS = list(
    product(
        ("constant", "lpop"),
        ("nevertreated", "notyettreated"),
        ("varying", "universal"),
        ("reg", "ipw", "dr"),
    )
)
TOL = 2e-8


def canonical(control, base, method):
    return (control, base, method) == ("nevertreated", "varying", "dr")


def compare_case(data, design, control, base, method):
    boot = canonical(control, base, method)
    f = did_callaway_santanna(
        "lemp ~ 1" if design == "constant" else "lemp ~ lpop",
        data,
        panel=("countyreal", "year"),
        cohort="first.treat",
        adjustment=public("adjustment", method),
        control_group=public("control_group", control),
        base_period=base,
        n_threads=2,
        multiplier_bootstrap_reps=999 if boot else 0,
        multiplier_bootstrap_seed=815 if boot else None,
        batch_units=73,
        preprocess_mb=64,
    )
    prefix = f"{design}_{control}_{base}_{method}"
    differences = {}

    def check(key, a, b):
        a, b = np.asarray(a), np.asarray(b)
        assert_allclose(a, b, atol=TOL, rtol=1e-8, err_msg=f"{prefix}: {key}")
        differences[key] = max(differences.get(key, 0.0), float(np.max(np.abs(a - b))))

    cells = read_csv(REF / f"{prefix}_cells.csv", null_values="NA")
    keys = list(zip(cells["g"], cells["t"], strict=True))
    ix = [keys.index(c) for c in f.cells]
    check("cell_estimates", f.beta, cells["att"].to_numpy()[ix])
    check("cell_se", f.se, cells["se"].to_numpy()[ix])
    check(
        "cell_covariance",
        f.vcov,
        read_csv(REF / f"{prefix}_cells_vcov.csv").to_numpy()[np.ix_(ix, ix)],
    )
    if boot:
        assert f.diagnostics["partitions"] == 1, (
            "common draws require a single sorted county-ID leaf"
        )
    rows = []
    for kind in ("dynamic", "simple", "cohort", "calendar", "balanced"):
        a = (
            f.aggregate(balanced_horizon=1)
            if kind == "balanced"
            else f.aggregate(public("kind", kind))
        )
        label = "group" if kind == "cohort" else kind
        z = read_csv(REF / f"{prefix}_{label}.csv", null_values="NA")
        order = [z["term"].to_list().index(n) for n in a.coefnames]
        check("aggregate_estimates", a.beta, z["estimate"].to_numpy()[order])
        check("aggregate_se", a.se, z["se"].to_numpy()[order])
        check(
            "aggregate_covariance",
            a.vcov,
            read_csv(REF / f"{prefix}_{label}_vcov.csv").to_numpy()[np.ix_(order, order)],
        )
        band = None
        if boot:
            draws = read_csv(REF / f"{prefix}_{label}_draws.csv.gz").to_numpy()[:, order]
            check("bootstrap_draws", a.multiplier_draws(), draws)
            band = a.multiplier_confint()
            zboot = read_csv(REF / f"{prefix}_{label}_bootstrap.csv")
            for col, refcol, key in [
                ("Std. Error", "se", "bootstrap_se"),
                ("critical_value", "critical", "bootstrap_critical"),
                ("2.5%", "lower", "bootstrap_bounds"),
                ("97.5%", "upper", "bootstrap_bounds"),
            ]:
                check(key, band[col].to_numpy(), zboot[refcol].to_numpy()[order])
        for j, name in enumerate(a.coefnames):
            rows.append(
                dict(
                    case=prefix,
                    aggregation=kind,
                    term=name,
                    estimate=float(a.beta[j]),
                    analytic_se=float(a.se[j]),
                    bootstrap_se=float(band["Std. Error"][j]) if band is not None else None,
                    band_lower=float(band["2.5%"][j]) if band is not None else None,
                    band_upper=float(band["97.5%"][j]) if band is not None else None,
                )
            )
    pre = f.pretrend_test()
    z = read_csv(REF / "pretests.csv").filter(pl.col("case") == prefix).row(0, named=True)
    check("pretrend_statistic", pre["statistic"], z["statistic"])
    check("pretrend_pvalue", pre["pvalue"], z["pvalue"])
    assert pre["df1"] == z["df"]
    assert f.n_units == 500 and f.n_obs == 2500
    return dict(
        case=prefix, status="passed", max_abs_difference=differences, pretrend_pvalue=pre["pvalue"]
    ), rows


def regenerate(rscript):
    REF.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("R_LIBS_USER", str(ROOT / "benchmarks/external/.r/library"))
    env.setdefault("R_USER_CONFIG_DIR", "/tmp/diff-mpdta-r-config")
    Path(env["R_USER_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    command = [rscript, str(ROOT / "benchmarks/external/mpdta_reference.R"), str(DATA), str(REF)]
    subprocess.run(command + ["--data-only"], env=env, check=True)
    from hdfe_stream.core.wildboot import WEIGHTS

    n = read_csv(DATA)["countyreal"].n_unique()
    xi = WEIGHTS["rademacher"](n * 999, np.random.default_rng(815)).reshape(n, 999)
    with gzip.open(REF / "multipliers.csv.gz", "wt") as stream:
        np.savetxt(stream, xi, delimiter=",", fmt="%d")
    subprocess.run(command, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument(
        "--rscript", default=os.environ.get("DIFF_RSCRIPT") or shutil.which("Rscript") or "Rscript"
    )
    args = parser.parse_args()
    if args.regenerate:
        regenerate(args.rscript)
    data = read_csv(DATA)
    OUT.mkdir(parents=True, exist_ok=True)
    results, rows = [], []
    for scenario in SCENARIOS:
        result, case_rows = compare_case(data, *scenario)
        results.append(result)
        rows.extend(case_rows)
        print(result["case"], result["status"], flush=True)
    maxima = {
        key: max(r["max_abs_difference"].get(key, 0) for r in results)
        for key in results[-1]["max_abs_difference"]
    }
    for r in results:
        for key, value in r["max_abs_difference"].items():
            maxima[key] = max(maxima.get(key, 0), value)
    metadata = dict(
        status="passed",
        specifications=len(results),
        units=data["countyreal"].n_unique(),
        rows=len(data),
        years=sorted(data["year"].unique().to_list()),
        columns=data.columns,
        cohort_sizes=read_csv(REF / "cohorts.csv").to_dicts(),
        tolerance=TOL,
        max_abs_difference=maxima,
        reference_versions=(REF / "versions.txt").read_text().splitlines(),
        files={
            str(p.relative_to(ROOT)): file_digest(p)
            for p in [
                DATA,
                ROOT / "benchmarks/external/mpdta_reference.R",
                Path(__file__),
                REF / "versions.txt",
                REF / "multipliers.csv.gz",
            ]
        },
        python_version=sys.version,
        numpy_version=np.__version__,
        polars_version=pl.__version__,
        bootstrap=dict(reps=999, seed=815, weights="rademacher", common_draws=True),
        stata_executed=False,
        results=results,
    )
    (OUT / "report.json").write_text(json.dumps(metadata, indent=2) + "\n")
    pl.DataFrame(rows).write_csv(OUT / "estimates.csv")
    print(
        json.dumps(dict(status="passed", specifications=len(results), max_abs_difference=maxima)),
        flush=True,
    )


if __name__ == "__main__":
    main()
