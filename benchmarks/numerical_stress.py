"""Offline numerical stress matrix for conditional CS and its shared solvers.

R did/DRDID references use canonical units for intentional unit-rescaling cases.
Both successes and explicit refusals are retained; no tolerances are widened
based on the observed discrepancies.
"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import resource_checks as h
from diff_stream import cs_stream

OUT = h.ROOT / "benchmarks/results/numerical_stress"
SCENARIOS = (
    "baseline",
    "x_small",
    "x_large",
    "y_small",
    "y_large",
    "collinear_mild",
    "collinear_border",
    "overlap_weak",
    "overlap_very_weak",
    "weights_wide",
    "weights_aligned",
    "rare_cohort",
    "singular",
    "large_offset",
    "large_offset_centered",
    "separated",
    "quasi_separated",
)
REFUSALS = {"singular", "large_offset", "separated", "quasi_separated"}


def sample(scenario, unbalanced=False, n=2400):
    rng = np.random.default_rng(271828)
    T = 5
    x0, z0 = rng.normal(size=(2, n))
    strength = {"overlap_weak": 3.0, "overlap_very_weak": 8.0}.get(scenario, 0.5)
    eta = x0[:, None] * np.array([-strength, 0.0, strength])
    if scenario == "rare_cohort":
        eta[:, 1:] -= 3
    p = np.exp(eta - eta.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    g = np.array([0, 3, 5])[(rng.random(n)[:, None] > p.cumsum(axis=1)).sum(axis=1)]
    t = np.arange(1, T + 1)
    x = x0[:, None] + 0.05 * t + 0.1 * rng.normal(size=(n, T))
    z = np.broadcast_to(z0[:, None], x.shape).copy()
    y = (
        rng.normal(size=(n, 1))
        + 0.2 * t
        + 0.15 * t * x0[:, None]
        + 0.1 * t * z0[:, None]
        + 0.3 * x
        + rng.normal(size=x.shape)
        + ((g[:, None] > 0) & (t >= g[:, None])) * (1 + 0.1 * t)
    )
    w = np.ones(n)
    if scenario == "weights_wide":
        w = np.exp(rng.uniform(-14, 14, n))
    elif scenario == "weights_aligned":
        w = np.exp(np.clip(5 * x0, -25, 25))
    if scenario.startswith("collinear_"):
        z = x + {"collinear_mild": 1e-3, "collinear_border": 1e-5}[scenario] * rng.normal(
            size=x.shape
        )
    if scenario == "singular":
        z = 2 * x
    if scenario in ("separated", "quasi_separated"):
        x = np.broadcast_to(np.where(g > 0, 1.0, -1.0)[:, None], x.shape).copy()
        if scenario == "quasi_separated":
            for cohort in (0, 3, 5):
                tied = np.flatnonzero(g == cohort)[:4]
                x[tied] = 0
                z[tied] = np.array([-1.0, 1.0, -1.0, 1.0])[:, None]
    frame = pl.DataFrame(
        dict(
            id=np.repeat(np.arange(n), T),
            time=np.tile(t, n),
            first_treat=np.repeat(g, T),
            x=x.ravel(),
            z=z.ravel(),
            y=y.ravel(),
            w=np.repeat(w, T),
        )
    )
    if unbalanced:
        frame = frame.filter(~((pl.col("id") % 4 == 0) & pl.col("time").is_in([2, 4])))
    canonical = frame
    if scenario in ("x_small", "x_large"):
        frame = frame.with_columns(pl.col("x") * (1e-12 if scenario == "x_small" else 1e12))
    if scenario in ("large_offset", "large_offset_centered"):
        frame = frame.with_columns(pl.col("x") + 1e8)
        canonical = frame.with_columns(pl.col("x") - 1e8)
    scale = {"y_small": 1e-6, "y_large": 1e6}.get(scenario, 1.0)
    frame = frame.with_columns(pl.col("y") * scale)
    return frame, canonical, scale


def result_payload(fit, scale):
    a = fit.aggregate("simple")
    return dict(
        cells=[list(c) for c in fit.cells],
        beta=(fit.beta / scale).tolist(),
        vcov=(fit.vcov / scale**2).tolist(),
        att=float(a.beta[0] / scale),
        se=float(a.se[0] / abs(scale)),
        diagnostics=fit.diagnostics,
    )


def run(out, rscript):
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".driver.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "run.json").exists():
        history = out / "history" / str(time.time_ns())
        history.mkdir(parents=True)
        for previous in out.iterdir():
            if previous.is_file() and not previous.name.startswith("."):
                shutil.copyfile(previous, history / previous.name)
    data_dir = h.ROOT / "benchmarks/data/numerical_stress"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows, references = [], []
    h.save(
        out / "environment.json",
        dict(
            h.environment(),
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            library_sha256={
                str(p.relative_to(h.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((h.ROOT / "diff_stream").rglob("*.py"))
            },
            r_exporter_sha256=hashlib.sha256(
                (h.ROOT / "benchmarks/external/numerical_stress.R").read_bytes()
            ).hexdigest(),
        ),
    )
    for mode in ("balanced", "unbalanced"):
        for scenario in SCENARIOS:
            d, canonical, scale = sample(scenario, mode == "unbalanced")
            path = data_dir / f"{mode}_{scenario}.csv"
            canonical.write_csv(path)
            fml = "y ~ I(x-1e8) + z" if scenario == "large_offset_centered" else "y ~ x + z"
            for method in ("reg", "ipw", "dr"):
                name = f"{mode}_{scenario}_{method}"
                start = time.perf_counter()
                row = dict(
                    name=name,
                    scenario=scenario,
                    method=method,
                    panel_mode=mode,
                    outcome_scale=scale,
                    expected_refusal=(
                        scenario in REFUSALS
                        and not (scenario == "quasi_separated" and method == "reg")
                    ),
                    canonical_input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
                fit = None
                try:
                    fit = cs_stream(
                        fml,
                        d,
                        panel=("id", "time"),
                        cohort="first_treat",
                        est_method=method,
                        panel_mode=mode,
                        weights="w",
                        threads=2,
                        batch_units=137,
                        preprocess_mb=4,
                        trim_level=0.8 if scenario.startswith("overlap_") else 0.995,
                    )
                    payload = result_payload(fit, scale)
                    assert np.isfinite(fit.beta).all() and np.isfinite(fit.vcov).all()
                    assert np.linalg.eigvalsh(fit.vcov).min() >= -1e-8 * max(
                        1.0, float(np.max(np.diag(fit.vcov)))
                    )
                    row.update(status="ok", fit=payload)
                except Exception as exc:
                    row.update(status="refused", error=f"{type(exc).__name__}: {exc}")
                finally:
                    if fit is not None:
                        fit.cleanup()
                row["wall_s"] = time.perf_counter() - start
                row["expected_outcome"] = row["status"] == (
                    "refused" if row["expected_refusal"] else "ok"
                )
                rows.append(row)
                if row["status"] == "ok" and not row["expected_refusal"]:
                    references.append(
                        dict(
                            name=name,
                            csv_path=str(path.relative_to(h.ROOT)),
                            method=method,
                            panel_mode=mode,
                            trim_level=0.8 if scenario.startswith("overlap_") else 0.995,
                        )
                    )
                h.save(out / (name + ".json"), row)
                print(f"{name}: {row['status']} {row.get('error', '')}", flush=True)
    h.save(out / "reference_jobs.json", references)
    env = dict(os.environ, R_LIBS_USER=str(h.RLIB), R_USER_CONFIG_DIR="/tmp/diff-stress-r-config")
    proc = subprocess.run(
        [
            str(rscript),
            str(h.ROOT / "benchmarks/external/numerical_stress.R"),
            str(out / "reference_jobs.json"),
            str(out / "r_reference.json"),
        ],
        env=env,
        cwd=h.ROOT,  # job paths are relative to the repository
        capture_output=True,
        text=True,
        timeout=900,
    )
    (out / "r.stdout").write_text(proc.stdout)
    (out / "r.stderr").write_text(proc.stderr)
    proc.check_returncode()
    refs = {r["name"]: r for r in json.loads((out / "r_reference.json").read_text())}
    for row in rows:
        reference = refs.get(row["name"])
        if reference:
            row["reference"] = reference
            row["comparison"] = h.compare(
                dict(status=row["status"], fit=row.get("fit", {})),
                dict(status=reference["status"], fit=reference.get("fit", {})),
            )
            if reference["status"] != "ok":
                row["expected_outcome"] = False
            elif not row["comparison"]["passed"]:
                row["expected_outcome"] = False
            h.save(out / (row["name"] + ".json"), row)
    failed = [r["name"] for r in rows if not r["expected_outcome"]]
    h.save(out / "run.json", dict(cases=len(rows), unexpected=failed, rows=rows))
    print("Unexpected outcomes:", failed)
    return bool(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=OUT)
    parser.add_argument("--rscript", type=Path, default=h.RSCRIPT)
    args = parser.parse_args()
    raise SystemExit(run(args.results.resolve(), args.rscript))
