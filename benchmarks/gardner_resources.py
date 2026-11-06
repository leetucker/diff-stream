"""Fresh-process local Gardner scaling probes through the shared launcher.

Uses cached richer-model datasets. --cgroup-parent opts into verified Linux
kernel enforcement; otherwise an explicit portable RSS watchdog is used.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stream_limits.launcher import run_limited  # noqa: E402
from stream_limits.testing import verify_memory_limit  # noqa: E402

from benchmarks._estimators import public  # noqa: E402


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def child(config):
    import polars as pl

    from diff_stream import did_two_stage

    start = time.monotonic()
    fit = did_two_stage(
        "y ~ x + I(x**2) | region^time",
        config["path"],
        panel=("id", "time"),
        cohort="first_treat",
        event_study=public("event_study", config["specification"]),
        incomplete_units="keep",
        weights="w",
        allow_time_varying_weights=True,
        n_threads=2,
        max_memory_gb=config["memory_gb"],
        batch_units=config["batch_units"],
        preprocess_mb=config["preprocess_mb"],
        workdir=config["workdir"],
    )
    print(
        json.dumps(
            dict(
                n_units=fit.n_units,
                n_obs=fit.n_obs,
                columns=len(fit.diagnostics["first_stage_columns"]),
                coefficients=len(fit.beta),
                fit_s=time.monotonic() - start,
                names=fit.coefnames,
                beta=fit.beta.tolist(),
                vcov=fit.vcov.tolist(),
                diagnostics=fit.diagnostics,
                polars_version=pl.__version__,
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--cgroup-parent", type=Path)
    parser.add_argument("--only", nargs="+")
    parser.add_argument(
        "--results", type=Path, default=ROOT / "benchmarks/results/gardner_resources"
    )
    args = parser.parse_args()
    if args.fit:
        child(json.loads(args.fit.read_text()))
        return
    from benchmarks import resource_checks as h
    from benchmarks.richer_resources import DATA

    jobs = [
        dict(name=f"dynamic_{n}", n=n, T=8, G=4, regions=4, specification="dynamic")
        for n in [10000, 100000, 1000000]
    ]
    jobs.extend(
        [
            dict(name="dynamic_wide", n=20000, T=16, G=6, regions=8, specification="dynamic"),
            dict(
                name="dynamic_100000_tight",
                n=100000,
                T=8,
                G=4,
                regions=4,
                specification="dynamic",
                batch_units=64,
                preprocess_mb=4,
            ),
            dict(name="static_100000", n=100000, T=8, G=4, regions=4, specification="static"),
        ]
    )
    if args.only and set(args.only) - {j["name"] for j in jobs}:
        raise ValueError("unknown case")
    env = dict(
        os.environ,
        STREAM_LIMITS_MAX_THREADS="2",
        POLARS_MAX_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        NUMBA_NUM_THREADS="2",
        OMP_NUM_THREADS="2",
        MPLCONFIGDIR="/tmp/diff-stream-mpl",
        PYTHONPATH=str(ROOT),
    )
    out = args.results.resolve()
    out.mkdir(parents=True, exist_ok=True)
    save(out / "environment.json", h.environment())
    if args.cgroup_parent:
        save(out / "cgroup_selfcheck.json", verify_memory_limit(args.cgroup_parent, env=env))
    source = hashlib.sha256()
    for path in sorted((ROOT / "diff_stream").rglob("*.py")):
        source.update(str(path.relative_to(ROOT)).encode())
        source.update(path.read_bytes())
    rows = []
    for job in jobs:
        if args.only and job["name"] not in args.only:
            continue
        spec = {k: job[k] for k in ["n", "T", "G", "regions"]}
        DATA.mkdir(parents=True, exist_ok=True)
        path = DATA / (h.identity(spec) + ".parquet")
        if not path.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmarks/richer_resources.py"),
                    "--generate",
                    json.dumps(dict(spec=spec, path=str(path))),
                ],
                env=env,
                check=True,
            )
        with TemporaryDirectory(prefix=job["name"] + "-", dir=out) as work:
            config = dict(memory_gb=1.5, batch_units=256, preprocess_mb=8)
            config.update(job)
            # Optional tighter buffers override the common defaults explicitly.
            config.update(
                path=str(path),
                workdir=work,
                library_sha256=source.hexdigest(),
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            )
            payload = out / (job["name"] + ".config.json")
            save(payload, config)
            print("RUN", job["name"], flush=True)
            result = run_limited(
                [sys.executable, __file__, "--fit", str(payload)],
                mode="hard" if args.cgroup_parent else "watchdog",
                cgroup_parent=args.cgroup_parent,
                memory_gb=config["memory_gb"],
                timeout=300,
                env=env,
            )
            record = dict(
                config=config,
                result=result.as_dict(),
                cleanup_ok=not list(Path(work).iterdir()),
                input=json.loads(path.with_suffix(".json").read_text()),
            )
            if result.status == "ok":
                record["fit"] = json.loads(result.stdout)
            save(out / (job["name"] + ".json"), record)
            rows.append(record)
            print(
                "DONE",
                job["name"],
                result.status,
                round(result.peak_rss_bytes / 2**20, 1),
                flush=True,
            )
    failures = [
        r["config"]["name"] for r in rows if r["result"]["status"] != "ok" or not r["cleanup_ok"]
    ]
    comparison = None
    byname = {r["config"]["name"]: r for r in rows}
    if not failures and {"dynamic_100000", "dynamic_100000_tight"} <= byname.keys():
        import numpy as np

        a, b = [byname[n]["fit"] for n in ["dynamic_100000", "dynamic_100000_tight"]]
        assert a["names"] == b["names"]
        comparison = {
            k: float(np.max(abs(np.asarray(a[k]) - np.asarray(b[k])))) for k in ["beta", "vcov"]
        }
        assert max(comparison.values()) < 2e-9
    save(
        out / "summary.json",
        dict(
            cases=len(rows),
            unexpected_failures=failures,
            buffer_invariance=comparison,
            rows=[
                dict(
                    name=r["config"]["name"],
                    status=r["result"]["status"],
                    peak_rss_mib=r["result"]["peak_rss_bytes"] / 2**20,
                    elapsed_s=r["result"]["elapsed_s"],
                    **{
                        k: r.get("fit", {}).get(k)
                        for k in ["n_units", "n_obs", "columns", "coefficients"]
                    },
                )
                for r in rows
            ],
        ),
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
