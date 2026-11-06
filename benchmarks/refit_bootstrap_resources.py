"""Serial/parallel full-refit bootstrap scaling through the shared launcher."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stream_limits.launcher import run_limited  # noqa: E402
from stream_limits.testing import verify_memory_limit  # noqa: E402


def child(c):
    from diff_stream import refit_bootstrap
    from diff_stream.simulate import simulate_conditional_panel

    r = refit_bootstrap(
        "y~x",
        simulate_conditional_panel(c["n"], 5, seed=17),
        panel=("id", "time"),
        cohort="first_treat",
        estimator=c["estimator"],
        reps=c["reps"],
        workers=c["workers"],
        seed=29,
        aggregate=None if c["estimator"] == "gardner" else "simple",
        **({"specification": "static"} if c["estimator"] == "gardner" else {}),
        ess_warn_threshold=None,
        memory_gb=c["memory_gb"],
        preprocess_mb=4,
        batch_units=256,
        threads=2,
        workdir=c["workdir"],
    )
    assert r.successful_reps == c["reps"]
    print(
        json.dumps(
            dict(
                units=r.diagnostics["unit_count"],
                rows=r.diagnostics["analysis_rows"],
                partitions=r.diagnostics["partitions"],
                beta=r.beta.tolist(),
                se=r.se.tolist(),
                draw_plan_sha256=r.diagnostics["draw_plan_sha256"],
                worker_resources=r.diagnostics["worker_resources"],
                parallel_batches=r.diagnostics["parallel_batches"],
            )
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fit", type=Path)
    p.add_argument("--cgroup-parent")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--memory-gb", type=float, default=1.5)
    p.add_argument(
        "--results", type=Path, default=ROOT / "benchmarks/results/refit_bootstrap_resources"
    )
    args = p.parse_args()
    if args.fit:
        child(json.loads(args.fit.read_text()))
        return
    out = args.results.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        STREAM_LIMITS_MAX_THREADS="2",
        POLARS_MAX_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        NUMBA_NUM_THREADS="2",
        OMP_NUM_THREADS="2",
        MPLCONFIGDIR="/tmp/diff-stream-mpl",
    )
    if args.cgroup_parent:
        (out / "selfcheck.json").write_text(
            json.dumps(verify_memory_limit(args.cgroup_parent, env=env), indent=2)
        )
    library = hashlib.sha256()
    for path in sorted((ROOT / "diff_stream").rglob("*.py")):
        library.update(path.read_bytes())
    records = []
    for n in [10000, 100000]:
        for estimator in ["cs", "bjs", "gardner"]:
            name = f"{estimator}_{n}"
            with TemporaryDirectory(prefix=name, dir=out) as work:
                config = dict(
                    n=n,
                    estimator=estimator,
                    workdir=work,
                    reps=args.reps,
                    workers=args.workers,
                    memory_gb=args.memory_gb,
                    library_sha256=library.hexdigest(),
                )
                config_path = out / (name + ".config.json")
                config_path.write_text(json.dumps(config, indent=2))
                print("RUN", name, flush=True)
                result = run_limited(
                    [sys.executable, __file__, "--fit", str(config_path)],
                    mode="hard" if args.cgroup_parent else "watchdog",
                    cgroup_parent=args.cgroup_parent,
                    memory_gb=args.memory_gb,
                    memory_measure="rss",
                    timeout=600,
                    env=env,
                )
                record = dict(
                    config=config,
                    result=result.as_dict(),
                    cleanup_ok=not list(Path(work).iterdir()),
                )
                if result.status == "ok":
                    record["fit"] = json.loads(result.stdout)
                (out / (name + ".json")).write_text(json.dumps(record, indent=2) + "\n")
                records.append(record)
                print(
                    "DONE", name, result.status, round(result.peak_rss_bytes / 2**20, 1), flush=True
                )
    (out / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
    assert all(r["result"]["status"] == "ok" and r["cleanup_ok"] for r in records)


if __name__ == "__main__":
    main()
