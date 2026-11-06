"""Offline fresh-process resource probes for richer SA/BJS and varying CS weights."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import resource_checks as h

OUT = h.ROOT / "benchmarks/results/richer_resources"
DATA = h.ROOT / "benchmarks/data/richer_resources"


def jobs():
    out = []
    for n in [10000, 100000]:
        for estimator in ["cs", "sunab", "bjs"]:
            out.append(
                dict(
                    name=f"{estimator}_{n}",
                    n=n,
                    T=8,
                    G=4,
                    regions=4,
                    estimator=estimator,
                    memory_gb=1.0,
                )
            )
    for estimator in ["sunab", "bjs"]:
        out.append(
            dict(
                name=f"{estimator}_1000000",
                n=1000000,
                T=8,
                G=4,
                regions=4,
                estimator=estimator,
                memory_gb=1.5,
            )
        )
        out.append(
            dict(
                name=f"{estimator}_wide",
                n=20000,
                T=16,
                G=6,
                regions=8,
                estimator=estimator,
                memory_gb=1.5,
            )
        )
    return out


def generate(spec, path):
    import polars as pl
    import pyarrow.parquet as pq

    base = path.with_suffix(".base.parquet")
    h.generate(
        h.dataset(spec["n"], periods=spec["T"], cohorts=spec["G"], pattern="entry_exit"), base
    )
    try:
        (
            pl.scan_parquet(base, low_memory=True)
            .with_columns((pl.col("id") % spec["regions"]).alias("region"))
            .with_columns(
                (0.2 * pl.col("x") + 0.04 * pl.col("time") * ((pl.col("id") % 5) - 2))
                .exp()
                .alias("w"),
                (
                    pl.col("y") + 0.2 * pl.col("x") ** 2 + 0.03 * pl.col("region") * pl.col("time")
                ).alias("y"),
            )
            .sink_parquet(path, engine="streaming", row_group_size=8192)
        )
    finally:
        base.unlink()
        base.with_suffix(".json").unlink()
    with pq.ParquetFile(path) as pf:
        rows = pf.metadata.num_rows
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    h.save(
        path.with_suffix(".json"),
        dict(spec=spec, n_rows=rows, sha256=digest.hexdigest(), parquet_bytes=path.stat().st_size),
    )


def run(only=None, out=OUT, parent=None):
    out.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    lock = (out / ".driver.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    library = hashlib.sha256()
    for path in sorted((h.ROOT / "diff_stream").rglob("*.py")):
        library.update(str(path.relative_to(h.ROOT)).encode())
        library.update(path.read_bytes())
    h.save(out / "environment.json", h.environment())
    env = dict(os.environ, PYTHONPATH=str(h.ROOT), MPLCONFIGDIR="/tmp/diff-stream-mpl")
    for key in [
        "STREAM_LIMITS_MAX_THREADS",
        "POLARS_MAX_THREADS",
        "NUMBA_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
    ]:
        env[key] = "2"
    if parent:
        from benchmarks.memory_checks import cgroup_selfcheck

        cgroup_selfcheck(parent, out, env)
    rows = []
    selected = [c for c in jobs() if not only or c["name"] in only]
    if only and set(only) - {c["name"] for c in selected}:
        raise ValueError("unknown case")
    for c in selected:
        c.update(
            cgroup_parent=str(parent) if parent else None,
            enforcement_version=__import__("stream_limits").__version__ if parent else None,
            library_sha256=library.hexdigest(),
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            fit_driver_sha256=hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest(),
        )
        destination = out / (c["name"] + ".json")
        if destination.exists():
            previous = json.loads(destination.read_text())
            if all(previous["config"].get(k) == v for k, v in c.items()) and h.accepted(previous):
                print("REUSE", c["name"], flush=True)
                rows.append(previous)
                continue
            archive = out / "history" / (c["name"] + "-" + str(time.time_ns()))
            archive.mkdir(parents=True)
            for p in out.glob(c["name"] + ".*"):
                if p.is_file():
                    archive.joinpath(p.name).write_bytes(p.read_bytes())
        spec = {k: c[k] for k in ["n", "T", "G", "regions"]}
        path = DATA / (h.identity(spec) + ".parquet")
        if not path.exists():
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--generate",
                    json.dumps(dict(spec=spec, path=str(path))),
                ],
                env=env,
                check=True,
            )
        with TemporaryDirectory(prefix=c["name"] + "-", dir=out) as work:
            config = dict(
                c,
                path=str(path),
                formula="y ~ x + z"
                if c["estimator"] == "cs"
                else "y ~ x + I(x**2) | id + time + region^time",
                panel_mode="unbalanced",
                method="dr",
                threads=2,
                batch_units=256,
                preprocess_mb=8,
                timeout=300,
                weights="w",
                allow_time_varying_weights=True,
                pretrends=1 if c["estimator"] == "bjs" else 0,
                avgeffectsby="all" if c["estimator"] == "bjs" else "cohort_time",
                leaveout=c["estimator"] == "bjs",
                extra_aggregations=True,
                workdir=work,
                progress_path=str(out / (c["name"] + ".progress.json")),
            )
            payload = out / (c["name"] + ".config.json")
            h.save(payload, config)
            print("RUN", c["name"], flush=True)
            command = [sys.executable, str(Path(h.__file__)), "--fit", str(payload)]
            from stream_limits.enforcement import MemoryCgroup

            group = MemoryCgroup(parent, c["memory_gb"] * 1e9) if parent else None
            try:
                result = h.monitor(
                    group.command(command) if group else command,
                    dict(env, STREAM_LIMITS_MEMORY_GB=str(c["memory_gb"])),
                    config,
                    out / c["name"],
                )
                if group:
                    result["cgroup"] = group.metrics()
                    if result["cgroup"]["events"].get("oom_kill", 0):
                        result["status"] = "cgroup_oom"
            finally:
                if group:
                    group.close()
            result["cleanup_ok"] = not list(Path(work).iterdir())
            result["input"] = json.loads(path.with_suffix(".json").read_text())
            h.save(destination, result)
        rows.append(result)
        print(
            "DONE",
            c["name"],
            result["status"],
            round(result["peak_rss_mib"], 1),
            round(result["process_s"], 1),
            flush=True,
        )
    failures = [r["config"]["name"] for r in rows if not h.accepted(r)]
    h.save(
        out / "summary.json",
        dict(
            cases=len(rows),
            unexpected_failures=failures,
            rows=[
                dict(
                    name=r["config"]["name"],
                    status=r["status"],
                    n_obs=r["fit"].get("n_obs"),
                    peak_rss_mib=r["peak_rss_mib"],
                    process_s=r["process_s"],
                    peak_disk_mb=r["peak_disk_mb"],
                    cleanup_ok=r["cleanup_ok"],
                    columns=len(r["fit"].get("diagnostics", {}).get("covariate_names", [])),
                )
                for r in rows
            ],
        ),
    )
    return bool(failures)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        a = json.loads(sys.argv[2])
        generate(a["spec"], Path(a["path"]))
    else:
        p = argparse.ArgumentParser(description=__doc__)
        p.add_argument("--only", nargs="+")
        p.add_argument("--results", type=Path, default=OUT)
        p.add_argument("--cgroup-parent", type=Path)
        args = p.parse_args()
        raise SystemExit(run(args.only, args.results.resolve(), args.cgroup_parent))
