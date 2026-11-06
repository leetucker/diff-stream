"""Fresh-process feTR scaling, unbalanced/reversing treatment and cell export."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stream_limits.launcher import run_limited  # noqa: E402
from stream_limits.testing import verify_memory_limit  # noqa: E402

from benchmarks.gardner_resources import save  # noqa: E402


def child(c):
    # Set scheduler defaults before importing Polars.
    from diff_stream import twfe_weights  # noqa: I001

    import polars as pl
    import pyarrow.parquet as pq

    source = pl.scan_parquet(c["path"], low_memory=True)
    if c["unbalanced"]:
        source = source.filter(~((pl.col("id") % 5 == 0) & (pl.col("time") == 3)))
    raw = ((pl.col("first_treat") > 0) & (pl.col("time") >= pl.col("first_treat"))).cast(pl.Float64)
    # Mostly staggered adoption (negative weights), plus explicit treatment exits.
    source = source.with_columns(
        pl.when((pl.col("id") % 23 == 0) & (pl.col("time") >= c["periods"] - 1))
        .then(1 - raw)
        .otherwise(raw)
        .alias("d"),
        (pl.col("w") * (1 + 0.05 * pl.col("time"))).alias("w"),
        *[
            (
                (pl.col("id") * 0.1 * (j + 1) + pl.col("time") * 0.3).sin()
                + 0.03 * pl.col("time") * (pl.col("first_treat") % 3)
            ).alias(f"x{j}")
            for j in range(c["width"])
        ],
    )
    export = Path(c["workdir"]) / "export.parquet"
    with twfe_weights(
        "y~" + "+".join(f"x{j}" for j in range(c["width"])),
        source,
        panel=("id", "time"),
        treatment="d",
        panel_mode="unbalanced" if c["unbalanced"] else "balanced",
        weights="w",
        allow_time_varying_weights=True,
        keep_cells=True,
        threads=2,
        memory_gb=1.5,
        preprocess_mb=8,
        batch_units=128,
        workdir=c["workdir"],
    ) as r:
        assert abs(r.summary_dict()["weights"]["raw_weight_sum"] - 1) < 1e-8
        assert r.summary_dict()["weights"]["negative_weight_mass"] > 0
        assert r.sensitivity().row(1, named=True)["status"] == "finite"
        assert r.summary_dict()["sensitivity"]["cutoff_iterations"] > 0
        r.write_cells(export)
        result = r.summary_dict()
    with pq.ParquetFile(export, pre_buffer=False) as pf:
        assert pf.metadata.num_rows == result["n_obs"]
    result["export_bytes"] = export.stat().st_size
    export.unlink()
    print(json.dumps(result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", type=Path)
    p.add_argument("--cgroup-parent", type=Path)
    p.add_argument(
        "--results", type=Path, default=ROOT / "benchmarks/results/twfe_weights_resources"
    )
    a = p.parse_args()
    if a.fit:
        child(json.loads(a.fit.read_text()))
        return
    from benchmarks import resource_checks as h

    out = a.results.resolve()
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
    if a.cgroup_parent:
        save(out / "cgroup_selfcheck.json", verify_memory_limit(a.cgroup_parent, env=env))
    save(out / "environment.json", h.environment())
    library = hashlib.sha256()
    for path in sorted((ROOT / "diff_stream").rglob("*.py")):
        library.update(path.read_bytes())
    rows = []
    data = ROOT / "benchmarks/data/bacon_weights_resources"
    data.mkdir(parents=True, exist_ok=True)
    for n, T, K, unbalanced in [
        (10000, 8, 2, False),
        (100000, 8, 2, False),
        (1000000, 8, 2, False),
        (1000000, 8, 2, True),
        (10000, 64, 2, True),
        (5000, 16, 32, True),
    ]:
        path = data / f"n{n}_t{T}.parquet"
        if not path.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmarks/bacon_resources.py"),
                    "--generate",
                    str(n),
                    str(T),
                    str(path),
                    "--weights",
                ],
                env=env,
                check=True,
            )
        name = f"n{n}_t{T}_k{K}_u{int(unbalanced)}"
        with TemporaryDirectory(prefix=name + "-", dir=out) as work:
            c = dict(
                name=name,
                path=str(path),
                width=K,
                periods=T,
                unbalanced=unbalanced,
                workdir=work,
                library_sha256=library.hexdigest(),
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            )
            config = out / (name + ".config.json")
            save(config, c)
            print("RUN", name, flush=True)
            result = run_limited(
                [sys.executable, __file__, "--fit", str(config)],
                mode="hard" if a.cgroup_parent else "watchdog",
                cgroup_parent=a.cgroup_parent,
                memory_gb=1.5,
                memory_measure="rss",
                timeout=600,
                env=env,
            )
            record = dict(
                config=c, result=result.as_dict(), cleanup_ok=not list(Path(work).iterdir())
            )
            if result.status == "ok":
                record["fit"] = json.loads(result.stdout)
            save(out / (name + ".json"), record)
            rows.append(record)
            print("DONE", name, result.status, round(result.peak_rss_bytes / 2**20, 1), flush=True)
    failed = [
        r["config"]["name"] for r in rows if r["result"]["status"] != "ok" or not r["cleanup_ok"]
    ]
    save(
        out / "summary.json",
        dict(
            cases=len(rows),
            unexpected_failures=failed,
            rows=[
                dict(
                    name=r["config"]["name"],
                    status=r["result"]["status"],
                    peak_rss_mib=r["result"]["peak_rss_bytes"] / 2**20,
                    elapsed_s=r["result"]["elapsed_s"],
                    n_units=r.get("fit", {}).get("n_units"),
                    n_obs=r.get("fit", {}).get("n_obs"),
                    export_bytes=r.get("fit", {}).get("export_bytes"),
                )
                for r in rows
            ],
        ),
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
