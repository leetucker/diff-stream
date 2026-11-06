"""Fresh-process covariate/always-treated Bacon scaling via the shared launcher."""

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
    # Apply scheduler memory defaults before importing Polars in this child.
    from diff_stream import bacon_decomp  # noqa: I001

    import polars as pl

    source = pl.scan_parquet(c["path"], low_memory=True).with_columns(
        pl.when(pl.col("first_treat") == 2)
        .then(1)
        .otherwise(pl.col("first_treat"))
        .alias("first_treat")
    )
    source = source.with_columns(
        *[
            (
                (pl.col("id") * 0.1 * (j + 1) + pl.col("time") * 0.3).sin()
                + 0.03 * pl.col("time") * (pl.col("first_treat") % 3)
            ).alias(f"x{j}")
            for j in range(c["width"])
        ]
    )
    formula = "y~" + "+".join(f"x{j}" for j in range(c["width"]))
    r = bacon_decomp(
        formula,
        source,
        panel=("id", "time"),
        cohort="first_treat",
        weights="w",
        threads=2,
        memory_gb=1.5,
        preprocess_mb=8,
        batch_units=256,
        workdir=c["workdir"],
    )
    assert abs(r.reconstruction_error) < 1e-8
    print(
        json.dumps(
            dict(
                twfe=r.twfe_estimate,
                within_weight=r.within_weight,
                reconstruction_error=r.reconstruction_error,
                n_units=r.n_units,
                n_obs=r.n_obs,
                pairs=r.comparisons().height,
                diagnostics=r.diagnostics,
            )
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", type=Path)
    p.add_argument("--cgroup-parent", type=Path)
    p.add_argument(
        "--results", type=Path, default=ROOT / "benchmarks/results/bacon_extended_resources"
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
    cases = [(10000, 8, 2), (100000, 8, 2), (1000000, 8, 2), (10000, 64, 2), (5000, 16, 32)]
    rows = []
    data = ROOT / "benchmarks/data/bacon_weights_resources"
    data.mkdir(parents=True, exist_ok=True)
    for n, T, K in cases:
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
        name = f"n{n}_t{T}_k{K}"
        with TemporaryDirectory(prefix=name + "-", dir=out) as work:
            c = dict(
                name=name,
                path=str(path),
                width=K,
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
                timeout=300,
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
    failures = [
        r["config"]["name"] for r in rows if r["result"]["status"] != "ok" or not r["cleanup_ok"]
    ]
    save(
        out / "summary.json",
        dict(
            cases=len(rows),
            unexpected_failures=failures,
            rows=[
                dict(
                    name=r["config"]["name"],
                    status=r["result"]["status"],
                    peak_rss_mib=r["result"]["peak_rss_bytes"] / 2**20,
                    elapsed_s=r["result"]["elapsed_s"],
                    **{
                        k: r.get("fit", {}).get(k)
                        for k in [
                            "n_units",
                            "n_obs",
                            "pairs",
                            "within_weight",
                            "reconstruction_error",
                        ]
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
