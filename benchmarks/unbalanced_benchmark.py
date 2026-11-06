"""Local observed-row scaling checks; fit plus one BJS-aware aggregation."""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from _harness import measure, run_child

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from diff_stream import diff_stream  # noqa: E402
from diff_stream.simulate import simulate_panel  # noqa: E402


def runner(config, n, path, workdir):
    start = time.perf_counter()
    r = diff_stream(
        "y ~ 1",
        path,
        panel=("id", "time"),
        cohort="first_treat",
        estimator=config,
        panel_mode="unbalanced",
        threads=2,
        preprocess_mb=16,
        workdir=workdir,
    )
    try:
        a = r.aggregate("simple")
        return dict(
            wall_s=time.perf_counter() - start,
            n_obs=r.n_obs,
            checksum=float(a.beta[0]),
            se=float(a.se[0]),
            partitions=r.diagnostics["partitions"],
        )
    finally:
        r.cleanup()


def main():
    import polars as pl

    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int, default=[10000, 100000])
    p.add_argument(
        "--estimators", nargs="+", choices=["cs", "sunab", "bjs"], default=["cs", "sunab", "bjs"]
    )
    a = p.parse_args()
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    env = {
        name: "2"
        for name in [
            "STREAM_LIMITS_MAX_THREADS",
            "POLARS_MAX_THREADS",
            "NUMBA_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
        ]
    }
    env["MPLCONFIGDIR"] = os.environ.get("MPLCONFIGDIR", "/tmp/diff-stream-mpl")
    fields = [
        "model",
        "n_units",
        "status",
        "wall_s",
        "n_obs",
        "peak_memory_mb",
        "baseline_memory_mb",
        "peak_disk_mb",
        "process_s",
        "checksum",
        "se",
        "partitions",
    ]
    with (ROOT / "results/unbalanced.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        for n in a.sizes:
            # Keep the first date, guaranteeing an untreated row for BJS.
            d = simulate_panel(n, 5, seed=872)
            d = d.filter(
                (pl.col("time") == 1) | ((pl.col("id") * 7 + pl.col("time") * 11) % 13 > 2)
            )
            path = ROOT / "data" / f"unbalanced_{n}.parquet"
            d.write_parquet(path, row_group_size=8192)
            del d
            for estimator in a.estimators:
                result = measure(
                    __file__,
                    estimator,
                    n,
                    path,
                    ROOT / "work" / f"unbalanced_{estimator}_{n}",
                    timeout=120,
                    env=env,
                )
                row = dict(model=estimator, n_units=n, **result)
                writer.writerow(row)
                f.flush()
                print(row, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        run_child(runner, *sys.argv[2:])
    else:
        main()
