"""Reproduce the fixed 100,000-unit anticipation=2 resource smoke check offline.

Input generation happens in a separate process and is excluded from the fit
budget. Fits/aggregation/cleanup run serially under a 1 GB sampled watchdog.
"""

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from resource_checks import ROOT, monitor, save

DATA = ROOT / "benchmarks/data/anticipation_resource_panel.parquet"
OUTPUT = ROOT / "benchmarks/results/anticipation_resources"


def generate(sampling_weights=False):
    sys.path.insert(0, str(ROOT))
    import polars as pl

    from diff_stream.simulate import simulate_conditional_panel

    p = simulate_conditional_panel(100000, 8, seed=741)
    p = p.with_columns(
        (
            pl.col("y")
            + pl.when(
                (pl.col("first_treat") > 0)
                & (pl.col("time") >= pl.col("first_treat") - 2)
                & (pl.col("time") < pl.col("first_treat"))
            )
            .then(0.4)
            .otherwise(0)
        ).alias("y")
    )
    p = p.filter(~((pl.col("id") % 4 == 0) & pl.col("time").is_in([3, 7])))
    if sampling_weights:
        p = p.with_columns((0.2 + pl.col("id") % 9).alias("w"))
    data = DATA.with_name("weights_resource_panel.parquet") if sampling_weights else DATA
    data.parent.mkdir(parents=True, exist_ok=True)
    p.write_parquet(data, row_group_size=8192)


def main(sampling_weights=False):
    data = DATA.with_name("weights_resource_panel.parquet") if sampling_weights else DATA
    output = OUTPUT.with_name("weights_resources") if sampling_weights else OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        STREAM_LIMITS_MAX_THREADS="2",
        POLARS_MAX_THREADS="2",
        NUMBA_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        OMP_NUM_THREADS="2",
        STREAM_LIMITS_MEMORY_GB="1",
        MPLCONFIGDIR="/tmp/diff-stream-mpl",
        PYTHONPATH=str(ROOT),
    )
    if not data.exists():
        subprocess.run(
            [sys.executable, __file__, "--generate"] + (["--weights"] if sampling_weights else []),
            env=env,
            check=True,
        )
    for estimator in ["cs", "sunab", "bjs"]:
        with TemporaryDirectory(prefix="diff-anticipation-watch-") as work:
            config = dict(
                estimator=estimator,
                anticipation=2,
                weights="w" if sampling_weights else None,
                path=str(data),
                workdir=work,
                memory_gb=1.0,
                timeout=120,
            )
            path = output / f"{estimator}.config.json"
            save(path, config)
            result = monitor(
                [sys.executable, str(ROOT / "benchmarks/anticipation_resource_fit.py"), str(path)],
                env,
                config,
                output / estimator,
            )
            result["cleanup_ok"] = not list(Path(work).iterdir())
            save(output / f"{estimator}.json", result)
            print(estimator, result["status"], round(result["peak_rss_mib"], 1), flush=True)
            if result["status"] != "ok" or not result["cleanup_ok"]:
                raise RuntimeError(f"{estimator}: anticipation resource check failed")
    save(
        output / "input.json",
        dict(
            n_units=100000,
            periods=8,
            seed=741,
            anticipation=2,
            missing_rows="id%4==0 and time in (3,7)",
            bjs_inference="all groups; leaveout",
            generator="simulate_conditional_panel; add .4 inside two-period anticipation window",
            network_required=False,
            sampling_weights="w=0.2+id%9" if sampling_weights else None,
        ),
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        generate(sampling_weights="--weights" in sys.argv)
    else:
        main()
