"""Offline 10k/100k-unit weighted unbalanced/complete BJS clean-placebo checks."""

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from resource_checks import ROOT, monitor, save

OUT = ROOT / "benchmarks/results/bjs_placebo_resources"
DATA = ROOT / "benchmarks/data/weights_resource_panel.parquet"
CASES = [
    (10000, 2, "unbalanced"),
    (100000, 2, "unbalanced"),
    (100000, 0, "unbalanced"),
    (100000, 2, "complete"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
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
    if not DATA.exists():
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "benchmarks/anticipation_resources.py"),
                "--generate",
                "--weights",
            ],
            env=env,
            check=True,
        )
    for n, A, mode in CASES:
        name = f"n{n}_a{A}_{mode}"
        with TemporaryDirectory(prefix="diff-bjs-placebo-watch-") as work:
            config = dict(
                path=str(DATA),
                n_units=n,
                anticipation=A,
                panel_mode=mode,
                workdir=work,
                memory_gb=1.0,
                timeout=180,
            )
            path = OUT / f"{name}.config.json"
            save(path, config)
            result = monitor(
                [sys.executable, str(ROOT / "benchmarks/bjs_placebo_resource_fit.py"), str(path)],
                env,
                config,
                OUT / name,
            )
            result["cleanup_ok"] = not list(Path(work).iterdir())
            save(OUT / f"{name}.json", result)
            print(name, result["status"], round(result["peak_rss_mib"], 1), flush=True)
            if result["status"] != "ok" or not result["cleanup_ok"]:
                raise RuntimeError(f"{name}: resource check failed")
    save(
        OUT / "input.json",
        dict(
            seed=741,
            periods=8,
            weights="0.2+id%9",
            pretrends=3,
            missing_rows="id%4==0 and time in (3,7)",
            network_required=False,
        ),
    )


if __name__ == "__main__":
    main()
