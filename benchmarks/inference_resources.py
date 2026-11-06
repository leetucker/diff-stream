"""Offline bounded-memory CS multiplier checks at 10k/100k units and 999/1999 draws."""

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from resource_checks import ROOT, monitor, save

DATA = ROOT / "benchmarks/data/weights_resource_panel.parquet"
OUT = ROOT / "benchmarks/results/inference_resources"


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
    for n, reps in [(10000, 999), (100000, 999), (100000, 1999)]:
        name = f"cs_{n}_{reps}"
        with TemporaryDirectory(prefix="diff-multiplier-watch-") as work:
            config = dict(
                path=str(DATA), n_units=n, reps=reps, workdir=work, memory_gb=1.0, timeout=180
            )
            path = OUT / f"{name}.config.json"
            save(path, config)
            result = monitor(
                [sys.executable, str(ROOT / "benchmarks/inference_resource_fit.py"), str(path)],
                env,
                config,
                OUT / name,
            )
            result["cleanup_ok"] = not list(Path(work).iterdir())
            save(OUT / f"{name}.json", result)
            print(name, result["status"], round(result["peak_rss_mib"], 1), flush=True)
            if result["status"] != "ok" or not result["cleanup_ok"]:
                raise RuntimeError(f"{name} failed")
    save(
        OUT / "input.json",
        dict(
            seed=741,
            periods=8,
            anticipation=2,
            weights="0.2+id%9",
            missing_rows="id%4==0 and time in (3,7)",
            bootstrap_seed=431,
            bootstrap_weights="rademacher",
            network_required=False,
        ),
    )


if __name__ == "__main__":
    main()
