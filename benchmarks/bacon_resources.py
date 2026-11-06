"""Offline 10k/100k/1m-unit and many-cohort Bacon checks under a 1 GB watchdog."""

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from resource_checks import ROOT, monitor, save

OUT = ROOT / "benchmarks/results/bacon_resources"
DATA = ROOT / "benchmarks/data/bacon_resources"
CASES = [
    (10000, 8, False),
    (100000, 8, False),
    (1000000, 8, False),
    (10000, 64, False),
    (1024, 512, True),
]


def generate(n, T, path, sampling_weights=False):
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(913)
    with pq.ParquetWriter(
        path,
        pa.schema(
            [
                ("id", pa.int64()),
                ("time", pa.int64()),
                ("first_treat", pa.int64()),
                ("y", pa.float64()),
                *([("w", pa.float64())] if sampling_weights else []),
            ]
        ),
    ) as writer:
        for start in range(0, n, 1000):
            ids = np.arange(start, min(n, start + 1000))
            g = ids % T
            g = np.where(g == 0, 0, g + 1)
            times = np.arange(1, T + 1)
            D = (g[:, None] > 0) & (times >= g[:, None])
            y = (
                0.4 * rng.normal(size=(len(ids), 1))
                + 0.1 * times
                + (0.5 + 0.15 * g[:, None] + 0.25 * (times - g[:, None])) * D
                + rng.normal(size=(len(ids), T))
            )
            writer.write_table(
                pa.table(
                    dict(
                        id=np.repeat(ids, T),
                        time=np.tile(times, len(ids)),
                        first_treat=np.repeat(g, T),
                        y=y.ravel(),
                        **(dict(w=np.repeat(0.2 + ids % 9, T)) if sampling_weights else {}),
                    )
                ),
                row_group_size=8192,
            )


def main(sampling_weights=False):
    out = OUT.with_name("bacon_weights_resources") if sampling_weights else OUT
    data_root = DATA.with_name("bacon_weights_resources") if sampling_weights else DATA
    out.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
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
    for n, T, guard in CASES:
        name = f"n{n}_t{T}"
        data = data_root / f"{name}.parquet"
        if not data.exists():
            subprocess.run(
                [sys.executable, __file__, "--generate", str(n), str(T), str(data)]
                + (["--weights"] if sampling_weights else []),
                env=env,
                check=True,
            )
        with TemporaryDirectory(prefix="diff-bacon-watch-") as work:
            config = dict(
                path=str(data),
                n_units=n,
                periods=T,
                expect_guard=guard,
                workdir=work,
                memory_gb=1.0,
                timeout=180,
                weights="w" if sampling_weights else None,
            )
            path = out / f"{name}.config.json"
            save(path, config)
            result = monitor(
                [sys.executable, str(ROOT / "benchmarks/bacon_resource_fit.py"), str(path)],
                env,
                config,
                out / name,
            )
            result["cleanup_ok"] = not list(Path(work).iterdir())
            save(out / f"{name}.json", result)
            print(name, result["status"], round(result["peak_rss_mib"], 1), flush=True)
            expected = "guard_refused" if guard else "ok"
            if result["status"] != expected or not result["cleanup_ok"]:
                raise RuntimeError(f"{name}: resource check failed")
    save(
        out / "input.json",
        dict(
            seed=913,
            generation="1000-unit NumPy blocks; deterministic cohort cycling",
            network_required=False,
            sampling_weights="w=0.2+id%9" if sampling_weights else None,
        ),
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        generate(
            int(sys.argv[2]),
            int(sys.argv[3]),
            sys.argv[4],
            sampling_weights="--weights" in sys.argv,
        )
    else:
        main()
