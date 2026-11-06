"""Small/medium subprocess benchmarks; opt in to larger sizes explicitly."""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

from _harness import measure, run_child

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from diff_stream import diff_stream  # noqa: E402


def runner(config, n, path, workdir):
    model, threads = config.split(":")
    start = time.perf_counter()
    r = diff_stream(
        "y ~ 1",
        path,
        panel=("id", "time"),
        cohort="first_treat",
        estimator=model,
        threads=int(threads),
        workdir=workdir,
    )
    return {
        "wall_s": time.perf_counter() - start,
        "checksum": float(r.beta.sum()),
        "n_obs": r.n_obs,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000])
    p.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    p.add_argument(
        "--models", nargs="+", choices=["cs", "sunab", "bjs"], default=["cs", "sunab", "bjs"]
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--rscript", default=os.environ.get("DIFF_RSCRIPT") or shutil.which("Rscript"))
    p.add_argument("--output", type=Path, default=ROOT / "results" / "local_scale.csv")
    args = p.parse_args()
    if args.repeats < 1 or any(t < 1 for t in args.threads) or any(n < 10 for n in args.sizes):
        p.error("use positive repeats/threads and sizes of at least 10 units")
    from diff_stream.simulate import simulate_panel

    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "n",
        "repeat",
        "configuration",
        "status",
        "wall_s",
        "peak_memory_mb",
        "baseline_memory_mb",
        "peak_disk_mb",
        "process_s",
        "checksum",
        "n_obs",
    ]
    deferred = set()
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for n in sorted(args.sizes):
            df = simulate_panel(n, 8)
            path = data_dir / f"sim_{n}.parquet"
            df.write_parquet(path)
            csvpath = path.with_suffix(".csv")
            df.write_csv(csvpath)
            for model in args.models:
                for repeat in range(args.repeats):
                    configs = [
                        (f"diff-stream ({t} thread{'s' if t != 1 else ''})", t)
                        for t in args.threads
                    ]
                    if args.rscript:
                        configs.append(("R reference (4 threads)", 4))
                    for label, threads in configs:
                        work = ROOT / "work" / f"{model}_{n}_{threads}"
                        command = None
                        if label.startswith("R"):
                            spec = dict(model=model, data=str(csvpath), threads=threads)
                            command = [
                                args.rscript,
                                str(ROOT / "external" / "benchmark.R"),
                                json.dumps(spec),
                            ]
                        env = {
                            "STREAM_LIMITS_MAX_THREADS": str(threads),
                            "POLARS_MAX_THREADS": str(threads),
                            "NUMBA_NUM_THREADS": str(threads),
                            "OPENBLAS_NUM_THREADS": str(threads),
                            "OMP_NUM_THREADS": str(threads),
                        }
                        r = (
                            {"status": "deferred after earlier timeout/out-of-memory"}
                            if (model, label) in deferred
                            else measure(
                                __file__,
                                f"{model}:{threads}",
                                n,
                                path,
                                work,
                                command=command,
                                timeout=args.timeout,
                                env=env,
                            )
                        )
                        if r["status"].startswith(("timeout", "out of memory", "killed")):
                            deferred.add((model, label))
                        row = dict(model=model, n=n, repeat=repeat, configuration=label, **r)
                        writer.writerow(row)
                        handle.flush()
                        print(row, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        run_child(runner, *sys.argv[2:])
    else:
        main()
