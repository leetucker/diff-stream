"""Local conditional-CS comparisons; shared-pass nuisance fits, numeric design."""

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
from benchmarks._estimators import public  # noqa: E402
from diff_stream import did_callaway_santanna  # noqa: E402
from diff_stream.simulate import simulate_conditional_panel  # noqa: E402


def runner(config, n, path, workdir):
    method, threads = config.split(":")
    start = time.perf_counter()
    r = did_callaway_santanna(
        "y ~ x + z",
        path,
        panel=("id", "time"),
        cohort="first_treat",
        adjustment=public("adjustment", method),
        n_threads=int(threads),
        workdir=workdir,
    )
    return dict(
        wall_s=time.perf_counter() - start,
        checksum=float(r.beta.sum()),
        n_obs=r.n_obs,
        nuisance_source_passes=r.diagnostics["nuisance_source_passes"],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000])
    p.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--rscript", default=os.environ.get("DIFF_RSCRIPT") or shutil.which("Rscript"))
    a = p.parse_args()
    out = ROOT / "results/conditional.csv"
    out.parent.mkdir(exist_ok=True)
    (ROOT / "data").mkdir(exist_ok=True)
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
        "nuisance_source_passes",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for n in a.sizes:
            d = simulate_conditional_panel(n)
            path = ROOT / "data" / f"conditional_{n}.parquet"
            d.write_parquet(path)
            d.write_csv(path.with_suffix(".csv"))
            for method in ["reg", "ipw", "dr"]:
                for repeat in range(a.repeats):
                    configs = [
                        (f"diff-stream ({t} thread" + ("s" if t != 1 else "") + ")", t)
                        for t in a.threads
                    ]
                    if a.rscript:
                        configs.append(("R reference (4 threads)", 4))
                    for label, threads in configs:
                        command = None
                        if label.startswith("R"):
                            command = [
                                a.rscript,
                                str(ROOT / "external/conditional_benchmark.R"),
                                json.dumps(dict(data=str(path.with_suffix(".csv")), method=method)),
                            ]
                        env = {
                            name: str(threads)
                            for name in [
                                "STREAM_LIMITS_MAX_THREADS",
                                "POLARS_MAX_THREADS",
                                "NUMBA_NUM_THREADS",
                                "OPENBLAS_NUM_THREADS",
                                "OMP_NUM_THREADS",
                            ]
                        }
                        r = measure(
                            __file__,
                            f"{method}:{threads}",
                            n,
                            path,
                            ROOT / "work" / f"cs_{method}_{n}_{threads}",
                            command=command,
                            timeout=60,
                            env=env,
                        )
                        row = dict(model=method, n=n, repeat=repeat, configuration=label, **r)
                        writer.writerow(row)
                        f.flush()
                        print(row, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        run_child(runner, *sys.argv[2:])
    else:
        main()
