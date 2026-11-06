"""Run the phase probe with a parent-enforced sampled RSS ceiling.

This watchdog terminates its own child if RSS exceeds the chosen limit. It does
not configure an OS cgroup; sampling can miss peaks shorter than the interval.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("--memory-gb", type=float, default=1.0)
    p.add_argument("--preprocess-mb", type=float, default=16)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--timeout", type=float, default=60)
    a = p.parse_args()
    env = dict(
        os.environ,
        STREAM_LIMITS_MEMORY_GB=str(a.memory_gb),
        STREAM_LIMITS_MAX_THREADS=str(a.threads),
        POLARS_MAX_THREADS=str(a.threads),
        NUMBA_NUM_THREADS=str(a.threads),
        OPENBLAS_NUM_THREADS=str(a.threads),
        OMP_NUM_THREADS=str(a.threads),
    )
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("memory_probe.py")),
        a.data,
        "--threads",
        str(a.threads),
        "--memory-gb",
        str(a.memory_gb),
        "--preprocess-mb",
        str(a.preprocess_mb),
    ]
    start, peak, status = time.monotonic(), 0, "ok"
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        child = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                peak = max(peak, child.memory_info().rss)
            except psutil.NoSuchProcess:
                break
            if peak > a.memory_gb * 1e9:
                status = "RSS ceiling exceeded"
                proc.kill()
                break
            if time.monotonic() - start > a.timeout:
                status = "timeout"
                proc.kill()
                break
            time.sleep(0.005)
        stdout, stderr = proc.communicate()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    result = dict(
        status=status if proc.returncode == 0 or status != "ok" else "fit failed",
        rss_limit_gb=a.memory_gb,
        sampled_peak_rss_gb=peak / 1e9,
        returncode=proc.returncode,
        stderr=stderr,
    )
    if proc.returncode == 0:
        result["fit"] = json.loads(stdout)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
