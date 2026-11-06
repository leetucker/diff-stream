"""Subprocess resource measurement, adapted from dpd-stream (MIT).

Wall time includes reading the source. Peak RSS is the process high-water mark;
peak allocated scratch disk is sampled, so short-lived files can be missed.
"""

import json
import os
import resource
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


def peak_rss_mb():
    """Peak resident set size of this process in MB, from /proc (VmHWM, which
    unlike ru_maxrss is not inherited across fork+exec)."""
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def allocated_bytes(root):
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_blocks * 512
            except OSError:
                pass
    return total


class DiskPoller:
    """The maximum allocated size of a directory, sampled in a thread."""

    def __init__(self, root, interval=0.2):
        self.root = Path(root)
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, allocated_bytes(self.root))
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, allocated_bytes(self.root))


def run_child(runner, config, n, path, workdir):
    """In the child: run one configuration and print its report as JSON."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    baseline = peak_rss_mb()
    with DiskPoller(workdir) as disk:
        result = runner(config, int(n), Path(path), workdir)
    result["peak_memory_mb"] = peak_rss_mb()
    result["baseline_memory_mb"] = baseline
    result["peak_disk_mb"] = max(disk.peak / 1e6, result.pop("reported_disk_mb", 0.0))
    print("RESULT " + json.dumps(result, default=float))


# how R and Python say that an allocation was refused
OUT_OF_MEMORY = (
    "MemoryError",
    "Unable to allocate",
    "OutOfMemory",
    "cannot allocate",
    "out of memory",
    "RESOURCE_EXHAUSTED",
)


def measure(script, config, n, path, workdir, timeout=None, command=None, env=None):
    """Run one configuration in a subprocess; its report, with ``status``.
    The script's own child, or `command` (from `external_command`)."""
    workdir = Path(workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = command or [
        sys.executable,
        str(script),
        "--child",
        config,
        str(n),
        str(path),
        str(workdir),
    ]
    env = dict(os.environ, PYTHONWARNINGS="ignore", **(env or {}))
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": f"timeout after {timeout:.0f} s"}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    elapsed = time.perf_counter() - t0
    if proc.returncode in (-9, 137):
        return {"status": "killed (out of memory)"}
    if proc.returncode != 0:
        lines = (proc.stderr or proc.stdout).strip().splitlines()
        oom = next((ln.strip() for ln in lines if any(s in ln for s in OUT_OF_MEMORY)), None)
        if oom:
            return {"status": "out of memory: " + oom[:200]}
        return {"status": "failed: " + " | ".join(lines[-3:])[:400]}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            rep = json.loads(line[len("RESULT ") :])
            rep["status"] = "ok"
            rep["process_s"] = elapsed
            return rep
    return {"status": "failed: no result line"}
