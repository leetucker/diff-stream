"""Local one-factor model-complexity sweeps with bounded fresh subprocesses.

Reuses the resource-check generator, fit instrumentation and RSS watchdog.
Results are resumable and keyed by the library/configuration hashes. No cloud,
network access, or modifications to sibling packages are required.
"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import resource_checks as h

OUT = h.ROOT / "benchmarks/results/complexity_checks"


def jobs():
    out = []

    def add(name, n=10000, T=8, G=3, K=3, pattern="balanced", **kwargs):
        c = h.case(name, h.dataset(n, periods=T, cohorts=G, width=K, pattern=pattern))
        c.update(timeout=240, memory_gb=2.0, **kwargs)
        out.append(c)

    add("baseline")
    for T in (16, 32, 64):
        add(f"periods_{T}", T=T)
    for G in (1, 7, 15):
        add(f"cohorts_{G}", T=16, G=G)
    add("cohorts_3", T=16)
    for K in (17, 33, 65, 129, 193, 256):
        add(f"columns_{K}", n=5000, K=K)
    add("columns_3", n=5000)
    add("columns_257_guard", n=2000, K=257, expected="fit_failed")
    for pattern in ("balanced", "entry_exit"):
        for K in (3, 33):
            add(f"cross_t24_g8_k{K}_{pattern}", n=4000, T=24, G=8, K=K, pattern=pattern)
    add("cross_t32_g15_k33", n=3000, T=32, G=15, K=33)
    add("cross_t24_g8_k33_entry_exit_larger", n=16000, T=24, G=8, K=33, pattern="entry_exit")
    add("cross_t64_g31_k3", n=6000, T=64, G=31)
    add("cross_t64_g31_k33", n=6000, T=64, G=31, K=33)
    for estimator in ("sunab", "bjs"):
        for T, G in ((16, 3), (32, 15)):
            add(f"{estimator}_t{T}_g{G}", n=10000, T=T, G=G, K=1, estimator=estimator)
    add("baseline_as3g", as_limit_bytes=3 * 2**30)
    add("cross_as3g", n=4000, T=24, G=8, K=33, as_limit_bytes=3 * 2**30)
    add("cross_as6g", n=4000, T=24, G=8, K=33, as_limit_bytes=6 * 2**30)
    for c in out:
        if c["data"]["periods"] >= 64 and c["data"]["cohorts"] >= 15:
            c["summary_only"] = True
    return out


def run(out, only=None):
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".driver.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    data_dir = h.ROOT / "benchmarks/data/complexity_checks"
    data_dir.mkdir(parents=True, exist_ok=True)
    library = hashlib.sha256()
    for p in sorted((h.ROOT / "diff_stream").rglob("*.py")):
        library.update(p.read_bytes())
    h.save(out / "environment.json", h.environment())
    rows = []
    for c in jobs():
        if only and c["name"] not in only:
            continue
        c["library_sha256"] = library.hexdigest()
        c["driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        c["fit_driver_sha256"] = hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest()
        destination = out / (c["name"] + ".json")
        if destination.exists():
            previous = json.loads(destination.read_text())
            if all(previous["config"].get(k) == v for k, v in c.items()):
                print(f"REUSE {c['name']}: {previous['status']}", flush=True)
                rows.append(previous)
                continue
            history = out / "history" / (c["name"] + "-" + str(time.time_ns()))
            history.mkdir(parents=True)
            for old in out.glob(c["name"] + ".*"):
                if old.is_file():
                    shutil.copyfile(old, history / old.name)
        env = dict(os.environ)
        env.update(
            {
                key: "2"
                for key in (
                    "STREAM_LIMITS_MAX_THREADS",
                    "POLARS_MAX_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "NUMBA_NUM_THREADS",
                )
            }
        )
        env.update(
            PYTHONPATH=str(h.ROOT),
            MPLCONFIGDIR="/tmp/diff-stream-mpl",
            STREAM_LIMITS_MEMORY_GB=str(c["memory_gb"]),
        )
        path = data_dir / (h.identity(c["data"]) + ".parquet")
        if not path.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(h.ROOT / "benchmarks/resource_checks.py"),
                    "--generate",
                    json.dumps(dict(spec=c["data"], path=str(path), csv_output=False)),
                ],
                env=env,
                check=True,
            )
        with TemporaryDirectory(prefix=c["name"] + "-", dir=out) as work:
            config = dict(
                c,
                path=str(path),
                workdir=work,
                formula=h.formula(c["data"]) if c["estimator"] == "cs" else "y ~ 1",
                panel_mode="balanced" if c["data"]["pattern"] == "balanced" else "unbalanced",
                progress_path=str(out / (c["name"] + ".progress.json")),
            )
            payload = out / (c["name"] + ".config.json")
            h.save(payload, config)
            command = [
                sys.executable,
                str(h.ROOT / "benchmarks/resource_checks.py"),
                "--fit",
                str(payload),
            ]
            if c.get("as_limit_bytes"):
                command = ["prlimit", f"--as={c['as_limit_bytes']}", "--", *command]
            print(f"RUN {c['name']}: {c['data']}", flush=True)
            result = h.monitor(command, env, config, out / c["name"])
            result["cleanup_ok"] = not list(Path(work).iterdir())
            if c["name"] == "columns_257_guard":
                result["guard_message_ok"] = "256" in result["fit"].get("error", "")
            h.save(destination, result)
        rows.append(result)
        print(
            f"DONE {c['name']}: {result['status']} {result['peak_rss_mib']:.1f} MiB {result['process_s']:.1f}s",
            flush=True,
        )
        h.save(
            out / "progress.json",
            dict(
                completed=len(rows),
                cases=[dict(name=r["config"]["name"], status=r["status"]) for r in rows],
            ),
        )
    failed = [
        r["config"]["name"] for r in rows if not h.accepted(r) or r.get("guard_message_ok") is False
    ]
    h.save(out / "run.json", dict(cases=len(rows), unexpected_failures=failed))
    return bool(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=OUT)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    raise SystemExit(run(args.results.resolve(), args.only))
