"""Offline source-layout/unit-scaling checks in optional delegated v1/v2 cgroups.

RSS watchdogs remain enabled. --cgroup-parent must name an existing delegated
directory with the memory controller enabled; no system-wide settings change.
"""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stream_limits.enforcement import MemoryCgroup, cgroup_capability
from stream_limits.testing import verify_memory_limit

from benchmarks import resource_checks as h

OUT = h.ROOT / "benchmarks/results/memory_checks"


def jobs():
    out = []

    def add(name, n=100000, width=3, estimator="cs", pattern="balanced", reference=None, **layout):
        spec = h.dataset(n, width=width, pattern=pattern)
        spec.update(layout)
        c = h.case(name, spec)
        c.update(memory_gb=1.0, timeout=300, estimator=estimator, reference=reference)
        out.append(c)

    for n in (10000, 100000, 1000000, 2000000):
        add(f"cs_n{n}", n=n)
        if n == 2000000:
            out[-1]["memory_gb"] = 1.5
    for label, layout in [
        ("1gb", {}),
        ("large_groups", dict(row_group=524288)),
        ("tight", {}),
    ]:
        add(f"cs_n2000000_{label}", n=2000000, reference="cs_n2000000", **layout)
        # A boundary probe may fit or refuse before exceeding its headroom.
        # A refusal is reported separately and never counts as a completed fit.
        out[-1]["allow_guard_refusal"] = True
        if label == "tight":
            out[-1].update(preprocess_mb=4, batch_units=128)
    for label, layout in [
        ("tiny_groups", dict(row_group=257)),
        ("large_groups", dict(row_group=524288)),
        ("strings", dict(strings=True, id_padding=96)),
        (
            "strings_shuffled_large",
            dict(strings=True, id_padding=96, shuffled=True, row_group=524288),
        ),
        ("unused_wide", dict(unused_columns=64, row_group=131072)),
    ]:
        add(f"cs_{label}", reference="cs_n100000", **layout)
    add("cs_wide", width=33, pattern="entry_exit")
    add(
        "cs_wide_strings_shuffled_large",
        width=33,
        pattern="entry_exit",
        strings=True,
        id_padding=96,
        shuffled=True,
        row_group=262144,
        reference="cs_wide",
    )
    add("cs_wide_tight", width=33, pattern="entry_exit", reference="cs_wide")
    out[-1].update(preprocess_mb=4, batch_units=128)
    for estimator in ("sunab", "bjs"):
        for n in (100000, 1000000):
            add(f"{estimator}_n{n}", n=n, width=1, estimator=estimator, pattern="entry_exit")
            add(
                f"{estimator}_strings_n{n}",
                n=n,
                width=1,
                estimator=estimator,
                pattern="entry_exit",
                strings=True,
                id_padding=96,
                shuffled=True,
                row_group=524288,
                reference=f"{estimator}_n{n}",
            )
    return out


def cgroup_selfcheck(parent, out, env):
    """Compatibility artifact adapter around stream-limits' shared verifier."""
    report = verify_memory_limit(parent, env=env)
    # Preserve the earlier report reader's control-fit field; enforcement and
    # measurement are owned entirely by stream-limits.
    report["control"]["fit"] = dict(status="ok", detected_memory_gb=report["detected_memory_gb"])
    h.save(out / "cgroup_selfcheck.json", report)
    return report


def run(out, only=None, parent=None):
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".driver.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    selected = [c for c in jobs() if not only or c["name"] in only]
    if only and set(only) - {c["name"] for c in selected}:
        raise ValueError("unknown case name")
    library = hashlib.sha256()
    for path in sorted((h.ROOT / "diff_stream").rglob("*.py")):
        library.update(str(path.relative_to(h.ROOT)).encode())
        library.update(path.read_bytes())
    env = dict(os.environ, PYTHONPATH=str(h.ROOT), MPLCONFIGDIR="/tmp/diff-stream-mpl")
    for key in (
        "STREAM_LIMITS_MAX_THREADS",
        "POLARS_MAX_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        env[key] = "2"
    h.save(out / "environment.json", dict(h.environment(), cgroup=cgroup_capability(parent)))
    h.watchdog_selfcheck(out)
    if parent:
        info = cgroup_capability(parent)
        if not (info["v1"] or info["v2"]) or not info["writable"]:
            raise ValueError(
                "--cgroup-parent must be an existing writable delegated memory cgroup v1/v2"
            )
        cgroup_selfcheck(parent, out, env)
    data_dir = h.ROOT / "benchmarks/data/memory_checks"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in selected:
        c.update(
            library_sha256=library.hexdigest(),
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            fit_driver_sha256=hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest(),
            cgroup_parent=str(parent) if parent else None,
        )
        destination = out / (c["name"] + ".json")
        if destination.exists():
            previous = json.loads(destination.read_text())
            if all(previous["config"].get(k) == v for k, v in c.items()) and h.accepted(previous):
                print(f"REUSE {c['name']}", flush=True)
                rows.append(previous)
                continue
            archive = out / "history" / (c["name"] + "-" + str(time.time_ns()))
            archive.mkdir(parents=True)
            for old in out.glob(c["name"] + ".*"):
                if old.is_file():
                    archive.joinpath(old.name).write_bytes(old.read_bytes())
        path = data_dir / (h.identity(c["data"]) + ".parquet")
        if not path.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(Path(h.__file__)),
                    "--generate",
                    json.dumps(dict(spec=c["data"], path=str(path), csv_output=False)),
                ],
                env=env,
                check=True,
            )
        manifest = json.loads(path.with_suffix(".json").read_text())
        # Regeneration is excluded from fit measurements, but record actual
        # physical row groups and input checksum, not just requested layout.
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
            command = [sys.executable, str(Path(h.__file__)), "--fit", str(payload)]
            group = MemoryCgroup(parent, c["memory_gb"] * 1e9) if parent else None
            try:
                print(f"RUN {c['name']}", flush=True)
                result = h.monitor(
                    group.command(command) if group else command,
                    dict(env, STREAM_LIMITS_MEMORY_GB=str(c["memory_gb"])),
                    config,
                    out / c["name"],
                )
                if result["returncode"] != 0 and result["status"] == "ok":
                    result["status"] = "process_failed"
                if group:
                    result["cgroup"] = group.metrics()
                    if result["cgroup"]["events"].get("oom_kill", 0):
                        result["status"] = "cgroup_oom"
            finally:
                if group:
                    group.close()
            result["cleanup_ok"] = not list(Path(work).iterdir())
            result["input"] = manifest
            result["safe_guard_refusal"] = (
                c.get("allow_guard_refusal", False)
                and result["status"] == "guard_refused"
                and result["cleanup_ok"]
                and "headroom" in result["fit"].get("error", "")
            )
            if result["status"] == "ok":
                decoded = result["fit"]["diagnostics"]["max_decoded_leaf_bytes"]
                result["leaf_budget_ok"] = decoded <= c["preprocess_mb"] * 1e6 / 4
            h.save(destination, result)
        rows.append(result)
        print(
            f"DONE {c['name']}: {result['status']} {result['peak_rss_mib']:.1f} MiB {result['process_s']:.1f}s",
            flush=True,
        )
    # Resolve comparisons even when a baseline was retained by an earlier run.
    for row in rows:
        name = row["config"].get("reference")
        if name and not row.get("safe_guard_refusal"):
            baseline = out / (name + ".json")
            if baseline.exists():
                row["invariance"] = h.compare(row, json.loads(baseline.read_text()))
            row["invariance_required"] = True
            h.save(out / (row["config"]["name"] + ".json"), row)
    failed = [
        row["config"]["name"]
        for row in rows
        if not row.get("safe_guard_refusal")
        and (
            not h.accepted(row)
            or row.get("leaf_budget_ok") is not True
            or (row.get("invariance_required") and not row.get("invariance"))
        )
    ]
    h.save(
        out / "summary.json",
        dict(
            cases=len(rows),
            completed_fits=sum(r["status"] == "ok" for r in rows),
            safe_guard_refusals=sum(bool(r.get("safe_guard_refusal")) for r in rows),
            unexpected_failures=failed,
            enforcement="kernel memory cgroup plus RSS watchdog"
            if parent
            else "sampled RSS watchdog plus process high-water RSS",
            rows=[
                dict(
                    name=r["config"]["name"],
                    status=r["status"],
                    peak_rss_mib=r["peak_rss_mib"],
                    process_s=r["process_s"],
                    peak_disk_mb=r["peak_disk_mb"],
                    n_rows=r["input"]["n_rows"],
                    max_row_group_rows=r["input"]["max_row_group_rows"],
                    invariance=r.get("invariance"),
                    cleanup_ok=r["cleanup_ok"],
                    leaf_budget_ok=r.get("leaf_budget_ok"),
                )
                for r in rows
            ],
        ),
    )
    return bool(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=OUT)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--cgroup-parent", type=Path)
    parser.add_argument("--selfcheck-only", action="store_true")
    args = parser.parse_args()
    if args.selfcheck_only:
        if args.cgroup_parent is None:
            parser.error("--selfcheck-only requires --cgroup-parent")
        args.results.mkdir(parents=True, exist_ok=True)
        cgroup_selfcheck(args.cgroup_parent, args.results, dict(os.environ))
        raise SystemExit(0)
    raise SystemExit(run(args.results.resolve(), args.only, args.cgroup_parent))
