"""Resumable, network-free conditional-CS and observed-panel resource checks.

Run --suite smoke first, then --suite scale. Each case has an atomic JSON result,
stdout/stderr logs, a process-tree RSS watchdog and a timeout. Existing completed
cases are reused unless --retry-failed is given. Data are generated in bounded
unit blocks. This is a local watchdog, not an OS/cgroup hard memory limit.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import psutil

try:
    import fcntl
except ImportError:  # Windows: no lock between drivers
    fcntl = None

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results/resource_checks"
DATA = ROOT / "benchmarks/data/resource_checks"
RSCRIPT = Path(os.environ.get("DIFF_RSCRIPT") or shutil.which("Rscript") or "Rscript")
RLIB = ROOT / "benchmarks/external/.r/library"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def identity(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def accepted(row):
    if row["status"] != row["config"].get("expected", "ok"):
        return False
    if not row.get("cleanup_ok", True):
        return False
    if row.get("reference_requested") and row.get("r_status") != "ok":
        return False
    return all(row.get(key) is None or row[key]["passed"] for key in ("r_comparison", "invariance"))


def dataset(
    n=10000,
    periods=5,
    cohorts=3,
    width=3,
    pattern="balanced",
    strings=False,
    row_group=8192,
    invariant=False,
):
    return dict(
        version=1,
        n=n,
        periods=periods,
        cohorts=cohorts,
        width=width,
        pattern=pattern,
        strings=strings,
        row_group=row_group,
        invariant=invariant,
        seed=314159,
    )


def formula(spec, r=False):
    if spec["width"] == 1:
        return "~1" if r else "y ~ 1"
    terms = ["x", "z"]
    if spec["width"] > 3:
        terms += [f"v{j}" for j in range(spec["width"] - 9)]
        cat = "factor(category)" if r else "C(category)"
        terms += [cat, f"x:{cat}"]
    return ("~" if r else "y ~ ") + "+".join(terms)


def generate(spec, path, csv_output=False):
    # The fit process never inherits generation arrays or allocator pages.
    sys.path.insert(0, str(ROOT))
    import stream_limits

    stream_limits.limit_threads()
    import numpy as np
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq

    if spec["width"] not in (1, 3) and spec["width"] < 9:
        raise ValueError("expanded designs need width >= 9")
    T = spec["periods"]
    choices = np.r_[0, np.linspace(2, T, spec["cohorts"], dtype=int)]
    if len(set(choices)) != len(choices):
        raise ValueError("too many cohorts for period grid")
    temporary = path.with_suffix(".parquet.tmp")
    writer = None
    csv_file = path.with_suffix(".csv.tmp").open("w") if csv_output else None
    rows = 0
    pending, pending_rows = [], 0
    try:
        for offset in range(0, spec["n"], 4096):
            size = min(4096, spec["n"] - offset)
            rng = np.random.default_rng(spec["seed"] + offset)
            ids = np.arange(offset, offset + size)
            x0 = rng.normal(size=size)
            z = rng.integers(0, 2, size=size)
            category = np.array(["a", "b", "c", "d"])[ids % 4]
            slope = np.linspace(-0.25, 0.25, len(choices))
            eta = x0[:, None] * slope + 0.12 * z[:, None] * np.cos(np.arange(len(choices)))
            prob = np.exp(eta - eta.max(axis=1, keepdims=True))
            prob /= prob.sum(axis=1, keepdims=True)
            g = choices[(rng.random(size)[:, None] > prob.cumsum(axis=1)).sum(axis=1)]
            g[: min(size, len(choices))] = choices[: min(size, len(choices))]
            t = np.arange(1, T + 1)
            x = np.broadcast_to(x0[:, None], (size, T)).copy()
            if not spec["invariant"]:
                x += 0.08 * t + 0.12 * rng.normal(size=x.shape)
            error = rng.normal(size=(size, T))
            for j in range(1, T):
                error[:, j] += 0.35 * error[:, j - 1]
            tau = np.where(
                (g[:, None] > 0) & (t >= g[:, None]),
                0.6 + 0.08 * g[:, None] + 0.15 * (t - g[:, None]),
                0,
            )
            y = (
                rng.normal(size=(size, 1))
                + 0.18 * t
                + 0.2 * t * x0[:, None]
                + 0.12 * t * z[:, None]
                + 0.3 * x
                + tau
                + error
            )
            fields = dict(
                id=np.repeat(ids, T),
                time=np.tile(t, size),
                first_treat=np.repeat(g, T),
                y=y.ravel(),
                x=x.ravel(),
                z=np.repeat(z, T),
                category=np.repeat(category, T),
            )
            for j in range(max(0, spec["width"] - 9)):
                v = np.broadcast_to(rng.normal(size=(size, 1)), (size, T)).copy()
                if not spec["invariant"]:
                    v += 0.03 * t + 0.1 * rng.normal(size=v.shape)
                fields[f"v{j}"] = v.ravel()
            keep = np.ones((size, T), bool)
            if spec["pattern"] == "gaps":
                keep = (rng.random((size, T)) > 0.25) | (t == 1)
            elif spec["pattern"] == "entry_exit":
                last_entry = np.where(g > 0, g - 1, T - 1)
                entry = 1 + (rng.random(size) * last_entry).astype(int)
                earliest_exit = np.maximum(entry, np.where(g > 0, g, entry))
                exit = earliest_exit + (rng.random(size) * (T - earliest_exit + 1)).astype(int)
                keep = (t >= entry[:, None]) & (t <= exit[:, None]) & (rng.random((size, T)) > 0.15)
                keep[np.arange(size), entry - 1] = True  # Every unit retains an untreated row.
            elif spec["pattern"] != "balanced":
                raise ValueError("unknown missing-row pattern")
            frame = pl.DataFrame(fields).filter(pl.Series(keep.ravel()))
            if spec["strings"]:
                label = pl.col("id").cast(pl.String)
                if spec.get("id_padding"):
                    label = label.str.pad_start(spec["id_padding"], "0")
                frame = frame.with_columns(("unit_" + label).alias("id"))
            if spec.get("unused_columns"):
                # Independent RNG keeps the estimation sample identical when
                # adding columns that the model should never decode.
                extra = np.random.default_rng(spec["seed"] + offset + 999999)
                frame = frame.with_columns(
                    [
                        pl.Series(f"unused{j}", extra.normal(size=len(frame)))
                        for j in range(spec["unused_columns"])
                    ]
                )
            if spec.get("shuffled"):
                frame = frame.sample(fraction=1, shuffle=True, seed=spec["seed"] + offset)
            table = frame.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            if spec["row_group"] > 32768:
                # ParquetWriter does not join row groups across write_table
                # calls. Buffer one requested group to exercise truly large
                # input row groups while keeping generation bounded.
                pending.append(table)
                pending_rows += table.num_rows
                if pending_rows >= spec["row_group"]:
                    combined = pa.concat_tables(pending)
                    writer.write_table(
                        combined.slice(0, spec["row_group"]), row_group_size=spec["row_group"]
                    )
                    rest = combined.slice(spec["row_group"])
                    pending = [rest] if rest.num_rows else []
                    pending_rows = rest.num_rows
            else:
                writer.write_table(table, row_group_size=spec["row_group"])
            if csv_file is not None:
                frame.write_csv(csv_file, include_header=offset == 0)
            rows += len(frame)
    finally:
        if writer is not None:
            if pending_rows:
                writer.write_table(pa.concat_tables(pending), row_group_size=spec["row_group"])
            writer.close()
        if csv_file is not None:
            csv_file.close()
    temporary.replace(path)
    if csv_output:
        path.with_suffix(".csv.tmp").replace(path.with_suffix(".csv"))
    with pq.ParquetFile(path) as pf:
        groups = pf.metadata.num_row_groups
        max_group = max(pf.metadata.row_group(j).num_rows for j in range(groups))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    save(
        path.with_suffix(".json"),
        dict(
            spec=spec,
            n_rows=rows,
            parquet_bytes=path.stat().st_size,
            row_groups=groups,
            max_row_group_rows=max_group,
            sha256=digest.hexdigest(),
        ),
    )


def highwater_mib():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    return psutil.Process().memory_info().rss / 2**20


def fit_child(config):
    sys.path.insert(0, str(ROOT))
    import numpy as np

    import diff_stream.api as api
    import diff_stream.cs as cs
    import diff_stream.cs_unbalanced as rc
    import diff_stream.linear as linear
    import diff_stream.unbalanced as ub

    report = dict(import_baseline_mib=psutil.Process().memory_info().rss / 2**20)
    if config.get("preload_formula"):
        from hdfe_stream.core.formula import pyfixest_formula_api

        pyfixest_formula_api()
        report["preloaded_formula_mib"] = psutil.Process().memory_info().rss / 2**20
    phase = ["estimation"]
    peaks = {}
    stop = threading.Event()
    process = psutil.Process()
    progress = dict(numerical_passes=0, batches=0)
    started = time.monotonic()
    progress_file = Path(config.get("progress_path", RESULTS / (config["name"] + ".progress.json")))

    def sample():
        last_saved = 0
        while not stop.is_set():
            name = phase[0]
            peaks[name] = max(peaks.get(name, 0), process.memory_info().rss / 2**20)
            now = time.monotonic()
            if progress_file is not None and now - last_saved > 5:
                save(
                    progress_file,
                    dict(
                        progress,
                        phase=name,
                        elapsed_s=now - started,
                        cpu_s=sum(process.cpu_times()[:2]),
                        phase_peaks_mib=dict(peaks),
                    ),
                )
                last_saved = now
            stop.wait(0.005)

    def wrap(original, label):
        def call(*args, **kwargs):
            previous = phase[0]
            phase[0] = label
            try:
                return original(*args, **kwargs)
            finally:
                phase[0] = previous

        return call

    replacements = []
    original_batches = api.PanelStore.batches

    def batches(self, **kwargs):
        progress["numerical_passes"] += 1
        for batch in original_batches(self, **kwargs):
            progress["batches"] += 1
            yield batch

    api.PanelStore.batches = batches
    replacements.append((api.PanelStore, "batches", original_batches))
    for owner, name, label in [
        (api.PanelStore, "__init__", "preparation"),
        (api.PanelStore, "_prepare", "partition"),
        (api.PanelStore, "_leaf", "leaf_validation"),
        (api.pl.LazyFrame, "sink_parquet", "snapshot"),
        (cs, "fit_systems", "nuisance"),
        (cs, "fit_contrasts", "contrast_moments"),
        (rc, "fit_systems", "nuisance"),
        (rc, "fit_linear_systems", "outcome_nuisance"),
        (ub, "fit_sunab", "sunab_systems"),
        (ub, "fit_bjs", "bjs_fit"),
        (linear, "fit_sunab", "richer_sunab_systems"),
        (linear, "fit_bjs", "richer_bjs_fit"),
        (linear, "fit_placebos", "richer_placebos"),
    ]:
        original = getattr(owner, name)
        replacements.append((owner, name, original))
        setattr(owner, name, wrap(original, label))
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    start = time.perf_counter()
    result = None
    try:
        result = api._fit(
            config["formula"],
            config["path"],
            panel=("id", "time"),
            cohort="first_treat",
            estimator=config["estimator"],
            est_method=config["method"],
            panel_mode=config["panel_mode"],
            batch_units=config["batch_units"],
            preprocess_mb=config["preprocess_mb"],
            memory_gb=config["memory_gb"],
            threads=config["threads"],
            workdir=config["workdir"],
            avgeffectsby=config.get("avgeffectsby", "cohort_time"),
            leaveout=config.get("leaveout", False),
            weights=config.get("weights"),
            allow_time_varying_weights=config.get("allow_time_varying_weights", False),
            pretrends=config.get("pretrends", 0),
        )
        report["fit_s"] = time.perf_counter() - start
        phase[0] = "aggregation"
        before = time.perf_counter()
        a = result.aggregate("simple")
        if config.get("extra_aggregations"):
            result.aggregate("calendar")
            result.aggregate(balance_e=1, horizons=[0, 1])
        report.update(
            status="ok",
            aggregation_s=time.perf_counter() - before,
            n_obs=result.n_obs,
            n_units=result.n_units,
            cells=[list(c) for c in result.cells],
            beta=result.beta.tolist(),
            vcov=None if config.get("summary_only") else result.vcov.tolist(),
            vcov_shape=list(result.vcov.shape),
            vcov_trace=float(np.trace(result.vcov)),
            vcov_sha256=hashlib.sha256(np.ascontiguousarray(result.vcov).tobytes()).hexdigest(),
            att=float(a.beta[0]),
            se=float(a.se[0]),
            diagnostics=result.diagnostics,
        )
        if config.get("pretrends"):
            placebo = result.placebos()
            if not np.isfinite(placebo.beta).all() or not np.isfinite(placebo.vcov).all():
                raise ValueError("non-finite clean placebo estimates/covariance")
            report["placebo"] = dict(beta=placebo.beta.tolist(), vcov=placebo.vcov.tolist())
        if (
            not np.isfinite(result.beta).all()
            or not np.isfinite(result.vcov).all()
            or not np.isfinite(a.se).all()
        ):
            raise ValueError("non-finite fitted estimates/covariance")
    except MemoryError as exc:
        report.update(status="guard_refused", error=str(exc))
    except Exception as exc:
        report.update(status="fit_failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        phase[0] = "cleanup"
        if result is not None:
            result.cleanup()
        stop.set()
        sampler.join()
        for owner, name, original in replacements:
            setattr(owner, name, original)
    report.update(
        wall_s=time.perf_counter() - start,
        phase_peaks_mib=peaks,
        peak_rss_mib=highwater_mib(),
        final_rss_mib=process.memory_info().rss / 2**20,
        numerical_work=dict(progress),
        python=platform.python_version(),
        polars=api.pl.__version__,
    )
    print("RESULT " + json.dumps(report, allow_nan=False))


def case(name, spec, **options):
    return dict(
        name=name,
        data=spec,
        estimator="cs",
        method="dr",
        threads=2,
        batch_units=512,
        preprocess_mb=16,
        memory_gb=1.5,
        timeout=240,
        **options,
    )


def cases(suite):
    if suite == "smoke":
        out = []
        for label, spec in [
            ("balanced", dataset(2000)),
            (
                "entry_wide",
                dataset(5000, width=17, pattern="entry_exit", strings=True, row_group=257),
            ),
        ]:
            for method in ("reg", "ipw", "dr"):
                c = case(f"smoke_{label}_{method}", spec)
                c["method"] = method
                c["r_reference"] = True
                out.append(c)
        return out
    out = []
    for n in (10000, 100000, 1000000):
        for pattern in ("balanced", "entry_exit"):
            out.append(case(f"cs_n{n}_{pattern}_k3", dataset(n, pattern=pattern)))
    for n in (10000, 100000):
        for pattern in ("balanced", "entry_exit"):
            for method in ("dr",) if n == 10000 else ("reg", "ipw", "dr"):
                c = case(f"cs_n{n}_{pattern}_k17_{method}", dataset(n, width=17, pattern=pattern))
                c["method"] = method
                out.append(c)
    out += [
        case("cs_k33", dataset(100000, width=33, pattern="entry_exit")),
        case("cs_more_periods", dataset(100000, periods=8, cohorts=4, pattern="gaps")),
        case("cs_more_cohorts_k3", dataset(50000, periods=12, cohorts=6, pattern="entry_exit")),
        case(
            "cs_more_cohorts_k17",
            dataset(50000, periods=12, cohorts=6, width=17, pattern="entry_exit"),
        ),
        case("cs_invariant", dataset(100000, invariant=True)),
        case("cs_invariant_entry_exit", dataset(100000, invariant=True, pattern="entry_exit")),
        case(
            "cs_strings_large_groups",
            dataset(100000, width=17, pattern="entry_exit", strings=True, row_group=131072),
        ),
    ]
    small = case("cs_tighter_buffers", dataset(100000, width=17, pattern="entry_exit"))
    small.update(batch_units=128, preprocess_mb=4, memory_gb=1.0)
    out.append(small)
    four = case("cs_four_threads", dataset(100000, width=17, pattern="entry_exit"))
    four["threads"] = 4
    out.append(four)
    preload = case("cs_preloaded_formula", dataset(100000, width=17, pattern="entry_exit"))
    preload["preload_formula"] = True
    out.append(preload)
    capped = case("cs_million_1gb", dataset(1000000))
    capped["memory_gb"] = 1.0
    out.append(capped)
    for est in ("sunab", "bjs"):
        for n, T, G in ((1000000, 5, 3), (250000, 12, 6)):
            c = case(
                f"{est}_n{n}_t{T}", dataset(n, periods=T, cohorts=G, width=1, pattern="entry_exit")
            )
            c["estimator"] = est
            out.append(c)
    c = case(
        "bjs_coarse_leaveout_aggregations",
        dataset(100000, periods=12, cohorts=6, width=1, pattern="gaps"),
    )
    c.update(estimator="bjs", avgeffectsby="all", leaveout=True, extra_aggregations=True)
    out.append(c)
    c = case("expected_budget_refusal", dataset(10000))
    c.update(memory_gb=0.35, expected="guard_refused")
    out.append(c)
    for pattern in ("balanced", "entry_exit"):
        c = case(f"compat137_{pattern}", dataset(100000, width=17, pattern=pattern))
        c["polars137"] = True
        out.append(c)
    for c in out:
        if c["name"] in (
            "cs_n100000_balanced_k3",
            "cs_n100000_entry_exit_k3",
            "cs_more_cohorts_k17",
        ):
            c["r_reference"] = True
        if c["name"] == "cs_more_cohorts_k17":
            c["r_memory_gb"] = 3.0
    return out


def disk_bytes(root):
    total = 0
    for path in Path(root).rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_blocks * 512
        except FileNotFoundError:
            pass  # Writers and cleanup can remove files between samples.
    return total


def monitor(command, env, config, stem):
    start = time.monotonic()
    peak = 0
    disk = 0
    status = None
    last_disk = 0
    with (
        stem.with_suffix(".stdout").open("w") as stdout,
        stem.with_suffix(".stderr").open("w") as stderr,
    ):
        proc = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
        process = psutil.Process(proc.pid)
        try:
            while proc.poll() is None:
                try:
                    tree = [process] + process.children(recursive=True)
                    rss = sum(p.memory_info().rss for p in tree if p.is_running())
                    peak = max(peak, rss)
                except psutil.NoSuchProcess:
                    pass
                now = time.monotonic()
                if now - last_disk > 0.2:
                    disk = max(disk, disk_bytes(config["workdir"]))
                    last_disk = now
                if peak > config["memory_gb"] * 1e9:
                    status = "rss_limit_exceeded"
                    break
                if now - start > config["timeout"]:
                    status = "timeout"
                    break
                time.sleep(0.005)
        finally:
            if proc.poll() is None:
                try:
                    children = process.children(recursive=True)
                except psutil.NoSuchProcess:
                    children = []
                for child in children:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                proc.kill()
            proc.wait()
    result = {}
    for line in stem.with_suffix(".stdout").read_text().splitlines():
        if line.startswith("RESULT "):
            result = json.loads(line[7:])
    status = status or result.get("status", "process_failed")
    peak = max(peak, result.get("peak_rss_mib", 0) * 2**20)
    if status == "ok" and peak > config["memory_gb"] * 1e9:
        status = "rss_limit_exceeded"
    return dict(
        config=config,
        status=status,
        fit=result,
        peak_rss_mib=peak / 2**20,
        peak_disk_mb=disk / 1e6,
        process_s=time.monotonic() - start,
        returncode=proc.returncode,
        rss_scope="child process tree; 5 ms sampling plus reported process HWM",
        stderr=stem.with_suffix(".stderr").read_text()[-4000:],
    )


def compare(left, right):
    import numpy as np

    if left.get("status") != "ok" or right.get("status") != "ok":
        return None
    a, b = left["fit"], right["fit"]
    keys = [tuple(c) for c in b["cells"]]
    ix = [keys.index(tuple(c)) for c in a["cells"]]
    beta = np.asarray(b["beta"])[ix]
    V = np.asarray(b["vcov"])[np.ix_(ix, ix)]
    be = float(np.max(np.abs(np.asarray(a["beta"]) - beta)))
    ve = float(np.max(np.abs(np.asarray(a["vcov"]) - V)))
    ae = abs(a["att"] - b["att"])
    se = abs(a["se"] - b["se"])
    return dict(
        beta_max_abs=be, vcov_max_abs=ve, att_abs=ae, se_abs=se, passed=max(be, ve, ae, se) < 2e-7
    )


def environment():
    versions = {
        p: importlib.metadata.version(p)
        for p in (
            "diff-stream",
            "hdfe-stream",
            "polars",
            "numpy",
            "scipy",
            "numba",
            "pyarrow",
            "psutil",
        )
    }
    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        versions=versions,
        available_memory_gb=psutil.virtual_memory().available / 1e9,
        cpu_count=os.cpu_count(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        network_required=False,
        clock="monotonic",
        date=time.strftime("%Y-%m-%d"),
        notes=[
            "Sequential fresh fit processes; data generation excluded from fit measurements.",
            "GB is decimal; MiB is binary. Process-tree RSS is sampled, not a cgroup guarantee.",
            "Imports and first-use initialization are included; preload case isolates formula imports.",
        ],
    )


def watchdog_selfcheck(results):
    with TemporaryDirectory(prefix="watchdog-check-", dir=results) as work:
        config = dict(workdir=work, memory_gb=0.05, timeout=5)
        memory = monitor(
            [sys.executable, "-c", "import time; x=bytearray(64*1024*1024); time.sleep(10)"],
            dict(os.environ),
            config,
            results / "watchdog_memory",
        )
        config = dict(workdir=work, memory_gb=0.5, timeout=0.2)
        timeout = monitor(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            dict(os.environ),
            config,
            results / "watchdog_timeout",
        )
    passed = memory["status"] == "rss_limit_exceeded" and timeout["status"] == "timeout"
    save(results / "watchdog_selfcheck.json", dict(passed=passed, memory=memory, timeout=timeout))
    if not passed:
        raise RuntimeError("resource watchdog self-check failed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=["smoke", "scale"], default="smoke")
    p.add_argument("--only", nargs="*")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--no-r", action="store_true")
    p.add_argument("--rscript", default=str(RSCRIPT))
    p.add_argument("--results", type=Path, default=RESULTS)
    a = p.parse_args()
    a.results.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    lock_file = (a.results / ".driver.lock").open("w")
    try:
        if fcntl is not None:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error("another resource-check driver is using this results directory")
    save(a.results / f"environment_{a.suite}.json", environment())
    jobs = [c for c in cases(a.suite) if not a.only or c["name"] in a.only]
    if a.only and set(a.only) - {c["name"] for c in jobs}:
        p.error("unknown case names: " + str(sorted(set(a.only) - {c["name"] for c in jobs})))
    failures = []
    rows = []
    source_hash = hashlib.sha256()
    for source in sorted((ROOT / "diff_stream").rglob("*.py")):
        source_hash.update(str(source.relative_to(ROOT)).encode())
        source_hash.update(source.read_bytes())
    library_hash = source_hash.hexdigest()
    if a.suite == "smoke":
        watchdog_selfcheck(a.results)
    for c in jobs:
        c["library_sha256"] = library_hash
        r_requested = c.get("r_reference", False) and not a.no_r
        destination = a.results / (c["name"] + ".json")
        # Only remove this runner's own abandoned scratch, never input data or
        # arbitrary folders supplied through an edited configuration file.
        old_config = a.results / (c["name"] + ".config.json")
        if old_config.exists():
            old_work = Path(json.loads(old_config.read_text())["workdir"])
            if (
                old_work.exists()
                and old_work.resolve().parent == a.results.resolve()
                and old_work.name.startswith(c["name"] + "-")
            ):
                shutil.rmtree(old_work)
        if destination.exists():
            prior = json.loads(destination.read_text())
            same = all(prior["config"].get(k) == v for k, v in c.items())
            references_done = not r_requested or prior.get("r_status") == "ok"
            if same and references_done and (not a.retry_failed or accepted(prior)):
                print(f"REUSE {c['name']}: {prior['status']}", flush=True)
                rows.append(prior)
                if not accepted(prior):
                    failures.append(c["name"])
                continue
            archive = a.results / "history" / (c["name"] + "-" + str(time.time_ns()))
            archive.mkdir(parents=True)
            for old in a.results.glob(c["name"] + ".*"):
                if old.is_file():
                    shutil.copyfile(old, archive / old.name)
        spec = c["data"]
        key = identity(spec)
        path = DATA / (key + ".parquet")
        env = dict(
            os.environ,
            **{
                k: str(c["threads"])
                for k in (
                    "STREAM_LIMITS_MAX_THREADS",
                    "POLARS_MAX_THREADS",
                    "NUMBA_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                )
            },
        )
        env.update(
            MPLCONFIGDIR="/tmp/diff-stream-mpl", PYTHONPATH=str(ROOT), PYTHONWARNINGS="ignore"
        )
        if (
            not path.exists()
            or not path.with_suffix(".json").exists()
            or (r_requested and not path.with_suffix(".csv").exists())
        ):
            payload = dict(spec=spec, path=str(path), csv_output=r_requested)
            subprocess.run(
                [sys.executable, __file__, "--generate", json.dumps(payload)], env=env, check=True
            )
        if c.get("polars137"):
            env["PYTHONPATH"] = "/tmp/diff-stream-polars137:" + str(ROOT)
        env["STREAM_LIMITS_MEMORY_GB"] = str(c["memory_gb"])
        with TemporaryDirectory(prefix=c["name"] + "-", dir=a.results) as work:
            config = dict(
                c,
                path=str(path),
                formula=formula(spec) if c["estimator"] == "cs" else "y ~ 1",
                panel_mode="balanced" if spec["pattern"] == "balanced" else "unbalanced",
                workdir=work,
                progress_path=str(a.results / (c["name"] + ".progress.json")),
            )
            payload = a.results / (c["name"] + ".config.json")
            save(payload, config)
            print(
                f"RUN {c['name']} ({spec['n']} units, T={spec['periods']}, K={spec['width']}, limit={c['memory_gb']} GB)",
                flush=True,
            )
            result = monitor(
                [sys.executable, __file__, "--fit", str(payload)],
                env,
                config,
                a.results / c["name"],
            )
            result["cleanup_ok"] = not list(Path(work).iterdir())
            result["reference_requested"] = r_requested
            save(destination, result)
        if r_requested and Path(a.rscript).exists():
            with TemporaryDirectory(prefix="r-" + c["name"] + "-", dir=a.results) as work:
                rc = dict(
                    config,
                    workdir=work,
                    csv_path=str(path.with_suffix(".csv")),
                    r_formula=formula(spec, True),
                )
                rc["memory_gb"] = c.get("r_memory_gb", c["memory_gb"])
                rconfig = a.results / (c["name"] + ".r.config.json")
                save(rconfig, rc)
                renv = dict(
                    env,
                    R_LIBS_USER=str(RLIB),
                    STREAM_LIMITS_MEMORY_GB=str(rc["memory_gb"]),
                    TMPDIR=work,
                )
                reference = monitor(
                    [a.rscript, str(ROOT / "benchmarks/external/resource_checks.R"), str(rconfig)],
                    renv,
                    rc,
                    a.results / (c["name"] + ".r"),
                )
                save(a.results / (c["name"] + ".r.json"), reference)
                result["r_comparison"] = compare(result, reference)
                result["r_status"] = reference["status"]
                save(destination, result)
        rows.append(result)
        if not accepted(result):
            failures.append(c["name"])
        print(
            f"DONE {c['name']}: {result['status']}, peak={result['peak_rss_mib']:.1f} MiB, process={result['process_s']:.1f}s, R={result.get('r_comparison')}",
            flush=True,
        )
    # Also compare resource settings/layout/thread/Polars changes on identical samples.
    base = a.results / "cs_n100000_entry_exit_k17_dr.json"
    if base.exists():
        baseline = json.loads(base.read_text())
        for name in (
            "cs_tighter_buffers",
            "cs_four_threads",
            "cs_preloaded_formula",
            "cs_strings_large_groups",
            "compat137_entry_exit",
        ):
            file = a.results / (name + ".json")
            if file.exists():
                row = json.loads(file.read_text())
                row["invariance"] = compare(baseline, row)
                save(file, row)
                if row["invariance"] and not row["invariance"]["passed"]:
                    failures.append(name)
    for baseline_name, variant_name in (
        ("cs_n1000000_balanced_k3", "cs_million_1gb"),
        ("cs_n100000_balanced_k17_dr", "compat137_balanced"),
    ):
        baseline_file, variant_file = (
            a.results / (baseline_name + ".json"),
            a.results / (variant_name + ".json"),
        )
        if baseline_file.exists() and variant_file.exists():
            row = json.loads(variant_file.read_text())
            row["invariance"] = compare(json.loads(baseline_file.read_text()), row)
            save(variant_file, row)
            if row["invariance"] and not row["invariance"]["passed"]:
                failures.append(variant_name)
    summary = []
    for file in sorted(a.results.glob("*.json")):
        value = json.loads(file.read_text())
        if "config" not in value or "fit" not in value:
            continue
        config = value["config"]
        fit = value["fit"]
        diag = fit.get("diagnostics", {})
        summary.append(
            dict(
                name=config["name"],
                status=value["status"],
                estimator=config["estimator"],
                method=config["method"],
                n=config["data"]["n"],
                periods=config["data"]["periods"],
                width=config["data"]["width"],
                pattern=config["data"]["pattern"],
                threads=config["threads"],
                polars=fit.get("polars"),
                fit_s=fit.get("fit_s"),
                aggregation_s=fit.get("aggregation_s"),
                peak_rss_mib=value["peak_rss_mib"],
                peak_disk_mb=value["peak_disk_mb"],
                passes=diag.get("nuisance_source_passes"),
                propensity_models=diag.get("propensity_models"),
                partitions=diag.get("partitions"),
                cells=len(fit.get("cells", [])),
                r_reference=file.name.endswith(".r.json"),
                validation=value.get("r_comparison", value.get("invariance")),
            )
        )
    if summary:
        with (a.results / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    save(a.results / f"run_{a.suite}.json", dict(cases=len(jobs), unexpected_failures=failures))
    return bool(failures)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        payload = json.loads(sys.argv[2])
        generate(payload["spec"], Path(payload["path"]), payload["csv_output"])
    elif len(sys.argv) > 1 and sys.argv[1] == "--fit":
        fit_child(json.loads(Path(sys.argv[2]).read_text()))
    else:
        sys.exit(main())
