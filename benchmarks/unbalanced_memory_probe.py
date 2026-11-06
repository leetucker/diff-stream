"""Separate fixed imports/first use from native SA data-dependent RSS.

Run in a fresh process per input. The default probes the specialized SA path;
--hdfe-reference temporarily restores the prior general-backend adapter for a
comparison. Instrumentation changes runtime callables, never library files. RSS includes resident mmap pages and retained
allocator pages, not just live NumPy allocations. Peaks are sampled every 5ms.
"""

import json
import sys
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from hdfe_stream.core.estimator import _StreamingBase  # noqa: E402
from hdfe_stream.core.formula import pyfixest_formula_api  # noqa: E402

import diff_stream.unbalanced as observed  # noqa: E402
from diff_stream import sunab_stream  # noqa: E402


def hdfe_reference(store, cells, anticipation=0):
    from hdfe_stream import feols_stream

    names = [f"effect{j}" for j in range(len(cells))]
    data = pl.scan_parquet(store.paths, low_memory=True).with_columns(
        [
            ((pl.col("g") == g) & (pl.col("t") == t)).cast(pl.Float64).alias(name)
            for name, (g, t) in zip(names, cells, strict=True)
        ]
    )
    result = feols_stream(
        "y ~ " + " + ".join(names) + " | u + t",
        data,
        workdir=str(store.root / "sunab"),
        vcov={"CRV1": "u"},
        fixef_rm="none",
        verbose=False,
        save_resid=False,
        n_threads=2,
        max_memory_gb=None,
        stream="u",
        batch_rows=min(store.read_rows, store.batch_units * store.periods),
        row_group_size=store.read_rows,
        rows_per_bucket=max(1, int(store.leaf_bytes / (8 * (len(names) + 8)))),
    )
    try:
        if result.collin_vars:
            raise ValueError(
                "Sun–Abraham interactions are unidentified on the observed rows: "
                + str(result.collin_vars)
            )
        order = [result.coefnames.index(name) for name in names]
        factor = result.diagnostics["vcov_factors"][result.vcov_type]
        counts = np.array([store.cell_counts[g][t - store.times[0]] for g, t in cells], float)
        return result.beta[order], counts, result.vcov[np.ix_(order, order)] / factor
    finally:
        result.cleanup()


def main():
    process = psutil.Process()
    report = {"input": sys.argv[1], "import_baseline_mib": process.memory_info().rss / 2**20}
    backend = "--hdfe-reference" in sys.argv
    report["implementation"] = "hdfe_reference" if backend else "cohort_time_systems"
    if backend:
        observed.fit_sunab = hdfe_reference
        pyfixest_formula_api()
        report["formula_imports_mib"] = process.memory_info().rss / 2**20
    kw = dict(
        panel=("id", "time"),
        cohort="first_treat",
        panel_mode="unbalanced",
        threads=2,
        preprocess_mb=16,
    )

    # A fixed tiny fit loads the JIT and estimator machinery before measuring N.
    with TemporaryDirectory(prefix="diff-memory-warm-") as work:
        r = sunab_stream(
            "y ~ 1", pl.scan_parquet(sys.argv[1], low_memory=True).head(500), workdir=work, **kw
        )
        r.cleanup()
    report["warmed_rss_mib"] = process.memory_info().rss / 2**20
    peaks, phase = {}, ["preparation"]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            label = phase[0]
            peaks[label] = max(peaks.get(label, 0), process.memory_info().rss / 2**20)
            stop.wait(0.005)

    names = ["_pass0_code", "_pass1_cells", "_load_ident", "_solve", "_assemble", "_pass2"]
    originals = {name: getattr(_StreamingBase, name) for name in names}

    def wrap(name, original):
        def call(self, *a, **kw):
            old = phase[0]
            phase[0] = name
            try:
                return original(self, *a, **kw)
            finally:
                phase[0] = old

        return call

    for name, original in originals.items():
        setattr(_StreamingBase, name, wrap(name, original))
    original_sa = observed.fit_sunab

    def fit_sa(*args, **kwargs):
        phase[0] = "sunab_systems"
        try:
            return original_sa(*args, **kwargs)
        finally:
            phase[0] = "preparation"

    observed.fit_sunab = fit_sa
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    start = time.perf_counter()
    try:
        with TemporaryDirectory(prefix="diff-memory-sa-") as work:
            r = sunab_stream("y ~ 1", sys.argv[1], workdir=work, **kw)
            report.update(n_units=r.n_units, n_obs=r.n_obs, checksum=float(r.beta.sum()))
            r.cleanup()
    finally:
        stop.set()
        sampler.join()
        observed.fit_sunab = original_sa
        for name, original in originals.items():
            setattr(_StreamingBase, name, original)
    report.update(
        wall_s=time.perf_counter() - start,
        phase_peaks_mib=peaks,
        final_rss_mib=process.memory_info().rss / 2**20,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
