"""Sample process RSS by execution phase; instrumentation stays outside the API.

Usage: STREAM_LIMITS_MAX_THREADS=4 python benchmarks/memory_probe.py panel.parquet --threads 4
RSS includes allocator-retained pages; phase peaks are not isolated live-object sizes.
"""

import argparse
import json
import platform
import threading
import time
from tempfile import TemporaryDirectory

import psutil

import diff_stream.api as api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-gb", type=float)
    parser.add_argument("--preprocess-mb", type=float, default=64)
    args = parser.parse_args()
    phase = ["baseline"]
    peaks = {}
    stop = threading.Event()
    process = psutil.Process()

    def sample():
        while not stop.is_set():
            label = phase[0]
            peaks[label] = max(peaks.get(label, 0), process.memory_info().rss / 2**20)
            stop.wait(0.005)

    original_inspect, original_init = api.inspect, api.PanelStore.__init__
    original_fit, original_scores = api.fit_means, api.change_scores
    original_prepare, original_leaf = api.PanelStore._prepare, api.PanelStore._leaf
    original_sink = api.pl.LazyFrame.sink_parquet

    def inspect(*a, **kw):
        phase[0] = "schema"
        return original_inspect(*a, **kw)

    def init(self, *a, **kw):
        phase[0] = "snapshot"
        original_init(self, *a, **kw)
        phase[0] = "prepared"

    def sink(self, *a, **kw):
        phase[0] = "snapshot"
        result = original_sink(self, *a, **kw)
        phase[0] = "row_validation"
        return result

    def prepare(self, *a, **kw):
        phase[0] = "hash_partition"
        return original_prepare(self, *a, **kw)

    def leaf(self, *a, **kw):
        phase[0] = "leaf_sort_and_validation"
        return original_leaf(self, *a, **kw)

    def fit(*a, **kw):
        phase[0] = "small_solve"
        return original_fit(*a, **kw)

    def scores(*a, **kw):
        phase[0] = "scores_and_covariance"
        return original_scores(*a, **kw)

    api.inspect, api.PanelStore.__init__ = inspect, init
    api.fit_means, api.change_scores = fit, scores
    api.PanelStore._prepare, api.PanelStore._leaf = prepare, leaf
    api.pl.LazyFrame.sink_parquet = sink
    peaks["baseline"] = process.memory_info().rss / 2**20
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    start = time.perf_counter()
    try:
        with TemporaryDirectory(prefix="diff-memory-") as work:
            r = api.cs_stream(
                "y ~ 1",
                args.data,
                panel=("id", "time"),
                cohort="first_treat",
                threads=args.threads,
                workdir=work,
                memory_gb=args.memory_gb,
                preprocess_mb=args.preprocess_mb,
            )
    finally:
        stop.set()
        sampler.join()
        api.inspect, api.PanelStore.__init__ = original_inspect, original_init
        api.fit_means, api.change_scores = original_fit, original_scores
        api.PanelStore._prepare, api.PanelStore._leaf = original_prepare, original_leaf
        api.pl.LazyFrame.sink_parquet = original_sink
    print(
        json.dumps(
            {
                "memory_gb": args.memory_gb,
                "python": platform.python_version(),
                "polars": api.pl.__version__,
                "preprocess_mb": args.preprocess_mb,
                "partitions": r.diagnostics["partitions"],
                "max_decoded_leaf_bytes": r.diagnostics["max_decoded_leaf_bytes"],
                "threads": args.threads,
                "n_obs": r.n_obs,
                "wall_s": time.perf_counter() - start,
                "sampled_peak_rss_mb": peaks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
