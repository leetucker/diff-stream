"""One fit with pyfixest, for benchmarks/_harness.py: read the Parquet file with
pandas (timed as "read"), estimate the interaction-weighted (Sun-Abraham) event
study with pf.event_study(estimator="saturated") and its aggregation, and print
one line,

    RESULT {"t_read": ..., "t_fit": ..., "peak_memory_mb": ..., "coefs": {"event::0": ...}, ...}

    python pyfixest_fit.py '<spec as JSON>'     (see the benchmarks' external_command)

spec["path"] and spec["columns"] as in did_fit.R. Estimates are "event::<k>", k
the periods since treatment. pyfixest runs on pandas, in memory.
"""

import json
import sys
import time


def hwm_mb():
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def main():
    spec = json.loads(sys.argv[1])
    baseline = hwm_mb()
    import pandas as pd
    import pyfixest as pf

    t0 = time.perf_counter()
    d = pd.read_parquet(spec["path"], columns=spec["columns"])
    t_read = time.perf_counter() - t0
    if spec["fit"] != "sunab":
        raise SystemExit(f"unknown fit: {spec['fit']}")
    t0 = time.perf_counter()
    fit = pf.event_study(
        d,
        yname="y",
        idname="id",
        tname="time",
        gname="first_treat",
        estimator="saturated",
        cluster="id",
    )
    table = fit.aggregate(weighting="shares")
    coefs = {f"event::{int(k)}": float(v) for k, v in table["Estimate"].items()}
    t_fit = time.perf_counter() - t0
    print(
        "RESULT "
        + json.dumps(
            dict(
                t_read=t_read,
                t_fit=t_fit,
                peak_memory_mb=hwm_mb(),
                baseline_memory_mb=baseline,
                peak_disk_mb=0.0,
                n_obs=len(d),
                coefs=coefs,
            )
        )
    )


if __name__ == "__main__":
    main()
