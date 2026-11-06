"""Fresh-process weighted BJS treatment/placebo fitting under the RSS watchdog."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import numpy as np
import polars as pl
from resource_checks import highwater_mib

from diff_stream import bjs_stream

c = json.loads(Path(sys.argv[1]).read_text())
d = pl.scan_parquet(c["path"]).filter(pl.col("id") < c["n_units"])
f = bjs_stream(
    "y ~ 1",
    d,
    panel=("id", "time"),
    cohort="first_treat",
    weights="w",
    panel_mode=c["panel_mode"],
    anticipation=c["anticipation"],
    pretrends=3,
    avgeffectsby="all",
    leaveout=True,
    threads=2,
    memory_gb=1,
    preprocess_mb=8,
    batch_units=512,
    workdir=c["workdir"],
)
try:
    p = f.placebos()
    test = f.pretrend_test()
    assert np.isfinite(p.beta).all() and np.isfinite(p.vcov).all()
    a = f.aggregate("simple")
    assert np.isfinite(a.vcov).all()
    f.cleanup()
    assert f.pretrend_test() == test
    print(
        "RESULT "
        + json.dumps(
            dict(
                status="ok",
                n_units=f.n_units,
                n_obs=f.n_obs,
                clean_rows=p.n_obs,
                placebo_beta=p.beta.tolist(),
                pretrend_pvalue=test["pvalue"],
                peak_rss_mib=highwater_mib(),
                partitions=f.diagnostics["partitions"],
            )
        )
    )
finally:
    f.cleanup()
