"""Fresh-process weighted conditional-CS multiplier resource check."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import numpy as np
import polars as pl
from resource_checks import highwater_mib

from diff_stream import cs_stream

c = json.loads(Path(sys.argv[1]).read_text())
source = pl.scan_parquet(c["path"]).filter(pl.col("id") < c["n_units"])
r = cs_stream(
    "y ~ x + z",
    source,
    panel=("id", "time"),
    cohort="first_treat",
    weights="w",
    anticipation=2,
    panel_mode="unbalanced",
    bootstrap_reps=c["reps"],
    bootstrap_seed=431,
    threads=2,
    memory_gb=1.0,
    preprocess_mb=8,
    batch_units=512,
    workdir=c["workdir"],
)
a = r.aggregate(balance_window=(-4, 1))
band = a.multiplier_confint()
assert np.isfinite(band.select("2.5%", "97.5%").to_numpy()).all()
assert r.multiplier_draws().shape == (c["reps"], len(r.cells))
pre = a.pretrend_test()
print(
    "RESULT "
    + json.dumps(
        dict(
            status="ok",
            n_units=r.n_units,
            n_obs=r.n_obs,
            cells=len(r.cells),
            reps=c["reps"],
            draw_state_bytes=r._multiplier_draws.nbytes + r._multiplier_shares.nbytes,
            critical=float(band["critical_value"][0]),
            pretrend_pvalue=pre["pvalue"],
            peak_rss_mib=highwater_mib(),
        )
    )
)
