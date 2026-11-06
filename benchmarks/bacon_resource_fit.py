"""Fresh-process balanced Goodman–Bacon resource check."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import numpy as np
from resource_checks import highwater_mib

from diff_stream import bacon_decomp

c = json.loads(Path(sys.argv[1]).read_text())
try:
    r = bacon_decomp(
        "y ~ 1",
        c["path"],
        panel=("id", "time"),
        cohort="first_treat",
        weights=c.get("weights"),
        n_threads=2,
        max_memory_gb=1,
        preprocess_mb=8,
        workdir=c["workdir"],
    )
except MemoryError as error:
    if not c.get("expect_guard") or "comparison table" not in str(error):
        raise
    result = dict(status="guard_refused", message=str(error), peak_rss_mib=highwater_mib())
else:
    assert not c.get("expect_guard")
    assert np.isclose(r.twfe_estimate, r.decomposed_estimate, atol=1e-10)
    assert np.isclose(r.comparisons()["weight"].sum(), 1)
    assert r.n_units == c["n_units"]
    assert r.diagnostics["sampling_weights"] == c.get("weights")
    result = dict(
        status="ok",
        n_units=r.n_units,
        n_obs=r.n_obs,
        n_comparisons=r.comparisons().height,
        twfe=r.twfe_estimate,
        reconstruction_error=r.reconstruction_error,
        partitions=r.diagnostics["partitions"],
        peak_rss_mib=highwater_mib(),
    )
print("RESULT " + json.dumps(result))
