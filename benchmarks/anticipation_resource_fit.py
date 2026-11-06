"""Child for the fixed anticipation resource check; invoked by its driver."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import numpy as np
from resource_checks import highwater_mib

from benchmarks._estimators import fit_estimator

c = json.loads(Path(sys.argv[1]).read_text())
r = fit_estimator(
    "y ~ x + z" if c["estimator"] == "cs" else "y ~ 1",
    c["path"],
    panel=("id", "time"),
    cohort="first_treat",
    estimator=c["estimator"],
    anticipation=2,
    weights=c.get("weights"),
    incomplete_units="keep",
    n_threads=2,
    max_memory_gb=1.0,
    preprocess_mb=8,
    batch_units=512,
    workdir=c["workdir"],
    residual_averaging="all" if c["estimator"] == "bjs" else "cohort_time",
    leave_out=c["estimator"] == "bjs",
)
try:
    a = r.aggregate("overall")
    w = r.aggregate(balance_window=(-2, 1))
    assert np.isfinite(r.beta).all() and np.isfinite(r.vcov).all() and np.isfinite(w.vcov).all()
    assert r.diagnostics["anticipation"] == 2
    print(
        "RESULT "
        + json.dumps(
            dict(
                status="ok",
                n_units=r.n_units,
                n_obs=r.n_obs,
                cells=len(r.cells),
                att=float(a.beta[0]),
                se=float(a.se[0]),
                peak_rss_mib=highwater_mib(),
                partitions=r.diagnostics["partitions"],
            )
        )
    )
finally:
    r.cleanup()
