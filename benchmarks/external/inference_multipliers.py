"""Generate common multipliers for the small R/Python inference fixture."""

import gzip
import sys
from pathlib import Path

import numpy as np
from hdfe_stream.core.wildboot import WEIGHTS

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
xi = WEIGHTS["rademacher"](400 * 199, np.random.default_rng(815)).reshape(400, 199)
with gzip.open(out / "multipliers.csv.gz", "wt") as f:
    np.savetxt(f, xi, delimiter=",", fmt="%.17g")
