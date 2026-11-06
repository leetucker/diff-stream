"""Covariates and fixed effects that fit a dense design: where R's did is slowest and
where the iterative fixed-effect solver can be compared with the dense one.

    python benchmarks/covariates_benchmark.py [units ...] [--scenario G] [--only LABEL ...]

G: Callaway-Sant'Anna with two covariates by regression, IPW and doubly robust adjustment
(R's did and diff-diff). H: Sun-Abraham, BJS and the two-stage estimator with a covariate and
region-by-period fixed effects (240 levels), with diff-stream's dense and iterative
solvers (fixest, didimputation and did2s).

Results go to benchmarks/results/covariates.csv; see _suites.py for the task every library
is given and _designs.py for the data.
"""

import sys
from pathlib import Path

from _harness import run_child
from _suites import main, runner

SCRIPT = Path(__file__).resolve()
SCENARIOS = ["G", "H"]
DEFAULT_SIZES = [10_000, 100_000, 1_000_000, 10_000_000]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        run_child(runner, *sys.argv[2:])
    else:
        main(SCRIPT, "covariates.csv", SCENARIOS, DEFAULT_SIZES)
