"""Time and peak memory as the number of units grows, for the event study of every estimator
with no covariates.

    python benchmarks/scale_benchmark.py [units ...] [--scenario A] [--only LABEL ...]

Scenarios A-E compare the six estimators on five panels (8 or 30 periods, few or many
cohorts, one with 10% of the rows missing); F compares the two diagnostics. diff-stream
runs on all cores and on one thread, and on B with a memory limit; R's did, fixest,
didimputation, did2s, etwfe, bacondecomp and TwoWayFEWeights, and Python's diff-diff and
pyfixest, run where they are installed.

Results go to benchmarks/results/scale.csv; see _suites.py for the task every library
is given and _designs.py for the data.
"""

import sys
from pathlib import Path

from _harness import run_child
from _suites import main, runner

SCRIPT = Path(__file__).resolve()
SCENARIOS = ["A", "B", "C", "D", "E", "F"]
DEFAULT_SIZES = [10_000, 100_000, 1_000_000, 10_000_000]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        run_child(runner, *sys.argv[2:])
    else:
        main(SCRIPT, "scale.csv", SCENARIOS, DEFAULT_SIZES)
