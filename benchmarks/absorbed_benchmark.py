"""Sun-Abraham and BJS with fixed effects of very many levels: county-by-period effects,
n/40 counties, so millions of levels at the largest sizes.

    python benchmarks/absorbed_benchmark.py [units ...] [--scenario I] [--only LABEL ...]

I: county-by-period. J: county-by-period and industry-by-period, with a covariate.
diff-stream absorbs them with hdfe-stream; fixest's sunab and didimputation (R) are
compared. diff-stream also runs under a 4 GB limit.

Results go to benchmarks/results/absorbed.csv; see _suites.py for the task every library
is given and _designs.py for the data.
"""

import sys
from pathlib import Path

from _harness import run_child
from _suites import main, runner

SCRIPT = Path(__file__).resolve()
SCENARIOS = ["I", "J"]
DEFAULT_SIZES = [10_000, 100_000, 1_000_000, 10_000_000]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        run_child(runner, *sys.argv[2:])
    else:
        main(SCRIPT, "absorbed.csv", SCENARIOS, DEFAULT_SIZES)
