"""Rerun the example scripts and save their output for the documentation pages.

Run from the repository root after changing an example or the library:

    python docs/examples/refresh.py            # every example
    python docs/examples/refresh.py castle     # one example

Each page includes examples/<name>.py and docs/examples/output/<name>.txt.
Warnings go to stderr and are not saved.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/examples/output"


def main(names):
    OUTPUT.mkdir(exist_ok=True)
    scripts = sorted((ROOT / "examples").glob("*.py"))
    if names:
        scripts = [s for s in scripts if s.stem in names]
    env = dict(os.environ, STREAM_LIMITS_MAX_THREADS="4", POLARS_FMT_MAX_ROWS="12")
    for script in scripts:
        run = subprocess.run(
            [sys.executable, str(script)], cwd=ROOT, env=env, capture_output=True, text=True
        )
        if run.returncode:
            sys.exit(f"{script.name} failed:\n{run.stderr[-2000:]}")
        (OUTPUT / f"{script.stem}.txt").write_text(run.stdout)
        print(f"{script.name}: {len(run.stdout.splitlines())} lines")


if __name__ == "__main__":
    main(sys.argv[1:])
