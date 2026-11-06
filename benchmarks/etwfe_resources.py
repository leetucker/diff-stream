"""Local linear ETWFE scaling checks through the opt-in shared launcher."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stream_limits.launcher import run_limited  # noqa: E402
from stream_limits.testing import verify_memory_limit  # noqa: E402

from benchmarks.gardner_resources import save  # noqa: E402


def child(c):
    from diff_stream import etwfe_stream

    start = time.monotonic()
    fit = etwfe_stream(
        "y~x+z",
        c["path"],
        panel=("id", "time"),
        cohort="first_treat",
        control_group=c["control_group"],
        fixed_effects=c["fixed_effects"],
        inference="unconditional",
        panel_mode="unbalanced",
        weights="w",
        allow_time_varying_weights=True,
        threads=2,
        batch_units=c["batch_units"],
        preprocess_mb=c["preprocess_mb"],
        memory_gb=c["memory_gb"],
        workdir=c["workdir"],
    )
    aggregates = {
        kind: dict(beta=fit.aggregate(kind).beta.tolist(), vcov=fit.aggregate(kind).vcov.tolist())
        for kind in ["simple", "dynamic", "cohort", "calendar"]
    }
    print(
        json.dumps(
            dict(
                n_units=fit.n_units,
                n_obs=fit.n_obs,
                columns=len(fit.diagnostics["regression_columns"]),
                expanded_columns=fit.diagnostics["expanded_columns"],
                coefficients=len(fit.beta),
                fit_s=time.monotonic() - start,
                cells=fit.cells,
                beta=fit.beta.tolist(),
                vcov=fit.vcov.tolist(),
                aggregates=aggregates,
                diagnostics=fit.diagnostics,
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--cgroup-parent", type=Path)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--results", type=Path, default=ROOT / "benchmarks/results/etwfe_resources")
    args = parser.parse_args()
    if args.fit:
        child(json.loads(args.fit.read_text()))
        return
    from benchmarks import resource_checks as h

    jobs = [
        dict(
            name=f"notyet_{n}",
            n=n,
            T=8,
            G=4,
            regions=4,
            control_group="notyettreated",
            fixed_effects="cohort",
        )
        for n in [10000, 100000, 1000000]
    ]
    jobs.extend(
        [
            dict(
                name="never_unit_100000",
                n=100000,
                T=8,
                G=4,
                regions=4,
                control_group="nevertreated",
                fixed_effects="unit",
            ),
            dict(
                name="never_wide",
                n=20000,
                T=16,
                G=6,
                regions=8,
                control_group="nevertreated",
                fixed_effects="cohort",
            ),
            dict(
                name="notyet_100000_tight",
                n=100000,
                T=8,
                G=4,
                regions=4,
                control_group="notyettreated",
                fixed_effects="cohort",
                batch_units=64,
                preprocess_mb=4,
            ),
        ]
    )
    if args.only and set(args.only) - {j["name"] for j in jobs}:
        raise ValueError("unknown case")
    out = args.results.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        STREAM_LIMITS_MAX_THREADS="2",
        POLARS_MAX_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        NUMBA_NUM_THREADS="2",
        OMP_NUM_THREADS="2",
        MPLCONFIGDIR="/tmp/diff-stream-mpl",
        PYTHONPATH=str(ROOT),
    )
    save(out / "environment.json", h.environment())
    if args.cgroup_parent:
        save(out / "cgroup_selfcheck.json", verify_memory_limit(args.cgroup_parent, env=env))
    source = hashlib.sha256()
    for p in sorted((ROOT / "diff_stream").rglob("*.py")):
        source.update(str(p.relative_to(ROOT)).encode())
        source.update(p.read_bytes())
    data = ROOT / "benchmarks/data/richer_resources"
    data.mkdir(parents=True, exist_ok=True)
    rows = []
    for j in jobs:
        if args.only and j["name"] not in args.only:
            continue
        spec = {k: j[k] for k in ["n", "T", "G", "regions"]}
        path = data / (h.identity(spec) + ".parquet")
        if not path.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmarks/richer_resources.py"),
                    "--generate",
                    json.dumps(dict(spec=spec, path=str(path))),
                ],
                env=env,
                check=True,
            )
        with TemporaryDirectory(prefix=j["name"] + "-", dir=out) as work:
            c = dict(batch_units=256, preprocess_mb=8, memory_gb=1.5)
            c.update(j)
            c.update(
                path=str(path),
                workdir=work,
                library_sha256=source.hexdigest(),
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            )
            payload = out / (j["name"] + ".config.json")
            save(payload, c)
            print("RUN", j["name"], flush=True)
            result = run_limited(
                [sys.executable, __file__, "--fit", str(payload)],
                mode="hard" if args.cgroup_parent else "watchdog",
                cgroup_parent=args.cgroup_parent,
                memory_gb=c["memory_gb"],
                timeout=300,
                env=env,
            )
            r = dict(
                config=c,
                result=result.as_dict(),
                cleanup_ok=not list(Path(work).iterdir()),
                input=json.loads(path.with_suffix(".json").read_text()),
            )
            if result.status == "ok":
                r["fit"] = json.loads(result.stdout)
            save(out / (j["name"] + ".json"), r)
            rows.append(r)
            print(
                "DONE",
                j["name"],
                result.status,
                round(result.peak_rss_bytes / 2**20, 1),
                flush=True,
            )
    failures = [
        r["config"]["name"] for r in rows if r["result"]["status"] != "ok" or not r["cleanup_ok"]
    ]
    byname = {r["config"]["name"]: r for r in rows}
    comparison = None
    if not failures and {"notyet_100000", "notyet_100000_tight"} <= byname.keys():
        import numpy as np

        a, b = [byname[n]["fit"] for n in ["notyet_100000", "notyet_100000_tight"]]
        assert a["cells"] == b["cells"]
        comparison = {
            k: float(np.max(abs(np.asarray(a[k]) - np.asarray(b[k])))) for k in ["beta", "vcov"]
        }
        for kind in ["simple", "dynamic", "cohort", "calendar"]:
            for k in ["beta", "vcov"]:
                comparison[f"{kind}_{k}"] = float(
                    np.max(
                        abs(
                            np.asarray(a["aggregates"][kind][k])
                            - np.asarray(b["aggregates"][kind][k])
                        )
                    )
                )
        assert max(comparison.values()) < 2e-9
    save(
        out / "summary.json",
        dict(
            cases=len(rows),
            unexpected_failures=failures,
            buffer_invariance=comparison,
            rows=[
                dict(
                    name=r["config"]["name"],
                    status=r["result"]["status"],
                    peak_rss_mib=r["result"]["peak_rss_bytes"] / 2**20,
                    elapsed_s=r["result"]["elapsed_s"],
                    **{
                        k: r.get("fit", {}).get(k)
                        for k in ["n_units", "n_obs", "columns", "expanded_columns", "coefficients"]
                    },
                )
                for r in rows
            ],
        ),
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
