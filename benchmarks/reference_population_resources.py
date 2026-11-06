"""BJS/Gardner/ETWFE last-reference staging and inference under a shared cap."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stream_limits.launcher import run_limited  # noqa: E402
from stream_limits.testing import verify_memory_limit  # noqa: E402

from benchmarks._estimators import fit_estimator, public  # noqa: E402
from benchmarks.gardner_resources import save  # noqa: E402


def child(c):
    # Apply scheduler defaults before loading Polars in the child.

    import polars as pl

    fit = fit_estimator(
        "y~x+z" if c["estimator"] == "etwfe" else "y~x+z|region^time",
        pl.scan_parquet(c["path"], low_memory=True)
        .filter(pl.col("first_treat") != 0)
        .with_columns((pl.col("first_treat") + 1).alias("first_treat")),
        panel=("id", "time"),
        cohort="first_treat",
        estimator=c["estimator"],
        incomplete_units="keep",
        control_group=public(
            "control_group", "notyettreated" if c["estimator"] == "etwfe" else "nevertreated"
        ),
        last_treated_controls="if_needed",
        anticipation=1,
        weights="w",
        allow_time_varying_weights=True,
        event_study=public("event_study", "static" if c["estimator"] == "gardner" else "dynamic"),
        pretrend_leads=2 if c["estimator"] == "bjs" else 0,
        effective_sample_warning=None,
        n_threads=2,
        preprocess_mb=8,
        batch_units=256,
        max_memory_gb=1.5,
        workdir=c["workdir"],
    )
    try:
        a = fit if c["estimator"] == "gardner" else fit.aggregate("overall")
        result = dict(
            n_units=fit.n_units,
            n_obs=fit.n_obs,
            placebo_rows=len(fit.placebos().coefnames) if c["estimator"] == "bjs" else 0,
            beta=fit.beta.tolist(),
            vcov=fit.vcov.tolist(),
            att=a.beta.tolist(),
            att_vcov=a.vcov.tolist(),
            diagnostics=fit.diagnostics,
        )
    finally:
        fit.cleanup()
    print(json.dumps(result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", type=Path)
    p.add_argument("--cgroup-parent", type=Path)
    p.add_argument(
        "--results", type=Path, default=ROOT / "benchmarks/results/reference_population_resources"
    )
    a = p.parse_args()
    if a.fit:
        child(json.loads(a.fit.read_text()))
        return
    from benchmarks import resource_checks as h

    out = a.results.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        STREAM_LIMITS_MAX_THREADS="2",
        POLARS_MAX_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        NUMBA_NUM_THREADS="2",
        OMP_NUM_THREADS="2",
        MPLCONFIGDIR="/tmp/diff-stream-mpl",
    )
    if a.cgroup_parent:
        save(out / "cgroup_selfcheck.json", verify_memory_limit(a.cgroup_parent, env=env))
    save(out / "environment.json", h.environment())
    library = hashlib.sha256()
    for path in sorted((ROOT / "diff_stream").rglob("*.py")):
        library.update(path.read_bytes())
    rows = []
    for estimator, n in [
        ("bjs", 10000),
        ("gardner", 10000),
        ("etwfe", 10000),
        ("bjs", 1000000),
        ("gardner", 1000000),
        ("etwfe", 1000000),
    ]:
        spec = dict(n=n, T=8, G=4, regions=4)
        data = ROOT / "benchmarks/data/richer_resources" / (h.identity(spec) + ".parquet")
        if not data.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "benchmarks/richer_resources.py"),
                    "--generate",
                    json.dumps(dict(spec=spec, path=str(data))),
                ],
                env=env,
                check=True,
            )
        name = f"{estimator}_{n}"
        with TemporaryDirectory(prefix=name + "-", dir=out) as work:
            c = dict(
                name=name,
                estimator=estimator,
                source_generated_units=n,
                cohort_shift=1,
                path=str(data),
                workdir=work,
                library_sha256=library.hexdigest(),
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            )
            config = out / (name + ".config.json")
            save(config, c)
            print("RUN", name, flush=True)
            result = run_limited(
                [sys.executable, __file__, "--fit", str(config)],
                mode="hard" if a.cgroup_parent else "watchdog",
                cgroup_parent=a.cgroup_parent,
                memory_gb=1.5,
                memory_measure="rss",
                timeout=600,
                env=env,
            )
            record = dict(
                config=c,
                result=result.as_dict(),
                cleanup_ok=not list(Path(work).iterdir()),
                input=json.loads(data.with_suffix(".json").read_text()),
            )
            if result.status == "ok":
                record["fit"] = json.loads(result.stdout)
            save(out / (name + ".json"), record)
            rows.append(record)
            print("DONE", name, result.status, round(result.peak_rss_bytes / 2**20, 1), flush=True)
    failures = [
        r["config"]["name"] for r in rows if r["result"]["status"] != "ok" or not r["cleanup_ok"]
    ]
    save(
        out / "summary.json",
        dict(
            cases=len(rows),
            unexpected_failures=failures,
            rows=[
                dict(
                    name=r["config"]["name"],
                    status=r["result"]["status"],
                    peak_rss_mib=r["result"]["peak_rss_bytes"] / 2**20,
                    elapsed_s=r["result"]["elapsed_s"],
                    **{k: r.get("fit", {}).get(k) for k in ["n_units", "n_obs", "placebo_rows"]},
                )
                for r in rows
            ],
        ),
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
