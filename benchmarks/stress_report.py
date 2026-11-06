"""Summarize retained numerical stress and local complexity results offline."""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.complexity_checks import jobs
from benchmarks.resource_checks import ROOT, save


def main():
    numeric_path = ROOT / "benchmarks/results/numerical_stress/run.json"
    numeric = json.loads(numeric_path.read_text())
    result_dir = ROOT / "benchmarks/results/complexity_checks"
    rows = []
    for config in jobs():
        path = result_dir / (config["name"] + ".json")
        if not path.exists():
            raise FileNotFoundError(path)
        r = json.loads(path.read_text())
        f, d = r["fit"], r["fit"].get("diagnostics", {})
        rows.append(
            dict(
                name=config["name"],
                status=r["status"],
                estimator=config["estimator"],
                n=config["data"]["n"],
                periods=config["data"]["periods"],
                cohorts=config["data"]["cohorts"],
                columns=config["data"]["width"],
                cells=len(f.get("cells", [])),
                fit_s=f.get("fit_s"),
                peak_rss_mib=r["peak_rss_mib"],
                peak_disk_mb=r["peak_disk_mb"],
                propensity_models=d.get("propensity_models"),
                source_passes=d.get("nuisance_source_passes"),
                error=f.get("error"),
            )
        )
    with (result_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparisons = [r["comparison"] for r in numeric["rows"] if r.get("comparison")]
    summary = dict(
        numerical_cases=numeric["cases"],
        numerical_outcomes=dict(Counter(r["status"] for r in numeric["rows"])),
        unexpected_numerical_outcomes=numeric["unexpected"],
        actual_R_comparisons=len(comparisons),
        max_reference_errors={
            key: max(c[key] for c in comparisons)
            for key in ("beta_max_abs", "vcov_max_abs", "att_abs", "se_abs")
        },
        complexity_cases=len(rows),
        complexity_outcomes=dict(Counter(r["status"] for r in rows)),
        complexity_non_successes=[r for r in rows if r["status"] != "ok"],
        largest_success_peak_rss_mib=max(r["peak_rss_mib"] for r in rows if r["status"] == "ok"),
        selected_resource_results=rows,
    )
    save(ROOT / "benchmarks/results/stress_summary.json", summary)
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in ("selected_resource_results", "complexity_non_successes")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
