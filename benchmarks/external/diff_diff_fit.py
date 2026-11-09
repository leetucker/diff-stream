"""One fit with the diff-diff package, for benchmarks/_harness.py: read the Parquet
file with pandas (timed as "read"), estimate the event study, and print one line,

    RESULT {"t_read": ..., "t_fit": ..., "peak_memory_mb": ..., "coefs": {"event::0": ...}, ...}

    python diff_diff_fit.py '<spec as JSON>'     (see the benchmarks' external_command)

spec["fit"] is cs, sunab, bjs, gardner, etwfe, stacked or bacon; the other keys are
as in did_fit.R (columns, adjustment, unbalanced, window, control_group, path).
Estimates are "event::<k>", k the periods since treatment. diff-diff runs on
pandas, in memory, on one thread.
"""

import json
import sys
import time


def hwm_mb():
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def event_study(results, aggregated=None):
    """{"event::k": estimate}: from the object `aggregate("event_study")` returned
    (arrays event_time and att), or the results' event_study_effects dict."""
    out = {}
    if aggregated is not None and hasattr(aggregated, "event_time"):
        for k, effect in zip(aggregated.event_time, aggregated.att, strict=True):
            out[f"event::{int(k)}"] = float(effect)
        return out
    for k, v in results.event_study_effects.items():
        if v.get("is_reference"):
            continue
        effect = v.get("effect", v.get("att"))
        if effect is not None:
            out[f"event::{int(k)}"] = float(effect)
    return out


def main():
    spec = json.loads(sys.argv[1])
    baseline = hwm_mb()
    import diff_diff as dd
    import pandas as pd

    t0 = time.perf_counter()
    d = pd.read_parquet(spec["path"], columns=spec["columns"])
    t_read = time.perf_counter() - t0
    names = dict(outcome="y", unit="id", time="time", first_treat="first_treat")
    control = spec.get("control_group", "never_treated")
    t0 = time.perf_counter()
    fit = spec["fit"]
    if fit == "cs":
        model = dd.CallawaySantAnna(
            estimation_method=spec["adjustment"],
            control_group=control,
            allow_unbalanced_panel=bool(spec.get("unbalanced")),
        )
        res = model.fit(d, **names, covariates=spec.get("covariates") or None)
        coefs = event_study(res, res.aggregate("event_study"))
    elif fit in ("sunab", "bjs", "gardner", "etwfe"):
        if fit == "sunab":
            model = dd.SunAbraham(control_group=control)
        elif fit == "bjs":
            model = dd.ImputationDiD()
        elif fit == "gardner":
            model = dd.TwoStageDiD()
        else:
            model = dd.ETWFE(control_group=control)
        # ETWFE is run without covariates: diff-diff's three covariate arguments are
        # specifications of their own, none of which matches Wooldridge's full
        # interaction of covariates with cohort and time that diff-stream and R's etwfe fit
        covariates = None if fit == "etwfe" else spec.get("covariates") or None
        res = model.fit(d, **names, covariates=covariates) if covariates else model.fit(d, **names)
        coefs = event_study(res, None if fit == "sunab" else res.aggregate("event_study"))
    elif fit == "stacked":
        lo, hi = spec["window"]
        # diff-diff's "not_yet_treated" is the clean-control rule of Wing, Freedman and
        # Hollingsworth's code (never treated, or treated after the window)
        model = dd.StackedDiD(kappa_pre=-lo, kappa_post=hi, control_group="not_yet_treated")
        res = model.fit(d, **names)
        coefs = event_study(res)
    elif fit == "bacon":
        d["treat"] = ((d.first_treat > 0) & (d.time >= d.first_treat)).astype(int)
        res = dd.BaconDecomposition().fit(d, **names)
        coefs = {"twfe": float(res.twfe_estimate)}
        for comparison in res.comparisons:
            key = f"weight::{comparison.comparison_type}"
            coefs[key] = coefs.get(key, 0.0) + float(comparison.weight)
    else:
        raise SystemExit(f"unknown fit: {fit}")
    t_fit = time.perf_counter() - t0
    print(
        "RESULT "
        + json.dumps(
            dict(
                t_read=t_read,
                t_fit=t_fit,
                peak_memory_mb=hwm_mb(),
                baseline_memory_mb=baseline,
                peak_disk_mb=0.0,
                n_obs=len(d),
                coefs=coefs,
            )
        )
    )


if __name__ == "__main__":
    main()
