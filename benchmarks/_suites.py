"""What the three benchmark suites fit: for each scenario (benchmarks/_designs.py),
the estimators, diff-stream's variants of each, and the other libraries compared.

The task is the same for every library: **the event study**, the estimates by
periods since treatment with their standard errors, as an applied user would
report it. diff-stream's fit includes `aggregate("event_time")` where the
estimator has one; R's did includes aggte(type = "dynamic"); etwfe includes
emfx(type = "event"). The data are read from the same Parquet file by every
library (diff-stream streams it inside its fit, so its read time is zero), and
each library gets the cores and options its documentation recommends. Each
competitor is compared with diff-stream's fit of the same estimator
(`max_abs_diff`, over the event times they share), which shows whether they
estimate the same thing.

A configuration's *kind* is "<scenario>:<library>:<fit>[:<options>]", options
comma-separated key=value; _harness.run_configs hands it to the child process
(the library "ds") or to `commands`, which builds the comparator's command.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

DS = "diff-stream"
ALL_CORES = os.cpu_count() or 1
WINDOW = (-3, 3)  # the stacked event window
TIMEOUT_S = 1800

FUNCTIONS = {
    "cs": "did_callaway_santanna",
    "sunab": "did_sun_abraham",
    "bjs": "did_bjs",
    "gardner": "did_two_stage",
    "etwfe": "did_extended_twfe",
    "stacked": "did_stacked",
    "bacon": "bacon_decomp",
    "twfe_weights": "twfe_weights",
}
ADJUSTMENTS = {"reg": "regression", "ipw": "ipw", "dr": "doubly_robust"}
ADJUSTMENT_NAMES = {"reg": "regression", "ipw": "IPW", "dr": "doubly robust"}


@dataclass(frozen=True)
class Model:
    fml: str  # diff-stream's formula
    covariates: tuple = ()  # the comparators' covariates
    fe: tuple = ()  # their extra fixed effects, in fixest syntax
    columns: tuple = ("id", "time", "first_treat", "y")  # what they read
    unbalanced: bool = False


PLAIN = Model("y ~ 1")
MODELS = {
    **{s: PLAIN for s in "ABCDF"},
    "E": Model("y ~ 1", unbalanced=True),
    "G": Model("y ~ x + z", ("x", "z"), (), ("id", "time", "first_treat", "y", "x", "z")),
    "H": Model(
        "y ~ x | region^time",
        ("x",),
        ("region^time",),
        ("id", "time", "first_treat", "y", "x", "region"),
    ),
    "I": Model(
        "y ~ 1 | county^time", (), ("county^time",), ("id", "time", "first_treat", "y", "county")
    ),
    "J": Model(
        "y ~ x | county^time + industry^time",
        ("x",),
        ("county^time", "industry^time"),
        ("id", "time", "first_treat", "y", "x", "county", "industry"),
    ),
}

# the other libraries, by fit: (label, library, options)
# libraries: "r" = benchmarks/external/did_fit.R, "dd" = diff_diff_fit.py,
# "pyfixest" = pyfixest_fit.py (its did2s fails on these panels in 0.60)
COMPETITORS = {
    "cs": [("did (R)", "r"), ("diff-diff CallawaySantAnna (Python)", "dd")],
    "sunab": [
        ("fixest sunab (R)", "r"),
        ("pyfixest saturated (Python)", "pyfixest"),
        ("diff-diff SunAbraham (Python)", "dd"),
    ],
    "bjs": [("didimputation (R)", "r"), ("diff-diff ImputationDiD (Python)", "dd")],
    "gardner": [("did2s (R)", "r"), ("diff-diff TwoStageDiD (Python)", "dd")],
    "etwfe": [("etwfe (R)", "r"), ("diff-diff ETWFE (Python)", "dd")],
    "stacked": [("fixest on stacked data (R)", "r"), ("diff-diff StackedDiD (Python)", "dd")],
    "bacon": [("bacondecomp (R)", "r"), ("diff-diff BaconDecomposition (Python)", "dd")],
    "twfe_weights": [("TwoWayFEWeights (R)", "r")],
}
# scenarios with covariates or fixed effects in a form only R's packages fit alike
# (diff-diff takes covariates but not fixed effects interacted with time; pyfixest's
# event study takes neither)
ONLY_R = {"H", "I", "J"}

# scenario -> fits compared
FITS = {
    **{s: ["cs", "sunab", "bjs", "gardner", "etwfe", "stacked"] for s in "ABCD"},
    "E": ["cs", "sunab", "bjs", "gardner"],
    "F": ["bacon", "twfe_weights"],
    "G": ["cs"],
    "H": ["sunab", "bjs", "gardner"],
    "I": ["sunab", "bjs"],
    "J": ["sunab", "bjs"],
}
# scenarios with a memory-limited variant of diff-stream's fits (max_memory_gb)
LIMITED = {"B": 2.0, "I": 4.0}


def parse(kind):
    scenario, library, fit, *rest = kind.split(":")
    options = dict(p.split("=", 1) for p in (rest[0].split(",") if rest and rest[0] else []))
    return scenario, library, fit, options


def kind(scenario, library, fit, **options):
    text = ",".join(f"{k}={v}" for k, v in options.items())
    return f"{scenario}:{library}:{fit}" + (f":{text}" if text else "")


def configs(scenario):
    """{label: kind} for a scenario, diff-stream's variants of each estimator first
    (the first of them is what the others are compared with)."""
    out = {}
    for fit in FITS[scenario]:
        function = FUNCTIONS[fit]
        if scenario == "G":
            for adjustment in ("reg", "ipw", "dr"):
                name = ADJUSTMENT_NAMES[adjustment]
                out[f"{DS} {function} ({name})"] = kind(scenario, "ds", fit, adjustment=adjustment)
                if adjustment == "dr":
                    out[f"{DS} {function} ({name}, 1 thread)"] = kind(
                        scenario, "ds", fit, adjustment=adjustment, threads=1
                    )
                for label, library in COMPETITORS[fit]:
                    if library in ("r", "dd"):
                        out[f"{label[:-1]}, {name})"] = kind(
                            scenario, library, fit, adjustment=adjustment
                        )
            continue
        out[f"{DS} {function}"] = kind(scenario, "ds", fit)
        out[f"{DS} {function} (1 thread)"] = kind(scenario, "ds", fit, threads=1)
        if scenario in LIMITED and fit in ("cs", "sunab", "bjs", "gardner", "etwfe", "stacked"):
            out[f"{DS} {function} ({LIMITED[scenario]:g} GB limit)"] = kind(
                scenario, "ds", fit, limit=LIMITED[scenario]
            )
        if scenario == "H" and fit in ("sunab", "bjs"):
            # 240 levels: "auto" is the iterative solver; the dense one is the comparison
            out[f"{DS} {function} (dense solver)"] = kind(scenario, "ds", fit, solver="dense")
        for label, library in COMPETITORS[fit]:
            if scenario in ONLY_R and library != "r":
                continue
            out[label] = kind(scenario, library, fit)
    return out


def references(scenario):
    """{label: the label whose event study it is compared with}: the diff-stream
    fit of the same estimator (and, in G, of the same adjustment)."""
    cfg = configs(scenario)
    out = {}
    first = {}
    for label, k in cfg.items():
        _s, library, fit, options = parse(k)
        key = (fit, options.get("adjustment"))
        if library == "ds" and not options:
            first[key] = label
        elif library == "ds" and set(options) == {"adjustment"}:
            first[key] = label
    for label, k in cfg.items():
        _s, library, fit, options = parse(k)
        ref = first.get((fit, options.get("adjustment")))
        out[label] = ref if ref != label else None
    return out


def estimator_of(scenario):
    """{label: fit} for the CSV's estimator column."""
    return {label: parse(k)[2] for label, k in configs(scenario).items()}


def library_of(scenario):
    names = {"ds": DS, "r": "R", "dd": "diff-diff", "pyfixest": "pyfixest"}
    return {label: names[parse(k)[1]] for label, k in configs(scenario).items()}


def child_env(label_kind):
    """Environment variables of a diff-stream child with a thread limit."""
    _s, library, _fit, options = parse(label_kind)
    if library != "ds" or "threads" not in options:
        return {}
    t = str(options["threads"])
    return {
        "STREAM_LIMITS_MAX_THREADS": t,
        "POLARS_MAX_THREADS": t,
        "NUMBA_NUM_THREADS": t,
        "OPENBLAS_NUM_THREADS": t,
        "OMP_NUM_THREADS": t,
    }


# --------------------------------------------------------------------------
# the comparators' commands
# --------------------------------------------------------------------------


def commands(path, external_command, external_missing):
    """run_configs' `command(label, kind)`: None for diff-stream's own child, else the
    comparator's command or a string saying why it is skipped."""

    def command(_label, k):
        scenario, library, fit, options = parse(k)
        if library == "ds":
            return None
        model = MODELS[scenario]
        spec = dict(
            fit=fit,
            path=path,
            columns=list(model.columns),
            covariates=list(model.covariates),
            fe=list(model.fe),
            unbalanced=model.unbalanced,
            window=list(WINDOW),
            control_group="never" if library == "r" else "never_treated",
            adjustment=options.get("adjustment", "dr"),
        )
        kinds = {"r": "R", "dd": "diff-diff", "pyfixest": "pyfixest"}
        return external_command(kinds[library], **spec) or external_missing(kinds[library])

    return command


# --------------------------------------------------------------------------
# diff-stream's child
# --------------------------------------------------------------------------


def run_diff_stream(scenario, fit, options, path, workdir):
    import diff_stream as ds

    model = MODELS[scenario]
    common = dict(
        panel=("id", "time"), cohort="first_treat", n_threads=int(options.get("threads", ALL_CORES))
    )
    if "limit" in options:
        common["max_memory_gb"] = float(options["limit"])
    function = getattr(ds, FUNCTIONS[fit])
    t0 = time.perf_counter()
    kw = {}
    if model.unbalanced and fit not in ("bacon", "twfe_weights"):
        kw["incomplete_units"] = "keep"
    if fit == "cs":
        kw["adjustment"] = ADJUSTMENTS[options.get("adjustment", "dr")]
    if fit in ("sunab", "bjs") and "solver" in options:
        kw["fixed_effect_solver"] = options["solver"]
    if fit == "etwfe":
        kw["control_group"] = "never_treated"
    if fit == "stacked":
        kw["event_window"] = WINDOW
    if fit in ("bacon", "twfe_weights"):
        result = function("y ~ 1", path, workdir=str(workdir), **common)
    else:
        result = function(model.fml, path, workdir=str(workdir), **common, **kw)
    extra = {}
    try:
        if fit in ("cs", "sunab", "bjs", "etwfe"):
            event = result.aggregate("event_time")
            coefs = dict(zip(event.coefnames, event.beta.tolist(), strict=True))
        elif fit in ("gardner", "stacked"):
            coefs = dict(zip(result.coefnames, result.beta.tolist(), strict=True))
        elif fit == "bacon":
            coefs = {"twfe": float(result.twfe_estimate)}
            for row in result.summary().iter_rows(named=True):
                coefs[f"weight::{row['comparison_type']}"] = float(row["weight"])
        else:
            rows = {r["sign"]: r for r in result.summary().iter_rows(named=True)}
            coefs = {
                "n_positive": rows["positive"]["n_cells"],
                "n_negative": rows["negative"]["n_cells"],
                "sum_negative": rows["negative"]["weight_sum"],
                "sensitivity": float(result.sensitivity()["value"][0]),
            }
        extra["fixed_effect_solver"] = getattr(result, "diagnostics", {}).get(
            "fixed_effect_solver", ""
        )
        n_obs = getattr(result, "n_obs", None)
    finally:
        cleanup = getattr(result, "cleanup", None)
        if cleanup:
            cleanup()
    return dict(t_read=0.0, t_fit=time.perf_counter() - t0, n_obs=n_obs, coefs=coefs, **extra)


def runner(config, n, path, workdir):
    """run_child's runner: the diff-stream configurations."""
    scenario, library, fit, options = parse(config)
    if library != "ds":
        raise ValueError(f"no Python runner for {config!r}")
    return run_diff_stream(scenario, fit, options, path, workdir)


# --------------------------------------------------------------------------
# the suites' main loop
# --------------------------------------------------------------------------


def main(script, csv_name, scenarios, default_sizes, argv=None):
    import argparse
    from pathlib import Path

    from _designs import DESIGNS, data_path
    from _harness import (
        DATA_DIR,
        RESULTS_DIR,
        external_command,
        external_missing,
        run_configs,
        write_machine_info,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("sizes", nargs="*", type=int, help="numbers of units")
    ap.add_argument("--scenario", choices=scenarios, action="append")
    ap.add_argument("--only", nargs="+", help="labels, or prefixes with a trailing *")
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S)
    args = ap.parse_args(argv)
    out = RESULTS_DIR / csv_name
    write_machine_info(RESULTS_DIR / "machine.json")
    for n in args.sizes or default_sizes:
        for scenario in args.scenario or scenarios:
            path = data_path(DESIGNS[scenario], n, DATA_DIR)
            estimator, library = estimator_of(scenario), library_of(scenario)
            kinds = configs(scenario)
            cmd = commands(path, external_command, external_missing)
            run_configs(
                Path(script),
                out,
                scenario,
                n,
                path,
                kinds,
                command=cmd,
                timeout=args.timeout,
                only=args.only,
                reference=references(scenario),
                extra=lambda label, estimator=estimator, library=library: {
                    "estimator": estimator[label],
                    "library": library[label],
                },
                env=lambda label, kinds=kinds: child_env(kinds[label]),
            )
