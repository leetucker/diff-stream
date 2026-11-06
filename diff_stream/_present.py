"""What a fit shows: public option names in diagnostics, original period labels.

The estimators work with internal option names and with consecutively
numbered periods (see periods.py). When a fit returns, its result keeps that
internal record for its own methods (``result._internal``) and shows the user
the same record with the public option names and values, and with cohorts and
periods under their original labels. Coefficient names and the cohort and time
columns of every table follow the same labels.
"""

import re

import polars as pl

from .periods import PeriodLabels

IDENTITY = PeriodLabels(0, 1)

PUBLIC_ESTIMATORS = {
    "cs": "did_callaway_santanna",
    "sunab": "did_sun_abraham",
    "bjs": "did_bjs",
    "gardner": "did_two_stage",
    "etwfe": "did_extended_twfe",
    "stacked": "did_stacked",
}

# internal diagnostics key -> public key
KEYS = {
    "reference_group_requested": "last_treated_controls",
    "reference_group": "reference",
    "balance_requested": "covariate_balance",
    "balance_ecdf": "balance_distributions",
    "pretrends": "pretrend_leads",
    "bootstrap_reps": "multiplier_bootstrap_reps",
    "bootstrap_seed": "multiplier_bootstrap_seed",
    "bootstrap_weights": "multiplier_bootstrap_weights",
    "bootstrap_unit_order": "multiplier_bootstrap_unit_order",
    "panel_mode": "incomplete_units",
    "est_method": "adjustment",
    "trim_level": "propensity_trim",
    "avgeffectsby": "residual_averaging",
    "leaveout": "leave_out",
    "ess_warn_threshold": "effective_sample_warning",
    "ess_suppress_threshold": "min_effective_sample",
    "specification": "event_study",
}
# public key -> {internal value: public value}
VALUES = {
    "last_treated_controls": {
        "auto": "if_needed",
        "nevertreated": "never",
        "lasttreated": "always",
    },
    "reference": {"nevertreated": "never_treated", "lasttreated": "last_treated"},
    "incomplete_units": {"balanced": "error", "complete": "drop", "unbalanced": "keep"},
    "control_group": {"nevertreated": "never_treated", "notyettreated": "not_yet_treated"},
    "adjustment": {"reg": "regression", "ipw": "ipw", "dr": "doubly_robust"},
    "event_study": {"dynamic": True, "static": False},
}
TIME_KEYS = {
    "time_min",
    "time_max",
    "input_time_min",
    "input_time_max",
    "reference_cutoff",
    "reference_last_clean_time",
    "covariate_time",
    "weight_time",
    "time",
}
COHORT_KEYS = {"reference_cohort", "cohort"}
COHORT_DICTS = {"cohort_sizes", "cohort_weight_totals"}
COHORT_LISTS = {"controls", "contributing_cohorts", "cohorts"}


def labels_of(result):
    return getattr(result, "_periods", None) or IDENTITY


def internal(result):
    """The internal record a result's methods read (public one if none was kept)."""
    record = getattr(result, "_internal", None)
    return record if record is not None else result.diagnostics


def present(result, labels):
    """Keep the internal record, show the public one; relabel names (in place)."""
    labels = labels or IDENTITY
    result._internal = dict(result.diagnostics)
    result._periods = labels
    result.diagnostics = public_diagnostics(result.diagnostics, labels, result.estimator)
    if labels.step != 1:
        result.coefnames = [label_name(n, labels) for n in result.coefnames]
        ess = getattr(result, "_ess_table", None)
        if ess is not None:
            # Joined with aggregation contributors by coefficient name.
            names = [
                pl.col(c).map_elements(lambda n: label_name(n, labels), return_dtype=pl.String)
                for c in ("coefficient", "cell")
            ]
            result._ess_table = label_frame(ess.with_columns(names), labels)
    placebos = getattr(result, "_bjs_placebos", None)
    if placebos is not None and getattr(placebos, "_internal", None) is None:
        present(placebos, labels)
    return result


def public_diagnostics(diagnostics, labels, estimator=None):
    out = {}
    for key, value in diagnostics.items():
        if (
            estimator == "etwfe"
            and key == "inference"
            and value in ("conditional", "unconditional")
        ):
            out["treat_sample_as_fixed"] = value == "conditional"
            continue
        key = KEYS.get(key, key)
        if key in VALUES and isinstance(value, str | bool) and value in VALUES[key]:
            value = VALUES[key][value]
        out[key] = _relabel(key, value, labels)
    out["period_step"] = labels.step
    return out


def _relabel(key, value, labels):
    if labels.step == 1 or value is None:
        return value
    if key in TIME_KEYS and isinstance(value, int):
        return labels.label_int(value)
    if key in COHORT_KEYS and isinstance(value, int):
        return labels.label_cohort(value)
    if key in COHORT_DICTS and isinstance(value, dict):
        return {str(labels.label_cohort(int(k))): v for k, v in value.items()}
    if key in COHORT_LISTS and isinstance(value, list):
        return [labels.label_cohort(v) if isinstance(v, int) else v for v in value]
    if key == "omitted_cells" and isinstance(value, list):
        return [(labels.label_cohort(g), labels.label_int(t)) for g, t in value]
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return [{k: _relabel(k, v, labels) for k, v in item.items()} for item in value]
    return value


_NAME = re.compile(r"\b(g|cohort|t|calendar)::(\d+)")
_CELL = re.compile(r"\bATT::(\d+)::(\d+)")  # extended TWFE's cell names


def label_name(name, labels):
    """g::<cohort>, t::<period>, cohort::<g> and calendar::<t> under original labels."""

    def replace(match):
        kind, value = match.group(1), int(match.group(2))
        label = labels.label_cohort(value) if kind in ("g", "cohort") else labels.label_int(value)
        return f"{kind}::{label}"

    name = _CELL.sub(
        lambda m: (
            f"ATT::{labels.label_cohort(int(m.group(1)))}::{labels.label_int(int(m.group(2)))}"
        ),
        name,
    )
    return _NAME.sub(replace, name)


def label_frame(frame, labels):
    """A table's cohort and time columns under their original labels."""
    if labels is None or labels.step == 1:
        return frame
    o, s = labels.origin, labels.step
    exprs = []
    for column in frame.columns:
        if column in ("cohort", "treated_cohort", "control_cohort"):
            c = pl.col(column)
            exprs.append(pl.when(c == 0).then(0).otherwise(o + (c - o) * s).alias(column))
        elif column in ("time", "calendar_time", "pre_start", "pre_end", "post_start", "post_end"):
            exprs.append((o + (pl.col(column) - o) * s).alias(column))
        elif column == "cohorts" and frame.schema[column] == pl.List(pl.Int64):
            exprs.append(
                pl.col(column).list.eval(
                    pl.when(pl.element() == 0).then(0).otherwise(o + (pl.element() - o) * s)
                )
            )
    return frame.with_columns(exprs) if exprs else frame


def present_plan(diagnostics, labels):
    """An aggregation's diagnostics (contributing and excluded cohorts) under labels."""
    labels = labels or IDENTITY
    return {key: _relabel(key, value, labels) for key, value in diagnostics.items()}


def public_profile(out):
    """A returned Profile's notes under public names; any other result as it is."""
    from hdfe_stream.core.profile import Profile

    if isinstance(out, Profile):
        out.notes = {k: v for k, v in public_diagnostics(out.notes, IDENTITY).items()}
        for code, name in PUBLIC_ESTIMATORS.items():
            if out.model == f"diff-stream {code}":
                out.model = f"diff-stream {name}"
    return out
