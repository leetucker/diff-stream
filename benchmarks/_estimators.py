"""The estimators by short code, for tests and benchmarks that loop over them.

Reference files and benchmark tables are keyed by these codes ("cs", "sunab",
"bjs", "gardner", "etwfe"); the public API has one function per estimator.
"""

from inspect import signature

from diff_stream import (
    _options,
    did_bjs,
    did_callaway_santanna,
    did_extended_twfe,
    did_sun_abraham,
    did_two_stage,
)

ESTIMATORS = {
    "cs": did_callaway_santanna,
    "sunab": did_sun_abraham,
    "bjs": did_bjs,
    "gardner": did_two_stage,
    "etwfe": did_extended_twfe,
}


# Options some estimators take, at their defaults. Loops over estimators pass
# them to every estimator; the ones an estimator does not take are dropped
# only at these values, so a meaningful setting still raises TypeError.
DEFAULTS = {
    "control_group": "never_treated",
    "base_period": "varying",
    "adjustment": "doubly_robust",
    "propensity_trim": 0.995,
    "multiplier_bootstrap_reps": 0,
    "multiplier_bootstrap_seed": None,
    "multiplier_bootstrap_weights": "rademacher",
    "effective_sample_warning": None,
    "min_effective_sample": None,
    "covariate_balance": False,
    "balance_distributions": False,
    "residual_averaging": "cohort_time",
    "leave_out": False,
    "pretrend_leads": 0,
    "event_study": True,
    "reference_period": None,
    "fixed_effects": "cohort",
    "treat_sample_as_fixed": False,
    "max_design_columns": 2048,
}


def fit_estimator(fml, data, *, estimator="cs", **options):
    """Fit the estimator with code ``estimator`` ("cs", "sunab", ...)."""
    function = ESTIMATORS[estimator]
    accepted = signature(function).parameters
    for name in [n for n in options if n not in accepted]:
        if name in DEFAULTS and options[name] == DEFAULTS[name]:
            del options[name]
    return function(fml, data, **options)


_PUBLIC = {
    name: {internal: public for public, internal in mapping.items()}
    for name, mapping in _options.VALUES.items()
}
_PUBLIC["kind"] = {"dynamic": "event_time", "simple": "overall", "group": "cohort"}


def public(option, value):
    """The public value of an option given by its short code.

    Reference files are named with the codes ("nevertreated", "unbalanced",
    "dynamic"); the API takes the spelled-out values ("never_treated",
    "keep", "event_time"). Public values pass through unchanged.
    """
    return _PUBLIC.get(option, {}).get(value, value)
