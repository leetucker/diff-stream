"""Public option names and values, translated to the estimators' internal ones.

The public functions spell options out (``incomplete_units="drop"``); the
estimation code was written against shorter internal names
(``panel_mode="complete"``). Translation happens once, here, so that the
estimator functions and refit_bootstrap() accept exactly the same options.
"""

# public option name -> internal option name
NAMES = {
    "n_threads": "threads",
    "max_memory_gb": "memory_gb",
    "incomplete_units": "panel_mode",
    "last_treated_controls": "reference_group",
    "adjustment": "est_method",
    "propensity_trim": "trim_level",
    "multiplier_bootstrap_reps": "bootstrap_reps",
    "multiplier_bootstrap_seed": "bootstrap_seed",
    "multiplier_bootstrap_weights": "bootstrap_weights",
    "effective_sample_warning": "ess_warn_threshold",
    "min_effective_sample": "ess_suppress_threshold",
    "covariate_balance": "balance",
    "balance_distributions": "balance_ecdf",
    "residual_averaging": "avgeffectsby",
    "leave_out": "leaveout",
    "pretrend_leads": "pretrends",
    "event_study": "specification",
    "treat_sample_as_fixed": "inference",
    "zero_tolerance": "zero_tol",
    "fixed_effect_solver": "fe_solver",
}

# public option name -> {public value: internal value}
VALUES = {
    "control_group": {"never_treated": "nevertreated", "not_yet_treated": "notyettreated"},
    "incomplete_units": {"error": "balanced", "drop": "complete", "keep": "unbalanced"},
    "last_treated_controls": {
        "if_needed": "auto",
        "never": "nevertreated",
        "always": "lasttreated",
    },
    "adjustment": {"regression": "reg", "ipw": "ipw", "doubly_robust": "dr"},
    "event_study": {True: "dynamic", False: "static"},
    "treat_sample_as_fixed": {False: "unconditional", True: "conditional"},
}


def internal(options):
    """Translate public keyword options to the internal estimator's keywords."""
    out = {}
    for name, value in options.items():
        if name in VALUES:
            choices = VALUES[name]
            if isinstance(next(iter(choices)), bool):
                if not isinstance(value, bool):
                    raise ValueError(f"{name} must be True or False")
            elif not isinstance(value, str) or value not in choices:
                listed = ", ".join(repr(c) for c in choices)
                raise ValueError(f"{name} must be one of {listed}; got {value!r}")
            value = choices[value]
        out[NAMES.get(name, name)] = value
    return out
