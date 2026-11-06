"""CS comparison support, warning policy and explicit coefficient suppression."""

import warnings
from numbers import Real

import numpy as np
import polars as pl


class LowEffectiveSampleWarning(UserWarning):
    """A CS comparison relies on a low effective treated/control sample."""


def threshold(value, name, *, optional=True):
    if value is None and optional:
        return value
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a finite nonnegative number" + (" or None" if optional else "")
        )
    return float(value)


def finish_result(result, warn_threshold, suppress_threshold):
    if result.estimator != "cs":
        return result
    rows = []
    cell_index = {cell: j for j, cell in enumerate(result.cells)}
    for comparison in result.diagnostics["comparisons"]:
        indices = comparison.get("cells")
        if indices is None:
            indices = [cell_index[(comparison["cohort"], comparison["time"])]]
        for index in indices:
            for group, key in (("treated", "effective_treated"), ("control", "effective_controls")):
                values = np.atleast_1d(comparison[key])
                labels = [group] if len(values) == 1 else [group + "_pre", group + "_post"]
                for label, value in zip(labels, values, strict=True):
                    if not np.isfinite(value) or value <= 0:
                        raise ValueError("CS comparison has undefined effective sample size")
                    rows.append(
                        dict(
                            coefficient=result.coefnames[index],
                            cell=result.coefnames[index],
                            cohort=result.cells[index][0],
                            time=result.cells[index][1],
                            stratum=label,
                            effective_n=float(value),
                        )
                    )
    result._ess_table = pl.DataFrame(rows)
    result.diagnostics["ess_warn_threshold"] = warn_threshold
    result.diagnostics["ess_suppress_threshold"] = suppress_threshold
    result.diagnostics["ess_definition"] = (
        "Kish weight ESS within each required treated/control comparison stratum; not regression-information or influence-score ESS"
    )
    minima = result._minimum_ess()
    bad = minima < (warn_threshold or 0)
    if bad.any():
        message = (
            f"Low effective sample size: {int(bad.sum())} CS coefficients have a required "
            f"treated/control stratum below {warn_threshold:g} (minimum {minima.min():.3g}). "
            "Inspect effective_sample_size(); suppress(min_effective_sample=...) masks coefficients without changing aggregation weights."
        )
        result.diagnostics["warnings"] = [*result.diagnostics.get("warnings", []), message]
        warnings.warn(message, LowEffectiveSampleWarning, stacklevel=3)
    if suppress_threshold is not None:
        return result.suppress(min_effective_sample=suppress_threshold)
    return result
