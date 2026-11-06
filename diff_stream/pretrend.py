"""Clean pre-treatment selection and transparent joint Wald tests."""

import numpy as np
import polars as pl

from .aggregation import integer


def pretrend_test(result, *, level="auto", horizons=None):
    if result.estimator == "bjs":
        return result.placebos().pretrend_test(level=level, horizons=horizons)
    if level not in ("auto", "cell", "dynamic"):
        raise ValueError("level must be 'auto', 'cell' or 'dynamic'")
    A = result.diagnostics.get("anticipation", 0)
    if result.cells:
        level = "cell" if level == "auto" else level
        events = np.array([t - g for g, t in result.cells])
    else:
        if result.diagnostics.get("aggregation") != "dynamic" or level == "cell":
            raise ValueError("pretrend_test requires group-time estimates or a dynamic aggregate")
        level = "dynamic"
        events = np.array([int(name.split("::")[-1]) for name in result.coefnames])
    clean = events < -A
    requested = None
    if horizons is not None:
        try:
            requested = list(horizons)
        except TypeError as exc:
            raise ValueError(
                "horizons must contain clean pre-treatment integer event times"
            ) from exc
        if not requested or not all(integer(h) and h < -A for h in requested):
            raise ValueError(
                "horizons must contain clean pre-treatment integer event times < -anticipation"
            )
        requested = sorted(set(int(h) for h in requested))
        missing = set(requested) - set(events[clean].tolist())
        if missing:
            raise ValueError(f"requested clean horizons are not estimated: {sorted(missing)}")
        clean &= np.isin(events, requested)
    if not clean.any():
        raise ValueError(
            "no estimated clean pre-treatment coefficients; normalized references and anticipation periods are excluded"
        )
    if result.cells and level == "dynamic":
        result = result.aggregate(horizons=sorted(set(events[clean].tolist())))
        events = np.array([int(name.split("::")[-1]) for name in result.coefnames])
        clean = np.ones(len(events), dtype=bool)
    ix = np.flatnonzero(clean)
    names = [result.coefnames[j] for j in ix]
    V = result.vcov[np.ix_(ix, ix)]
    sd = np.sqrt(np.maximum(np.diag(V), 0))
    if not np.isfinite(V).all() or np.any(sd <= 0):
        raise ValueError(
            "pre-trend covariance has zero or non-finite variance; select estimable horizons"
        )
    correlation = V / np.outer(sd, sd)
    if np.linalg.eigvalsh((correlation + correlation.T) / 2)[0] <= 1e-10:
        raise ValueError(
            "pre-trend covariance is singular or ill-conditioned; select fewer cells/horizons or use level='dynamic'"
        )
    # Rescaling restrictions leaves the null unchanged and lets the inherited
    # Wald solver work on the correlation scale. Otherwise its pseudoinverse
    # can discard a small-variance restriction solely because of units/scales.
    R = np.zeros((len(ix), len(result.beta)))
    R[np.arange(len(ix)), ix] = 1 / sd
    test = result.wald_test(R=R, distribution="chi2")
    if result.cells:
        cells = [dict(cohort=int(result.cells[j][0]), time=int(result.cells[j][1])) for j in ix]
    else:
        table = (
            result.support() if result.estimator == "bjs_placebo" else result.contributions()
        ).filter(pl.col("aggregate").is_in(names))
        cells = table.select("cohort", "time").unique().sort(["cohort", "time"]).to_dicts()
    return dict(
        **test,
        level=level,
        coefficients=names,
        horizons=sorted(set(int(h) for h in events[ix])),
        cells=cells,
        n_restrictions=len(ix),
        anticipation=A,
        reference_event_time=result.diagnostics.get("reference_event_time", -A - 1),
        null="all selected clean pre-treatment effects equal zero",
        base_period=result.diagnostics.get("base_period"),
        **(
            dict(
                fitted_horizons=result.diagnostics["fitted_horizons"],
                sample="separate clean-sample lead regression",
            )
            if result.estimator == "bjs_placebo"
            else {}
        ),
    )
