"""Covariates, fixed effects and the experimental warning of did_continuous.

The formula's covariates and one absorbed fixed-effect term are parsed here, and the
fixed effect's (or a cluster's) distinct values are numbered so that they can be
staged as a design column. The fit itself is ``continuous_general``.
"""

import re

import polars as pl
from hdfe_stream.core.report import ExperimentalWarning as _Experimental

FE = "__continuous_fe"
CLUSTER = "__continuous_cluster"


class ExperimentalWarning(_Experimental):
    """A diff-stream fit uses a method its authors have not developed.

    It subclasses ``hdfe_stream.ExperimentalWarning``: filter this class to silence
    diff-stream's experimental warnings, or that one for every package built on
    hdfe-stream.
    """


DOCS = "https://leetucker.github.io/diff-stream/estimators/continuous.html#experimental-extensions"


def experimental_message(parts):
    """The warning for a fit that uses the extensions ``parts`` (a list of phrases)."""
    listed = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return (
        f"did_continuous with {listed} goes beyond what Callaway, Goodman-Bacon and "
        "Sant'Anna developed, and the estimates rely on further assumptions; see "
        f"{DOCS}"
    )


def parse(fml, dose, panel):
    """The outcome, the covariates' formula (or None) and the fixed effect's columns (or None)."""
    if not isinstance(fml, str) or fml.count("~") != 1:
        raise ValueError("use one outcome formula: 'y ~ 1', 'y ~ x' or 'y ~ x | region'")
    y, rhs = [s.strip() for s in fml.split("~")]
    if not y.isidentifier():
        raise ValueError("the outcome must be a column name")
    parts = [s.strip() for s in rhs.split("|")]
    if len(parts) > 2:
        raise ValueError("give at most one '|' (the fixed effect): 'y ~ x | region'")
    covariates = parts[0] if parts[0] not in ("1", "") else None
    if covariates is not None and dose in re.findall(r"[A-Za-z_]\w*", covariates):
        raise ValueError("the dose enters through dose=, not as a covariate")
    fe = None
    if len(parts) == 2:
        term = parts[1]
        if "+" in term:
            raise ValueError(
                "did_continuous absorbs one fixed-effect term (which may be an "
                "interaction, region^industry); give the others as covariates"
            )
        fe = [c.strip() for c in re.split(r"[\^:]", term)]
        if not all(c.isidentifier() for c in fe) or len(set(fe)) != len(fe):
            raise ValueError(f"cannot read the fixed effect {term!r}")
        unit, time = panel
        if unit in fe:
            raise ValueError(
                f"{unit} (the unit) cannot be a fixed effect: unit effects are already "
                "differenced out"
            )
        if time in fe:
            raise ValueError(
                f"{time} (the period) cannot be a fixed effect: each cohort-period has "
                "its own regression, so a fixed effect already differs by period "
                "(| region is region-by-period)"
            )
        if dose in fe or y in fe:
            raise ValueError("the fixed effect cannot be the dose or the outcome")
    return y, covariates, fe


def join_levels(src, columns, name=FE, what="fixed-effect"):
    """The source with each distinct value of ``columns`` numbered as ``name``, and how many."""
    missing = set(columns) - set(src.collect_schema())
    if missing:
        raise ValueError(f"unknown {what} columns: {sorted(missing)}")
    levels = src.select(columns).unique().collect(engine="streaming")
    if levels.null_count().sum_horizontal().item():
        raise ValueError(
            f"the {what.replace('-', ' ')} has missing values; fill or drop them first"
        )
    levels = levels.sort(columns).with_row_index(name).with_columns(pl.col(name).cast(pl.Float64))
    return src.join(levels.lazy(), on=columns, how="left"), len(levels)
