"""Parameter descriptions shared by the public functions' docstrings."""

DATA = """\
    data : polars.DataFrame, polars.LazyFrame, str or Path
        The panel, or a Parquet path or glob. Rows are read in streaming
        batches; the data need not fit in memory.
    panel : tuple of str
        The unit and time columns, ``(unit, time)``. Periods are positive
        integers, evenly spaced (every year, every second year, ...). Event
        times, ``anticipation`` and ``pretrend_leads`` count periods; results
        label cohorts and periods as in the data.
    cohort, treatment : str
        Exactly one. ``cohort``: each unit's first treated period, constant
        within the unit; 0 or null marks a never-treated unit. ``treatment``: a
        0/1 column, absorbing (once 1, a unit stays treated), from which each
        unit's first treated period is taken."""

ANTICIPATION = """\
    anticipation : int, default 0
        Periods before treatment in which units may already respond. Clean
        (untreated) observations are those before ``cohort - anticipation``;
        event times stay relative to actual treatment."""

WEIGHTS = """\
    weights : str, optional
        A column of sampling weights: positive, finite, and constant within
        each unit unless ``allow_time_varying_weights=True``. They enter the
        estimates, aggregation and clustered standard errors.
    allow_time_varying_weights : bool, default False
        Accept weights that change over time within a unit."""

INCOMPLETE_UNITS = """\
    incomplete_units : {"error", "drop", "keep"}, default "error"
        What to do with units missing some periods: refuse the data, drop
        those units, or keep their observed rows (an unbalanced panel)."""

LAST_TREATED = """\
    last_treated_controls : {"if_needed", "never", "always"}, default "if_needed"
        When to use the last-treated cohort as a comparison group, keeping
        only periods before it is treated. "if_needed" does so only when there
        are no never-treated units (with a ``ReferenceWindowWarning``);
        "never" requires never-treated units; "always" uses it even when
        never-treated units exist (they remain controls too)."""

FIXED_EFFECT_SOLVER = """\
    fixed_effect_solver : {"auto", "dense", "iterative"}, default "auto"
        How additional fixed effects (after the bar in ``fml``) are
        estimated. "dense" adds a dummy for each level to the regression,
        which is exact and fastest with few levels, and limited to 256 columns in
        all. "iterative" absorbs them by alternating projections (hdfe-stream),
        with a time that does not grow with the number of levels and no limit
        on it, such as county-by-year. "auto" uses dense for up to 40 columns
        (levels and covariates) and iterative above that. Both give the same
        estimates and standard errors."""

VCOV = """\
    vcov : {"CRV0", "CRV1"}, default "CRV0"
        Standard errors clustered by unit. CRV1 multiplies the variance by
        N/(N-1)."""

RESOURCES = """\
    n_threads : int, optional
        Threads for numba and BLAS during the fit (stream-limits'
        ``thread_scope``). Default: every core the job has. Polars' threads
        are fixed when polars is imported: set ``STREAM_LIMITS_MAX_THREADS``
        in the job script.
    max_memory_gb, memory_reserve : float, optional
        The memory limit and the share of it kept in reserve for allocations
        the estimates do not model (stream-limits' ``MemoryBudget``; default:
        the limits it detects, including a scheduler job's, and a 15%
        reserve). Fits that would exceed the limit stop with ``MemoryError``
        before allocating.
    workdir : str or Path, optional
        Where temporary files are written. Default: the system temporary
        directory.
    profile : bool, default False
        Return a ``Profile`` (hdfe-stream's) of the time, memory and disk the
        fit needs, without fitting.
    batch_units : int, optional
        Advanced: units per numerical batch.
    preprocess_mb : float, optional
        Advanced: the working memory for sorting and validating the data, in
        MB. Default: a quarter of the memory left under ``max_memory_gb``, so a
        panel that fits is sorted in one piece; larger panels are split into
        whole-unit pieces on disk. A smaller value uses less memory and more
        time."""


def document(function):
    """Fill the shared parameter blocks into a public function's docstring."""
    blocks = {
        name: value
        for name, value in globals().items()
        if name.isupper() and isinstance(value, str)
    }
    text = function.__doc__
    for name, value in blocks.items():
        text = text.replace("    {" + name + "}\n", value + "\n")
    function.__doc__ = text
    return function
