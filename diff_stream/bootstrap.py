"""Ordinary unit-cluster pairs bootstrap with serial or parallel full refits."""

import json
import pickle
import sys
import warnings
from concurrent.futures import as_completed
from contextlib import closing, nullcontext
from dataclasses import dataclass, field
from inspect import signature
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import pyarrow as pa
from scipy.stats import norm
from stream_limits import MemoryBudget, detect_cpus, thread_scope

from . import _options
from ._core.partition import preprocess_budget_mb
from ._core.process import ProcessBatch
from ._core.resample import DiskGroupResampler
from .api import _fit, did_bjs, did_callaway_santanna, did_sun_abraham
from .etwfe import did_extended_twfe
from .gardner import did_two_stage
from .store import inspect, source


class BootstrapFailureWarning(UserWarning):
    """Some ordinary resamples failed; inference requires explicit acknowledgment."""


@dataclass(kw_only=True)
class RefitBootstrapResult:
    """Point estimates, ordinary refit draws and pointwise bootstrap inference.

    No source or fit scratch survives. Failed attempts remain in the ledger and
    are never redrawn. Inference on successful draws requires allow_incomplete
    when any attempt failed; such inference is conditional on successful fits.
    """

    fml: str
    estimator: str
    coefnames: list
    beta: np.ndarray
    diagnostics: dict
    _draws: np.ndarray = field(repr=False)
    _failures: list = field(repr=False)
    _warnings: list = field(repr=False)

    @property
    def successful_reps(self):
        return int(np.isfinite(self._draws).all(axis=1).sum())

    @property
    def se(self):
        """Sample SD of refitted estimates; refuses incomplete bootstrap runs."""
        return self.standard_errors()

    def _valid(self, allow_incomplete):
        if self._failures and not allow_incomplete:
            raise ValueError(
                "some refits failed; inspect failures() and explicitly set allow_incomplete=True for inference conditional on success"
            )
        data = self._draws[np.isfinite(self._draws).all(axis=1)]
        if len(data) < 2:
            raise ValueError("at least two successful refits are required for bootstrap inference")
        return data

    def standard_errors(self, *, allow_incomplete=False):
        return np.std(self._valid(allow_incomplete), axis=0, ddof=1)

    def confint(self, alpha=0.05, *, method="percentile", allow_incomplete=False):
        """Pointwise percentile/type-7, basic, or uncorrected normal intervals.

        Normal intervals are point estimate +/- normal critical value times
        bootstrap SD, without boot.ci's normal bias correction. No simultaneous,
        studentized or BCa intervals are supplied.
        """
        if isinstance(alpha, bool) or not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must be between zero and one")
        if method not in ("percentile", "basic", "normal"):
            raise ValueError("method must be percentile, basic or normal")
        draws = self._valid(allow_incomplete)
        if method == "normal":
            spread = norm.ppf(1 - alpha / 2) * np.std(draws, axis=0, ddof=1)
            lower, upper = self.beta - spread, self.beta + spread
        else:
            low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0, method="linear")
            lower, upper = (
                (low, high)
                if method == "percentile"
                else (2 * self.beta - high, 2 * self.beta - low)
            )
        return pl.DataFrame(
            {"Coefficient": self.coefnames, "Estimate": self.beta, "Lower": lower, "Upper": upper}
        )

    def tidy(self, alpha=0.05, *, method="percentile", allow_incomplete=False):
        draws = self._valid(allow_incomplete)
        return self.confint(alpha, method=method, allow_incomplete=allow_incomplete).with_columns(
            pl.Series("Std. Error", np.std(draws, axis=0, ddof=1)),
            pl.Series("Bootstrap Mean", draws.mean(axis=0)),
            pl.Series("Bias", draws.mean(axis=0) - self.beta),
            pl.lit(len(draws)).alias("Successful Reps"),
        )

    def draws(self):
        """Copy attempted estimates; failed rows are NaN for every coefficient."""
        return pl.DataFrame(
            {
                "draw": np.arange(len(self._draws)),
                "successful": np.isfinite(self._draws).all(axis=1),
                **{n: self._draws[:, j].copy() for j, n in enumerate(self.coefnames)},
            }
        )

    def failures(self):
        return pl.DataFrame(
            self._failures, schema={"draw": pl.Int64, "error_type": pl.String, "reason": pl.String}
        )

    def refit_warnings(self):
        return pl.DataFrame(
            self._warnings, schema={"draw": pl.Int64, "category": pl.String, "message": pl.String}
        )

    def summary_dict(self):
        return dict(
            fml=self.fml,
            estimator=self.estimator,
            coefficients=dict(zip(self.coefnames, self.beta.tolist(), strict=True)),
            requested_reps=len(self._draws),
            successful_reps=self.successful_reps,
            failures=self.failures().to_dicts(),
            refit_warnings=self.refit_warnings().to_dicts(),
            diagnostics=self.diagnostics,
        )

    def summary(self):
        """Run/sample/failure diagnostics; inference is available through tidy()."""
        return self.summary_dict()

    def summary_json(self, path=None):
        content = json.dumps(self.summary_dict(), indent=2, allow_nan=False, default=str)
        if path is not None:
            Path(path).write_text(content + "\n")
        return content


def _target(result, aggregate, aggregate_options, coefficients):
    target = result.aggregate(aggregate, **aggregate_options) if aggregate is not None else result
    names = list(target.coefnames) if coefficients is None else list(coefficients)
    if not names or len(set(names)) != len(names) or not set(names) <= set(target.coefnames):
        raise ValueError("requested coefficient names are missing, duplicated or empty")
    values = target.beta[[target.coefnames.index(n) for n in names]].copy()
    if not np.isfinite(values).all():
        raise ValueError("requested coefficients are nonfinite or suppressed")
    return names, values


def _replicate_formula(fml, unit):
    if "|" not in fml:
        return fml
    regression, fe = fml.split("|")
    terms = [x.strip() for x in fe.split("+") if x.strip() != unit]
    return regression if not terms else regression + "|" + "+".join(terms)


def _refit_attempt(task):
    """Identical fit/sample/failure policy for in-process and subprocess draws."""
    result, values, failure, captured = None, None, None, []
    b = task["draw"]
    try:
        if task["dates"] != task["expected_dates"]:
            raise ValueError("resample loses dates from the fixed analysis window")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = _fit(
                    task["fml"],
                    task["path"],
                    panel=task["panel"],
                    cohort=task["cohort"],
                    treatment=task["treatment"],
                    estimator=task["estimator"],
                    workdir=task["workdir"],
                    **task["options"],
                )
                if result.n_units != task["n_units"] or result.n_obs != task["draw_rows"]:
                    raise ValueError("refit changed the fixed sampled population")
                _, values = _target(
                    result, task["aggregate"], task["aggregate_options"], task["names"]
                )
            finally:
                captured.extend(
                    dict(draw=b, category=w.category.__name__, message=str(w.message)[:2000])
                    for w in caught
                )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        failure = dict(draw=b, error_type=type(exc).__name__, reason=str(exc)[:2000])
    finally:
        if result is not None:
            result.cleanup()
        pa.default_memory_pool().release_unused()
    return values, failure, captured


def _parallel_refits(tasks, workers, threads, memory_gb):
    """Yield completed bounded-batch results; stop children before scratch cleanup."""
    with ProcessBatch(workers) as batch:
        pending = {}
        for task in tasks:
            root = Path(task["workdir"]).parent
            job, output, log = root / "job.pkl", root / "result.pkl", root / "worker.log"
            task = dict(task, options=dict(task["options"], threads=threads, memory_gb=memory_gb))
            with job.open("wb") as f:
                pickle.dump(task, f, protocol=pickle.HIGHEST_PROTOCOL)
            command = [
                sys.executable,
                str(Path(__file__).with_name("_bootstrap_worker.py")),
                str(job),
                str(output),
                str(threads),
                repr(memory_gb),
            ]
            pending[batch.submit(command, log)] = (task["draw"], output, log)
        for future in as_completed(pending):
            b, output, log = pending[future]
            returncode = future.result()
            if returncode or not output.exists():
                with log.open("rb") as f:
                    f.seek(max(0, log.stat().st_size - 4000))
                    detail = f.read().decode(errors="replace")
                raise RuntimeError(f"bootstrap worker for draw {b} exited {returncode}: {detail}")
            with output.open("rb") as f:
                payload = pickle.load(f)  # Private output of our own worker.
            if "error" in payload:
                error = payload["error"]
                # Keep the exception type (notably MemoryError) and remote trace.
                raise error from RuntimeError(payload["traceback"])
            yield b, payload["result"], payload["resources"]


def _refit_bootstrap(
    fml,
    data,
    *,
    panel,
    cohort=None,
    treatment=None,
    estimator,
    public_estimator,
    public_options,
    reps=199,
    seed=None,
    aggregate=None,
    aggregate_options=None,
    coefficients=None,
    workers=1,
    threads=None,
    memory_gb=None,
    memory_reserve=None,
    preprocess_mb=None,
    batch_units=256,
    workdir=None,
    **fit_options,
):
    """refit_bootstrap with internal estimator codes and option names."""
    for name, value, minimum in [
        ("reps", reps, 2),
        ("batch_units", batch_units, 1),
        ("n_workers", workers, 1),
    ]:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    if threads is not None and (
        isinstance(threads, bool) or not isinstance(threads, int) or threads < 1
    ):
        raise ValueError("n_threads must be a positive integer")
    active_workers = min(workers, reps)
    job_threads = threads if threads is not None else detect_cpus()
    if active_workers > 1 and active_workers > job_threads:
        raise ValueError(
            "n_workers cannot exceed the total thread budget; increase n_threads or reduce n_workers"
        )
    worker_threads = max(1, job_threads // active_workers)
    if aggregate not in (None, "event_time", "overall", "cohort", "calendar"):
        raise ValueError("aggregate must be 'event_time', 'overall', 'cohort', 'calendar' or None")
    if estimator == "gardner" and aggregate is not None:
        raise ValueError(
            "did_two_stage coefficients are reported directly: use aggregate=None "
            "(and event_study=False for a single ATT)"
        )
    aggregate_options = {} if aggregate_options is None else dict(aggregate_options)
    if aggregate is None and aggregate_options:
        raise ValueError("aggregate_options requires an aggregate target")
    if isinstance(coefficients, str):
        raise ValueError("coefficients must be a sequence of exact coefficient names")
    if coefficients is not None:
        coefficients = list(coefficients)
    if (
        fit_options.get("profile", False)
        or fit_options.get("bootstrap_reps", 0)
        or fit_options.get("bootstrap_seed") is not None
    ):
        raise ValueError(
            "refit_bootstrap fits the model itself: profile and the multiplier bootstrap "
            "options cannot be passed to it"
        )
    budget = MemoryBudget(memory_gb, memory_reserve)
    # Refits size their own buffers from their own budgets (workers have less).
    # The draw plan follows the sampler's unit order, which depends on its
    # partitions: a fixed default keeps seeded draws the same on every machine.
    refit_preprocess_mb = preprocess_mb
    preprocess_mb = preprocess_budget_mb(64 if preprocess_mb is None else preprocess_mb, budget)
    if not budget.fits((preprocess_mb * 1e6 + 96e6) / 1e9):
        raise MemoryError("bootstrap preprocessing buffers exceed job headroom")
    weights = fit_options.get("weights")
    _, y, design, absorbed, _ = inspect(
        fml,
        data,
        panel,
        cohort,
        treatment=treatment,
        allow_covariates=True,
        weights=weights,
        allow_fe=estimator in ("sunab", "bjs", "gardner"),
        fe_solver=fit_options.get("fe_solver", "auto"),
    )
    required = list(
        dict.fromkeys(
            [
                *panel,
                cohort if cohort is not None else treatment,
                y,
                *([] if weights is None else [weights]),
                *([] if design is None else design.required),
                *[c for _, written in absorbed or [] for c in written.split("^")],
            ]
        )
    )
    src = source(data)
    used = set(src.collect_schema().names())
    index_column = "__diff_source_unit"
    while index_column in used:
        index_column += "_"
    used.add(index_column)
    draw_column = "__diff_boot_unit"
    while draw_column in used:
        draw_column += "_"
    options = dict(
        fit_options,
        threads=threads,
        memory_gb=memory_gb,
        memory_reserve=memory_reserve,
        preprocess_mb=refit_preprocess_mb,
        batch_units=batch_units,
    )
    seed_sequence = np.random.SeedSequence(seed)
    rng = np.random.default_rng(seed_sequence)
    if workdir is not None:
        Path(workdir).mkdir(parents=True, exist_ok=True)
    with thread_scope(threads), TemporaryDirectory(prefix="diff-stream-refit-", dir=workdir) as run:
        run = Path(run)
        snapshot = run / "source.parquet"
        src.select(required).sink_parquet(snapshot, engine="streaming", row_group_size=32768)
        original_dates = (
            pl.scan_parquet(snapshot)
            .select(
                pl.col(panel[1]).min().alias("lo"),
                pl.col(panel[1]).max().alias("hi"),
                pl.len().alias("rows"),
            )
            .collect(engine="streaming")
            .row(0)
        )
        original = _fit(
            fml,
            snapshot,
            panel=panel,
            cohort=cohort,
            treatment=treatment,
            estimator=estimator,
            workdir=run / "fit",
            **options,
        )
        try:
            names, beta = _target(original, aggregate, aggregate_options, coefficients)
            n_units, n_obs = original.n_units, original.n_obs
            diag = dict(original.diagnostics)
        finally:
            original.cleanup()
        del original
        step = diag.get("period_step", 1)
        expected_dates = list(range(diag["time_min"], diag["time_max"] + 1, step))
        cutoff = diag.get("reference_cutoff")
        mode = options.get("panel_mode", "balanced")

        def select(frame):
            if mode == "complete":
                ids = frame[panel[0]].to_numpy()
                starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
                lengths = np.diff(np.r_[starts, len(frame)])
                frame = frame.filter(
                    pl.Series(
                        np.repeat(
                            lengths == (original_dates[1] - original_dates[0]) // step + 1, lengths
                        )
                    )
                )
            if cutoff is not None:
                frame = frame.filter(pl.col(panel[1]) < cutoff)
            return frame

        stage = run / "units"
        stage.mkdir()
        sampler = DiskGroupResampler(
            snapshot,
            stage,
            budget,
            preprocess_mb,
            key=panel[0],
            sort_columns=list(panel),
            select=select,
            index_column=index_column,
            draw_column=draw_column,
        )
        if (sampler.n_units, sampler.n_rows) != (n_units, n_obs):
            raise RuntimeError("bootstrap staging differs from the original fitted sample")
        numeric = 8 * 4 * reps * len(names) + 48e6
        pa.default_memory_pool().release_unused()
        if not budget.fits(numeric / 1e9):
            raise MemoryError(
                "bootstrap draw matrix exceeds job headroom; reduce reps/coefficients"
            )
        draws = np.full((reps, len(names)), np.nan)
        failures, captured, hashes = [], [], []
        options.update(
            reference_group="nevertreated",
            panel_mode="unbalanced" if mode == "unbalanced" else "balanced",
        )
        replicate_fml = _replicate_formula(fml, panel[0])
        batches, worker_resources = [], []
        for start in range(0, reps, active_workers):
            with TemporaryDirectory(prefix="batch-", dir=run) as batch_root:
                tasks = []
                for b in range(start, min(start + active_workers, reps)):
                    attempt = Path(batch_root) / f"draw-{b}"
                    attempt.mkdir()
                    path, dates, draw_rows, identity = sampler.materialize(
                        rng, attempt / "data", time_column=panel[1]
                    )
                    hashes.append(identity)
                    tasks.append(
                        dict(
                            draw=b,
                            fml=replicate_fml,
                            path=path,
                            panel=(draw_column, panel[1]),
                            cohort=cohort,
                            treatment=treatment,
                            estimator=estimator,
                            workdir=attempt / "fit",
                            options=options,
                            dates=dates,
                            expected_dates=expected_dates,
                            n_units=n_units,
                            draw_rows=draw_rows,
                            aggregate=aggregate,
                            aggregate_options=aggregate_options,
                            names=names,
                        )
                    )
                if active_workers == 1:
                    completed = [(start, _refit_attempt(tasks[0]), None)]
                else:
                    pa.default_memory_pool().release_unused()
                    # Parent staging is idle while children fit. Reserve its
                    # current RSS plus the job reserve; workers share only the
                    # remainder and also retain their own MemoryBudget reserve.
                    worker_memory = budget.headroom_gb() / active_workers
                    minimum = 0.35 + (refit_preprocess_mb or 64) / 1000 + 0.096
                    if worker_memory * (1 - budget.reserve) < minimum:
                        raise MemoryError(
                            "insufficient whole-job memory for parallel refits including worker runtimes; "
                            "reduce n_workers/preprocess_mb or increase max_memory_gb"
                        )
                    batches.append(
                        dict(first_draw=start, refits=len(tasks), worker_memory_gb=worker_memory)
                    )
                    completed = _parallel_refits(
                        tasks, active_workers, worker_threads, worker_memory
                    )
                # Closing the generator stops/reaps workers before the batch
                # temporary directory is removed, including on interruption.
                with (
                    closing(completed) if active_workers > 1 else nullcontext(completed) as results
                ):
                    for b, (values, failure, notes), resources in results:
                        if values is not None:
                            draws[b] = values
                        if failure is not None:
                            failures.append(failure)
                        captured.extend(notes)
                        if resources is not None:
                            worker_resources.append(dict(draw=b, **resources))
        failures.sort(key=lambda x: x["draw"])
        captured.sort(key=lambda x: x["draw"])
        worker_resources.sort(key=lambda x: x["draw"])
        diagnostics = dict(
            workers_requested=workers,
            workers=active_workers,
            job_threads=job_threads,
            worker_threads=worker_threads if active_workers > 1 else threads,
            parallel_batches=batches,
            worker_resources=worker_resources,
            seed=seed,
            seed_entropy=seed_sequence.entropy,
            bit_generator=type(rng.bit_generator).__name__,
            reps=reps,
            unit_count=n_units,
            original_rows=n_obs,
            analysis_rows=n_obs,
            source_rows=original_dates[2],
            source_panel=list(panel),
            cohort=cohort,
            fit_options=public_options,
            aggregate=aggregate,
            aggregate_options=aggregate_options,
            incomplete_units={"balanced": "error", "complete": "drop", "unbalanced": "keep"}[mode],
            fixed_reference=diag.get("reference"),
            fixed_reference_cohort=diag.get("reference_cohort"),
            reference_cutoff=cutoff,
            analysis_time_min=diag["time_min"],
            analysis_time_max=diag["time_max"],
            replicate_reference="never_treated",
            sampling="ordinary uniform unit pairs with replacement; fresh ID for each copy",
            inference="pointwise; complete successful rows only; failed attempts never redrawn",
            unit_order="hash leaf traversal, sorted source IDs within leaf; fixed input/layout reproducibility",
            unit_order_sha256=sampler.identity.hexdigest(),
            draw_plan_sha256=hashes,
            source_columns=required,
            partitions=sampler.partition_count,
            preprocess_mb=preprocess_mb,
            batch_units=batch_units,
            added_generic_infrastructure="bounded whole-group resampling adapter in diff_stream._core.resample",
        )
    out = RefitBootstrapResult(
        fml=fml,
        estimator=public_estimator,
        coefnames=names,
        beta=beta,
        diagnostics=diagnostics,
        _draws=draws,
        _failures=failures,
        _warnings=captured,
    )
    if failures:
        warnings.warn(
            f"{len(failures)} of {reps} unit-cluster refits failed. Inspect failures(); incomplete-run inference requires allow_incomplete=True and is conditional on successful fits.",
            BootstrapFailureWarning,
            stacklevel=2,
        )
    return out


# refit_bootstrap() sets these itself, for the original fit and every refit.
_CONTROLLED = {
    "fml",
    "data",
    "panel",
    "cohort",
    "n_threads",
    "max_memory_gb",
    "memory_reserve",
    "workdir",
    "profile",
    "batch_units",
    "preprocess_mb",
    "multiplier_bootstrap_reps",
    "multiplier_bootstrap_seed",
    "multiplier_bootstrap_weights",
}


def refit_bootstrap(
    fml,
    data,
    *,
    panel,
    cohort=None,
    treatment=None,
    estimator,
    reps=199,
    seed=None,
    aggregate=None,
    aggregate_options=None,
    coefficients=None,
    n_workers=1,
    n_threads=None,
    max_memory_gb=None,
    memory_reserve=None,
    workdir=None,
    batch_units=256,
    preprocess_mb=None,
    **fit_options,
):
    """Bootstrap by resampling units and refitting the estimator.

    Draws ``reps`` samples of the units with replacement (each copy of a unit
    gets its own ID) and refits the whole estimator on each, including its
    propensity scores, trimming and aggregation weights. Reports bootstrap
    standard errors and percentile, basic or normal intervals for each
    coefficient. The sample and comparison group chosen by the original fit
    (incomplete units dropped, a last-treated reference window) are held
    fixed across draws. This is slower than analytic standard errors, by
    about ``reps`` fits; for CS, the multiplier bootstrap
    (``multiplier_bootstrap_reps``) is far cheaper.

    Parameters
    ----------
    fml, data, panel, cohort
        As for the estimator.
    estimator : function
        One of the estimator functions, e.g. ``did_bjs``.
    reps : int, default 199
        Bootstrap draws.
    seed : int, optional
        Seed for the draws. With the same data layout, the same seed gives
        the same draws.
    aggregate : {"event_time", "overall", "cohort", "calendar"}, optional
        Bootstrap an aggregate (``result.aggregate(aggregate,
        **aggregate_options)``) instead of the fitted coefficients. Not
        available for ``did_two_stage``, whose coefficients are reported
        directly.
    aggregate_options : dict, optional
        Options for ``aggregate()``, such as ``{"balanced_horizon": 2}``.
    coefficients : list of str, optional
        Bootstrap only these coefficients (exact names).
    n_workers : int, default 1
        Refits run in parallel, each in a fresh process. ``n_threads`` and
        ``max_memory_gb`` are shared out among them.
    n_threads, max_memory_gb, memory_reserve, workdir, batch_units, preprocess_mb
        As for the estimator; they cover the whole job.
    **fit_options
        The estimator's other options, e.g. ``adjustment="ipw"``.

    Returns
    -------
    RefitBootstrapResult
        ``tidy()``, ``confint()`` and ``standard_errors()``. Failed draws are
        recorded in ``failures()``, never redrawn; inference then requires
        ``allow_incomplete=True``.
    """
    code = _ESTIMATOR_CODES.get(estimator)
    if code is None:
        raise TypeError(
            "estimator must be one of diff-stream's estimator functions, such as did_bjs"
        )
    allowed = set(signature(estimator).parameters) - _CONTROLLED
    unknown = sorted(set(fit_options) - allowed)
    if any(name.startswith("multiplier_bootstrap") or name == "profile" for name in unknown):
        raise ValueError(
            "refit_bootstrap fits the model itself: profile and the multiplier bootstrap "
            "options cannot be passed to it"
        )
    if unknown:
        raise TypeError(f"{estimator.__name__} has no option {', '.join(unknown)}")
    return _refit_bootstrap(
        fml,
        data,
        panel=panel,
        cohort=cohort,
        treatment=treatment,
        estimator=code,
        public_estimator=estimator.__name__,
        public_options=dict(fit_options),
        reps=reps,
        seed=seed,
        aggregate=aggregate,
        aggregate_options=aggregate_options,
        coefficients=coefficients,
        workers=n_workers,
        threads=n_threads,
        memory_gb=max_memory_gb,
        memory_reserve=memory_reserve,
        workdir=workdir,
        batch_units=batch_units,
        preprocess_mb=preprocess_mb,
        **_options.internal(fit_options),
    )


_ESTIMATOR_CODES = {
    did_callaway_santanna: "cs",
    did_sun_abraham: "sunab",
    did_bjs: "bjs",
    did_two_stage: "gardner",
    did_extended_twfe: "etwfe",
}
