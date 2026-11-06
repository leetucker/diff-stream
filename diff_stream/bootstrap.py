"""Ordinary unit-cluster pairs bootstrap with serial or parallel full refits."""

import json
import pickle
import sys
import warnings
from concurrent.futures import as_completed
from contextlib import closing, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import pyarrow as pa
from scipy.stats import norm
from stream_limits import MemoryBudget, detect_cpus, thread_scope

from ._core.process import ProcessBatch
from ._core.resample import DiskGroupResampler
from .api import diff_stream
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
                result = diff_stream(
                    task["fml"],
                    task["path"],
                    panel=task["panel"],
                    cohort=task["cohort"],
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


def refit_bootstrap(
    fml,
    data,
    *,
    panel,
    cohort,
    estimator="cs",
    reps=199,
    seed=None,
    aggregate=None,
    aggregate_options=None,
    coefficients=None,
    workers=1,
    threads=None,
    memory_gb=None,
    preprocess_mb=64,
    batch_units=256,
    workdir=None,
    **fit_options,
):
    """Ordinary uniform unit-cluster resampling and full-estimator refits.

    Fits the original statistic once. Its selected panel/reference window is
    fixed before sampling. Each draw contains N units sampled with replacement,
    with distinct synthetic unit IDs; original columns and weights are carried.
    Nuisance models, trimming and aggregation shares are refitted on every draw.
    Reference selection cannot switch to an earlier cohort when controls vanish.

    aggregate=None targets the fitted coefficients; aggregate='dynamic'/'simple'/
    'cohort'/'calendar' invokes DIDResult.aggregate with aggregate_options. Gardner
    targets its static/dynamic coefficients directly and requires aggregate=None.
    coefficients optionally selects exact target names. Failed draws are logged,
    never replaced, and block SEs/intervals until allow_incomplete=True is explicit.
    At least two successful refits are required for any inference.

    Source data is projected/snapshotted once. Unit manifests/plans live on disk;
    resident bootstrap state is O(reps*target coefficients) plus bounded buffers.
    Seeds reproduce a fixed input/partition layout and NumPy RNG version; unit
    order and draw-plan hashes are recorded. Scratch is removed before return.
    This is not a multiplier/wild bootstrap or a complex-survey replicate design.

    workers=1 preserves in-process serial execution. workers>1 runs bounded
    batches of fresh Python processes, portable across Linux/macOS/Windows and
    interactive sessions. threads is the total fit-thread budget (detected when
    omitted), divided between workers. Parent staging pauses during each batch.
    memory_gb is the whole-job cooperative budget: parent RSS/reserve is removed
    before dividing the remaining budget between workers. For OS enforcement,
    run the whole call through stream-limits' process-tree launcher.
    """
    if estimator not in ("cs", "sunab", "bjs", "gardner", "etwfe"):
        raise ValueError("estimator must be cs, sunab, bjs, gardner or etwfe")
    for name, value, minimum in [
        ("reps", reps, 2),
        ("batch_units", batch_units, 1),
        ("workers", workers, 1),
    ]:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    if threads is not None and (
        isinstance(threads, bool) or not isinstance(threads, int) or threads < 1
    ):
        raise ValueError("threads must be a positive integer")
    active_workers = min(workers, reps)
    job_threads = threads if threads is not None else detect_cpus()
    if active_workers > 1 and active_workers > job_threads:
        raise ValueError(
            "workers cannot exceed the total thread budget; increase threads or reduce workers"
        )
    worker_threads = max(1, job_threads // active_workers)
    if isinstance(preprocess_mb, bool) or not np.isfinite(preprocess_mb) or preprocess_mb < 1:
        raise ValueError("preprocess_mb must be finite and at least 1 MB")
    if aggregate not in (None, "dynamic", "simple", "cohort", "calendar"):
        raise ValueError("aggregate must be dynamic, simple, cohort, calendar or None")
    if estimator == "gardner" and aggregate is not None:
        raise ValueError(
            "Gardner coefficients are direct; use aggregate=None and specification=static/dynamic"
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
        raise ValueError("refit bootstrap requires actual fits without nested multiplier draws")
    budget = MemoryBudget(memory_gb)
    if not budget.fits((preprocess_mb * 1e6 + 96e6) / 1e9):
        raise MemoryError("bootstrap preprocessing buffers exceed job headroom")
    weights = fit_options.get("weights")
    _, y, design = inspect(
        fml,
        data,
        panel,
        cohort,
        allow_covariates=True,
        weights=weights,
        allow_fe=estimator in ("sunab", "bjs", "gardner"),
    )
    required = list(
        dict.fromkeys(
            [
                *panel,
                cohort,
                y,
                *([] if weights is None else [weights]),
                *([] if design is None else design.required),
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
        preprocess_mb=preprocess_mb,
        batch_units=batch_units,
    )
    seed_sequence = np.random.SeedSequence(seed)
    rng = np.random.default_rng(seed_sequence)
    original_options = dict(options)
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
        original = diff_stream(
            fml,
            snapshot,
            panel=panel,
            cohort=cohort,
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
        expected_dates = list(range(diag["time_min"], diag["time_max"] + 1))
        cutoff = diag.get("reference_cutoff")
        mode = options.get("panel_mode", "balanced")

        def select(frame):
            if mode == "complete":
                ids = frame[panel[0]].to_numpy()
                starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
                lengths = np.diff(np.r_[starts, len(frame)])
                frame = frame.filter(
                    pl.Series(
                        np.repeat(lengths == original_dates[1] - original_dates[0] + 1, lengths)
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
                    minimum = 0.35 + preprocess_mb / 1000 + 0.096
                    if worker_memory * (1 - budget.reserve) < minimum:
                        raise MemoryError(
                            "insufficient whole-job memory for parallel refits including worker runtimes; "
                            "reduce workers/preprocess_mb or increase memory_gb"
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
            original_fit_options=original_options,
            aggregate=aggregate,
            aggregate_options=aggregate_options,
            panel_mode=mode,
            fixed_reference_group=diag.get("reference_group"),
            fixed_reference_cohort=diag.get("reference_cohort"),
            reference_cutoff=cutoff,
            analysis_time_min=diag["time_min"],
            analysis_time_max=diag["time_max"],
            replicate_reference_group="nevertreated",
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
        estimator=estimator,
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
