"""Goodman–Bacon decomposition from balanced cohort-time sufficient statistics.

Independent implementation of the published two-group comparison identity.
Treatment timing and decomposition rules are DiD-specific; bounded source
validation/partitioning is shared with the estimation entry points.
"""

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
from hdfe_stream.core.profile import Component, Profile
from stream_limits import MemoryBudget, current_rss_gb, detect_cpus, thread_scope

from .store import PanelStore, inspect


@dataclass(kw_only=True)
class BaconResult:
    """Descriptive TWFE decomposition; no separate causal ATT or standard errors."""

    fml: str
    depvar: str
    twfe_estimate: float
    decomposed_estimate: float
    n_units: int
    n_obs: int
    diagnostics: dict
    _table: pl.DataFrame = field(repr=False)
    within_weight: float = 0.0
    within_estimate: float | None = None
    within_contribution: float = 0.0

    @property
    def reconstruction_error(self):
        return self.decomposed_estimate - self.twfe_estimate

    def comparisons(self):
        """Two-block comparisons, or full-period dyads when covariate-adjusted."""
        return self._table.clone()

    def tidy(self):
        """Alias for comparisons()."""
        return self.comparisons()

    def within_timing(self):
        """Within-timing component of a covariate-adjusted decomposition.

        The estimate is undefined/null when its weight is zero. Contributions
        plus the contributions in comparisons() reconstruct the target TWFE.
        """
        if not self.diagnostics.get("covariate_adjusted", False):
            raise ValueError("within_timing is available for covariate-adjusted decompositions")
        return pl.DataFrame(
            {
                "comparison_type": ["within_timing"],
                "weight": [self.within_weight],
                "estimate": [self.within_estimate],
                "contribution": [self.within_contribution],
            },
            schema={
                "comparison_type": pl.String,
                "weight": pl.Float64,
                "estimate": pl.Float64,
                "contribution": pl.Float64,
            },
        )

    def summary(self):
        """Weights, within-type weighted estimates and contributions by type."""
        table = self._table.group_by("comparison_type", maintain_order=True).agg(
            pl.len().alias("n_comparisons"), pl.col("weight").sum(), pl.col("contribution").sum()
        )
        if self.diagnostics.get("covariate_adjusted", False):
            within = self.within_timing().with_columns(
                pl.lit(1, dtype=pl.UInt32).alias("n_comparisons")
            )
            table = pl.concat([table, within.select(table.columns)])
        return table.with_columns(
            pl.when(pl.col("weight") > 0)
            .then(pl.col("contribution") / pl.col("weight"))
            .otherwise(None)
            .alias("estimate")
        ).select("comparison_type", "n_comparisons", "weight", "estimate", "contribution")

    def summary_dict(self, *, comparisons=False):
        out = dict(
            fml=self.fml,
            depvar=self.depvar,
            twfe_estimate=self.twfe_estimate,
            decomposed_estimate=self.decomposed_estimate,
            reconstruction_error=self.reconstruction_error,
            n_units=self.n_units,
            n_obs=self.n_obs,
            diagnostics=self.diagnostics,
            by_type=self.summary().to_dicts(),
        )
        if self.diagnostics.get("covariate_adjusted", False):
            out["within_timing"] = self.within_timing().to_dicts()[0]
        if comparisons:
            out["comparisons"] = self.comparisons().to_dicts()
        return out

    def summary_json(self, path=None, *, comparisons=False):
        """Return JSON and optionally save it to a file."""
        content = json.dumps(self.summary_dict(comparisons=comparisons), indent=2, allow_nan=False)
        if path is not None:
            Path(path).write_text(content + "\n")
        return content


def _decompose(store):
    groups = sorted(store.gc)
    index = {g: j for j, g in enumerate(groups)}
    sizes = np.array([store.gc[g] for g in groups], dtype=float)
    masses = np.array([store.gw[g] for g in groups], dtype=float)
    proportions = masses / masses.sum()
    raw_masses = masses * (store.weight_scale / store.weight_normalizer)
    times, T = store.times, store.periods
    first, last = int(times[0]), int(times[-1])
    # Cohort constants cancel from every contrast. Removing them before prefix
    # sums reduces cancellation from large unit/cohort intercepts.
    means = store.means - store.means[:, :1]
    prefix = np.column_stack([np.zeros(len(groups)), np.cumsum(means, axis=1)])
    rows = []

    def add(treated, control, pre_start, pre_end, post_start, post_end, kind):
        i, j = index[treated], index[control]
        pre, post = pre_end - pre_start + 1, post_end - post_start + 1

        def change(k):
            return (prefix[k, post_end - first + 1] - prefix[k, post_start - first]) / post - (
                prefix[k, pre_end - first + 1] - prefix[k, pre_start - first]
            ) / pre

        rows.append(
            dict(
                treated_cohort=int(treated),
                control_cohort=int(control),
                comparison_type=kind,
                control_already_treated=kind in ("later_vs_earlier", "later_vs_always"),
                pre_start=int(pre_start),
                pre_end=int(pre_end),
                post_start=int(post_start),
                post_end=int(post_end),
                pre_periods=int(pre),
                post_periods=int(post),
                n_treated_units=int(sizes[i]),
                n_control_units=int(sizes[j]),
                treated_weight_total=float(raw_masses[i]),
                control_weight_total=float(raw_masses[j]),
                estimate=float(change(i) - change(j)),
                weight=float(proportions[i] * proportions[j] * (pre / T) * (post / T)),
            )
        )

    treated = [g for g in groups if g > first]
    if 0 in index:
        for g in treated:
            add(g, 0, first, g - 1, g, last, "treated_vs_never")
    if first in index:
        for g in treated:
            add(g, first, first, g - 1, g, last, "later_vs_always")
    for early, late in combinations(treated, 2):
        add(early, late, first, early - 1, early, late - 1, "earlier_vs_later")
        add(late, early, early, late - 1, late, last, "later_vs_earlier")
    weight_sum = sum(row["weight"] for row in rows)
    D = np.array([(times >= g) & (g != 0) for g in groups], dtype=float)
    d = D - D.mean(axis=1, keepdims=True)
    d -= proportions @ d
    y = means - means.mean(axis=1, keepdims=True)
    y -= proportions @ y
    variance = float(np.sum(proportions[:, None] * d**2) / T)
    if not weight_sum > 0 or not variance > 0:
        raise ValueError(
            "treatment is collinear with unit/time effects; no identified TWFE decomposition"
        )
    if not np.isclose(weight_sum, variance, rtol=1e-10, atol=0):
        raise FloatingPointError(
            "decomposition weights do not reconstruct the residualized treatment variance"
        )
    twfe = float(np.sum(proportions[:, None] * d * y) / T / variance)
    for row in rows:
        row["weight"] /= weight_sum
        row["contribution"] = row["weight"] * row["estimate"]
    reconstructed = sum(row["contribution"] for row in rows)
    tolerance = 1e-10 * max(1.0, abs(twfe), max(abs(row["estimate"]) for row in rows))
    if (
        not np.isfinite(twfe)
        or not np.isfinite(reconstructed)
        or abs(twfe - reconstructed) > tolerance
    ):
        raise FloatingPointError(
            "Goodman–Bacon reconstruction lost numerical precision; rescale the outcome"
        )
    return pl.DataFrame(rows), twfe, float(reconstructed), variance


def bacon_decomp(
    fml,
    data,
    *,
    panel,
    cohort,
    panel_mode="balanced",
    weights=None,
    threads=None,
    memory_gb=None,
    workdir=None,
    profile=False,
    preprocess_mb=64,
    batch_units=1024,
):
    """Decompose the OLS/WLS coefficient on treatment with unit/time effects.

    fml is an outcome formula ``'y ~ 1'`` or ``'y ~ x + z'``; treatment comes
    from cohort (0=never). Requires balanced absorbing-treatment panels. Optional
    weights='column' supplies positive, finite sampling weights constant within
    unit. All fits/comparison weights use the same sampling weights; physical
    unit counts remain separate from input-scale weight totals. Common positive
    rescaling leaves estimates and decomposition weights unchanged.
    panel_mode='complete' explicitly drops
    incomplete units; native unbalanced mode is unsupported. Positive cohorts
    on/before the first date normalize to an always-treated group. Future
    cohorts normalize to never-treated within the sample.
    A never-treated group is optional when distinct treated timings identify
    the regression. The result reports comparisons/weights/contributions and
    a TWFE reconstruction, without causal ATT inference. With covariates, the
    within-timing component and full-period timing-group dyads jointly recover
    TWFE; dyad weights alone sum to 1-within_weight. See docs/bacon_extensions.md.

    Resource/source parameters follow diff_stream(). profile=True returns a
    Profile after disk-backed validation, before allocating the comparison
    table. Without covariates numerical state is O(groups * periods + groups**2).
    Covariates add a dense projection Gram and bounded unit batches; batch_units
    controls the latter. Preprocessing uses bounded whole-unit partitions.
    """
    if panel_mode not in ("balanced", "complete"):
        raise ValueError(
            "Goodman–Bacon requires panel_mode='balanced' or 'complete'; native unbalanced panels are unsupported"
        )
    if isinstance(batch_units, bool) or not isinstance(batch_units, int) or batch_units < 1:
        raise ValueError("batch_units must be a positive integer")
    if threads is not None and (
        isinstance(threads, bool) or not isinstance(threads, int) or threads < 1
    ):
        raise ValueError("threads must be a positive integer")
    if isinstance(preprocess_mb, bool) or not np.isfinite(preprocess_mb) or preprocess_mb < 1:
        raise ValueError("preprocess_mb must be finite and at least 1 MB")
    budget = MemoryBudget(memory_gb)
    baseline = current_rss_gb() or 0
    with thread_scope(threads):
        lf, yname, design = inspect(
            fml, data, panel, cohort, weights=weights, allow_covariates=True
        )
        if not budget.fits((preprocess_mb * 1e6 + 96e6) / 1e9):
            raise MemoryError(
                "preprocessing buffers exceed job headroom before staging; reduce preprocess_mb or request more memory"
            )
        if workdir is not None:
            Path(workdir).mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="diff-stream-bacon-", dir=workdir) as run:
            store = PanelStore(
                lf,
                run,
                batch_units,
                budget,
                preprocess_mb,
                design=design,
                panel_mode=panel_mode,
                require_never_treated=False,
                allow_always_treated=True,
            )
            G, T = len(store.gc), store.periods
            always = int(store.times[0]) in store.gc
            K = G - int(0 in store.gc) - int(always)
            covariates = (
                [j for j, name in enumerate(design.names) if name != "Intercept"]
                if design is not None
                else []
            )
            adjusted = bool(covariates)
            comparisons = (
                G * (G - 1) // 2
                if adjusted
                else K * (K - 1) + K * (int(0 in store.gc) + int(always))
            )
            numeric = 8 * 24 * G * T + 4096 * comparisons + 16e6
            if adjusted:
                P = T - 1 + len(covariates)
                B = min(batch_units, store.n_units)
                numeric += 8 * (10 * P * P + 12 * B * T * (P + 3) + 8 * G * T * len(covariates))
            diagnostics = dict(
                covariate_adjusted=adjusted,
                covariate_names=[design.names[j] for j in covariates],
                always_treated_units=store.gc.get(int(store.times[0]), 0),
                always_treated_cohort=int(store.times[0]) if always else None,
                batch_units=batch_units,
                sampling_weights=weights,
                weight_convention="positive time-invariant unit sampling weights"
                if store.weighted
                else "equal unit weights",
                weight_scale=store.weight_scale / store.weight_normalizer,
                total_unit_weight=float(
                    store.n_units * (store.weight_scale / store.weight_normalizer)
                ),
                cohort_weight_totals={
                    str(g): float(m * (store.weight_scale / store.weight_normalizer))
                    for g, m in store.gw.items()
                },
                panel_mode=panel_mode,
                input_units=store.n_units_input,
                input_rows=store.n_rows,
                dropped_units=store.n_units_input - store.n_units,
                dropped_rows=store.n_rows - store.n_rows_used,
                time_min=int(store.times[0]),
                time_max=int(store.times[-1]),
                periods=T,
                cohort_sizes={str(g): int(n) for g, n in store.gc.items()},
                n_cohorts=G,
                n_comparisons=comparisons,
                partitions=store.partition_count,
                preprocess_mb=preprocess_mb,
                max_decoded_leaf_bytes=store.max_leaf_bytes,
                treatment="1[cohort > 0 and time >= cohort]",
                scope="balanced, covariate-adjusted unit/time fixed effects"
                if adjusted
                else "balanced, no covariates; weighted unit/time fixed effects"
                if store.weighted
                else "balanced, no covariates, equal weights; OLS unit/time fixed effects",
            )
            if profile:
                return Profile(
                    model="diff-stream Goodman–Bacon",
                    threads=threads or detect_cpus(),
                    memory_limit_gb=budget.limit_gb,
                    memory_reserve=budget.reserve,
                    polars_threads=pl.thread_pool_size(),
                    baseline_gb=baseline,
                    components=[
                        Component(
                            "cohort-time reductions, projection and comparison table"
                            if adjusted
                            else "cohort-time reductions and comparison table",
                            numeric,
                            numeric * 1.5,
                        ),
                        Component(
                            "bounded partition preprocessing",
                            preprocess_mb * 1e6 + 96e6,
                            preprocess_mb * 1e6 + 256e6,
                        ),
                        Component(
                            "snapshot, hash partitions and sorted leaves",
                            64 * store.n_rows,
                            128 * store.n_rows,
                            where="disk",
                        ),
                    ],
                    notes=diagnostics,
                    warnings=[
                        "Heuristic estimate; source readers and allocators have separate overhead. Comparison output grows quadratically with cohorts."
                    ],
                )
            if not budget.fits(numeric / 1e9):
                raise MemoryError(
                    "Goodman–Bacon comparison table exceeds job headroom; reduce the number of cohorts or request more memory"
                )
            within_weight, within_estimate, within_contribution = 0.0, None, 0.0
            if adjusted:
                from .bacon_controls import decompose_controls

                (
                    table,
                    twfe,
                    decomposed,
                    variance,
                    within_weight,
                    within_estimate,
                    within_contribution,
                    metadata,
                ) = decompose_controls(store, covariates)
                diagnostics.update(metadata)
            else:
                table, twfe, decomposed, variance = _decompose(store)
            diagnostics.update(
                residualized_treatment_variance=variance,
                weight_sum=float(table["weight"].sum()) + within_weight,
                between_weight_sum=float(table["weight"].sum()),
                already_treated_control_weight=(
                    None
                    if adjusted
                    else float(table.filter(pl.col("control_already_treated"))["weight"].sum())
                ),
            )
            return BaconResult(
                fml=fml,
                depvar=yname,
                twfe_estimate=twfe,
                decomposed_estimate=decomposed,
                n_units=store.n_units,
                n_obs=store.n_rows_used,
                diagnostics=diagnostics,
                _table=table,
                within_weight=within_weight,
                within_estimate=within_estimate,
                within_contribution=within_contribution,
            )
