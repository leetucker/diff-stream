"""Explicit clean last-treated reference windows on an already validated panel."""

import warnings
from dataclasses import dataclass

import polars as pl

from .store import PanelStore


class ReferenceWindowWarning(UserWarning):
    """Estimation uses only dates before the last reference cohort is affected."""


@dataclass
class FrozenDesign:
    """Reuse validated model columns when restaging a restricted panel."""

    names: list

    def discover(self, frame):
        pass

    def compile(self):
        pass

    def evaluate(self, frame):
        return frame.select([f"__x{j}" for j in range(len(self.names))])


def reference_store(store, policy, anticipation):
    """Auto selects last-treated only when no clean never-treated group exists.

    Complete-panel selection precedes reference selection. The reference cohort
    and existing never-treated units are internally pooled as clean controls;
    the original reference date and all restrictions remain explicit metadata.
    """
    use_last = policy == "lasttreated" or (policy == "auto" and 0 not in store.gc)
    if not use_last:
        return store, dict(
            reference_group_requested=policy,
            reference_group="nevertreated",
            reference_cohort=0,
            reference_cutoff=None,
            reference_dropped_rows=0,
            reference_dropped_units=0,
        )
    treated = [g for g in store.gc if g]
    if len(treated) < 2:
        raise ValueError(
            "last-treated reference requires at least two treated cohorts after panel selection"
        )
    reference = max(treated)
    cutoff = reference - anticipation
    if cutoff - int(store.times[0]) < 2:
        raise ValueError(
            "fewer than two clean dates remain before the last-treated reference is affected"
        )
    data = (
        pl.scan_parquet(store.paths, low_memory=True)
        .filter(pl.col("t") < cutoff)
        .with_columns(pl.when(pl.col("g") == reference).then(0).otherwise(pl.col("g")).alias("g"))
    )
    restricted = restage(
        store,
        data,
        "reference",
        "balanced" if store.panel_mode in ("balanced", "complete") else "unbalanced",
        anticipation,
    )
    dropped_rows = store.n_rows_used - restricted.n_rows_used
    dropped_units = store.n_units - restricted.n_units
    metadata = dict(
        reference_group_requested=policy,
        reference_group="lasttreated",
        reference_cohort=int(reference),
        reference_cohort_units=store.gc[reference],
        reference_never_units=store.gc.get(0, 0),
        input_time_min=int(store.times[0]),
        input_time_max=int(store.times[-1]),
        reference_cutoff=int(cutoff),
        reference_last_clean_time=int(restricted.times[-1]),
        reference_dropped_rows=dropped_rows,
        reference_dropped_units=dropped_units,
        reference_includes_never_treated=0 in store.gc,
        reference_selection="after original balanced/complete/unbalanced panel policy; before reference is affected",
    )
    for path in store.paths:
        path.unlink()
    warnings.warn(
        f"Last-treated cohort {reference} is a reference only: retain time < {cutoff} (anticipation={anticipation}); "
        f"excluded {dropped_rows} observed rows. Effects after the reference becomes affected are not identified by this design.",
        ReferenceWindowWarning,
        stacklevel=3,
    )
    return restricted, metadata


def restage(store, data, name, panel_mode, anticipation):
    """Stage a subset of a store's rows (``data``, with the store's columns) as
    a new store, keeping the input counts and the original period labels."""
    root = store.root / name
    root.mkdir()
    data = data.with_columns(pl.col("u").alias("id")).drop("u")
    if store.weighted:
        data = data.with_columns(pl.col("w") * store.weight_scale)
    design = FrozenDesign(list(store.design.names)) if store.design is not None else None
    restricted = PanelStore(
        data,
        root,
        store.batch_units,
        store.budget,
        store.leaf_bytes * 4 / 1e6,
        design,
        panel_mode,
        anticipation,
        allow_time_varying_weights=store.allow_time_varying_weights,
        retain_actual_cohort=store.retain_actual_cohort,
        actual_cohort_column="g_actual" if store.retain_actual_cohort else None,
    )
    # The restaged rows are already numbered consecutively; keep the original labels.
    restricted.period_labels = store.period_labels
    restricted.n_rows = store.n_rows
    restricted.n_units_input = store.n_units_input
    return restricted
