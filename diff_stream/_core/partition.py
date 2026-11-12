"""The memory of staging a panel (the partitioning itself is hdfe_stream.core.partition)."""


def staging_bytes(preprocess_mb, store):
    """Expected memory of staging: the preprocessing allowance, or less when the
    staged rows are small. Sorting a panel in one piece peaked at about six
    times its rows' size in the measurements (more on small panels, where the
    runtime's own use dominates), so eight times plus a fixed 0.4 GB covers
    them; the allowance stays the upper bound."""
    rows = getattr(store, "n_rows_used", None)
    if not rows:
        return preprocess_mb * 1e6
    design = getattr(store, "design", None)
    columns = 4 + (len(design.names) if design is not None else 0)
    columns += int(getattr(store, "weighted", False))
    return min(preprocess_mb * 1e6, 8.0 * rows * 8 * columns + 0.4e9)
