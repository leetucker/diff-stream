"""Bounded whole-group hash partitions, independent of panel/DiD semantics.

Consumers implement _leaf and optionally _oversized_group. Rows of a group
always remain together; the decoded leaf budget can include projected columns.
Intended for eventual hdfe_stream.core.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq


def preprocess_budget_mb(preprocess_mb, budget):
    """The working memory for partitioning and sorting, in decimal MB.

    An explicit value is validated and used as given. None (the default) uses
    a quarter of the job's current memory headroom, between 16 MB and 32 GB,
    so a panel that fits in memory is sorted in one piece instead of being
    hash-partitioned through disk, and a smaller ``max_memory_gb`` gives
    smaller partitions automatically. A quarter, not more: headroom shrinks
    while the data are staged, and each leaf is checked against it again.
    """
    if preprocess_mb is None:
        return float(min(32000.0, max(16.0, budget.headroom_gb() * 250)))
    if isinstance(preprocess_mb, bool) or not np.isfinite(preprocess_mb) or preprocess_mb < 1:
        raise ValueError("preprocess_mb must be finite and at least 1 MB, or None")
    return float(preprocess_mb)


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


@dataclass
class Part:
    path: Path
    rows: int
    size: int


def whole_group_batches(path, key, max_groups, read_rows=32768):
    """Read sorted Parquet without splitting a group across yielded frames.

    Memory is bounded by one reader batch, one unfinished group, and the
    consumer's batch. Keys need only be unique within this file.
    """
    pending = None
    with pq.ParquetFile(path, pre_buffer=False) as pf:
        for batch in pf.iter_batches(batch_size=read_rows):
            frame = pl.from_arrow(batch)
            if pending is not None:
                frame = pl.concat([pending, frame])
            values = frame[key].to_numpy()
            starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
            end = int(starts[-1])
            for j in range(0, len(starts) - 1, max_groups):
                stop = int(starts[min(j + max_groups, len(starts) - 1)])
                yield frame.slice(int(starts[j]), stop - int(starts[j]))
            pending = frame.slice(end)
        if pending is not None:
            yield pending


class GroupPartitions:
    def __init__(self, root, budget, preprocess_mb, *, key="id"):
        self.root = Path(root)
        self.budget = budget
        self.key = key
        self.leaf_bytes = int(preprocess_mb * 1e6 / 4)
        self.read_rows = max(256, min(262144, self.leaf_bytes // 128))
        self.extra_row_bytes = 0
        self.max_leaf_bytes = 0
        self.max_depth = 0
        self.partition_count = 0
        self._serial = 0

    def _oversized_group(self, part):
        raise MemoryError("a single group exceeds preprocess_mb; increase the preprocessing budget")

    def _guard(self, size):
        if not self.budget.fits(size / 1e9):
            threads = pl.thread_pool_size()
            hint = (
                f" (Polars is using {threads} threads, and the memory it holds while reading grows "
                "with them, about 130 MB each for row groups of a million rows; set "
                "STREAM_LIMITS_MAX_THREADS or POLARS_MAX_THREADS before Python starts to lower it)"
                if threads > 4
                else ""
            )
            raise MemoryError(
                "preprocessing buffers exceed job headroom; reduce preprocess_mb or request more memory"
                + hint
            )

    def _name(self, prefix):
        self._serial += 1
        return self.root / f"{prefix}-{self._serial}.parquet"

    def _prepare(self, part, depth):
        effective = part.size + part.rows * self.extra_row_bytes
        if effective <= self.leaf_bytes:
            self._leaf(part)
            return
        if depth >= 16:
            raise MemoryError("unit/hash skew exceeds the partition budget; increase preprocess_mb")
        self.max_depth = max(self.max_depth, depth + 1)
        fanout = min(16, max(2, int(np.ceil(2 * effective / self.leaf_bytes))))
        children = {}
        writers = {}
        first_id, same_id = None, True
        try:
            with pq.ParquetFile(part.path, pre_buffer=False) as pf:
                for b in pf.iter_batches(batch_size=self.read_rows):
                    self._guard(32e6)
                    d = pl.from_arrow(b)
                    if first_id is None:
                        first_id = d[self.key][0]
                    same_id = same_id and (d[self.key] == first_id).all()
                    d = d.with_columns(
                        (pl.col(self.key).hash(seed=72931 + depth) % fanout).alias("bucket")
                    )
                    for key, frame in d.partition_by("bucket", as_dict=True).items():
                        k = key[0]
                        table = frame.drop("bucket").to_arrow()
                        if k not in writers:
                            p = self._name("hash")
                            children[k] = Part(p, 0, 0)
                            writers[k] = pq.ParquetWriter(
                                p, table.schema, compression="zstd", use_dictionary=False
                            )
                        writers[k].write_table(table, row_group_size=self.read_rows)
                        children[k].rows += table.num_rows
                        children[k].size += table.nbytes
        finally:
            for w in writers.values():
                w.close()
        part.path.unlink()
        if same_id:
            self._oversized_group(part)
        # Do not retain the parent's last decoded batch and writer metadata
        # while recursively processing its children.
        del writers, d, frame, table, b, w
        for child in children.values():
            self._prepare(child, depth + 1)
