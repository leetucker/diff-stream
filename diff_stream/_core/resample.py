"""Bounded, disk-backed ordinary whole-group resampling.

Generic adapter intended for eventual hdfe_stream.core: consumers own sample
selection and model semantics. Uniform draws use N independent integer indices;
copied groups receive distinct draw IDs, rather than frequency weights.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from .partition import GroupPartitions, Part


@dataclass
class ResampleLeaf:
    path: Path
    start: int
    stop: int


class DiskGroupResampler(GroupPartitions):
    def __init__(
        self,
        snapshot,
        root,
        budget,
        preprocess_mb,
        *,
        key,
        sort_columns,
        select,
        index_column,
        draw_column,
    ):
        super().__init__(root, budget, preprocess_mb, key=key)
        self.sort_columns, self.select = sort_columns, select
        self.index_column, self.draw_column = index_column, draw_column
        self.leaves = []
        self.n_units = self.n_rows = 0
        self.identity = hashlib.sha256()
        size = 0
        with pq.ParquetFile(snapshot, pre_buffer=False) as pf:
            rows = pf.metadata.num_rows
            for batch in pf.iter_batches(batch_size=self.read_rows):
                self._guard(32e6 + batch.nbytes * 3)
                size += batch.nbytes
        self._prepare(Part(Path(snapshot), rows, size), 0)
        self.ends = np.array([leaf.stop for leaf in self.leaves], dtype=np.int64)

    def _leaf(self, part):
        self._guard(part.size * 4 + 32e6)
        with pq.ParquetFile(part.path, pre_buffer=False) as pf:
            data = pl.from_arrow(pf.read()).sort(self.sort_columns)
        data = self.select(data)
        if len(data):
            ids = data[self.key].to_numpy()
            starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
            lengths = np.diff(np.r_[starts, len(data)])
            for value in ids[starts]:
                encoded = repr(value.item() if isinstance(value, np.generic) else value).encode()
                self.identity.update(len(encoded).to_bytes(8, "little"))
                self.identity.update(encoded)
            data = data.with_columns(
                pl.Series(
                    self.index_column,
                    np.repeat(np.arange(self.n_units, self.n_units + len(starts)), lengths),
                )
            )
            path = self._name("resample-source")
            data.write_parquet(path, row_group_size=32768)
            self.leaves.append(ResampleLeaf(path, self.n_units, self.n_units + len(starts)))
            self.n_units += len(starts)
            self.n_rows += len(data)
            self.partition_count += 1
            self.max_leaf_bytes = max(self.max_leaf_bytes, part.size)
        part.path.unlink()

    def materialize(self, rng, root, *, time_column, chunk_rows=32768):
        """Write one exact N-group resample. Return path, dates, rows, plan hash.

        Integer plans are routed to leaf-specific binary files, never collected
        into an N-row in-memory manifest. Output buffers and join multiplicity
        are bounded separately from the number of groups in the population.
        """
        root = Path(root)
        root.mkdir()
        digest = hashlib.sha256()
        for offset in range(0, self.n_units, chunk_rows):
            count = min(chunk_rows, self.n_units - offset)
            indices = rng.integers(0, self.n_units, size=count, dtype=np.int64)
            digest.update(indices.astype("<i8", copy=False).tobytes())
            buckets = np.searchsorted(self.ends, indices, side="right")
            order = np.argsort(buckets, kind="stable")
            ordered = buckets[order]
            starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1, count]
            for a, b in zip(starts[:-1], starts[1:], strict=True):
                chosen = order[a:b]
                pairs = np.column_stack([indices[chosen], offset + chosen]).astype("<i8")
                with (root / f"plan-{ordered[a]}.bin").open("ab") as f:
                    pairs.tofile(f)
        output = root / "resample.parquet"
        writer, pending, pending_rows, pending_bytes = None, [], 0, 0
        dates, rows = set(), 0
        try:
            for j, leaf in enumerate(self.leaves):
                plan = root / f"plan-{j}.bin"
                if not plan.exists():
                    continue
                with pq.ParquetFile(leaf.path, pre_buffer=False) as pf:
                    data = pl.from_arrow(pf.read())
                ids = data[self.index_column].to_numpy()
                starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
                lengths = np.diff(np.r_[starts, len(data)])
                leaf_bytes = data.estimated_size()

                def blocks(pairs, leaf=leaf, lengths=lengths, leaf_bytes=leaf_bytes, data=data):
                    counts = np.bincount(pairs[:, 0] - leaf.start, minlength=leaf.stop - leaf.start)
                    nrows = int(counts @ lengths)
                    upper = leaf_bytes * int(counts.max()) + 16 * nrows
                    if upper > self.leaf_bytes * 2 and len(pairs) > 1:
                        mid = len(pairs) // 2
                        yield from blocks(pairs[:mid])
                        yield from blocks(pairs[mid:])
                        return
                    self._guard(4 * upper + 2 * leaf_bytes + 32e6)
                    selection = pl.DataFrame(
                        {self.index_column: pairs[:, 0], self.draw_column: pairs[:, 1]}
                    )
                    yield data.join(selection, on=self.index_column, how="inner").drop(
                        self.index_column
                    )

                with plan.open("rb") as f:
                    while True:
                        pairs = np.fromfile(f, dtype="<i8", count=512).reshape(-1, 2)
                        if not len(pairs):
                            break
                        for frame in blocks(pairs):
                            dates.update(frame[time_column].unique().to_list())
                            rows += len(frame)
                            if writer is None:
                                writer = pq.ParquetWriter(
                                    output,
                                    frame.to_arrow().schema,
                                    compression="zstd",
                                    use_dictionary=False,
                                )
                            pending.append(frame)
                            pending_rows += len(frame)
                            pending_bytes += frame.estimated_size()
                            if (
                                pending_rows >= 32768
                                or pending_bytes >= self.leaf_bytes * 2
                                or len(pending) >= 128
                            ):
                                writer.write_table(
                                    pl.concat(pending).to_arrow(), row_group_size=32768
                                )
                                pending, pending_rows, pending_bytes = [], 0, 0
                plan.unlink()
            if pending:
                writer.write_table(pl.concat(pending).to_arrow(), row_group_size=32768)
        finally:
            if writer is not None:
                writer.close()
        return output, sorted(dates), rows, digest.hexdigest()
