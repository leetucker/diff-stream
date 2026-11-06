"""Reference CSVs that are stored gzipped.

Frozen reference outputs (R, Stata) are committed as ``name.csv.gz`` to keep
the repository small; input panels stay plain CSV because the external
exporters read them by path. Code names a reference by its logical ``.csv``
path, and these helpers fall back to the ``.csv.gz`` file next to it. A
freshly regenerated plain ``.csv`` takes precedence, so regeneration works
unchanged; run ``python benchmarks/_files.py gzip <folder>`` afterwards.

Hashes recorded in provenance manifests are of the CSV content, so a file's
digest is the same before and after compression.
"""

import gzip
import hashlib
import sys
from pathlib import Path

import polars as pl


def resolve(path):
    """The file that holds ``path``: itself if present, else ``path.gz``."""
    if not isinstance(path, (str, Path)):
        return path
    path = Path(path)
    if path.exists():
        return path
    compressed = path.with_name(path.name + ".gz")
    return compressed if compressed.exists() else path


def read_csv(path, **options):
    """polars.read_csv on a reference that may be stored gzipped."""
    return pl.read_csv(resolve(path), **options)


def exists(path):
    return resolve(path).exists()


def read_bytes(path):
    """The content of ``path``, decompressed when it is stored as ``path.gz``.

    A file named ``.gz`` on purpose (for example ``multipliers.csv.gz``) is
    read as it is, so hashes recorded for it are unchanged.
    """
    found = resolve(path)
    data = found.read_bytes()
    return gzip.decompress(data) if found != Path(path) else data


def digest(path):
    return hashlib.sha256(read_bytes(path)).hexdigest()


def csv_files(directory, pattern="*.csv"):
    """Logical ``.csv`` paths in ``directory``, whether stored plain or gzipped."""
    directory = Path(directory)
    plain = set(directory.glob(pattern))
    stored = {p.with_name(p.name[:-3]) for p in directory.glob(pattern + ".gz")}
    return sorted(plain | stored)


def gzip_csvs(directory, *, recursive=True):
    """Replace each plain ``.csv`` under ``directory`` by a reproducible ``.csv.gz``.

    No timestamp or file name is stored in the gzip header, so compressing the
    same content always gives the same bytes.
    """
    directory = Path(directory)
    files = sorted(directory.rglob("*.csv") if recursive else directory.glob("*.csv"))
    for path in files:
        target = path.with_name(path.name + ".gz")
        target.write_bytes(gzip.compress(path.read_bytes(), compresslevel=9, mtime=0))
        path.unlink()
    return files


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "gzip":
        sys.exit("usage: python benchmarks/_files.py gzip <folder> [<folder> ...]")
    for folder in sys.argv[2:]:
        print(f"{folder}: {len(gzip_csvs(folder))} files compressed")
