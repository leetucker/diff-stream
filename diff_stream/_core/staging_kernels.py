"""One compiled pass over a sorted unit-period table for the staging checks.

Staging validates the panel and sums outcomes and weights by cohort and period.
In NumPy that is about fifteen passes over every row, each single-threaded.
The kernel here does the checks and sums in one parallel pass over chunks of
units; the chunks' partial sums are added in order, so the result does not
depend on the number of threads. ``PanelStore._leaf`` keeps the NumPy code for
the cases the kernel does not cover (treatment columns, spaced periods,
complete-case selection, original cohorts), and the tests compare the two.
"""

import numpy as np
from numba import njit, prange

ENABLED = True  # tests switch it off to compare with the NumPy path
CHUNKS = 256  # at most; fixed so results do not depend on the number of threads

DUPLICATE, COHORT, WEIGHT, BAD_WEIGHT, PERIOD = range(5)


@njit(parallel=True, cache=True)
def leaf_pass(starts, ends, t, g, y, w, has_w, scale, code, t0, per_chunk, flags, sums, mass, counts, mean_w):  # fmt: skip
    """Checks and sums of one sorted table; ``starts``/``ends`` delimit the units
    (``code`` is each unit's cohort index), ``flags[chunk]`` marks the failed
    checks and ``sums``, ``mass``, ``counts`` are per chunk by cohort and period."""
    n_units = starts.shape[0]
    T = sums.shape[2]
    chunks = flags.shape[0]
    for c in prange(chunks):
        lo = c * per_chunk
        hi = min(n_units, lo + per_chunk)
        for u in range(lo, hi):
            s = starts[u]
            e = ends[u]
            k = code[u]
            wsum = 0.0
            for r in range(s, e):
                if r > s and t[r] == t[r - 1]:
                    flags[c, DUPLICATE] = 1
                if g[r] != g[s]:
                    flags[c, COHORT] = 1
                ti = t[r] - t0
                if ti < 0 or ti >= T:
                    flags[c, PERIOD] = 1
                    continue
                wr = 1.0
                if has_w:
                    if w[r] != w[s]:
                        flags[c, WEIGHT] = 1
                    wr = w[r] / scale
                    if not (np.isfinite(wr) and wr > 0.0):
                        flags[c, BAD_WEIGHT] = 1
                wsum += wr
                sums[c, k, ti] += y[r] * wr
                mass[c, k, ti] += wr
                counts[c, k, ti] += 1
            mean_w[u] = wsum / (e - s)


def summarize(starts, lengths, t, g, y, w, scale, code, n_cohorts, t0, periods):
    """Flags and sums (``(cohorts, periods)`` arrays) of a sorted table; ``w`` is
    None when unweighted."""
    n_units = len(starts)
    chunks = min(CHUNKS, max(1, n_units))
    per_chunk = -(-n_units // chunks)
    chunks = -(-n_units // per_chunk)
    flags = np.zeros((chunks, 5), dtype=np.int8)
    shape = (chunks, n_cohorts, periods)
    sums, mass = np.zeros(shape), np.zeros(shape)
    counts = np.zeros(shape, dtype=np.int64)
    mean_w = np.zeros(n_units)
    leaf_pass(
        starts.astype(np.int64),
        (starts + lengths).astype(np.int64),
        np.ascontiguousarray(t, dtype=np.int64),
        np.ascontiguousarray(g, dtype=np.int64),
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(w, dtype=np.float64) if w is not None else np.zeros(1),
        w is not None,
        float(scale),
        np.ascontiguousarray(code, dtype=np.int64),
        int(t0),
        per_chunk,
        flags,
        sums,
        mass,
        counts,
        mean_w,
    )  # fmt: skip
    return flags.max(axis=0), sums.sum(axis=0), mass.sum(axis=0), counts.sum(axis=0), mean_w
