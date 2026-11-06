"""independent_columns: the ordered basis of a Gram matrix, against a direct oracle."""

import time

import numpy as np
import pytest

from diff_stream._core.within import independent_columns


def oracle(gram, tolerance=1e-10):
    """Keep a column when its squared distance from the span of those kept before it
    (after scaling to unit diagonal) exceeds the tolerance: least squares per column."""
    scale = np.sqrt(np.maximum(np.diag(gram), 0))
    live = np.diag(gram) > 0
    safe = np.where(live, scale, 1)
    corr = gram / safe[:, None] / safe[None, :]
    kept = []
    for j in np.flatnonzero(live):
        if kept:
            coef = np.linalg.lstsq(corr[np.ix_(kept, kept)], corr[kept, j], rcond=None)[0]
            remaining = corr[j, j] - corr[kept, j] @ coef
        else:
            remaining = corr[j, j]
        if remaining > tolerance:
            kept.append(j)
    return np.array(kept, dtype=int)


@pytest.mark.parametrize("seed", range(20))
def test_matches_the_direct_selection(seed):
    rng = np.random.default_rng(seed)
    n, k = int(rng.integers(20, 80)), int(rng.integers(5, 40))
    X = rng.normal(size=(n, k))
    for _ in range(int(rng.integers(1, 5))):  # exact aliases and a constant combination
        a, b, c = rng.integers(0, k, 3)
        X[:, a] = 2.0 * X[:, b] - X[:, c]
    X[:, int(rng.integers(0, k))] = 0.0  # a column with no variation
    gram = X.T @ X
    assert np.array_equal(independent_columns(gram), oracle(gram))
    norms = np.diag(gram).copy()
    assert np.array_equal(independent_columns(gram, original_norm=norms), oracle(gram))


def test_earlier_columns_have_priority():
    base = np.random.default_rng(1).normal(size=(50, 3))
    # column 3 is the sum of columns 0 and 1, column 4 repeats column 2
    X = np.column_stack([base, base[:, 0] + base[:, 1], base[:, 2]])
    assert independent_columns(X.T @ X).tolist() == [0, 1, 2]


def test_hundreds_of_columns_are_not_slow():
    # 841 columns, as in an extended TWFE with 29 cohorts over 30 periods: the
    # selection is a single pass of triangular solves, well under a second
    X = np.random.default_rng(0).normal(size=(3000, 841))
    start = time.perf_counter()
    kept = independent_columns(X.T @ X)
    assert len(kept) == 841
    assert time.perf_counter() - start < 10
