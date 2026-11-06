"""Large objective sums must not stall an otherwise converged logit fit."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from diff_stream._core.nuisance import fit_systems


@pytest.mark.parametrize("scale", [1e-12, 1e-8, 1.0, 1e8, 1e12])
@pytest.mark.parametrize("batch_size", [137, 4096])
@pytest.mark.parametrize("weighted", [False, True])
def test_logit_convergence_is_invariant_to_covariate_units(scale, batch_size, weighted):
    # These empirical success rates give the exact MLE logit(.8) at x=1.
    # Raw slope steps are tiny when x is huge, even far from the solution.
    n = 1000
    x = np.repeat([-1.0, 0.0, 1.0], n)
    X = np.column_stack([np.ones(len(x)), x * scale])
    d = np.concatenate([np.arange(n) < count for count in [200, 500, 800]])
    weights = np.repeat([0.01, 1.0, 100.0], n) if weighted else np.ones(len(x))

    def batches():
        for start in range(0, len(x), batch_size):
            stop = start + batch_size
            yield (
                X[start:stop],
                d[start:stop],
                np.zeros((len(d[start:stop]), 1)),
                weights[start:stop],
            )

    models, passes = fit_systems(batches, [lambda batch: batch], "ipw", ["Intercept", "x"])
    assert_allclose(models[0].beta * [1, scale], [0.0, np.log(4.0)], atol=1e-9)
    assert passes <= 20


def test_quasi_separation_with_opposing_labels_at_boundary_is_refused():
    # x=0 has both labels, so strict separation's positive-margin check cannot
    # detect this design. The slope nevertheless has no finite MLE.
    x = np.repeat([-1.0, 0.0, 1.0], 100)
    d = np.r_[np.zeros(100, dtype=bool), np.arange(100) < 50, np.ones(100, dtype=bool)]
    X = np.column_stack([np.ones(len(x)), x])

    def batches():
        yield X, d, np.zeros((len(x), 1))

    with pytest.raises(ValueError, match="quasi-separation|singular|separation"):
        fit_systems(batches, [lambda b: b], "ipw", ["Intercept", "x"])


def test_large_logit_matches_known_solution_across_batches():
    # Exact success rates .3/.5/.7 at x=-1/0/1 give a known, non-separated MLE.
    # The final Newton improvement is tiny compared with the summed loss.
    per_level = 200000
    x = np.repeat([-1.0, 0.0, 1.0], per_level)
    X = np.column_stack([np.ones(len(x)), x])
    d = np.concatenate([np.arange(per_level) < count for count in [60000, 100000, 140000]])
    estimates = []
    for batch_size in [137, 4096]:

        def batches(batch_size=batch_size):
            for start in range(0, len(x), batch_size):
                stop = start + batch_size
                yield X[start:stop], d[start:stop], np.zeros((len(d[start:stop]), 1))

        models, passes = fit_systems(
            batches, [lambda batch: batch], "ipw", ["Intercept", "x"], maxiter=10
        )
        estimates.append(models[0].beta)
        assert passes <= 10
        assert_allclose(models[0].beta, [0.0, np.log(7 / 3)], atol=1e-9)
    assert_allclose(estimates[0], estimates[1], atol=1e-9)
