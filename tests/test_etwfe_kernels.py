"""The compiled extended TWFE passes give the array path's numbers."""

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from diff_stream import did_extended_twfe
from diff_stream._core import etwfe_kernels
from diff_stream.simulate import simulate_conditional_panel


@pytest.fixture(scope="module")
def panels():
    p = simulate_conditional_panel(900, 7, seed=17).with_columns(
        (0.4 + pl.col("id") % 7).alias("w")
    )
    keep = np.random.default_rng(1).random(p.height) > 0.12
    return p, p.filter(pl.Series(keep))


def fit(data, enabled, **options):
    old = etwfe_kernels.ENABLED
    etwfe_kernels.ENABLED = enabled
    try:
        fml = options.pop("fml", "y ~ x + z")
        return did_extended_twfe(
            fml, data, panel=("id", "time"), cohort="first_treat", batch_units=97, **options
        )
    finally:
        etwfe_kernels.ENABLED = old


CASES = [
    dict(),
    dict(fml="y ~ 1"),
    dict(fml="y ~ x"),
    dict(treat_sample_as_fixed=True),
    dict(fixed_effects="unit"),
    dict(fixed_effects="unit", fml="y ~ 1"),
    dict(control_group="not_yet_treated"),
    dict(control_group="not_yet_treated", fixed_effects="unit", treat_sample_as_fixed=True),
    dict(weights="w"),
    dict(anticipation=1),
    dict(incomplete_units="keep"),
    dict(incomplete_units="keep", fixed_effects="unit", weights="w", fml="y ~ x"),
    dict(vcov="CRV1"),
]


@pytest.mark.parametrize("options", CASES, ids=lambda o: "-".join(f"{k}={v}" for k, v in o.items()))
def test_extended_twfe_matches_the_array_path(panels, options):
    data = panels[1] if options.get("incomplete_units") == "keep" else panels[0]
    a = fit(data, False, **options)
    b = fit(data, True, **options)
    assert a.coefnames == b.coefnames
    assert_allclose(b.beta, a.beta, rtol=1e-9, atol=1e-11)
    assert_allclose(b.vcov, a.vcov, rtol=1e-8, atol=1e-12)


def test_results_do_not_depend_on_threads(panels):
    results = []
    for threads in (1, 3, 8):
        r = fit(panels[0], True, n_threads=threads)
        results.append((np.asarray(r.beta), np.asarray(r.vcov)))
    for other in results[1:]:
        for x, y in zip(results[0], other, strict=True):
            assert np.array_equal(x, y)


def test_unconditional_scores_match_the_dense_accumulation():
    """The sparse accumulation of [score, share - c] gives the sums of the
    dense stack, over several batches and cohorts."""
    from hdfe_stream.core.influence import ClusterScoreAccumulator

    from diff_stream.etwfe import UnconditionalScores

    rng = np.random.default_rng(3)
    cells = [(g, t) for g in (2, 3, 5) for t in range(1, 7) if t != g - 1]
    M = len(cells)
    mass = rng.uniform(5, 20, M)
    c = mass / 400
    sparse_acc, dense_acc = UnconditionalScores(c, cells), ClusterScoreAccumulator()
    cohort_cells = {g: [j for j, (cg, _) in enumerate(cells) if cg == g] for g in (2, 3, 5)}
    for n in (70, 33, 120):
        cohort = rng.choice([0, 2, 3, 5], n)
        score = rng.standard_normal((n, M))
        rows, index, weight = [], [], []
        for u in range(n):
            for j in cohort_cells.get(int(cohort[u]), []):
                if rng.random() < 0.8:  # a unit can miss a cell's period
                    rows.append(u)
                    index.append(j)
                    weight.append(rng.uniform(0.5, 2))
        rows, index, weight = np.array(rows), np.array(index), np.array(weight)
        share = np.zeros((n, M))
        share[rows, index] = weight
        sparse_acc.add(score, cohort, rows, index, weight)
        dense_acc.add_clusters(score, share - c)
    assert_allclose(sparse_acc.sums, dense_acc.sums, rtol=1e-11, atol=1e-11)
    assert sparse_acc.n_clusters == dense_acc.n_clusters
