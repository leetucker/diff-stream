"""Exact identities between the estimators (balanced panel, no covariates).

docs/choosing.md states these; they follow from the estimators' definitions.
"""

from numpy.testing import assert_allclose

from diff_stream import (
    did_bjs,
    did_callaway_santanna,
    did_extended_twfe,
    did_sun_abraham,
    did_two_stage,
)
from diff_stream.simulate import simulate_panel

KW = dict(panel=("id", "time"), cohort="first_treat", n_threads=2)
DATA = simulate_panel(600, 7, seed=11)


def matched(a, b):
    common = [c for c in a.cells if c in b.cells]
    assert common
    return (
        a.beta[[a.cells.index(c) for c in common]],
        b.beta[[b.cells.index(c) for c in common]],
    )


def test_sun_abraham_is_callaway_santanna_with_universal_base():
    sa = did_sun_abraham("y ~ 1", DATA, **KW)
    cs = did_callaway_santanna("y ~ 1", DATA, base_period="universal", **KW)
    assert_allclose(*matched(sa, cs), atol=1e-12, rtol=0)


def test_two_stage_event_study_is_bjs_imputation():
    event = did_bjs("y ~ 1", DATA, **KW).aggregate()
    two_stage = did_two_stage("y ~ 1", DATA, **KW)
    names = [n for n in two_stage.coefnames if n in event.coefnames]
    assert len(names) == len(event.coefnames)
    assert_allclose(
        two_stage.beta[[two_stage.coefnames.index(n) for n in names]],
        event.beta[[event.coefnames.index(n) for n in names]],
        atol=1e-12,
        rtol=0,
    )


def test_extended_twfe_with_not_yet_treated_controls_is_bjs_imputation():
    etwfe = did_extended_twfe("y ~ 1", DATA, control_group="not_yet_treated", **KW)
    bjs = did_bjs("y ~ 1", DATA, **KW)
    assert etwfe.cells == bjs.cells
    assert_allclose(*matched(etwfe, bjs), atol=1e-11, rtol=0)
