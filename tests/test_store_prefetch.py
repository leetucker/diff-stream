"""The store reads its next batch in a background thread."""

import threading

import numpy as np
import pytest

import diff_stream.store as store_module
from diff_stream import did_bjs
from diff_stream.simulate import simulate_conditional_panel
from diff_stream.store import PanelStore


def bare(source):
    store = object.__new__(PanelStore)
    store._batches = lambda **options: source()
    return store


def test_batches_come_in_order_whatever_the_setting(monkeypatch):
    def source():
        for i in range(7):
            yield (np.full(3, i),)

    for flag in (False, True):
        monkeypatch.setattr(store_module, "PREFETCH", flag)
        assert [int(b[0][0]) for b in bare(source).batches()] == list(range(7))


def test_the_next_batch_is_read_in_another_thread():
    seen = []

    def source():
        for i in range(3):
            seen.append(threading.current_thread() is threading.main_thread())
            yield i

    assert list(bare(source).batches()) == [0, 1, 2]
    assert not any(seen)


def test_stopping_early_closes_the_source():
    closed = []

    def source():
        try:
            yield from range(10)
        finally:
            closed.append(True)

    for _ in bare(source).batches():
        break
    assert closed == [True]


def test_errors_reach_the_consumer():
    def source():
        yield 1
        raise ValueError("unreadable batch")

    it = bare(source).batches()
    assert next(it) == 1
    with pytest.raises(ValueError, match="unreadable batch"):
        next(it)


def test_fits_do_not_depend_on_it(monkeypatch):
    panel = simulate_conditional_panel(300, 6, seed=3)
    results = []
    for flag in (False, True):
        monkeypatch.setattr(store_module, "PREFETCH", flag)
        r = did_bjs("y ~ x + z", panel, panel=("id", "time"), cohort="first_treat", batch_units=50)
        results.append((np.asarray(r.beta), np.asarray(r.vcov)))
    assert np.array_equal(results[0][0], results[1][0])
    assert np.array_equal(results[0][1], results[1][1])
