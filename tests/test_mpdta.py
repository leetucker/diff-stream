"""Offline replication of the documented did mpdta empirical specifications."""

import polars as pl
import pytest

from benchmarks._files import read_csv
from benchmarks.mpdta_replication import DATA, SCENARIOS, compare_case


@pytest.fixture(scope="module")
def mpdta():
    return read_csv(DATA)


def test_published_data_snapshot(mpdta):
    assert len(mpdta) == 2500
    assert mpdta["countyreal"].n_unique() == 500
    assert sorted(mpdta["year"].unique().to_list()) == list(range(2003, 2008))
    assert sorted(mpdta["first.treat"].unique().to_list()) == [0, 2004, 2006, 2007]
    assert mpdta.select(pl.struct("countyreal", "year").n_unique()).item() == 2500
    assert mpdta.group_by("countyreal").len()["len"].to_list() == [5] * 500
    assert mpdta.null_count().sum_horizontal().item() == 0
    assert (mpdta["treat"] == (mpdta["first.treat"] > 0)).all()
    # Original package column names/calendar values remain intact.
    assert mpdta.columns == ["year", "countyreal", "lpop", "lemp", "first.treat", "treat"]


@pytest.mark.parametrize("design,control,base,method", SCENARIOS)
def test_r_mpdta_replication(mpdta, design, control, base, method):
    result, rows = compare_case(mpdta, design, control, base, method)
    assert result["status"] == "passed"
    assert set(row["aggregation"] for row in rows) == {
        "dynamic",
        "simple",
        "cohort",
        "calendar",
        "balanced",
    }
