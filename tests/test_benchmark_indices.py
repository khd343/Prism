"""
test_benchmark_indices.py
=========================
The seven benchmark indices are not stocks (2026-10-08).

The user added seven NSE indices (Nifty 500, Smallcap 250, Microcap 250, Nifty500 Momentum 50,
Nifty200 Momentum 30, Nifty200 Quality 30, Smallcap250 Momentum Quality 100) to the PRISM watchlists,
so the scans can later be judged against "could I buy the index fund instead?". The sync's own fetch
that day showed what they look like: a row like a stock, technicals filled (price, VSTOP, relative
strength, moving averages), every fundamental — and the sector, industry and market cap — 'null'.

Left in the universe they were ranked as stocks (Microcap 250 at rank 78) and, through every
percentile and median, moved the composite of 2,703 real stocks and the rank of 2,637. Set aside at
the merge, the 2,712 stocks came out identical on all 757 columns to a fetch without the rows.

Pinned here: who counts as a benchmark (declared ids, plus a safety net for an index added later);
that a real stock with one gap is never caught; that the merge — the one place every consumer reads
from — gives exactly the frame it would give without the index rows.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

import core.data_engine as de  # noqa: E402
from config import BENCHMARK_INDICES  # noqa: E402


def test_the_seven_benchmarks_are_declared_by_the_vendors_ids():
    assert BENCHMARK_INDICES == {
        "NSE:CNX500": "Nifty 500",
        "NSE:SMALLCA250": "Nifty Smallcap 250",
        "NSE:NFMICRO250": "Nifty Microcap 250",
        "NSE:NIF500MO50": "Nifty500 Momentum 50",
        "NSE:NI200MOM30": "Nifty200 Momentum 30",
        "NSE:NQUALITY30": "Nifty200 Quality 30",
        "NSE:NS250MQ100": "Nifty Smallcap250 Momentum Quality 100",
    }


def _rows():
    """Two stocks (one with a single gap), a declared index, an undeclared bare row."""
    return pd.DataFrame({
        "company_id": ["NSE:TCS", "BSE:GAPCO", "NSE:CNX500", "NSE:NEWINDEX"],
        "name": ["TCS", "Gap Co", "Nifty 500", "Some New Index"],
        "sector": ["IT Software", np.nan, np.nan, np.nan],
        "industry": ["IT - Software", "Steel", np.nan, np.nan],
        "market_cap": [1.2e6, 450.0, np.nan, np.nan],
    })


def test_declared_ids_and_bare_rows_are_benchmarks_and_a_stock_with_one_gap_is_not():
    assert de.benchmark_rows(_rows()).tolist() == [False, False, True, True]


def test_a_frame_without_the_columns_matches_by_id_only():
    """An archived vintage or a small test frame may lack market_cap / sector / industry: the safety net
    must not read every row of it as 'bare' and empty the universe."""
    df = pd.DataFrame({"company_id": ["NSE:TCS", "NSE:NQUALITY30"], "name": ["TCS", "Nifty200 Quality 30"]})
    assert de.benchmark_rows(df).tolist() == [False, True]


def test_an_unknown_index_like_row_is_named_never_dropped_silently(capsys):
    out = de._set_aside_benchmarks(_rows())
    assert out["company_id"].tolist() == ["NSE:TCS", "BSE:GAPCO"]
    log = capsys.readouterr().out
    assert "2 benchmark index row(s) set aside" in log
    assert "NSE:NEWINDEX" in log and "not in config.BENCHMARK_INDICES" in log
    assert "NSE:CNX500" not in log.split("not in config")[1], "a declared index is not a warning"


def _datasets(with_index: bool):
    """The six tabs as the loader hands them to the merge, in the shape the sync fetched on 2026-10-08:
    an index row carries an id and a name, technicals, and NaN ('null') for everything else."""
    ids = ["NSE:AAA", "NSE:BBB"] + (list(BENCHMARK_INDICES) if with_index else [])
    n_idx = len(ids) - 2
    ratio = pd.DataFrame({"company_id": ids,
                          "name": ["Alpha Ltd", "Beta Ltd"] + [BENCHMARK_INDICES[i] for i in ids[2:]],
                          "industry": ["Steel", "IT - Software"] + [np.nan] * n_idx,
                          "sector": ["Steel", "IT Software"] + [np.nan] * n_idx,
                          "roce": [18.0, 25.0] + [np.nan] * n_idx})
    tech = pd.DataFrame({"company_id": ids, "market_cap": [900.0, 5000.0] + [np.nan] * n_idx,
                         "close_price": [120.0, 800.0] + [21616.15] * n_idx})
    other = {k: pd.DataFrame({"company_id": ids, f"{k}_x": [1.0, 2.0] + [np.nan] * n_idx})
             for k in ("income", "balance", "cashflow", "shareholding")}
    return {"ratio": ratio, "technical": tech, **other}


def test_the_merge_gives_exactly_the_frame_it_gives_without_the_index_rows():
    """The one place every consumer reads from: with the seven rows present, the merged frame equals the
    merge of the same data without them — so nothing downstream can differ."""
    with_rows = de.merge_datasets(_datasets(with_index=True))
    without = de.merge_datasets(_datasets(with_index=False))
    pd.testing.assert_frame_equal(with_rows.reset_index(drop=True), without.reset_index(drop=True))
    assert not with_rows["company_id"].isin(BENCHMARK_INDICES).any()


def test_the_set_aside_runs_before_anything_reads_sector_or_name():
    """It must precede the dual-listing collapse (reads name) and the taxonomy (reads sector)."""
    import inspect
    src = inspect.getsource(de.merge_datasets)
    assert src.index("_set_aside_benchmarks(") < src.index("_collapse_dual_listings(") < src.index(
        "canonicalize_taxonomy(")


def test_no_real_stock_is_bare_in_the_local_data():
    """The safety net is only safe while no real company lacks market cap, sector AND industry at
    once (2026-10-08: none of 2,712). A future vintage where one does fails here, by name, instead of
    the stock vanishing from the universe."""
    try:
        data = de.load_all_csvs("local")
    except Exception as exc:                                     # the CSVs are gitignored
        pytest.skip(f"local data not available: {exc}")
    merged = data["ratio"][["company_id", "sector", "industry"]].merge(
        data["technical"][["company_id", "market_cap"]], on="company_id", how="left")
    stocks = merged[~merged["company_id"].astype(str).isin(BENCHMARK_INDICES)]
    bare = stocks[de.benchmark_rows(stocks)]
    assert bare.empty, f"real stocks with no market cap, sector or industry: {bare['company_id'].tolist()}"
