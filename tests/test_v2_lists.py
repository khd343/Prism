"""
test_v2_lists.py — Kite v2's lists (2026-10-17): the 🚀 Playbook list and the rank list from ONE PRISM commit
(Kite-v2 design §3.1, addendum A5), written once to the `v2-lists` branch by ui/v2_lists_job.py.

The rank list is every scored stock with the columns Kite v2 reads. Its momentum percentile must read exactly as the
🚀 recipe reads it, its two result dates come from days_from_result (measured 15 Oct: session + days equals the
vendor's own Next Result Date on 326 of 326 stocks), and a value it cannot vouch for is written empty, never guessed.

Run with: pytest tests/test_v2_lists.py -v
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import SHARIAH_STATUSES  # noqa: E402
from ui import playbook_feed_job as F  # noqa: E402
from ui import ui_scans as S  # noqa: E402
from ui import v2_lists_job as V  # noqa: E402

SESSION = "2026-10-15"


@pytest.fixture(autouse=True)
def _small_lists(monkeypatch):
    """The real bounds are 1,800-3,500 stocks (and 100-450 Playbook members); the toy market has 30."""
    monkeypatch.setattr(V, "MIN_RANK_ROWS", 1)
    monkeypatch.setattr(F, "MIN_ROWS", 1)


def _scored(n=30, members=10):
    """The feed tests' toy market (tests/test_playbook_feed.py _scored: the last `members` stocks pass every Playbook
    leg) plus what the rank list reads: 9 Shariah of 30 (30%, inside the measured band), a stop under each close, and
    a past result 20 days back."""
    i = np.arange(n, dtype=float)
    m = i >= n - members
    status = (["Shariah"] * 9 + ["Non-Shariah"] * 12 + ["Review"] * 5 + ["Doubtful"] * 4) * (n // 30 + 1)
    return pd.DataFrame({
        "company_id": [f"NSE:S{k:02d}" for k in range(n)],
        "name": [f"Stock {k:02d}" for k in range(n)],
        "sector": "Alpha", "industry": "Widgets",
        "market_category": np.where(i % 2 == 0, "Small Cap", "Mid Cap"),
        "weinstein_stage": "📈 Stage 2 Advancing",
        "composite_score": 40 + i, "crs_52w": 1 + i / 100, "momentum_score": i, "breakout_score": i,
        "improvement_score": i, "dist_to_vstop": np.where(m, 5.0, -1.0),
        "traded_value_cr": 2.5, "rsi_14d": 60.0, "close_price": 100.0 + i, "equity_shares": 1e7 + i,
        "rev_pyq": 100.0, "rev_lq": np.where(m, 130.0, 101.0),
        "pat_pyq": 10.0, "pat_lq": np.where(m, 15.0, 10.0),
        "ebitda_pyq": 15.0, "ebitda_lq": np.where(m, 26.0, 15.0),
        "pat": 12.0, "pat_1yb": 10.0, "revenue": 110.0, "revenue_1yb": 100.0, "opm": 15.0, "opm_1yb": 14.0,
        "shariah_status": status[:n], "vstop_value": 90.0 + i, "days_from_result": -20.0,
    })


def _rank(scored=None, previous=None, session=SESSION):
    frame, stats = V.build_rank(_scored() if scored is None else scored, session, previous)
    return frame, stats


def _row(frame, symbol):
    return frame.set_index("symbol").loc[symbol]


def _as_text(frame):
    """A rank list as the next session reads it back from the branch: text, blanks as ''."""
    return pd.read_csv(io.BytesIO(V.rank_csv_bytes(frame)), dtype=str, keep_default_na=False)


# ── 1. The contract ───────────────────────────────────────────────────────────────────────────────
def test_columns_are_the_contract_in_order():
    frame, _ = _rank()
    assert list(frame.columns) == ["symbol", "shariah_status", "cap", "traded_value_cr", "momentum_score",
                                   "momentum_pct", "vstop_value", "close_price", "sector", "stage",
                                   "last_result_date", "next_result_date", "equity_shares"]
    assert V.RANK_COLUMNS == tuple(frame.columns)


def test_one_row_per_scored_stock_sorted_by_symbol():
    df = _scored()
    frame, _ = _rank(df.iloc[::-1])
    assert len(frame) == len(df)
    assert frame["symbol"].tolist() == sorted(df["company_id"])


def test_a_missing_source_column_fails():
    with pytest.raises(F.FeedError, match="days_from_result"):
        _rank(_scored().drop(columns="days_from_result"))


# ── 2. The momentum percentile is the recipe's own ────────────────────────────────────────────────
def test_momentum_pct_is_the_recipes_own_percentile():
    df = _scored()
    frame, _ = _rank(df)
    want = pd.Series(S._pct(df, "momentum_score").to_numpy(), index=df["company_id"])
    got = frame.set_index("symbol")["momentum_pct"].astype(float)
    assert (got == want.reindex(got.index)).all()


def test_momentum_pct_round_trips_exactly():
    df = _scored()
    frame, _ = _rank(df)
    back = _as_text(frame).set_index("symbol")["momentum_pct"].astype(float)
    want = pd.Series(S._pct(df, "momentum_score").to_numpy(), index=df["company_id"])
    assert (back == want.reindex(back.index)).all()


def test_every_member_reads_at_or_above_two_thirds():
    df = _scored()
    members = df.loc[S.playbook_shortlist(df), "company_id"]
    back = _as_text(_rank(df)[0]).set_index("symbol")["momentum_pct"].astype(float)
    assert len(members) == 10 and (back.loc[members] >= 2 / 3).all()


# ── 3. The two result dates ───────────────────────────────────────────────────────────────────────
def _with_days(days):
    df = _scored()
    df.loc[:len(days) - 1, "days_from_result"] = days
    return df


def test_next_result_is_session_plus_days_when_not_past():
    frame, _ = _rank(_with_days([0.0, 5.0, -3.0]))
    assert _row(frame, "NSE:S00")["next_result_date"] == "2026-10-15"
    assert _row(frame, "NSE:S01")["next_result_date"] == "2026-10-20"
    assert _row(frame, "NSE:S02")["next_result_date"] == ""


def test_last_result_is_derived_when_past_or_today():
    frame, _ = _rank(_with_days([-20.0, 0.0]))
    assert _row(frame, "NSE:S00")["last_result_date"] == "2026-09-25"
    assert _row(frame, "NSE:S01")["last_result_date"] == "2026-10-15"


def test_an_announced_or_blank_day_carries_the_last_result_forward():
    yesterday = _as_text(_rank(_with_days([-30.0, -40.0, -50.0]), session="2026-10-14")[0])
    frame, stats = _rank(_with_days([6.0, np.nan, -1.0]), previous=yesterday)
    assert _row(frame, "NSE:S00")["last_result_date"] == "2026-09-14"        # announced: carried
    assert _row(frame, "NSE:S01")["last_result_date"] == "2026-09-04"        # blank: carried
    assert _row(frame, "NSE:S02")["last_result_date"] == "2026-10-14"        # reported: derived, not carried
    assert stats["carried"] == 2
    alone, alone_stats = _rank(_with_days([6.0]))
    assert _row(alone, "NSE:S00")["last_result_date"] == "" and alone_stats["carried"] == 0


def test_a_carry_never_takes_an_empty_value_as_a_date():
    yesterday = _as_text(_rank(_with_days([7.0]), session="2026-10-14")[0])   # S00 had no known last result
    frame, stats = _rank(_with_days([8.0]), previous=yesterday)
    assert _row(frame, "NSE:S00")["last_result_date"] == "" and stats["carried"] == 0


def test_a_non_integral_day_is_unknown():
    frame, stats = _rank(_with_days([-2.5, 3.4]))
    assert _row(frame, "NSE:S00")["last_result_date"] == "" and _row(frame, "NSE:S00")["next_result_date"] == ""
    assert _row(frame, "NSE:S01")["next_result_date"] == ""
    assert stats["unknown_days"] == 2


# ── 4. Labels: codes, never guesses ───────────────────────────────────────────────────────────────
def test_an_unknown_status_refuses_the_list():
    df = _scored()
    df.loc[3, "shariah_status"] = "Halal-ish"
    with pytest.raises(F.FeedError, match="Halal-ish"):
        _rank(df)


def test_an_unknown_cap_or_stage_label_refuses_the_list():
    df = _scored()
    df.loc[3, "market_category"] = "Giga Cap"
    with pytest.raises(F.FeedError, match="Giga Cap"):
        _rank(df)
    df = _scored()
    df.loc[3, "weinstein_stage"] = "Stage 5"
    with pytest.raises(F.FeedError, match="Stage 5"):
        _rank(df)


def test_a_blank_status_is_written_empty():
    df = _scored()
    df.loc[3, "shariah_status"] = np.nan
    df.loc[4, "shariah_status"] = "  "
    frame, _ = _rank(df)
    assert _row(frame, "NSE:S03")["shariah_status"] == "" and _row(frame, "NSE:S04")["shariah_status"] == ""
    assert set(frame["shariah_status"]) <= set(SHARIAH_STATUSES) | {""}


# ── 5. Validation and the input checks ────────────────────────────────────────────────────────────
def test_validate_rank_refuses_duplicates_bad_symbols_and_out_of_bounds(monkeypatch):
    frame, _ = _rank()
    V.validate_rank(frame)
    dup = frame.copy()
    dup.loc[1, "symbol"] = dup.loc[0, "symbol"]
    with pytest.raises(F.FeedError, match="duplicate"):
        V.validate_rank(dup)
    bad = frame.copy()
    bad.loc[1, "symbol"] = "XNSE S01"
    with pytest.raises(F.FeedError, match="malformed"):
        V.validate_rank(bad)
    monkeypatch.setattr(V, "MIN_RANK_ROWS", 31)
    with pytest.raises(F.FeedError, match="30 stocks"):
        V.validate_rank(frame)


def test_the_real_bounds_are_the_measured_ones():
    assert V._MEASURED_RANK_BOUNDS == (1800, 3500)            # the fixture lowers MIN_RANK_ROWS, never this
    assert V.MAX_RANK_ROWS == 3500
    assert (V.SHARIAH_SHARE_MIN, V.SHARIAH_SHARE_MAX) == (0.10, 0.40)


def _check(df):
    return {r.name: r for r in V.check_rank_inputs(df)}


def test_check_rank_inputs_fails_below_a_floor_and_on_an_absent_column():
    assert all(r.status == "ok" for r in V.check_rank_inputs(_scored()))
    thin = _scored()
    thin.loc[:3, "vstop_value"] = np.nan                       # 26 of 30 filled: 86.7%, under the 0.90 floor
    res = _check(thin)["Rank list inputs"]
    assert res.status == "fail" and "vstop_value" in res.detail
    absent = _scored().drop(columns="traded_value_cr")
    res = _check(absent)["Rank list inputs"]
    assert res.status == "fail" and "traded_value_cr" in res.detail
    unknown = _scored()
    unknown.loc[:4, "weinstein_stage"] = "❔ Unknown"            # 25 of 30 known: 83.3%, under the 0.85 floor
    assert _check(unknown)["Rank list inputs"].status == "fail"


def test_check_rank_inputs_fails_outside_the_shariah_band():
    broken = _scored()
    broken["shariah_status"] = "Doubtful"                     # a screen that labelled everything Doubtful
    res = _check(broken)["Shariah share"]
    assert res.status == "fail" and "0.0%" in res.detail
    wide = _scored()
    wide.loc[:20, "shariah_status"] = "Shariah"               # 21 of 30: 70%
    assert _check(wide)["Shariah share"].status == "fail"


# ── 6. Bytes ──────────────────────────────────────────────────────────────────────────────────────
def test_same_input_same_bytes_whatever_the_row_order():
    df = _scored()
    a = V.rank_csv_bytes(_rank(df)[0])
    b = V.rank_csv_bytes(_rank(df.sample(frac=1.0, random_state=7))[0])
    assert a == b and a.endswith(b"\n") and b"\r" not in a


def test_nan_is_written_empty_never_zero():
    df = _scored()
    df.loc[2, "vstop_value"] = np.nan
    df.loc[2, "traded_value_cr"] = np.nan
    row = _as_text(_rank(df)[0]).set_index("symbol").loc["NSE:S02"]
    assert row["vstop_value"] == "" and row["traded_value_cr"] == ""
