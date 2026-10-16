"""
test_qmom_gray_momentum.py
==========================
Contract for the 2026-10-16 fidelity fix of Quality Momentum (fw_qmom; Gray & Vogel, Quantitative Momentum).

Gray's generic momentum (Ch.5) is the "2-12" return: the last 12 months SKIPPING the most recent month,
(1 + R12) / (1 + R1) - 1, and his recipe (Ch.8) keeps the TOP DECILE of a liquid universe on it. PRISM could not
compute it (the six tabs carried no 1-year or 1-month return), so fw_qmom's primary gate was a proxy: the top 20%
of d47_rs_composite, a blend of the vendor's CRS STREAK counters (how long a stock has outperformed, not by how
much). The operator added "Returns 1M" and "Returns 1Y" to the Technicals tab on 15 Oct 2026, so the book's measure
replaces the proxy: gray_mom_2_12 (data_engine, D52) and, in fw_qmom, its top decile among the framework's own
universe (market cap >= Rs 500 Cr). Every other leg is unchanged: all three CRS readings positive (the stand-in for
Gray's frog-in-the-pan smoothness, which needs daily returns), ROCE, D/E, CFO/PAT, pledge, size.

A copy without the two columns (every vintage before 15 Oct) cannot show 2-12 momentum, so nobody passes there:
unverifiable is not passed (CLAUDE.md §5). Quality Momentum feeds no score; it is a badge and a filter.

Run with: pytest tests/test_qmom_gray_momentum.py -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import pytest

from test_data_quality_fixes import _frame
from data_engine import TECHNICAL_COLS, compute_derived_signals


# ── 1. The two returns are read, and 2-12 momentum is built from them ───────────────────────────────
def test_the_two_returns_are_mapped_from_the_technicals_tab():
    assert TECHNICAL_COLS.get("Returns 1M") == "ret_1m"
    assert TECHNICAL_COLS.get("Returns 1Y") == "ret_1y"


def test_two_twelve_momentum_skips_the_most_recent_month():
    """+50% over the year with +20% of it in the last month = +25% from 12 months ago to 1 month ago; +10% with a
    -10% last month = +22.2%. Either return missing = no 2-12 return (a stock listed under a year has none)."""
    out = compute_derived_signals(_frame(n=4, ret_1y=[50.0, 10.0, np.nan, 50.0], ret_1m=[20.0, -10.0, 5.0, np.nan]))
    assert "gray_mom_2_12" in out.columns
    got = out["gray_mom_2_12"].to_numpy()
    assert got[0] == pytest.approx(25.0) and got[1] == pytest.approx(100.0 * (1.1 / 0.9 - 1.0))
    assert np.isnan(got[2]) and np.isnan(got[3])


def test_a_copy_without_the_columns_has_no_momentum_and_does_not_crash():
    frame = _frame(n=3).drop(columns=["ret_1m", "ret_1y"], errors="ignore")
    out = compute_derived_signals(frame)
    assert "gray_mom_2_12" in out.columns and out["gray_mom_2_12"].isna().all()


# ── 2. The gate ─────────────────────────────────────────────────────────────────────────────────────
def _row(cid, mom, **overrides) -> dict:
    """A company that clears every Quality Momentum leg except, possibly, the momentum decile."""
    row = {
        "company_id": cid, "gray_mom_2_12": mom, "d47_rs_composite": 0.0, "market_cap": 1000.0,
        "crs_aligned": 1, "roce": 20.0, "debt_to_equity": 0.2, "is_financial": False, "cfo_to_pat": 90.0,
        "pledged_percentage": 0.0,
        # read elsewhere in compute_qglp_score
        "roce_med_10y": 20.0, "roce_med_5y": 20.0, "roe": 18.0, "pat_gr_5y": 20.0, "eps_gr_5y": 18.0,
        "peg": 1.0, "forensic_score": 90.0, "forensic_multiplier": 1.0, "piotroski_fscore": 7,
        "d35_roce_trend": 1.0, "fcf_yield": 6.0,
    }
    row.update(overrides)
    return row


def _universe(extra=()):
    """40 qualifying companies with 2-12 momentum 1..40: the top decile is the four at 37..40."""
    return [_row(f"U{i:02d}", float(i + 1)) for i in range(40)] + list(extra)


def _passers(rows) -> list:
    from config import MASTER_PROFILES
    from core.scoring_engine import compute_qglp_score

    df = pd.DataFrame(rows)
    df.attrs["detected_market_regime"] = "SIDEWAYS"
    out = compute_qglp_score(df, MASTER_PROFILES["Balanced"])
    hit = out["frameworks_passed"].str.split(", ").apply(lambda parts: "Quality Momentum" in parts)
    return sorted(out.loc[hit, "company_id"])


def test_the_top_decile_of_two_twelve_momentum_within_the_universe_passes():
    """A Rs 300 Cr company with the best momentum is outside the universe: it neither passes nor takes a slot."""
    small = _row("SMALL", 100.0, market_cap=300.0)
    assert _passers(_universe([small])) == ["U36", "U37", "U38", "U39"]


def test_no_momentum_figure_never_passes():
    assert "NOMOM" not in _passers(_universe([_row("NOMOM", np.nan)]))


def test_the_streak_proxy_no_longer_decides():
    """The old gate (top 20% of d47_rs_composite) would pick the companies carrying the highest composite; the
    book's measure picks by 2-12 momentum, whatever the streaks say."""
    rows = _universe()
    for r in rows[:8]:
        r["d47_rs_composite"] = 99.0
    assert _passers(rows) == ["U36", "U37", "U38", "U39"]


def test_the_other_legs_still_apply():
    rows = _universe()
    rows[-1]["crs_aligned"] = 0          # the strongest 2-12 momentum, but not all three CRS readings positive
    rows[-2]["pledged_percentage"] = 45.0
    assert _passers(rows) == ["U36", "U37"]


def test_a_copy_without_two_twelve_momentum_has_no_passer():
    rows = _universe()
    for r in rows:
        r["gray_mom_2_12"] = np.nan
        r["d47_rs_composite"] = 99.0
    assert _passers(rows) == []
