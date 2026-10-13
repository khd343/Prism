"""
test_scan_order.py — the order inside the 🚀 and 🌬️ tables (2026-10-13, the operator's call).

🚀 Playbook and 🌬️ Leading-Sector list by momentum, high to low, in both views; 🌱 keeps PRISM Rank (its
3-year study found momentum inside it flips year to year). A display order only: it changes no list, no
record and no fingerprint, and the Playbook feed keeps its own order (rows by symbol).

Measured before the call (docs/december-2026-plan.md rule 26 E-F): on windows that do not overlap, the top
25 by momentum sat at the 93/63/83rd percentile of random picks inside the Playbook and 68/69/79 inside
Leading-Sector; the top 25 by Rank at 38/33/73 and 38/60/46. ORDER_RULE, written the same day to decide
that switch, now CHECKS the order in use with its bar unchanged (tests/test_scan_hold_reading.py).

Run with: pytest tests/test_scan_order.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ui import ui_scans as S  # noqa: E402

_BY_KEY = {s["key"]: s for s in S.SCANS}


def _frame():
    """Eight matches. Two share momentum 70: E has the better Rank, though C comes first in the frame. D has
    no momentum reading but the second-best Rank."""
    return pd.DataFrame({
        "company_id": [f"NSE:{c}" for c in "ABCDEFGH"], "name": [f"Stock {c}" for c in "ABCDEFGH"],
        "sector": "Alpha", "composite_score": 50.0, "vol_sma_20d": 200_000.0, "close_price": 500.0,
        "momentum_score": [40, 90, 70, np.nan, 70, 10, 95, 55],
        "rank": [1, 8, 6, 2, 3, 4, 7, 5]})


def _every(key):
    """The real scan's settings over every row, so the order is tested apart from its recipe."""
    return dict(_BY_KEY[key], mask=lambda d: pd.Series(True, index=d.index))


def _ids(t):
    return "".join(c.split(":")[1] for c in t["company_id"])


def test_playbook_and_leading_sector_list_by_momentum_high_to_low():
    """A tie goes to the better Rank (E before C); a stock with no momentum reading goes last, whatever
    its Rank."""
    for key in ("playbook", "leaders"):
        assert _ids(S.scan_table(_frame(), _every(key))) == "GBECHAFD", key


def test_the_compounder_keeps_rank_order():
    assert _ids(S.scan_table(_frame(), _every("compounder"))) == "ADEFHCGB"


def test_the_order_check_measures_exactly_the_order_in_use():
    momentum = sorted(k for k, s in _BY_KEY.items() if S.momentum_ordered(s))
    assert momentum == ["leaders", "playbook"] == sorted(S.ORDER_RULE["scans"]), (
        "the scans listed by momentum are the ones whose order the card checks — no more, no fewer")


def test_the_sorted_column_is_on_screen_in_both_views():
    """Sort-by-visible (app.py, 2026-10-05): whatever a table is sorted by is on screen — and Rank stays a
    column in every view, so one click sorts by it."""
    for key, scan in _BY_KEY.items():
        for view in S.SCAN_VIEWS:
            cols = S.table_columns(scan, view)
            assert "rank" in cols and ("momentum_score" in cols or not S.momentum_ordered(scan)), (key, view)


def test_the_order_is_display_only():
    """It moves no record: the fingerprint does not follow it."""
    for s in S.SCANS:
        flipped = dict(s, order=None if S.momentum_ordered(s) else "momentum")
        assert S.recipe_fingerprint(flipped) == S.recipe_fingerprint(s), s["key"]


def test_the_rank_tip_says_what_the_list_is_sorted_by():
    for key in ("playbook", "leaders"):
        tip = S.table_config("x", _BY_KEY[key])["rank"]["help"]
        assert "sorted by Momentum, high to low" in tip and "click this header" in tip, key
        assert "for reading only" not in tip, key
    tip = S.table_config("x", _BY_KEY["compounder"])["rank"]["help"]
    assert "sorted by it for reading only" in tip and "Momentum" not in tip
