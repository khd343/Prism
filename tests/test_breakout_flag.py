"""
test_breakout_flag.py
=====================
Contract for the 🚀/Momentum breakout flag (2026-10-15).

THE BUG. StockScans' "Breakout Window" = "Number of days since the last close was greater than or equal to today's
close" (its own column dictionary): never below 15 (the vendor's floor), blank when no earlier close was that high.
_compute_breakout_score read it as `> 0` — true for 92-96% of stocks (anything with a value) and FALSE for the
blanks, which are the stocks in all-time-high territory (every blank's 52-week high is its all-time high).

THE FIX. A breakout is StockScans' own two scans joined: a window of BREAKOUT_MIN_WINDOW_DAYS (40) days or more, or
blank, AND the close within BREAKOUT_MAX_HIGH_GAP_PCT (10%) of the 52-week high. config.py has the measurement and
the three deliberate differences from the vendor's scans.

Run with: pytest tests/test_breakout_flag.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.scoring_engine import _compute_breakout_score, breakout_flag  # noqa: E402


@pytest.mark.parametrize("window,gap,expected", [
    (15.0, 5.0, False),     # the vendor's floor: 15 days or fewer, no base broken
    (39.0, 5.0, False),
    (40.0, 5.0, True),      # StockScans' short-term band starts at 40
    (250.0, 10.0, True),    # long-term band, exactly 10% below the 52-week high
    (250.0, 10.01, False),
    (np.nan, 3.0, True),    # blank: no earlier close that high — the longest base of all
    (np.nan, 15.0, False),
])
def test_breakout_is_a_long_enough_base_near_the_52_week_high(window, gap, expected):
    df = pd.DataFrame({"breakout_window": [window], "dist_52wh": [gap]})
    assert bool(breakout_flag(df).iloc[0]) is expected


def test_blank_without_distance_is_not_a_breakout():
    """Unverifiable is not passed: a blank window with no 52-week distance proves nothing."""
    df = pd.DataFrame({"breakout_window": [np.nan, 100.0], "dist_52wh": [np.nan, np.nan]})
    assert breakout_flag(df).tolist() == [False, False]


def test_the_flag_is_worth_the_breakout_window_weight():
    df = pd.DataFrame({"breakout_window": [100.0, 15.0], "dist_52wh": [5.0, 5.0],
                       "dist_52wh_days": [10.0, 10.0], "dist_13wh": [2.0, 2.0], "dist_ath": [5.0, 5.0]})
    s = _compute_breakout_score(df)
    assert round(s.iloc[0] - s.iloc[1], 6) == 20.0
