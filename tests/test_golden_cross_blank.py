"""
test_golden_cross_blank.py
==========================
Contract for the golden-cross part of the trend score (2026-10-15).

StockScans' "Last Goldencrossover 50D 200D" = "Trading days since EMA 50D crossed above EMA 200D (1 = latest day)".
A blank therefore means no golden cross IN FORCE — 76% of blanks (9 Oct, both averages present) have the 50D below
the 200D. The trend score filled every blank with a neutral 50, which ranks a stock in a death cross above an
uptrend whose cross is old. With both averages present a blank now scores lowest (0); without them the stock's
history is too short to tell (a new listing) and it stays neutral.

Run with: pytest tests/test_golden_cross_blank.py -v
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TREND_SIGNALS  # noqa: E402
from core.scoring_engine import _compute_trend_score  # noqa: E402


def test_blank_golden_cross_is_lowest_only_with_history():
    df = pd.DataFrame({"golden_cross_days": [10.0, np.nan, np.nan],
                       "sma_50d": [100.0, 90.0, np.nan], "sma_200d": [95.0, 100.0, np.nan]})
    s = _compute_trend_score(df)
    w = TREND_SIGNALS["golden_cross"]
    assert s.iloc[0] == 100 * w          # the only cross: the top rank
    assert s.iloc[1] == 0                # a blank with both averages: no cross in force
    assert s.iloc[2] == 50 * w           # a blank without the history: unknown, neutral
