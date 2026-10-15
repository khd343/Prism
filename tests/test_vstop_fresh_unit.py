"""
test_vstop_fresh_unit.py
========================
Contract for the 2026-10-15 unit fix of `vstop_fresh` (core/data_engine.py).

THE BUG. `vstop_fresh` is documented as "the V-Stop changed within the last 30 days" (config.TREND_SIGNALS),
but the vendor column it reads, "Last VSTOP Change 14W 2.5" (`last_vstop_change`), counts WEEKLY BARS since
the V-Stop last flipped, not days. Measured on the Daily workbooks: from 5 Oct to 12 Oct the count rose by
exactly 1 for 2,445 of 2,710 stocks, and a brand-new flip reads 1. The old `<= 30` therefore meant "flipped
within 30 WEEKS" (about seven months) and fired for 75% of the market, a binary signal above the 60% noise
line that the 2026-06-12 census accepted as "flip-frequency reality" without measuring the unit.

THE FIX. Fresh = flipped within config.VSTOP_FRESH_MAX_WEEKS = 4 weekly bars: the flip bar and the three
before it, the largest whole-week window that never exceeds the documented 30 days (4 x 7 = 28; 5 x 7 = 35).
Live sheet of 2026-10-14: 14.8% of stocks. A missing count is no evidence of a flip, so it is not fresh.

Run with: pytest tests/test_vstop_fresh_unit.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

import config
from test_data_quality_fixes import _frame
from data_engine import compute_derived_signals
from scoring_engine import detect_catalysts_and_tsunami


def test_fresh_means_flipped_within_four_weekly_bars():
    """1 = flipped this week, 4 = three weeks before that. A count of 5 can be 35 days old and 30 is seven
    months, so neither is fresh; a missing count proves nothing. Fails if the count is read as days again."""
    out = compute_derived_signals(_frame(n=6, last_vstop_change=[1.0, 4.0, 5.0, 18.0, 30.0, np.nan]))
    assert out["vstop_fresh"].tolist() == [1, 1, 0, 0, 0, 0]


def test_fresh_window_is_the_largest_whole_week_span_inside_30_days():
    """Pins the documented meaning (within 30 days) to the column's real unit (weekly bars)."""
    weeks = config.VSTOP_FRESH_MAX_WEEKS
    assert weeks * 7 <= 30 < (weeks + 1) * 7


def test_tsunami_needs_a_flip_inside_the_corrected_window():
    """End to end, from the vendor count: an uptrend that turned green 4 weekly bars ago is a fresh breakout,
    one that turned 5 bars ago is not. Every other Tsunami condition is aligned on both rows."""
    derived = compute_derived_signals(_frame(n=2, last_vstop_change=[4.0, 5.0]))
    frame = derived.assign(gate_pass=1, above_sma200=1, vstop_green=1, promoter_buying=1,
                           change_fii_lq=0.5, quality_score=80.0, crs_aligned=1, market_cap=3000.0)
    out = detect_catalysts_and_tsunami(frame)
    assert out["tsunami_signal"].tolist() == [1, 0]
