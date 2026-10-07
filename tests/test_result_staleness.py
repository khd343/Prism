"""
test_result_staleness.py
========================
Contract for the result-staleness data-recency guard (compute_derived_signals).

`days_from_result` in the CSV: negative = days SINCE the last reported result (past),
positive = an upcoming scheduled result. We derive:
  • result_age_days  = -days_from_result   (positive = financials this many days stale)
  • result_stale_flag = 1 when result_age_days > config.STALE_RESULT_DAYS (240): no result past every
    listing deadline, quarterly or SME half-yearly (scored on outdated fundamentals; often a distress
    tell — Gensol Engineering sat 477 days stale before its collapse).

Display-only companion to data_coverage_pct: it must NEVER gate or score, and a MISSING
days_from_result must NOT be asserted as stale (semantic truth).

Run with: pytest tests/test_result_staleness.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

from datetime import date

import numpy as np
from data_engine import compute_derived_signals
from test_data_quality_fixes import _frame
from config import STALE_RESULT_DAYS


def _stale(days_from_result_vals):
    n = len(days_from_result_vals)
    return compute_derived_signals(_frame(n=n, days_from_result=days_from_result_vals))


def test_fresh_result_not_flagged():
    """Reported 13 days ago → age 13 → not stale."""
    out = _stale([-13.0])
    assert out["result_age_days"].iloc[0] == 13.0
    assert out["result_stale_flag"].iloc[0] == 0


def test_stale_result_flagged():
    """Last result 300 days ago → stale (Gensol-class data-recency risk; Gensol sat at 602 days)."""
    out = _stale([-300.0, -602.0])
    assert out["result_age_days"].iloc[0] == 300.0
    assert out["result_stale_flag"].tolist() == [1, 1]


def test_threshold_is_strict_240():
    """Exactly 240 days = NOT stale (>240); 241 days = stale (Kody Technolab on 2026-10-06: no FY26
    annual by 30 May)."""
    assert STALE_RESULT_DAYS == 240
    out = _stale([-240.0, -241.0])
    assert out["result_stale_flag"].tolist() == [0, 1]


def test_no_on_time_filer_is_stale_under_either_calendar():
    """The threshold is set by the listing deadlines (SEBI LODR Reg 33), computed here from dates: the
    longest gap an ON-TIME filer shows stays within it — quarterly (Oct-Dec results ~10 Jan -> annual
    30 May) and SME half-yearly (Apr-Sep results ~10 Oct -> annual 30 May; even 2-3 Oct, leap year or
    not). The old 120 failed both: 2026-10-06 flagged ~270 SMEs on their normal calendar."""
    quarterly = (date(2027, 5, 30) - date(2027, 1, 10)).days                  # 140
    half_yearly = (date(2027, 5, 30) - date(2026, 10, 10)).days               # 232
    earliest = max((date(2027, 5, 30) - date(2026, 10, 2)).days,              # 240
                   (date(2028, 5, 30) - date(2027, 10, 3)).days)              # 240 (leap)
    assert quarterly > 120 and half_yearly > 120, "why 120 was wrong"
    assert max(quarterly, half_yearly, earliest) <= STALE_RESULT_DAYS
    out = _stale([-float(quarterly), -float(half_yearly), -132.0, -float(earliest)])
    assert out["result_stale_flag"].tolist() == [0, 0, 0, 0], "an SME's ~132 days in October is its calendar"


def test_every_label_says_the_same_threshold():
    """The Discovery label ('8 months'), its help and the Reference text follow the one constant."""
    import pathlib
    from ui.ui_reference_data import CONCEPT_REFERENCE
    root = pathlib.Path(__file__).resolve().parents[1]
    disc = (root / "ui" / "ui_discovery.py").read_text(encoding="utf-8")
    assert "Hide stale results (no result in 8 months)" in disc and round(STALE_RESULT_DAYS / 30) == 8
    assert "over {STALE_RESULT_DAYS} days" in disc and ">120d" not in disc
    stale = dict(CONCEPT_REFERENCE["⏳ Result Recency"])["⏳ Stale"]
    assert f"more than {STALE_RESULT_DAYS} days" in stale and "half-yearly" in stale


def test_upcoming_result_not_stale():
    """A future scheduled result (positive days_from_result) → negative age → not stale."""
    out = _stale([12.0])
    assert out["result_stale_flag"].iloc[0] == 0


def test_missing_not_asserted_stale():
    """Missing days_from_result must NOT be flagged stale (semantic truth)."""
    out = _stale([np.nan])
    assert out["result_stale_flag"].iloc[0] == 0
