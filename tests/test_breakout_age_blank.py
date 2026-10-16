"""
test_breakout_age_blank.py
==========================
Contract: a blank 52-week-high age is UNKNOWN, so it scores the neutral 50 (2026-10-16).

THE BUG. _compute_breakout_score filled a blank dist_52wh_days with 9999 before zoning it, so a blank read as "the
high was set more than a year ago": 5 points, the worst zone. config.HIGH_AGE_ZONES documents "NaN -> default 50
(neutral) — missing data never penalises", and the line's own trailing .fillna(50) could never fire. Measured
2026-10-16, structure only: on the 15 Oct copy one blank age would knock 134 of the 267 🚀 members off the list; the
24 in-sample archive copies (29 Jun - 9 Oct 2026) carry no blank age at all, so no list they made changes.

Run with: pytest tests/test_breakout_age_blank.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import BREAKOUT_SIGNALS, HIGH_AGE_ZONES  # noqa: E402
from core.scoring_engine import _compute_breakout_score  # noqa: E402

W = BREAKOUT_SIGNALS["52wh_days"]


def test_a_blank_age_scores_neutral_never_a_year_old_high():
    df = pd.DataFrame({"dist_52wh_days": [np.nan, 10.0, 300.0]})          # only the age leg is present
    assert _compute_breakout_score(df).tolist() == pytest.approx([50 * W, 100 * W, 5 * W])


def test_every_real_age_still_scores_its_zone():
    zones = list(HIGH_AGE_ZONES.values())
    df = pd.DataFrame({"dist_52wh_days": [float(z["min"]) for z in zones]})
    assert _compute_breakout_score(df).tolist() == pytest.approx([z["score"] * W for z in zones])
