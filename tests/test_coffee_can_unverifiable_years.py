"""
test_coffee_can_unverifiable_years.py
=====================================
Contract: a year that cannot be measured is not a year that passed (CLAUDE.md §5, "Unverifiable is not
passed").

Coffee Can's year-by-year check (`_cc_each_year`: revenue growth in EACH of years 2-5 at least 5%, the
pandemic-adjusted floor) filled a missing year with 5, exactly the floor, so a company whose year could not
be measured passed that year. Unusual Billionaires reuses the same check. Measured 2026-10-15 on the live
sheet: none of the 9 Coffee Can or 61 Unusual Billionaires passes depends on a missing year (their other legs
already need a long history), so this changes no stock today; it stops the next data gap from certifying a
company on absent evidence.

Run with: pytest tests/test_coffee_can_unverifiable_years.py -v
"""

import numpy as np
import pandas as pd


def _row(**overrides) -> dict:
    """A non-financial company that clears every Coffee Can and Unusual Billionaires leg."""
    row = {
        "is_financial": False,
        "roce_med_10y": 20.0, "roce_med_5y": 20.0,                    # capital efficiency >= 15
        "rev_gr_10y": 15.0, "rev_gr_5y": 12.0, "rev_gr_yoy": 10.0,    # 10Y >= 10, 5Y >= 8, not shrinking
        "rev_gr_y2": 12.0, "rev_gr_y3": 12.0, "rev_gr_y4": 12.0, "rev_gr_y5": 12.0,
        "cfo_to_ebitda": 95.0, "debt_to_equity": 0.3, "pledged_percentage": 0.0,
        "promoter_holdings": 60.0, "opm_stable": 1,
        # read elsewhere in compute_qglp_score
        "roce": 20.0, "roe": 18.0, "pat_gr_5y": 20.0, "eps_gr_5y": 18.0, "peg": 1.0,
        "forensic_score": 90.0, "forensic_multiplier": 1.0, "piotroski_fscore": 7,
        "d35_roce_trend": 1.0, "fcf_yield": 6.0, "cfo_to_pat": 85.0,
    }
    row.update(overrides)
    return row


def _score(rows: list) -> pd.DataFrame:
    from config import MASTER_PROFILES
    from core.scoring_engine import compute_qglp_score

    df = pd.DataFrame(rows)
    df.attrs["detected_market_regime"] = "SIDEWAYS"
    return compute_qglp_score(df, MASTER_PROFILES["Balanced"])


def test_coffee_can_does_not_pass_a_year_it_cannot_measure():
    """Identical companies, except year 3's growth is missing on the second. Fails if a missing year is
    filled with a passing value again."""
    out = _score([_row(), _row(rev_gr_y3=np.nan)])
    assert out["coffee_can_pass"].tolist() == [1, 0]


def test_unusual_billionaires_does_not_pass_a_year_it_cannot_measure():
    """Unusual Billionaires reads the same year-by-year check, so it inherits the same rule."""
    out = _score([_row(), _row(rev_gr_y5=np.nan)])
    assert out["ub_pass"].tolist() == [1, 0]
