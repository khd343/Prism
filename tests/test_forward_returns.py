"""
test_forward_returns.py — the ONE forward-return definition (core/forward_returns.py, 2026-10-06).

Moved out of the private tools/validate.py so the automated scan-records job and the local harness
compute a return the same way. A share count that moved between two vintages is a corporate action:
the raw price ratio is untrustworthy in a direction the share count cannot resolve (a bonus re-bases
the price, a cash raise does not), so the row is FLAGGED and readings exclude it rather than guess.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.forward_returns import MIN_HORIZON_DAYS, SHARE_TOL, forward_returns  # noqa: E402


def _pair():
    early = pd.DataFrame({"company_id": ["A", "B", "C", "D", "E", "F"],
                          "close_price": [100.0, 100.0, 100.0, 100.0, 0.0, 100.0],
                          "equity_shares": [10.0, 10.0, 10.0, np.nan, 10.0, 10.0]})
    late = pd.DataFrame({"company_id": ["A", "B", "C", "D", "E", "G"],
                         "close_price": [110.0, 20.0, 99.0, 150.0, 50.0, 100.0],
                         "equity_shares": [10.1, 50.0, 10.0, 10.0, 10.0, 10.0]})
    return forward_returns(early, late).set_index("company_id")


def test_a_plain_return_is_the_price_ratio():
    m = _pair()
    assert np.isclose(m.loc["A", "fwd_return"], 0.10) and np.isclose(m.loc["C", "fwd_return"], -0.01)


def test_a_share_count_move_beyond_the_tolerance_is_a_corporate_action():
    m = _pair()
    assert bool(m.loc["B", "corp_action"]), "a 5:1 split must be flagged, never read as -80%"
    assert np.isclose(m.loc["B", "fwd_return_adj"], 0.0), "the adjusted reading undoes the split"
    assert not bool(m.loc["A", "corp_action"]), f"a 1% wobble is within the {SHARE_TOL:.0%} tolerance"


def test_a_missing_share_count_is_not_evidence_of_an_action():
    assert not bool(_pair().loc["D", "corp_action"])


def test_a_non_positive_price_or_a_one_sided_stock_has_no_return():
    m = _pair()
    assert "E" not in m.index, "a zero price cannot anchor a return"
    assert "F" not in m.index and "G" not in m.index, "a stock in only one vintage has no return"


def test_the_harness_and_the_job_share_the_constants():
    assert SHARE_TOL == 0.02 and MIN_HORIZON_DAYS == 25
