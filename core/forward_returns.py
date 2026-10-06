"""
Forward returns between two vintages of the PRISM frame — ONE definition (2026-10-06).

Moved here from tools/validate.py so the automated scan-records job (ui/scan_records_job.py, run by
GitHub Actions) and the local measurement harness compute a return the same way; tools/validate.py
imports it.

A forward return is the price change between two vintages, matched on a stable key. A row whose
SHARE COUNT moved between them is a corporate action: a bonus or split re-bases the price (the holder
is unharmed), a cash issue does not, and the share count alone cannot tell the two apart. Such rows
are FLAGGED and excluded from headline readings rather than guessed at (CLAUDE.md §5, "unverifiable
is not passed"): guessing pro-rata would fabricate gains for every cash raiser, leaving them raw
fabricates -80% losses (Tembo Global -89.4% raw, +6.5% once its 10x split is undone).
"""
import numpy as np
import pandas as pd

SHARE_TOL = 0.02          # |share_mult - 1| above this = a real action, not a rounding wobble
MIN_HORIZON_DAYS = 25     # a window shorter than this is noise, not evidence


def forward_returns(early: pd.DataFrame, late: pd.DataFrame, key: str = "company_id") -> pd.DataFrame:
    """Inner-join two vintages on `key`; fwd_return = close_late / close_early - 1 (both prices > 0).

    Returns columns: key · close_early · close_late · fwd_return (raw) · share_mult · corp_action ·
    fwd_return_adj (split-adjusted, for sensitivity readings only). A missing share count is not
    evidence of an action, so such rows stay, unflagged."""
    def _cols(df):
        return [key, "close_price"] + (["equity_shares"] if "equity_shares" in df.columns else [])
    e = early[_cols(early)].rename(columns={"close_price": "close_early", "equity_shares": "shares_early"})
    lt = late[_cols(late)].rename(columns={"close_price": "close_late", "equity_shares": "shares_late"})
    m = e.merge(lt, on=key, how="inner")
    ok = (m["close_early"] > 0) & (m["close_late"] > 0)
    m = m[ok].copy()
    m["fwd_return"] = m["close_late"] / m["close_early"] - 1.0

    if "shares_early" in m.columns and "shares_late" in m.columns:
        se = pd.to_numeric(m["shares_early"], errors="coerce")
        sl = pd.to_numeric(m["shares_late"], errors="coerce")
        # §5 defensive denominator: a zero/missing prior count yields NaN, never a ratio.
        mult = np.where(se > 0, sl / se, np.nan)
        m["share_mult"] = mult
        m["corp_action"] = np.where(np.isnan(mult), False, np.abs(mult - 1.0) > SHARE_TOL).astype(bool)
        m["fwd_return_adj"] = np.where(m["corp_action"], m["close_late"] * mult / m["close_early"] - 1.0,
                                       m["fwd_return"])
    else:
        m["share_mult"] = np.nan
        m["corp_action"] = False
        m["fwd_return_adj"] = m["fwd_return"]
    return m
