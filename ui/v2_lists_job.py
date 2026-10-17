"""
📋 Kite v2's lists (2026-10-17): PRISM's two lists for Kite v2, both from ONE commit (Kite-v2 design §3.1, addendum A5).

    python -m ui.v2_lists_job --lists <a v2-lists checkout>                        # live, needs SHEET_ID
    python -m ui.v2_lists_job --lists <dir> --backfill <folder of "PRISM YYYY-MM-DD.xlsx">

For every session, two files go to the orphan branch `v2-lists`, written once:
- the 🚀 Playbook list, exactly the bytes the feed builds (ui/playbook_feed_job.py);
- the rank list: every scored stock with the columns v2 reads, among them its momentum percentile over the WHOLE
  market, and the dates of its last and next results.

Everything that decides a list is the feed's own and is imported, never copied: the scoring, the 🩺 gate, the
recipe, the Playbook list's builder, the stale-copy guard and the session rules. This module adds the rank list, its
checks, the two-file store and the carry-forward of `last_result_date`. The feed itself (branch `playbook-feed`,
pinned 68b38e0) is untouched: Kite v1 reads it until it retires, and v2 never mixes the two recipes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import SHARIAH_STATUSES
from ui import data_check_job as D
from ui import playbook_feed_job as F
from ui import ui_scans as S

# ── The rank list: one row per scored stock (design §3.1) ────────────────────────────────────────────────────────────
RANK_COLUMNS = ("symbol", "shariah_status", "cap", "traded_value_cr", "momentum_score", "momentum_pct", "vstop_value",
                "close_price", "sector", "stage", "last_result_date", "next_result_date", "equity_shares")
RANK_NUMBERS = ("traded_value_cr", "momentum_score", "vstop_value", "close_price")
RANK_SOURCE = ("company_id", "shariah_status", "market_category", "traded_value_cr", "momentum_score", "vstop_value",
               "close_price", "sector", "weinstein_stage", "days_from_result", "equity_shares")
# Measured 2026-10-17 on all 29 saved sheets (29 Jun - 16 Oct 2026): 2,107 to 2,716 scored stocks, no company id ever
# repeated or outside Kite's EXCHANGE:SYMBOL form. Wide enough that a normal day never trips it; an empty, truncated
# or exploded market always does.
_MEASURED_RANK_BOUNDS = (1800, 3500)
MIN_RANK_ROWS, MAX_RANK_ROWS = _MEASURED_RANK_BOUNDS
# THE RANK LIST'S OWN INPUTS, floored by the feed's rule (RECIPE_FLOORS): the lowest share on the 29 sheets less 8
# points, rounded down to 0.05. Lowest shares: shariah_status 100% (every sheet), market_category 99.96% (14 Sep),
# traded_value_cr 98.86% (10 Sep), momentum_score 100%, vstop_value 99.62% (29 Jun), sector 100%, days_from_result
# 99.89% (10 Sep), equity_shares 99.96% (14 Sep), a positive close 100%, a known Weinstein stage 93.04% (10 Sep).
RANK_FLOORS = {"shariah_status": 0.90, "market_category": 0.90, "traded_value_cr": 0.90, "momentum_score": 0.90,
               "vstop_value": 0.90, "sector": 0.90, "days_from_result": 0.90, "equity_shares": 0.90,
               "close_price": 0.90, "weinstein_stage": 0.85}
TEXT_FLOORS = ("shariah_status", "market_category", "sector")
UNKNOWN_STAGE = "❔ Unknown"
# A SCREEN THAT LABELS EVERYTHING DOUBTFUL STILL FILLS EVERY ROW, so the fill floor cannot see it. The share labelled
# Shariah ran 20.98% (29 Jun) to 30.91% (3 Sep) on the 29 sheets; less and plus 8 points, rounded outward to 0.05.
SHARIAH_SHARE_MIN, SHARIAH_SHARE_MAX = 0.10, 0.40


def build_rank(scored: pd.DataFrame, session: str, previous: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    """The rank list of `session`, and {"carried": n, "unknown_days": n}.

    `previous` is the newest earlier rank list on the branch, read as text (None when there is none). The momentum
    percentile is the 🚀 recipe's own (`S._pct` over the whole frame), so the 2/3 line reads exactly as the recipe
    reads it. The result dates come from days_from_result (negative = days since the last result):
    - next_result_date = session + days when days >= 0 (an announced date, or today);
    - last_result_date = session + days when days <= 0; otherwise (an announced date, or a blank) that symbol's value
      on `previous`, or '' when there is none.
    A non-integral or blank day is unknown. Every label is a code, and an unknown one refuses the list."""
    missing = sorted(set(RANK_SOURCE) - set(scored.columns))
    if missing:
        raise F.FeedError(f"the scored data lacks {missing}")
    status = scored["shariah_status"]
    named = status.notna() & (status.astype(str).str.strip() != "")
    unknown = sorted(set(status[named].astype(str)) - set(SHARIAH_STATUSES))
    if unknown:
        raise F.FeedError(f"unmapped shariah_status value(s) {unknown} — add them to the screen, never guess")
    out = pd.DataFrame(index=scored.index)
    out["symbol"] = scored["company_id"].astype(str)
    out["shariah_status"] = status.astype(str).where(named, "")
    out["cap"] = F._codes(scored["market_category"], F.CAP_CODES, "market_category")
    for c in RANK_NUMBERS:
        out[c] = pd.to_numeric(scored[c], errors="coerce")
    pct = S._pct(scored, "momentum_score")                     # the WHOLE market, as the recipe ranks it
    out["momentum_pct"] = pct.astype(str).where(pct.notna(), "")   # str(float) is its shortest exact repr
    out["sector"] = scored["sector"].where(scored["sector"].notna(), "").astype(str)
    out["stage"] = F._codes(scored["weinstein_stage"], F.STAGE_CODES, "weinstein_stage")

    raw = pd.to_numeric(scored["days_from_result"], errors="coerce")
    integral = raw.notna() & np.isclose(raw, np.round(raw))
    days = np.round(raw).where(integral)                       # NaN = unknown
    when = (pd.Timestamp(session) + pd.to_timedelta(days, unit="D")).dt.strftime("%Y-%m-%d")
    out["next_result_date"] = when.where(days >= 0, "")
    if previous is None or previous.empty:
        carried = pd.Series(np.nan, index=scored.index, dtype=object)
    else:
        carried = out["symbol"].map(previous.set_index("symbol")["last_result_date"]).replace("", np.nan)
    out["last_result_date"] = when.where(days <= 0, carried).fillna("")
    out["equity_shares"] = pd.to_numeric(scored["equity_shares"], errors="coerce").round().astype("Int64")
    stats = {"carried": int((~(days <= 0) & carried.notna()).sum()), "unknown_days": int((~integral).sum())}
    frame = out[list(RANK_COLUMNS)].sort_values("symbol", kind="mergesort").reset_index(drop=True)
    return frame, stats


def rank_csv_bytes(frame: pd.DataFrame) -> bytes:
    """UTF-8, \\n, numbers as the feed writes them; momentum_pct is already its exact text."""
    return frame.to_csv(index=False, float_format=F.FLOAT_FORMAT, lineterminator="\n").encode("utf-8")


def validate_rank(frame: pd.DataFrame) -> None:
    n = len(frame)
    if not MIN_RANK_ROWS <= n <= MAX_RANK_ROWS:
        raise F.FeedError(f"{n} stocks in the rank list, outside the measured {MIN_RANK_ROWS}-{MAX_RANK_ROWS}")
    dup = sorted(frame.loc[frame["symbol"].duplicated(), "symbol"].unique())
    if dup:
        raise F.FeedError(f"duplicate symbol(s) in the rank list {dup[:5]}")
    bad = sorted(frame.loc[~frame["symbol"].str.fullmatch(F.SYMBOL.pattern), "symbol"])
    if bad:
        raise F.FeedError(f"malformed symbol(s) in the rank list {bad[:5]}")


def _share(scored: pd.DataFrame, col: str) -> float:
    """The share of stocks carrying the reading; an absent column carries none (never skipped)."""
    if col not in scored.columns:
        return 0.0
    s = scored[col]
    if col in TEXT_FLOORS:
        ok = D._filled(s)
    elif col == "close_price":
        ok = pd.to_numeric(s, errors="coerce") > 0
    elif col == "weinstein_stage":
        ok = D._filled(s) & (s.astype(str) != UNKNOWN_STAGE)
    else:
        ok = pd.to_numeric(s, errors="coerce").notna()
    return float(ok.mean())


def check_rank_inputs(scored: pd.DataFrame) -> list:
    """Two 🩺 results in the feed's own form: every rank-list input at or above its floor, and the Shariah share
    inside its measured band."""
    share = {c: _share(scored, c) for c in RANK_FLOORS}
    thin = [f"{c} {share[c]:.1%}" for c in RANK_FLOORS if share[c] < RANK_FLOORS[c]]
    sh = (float((scored["shariah_status"].astype(str) == SHARIAH_STATUSES[0]).mean())
          if "shariah_status" in scored.columns else 0.0)
    return [D._ok(not thin, "Rank list inputs", f"below the floor: {thin}" if thin
                  else f"all {len(RANK_FLOORS)} at or above their floor"),
            D._ok(SHARIAH_SHARE_MIN <= sh <= SHARIAH_SHARE_MAX, "Shariah share",
                  f"{sh:.1%} labelled Shariah, the measured band is {SHARIAH_SHARE_MIN:.0%}-{SHARIAH_SHARE_MAX:.0%}")]
