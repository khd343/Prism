"""
🎯 Scans — measured stock lists, one definition each (2026-10-06).

A screener usually shows a list with no evidence. Each scan here is shown beside ITS OWN record: how
its stocks do against the market on the PRISM Archive's month-end copies, in windows nobody looked at
while choosing its recipe. Candidates with no card wait as "being measured" until December's review.
Every record starts fresh from 2026-11-05 (the operator's call; DESIGNED_ON).

THREE RULES, each pinned in tests/test_scans.py:
1. ONE DEFINITION. Each recipe is one function below. The tab calls it on the whole market and the
   records job (ui/scan_records_job.py) imports the same function, so a record always describes
   exactly the stocks on screen.
2. A RECORD NEVER OUTLIVES ITS RECIPE. scan_records.json stores a fingerprint of the logic it
   measured (the recipe plus every engine function it reads). Change the logic and the card shows no
   numbers until it is re-measured.
3. NOTHING HIDDEN, NOTHING PROMISED. A flagged stock stays in its list with the flag named; a missing
   value never passes a condition; the wording is "matched / record / provisional".

Each window also carries what the headline leaves out (2026-10-07): how many matches had no return
(no price at the end: delisted, suspended or missing) or were dropped as corporate actions, and a NET
record: only matches trading NET_MIN_CR a day or more, each charged NET_COST_PP for the round trip.

Display only, like ui_movers: no widgets here. app.py owns the scan choice, the liquidity floor and
the row click.
"""
import ast
import hashlib
import html
import inspect
import json
import re
import textwrap
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from config import BENCHMARK_INDICES, COLORS, SHARIAH_STATUSES
from core.data_engine import price_trend_inputs, traded_value_cr   # ONE liquidity number (moved there 2026-10-09)
from core.data_engine import ram_date_text, ram_day                 # 🔢 RAM's date words, the Tear-Sheet's own
from core.forward_returns import MIN_HORIZON_DAYS
from core.scoring_engine import (_compute_improvement_score, compute_momentum_score, improvement_evidence,
                                 strong_quarter)
from ui.ui_group_views import MEASURES, group_measures
from ui.ui_scanner import _SCANNER_HEADER_TIPS        # the Deep Scanner's header tips: one tip per column
# LIQUIDITY_STEPS and step_label are also re-exported for app.py's Scans selector (its one ui_scans import).
from ui.ui_liquidity import (HEADER as LIQUIDITY_HEADER, LIQUIDITY_STEPS, ORDER_SHARE, SURVEILLANCE_NOTE,
                             order_guide, order_size_lakh, step_label)

RECORDS_PATH = Path(__file__).with_name("scan_records.json")
# The hand checks of the falls the records name (2026-10-18, the operator's demerger rule): each one the corporate-
# action list of a stock's exchange (NSE's, or BSE's for a BSE: id, which NSE's cannot show) read over a span of
# ex-dates, and the demerger found there, if any, in the list's own words (subject). Kept by hand; the job only reads
# it (load_checks), and a month is written final only once every fall it names is covered by one.
CHECKS_PATH = Path(__file__).with_name("scan_record_checks.json")
CHECK_FIELDS = ("company_id", "from", "through", "demerger", "subject", "checked_on")
CHECK_ID = re.compile(r"(NSE|BSE):[^\s:]+")       # a company_id as the job spells it: every id on the sheet is one
# The net record (2026-10-07): what a buyer could have kept. Only matches trading at least NET_MIN_CR
# a day count, and each is charged NET_COST_PP for the round trip (0.5% in + 0.5% out: taxes, spread,
# impact). The benchmark (the market median) is charged nothing, so the cost side errs conservative.
NET_MIN_CR = 2.0
NET_COST_PP = 1.0
# Each scan against the index fund you could hold instead (2026-10-08, docs/december-2026-plan.md rule
# 23, written before any result). The portfolio you would actually hold — the scan's NET_MIN_CR+ matches,
# equal-weighted, charged the NET_COST_PP round trip each window — against the fund you would actually
# own: the index's PLAIN level change (not total return; the matches' returns carry no dividends either)
# less FUND_FEE_PCT_YR for the window's days. 0.2% a year is about what the cheapest direct plans charge
# (Nifty 500 ~0.17%, Nifty500 Momentum 50 0.16-0.19%, Oct 2026); funds on the smaller factor indices
# charge more (Smallcap250 Momentum Quality 100: 0.47-0.64%), so for those the test errs strict — it
# never flatters a scan. Each scan's main index is its `benchmark`; every declared index is recorded.
FUND_FEE_PCT_YR = 0.2
# "What if you held 20 of them?" (2026-10-13, fixed before the first unseen window; December plan rule
# 26 E-F, rule 27). Each unseen window also records HOLD_K stocks picked at random HOLD_DRAWS
# times with a FIXED seed, from the matches a Rs 1 lakh account can hold — a Mega, Large, Mid or Small
# cap (HOLD_CAPS) AND one share at most HOLD_MAX_PRICE (13 of 185 buyable stocks cost more on 2026-10-12)
# — after NET_COST_PP: the typical result, its 10th-90th range, how often it lost money, and where the
# top HOLD_K by Rank lands among those picks (by momentum too, until the order check moved to RAM Rank, 2026-11-05).
# HOLD_CAPS replaced a NET_MIN_CR (Rs 2 Cr a day) floor on this reading the same day, before any unseen
# window (the user's call): at ~Rs 5,000 a stock, turnover is not what limits a Rs 1 lakh account. The
# after-costs and index-fund readings keep NET_MIN_CR. Measured 2026-10-13: NEITHER rule keeps out SME
# stocks, which trade only in lots (Playbook holdable: 30 of 216 SME under HOLD_CAPS, 17 of 173 under the
# floor; a Rs 1 Cr floor still leaves 20) — a real fix needs a lot-size column, which the data lacks.
# The Shariah reading (scan_records_job.shariah_reading) is read on Kite v2's universe (2026-10-18, before any unseen
# window): its §4.1 floor, traded value of SHARIAH_MIN_CR a day or more, as kite_v2/engine/select.py reads the same
# core.data_engine.traded_value_cr (`>=`; an unknown value never passes). Measured on the four snapshots of 17 Jun -
# 3 Oct, 6-13% of the Playbook's Mega-to-Small matches trade under it: stocks Kite v2 never buys. Its SME stocks (A15) stay
# in: PRISM's data has no lot size, so they cannot be told apart, and each reading says so (sme_excluded).
SHARIAH_MIN_CR = 1.0
# The hand check (2026-10-18, before any unseen window). A demerger lowers the price while the share count stays, so
# core.forward_returns cannot flag it, and a window books the holder's new shares as a loss: NSE's own list showed
# HEG -66.2% (ex 7 Sep) and Triveni -33.6% (ex 22 Jul), neither flagged, and the vendor's own returns fell with them,
# so nothing in PRISM's data tells a demerger from a crash. Each unseen window names its matches that fell
# FALL_CHECK_PCT or more with the share count unmoved, to be checked against its exchange's list (NSE's, or BSE's for
# a BSE: id) before the window is read. Every fall stays named: a real crash stays in, as evidence, and a demerger its
# check confirms is set aside (the operator's rule, CHECKS_PATH). HEG was a 🌱 match on 17 Jun: a window across its
# demerger books -56.4% (543.40 -> 237.05 by 19 Sep, 192,977,530 shares at both ends).
FALL_CHECK_PCT = 30.0
HOLD_K = 20
HOLD_MAX_PRICE = 5000.0
HOLD_CAPS = ("Mega Cap", "Large Cap", "Mid Cap", "Small Cap")
HOLD_DRAWS = 2000
HOLD_SEED = 20261013
# The order check (2026-10-13). Written before any unseen window as the rule that would SWITCH a scan's
# table to momentum order; the same day, still before any result, the operator switched 🚀 and 🌬️ to
# momentum order (SCANS `order`), so it then CHECKED the order in use, its bar unchanged. Since 2026-11-05 (the
# operator's call, before any unseen window had a result) it reads RAM Rank instead, on what Kite can trade
# (scan_records_job.order_reading), whichever way the table lists: over the scan's FIRST `windows` separate
# (non-overlapping), finished unseen windows with at least `min_n` such stocks, RAM Rank's top HOLD_K must beat on
# average at least `min_avg_pct`% of the random picks AND average above Rank's top HOLD_K — "held"; otherwise "did
# not hold", which goes to the operator as evidence (nothing switches by itself; December plan rule 27). A window
# with no RAM has no place to count, so the check starts again from the first window that has it. Luck alone passes
# a 70% average over four windows about 1 time in 12 (8.6%); "3 of 4 above half" would pass 31% of the time. 🌱 lists
# by Rank and is not checked.
ORDER_RULE = dict(scans=("playbook", "leaders"), windows=4, min_avg_pct=70.0, min_n=30)
# Kite's buy filters beyond HOLD_CAPS, SHARIAH_MIN_CR and HOLD_MAX_PRICE, as far as PRISM can see them
# (kite_v2/engine/select.py and rules.py read zone_lo, zone_hi and stages the same way): an NSE stock, 0-25% above its
# V-Stop inclusive, Weinstein stage 1 or 2, in the codes the feed writes (playbook_feed_job.STAGE_CODES).
ORDER_EXCHANGE = "NSE"
ORDER_ZONE = (0.0, 25.0)
ORDER_STAGES = ("1", "2")
ORDER_NOT_IN_DATA = "The price band, ASM, GSM and the SME lot size are not in PRISM's data."
ORDER_FROM_RAM = "A window on a copy with no RAM does not count, so the check counts from the first window that has it"
# On 🚀 the check reads list C, the ladder's last layer, because that is the list Kite trades from (the operator, 2026-11-05:
# "only on list C ... and that also after filter"). The rest of 🚀's record stays on the 🚀 list until list C has a record of
# its own, in May. Every other checked scan reads its own matches.
ORDER_LIST = {"playbook": "price_only"}
ORDER_ON_C = "On 🚀 it reads list C, the list Kite trades from, not the 🚀 list."
# The seal (Kite's addendum 37, "Sealed until then"): before its one sealed run, nothing may work out a book that orders by RAM
# over a session from 30 Nov 2026, in any code, and a figure of one, computed or seen, makes that paper test no evidence. RAM's
# top HOLD_K among what Kite can trade is such a book. So while this is True the job draws nothing and records no figure for the
# check, and the card shows none. Set it to False only after that run's read-out, and rebuild the windows then.
ORDER_SEALED = True
ORDER_SEALED_WORDS = "Its figures stay sealed until Kite's one sealed run, since they would show that test's result early."
# The revenue-led check (registered 2026-10-15, before any unseen window has a result). Inside 🚀, do the matches
# that are also revenue_led (the test 🌱 uses) beat the rest? A Kite-side plan proposed trading that in Kite's window
# 2; the frozen go-rule needs the SAME design in two windows in a row, so the operator kept Kite as it is and chose to
# measure the idea here instead. Every unseen window records both arms (scan_records_job.revenue_led_reading). It
# HOLDS if, over the scan's first `windows` separate finished windows with both arms at least `min_n`, the
# revenue-led matches beat the rest by at least `min_gap_pp` on the mean AND on the median in EVERY one; otherwise
# "did not hold", decided once. Three monthly windows because the archive keeps month-end copies. Registered on 31 Oct
# -> 30 Nov -> 31 Dec -> 31 Jan, the months of Kite's windows 1 and 2; the fresh start of 2026-11-05 (DESIGNED_ON) moved it
# a month on, to 30 Nov -> 31 Dec -> 31 Jan -> 28 Feb, so it no longer reads November. It decides nothing by
# itself: nothing reaches the list, the feed or Kite; a hold makes it a candidate for Kite's next design, tested in
# that design's own windows.
REVENUE_LED_RULE = dict(scans=("playbook",), windows=3, min_gap_pp=1.0, min_n=30)
# Both checks read months whose named falls were all checked by hand (the operator, 2026-10-18, before any unseen
# window): a demerger confirmed on its exchange's list is set aside from its month, like a split (the kept-in reading
# is recorded beside it, but the rules read the month without it), and a month is written final only once every fall
# it names is checked (scan_records_job), so neither rule ever decides on a month with an unchecked named fall.
RECORD_VERSION = 2
# The day every record starts (the operator, 2026-11-05: "Remove old record of scan record completely, and from today is
# the fresh"). Only a month-end copy dated after it can start a window, so the first is the 30 Nov 2026 copy and the
# 31 Oct one never starts any: data nobody looked at while choosing the recipes, which were chosen on 6 and 7 Oct. The records
# from before it stay in git's history, at b19d485. A scan that needs a later date, after a recipe change, carries its own
# `designed_on` (scan_records_job.check_recipe_dates).
DESIGNED_ON = "2026-11-05"


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """The column as numbers; all-NaN when absent, so a missing input passes nothing."""
    return (pd.to_numeric(df[col], errors="coerce") if col in df.columns
            else pd.Series(np.nan, index=df.index))


def _pct(df: pd.DataFrame, col: str) -> pd.Series:
    """Percentile rank (0-1] across the frame given — the whole market, never a filtered slice."""
    return _num(df, col).rank(pct=True)


# ── The recipes ────────────────────────────────────────────────────────────────────────────────────
def playbook_shortlist(df: pd.DataFrame) -> pd.Series:
    """Momentum and Breakout in the market's top third, a 🔥 Strong Quarter (the engine's own
    definition; a quarter that cannot be judged is not strong), Improvement in the top half, and the
    price at or above its stop. A stock with no measurable Improvement leg never passes on the neutral
    fill (2026-10-15)."""
    return playbook_conditions(df).all(axis=1)


def playbook_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """The 🚀 recipe's six conditions, one column each, True where a stock meets it; the shortlist is the stocks
    meeting all six. One definition: the Playbook feed reads which condition a stock missed, to say which blank input
    its absence rests on (ui/playbook_feed_job.py, 2026-10-16). Row by row, never aligned on the index."""
    return pd.DataFrame({
        "momentum": (_pct(df, "momentum_score") >= 2 / 3).to_numpy(),
        "breakout": (_pct(df, "breakout_score") >= 2 / 3).to_numpy(),
        "quarter": (strong_quarter(df) == 1).to_numpy(),
        "improvement": (_pct(df, "improvement_score") >= 0.5).to_numpy(),
        "evidence": improvement_evidence(df).to_numpy(),
        "stop": (_num(df, "dist_to_vstop") >= 0).to_numpy(),
    }, index=df.index).fillna(False).astype(bool)


def leading_sectors(df: pd.DataFrame, n: int = 5, min_stocks: int = 5) -> list:
    """The n sectors (of at least min_stocks stocks) ranked best on the 📈 Sectors tab's own two
    breadth shares: stocks within 5% of their 52-week high, and stocks reporting a strong quarter.
    Ties break on the near-high share, then the sector name, so the answer never depends on row
    order."""
    if "sector" not in df.columns:
        return []
    g = group_measures(df, "sector")
    if not {"grp_near_high", "grp_strong_quarter"} <= set(g.columns):
        return []
    g = g.join(df["sector"].value_counts().rename("_n"))
    g = g[g["_n"] >= min_stocks].dropna(subset=["grp_near_high", "grp_strong_quarter"])
    order = pd.DataFrame({
        "score": (g["grp_near_high"].rank(ascending=False) + g["grp_strong_quarter"].rank(ascending=False)).to_numpy(),
        "near": g["grp_near_high"].to_numpy(), "sector": g.index.astype(str)})
    order = order.sort_values(["score", "near", "sector"], ascending=[True, False, True])
    return order["sector"].head(n).tolist()


def _growth(df: pd.DataFrame, now: str, prev: str) -> pd.Series:
    """now / prev - 1, only where BOTH years are positive: a loss year has no growth RATE, so it is no
    evidence (NaN, which passes nothing). The guard PRISM's improvement score uses."""
    a, b = _num(df, now), _num(df, prev)
    return (a / b - 1.0).where((a > 0) & (b > 0))


def revenue_led(df: pd.DataFrame) -> pd.Series:
    """Sales up more than 10% over the last year AND profit growing faster than sales: margin expansion
    the business earned by selling more, not by cutting costs on a shrinking business."""
    sales, profit = _growth(df, "revenue", "revenue_1yb"), _growth(df, "pat", "pat_1yb")
    return ((sales > 0.10) & (profit > sales)).fillna(False).astype(bool)


def compounder_candidates(df: pd.DataFrame) -> pd.Series:
    """Profit growth AND margin widening over the last year, each in the top 40% of the stocks that can
    be judged on it (whole market), AND revenue-led (sales up more than 10%, profit growing faster).
    Revenue-led is part of the recipe, not an option."""
    profit = _growth(df, "pat", "pat_1yb")
    margin = _num(df, "opm") - _num(df, "opm_1yb")
    # Top 40% = percentile rank ABOVE 0.6: of 10 stocks, ranks 7-10 (rank 6 sits at exactly 0.6). The
    # percentiles rank the whole market BEFORE the revenue-led leg narrows it.
    top = (profit.rank(pct=True) > 0.6) & (margin.rank(pct=True) > 0.6)
    return (top & revenue_led(df)).fillna(False).astype(bool)


# ── Why a stock LEFT 🌱 (2026-10-07) ──────────────────────────────────────────────────────────────
# Order = how bad the news is: a stock that fell AND stopped being revenue-led is listed under "profit fell".
EXIT_REASONS = {
    "profit_fell": ("🔴", "profit fell", "its latest annual profit is below the year before"),
    "not_revenue_led": ("🟠", "no longer revenue-led",
                        "profit did not fall, but sales are no longer up 10%+ with profit growing faster"),
    "rank_slipped": ("⚪", "others grew faster",
                     "still revenue-led with profit up, but its profit growth or margin change slipped out of "
                     "the market's top 40%"),
    "unjudged": ("⚪", "a figure is missing", "a number the recipe needs is missing"),
    "gone": ("⚪", "not in today's data", "delisted, suspended or renamed"),
}


def compounder_exit_reason(df: pd.DataFrame) -> pd.Series:
    """For each row, why it is NOT a 🌱 match (read only for stocks that left): a key of EXIT_REASONS."""
    pat, pat_1yb = _num(df, "pat"), _num(df, "pat_1yb")
    fell = pat.notna() & pat_1yb.notna() & (pat < pat_1yb)
    needed = ("pat", "pat_1yb", "revenue", "revenue_1yb", "opm", "opm_1yb")
    unjudged = pd.concat([_num(df, c).isna() for c in needed], axis=1).any(axis=1)
    reason = np.select([fell, unjudged, ~revenue_led(df)], ["profit_fell", "unjudged", "not_revenue_led"],
                       default="rank_slipped")
    return pd.Series(reason, index=df.index, dtype=object)


def cycle_shortened(df: pd.DataFrame) -> pd.Series:
    """Cash conversion cycle down 15+ days AND sales up more than 15%, BOTH in the same fiscal year as
    the P&L 🌱 is judged on — supply-chain velocity, the December candidate, as a READING AID on the 🌱
    table (not part of its recipe or record).

    The vendor rolls the balance sheet months after the P&L (cf_year_lag == 1 on ~83% of stocks in
    October), so the ✓ needs the balance sheet ON the P&L's year (cf_year_lag == 0): a lagging or undecided
    stock gets no ✓ until its balance sheet rolls."""
    same_year = _num(df, "cf_year_lag") == 0
    change = _num(df, "ccc") - _num(df, "ccc_1yb")
    sales = _growth(df, "revenue", "revenue_1yb")
    return (same_year & (change <= -15) & (sales > 0.15)).fillna(False).astype(bool)


def leading_sector_stocks(df: pd.DataFrame) -> pd.Series:
    """Every stock in the leading sectors — the whole sector, not only its leaders."""
    if "sector" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["sector"].isin(leading_sectors(df)).astype(bool)


SCANS = (
    # `fixed`: table columns the recipe itself fixes, so they would repeat one value on every row — the
    # Playbook requires a 🔥 Strong Quarter, so that column is ✓ on 100% of its stocks. Only columns
    # fixed BY CONSTRUCTION are hidden; Trend stays, because "above its stop" is not "above its 30-week
    # average" and a Stage 3 or 4 is possible (tests/test_scans.py). `widths` gives the hidden column's
    # 40px to Stock and Sector (measured 2026-10-07: names fit 79% -> 90%, sectors 51% -> 75%; re-measured
    # 2026-10-14 in the 14px font, with Category: names 80%, sectors 72%).
    # `inputs`: the engine functions that compute the columns a recipe reads. The fingerprint follows
    # them, and everything they call, so a re-defined score makes the record stale (2026-10-07).
    # The derived price inputs (dist_to_vstop, vstop_fresh, vstop_green, above_sma200, vol_ratio) ARE
    # followed, through data_engine.price_trend_inputs (2026-10-15: the V-Stop unit fix had changed the list
    # with the fingerprint unchanged); the rest of compute_derived_signals is not — it computes hundreds of
    # columns, so following it would stale the record on every unrelated engine change.
    # Nor is the regime momentum boost in run_scoring_pipeline (one factor for every stock, so it
    # moves no percentile below the 100 clip).
    # `benchmark`: the index fund each scan is judged against on its card — fixed 2026-10-08, before any
    # result (December plan rule 23). Not part of the recipe, so not in the fingerprint.
    # `order`: "momentum" lists the table by momentum, high to low (2026-10-13, the operator's call); a
    # scan without it lists by PRISM Rank. A display order — no list, record or fingerprint moves — and
    # ORDER_RULE checks exactly these scans (tests/test_scan_order.py).
    dict(key="playbook", emoji="🚀", name="Playbook Shortlist", mask=playbook_shortlist,
         benchmark="NSE:NIF500MO50",                  # a momentum list across the whole market
         order="momentum",
         uses=(strong_quarter, improvement_evidence, _pct, _num),
         inputs=(compute_momentum_score, _compute_improvement_score, price_trend_inputs),
         fixed=("_strong_q",), widths={"name": 185, "sector": 125},
         recipe="Momentum and Breakout in the top third of the market, a 🔥 Strong Quarter, Improvement "
                "in the top half, and the price above its stop."),
    dict(key="leaders", emoji="🌬️", name="Leading-Sector Stocks", mask=leading_sector_stocks,
         benchmark="NSE:CNX500",                      # whole sectors, every size
         order="momentum",
         uses=(leading_sectors, group_measures, strong_quarter),
         # It reads two of the Sectors measures, so it follows those two, not the whole table: a new
         # Sectors column must not stale this record (2026-10-07).
         values={f"MEASURES[{k}]": MEASURES[k] for k in ("grp_near_high", "grp_strong_quarter")},
         skip=("MEASURES",),
         recipe="Every stock in the 5 sectors (of 5+ stocks) with the most stocks near their 52-week "
                "high and reporting strong quarters, as ranked on 📈 Sectors."),
    # 🌱 (2026-10-07): a LONG-horizon list, so its unseen windows are 12 months long (`horizon_days`).
    dict(key="compounder", emoji="🌱", name="Compounder Candidates", mask=compounder_candidates,
         benchmark="NSE:NS250MQ100",                  # the plan's main test for 🌱
         uses=(revenue_led, _growth, _num), horizon_days=365,
         # One unseen window a YEAR, starting each November (the operator, 2026-11-05: October until then, but the record's
         # fresh start would have made its first window wait for October 2027; annual results are just as public by
         # November). A window from every month-end would put ~12 bars on the card
         # by 2028 that overlap by 11 months — one year of evidence that looks like twelve.
         start_month=11,
         # "Cycle ↓" (2026-10-07): a reading aid, not part of the recipe or record; its 60px come from
         # Stock, Sector, Score, ⚠ flags and Trend so the table keeps the laptop budget (2026-10-14, with
         # Category: names in full 46%, sectors 37%, from 52% and 52% — 🌱 carries two extra columns; each
         # stays at or above its measured floor, tests/test_scan_views.py).
         extra=("_cycle",), widths={"name": 150, "sector": 80, "composite_score": 65, "_measured_flags": 120,
                                    "weinstein_stage": 85},
         # "Left 🌱" (2026-10-07): why a stock in 🌱 on the 🆕 baseline is not in it now. Not part of the
         # recipe, so not in the fingerprint (which follows mask, uses and inputs only).
         exit_reason=compounder_exit_reason,
         recipe="Sales up more than 10% with profit growing faster, and profit growth and margin "
                "widening each in the top 40% of the market. Starting points for research, not a promise."),
)


# ── The 🚀 ladder (2026-10-30, the operator's call) ─────────────────────────────────────────────────────────────
# Two wider lists of 🚀's own recipe, shown on its table as plain lists: B drops the 🔥 Strong Quarter, C keeps the
# price legs only. They are exactly the L0/L1/L2 lists of Kite's frozen ladder test (Kite v2 addendum 36), whose comparison
# addendum (37) no longer runs, and B and C carry NO record of their own: they are no scan — not in SCANS, so the Playbook feed,
# Kite's v2 lists and the records' windows never list them (tests/test_scan_ladder.py). The one exception, since 2026-11-05, is
# the order check, sealed, which reads C's mask (ORDER_LIST). Tightest first; `conditions` are the columns of
# playbook_conditions, so the 🚀 layer (all six) is the shortlist itself. `note` is the line under the control: ONE
# line on a 1536px laptop (≤ 148 characters, measured 2026-10-30), with no addendum number and no promise of a record.
_NO_RECORD = "No record of its own: the card and its record are the 🚀 list's."
PLAYBOOK_LAYERS = (
    dict(key="playbook", tag="🚀", label="🚀 Playbook", note="",
         conditions=("momentum", "breakout", "quarter", "improvement", "evidence", "stop")),
    dict(key="no_quarter", tag="B", label="B · without 🔥",
         note=f"B — the 🚀 recipe without the 🔥 Strong Quarter. {_NO_RECORD}",
         conditions=("momentum", "breakout", "improvement", "evidence", "stop")),
    dict(key="price_only", tag="C", label="C · price only",
         note=f"C — Momentum and Breakout in the top third and the price above its stop. {_NO_RECORD}",
         conditions=("momentum", "breakout", "stop")),
)
# B's and C's Rank tip (2026-10-30): they take 🚀's order but none of its record, so no order check or test of 🚀's.
_UNRECORDED_RANK_TIP = ("PRISM's overall rank — click this header, then Sort ascending, to list by it. The list is "
                        "sorted by Momentum, high to low, as 🚀 is; it has no record, so nothing here checks that "
                        "order against random picks.")


def playbook_layer_mask(df: pd.DataFrame, key: str) -> pd.Series:
    """True where a stock meets every condition of the ladder layer `key`, ranked on the frame given (the whole
    market, as the recipe is). The 🚀 layer's mask is the shortlist itself."""
    conditions = next(layer["conditions"] for layer in PLAYBOOK_LAYERS if layer["key"] == key)
    return playbook_conditions(df)[list(conditions)].all(axis=1)


def playbook_layer_tag(df: pd.DataFrame) -> pd.Series:
    """Each stock's TIGHTEST ladder layer, '🚀', 'B' or 'C', and '' outside C: the List column. Ranked on the frame
    given; the layers are tried tightest first, so the first one a stock meets is its tightest."""
    met = playbook_conditions(df)
    return pd.Series(np.select([met[list(layer["conditions"])].all(axis=1).to_numpy() for layer in PLAYBOOK_LAYERS],
                               [layer["tag"] for layer in PLAYBOOK_LAYERS], default=""), index=df.index, dtype=object)


def playbook_layer_scan(scan: dict, key: str) -> dict:
    """What 🚀's table shows for the ladder layer `key`. The 🚀 layer: the scan itself, its table unchanged. B or C: a
    TABLE-ONLY scan dict — the layer's mask, 🚀's order, the List column after Rank and 🆕 hidden (`fixed`: a list with
    no record has no 🆕 baseline). 🔥 shows only where it says something: B lacks only 🚀's Strong Quarter, so there its
    ✓ falls exactly on the rows List marks 🚀 — B hides it as 🚀 does and takes 🚀's widths, which give its 40px to
    Stock and Sector; C shows it with the shared widths. Core keeps its 1,030px either way (List takes New's 55px). No
    recipe words, record inputs or exit reasons: it is in no SCANS, and only the table reads it. Its key is the
    layer's, so each list is its own table widget."""
    if key == PLAYBOOK_LAYERS[0]["key"]:
        return scan
    layer = next(layer for layer in PLAYBOOK_LAYERS if layer["key"] == key)
    lacks_only_quarter = set(PLAYBOOK_LAYERS[0]["conditions"]) - set(layer["conditions"]) == {"quarter"}
    return dict(key=key, order=scan.get("order"), mask=lambda df: playbook_layer_mask(df, key),
                fixed=("_new", "_strong_q") if lacks_only_quarter else ("_new",), extra=("_layer",),
                widths=scan.get("widths", {}) if lacks_only_quarter else {})


def scan_designed_on(scan: dict) -> str:
    """The day this scan's record starts: only a window starting after it counts."""
    return scan.get("designed_on", DESIGNED_ON)


def scan_horizon(scan: dict) -> int:
    """The shortest window that counts as evidence for this scan, in days."""
    return int(scan.get("horizon_days", MIN_HORIZON_DAYS))


def scan_start_month(scan: dict):
    """The one calendar month this scan's unseen windows start in (1-12), or None: every month-end."""
    return scan.get("start_month")


def _month_end(year: int, month: int) -> date:
    return date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)

# Recorded in tools/validate.py; December decides (docs/december-2026-plan.md, sections 2d and 8).
CANDIDATES = (
    ("Supply-chain velocity", "cash conversion cycle down 15+ days and revenue up more than 15%"),
    ("WC Bullies", "3-year revenue growth above 20%, cash cycle not worse, cash flow at least profit"),
    ("J-Curve inflection", "operating margin and profit accelerating while capital work converts"),
)


# ── A record never outlives its recipe ─────────────────────────────────────────────────────────────
def _recipe_functions(scan: dict) -> list:
    return [scan["mask"], *scan.get("uses", ())]


_PROJECT_MODULES = ("config", "core", "ui")
_PLAIN = (bool, int, float, str, type(None), dict, list, tuple, set, frozenset)


def _is_project(obj) -> bool:
    return (getattr(obj, "__module__", None) or "").split(".")[0] in _PROJECT_MODULES


class SourceMismatch(RuntimeError):
    """The file on disk no longer holds the code that is running — the app was updated while it ran
    (Streamlit Cloud, 2026-10-09: the old module stayed in memory, the new file arrived, and reading
    the logic by the old line numbers crashed the Scans tab and every tab after it). No fingerprint is
    made from a mismatch; the record is shown as one that could not be checked."""


def _same_code(a, b) -> bool:
    """Two code objects that do the same thing: instructions, names and constants (nested code compared
    the same way). Line numbers are not compared — they are what a twin compiled elsewhere differs in."""
    if a.co_code != b.co_code or a.co_names != b.co_names or len(a.co_consts) != len(b.co_consts):
        return False
    return all(_same_code(x, y) if inspect.iscode(x) and inspect.iscode(y) else repr(x) == repr(y)
               for x, y in zip(a.co_consts, b.co_consts))


def _twin(node: ast.Lambda, code):
    """The lambda's source compiled again, like for like: a lambda that uses its enclosing function's
    variables (a closure) is compiled inside a function that binds them, as where it was written."""
    if code.co_freevars:
        outer = ast.FunctionDef(
            name="_outer", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[ast.Assign(targets=[ast.Name(id=v, ctx=ast.Store())], value=ast.Constant(None))
                  for v in code.co_freevars] + [ast.Return(value=node)],
            decorator_list=[], type_params=[])
        compiled = compile(ast.fix_missing_locations(ast.Module(body=[outer], type_ignores=[])),
                           code.co_filename, "exec")
    else:
        compiled = compile(ast.Expression(body=node), code.co_filename, "eval")
    todo = [compiled]
    while todo:
        for c in todo.pop().co_consts:
            if inspect.iscode(c):
                if c.co_name == "<lambda>":
                    return c
                todo.append(c)
    return None


def _lambda_node(fn) -> ast.Lambda:
    """A lambda's own source node: found in its module by the line it starts on, and confirmed by
    compiling it back to the code that is running — so two lambdas on one line are told apart and a
    file that changed under the running code is refused, never misread."""
    code = fn.__code__
    try:
        tree = ast.parse(inspect.getsource(inspect.getmodule(fn)))
    except (OSError, TypeError, SyntaxError) as e:
        raise SourceMismatch(f"no readable source for the lambda at line {code.co_firstlineno}") from e
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda) and node.lineno == code.co_firstlineno:
            twin = _twin(node, code)
            if twin is not None and _same_code(twin, code):
                return node
    raise SourceMismatch(f"{code.co_filename}:{code.co_firstlineno} no longer holds this lambda")


def _logic(fn) -> str:
    """A function's LOGIC: its parsed SOURCE without the docstring or its own name, so a reworded
    comment or a rename keeps a record valid and a changed threshold does not. A lambda (one that sits
    inside a settings table, e.g. a Sectors measure) is read from its module's source the same way.
    Never the compiled bytecode: it differs between Python versions (3.13 here and in the weekly job,
    3.14 on Streamlit Cloud), and a record must read the same everywhere — on bytecode the 🌬️ record
    read 'recipe changed' on the live app from 2026-10-07 to 2026-10-09."""
    if fn.__name__ == "<lambda>":
        return ast.dump(_lambda_node(fn))
    try:
        node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    except (OSError, TypeError, SyntaxError) as e:
        raise SourceMismatch(f"no readable source for {fn.__module__}.{fn.__qualname__}") from e
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != fn.__name__:
        raise SourceMismatch(f"{fn.__module__}.{fn.__qualname__}: line {fn.__code__.co_firstlineno} of its "
                             f"file no longer holds it")
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)):
        node.body = node.body[1:]
    node.name = "_"
    return ast.dump(node)


def _canon(obj, found: list) -> str:
    """A settings value as stable text: dicts and sets sorted (set order changes between runs), a
    function inside it reduced to its logic and followed like any other dependency."""
    if isinstance(obj, dict):
        return "{" + ",".join(sorted(f"{_canon(k, found)}:{_canon(v, found)}" for k, v in obj.items())) + "}"
    if isinstance(obj, (set, frozenset)):
        return "{" + ",".join(sorted(_canon(v, found) for v in obj)) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_canon(v, found) for v in obj) + "]"
    if inspect.isfunction(obj):
        found.append(obj)
        return f"<fn {_fn_key(obj)}>"
    return repr(obj)


def _fn_key(fn) -> str:
    """A function's name for the record; lambdas share one name, so theirs carries their logic."""
    key = f"{fn.__module__}.{fn.__qualname__}"
    if fn.__name__ == "<lambda>":
        key += "#" + hashlib.sha256(_logic(fn).encode("utf-8")).hexdigest()[:12]
    return key


def _referenced_globals(fn) -> set:
    """The global names a function's code mentions: its own, its lambdas' and its comprehensions'."""
    names, todo = set(), [fn.__code__]
    while todo:
        code = todo.pop()
        names |= set(code.co_names)
        todo += [c for c in code.co_consts if inspect.iscode(c)]
    return names


def _dependencies(funcs, values=None, skip=()) -> dict:
    """Every PRISM function and settings value the given functions reach, followed to the end: the
    helpers they call, the helpers those call, the weight tables they read (config included). Names
    are looked up where each function lives, so a value imported from config counts wherever it is
    read. Libraries (numpy, pandas) are not followed; their versions are pinned in requirements.txt.
    `values` adds named settings directly (and follows the functions inside them); `skip` names a
    settings table that is covered that way instead of whole."""
    out, todo = {}, list(funcs)
    for label, value in sorted((values or {}).items()):
        found = []
        out[f"value {label}"] = _canon(value, found)
        todo += [f for f in found if _is_project(f)]
    while todo:
        fn = todo.pop()
        key = f"fn {_fn_key(fn)}"
        if key in out:
            continue
        out[key] = _logic(fn)
        for name in sorted(_referenced_globals(fn)):
            if name not in fn.__globals__:      # a method name or a builtin, not a module-level value
                continue
            obj = fn.__globals__[name]
            if inspect.isfunction(obj) and _is_project(obj):
                todo.append(obj)
            elif isinstance(obj, _PLAIN) and not name.startswith("__") and name not in skip:
                found = []
                out[f"value {name}"] = _canon(obj, found)
                todo += [f for f in found if _is_project(f)]
    return out


def _fingerprint_of(funcs, values=None, skip=()) -> str:
    """Hash of the LOGIC a record depends on: the given functions and everything they reach
    (_dependencies). Widened 2026-10-07: the fingerprint used to cover the recipe's own code only, so a
    change to how momentum_score or improvement_score is computed changed the list on screen while the
    card still read 'measured'."""
    deps = _dependencies(funcs, values, skip)
    roots = [_logic(fn) for fn in funcs]
    for fn in funcs:                         # hashed above without their names; a rename is no change
        deps.pop(f"fn {_fn_key(fn)}", None)
    text = "\n".join(roots + [f"{k}={v}" for k, v in sorted(deps.items())])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def recipe_fingerprint(scan: dict) -> str:
    return _fingerprint_of([*_recipe_functions(scan), *scan.get("inputs", ())], scan.get("values"),
                           scan.get("skip", ()))


def load_records(path: Path = RECORDS_PATH) -> dict:
    """The committed records; {} when missing, unreadable or another version (cards then say "Not yet
    measured" rather than guess)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("version") != RECORD_VERSION or "scans" not in data:
        return {}
    return data


def load_checks(path: Path = CHECKS_PATH) -> list:
    """The hand checks (CHECKS_PATH). Anything it cannot read raises — a misread check would hold a month for ever or
    let an unchecked one through, so the job stops instead (main): a missing file (OSError: unverifiable, not "no
    checks", or every recorded demerger of a stock that fell less than the bar would be dropped), and ValueError for
    a check without every field of CHECK_FIELDS, a company_id not spelled as the job spells it (CHECK_ID), a date
    that is not a real YYYY-MM-DD, `from` after `through`, `through` after the day the list was read (checked_on: a
    check vouches only for dates its list could show then), a demerger (or null) outside them, or a demerger without
    the list's own words for it (subject). A BOM, as PowerShell writes one, is read."""
    doc = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    checks = doc.get("checks") if isinstance(doc, dict) else None
    if not isinstance(checks, list):
        raise ValueError("no list of checks under 'checks'")
    for c in checks:
        if not isinstance(c, dict):
            raise ValueError(f"a check is not an object: {c!r}")
        missing = [f for f in CHECK_FIELDS if f not in c]
        if missing:
            raise ValueError(f"a check misses {missing}: {c}")
        if not isinstance(c["company_id"], str) or not CHECK_ID.fullmatch(c["company_id"]):
            raise ValueError(f"a check's company_id is not an id as the job spells it (NSE:… or BSE:…): {c}")
        frm, through, read = _check_date(c, "from"), _check_date(c, "through"), _check_date(c, "checked_on")
        if frm > through:
            raise ValueError(f"a check's from is after its through: {c}")
        if through > read:
            raise ValueError(f"a check's through is after the day it was read (checked_on): {c}")
        if c["demerger"] is not None and not frm <= _check_date(c, "demerger") <= through:
            raise ValueError(f"a check's demerger is outside the dates it read: {c}")
        if not isinstance(c["subject"], str):
            raise ValueError(f"a check's subject is not text: {c}")
        if c["demerger"] is not None and not c["subject"].strip():
            raise ValueError(f"a check's demerger needs its subject, the list's own words for it: {c}")
    return checks


def _check_date(c: dict, field: str) -> date:
    """A check's date, in the one form the job compares as text (YYYY-MM-DD): fromisoformat alone also reads 20260907
    or a week date, which would sort wrongly against the window's dates."""
    v = c[field]
    try:
        d = date.fromisoformat(v) if isinstance(v, str) else None
    except ValueError:
        d = None
    if d is None or d.isoformat() != v:
        raise ValueError(f"a check's {field} is not a YYYY-MM-DD date: {c}")
    return d


def record_state(records: dict, scan: dict):
    """("measured", rec) · ("stale", rec) when the recipe changed since it was measured · ("unverified",
    rec) when the recipe's logic could not be read just now · ("unmeasured", None). A record nobody
    could check shows no numbers, and it never takes the app down: on Streamlit Cloud a push can land
    while the app runs (old code in memory, new files on disk), and on 2026-10-09 reading the logic
    then crashed the Scans tab and every tab after it. A page reload clears it."""
    rec = (records.get("scans") or {}).get(scan["key"])
    if not rec:
        return "unmeasured", None
    try:
        current = recipe_fingerprint(scan)
    except Exception:          # SourceMismatch, or a file unreadable mid-deploy: never a crash
        return "unverified", rec
    if rec.get("fingerprint") != current:
        return "stale", rec
    return "measured", _current_recipe_only(rec, current)


def _current_recipe_only(rec: dict, fingerprint: str) -> dict:
    """The record as every reader counts it (2026-10-15): unseen windows measured with THIS recipe only. Each
    window carries the fingerprint it was measured with, and a final window is kept forever, so after a
    recipe change the file still holds the old recipe's windows — set aside under out_of_sample_earlier, never
    counted beside the new recipe's. A window with no fingerprint cannot be verified, so it is not counted
    either. The file itself is never edited; with nothing to set aside, the record comes back unchanged."""
    ws = rec.get("out_of_sample") or []
    now = [w for w in ws if w.get("fingerprint") == fingerprint]
    if len(now) == len(ws):
        return rec
    return {**rec, "out_of_sample": now,
            "out_of_sample_earlier": [w for w in ws if w.get("fingerprint") != fingerprint]}


# ── The list ───────────────────────────────────────────────────────────────────────────────────────
MEASURED_FLAGS = {"rf_wc_double_squeeze": "Squeeze", "rf_receivables_bloat": "Receivables",
                  "rf_negative_fcf": "Neg FCF"}
# The liquidity floor's steps, the order-size rule and the traded-value number itself are shared with
# the sidebar, the Deep Scanner and the Tear-Sheet (ui/ui_liquidity.py, core.data_engine.traded_value_cr).


def liquidity_help() -> str:
    """The Liquidity selector's help: what the number is, that it is a preference, and which step fits
    an order size — plus the one tradability risk the data cannot see."""
    guide = order_guide()
    return ("Minimum average daily traded value over 20 days (average volume × price). Your preference, "
            "not part of the record: the record counts every match.\n\n"
            f"**Which step?** Keep one order under about {ORDER_SHARE:.0%} of a day's traded value, so the "
            f"market can take it without moving the price much:\n\n{guide}\n\n"
            + SURVEILLANCE_NOTE)

TABLE_COLUMNS = ("rank", "_new", "name", "market_category", "sector", "composite_score", "momentum_score",
                 "improvement_score", "_strong_q", "_measured_flags", "weinstein_stage", "_liq_cr")
# Set widths (an auto-sized text column grows to its longest cell): 1,030px, the room a 1536px laptop
# leaves after the row-selection column (the Deep Scanner's measured budget; the live grid measured
# 1,038px for the columns on 2026-10-14). Category (2026-10-14, the user's ask, before Sector) took its 75px
# from width no header or common value used, measured that day in the grid's own font (14px Source Sans +
# 16px padding; tests/test_scan_views.py holds the floors): Rank 55 → 50 (header 46, "2717" 44), ₹ Cr/day
# 80 → 65 (header 64, "1234.56" 62), ⚠ Measured flags 135 → 125 (header 119, "Squeeze · Neg FCF" 122; the
# common "Receivables · Neg FCF" needs 142 and was cut at 135 too) — and from the Score bar, 105 → 75, the
# width 🌱 already showed it at. Stock and Sector gained on 🌬️ (names in full 68% → 72%; its sector names
# fit 27% anywhere from 90 to 130px).
TABLE_WIDTHS = {"rank": 50, "_new": 55, "name": 175, "market_category": 75, "sector": 95,
                "composite_score": 75, "momentum_score": 85, "improvement_score": 100, "_strong_q": 40,
                "_measured_flags": 125, "weinstein_stage": 90, "_liq_cr": 65, "_cycle": 60,
                # List (2026-10-30): the 🚀 ladder's B and C tables, in New's 55px (New is hidden there).
                "_layer": 55,
                # 📈 Technical (2026-10-13): the Deep Scanner's widths for the same columns, Breakout as a
                # number like Momentum here; "🔥 OVERHEATED" fits 110.
                "close_price": 80, "dist_to_vstop": 60, "d49_momentum_quality": 110, "rsi_14d": 50,
                "breakout_score": 85, "crs_52w": 60,
                # 🔢 RAM (2026-11-05), each column's header and widest value on the 4 Nov copy, by the method above: # 45 ("1000"
                # 44), RAM Rank 75 (header 74), Symbol 105 (widest on the five lists 103), RAM % 57 (header 56), RAM 46 ("-0.12" 45).
                "#": 45, "ram_rank": 75, "symbol": 105, "ram_pct": 57, "ram_score": 46}
# The table's views (2026-10-13), by the app-wide names (ui_group_views.VIEWS — the Deep Scanner's and
# Market Pulse's). 🏆 Core answers "why is it in the list"; 📈 Technical answers "is now a
# good moment, how far is the stop, can I buy one share": the Deep Scanner's Technical columns (without
# Readiness and Off ATH, both folded into Breakout) plus Price and ₹ Cr/day, which decide whether a
# small account can buy a share at all. Mom. Quality sits between the two numbers it reads. Adding these
# to Core would have pushed it ~450px past a 1536px laptop. A display choice: it changes no list, record
# or order, and the 🧹 Clear never resets it.
SCAN_VIEWS = ("🏆 Core", "📈 Technical")
TECHNICAL_COLUMNS = ("rank", "_new", "name", "close_price", "_liq_cr", "dist_to_vstop", "momentum_score",
                     "d49_momentum_quality", "rsi_14d", "breakout_score", "crs_52w", "weinstein_stage")
# 🔢 RAM (2026-11-05, the operator's call): a list you act on answers one question, so every list opens on one table of its
# own, by RAM Rank, strongest first, no RAM last: the 🚀 Playbook, B and C first, 🌬️ and 🌱 the same evening. Core and Technical
# stay as they are, on every list, and one choice of a view carries from list to list. Price
# is there because one share must fit the account. Display only: only the tab reads this view and its order, and no
# list, record, feed or fingerprint moves (tests/test_ram_display.py); the order check sorts RAM Rank on its own frame
# (scan_records_job.order_reading). It takes the 1,030px with nothing
# shrunk: every name on the five lists fits Stock 312 (5 Nov: 🌱's longest needs 310), and Sector 170 shows 79-82% of its
# names in full where 125 showed 73% (seen cut in the browser: "Capital Goods - Elec", "Air Transport Servic"). On 🌬️ it
# shows 57%: one long name, "E-Commerce/App based Aggregator", fills 32 of its 74 rows that day, and the five sectors change.
RAM_VIEW = "🔢 RAM"
RAM_VIEW_COLUMNS = ("#", "ram_rank", "symbol", "name", "ram_pct", "ram_score", "close_price", "market_category", "sector",
                    "_liq_cr")
RAM_VIEW_WIDTHS = {"name": 312, "sector": 170}
TABLE_VIEWS = (RAM_VIEW, *SCAN_VIEWS)         # what every list offers, the first its default
# The engine's Weinstein labels, shortened so the column fits; the header tip carries the words.
TREND_SHORT = {"📈 Stage 2 Advancing": "📈 Stage 2", "🔄 Stage 1 Basing": "🔄 Stage 1",
               "⚠️ Stage 3 Top": "⚠️ Stage 3", "📉 Stage 4 Declining": "📉 Stage 4"}


# Columns a scan may ADD to the shared table (2026-10-07), each placed after an anchor column. Only a
# scan that lists one in `extra` shows it; the others keep exactly the shared table. List (2026-10-30): the 🚀
# ladder's B and C tables name each row's tightest layer, right after Rank in both views.
EXTRA_COLUMNS = {"_cycle": "_strong_q", "_layer": "rank"}


def table_columns(scan: dict, view: str = SCAN_VIEWS[0]) -> tuple:
    """The columns this scan shows in `view`. 🏆 Core: every table column except those its recipe fixes,
    plus its own extra columns, each right after its anchor. 📈 Technical: TECHNICAL_COLUMNS except the
    fixed ones (none is fixed by a recipe; the ladder's B and C hide New), plus an extra only where the view
    shows its anchor (2026-10-30): List after Rank, while 🌱's Cycle ↓, anchored on 🔥, stays out. 🔢 RAM: its own columns,
    the same on every list."""
    if view == RAM_VIEW:
        return RAM_VIEW_COLUMNS
    if view == SCAN_VIEWS[1]:
        cols = [c for c in TECHNICAL_COLUMNS if c not in scan.get("fixed", ())]
        extras = [e for e in scan.get("extra", ()) if EXTRA_COLUMNS[e] in cols]
    else:
        cols = [c for c in TABLE_COLUMNS if c not in scan.get("fixed", ())]
        extras = scan.get("extra", ())
    for extra in extras:
        cols.insert(cols.index(EXTRA_COLUMNS[extra]) + 1, extra)
    return tuple(cols)


def table_widths(scan: dict) -> dict:
    """Column widths for this scan: the shared widths, with the scan's own overrides."""
    return {**TABLE_WIDTHS, **scan.get("widths", {})}


def ram_view_scan(scan: dict) -> dict:
    """The scan as the 🔢 RAM view draws it: its widths are the view's, whichever list it is."""
    return dict(scan, widths=RAM_VIEW_WIDTHS)


def ram_view_table(table: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """The 🔢 RAM view of a finished list (scan_table's, filters and floor applied), with RAM from the market frame beside it:
    RAM Rank low to high, no RAM last. The sort is stable, so on a tie and among the rows with no RAM the list keeps the
    order it came in (🚀's and 🌬️'s: momentum high to low, then Rank; 🌱's: Rank), and a list with no RAM at all reads as it came. # is
    the place as shown, so a filter renumbers it. Symbol is the id without its exchange."""
    ram = df.reindex(columns=["company_id", "ram_rank", "ram_pct", "ram_score", "ram_date"])
    out = (table.merge(ram, on="company_id", how="left")
           .sort_values("ram_rank", kind="mergesort", na_position="last").reset_index(drop=True))
    return out.assign(**{"#": np.arange(1, len(out) + 1),
                         "symbol": out["company_id"].astype(str).str.split(":", n=1).str[-1]})


def ram_view_order_words(table: pd.DataFrame, today: str, df: pd.DataFrame) -> str:
    """What the view's caption says it is sorted by and how old its ranks are: the newest RAM Date among the rows shown. With
    no RAM in the list it reads `today`, the list's own order, and says whether the whole copy `df` has none (an old copy,
    or before the first RAM night) or only this list (a BSE-only list on a copy that has RAM)."""
    if table["ram_rank"].notna().any():
        newest = ram_day(ram_date_text(table["ram_date"]).dropna().max())
        return "RAM Rank, strongest first · no RAM last" + (f" · RAM of {newest}" if newest else "")
    where = "in this list" if df.reindex(columns=["ram_rank"])["ram_rank"].notna().any() else "on this copy"
    return f"{today} · no RAM {where}"


def momentum_ordered(scan: dict) -> bool:
    """True for a scan whose table lists by momentum, high to low (its `order`), not by PRISM Rank."""
    return scan.get("order") == "momentum"


def scan_table(df: pd.DataFrame, scan: dict, members_at_latest=None, min_cr=None,
               keep_ids=None, result_window_days=None) -> pd.DataFrame:
    """The scan's stocks — by momentum, high to low, for a momentum-ordered scan (a tie goes to the better
    Rank, a stock with no momentum reading last), by PRISM Rank for the rest. The recipe runs on the
    frame given (the whole market); the liquidity floor and the sidebar's kept stocks (keep_ids) only
    narrow the FINISHED list, so a listed stock is exactly the match its record measured. A stock with
    no traded value cannot clear a floor. Flagged stocks stay, with their measured flags named. 🆕 marks
    a stock that was not in the scan on the 🆕 baseline (the last archived month-end copy; no baseline =
    nothing marked). 📄 marks a stock whose quarterly result was DECLARED inside the last
    `result_window_days` (the days since that baseline): result_age_days counts days since the last
    declared result and is negative for one only scheduled, so 0 <= age < window is "declared since the
    baseline" — the rule Movers uses. The two clocks differ on purpose: technical legs move daily,
    fundamentals only when a result lands, so 🆕 + 📄 together means the stock entered on a fresh result.
    On the 🚀 ladder's B and C tables (2026-10-30) List names each row's tightest layer, ranked on the whole
    market like the recipe, never on the narrowed list; only a table that shows it carries it."""
    keep = scan["mask"](df)
    out = df.loc[keep].copy()
    if "_layer" in scan.get("extra", ()):
        out["_layer"] = playbook_layer_tag(df)[keep].to_numpy()
    out["_liq_cr"] = traded_value_cr(out)
    if min_cr is not None:
        out = out[out["_liq_cr"] >= min_cr]
    if keep_ids is not None:
        out = out[out["company_id"].astype(str).isin(keep_ids)]
    named = pd.Series("", index=out.index, dtype=object)
    for col, lab in MEASURED_FLAGS.items():                 # three columns, never a loop over rows
        named = named.where(~(_num(out, col) == 1), np.where(named == "", lab, named + " · " + lab))
    out["_measured_flags"] = named
    new = (pd.Series(False, index=out.index) if members_at_latest is None
           else ~out["company_id"].astype(str).isin(set(members_at_latest)))
    age = _num(out, "result_age_days")
    fresh = ((age >= 0) & (age < result_window_days) if result_window_days
             else pd.Series(False, index=out.index))
    out["_new"] = np.select([new & fresh, new, fresh], ["🆕 📄", "🆕", "📄"], default="")
    out["_strong_q"] = np.where(strong_quarter(out) == 1, "✓", "")
    out["_cycle"] = np.where(cycle_shortened(out), "✓", "")
    if "weinstein_stage" in out.columns:
        out["weinstein_stage"] = out["weinstein_stage"].replace(TREND_SHORT)
    shown = [*TABLE_COLUMNS, *(c for c in TECHNICAL_COLUMNS if c not in TABLE_COLUMNS),
             *(c for c in EXTRA_COLUMNS if c in out.columns)]
    for c in shown:
        if c not in out.columns:
            out[c] = np.nan
    out = (out.sort_values(["momentum_score", "rank"], ascending=[False, True], kind="mergesort")
           if momentum_ordered(scan) else out.sort_values("rank", kind="mergesort"))
    return out[[*shown, "company_id"]].reset_index(drop=True)


def _change(df: pd.DataFrame, now: str, prev: str) -> pd.Series:
    """now / prev - 1 in %, wherever the earlier year is positive (a fall into a loss still shows)."""
    a, b = _num(df, now), _num(df, prev)
    return ((a / b - 1.0) * 100).where(b > 0)


def left_scan(df: pd.DataFrame, scan: dict, members_at_latest=None, keep_ids=None) -> pd.DataFrame:
    """Stocks in the scan on the 🆕 baseline that are not in it now, each with why: columns company_id,
    name, sector, _why, _sales_chg, _profit_chg, reason — the worst news first, then by name.

    Empty for a scan without an exit_reason or without a baseline (members_at_latest is None, as it is
    whenever the record is stale — the same rule the 🆕 marks follow). The recipe runs on the whole
    market; the liquidity floor does NOT narrow this list (a stock does not leave 🌱 by trading less),
    keep_ids does, and a stock gone from today's data is kept only when nothing narrows the list."""
    cols = ["company_id", "name", "sector", "_why", "_sales_chg", "_profit_chg", "reason"]
    if members_at_latest is None or "exit_reason" not in scan:
        return pd.DataFrame(columns=cols)
    ids = df["company_id"].astype(str)
    left = set(map(str, members_at_latest)) - set(ids[scan["mask"](df)])
    if keep_ids is not None:
        left &= set(map(str, keep_ids))
    here = df.loc[ids.isin(left)]
    out = pd.DataFrame({"company_id": ids[here.index], "name": here.get("name"), "sector": here.get("sector"),
                        "_sales_chg": _change(here, "revenue", "revenue_1yb"),
                        "_profit_chg": _change(here, "pat", "pat_1yb"),
                        "reason": scan["exit_reason"](here)})
    gone = sorted(left - set(out["company_id"]))
    if gone:
        out = pd.concat([out, pd.DataFrame({"company_id": gone, "name": gone, "sector": "",
                                            "reason": "gone"})], ignore_index=True)
    out["_why"] = out["reason"].map(lambda r: f"{EXIT_REASONS[r][0]} {EXIT_REASONS[r][1]}")
    order = {r: k for k, r in enumerate(EXIT_REASONS)}
    out = out.assign(_o=out["reason"].map(order)).sort_values(["_o", "name"], kind="mergesort")
    return out[cols].reset_index(drop=True)


def left_summary(left: pd.DataFrame) -> str:
    """'🔴 2 profit fell · 🟠 1 no longer revenue-led' — the counts, worst first; '' when none left."""
    counts = left["reason"].value_counts()
    return " · ".join(f"{EXIT_REASONS[r][0]} {counts[r]} {EXIT_REASONS[r][1]}"
                      for r in EXIT_REASONS if r in counts)


def left_legend() -> str:
    """One line per reason: its mark, its words and what it means."""
    return "\n".join(f"- {m} **{lab}** — {why}" for m, lab, why in EXIT_REASONS.values())


def table_config(members_date_label: str, scan: dict = None) -> dict:
    """Column headers — the same header each column carries in the Deep Scanner — widths and tips."""
    w = table_widths(scan or {})
    cc = st.column_config
    since = (f"on {members_date_label}, the last archived month-end copy" if members_date_label
             else "on the last archived month-end copy")
    return {
        "rank": cc.NumberColumn("Rank", format="%.0f", width=w["rank"], help=(
            _UNRECORDED_RANK_TIP if "_layer" in (scan or {}).get("extra", ()) else      # 🚀's B or C
            "PRISM's overall rank — click this header, then Sort ascending, to list by it. The list is "
            "sorted by Momentum, high to low (since 2026-10-13); " + (
                "the card's order check reads RAM Rank, and its figures are sealed for now" if ORDER_SEALED else
                "the card's order check records whether RAM Rank's order keeps beating random picks") +
            ". Every match is the scan." if momentum_ordered(scan or {}) else
            "PRISM's overall rank. The list is sorted by it for reading only: every match is the scan.")),
        "_new": cc.TextColumn("New", width=w["_new"],
                              help=f"🆕 = not in this scan {since}. 📄 = its quarterly result was declared "
                                   f"since then, so its Strong Quarter and Improvement readings are fresh. "
                                   f"🆕 📄 together = it entered on a new result; 🆕 alone = it entered on "
                                   f"price movement (technical legs move daily — check it is not flickering "
                                   f"across the line)."),
        # List (2026-10-30): the 🚀 ladder's B and C tables only (PLAYBOOK_LAYERS).
        "_layer": cc.TextColumn("List", width=w["_layer"], help=(
            "The tightest of the three lists the stock is in: 🚀 = the whole 🚀 recipe · B = all of it but the 🔥 "
            "Strong Quarter · C = price only — Momentum and Breakout in the top third of the market and the price "
            "above its stop. Each list holds the one before it. B and C have no record of their own: the card and its "
            "record are the 🚀 list's.")),
        "name": cc.TextColumn("Stock", width=w["name"]),
        # "Category" (2026-10-14): the user's word and the data vendor's own ("Market Category"); the Deep
        # Scanner shows the same column as Market Cap.
        "market_category": cc.TextColumn(
            "Category", width=w["market_category"],
            help="The company's market-cap category from the data vendor: Mega, Large, Mid, Small, Micro or Nano "
                 "Cap — the size band the Market cap filter above narrows by. The Deep Scanner calls this "
                 "column Market Cap."),
        "sector": cc.TextColumn("Sector", width=w["sector"]),
        "composite_score": cc.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f",
                                             width=w["composite_score"]),
        "momentum_score": cc.NumberColumn("Momentum", format="%.0f", width=w["momentum_score"]),
        # 🔢 RAM's own columns (2026-11-05); the rest of its table is the columns below, with their own headers and tips.
        "#": cc.NumberColumn("#", format="%d", width=w["#"],
                             help="This row's place in the list as shown, after the filters and the sort. A filter renumbers it."),
        "ram_rank": cc.NumberColumn("RAM Rank", format="%.0f", width=w["ram_rank"], help=_SCANNER_HEADER_TIPS["ram_rank"]),
        "symbol": cc.TextColumn("Symbol", width=w["symbol"],
                                help="The stock's symbol without its exchange, to type into a broker or a chart."),
        "ram_pct": cc.NumberColumn("RAM %", format="%.1f", width=w["ram_pct"], help=_SCANNER_HEADER_TIPS["ram_pct"]),
        "ram_score": cc.NumberColumn("RAM", format="%.2f", width=w["ram_score"], help=_SCANNER_HEADER_TIPS["ram_score"]),
        "improvement_score": cc.NumberColumn("Improvement", format="%.0f", width=w["improvement_score"]),
        "_strong_q": cc.TextColumn("🔥", width=w["_strong_q"],
                                   help="🔥 Strong Quarter: latest quarter vs the same quarter last year — "
                                        "revenue up more than 15%, profit up more than 20%, margin wider."),
        "_measured_flags": cc.TextColumn(
            "⚠ Measured flags", width=w["_measured_flags"],
            help="The red flags that went with weaker returns in the 2026-10-06 measurement (provisional; "
                 "December re-tests them): working-capital double squeeze (customers pay 10+ days slower "
                 "AND suppliers are paid 10+ days faster) — weaker even for improving businesses; "
                 "receivables bloat and negative free cash flow — weaker mainly for businesses that are "
                 "not improving, and negative free cash flow alongside capex mirage was not weaker. "
                 "Shown, never removed: open the Tear-Sheet's Forensics tab before deciding."),
        "weinstein_stage": cc.TextColumn(
            "Trend", width=w["weinstein_stage"],
            help="Weinstein stage — the price against its 30-week average, and that average against the "
                 "200-day: 📈 Stage 2 advancing · 🔄 Stage 1 basing · ⚠️ Stage 3 topping · 📉 Stage 4 "
                 "declining · ❔ not enough price history. Being above its stop is not the same line, so "
                 "a Stage 3 or 4 can appear in a scan; open the Tear-Sheet before deciding."),
        "_cycle": cc.TextColumn(
            "Cycle ↓", width=w["_cycle"],
            help="✓ = cash conversion cycle down 15+ days with sales up more than 15%, both in the same "
                 "year as the profit and sales 🌱 is judged on (supply-chain velocity). A company whose "
                 "balance sheet is still last year's stays blank until it updates. A "
                 "reading aid, not part of 🌱's recipe or record."),
        "_liq_cr": cc.NumberColumn(LIQUIDITY_HEADER, format="%.2f", width=w["_liq_cr"],
                                   help="Average daily traded value over 20 days (average volume × price), "
                                        f"in ₹ crore. Keep one order under about {ORDER_SHARE:.0%} of it: "
                                        f"2.0 → about ₹{order_size_lakh(2.0):g} lakh."),
        # 📈 Technical (2026-10-13): the Deep Scanner's header, format and tip for each column.
        "close_price": cc.NumberColumn("Price ₹", format="%.2f", width=w["close_price"],
                                       help=_SCANNER_HEADER_TIPS["close_price"]),
        "dist_to_vstop": cc.NumberColumn("Stop Δ", format="%.1f%%", width=w["dist_to_vstop"],
                                         help=_SCANNER_HEADER_TIPS["dist_to_vstop"]),
        "d49_momentum_quality": cc.TextColumn("Mom. Quality", width=w["d49_momentum_quality"],
                                              help=_SCANNER_HEADER_TIPS["d49_momentum_quality"]),
        "rsi_14d": cc.NumberColumn("RSI", format="%.0f", width=w["rsi_14d"],
                                   help=_SCANNER_HEADER_TIPS["rsi_14d"]),
        "breakout_score": cc.NumberColumn("Breakout", format="%.0f", width=w["breakout_score"],
                                          help=_SCANNER_HEADER_TIPS["breakout_score"]),
        "crs_52w": cc.NumberColumn("RS 52W", format="%.0f", width=w["crs_52w"],
                                   help=_SCANNER_HEADER_TIPS["crs_52w"]),
    }


# ── The card ───────────────────────────────────────────────────────────────────────────────────────
def _month_span(w: dict) -> str:
    a, b = date.fromisoformat(w["start"]), date.fromisoformat(w["end"])
    return f"{a:%b}→{b:%b}"


def _range(vals, fmt) -> str:
    lo, hi = min(vals), max(vals)
    return fmt.format(lo) if lo == hi else f"{fmt.format(lo)}–{fmt.format(hi)}"


def non_overlapping(ws) -> int:
    """The most windows that share no days (2026-10-13). A window ending on the day the next starts is
    separate — the first's end price is the second's start. Greedy by earliest end, which is optimal. Two
    windows from one start date share days: one stretch of the market, not two results."""
    n, last = 0, ""
    for w in sorted(ws, key=lambda w: (w["end"], w["start"])):
        if w["start"] >= last:
            n, last = n + 1, w["end"]
    return n


def order_rule(ws) -> dict:
    """ORDER_RULE applied to a scan's unseen windows: state 'pending' until it has ORDER_RULE['windows']
    separate (non-overlapping, earliest end first — the same greedy choice as non_overlapping), finished
    windows whose order reading has at least min_n stocks and places both RAM Rank and Rank; then 'held' or
    'did not hold' on those first windows only, so a later window can never reopen it. A window from before the
    reading, or from a copy with no RAM, does not count. counted = windows that count so far."""
    read = lambda w: w.get("order") or {}
    usable = [w for w in ws if w.get("final") and int(read(w).get("n", 0)) >= ORDER_RULE["min_n"]
              and read(w).get("ram_pct") is not None and read(w).get("rank_pct") is not None]
    picked, last = [], ""
    for w in sorted(usable, key=lambda w: (w["end"], w["start"])):
        if w["start"] >= last and len(picked) < ORDER_RULE["windows"]:
            picked.append(w)
            last = w["end"]
    out = dict(counted=len(picked), state="pending")
    if len(picked) == ORDER_RULE["windows"]:
        ram = float(np.mean([float(w["order"]["ram_pct"]) for w in picked]))
        rank = float(np.mean([float(w["order"]["rank_pct"]) for w in picked]))
        held = ram >= ORDER_RULE["min_avg_pct"] and ram > rank
        out.update(state="held" if held else "did not hold", ram_avg=ram, rank_avg=rank)
    return out


def revenue_led_rule(ws) -> dict:
    """REVENUE_LED_RULE applied to a scan's unseen windows: 'pending' until it has REVENUE_LED_RULE['windows']
    separate finished windows whose revenue-led reading has both arms of at least min_n (a reading carries its
    gaps only then; the same greedy choice as order_rule); then 'held' or 'did not hold' on those first windows
    only, so a later window can never reopen it. counted = windows that count so far."""
    rule = REVENUE_LED_RULE
    usable = [w for w in ws if w.get("final") and "mean_gap_pp" in (w.get("revenue_led") or {})]
    picked, last = [], ""
    for w in sorted(usable, key=lambda w: (w["end"], w["start"])):
        if w["start"] >= last and len(picked) < rule["windows"]:
            picked.append(w)
            last = w["end"]
    out = dict(counted=len(picked), state="pending")
    if len(picked) == rule["windows"]:
        mean = [float(w["revenue_led"]["mean_gap_pp"]) for w in picked]
        median = [float(w["revenue_led"]["median_gap_pp"]) for w in picked]
        held = all(a >= rule["min_gap_pp"] and b >= rule["min_gap_pp"] for a, b in zip(mean, median))
        out.update(state="held" if held else "did not hold", mean_gaps=mean, median_gaps=median)
    return out


def order_pool_words() -> str:
    """What the order check's pool passes, from the settings the job reads: one line for the card and the guide."""
    first, last = (str(HOLD_CAPS[i]).replace(" Cap", "") for i in (0, -1))
    return (f"{SHARIAH_STATUSES[0]}, {ORDER_EXCHANGE}, {first} to {last}, ₹{SHARIAH_MIN_CR:g} Cr a day, up to "
            f"₹{HOLD_MAX_PRICE:,.0f} a share, {ORDER_ZONE[0]:g}–{ORDER_ZONE[1]:g}% above the V-Stop, "
            f"Stage {' or '.join(ORDER_STAGES)}")


def _hold_line(scan: dict, ws, c: dict, esc) -> str:
    """Under the unseen record: per window, HOLD_K matches a Rs 1 lakh account can hold, picked at random
    — the typical result after costs and its 10th-90th range — and where Rank's top HOLD_K lands among those
    picks; then, for the scans ORDER_RULE checks, where RAM Rank's and Rank's top HOLD_K land among random picks of
    what Kite can trade, "–" for a window with no such reading, and where the check stands. Empty until a window
    carries the reading."""
    rows = [w for w in ws if "median_pp" in (w.get("hold") or {})]
    if not rows:
        return ""
    k = int(rows[0]["hold"]["k"])
    res = " / ".join(f'{float(w["hold"]["median_pp"]):+.1f}% ({float(w["hold"]["p10_pp"]):+.1f} to '
                     f'{float(w["hold"]["p90_pp"]):+.1f})' for w in rows)
    loss = " / ".join(f'{int(w["hold"]["loss_pct"])}%' for w in rows)
    pct = lambda v: "–" if v is None else f"{int(v)}%"
    rnk = " / ".join(pct(w["hold"].get("rank_pct")) for w in rows)
    h0 = rows[0]["hold"]
    names = [str(v).replace(" Cap", "") for v in h0.get("caps", HOLD_CAPS)]
    caps = ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else "".join(names)
    rule = note = ""
    if scan.get("key") in ORDER_RULE["scans"]:
        # 2026-11-05: every list opens on 🔢 RAM and the check reads RAM Rank whichever way the table lists, so the sentence says
        # whose order is whose.
        on_c = f"{ORDER_ON_C} " if scan.get("key") in ORDER_LIST else ""
        note = (f' 🏆 Core and 📈 Technical list by momentum since 13 Oct and {RAM_VIEW} by RAM Rank; the order check reads RAM Rank. '
                f'What Kite can trade: {order_pool_words()}. {ORDER_NOT_IN_DATA} {on_c}'
                f'The check holds if RAM Rank\'s top {k} beats {ORDER_RULE["min_avg_pct"]:.0f}%+ of random picks of those '
                f'stocks on average over {ORDER_RULE["windows"]} separate windows, and beats Rank\'s. {ORDER_FROM_RAM}, and '
                f'only with {ORDER_RULE["min_n"]}+ of those stocks.' + (f' {ORDER_SEALED_WORDS}' if ORDER_SEALED else ''))
        placed = lambda key: " / ".join(pct((w.get("order") or {}).get(key)) for w in rows)
        v = order_rule(ws)
        rule = " · Order check: sealed until Kite's sealed run" if ORDER_SEALED else (
                f' · Among what Kite can trade, RAM Rank\'s top {k} beat {placed("ram_pct")} of random picks, '
                f'Rank\'s top {k} {placed("rank_pct")}'
                + (f' · Order check: {v["counted"]} of {ORDER_RULE["windows"]} separate windows in'
                   if v["state"] == "pending" else
                   f' · Order check after {ORDER_RULE["windows"]} separate windows: RAM Rank\'s order {v["state"]} '
                   f'(its top {k} beat {v["ram_avg"]:.0f}% of random picks on average, Rank\'s '
                   f'{v["rank_avg"]:.0f}%)'))
    return (f'<div style="font-size:0.7rem;color:{c["text_secondary"]};margin-top:5px;">Holding {k} of them, '
            f'picked at random: {esc(res)} after costs · lost money in {esc(loss)} of picks · '
            f'Rank\'s top {k} beat {esc(rnk)} of them{esc(rule)}</div>'
            f'<div style="font-size:0.66rem;color:{c["text_muted"]};margin-top:2px;">From the {esc(caps)} cap '
            f'matches with one share at most ₹{float(h0["max_price"]):,.0f} (what ₹1 lakh across {k} can hold, '
            f'whatever they trade a day), {int(h0["draws"]):,} random picks each window with a fixed seed, a '
            f'{float(h0["cost_pp"]):.1f}pp round trip.{esc(note)}</div>')


def _points(x: float) -> str:
    return f"{x:g} point" + ("" if x == 1 else "s")


def _revenue_led_line(scan: dict, ws, c: dict, esc) -> str:
    """Under the unseen record of a scan REVENUE_LED_RULE checks: per window, its matches that are revenue-led
    against the rest — the gap in their mean and median excess, and each side's size — then where the check
    stands. Empty for other scans and until a window carries the reading."""
    if scan.get("key") not in REVENUE_LED_RULE["scans"]:
        return ""
    rows = [w["revenue_led"] for w in ws if "n_led" in (w.get("revenue_led") or {})]
    if not rows:
        return ""
    rule = REVENUE_LED_RULE
    gaps = lambda key: " / ".join("–" if key not in r else f"{float(r[key]):+.1f}" for r in rows)
    sizes = " / ".join(f'{int(r["n_led"])} vs {int(r["n_rest"])}' for r in rows)
    v = revenue_led_rule(ws)
    state = (f'Revenue-led check: {v["counted"]} of {rule["windows"]} separate windows in' if v["state"] == "pending"
             else f'Revenue-led check after {rule["windows"]} separate windows: {v["state"]}')
    return (f'<div style="font-size:0.7rem;color:{c["text_secondary"]};margin-top:5px;">Revenue-led vs the rest of '
            f'the list: {esc(gaps("mean_gap_pp"))} points on the mean, {esc(gaps("median_gap_pp"))} on the middle '
            f'stock ({esc(sizes)} stocks) · {esc(state)}</div>'
            f'<div style="font-size:0.66rem;color:{c["text_muted"]};margin-top:2px;">Revenue-led = sales up more than '
            f'10% and profit growing faster, the test 🌱 uses. The check holds if the revenue-led matches beat the '
            f'rest by at least {esc(_points(rule["min_gap_pp"]))} on the mean and on the middle stock in each of the '
            f'first {rule["windows"]} separate windows with {rule["min_n"]}+ stocks on each side. Registered on 15 Oct '
            f'2026, before any of these windows; it changes nothing in the list or in Kite.</div>')


def _overlap_note(ws, sep: str, end: str = "") -> str:
    """'<sep>K non-overlapping<end>' when the windows overlap; nothing when every one is independent."""
    k = non_overlapping(ws)
    return f"{sep}{k} non-overlapping{end}" if k < len(ws) else ""


def _bars(ws, c, esc) -> str:
    """One bar per window: height by size (12 points fills it), green above the median, red below. A
    window still RUNNING (it ends on the latest copy, which changes until its month is over) reads
    "so far" under its bar (2026-10-15), so a reading in progress never looks like a result. That line is kept
    under every bar of such a row, blank under a finished one, so the bars still stand on one baseline."""
    out = ""
    tagged = any(not w.get("final") for w in ws)
    for w in ws:
        v = float(w["median_excess_pp"])
        h = 6 + round(28 * min(abs(v) / 12.0, 1.0))
        col = c["green"] if v > 0 else c["red"]
        running = not w.get("final")
        tip = (f'{w["start"]} → {w["end"]}, {int(w["days"])} days, {int(w["n"])} stocks'
               + (", still running" if running else ""))
        out += (f'<div style="display:flex;flex-direction:column;align-items:center;justify-content:flex-end;'
                f'min-width:46px;" title="{esc(tip)}">'
                f'<div style="width:14px;height:{h}px;background:{col};border-radius:3px;"></div>'
                f'<div style="font-size:0.78rem;font-weight:800;color:{col};margin-top:3px;">{v:+.1f}</div>'
                f'<div style="font-size:0.6rem;color:{c["text_muted"]};">{_month_span(w)}</div>'
                + (f'<div style="font-size:0.6rem;color:{c["gold"]};">{"so far" if running else "&nbsp;"}</div>'
                   if tagged else "")
                + '</div>')
    return out


def _unseen_notes(rec: dict, ws, c: dict, esc) -> str:
    """Under the unseen record (2026-10-15): what "so far" means when a window is still running, how many windows
    an earlier recipe left on file, not counted here, and (2026-10-18) each window's falls to check by hand against
    its exchange's list, since a demerger books like a crash (FALL_CHECK_PCT), with what each check found and every
    stock set aside. Nothing when none applies."""
    muted = f'font-size:0.66rem;color:{c["text_muted"]};margin-top:2px;'
    out = ""
    if any(not w.get("final") for w in ws):
        out += (f'<div style="{muted}">"So far": that window ends on the latest copy, so it is re-measured each '
                f'week until its month is over, then written once and never again.</div>')
    k = len(rec.get("out_of_sample_earlier") or [])
    if k:
        out += (f'<div style="{muted}">{k} unseen window{"s" if k != 1 else ""} measured with an earlier recipe '
                f'{"are" if k != 1 else "is"} kept on file, not counted here: this recipe\'s record starts '
                f'after it was chosen.</div>')
    words = [(f"{esc(str(i))} {float(v):+.1f}% ({_month_span(w)})", _fall_word(w, str(i)))
             for w in ws for i, v in (w.get("falls") or [])]
    if words:
        # Each window's own bar (falls_pct); one recorded before the bar was stored reads today's. "Check" only while
        # one is left to check: every final month has none (the job's stop), and its card must not ask for one.
        bars = sorted({float(w.get("falls_pct", FALL_CHECK_PCT)) for w in ws if w.get("falls")})
        lead = "Check" if any(word == "to check" for _, word in words) else "Checked"
        out += (f'<div style="{muted}">{lead} by hand against its exchange\'s corporate-action list (NSE\'s, or '
                f'BSE\'s for a BSE: id): fell {" or ".join(f"{b:g}%+" for b in bars)} with no change in share '
                f'count, which a demerger shows as well as a crash. A demerger is set aside from its month, like a '
                f'split; a month is written final only once all its falls are checked: '
                f'{" · ".join(f"{fell} {word}" for fell, word in words)}</div>')
    gone = [_set_aside_words(w, esc) for w in ws if w.get("set_aside")]
    if gone:
        out += (f'<div style="{muted}">Set aside as a demerger on its exchange\'s list, like a split: '
                f'{" · ".join(gone)}</div>')
    return out


def _set_aside_words(w: dict, esc) -> str:
    """A month's stocks set aside (the operator's demerger rule, 2026-10-18), named fall or not: each with its ex-date
    and the list's own words, typed by hand and so escaped, then what the month read with them kept in — one reading a
    month, never said of one stock when two were set aside."""
    names = "; ".join(f'{esc(str(i))} ex {date.fromisoformat(d):%d %b} "{esc(str(s))}"' for i, d, s in w["set_aside"])
    them = "it" if len(w["set_aside"]) == 1 else "them"
    return (f"{names} ({_month_span(w)}; with {them} kept in, the month read "
            f"{float(w['as_recorded']['median_excess_pp']):+.1f})")


def _fall_word(w: dict, sid: str) -> str:
    """What a named fall came to (the operator's demerger rule, 2026-10-18): set aside as a demerger (named again
    below with its ex-date and the list's words), checked with none, or still to check — also for a window recorded
    before checks."""
    if sid in {s for s, *_ in (w.get("set_aside") or [])}:
        return "set aside: a demerger"
    return "checked: no demerger" if sid in (w.get("checked") or []) else "to check"


def _first_unseen_start(scan: dict) -> date:
    """The month-end the first unseen window can start on: the first after the scan's record date (scan_designed_on), in its
    start month if it has one. A calendar month-end: the archive dates its copy by the month's last trading day."""
    d = date.fromisoformat(scan_designed_on(scan))
    m = scan_start_month(scan)
    start = _month_end(d.year, m or d.month)
    if start > d:
        return start
    return _month_end(d.year + 1, m) if m else _month_end(d.year, d.month + 1)


def no_record_words(scan: dict) -> str:
    """What a card says before its scan has any window (2026-11-05). The year is shown when it differs from the record date's."""
    start = _first_unseen_start(scan)
    day = f"{start:%d %b}" if start.year == date.fromisoformat(scan_designed_on(scan)).year else f"{start:%d %b %Y}"
    return f"No record yet. The first window starts on the {day} copy."


def _extras(ws, c, esc) -> str:
    """Under a record: the NET reading (liquid matches only, charged the round trip) and how many matches
    the headline left out. Windows measured before these were recorded say so instead of showing 0."""
    nets = [w.get("net") for w in ws]
    if not any(nets):
        return (f'<div style="font-size:0.66rem;color:{c["text_muted"]};margin-top:3px;">Net-of-costs record '
                f'and left-out counts: not measured for these windows yet</div>')
    cells = []
    for nt in nets:
        if nt and "median_excess_pp" in nt:
            col = c["green"] if nt["median_excess_pp"] > 0 else c["red"]
            cells.append(f'<span style="color:{col};font-weight:700;">{nt["median_excess_pp"]:+.1f}</span>')
        else:
            cells.append('<span title="fewer than 30 liquid matches">–</span>')
    nt0 = next(nt for nt in nets if nt)
    measured = [nt for nt in nets if nt and "median_excess_pp" in nt]
    tail = (f' · {_range([int(nt["beat_pct"]) for nt in measured], "{}")}% beat · '
            f'{_range([int(nt["n"]) for nt in measured], "{}")} stocks' if measured else
            ' · too few liquid matches to measure')
    out = (f'<div style="font-size:0.7rem;color:{c["text_secondary"]};margin-top:5px;">Net of a '
           f'{float(nt0["cost_pp"]):.1f}pp round trip, ₹{float(nt0["min_cr"]):g} Cr+/day matches only: '
           f'{" / ".join(cells)}{tail}</div>')
    if all("dropped_no_return" in w for w in ws):
        gone = _range([int(w["dropped_no_return"]) for w in ws], "{}")
        ca = _range([int(w["dropped_corp_action"]) for w in ws], "{}")
        out += (f'<div style="font-size:0.66rem;color:{c["text_muted"]};margin-top:2px;">Left out per window: '
                f'{gone} with no price at the end (delisted, suspended or missing) · {ca} corporate '
                f'actions</div>')
    return out


def _index_readings(scan: dict, ws) -> list:
    """For each unseen window that carries the comparison: (window, edge) — the portfolio's return after
    costs minus the scan's own index after the fund fee, in points; edge None where the window had too
    few liquid matches for a portfolio or no reading of that index."""
    code = scan.get("benchmark")
    out = []
    for w in ws:
        v = w.get("vs_index")
        if not code or not v:
            continue
        p, i = v.get("portfolio_pp"), (v.get("indices") or {}).get(code)
        out.append((w, None if p is None or i is None else float(p) - float(i)))
    return out


def _vs_index(scan: dict, ws, c: dict, esc) -> str:
    """Under the unseen record (December plan rule 23): the scan's liquid matches as one equal-weight
    portfolio against the index fund it is judged on, both after costs, per window — every recorded
    index one hover away. A progress reading: the plan's verdict needs 3+ years. Empty until a window
    carries the comparison (the indices are archived from 2026-10-08)."""
    rows = _index_readings(scan, ws)
    if not rows:
        return ""
    code = scan["benchmark"]
    name = BENCHMARK_INDICES.get(code, code)
    cells = []
    for w, edge in rows:
        v = w["vs_index"]
        if edge is None:
            why = "fewer than 30 liquid matches" if v.get("portfolio_pp") is None else f"no {name} reading"
            cells.append(f'<span title="{esc(why)}">–</span>')
            continue
        p = float(v["portfolio_pp"])
        every = " · ".join(f"{BENCHMARK_INDICES.get(k, k)} {p - float(x):+.1f}" for k, x in v["indices"].items())
        tip = (f'{w["start"]} → {w["end"]}: portfolio {p:+.1f}% vs {name} {float(v["indices"][code]):+.1f}%, '
               f'both after costs, {int(v["n"])} stocks. Against every index: {every}')
        col = c["green"] if edge > 0 else c["red"]
        cells.append(f'<span title="{esc(tip)}" style="color:{col};font-weight:700;">{edge:+.1f}</span>')
    measured = [e for _, e in rows if e is not None]
    tail = (f' · ahead in {sum(e > 0 for e in measured)} of {len(measured)}' if measured
            else ' · too few liquid matches to measure')
    v0 = rows[0][0]["vs_index"]
    return (f'<div style="font-size:0.7rem;color:{c["text_secondary"]};margin-top:5px;">Against the '
            f'{esc(name)} index fund — what you could hold instead — after costs: {" / ".join(cells)} pts'
            f'{tail} · a progress reading, the verdict needs 3+ years</div>'
            f'<div style="font-size:0.66rem;color:{c["text_muted"]};margin-top:2px;">Equal-weight portfolio of '
            f'the ₹{float(v0["min_cr"]):g} Cr+/day matches, charged a {float(v0["cost_pp"]):.1f}pp round trip '
            f'each window, against the plain index (no dividends, like the stocks) less a '
            f'{float(v0["fee_pct_yr"]):.1f}%/yr fund fee. Hover a number for every index.</div>')


def _unseen_body(scan: dict, unseen: list, c: dict, esc) -> str:
    """The record, once it has a window: one row, the words on the left and a bar per window on the right, so the card is
    about half the height of a stacked one and the list starts on the first screen (2026-10-07); then each window's other
    readings."""
    since = f"{date.fromisoformat(scan_designed_on(scan)):%d %b %Y}"
    u_beat = _range([int(w["beat_pct"]) for w in unseen], "{}") + "%"
    bars_tip = "Points above the market median in each window, before costs; hover a bar for its dates and size."
    return (f'<div style="display:flex;flex-wrap:wrap;gap:6px 14px;align-items:flex-end;justify-content:space-between;">'
            f'<div style="flex:1 1 200px;min-width:0;">'
            f'<div style="font-size:0.66rem;font-weight:700;color:{c["text_muted"]};letter-spacing:.3px;">'
            f'OUT-OF-SAMPLE · SINCE {esc(since.upper())} — WINDOWS IT NEVER SAW</div>'
            f'<div style="font-size:0.74rem;color:{c["text_secondary"]};margin-top:3px;">'
            f'{u_beat} of matches beat the market · before costs{_overlap_note(unseen, " · ")}</div></div>'
            f'<div style="display:flex;gap:4px;align-items:flex-end;" title="{esc(bars_tip)}">{_bars(unseen, c, esc)}</div></div>'
            f'{_extras(unseen, c, esc)}{_vs_index(scan, unseen, c, esc)}{_hold_line(scan, unseen, c, esc)}'
            f'{_revenue_led_line(scan, unseen, c, esc)}')


def _no_record_body(scan: dict, c: dict, esc) -> str:
    """The record before its first window: that there is none yet and when the first starts; for a scan judged once a year,
    why that may be a long way off."""
    body = f'<div style="font-size:0.74rem;color:{c["text_secondary"]};">{esc(no_record_words(scan))}</div>'
    m = scan_start_month(scan)
    if m:
        body += (f'<div style="font-size:0.66rem;color:{c["text_muted"]};margin-top:2px;">This list is judged on one '
                 f'{scan_horizon(scan) / 365:g}-year window a year, each starting at the end of {date(2000, m, 1):%B}.</div>')
    return body


def _card_head(scan: dict, n_now: int, c: dict, esc) -> str:
    """Name, today's count and the recipe in words — the same on the summary card and the full record."""
    return (f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px;">'
            f'<span style="font-size:1.0rem;font-weight:800;color:{c["text_primary"]};">'
            f'{scan["emoji"]} {esc(scan["name"])}</span>'
            f'<span style="font-size:1.35rem;font-weight:900;color:{c["gold"]};">{n_now:,}'
            f'<span style="font-size:0.7rem;font-weight:600;color:{c["text_muted"]};"> now</span></span></div>'
            f'<div style="font-size:0.76rem;color:{c["text_secondary"]};margin:4px 0 10px 0;line-height:1.35;">'
            f'{esc(scan["recipe"])}</div>')


def card_html(scan: dict, n_now: int, state: str, rec, selected: bool, with_head: bool = True) -> str:
    """One scan card: name, today's count, the recipe in words, and the record: its unseen windows (windows that start after
    DESIGNED_ON, measured automatically from the PRISM Archive), one bar per window, the share of matches that beat the
    market, the net-of-costs reading and what was left out. A record with no window yet says so and when the first starts; a
    stale or missing record shows no numbers at all. It reads only `out_of_sample`: a design-period window or a study left
    in a file is never shown (2026-11-05)."""
    c = COLORS
    esc = html.escape
    border = c["gold"] if selected else c["border"]
    head = _card_head(scan, n_now, c, esc) if with_head else ""
    if state == "measured" and rec:
        unseen = rec.get("out_of_sample") or []
        body = (_unseen_body(scan, unseen, c, esc) if unseen else _no_record_body(scan, c, esc)) + _unseen_notes(rec, unseen, c, esc)
    elif state == "unverified":
        body = (f'<div style="font-size:0.74rem;color:{c["text_muted"]};">The record could not be checked against '
                f'the recipe just now — the app may be updating. Reload the page in a minute.</div>')
    elif state == "stale":
        body = (f'<div style="font-size:0.74rem;color:{c["gold"]};">The recipe changed since its record was '
                f'measured — no record is shown until it is re-measured.</div>')
    else:
        body = f'<div style="font-size:0.74rem;color:{c["text_muted"]};">Not yet measured.</div>'
    return (f'<div style="background:{c["bg_secondary"]};border:1px solid {border};border-radius:10px;'
            f'padding:12px 14px;">{head}{body}</div>')


def _pts(vals) -> str:
    """'+2.9 to +10.3' (or one value): points above the market's median stock."""
    lo, hi = min(vals), max(vals)
    return f"{lo:+.1f}" if lo == hi else f"{lo:+.1f} to {hi:+.1f}"


def _running_tail(ws) -> str:
    """The summary's word on unseen windows still running (2026-10-15): ' · still running, so far' when every
    counted window is, ' · K of them still running' when some are, nothing once all are finished."""
    k = sum(not w.get("final") for w in ws)
    if not k:
        return ""
    return " · still running, so far" if k == len(ws) else f" · {k} of them still running"


# Every summary card is this tall, so the three "Show list" buttons line up in every state. Set 2026-10-09 for the tallest
# summary of that day, at 345px wide. The old record's lines went on 2026-11-05: measured that day in the app at 1536px the cards
# need 155–171 with windows against the index fund, so 224 leaves room for a longer line.
SUMMARY_MIN_HEIGHT = 224


def card_summary_html(scan: dict, n_now: int, state: str, rec, selected: bool) -> str:
    """The card the user CHOOSES from (2026-10-08): name, count, recipe, and one honest headline — how many unseen windows
    it was ahead in, before and after costs, or that it has no record yet and when its first window starts. Every number
    here is computed from the same record card_html shows in full under the cards, so the two cannot disagree. Three full
    cards side by side were 406 / 408 / 596px tall and pushed the list ~240px below a laptop's first screen."""
    c = COLORS
    esc = html.escape
    border = c["gold"] if selected else c["border"]
    main = f'font-size:0.74rem;color:{c["text_secondary"]};margin-top:3px;line-height:1.35;'
    unseen = (rec or {}).get("out_of_sample") or []
    if unseen:
        # Before costs, and after costs whenever the windows carry the net reading (2026-10-15) — the audit's rehearsal
        # read "ahead (+0.5 pts)" while after costs it was -0.5.
        gross = [float(w["median_excess_pp"]) for w in unseen]
        nets = [float(w["net"]["median_excess_pp"]) for w in unseen
                if (w.get("net") or {}).get("median_excess_pp") is not None]
        after = ""
        if nets:
            after = f", {_pts(nets)} after costs" + ("" if len(nets) == len(unseen) else
                                                    f" ({len(nets)} of {len(unseen)} windows)")
        oos = (f'Out-of-sample: ahead in {sum(v > 0 for v in gross)} of {len(unseen)} unseen window'
               f'{"s" if len(unseen) != 1 else ""}: {_pts(gross)} pts before costs{after}'
               f'{_overlap_note(unseen, " · ")}{_running_tail(unseen)}')
    else:
        oos = no_record_words(scan)
    # Once the unseen windows carry the index comparison (December plan rule 23), it IS the out-of-sample
    # line: after costs, against the fund the user could hold instead — the strictest reading, in the
    # same space, so the three cards stay one height (the full record below keeps both readings).
    read = [(w, e) for w, e in _index_readings(scan, unseen) if e is not None]
    edges = [e for _, e in read]
    if edges:
        oos = (f"Out-of-sample, after costs: ahead of the {BENCHMARK_INDICES.get(scan['benchmark'], scan['benchmark'])}"
               f" index fund in {sum(e > 0 for e in edges)} of {len(edges)} window{'s' if len(edges) != 1 else ''}"
               f" ({_pts(edges)} pts){_overlap_note([w for w, _ in read], ' · ')}{_running_tail([w for w, _ in read])}")
    if state == "measured" and rec:
        body = f'<div style="{main}">{esc(oos)}</div>'
    elif state == "unverified":
        body = (f'<div style="font-size:0.74rem;color:{c["text_muted"]};">The record could not be checked against '
                f'the recipe just now — the app may be updating. Reload the page in a minute.</div>')
    elif state == "stale":
        body = (f'<div style="font-size:0.74rem;color:{c["gold"]};">The recipe changed since its record was '
                f'measured — no record is shown until it is re-measured.</div>')
    else:
        body = f'<div style="font-size:0.74rem;color:{c["text_muted"]};">Not yet measured.</div>'
    return (f'<div style="background:{c["bg_secondary"]};border:1px solid {border};border-radius:10px;'
            f'padding:12px 14px;min-height:{SUMMARY_MIN_HEIGHT}px;">{_card_head(scan, n_now, c, esc)}{body}</div>')


def members_label(records: dict) -> str:
    """'30 Sep' — the archived month-end copy the 🆕 marks compare against; '' without one."""
    d = records.get("members_date") if records else None
    try:
        return f"{date.fromisoformat(d):%d %b}"
    except (TypeError, ValueError):
        return ""
