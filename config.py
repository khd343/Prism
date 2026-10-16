"""
Multibagger Discovery System — Configuration
=============================================
All thresholds, weights, gate conditions, and scoring parameters.
Single source of truth — every magic number lives here.
"""

# ═══════════════════════════════════════════════════════════════
# 1. DATA PATHS
# ═══════════════════════════════════════════════════════════════
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import datetime as _dt
import re as _re

_VINTAGE_RE = _re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def vintage_date(path):
    """The DATA date stamped in a drop's filename, or None when the name carries none.

    The ingestion pipeline names every drop after its data session — "PRISM 2026-09-18 Fri.xlsx",
    "PRISM 2026-08-28 Fri - Ratio.csv", and the bare "PRISM 2026-06-29.xlsx" shape — so the
    vintage is a fact of the NAME. A file's modification time is a fact of when it was last
    COPIED, which is not the same thing: on 2026-09-19 an older workbook copied into the folder
    became "newest" by mtime and every local run silently scored June's universe (2,107 stocks)
    in place of September's (2,717). The legacy "Prism - Ratio.csv" has no date and returns None;
    an impossible date such as 2026-13-45 also returns None rather than raising — a resolver must
    never crash on a stranger's filename. Pinned by tests/test_vintage_resolution.py."""
    m = _VINTAGE_RE.search(os.path.basename(str(path)))
    if not m:
        return None
    try:
        return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _vintage_sort_key(path):
    """NEWEST VINTAGE WINS: date in the name first, then mtime, then name — so mtime only decides
    between two copies of the SAME session (a re-fetch) or between undated legacy names."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    return (vintage_date(path) or _dt.date.min, mtime, os.path.basename(str(path)))


def newer_vintage(a, b):
    """True when drop `a` carries newer DATA than drop `b`. When both names are dated the dates
    decide; otherwise the clock is the only evidence and mtime decides (legacy undated CSVs)."""
    da, db = vintage_date(a), vintage_date(b)
    if da is not None and db is not None and da != db:
        return da > db
    return os.path.getmtime(a) > os.path.getmtime(b)


def _get_actual_path(base, folder_name, file_name):
    # Case-insensitive resolution for Linux (Streamlit Cloud).
    # Supports multi-level folder_name (e.g. "Other Resources/CSV Data") — splits on
    # forward slash and resolves each level independently so the same code works on
    # both Windows (backslash native) and Linux (Streamlit Cloud).
    try:
        current = base
        for part in folder_name.replace("\\", "/").split("/"):
            dir_map = {
                item.lower(): item
                for item in os.listdir(current)
                if os.path.isdir(os.path.join(current, item))
            }
            current = os.path.join(current, dir_map.get(part.lower(), part))
        file_map = {item.lower(): item for item in os.listdir(current)}
        actual_file = file_map.get(file_name.lower())
        if actual_file is None:
            # DATED-EXPORT RESOLUTION (2026-08-30): the CSV drops are now named by the
            # ingestion pipeline after their data session — "PRISM 2026-08-28 Fri - Ratio.csv"
            # — where the date/day part changes on every refresh but the "PRISM" prefix and
            # the "- <Tab>.csv" suffix are fixed. Match prefix+suffix case-insensitively and,
            # if several vintages coexist, take the newest DATA — the date in the name — with
            # mtime and then name only as tie-breaks (corrected 2026-09-19: ranking by mtime
            # let an older drop copied in later win; see vintage_date above). The exact legacy
            # name above still wins when present.
            tab_suffix = "- " + file_name.lower().split("- ", 1)[-1]   # "- ratio.csv"
            candidates = sorted(
                (item for low, item in file_map.items()
                 if low.startswith("prism") and low.endswith(tab_suffix)),
                key=lambda it: _vintage_sort_key(os.path.join(current, it)),
                reverse=True,
            )
            actual_file = candidates[0] if candidates else file_name
        return os.path.join(current, actual_file)
    except Exception:
        return os.path.join(base, *folder_name.replace("\\", "/").split("/"), file_name)

DATA_DIR_NAME = "Other Resources/CSV Data"

# ═══════════════════════════════════════════════════════════════
# 2. MACRO ECONOMIC CONSTANTS
# ═══════════════════════════════════════════════════════════════

# India institutional Cost of Equity — Motilal Oswal 23rd WCS baseline.
# Used in Economic Profit: EP = equity base × (RoE − COST_OF_EQUITY). The 28th WCS's Eq 3 says
# NET WORTH; the engine uses RESERVES for both years instead — a data-forced deviation (the CSV
# has no historical P/B, so no prior-year net worth on the same definition exists). Documented
# at the EP block in core/data_engine.py.
# Referenced in data_engine.py; update here to stress-test against 12% or 15% hurdle rates.
COST_OF_EQUITY = 12.0
INDIA_GSEC_YIELD = 7.0   # India 10-year G-Sec proxy (update when RBI policy shifts)

CSV_FILES = {
    "ratio":         _get_actual_path(BASE_DIR, DATA_DIR_NAME, "Prism - Ratio.csv"),
    "income":        _get_actual_path(BASE_DIR, DATA_DIR_NAME, "Prism - Income Statement.csv"),
    "balance":       _get_actual_path(BASE_DIR, DATA_DIR_NAME, "Prism - Balance Sheet.csv"),
    "cashflow":      _get_actual_path(BASE_DIR, DATA_DIR_NAME, "Prism - Cashflow.csv"),
    "shareholding":  _get_actual_path(BASE_DIR, DATA_DIR_NAME, "Prism - Shareholdings.csv"),
    "technical":     _get_actual_path(BASE_DIR, DATA_DIR_NAME, "Prism - Technicals.csv"),
}

# Exact tab names inside the user's Google Spreadsheet.
# INVARIANT: These names must never be changed — the data pipeline loads by tab name,
# not by GID. GIDs are per-spreadsheet and always wrong for a different user's sheet.
SHEET_TAB_NAMES = {
    "ratio":        "Ratio",
    "income":       "Income Statement",
    "balance":      "Balance Sheet",
    "cashflow":     "Cashflow",
    "shareholding": "Shareholdings",
    "technical":    "Technicals",
}

# ═══════════════════════════════════════════════════════════════
# 3. MARKET CAP TIERS  (₹ Crores)
# ═══════════════════════════════════════════════════════════════
MCAP_TIERS = {
    "Mega Cap":   {"label": "Mega Cap",   "min": 200_000, "emoji": "🏛️"},
    "Large Cap":  {"label": "Large Cap",  "min": 20_000,  "emoji": "🏢"},
    "Mid Cap":    {"label": "Mid Cap",    "min": 5_000,   "emoji": "🏗️"},
    "Small Cap":  {"label": "Small Cap",  "min": 500,     "emoji": "🔬"},
    "Micro Cap":  {"label": "Micro Cap",  "min": 100,     "emoji": "⚗️"},
    "Nano Cap":   {"label": "Nano Cap",   "min": 0,       "emoji": "🔭"},
}

# ═══════════════════════════════════════════════════════════════
# 3. HARD GATES — Binary Pass/Fail (Layer 1)
# ═══════════════════════════════════════════════════════════════
# Every stock must pass ALL gates before scoring begins.
# Source frameworks tagged for audit trail.

HARD_GATES = {
    "debt_safety": {
        "column": "debt_to_equity",
        "operator": "<=",
        "threshold": 1.0,
        "penalty": -0.20,
        "source": "Coffee Can / Forensic / Baid",
        "description": "D/E ≤ 1.0 — balance sheet risk gate (Baid prefers ≤0.5)",
    },
    "current_ratio": {
        "column": "current_ratio",
        "operator": ">=",
        "threshold": 1.0,
        "source": "Forensic Shenanigans",
        "description": "CR ≥ 1.0 — liquidity safety gate",
    },
    "pledge_safety": {
        "column": "pledged_percentage",
        "operator": "<=",
        "threshold": 20.0,
        "source": "Fisher Point 15 / SQGLP",
        "description": "Pledged % ≤ 20% — promoter collateral risk",
    },
    "pledge_direction": {
        "column": "pledge_rising",  # derived: pledged > pledged_1qb
        "operator": "==",
        "threshold": 0,
        "source": "Forensic Shenanigans",
        "description": "Pledge not increasing QoQ",
    },
    "promoter_alignment": {
        "column": "promoter_holdings",
        "operator": ">=",
        "threshold": 30.0,
        "source": "Fisher / SQGLP",
        "description": "Promoter holdings ≥ 30% — skin in the game",
    },
    "cash_quality": {
        "column": "cfo_to_pat",
        "operator": ">=",
        "threshold": 70.0,  # cfo_to_pat is PERCENTAGE in CSV (e.g. 73.04 = 73%). Was 0.7 (ratio) — gate always passed.
        "source": "Coffee Can Clean Accounts",
        "description": "CFO/PAT ≥ 70% — earnings are real cash",
    },
    # NOTE: SMA200 was here as a hard gate. REMOVED.
    # Rationale: Binary elimination conflicts with the fundamental philosophy.
    # HDFC Bank, Asian Paints, Page Industries all broke below 200D SMA in March 2020.
    # The GOD SCREEN would have eliminated them at their best buying opportunity.
    # SMA200 is now a CONTINUOUS TREND_SIGNAL (20% of trend score) — it penalizes,
    # not eliminates. A quality stock in a correction naturally scores lower on momentum.
    # The human investor's conviction decides whether to act on the fundamental signal.
    # The signal lives in TREND_SIGNALS["above_sma200"] and `d45_trend_structure`.

    "no_dilution": {
        "column": "dilution_flag",   # 0=Clean, 1=ESOP-level, 2=Meaningful, 3=Large (>10%)
        "operator": "<",
        "threshold": 3,
        "source": "Fisher Point 13 — Materiality-adjusted",
        "description": "Dilution <10% a year tolerated (ESOPs pass). 10–50% fails, whatever the cause. 1.5x+ passes as a bonus or split.",
    },
    "positive_ocf": {
        "column": "operating_cash_flow",
        "operator": ">",
        "threshold": 0,
        "source": "Forensic Shenanigans",
        "description": "Operating cash flow must be positive",
    },
    # HG-07: No loss-making companies (PAT must be positive)
    "positive_pat": {
        "column": "pat",
        "operator": ">",
        "threshold": 0,
        "source": "Handbook HG-07 / Buffett quality",
        "description": "Annual PAT > 0 — no loss-makers pass the screen",
    },
    # HG-08: Revenue floor — no revenue-negative or collapsing businesses
    "revenue_floor": {
        "column": "rev_gr_yoy",
        "operator": ">=",
        "threshold": -20.0,
        "source": "Handbook HG-08",
        "description": "Revenue growth YoY ≥ -20% — excludes businesses in freefall",
    },
}

# ═══════════════════════════════════════════════════════════════

# Financial sector stocks get a separate gate set
# Exact INDUSTRY names for df["industry"].isin() — verified against 354 CSV industry values.
# Previous list ("Banking", "NBFC", etc.) matched 0/10 — all were sector names, not industry names.
FINANCIAL_SECTORS = frozenset([
    "Finance - Capital Markets",
    "Finance - Capital Markets - Brokers",
    "Finance - Capital Markets - RTA",
    "Finance - Capital Markets - Wealth Management",
    "Finance - Holding Company",
    "Finance - Housing",
    "Finance - Investment/Others",
    "Finance - PSU Lending",
    "Finance & Investments - Others",
    "NBFC - Holding Companies",
    "NBFC - Others",
    "Insurance - Proxy",
    "Exchanges",
    "Credit Rating Agencies",
    "Infra/Real Estate Investment Trust",
    "Infrastructure Investment Trusts",
    "Real Estate Investment Trusts",
])

# Exact SECTOR names for df["sector"].isin() — verified against 81 CSV sector values.
# Covers: Financial Services, Finance, brokers, and credit agencies.
FINANCIAL_SECTOR_NAMES = frozenset([
    "Financial Services",
    "Finance",
    "Stock/ Commodity Brokers",
    "Credit Rating Agencies",
])

# Regulated utilities — Greenblatt's Magic Formula explicitly excludes "utilities and
# financial stocks" because their returns are rate-capped, distorting the ROC ranking.
# ONLY regulated power/gas distribution — NOT electrical-equipment makers or upstream E&P.
UTILITY_SECTOR_NAMES = frozenset([
    "Power Generation & Distribution",
    "Gas Distribution",
    "Power Infrastructure",
])

# ═══════════════════════════════════════════════════════════════
# SHARIAH SCREEN (2026-10-16) — a label and a filter, ZERO scoring
# ═══════════════════════════════════════════════════════════════
# The KMI-30 / Meezan method (Pakistan Stock Exchange) with the operator's thresholds: debt 33% and illiquid
# assets 20% of total assets. NOT AAOIFI, which measures debt against 30% of market value. core/shariah.py
# reads every number and list here, and so do the UI and the Reference tab — nothing types a copy. No scorer,
# gate, framework or scan recipe reads shariah_* (tests/test_shariah_screen.py). Kite will later buy only
# "Shariah", sell on "Non-Shariah" and never sell on "Review" or "Doubtful": an open question must never sell a
# holding.
#   Review    nothing visible fails, but a rule the sheet cannot prove (investments, other income) or a business
#             awaiting a scholar's ruling is open — the operator's check settles it (added 2026-10-16, the operator's
#             call: 236 of 1,484 in the ₹500–20,000 Cr window on the 15 Oct sheet, Infosys / HCL / Wipro among them)
#   Doubtful  a figure is MISSING, so PRISM cannot screen the stock fully. A missing figure beats a review item.
# Each rule reads the same way: pass / fail / review / unknown (missing).
SHARIAH_STATUSES = ("Shariah", "Non-Shariah", "Review", "Doubtful")
SHARIAH_RULE_STATES = ("pass", "fail", "review", "unknown")
# The six rules in the order the reasons are read. `limit` is a share of the rule's base (0.33 = 33%).
SHARIAH_RULES = {
    "business":    {"n": 1, "name": "Business",                  "limit": None},
    "income":      {"n": 2, "name": "Impermissible income",      "limit": 0.05},   # of revenue, at most
    "debt":        {"n": 3, "name": "Interest-bearing debt",     "limit": 0.33},   # of total assets, at most
    "illiquid":    {"n": 4, "name": "Illiquid assets",           "limit": 0.20},   # of total assets, at least
    "nla":         {"n": 5, "name": "Net liquid assets",         "limit": None},   # at most the market cap
    "investments": {"n": 6, "name": "Non-compliant investments", "limit": 0.33},   # of total assets, below
}
# Industry names exactly as core/cyclicality_map.INDUSTRY_TIER spells them (after core.taxonomy translates the
# vendor's names). Financials need no list: Rule 1 reads the map's tier E (banks, NBFCs, insurers, AMCs,
# brokers), never FINANCIAL_SECTORS, which lacks banks and insurers and holds the REITs below.
SHARIAH_PROHIBITED_INDUSTRIES = frozenset([
    "Alcoholic Beverages",
    "Cigarettes & Tobacco Products",
    "Hotels",
    "Hotels - Resorts",
    "Resorts",
    "Entertainment & Media",
    "Music Licensing",
    "Aerospace & Defence - Equipments",
])
# Review, "awaiting a scholar's ruling" — never a pass, never a fail.
SHARIAH_REVIEW_INDUSTRIES = frozenset([
    "Restaurants",
    "Quick Service Restaurant - QSR",
    "Airport Restaurants and Lounges",
    "Amusement Parks",
    "Chemicals - Organic - Alcohol Based",
    "Real Estate Investment Trusts",
    "Infrastructure Investment Trusts",
    "Infra/Real Estate Investment Trust",
])
# The operator's call, 2026-10-16: 20 stocks carry this sector with another industry, and 18 of them describe
# their own business as defence (warships, missile propellants, "defence-focused"); Cochin Shipyard and JTL
# Defence (copper and brass products) are for the scholar to clear.
SHARIAH_REVIEW_SECTORS = frozenset([
    "Aerospace & Defence",
])
# A prohibited business the vendor files under a neutral industry. Each entry carries its reason, a fact about
# the business from its own filings — never a price or a return.
SHARIAH_PROHIBITED_COMPANIES = {
    "NSE:DELTACORP": "casino gaming in Goa and Sikkim",
}
# YOUR LISTS (the operator, 2026-10-16: "whatever I said in config, that's it — and if after a new result all six
# rules pass, give it Shariah"). Only the ticker, no dates: "NSE:SYMBOL", or "BSE:code" for a BSE-only stock. A
# company sits in at most one list. One rule, read after the six:
#   the latest data passes all six rules  ->  Shariah, whatever a list says
#   otherwise your list decides:
#     SHARIAH_APPROVED          Shariah — even when the data shows a fail (its reason then says "though the data
#                               shows: ...", so it is never silent)
#     SHARIAH_REJECTED          Non-Shariah
#     SHARIAH_REVIEW_COMPANIES  Review
#   otherwise PRISM's own label.
# A haram BUSINESS the data cannot see belongs in SHARIAH_PROHIBITED_COMPANIES (above) instead: that is Rule 1
# itself, so no new result can clear it.
SHARIAH_APPROVED = frozenset([
    # "NSE:EXAMPLE",
])
SHARIAH_REJECTED = frozenset([
])
SHARIAH_REVIEW_COMPANIES = frozenset([
])
# Industries whose NAME holds a word the classification guard looks for, decided allowed: "film" here is
# plastic film, and "media" sits inside "intermediate".
SHARIAH_ALLOWED_NAME_MATCHES = frozenset([
    "Packaging - Films",
    "Plastics - Sheets/Films",
    "Detergents/Intermediates",
    "Dyes - Intermediate",
])

# ═══════════════════════════════════════════════════════════════
# 4. QUALITY SCORE WEIGHTS (Layer 2) — 0 to 100
# ═══════════════════════════════════════════════════════════════
# IMPROVEMENT joined 2026-10-03 at 0.20; the six level facets were each scaled by 0.8 (moat 0.22 ->
# 0.176 ...), so levels stay in. Evidence and the deliberate exclusions (ROCE change: the vendor's
# balance sheet lags its P&L by a fiscal year) are in tests/test_improvement_facet.py. 0.20 is a
# prior, not a fit.
QUALITY_WEIGHTS = {
    "moat":          0.176,  # ROCE trajectory, ROE — SQGLP Quality
    "growth":        0.176,  # Revenue/PAT/EPS CAGR — SQGLP Growth
    "improvement":   0.20,   # last-year profit growth + revenue growth + margin change — getting BETTER?
    "cash":          0.16,   # CFO/PAT, FCF yield, self-funding — Coffee Can
    "margin":        0.104,  # NPM, OPM, GPM medians + acceleration — Fisher
    "balance_sheet": 0.104,  # Net debt, reserves growth, CWIP — Baid/Marks
    "valuation":     0.08,   # PE discount, PEG, FCF yield — Marks/Baid Entry Price
}

# NOTE (2026-06-19): the moat / growth / cash / margin / balance_sheet COMPONENT weights are NOT
# here — they live inline in core/scoring_engine._compute_*_score. Each weight is inseparable from
# its per-signal logic (sector-vs-universe rank, ascending direction, binary ×100, FCF/CFO's
# negative-OCF special case), so a flat weights-dict can never be their full source; the engine is.
# The dead duplicate dicts (MOAT/GROWTH/CASH/MARGIN/BALANCE_SHEET_SIGNALS) that used to sit here were
# import-only — editing them did nothing — so they were removed to end the single-source-of-truth
# breach. Only the VALUATION/RS/TREND/BREAKOUT/SECTOR signal-weight dicts (below) are config-driven.

# ═══════════════════════════════════════════════════════════════
# 4b. VALUATION SCORE SIGNALS (Marks + Baid Entry Price Discipline)
# ═══════════════════════════════════════════════════════════════
VALUATION_SIGNALS = {
    "pe_discount":     0.20,   # PE vs 10Y median — higher discount = better
    "peg_ratio":       0.25,   # PEG < 1.0 = cheap (confirmed in all 30 MOSL studies)
    "payback_ratio":   0.15,   # Payback < 1x = most reliable MOSL supernormal-return signal
    "ev_compression":  0.15,   # EV/EBITDA falling = value creation
    "fcf_yield_val":   0.15,   # FCF Yield > 3% = attractive
    "de_fortress":     0.10,   # D/E < 0.5 (Baid's fortress gate) = bonus
}
# Weights sum: 0.20+0.25+0.15+0.15+0.15+0.10 = 1.00 ✓

# Payback Ratio zones (market_cap / 5Y cumulative estimated PAT)
# < 1.0: market cap recovered within 5Y of profits = supernormal return territory (MOSL)
# < 2.0: attractive; < 3.0: fair; > 3.0: expensive; > 5.0: very expensive
PAYBACK_ZONES = {
    "supernormal":  {"min": 0,   "max": 1.0, "score": 100},
    "attractive":   {"min": 1.0, "max": 2.0, "score": 80},
    "fair":         {"min": 2.0, "max": 3.0, "score": 60},
    "expensive":    {"min": 3.0, "max": 5.0, "score": 35},
    "very_exp":     {"min": 5.0, "max": 999, "score": 10},
}

# PEG zone scoring (Baid + Marks). MOSL lineage note (2026-08-29, 4th WCS full read): within the
# WCS series the PEG yardstick ORIGINATES in the 4th study (1994-99, pub Feb 2000) — its low-PEG
# quadrant (below the 0.83 median) returned 45-46%/yr with a 68% hit rate vs 12-13%/yr for high-PEG,
# its Infosys-vs-HLL contrast (PEG 0.52 -> 105%/yr vs 1.46 -> 24%/yr), and its explicit screen
# ("PEG well below 1.0") are the earliest source of the PEG<=1 anchor the 23rd/QGLP later carry.
PEG_ZONES = {
    "deep_value":  {"min": 0,   "max": 0.8,  "score": 100},
    "undervalued": {"min": 0.8, "max": 1.2,  "score": 85},
    "fair":        {"min": 1.2, "max": 1.5,  "score": 70},
    "full":        {"min": 1.5, "max": 2.0,  "score": 45},
    "expensive":   {"min": 2.0, "max": 2.5,  "score": 20},
    "extreme":     {"min": 2.5, "max": 999,  "score": 5},
}

# ═══════════════════════════════════════════════════════════════
# 4c. MEAN REVERSION RISK (Marks: "Extremes revert")
# ═══════════════════════════════════════════════════════════════
# Flag stocks where current margins >> 5Y median as cyclical peak risk
MEAN_REVERSION = {
    "opm_spike_threshold": 1.3,    # if OPM_LQ / OPM_Med_5Y > 1.3 = cyclical peak risk
    "npm_spike_threshold": 1.3,    # if NPM_LQ / NPM_Med_5Y > 1.3 = cyclical peak risk
    "penalty_factor":      0.85,   # multiply quality score by this if cyclical peak
}

# ═══════════════════════════════════════════════════════════════
# 5. MOMENTUM SCORE WEIGHTS (Layer 3) — 0 to 100
# ═══════════════════════════════════════════════════════════════
MOMENTUM_WEIGHTS = {
    "relative_strength": 0.30,
    "trend_quality":     0.25,
    "breakout_proximity":0.20,
    "volume_confirm":    0.10,
    "sector_leadership": 0.15,
}

RS_SIGNALS = {
    "crs_50d":          0.40,
    "crs_52w":          0.30,
    "crs_26w":          0.30,
}

# "Last VSTOP Change 14W 2.5" (last_vstop_change) counts WEEKLY BARS since the V-Stop last flipped, not days:
# measured 2026-10-15, from 5 Oct to 12 Oct it rose by exactly 1 for 2,445 of 2,710 stocks, and a new flip
# reads 1. "Fresh" = flipped within this many bars, the largest whole-week window inside the documented 30 days
# (4 x 7 = 28; 5 x 7 = 35). Read as days, the old `<= 30` meant 30 WEEKS and fired for 75% of the market.
VSTOP_FRESH_MAX_WEEKS = 4

TREND_SIGNALS = {
    # above_sma200 moved FROM hard gate TO scoring signal.
    # Penalises stocks below 200D SMA continuously (−20 pts on trend score)
    # rather than eliminating them. Fundamental quality carries through corrections.
    "above_sma200":     0.20,   # Price > SMA 200D — trend direction (was hard gate)
    "vstop_green":      0.20,   # VSTOP 14W 2.5 = green (reduced: correlated with sma200)
    "vstop_fresh":      0.15,   # flipped UP within VSTOP_FRESH_MAX_WEEKS weekly bars (price above the stop), ≤ 30 days (reduced)
    "adx_strong":       0.20,   # ADX 14W > 25 — trend strength (independent signal)
    "rsi_zone":         0.15,   # RSI 55-70 sweet spot
    "golden_cross":     0.10,   # recent golden cross — trend recovery signal
}
# Weights sum: 0.20+0.20+0.15+0.20+0.15+0.10 = 1.00 ✅

BREAKOUT_SIGNALS = {
    "52wh_distance":    0.30,   # % below 52WH — #1 backtested alpha signal (+0.55%/wk)
    "52wh_days":        0.20,   # days since 52WH was set — recency of momentum peak (new)
    "13wh_distance":    0.20,   # quarterly high proximity — near-term breakout
    "breakout_window":  0.20,   # a 40+ day base broken near the 52-week high (breakout_flag)
    "ath_distance":     0.10,   # all-time high proximity (reduced: ATH can be years old)
}
# Weights sum: 0.30+0.20+0.20+0.20+0.10 = 1.00 ✅
# The breakout flag (2026-10-15). StockScans' "Breakout Window" = "Number of days since the last close was greater than
# or equal to today's close" (its own column dictionary): never below 15 (the vendor's floor) and blank when no
# earlier close was that high (every blank stock's 52-week high is its all-time high: 18 Sep and 9 Oct). The flag
# read it as `> 0` — true for 92-96% of stocks and false for those blanks. A breakout is now StockScans' own two scans
# joined (Long Term: window >= 200 and within 10% of the 52-week high; Short Term: window 40-200 and within 10% of the
# 26-week high), with three deliberate differences: no Market Cap >= 1,000 Cr filter (size is Kite's cap rule; it
# would remove 40 of the 121 buyable breakouts on 9 Oct), a blank counts as the longest window, and the 52-week high
# stands in for the 26-week one, which the saved sheets before 15 Oct lack — within 10% of the 52-week high is the
# only case that PROVES within 10% of the 26-week high (§5). Fires on 181 of 2,712 stocks (6.7%) on 9 Oct.
BREAKOUT_MIN_WINDOW_DAYS = 40
BREAKOUT_MAX_HIGH_GAP_PCT = 10.0

SECTOR_SIGNALS = {
    "ret_vs_industry_1y":  0.55,
    "ret_vs_industry_3m":  0.45,
}

# ═══════════════════════════════════════════════════════════════
# 6. COMPOSITE SCORE BLEND (Layer 4)
# ═══════════════════════════════════════════════════════════════
# Only governance weight lives here — quality/momentum weights are per-ANALYSIS_MODES,
# not global constants. (Previously had "quality": 0.55 / "momentum": 0.30 here,
# but they were never read by the scoring engine and created false documentation.)
COMPOSITE_WEIGHTS = {
    "governance": 0.15,
}

# Governance bonus components — positive values sum to 133 (raw max), engine clamps to 100 via _safe_clip.
# Dilution penalties are DEDUCTIONS applied inside compute_governance_bonus.
# Tier 3 (>10%) never reaches scoring — hard gate eliminates it first.
# Tier 2 (3-10%): passes gate, but loses 25 governance pts — visible, proportional.
# Tier 1 (<3% ESOP): passes gate, -5 pts — distinguishes from zero-dilution companies.
# Tier 0 (clean): no penalty.
GOVERNANCE_BONUS = {
    "promoter_buying":         20,   # promoter increased holding this Q
    "fii_accumulating":        15,   # FII buying this Q
    "dii_accumulating":        10,   # DII buying this Q
    "inst_convergence":        15,   # FII + DII both buying same Q — intentional triple-stack (FII+DII+convergence=40pts): convergence of smart money is qualitatively stronger than either alone
    "insider_trading_present": 15,   # insiders net-buying (data_engine.insider_net_buyer)
    "pledge_falling_1y":       10,   # pledge reduced over 1 year
    "undiscovered_alpha":      15,   # low FII + Tier C mcap
    # Promoter holding alignment — Mayer 100-Bagger: present in 10/10 Indian 100-baggers.
    # Rewards the BASELINE alignment level, not just quarterly buying activity.
    # Dynasty mode (≥60%): founder's wealth IS the stock — decades-horizon thinking.
    # Well-aligned (50-60%): meaningful skin in game without full dynasty mode.
    # Selling from low base (<40% + declining): promoter telling you something the price hasn't yet reflected.
    "promoter_high_alignment":  15,  # holdings ≥ 60%: dynasty mode
    "promoter_good_alignment":   8,  # holdings 50-60%: well-aligned owner-operator
    # 3-year promoter trend — most powerful ownership signal; single quarter buys/sells are noise
    "promoter_3y_accumulation": 10,  # 3Y net buying > 3%: sustained conviction = dynasty building
    # Dilution: tier-1 minor ESOP stays a small additive deduction; tier-2 (3-10%) is a
    # HARD RISK SIGNAL handled by GOVERNANCE_RISK_MULTIPLIERS below, not additive points.
    "dilution_tier1_minor":    -5,   # <3% ESOP dilution: minor deduction vs zero-dilution
}

# ── Forensic Red-Flag Cascade ──
# The HARSHER of the two asymmetric penalty multipliers (the governance shield below is milder).
# red_flag_count → multiplier on composite_score. Range-based UPPER bounds (max_flags); the tier
# with max_flags=None is the open-ended top bucket (5+). Tiers MUST be ascending by max_flags —
# the engine applies them as a first-match np.select cascade. This is the SINGLE source for the
# schedule; forensic_engine.compute_cascading_forensic_filter reads it (no inline magic numbers).
#   0 flags → 1.00 (clean) · 1–2 → 0.90 (watch) · 3–4 → 0.75 (caution) · 5+ → 0.50 (high risk)
FORENSIC_PENALTY_TIERS = [
    {"max_flags": 0,    "multiplier": 1.00, "label": "Clean"},
    {"max_flags": 2,    "multiplier": 0.90, "label": "Watch"},
    {"max_flags": 4,    "multiplier": 0.75, "label": "Caution"},
    {"max_flags": None, "multiplier": 0.50, "label": "High Risk"},
]

# Below this listing age the YoY share-count delta compares today's public float against a
# PRE-LISTING share count, so it measures nothing — see the block comment above
# `_prelisting_baseline` in core/data_engine.py for the measured evidence. NOT a tuned number:
# `equity_shares_1yb` is a prior FISCAL-YEAR figure whose age runs ~365-730 days depending on
# where in the year the vintage is taken, and 730 is the maximum, so it is the only bound that
# holds on every vintage. The 2026-09-18 vintage puts the observed boundary at ~537 days
# (FY2025 closed 31 Mar 2025); tests/test_prelisting_dilution.py re-measures that each run and
# fails if a future vintage pushes the distortion past this constant.
PRELISTING_BASELINE_DAYS = 730

# A company's results are STALE once it has filed none for this many days (2026-10-07). Set by the
# listing rules, not by taste. SEBI LODR Reg 33: a quarter's results within 45 days of its end, the
# March quarter with the annual within 60 days; companies on the SME platforms file HALF-YEARLY
# (Apr-Sep within 45 days, the annual within 60 days of March). The longest gap an ON-TIME filer can
# show: quarterly ~140 days (Oct-Dec results filed around 10 January -> the annual on 30 May);
# half-yearly ~232 days (Apr-Sep results filed around 10 October -> the annual on 30 May; only a filing
# in the first two days of October could pass 240, and a half-year audited that fast is not seen). 240
# is past both, so it marks a company that missed a deadline under ANY calendar. The previous 120 flagged 282
# stocks on the 2026-10-06 vintage — about 270 of them SME-platform companies on their normal
# half-yearly calendar, including 89 of 🌱's 255 — and quarterly filers too every May; 240 flags 10,
# each a real story: Gensol (602 days), Jet Airways (648), absorbed or merged companies (Cigniti,
# Sanghi, GSPL), Kody Technolab (no FY26 annual by 30 May). The price, accepted: a quarterly filer that
# misses ONE deadline is flagged at ~8 months, not ~4; its Result Age still shows the days.
STALE_RESULT_DAYS = 240

# ── Benchmark indices (2026-10-08) ──
# The NSE indices the scans are judged against ("could I buy the index fund instead?"). They
# arrive in the PRISM watchlists, so the sheet carries them as rows like a stock: price and
# technicals filled, every fundamental, the sector, the industry and the market cap 'null'. They are
# set aside at the merge (core/data_engine._set_aside_benchmarks) and never enter the stock universe.
# Keyed by the vendor's id, which is steadier than a name (the vendor re-spelled every sector name in
# early October 2026). Read from the sync's own fetch of 2026-10-08 (seven), and of 2026-10-13 for the
# two MidSmallcap 400 indices: the Kite Playbook trades mid and small caps only, and its frozen test is
# judged against Nifty MidSmallcap 400 with its Momentum Quality 100 as the reference — so PRISM shows the
# same yardstick. No scan card's declared benchmark changed (ui_scans.SCANS, fixed 2026-10-08).
BENCHMARK_INDICES = {
    "NSE:CNX500": "Nifty 500",
    "NSE:SMALLCA250": "Nifty Smallcap 250",
    "NSE:NFMICRO250": "Nifty Microcap 250",
    "NSE:MIDSMAL400": "Nifty MidSmallcap 400",
    "NSE:NIF500MO50": "Nifty500 Momentum 50",
    "NSE:NI200MOM30": "Nifty200 Momentum 30",
    "NSE:NQUALITY30": "Nifty200 Quality 30",
    "NSE:NS250MQ100": "Nifty Smallcap250 Momentum Quality 100",
    "NSE:NMS400MQ10": "Nifty MidSmallcap400 Momentum Quality 100",
}

# ── Asymmetric Governance Risk Shield ──
# Negative ownership signals predict DISASTERS far better than positive signals predict
# winners (Yes Bank, DHFL, Zee, Manpasand all showed promoter exit / pledge / dilution
# patterns before collapse, while fundamentals still screened fine). Therefore:
#   Positive signals → additive governance_bonus (engine, GOVERNANCE_BONUS above)
#   Negative signals → composite MULTIPLIER (shield, this table)
# A multiplier scales with conviction: a 90-composite stock loses more absolute points
# than a 20-composite stock — exactly right, because the risk threatens a larger position.
# The four hard risk signals counted in gov_risk_count (scoring_engine.compute_governance_bonus):
#   1. Tier-2 dilution (dilution_flag == 2: 3-10% share dilution)
#   2. Promoter 3Y systematic exit  (change_promoter_3y < -5)
#   3. Promoter 2Y recent exit      (change_promoter_2y < -3 AND 3Y >= -5) — mutually
#      exclusive with #2 by construction; the same exit is never double-counted
#   4. Low + declining promoter     (promoter_holdings < 40 AND change_promoter_1y < 0)
# Deliberately milder than the forensic cascade (x0.50 floor): ownership signals are
# warnings; forensic red flags are evidence.
GOVERNANCE_RISK_MULTIPLIERS = {
    0: 1.00,   # no ownership risk signals
    1: 0.92,   # one signal — caution
    2: 0.82,   # two signals — structural concern
    3: 0.70,   # three or more — the promoter is telling you something
}

# ═══════════════════════════════════════════════════════════════
# 7. CONVICTION TIERS
# ═══════════════════════════════════════════════════════════════
CONVICTION_TIERS = [
    {"min": 85, "tier": 1, "label": "Crown Jewels",       "emoji": "🏆", "color": "#FFD700",
     "description": "Highest conviction compounders — deep-dive and build position"},
    {"min": 70, "tier": 2, "label": "Strong Compounders",  "emoji": "🥇", "color": "#3fb950",
     "description": "Quality with momentum confirmation — watchlist priority"},
    {"min": 55, "tier": 3, "label": "Emerging Quality",    "emoji": "🥈", "color": "#58a6ff",
     "description": "Quality building, momentum developing — monitor for upgrade"},
    {"min": 40, "tier": 4, "label": "On Radar",            "emoji": "🥉", "color": "#d29922",
     "description": "Some quality signals, needs time — early watchlist"},
    {"min": 0,  "tier": 5, "label": "Not Ready",           "emoji": "❌", "color": "#f85149",
     "description": "Insufficient quality or momentum — ignore"},
]


# Baid's 3 Sell Triggers (alert system)
BAID_SELL_TRIGGERS = {
    "thesis_broken": {
        "description": "ROCE declining structurally (3Y trajectory negative)",
        "check": "roce_trajectory < -3",
    },
    "management_deteriorated": {
        "description": "Pledge rising + promoter selling + D/E rising",
        "check": "pledge_rising AND change_promoter_lq < 0 AND de_slope_3y > 0",
    },
    "cash_quality_collapse": {
        "description": "CFO/PAT dropped below 50% (was above 70%) — cfo_to_pat is stored as PERCENTAGE",
        "check": "cfo_to_pat < 50",
    },
}

# ═══════════════════════════════════════════════════════════════
# 7c. ANALYSIS MODES — Controls Fundamental vs Technical balance
#     Blend only — the QGLP screen is fixed and no longer varies by mode (2026-09-20).
# ═══════════════════════════════════════════════════════════════
# THE DEFAULT ANALYSIS MODE — read by app.py, core.run_scoring_pipeline, run_full_scoring,
# tools/snapshot.py, /census and /verify. ONE constant on purpose: before this existed the app
# and the engine carried the literal "Hybrid" independently, so changing the app alone would have
# left every snapshot, census and verify run scored under a mode the user no longer sees — the
# silent-divergence class. Changed 2026-09-19 Hybrid -> Breakout on three-window forward evidence
# (see the Breakout entry below), and 2026-10-15 Breakout -> MomentumBreakout, the user's decision
# ahead of December (see the MomentumBreakout entry); revert by changing this line alone.
DEFAULT_ANALYSIS_MODE = "MomentumBreakout"

ANALYSIS_MODES = {
    "Hybrid": {
        "label": "🔀 Hybrid (Quantamental)",
        "fundamental_w": 0.70,
        "momentum_w": 0.30,
        "description": "Best of both — great business + institutions are buying it now",
    },
    "Fundamental": {
        "label": "📚 Fundamental Only",
        "fundamental_w": 1.00,
        "momentum_w": 0.00,
        "description": "Pure business quality — for long-term buy-and-hold Coffee Can investors",
    },
    "Technical": {
        "label": "📈 Technical Only",
        "fundamental_w": 0.10,
        "momentum_w": 0.90,
        "description": "Pure price action — follow institutional money flow with O'Neil rules",
    },
    # ── THE DEFAULT 2026-09-19 → 2026-10-15 (a candidate that morning, promoted hours later) ──
    # Equal thirds of quality, momentum and breakout (the composite's third leg, `breakout_w`;
    # absent from every other mode, so those are byte-identical to before it existed).
    #
    # THE EVIDENCE, measured in EXACTLY the form the engine computes (governance 15%, framework
    # boosts and the forensic multiplier all kept — the bare rank blend flatters every candidate).
    # A 66-point sweep of the whole Q/M/B simplex on all THREE forward windows now available:
    #                           66d(06-17→08-22)   28d(08-22→09-19)   94d(06-17→09-19)
    #   Fundamental 100/0/0       +0.118 #66/66      -0.008 #66/66      +0.068 #66/66
    #   Hybrid      70/30/0       +0.134 #56         +0.034 #60         +0.096 #60
    #   THIS MODE   33/33/33      +0.150 #17         +0.099 #32         +0.138 #33
    # Hybrid is not merely beaten — it sits near the BOTTOM of the grid on every window, and 14
    # of the 21 quality-led (>=50% Q) weightings beat it in all three. "Hybrid is over-weighted
    # on quality for any horizon we can measure" is therefore a broad, robust finding, not one
    # lucky grid point.
    #
    # WHY EQUAL THIRDS AND NOT THE OPTIMUM — THE WEIGHTS ARE A PRIOR, NOT A FIT. The grid optimum
    # is unstable AND absurd: 66d wants (0.1, 0.0, 0.9), 28d wants (0.0, 0.4, 0.6), 94d wants
    # (0.0, 0.0, 1.0) — the two short optima sit 40% of the weight budget apart and their IC
    # surfaces correlate only +0.417, yet all three agree on "delete the fundamentals". Following
    # that would not be PRISM. Thirds ranks #17/#32/#33 — never optimal, never bad — which is what
    # a neutral prior should look like when the instrument cannot answer the question. DO NOT TUNE
    # THESE WEIGHTS on sub-quarterly windows; the sweep is in docs/known-issues.md.
    #
    # WHAT IS KNOWINGLY GIVEN UP: breakout_score is ALREADY 40% of momentum_score
    # (breakout_proximity + breakout_window, rho 0.875), so this mode puts ~2/3 of the composite
    # on price strength and ~1/3 on the fundamental work. That is a deliberate hedge, not a
    # measurement: every horizon we can see says price strength, and the horizon the engine was
    # built for (multi-year compounding) cannot be seen yet. The 1/3 keeps it in the ranking.
    #
    # DECEMBER IS A REVIEW, NOT A CORONATION — the rule, pre-declared so it is not re-argued by
    # taste: on the first 6-month window, if a quality-led weighting beats 33/33/33, move back
    # toward it; if thirds still wins, it stays. tools/validate.py reconstructs EVERY mode on
    # EVERY snapshot (composite_mode_*) so the comparison is apples-to-apples whatever mode a
    # snapshot was taken under. Reverting is one constant. Contract: tests/test_breakout_mode.py.
    "Breakout": {
        "label": "⚡ Quantamental + Breakout",
        "fundamental_w": 1.0 / 3.0,
        "momentum_w":    1.0 / 3.0,
        "breakout_w":    1.0 / 3.0,
        "description": "Quality, momentum and breakout in equal thirds — price strength at two-thirds "
                       "of the score. It beat Hybrid on all three forward windows, which rank Hybrid "
                       "#56/#60/#60 of 66 weightings. December's 6-month window compares it with "
                       "Momentum + Breakout",
    },
    # ── THE DEFAULT SINCE 2026-10-15 (selectable since 2026-10-10) ──────────────────────────
    # Momentum and breakout in halves, quality OUT of the blend. Measured in exactly this form
    # (governance 15%, framework boosts, governance shield and forensic multiplier all kept) on
    # the four forward windows PRISM holds — rank-IC, all stocks / stocks trading >= Rs 1 Cr a day:
    #                               typical stock   Breakout 33/33/33   THIS MODE
    #   06-17→08-22 (66 days)            +0.2%       +0.147 / +0.132     +0.143 / +0.137
    #   08-22→09-19 (28 days)            -2.0%       +0.091 / +0.072     +0.135 / +0.134
    #   06-29→10-09 (102 days, archive)  -6.2%       +0.110 / +0.073     +0.133 / +0.111
    #   06-17→10-03 (108 days)           -3.6%       +0.128 / +0.126     +0.157 / +0.172
    # Ties thirds once, beats it three times, wins all four on the buyable stocks.
    # Held back from the default on 2026-10-10 because the two long windows overlap almost
    # entirely, three of the four were falling markets (where price strength protects best) and
    # the longest is ~3.5 months against a multi-year aim; December's 6-month window was to decide
    # (docs/december-2026-plan.md rule 26). 2026-10-15: THE USER MADE IT THE DEFAULT, ahead of
    # December. The caveats stand, so December now REVIEWS it: if thirds beats it on that window,
    # quality earns its third back — one constant above, and tools/validate.py rebuilds every
    # mode on every snapshot, so the comparison does not depend on which mode is the default.
    # Quality + Breakout (no momentum), measured the same day, trailed thirds in three of four and
    # is not a mode. Contract: tests/test_momentum_breakout_mode.py.
    "MomentumBreakout": {
        "label": "🚀 Momentum + Breakout",
        "fundamental_w": 0.0,
        "momentum_w":    0.5,
        "breakout_w":    0.5,
        "description": "Momentum and breakout in halves, quality left out of the blend — the forensic "
                       "penalty, hard gates and framework boosts still apply. The default since "
                       "2026-10-15: measured ahead of 33/33/33 on 3 of 4 forward windows (all falling "
                       "or flat markets, the longest ~3.5 months); December's 6-month window reviews it",
    },
}

# The Analysis Mode selectbox's label, and the way back from it (2026-10-16). Streamlit 1.54 syncs a selectbox by its
# DISPLAY label and hands back a label it cannot match as it is; a tab open across the 15 Oct deploy that moved the
# default re-sent "⚡ Quantamental + Breakout · default", which no option carried any more, and app.py, reading
# cfg_mode before the selectbox could check it, crashed on ANALYSIS_MODES[...] (KeyError, live, 16 Oct). The label
# round-trip rule the sidebar already follows (ui_discovery.recover_labels, ui_liquidity.step_value). Pinned by
# tests/test_analysis_mode_roundtrip.py.
_DEFAULT_MARK = " · default"


def mode_label(key) -> str:
    """The selectbox's label for a mode key: its label, the default marked. The ONE label — mode_value reverses it."""
    return ANALYSIS_MODES[key]["label"] + (_DEFAULT_MARK if key == DEFAULT_ANALYSIS_MODE else "")


def mode_value(stored) -> str:
    """A real mode key from whatever session state holds: a key passes through; the label of any mode, with or
    without the default mark, recovers its key (recover, do not drop: the tab keeps the mode it showed); anything
    else reads as the default. Never raises."""
    if not isinstance(stored, str):
        return DEFAULT_ANALYSIS_MODE
    if stored in ANALYSIS_MODES:
        return stored
    stem = stored.removesuffix(_DEFAULT_MARK)
    return next((k for k, m in ANALYSIS_MODES.items() if m["label"] == stem), DEFAULT_ANALYSIS_MODE)

# ═══════════════════════════════════════════════════════════════
# 7d. THE QGLP PROFILE — Agrawal's own weights and gates, FIXED
# ═══════════════════════════════════════════════════════════════
# SEVEN PROFILES AND THEIR SELECTOR REMOVED 2026-09-20. The control re-ranked NOTHING:
# composite_score, rank, conviction_tier and the top-50 were byte-identical across all eight
# on the live 2,717-stock universe. What it DID move was qglp_pass (307 under Quality ->
# 1,274 under Turnaround, 4.1x) and frameworks_passed (up to 861 rows) — which made QGLP the
# only one of 38 frameworks whose verdict was a user preference rather than a fact about the
# company, and quietly moved the MOSL tab that counts QGLP as one of its ten lenses.
#
# THE DEFECT THAT DECIDED IT: the profile changed WITHOUT BEING TOUCHED. Each ANALYSIS_MODES
# entry carried a list of permitted profiles and app.py snapped the profile into it on every
# mode change, one-way. "Technical Only" permitted only [Momentum, Turnaround], so the round
# trip Breakout -> Technical Only -> Breakout left the QGLP screen on Momentum and it STUCK
# (passers 413 -> 631, MOSL convergence 709 -> 789, 218 stocks moved) while composite, rank
# and every header tile stayed identical — nothing moved on screen, and nothing warned.
#
# KEPT AS A DICT rather than inlined, for two live readers: get_adaptive_weights layers the
# regime deltas on top of these numbers, and ui_tearsheet's QGLP radar resolves its
# thresholds from here. Both reach it via `.get(name, MASTER_PROFILES["Balanced"])`, so a
# stale caller degrades to the book instead of raising. Pins: tests/test_qglp_profile_fixed.py.
#
# `forensic_boost` and `priority_cols` were DROPPED 2026-09-20. Nothing outside this file ever
# read them: get_adaptive_weights copied them into its return and no consumer touched the
# result, and the only other reference in the repo was a test asserting they must exist — a
# closed loop keeping a fossil alive. forensic_boost was designed to scale forensic
# sensitivity per profile (0.7-1.8 across the eight); with one profile left its only value was
# 1.0, the identity, so it could not have done anything even once wired.
MASTER_PROFILES = {
    "Balanced": {
        "label": "Balanced (QGLP)",    "icon": "⚖️",
        "description": "Raamdeo Agrawal's QGLP — balanced Quality, Growth, Longevity, Price",
        "quality_w": 0.35, "growth_w": 0.35, "longevity_w": 0.15, "price_w": 0.15,
        "roce_gate": 15.0, "growth_gate": 15.0, "peg_gate": 1.5,
    },
}

# ═══════════════════════════════════════════════════════════════
# 7e. REGIME-ADAPTIVE WEIGHT ADJUSTMENTS
# When the market regime is auto-detected, these adjustments are
# applied ON TOP of the fixed QGLP profile's base weights.
# Positive = boost that factor, Negative = suppress that factor.
# Gate moves are per-gate, not uniform: BEAR tightens ROCE but RELAXES growth and PEG
# (see each delta's comment). Pinned direction-by-direction in test_qglp_profile_fixed.py.
# ═══════════════════════════════════════════════════════════════
REGIME_ADJUSTMENTS = {
    "BULL": {
        "label": "🟢 Bull Market — Offence Mode",
        # Shift weights: boost Growth + reduce Price conservatism
        "quality_delta":   -0.05,   # slightly less defensive
        "growth_delta":    +0.10,   # chase earnings acceleration
        "longevity_delta": -0.05,   # longevity less critical in bull
        "price_delta":     +0.00,   # keep price neutral
        # Gates loosen slightly — rising tide lifts quality boats
        "roce_gate_delta":   0.0,
        "growth_gate_delta": +5.0,  # demand even higher growth in bull
        "peg_gate_delta":   +0.5,   # tolerate slightly higher PEG
        # Momentum gets extra weight in composite blend
        "momentum_boost": 1.10,
    },
    "BEAR": {
        "label": "🔴 Bear Market — Defence Mode",
        # Shift weights: boost Quality + Longevity, suppress Growth
        "quality_delta":   +0.15,   # fortress quality demanded
        "growth_delta":    -0.10,   # growth doesn't matter if market is crashing
        "longevity_delta": +0.05,   # survivors with 10Y track record
        "price_delta":     -0.10,   # ignore valuation (everything looks cheap)
        # Gates tighten — only the best survive
        "roce_gate_delta":   +5.0,  # ROCE > 20% demanded
        "growth_gate_delta": -5.0,  # relax growth gate (everyone is suffering)
        "peg_gate_delta":   +1.0,   # relax PEG (denominator is depressed)
        # Momentum is dangerous in bear — suppress
        "momentum_boost": 0.70,
    },
    "SIDEWAYS": {
        "label": "🟡 Sideways Market — Neutral",
        # No adjustments — pure profile weights apply
        "quality_delta":   0.0,
        "growth_delta":    0.0,
        "longevity_delta": 0.0,
        "price_delta":     0.0,
        "roce_gate_delta":   0.0,
        "growth_gate_delta": 0.0,
        "peg_gate_delta":   0.0,
        "momentum_boost": 1.0,
    },
}


# Does the DETECTED regime change scores? Not until evidence says it should (2026-10-08). The detector
# (scoring_engine.detect_market_regime) votes 2 of 3 on the share of stocks BEATING the Nifty 500, which
# sits near 50% by construction: on 2026-10-08 it read 59.5% / 56.3% / 46.0% against BULL bars of 60 /
# 55 / 55 — half a point from "Offence Mode" — while the Nifty 500 itself was in a Weinstein Stage-4
# decline below its 200-day average. A flip would re-score every stock through REGIME_ADJUSTMENTS,
# which were never tested, and BEAR's momentum ×0.70 runs against PRISM's own measurement (momentum
# was the strongest signal in the falling Aug→Sep window). So scoring uses SIDEWAYS — every adjustment
# neutral, exactly what scored the universe until now — and the detected regime is shown, not applied.
# December's review decides whether a regime (perhaps the index's own trend) should touch scores at all.
REGIME_DRIVES_SCORING = False


def get_adaptive_weights(profile_name: str, regime: str = "SIDEWAYS") -> dict:
    """The Weight Factory — cascades Profile → Regime → Final Weights.
    
    Returns a dict with final QGLP weights, gate thresholds, and momentum boost,
    all adjusted for the current market regime.
    """
    # Resolve the NAME first, not just the dict: the return echoes profile_name into the scoring
    # banner, so falling back to Balanced's numbers while still reporting the caller's string
    # would print "Profile: Turnaround" over Balanced's gates — the engine-says-one-thing class.
    # Unknown names became reachable on 2026-09-20 when the other seven profiles were removed.
    profile_name = profile_name if profile_name in MASTER_PROFILES else "Balanced"
    profile = MASTER_PROFILES[profile_name]
    adj = REGIME_ADJUSTMENTS.get(regime, REGIME_ADJUSTMENTS["SIDEWAYS"])

    # 1. Apply regime deltas to base profile weights
    raw_q = profile["quality_w"]   + adj["quality_delta"]
    raw_g = profile["growth_w"]    + adj["growth_delta"]
    raw_l = profile["longevity_w"] + adj["longevity_delta"]
    raw_p = profile["price_w"]     + adj["price_delta"]

    # 2. Clamp to [0.05, 0.80] — never zero out a factor completely
    raw_q = max(0.05, min(0.80, raw_q))
    raw_g = max(0.05, min(0.80, raw_g))
    raw_l = max(0.05, min(0.80, raw_l))
    raw_p = max(0.05, min(0.80, raw_p))

    # 3. Re-normalize so they sum to exactly 1.0
    total = raw_q + raw_g + raw_l + raw_p
    final_q = round(raw_q / total, 3)
    final_g = round(raw_g / total, 3)
    final_l = round(raw_l / total, 3)
    final_p = round(1.0 - final_q - final_g - final_l, 3)  # absorb rounding error

    # 4. Apply regime deltas to gate thresholds
    final_roce_gate   = max(5.0, profile["roce_gate"]   + adj["roce_gate_delta"])
    final_growth_gate = max(0.0, profile["growth_gate"]  + adj["growth_gate_delta"])
    final_peg_gate    = max(0.5, profile["peg_gate"]     + adj["peg_gate_delta"])

    return {
        "quality_w":     final_q,
        "growth_w":      final_g,
        "longevity_w":   final_l,
        "price_w":       final_p,
        "roce_gate":     final_roce_gate,
        "growth_gate":   final_growth_gate,
        "peg_gate":      final_peg_gate,
        "momentum_boost": adj["momentum_boost"],
        "regime":         regime,
        "profile_name":   profile_name,
        "regime_label":   adj["label"],
    }


# ═══════════════════════════════════════════════════════════════
# EPOCH 2: REINVESTMENT MOAT THRESHOLDS (7th–12th WCS, 2002–2007)
# Three mathematical identities for self-funding compound machines:
#   Identity A: Reinvestment Rate (RR) = 1 − DPR
#   Identity B: Fundamental Growth Capacity = ROE × RR
#   Identity C: Buffett Value Creation Ratio (VCR) ≥ 1.0 to pass
# ═══════════════════════════════════════════════════════════════
EPOCH2_REINVESTMENT = {
    "min_reinvestment_rate":    0.60,   # DPR < 40% — retaining ≥60% of earnings
    "min_capital_efficiency":   20.0,   # ROCE/ROE baseline hurdle rate (%)
    "min_value_creation_ratio": 1.0,    # Buffett 1-to-1 dollar baseline test hurdle
    "elite_vcr_threshold":      2.0,    # VCR ≥ 2.0 = elite capital allocator
    "quality_boost_pts":        10.0,   # Quality score boost for flag_epoch2_compounder
    "misallocation_penalty":    0.90,   # 10% quality score penalty for capital misallocators
}


# ═══════════════════════════════════════════════════════════════
# EPOCH 3: TAXONOMY & MOAT ENDURANCE (13th–18th WCS, 2008–2013)
# ═══════════════════════════════════════════════════════════════
EPOCH3_TAXONOMY = {
    # Great/Good/Gruesome classification boundaries
    "great_roce_spread_floor": 10.0,    # ROCE - CoC ≥ 10% = structural monopoly
    "great_fcf_velocity_min":  0.60,    # FCF/OCF ≥ 60% = organic cash machine
    "good_roce_spread_floor":  5.0,     # ROCE - CoC ≥ 5% = efficient operator
    "good_fcf_velocity_max":   0.60,    # FCF/OCF < 60% = capex-heavy growth model
    "gruesome_roce_ceiling":   12.0,    # 10Y median ROCE < 12% = value destruction zone

    # Moat Endurance Factor (MEF) — 17th WCS
    "mef_expanding_threshold": 1.0,     # MEF ≥ 1.0 = moat intact/expanding
    "mef_eroding_threshold":   0.80,    # MEF < 0.80 = severe moat degradation

    # Payback Ratio enhancement thresholds (15th WCS)
    "payback_dislocation_max": 2.0,     # Payback < 2.0 during crises = asymmetric setup

    # Capital Return Floor (Epoch 3 structural filter)
    "capital_return_floor_10y": 20.0,   # 10Y median ROCE ≥ 20%
    "capital_return_floor_7y":  20.0,   # 7Y median ROCE ≥ 20%

    # Debt Solvency Perimeter
    "min_interest_coverage":   4.0,     # ICR ≥ 4.0x (stricter than Malik's 3x)
    "max_debt_to_equity":      1.0,     # D/E < 1.0 absolute ceiling
    "preferred_de_non_fin":    0.5,     # D/E < 0.5 preferred for industrials

    # Free Cash Verification Gate
    "cfo_pat_structural_min":  80.0,    # CFO ≥ 80% of PAT (percentage)

    # Anti-Pattern: Cyclical Profit Mirage
    "mirage_rev_growth_min":   25.0,    # Revenue YoY > 25%
    "mirage_roce_10y_max":     12.0,    # 10Y median ROCE < 12%

    # Scoring impact
    "gruesome_quality_penalty": 0.50,   # 50% haircut to quality_score
    "great_quality_boost":      1.10,   # 10% boost for Great companies
}


# ═══════════════════════════════════════════════════════════════
# EPOCH 4: SQGLP & ACCOUNTING INTEGRITY (19th–25th WCS, 2014–2020)
# ═══════════════════════════════════════════════════════════════
EPOCH4_SQGLP = {
    "min_cfo_to_pat_ratio": 80.0,       # Saurabh Mukherjea / 24th WCS Cash Quality Floor — PERCENTAGE (cfo_to_pat is stored as 73.04, not 0.73)
    "max_promoter_pledge": 10.0,        # Strict 10% maximum pledging boundary
    "max_peg_ratio": 1.5,               # 23rd WCS anchor — relaxed from 1.0: Indian quality mid-caps rarely trade at PEG < 1
}


# ═══════════════════════════════════════════════════════════════
# EPOCH 5: MODERN DIGITAL & MACRO FRONTIER (26th–30th WCS, 2011–2025)
# Atoms vs Bits paradigm, Consistents vs Volatiles classification,
# Bruised Blue Chip entry matrix, Multi-Trillion Tipping Point velocity.
# ═══════════════════════════════════════════════════════════════
EPOCH5_MODERN = {
    "bbc_roce_floor":    20.0,          # 29th WCS: minimum 10Y median ROCE for BBC flag
    "bbc_pb_ceiling":    2.0,           # 29th WCS: "typically P/B less than 2x" — verbatim from p.285
    "tipping_sectors": [                # 30th WCS: sectors hitting multi-trillion tipping point
        # Names verified against CSV Sector column (81 unique values)
        "Financial Services",
        "Stock/ Commodity Brokers",     # CSV has space after /
        "Consumer Durables",
        "Automobile",                   # CSV has no trailing 's'
        "Quick Service Restaurant",     # CSV has no trailing 's'
    ],
    "max_volatility_ratio": 0.40,       # 27th WCS: Consistency Coefficient ceiling for Consistents
}

# ═══════════════════════════════════════════════════════════════
# EPOCH 3.5: UNUSUAL BILLIONAIRES — Greatness Formula Config
# Source: Mukherjea, "The Unusual Billionaires" (2016)
# Sector-routed growth hurdles: non-financials use revenue CAGR;
# financials use loan-book expansion proxy (same Greatness Formula
# research base as Coffee Can's explicit 15% financial hurdle).
# ═══════════════════════════════════════════════════════════════
EPOCH35_UNUSUAL_BILLIONAIRES = {
    "non_financial_growth_hurdle": 10.0,  # Revenue CAGR ≥ 10% for industrials/consumer/pharma
    "financial_growth_hurdle":     15.0,  # Loan/Revenue CAGR ≥ 15% for banks and NBFCs
    "capital_efficiency_hurdle":   15.0,  # ROCE (non-fin) / ROE (fin) ≥ 15% — Greatness Formula floor
    "min_promoter_stake":          45.0,  # Owner-operator alignment barrier (UB case studies)
}

# 27th WCS: Sectors MOSL classified as Consistent compounders (sustained earnings 2007–2022).
# Verified against 81 CSV Sector column values — previous inline set in data_engine had 7 wrong names.
# Removed: "Banks - Private Sector" (no such sector), "Utilities" (no such sector).
# Fixed: "Cigarettes"→"Tobacco Products", "Diamonds/Gems/Jewellery"→"Diamond, Gems and Jewellery",
#        "IT"→"IT - Software"+"IT - Hardware", "Oil & Gas"→"Crude Oil & Natural Gas", "Pharma"→"Pharmaceuticals".
CONSISTENT_SECTORS = frozenset([
    "Agro Chemicals",
    "Auto Ancillaries",
    "Cement",
    "Chemicals",
    "Credit Rating Agencies",
    "Diamond, Gems and Jewellery",
    "Engineering",
    "Finance",
    "FMCG",
    "IT - Software",
    "IT - Hardware",
    "Logistics",
    "Crude Oil & Natural Gas",
    "Paints/Varnish",
    "Pharmaceuticals",
    "Refineries",
    "Tobacco Products",
])


# ═══════════════════════════════════════════════════════════════
# 8. FORENSIC ENGINE THRESHOLDS
# ═══════════════════════════════════════════════════════════════
# Fixed denominator for forensic_score: (max_flags - red_flag_count) / max_flags × 100.
# Must be updated manually when a new rf_ flag is added to forensic_engine.py.
# Current flags: 28 active rf_ columns (rf_snoa added 2026-06-13, Quantitative Value audit).
# rf_capex_mirage, rf_ccc_worsening, rf_cwip_bloat, rf_debt_ebitda_high, rf_dilution,
# rf_expense_rising, rf_fcf_to_cfo_low, rf_high_accruals, rf_high_cash_debt, rf_high_receivables,
# rf_inventory_bloat, rf_itr_declining, rf_lease_inflation, rf_low_cfo_ebitda, rf_low_cfo_pat,
# rf_low_fcf_ebitda, rf_margin_squeeze, rf_negative_fcf, rf_nfat_very_low, rf_opm_volatile,
# rf_pledge_elevated, rf_psu_value_destruction, rf_receivables_bloat, rf_rising_debt,
# rf_snoa, rf_ssgr_deficit, rf_tax_panic, rf_wc_double_squeeze.
FORENSIC_MAX_FLAGS = 28
FORENSIC = {
    "cfo_pat_alert":              70.0,   # below this = Level 1 red flag (percentage, e.g. 73.04 = 73%)
    # FORENSIC CFO/EBITDA floor — recalibrated 90 → 50 (2026-06-12 Schilit audit).
    # CFO is AFTER tax + interest + working capital; EBITDA is BEFORE all three, so
    # ~75% is mathematical PAR for a clean 25%-tax company (live universe median: 68.8%).
    # The old 90% line was Mukherjea's Coffee Can ELITE-QUALITY gate (it stays in
    # fw_coffee_can, untouched) — as a RED FLAG it fired for 54% of the universe,
    # counting ordinary tax mathematics toward red_flag_count and the cascading
    # multiplier. Schilit's actual cash detections are CFFO vs Net Income
    # (rf_low_cfo_pat < 70%) and FCF/EBITDA < 0.3 (rf_low_fcf_ebitda) — both exact.
    # 50% = conversion clearly below what taxes alone explain → genuine anomaly only.
    "cfo_ebitda_clean_threshold": 50.0,  # percentage threshold; rf_low_cfo_ebitda fires below this
    "receivable_rise_days":  15,    # DSO rising more than this = flag
    "inventory_vs_revenue":  True,  # inv growth > rev growth = flag
    "capex_depr_ratio_max":  3.0,   # capex/depr > 3 without rev jump
    "pledge_watch":          10.0,  # above this = watch
    "pledge_critical":       20.0,  # above this = critical
    "expense_ratio_rising":  True,  # rising expense ratio = flag
}

# Piotroski F-Score thresholds
PIOTROSKI = {
    "strong": 7,    # F-Score ≥ 7 = strong
    "moderate": 5,  # F-Score 5-6 = moderate
    "weak": 0,      # F-Score ≤ 4 = weak
}

# ═══════════════════════════════════════════════════════════════
# 9. RSI ZONE SCORING
# ═══════════════════════════════════════════════════════════════
RSI_ZONES = {
    "overbought":   {"min": 80, "max": 100, "score": 10},
    "strong_trend":  {"min": 70, "max": 80,  "score": 60},
    "sweet_spot":    {"min": 55, "max": 70,  "score": 100},
    "neutral":       {"min": 45, "max": 55,  "score": 50},
    "weak":          {"min": 30, "max": 45,  "score": 20},
    "oversold":      {"min": 0,  "max": 30,  "score": 40},  # mean-reversion potential
}

# ═══════════════════════════════════════════════════════════════
# 9b. 52WH AGE ZONE SCORING
# ═══════════════════════════════════════════════════════════════
# Scores the FRESHNESS of the 52-week high (how many days ago it was set).
# Pairs with 52wh_distance (magnitude): both needed for complete breakout picture.
# A stock 5% from its 52WH set yesterday is completely different from
# one 5% from its 52WH set 280 days ago (stale overhead supply).
HIGH_AGE_ZONES = {
    "breakout_fresh":  {"min": 0,   "max": 31,   "score": 100},  # 0-30d: just made/near new high
    "consolidating":   {"min": 31,  "max": 91,   "score": 75},   # 1-3m: classic base/flag setup
    "building":        {"min": 91,  "max": 181,  "score": 45},   # 3-6m: overhead supply forming
    "stale":           {"min": 181, "max": 253,  "score": 20},   # 6-12m: high becoming resistance
    "very_stale":      {"min": 253, "max": 9999, "score": 5},    # 1y+: failed breakout or downtrend
}
# NaN → default 50 (neutral) — missing data never penalises

# ═══════════════════════════════════════════════════════════════
# 9c. MINERVINI LINES MORE THAN ONE MODULE READS (2026-10-15)
# ═══════════════════════════════════════════════════════════════
# One definition each: the engine applies them and every screen that states them reads them, so a
# retuned line cannot leave stale words behind (the SEPA panel said "10D < 50D" for months after the
# dry-up moved to 70%). Trend Template criterion 7: within this % of the 52-week high — the SEPA
# certification gate and the ⏳ Consolidation Base ⭐ setup both apply it.
TREND_TEMPLATE_MAX_BELOW_HIGH = 25
# SEPA's India market-cap floor (₹ Cr) for certification.
SEPA_MIN_MARKET_CAP_CR = 500

# ═══════════════════════════════════════════════════════════════
# 10. UI CONFIGURATION
# ═══════════════════════════════════════════════════════════════
UI = {
    "app_title": "PRISM",
    "app_icon": "🔷",
    "app_subtitle": "Every lens. One verdict.",
    "version": "1.0.0",
    "max_display_default": 100,
    "font_url": "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap",
}

# Color palette
COLORS = {
    "bg_primary":    "#0d1117",
    "bg_secondary":  "#161b22",
    "bg_tertiary":   "#21262d",
    "border":        "#30363d",
    "border_hover":  "#484f58",
    "text_primary":  "#e6edf3",
    "text_secondary":"#8b949e",
    "text_muted":    "#6e7681",
    "gold":          "#e3b341",
    "green":         "#3fb950",
    "blue":          "#58a6ff",
    "red":           "#f85149",
    "purple":        "#8b5cf6",
    "orange":        "#FF6B35",
    "cyan":          "#00CED1",
    "gradient_start":"#1a1a2e",
    "gradient_mid":  "#16213e",
    "gradient_end":  "#0f3460",
}

# Tier-specific colors for the conviction table
TIER_COLORS = {
    1: {"bg": "rgba(255,215,0,0.06)",  "border": "rgba(255,215,0,0.3)",  "text": "#FFD700"},
    2: {"bg": "rgba(63,185,80,0.06)",  "border": "rgba(63,185,80,0.3)",  "text": "#3fb950"},
    3: {"bg": "rgba(88,166,255,0.06)", "border": "rgba(88,166,255,0.3)", "text": "#58a6ff"},
    4: {"bg": "rgba(210,153,34,0.06)", "border": "rgba(210,153,34,0.3)", "text": "#d29922"},
    5: {"bg": "rgba(248,81,73,0.06)",  "border": "rgba(248,81,73,0.3)",  "text": "#f85149"},
}

# Framework taxonomy — the 5 §7 (CLAUDE.md) groups. Used to categorize the 37 frameworks in the UI:
# compact category COUNTS on cards, grouped chips on the tearsheet. Reveals a stock's conviction
# CHARACTER (quality-moat vs momentum vs value play) instead of a flat undifferentiated pill list.
# Names MUST match _FW_META / frameworks_passed exactly (the §7 zero-duplicate contract;
# the 36 live names were verified against frameworks_passed 2026-06-14; "Blue Chip Quality" is the
# 37th — currently DPR-dead so it never appears, but it is mapped for when the source column is fixed).
FRAMEWORK_CATEGORIES = [
    ("🏛️", "MOSL",     COLORS["blue"], [
        "QGLP", "MOSL Wealth Creator", "SQGLP Century Stock", "100x Candidate",
        "Fallen Quality", "CAP-GAP Compounder", "Economic Moat", "Blue Chip Quality",
        "Consistent in Volatile", "EP Hockey Stick", "Bruised Blue Chip 29", "Multi-Trillion Cap",
    ]),
    ("📚", "Moats",    COLORS["green"], [
        "Coffee Can", "Diamond", "Peaceful Investing", "Unusual Billionaires",
        "Long Game Quality", "Baid Compounder", "Basant 30% Club", "Quality Compounder",
    ]),
    ("⚡", "Momentum", COLORS["orange"], [
        "CAN SLIM", "SEPA Momentum", "Quality Momentum", "Lynch Dream", "EP Improver", "SMILE",
    ]),
    ("🛡️", "Value",    COLORS["gold"], [
        "Magic Formula", "Dhandho Asymmetry", "Parikh Contrarian", "Wide Moat",
        "Outsider CEO", "Expectations Matrix", "Financial Shenanigans", "Marks Cycle Shield",
    ]),
    ("🎣", "Fisher",   COLORS["cyan"], [
        "Fisher Quality", "Fisher Scalability", "100-Bagger",
    ]),
]
