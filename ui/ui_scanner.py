"""Scanner column header-tip map.

This module once held an AgGrid `render_scanner_grid`; that grid was retired (superseded by the
`st.dataframe` Deep Scanner in app.py) and now lives only in git history. What remains is the live,
single-source tooltip map the Deep Scanner wires into its column_config `help=`.
"""

# Plain-language header tooltips for the scanner's machine-named columns. Reuse the SAME glossary
# the tearsheet "?" chip uses (single source of truth — a term can never drift between the grid
# header and the tearsheet); bespoke sentences only for scanner-only columns with no tearsheet cell.
from ui.ui_components import _RAW_GLOSSARY as _GLOSSARY

_SCANNER_HEADER_TIPS = {
    "rank":             "Overall PRISM rank by Score across the whole universe (1 = highest). It stays the same whatever the table is sorted or filtered by.",
    "result_when":      "How current this stock's financials are — '8d ago' = last results reported 8 days back; '📅 due 4d' = a result is SCHEDULED in 4 days and not yet declared, so every number here is still the previous quarter's. Appears when sorting by 🆕 Results.",
    "d48_breakout_readiness": "How close price is to breaking out — 🎯 IMMINENT: within 10% of the 52-week high AND 5% of the 13-week high; NEAR: within 20% of the 52-week high; FAR: more than 20% below. The categorical read of the Breakout score beside it.",
    "d49_momentum_quality": "Whether the momentum is healthy or exhausted — ⚡ HIGH: RSI in the 50–70 zone with a confirmed trend (ADX > 20); 🔥 OVERHEATED: RSI above 70, strong but stretched, chase risk elevated; WEAK: neither. Reads the Momentum and RSI columns beside it.",
    "verdict_direction":"The engine's one-word gate (SOUND / MIXED / FLAWED) synthesised from all 6 axes after the forensic penalty.",
    "wealth_tier":"The wealth-engine tier (BUY★ / BUY / WATCH★ / WATCH / AVOID / N/A) — is the business becoming more valuable? Three clocks: EP% (ROE − cost of equity), Vel% (this year's change in it), 5-year margin tau. Price-blind and forensics-blind; a description, not a recommendation.",
    "corporate_class":  _GLOSSARY["Corporate Class"],
    "composite_score":  _GLOSSARY["Composite Score"],
    "conviction_tier":  _GLOSSARY["Conviction Tier"],
    "data_coverage_pct": _GLOSSARY["Evidence Coverage"],
    "forensic_multiplier": _GLOSSARY["Forensic Mult"],
    "schilit_pass":     _GLOSSARY["Schilit Pass"],
    "moat_growth_quad": "Where the stock sits on the moat-versus-growth map (e.g. Wealth Creator, Quality Trap, Growth Trap).",
    "fisher_lifecycle_quadrant": "Phil Fisher's growth-versus-quality lifecycle quadrant for the business (e.g. Catalyst Play, Laggard).",
    "cash_score":       "A 0-100 score for how strongly the business converts reported profit into real operating cash.",
    "buy_zone_label":   _GLOSSARY["Buy Zone"],
    "forensic_score":   _GLOSSARY["Forensic Scr"],
    "red_flag_count":   _GLOSSARY["Red Flags"],
    "red_flag_list":    "The names of the forensic red flags this stock is carrying (the 'which', beside the count) — 'Clean ✅' when none fired.",
    "momentum_score":   _GLOSSARY["Momentum Scr"],
    # ── Deep Scanner view-preset columns (Quality / Valuation / Forensic / Technical) ──
    # Reuse the SAME glossary the tearsheet "?" chip uses (single source of truth); bespoke text only
    # where a column has no tearsheet-cell glossary entry. Coverage is pinned by
    # tests/test_tooltip_coverage.py::test_every_scanner_preset_column_has_header_tip.
    "smart_money_flow":       _GLOSSARY["Smart Money"],
    "quality_score":          "Overall fundamental quality sub-score (0-100): moat, growth, improvement, cash, margins, balance sheet and valuation, before the forensic penalty.",
    "moat_score":             _GLOSSARY["Moat Score"],
    "growth_score":           _GLOSSARY["Growth Score"],
    "improvement_score":      _GLOSSARY["Improvement Score"],
    "governance_bonus":       _GLOSSARY["Governance Score"],
    "piotroski_fscore":       _GLOSSARY["Piotroski"],
    "roce":                   _GLOSSARY["ROCE Current"],
    "roce_expansion":         _GLOSSARY["ROCE Δ 10Y"],
    "roce_inflection":        _GLOSSARY["ROCE Δ 2Y"],
    "opm":                    _GLOSSARY["OPM"],
    "opm_acceleration":       _GLOSSARY["OPM Δ YoY-Q"],
    "cfo_to_pat":             _GLOSSARY["CFO/PAT"],
    "valuation_score":        _GLOSSARY["Valuation Scr"],
    "expected_excess_return": "Mauboussin expected-value excess return (%): the probability-weighted upside-versus-downside payoff over the base case.",
    "pe":                     _GLOSSARY["PE"],
    "pb_ratio":               _GLOSSARY["P/B"],
    "peg":                    _GLOSSARY["PEG"],
    "earnings_yield":         _GLOSSARY["Earnings Yield"],
    "dividend_yield":         _GLOSSARY["Dividend Yield"],
    "fcf_yield":              _GLOSSARY["FCF Yield"],
    "market_cap":             "Total market value of the company's equity (price × shares), in ₹ crore.",
    "close_price":            "Last closing price (₹), as of the most recent data refresh.",
    "fair_value_qglp":        "QGLP fair value (₹): the quality-justified fair P/E × EPS. Blank for loss-makers — an earnings-multiple fair value is undefined without positive earnings.",
    "dist_to_vstop":          "How far the price sits above (+) or below (−) its Volatility Stop, in %. Negative = trend broken.",
    "accruals_ratio":         _GLOSSARY["Accruals Ratio"],
    "debt_to_equity":         _GLOSSARY["D/E Ratio"],
    "promoter_holdings":      _GLOSSARY["Promoter %"],
    "pledged_percentage":     _GLOSSARY["Pledge %"],
    "change_promoter_1y":     _GLOSSARY["Promoter 1Y Δ"],
    "fii_holdings":           _GLOSSARY["FII %"],
    "change_fii_lq":          _GLOSSARY["FII Chg"],
    "dii_holdings":           _GLOSSARY["DII %"],
    "change_dii_lq":          _GLOSSARY["DII Chg"],
    "rsi_14d":                _GLOSSARY["RSI 14D"],
    "dist_52wh":              _GLOSSARY["Dist 52WH"],
    "crs_52w":                _GLOSSARY["CRS 52W"],
    "dist_ath":               _GLOSSARY["Off ATH"],
    "weinstein_stage":        _GLOSSARY["Weinstein Stage"],
    "breakout_score":         _GLOSSARY["Breakout Scr"],
    "vstop_green":            _GLOSSARY["VSTOP Green"],
    "gate_pass":              "Whether the stock clears ALL the engine's hard quality gates (e.g. ROCE, growth, positive-PAT floors).",
    "tsunami_signal":         "A rare confluence flag — a quality breakout meeting institutional accumulation; fires for only a handful of stocks by design.",
}


# ── Deep Scanner column rule (2026-10-05) ──────────────────────────────────────────────────────
# The column that makes each sort READABLE on screen. composite_score's order IS rank (the engine
# re-derives rank from the post-penalty score — pinned by test_rank_tracks_post_penalty_composite)
# and rank leads every view, so the Score sort adds nothing; the 🆕 Results sort orders by a raw day
# count the app shows as readable text (result_when).
_SORT_WITNESS = {"composite_score": "rank", "result_age_days": "result_when"}


def ds_view_columns(view, sort_col, available):
    """The columns the Deep Scanner shows for a view: the view's own list (it leads with Rank · Stock)
    plus whatever the table is sorted by, placed right after Stock when the view does not already
    carry it — so the order on screen can always be read. A view that carries it is left exactly as
    curated. Only columns in `available` are returned; an absent column is never invented."""
    cols = [c for c in view if c in available]
    shown = _SORT_WITNESS.get(sort_col, sort_col)
    if shown not in available:
        shown = sort_col
    if shown in available and shown not in cols:
        cols.insert(cols.index("name") + 1 if "name" in cols else 0, shown)
    return cols
