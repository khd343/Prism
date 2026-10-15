"""
PRISM — Tearsheet Visualization Layer
=====================================
Deep-dive charts and WCS 28/29/30 framework cards for individual stocks.
All functions are PURE DISPLAY — zero sorting, grouping, or math.
Pre-calculated vectors arrive from data_engine + scoring_engine + forensic_engine.
"""

import re
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import html as _html
from config import (COLORS, CONVICTION_TIERS, TIER_COLORS, FORENSIC_MAX_FLAGS,
                    FRAMEWORK_CATEGORIES, MASTER_PROFILES, INDIA_GSEC_YIELD, COST_OF_EQUITY,
                    PRELISTING_BASELINE_DAYS, TREND_TEMPLATE_MAX_BELOW_HIGH, SEPA_MIN_MARKET_CAP_CR)
# Single source of truth for the "?" help chip lives in ui_components (which owns the .ts-help CSS).
# Re-imported here so this module's renderers AND existing `from ui.ui_tearsheet import ...` callers
# (the scanner, the tests) resolve against the SAME objects — one definition, zero drift.
from ui.ui_components import help_chip, _RAW_GLOSSARY
from ui.ui_liquidity import trade_line
# MIRROR, never restate: rf_high_receivables fires on a PER-SECTOR DSO gate, so the evidence row
# reads the engine's own table. A local copy would drift the moment a sector norm is retuned —
# which is exactly how the ">75d / >120d" string below got stranded (see the handler's comment).
from core.forensic_engine import _SECTOR_DSO_THRESHOLDS, _DEFAULT_DSO_THRESHOLD
# Same rule for the ⏳ Consolidation Base card and the SEPA panel's V box: the engine's lines and labels.
from core.data_engine import CONSOLIDATION_BASE_LABELS, VCP_DRYUP_SHARE


# ─── Display Utilities ──────────────────────────────────────────────────────

def _esc(val) -> str:
    """Escape HTML special characters before raw-HTML markdown injection.
    Prevents XSS and broken layouts from corporate strings like industry descriptions
    or insider-trading comments that contain quotes, ampersands, or angle brackets.
    """
    return _html.escape(str(val)) if val is not None else ""


def _g(stock: pd.Series, key: str, default=0):
    """Null-safe Series lookup. Returns default if key is missing or NaN."""
    v = stock.get(key, default)
    return default if (v is None or (isinstance(v, float) and np.isnan(v))) else v


def _shown(value, dp: int = 0) -> float:
    """The number AS THE READER SEES IT — rounded to the displayed precision.

    Classify colours, zone words and bands on THIS, never on the raw value. Otherwise two cells
    print the identical string and wear different colours, with nothing on screen to explain the
    difference: growth_score 69.6744 and 70.3481 both render "70" while one is gold and the other
    green. Measured 2026-08-27 across the tearsheet: 540 such renders on 10 surfaces.

    WHEN THIS IS THE RIGHT FIX: the threshold is a DISPLAY BAND — Strong/Average/Weak at 70/40,
    "cheaper than X% of peers" at 75/25. Those cuts are presentation conventions, so moving a value
    to the band its printed form implies loses nothing.

    WHEN IT IS NOT: the threshold is a real economic line. VCR's 1.00 is Buffett's one-dollar
    premise — blurring its edge would destroy the signal, so that chip shows MORE PRECISION instead
    (tests/test_deep_signal_chips.py). Nor does it apply where the colour comes from an ENGINE
    classification that is separately labelled on the page: the hero ring is coloured by
    conviction_tier, and the tier's own name sits beside it, so the number is not what the reader
    is decoding. Ask what the threshold MEANS before reaching for this.
    """
    return round(float(value), dp)


def _same_side(raw: float, rule) -> str:
    """`raw` printed with the fewest decimals (0-2) that leave `rule` as true or false as it is for the
    raw value — the "real economic line" case in _shown's docstring, where precision is added rather
    than the value moved. The ⏳ Consolidation Base card's lines are the engine's own: a 69.7% volume
    ratio is a dry-up (below 70%), so it never prints as "70%"; 25.3% below the high missed the 25%
    line, so it never prints as "25%"."""
    for dp in (0, 1, 2):
        if bool(rule(round(raw, dp))) == bool(rule(raw)):
            return f"{raw:.{dp}f}"
    return f"{raw:.3f}"


def _parse_frameworks(fw_str, exclude: set = None) -> list:
    """Split the `frameworks_passed` string into clean, whole-token framework names.

    Contract: the engine joins framework names with the exact separator ", ". Splitting on
    a strict ", "-with-flexible-whitespace boundary (re.split r'\\s*,\\s*') yields complete
    tokens only — never partial substrings. This guarantees that a compound name such as
    "Bruised Blue Chip 29" is treated as ONE atomic token and can never cross-contaminate a
    shorter standalone variant ("Bruised Blue Chip") in any downstream membership test or
    dropdown option. Empty fragments and the sentinel "None" are dropped; `exclude` removes
    dedicated-pill names that are rendered separately (exact whole-token match only).
    """
    if not fw_str or str(fw_str).strip() in ("", "None"):
        return []
    exclude = exclude or set()
    tokens = re.split(r"\s*,\s*", str(fw_str).strip())
    return [t for t in (tok.strip() for tok in tokens) if t and t != "None" and t not in exclude]


# ─── Forensic flag badge registry (mirrors flag_descriptions in forensic_engine) ──
_FLAG_DISPLAY = {
    "rf_low_cfo_pat":       ("Low CFO/PAT (<70%) — earnings not backed by cash",           "🔴"),
    "rf_high_receivables":  ("High DSO for its sector (45–120d gate)",                    "🟠"),
    "rf_inventory_bloat":   ("Inventory growing faster than revenue",                        "🟡"),
    "rf_rising_debt":       ("D/E rising materially (>10% relative rise AND D/E >0.3)",     "🟠"),
    "rf_ccc_worsening":     ("Cash conversion cycle worsening by >10 days",                 "🟡"),
    "rf_expense_rising":    ("Expense ratio rising >3 percentage points",                    "🟡"),
    "rf_pledge_elevated":   ("Promoter pledge >10% of shares",                              "🔴"),
    "rf_dilution":          ("Meaningful share dilution (>3% Tier 2+)",                     "🟠"),
    "rf_negative_fcf":      ("Negative FCF AND negative OCF — true cash burn",              "🔴"),
    "rf_margin_squeeze":    ("Revenue growth +5% but PAT declining — margin collapse",      "🟠"),
    "rf_high_cash_debt":    ("High cash + high debt simultaneously (Malik Shenanigan 4)",   "🟡"),
    "rf_itr_declining":     ("Inventory turnover declining >10% YoY (Malik Shenanigan 3)", "🟡"),
    "rf_ssgr_deficit":      ("Actual growth exceeds SSGR by >5% — debt-dependent growth",  "🔴"),
    "rf_high_accruals":     ("High accruals >5% of assets — Beneish TATA forensic signal",  "🔴"),
    "rf_low_fcf_ebitda":    ("FCF/EBITDA <30% — EBITDA significantly overstates real cash", "🟠"),
    "rf_fcf_to_cfo_low":    ("FCF/CFO <15% — capital trap: capex consuming all OCF",        "🟠"),
    "rf_opm_volatile":      ("OPM deviates >30% from 5Y median — unstable margins, no pricing power", "🟡"),
    "rf_nfat_very_low":     ("NFAT <1.5 — extreme capital intensity, growth destroys value","🟡"),
    "rf_debt_ebitda_high":  ("Debt/EBITDA >5× — Amtek Auto collapse pattern",              "🔴"),
    "rf_cwip_bloat":        ("CWIP share of assets grew >50% YoY — IL&FS balance-sheet parking", "🟠"),
    "rf_capex_mirage":      ("Rev growth >20% but capex <0.5× dep — deferred-maintenance time bomb", "🟠"),
    "rf_tax_panic":         ("Effective tax rate <10% despite PAT >0 — Sharp Practices (WCS 24)", "🔴"),
    "rf_receivables_bloat":        ("DSO expansion >20 days above sector median — relative receivables manipulation", "🟡"),
    "rf_psu_value_destruction":    ("PSU Value-Destruction Loop — low capital spread + high payout + CWIP delays", "🟠"),
    "rf_lease_inflation":          ("Ind AS 116 lease mirage — EBITDA inflated by RoU capitalisation (QSR/Retail/Aviation)", "🟡"),
    "rf_low_cfo_ebitda":           ("CFO/EBITDA <50% — cash conversion far below tax-math par; EBITDA likely inflated", "🟠"),
    "rf_wc_double_squeeze":        ("DSO rising >10 days AND DPO falling >10 days simultaneously — double working capital squeeze", "🟠"),
    "rf_snoa":                     ("Net operating assets bloat — cumulative accrual build-up (QV SNOA >1.0)", "🟡"),
}


# ═══════════════════════════════════════════════════════════════
# MOAT-GROWTH MATRIX (22nd WCS)
# ═══════════════════════════════════════════════════════════════

def _moat_growth_plot_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the Moat-Growth scatter frame. Moat_Y = ROCE (5y-median → current fallback),
    Growth_X = PAT CAGR (5y → 3y fallback). REAL-value fallback ONLY — no sentinel fill, so a stock
    missing BOTH sources for an axis stays NaN and is EXCLUDED (CLAUDE.md §5 semantic-truth; Plotly
    cannot plot NaN, and 0 would be a false 'destroyer' coordinate). Pure + vectorized — unit-tested
    by tests/test_moat_growth_matrix.py."""
    out = df.copy()
    out["Moat_Y"]   = out["roce_med_5y"].fillna(out["roce"])
    out["Growth_X"] = out["pat_gr_5y"].fillna(out["pat_gr_3y"])
    return out[out["Moat_Y"].notna() & out["Growth_X"].notna()]


def render_moat_growth_matrix(df: pd.DataFrame, highlight_stock: str = None):
    """
    2D scatter — ROCE (Moat) vs PAT CAGR (Growth) for the entire filtered universe.

    Data integrity notes:
    - A stock missing EITHER axis (no ROCE after the 5y→current fallback, OR no growth after the
      5y→3y fallback) is EXCLUDED — Plotly cannot plot NaN, and 0 would be a false "destroyer"
      coordinate (CLAUDE.md §5 semantic-truth). The frame is built by _moat_growth_plot_frame().
    - Stocks with Growth_X > viewport are fully RETAINED in the dataset; the axis range
      parameter clips the visual canvas only (G9 FIX). No rows are dropped for outliers.
    """
    # Chip text is the STUDY'S OWN outcome language (22nd WCS, Exhibit 5 and the paragraph above
    # it), which the glossary's mechanical definitions ("ROCE >= 15% and growth >= 15%") do not
    # carry: the book's point is not where a company plots, it is what each quadrant DOES to you.
    # Note the book draws Moat on X and Growth on Y; this chart transposes them, but every
    # quadrant keeps the book's own name and meaning.
    st.markdown(
        "<div class='sec-head'>🧭 Moat-Growth Matrix (22nd WCS)"
        + help_chip("Moat-Growth Matrix",
                    "22nd WCS, Exhibit 5 — moat and growth are the two dimensions of longevity, "
                    "split at MOSL's 15% cost of equity. The study's own reading of each corner: "
                    "⭐ Wealth Creators are ENDURING multi-baggers · ⚡ Growth Traps are TRANSITORY "
                    "multi-baggers that 'may slip all the way back to where they started off from, "
                    "or even lower' unless sold at the right price · 🛡️ Quality Traps are "
                    "underperformers — a real moat whose engine has stopped compounding · "
                    "💀 Wealth Destroyers are permanent capital loss. "
                    "\"Moat without growth will underperform; growth without moat will end soon.\"")
        + "</div>",
        unsafe_allow_html=True,
    )

    plot_df = _moat_growth_plot_frame(df)

    if len(plot_df) == 0:
        st.warning("Not enough valid data to plot the matrix.")
        return

    # VIEWPORT FITS THE DATA (2026-08-26). This clipped at a flat 300% growth, but the universe's
    # p99 is ~178% and its max ~1,568%, so the canvas stretched to 300 to accommodate under 1% of
    # stocks — and 83.6% of the points (1,704 of 2,038) ended up inside a box worth 8.8% of the
    # plot area, an unreadable solid mass exactly where the median stock lives (growth 11.9%).
    # Clipping at the 98th percentile roughly triples the resolution where 96.6% of stocks sit.
    # Rows are NEVER dropped — the axis range clips the CANVAS only (the original G9 fix) — and
    # the count beyond the edge is stated on the chart, so nothing is hidden silently.
    _x_p98  = float(plot_df["Growth_X"].quantile(0.98))
    _x_true = float(plot_df["Growth_X"].max())
    x_max   = max(min(_x_p98 * 1.02, 300.0), 50.0)   # floor 50 prevents axis collapse at zero growth
    _beyond = int((plot_df["Growth_X"] > x_max).sum())

    fig = px.scatter(
        plot_df, x="Growth_X", y="Moat_Y",
        color="moat_growth_quad",
        color_discrete_map={
            "⭐ Wealth Creator":    COLORS["green"],
            "🛡️ Quality Trap":     COLORS["gold"],
            "⚡ Growth Trap":      COLORS["blue"],
            "💀 Wealth Destroyer": COLORS["red"],
        },
        hover_name="name",
        hover_data={"Growth_X": ":.1f", "Moat_Y": ":.1f", "moat_growth_quad": False},
        labels={"Growth_X": "Growth (PAT CAGR %)", "Moat_Y": "Moat (ROCE %)"},
        # Fully opaque markers made the dense core a flat silhouette — you could not tell one
        # stock from fifty in the very region holding five of every six. Alpha turns overlap
        # into shading, so crowding becomes information instead of a blob.
        opacity=0.55,
    )
    fig.add_vline(x=15, line_width=1, line_dash="dash", line_color=COLORS["border"])
    fig.add_hline(y=15, line_width=1, line_dash="dash", line_color=COLORS["border"])
    # Annotation x-coords are data-relative: right labels at 85% of x_max, left labels at
    # 70% of the -50 left bound. Y coords are absolute (range fixed at [-25, 105]).
    # Hardcoding x=80 clips labels when x_max=50 (low-growth filtered universes).
    _ann_x_right = x_max * 0.85
    _ann_x_left  = -35
    fig.add_annotation(x=_ann_x_right, y=90,  text="⭐ Wealth Creators", showarrow=False,
                       font=dict(color=COLORS["green"], size=16), opacity=0.3)
    fig.add_annotation(x=_ann_x_left,  y=90,  text="🛡️ Quality Traps",  showarrow=False,
                       font=dict(color=COLORS["gold"],  size=16), opacity=0.3)
    fig.add_annotation(x=_ann_x_right, y=-18, text="⚡ Growth Traps",   showarrow=False,
                       font=dict(color=COLORS["blue"],  size=16), opacity=0.3)
    fig.add_annotation(x=_ann_x_left,  y=-18, text="💀 Destroyers",     showarrow=False,
                       font=dict(color=COLORS["red"],   size=16), opacity=0.3)

    if highlight_stock:
        hl = plot_df[plot_df["name"] == highlight_stock]
        if not hl.empty:
            fig.add_trace(go.Scatter(
                x=hl["Growth_X"], y=hl["Moat_Y"],
                mode="markers+text",
                marker=dict(color="white", size=15, line=dict(color="black", width=2)),
                # No emoji in the text: 🎯 renders as a filled circular glyph indistinguishable
                # from a plotted marker, so one stock appeared as two dots — a phantom point
                # beside the real one. The white marker already marks the position.
                text=[_esc(highlight_stock)],
                textposition="top center",
                name="Selected Stock",
                showlegend=False,
            ))

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text_primary"]),
        margin=dict(l=0, r=0, t=30, b=0), height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        # px.scatter titles the legend with the COLUMN it colours by, so the chart printed the
        # literal string "moat_growth_quad" above the swatches.
        legend_title_text="Quadrant",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=COLORS["border"],
                     zeroline=True, zerolinewidth=2, zerolinecolor=COLORS["border"],
                     range=[-50, x_max])
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=COLORS["border"],
                     zeroline=True, zerolinewidth=2, zerolinecolor=COLORS["border"],
                     range=[-25, 105])
    st.plotly_chart(fig, width="stretch")

    # WHERE DOES *THIS* STOCK SIT — the question a single-stock tearsheet is actually asking.
    # The scatter spends all its resolution on the other ~2,037 dots, and the quadrant label is
    # already a pill elsewhere on the tearsheet, so the chart's unique job is placing this stock
    # in the distribution. It also stays truthful where the quadrant label cannot: 51.6% of the
    # universe sits within ±5pp of a 15/15 dividing line and the median stock is at (11.9, 12.9),
    # essentially ON the crossing — so for half the universe the categorical verdict turns on
    # noise, while a percentile does not. Ranked against the PLOTTED frame, so the numbers always
    # describe exactly what is drawn above.
    if highlight_stock:
        _me = plot_df[plot_df["name"] == highlight_stock]
        if not _me.empty:
            _g, _m = float(_me["Growth_X"].iloc[0]), float(_me["Moat_Y"].iloc[0])
            _gp = float((plot_df["Growth_X"] < _g).mean() * 100.0)
            _mp = float((plot_df["Moat_Y"] < _m).mean() * 100.0)

            def _ordinal(n: int) -> str:
                """1st/2nd/3rd/4th — a hardcoded 'th' printed '93th percentile'.
                The teens are the trap: 11/12/13 take 'th', not st/nd/rd."""
                return "th" if 11 <= (n % 100) <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

            def _pct_cell(lbl, val, pct, dur_html=""):
                # Colour from the percentile the cell PRINTS, not the raw one: pct 74.6 renders
                # "75th percentile" and must not sit in a different band from a pct of 75.0 that
                # renders the identical text. _p is that printed figure.
                _p = int(_shown(pct, 0))
                _c = (COLORS["green"] if _p >= 75 else
                      COLORS["gold"] if _p >= 40 else COLORS["red"])
                return (
                    f'<div style="flex:1;min-width:180px;">'
                    f'<span style="font-size:0.56rem;font-weight:700;color:{COLORS["text_muted"]};'
                    f'text-transform:uppercase;letter-spacing:0.7px;">{_esc(lbl)}</span><br>'
                    f'<span style="font-size:1.0rem;font-weight:900;color:{_c};">{val:.1f}%</span>'
                    f'<span style="font-size:0.72rem;font-weight:700;color:{COLORS["text_secondary"]};'
                    f'margin-left:7px;">{_p}{_ordinal(_p)} percentile</span>'
                    f'{dur_html}</div>'
                )

            # ── LONGEVITY: the 22nd WCS's other half ──────────────────────────────────
            # "Longevity of Moat can be measured as CAP (Competitive Advantage Period), and
            # longevity of earnings growth as GAP." Exhibit 5 says WHERE a company sits; CAP/GAP
            # say HOW LONG it has stayed there. This caption used to disclaim CAP/GAP outright
            # while the engine computed both at 100% coverage — they were only reachable in the
            # All Data raw dump, two tabs away.
            #
            # NAMING HAZARD (the reason this is worded so carefully): cap_years_proxy and
            # gap_years_proxy are NOT years despite their names. They COUNT WINDOWS clearing the
            # hurdle — CAP over 5 ROCE lookbacks (10y/7y/5y/prior/current), GAP over 3 PAT-growth
            # lookbacks (10y/5y/3y). A CAP of 5 therefore spans TEN years, not five, so rendering
            # "5 years" would be flatly wrong. Say windows, and name them in the tooltip.
            def _dur_html(kind, n, total, hurdle, tip):
                # HURDLE IS A PARAMETER, not the literal "15%" this printed until 2026-08-26.
                # CAP and GAP do NOT share a hurdle: CAP tests >= COST_OF_EQUITY (config: 12.0),
                # GAP tests >= 15.0 (hardcoded, and book-correct — the 22nd WCS's benchmark PAT
                # growth rate). Labelling both "≥15%" overstated CAP's bar, and 718 of 2,117
                # stocks (33.9%) have a DIFFERENT CAP count at 15 than at 12. Rendering the number
                # the engine actually uses means the label can never drift from the constant again.
                if n is None:
                    return ""
                _dc = (COLORS["green"] if n >= total else
                       COLORS["gold"] if n > 0 else COLORS["text_muted"])
                return (
                    f'<br><span style="font-size:0.62rem;font-weight:700;color:{_dc};">'
                    f'{_esc(kind)} {int(n)}/{total}</span>'
                    f'<span style="font-size:0.62rem;color:{COLORS["text_muted"]};margin-left:5px;">'
                    f'windows ≥{hurdle:g}%</span>{help_chip(kind + " longevity", tip)}'
                )

            def _dur_val(col):
                # Read from the highlighted ROW of the plotted frame — this renderer receives the
                # universe plus a name, not a stock Series (the cockpit's `stock` does not exist here).
                if col not in _me.columns:
                    return None
                v = _me[col].iloc[0]
                return None if pd.isna(v) else int(v)

            _cap_n = _dur_val("cap_years_proxy")
            _gap_n = _dur_val("gap_years_proxy")
            _shown_longevity = (_cap_n is not None) or (_gap_n is not None)
            _cap_html = _dur_html(
                "CAP", _cap_n, 5, COST_OF_EQUITY,
                "Competitive Advantage Period — the 22nd WCS's measure of MOAT longevity: the "
                "time a company earns above its cost of capital. The study defines it as the "
                "SUCCESSIVE years of RoE above 15%, and found a median CAP of 9-11 years among "
                "its wealth creators, with 171 of 223 clearing 5 years. This is a PROXY on two "
                "counts, and both matter: it counts WINDOWS not a consecutive run — how many of "
                "five ROCE lookbacks (10-year median, 7-year, 5-year, prior year, current) clear "
                f"the hurdle — and the hurdle here is this engine's cost of equity "
                f"({COST_OF_EQUITY:g}%, config.COST_OF_EQUITY), NOT the study's 15%. Those differ, "
                "so a 5/5 spans a decade of evidence above 12% rather than an unbroken streak "
                "above 15%.")
            _gap_html = _dur_html(
                "GAP", _gap_n, 3, 15.0,   # hardcoded in the engine, and book-correct
                "Growth Advantage Period — the 22nd WCS's measure of GROWTH longevity: the time "
                "profits outgrow the benchmark. The study's hurdle is 15% PAT growth, deliberately "
                "the same number as the cost of equity. Counted here as WINDOWS, not years: how "
                "many of three PAT-growth lookbacks (10-year, 5-year, 3-year) clear 15%. Read it "
                "against CAP — \"in most cases, CAP is the foundation for sustained GAP\", and "
                "\"end of CAP is a certain cause for end of GAP\". A full CAP with a slipping GAP "
                "is the shape to notice: the moat still holds, the compounding has stopped.")

            st.markdown(
                f'<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;'
                f'background:{COLORS["bg_secondary"]};border:1px solid {COLORS["border"]};'
                f'border-radius:10px;padding:9px 14px;margin-top:2px;">'
                f'<div style="font-size:0.72rem;font-weight:800;color:{COLORS["text_primary"]};'
                f'min-width:150px;">🎯 {_esc(highlight_stock)}</div>'
                f'{_pct_cell("Moat · ROCE", _m, _mp, _cap_html)}'
                f'{_pct_cell("Growth · PAT CAGR", _g, _gp, _gap_html)}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Silent omission is the one thing this app refuses to do: name what is not on the chart.
    _omitted = int(len(df) - len(plot_df))
    _notes = []
    if _omitted:
        _notes.append(f"{_omitted:,} of {len(df):,} stocks are not plotted — no ROCE or no growth "
                      f"figure to place them.")
    if _beyond:
        _notes.append(f"{_beyond:,} sit beyond {x_max:.0f}% growth (up to {_x_true:,.0f}%) and are "
                      f"drawn at the right edge; the view is clipped so the crowded core stays "
                      f"readable, never to drop a stock.")
    _note_html = (" ".join(_notes) + " ") if _notes else ""
    # Only claim CAP/GAP when they were actually rendered — on a frame without those columns
    # (an old snapshot, a minimal fixture) the caption must not point at an absent strip.
    _longev_html = (
        "The quadrant is position <b>today</b>; CAP/GAP above it are the study's longevity "
        "measures — how many lookback windows have held the line, not how many years."
        if locals().get("_shown_longevity") else
        "Quadrant = position <b>today</b>; longevity (CAP/GAP duration) is a separate measure."
    )
    st.markdown(
        f"<div style='font-size:0.6rem;color:{COLORS['text_muted']};margin-top:6px;margin-bottom:10px;'>"
        f"{_note_html}"
        f"Moat (5Y-median ROCE) × Growth (5Y PAT CAGR), split at MOSL's 15% cost-of-equity line — "
        f"22nd WCS, Exhibit 5. {_longev_html}</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# EP POWER CURVE MODULE — 28th WCS
# ═══════════════════════════════════════════════════════════════

def render_ep_power_curve_module(stock: pd.Series):
    """
    Economic Profit Power Curve card (28th WCS).
    Displays the stock's quintile position, EP velocity, and the Hockey-Stick
    Breakthrough badge when ep_hockey_stick_breakout fires (Q2/Q3 ascending
    the curve with institutional volume confirmation).
    """
    st.markdown("<div class='sec-head'>📈 Economic Profit Power Curve (28th WCS)</div>",
                unsafe_allow_html=True)

    # EP is UNDEFINED when equity is gone or RoE is unreported — the engine leaves it NaN.
    # Never render that as ₹0 / Value Trap; a data hole is not a verdict.
    ep_known     = pd.notna(stock.get("economic_profit"))
    ep_vel_known = pd.notna(stock.get("economic_profit_velocity",
                                      stock.get("economic_profit_delta")))
    ep_val      = _g(stock, "economic_profit",          0)
    ep_vel      = _g(stock, "economic_profit_velocity",
                    _g(stock, "economic_profit_delta",  0))
    ep_curve    = stock.get("ep_power_curve", "") or "❔ Not reported"
    ep_q        = stock.get("ep_quintile",    None)
    hs_breakout = int(_g(stock, "ep_hockey_stick_breakout", 0))
    ep_positive = int(_g(stock, "economic_profit_positive",  0))

    # ── Hockey-Stick Breakthrough Banner ──────────────────────────────────
    if hs_breakout:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0b2214,#0e3320);
                    border:2px solid {COLORS['green']}; border-radius:14px;
                    padding:16px 22px; margin-bottom:14px; text-align:center;
                    box-shadow:0 0 22px rgba(63,185,80,0.40);">
            <div style="font-size:1.7rem; margin-bottom:4px;">🚀</div>
            <div style="font-size:1.05rem; font-weight:900; color:{COLORS['green']};
                        letter-spacing:1.2px;">
                HOCKEY-STICK EP BREAKTHROUGH
            </div>
            <div style="font-size:0.75rem; color:{COLORS['text_muted']}; margin-top:6px;">
                Q2/Q3 company ascending the Economic Profit Power Curve with institutional
                volume confirmation — 28th WCS structural alpha inflection signal
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Quintile Position Bar ─────────────────────────────────────────────
    _Q_LABELS = {
        1: "Alpha Creators",
        2: "Value Creators",
        3: "Emerging",
        4: "Destroyers",
        5: "Capital Destroyers",
    }
    _Q_COLORS = {
        1: COLORS["green"],
        2: "#6ec97f",
        3: COLORS["gold"],
        4: COLORS["orange"],
        5: COLORS["red"],
    }

    ep_q_int = None
    if ep_q is not None:
        try:
            ep_q_int = int(float(ep_q))
        except (TypeError, ValueError):
            ep_q_int = None

    segs_html = ""
    for q in range(1, 6):
        is_cur  = (ep_q_int == q)
        opacity = "1.0" if is_cur else "0.38"
        border  = (f"box-shadow:0 0 0 3px #0d1117,0 0 0 5px {_Q_COLORS[q]};"
                   "font-size:0.82rem;") if is_cur else "font-size:0.7rem;"
        segs_html += f"""
            <div style="flex:1;height:36px;background:{_Q_COLORS[q]};opacity:{opacity};
                        border-radius:8px;display:flex;align-items:center;
                        justify-content:center;font-weight:800;color:#0d1117;{border}">
                Q{q}
            </div>"""

    q_label = _Q_LABELS.get(ep_q_int, "Unknown") if ep_q_int else "Not ranked"
    q_color = _Q_COLORS.get(ep_q_int, COLORS["gold"]) if ep_q_int else COLORS["gold"]

    st.markdown(f"""
    <div style="margin:8px 0 12px 0;">
        <div style="display:flex;gap:6px;margin-bottom:6px;">{segs_html}</div>
        <div style="text-align:center;font-size:0.72rem;color:{COLORS['text_muted']};">
            Q1 = Top 20% Economic Profit Earners &nbsp;·&nbsp;
            Q2/Q3 = Breakout Launch Zone (the study's best starting point) &nbsp;·&nbsp;
            Q5 = Bottom 20% Capital Destroyers
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── EP Metrics Strip — packed HTML flex (no st.columns/st.metric padding) ──
    # EP Velocity prints WHOLE CRORE, so sign, colour and the Ascending/Descending word are all
    # decided on that rounded figure — +0.4 and -0.4 both render "0 Cr" and must not wear opposite
    # colours (99 stocks did). A value that rounds to zero at the displayed scale is neither
    # ascending nor descending, so it reads FLAT rather than being forced onto a side: this
    # threshold is a SIGN, and absent movement is a real third state, not a band edge.
    # `_vel_shown != 0` normalises -0.0 (round(-0.4) is -0.0, which formats as the ugly "-0").
    _vel_shown = _shown(ep_vel, 0) if ep_vel_known else 0.0
    if _vel_shown == 0:
        _vel_shown = 0.0
    # The display's own "{:,.0f}" already carries the minus for negatives — only "+" is added here.
    vel_sign = "+" if _vel_shown > 0 else ""
    ep_clr  = COLORS["green"] if ep_positive else COLORS["red"]
    vel_clr = (COLORS["green"] if _vel_shown > 0 else
               COLORS["red"]   if _vel_shown < 0 else COLORS["text_secondary"])

    def _ep_metric(label: str, value: str, sub: str, val_clr: str, sub_clr: str,
                   tip: str = "") -> str:
        # Value type-size follows value LENGTH. Three of these cards hold short numbers
        # ("₹4 Cr", "Q2") but EP Trajectory holds a text label, and at a shared 1.3rem the
        # 25-character "➖ EP Positive, Not Rising" ran to the card edge and visually outweighed
        # the measurements beside it — a label shouting louder than the numbers it annotates.
        # Scaling by length keeps every card balanced and handles all four curve labels
        # ("🚀 Hockey Stick", "📈 Improving", "📉 Value Trap") without a per-card special case.
        _vl = len(value)
        _fs = "1.3rem" if _vl <= 14 else "1.02rem" if _vl <= 20 else "0.88rem"
        return (
            f'<div style="flex:1;min-width:120px;background:{COLORS["bg_secondary"]};'
            f'border:1px solid {COLORS["border"]};border-radius:10px;padding:10px 14px;">'
            f'<div style="font-size:0.56rem;font-weight:700;color:{COLORS["text_muted"]};'
            f'text-transform:uppercase;letter-spacing:0.7px;">{_esc(label)}'
            f'{help_chip(label, tip)}</div>'
            f'<div style="font-size:{_fs};font-weight:900;color:{val_clr};'
            f'line-height:1.15;margin-top:3px;white-space:nowrap;">{_esc(value)}</div>'
            f'<div style="font-size:0.62rem;font-weight:600;color:{sub_clr};'
            f'margin-top:2px;white-space:nowrap;">{_esc(sub)}</div>'
            f'</div>'
        )

    # ── 💹 Wealth Tier card inputs — computed HERE, before the strip expression (the four EP
    # cards are assembled inside ONE concatenated expression; an earlier edit that inserted
    # statements mid-expression broke the module and was reverted). This card closes the loop the
    # strip opens: its four siblings ARE the tier's raw inputs, and the normalized EP% / Vel% —
    # the scale-free versions (EP ÷ reserves = ROE − CoE, exactly) that make a ₹200 Cr and a
    # ₹2,000 Cr business comparable — are displayed on NO other per-stock surface.
    _wt_val = str(stock.get("wealth_tier", "") or "")
    _wt_warn = " ⚠" if int(stock.get("wealth_warn", 0) or 0) == 1 else ""
    _wt_ep, _wt_vel = stock.get("wealth_ep_pct"), stock.get("wealth_vel_pct")
    if _wt_val in ("BUY★", "BUY"):
        _wt_clr = COLORS["green"]
    elif _wt_val in ("WATCH★", "WATCH"):
        _wt_clr = COLORS["gold"]
    elif _wt_val == "AVOID":
        _wt_clr = COLORS["text_secondary"]
    else:
        _wt_val, _wt_clr = "N/A", COLORS["text_muted"]
    _wt_sub = ("EP% " + (f"{float(_wt_ep):+.1f}" if pd.notna(_wt_ep) else "—")
               + " · Vel% " + (f"{float(_wt_vel):+.1f}" if pd.notna(_wt_vel) else "—"))

    ep_strip = (
        _ep_metric("Economic Profit",
                   f"₹{ep_val:,.0f} Cr" if ep_known else "—",
                   ("EP Positive ✅" if ep_positive else "EP Negative ❌")
                       if ep_known else "Equity or RoE not reported",
                   ep_clr if ep_known else COLORS["text_muted"],
                   ep_clr if ep_known else COLORS["text_muted"],
                   tip="Profit left after charging for ALL capital, equity included: "
                       "equity (reserves) × (RoE − cost of equity). Accounting profit can be healthy "
                       "while economic profit is negative — that is a company earning less than "
                       "its shareholders' money costs.") +
        _ep_metric("EP Velocity (YoY)",
                   f"{vel_sign}₹{_vel_shown:,.0f} Cr" if ep_vel_known else "—",
                   ("Ascending ↑" if _vel_shown > 0 else
                    "Descending ↓" if _vel_shown < 0 else "Flat — under ₹1 Cr")
                       if ep_vel_known else "No prior-year equity",
                   vel_clr if ep_vel_known else COLORS["text_muted"],
                   vel_clr if ep_vel_known else COLORS["text_muted"],
                   tip="Change in economic profit versus a year ago, in rupees. The 28th WCS's "
                       "finding is that DIRECTION matters more than level: a company climbing "
                       "from a low base has historically outperformed a high earner sliding "
                       "backwards.") +
        _ep_metric("Quintile Position", f"Q{ep_q_int}" if ep_q_int else "N/A",
                   q_label, q_color, q_color,
                   tip="Where this company sits when the whole universe is ranked by ABSOLUTE "
                       "economic profit. Q1 = top 20% of earners, Q5 = the deepest destroyers. "
                       "The 28th WCS's best starting point is Q2/Q3, not Q1 — the launch zone, "
                       "where there is still room to climb.") +
        _ep_metric("EP Trajectory", ep_curve, "28th WCS curve position",
                   COLORS["blue"], COLORS["text_muted"],
                   tip="A 2×2 of economic-profit LEVEL against its DIRECTION. "
                       "🚀 Hockey Stick = positive AND rising (the best state) · "
                       "➖ EP Positive, Not Rising = positive but not growing · "
                       "📈 Improving = still negative, but turning up · "
                       "📉 Value Trap = negative and not improving. "
                       "Note: the 🏒 EP Hockey Stick FRAMEWORK is stricter — it also requires a "
                       "P/E of 20x or less (the 28th WCS's TEM-P: Trends/Endowment/Moves lift "
                       "economic profit, but the RETURN needs a cheap entry). 663 stocks show 🚀 "
                       "here; 268 pass the framework. So a stock can be a Hockey Stick on this "
                       "card and still have no pill in the Frameworks tab — that is the price "
                       "gate, not a disagreement.") +
        _ep_metric("Wealth Tier", f"{_wt_val}{_wt_warn}", _wt_sub, _wt_clr, COLORS["text_muted"],
                   tip="The wealth-engine tier — three clocks, nothing else: EP% (excess return, "
                       "ROE − cost of equity), Vel% (this year's change in it), and the 5-year "
                       "margin tau. The four cards to the left are its raw inputs; EP% and Vel% "
                       "here are those figures ÷ reserves (the same equity base EP is computed "
                       "on), so businesses of any size compare "
                       "fairly. Price-blind and forensics-blind: the ⚠ marks 8+ red flags or a "
                       "Schilit fail and never alters the tier. A description, not a "
                       "recommendation — see the 💹 Wealth tab for the whole universe.")
    )
    st.markdown(
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:4px;">{ep_strip}</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# BRUISED BLUE CHIP BADGE — 29th WCS
# ═══════════════════════════════════════════════════════════════

def render_bruised_blue_chip_badge(stock: pd.Series):
    """
    Bruised Blue Chip premium franchise badge (29th WCS).
    Renders ONLY when bruised_blue_chip_29 (Agent 9 large-cap elite) or
    bruised_blue_chip (fallen-quality signal) is triggered.
    No-op when neither is active — keeps the tearsheet clean.
    """
    bbc29  = int(_g(stock, "bruised_blue_chip_29", 0))
    bbc_og = int(_g(stock, "bruised_blue_chip",    0))

    if not bbc29 and not bbc_og:
        return

    # ── Agent 9 variant: Large-Cap Elite blue chip + P/B ≤ 2.0 (primary badge) ──
    if bbc29:
        mcap     = _g(stock, "market_cap",    0)
        roe_10y  = _g(stock, "roe_med_10y",   0)   # engine gates on ROE (not ROCE) — blue chips incl. banks
        pb       = _g(stock, "pb_ratio",      0)
        dist52   = _g(stock, "dist_52wh",     0)
        pe_disc  = _g(stock, "pe_discount",   0)
        # "Bruised" proxy — show whichever condition actually gated (the card's previously-missing thesis)
        _bruise_bits = []
        if dist52 > 25:   _bruise_bits.append(f"{dist52:.0f}% off 52-wk high")
        if pe_disc >= 25: _bruise_bits.append(f"PE {pe_disc:.0f}% below history")
        _bruise_txt = " · ".join(_bruise_bits) if _bruise_bits else "deep historical drawdown"
        # ROE ≥ 20 only gates the top-250 branch (top-50 has NO floor) → conditional, never a false ✅
        if roe_10y >= 20:
            _roe_html = (f'<span style="font-size:0.78rem;color:{COLORS["green"]};">'
                         f'✅ ROE 10Y ≥ 20% &nbsp;({roe_10y:.1f}%)</span>')
        else:
            _roe_html = (f'<span style="font-size:0.78rem;color:{COLORS["text_muted"]};">'
                         f'ROE 10Y: {roe_10y:.1f}%</span>')

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#091624,#0c2040);
                    border:2px solid {COLORS['blue']};border-radius:16px;
                    padding:18px 22px;margin:10px 0;
                    box-shadow:0 0 26px rgba(88,166,255,0.22);">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
                <div style="font-size:1.8rem;">💙</div>
                <div>
                    <div style="font-size:1.05rem;font-weight:900;color:{COLORS['blue']};">
                        BRUISED BLUE CHIP — 29th WCS
                    </div>
                    <div style="font-size:0.75rem;color:{COLORS['text_muted']};">
                        Elite franchise at a historical valuation floor
                    </div>
                </div>
            </div>
            <div style="display:flex;gap:14px;flex-wrap:wrap;">
                <span style="font-size:0.78rem;color:{COLORS['green']};">
                    ✅ Top-50 / Top-250 Quality MCap &nbsp;(₹{mcap:,.0f} Cr)
                </span>
                <span style="font-size:0.78rem;color:{COLORS['green']};">
                    ✅ Bruised — {_esc(_bruise_txt)}
                </span>
                <span style="font-size:0.78rem;color:{COLORS['green']};">
                    ✅ P/B ≤ 2.0× &nbsp;({pb:.2f}×)
                </span>
                {_roe_html}
            </div>
            <div style="margin-top:10px;font-size:0.75rem;color:{COLORS['text_secondary']};
                        border-top:1px solid rgba(88,166,255,0.2);padding-top:8px;">
                Premium franchise at a historical valuation floor — P/B {pb:.2f}×, {_esc(_bruise_txt)} —
                asymmetric risk/reward per MOSL 29th Annual Wealth Creation Study.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Original fallen-quality signal (secondary) ──
    if bbc_og:
        dist_52wh = _g(stock, "dist_52wh",         0)
        d32       = _g(stock, "d32_pe_vs_median",   0)
        roce_5y   = _g(stock, "roce_med_5y",        0)
        st.markdown(f"""
        <div style="background:rgba(88,166,255,0.07);
                    border:1px solid rgba(88,166,255,0.3);
                    border-radius:10px;padding:10px 16px;margin:6px 0;">
            <span style="font-weight:700;color:{COLORS['blue']};">💙 Fallen Quality Signal:</span>
            <span style="font-size:0.8rem;color:{COLORS['text_secondary']};">
                {dist_52wh:.0f}% off 52W high ·
                {abs(d32):.0f}% below 10Y median PE ·
                ROCE 5Y: {roce_5y:.1f}%
            </span>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MULTI-TRILLION TIPPING POINT CARD — 30th WCS
# ═══════════════════════════════════════════════════════════════

def render_multitrillioncap_card(stock: pd.Series):
    """
    Multi-Trillion Economy Compounding Tipping Point card (30th WCS).
    Renders the full signal grid when multitrillioncap_tipping_point == 1.
    Renders a lightweight 'Sunrise Sector' note for sector_tailwind == 1 only.
    Silent when neither applies.
    """
    mtp        = int(_g(stock, "multitrillioncap_tipping_point", 0))
    sector_tw  = int(_g(stock, "sector_tailwind",                0))

    if not mtp and not sector_tw:
        return

    # ── In sunrise sector but not at tipping point yet ──
    if not mtp and sector_tw:
        sector_nm = _esc(stock.get("sector", ""))
        st.markdown(f"""
        <div style="background:rgba(139,92,246,0.07);
                    border:1px dashed rgba(139,92,246,0.4);
                    border-radius:10px;padding:10px 14px;margin:6px 0;">
            <span style="font-size:0.8rem;color:{COLORS['purple']};">🌐 Sunrise Sector</span>
            <span style="font-size:0.75rem;color:{COLORS['text_muted']};margin-left:8px;">
                {sector_nm} — structural tailwind sector (30th WCS Multi-Trillion opportunity).
                Tipping point signals not yet fully triggered.
            </span>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Full tipping point — render the complete signal grid ──
    vol_ratio  = _g(stock, "vol_ratio",  0)
    pat_gr_3y  = _g(stock, "pat_gr_3y",  0)
    q_pat_yoy  = _g(stock, "q_pat_yoy",  0)
    dist_52wh  = _g(stock, "dist_52wh",  999)
    sector_nm  = _esc(stock.get("sector",   ""))
    industry_nm= _esc(stock.get("industry", ""))

    vol_ok   = vol_ratio >= 1.5
    earn_ok  = q_pat_yoy > 25 or pat_gr_3y > 25
    break_ok = dist_52wh <= 15

    def _sig_card(label: str, fired: bool, detail: str) -> str:
        col = COLORS["green"] if fired else COLORS["text_muted"]
        bdr = "rgba(139,92,246,0.40)" if fired else "rgba(139,92,246,0.15)"
        ico = "✅" if fired else "⬜"
        return f"""
        <div style="flex:1;min-width:130px;background:rgba(139,92,246,0.09);
                    border:1px solid {bdr};border-radius:8px;
                    padding:8px 12px;text-align:center;">
            <div style="font-size:1.1rem;">{ico}</div>
            <div style="font-size:0.72rem;font-weight:700;color:{col};margin-top:2px;">{label}</div>
            <div style="font-size:0.66rem;color:{COLORS['text_muted']};">{detail}</div>
        </div>"""

    sigs_html = (
        _sig_card("Volume Surge",          vol_ok,   f"{vol_ratio:.1f}× 20D SMA") +
        _sig_card("Earnings Acceleration", earn_ok,  f"PAT 3Y CAGR {pat_gr_3y:.0f}%") +
        _sig_card("Near Breakout",         break_ok, f"{dist_52wh:.0f}% from 52W high")
    )

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#110830,#1c0d46);
                border:2px solid {COLORS['purple']};border-radius:16px;
                padding:18px 22px;margin:10px 0;
                box-shadow:0 0 26px rgba(139,92,246,0.28);">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
            <div style="font-size:1.6rem;">🌐</div>
            <div>
                <div style="font-size:1.0rem;font-weight:900;color:{COLORS['purple']};">
                    MULTI-TRILLION COMPOUNDING TIPPING POINT — 30th WCS
                </div>
                <div style="font-size:0.74rem;color:{COLORS['text_muted']};">
                    {sector_nm} · {industry_nm} · Structural tailwind sector at critical velocity
                </div>
            </div>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">{sigs_html}</div>
        <div style="margin-top:10px;font-size:0.74rem;color:{COLORS['text_muted']};
                    border-top:1px solid rgba(139,92,246,0.2);padding-top:8px;">
            {_esc(sector_nm)} screens as a structural-tailwind sector — MOSL's 30th Study projects
            India's high-growth sectors to 3× their market cap by 2030. Stocks reaching volume +
            earnings + breakout confluence are the structural compounders at tipping velocity.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# FORENSIC FRAUD PERIMETER — WCS 24 / FORENSIC_MAX_FLAGS-flag cascade (28, self-updating)
# ═══════════════════════════════════════════════════════════════

def _get_flag_context(stock: pd.Series, rf_col: str) -> str:
    """Return stock-specific metric values for a fired forensic flag.
    Matches design: 'cfo_to_pat: 54.2%  ·  threshold: ≥70%' beneath each flag title.
    Returns empty string when data is unavailable (flag row still renders without sub).
    """
    def _v(col, fmt):
        raw = stock.get(col)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return None
        try:
            return fmt.format(float(raw))
        except Exception:
            return None

    if rf_col == "rf_low_cfo_pat":
        v = _v("cfo_to_pat", "{:.1f}%")
        return f"CFO/PAT: {v}  ·  threshold: <70%" if v else ""
    if rf_col == "rf_high_receivables":
        # STALE THRESHOLD (fixed 2026-08-25): this printed a fixed ">75d (products) / >120d
        # (services)" — the binary rule the engine REPLACED in BUG #11 with an per-sector lookup
        # (_SECTOR_DSO_THRESHOLDS). Seven distinct gates are live on firing rows (45/70/75/80/
        # 90/110/120d), so the sentence was wrong for 14 of 664 firing stocks outright — JTL
        # Industries, Steel, gate 70d, showed "DSO: 71d · threshold: >75d", a flag apparently
        # contradicting itself — and uninformative for the rest, since it never named the gate
        # actually applied. Read the engine's table so the two can never drift again.
        _dso = stock.get("days_receivable")
        if pd.isna(_dso):
            return ""
        _sec = str(stock.get("sector") or "Unknown")
        _gate = _SECTOR_DSO_THRESHOLDS.get(_sec, _DEFAULT_DSO_THRESHOLD)
        # Name the sector only when it HAS a calibrated norm. On the fallback, "(Infrastructure
        # Developers & Operators default)" ran to 77 chars, named a sector that is precisely the
        # one NOT calibrated (so the name carries no information), risked clipping in this narrow
        # column, and pushed a raw "&" into the inline HTML. "(default)" says the same thing.
        _band = (f"({_sec})" if _sec in _SECTOR_DSO_THRESHOLDS else "(sector default)")
        return f"DSO: {float(_dso):.0f}d  ·  threshold: >{_gate:.0f}d {_band}"
    if rf_col == "rf_inventory_bloat":
        # WRONG METRIC (fixed 2026-08-24): showed rev_gr_yoy while the engine tests
        # inv_vs_rev_gap > 10 (median 31.7pp on firing rows). It also rendered a broken "+-4.2%"
        # on the 189 firing stocks whose revenue growth is NEGATIVE, because the format hardcoded
        # a "+" prefix. The gap IS the tested quantity, so show it.
        _gap_i = stock.get("inv_vs_rev_gap")
        if pd.isna(_gap_i):
            return ""
        return (f"inventory outgrew revenue by {float(_gap_i):.1f}pp"
                f"  ·  threshold: >10pp")
    if rf_col == "rf_rising_debt":
        de = _v("debt_to_equity", "{:.2f}")
        de1 = _v("debt_to_equity_1yb", "{:.2f}")
        parts = []
        if de:  parts.append(f"D/E current: {de}")
        if de1: parts.append(f"prior year: {de1}")
        return "  ·  ".join(parts)
    if rf_col == "rf_pledge_elevated":
        v = _v("pledged_percentage", "{:.1f}%")
        return f"pledge: {v}  ·  threshold: >10%" if v else ""
    if rf_col == "rf_negative_fcf":
        ocf = _v("operating_cash_flow", "₹{:,.0f} Cr")
        fcf = _v("free_cash_flow",      "₹{:,.0f} Cr")
        parts = []
        if ocf: parts.append(f"OCF: {ocf}")
        if fcf: parts.append(f"FCF: {fcf}")
        return "  ·  ".join(parts)
    if rf_col == "rf_margin_squeeze":
        rg = _v("rev_gr_yoy", "{:.1f}%")
        pg = _v("pat_gr_yoy", "{:.1f}%")
        parts = []
        if rg: parts.append(f"rev_gr: +{rg}")
        if pg: parts.append(f"pat_gr: {pg}")
        return "  ·  ".join(parts)
    if rf_col == "rf_ssgr_deficit":
        # WRONG METRIC (fixed 2026-08-24): this printed pat_gr_yoy — PROFIT growth year-over-year
        # — while the flag fires on ssgr_cushion, which the engine builds from REVENUE growth 5Y.
        # Sarda rendered "actual_gr: 9.4% · SSGR: 13.5%" beneath the title "Actual growth exceeds
        # SSGR by >5%": evidence that appears to REFUTE its own accusation, making a correct
        # forensic flag look like a false positive. The true pair is 20.9% vs 13.5%.
        # Derived from the cushion (actual = ssgr − cushion) so the evidence is self-consistent
        # with the firing condition BY CONSTRUCTION and can never drift from it again.
        _ss, _cu = stock.get("ssgr"), stock.get("ssgr_cushion")
        if pd.isna(_ss) or pd.isna(_cu):
            return ""
        return (f"growth: {float(_ss) - float(_cu):.1f}%  ·  SSGR: {float(_ss):.1f}%"
                f"  ·  exceeds by {abs(float(_cu)):.1f}%")
    if rf_col == "rf_opm_volatile":
        # WRONG METRIC (fixed 2026-08-24, same class as rf_ssgr_deficit): this printed the CURRENT
        # opm, but the engine tests |opm_1yb − opm_med_5y| / opm_med_5y > 0.30 (annual vs annual;
        # the quarterly fallback was removed as seasonally distorted). On 52 of 474 firing rows
        # (11.0%) the displayed pair does NOT breach 30%, so the evidence appeared to refute its
        # own flag. Now shows the compared annual figure and the deviation the flag measured.
        _o5 = stock.get("opm_med_5y")
        _o1 = stock.get("opm_1yb")
        if pd.isna(_o5) or float(_o5) <= 0:
            return ""
        _cmp = float(_o1) if pd.notna(_o1) else float(_o5)
        _dev = abs(_cmp - float(_o5)) / float(_o5) * 100.0
        return (f"OPM {_cmp:.1f}% vs 5Y median {float(_o5):.1f}%"
                f"  ·  deviation {_dev:.1f}%  ·  threshold: >30%")
    if rf_col == "rf_high_accruals":
        # engine (forensic_engine rf_high_accruals): (PAT − OCF) / avg_total_assets > 0.05, with the
        # PAT of the CASH-FLOW year — the same figure the flag was computed from.
        _pat_a = stock.get("pat_cf_year", stock.get("pat")); _ocf_a = stock.get("operating_cash_flow")
        # The engine's own denominator (KI-8): the years that exist and are above 0, averaged only when
        # both do, so a stock whose assets this year are blank shows the number its flag was judged on.
        _yrs = [float(v) for v in (stock.get("total_assets"), stock.get("total_assets_1yb"))
                if pd.notna(v) and float(v) > 0]
        if pd.isna(_pat_a) or pd.isna(_ocf_a) or not _yrs:
            return ""
        _avg_ta = sum(_yrs) / len(_yrs)
        _acc = (float(_pat_a) - float(_ocf_a)) / _avg_ta * 100.0
        _yr_a = (" (last year's PAT — the latest cash flow is a year older)"
                 if stock.get("cf_year_lag") == 1 else "")
        return (f"accruals: {_acc:.1f}% of assets  ·  threshold: >5%"
                f"  ·  PAT ₹{float(_pat_a):,.0f}cr vs OCF ₹{float(_ocf_a):,.0f}cr{_yr_a}")
    if rf_col == "rf_low_fcf_ebitda":
        # engine (~L320): FCF / EBITDA < 0.30, only when EBITDA > 0
        # EBITDA of the CASH-FLOW year — the figure the flag divided by.
        _fcf_e, _ebd = stock.get("free_cash_flow"), stock.get("ebitda_cf_year", stock.get("ebitda"))
        if pd.isna(_fcf_e) or pd.isna(_ebd) or float(_ebd) <= 0:
            return ""
        _yr_e = (" (last year's EBITDA — the latest cash flow is a year older)"
                 if stock.get("cf_year_lag") == 1 else "")
        return (f"FCF/EBITDA: {100.0 * float(_fcf_e) / float(_ebd):.0f}%  ·  threshold: <30%"
                f"  ·  FCF ₹{float(_fcf_e):,.0f}cr vs EBITDA ₹{float(_ebd):,.0f}cr{_yr_e}")
    if rf_col == "rf_debt_ebitda_high":
        # engine (~L384): debt / EBITDA > 5.0, non-financials only
        _dbt, _ebd2 = stock.get("debt"), stock.get("ebitda")
        if pd.isna(_dbt) or pd.isna(_ebd2) or float(_ebd2) <= 0:
            return ""
        return (f"Debt/EBITDA: {float(_dbt) / float(_ebd2):.1f}×  ·  threshold: >5×"
                f"  ·  debt ₹{float(_dbt):,.0f}cr")
    if rf_col == "rf_cwip_bloat":
        # engine (~L409): CWIP share of assets grew >50% YoY
        _cw, _ta_c = stock.get("cwip"), stock.get("total_assets")
        _cw1, _ta1c = stock.get("cwip_1yb"), stock.get("total_assets_1yb")
        if any(pd.isna(v) for v in (_cw, _ta_c, _cw1, _ta1c)) or float(_ta_c) <= 0 or float(_ta1c) <= 0:
            return ""
        _now = 100.0 * float(_cw) / float(_ta_c)
        _then = 100.0 * float(_cw1) / float(_ta1c)
        if _then <= 0:
            return ""
        return (f"CWIP {_then:.1f}% → {_now:.1f}% of assets"
                f"  ·  +{100.0 * (_now / _then - 1.0):.0f}% YoY  ·  threshold: >+50%")
    if rf_col == "rf_dilution":
        # engine (~L260): dilution_flag >= 2 (Tier 2+ = >3% meaningful dilution).
        # The old handler returned "" whenever shares_gr_yoy was missing AND this flag's
        # description carries no fallback text — so the row rendered a title above a BLANK line
        # (626 firing stocks, user-reported). Tier is always present when the flag fires.
        _sh_gr = stock.get("shares_gr_yoy")
        _tier_d = int(_g(stock, "dilution_flag", 0))
        # Not "predatory" (2026-10-09): a bonus or rights issue below 1.5x lands in Tier 3 too, and the
        # data cannot tell them from a placement (docs/known-issues.md, the dilution entry). Nor from shares
        # held by an employee trust: today's count includes them, last year's (EPS basis) does not.
        _tier_txt = ("Tier 3 (>10% in a year: a placement, rights issue, bonus or shares held by an employee "
                     "trust that only this year's count includes — the data can't tell which)"
                     if _tier_d >= 3 else "Tier 2 (3–10%)")
        if pd.notna(_sh_gr):
            return f"share count grew: {float(_sh_gr):.1f}%  ·  {_tier_txt}  ·  threshold: >3%"
        return f"{_tier_txt} dilution  ·  threshold: >3% share-count growth"
    if rf_col == "rf_nfat_very_low":
        v = _v("nfat", "{:.2f}×")
        return f"NFAT: {v}  ·  threshold: <1.5×" if v else ""
    if rf_col == "rf_debt_ebitda_high":
        v = _v("debt_to_ebitda", "{:.1f}×")
        return f"Debt/EBITDA: {v}  ·  threshold: >5×" if v else ""
    if rf_col == "rf_tax_panic":
        v = _v("tax_rate_est", "{:.1f}%")
        return f"tax_rate: {v}  ·  threshold: <10% despite positive PAT" if v else ""
    if rf_col == "rf_fcf_to_cfo_low":
        v = _v("fcf_to_cfo_pct", "{:.1f}%")
        return f"FCF/CFO: {v}  ·  threshold: <15%" if v else ""
    if rf_col == "rf_low_fcf_ebitda":
        v = _v("fcf_to_ebitda_pct", "{:.1f}%")
        return f"FCF/EBITDA: {v}  ·  threshold: <30%" if v else ""
    if rf_col == "rf_high_cash_debt":
        # WRONG METRIC (fixed 2026-08-24): showed D/E, but the engine tests
        # cash > 0 AND debt > 0 AND cash > debt × 0.3 (data_engine ~L1690). D/E never determines
        # it — 443 of the 878 firing stocks have D/E < 0.1, so the row read "High cash + high
        # debt · D/E: 0.01", contradicting its own title (Kothari: cash ₹18cr vs debt ₹2cr).
        _cash_h, _debt_h = stock.get("cash_equivalents"), stock.get("debt")
        if pd.isna(_cash_h) or pd.isna(_debt_h) or float(_debt_h) <= 0:
            return ""
        return (f"cash ₹{float(_cash_h):,.0f}cr vs debt ₹{float(_debt_h):,.0f}cr"
                f"  ·  cash is {100.0 * float(_cash_h) / float(_debt_h):.0f}% of debt"
                f"  ·  threshold: >30%")
    if rf_col == "rf_receivables_bloat":
        # The engine tests (DSO − DSO_1yb) > sector-median expansion + 20 — an EXPANSION vs peers,
        # not an absolute level (that is rf_high_receivables). Show the expansion the flag measured.
        _d_now, _d_prev = stock.get("days_receivable"), stock.get("days_receivable_1yb")
        if pd.isna(_d_now) or pd.isna(_d_prev):
            return ""
        return (f"DSO {float(_d_now):.0f}d vs {float(_d_prev):.0f}d prior"
                f"  ·  expanded {float(_d_now) - float(_d_prev):+.0f}d"
                f"  ·  fires at >20d above the SECTOR median expansion")
    if rf_col == "rf_high_accruals":
        v = _v("accruals_to_assets", "{:.1f}%")
        return f"accruals/assets: {v}  ·  threshold: >5%" if v else ""
    if rf_col == "rf_ccc_worsening":
        # The engine tests ccc > ccc_1yb + 10 — a DELTA. Showing only the current value left the
        # reader unable to see the change the flag measured.
        _c_now, _c_prev = stock.get("ccc"), stock.get("ccc_1yb")
        if pd.isna(_c_now) or pd.isna(_c_prev):
            return ""
        return (f"CCC {float(_c_now):.0f}d vs {float(_c_prev):.0f}d prior"
                f"  ·  worsened {float(_c_now) - float(_c_prev):.0f}d"
                f"  ·  threshold: >10d")
    if rf_col == "rf_expense_rising":
        v = _v("expense_ratio", "{:.1f}%")
        return f"expense_ratio: {v}  ·  rose >3pp" if v else ""
    if rf_col == "rf_dilution":
        shares_gr = _v("shares_gr_yoy", "{:.1f}%")
        return f"share count grew: {shares_gr}  ·  Tier 2+ dilution (>3%)" if shares_gr else ""
    if rf_col == "rf_itr_declining":
        itr = _v("inventory_turnover", "{:.2f}×")
        itr1 = _v("inventory_turnover_1yb", "{:.2f}×")
        parts = []
        if itr:  parts.append(f"ITR current: {itr}")
        if itr1: parts.append(f"prior year: {itr1}")
        return "  ·  ".join(parts) if parts else ""
    if rf_col == "rf_cwip_bloat":
        cwip = _v("cwip_to_assets", "{:.1f}%")
        return f"CWIP/assets: {cwip}  ·  grew >50% YoY — balance-sheet parking risk" if cwip else ""
    if rf_col == "rf_capex_mirage":
        rg = _v("rev_gr_yoy", "{:.1f}%")
        dep = _v("depreciation", "₹{:,.0f} Cr")
        parts = []
        if rg: parts.append(f"rev_gr: +{rg}")
        if dep: parts.append(f"dep: {dep}  ·  capex <0.5× dep")
        return "  ·  ".join(parts) if parts else ""
    if rf_col == "rf_psu_value_destruction":
        roce = _v("roce", "{:.1f}%")
        de = _v("debt_to_equity", "{:.2f}")
        parts = []
        if roce: parts.append(f"ROCE: {roce}")
        if de:   parts.append(f"D/E: {de}")
        return "  ·  ".join(parts) + "  ·  PSU capital spread < cost of capital" if parts else ""
    if rf_col == "rf_lease_inflation":
        opm = _v("opm", "{:.1f}%")
        return f"EBITDA-level OPM: {opm}  ·  Ind AS 116 RoU removes operating lease costs from EBITDA" if opm else ""
    if rf_col == "rf_low_cfo_ebitda":
        v = _v("cfo_to_ebitda", "{:.1f}%")
        return f"CFO/EBITDA: {v}  ·  threshold: <50%" if v else ""
    if rf_col == "rf_wc_double_squeeze":
        dso = _v("days_receivable", "{:.0f}d")
        dpo = _v("days_payable",    "{:.0f}d")
        parts = []
        if dso: parts.append(f"DSO: {dso} (rising >10d)")
        if dpo: parts.append(f"DPO: {dpo} (falling >10d)")
        return "  ·  ".join(parts)
    if rf_col == "rf_snoa":
        v = _v("scaled_net_operating_assets", "{:.3f}")
        return f"SNOA: {v}  ·  threshold: >1.0 (net op assets exceed lagged asset base)" if v else ""
    return ""


# The 7 flags the Fraud Perimeter groups under "🔴 Critical" — derived from _FLAG_DISPLAY so the
# severity map stays the single source of truth (add a critical flag there and this follows).
_CRITICAL_FLAG_COLS = tuple(c for c, (_d, _sev) in _FLAG_DISPLAY.items() if _sev == "🔴")


def _forensic_status(forensic_score: float, flag_count: int, has_critical: bool = False):
    """Selective forensic verdict from the cascade's OWN metrics (forensic_score + red_flag_count
    + critical-severity presence).

    NOT the forensic_label column — that fires its negative band for ~98.6% of the
    universe (only 29/2107 are "🟢 Clean"), so it cried wolf on clean Crown Jewels and CONTRADICTED
    the Schilit shield's "Clean Audit". Census 2026-06-15 on the 2107-stock universe:
    🔴 Sharp 474 (22%) · 🟡 Watch 879 (42%) · 🟢 Clean 754 (36%). Returns (text, color, is_clean).

    SEVERITY GATE (added 2026-08-24): the Clean band judged COUNT and SCORE only, so a stock could
    print "🟢 Clean — No Material Red Flags" directly above a panel headed "🔴 CRITICAL — 1 FLAG"
    (Sarda: score 89, 3 flags, one of them critical). "Material" is exactly what a critical flag
    IS. Measured: 639 stocks were labelled Clean and 201 of them (31.5%) carried a critical flag.
    Those now fall to the existing Elevated band — which is what they always were.
    `has_critical` defaults False so callers that cannot compute it keep the old behaviour.
    """
    if forensic_score < 60 or flag_count >= 8:
        return ("🚨 Sharp Practices Detected", COLORS["red"], False)
    if forensic_score >= 80 and flag_count <= 3 and not has_critical:
        return ("🟢 Clean — No Material Red Flags", COLORS["green"], True)
    return ("🟡 Elevated — Watch the Accounts", COLORS["gold"], False)


def _has_critical_flag(stock: pd.Series) -> bool:
    """True when any 🔴-severity forensic flag fired — the severity input to _forensic_status."""
    return any(int(_g(stock, c, 0)) == 1 for c in _CRITICAL_FLAG_COLS)


def render_forensic_perimeter(stock: pd.Series):
    """
    Vectorized Fraud Perimeter Display.
    Outputs structured, named red-flag badges (not just a count) for every fired
    forensic signal. Connects directly to the cascading forensic filter multiplier.
    """
    flag_count     = int(_g(stock, "red_flag_count",         0))
    forensic_score = _g(stock,  "forensic_score",            100)
    status_txt, status_clr, _ = _forensic_status(forensic_score, flag_count,
                                                 _has_critical_flag(stock))
    f_mult         = _g(stock,  "forensic_multiplier",       1.0)
    piotroski      = int(_g(stock, "piotroski_fscore",        0))
    pio_label      = stock.get("piotroski_label",  "") or ""
    mgmt_int       = int(_g(stock, "management_integrity_score", 0))

    mult_color = (COLORS["green"]  if f_mult == 1.0 else
                  COLORS["gold"]   if f_mult >= 0.90 else
                  COLORS["orange"] if f_mult >= 0.75 else
                  COLORS["red"])

    flag_color = ("#3fb950" if flag_count == 0 else
                  "#d29922" if flag_count <= 2 else
                  "#ff6b35" if flag_count <= 4 else
                  "#f85149")

    # ── KPI strip ────────────────────────────────────────────────────────
    pio_clr  = (COLORS["green"] if piotroski >= 7 else
                COLORS["gold"]  if piotroski >= 5 else COLORS["red"])
    # 0 contradictions on today's data, but forensic_score is NOT integer-valued (18 distinct
    # values) -- that zero is luck, not structure, so it bands on the shown figure like the rest.
    _fsc_shown = _shown(forensic_score, 0)
    fsc_clr  = (COLORS["green"] if _fsc_shown >= 80 else
                COLORS["gold"]  if _fsc_shown >= 60 else COLORS["red"])

    st.markdown(f"""
    <div class="ts-kpi-strip">
      <div class="ts-kpi-cell" style="border-top:3px solid {flag_color};">
        <div class="ts-kpi-val" style="color:{flag_color};">{flag_count}</div>
        <div class="ts-kpi-lbl">Red Flags / {FORENSIC_MAX_FLAGS}{help_chip('Red Flags')}</div>
      </div>
      <div class="ts-kpi-cell" style="border-top:3px solid {fsc_clr};">
        <div class="ts-kpi-val" style="color:{fsc_clr};">{forensic_score:.0f}</div>
        <div class="ts-kpi-lbl">Forensic Score{help_chip('Forensic Scr')}</div>
      </div>
      <div class="ts-kpi-cell" style="border-top:3px solid {mult_color};">
        <div class="ts-kpi-val" style="color:{mult_color};">{f_mult:.0%}</div>
        <div class="ts-kpi-lbl">Score Multiplier{help_chip('Forensic Mult')}</div>
      </div>
      <div class="ts-kpi-cell" style="border-top:3px solid {pio_clr};">
        <div class="ts-kpi-val" style="color:{pio_clr};">{piotroski}/9</div>
        <div class="ts-kpi-lbl">Piotroski F-Score{help_chip('Piotroski')}</div>
      </div>
      <div class="ts-kpi-cell" style="border-top:3px solid {COLORS['purple']};">
        <div class="ts-kpi-val" style="color:{COLORS['purple']};">{mgmt_int}/3</div>
        <div class="ts-kpi-lbl">Mgmt Integrity{help_chip('Mgmt Integrity')}</div>
      </div>
    </div>
    <div style="font-size:0.72rem;color:{COLORS['text_muted']};margin-bottom:12px;">
      Status: <strong style="color:{status_clr};">{_esc(status_txt)}</strong>
      &nbsp;·&nbsp; Piotroski: <strong style="color:{pio_clr};">{_esc(pio_label)}</strong>
    </div>
    """, unsafe_allow_html=True)

    if flag_count == 0:
        st.markdown(f"""
        <div style="background:rgba(63,185,80,0.08);border:1px solid rgba(63,185,80,0.35);
                    border-radius:10px;padding:14px 18px;text-align:center;">
          <div style="font-size:1.2rem;margin-bottom:4px;">✅</div>
          <div style="font-size:0.85rem;font-weight:700;color:{COLORS['green']};">
            Clean Bill of Health
          </div>
          <div style="font-size:0.72rem;color:{COLORS['text_muted']};margin-top:4px;">
            Zero forensic red flags across all {FORENSIC_MAX_FLAGS} accounting checks —
            the full Schilit / Malik / WCS-24 forensic perimeter.
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    if f_mult < 1.0:
        st.markdown(f"""
        <div style="background:rgba(255,107,53,0.07);border:1px solid rgba(255,107,53,0.4);
                    border-radius:8px;padding:10px 14px;margin-bottom:10px;font-size:0.78rem;">
          ⚠️ <strong style="color:{COLORS['orange']};">Cascading Forensic Filter active:</strong>
          <span style="color:{COLORS['text_secondary']};">
            composite score × {f_mult:.0%} ({flag_count} flags fired).
            Engine multiplier preserves rank ordering while proportionally penalising risk.
          </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Flags grouped by severity ─────────────────────────────────────────
    _SEV_ORDER = ["🔴", "🟠", "🟡"]
    _SEV_META  = {
        "🔴": (COLORS["red"],    "rgba(248,81,73,0.08)",   "rgba(248,81,73,0.5)",  "Critical"),
        "🟠": (COLORS["orange"], "rgba(255,107,53,0.08)",  "rgba(255,107,53,0.5)", "High"),
        "🟡": (COLORS["gold"],   "rgba(228,179,65,0.08)",  "rgba(228,179,65,0.5)", "Medium"),
    }

    for sev in _SEV_ORDER:
        sev_flags = [
            (rf_col, desc)
            for rf_col, (desc, s) in _FLAG_DISPLAY.items()
            if s == sev and int(_g(stock, rf_col, 0)) == 1
        ]
        if not sev_flags:
            continue

        clr, bg, bdr, label = _SEV_META[sev]
        st.markdown(
            f'<div style="font-size:0.65rem;font-weight:800;color:{clr};'
            f'text-transform:uppercase;letter-spacing:1px;margin:14px 0 6px 0;">'
            f'{sev} {label} — {len(sev_flags)} flag{"s" if len(sev_flags)>1 else ""}</div>',
            unsafe_allow_html=True,
        )
        for rf_col, desc in sev_flags:
            parts    = desc.split(" — ", 1)
            title    = parts[0].strip()
            fallback = parts[1].strip() if len(parts) > 1 else ""
            val_ctx  = _get_flag_context(stock, rf_col)
            sub_text = val_ctx if val_ctx else fallback
            st.markdown(
                f'<div class="ts-flag-row" style="background:{bg};border-left-color:{bdr};">'
                f'<div class="ts-flag-sev">{sev}</div>'
                f'<div>'
                f'<div class="ts-flag-title" style="color:{clr};">{_esc(title)}</div>'
                f'{"<div class=ts-flag-sub>" + _esc(sub_text) + "</div>" if sub_text else ""}'
                f'</div></div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════
# GURU FRAMEWORK CHECKLIST — Coffee Can / QGLP / WCS 28-30
# ═══════════════════════════════════════════════════════════════

# Plain-language "the idea" one-liner per framework — distilled from docs/handbook/08-the-frameworks.md
# ("*The idea:*" lines). The card already shows the terse GATE SPEC (_FW_META desc); this explains the
# INSIGHT a beginner needs (what it's for / why a pass matters), surfaced via the "?" idea tooltip.
# Every name in _FW_META must appear here — enforced by test_every_framework_card_has_an_idea_tooltip.
_FW_IDEA = {
    # 🏛️ Motilal Oswal wealth-creation family
    "QGLP":                   "Buy Quality + Growth + Longevity at a reasonable Price — Motilal Oswal's flagship long-horizon compounder screen.",
    "MOSL Wealth Creator":    "The profile of a proven long-run wealth creator — profit growth consistent across all horizons with a wide economic-profit spread.",
    "SQGLP Century Stock":    "The strictest bar — QGLP plus a small Size base that can still multiply many times. Passes are very rare by design.",
    "100x Candidate":         "An early-stage, small-base business with the structural setup to compound roughly 100× over a long runway (high ceiling, high uncertainty).",
    "Fallen Quality":         "A genuinely high-quality business temporarily beaten down — quality on sale, not a permanent decliner.",
    "CAP-GAP Compounder":     "22nd WCS longevity proof: RoE at or above 15% across the decade, five years AND today (CAP), with PAT growth at or above 15% across all three windows (GAP).",
    "Economic Moat":          "17th WCS sector-relative moat: RoE above the SECTOR average in at least 4 of 5 windows — a durable edge over direct peers, not an absolute-return test.",
    "Blue Chip Quality":      "An established, high-quality large-cap — a proven, lower-risk compounder.",
    "Consistent in Volatile": "A steady performer that holds up through volatile markets — low earnings/return volatility alongside solid quality.",
    "EP Hockey Stick":        "28th WCS TEMP setup: economic profit positive AND rising, bought at a P/E of 20x or less.",
    "Bruised Blue Chip 29":   "A blue chip fallen hard and cheap versus its own history — a quality name the market has temporarily punished.",
    "Multi-Trillion Cap":     "The very largest, most-proven compounders — mega-cap size with elite, durable quality.",
    # 📚 Fundamental & cash-quality moats
    "Coffee Can":             "Own clean, consistent compounders and forget them — Mukherjea's buy-and-hold quality filter.",
    "Diamond":                "Forensic-verified high-quality compounders — quality that has also survived an accounting screen.",
    "Peaceful Investing":     "Vijay Malik's systematic forensic-quality filter — fundamentally sound, sleep-at-night businesses.",
    "Unusual Billionaires":   "Mukherjea's 'Greatness Formula' — sustained high returns AND consistent growth reinvested over years.",
    "Long Game Quality":      "Fort-like businesses built to compound for years — the strictest balance-sheet bar plus strong free cash flow.",
    "Baid Compounder":        "Gautam Baid's steady, sensible compounding — solid quality with balance-sheet strength and consistent growth.",
    "Basant 30% Club":        "A fast grower bought at a reasonable price — high sustained growth where the price hasn't yet caught up.",
    "Quality Compounder":     "The asset-light, cash-generative quality business — low capital intensity, strong free cash flow, high returns.",
    # ⚡ Technical momentum & growth sieves
    "CAN SLIM":               "O'Neil's screen for an elite grower breaking out with institutional backing — all seven criteria, very rare to pass.",
    "SEPA Momentum":          "Minervini's Specific Entry Point Analysis — buy strength in a Stage 2 uptrend near its highs. A volatility contraction (VCP) is his ideal entry; here it adds to the score but is not required.",
    "Quality Momentum":       "Gray's strongest, smoothest uptrends among quality names — top-20% relative strength with a governance guard.",
    "Lynch Dream":            "Peter Lynch's fast grower at a fair price — growth-at-a-reasonable-price the big funds haven't crowded yet.",
    "EP Improver":            "28th WCS turnaround stage: economic profit still negative but climbing, with RoE, ROCE and margins all turning. Speculative by the study's own word.",
    "SMILE":                  "Vijay Kedia's small-cap with Integrity, large aspiration and extra-large potential — an under-the-radar scaler.",
    # 🛡️ Valuation, capital allocation & defense shields
    "Magic Formula":          "Greenblatt's cheap-and-good — a high-return business (ROCE) at a genuinely cheap enterprise price (EBIT/EV).",
    "Dhandho Asymmetry":      "Pabrai's heads-I-win, tails-I-don't-lose-much — a low-downside, real-upside bet where fear has outrun the facts.",
    "Parikh Contrarian":      "Parag Parikh's out-of-favour but fundamentally sound — a sensible contrarian value the crowd has overlooked.",
    "Wide Moat":              "Dorsey's structural, durable moats — high returns on capital with a moat that is wide and still holding.",
    "Outsider CEO":           "Thorndike's elite capital allocators — buybacks without dilution, strong cash generation, debt discipline.",
    "Expectations Matrix":    "Mauboussin's expectations gap — the market is implying less than the business can likely deliver.",
    "Schilit Clean":          "Passes Schilit's Financial Shenanigans screen — at most 2 of the 4 accounting-manipulation checkers fire.",
    "Marks Cycle Shield":     "Howard Marks' cycle posture — risk/reward is favourable, not late-cycle euphoric.",
    # 🎣 Fisher & Mayer
    "Fisher Quality":         "Fisher's 15 qualitative points on a truly excellent business, run as automated quantitative proxies.",
    "Fisher Scalability":     "Does the business still have room to grow? — revenue runway, operating leverage, pricing power, no dilution.",
    "100-Bagger":             "Phelps/Mayer's long-compounding small-base setup that can multiply 100× — consistent growth, low payback, low pledge.",
}


# ── 37-FRAMEWORK EMOJI MATRIX — absolute zero-duplicate uniqueness contract ──────
# Every emoji below is unique across all 37 frameworks (visual-sanitization mandate).
# Names must match exactly what scoring_engine writes into frameworks_passed column.
# NOTE: "Fisher Scalability" was moved 📡 → 📶 because CAN SLIM now owns 📡 (radar);
#        📶 (ascending signal bars) cleanly reads as operating-leverage scaling.
# Module-level (importable) so the Reference → Markdown export emits the full framework registry from
# this SAME single source. Read statically (AST) by the emoji/concept tests; used live by the tearsheet.
_FW_META = {
        # ── 🏛️ Motilal Oswal Wealth Creation Frameworks ──
        "QGLP":                    (COLORS["purple"], "🥇", "Quality + Growth + Longevity + Price — Raamdeo"),
        "MOSL Wealth Creator":     (COLORS["gold"],   "🌟", "Raamdeo's Wealth Creator criteria from annual WCS"),
        "SQGLP Century Stock":     (COLORS["gold"],   "👑", "MOSL 19th: ≥4 of 5 SQGLP pillars (Size·Quality·Growth·Longevity·Price)"),
        # PROVENANCE CORRECTED 2026-08-30: was "17th WCS Mouse-to-Elephant". The 19th-study audit
        # established 100x is the 19th WCS (the 17th is Economic Moat) and that "Mouse to Elephant"
        # appears 0× in it; the gates below are the ENGINE's construction, not the study's screen
        # (the study's own is mcap<₹30b + value-migration + P/E<25x). The engine comment was fixed
        # then; this pill text was not — pinned now by test_no_refuted_study_attribution_survives.
        "100x Candidate":          (COLORS["gold"],   "🐘", "19th WCS lineage, engine-constructed gates: PAT CAGR ≥20% + ROCE ≥20% + mcap ≤₹15k Cr + D/E <0.5 + ROE ≥15%"),
        "Fallen Quality":          (COLORS["cyan"],   "🩹", "All-cap fallen quality: ROCE≥15% + PAT CAGR≥10%, >40% off 52WH, cheap vs own 10Y PE"),
        "CAP-GAP Compounder":      (COLORS["green"],  "📐", "22nd WCS: RoE ≥ 15% (10Y/5Y/now) + PAT growth ≥ 15% (10/5/3Y) — longevity proof"),
        "Economic Moat":           (COLORS["purple"], "🏰", "17th WCS: RoE above sector AVERAGE in ≥4 of 5 windows — sector-relative moat"),
        "Blue Chip Quality":       (COLORS["blue"],   "💙", "MOSL 16th: 10Y ROE ≥15% + dividend payout ≥20% + PAT no-crash consistency + ≥5M shares"),
        "Consistent in Volatile":  (COLORS["orange"], "🌪️", "27th WCS: consistent compounder in volatile sector — 19% CAGR"),
        "EP Hockey Stick":         (COLORS["green"],  "🏒", "28th WCS TEMP: Economic Profit positive AND rising YoY, entered at P/E <= 20x"),
        "Bruised Blue Chip 29":    (COLORS["blue"],   "🏛️", "Elite ROCE + large-cap at P/B ≤2× — 29th WCS"),
        "Multi-Trillion Cap":      (COLORS["purple"], "🌐", "Sunrise sector at compounding velocity — 30th WCS"),
        # ── 📚 Fundamental & Cash Quality Moats ──
        "Coffee Can":              (COLORS["gold"],   "☕", "ROCE ≥15% for 10Y + Rev CAGR ≥10% — Mukherjea"),
        # FOUR GATE SPECS CORRECTED 2026-08-30 against their engine gates (final-build sign-off).
        # Each old text described a screen the engine does not run — Diamond was written as a
        # deep-value earnings-yield test (it is Mukherjea's three-lens forensic compounder),
        # Peaceful Investing borrowed Quality Compounder's NFAT signal (Malik gates 5 pillars),
        # Long Game cited PAT-CAGR/volatility (its gate is ICR≥5 + FCF/PAT≥60), and Quality
        # Compounder cited ROCE≥20/PAT-CAGR (its gate is ROCE≥15 + NFAT>4 + FCF-yield≥2).
        # These strings show on the tearsheet pill, the Reference registry AND the download.
        "Diamond":                 (COLORS["cyan"],   "💎", "Mukherjea Diamonds: ROCE ≥15% (10Y+5Y) + D/E <0.5 + CFO/PAT ≥80% + FCF/CFO ≥25% + zero forensic flags"),
        "Peaceful Investing":      (COLORS["gold"],   "🕊️", "Vijay Malik: all 5 pillars — growth runway + profit stability + debt fortress + cash generation + self-funded"),
        "Unusual Billionaires":    (COLORS["purple"], "💰", "Saurabh Mukherjea: promoter-run compounders"),
        "Long Game Quality":       (COLORS["purple"], "⏳", "Khandelwal fortress: interest cover ≥5× + FCF/PAT ≥60% after capex + ROCE ≥15% + D/E ≤0.5"),
        "Baid Compounder":         (COLORS["green"],  "📚", "Gautam Baid: 7Y ROCE ≥ 15% + 10Y Rev CAGR ≥ 12% + no-stumble consistency"),
        "Basant 30% Club":         (COLORS["gold"],   "🏅", "Basant Maheshwari: PAT CAGR ≥ 30% for 5Y + promoter"),
        "Quality Compounder":      (COLORS["green"],  "⭐", "AKO three-circle: NFAT >4 (asset-light) + FCF yield ≥2% + ROCE ≥15% (10Y+5Y) + CFO/PAT ≥80%"),
        # ── ⚡ Technical Momentum & Growth Sieves ──
        "CAN SLIM":                (COLORS["blue"],   "📡", "O'Neil: EPS + Revenue + Institutional + Near High"),
        "SEPA Momentum":           (COLORS["blue"],   "⚡", "Mark Minervini: Stage 2 + RS + Earnings acceleration"),
        "Quality Momentum":        (COLORS["green"],  "🚀", "High quality fundamentals + price momentum confluence"),
        "Lynch Dream":             (COLORS["green"],  "👓", "PEG ≤1.0 + Rev outpacing costs — Peter Lynch"),
        "EP Improver":             (COLORS["orange"], "📈", "28th WCS Q4/Q5 turnaround: EP negative but climbing, RoE+ROCE+margins all turning — speculative stage"),
        "SMILE":                   (COLORS["green"],  "😊", "Vijay Kedia: Small + Integrity + Large aspiration + Extra-large potential"),
        # ── 🛡️ Valuation, Capital Allocation & System Defense Shields ──
        "Magic Formula":           (COLORS["gold"],   "🧮", "High ROCE + High Earnings Yield — Joel Greenblatt"),
        "Dhandho Asymmetry":       (COLORS["gold"],   "🎲", "Pabrai: Heads I win, tails I don't lose much"),
        "Parikh Contrarian":       (COLORS["orange"], "🔄", "Rajeev Parikh: contrarian with forensic clean bill"),
        "Wide Moat":               (COLORS["purple"], "🌊", "Pat Dorsey: structural moat with ROCE expanding"),
        "Outsider CEO":            (COLORS["orange"], "🎯", "Thorndike: buybacks + decentralised capital allocation"),
        "Expectations Matrix":     (COLORS["purple"], "🔮", "Mauboussin PIE: implied CAP realistic + treadmill safe + operating leverage intact"),
        "Schilit Clean":           (COLORS["red"],    "🕵️", "Passes Schilit's Financial Shenanigans screen — accounting-manipulation perimeter clear"),
        "Marks Cycle Shield":      (COLORS["cyan"],   "🛡️", "Howard Marks: not at cyclical-peak margins; mean-reversion risk low"),
        # ── 🎣 Fisher dual-engine + Mayer 100-Bagger (not in the 34-row matrix; kept unique) ──
        "Fisher Quality":          (COLORS["green"],  "🎣", "Phil Fisher 15-point scuttlebutt quality check"),
        "Fisher Scalability":      (COLORS["purple"], "📶", "Fisher operating leverage inflection — Rev runway + OpLev + Pricing + Anti-dilution"),
        "100-Bagger":              (COLORS["gold"],   "💯", "Mayer: owner-operator + small + high ROCE + low payout"),
}


def render_guru_frameworks(stock: pd.Series):
    """
    Displays which institutional Guru frameworks the stock passes.
    Reads pre-computed framework flags from scoring_engine — no re-computation.
    """
    fw_list = _parse_frameworks(stock.get("frameworks_passed", "None"))

    if not fw_list:
        st.info("No institutional Guru frameworks fully met in current market configuration.")
        return

    total_fw = len(_FW_META)
    passed_n = len(fw_list)
    pct = int(passed_n / total_fw * 100)
    bar_clr = COLORS["green"] if pct >= 30 else COLORS["gold"] if pct >= 10 else COLORS["orange"]

    # Group the passed frameworks under the 5 §7 category headers (not a flat grid) —
    # so the drill-down reads as "this stock's conviction comes from X, Y, Z styles".
    fw_set = set(fw_list)

    def _fw_card(fw):
        color, icon, desc = _FW_META.get(fw, (COLORS["text_muted"], "✅", fw))
        return (
            f'<div class="ts-fw-card" style="background:{color}10;border-color:{color}40;">'
            f'<div class="ts-fw-card-head">'
            f'<span style="font-size:1.1rem;">{icon}</span>'
            f'<span class="ts-fw-card-name" style="color:{color};">{_esc(fw)}</span>'
            f'{help_chip(tip=_FW_IDEA.get(fw, ""))}'
            f'</div>'
            f'<div class="ts-fw-card-desc">{_esc(desc)}</div>'
            f'</div>'
        )

    grid_cards = ""
    _categorized = set()
    for _cemoji, _clbl, _cclr, _cfws in FRAMEWORK_CATEGORIES:
        _categorized.update(_cfws)
        _hits = [f for f in _cfws if f in fw_set]
        if not _hits:
            continue
        grid_cards += (
            f'<div style="font-size:0.68rem;font-weight:800;color:{_cclr};letter-spacing:0.6px;'
            f'margin:12px 0 6px 0;">{_cemoji} {_clbl.upper()} · {len(_hits)}</div>'
            f'<div class="ts-fw-grid">{"".join(_fw_card(f) for f in _hits)}</div>'
        )
    _other = [f for f in fw_list if f not in _categorized]
    if _other:
        grid_cards += (
            f'<div style="font-size:0.68rem;font-weight:800;color:{COLORS["text_muted"]};'
            f'letter-spacing:0.6px;margin:12px 0 6px 0;">OTHER · {len(_other)}</div>'
            f'<div class="ts-fw-grid">{"".join(_fw_card(f) for f in _other)}</div>'
        )

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;
                background:{COLORS['bg_secondary']};border:1px solid {COLORS['border']};
                border-radius:10px;padding:12px 16px;">
      <div style="font-size:1.8rem;font-weight:900;color:{bar_clr};">{passed_n}</div>
      <div style="flex:1;">
        <div style="font-size:0.75rem;font-weight:700;color:{COLORS['text_primary']};">
          of {total_fw} Guru Frameworks Passed
        </div>
        <div style="height:6px;background:{COLORS['bg_tertiary']};border-radius:3px;
                    margin-top:6px;overflow:hidden;">
          <div style="width:{pct}%;height:6px;background:{bar_clr};border-radius:3px;"></div>
        </div>
      </div>
      <div style="font-size:0.7rem;color:{COLORS['text_muted']};">{pct}%</div>
    </div>
    {grid_cards}
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PIOTROSKI F-SCORE CHECKLIST
# ═══════════════════════════════════════════════════════════════

# (column, label, operative rule, tooltip). Passable checks first, trend checks after; the score
# itself is read from piotroski_fscore and never recounted here.
_PIOTROSKI_9 = [
    ("f_roa_positive",         "ROA positive",             "PAT > 0",
     "Profitability, the first of Piotroski's nine. A company earning nothing cannot be a value "
     "recovery — this is the floor the other eight build on."),
    ("f_ocf_positive",         "Operating CF positive",    "CFO > 0",
     "Cash from operations above zero. Reported profit can be an accounting opinion; operating "
     "cash flow is much harder to manufacture."),
    ("f_accrual_quality",      "Accruals clean",           "CFO > PAT",
     "Cash flow exceeding reported profit. When profit runs ahead of cash the gap is accruals, "
     "and Piotroski treats that as the clearest quality warning in the set."),
    ("f_no_dilution",          "No equity issued",         "share count not up",
     "Piotroski's wording is that the firm did not ISSUE common equity — an offering. A bonus "
     "issue or a split is not an offering: no shares are sold, no capital is raised, and every "
     "holder's stake is unchanged, so corporate actions are exempt from this check."),
    ("f_roa_improving",        "ROA improving",            "vs last year",
     "Return on assets higher than a year ago — the business is getting more out of what it owns."),
    ("f_leverage_declining",   "Leverage declining",       "debt/assets down",
     "Long-term debt falling as a share of assets. Rising leverage during a recovery is the "
     "opposite of the deleveraging Piotroski looks for."),
    ("f_margin_improving",     "Margin improving",         "op. margin up (full year)",
     "Full-year operating margin above the previous full year's — pricing power or cost discipline, "
     "not volume alone. Piotroski uses gross margin; no annual gross-margin figure exists in the "
     "data, so operating margin stands in."),
    ("f_efficiency_improving", "Asset turnover improving", "revenue/assets up",
     "Revenue per rupee of assets rising — the balance sheet is working harder."),
    ("f_liquidity_improving",  "Liquidity improving",      "current ratio up",
     "Current ratio above last year's. Needs a PRIOR-year current ratio to compare against."),
]

# A check whose two inputs are IDENTICAL cannot be evaluated — that is a data limitation, not a
# company failure. Derived per stock from the same columns the engine compares, so it self-heals
# the moment the source does; no column is hardcoded as "known broken".
_UNEVALUABLE_WHEN_EQUAL = {"f_liquidity_improving": ("current_ratio", "current_ratio_1yb")}


def _pio_row(label: str, passed, value_str: str, context: str = "", tip: str = "") -> str:
    """Row renderer for the checklist.

    Deliberately a SEPARATE function from `_row` in render_financial_insights: that one is nested
    by design and five contracts assert it stays there, so promoting it to share would rewrite
    working tests to suit a refactor. The two are pinned to behave identically instead —
    test_piotroski_checklist.py drives both over every input shape and compares the icons.

    NUMPY BOOL TRAP: `np.float64(1) == 1` yields np.bool_, for which `x is True` is FALSE. Coerce
    once here so no caller can reintroduce the bug that silently greyed out 13 of 17 rows on every
    stock in August 2026; None is preserved as the genuine unknown.
    """
    if passed is not None:
        passed = bool(passed)
    if passed is True:
        ico, clr = "✅", COLORS["green"]
    elif passed is False:
        ico, clr = "❌", COLORS["red"]
    else:
        ico, clr = "⚪", COLORS["text_muted"]
    _tip = help_chip("", tip) if tip else ""
    ctx = (
        f'<span style="font-size:0.68rem;color:{COLORS["text_muted"]};flex:1;'
        f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'
        f'margin-left:6px;">{_esc(context)}</span>'
    ) if context else ""
    return (
        f'<div style="display:flex;align-items:center;gap:6px;padding:5px 0;'
        f'border-bottom:1px solid rgba(255,255,255,0.04);">'
        f'<span style="font-size:0.85rem;width:18px;flex-shrink:0;">{ico}</span>'
        f'<span style="font-size:0.76rem;color:{COLORS["text_secondary"]};width:175px;'
        f'flex-shrink:0;">{_esc(label)}{_tip}</span>'
        f'<span style="font-size:0.80rem;font-weight:700;color:{clr};'
        f'white-space:nowrap;flex-shrink:0;">{_esc(value_str)}</span>'
        f'{ctx}</div>'
    )


def render_piotroski_checklist(stock: pd.Series):
    """The nine components behind the number already shown as "Piotroski n/9".

    The F-Score is a CHECKLIST, not a score: "4/9" without the failing five is the least useful
    form it can take. All nine columns are computed at 100% coverage and none of them reached the
    screen — exactly the orphan class tools/ui_coverage.py exists to find.

    The ⚪ state is the point of this panel. `f_liquidity_improving` passes for 0 of 2,117 stocks,
    because the sheet's "Current Ratio 1 Year Back" equals the current ratio on every row and the
    engine's `_cr != _cr_1yb` guard correctly refuses to score it. That caps the ENTIRE universe at
    8/9 — the observed maximum is 8, reached by 93 stocks — and nothing on screen said so.
    Rendering it ❌ would blame every company for a broken source column: "unverifiable is not
    passed", applied in the other direction — do not CONDEMN on absent evidence either.
    """
    _score = stock.get("piotroski_fscore")
    if pd.isna(_score):
        return
    _score = int(_score)

    rows, n_dead = "", 0
    for _col, _lbl, _ctx, _tip in _PIOTROSKI_9:
        _v = stock.get(_col)
        _pair = _UNEVALUABLE_WHEN_EQUAL.get(_col)
        _blind = False
        if _pair:
            _a, _b = stock.get(_pair[0]), stock.get(_pair[1])
            _blind = pd.isna(_a) or pd.isna(_b) or float(_a) == float(_b)
        if _blind:
            n_dead += 1
            rows += _pio_row(
                _lbl, None, "not evaluable", "source data cannot support this check",
                _tip + " Here the prior-year figure is identical to the current one, so the "
                       "comparison cannot run and the engine correctly declines to award the "
                       "point. A DATA limitation, not a company failure — it clears itself when "
                       "the source is fixed.")
        else:
            rows += _pio_row(_lbl, None if pd.isna(_v) else bool(_v), "", _ctx, _tip)

    _max = 9 - n_dead
    _clr = (COLORS["green"] if _score >= 7 else
            COLORS["gold"] if _score >= 5 else
            COLORS["orange"] if _score >= 3 else COLORS["red"])
    _cap = (f'<span style="font-size:0.62rem;color:{COLORS["text_muted"]};margin-left:auto;'
            f'text-align:right;">{n_dead} check{"s" if n_dead != 1 else ""} not evaluable · '
            f'attainable maximum {_max}</span>') if n_dead else ""

    st.markdown(
        f'<div style="background:{COLORS["bg_secondary"]};border:1px solid {COLORS["border"]};'
        f'border-left:3px solid {_clr};border-radius:10px;padding:14px 16px;margin-bottom:12px;">'
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
        f'<span style="font-size:0.68rem;font-weight:800;color:{_clr};letter-spacing:0.6px;">'
        f'🧾 PIOTROSKI F-SCORE</span>'
        f'<span style="font-size:1.0rem;font-weight:900;color:{_clr};">{_score}</span>'
        f'<span style="font-size:0.68rem;color:{COLORS["text_muted"]};">of 9</span>'
        f'{help_chip("Piotroski F-Score detail", "Joseph Piotroski\'s nine-point accounting "
                    "checklist — profitability, leverage and liquidity, operating efficiency. It "
                    "is a CHECKLIST: WHICH nine pass matters more than the total. 7-9 is strong, "
                    "0-3 weak.")}'
        f'{_cap}</div>{rows}</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# SYSTEMATIC FISHER PROXY — 100% Automated from CSV
# ═══════════════════════════════════════════════════════════════

def _fisher_quality_proxies(stock: pd.Series):
    """The 7 Fisher Quality gates EXACTLY as scoring_engine.fw_fisher (Framework 11) computes them —
    same columns, same fillna defaults, same thresholds — so the on-screen pass/fail can NEVER diverge
    from the engine's `fisher_quality_pass` / the "Fisher Quality" framework pill. Pure: returns a list
    of (label, is_pass: bool, value_str); no Streamlit, no recomputation drift. Pinned across the whole
    universe by test_fisher_module_consistency.

    P15 is Fisher's STRICTER integrity bar (forensic_score >= 90) — deliberately tighter than the
    forensic cascade's "Clean" verdict (>=80 & <=3 flags) shown in the Fraud Perimeter above, because
    Fisher treats integrity as the master filter. So a cascade-"Clean" stock (e.g. forensic 89) can
    still read ❌ here and correctly miss the Fisher Quality framework — that's intended, not a bug.
    """
    rev  = _g(stock, "rev_gr_5y", 0.0)
    npm  = _g(stock, "npm", 0.0)
    npm1 = _g(stock, "npm_1yb", 0.0)
    cfo  = _g(stock, "cfo_to_pat", 0.0)
    dil  = int(_g(stock, "dilution_flag", 1))
    opl  = int(_g(stock, "operating_leverage", 0))
    fsc  = _g(stock, "forensic_score", 999.0)
    return [
        ("P1: Market Potential (Sales Growth >15%)",        bool(rev >= 15),   f"{rev:.1f}%"),
        ("P4: Sales Org Efficiency (Profit Gr > Sales Gr)", bool(opl == 1),    "Passed" if opl == 1 else "Failed"),
        ("P5: Worthwhile Margins (NPM >10%)",               bool(npm >= 10),   f"{npm:.1f}%"),
        ("P6: Margin Trajectory (NPM ≥ Last Year)",         bool(npm >= npm1), "Improving" if npm >= npm1 else "Declining"),
        ("P10: Accounting Controls (CFO/PAT ≥70%)",         bool(cfo >= 70),   f"{cfo:.1f}%"),
        ("P13: No Equity Dilution (Share Count Stable)",    bool(dil == 0),    "Clean" if dil == 0 else "Diluted"),
        ("P15: Fisher Integrity (Forensic Score ≥90)",      bool(fsc >= 90),   f"{fsc:.0f} / 90"),
    ]


def render_fisher_module(stock: pd.Series):
    """
    Translates Philip Fisher's 15 qualitative principles into strict quantitative
    proxies using ONLY pre-derived CSV columns. Zero manual input; zero re-computation.
    The 7 pass/fail checks come from _fisher_quality_proxies (mirrors the engine's fw_fisher gate);
    the headline verdict reads the engine's binary `fisher_quality_pass` — never a softer recomputation.
    """
    # ── Fisher Lifecycle Quadrant Banner ─────────────────────────────────────
    # Materialised by scoring_engine fw_fisher_scalability + fw_fisher dual-engine.
    # Placed at the TOP of the Fisher module so the strategic classification is
    # the first thing a user reads before the P1-P15 proxy detail below it.
    quadrant = stock.get("fisher_lifecycle_quadrant", "⚪ Laggard") or "⚪ Laggard"
    f_score  = int(float(_g(stock, "fisher_score", 0)))

    _q_colors = {
        "👑 Apex Winner":       "#bc8cff",   # purple  — quality + scalability firing
        "🐢 Steady Compounder": "#58a6ff",   # blue    — quality proven, no inflection
        "⚡ Catalyst Play":     "#d29922",   # gold    — inflection without structural quality
        "⚪ Laggard":           "#8b949e",   # grey    — neither gate passing
    }
    _q_descriptions = {
        "👑 Apex Winner":       "Elite quality business AT its operating leverage peak — prime entry signal",
        "🐢 Steady Compounder": "Structural quality proven; no current scalability inflection — steady long hold",
        "⚡ Catalyst Play":     "Inflection firing but structural quality absent — trading candidate, cap position size",
        "⚪ Laggard":           "Fails both Fisher Quality and Scalability gates — structural irrelevance",
    }
    q_clr  = _q_colors.get(quadrant, "#8b949e")
    q_desc = _q_descriptions.get(quadrant, "")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1117,#161b22);
                border:1px solid {q_clr}44;
                border-left:4px solid {q_clr};
                border-radius:12px;
                padding:14px 20px;margin-bottom:16px;
                box-shadow:0 2px 12px {q_clr}22;">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
        <div style="flex:1;">
          <div style="font-size:0.62rem;font-weight:800;color:{q_clr};
                      text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;">
            Fisher Lifecycle Quadrant
          </div>
          <div style="font-size:1.05rem;font-weight:900;color:{q_clr};">
            {_esc(quadrant)}
          </div>
          <div style="font-size:0.71rem;color:#8b949e;margin-top:4px;">
            {_esc(q_desc)}
          </div>
        </div>
        <div style="text-align:center;flex-shrink:0;">
          <div style="font-size:1.5rem;font-weight:900;color:{q_clr};line-height:1;">
            {f_score}/4
          </div>
          <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                      letter-spacing:0.8px;margin-top:2px;">
            Scalability Gates
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # The 7 Fisher Quality proxies — sourced from _fisher_quality_proxies, which mirrors the engine's
    # fw_fisher gate EXACTLY (same columns, fillna defaults, thresholds). So the on-screen pass/fail can
    # never diverge from fisher_quality_pass / the "Fisher Quality" pill (pinned by
    # test_fisher_module_consistency). P15 is Fisher's STRICTER integrity bar (forensic ≥90).
    proxies = _fisher_quality_proxies(stock)
    passed  = sum(1 for _, is_pass, _ in proxies if is_pass)
    total   = len(proxies)
    score_pct = (passed / total) * 100

    # ── Score summary bar ─────────────────────────────────────────────────
    # Verdict = the ENGINE's binary fisher_quality_pass (single source of truth), NOT a soft %-gradient.
    # So the module can never say "qualifies" when the framework gate — and the Lifecycle banner above —
    # say it doesn't. passed == total ⟺ fisher_quality_pass (drift-pinned).
    q_pass = int(_g(stock, "fisher_quality_pass", 0)) == 1
    if q_pass:
        verdict     = "🟢 Fisher Quality — Framework MET"
        gauge_color = COLORS["green"]
    else:
        _short      = total - passed
        verdict     = f"⚪ Fisher Quality — Not Met ({_short} gate{'s' if _short != 1 else ''} short)"
        gauge_color = COLORS["gold"] if passed >= 4 else COLORS["red"]
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:16px;
                background:{COLORS['bg_secondary']};border:1px solid {COLORS['border']};
                border-radius:10px;padding:12px 16px;margin-bottom:14px;">
      <div style="font-size:2.2rem;font-weight:900;color:{gauge_color};">{passed}/{total}</div>
      <div style="flex:1;">
        <div style="font-size:0.75rem;font-weight:700;color:{COLORS['text_primary']};">{verdict}</div>
        <div style="height:6px;background:{COLORS['bg_tertiary']};border-radius:3px;
                    margin-top:6px;overflow:hidden;">
          <div style="width:{score_pct:.0f}%;height:6px;background:{gauge_color};
                      border-radius:3px;"></div>
        </div>
      </div>
      <div style="font-size:0.7rem;color:{COLORS['text_muted']};">Fisher Score</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Card grid ─────────────────────────────────────────────────────────
    cards_html = ""
    for desc, is_pass, val in proxies:
        clr  = COLORS["green"] if is_pass else COLORS["red"]
        bg   = f"{clr}0d"
        bdr  = f"{clr}40"
        ico  = "✅" if is_pass else "❌"
        # Extract short key (e.g. "P1: Market Potential" → "P1")
        short = desc.split(":")[0].strip() if ":" in desc else desc[:4]
        long  = desc.split(":", 1)[1].strip() if ":" in desc else desc
        cards_html += (
            f'<div class="ts-fisher-card" style="background:{bg};border-color:{bdr};">'
            f'<div class="ts-fisher-head">'
            f'<span style="font-size:0.85rem;">{ico}</span>'
            f'<span class="ts-fisher-key" style="color:{clr};">{_esc(short)}</span>'
            f'<span style="font-size:0.66rem;color:{COLORS["text_muted"]};">'
            f'{_esc(long)}</span>'
            f'</div>'
            f'<div class="ts-fisher-val" style="color:{clr};">{_esc(val)}</div>'
            f'</div>'
        )

    st.markdown(f'<div class="ts-fisher-grid">{cards_html}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# FINANCIAL INSIGHTS PANEL — Translated Business Analysis
# ═══════════════════════════════════════════════════════════════

def render_financial_insights(stock: pd.Series):
    """
    Translates raw CSV metrics into human-language grouped verdicts.
    4 cards: Business Quality · Cash & Debt · Valuation · Ownership.
    Replaces the raw metric grid rows in the Tear-Sheet main view.
    """

    def _row(label: str, passed, value_str: str, context: str = "", tip: str = "") -> str:
        # NUMPY BOOL TRAP (fixed 2026-08-24): every threshold here compares a NumPy float, and
        # `np.float64(18.4) >= 15` returns np.bool_ — for which `x is True` is FALSE. The identity
        # checks below therefore fell through to the neutral ⚪ branch for 13 of 17 rows on EVERY
        # stock, silently turning computed verdicts into "no data" — and ⚪ is this codebase's
        # honest-blank signal, so the panel actively lied about having no answer. Normalizing HERE
        # (not at each call site) immunizes every present and future caller; None is preserved as
        # the genuine unknown state. Pinned by tests/test_financial_insights_display.py.
        if passed is not None:
            passed = bool(passed)
        if passed is True:
            ico, clr = "✅", COLORS["green"]
        elif passed is False:
            ico, clr = "❌", COLORS["red"]
        else:
            ico, clr = "⚪", COLORS["text_muted"]
        c_sec = COLORS["text_secondary"]
        c_mut = COLORS["text_muted"]
        # TRUNCATION FIX (2026-08-24): the context column is flex:1 with ellipsis, so in a
        # half-width card every multi-band rule died mid-sentence ("≥70%: real cash | 50-70%:
        # watch | <50%: accr…"). That was tolerable while the icons were meaningless; once ✅/❌
        # became driven by exactly those rules, hiding them broke the panel's own logic. Rules are
        # now SPLIT: the operative threshold stays visible in `context` (short enough never to
        # clip), and the full multi-band explanation moves into `tip` — rendered via the same
        # help_chip "?" affordance the scorecard and All-Data cells already use.
        _tip = help_chip("", tip) if tip else ""
        ctx = (
            f'<span style="color:{c_mut};font-size:0.68rem;flex:1;min-width:0;'
            f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'
            f'margin-left:6px;">{_esc(context)}</span>'
        ) if context else ""
        return (
            f'<div style="display:flex;align-items:center;gap:6px;padding:5px 0;'
            f'border-bottom:1px solid rgba(255,255,255,0.04);">'
            f'<span style="font-size:0.85rem;width:18px;flex-shrink:0;">{ico}</span>'
            f'<span style="font-size:0.76rem;color:{c_sec};width:155px;flex-shrink:0;">'
            f'{_esc(label)}{_tip}</span>'
            f'<span style="font-size:0.80rem;font-weight:700;color:{clr};'
            f'white-space:nowrap;flex-shrink:0;">'
            f'{_esc(value_str)}</span>'
            f'{ctx}</div>'
        )

    def _card(title: str, icon: str, rows_html: str, border: str) -> str:
        bg = COLORS["bg_secondary"]
        bdr = COLORS["border"]
        return (
            f'<div style="background:{bg};border:1px solid {bdr};'
            f'border-left:3px solid {border};border-radius:10px;'
            f'padding:14px 16px;margin-bottom:12px;">'
            f'<div style="font-size:0.68rem;font-weight:800;color:{border};'
            f'text-transform:uppercase;letter-spacing:1.2px;margin-bottom:9px;">'
            f'{icon}&nbsp; {_esc(title)}</div>'
            f'{rows_html}'
            f'</div>'
        )

    # ── Business Quality ──
    _r10y_raw = _g(stock, "roce_med_10y", None)   # None = missing; 0 = real zero
    roce_10y  = _r10y_raw if _r10y_raw is not None else _g(stock, "roce", 0)
    roce_curr = _g(stock, "roce", 0)
    pat_5y    = _g(stock, "pat_gr_5y", 0)
    rev_5y    = _g(stock, "rev_gr_5y", 0)
    npm       = _g(stock, "npm", 0)
    npm_5y    = _g(stock, "npm_med_5y", npm)

    bq = ""
    bq += _row(
        "ROCE — 10Y Median",
        roce_10y >= 15,
        f"{roce_10y:.1f}%",
        f"Current {roce_curr:.1f}% · {'Accelerating ↑' if roce_curr >= roce_10y else 'Decelerating ↓'}",
        "Bar: 10-year median ROCE ≥ 15%. The decade median proves durability — a single good "
        "year does not. 'Accelerating' compares the CURRENT ROCE against that median.",
    )
    bq += _row(
        "Profit CAGR — 5 Years",
        pat_5y >= 15,
        f"{pat_5y:.1f}% p.a.",
        f"vs Revenue {rev_5y:.1f}% · {'Expanding margin ✅' if pat_5y > rev_5y else 'Margin pressure ⚠️'}",
    )
    # The help text used to read "PAT CAGR > Revenue CAGR" — the FALLBACK basis only. The engine
    # prefers EBIT 3Y − Revenue 3Y (capital-structure-neutral), so a stock could show +18.2pp here
    # while the row ABOVE showed 5-year PAT trailing revenue ("Margin pressure ⚠️") — two true
    # statements reading as a contradiction (EPack Prefab, reported 2026-08-24). Now the row names
    # its real basis and surfaces the computed value instead of a bare "Positive".
    _spc = stock.get("sales_profit_conversion")
    _spc_known = pd.notna(_spc)
    bq += _row(
        "Sales→Profit Conversion",
        (float(_spc) > 0) if _spc_known else None,
        (f"{float(_spc):+.1f}pp" if _spc_known else "Not reported"),
        "EBIT 3Y vs Revenue 3Y",
        "Operating leverage: how many percentage points faster the profit engine grew than "
        "sales. Measured on EBIT over 3 years (capital-structure-neutral); falls back to PAT "
        "over 5 years when EBIT history is unavailable. Positive = a scalable cost structure.",
    )
    bq += _row(
        "Net Margin — 5Y Median",
        # Book + ENGINE parity (fixed 2026-08-25): Malik's checklist says NPM >8% and the engine's
        # own malik_profit_stability pillar uses 8 — this row demanded 10, so the panel was
        # stricter than the engine sitting beside it.
        npm_5y >= 8,
        f"{npm_5y:.1f}%",
        f"Current {npm:.1f}% · Bar: ≥8% · "
        f"{'Stable/Improving' if npm >= npm_5y * 0.95 else 'Declining'}",
    )

    # ── Cash & Debt Quality ──
    cfo_pat = _g(stock, "cfo_to_pat", 0)
    ssgr    = _g(stock, "ssgr", 0)
    ssgr_c  = _g(stock, "ssgr_cushion", 0)
    ssgr_sf = int(_g(stock, "ssgr_self_funded", 0))
    de      = _g(stock, "debt_to_equity", 0)
    tax     = _g(stock, "tax_rate_est", 0)

    cd = ""
    cd += _row(
        "Cash Earnings (CFO/PAT)",
        cfo_pat >= 70,
        f"{cfo_pat:.1f}%",
        "Bar: ≥70%",
        "Cash Earnings = CFO ÷ PAT. ≥70%: profits arrive as real cash. 50–70%: watch. "
        "<50%: accrual risk — reported profit is not converting to cash.",
    )
    # SSGR — three HONEST states (fixed 2026-08-24):
    #  * UNKNOWN: 428 live rows (20.2%) carry a NaN SSGR, and the old `_g(..., 0)` default printed
    #    a red ✗ "External Capital — actual growth exceeds SSGR 0.0%" — a fabricated accusation on
    #    absent data, hitting ranks #1/#3/#4/#6. Only 28 rows have a genuine 0.0% SSGR.
    #  * The fail branch stated only the LEVEL in a sentence that reads as the GAP: "exceeds SSGR
    #    13.5%" parses as "exceeds BY 13.5%" (Sarda: SSGR 13.5%, growth 21.0%, shortfall −7.4%).
    #    Both branches now state the level AND the gap explicitly, mirroring each other.
    _ssgr_raw, _cush_raw = stock.get("ssgr"), stock.get("ssgr_cushion")
    if pd.isna(_ssgr_raw) or pd.isna(_cush_raw):
        cd += _row("Growth Funding (SSGR)", None, "SSGR unavailable",
                   "Needs net margin, asset turnover and payout — one or more not reported")
    else:
        # LAYOUT (fixed 2026-08-24): _row lays out [icon][label][value nowrap][context ellipsis],
        # so a sentence in the VALUE slot eats the whole row and pushes the context off entirely —
        # which is what the first version of this fix did. Every sibling row is short-value +
        # explanatory-context; this one now matches: the SSGR level is the value, the gap and the
        # verdict are the context.
        _ssgr_v, _cush_v = float(_ssgr_raw), float(_cush_raw)
        _self_funded = _cush_v > 0
        cd += _row(
            "Growth Funding (SSGR)",
            _self_funded,
            f"{_ssgr_v:.1f}%",
            (f"Self-funded — covers growth by {_cush_v:.1f}%" if _self_funded else
             f"Growth exceeds it by {abs(_cush_v):.1f}% — externally funded"),
        )
    # Distinguish truly debt-free (int_cov data absent AND D/E near zero) from
    # missing coverage data (company has debt but interest_coverage not in CSV).
    _int_raw = _g(stock, "interest_coverage", None)  # None = NaN/missing
    if _int_raw is None:
        int_pass = de < 0.05
        int_str  = "Debt-free" if de < 0.05 else "Coverage N/A"
    elif _int_raw > 0.01:
        int_pass = _int_raw >= 3
        int_str  = f"{_int_raw:.1f}×"
    else:
        int_pass = de < 0.05
        int_str  = "Debt-free" if de < 0.05 else "Coverage 0×"
    cd += _row(
        "Debt Safety",
        int_pass,
        int_str,
        f"D/E {de:.2f} · Bar: ≥3×",
        "Interest coverage = EBIT ÷ interest expense (Malik Parameter 4: >3). A company with "
        "near-zero debt passes regardless — there is nothing to cover.",
    )
    # STALE REGIME (fixed 2026-08-25). Malik's rule is RELATIVE — "the tax rate should be near
    # general corporate tax rate" — and his ">30%" was simply that rate's value when he wrote it
    # ("In India, the corporate tax rate is 30%", Peaceful Investing p.45). India cut it in 2019 to
    # 22% + surcharge + cess ≈ 25.17%, and the live data confirms the migration completed: median
    # effective rate 25.4%, p25–p75 = 23.4–27.4%. The old 30–55% band therefore FAILED 88.1% of the
    # universe — 1,496 stocks flagged purely for paying the current legal rate, a fire rate the
    # census discipline calls noise. 20–40% spans both regimes (new ≈25%, old ≈35%) and passes
    # 81%. This RESTORES the book's rule rather than overriding it; <10% sharp-practices is
    # untouched and remains the real forensic signal (rf_tax_panic, fires 8.9%).
    tax_ok = (20 <= tax <= 40) if tax > 5 else None
    cd += _row(
        "Tax Rate — Malik P3 proxy",
        tax_ok,
        f"{tax:.1f}%",
        "Normal band 20–40%",
        "Effective tax rate (Malik Parameter 3). Malik's rule is that it should sit NEAR the "
        "general corporate tax rate — which India cut in 2019 to ~25% (22% + surcharge + cess), "
        "with the older regime near 35%. The 20–40% band spans both. Abnormally low payouts "
        "(<10%) raise a separate sharp-practices flag.",
    )

    # ── Valuation ──
    pe_disc  = _g(stock, "pe_discount", 0)
    fcf_y    = _g(stock, "fcf_yield", 0)
    ey       = _g(stock, "earnings_yield", 0)
    peg      = _g(stock, "peg", 0)
    peg_zone = str(stock.get("peg_zone", "") or "")   # _row() escapes it

    vl = ""
    if pe_disc > 1:
        vl += _row("P/E vs 10Y Average", pe_disc >= 15,
                   f"{pe_disc:.1f}% below avg", "≥20%: historically cheap  |  0–20%: fair")
    elif pe_disc < -1:
        vl += _row("P/E vs 10Y Average", False,
                   f"{abs(pe_disc):.1f}% above avg", "Premium to 10Y median",
                   "P/E measured against the stock's OWN 10-year median, not the market's — a "
                   "premium here means it is expensive versus its own history.")
    else:
        vl += _row("P/E vs 10Y Average", None, "At median", "Fair value territory")

    vl += _row(
        "PEG Ratio",
        (0 < peg <= 1.0) if peg > 0 else None,
        f"{peg:.2f}×" if peg > 0 else "N/A",
        (peg_zone or "Bar: ≤1.0"),
        "PEG = P/E ÷ earnings growth. Peter Lynch's rule: ≤1.0 means you are paying no more "
        "for the growth than the growth rate itself — a bargain. Above 2.0 is expensive.",
    )
    # FALSE GREEN (fixed 2026-08-25): the bar was a hardcoded 4%, so a stock yielding 4–7%
    # printed "✅ justifies equity risk" while risk-free G-Secs paid 7% — the panel endorsing a
    # sub-risk-free return (Sarda: ✅ at 6.0%). Malik states the rule RELATIVELY — "EY should be
    # greater than long-term government bond yields" — and config already carries the constant
    # (INDIA_GSEC_YIELD, maintained, used by two engine formulas) which this row simply ignored.
    vl += _row(
        "Earnings Yield",
        ey >= INDIA_GSEC_YIELD,
        f"{ey:.1f}%",
        f"Bar: ≥{INDIA_GSEC_YIELD:.0f}% (10Y G-Sec)",
        f"Earnings Yield = EPS ÷ price. Malik's rule is RELATIVE: it must beat the 10-year "
        f"government bond yield (currently {INDIA_GSEC_YIELD:.1f}%, config.INDIA_GSEC_YIELD) — "
        f"below that you are taking equity risk for less than a risk-free return.",
    )
    # MOSL Study 16's buy signal — dividend yield at/above the G-Sec yield — is RARE (p99 of the
    # live universe is 7.0%), so the row is ✅ only there and ⚪ everywhere else: not paying a
    # dividend is not a failure and a 2% yield is not a ❌. No yield column (archived vintage) →
    # no row; a 0 IS a value ("does not pay", TTM) and prints as 0.00%.
    _dy_ins = stock.get("dividend_yield")
    if _dy_ins is not None and pd.notna(_dy_ins):
        _dy_ins = float(_dy_ins)
        vl += _row(
            "Dividend Yield",
            True if _dy_ins >= INDIA_GSEC_YIELD else None,
            f"{_dy_ins:.2f}%",
            (f"Study 16 buy signal: ≥{INDIA_GSEC_YIELD:.0f}% (10Y G-Sec)" if _dy_ins > 0
             else "Does not pay a dividend (last 12 months)"),
            f"Dividend yield = dividends per share over the last 12 months ÷ price. MOSL's 16th "
            f"Wealth Creation study: a yield at or above the 10-year G-Sec ({INDIA_GSEC_YIELD:.1f}%, "
            f"config.INDIA_GSEC_YIELD) is a rare buy signal — the risk-free rate paid in cash while "
            f"you hold the equity. Below it is simply no signal, never a failure: growth businesses "
            f"reinvest instead.",
        )
    if fcf_y > 0:
        vl += _row(
            "FCF Yield",
            fcf_y >= 3,
            f"{fcf_y:.1f}%",
            "Bar: ≥3%",
            "Free-cash-flow yield = FCF ÷ market cap. ≥4%: excellent. 2–4%: reasonable. "
            "<2%: low — little owner cash generated per rupee of price.",
        )

    # ── Ownership Alignment ──
    prom    = _g(stock, "promoter_holdings", 0)
    pledge  = _g(stock, "pledged_percentage", 0)
    fii     = _g(stock, "fii_holdings", 0)
    dii     = _g(stock, "dii_holdings", 0)
    ch_prom = _g(stock, "change_promoter_lq", 0)
    smart   = str(stock.get("smart_money_flow", "⚪ Neutral") or "⚪ Neutral")  # _row() escapes it

    _dir = "↑" if ch_prom > 0.05 else ("↓" if ch_prom < -0.05 else "→")
    prom_ctx = (
        f"{_dir} {abs(ch_prom):.1f}% last Q  |  "
        f"{'Dynasty ≥60%' if prom >= 60 else ('Well-aligned ≥50%' if prom >= 50 else 'Below ideal <50%')}"
    )
    ow = ""
    ow += _row("Promoter Holding", prom >= 50, f"{prom:.1f}%", prom_ctx)
    ow += _row("Promoter Pledge",  pledge <= 10, f"{pledge:.1f}%", "Bar: ≤10%",
               "Percentage of promoter shares pledged as loan collateral. 0%: clean. <5%: low "
               "risk. >10%: a red flag — a price fall can force lender selling.")
    # Presence (the icon's test) and FLOW (the label) are different measurements — stating both
    # explicitly stops the row reading as a contradiction when a ❌ presence sits beside a ✅ flow.
    ow += _row("FII + DII Holdings", fii >= 5 or dii >= 5,
               f"FII {fii:.1f}%  ·  DII {dii:.1f}%",
               f"≥5% · Flow: {smart}",
               "The icon judges institutional PRESENCE (FII or DII holding ≥5%). The flow label "
               "is a separate read of whether institutions are currently accumulating or "
               "distributing — a low holding can still show positive flow, and vice versa.")

    # ── Render in a balanced 2×2 CSS grid (no st.columns gutter padding) ──
    # FIXED 2 columns so the 4 cards always lay out 2×2 — auto-fit fit 3 per row on wide desktops,
    # which orphaned the 4th card (Ownership) alone on row 2 with empty space beside it. 2 cols also
    # gives each card half-width (vs third) → roomier rows, fewer truncated context notes.
    grid_cards = (
        _card("Business Quality",    "🏭", bq, COLORS["purple"]) +
        _card("Cash & Debt Quality", "💵", cd, COLORS["green"])  +
        _card("Valuation",           "💰", vl, COLORS["gold"])   +
        _card("Ownership Alignment", "👥", ow, COLORS["blue"])
    )
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));'
        f'gap:10px;align-items:start;">{grid_cards}</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# VERDICT SCORECARD — the 6-axis evidence grid (Layer 2 of the verdict)
# ═══════════════════════════════════════════════════════════════

def render_verdict_scorecard(stock: pd.Series):
    """6-axis decision scorecard, mounted right under the verdict header.
    Each cell = the engine's axis pill (verdict_axis_*) + the key supporting metrics, and it
    deliberately surfaces signals that were computed-but-never-shown (IBAS moat, Magic-Formula
    earnings yield, payback ratio, Emerging-VC, SNOA). PURE DISPLAY — pre-materialized columns only.
    """
    def _v(k, d=np.nan):
        x = stock.get(k, d)
        return d if (x is None or (isinstance(x, float) and np.isnan(x))) else x

    def _n(k, fmt="{:.0f}", suf=""):
        x = _v(k)
        try:
            return (fmt.format(float(x)) + suf) if (x == x and x is not None) else "—"
        except Exception:
            return "—"

    def _pill(k):
        return str(stock.get(k, "") or "")

    def _share_change_str():
        """Share-count change, named for what it actually is.

        "Dilution 2496.6%" was an interpretation the data cannot support: EPack Prefab's share
        count went 3.9M → 100.6M because it LISTED, and 85% of that increase was pre-IPO
        restructuring, not issuance. Bonus issues read the same way — Nestle's 1:1 printed
        "Dilution 100%" for handing shareholders free stock. When the engine has classified the
        move as a corporate action (bonus/split/rights/IPO re-basing, `dilution_is_corporate
        _action`), show the raw counts and say so; only ordinary issuance keeps the percentage.
        Mirrors the engine's own flag — never re-derives the 1.5x threshold here.
        """
        # EXTENDED 2026-09-19 TO THE SECOND ARM. The EPack case above IS a pre-listing baseline,
        # but until Phase 2 the only rows reaching this branch were those over the 1.5x corporate-
        # action line. Belrise sits at 1.4857x — UNDER it — so the engine classified it as a
        # pre-listing baseline (Tier 1, unmeasurable) while this function went on printing
        # "Dilution 48.6%". Engine and display then said opposite things about the same row, which
        # is exactly the drift the Fisher module/engine precedent exists to stop. Both arms now
        # route here, each named for what it actually is, and neither threshold is re-derived.
        if _v("dilution_prelisting_baseline", 0) == 1:
            _a, _b = stock.get("equity_shares_1yb"), stock.get("equity_shares")
            if pd.notna(_a) and pd.notna(_b) and float(_a) > 0:
                def _cmpct_pl(n):
                    n = float(n)
                    return (f"{n/1e7:.1f}Cr" if n >= 1e7 else
                            f"{n/1e5:.1f}L"  if n >= 1e5 else f"{n:,.0f}")
                return f"Shares {_cmpct_pl(_a)} → {_cmpct_pl(_b)} (pre-listing baseline)"
            return "Pre-listing baseline"
        if _v("dilution_is_corporate_action", 0) == 1:
            _a, _b = stock.get("equity_shares_1yb"), stock.get("equity_shares")
            if pd.notna(_a) and pd.notna(_b) and float(_a) > 0:
                def _cmpct(n):
                    n = float(n)
                    return (f"{n/1e7:.1f}Cr" if n >= 1e7 else
                            f"{n/1e5:.1f}L"  if n >= 1e5 else f"{n:,.0f}")
                return f"Shares {_cmpct(_a)} → {_cmpct(_b)} (restructuring)"
            return "Shares restructured"
        return f"Dilution {_n('dilution_pct', '{:.1f}', suf='%')}"

    _emerg = "🌱 Emerging VC" if _v("emerging_vc_flag", 0) == 1 else "Mature"
    _snoa  = "⚠ bloating" if _v("rf_snoa", 0) == 1 else "✓ clean"
    _netcash = "✓ net cash" if _v("net_debt_negative", 0) == 1 else "net debt"
    # 6 ORTHOGONAL axes (no double-counting): Moat·Growth·Valuation·Balance·Governance·Forensics.
    # (axis-concept key for the "?" tooltip, the engine's axis pill, supporting metrics)
    axes = [
        ("Moat Axis", _pill("verdict_axis_moat"),
         f"ROCE {_n('roce_med_10y', suf='%')} · ROE {_n('roe_med_10y', suf='%')} · IBAS {_n('ibas_moat_score')}"),
        ("Growth Axis", _pill("verdict_axis_growth"),
         f"EPS·5y {_n('eps_gr_5y', suf='%')} · Rev·5y {_n('rev_gr_5y', suf='%')} · {_emerg}"),
        ("Valuation Axis", _pill("verdict_axis_valuation"),
         f"PE {_n('pe', '{:.1f}')} vs Fair {_n('fair_pe_qglp', '{:.1f}')} · Magic-Yld {_n('magic_formula_earnings_yield', '{:.1f}', suf='%')} · Payback {_n('payback_ratio', '{:.1f}', suf='x')}"),
        ("Balance Axis", _pill("verdict_axis_balance"),
         f"D/E {_n('debt_to_equity', '{:.2f}')} · Int-Cov {_n('interest_coverage', '{:.1f}', suf='x')} · {_netcash}"),
        ("Governance Axis", _pill("verdict_axis_governance"),
         f"Promoter {_n('promoter_holdings', suf='%')} · Pledge {_n('pledged_percentage', suf='%')} · {_share_change_str()}"),
        ("Forensics Axis", _pill("verdict_axis_forensics"),
         f"Piotroski {_n('piotroski_fscore')}/9 · Red flags {_n('red_flag_count')} · SNOA {_snoa}"),
    ]
    cells = "".join(
        f'<div style="flex:1 1 30%;min-width:185px;background:{COLORS["bg_secondary"]};'
        f'border:1px solid {COLORS["border"]};border-radius:8px;padding:8px 11px;">'
        f'<div style="font-size:0.73rem;font-weight:800;color:{COLORS["text_primary"]};'
        f'white-space:nowrap;">{_esc(hdr)}{help_chip(axis_key)}</div>'
        f'<div style="font-size:0.65rem;color:{COLORS["text_secondary"]};margin-top:3px;'
        f'line-height:1.5;">{metrics}</div>'
        f'</div>'
        for axis_key, hdr, metrics in axes
    )
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:7px;margin:0 0 7px 0;">{cells}</div>',
        unsafe_allow_html=True,
    )

    # ── Deep Signals: cross-cutting synthesis metrics that were computed-but-invisible ──
    # (WCS wealth-creation composite, economic profit, Buffett VCR, terms-of-trade, cash machine).
    # Scales verified 2026-06-14: wcs 0-10, EP ₹Cr, VCR ~1x, ToT days, cash 0/50/100.
    def _ds(label, val_str, good):
        # Same NumPy-bool trap as _row (fixed 2026-08-24): the Deep Signals chips pass raw
        # comparisons ((_wcs >= 5) etc.) so all 5 rendered permanently neutral on every stock,
        # while the Entry-Timing chips were correct only because _good() returns Python bools.
        if good is not None:
            good = bool(good)
        clr = (COLORS["green"] if good is True else
               COLORS["red"] if good is False else COLORS["text_secondary"])
        return (f'<span style="font-size:0.62rem;font-weight:700;padding:2px 8px;border-radius:10px;'
                f'background:{COLORS["bg_tertiary"]};border:1px solid {COLORS["border"]};'
                f'color:{clr};white-space:nowrap;">{label}{help_chip(label)}&nbsp;{val_str}</span>')

    _wcs, _ep = _v("wcs_score"), _v("economic_profit")
    _vcr, _tot, _cash = _v("value_creation_ratio"), _v("terms_of_trade_spread"), _v("cash_machine_score")
    _ep_str = (f"₹{_ep:,.0f}cr" if _ep == _ep else "—")
    deep = "".join([
        _ds("WCS",            (f"{_wcs:.0f}/10" if _wcs  == _wcs  else "—"), (_wcs  >= 5)   if _wcs  == _wcs  else None),
        _ds("Econ-Profit",    _ep_str,                                       (_ep   > 0)    if _ep   == _ep   else None),
        # VCR prints TWO decimals, unlike its neighbours. The threshold here is not a noise floor
        # but Buffett's one-dollar premise itself — >=1.00 is the whole test — so the exact side of
        # 1.0 is the point. At one decimal, 150 stocks (7.1%) printed "1.0x" while being coloured
        # from values as far apart as 1.0433 and 0.9542: the colours were right and the label was
        # too coarse to show why two chips differed. Two decimals cuts that to 16 (0.8%).
        # NOT the Trajectory-card fix (round-then-classify): that band WAS a noise floor, so
        # collapsing its edge to neutral was correct. Here the boundary is real and must be shown,
        # not blurred. Same symptom, opposite remedy — pinned by tests/test_deep_signal_chips.py.
        _ds("VCR",            (f"{_vcr:.2f}x"  if _vcr  == _vcr  else "—"), (_vcr  >= 1.0) if _vcr  == _vcr  else None),
        _ds("Terms-of-Trade", (f"{_tot:+.0f}d"  if _tot  == _tot  else "—"), (_tot  > 0)    if _tot  == _tot  else None),
        _ds("Cash-Machine",   (f"{_cash:.0f}"   if _cash == _cash else "—"), (_cash >= 50)  if _cash == _cash else None),
    ])
    st.markdown(
        f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin:0 0 6px 0;">'
        f'<span style="font-size:0.6rem;font-weight:800;color:{COLORS["text_muted"]};'
        f'letter-spacing:0.5px;">🔬 DEEP SIGNALS</span>{deep}</div>',
        unsafe_allow_html=True,
    )

    # ── ⏱️ Entry Timing: momentum reads — the WHEN, NOT part of the WHAT verdict above ──
    # The 6 axes weigh SELECTION and are blind to momentum (fundamentals select, technicals time).
    # Orphans verified alive + orthogonal on live data: relative strength, price trajectory,
    # earnings acceleration, volume confirmation. Thresholds are quartile-grounded. Reuses the
    # _ds() chip so it reads as auxiliary, not a 7th verdict axis.
    #
    # 5th CHIP — Base, ADDED 2026-09-10 by the strip's own tripwire, not by taste. The strip was
    # declared closed on evidence in 2026-08-27, and tests/test_deep_signal_chips.py holds that
    # evidence as a live rule: if any rejected candidate ever becomes MORE orthogonal to the shown
    # chips than they are to each other, it FAILS and demands a fresh judgement. On the 2026-09-09
    # refresh it fired. dist_52wl now reaches max |corr| 0.140 against the four (0.031 traj · 0.043
    # vol · 0.076 accel · 0.140 rs) where those four reach 0.238 among THEMSELVES — so it is more
    # independent of the strip than the strip is of itself. Every other candidate stayed redundant
    # (above_sma200 0.789 · breakout_score 0.806 · dist_52wh 0.754 · crs_52w 0.722 · rsi_14d 0.598).
    #
    # WHY IT PASSES THE BAR THAT REJECTED trend_breakout, which was ALSO orthogonal (0.088): that
    # one was rejected for being rare (fires 0.37%) AND already named in trend_modifier. Neither
    # holds here. dist_52wl is 0% NaN and spreads across the whole universe (deciles 6.6 → 132.8),
    # and its only existing surface is SEPA Pillar L — a BINARY gated at ≥30% that fires 58%, a
    # coin-flip buried in a framework card, not a reading. It answers the one question the other
    # four cannot: is there a base under this, or have I already missed the move?
    #
    # THRESHOLDS ARE THE BOOK'S, not fitted: green ≥30% is Minervini Trend Template Criterion 6
    # verbatim (the same constant behind sepa_low_base), red ≤10% is "no base under it at all".
    # Live tri-state 58% green · 26.4% amber · 15.6% red — no bucket owns the universe.
    #
    # The five chips now reach max pairwise corr 0.24 among themselves (rs↔traj, unchanged — adding
    # Base did not tighten the strip), and THAT is the bar any sixth candidate has to beat.
    def _good(green: bool, red: bool):
        return True if green else (False if red else None)

    _rs, _traj   = _v("rs_score"), _v("trajectory_score")
    _accel, _vsc = _v("eps_acceleration"), _v("volume_score")
    _base        = _v("dist_52wl")
    _rs_ok    = None if _rs    != _rs    else _good(_rs   >= 70,  _rs   <= 30)
    _traj_ok  = None if _traj  != _traj  else _good(_traj >= 0.5, _traj <  0)
    _accel_ok = None if _accel != _accel else _good(_accel >= 10, _accel <  0)
    _vol_ok   = None if _vsc   != _vsc   else _good(_vsc  >= 60,  _vsc  <= 20)
    _base_ok  = None if _base  != _base  else _good(_base >= 30,  _base <= 10)
    timing = "".join([
        _ds("RS",        (f"{_rs:.0f}"    if _rs   == _rs   else "—"), _rs_ok),
        _ds("Traj",      (f"{_traj:+.2f}" if _traj == _traj else "—"), _traj_ok),
        _ds("EPS-Accel", ("▲" if _accel_ok is True else "▼" if _accel_ok is False
                          else "·" if _accel == _accel else "—"),       _accel_ok),
        _ds("Vol",       (f"{_vsc:.0f}"   if _vsc  == _vsc  else "—"),  _vol_ok),
        _ds("Base",      (f"+{_base:.0f}%" if _base == _base else "—"), _base_ok),
    ])
    st.markdown(
        f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin:0 0 12px 0;">'
        f'<span style="font-size:0.6rem;font-weight:800;color:{COLORS["text_muted"]};'
        f'letter-spacing:0.5px;">⏱️ ENTRY TIMING</span>{timing}</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# STOCK HERO HEADER — Premium identity card
# ═══════════════════════════════════════════════════════════════

def render_stock_hero(stock: pd.Series, regime: str = "SIDEWAYS", tier_colors: dict = None):
    """
    Full-width premium hero header. Displays stock identity, composite score ring,
    tier badge, moat quad, forensic status, regime, and all active pills in one card.
    Pure display — reads pre-computed columns only.
    """
    tier_num   = int(_g(stock, "conviction_tier", 5))
    tc         = (tier_colors or TIER_COLORS).get(tier_num, TIER_COLORS[5])
    tcfg       = next((t for t in CONVICTION_TIERS if t["tier"] == tier_num), CONVICTION_TIERS[-1])
    comp       = float(_g(stock, "composite_score", 0))
    name       = _esc(stock.get("name", "N/A") or "N/A")
    rank       = int(_g(stock, "rank", 0))
    sector     = _esc(stock.get("sector", "") or "")
    industry   = _esc(stock.get("industry", "") or "")
    mcap       = _g(stock, "market_cap", 0)
    # Price chip (added 2026-08-24): the tearsheet showed a ₹ stop and ₹ allocation but never the
    # price itself. NaN -> omitted entirely (no fabricated zero).
    _px_hero   = stock.get("close_price")
    px_chip    = (f' &nbsp;·&nbsp; ₹{float(_px_hero):,.2f}'
                  if pd.notna(_px_hero) and float(_px_hero) > 0 else "")
    # Liquidity (2026-10-09): can you buy it in your size? The engine's 20-day traded value and the order
    # that stays under ui_liquidity.ORDER_SHARE of it. Unknown -> omitted (no fabricated zero).
    _liq_line  = trade_line(pd.to_numeric(stock.get("traded_value_cr"), errors="coerce"))
    liq_chip   = f' &nbsp;·&nbsp; {_liq_line}{help_chip("Liquidity")}' if _liq_line else ""
    mcat       = _esc(stock.get("market_category", "") or "")
    mg_quad    = _esc(stock.get("moat_growth_quad", "") or "")
    # Selective forensic badge (forensic_score + red_flag_count) — the forensic_label column reads
    # its negative band for 98.6% of the universe, which contradicted the BUY/Clean-Audit
    # verdict in the hero. See _forensic_status() (same logic powers the Perimeter + Fisher P15).
    f_txt, f_clr, _ = _forensic_status(_g(stock, "forensic_score", 100),
                                       int(_g(stock, "red_flag_count", 0)),
                                       _has_critical_flag(stock))
    # Label as "… Market" so this market-wide regime badge isn't mistaken for a per-stock trait
    # (it's the same on every tearsheet by design — one breadth-derived regime for the whole universe).
    reg_map    = {"BULL": ("🟢 Bull Market", COLORS["green"]), "BEAR": ("🔴 Bear Market", COLORS["red"])}
    reg_txt, reg_clr = reg_map.get(regime, ("🟡 Sideways Market", COLORS["gold"]))

    # Score ring color — matches tier
    ring_clr = tc["text"]
    ring_bdr = tc["border"]

    # ── Active pills (catalysts + frameworks + special signals) ──
    pill_items = []
    _CAT_PILLS = [
        ("cat_capacity",        COLORS["blue"],   "🔥 Capacity Explosion"),
        ("cat_oplev",           COLORS["green"],  "🔥 OpLev Inflection"),
        ("cat_inst_discovery",  COLORS["purple"], "🔥 Inst Discovery"),
        ("cat_deleveraging",    COLORS["gold"],   "🔥 Deleveraging"),
        ("cat_lynch_dream",     COLORS["green"],  "🔥 Lynch Dream"),
        ("cat_strong_quarter",  COLORS["orange"], "🔥 Strong Quarter"),
    ]
    for col, clr, lbl in _CAT_PILLS:
        if int(_g(stock, col, 0)) == 1:
            pill_items.append((lbl, clr))

    if int(_g(stock, "tsunami_signal", 0)) == 1:
        pill_items.append(("🌊 Tsunami", COLORS["purple"]))
    if int(_g(stock, "net_debt_negative", 0)) == 1:
        pill_items.append(("💰 Net Cash", COLORS["green"]))
    # Dedicated colour pills for these two — the generic loop below skips them (no duplicate display).
    if int(_g(stock, "bruised_blue_chip_29", 0)) == 1:
        pill_items.append(("🏛️ Bruised Blue Chip", COLORS["blue"]))
    if int(_g(stock, "mosl_100x_candidate", 0)) == 1:
        pill_items.append(("🐘 100x Candidate", COLORS["gold"]))

    # Dedicated colour pills above already render these two — exclude them as whole tokens
    # so "Bruised Blue Chip 29" can never bleed into a generic "Bruised Blue Chip" pill.
    _DEDICATED_FW = {"100x Candidate", "Bruised Blue Chip 29"}
    fw_list = _parse_frameworks(stock.get("frameworks_passed", "None"), exclude=_DEDICATED_FW)
    for fw in fw_list[:8]:  # cap pills at 8 frameworks to avoid overflow
        pill_items.append((f"🏛️ {_esc(fw)}", COLORS["text_secondary"]))

    pills_html = "".join(
        f'<span style="display:inline-block;padding:3px 10px;border-radius:20px;'
        f'font-size:0.68rem;font-weight:600;margin:2px 3px;'
        f'background:{clr}18;border:1px solid {clr}55;color:{clr};">{lbl}</span>'
        for lbl, clr in pill_items
    )

    # ── Tier / status badges ──
    def _badge(txt, clr):
        return (f'<span style="display:inline-flex;align-items:center;'
                f'padding:4px 12px;border-radius:20px;font-size:0.72rem;font-weight:700;'
                f'background:{clr}18;border:1px solid {clr}55;color:{clr};margin:2px 3px;">'
                f'{txt}</span>')

    _mg_badge = (
        _badge(mg_quad, COLORS["green"] if "Wealth Creator" in mg_quad else
                        COLORS["gold"]  if "Quality Trap"   in mg_quad else
                        COLORS["blue"]  if "Growth Trap"    in mg_quad else COLORS["red"])
    ) if mg_quad else ""
    # Governance risk shield badge — shown only when ownership risk signals fired.
    # gov_risk_count / governance_risk_multiplier are pre-materialized by the engine
    # (compute_governance_bonus); pure display, no threshold re-computation here.
    _gov_n    = int(_g(stock, "gov_risk_count", 0))
    _gov_mult = float(_g(stock, "governance_risk_multiplier", 1.0))
    _gov_badge = (
        _badge(f"⚠️ Governance Risk ×{_gov_mult:.2f} ({_gov_n} signal{'s' if _gov_n > 1 else ''})",
               COLORS["red"] if _gov_n >= 2 else COLORS["orange"])
    ) if _gov_n >= 1 else ""

    # Score-confidence badge — engine-materialized evidence coverage of the ranked
    # inputs (data_coverage_pct / data_coverage_label, see CORE_SCORING_INPUTS).
    # Distinguishes a true mid-score from a data-starved one whose missing inputs
    # became neutral 50s. Pure display: no thresholds, neutral colour, hidden only
    # when the engine columns are absent (legacy cached frames).
    _cov_raw = stock.get("data_coverage_pct")
    _cov_badge = ""
    if _cov_raw is not None and pd.notna(_cov_raw):
        _cov_lbl = str(stock.get("data_coverage_label", "") or "")
        _cov_badge = _badge(
            f"🔍 Evidence {float(_cov_raw):.0f}%"
            + (f" · {_esc(_cov_lbl)}" if _cov_lbl else "")
            + help_chip("Evidence Coverage"),
            COLORS["blue"],
        )

    # Data-recency companion to the Evidence badge: the score rests on the last reported result,
    # so a stock that has filed nothing for config.STALE_RESULT_DAYS (240: a missed deadline under any
    # listing calendar, quarterly or SME half-yearly) is scored on stale fundamentals (and is often
    # in distress — Gensol Engineering sat 477 days stale before its collapse). Shown only when
    # stale; display-only, no scoring impact.
    _stale_badge = ""
    if int(stock.get("result_stale_flag", 0) or 0) == 1:
        _age = stock.get("result_age_days")
        if _age is not None and pd.notna(_age):
            _stale_badge = _badge(f"⏳ Stale {int(_age)}d", COLORS["orange"])

    # Recently-listed marker. The engine now treats a share-count delta as UNMEASURABLE when the
    # prior-year count predates listing (core/data_engine._prelisting_baseline), and the share row
    # above already prints "(pre-listing baseline)" instead of a dilution percentage — but nothing
    # on the page said WHY, so the effect was visible and its cause was not. This closes that, and
    # carries its own weight besides: a business listed 74 days ago has no public track record,
    # whatever its scores say.
    #
    # CONDITIONAL, NEVER A RAW COLUMN, AND THAT IS THE WHOLE DESIGN. listed_days is CENSORED at the
    # top — 1,167 stocks (43%) sit at exactly 3998 — so rendering it as "company age" would be a
    # lie for nearly half the universe. It is trustworthy only at the young end (min 10 days, a
    # clean gradient), which is exactly the end this shows. Fires on 15.9% of the universe.
    #
    # The bound is IMPORTED, never retyped: re-deriving a threshold in the UI is the drift that
    # produced the "Dilution 48.6%" contradiction this very card was fixed for.
    _listed_badge = ""
    _ld = stock.get("listed_days")
    if _ld is not None and pd.notna(_ld) and float(_ld) < PRELISTING_BASELINE_DAYS:
        _listed_badge = _badge(f"🆕 Listed {int(float(_ld))}d" + help_chip("Recently Listed"),
                               COLORS["text_secondary"])

    # Cyclicality context — a-priori business type (industry tier) + realized 5Y earnings drawdown.
    # NEUTRAL by design: cyclical ≠ bad, it's a holding-regime hint (timing-overlay vs hold-through-
    # cycle). NaN-safe: earn-DD appended only when ≥4 of 6 PAT years exist. No threshold, no coloring
    # — the "Defensive-but-cyclical" flag idea was vetoed by census (fired 40% of C/D = noise).
    # Hidden on legacy cached frames lacking the engine column (mirrors the Evidence badge).
    _cyc_badge = ""
    _cyc_tier = stock.get("cyclicality_tier")
    if _cyc_tier is not None and pd.notna(_cyc_tier):
        _cyc_txt = f"🔄 {_esc(str(_cyc_tier))}"
        _cyc_code = str(stock.get("cyclicality_tier_code", "") or "")
        if _cyc_code:
            _cyc_txt += f" ({_esc(_cyc_code)})"
        _cyc_dd = stock.get("max_earnings_drawdown_5y")
        if _cyc_dd is not None and pd.notna(_cyc_dd):
            _cyc_txt += f" · earn-DD {float(_cyc_dd) * 100:.0f}%"
        _cyc_badge = _badge(_cyc_txt + help_chip("Cyclicality Tier"), COLORS["text_secondary"])

    # Per-stock TREND badge — the actionable per-stock counterpart to the market-wide regime badge
    # (which is identical on every tearsheet). Composes weinstein_stage DIRECTION + d45_trend_structure
    # STRENGTH (x/5) + the trend_modifier path chip (↩️ Pullback / 🚀 Breakout / ⚠️ Bounce / ⚠️ Extended).
    # Color tracks the stage (Stage 2 green / Stage 4 red / else gold). Hidden on legacy cached frames
    # lacking the column (mirrors the Cyclicality / Evidence badges).
    _trend_badge = ""
    _tr_stage = stock.get("weinstein_stage")
    if _tr_stage is not None and pd.notna(_tr_stage) and str(_tr_stage) != "❔ Unknown":
        _tr_txt = _esc(str(_tr_stage))
        _tr_str = stock.get("d45_trend_structure")
        if _tr_str is not None and pd.notna(_tr_str):
            _tr_txt += f" · {int(_tr_str)}/5"
        _tr_mod = str(stock.get("trend_modifier", "") or "")
        if _tr_mod:
            _tr_txt += f" · {_esc(_tr_mod)}"
        _tr_clr = (COLORS["green"] if "Stage 2" in str(_tr_stage)
                   else COLORS["red"] if "Stage 4" in str(_tr_stage)
                   else COLORS["gold"])
        _trend_badge = _badge(_tr_txt + help_chip("Weinstein Stage"), _tr_clr)

    badges_html = (
        _badge(f"{tcfg['emoji']} {tcfg['label']}{help_chip('Conviction Tier')}", ring_clr) +
        _trend_badge +
        _mg_badge +
        _badge(f_txt, f_clr) +
        _badge(reg_txt, reg_clr) +
        _gov_badge +
        _cov_badge +
        _stale_badge +
        _listed_badge +
        _cyc_badge
    )

    st.markdown(f"""
    <div class="ts-hero">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:20px;position:relative;">
        <!-- Identity column -->
        <div style="flex:1;min-width:0;">
          <div style="font-size:0.65rem;color:{COLORS['text_muted']};letter-spacing:2px;
                      text-transform:uppercase;margin-bottom:5px;">
            #{rank} &nbsp;·&nbsp; {sector} &nbsp;·&nbsp; {mcat}
          </div>
          <div style="font-size:2.1rem;font-weight:900;color:{COLORS['text_primary']};
                      line-height:1.1;word-break:break-word;">{name}</div>
          <div style="font-size:0.78rem;color:{COLORS['text_muted']};margin-top:4px;">
            {industry} &nbsp;·&nbsp; ₹{mcap:,.0f} Cr{px_chip}{liq_chip}
          </div>
          <div style="margin-top:12px;">{badges_html}</div>
        </div>
        <!-- Score ring -->
        <div class="ts-score-ring" style="border-color:{ring_bdr};
             box-shadow:0 0 28px {ring_bdr},inset 0 0 20px rgba(0,0,0,0.4);">
          <div class="ts-score-val" style="color:{ring_clr};">{comp:.0f}</div>
          <div class="ts-score-lbl" style="color:{ring_clr};">/ 100{help_chip("Composite Score")}</div>
        </div>
      </div>
      <!-- Pills row -->
      <div style="margin-top:16px;border-top:1px solid {COLORS['border']};
                  padding-top:12px;line-height:2.2;">
        {pills_html if pills_html else
         f'<span style="font-size:0.7rem;color:{COLORS["text_muted"]};">No active catalyst or framework signals</span>'}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SCORE STRIP — 5-score visual overview
# ═══════════════════════════════════════════════════════════════

def render_score_strip(stock: pd.Series):
    """
    Horizontal 5-cell strip: Moat · Growth · Cash · Momentum · Governance.
    Each cell has big number + colored mini progress bar.
    """
    moat  = float(_g(stock, "moat_score",       0))
    grow  = float(_g(stock, "growth_score",      0))
    cash  = float(_g(stock, "cash_score",        0))
    mom   = float(_g(stock, "momentum_score",    0))
    gov   = float(_g(stock, "governance_bonus",  0))

    def _cell(label: str, icon: str, val: float, color: str) -> str:
        w   = max(0.0, min(100.0, val))
        neg = val < 0
        disp = f"{val:+.0f}" if neg else f"{val:.0f}"
        # Band on the DISPLAYED number: 69.67 and 70.35 both print "70", so both must read
        # Strong/green. The zone WORD is classified here too — it had the same split.
        _v   = _shown(val, 0)
        zone = ("Strong" if _v >= 70 else "Average" if _v >= 40 else "Weak") if not neg else "Penalty"
        zone_clr = (COLORS["green"] if _v >= 70 else
                    COLORS["gold"]  if _v >= 40 else COLORS["red"])
        return (
            f'<div class="ts-score-cell" style="border-top:3px solid {color};">'
            f'<div class="ts-score-cell-lbl">{icon} {label}{help_chip(f"{label} Score")}</div>'
            f'<div class="ts-score-cell-val" style="color:{color};">{disp}</div>'
            f'<div class="ts-score-bar-bg"><div class="ts-score-bar-fill" '
            f'style="width:{w:.1f}%;background:{color};"></div></div>'
            f'<div style="font-size:0.52rem;color:{zone_clr};margin-top:4px;'
            f'text-transform:uppercase;letter-spacing:0.6px;font-weight:700;">{zone}</div>'
            f'</div>'
        )

    cells = (
        _cell("Moat",       "🏰", moat, COLORS["purple"]) +
        _cell("Growth",     "🌱", grow, COLORS["green"])  +
        _cell("Cash",       "💵", cash, COLORS["blue"])   +
        _cell("Momentum",   "⚡", mom,  COLORS["orange"]) +
        _cell("Governance", "👑", gov,  COLORS["gold"])
    )
    st.markdown(f'<div class="ts-score-strip">{cells}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SELL ALERTS PANEL — Prominent multi-alert display
# ═══════════════════════════════════════════════════════════════

def render_sell_alerts_panel(stock: pd.Series):
    """
    Renders sell alert banners prominently.
    Shows a green confirmation when no alerts are active — silence was confusing users.
    Shows each fired alert as a distinct colored banner with explanation.
    """
    has_any = int(_g(stock, "sell_alert_any", 0)) == 1
    if not has_any:
        st.markdown(f"""
        <div style="background:rgba(63,185,80,0.07);border:1px solid rgba(63,185,80,0.3);
                    border-radius:10px;padding:11px 18px;margin:4px 0 12px 0;
                    display:flex;align-items:center;gap:10px;">
          <span style="font-size:1rem;">✅</span>
          <div>
            <div style="font-size:0.78rem;font-weight:700;color:{COLORS['green']};">
              No Exit Signals Active
            </div>
            <div style="font-size:0.68rem;color:{COLORS['text_muted']};margin-top:1px;">
              All 6 Baid/Howard Marks/Mauboussin sell triggers checked — none fired.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    _ALERTS = [
        ("sell_alert_thesis_broken",    COLORS["red"],    "🔴",
         "Investment Thesis Broken",
         "ROCE trajectory is declining structurally — the business moat is eroding. "
         "Quality score is automatically penalized. Re-evaluate the competitive position."),
        ("sell_alert_mgmt_deteriorated", COLORS["orange"], "🟠",
         "Management Deterioration",
         "Pledge rising + promoter selling + D/E rising simultaneously — "
         "insider confidence is collapsing. Three independent signals, all firing together."),
        ("sell_alert_cash_collapse",     COLORS["red"],    "🔴",
         "Cash Quality Collapse",
         "CFO/PAT dropped below 50% — reported profits are no longer backed by cash. "
         "Baid's #1 red flag: earnings may be fictional."),
        ("sell_alert_overvalued",        COLORS["gold"],   "🟡",
         "Price Excess (Howard Marks)",
         "PEG > 2.5 or P/E > 30% above own 10Y median. Even great businesses are terrible "
         "investments at extreme prices. Howard Marks' extreme caution zone."),
        ("sell_alert_treadmill",         COLORS["orange"], "🟠",
         "Growth Treadmill (Mauboussin)",
         "P/E > 50× implies 15-20 year CAP assumption, but growth is decelerating AND ROCE "
         "is declining. The machine is priced for perfection and visibly slipping."),
        ("sell_alert_sequential_decline", COLORS["red"],   "🔴",
         "Sequential Revenue Collapse",
         "Current year revenue negative + 3Y CAGR also negative + PAT declining. "
         "Not a one-bad-year blip — this is structural multi-year collapse. Exit signal."),
    ]

    banners_html = ""
    for col, clr, sev, title, body in _ALERTS:
        if int(_g(stock, col, 0)) != 1:
            continue
        banners_html += (
            f'<div class="ts-sell-banner" style="background:{clr}0d;border-color:{clr}55;">'
            f'<div class="ts-sell-icon">{sev}</div>'
            f'<div>'
            f'<div class="ts-sell-title" style="color:{clr};">{title}</div>'
            f'<div class="ts-sell-body" style="color:{COLORS["text_secondary"]};">{body}</div>'
            f'</div>'
            f'</div>'
        )

    st.markdown(f"""
    <div style="background:rgba(248,81,73,0.06);border:1px solid rgba(248,81,73,0.4);
                border-radius:14px;padding:14px 18px;margin:8px 0 16px 0;">
      <div style="font-size:0.72rem;font-weight:800;letter-spacing:1.5px;color:{COLORS['red']};
                  text-transform:uppercase;margin-bottom:12px;">
        🚨 &nbsp;Sell Alert(s) Active — Review Before Holding
      </div>
      {banners_html}
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# RAW SIGNALS PANEL — Structured metric grid (replaces expander)
# ═══════════════════════════════════════════════════════════════

# Plain-language "?" help: the chip renderer (help_chip) and the _RAW_GLOSSARY single source of truth
# now live in ui/ui_components.py (which owns the .ts-help CSS). They are re-imported at the top of
# this module, so existing `from ui.ui_tearsheet import help_chip / _RAW_GLOSSARY` callers (incl. the
# scanner and the tests) keep resolving against the SAME objects — one definition, zero drift.


_CELL_OPEN = '<div class="ts-raw-cell"'      # split marker AND each cell's opening tag


def render_raw_signals(stock: pd.Series, query: str = ""):
    """
    Renders all raw numeric signals in a clean labeled grid.
    Grouped into logical clusters. Used inside the 'All Data' inner tab.
    Each label that appears in _RAW_GLOSSARY auto-renders a plain-language "?" hover tooltip.

    `query` filters the grid — 143 signals across 9 sections is the densest surface in the app and
    had no way to reach one except scrolling (the Deep Scanner and Reference tab both have search).
    Empty query renders exactly as before: every cell, every section.

    WHY A PARAMETER AND NOT A WIDGET HERE: this module is bound by the stateless contract
    (CLAUDE.md §5, pinned by tests/test_tearsheet_stateless_contract.py) — app.py owns
    st.session_state, so the text_input lives there and the string arrives as a plain argument.
    """
    # Token-AND, the same matching the Reference tab's search uses: every word must appear in the
    # cell's label OR its glossary text. Searching the MEANING is the point — "cost of equity"
    # finds CAP Years, whose label contains none of those words.
    _tokens = [t for t in (query or "").lower().split() if t]
    _hidden_sections: list = []
    _shown = [0]
    _total = [0]

    def _match(cell_html: str) -> bool:
        m = re.search(r'data-s="([^"]*)"', cell_html)
        return all(t in (m.group(1) if m else "") for t in _tokens)

    def _cell(label: str, val, fmt: str = "", help: str = "") -> str:
        if isinstance(val, float) and np.isnan(val):
            disp = "N/A"
        elif fmt:
            try:
                disp = fmt.format(val)
            except Exception:
                disp = str(val)
        else:
            disp = str(val) if val is not None else "N/A"
        # Plain-language "?" tooltip via the shared help_chip() (explicit help= overrides the glossary).
        help_html = help_chip(label, help)
        # data-s carries what search matches on: the label PLUS its tooltip text. An attribute
        # rather than a side-list because the 143 _cell() call sites concatenate into one string —
        # the marker has to travel WITH the cell.
        _hay = _esc(f"{label} {help or _RAW_GLOSSARY.get(label, '')}".lower())
        return (
            f'{_CELL_OPEN} data-s="{_hay}">'
            f'<div class="ts-raw-lbl">{_esc(label)}{help_html}</div>'
            f'<div class="ts-raw-val">{_esc(disp)}</div>'
            f'</div>'
        )

    def _section(title: str, color: str, cells_html: str):
        _parts = [p for p in cells_html.split(_CELL_OPEN) if p.strip()]
        _total[0] += len(_parts)
        if _tokens:
            _parts = [p for p in _parts if _match(_CELL_OPEN + p)]
            if not _parts:
                _hidden_sections.append(title)
                return
        _shown[0] += len(_parts)
        cells_html = "".join(_CELL_OPEN + p for p in _parts)
        st.markdown(
            f'<div style="font-size:0.7rem;font-weight:800;color:{color};'
            f'text-transform:uppercase;letter-spacing:1px;margin:18px 0 8px 0;">'
            f'{title}</div>'
            f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));'
            f'gap:6px;">{cells_html}</div>',
            unsafe_allow_html=True,
        )

    g = lambda k, d=0: _g(stock, k, d)

    # Quality
    _section("🏭 Business Quality", COLORS["purple"],
        _cell("ROCE Current",  g("roce"),           "{:.1f}%") +
        _cell("ROCE 10Y Med",  g("roce_med_10y"),   "{:.1f}%") +
        _cell("ROCE 5Y Med",   g("roce_med_5y"),    "{:.1f}%") +
        _cell("ROE Current",   g("roe"),            "{:.1f}%") +
        _cell("ROE 10Y Med",   g("roe_med_10y"),    "{:.1f}%") +
        _cell("NPM",           g("npm"),            "{:.1f}%") +
        _cell("NPM 5Y Med",    g("npm_med_5y"),     "{:.1f}%") +
        _cell("OPM",           g("opm"),            "{:.1f}%") +
        _cell("Malik Score",   g("malik_score"),    "{:.0f}/5") +
        _cell("Malik Pass",    "Yes ✅" if g("malik_pass") == 1 else "No", "") +
        _cell("Malik Label",   stock.get("malik_label", "") or "", "") +
        # Lynch Score/Pass live in their own 🚀 Lynch Fast Grower Pillars block below (de-duped).
        _cell("Piotroski",     g("piotroski_fscore"),"{:.0f}/9") +
        _cell("Fisher Scal. Score", g("fisher_score"),   "{:.0f}/4") +
        _cell("Fisher Quadrant",    stock.get("fisher_lifecycle_quadrant", "⚪ Laggard") or "⚪ Laggard", "") +
        # IBAS moat decomposition — the 4 sub-scores that average to the scorecard's "IBAS" aggregate
        # (Mukherjea's Intangibles/Brand/Architecture/Strategic-assets moat lens), previously orphaned.
        _cell("IBAS Architecture",   g("ibas_architecture_score"),     "{:.0f}") +
        _cell("IBAS Innovation",     g("ibas_innovation_score"),       "{:.0f}") +
        _cell("IBAS Reputation",     g("ibas_reputation_score"),       "{:.0f}") +
        _cell("IBAS Strategic",      g("ibas_strategic_assets_score"), "{:.0f}") +
        _cell("Moat Endurance",      stock.get("mef_label", "") or "", "") +  # widening / intact / eroding / degrading
        _cell("Moat Endur ×",        g("moat_endurance_factor"), "{:.2f}×") +  # current ÷ 10y-median ROCE
        _cell("Elite ROE",           "Yes ✅" if g("roe_elite_flag") == 1 else "No", "") +
        _cell("ROE Rising",          "Yes ✅" if g("roe_trend_rising_flag") == 1 else "No", "") +
        _cell("Mcap Tier",           stock.get("mcap_tier", "") or "", "") +
        _cell("Cyclicality Tier",    stock.get("cyclicality_tier", "") or "", "") +        # a-priori industry type (display-only)
        _cell("Earn Drawdown 5Y",    g("max_earnings_drawdown_5y", np.nan), "{:.0%}") +    # realized worst PAT peak-to-trough; N/A if <4y
        _cell("ROE Turnaround",      "Yes ✅" if g("roe_turnaround_flag") == 1 else "No", "") +
        _cell("Category Winner",     "Yes ✅" if g("category_winner_flag") == 1 else "No", "") +
        _cell("Enduring VC",         "Yes ✅" if g("enduring_vc_flag") == 1 else "No", "") +
        _cell("Compound Power",      "Yes ✅" if g("compound_growth_power_flag") == 1 else "No", "") +
        _cell("Steady in Volatile",  "Yes ✅" if g("consistent_in_volatile_flag") == 1 else "No", "") +
        _cell("QMOM Quality",        g("d51_qmom_quality_score"), "{:.2f}")
    )

    # Growth
    _section("🌱 Growth", COLORS["green"],
        _cell("PAT 5Y CAGR",   g("pat_gr_5y"),      "{:.1f}%") +
        _cell("PAT 3Y CAGR",   g("pat_gr_3y"),      "{:.1f}%") +
        # "PAT YoY" REMOVED 2026-08-26 — it rendered pat_gr_yoy, which is IDENTICAL to the
        # "Q PAT YoY" cell below on 2,029 of 2,085 rows (97.3%). Two cells, one number, and the
        # unprefixed label implied the ANNUAL year beside "PAT 5Y/3Y CAGR" — so a reader comparing
        # "PAT YoY 108.5%" against "3Y CAGR 39.0%" believed they were comparing annual to annual
        # when the first figure was a single quarter. The honestly-named Q cell survives; nothing
        # is lost. (`pat_gr_yoy` is quarterly under an annual-sounding name — docs/known-issues.md.)
        _cell("Rev 10Y CAGR",  g("rev_gr_10y"),     "{:.1f}%") +
        _cell("Rev 5Y CAGR",   g("rev_gr_5y"),      "{:.1f}%") +
        # Relabelled, NOT removed: q_rev_yoy is not rendered anywhere on this tab, so deleting
        # this cell would drop the number entirely. rev_gr_yoy is quarterly (identical to q_rev_yoy
        # on 97.3% of rows), so the "Q" prefix simply makes the existing label truthful.
        _cell("Q Rev YoY",     g("rev_gr_yoy"),     "{:.1f}%") +
        _cell("EPS 5Y CAGR",   g("eps_gr_5y"),      "{:.1f}%") +
        _cell("Q EPS YoY",     g("eps_gr_yoy"),     "{:.1f}%") +   # quarterly, like the two above
        _cell("Q PAT YoY",     g("q_pat_yoy"),      "{:.1f}%") +
        _cell("Op Leverage",   "Yes" if g("operating_leverage") == 1 else "No", "") +
        _cell("Lynch Category", stock.get("lynch_category", "") or "", "") +  # Fast Grower / Stalwart / Slow Grower / Turnaround
        _cell("Op Lev (3Y)",    g("ebit_vs_rev_spread_3y"), "{:.1f}%") +  # 3Y EBIT-minus-revenue growth spread; + = operating leverage
        _cell("PAT 1Y Δ %",     g("pat_decline_1y_pct"), "{:+.1f}%") +
        _cell("Value Migration", "Yes ✅" if g("value_migration_flag") == 1 else "No", "") +
        _cell("EPS Accelerating","Yes ✅" if g("eps_strong_acceleration") == 1 else "No", "") +
        _cell("UU Setup",        "Yes ✅" if g("uu_setup_flag") == 1 else "No", "") +
        _cell("Fast Creator",    "Yes ✅" if g("fast_creator_setup") == 1 else "No", "")
    )

    # Cash & Debt
    _section("💵 Cash & Debt", COLORS["blue"],
        _cell("CFO/PAT",       g("cfo_to_pat"),      "{:.1f}%") +
        _cell("FCF Yield",     g("fcf_yield"),       "{:.1f}%") +
        _cell("FCF/CFO",       g("fcf_to_cfo_pct"),  "{:.1f}%") +
        _cell("FCF/PAT",       g("d28_fcf_to_pat_pct"), "{:.1f}%") +
        # FCF provenance — tells a verifier the FCF above is NOT raw (imputed from OCF / reconstructed).
        _cell("FCF Imputed",      "Yes" if g("fcf_imputed_flag") == 1 else "No", "") +
        _cell("FCF Reconstructed","Yes" if g("fcf_reconstructed_flag") == 1 else "No", "") +
        _cell("SSGR",          g("ssgr"),            "{:.1f}%") +
        _cell("SSGR Cushion",  g("ssgr_cushion"),    "{:.1f}%") +
        _cell("D/E Ratio",     g("debt_to_equity"),  "{:.2f}") +
        _cell("Int Coverage",  g("interest_coverage"),"{:.1f}×") +
        _cell("Current Ratio", g("current_ratio"),   "{:.2f}") +
        _cell("Tax Rate Est",  g("tax_rate_est"),    "{:.1f}%") +
        _cell("Asset Growth",  g("asset_growth_yoy"), "{:.1f}%") +  # Capital cycle: low = disciplined
        _cell("CFROIC",        g("cfroic"),           "{:.1f}%") +  # Tortoriello: cash return on invested capital
        _cell("Ext Financing", g("external_financing_to_assets"), "{:.1f}%") +  # Tortoriello: neg = returning capital
        _cell("Capital Alloc", stock.get("capital_allocation_signal","") or "", "") +
        _cell("Sector Capital", stock.get("sector_capital_phase","") or "", "") +  # Chancellor sectoral cycle
        _cell("CWIP/FA %",      g("cwip_ratio"),             "{:.1f}%") +  # capital-work-in-progress ÷ fixed assets
        _cell("EBITDA→PAT Gap", g("ebitda_to_pat_gap_pct"),  "{:.1f}%") +  # (Dep+Int+Tax)/EBITDA — NOT a tax rate
        _cell("Supplier Float", g("supplier_float_score"),   "{:.0f}/100") +  # negative-CCC float moat
        _cell("Negative WC",    "Yes ✅" if g("negative_wc_flag") == 1 else "No", "")
    )

    # Valuation
    _section("💰 Valuation", COLORS["gold"],
        _cell("PE",            g("pe"),              "{:.1f}×") +
        _cell("Fair PE (QGLP)",g("fair_pe_qglp"),    "{:.1f}×") +
        _cell("Industry PE",   g("industry_pe"),     "{:.1f}×") +
        _cell("P/B",           g("price_to_book"),   "{:.2f}×") +
        _cell("P/S",           g("ps_ratio"),        "{:.2f}×") +
        _cell("FGV",           g("fgv_pct"),         "{:.0%}") +
        _cell("PEG",           g("peg"),             "{:.2f}") +
        _cell("PEG Zone",      stock.get("peg_zone","") or "N/A", "") +
        _cell("Earnings Yield",g("earnings_yield"),  "{:.1f}%") +
        _cell("Dividend Yield",g("dividend_yield", np.nan), "{:.2f}%") +   # vendor TTM yield; NaN default so an absent column reads N/A, never a fabricated 0.00%
        _cell("PE vs 10Y Med", g("pe_discount"),     "{:.1f}%") +
        _cell("EV/EBITDA Dir", g("ev_ebitda_direction"), "{:.2f}") +
        _cell("Payback Ratio", g("payback_ratio"),   "{:.1f}y") +
        _cell("P/E vs ROE MoS",g("pe_vs_roe_mos"),  "{:.1f}") +
        _cell("Valuation Scr", g("valuation_score"), "{:.0f}/100") +
        _cell("O'Shaughnessy VC", g("oshaughnessy_value_composite"), "{:.0f}/100") +  # 5-factor value composite
        _cell("Trending Value", "Yes ✅" if g("trending_value_flag") == 1 else "No", "") +  # cheap + 6M momentum
        _cell("Buy Zone",      stock.get("buy_zone_label","") or "", "") +
        _cell("Payoff Ratio",  g("payoff_ratio_proxy"),    "{:.2f}×") +  # Mauboussin upside/downside payoff
        _cell("Exp Gap Rank",  g("expectations_gap_rank"), "{:.0f}/100")  # market-implied expectations gap rank
    )

    # Ownership — 3Y promoter change shown because the governance engine itself
    # gates on it (accumulation > +3 / exit < −5): the user must see what the shield sees.
    _section("👥 Ownership & Governance", COLORS["orange"],
        _cell("Promoter %",    g("promoter_holdings"),    "{:.1f}%") +
        _cell("Pledge %",      g("pledged_percentage"),   "{:.1f}%") +
        _cell("FII %",         g("fii_holdings"),         "{:.1f}%") +
        _cell("DII %",         g("dii_holdings"),         "{:.1f}%") +
        _cell("Promoter Chg", g("change_promoter_lq"),   "{:+.1f}%") +
        _cell("Promoter 3Y Δ", g("change_promoter_3y"),   "{:+.1f}%") +
        _cell("FII Chg",       g("change_fii_lq"),        "{:+.1f}%") +
        _cell("DII Chg",       g("change_dii_lq"),        "{:+.1f}%") +
        _cell("Smart Money",   stock.get("smart_money_flow","") or "","") +
        _cell("Gov Bonus",     g("governance_bonus"),     "{:.0f}") +
        _cell("Mgmt Integrity",g("management_integrity_score"),"{:.0f}/3") +
        _cell("Dilution Flag", "Yes ⚠️" if g("dilution_flag") == 1 else "Clean ✅","") +
        _cell("Pledge Re-rate","Yes ✅" if g("pledge_rerate_catalyst") == 1 else "No","")
    )

    # Technical
    _section("⚡ Technical & Momentum", COLORS["cyan"],
        _cell("CRS 50D",       g("crs_50d"),         "{:.0f}") +
        _cell("CRS 26W",       g("crs_26w"),         "{:.0f}") +
        _cell("CRS 52W",       g("crs_52w"),         "{:.0f}") +
        _cell("RS Composite",  g("d47_rs_composite"),"{:.1f}") +
        _cell("RSI 14D",       g("rsi_14d"),         "{:.1f}") +
        _cell("Vol Ratio",     g("vol_ratio"),       "{:.2f}×") +
        _cell("Dist 52WH",     g("dist_52wh"),       "{:.1f}%") +
        _cell("VSTOP Green",   "Yes ✅" if g("vstop_green") == 1 else "No","") +
        _cell("Breakout Scr",  g("breakout_score"),  "{:.0f}") +
        _cell("Momentum Scr",  g("momentum_score"),  "{:.0f}/100") +
        _cell("Weinstein Stage", stock.get("weinstein_stage","") or "", "") +  # 30W-MA stage analysis
        _cell("Trend Score",   g("trend_score"),  "{:.0f}/100")  # SMA200 dir + VSTOP + ADX + RSI zone + golden cross
    )

    # Forensic flags summary
    _section("🔬 Forensic Summary", COLORS["red"],
        _cell("Red Flags",     g("red_flag_count"),      f"{{:.0f}}/{FORENSIC_MAX_FLAGS}") +
        _cell("Forensic Scr",  g("forensic_score"),      "{:.0f}/100") +
        _cell("Forensic Mult", g("forensic_multiplier"), "{:.0%}") +
        _cell("Accruals Ratio",g("accruals_ratio"),      "{:.2f}") +  # Sloan accruals — negative = conservative
        # Piotroski shown in 🏭 Business Quality; EP Quintile in 🏛️ MOSL Signals (both de-duped).
        _cell("Econ Profit",   g("economic_profit", float("nan")), "₹{:,.0f} Cr") +
        _cell("EP Spread",     g("economic_profit_spread", float("nan")), "{:.1f}%") +  # ROIC − WACC spread (EP per capital)
        _cell("Earnings Power",stock.get("earnings_power_box","") or "","") +  # Heiserman defensive×enterprising box

        _cell("QGLP Score",    g("qglp_score"),          "{:.0f}/100") +
        _cell("QGLP Pass",     "Yes ✅" if g("qglp_pass") == 1 else "No","") +
        _cell("Composite Scr", g("composite_score"),     "{:.0f}/100") +
        _cell("Conviction Tier",g("conviction_tier"),    "Tier {:.0f}") +
        _cell("Diamond Flags",  g("dm_forensic_flag_count"), "{:.0f}") +  # Mukherjea 'Diamonds' forensic checks fired
        _cell("Cyclical Mirage", "Yes ⚠️" if g("cyclical_mirage_flag") == 1 else "No", "") +
        _cell("Dilution Vampire","Yes ⚠️" if g("dilution_vampire_flag") == 1 else "Clean ✅", "")
    )

    # MOSL Wealth Creation signals (9 Annual Wealth Creation Studies extracted into the engine)
    _yn = lambda k: "Yes ✅" if g(k) == 1 else "No"
    _section("🏛️ MOSL Wealth Creation Signals", COLORS["gold"],
        # 💹 The wealth tier + its two normalized inputs (engine columns, verdict_engine.py).
        # Here and not in a new section: the tier IS the Wealth Creation studies' formula
        # (ROCE expansion + earnings growth) operationalized, so this is its conceptual home.
        _cell("Wealth Tier",     (str(stock.get("wealth_tier","") or "N/A")
                                  + (" ⚠" if int(stock.get("wealth_warn",0) or 0)==1 else "")), "") +
        _cell("Wealth EP%",      g("wealth_ep_pct"),  "{:+.1f}") +
        _cell("Wealth Vel%",     g("wealth_vel_pct"), "{:+.1f}") +
        # 13th — Great/Good/Gruesome taxonomy
        _cell("Corporate Class", stock.get("corporate_class","") or "N/A", "") +
        # 17th — Economic Moat persistence (sector-relative ROE across 5 timeframes)
        _cell("EMC Sector-Beat",  g("emc_sector_beat_count"), "{:.0f}/5") +
        _cell("EMC Flag",         _yn("emc_flag"), "") +
        # 22nd — CAP & GAP longevity (duration above cost of capital / 15% growth)
        _cell("CAP Years",        g("cap_years_proxy"), "{:.0f}/5") +
        _cell("GAP Years",        g("gap_years_proxy"), "{:.0f}/3") +
        _cell("CAP-GAP Score",    g("cap_gap_score"),   "{:.0f}/4") +
        # 27th — Consistents vs Volatiles
        _cell("Consistency Champ", _yn("consistency_champion"), "") +
        _cell("PAT Falls >10%",   g("pat_decline_count_5y"), "{:.0f}/5") +
        _cell("Volatile Flag",    _yn("mosl_volatile_flag"), "") +
        # 28th — EP Power Curve quintile (1=highest EP)
        # ep_quintile is a float64, so it printed "1.0". Everywhere else the quintile reads "Q1".
        _cell("EP Quintile",      (f"Q{int(float(stock['ep_quintile']))}"
                                   if pd.notna(stock.get("ep_quintile")) else "N/A"), "") +
        _cell("EP Top Q1/Q2",     _yn("ep_top_quintile_flag"), "") +
        # 14th — Winner Category (sector tailwind) × Category Winner (leader)
        _cell("Winner Category",  _yn("winner_category_flag"), "") +
        _cell("Sector Leader",    g("sector_leader_score"), "{:.0f}") +  # leadership rank within its own sector (0-100)
        _cell("Winning Invest.",  _yn("category_winner_in_winner_cat"), "") +
        # 19th — 100x candidate (SQGLP, small-cap) + 20th — Mid→Mega (MQGLP, mid-cap rank 101-300)
        _cell("100x Candidate",   _yn("mosl_100x_candidate"), "") +
        _cell("Mid→Mega",         _yn("mid_to_mega_candidate"), "") +
        # 29th — Bruised Blue Chip (P/B < 2x quality fallen)
        _cell("Bruised Blue Chip", _yn("bruised_blue_chip_29"), "") +
        # 23rd — Growth-Value trap (growth + ROE < cost of equity)
        _cell("Growth-Value Trap", _yn("growth_value_trap"), "") +
        # 9th — Cyclical peak trap (commodity at peak-cycle deceptive low P/E)
        _cell("Cyclical Peak Trap", _yn("cyclical_peak_trap"), "") +
        # 26th — Atoms vs Bits business design + PSG (Price/Sales-to-Growth), the study's
        # signature valuation lens for Bits cos whose PE/PEG mislead under "optical losses".
        # Study uses PSG peer-relative ("compared with suitable peers") — NO absolute cutoff,
        # so we show the raw value for cross-stock comparison, not a fabricated verdict.
        _cell("Atoms/Bits",       stock.get("atoms_to_bits_label","") or "N/A", "") +
        _cell("PSG",              g("psg_ratio"),       "{:.2f}") +
        _cell("SQGLP Score",      g("sqglp_score"),     "{:.0f}/5") +    # Size+Quality+Growth+Longevity+Price
        _cell("QV Score",         g("vqs_score"),       "{:.0f}") +      # Gray quantitative value-quality composite
        _cell("Sector Type",      stock.get("sector_consistent_type","") or "", "")  # Consistent vs Volatile sector
    )

    # Data Trust — meta-signals on how COMPLETE & FRESH this row's inputs are. Mirrors the verdict
    # band's 🔍 Evidence badge + ⏳ Stale chip, surfaced here so the raw-data view answers "how much do
    # I trust the numbers above?" (coverage % of the 44 core scoring inputs; days since last result).
    # NaN-safe: np.nan default → "N/A" (never a fabricated 0%/0d) when the engine column is absent.
    _section("🔍 Data Trust", COLORS["blue"],
        _cell("Evidence Coverage", g("data_coverage_pct", np.nan), "{:.0f}%") +
        _cell("Coverage Label",    stock.get("data_coverage_label", "") or "", "") +
        _cell("Result Age",        g("result_age_days", np.nan), "{:.0f}d") +
        _cell("Result Stale",      "Yes ⏳" if g("result_stale_flag") == 1 else "No", "")
    )

    # Framework PILLAR breakdowns (Dorsey / Outsider / Marks / Lynch / Mauboussin / CAN SLIM /
    # SEPA) deliberately live in the 🏛️ Frameworks tab as radars (render_*_radar) — shown there
    # with labels + thresholds + context the bare ✅/❌ grid lacked. Every underlying column still
    # ships in the Export below, so nothing is lost. This keeps All Data = raw fundamental +
    # engine signals (Business Quality → MOSL), not a second, worse copy of the Frameworks tab.


# ═══════════════════════════════════════════════════════════════
# CAN SLIM® TACTICAL MOMENTUM RADAR — O'Neil
# ═══════════════════════════════════════════════════════════════


    # Search feedback. Naming the sections that matched NOTHING is the useful half — it says where
    # the signal you wanted is not, rather than leaving a silently short page.
    if _tokens:
        if _shown[0] == 0:
            st.markdown(
                f'<div style="font-size:0.72rem;color:{COLORS["text_muted"]};padding:12px 0;">'
                f'No signal matches <b>{_esc(query)}</b> — searched {_total[0]} signal names '
                f'and their descriptions.</div>',
                unsafe_allow_html=True,
            )
        else:
            _hid = (f' &nbsp;·&nbsp; no match in {" · ".join(_esc(h) for h in _hidden_sections)}'
                    if _hidden_sections else "")
            st.markdown(
                f'<div style="font-size:0.66rem;color:{COLORS["text_muted"]};margin-top:12px;">'
                f'Showing <b>{_shown[0]}</b> of {_total[0]} signals for '
                f'<b>{_esc(query)}</b>{_hid}</div>',
                unsafe_allow_html=True,
            )

def render_canslim_radar(stock: pd.Series):
    """
    Renders William O'Neil's 7-pillar CAN SLIM tactical momentum radar panel.
    PURE DISPLAY — Reads pre-materialized binary pillar columns from scoring_engine.py.
    Zero threshold re-computation; zero scoring logic; immune to parameter drift.
    """
    st.markdown("<div class='sec-head'>📊 CAN SLIM® Tactical Momentum Radar</div>",
                unsafe_allow_html=True)

    cs_pass  = int(_g(stock, "can_slim_pass", 0))
    cs_score = int(_g(stock, "can_slim_score", 0))
    regime   = str(stock.get("market_regime", "SIDEWAYS") or "SIDEWAYS").upper()

    pillars = [
        ("C", "Current Earnings", int(_g(stock, "can_slim_c", 0)) == 1,
         "Quarterly EPS & Sales Growth ≥ 25% YoY"),
        ("A", "Annual Growth",    int(_g(stock, "can_slim_a", 0)) == 1,
         "5Y EPS CAGR ≥ 25% · ROE ≥ 17% · 3Y Unbroken Step"),
        ("N", "New Breakout",     int(_g(stock, "can_slim_n", 0)) == 1,
         "Price Within 15% of 52-Week High"),
        ("S", "Supply & Demand",  int(_g(stock, "can_slim_s", 0)) == 1,
         "Breakout Session Volume Surge ≥ 1.5×"),
        ("L", "Leader / Laggard", int(_g(stock, "can_slim_l", 0)) == 1,
         "IBD RS Composite Percentile Rank ≥ 80"),
        ("I", "Institutional",    int(_g(stock, "can_slim_i", 0)) == 1,
         "Active Smart-Money Inflow (FII or DII +)"),
        ("M", "Market Direction", int(_g(stock, "can_slim_m", 0)) == 1,
         f"Regime: {_esc(regime)} — PAUSED if BEAR"),
    ]

    vcp_active = int(_g(stock, "can_slim_vcp",      0)) == 1
    rs_active  = int(_g(stock, "can_slim_rs_trend", 0)) == 1

    hdr_color  = COLORS["blue"] if cs_pass else COLORS["text_muted"]
    status_msg = "🟢 PASSED BREAKOUT GATE" if cs_pass else "⚪ Tactical Sieve Hold"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);
                border:1px solid {COLORS['border']};border-top:3px solid {hdr_color};
                border-radius:12px;padding:14px 18px;margin-bottom:12px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
          <div style="font-size:0.95rem;font-weight:800;color:#e6edf3;">
            O'Neil Momentum Compliance Profile
          </div>
          <div style="font-size:0.72rem;color:#8b949e;margin-top:2px;">
            Status: <strong style="color:{hdr_color};">{status_msg}</strong>
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:1.5rem;font-weight:900;color:{hdr_color};line-height:1.0;">
            {cs_score}
            <span style="font-size:0.85rem;color:#8b949e;font-weight:400;">&thinsp;/ 17</span>
          </div>
          <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                      letter-spacing:0.5px;margin-top:2px;">Total Tactical Components</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    grid_html = ""
    for letter, title, passed, baseline in pillars:
        # Critical pillars (C, A, M) get red on failure; others get muted grey
        clr = COLORS["green"] if passed else ("#f85149" if letter in ("C", "A", "M") else "#8b949e")
        bg_opacity = "15" if passed else "08"
        ico = "✅" if passed else "❌"

        desc = baseline
        if letter == "S" and vcp_active:
            desc += " <span style='color:#bc8cff;'>(🔥 VCP Dryup)</span>"
        if letter == "L" and rs_active:
            desc += " <span style='color:#58a6ff;'>(🔥 RS Uptrend)</span>"

        grid_html += (
            f"<div style='background:{clr}{bg_opacity};border:1px solid {clr}40;"
            f"border-radius:8px;padding:10px;text-align:center;min-width:110px;flex:1;'>"
            f"<div style='font-size:1.6rem;font-weight:900;color:{clr};line-height:1.1;'>{letter}</div>"
            f"<div style='font-size:0.68rem;font-weight:700;color:#e6edf3;margin-top:4px;"
            f"white-space:nowrap;'>{_esc(title)}</div>"
            f"<div style='font-size:0.58rem;color:#8b949e;margin-top:2px;line-height:1.2;'>{desc}</div>"
            f"<div style='font-size:1.0rem;margin-top:4px;'>{ico}</div>"
            f"</div>"
        )

    st.markdown(
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;'>{grid_html}</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# SCHILIT FORENSIC SHIELD — Accounting Shenanigans Audit
# ═══════════════════════════════════════════════════════════════

def _get_schilit_context(stock: pd.Series, checker_col: str) -> str:
    """Returns a compact metric string to show beneath an active Schilit flag."""
    # HUMAN LABELS (fixed 2026-08-24): these cards printed raw internal column names —
    # "accruals_warning: 1 · inv_gap: 30.2%", "high_cash_high_debt: 1" — developer snake_case
    # leaking into a user-facing panel, with booleans rendered as a bare "1" that means nothing
    # to a reader. Every other panel in the app uses plain-language labels; these now match.
    if checker_col == "schilit_ems_flag":
        aw  = int(_g(stock, "accruals_warning", 0))
        igp = _g(stock, "inv_vs_rev_gap", 0)
        return (f"Accruals warning: {'yes' if aw else 'no'}"
                f"  ·  Inventory outgrew revenue by {igp:.1f}%")
    if checker_col == "schilit_cfs_flag":
        pg  = _g(stock, "pat_gr_yoy",  0)
        ocf = _g(stock, "ocf_growth",  0)
        return f"Profit growth {pg:+.1f}%  ·  Operating cash flow {ocf:+.1f}%"
    if checker_col == "schilit_kms_lev_flag":
        hcd = int(_g(stock, "high_cash_high_debt", 0))
        return ("Holding high cash AND high debt at once" if hcd else
                "Leverage structure flagged")
    if checker_col == "schilit_kms_bloat_flag":
        dso = _g(stock, "dso_delta_3y",           0)
        idc = _g(stock, "inventory_days_change",  0)
        return (f"Receivable days {dso:+.0f}d over 3Y  ·  Inventory days {idc:+.0f}d YoY")
    return ""


def render_schilit_shield(stock: pd.Series):
    """
    Renders Howard Schilit's Financial Shenanigans Accounting Audit Shield.
    PURE DISPLAY — Reads pre-materialized boolean flags from forensic_engine.py.
    All 6 column names verified against forensic_engine.py lines 820-826.
    """
    st.markdown("<div class='sec-head'>🛡️ Schilit Accounting Anomaly Shield</div>",
                unsafe_allow_html=True)

    f_score = _g(stock, "schilit_forensic_score", 100.0)
    f_pass  = int(_g(stock, "schilit_pass", 1))

    checkers = [
        ("EMS Anomaly Gimmick",
         int(_g(stock, "schilit_ems_flag",       0)) == 1,
         "schilit_ems_flag",
         "Revenue Recognition / Expense Capitalization Metrics"),
        ("CFS Cash Flow Trap",
         int(_g(stock, "schilit_cfs_flag",       0)) == 1,
         "schilit_cfs_flag",
         "Operating Cash Divergence / Paper Profit Shifts"),
        ("KMS Leverage Mirage",
         int(_g(stock, "schilit_kms_lev_flag",   0)) == 1,
         "schilit_kms_lev_flag",
         "Off-Balance Sheet Guarantees & Pledged Cash Mismatches"),
        ("KMS Operational Bloat",
         int(_g(stock, "schilit_kms_bloat_flag", 0)) == 1,
         "schilit_kms_bloat_flag",
         "Channel Stuffing / Asset Aging Accumulation"),
    ]

    shield_clr  = COLORS["green"] if f_pass else COLORS["red"]
    status_txt  = ("🛡️ PERIMETER SECURE — CLEAN AUDIT"
                   if f_pass else
                   "🚨 COGNITIVE RISK CAPTURED — SHENANIGAN ALERT")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);
                border:2px solid {shield_clr}40;border-left:6px solid {shield_clr};
                border-radius:12px;padding:14px 20px;margin-bottom:14px;
                box-shadow:0 4px 20px rgba(0,0,0,0.2);">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:15px;">
        <div>
          <div style="font-size:0.65rem;color:#8b949e;letter-spacing:1.5px;
                      text-transform:uppercase;">Accounting Security Shield</div>
          <div style="font-size:1.15rem;font-weight:900;color:{shield_clr};margin-top:2px;">
            {status_txt}
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:1.6rem;font-weight:900;color:{shield_clr};line-height:1.0;">
            {f_score:.0f}
            <span style="font-size:0.85rem;color:#8b949e;font-weight:400;">&thinsp;/ 100</span>
          </div>
          <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                      letter-spacing:0.5px;margin-top:2px;">Forensic Credibility</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    cells_html = ""
    for title, triggered, col_name, narrative in checkers:
        clr = COLORS["red"] if triggered else COLORS["green"]
        ico = "⚠️ Triggered" if triggered else "🎯 Clear"
        bg  = f"{clr}0d"
        bdr = f"{clr}30"

        ctx_val = _get_schilit_context(stock, col_name) if triggered else ""
        sub_desc = (
            f"<div style='font-size:0.58rem;color:{clr};font-weight:700;margin-top:2px;'>"
            f"{_esc(ctx_val)}</div>"
            if ctx_val else
            f"<div style='font-size:0.58rem;color:#8b949e;margin-top:2px;'>"
            f"{_esc(narrative)}</div>"
        )

        cells_html += (
            f"<div style='background:{bg};border:1px solid {bdr};border-radius:8px;"
            f"padding:10px;flex:1;min-width:220px;'>"
            f"<div style='display:flex;align-items:center;justify-content:space-between;'>"
            f"<span style='font-size:0.78rem;font-weight:800;color:#e6edf3;'>"
            f"{_esc(title)}</span>"
            f"<span style='font-size:0.68rem;font-weight:700;color:{clr};"
            f"background:{clr}15;padding:2px 8px;border-radius:12px;'>{ico}</span>"
            f"</div>"
            f"{sub_desc}"
            f"</div>"
        )

    st.markdown(
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;'>{cells_html}</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# MARK MINERVINI SEPA MOMENTUM RADAR — 7-Pillar Technical-Momentum Audit
# ═══════════════════════════════════════════════════════════════

def render_sepa_radar(stock: pd.Series):
    """
    Renders Mark Minervini's SEPA Momentum 7-pillar technical-momentum audit card.
    PURE DISPLAY — Reads pre-materialized binary pillar columns from scoring_engine.py.
    Zero threshold re-computation; zero scoring logic; immune to parameter drift.
    Source: docs/sepa_momentum_specs.json v1.0-sepa-momentum-codex.
    Pillars (T/A/L/R/E/I hard gates + V score bonus):
      T — Trend Template:  sepa_trend_template (5-pt Minervini Trend Template ≥ 4)
      A — ADX Confirmed:   sepa_adx_confirmed (ADX 14W ≥ 20 trend strength)
      L — Low Base:        sepa_low_base (≥ 30% above 52-week low — Criterion 6)
      R — RS Aligned:      sepa_rs_confirmed (all 3 CRS timeframes positive — Criterion 8)
      E — Earnings Fuel:   sepa_earnings_fuel (EPS ≥25% + Rev ≥20% + ROE ≥17%)
      I — Institutional:   sepa_institutional (FII or DII stake increasing QoQ)
      V — VCP Volume:      sepa_vcp_dryup (BONUS — 10D vol below 70% of 50D; never a hard gate)
    """
    st.markdown("<div class='sec-head'>⚡ Mark Minervini — SEPA Momentum Radar</div>",
                unsafe_allow_html=True)

    s_pass  = int(_g(stock, "sepa_pass",  0))
    s_score = int(_g(stock, "sepa_score", 0))

    # 6 hard-gate pillars (T/A/L/R/E/I) — green pass / red fail
    hard_pillars = [
        ("T", "Trend Template",
         int(_g(stock, "sepa_trend_template", 0)) == 1,
         "4 of 5 trend checks: price above its 30-week & 200-day averages, 30-week above 200-day, "
         "50-day above both, price above its stop"),
        ("A", "ADX Confirmed",
         int(_g(stock, "sepa_adx_confirmed", 0)) == 1,
         "Trend Strength Gate: ADX 14W ≥ 20 — confirmed directional trend"),
        ("L", "Low Base",
         int(_g(stock, "sepa_low_base", 0)) == 1,
         "Breakout Foundation: Price ≥ 30% above 52-week low (C6)"),
        ("R", "RS Aligned",
         int(_g(stock, "sepa_rs_confirmed", 0)) == 1,
         "Relative Strength: All 3 CRS timeframes beating Nifty 500 (C8)"),
        ("E", "Earnings Fuel",
         int(_g(stock, "sepa_earnings_fuel", 0)) == 1,
         "Fundamental Acceleration: EPS ≥25% + Rev ≥20% + ROE ≥17%"),
        ("I", "Institutional",
         int(_g(stock, "sepa_institutional", 0)) == 1,
         "Smart Money Entering: FII or DII quarterly stake increasing (REQ 7)"),
    ]

    _SEPA_BLUE = "#58a6ff"
    hdr_color  = _SEPA_BLUE if s_pass else COLORS["text_muted"]
    status_msg = "SEPA MOMENTUM CERTIFIED" if s_pass else "SEPA Criteria Not Met"

    # ── What certification also needs (2026-10-15): criterion 7 and the ₹500 Cr floor gate the pass but
    # have no pillar box, so six green boxes could read "Criteria Not Met" with nothing saying why (9 stocks
    # on the 14 Oct sheet). ✓/✗ are the engine's flags (sepa_near_high / sepa_mcap_ok), never re-judged
    # here; the stock's own numbers sit beside them, each printed on its mark's side of the line. Hidden on
    # a frame scored before the flags existed: never a ✗ for a gate that was not measured.
    _also_html = ""
    if all(k in stock.index for k in ("sepa_near_high", "sepa_mcap_ok")):
        def _gate(ok: bool, words: str, number: str) -> str:
            mark, clr = ("✓", "#3fb950") if ok else ("✗", "#f85149")
            return (f'<span style="color:{clr};font-weight:800;">{mark}</span> {_esc(words)} '
                    f'<span style="color:#6e7681;">({_esc(number)})</span>')
        _d52 = stock.get("dist_52wh")
        if pd.isna(_d52):
            _d52_txt = "not reported"
        else:
            _far = _same_side(float(_d52), lambda v: v <= TREND_TEMPLATE_MAX_BELOW_HIGH)
            _d52_txt = "at its high" if float(_far) <= 0 else f"{_far}% below"
        _mc = stock.get("market_cap")
        if pd.isna(_mc):
            _mc_txt = "not reported"
        else:
            _mc_s = _same_side(float(_mc), lambda v: v >= SEPA_MIN_MARKET_CAP_CR)
            _mc_txt = f"₹{float(_mc):,.{len(_mc_s.partition('.')[2])}f} Cr"
        _also_html = (
            '<div style="font-size:0.68rem;color:#8b949e;margin-top:4px;">Also required: '
            + _gate(int(_g(stock, "sepa_near_high", 0)) == 1,
                    f"within {TREND_TEMPLATE_MAX_BELOW_HIGH}% of its 52-week high", _d52_txt)
            + " · "
            + _gate(int(_g(stock, "sepa_mcap_ok", 0)) == 1,
                    f"market cap ₹{SEPA_MIN_MARKET_CAP_CR:,} Cr+", _mc_txt)
            + "</div>")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1117 0%,#0a1422 100%);
                border:1px solid {COLORS['border']};border-top:3px solid {hdr_color};
                border-radius:12px;padding:14px 18px;margin-bottom:12px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
          <div style="font-size:0.95rem;font-weight:800;color:#e6edf3;">
            Minervini SEPA Specific Entry Point Profile
          </div>
          <div style="font-size:0.72rem;color:#8b949e;margin-top:2px;">
            Status: <strong style="color:{hdr_color};">{_esc(status_msg)}</strong>
          </div>{_also_html}
        </div>
        <div style="text-align:right;">
          <div style="font-size:1.5rem;font-weight:900;color:{hdr_color};line-height:1.0;">
            {s_score}
            <span style="font-size:0.85rem;color:#8b949e;font-weight:400;">&thinsp;/ 7</span>
          </div>
          <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                      letter-spacing:0.5px;margin-top:2px;">SEPA Pillars (6 gates + VCP bonus)</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _sp_grid = ""
    for letter, title, passed, baseline in hard_pillars:
        clr_sp = _SEPA_BLUE if passed else "#f85149"
        bg_sp  = "18" if passed else "08"
        ico_sp = "✅" if passed else "❌"
        _sp_grid += (
            f"<div style='background:{clr_sp}{bg_sp};border:1px solid {clr_sp}40;"
            f"border-radius:8px;padding:10px;text-align:center;min-width:110px;flex:1;'>"
            f"<div style='font-size:1.6rem;font-weight:900;color:{clr_sp};line-height:1.1;'>"
            f"{_esc(letter)}</div>"
            f"<div style='font-size:0.68rem;font-weight:700;color:#e6edf3;margin-top:4px;"
            f"white-space:nowrap;'>{_esc(title)}</div>"
            f"<div style='font-size:0.58rem;color:#8b949e;margin-top:2px;line-height:1.2;'>"
            f"{_esc(baseline)}</div>"
            f"<div style='font-size:1.0rem;margin-top:4px;'>{ico_sp}</div>"
            f"</div>"
        )

    # Pillar V — VCP Volume (SCORE BONUS): never shown as a red fail. It scores the plain 10-day dry-up
    # at any stage; the cockpit's ⏳ Consolidation Base card tells a setup from a quiet decline.
    # Dried up (1) → green ⭐. Not (0) → amber ⏳, with a missing volume average named, never read as "no".
    # The line is the engine's (VCP_DRYUP_SHARE): this box said "10D < 50D" long after it moved to 70%.
    _vcp_on  = int(_g(stock, "sepa_vcp_dryup", 0)) == 1
    clr_v    = "#3fb950" if _vcp_on else "#e3b341"   # green active / amber forming (NOT red)
    bg_v     = "18" if _vcp_on else "12"
    ico_v    = "⭐" if _vcp_on else "⏳"
    sub_v    = (f"Volume dried up: 10-day average below {VCP_DRYUP_SHARE:.0%} of the 50-day"
                if _vcp_on else
                "Volume averages not reported (a bonus, not required)"
                if pd.isna(stock.get("vcp_volume_ratio")) else
                f"No dry-up: 10-day average at or above {VCP_DRYUP_SHARE:.0%} of the 50-day "
                f"(a bonus, not required)")
    _sp_grid += (
        f"<div style='background:{clr_v}{bg_v};border:1px dashed {clr_v}55;"
        f"border-radius:8px;padding:10px;text-align:center;min-width:110px;flex:1;'>"
        f"<div style='font-size:1.6rem;font-weight:900;color:{clr_v};line-height:1.1;'>V</div>"
        f"<div style='font-size:0.68rem;font-weight:700;color:#e6edf3;margin-top:4px;"
        f"white-space:nowrap;'>VCP Volume</div>"
        f"<div style='font-size:0.58rem;color:#8b949e;margin-top:2px;line-height:1.2;'>"
        f"{_esc(sub_v)}</div>"
        f"<div style='font-size:1.0rem;margin-top:4px;'>{ico_v}</div>"
        f"</div>"
    )

    st.markdown(
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;'>{_sp_grid}</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# PAT DORSEY WIDE MOAT RADAR — 5-Pillar Economic Moat Audit
# ═══════════════════════════════════════════════════════════════

def render_dorsey_radar(stock: pd.Series):
    """
    Renders Pat Dorsey's Wide Moat 5-pillar economic moat audit card.
    PURE DISPLAY — Reads pre-materialized binary pillar columns from scoring_engine.py.
    Zero threshold re-computation; zero scoring logic; immune to parameter drift.
    Source: docs/dorsey_moat_specs.json
    """
    st.markdown("<div class='sec-head'>🏰 Dorsey Wide Moat Radar</div>",
                unsafe_allow_html=True)

    d_pass  = int(_g(stock, "dorsey_pass",  0))
    d_score = int(_g(stock, "dorsey_score", 0))

    pillars = [
        ("M", "Moat Return Level",
         int(_g(stock, "dorsey_moat_level",    0)) == 1,
         "Wide Moat Return Hurdle: 10Y & 5Y ROCE ≥ 20%"),
        ("D", "Moat Trajectory",
         int(_g(stock, "dorsey_moat_direction", 0)) == 1,
         "Advantage Direction: Stable or Widening Trajectory"),
        ("V", "FCF Valuation Yield",
         int(_g(stock, "dorsey_fcf_valuation",  0)) == 1,
         "Margin of Safety Floor: Free Cash Flow Yield ≥ 5%"),
        ("Q", "Cash Realization Quality",
         int(_g(stock, "dorsey_cash_quality",   0)) == 1,
         "Earnings Conversion Base: CFO/PAT Conversion ≥ 80%"),
        ("C", "Capital Structure Cushion",
         int(_g(stock, "dorsey_cap_structure",  0)) == 1,
         "Leverage Cushion Guard: D/E < 1.0 / Financial Exempt"),
    ]

    hdr_color  = COLORS["purple"] if d_pass else COLORS["text_muted"]
    status_msg = "🏰 CONFIRMED WIDE MOAT" if d_pass else "⚪ Moat Unconfirmed"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);
                border:1px solid {COLORS['border']};border-top:3px solid {hdr_color};
                border-radius:12px;padding:14px 18px;margin-bottom:12px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
          <div style="font-size:0.95rem;font-weight:800;color:#e6edf3;">
            Pat Dorsey Economic Moat Compliance Profile
          </div>
          <div style="font-size:0.72rem;color:#8b949e;margin-top:2px;">
            Status: <strong style="color:{hdr_color};">{status_msg}</strong>
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:1.5rem;font-weight:900;color:{hdr_color};line-height:1.0;">
            {d_score}
            <span style="font-size:0.85rem;color:#8b949e;font-weight:400;">&thinsp;/ 5</span>
          </div>
          <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                      letter-spacing:0.5px;margin-top:2px;">Moat Gates Cleared</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    grid_html = ""
    for letter, title, passed, baseline in pillars:
        # All 5 pillars are equal weight — red on failure (each is a hard gate)
        clr = COLORS["purple"] if passed else "#f85149"
        bg_opacity = "15" if passed else "08"
        ico = "✅" if passed else "❌"

        grid_html += (
            f"<div style='background:{clr}{bg_opacity};border:1px solid {clr}40;"
            f"border-radius:8px;padding:10px;text-align:center;min-width:110px;flex:1;'>"
            f"<div style='font-size:1.6rem;font-weight:900;color:{clr};line-height:1.1;'>{letter}</div>"
            f"<div style='font-size:0.68rem;font-weight:700;color:#e6edf3;margin-top:4px;"
            f"white-space:nowrap;'>{_esc(title)}</div>"
            f"<div style='font-size:0.58rem;color:#8b949e;margin-top:2px;line-height:1.2;'>"
            f"{_esc(baseline)}</div>"
            f"<div style='font-size:1.0rem;margin-top:4px;'>{ico}</div>"
            f"</div>"
        )

    st.markdown(
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;'>{grid_html}</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# WILLIAM THORNDIKE OUTSIDER CEO RADAR — 4-Pillar Capital Allocation Audit
# ═══════════════════════════════════════════════════════════════

def render_outsider_radar(stock: pd.Series):
    """
    Renders William Thorndike Outsider CEO 4-pillar capital allocation audit card.
    PURE DISPLAY — Reads pre-materialized binary pillar columns from scoring_engine.py.
    Zero threshold re-computation; zero scoring logic; immune to parameter drift.
    Source: docs/outsider_specs.json
    """
    st.markdown("<div class='sec-head'>🏆 Thorndike Outsider CEO Radar</div>",
                unsafe_allow_html=True)

    o_pass  = int(_g(stock, "outsider_pass",  0))
    o_score = int(_g(stock, "outsider_score", 0))

    pillars = [
        ("S", "Share Retirement",
         int(_g(stock, "outsider_share_retirement", 0)) == 1,
         "Anti-Dilution Shield: share count stable or active repurchases"),
        ("D", "Debt Discipline",
         int(_g(stock, "outsider_debt_discipline",  0)) == 1,
         "Deleveraging Trend: 3Y D/E slope declining or stable"),
        ("C", "Cash Generation",
         int(_g(stock, "outsider_cash_generation",  0)) == 1,
         "Realization Floor: CFO/PAT conversion ≥ 85% (strictest base)"),
        ("R", "Capital Efficiency",
         int(_g(stock, "outsider_capital_returns",  0)) == 1,
         "Full-Cycle Returns: 10-Year ROCE Median ≥ 15%"),
    ]

    _OUTSIDER_GOLD = "#f0a500"
    hdr_color  = _OUTSIDER_GOLD if o_pass else COLORS["text_muted"]
    status_msg = "🏆 CONFIRMED OUTSIDER CEO" if o_pass else "⚪ Outsider Standard Not Met"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1117 0%,#1a1400 100%);
                border:1px solid {COLORS['border']};border-top:3px solid {hdr_color};
                border-radius:12px;padding:14px 18px;margin-bottom:12px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
          <div style="font-size:0.95rem;font-weight:800;color:#e6edf3;">
            Thorndike Capital Allocation Compliance Profile
          </div>
          <div style="font-size:0.72rem;color:#8b949e;margin-top:2px;">
            Status: <strong style="color:{hdr_color};">{status_msg}</strong>
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:1.5rem;font-weight:900;color:{hdr_color};line-height:1.0;">
            {o_score}
            <span style="font-size:0.85rem;color:#8b949e;font-weight:400;">&thinsp;/ 4</span>
          </div>
          <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                      letter-spacing:0.5px;margin-top:2px;">Capital Gates Cleared</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    grid_html = ""
    for letter, title, passed, baseline in pillars:
        clr        = _OUTSIDER_GOLD if passed else "#f85149"
        bg_opacity = "18" if passed else "08"
        ico        = "✅" if passed else "❌"

        grid_html += (
            f"<div style='background:{clr}{bg_opacity};border:1px solid {clr}40;"
            f"border-radius:8px;padding:10px;text-align:center;min-width:110px;flex:1;'>"
            f"<div style='font-size:1.6rem;font-weight:900;color:{clr};line-height:1.1;'>{letter}</div>"
            f"<div style='font-size:0.68rem;font-weight:700;color:#e6edf3;margin-top:4px;"
            f"white-space:nowrap;'>{_esc(title)}</div>"
            f"<div style='font-size:0.58rem;color:#8b949e;margin-top:2px;line-height:1.2;'>"
            f"{_esc(baseline)}</div>"
            f"<div style='font-size:1.0rem;margin-top:4px;'>{ico}</div>"
            f"</div>"
        )

    st.markdown(
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;'>{grid_html}</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# MARKS MARKET CYCLE & RISK DEFENSIVE SHIELD — Howard Marks
# ═══════════════════════════════════════════════════════════════

def render_marks_radar(stock: pd.Series):
    """
    Renders Howard Marks Market Cycle & Risk Defensive 4-pillar shield card.
    PURE DISPLAY — Reads pre-materialized binary pillar columns from scoring_engine.py.
    Zero threshold re-computation; zero scoring logic; immune to parameter drift.
    Source: docs/marks_cycle_specs.json v1.1-marks-cycle-codex-india-calibrated
    Thresholds (v1.1 India-calibrated): see docs/marks_cycle_specs.json for exact values.
    Companion Ch.9: D/E and CFO/PAT gates tightened for India defensive floor.
    """
    st.markdown("<div class='sec-head'>🛡️ Howard Marks Cycle & Risk Defensive Radar</div>",
                unsafe_allow_html=True)

    m_pass  = int(_g(stock, "marks_pass",  0))
    m_score = int(_g(stock, "marks_score", 0))

    pillars = [
        ("M", "Margin Extreme",
         int(_g(stock, "marks_margin_spike",   0)) == 1,
         "Pendulum Spike Guard: margins sit within sustainable historical limits"),
        ("P", "Price vs Value",
         int(_g(stock, "marks_price_value",    0)) == 1,
         "Asymmetry Margin: asset trades within a disciplined entry buy zone"),
        ("L", "Leverage Cushion",
         int(_g(stock, "marks_leverage_trap",  0)) == 1,
         "Risk Avoidance Line: balance sheet debt stays safely below caps"),
        ("D", "Defensive Cushion",
         int(_g(stock, "marks_defensive_base", 0)) == 1,
         "Margin for Error Base: CFO/PAT cash generation clears 70% floor"),
    ]

    _MARKS_CYAN = "#00CED1"
    hdr_color  = _MARKS_CYAN if m_pass else COLORS["text_muted"]
    status_msg = "🛡️ MARKS CYCLE SHIELD CONFIRMED" if m_pass else "⚪ Cycle Shield Not Cleared"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1117 0%,#001a1a 100%);
                border:1px solid {COLORS['border']};border-top:3px solid {hdr_color};
                border-radius:12px;padding:14px 18px;margin-bottom:12px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
          <div style="font-size:0.95rem;font-weight:800;color:#e6edf3;">
            Howard Marks Cycle Defence Compliance Profile
          </div>
          <div style="font-size:0.72rem;color:#8b949e;margin-top:2px;">
            Status: <strong style="color:{hdr_color};">{status_msg}</strong>
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:1.5rem;font-weight:900;color:{hdr_color};line-height:1.0;">
            {m_score}
            <span style="font-size:0.85rem;color:#8b949e;font-weight:400;">&thinsp;/ 4</span>
          </div>
          <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                      letter-spacing:0.5px;margin-top:2px;">Cycle Gates Cleared</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    grid_html = ""
    for letter, title, passed, baseline in pillars:
        clr        = _MARKS_CYAN if passed else "#f85149"
        bg_opacity = "18" if passed else "08"
        ico        = "✅" if passed else "❌"

        grid_html += (
            f"<div style='background:{clr}{bg_opacity};border:1px solid {clr}40;"
            f"border-radius:8px;padding:10px;text-align:center;min-width:110px;flex:1;'>"
            f"<div style='font-size:1.6rem;font-weight:900;color:{clr};line-height:1.1;'>{letter}</div>"
            f"<div style='font-size:0.68rem;font-weight:700;color:#e6edf3;margin-top:4px;"
            f"white-space:nowrap;'>{_esc(title)}</div>"
            f"<div style='font-size:0.58rem;color:#8b949e;margin-top:2px;line-height:1.2;'>"
            f"{_esc(baseline)}</div>"
            f"<div style='font-size:1.0rem;margin-top:4px;'>{ico}</div>"
            f"</div>"
        )

    st.markdown(
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;'>{grid_html}</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# DR. VIJAY MALIK PEACEFUL INVESTING RADAR — 5-Pillar Financial Quality Audit
# ═══════════════════════════════════════════════════════════════

def render_malik_radar(stock: pd.Series):
    """
    Renders Dr. Vijay Malik's Peaceful Investing 5-pillar financial quality audit card.
    PURE DISPLAY — Reads pre-materialized binary pillar columns from scoring_engine.py.
    Zero threshold re-computation; zero scoring logic; immune to parameter drift.
    Source: docs/malik_peaceful_specs.json v1.0-malik-peaceful-codex (all thresholds there).
    Pillars (G/P/F/C/S):
      G — Growth Runway:    malik_growth_runway    (Sales CAGR gate; see pillar_g_growth_runway)
      P — Margin Stability: malik_profit_stability (NPM stable gate; see pillar_p_profit_stability)
      F — Debt Fortress:    malik_debt_fortress    (ICR + D/E + CR; fin exempt; see pillar_f_debt_fortress)
      C — Cash Realization: malik_cash_generation  (CFO/PAT PERCENTAGE gate; see pillar_c_cash_generation)
      S — Self-Funded:      malik_self_funded      (SSGR binary flag; see pillar_s_self_funded)
    """
    st.markdown("<div class='sec-head'>🕊️ Dr. Vijay Malik — Peaceful Investing Radar</div>",
                unsafe_allow_html=True)

    m_pass  = int(_g(stock, "malik_pass",  0))
    m_score = int(_g(stock, "malik_score", 0))

    pillars = [
        ("G", "Growth Runway",
         int(_g(stock, "malik_growth_runway",    0)) == 1,
         "Sales Growth Hurdle: 10Y/5Y revenue CAGR clears the self-funding floor"),
        ("P", "Margin Stability",
         int(_g(stock, "malik_profit_stability", 0)) == 1,
         "Pricing Power Shield: current NPM stable and prior year not deteriorating"),
        ("F", "Debt Fortress",
         int(_g(stock, "malik_debt_fortress",    0)) == 1,
         "Leverage Cushion Gate: ICR, D/E, and Current Ratio all within safe range"),
        ("C", "Cash Realization",
         int(_g(stock, "malik_cash_generation",  0)) == 1,
         "Audited Reality Floor: operating cash flow substantively backs reported profit"),
        ("S", "Self-Funded Growth",
         int(_g(stock, "malik_self_funded",      0)) == 1,
         "Sustainable Growth Core: SSGR covers actual sales growth without new debt"),
    ]

    _MALIK_GREEN = "#2ecc71"
    hdr_color  = _MALIK_GREEN if m_pass else COLORS["text_muted"]
    status_msg = "PEACEFUL INVESTING CERTIFIED" if m_pass else "Peaceful Investing Criteria Not Met"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1117 0%,#0b1a0f 100%);
                border:1px solid {COLORS['border']};border-top:3px solid {hdr_color};
                border-radius:12px;padding:14px 18px;margin-bottom:12px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
          <div style="font-size:0.95rem;font-weight:800;color:#e6edf3;">
            Vijay Malik Financial Quality Compliance Profile
          </div>
          <div style="font-size:0.72rem;color:#8b949e;margin-top:2px;">
            Status: <strong style="color:{hdr_color};">{_esc(status_msg)}</strong>
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:1.5rem;font-weight:900;color:{hdr_color};line-height:1.0;">
            {m_score}
            <span style="font-size:0.85rem;color:#8b949e;font-weight:400;">&thinsp;/ 5</span>
          </div>
          <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                      letter-spacing:0.5px;margin-top:2px;">Peaceful Parameters Cleared</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _mk_grid = ""
    for letter, title, passed, baseline in pillars:
        clr_mk     = _MALIK_GREEN if passed else "#f85149"
        bg_mk      = "18" if passed else "08"
        ico_mk     = "✅" if passed else "❌"

        _mk_grid += (
            f"<div style='background:{clr_mk}{bg_mk};border:1px solid {clr_mk}40;"
            f"border-radius:8px;padding:10px;text-align:center;min-width:110px;flex:1;'>"
            f"<div style='font-size:1.6rem;font-weight:900;color:{clr_mk};line-height:1.1;'>"
            f"{_esc(letter)}</div>"
            f"<div style='font-size:0.68rem;font-weight:700;color:#e6edf3;margin-top:4px;"
            f"white-space:nowrap;'>{_esc(title)}</div>"
            f"<div style='font-size:0.58rem;color:#8b949e;margin-top:2px;line-height:1.2;'>"
            f"{_esc(baseline)}</div>"
            f"<div style='font-size:1.0rem;margin-top:4px;'>{ico_mk}</div>"
            f"</div>"
        )

    st.markdown(
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;'>{_mk_grid}</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PETER LYNCH FAST GROWER RADAR — 4-Pillar Tenbagger Discovery Audit
# ══════════════════════════════════════════════════════════════════════════════

def render_lynch_radar(stock: pd.Series):
    """
    Renders Peter Lynch's Fast Grower 4-pillar tenbagger discovery audit card.
    PURE DISPLAY — Reads pre-materialized binary pillar columns from scoring_engine.py.
    Zero threshold re-computation; zero scoring logic; immune to parameter drift.
    Source: docs/lynch_growth_specs.json v1.1-india-calibrated-fastgrower (all thresholds there).
    Pillars (V/P/D/F):
      V — Growth Velocity: lynch_growth_velocity (Rev 5Y CAGR + EPS per share + FCF cash gate)
      P — Valuation PEG:   lynch_valuation_peg   (positive PEG in sweet spot corridor)
      D — Pre-Discovery:   lynch_pre_discovery   (FII+DII combined below institutional threshold)
      F — Fortress Owner:  lynch_fortress_owner  (D/E balance sheet + promoter level OR active buying)
    """
    st.markdown("<div class='sec-head'>🚀 Peter Lynch — Fast Grower Tenbagger Radar</div>",
                unsafe_allow_html=True)

    l_pass  = int(_g(stock, "lynch_pass",  0))
    l_score = int(_g(stock, "lynch_score", 0))

    pillars = [
        ("V", "Growth Velocity",
         int(_g(stock, "lynch_growth_velocity", 0)) == 1,
         "Hyper-Growth Runway: revenue speed confirmed by EPS per share acceleration and positive free cash flow"),
        ("P", "PEG Sweet Spot",
         int(_g(stock, "lynch_valuation_peg",   0)) == 1,
         "GARP Entry Corridor: growth significantly outpaces the price paid for it"),
        ("D", "Pre-Discovery",
         int(_g(stock, "lynch_pre_discovery",   0)) == 1,
         "Combined Institutional Weight: FII plus DII below the pre-discovery threshold"),
        ("F", "Fortress Owner",
         int(_g(stock, "lynch_fortress_owner",  0)) == 1,
         "Skin In The Game Shield: conservative balance sheet and owner conviction by level or active buying"),
    ]

    # COLOUR INVERSION (fixed 2026-08-25): this radar used Lynch's brand red (#e74c3c) for a
    # PASSING pillar while six of the eight radars (CAN SLIM, SEPA, Dorsey, Outsider, Marks,
    # Malik) use red for FAILURE. On NALCO the two ✅ pillars rendered in red boxes and the two
    # ❌ pillars in neutral grey — the tab's colour language flipped on one panel, so a scan read
    # the passes as failures. Red now means exactly one thing across the whole Frameworks tab.
    _LYNCH_GREEN = COLORS["green"]      # pass — matches Malik / CAN SLIM
    _LYNCH_FAIL  = "#f85149"            # fail — the shared red used by the other radars
    hdr_color   = _LYNCH_GREEN if l_pass else COLORS["text_muted"]
    status_msg  = "LYNCH TENBAGGER CERTIFIED" if l_pass else "Lynch Fast Grower Criteria Not Met"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1117 0%,#1a0a0a 100%);
                border:1px solid {COLORS['border']};border-top:3px solid {hdr_color};
                border-radius:12px;padding:14px 18px;margin-bottom:12px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
          <div style="font-size:0.95rem;font-weight:800;color:#e6edf3;">
            Peter Lynch Fast Grower Tenbagger Discovery Profile
          </div>
          <div style="font-size:0.72rem;color:#8b949e;margin-top:2px;">
            Status: <strong style="color:{hdr_color};">{_esc(status_msg)}</strong>
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:1.5rem;font-weight:900;color:{hdr_color};line-height:1.0;">
            {l_score}
            <span style="font-size:0.85rem;color:#8b949e;font-weight:400;">&thinsp;/ 4</span>
          </div>
          <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                      letter-spacing:0.5px;margin-top:2px;">Lynch Gates Cleared</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _ly_grid = ""
    for letter, title, passed, baseline in pillars:
        clr_ly  = _LYNCH_GREEN if passed else _LYNCH_FAIL
        bg_ly   = "18" if passed else "08"
        ico_ly  = "✅" if passed else "❌"

        _ly_grid += (
            f"<div style='background:{clr_ly}{bg_ly};border:1px solid {clr_ly}40;"
            f"border-radius:8px;padding:10px;text-align:center;min-width:110px;flex:1;'>"
            f"<div style='font-size:1.6rem;font-weight:900;color:{clr_ly};line-height:1.1;'>"
            f"{_esc(letter)}</div>"
            f"<div style='font-size:0.68rem;font-weight:700;color:#e6edf3;margin-top:4px;"
            f"white-space:nowrap;'>{_esc(title)}</div>"
            f"<div style='font-size:0.58rem;color:#8b949e;margin-top:2px;line-height:1.2;'>"
            f"{_esc(baseline)}</div>"
            f"<div style='font-size:1.0rem;margin-top:4px;'>{ico_ly}</div>"
            f"</div>"
        )

    st.markdown(
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;'>{_ly_grid}</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# QGLP DEEP-DIVE RADAR — the namesake methodology (Raamdeo Agrawal)
# ═══════════════════════════════════════════════════════════════

def render_qglp_radar(stock: pd.Series, profile_name: str = "Balanced"):
    """Deep-dive audit card for QGLP — Quality, Growth, Longevity, reasonable Price
    (25th WCS: "QGL is the Value component which is then juxtaposed with P").

    PURE DISPLAY over pre-materialized engine columns: the four 0-100 sub-scores
    (qglp_quality/growth/longevity/price), qglp_score/qglp_pass, the raw gate inputs,
    and the 19th-WCS SQGLP letter screen. Hard-gate thresholds are read from
    MASTER_PROFILES[profile_name] — the SAME source compute_qglp_score used — so the
    card can never drift from the gate (the Fisher module-vs-engine lesson, 7fff308).
    Missing data renders honest blanks ("—"), never fabricated zeros.
    """
    _Q_GOLD = "#d4a017"
    prof = MASTER_PROFILES.get(profile_name, MASTER_PROFILES["Balanced"])
    roce_gate  = prof.get("roce_gate", 15.0)
    growth_gate = prof.get("growth_gate", 15.0)
    peg_gate   = prof.get("peg_gate", 1.5)

    st.markdown("<div class='sec-head'>👑 QGLP — Raamdeo's Process (Q·G·L·P)</div>",
                unsafe_allow_html=True)

    q_pass  = int(_g(stock, "qglp_pass", 0)) == 1
    q_score = stock.get("qglp_score")
    _score_known = pd.notna(q_score)
    hdr_color  = _Q_GOLD if q_pass else COLORS["text_muted"]
    status_msg = "QGLP-COMPLIANT" if q_pass else "Hard Gates Pending"
    big = f"{float(q_score):.0f}" if _score_known else "—"

    st.markdown(
        f'<div style="background:linear-gradient(135deg,#0d1117 0%,#171204 100%);'
        f'border:1px solid {COLORS["border"]};border-top:3px solid {hdr_color};'
        f'border-radius:12px;padding:14px 18px;margin-bottom:12px;">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;">'
        f'<div><div style="font-size:0.95rem;font-weight:800;color:#e6edf3;">'
        f'Raamdeo Agrawal QGLP Compliance Profile</div>'
        f'<div style="font-size:0.72rem;color:#8b949e;margin-top:2px;">'
        f'Status: <strong style="color:{hdr_color};">{_esc(status_msg)}</strong>'
        f'</div></div>'
        f'<div style="text-align:right;">'
        f'<div style="font-size:1.5rem;font-weight:900;color:{hdr_color};line-height:1.0;">'
        f'{big}<span style="font-size:0.85rem;color:#8b949e;font-weight:400;">&thinsp;/ 100</span></div>'
        f'<div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;'
        f'letter-spacing:0.5px;margin-top:2px;">Weighted QGLP Score</div>'
        f'</div></div></div>',
        unsafe_allow_html=True)

    # ── The four legs (0-100 percentile sub-scores; Q-leg at full parity since 2026-08-22) ──
    _legs = [("Q", "Quality",   "qglp_quality",   "ROCE percentile · promoter conduct"),
             ("G", "Growth",    "qglp_growth",    "PAT + EPS 5-yr growth percentiles"),
             ("L", "Longevity", "qglp_longevity", "Decade-RoE consistency percentile"),
             ("P", "Price",     "qglp_price",     "PEG zone (the final check on QGL)")]
    _rows = ""
    for letter, title, col, sub in _legs:
        v = stock.get(col)
        known = pd.notna(v)
        pct   = max(0.0, min(100.0, float(v))) if known else 0.0
        _p    = _shown(pct, 0)                      # band on what the leg PRINTS, not the raw pct
        clr   = (_Q_GOLD if _p >= 70 else COLORS["gold"] if _p >= 40
                 else COLORS["orange"]) if known else COLORS["text_muted"]
        disp  = f"{float(v):.0f}" if known else "—"
        _rows += (
            f"<div style='display:flex;align-items:center;gap:10px;margin:6px 0;'>"
            f"<div style='width:22px;font-size:1.05rem;font-weight:900;color:{clr};'>{letter}</div>"
            f"<div style='width:86px;font-size:0.74rem;font-weight:700;color:#e6edf3;'>{title}</div>"
            f"<div style='flex:1;height:10px;background:#161b22;border-radius:5px;overflow:hidden;'>"
            f"<div style='width:{pct:.0f}%;height:100%;background:{clr};'></div></div>"
            f"<div style='width:34px;text-align:right;font-size:0.85rem;font-weight:800;"
            f"color:{clr};'>{disp}</div>"
            f"<div style='width:230px;font-size:0.6rem;color:#8b949e;'>{_esc(sub)}</div>"
            f"</div>")
    st.markdown(f"<div style='margin-bottom:12px;'>{_rows}</div>", unsafe_allow_html=True)

    # ── Hard gates: actual vs the ACTIVE profile's thresholds (mirrors qglp_pass exactly) ──
    _roce = stock.get("roce")
    _gr   = stock.get("pat_gr_5y")
    _peg  = stock.get("peg")

    def _gate(label, actual, thr_txt, ok):
        clr = _Q_GOLD if ok else COLORS["text_muted"]
        a   = f"{float(actual):.1f}" if pd.notna(actual) else "—"
        ico = "✅" if ok else "❌"
        return (f"<div style='background:{clr}12;border:1px solid {clr}40;border-radius:8px;"
                f"padding:9px 12px;text-align:center;min-width:130px;flex:1;'>"
                f"<div style='font-size:0.62rem;font-weight:700;color:#8b949e;"
                f"text-transform:uppercase;'>{_esc(label)}</div>"
                f"<div style='font-size:1.05rem;font-weight:900;color:{clr};margin-top:2px;'>"
                f"{a} <span style='font-size:0.66rem;color:#8b949e;font-weight:600;'>"
                f"{_esc(thr_txt)}</span></div>"
                f"<div style='font-size:0.9rem;margin-top:2px;'>{ico}</div></div>")

    gates = (
        _gate("ROCE", _roce, f"vs ≥ {roce_gate:.0f}",
              bool(pd.notna(_roce) and float(_roce) >= roce_gate)) +
        _gate("PAT 5Y CAGR", _gr, f"vs ≥ {growth_gate:.0f}",
              bool(pd.notna(_gr) and float(_gr) >= growth_gate)) +
        _gate("PEG", _peg, f"vs 0–{peg_gate:.1f}",
              bool(pd.notna(_peg) and 0.0 <= float(_peg) <= peg_gate))
    )
    st.markdown(
        f"<div style='font-size:0.62rem;font-weight:800;color:#8b949e;text-transform:uppercase;"
        f"letter-spacing:0.8px;margin:4px 0 6px 0;'>Hard Gates</div>"
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;'>{gates}</div>",
        unsafe_allow_html=True)

    # ── SQGLP letter strip (19th WCS century-stock variant — same DNA, small-cap 100x form) ──
    _letters = [("S", "Size",      "sqglp_s", "< ₹5,000 Cr small base"),
                ("Q", "Quality",   "sqglp_q", "ROCE·RoE ≥ 15 · CFO/PAT ≥ 70"),
                ("G", "Growth",    "sqglp_g", "PAT ≥ 20 · Rev ≥ 15 (5Y)"),
                ("L", "Longevity", "sqglp_l", "10-yr growth ≥ 12"),
                ("P", "Price",     "sqglp_p", "P/E ≤ 15 entry")]
    _grid = ""
    for letter, title, col, base in _letters:
        on  = int(_g(stock, col, 0)) == 1
        clr = _Q_GOLD if on else COLORS["text_muted"]
        bg  = "18" if on else "08"
        ico = "✅" if on else "❌"
        _grid += (
            f"<div style='background:{clr}{bg};border:1px solid {clr}40;"
            f"border-radius:8px;padding:10px;text-align:center;min-width:105px;flex:1;'>"
            f"<div style='font-size:1.5rem;font-weight:900;color:{clr};line-height:1.1;'>{letter}</div>"
            f"<div style='font-size:0.66rem;font-weight:700;color:#e6edf3;margin-top:3px;'>{_esc(title)}</div>"
            f"<div style='font-size:0.56rem;color:#8b949e;margin-top:2px;'>{_esc(base)}</div>"
            f"<div style='font-size:0.95rem;margin-top:3px;'>{ico}</div></div>")
    _cent = int(_g(stock, "century_stock_flag", 0)) == 1
    _sq   = stock.get("sqglp_score")
    _sq_s = f"{int(_sq)}/5" if pd.notna(_sq) else "—"
    _cent_s = " · 🐘 Century Candidate" if _cent else ""
    st.markdown(
        f"<div style='font-size:0.62rem;font-weight:800;color:#8b949e;text-transform:uppercase;"
        f"letter-spacing:0.8px;margin:2px 0 6px 0;'>SQGLP Century-Stock Screen (19th WCS) · "
        f"{_sq_s}{_cent_s}</div>"
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px;'>{_grid}</div>",
        unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════
# MAUBOUSSIN EXPECTATIONS INVESTING RADAR — Framework 34
# ═══════════════════════════════════════════════════════════════

def render_mauboussin_radar(stock: pd.Series):
    """
    Renders Mauboussin & Rappaport's Expectations Investing 3-layer audit card.
    PURE DISPLAY — Reads pre-materialized binary pillar columns from scoring_engine.py.
    Zero threshold re-computation; zero scoring logic; immune to parameter drift.
    Source: docs/mauboussin_expectations_specs.json v2.0-expectations-gap.
    Pillars (G/T/C) — v2.0 REPOINT 2026-08-28: G is the book's ch.7 buy standard and the
    CERTIFICATION leg (strict inputs); T and C are DISQUALIFIERS (fire on evidence only):
      G — Expectations Discount:  mauboussin_gap_opportunity  (expectations_gap ≤ −5, inputs required)
      T — Treadmill Safety:       mauboussin_treadmill_breach (sell_alert_treadmill gate)
      C — CAP Trap Clear:         mauboussin_cap_trap==0      (implied_cap > 15 + ROCE 3Y slope < -1)
    Layer 3: Interactive Reverse DCF Expected Value Calculator (on-demand, single-stock only).
    """
    _MAUB_COLOR = "#8b5cf6"

    st.markdown("<div class='sec-head'>🔮 Mauboussin — Expectations Investing Radar</div>",
                unsafe_allow_html=True)

    m_pass   = int(_g(stock, "mauboussin_pass",         0))
    m_score  = int(_g(stock, "mauboussin_score",        0))
    m_cap    = _g(stock, "mauboussin_implied_cap",      0.0)
    _nopat_raw = _g(stock, "mauboussin_nopat_margin",   None)
    m_nopat_str = f"{_nopat_raw:.1f}%" if _nopat_raw is not None and _nopat_raw == _nopat_raw else "—"

    _m_gap_raw = _g(stock, "expectations_gap", None)
    _m_gap_str = (f"{_m_gap_raw:+.1f}pp" if _m_gap_raw is not None
                  and _m_gap_raw == _m_gap_raw else "—")
    pillars = [
        ("G", "Expectations Discount",
         int(_g(stock, "mauboussin_gap_opportunity",  0)) == 1,
         "The book's ch.7 buy standard: the growth the price implies sits at least 5 points "
         "below what the business can sustainably deliver — a margin of safety in expectations. "
         "Requires all inputs present: unverifiable is not certified."),
        ("T", "Treadmill Safety",
         int(_g(stock, "mauboussin_treadmill_breach", 0)) == 1,
         "Expectations Treadmill Safe: stock not priced for indefinite perfection requiring continuous positive surprises"),
        ("C", "CAP Trap Clear",
         int(_g(stock, "mauboussin_cap_trap",         0)) == 0,
         "Competitive Advantage Period Realistic: no high-CAP expectations paired with ROCE deceleration"),
    ]

    hdr_color  = _MAUB_COLOR if m_pass else COLORS["text_muted"]
    status_msg = "EXPECTATIONS MATRIX CERTIFIED" if m_pass else "Expectations Investing Criteria Not Met"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1117 0%,#120a1a 100%);
                border:1px solid {COLORS['border']};border-top:3px solid {hdr_color};
                border-radius:12px;padding:14px 18px;margin-bottom:12px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
          <div style="font-size:0.95rem;font-weight:800;color:#e6edf3;">
            Mauboussin Price-Implied Expectations (PIE) Audit
          </div>
          <div style="font-size:0.72rem;color:#8b949e;margin-top:2px;">
            Expectations Gap: <strong style="color:{hdr_color};">{_m_gap_str}</strong>
            &nbsp;·&nbsp;
            Implied CAP Proxy: <strong style="color:{hdr_color};">{m_cap:.2f}</strong>
            &nbsp;·&nbsp;
            NOPAT Margin: <strong style="color:{hdr_color};">{m_nopat_str}</strong>
            &nbsp;·&nbsp;
            Status: <strong style="color:{hdr_color};">{_esc(status_msg)}</strong>
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:1.5rem;font-weight:900;color:{hdr_color};line-height:1.0;">
            {m_score}
            <span style="font-size:0.85rem;color:#8b949e;font-weight:400;">&thinsp;/ 3</span>
          </div>
          <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                      letter-spacing:0.5px;margin-top:2px;">Expectations Gates Cleared</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _mb_grid = ""
    for letter, title, passed, baseline in pillars:
        # Fail was grey — but grey is this codebase's honest-blank/unknown signal everywhere else
        # (⚪ N/A pills, neutral rows), so a FAILED gate read as "no data". Red = failed, matching
        # the other radars; the purple brand colour still marks a pass.
        clr_mb = _MAUB_COLOR if passed else "#f85149"
        bg_mb  = "18" if passed else "08"
        ico_mb = "✅" if passed else "❌"

        _mb_grid += (
            f"<div style='background:{clr_mb}{bg_mb};border:1px solid {clr_mb}40;"
            f"border-radius:8px;padding:10px;text-align:center;min-width:110px;flex:1;'>"
            f"<div style='font-size:1.6rem;font-weight:900;color:{clr_mb};line-height:1.1;'>"
            f"{_esc(letter)}</div>"
            f"<div style='font-size:0.68rem;font-weight:700;color:#e6edf3;margin-top:4px;"
            f"white-space:nowrap;'>{_esc(title)}</div>"
            f"<div style='font-size:0.58rem;color:#8b949e;margin-top:2px;line-height:1.2;'>"
            f"{_esc(baseline)}</div>"
            f"<div style='font-size:1.0rem;margin-top:4px;'>{ico_mb}</div>"
            f"</div>"
        )

    st.markdown(
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;'>{_mb_grid}</div>",
        unsafe_allow_html=True,
    )

    # ── Layer 3: Per-Stock Payoff Framework (Mauboussin ch.7, stateless display) ──
    # Citation corrected 2026-08-27: was "Ch.13"; the book has 12 chapters and the EV method is
    # ch.7 "Buy, Sell, or Hold?". The SIZING BANDS are PRISM's own, not the book's.
    # Replaced the old STATIC 3-scenario matrix (identical hardcoded numbers for every
    # stock — decoration, not analysis). All values below are pre-materialized per stock
    # by the engine (MOD 5): P(up) = trajectory-calibrated win_rate_proxy; Upside% =
    # re-rating gap to quality-justified fair PE; Downside% = distance to the volatility
    # stop. Zero math in the UI beyond the book's sizing verdict zoning.
    p_up   = float(_g(stock, "win_rate_proxy", 0.5)) * 100.0
    up_pct = float(_g(stock, "mauboussin_ev_upside_pct", 0.0))
    dn_pct = float(_g(stock, "mauboussin_ev_downside_pct", 20.0))
    ev     = float(_g(stock, "expected_excess_return", 0.0))
    # Verdict + sizing pre-materialized by the engine (PRISM's thesis table) — pure display here
    ev_verdict = str(_g(stock, "mauboussin_ev_verdict", "Insufficient Edge · No position"))
    ev_color   = "#e74c3c" if "Insufficient" in ev_verdict else _MAUB_COLOR

    st.markdown(
        f"<div style='font-size:0.7rem;font-weight:800;color:{_MAUB_COLOR};"
        f"text-transform:uppercase;letter-spacing:1px;margin:10px 0 4px 0;'>"
        f"🧮 Payoff Framework — Per-Stock Expected Excess Return</div>"
        f"<div style='font-size:0.62rem;color:{COLORS['text_muted']};margin-bottom:8px;'>"
        f"EV = P(Upside) × Upside% − P(Downside) × Downside% · book minimum: 5% edge "
        f"· inputs computed per stock by the engine</div>",
        unsafe_allow_html=True,
    )

    def _ev_tile(label: str, big: str, sub: str, clr: str) -> str:
        return (
            f"<div style='flex:1;min-width:150px;background:{COLORS['bg_secondary']};"
            f"border:1px solid {clr}55;border-top:3px solid {clr};"
            f"border-radius:10px;padding:11px 14px;'>"
            f"<div style='font-size:0.58rem;font-weight:700;color:{COLORS['text_muted']};"
            f"text-transform:uppercase;letter-spacing:0.7px;'>{label}</div>"
            f"<div style='font-size:1.5rem;font-weight:900;color:{clr};"
            f"line-height:1.1;margin-top:3px;'>{big}</div>"
            f"<div style='font-size:0.58rem;color:{clr};font-weight:600;'>{sub}</div>"
            f"</div>"
        )

    _ev_tiles = (
        _ev_tile("Upside Leg", f"+{up_pct:.1f}%",
                 f"P↑ {p_up:.0f}% · re-rating to fair P/E", _MAUB_COLOR) +
        _ev_tile("Downside Leg", f"−{dn_pct:.1f}%",
                 f"P↓ {100 - p_up:.0f}% · distance to volatility stop", "#e74c3c") +
        _ev_tile("Expected Excess Return", f"{ev:+.1f}%",
                 _esc(f"{ev_verdict} · executable today: "
                      f"{float(_g(stock, 'optimal_portfolio_weight_pct', 0.0)):.2f}% "
                      f"(see Sizing Cockpit)"), ev_color)
    )
    st.markdown(
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-top:4px;'>{_ev_tiles}</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# LEVEL × TRAJECTORY × MISPRICING COCKPIT — Pass 3 integration
# Reads exclusively from pre-materialized engine columns via _g(); no engine scoring is
# re-derived here. The only inline conditionals are display colour/label choices. Renders
# 100% stateless via inline flex/grid — no st.columns/st.metric (CLAUDE.md §5).
# ═══════════════════════════════════════════════════════════════

def render_valuation_inversion_and_sizing_cockpit(stock: pd.Series):
    """Render high-dimensional Level × Trajectory × Mispricing parameters and portfolio sizing."""
    st.markdown("### 🔮 Value Creation & Expected Return Identity Cockpit")

    # Extract pre-materialized metrics — zero inline arithmetic
    exp_cagr = _g(stock, "expected_cagr_engine", 0.0)
    _id_g    = _g(stock, "expected_cagr_growth_term", 0.0)
    _id_y    = _g(stock, "expected_cagr_yield_term", 0.0)
    _id_r    = _g(stock, "expected_cagr_rerate_term", 0.0)
    moat_tau = stock.get("moat_tau")                    # NaN-aware: 35 live rows have no ladder
    val_rank = stock.get("valuation_residual_rank")     # MOD 2 percentile of the OLS residual, 0-100
    sepa_scr = int(_g(stock, "sepa_score", 0))
    sepa_pss = int(_g(stock, "sepa_pass", 0))

    # Compact inline flex card — mirrors the EP-strip idiom (no st.columns/st.metric padding,
    # CLAUDE.md §5). The sub-line is dropped when empty so single-value cards stay clean.
    def _cockpit_card(label: str, value: str, sub: str, val_clr: str, sub_clr: str,
                      tip: str = "") -> str:
        sub_html = (
            f'<div style="font-size:0.62rem;font-weight:600;color:{sub_clr};'
            f'margin-top:2px;white-space:normal;">{_esc(sub)}</div>'
        ) if sub else ""
        # help_chip returns "" when there is nothing to explain, so no bare "?" ever renders.
        # It must sit OUTSIDE _esc() — it is markup, not text.
        return (
            f'<div style="flex:1;min-width:150px;background:{COLORS["bg_secondary"]};'
            f'border:1px solid {COLORS["border"]};border-radius:10px;padding:12px 16px;">'
            f'<div style="font-size:0.56rem;font-weight:700;color:{COLORS["text_muted"]};'
            f'text-transform:uppercase;letter-spacing:0.7px;">{_esc(label)}'
            f'{help_chip(label, tip)}</div>'
            f'<div style="font-size:1.35rem;font-weight:900;color:{val_clr};'
            f'line-height:1.15;margin-top:3px;white-space:nowrap;">{_esc(value)}</div>'
            f'{sub_html}</div>'
        )

    # Row 1 — three honest reads: can the business compound / is the margin trending / is the
    # price justified. Post-audit 2026-08-22: the CAGR tile shows its capped decomposition (the
    # raw sum reached ±300%/yr on degenerate inputs); the valuation tile shows the engine's
    # bounded PERCENTILE, not the raw mean-zero OLS residual (whose negative half — 51.6% of
    # the universe by construction — used to wear a green "Alpha" badge); the tau tile names
    # what moat_tau actually measures (operating margins over ~5y, not a decade of ROCE).
    cagr_good  = exp_cagr > 15.0
    _tau_known = pd.notna(moat_tau)
    tau_good   = bool(_tau_known and float(moat_tau) > 0.25)
    _rank_known = pd.notna(val_rank)
    _cheaper    = (100.0 - float(val_rank)) if _rank_known else None   # low rank = cheapest
    if _rank_known:
        _c_shown = _shown(_cheaper, 0)          # the tile prints 0dp; band on that
        _val_clr = (COLORS["green"] if _c_shown >= 75.0
                    else COLORS["gold"] if _c_shown >= 25.0 else COLORS["red"])
        _val_txt, _val_sub = f"{_cheaper:.0f}%", (
            f"Cheaper than {_cheaper:.0f}% of fundamentals-matched peers")
    else:
        _val_clr, _val_txt, _val_sub = COLORS["text_muted"], "—", "No cross-sectional rank"
    row1 = (
        _cockpit_card("👑 Expected Return Estimate", f"{exp_cagr:+.1f}% /yr",
                      f"Growth {_id_g:+.0f} · Cash {_id_y:+.0f} · Re-rating {_id_r:+.0f} (capped)",
                      COLORS["green"] if cagr_good else COLORS["orange"],
                      COLORS["text_muted"],
                      tip="A decomposition, not a forecast: earnings growth + cash yield + the "
                          "re-rating available if the P/E moved to fair value. Each leg is "
                          "capped — the uncapped sum reached ±300%/yr on degenerate inputs.") +
        _cockpit_card("⏳ Margin Trend (5Y Tau)",
                      f"{float(moat_tau):+.2f}" if _tau_known else "—",
                      ("Operating margins trending up" if tau_good
                       else "Operating margins flat / fading" if _tau_known
                       else "Not enough margin history"),
                      COLORS["green"] if tau_good
                      else COLORS["orange"] if _tau_known else COLORS["text_muted"],
                      COLORS["green"] if tau_good
                      else COLORS["orange"] if _tau_known else COLORS["text_muted"],
                      tip="Kendall's tau over roughly five years of OPERATING MARGINS — a rank "
                          "measure of trend direction, +1 (rising every year) to −1 (falling "
                          "every year). It measures margins, not returns on capital.") +
        _cockpit_card("📊 Price vs Fundamentals", _val_txt, _val_sub, _val_clr, _val_clr,
                      tip="How much of the fundamentally-matched peer group this stock is "
                          "cheaper than — a bounded cross-sectional percentile, not a raw "
                          "valuation residual. 50% is the middle of the pack, not a bargain.")
    )
    st.markdown(
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:4px;">{row1}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### ⚡ Mark Minervini SEPA® Risk & Allocation Matrix")

    if sepa_pss == 1:
        st.success(f"🚀 SEPA MOMENTUM BREAKOUT COMPLIANT — INDIVIDUAL PROFILE SCORE: {sepa_scr}/7")
    else:
        st.info(f"⏳ Watchlist Setup Mode — Individual Profile Score: {sepa_scr}/7 (Hard Gates Pending)")

    weight_pct = _g(stock, "optimal_portfolio_weight_pct", 0.0)
    allocation = _g(stock, "rupee_capital_allocation", 0.0)
    stop_loss  = _g(stock, "vstop_value", 0.0)

    # HARDCODED CLAIM (fixed 2026-08-25): this card's subtitle was the literal string
    # "-7-8% Active Perimeter Shield" — a number stated as if measured but never computed. It is
    # true for 95 of 2,117 stocks (4.5%); the real distance spans -20% to +27% (p10-p90). On EPack
    # Prefab it read "-7-8%" while the stop sat 25.5% BELOW the price, and the card immediately to
    # its right correctly said "+34.2% vs stop" — the same row contradicting itself. Compute it.
    # ~26% of the universe trades AT or BELOW its stop, where a negative "distance" would be a lie
    # in the other direction, so that state gets its own wording.
    # A STOP ABOVE THE PRICE IS NOT A STOP (fixed 2026-08-25). A stop-loss is the level you exit
    # at to cap a loss, so it must sit BELOW the position — one above the price would trigger
    # instantly. A volatility stop flips to the short side once price breaks down, so for 876
    # stocks (41.4% of the universe, every one "🔻 Below Stop (Trend Broken)") this card printed
    # a short-side level under a long-side label: Avanti Feeds, price ₹861.40, headline
    # "HARD VOLATILITY STOP-LOSS LEVEL ₹1,153.36". The number is real — it is the level price must
    # RECLAIM — but calling it your stop is wrong, so the label changes with the state.
    #
    # It also states the gap from the SAME side as the price card. The first version of this
    # subtitle recomputed the distance against the price ((vstop-close)/close) while the price
    # card uses the engine's dist_to_vstop ((close-vstop)/vstop), so one screen showed "33.9%
    # ABOVE price" and "-25.3% vs stop" for a single gap — arithmetically fine, visibly
    # contradictory. Mirror the engine's column instead of inventing a second denominator.
    _cp_stop, _vs_stop = stock.get("close_price"), stock.get("vstop_value")
    _dist_stop = stock.get("dist_to_vstop")
    _stop_lbl = "🚨 Hard Volatility Stop-Loss Level"
    if pd.isna(_cp_stop) or pd.isna(_vs_stop) or float(_cp_stop) <= 0 or float(_vs_stop) <= 0:
        _stop_sub, _stop_clr = "", COLORS["red"]          # no verdict from a data hole
    elif float(_vs_stop) > float(_cp_stop):
        _stop_lbl  = "⚠️ Volatility Stop — Trend Broken"
        _stop_sub  = (f"price sits {abs(float(_dist_stop)):.1f}% below it · no long stop to set"
                      if pd.notna(_dist_stop) else "price is below it · no long stop to set")
        _stop_clr  = COLORS["orange"]
    else:
        _stop_sub = (f"price sits {abs(float(_dist_stop)):.1f}% above it · trailing perimeter"
                     if pd.notna(_dist_stop) else "trailing perimeter")
        _stop_clr = COLORS["red"]

    # ── Thesis vs Executable reconciliation (mirror-and-explain — Fisher precedent, 7fff308) ──
    # The tearsheet carries TWO sizes for one stock: the Mauboussin EV verdict (the THESIS size —
    # what the expected value justifies) and this cockpit's Kelly×Minervini weight (the EXECUTABLE
    # size today — what trend + stop allow). They legitimately diverge: a 0% weight beside a
    # "High Conviction · 8–12%" band means "target — no entry now", not a contradiction (190 of the
    # 426 High-Conviction stocks sat exactly there on 2026-08-22). The strip states the band, the
    # executable weight, and the REASON for any gap. Display-only: it reads engine columns and
    # names which engine constraint bound — zero sizing math is re-derived here.
    _ev_verdict = str(stock.get("mauboussin_ev_verdict", "") or "")
    _EV_BANDS = {                                   # engine's EV verdict strings → thesis band
        "High Conviction · 8–12% position":           (8.0, 12.0),
        "Moderate-High · 5–8% position":              (5.0, 8.0),
        "Moderate · 3–5% position":                   (3.0, 5.0),
        "Insufficient Edge · No position (< 5% min)": (0.0, 0.0),
    }
    _band = _EV_BANDS.get(_ev_verdict)
    _close_rc = stock.get("close_price")
    _vstop_rc = stock.get("vstop_value")
    _stopped  = bool(pd.notna(_close_rc) and pd.notna(_vstop_rc)
                     and float(_close_rc) <= float(_vstop_rc))
    if weight_pct <= 0.0:
        _gap_reason = ("price at/below its volatility stop — no entry now"
                       if _stopped else
                       "Kelly edge ≤ 0 at current proxy odds — the formula sees no positive expectancy")
    elif _band == (0.0, 0.0):
        _gap_reason = ("EV below the 5% book minimum — thesis says no position; "
                       "technical sizing shown for reference only")
    elif _band and weight_pct < _band[0]:
        _gap_reason = "risk-capped below the thesis band (quarter-Kelly × 1%-risk rule)"
    elif _band and weight_pct > _band[1]:
        _gap_reason = "executable weight above the thesis band — trim to the band"
    elif _band:
        _gap_reason = "aligned with the thesis band"
    else:
        _gap_reason = ""
    if _ev_verdict:
        _recon_clr = (COLORS["orange"] if (weight_pct <= 0.0 or _band == (0.0, 0.0))
                      else COLORS["green"])
        st.markdown(
            f'<div style="background:{COLORS["bg_secondary"]};border:1px solid {_recon_clr}55;'
            f'border-left:3px solid {_recon_clr};border-radius:8px;padding:8px 14px;'
            f'margin:6px 0 8px 0;font-size:0.72rem;color:{COLORS["text_muted"]};">'
            f'🧭 <strong style="color:#e6edf3;">EV Thesis:</strong> {_esc(_ev_verdict)}'
            f' &nbsp;·&nbsp; <strong style="color:#e6edf3;">Executable today:</strong> '
            f'<strong style="color:{_recon_clr};">{weight_pct:.2f}%</strong>'
            + (f' — {_esc(_gap_reason)}' if _gap_reason else "")
            + '</div>',
            unsafe_allow_html=True,
        )

    # Row 2 — Fractional-Kelly Capital Allocation Matrix (inline flex; no st.columns/st.metric).
    # Copy is deliberately modest: p and b are UNCALIBRATED PROXIES (see the MOD 5 docstring in
    # scoring_engine.py), so "Quarter-Kelly Risk Managed" overclaimed — this is a sizing heuristic
    # whose real risk control is the Minervini 1%-risk cap.
    row2 = (
        _cockpit_card("🎯 Executable Capital Weight", f"{weight_pct:.2f}%",
                      "Fractional-Kelly heuristic · proxy odds", COLORS["blue"], COLORS["text_muted"]) +
        _cockpit_card("💰 Capital Deployment (10L Base)", f"₹ {allocation:,.2f}",
                      "", COLORS["gold"], COLORS["text_muted"]) +
        _cockpit_card(_stop_lbl, f"₹ {stop_loss:,.2f}", _stop_sub, _stop_clr, _stop_clr,
                      tip="The volatility stop (Chandelier-style trailing level). While price is "
                          "ABOVE it, this is your exit level — the most you intend to lose. Once "
                          "price falls BELOW it the trend is broken and it stops being a stop: "
                          "it becomes the level price must reclaim, which is why the label "
                          "changes. A long stop above the market would trigger instantly.")
    )
    st.markdown(
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:4px;">{row2}</div>',
        unsafe_allow_html=True,
    )

    # ── Row 3 · EXECUTION STRIP (added 2026-08-24) — the price, at last, against the engine's
    # own levels. The cockpit printed a ₹ stop and a ₹ allocation while the price itself appeared
    # NOWHERE on the tearsheet — the reader couldn't verify the stop distance or turn the
    # allocation into shares. Pure display over pre-materialized columns (fair_value_qglp is
    # engine-guarded: loss-makers propagate NaN → honest "—"). Price is as of the last data
    # refresh — there is no price-date column, so no timestamp is fabricated.
    _px_ex   = stock.get("close_price")
    _fv_ex   = stock.get("fair_value_qglp")
    _d52_ex  = stock.get("dist_52wh")
    _dvs_ex  = stock.get("dist_to_vstop")
    _px_ok   = pd.notna(_px_ex) and float(_px_ex) > 0
    _fv_txt, _fv_sub, _fv_clr = "—", "needs positive EPS", COLORS["text_muted"]
    if _px_ok and pd.notna(_fv_ex):
        _upside = (float(_fv_ex) / float(_px_ex) - 1.0) * 100.0
        _up_shown = _shown(_upside, 0)          # the sub-line prints 0dp; colour on that
        _fv_txt = f"₹ {float(_fv_ex):,.0f}"
        _fv_sub = f"{_upside:+.0f}% vs price · QGLP fair PE × EPS"
        _fv_clr = COLORS["green"] if _up_shown >= 0 else COLORS["red"]
    # A COMPUTED zero (sizing ruled the position out — price at/below stop, or zero Kelly
    # weight) is a KNOWN answer, not missing data. The em-dash is this codebase's honest-blank
    # glyph for "unknown", so a ruled-out entry must not wear it (the VCV full-payout lesson,
    # same day). Mirrors the ENGINE value: raw rupee_capital_allocation present-and-zero = the
    # rule spoke; missing = "—" as before.
    _sh_txt, _sh_sub = "—", "deployment ÷ price"
    _alloc_raw = stock.get("rupee_capital_allocation")
    if _px_ok and pd.notna(_alloc_raw):
        if float(_alloc_raw) > 0:
            _sh_txt = f"{int(float(_alloc_raw) // float(_px_ex)):,} shares"
        else:
            _sh_txt, _sh_sub = "0 shares", "no entry — sizing rules it out at this price"
    _dvs_sub = (f"{float(_dvs_ex):+.1f}% vs stop" if pd.notna(_dvs_ex) else "stop distance unknown")
    _d52_sub = (f"{float(_d52_ex):.1f}% off 52w high" if pd.notna(_d52_ex) else "")
    row3 = (
        _cockpit_card("💹 Price (last data refresh)",
                      f"₹ {float(_px_ex):,.2f}" if _px_ok else "—",
                      f"{_dvs_sub}" + (f" · {_d52_sub}" if _d52_sub else ""),
                      COLORS["text_primary"], COLORS["text_muted"]) +
        _cockpit_card("⚖️ Fair Value (QGLP)", _fv_txt, _fv_sub, _fv_clr, _fv_clr) +
        _cockpit_card("🧾 Executable at 10L Base", _sh_txt,
                      _sh_sub, COLORS["blue"], COLORS["text_muted"])
    )
    st.markdown(
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px;">{row3}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:0.64rem;color:{COLORS["text_muted"]};margin-top:6px;">'
        f'Per-stock sizing: each weight is computed independently for this stock alone and is '
        f'<strong>not portfolio-normalized</strong> — weights across many stocks will not sum '
        f'to 100% of capital.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Secondary Structural Decomposition.
    # UNSTYLED ORPHANS (fixed 2026-08-25): these three were raw st.write / st.markdown lines —
    # the only content on the tab that was not a card, dangling under a polished grid and reading
    # like debug output that never got designed. They now use the cockpit's own _cockpit_card.
    # HONESTY (same edit): both numbers defaulted to 0.0 via _g(), so a MISSING value rendered as
    # a confident "+0.00%". expectations_gap is only 68.4% populated — 669 stocks were shown a
    # fabricated zero — and value_creation_velocity 96.0%. A data hole is not a measurement, so
    # absent inputs now render "N/A" (§5 semantic truth).
    _vcv_raw  = stock.get("value_creation_velocity")
    _egap_raw = stock.get("expectations_gap")
    _vcv_ok, _egap_ok = pd.notna(_vcv_raw), pd.notna(_egap_raw)
    _vcv, _egap = (float(_vcv_raw) if _vcv_ok else 0.0), (float(_egap_raw) if _egap_ok else 0.0)

    # ⏳ CONSOLIDATION BASE (2026-10-15): the engine's label, never re-derived here, so the card always
    # agrees with the sidebar filter that offers the same states. It used to read "VCP Firing" for every
    # 10-day volume dry-up (1,218 stocks on the 14 Oct data, 377 of them Stage 4 declines) and "Active"
    # for no dry-up. The
    # sub-line states the facts the label was decided on, each printed on its own side of the line.
    _cb = stock.get("consolidation_base")
    _cb = _cb if isinstance(_cb, str) else ""
    _cb_ratio = stock.get("vcp_volume_ratio")
    _cb_facts = []
    if pd.notna(_cb_ratio):
        _cb_pct = _same_side(float(_cb_ratio) * 100, lambda v: v < VCP_DRYUP_SHARE * 100)
        _cb_facts.append(f"10-day volume at {_cb_pct}% of the 50-day")
    if _cb in CONSOLIDATION_BASE_LABELS[:2]:            # a dry-up: the facts that make it ⭐ or 💤
        _cb_stage = re.search(r"Stage \d", str(stock.get("weinstein_stage") or ""))
        _cb_facts.append(_cb_stage.group(0) if _cb_stage else "stage unknown")
        _cb_d52 = stock.get("dist_52wh")
        if pd.notna(_cb_d52):
            _cb_far = _same_side(float(_cb_d52), lambda v: v <= TREND_TEMPLATE_MAX_BELOW_HIGH)
            _cb_facts.append("at its 52-week high" if float(_cb_far) <= 0
                             else f"{_cb_far}% below its 52-week high")
        _cb_stop = stock.get("dist_to_vstop")
        if pd.notna(_cb_stop) and float(_cb_stop) < 0:
            _cb_facts.append("below its stop")
    _cb_sub = " · ".join(_cb_facts) if _cb else "volume averages not reported"
    _cb_clr = (COLORS["gold"] if _cb == CONSOLIDATION_BASE_LABELS[0]
               else COLORS["text_primary"] if _cb else COLORS["text_muted"])

    # Polarity per the engine's own definition (data_engine ~L3308): "Positive gap = priced for
    # more than it can deliver (expectations risk); negative = pessimism / margin of safety."
    _egap_clr = COLORS["text_muted"] if not _egap_ok else (
        COLORS["green"] if _shown(_egap, 2) < 0 else COLORS["orange"])

    # ── VCV display honesty (2026-08-28, audited on the live BUY★ cohort). Three states beyond
    # the number, each MIRRORING an engine value (never re-derived — the Fisher drift lesson):
    #   FULL PAYOUT  reinvestment_rate == 0 (DPR ≥ 100): the formula truthfully reads 0.00 —
    #                nothing retained to compound — but 11 live stocks with real above-hurdle
    #                spread (median ROCE 48.4; Castrol 75.8, P&G Health 85.9) rendered a RED zero
    #                beside a green Moat axis. A full distributor is not a value destroyer → gold.
    #   ASSUMED RR   dividend_payout_ratio missing (58.8% of the universe): RR is fillna-
    #                fabricated at 1.0, so the shown number is really just ROCE − CoE. The
    #                sub-line stops claiming a measured "Reinvestment rate ×" and names the
    #                assumption instead.
    #   ZERO ≠ RED   the engine defines red as "compounding below hurdle" — the shown value
    #                being NEGATIVE. A 0.00 is a third state (full payout, or spread exactly at
    #                the hurdle): muted, not condemned.
    _vcv_rr  = stock.get("reinvestment_rate")
    _vcv_dpr = stock.get("dividend_payout_ratio")
    _vcv_full_payout = bool(_vcv_ok and pd.notna(_vcv_rr) and float(_vcv_rr) == 0.0)
    _vcv_assumed_rr  = bool(_vcv_ok and pd.isna(_vcv_dpr))
    if not _vcv_ok:
        _vcv_sub = "not reported"
    elif _vcv_full_payout:
        _vcv_sub = "distributes all earnings — spread returned as cash, not retained"
    elif _vcv_assumed_rr:
        _vcv_sub = "assumed full retention — payout unreported"
    else:
        _vcv_sub = "Reinvestment rate × capital spread"
    _vcv_clr = (COLORS["text_muted"] if not _vcv_ok else
                COLORS["gold"] if _vcv_full_payout else
                COLORS["green"] if _shown(_vcv, 2) > 0 else
                COLORS["red"] if _shown(_vcv, 2) < 0 else
                COLORS["text_muted"])

    _struct = (
        _cockpit_card(
            "🔄 Value Creation Velocity", f"{_vcv:+.2f}%" if _vcv_ok else "N/A",
            _vcv_sub,
            _vcv_clr, COLORS["text_muted"],
            tip="How fast the business compounds its own capital: the share of profit it "
                "reinvests multiplied by the spread it earns above its cost of capital. "
                "Positive means every retained rupee is creating value. A 0.00 on a full "
                "payout means nothing is retained to compound — not value destruction. When "
                "the payout ratio is unreported the engine takes full retention as given, so "
                "the figure shown is the raw spread above the 12% hurdle.") +
        _cockpit_card(
            "📊 Expectations Gap", f"{_egap:+.2f}%" if _egap_ok else "N/A",
            ("priced above what it can deliver" if _egap > 0 else "margin of safety")
            if _egap_ok else "needs P/B + RoE + SSGR",
            _egap_clr, COLORS["text_muted"],
            tip="Growth the market PRICES IN (g_implied, inverted from the P/B-Gordon identity) "
                "minus the growth the business can SUSTAIN (g★, the lower of RoE×reinvestment "
                "and SSGR). Positive = expectations risk; negative = pessimism you can buy.") +
        _cockpit_card(
            "⏳ Consolidation Base", _cb or "N/A", _cb_sub, _cb_clr, COLORS["text_muted"],
            tip=(f"Minervini's Volatility Contraction Pattern, as far as one day's data can see it. "
                 f"⭐ VCP Setup: the 10-day average volume below {VCP_DRYUP_SHARE:.0%} of the 50-day "
                 f"(supply drying up) in a 📈 Stage 2 stock within {TREND_TEMPLATE_MAX_BELOW_HIGH}% of its "
                 f"52-week high and not below its stop. 💤 Drying Up: the same dry-up anywhere else — "
                 f"quiet alone is not a base. ⚪ Normal Volume: no dry-up. The narrowing price swings "
                 f"that complete a VCP need a chart. The sidebar's ⏳ Consolidation Base filter offers "
                 f"these same labels."))
    )
    st.markdown(
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:4px;">{_struct}</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# vs SECTOR PEERS — per-stock sector-relative context (the value-trap guard)
# ═══════════════════════════════════════════════════════════════

def _roce_sector_label(raw_rank):
    """Pure: sector_roce_pct_rank (0-1, or None/NaN) -> (band, value_str, subtitle). The subtitle's
    "Top X%" is DERIVED from the actual percentile (100 - pct), never a fixed band string — so it can
    no longer contradict the displayed value: a 100th-pctile stock reads "Top 1% — sector ROCE
    leader", a 71st reads "Top 29% in sector" (the old code hardcoded "Top 30% — sector ROCE leader"
    for the entire 70-100 band, mislabelling both). `band` ∈ {none, leader, above, below} drives the
    tile colour in the caller. Bands are the engine's own lines (0.70 = category-winner top-30%,
    0.50 = sector median)."""
    if raw_rank is None or raw_rank != raw_rank:        # None or NaN
        return "none", "—", "No sector peer rank"
    pct = float(raw_rank) * 100.0
    top = max(1, round(100.0 - pct))                    # the stock's ACTUAL top-X% position
    val = f"{pct:.0f}"
    if pct >= 70.0:
        sub = f"Top {top}% — sector ROCE leader" if pct >= 95.0 else f"Top {top}% in sector"
        return "leader", val, sub
    if pct >= 50.0:
        return "above", val, "Above sector median"
    return "below", val, "Below sector median — value-trap check"


def _sector_peer_strip_html(stock: pd.Series) -> str:
    """Build the 'vs Sector Peers' HTML strip — PURE function, zero st calls (unit-testable).

    Surfaces four already-computed-but-orphaned sector-relative columns so the Overview can
    answer the one question the absolute 6-axis scorecard structurally cannot: is this stock
    genuinely strong, or only strong *for a weak sector* (and vice-versa) — the value-trap guard.

    Thresholds are the engine's OWN, not invented: 0.70 = the `category_winner_flag` top-30%
    sector-ROCE line; 0.50 = the sector median. Reads sector_roce_pct_rank (0-1, fillna 0.5 in
    the engine), emc_flag / emc_sector_beat_count (ROE beats sector AVERAGE in N of 5 windows),
    and sector_capital_phase (Chancellor capital-cycle: Hot / Starved / Neutral).
    """
    GOLD, GREEN, RED, MUTE = (
        COLORS["gold"], COLORS["green"], COLORS["red"], COLORS["text_muted"]
    )

    # ── Tile 1: ROCE percentile within sector (the value-trap signal) ───────────
    # Subtitle's "Top X%" is derived from the actual percentile in _roce_sector_label, so it can
    # never contradict the displayed value (the old fixed "Top 30%" did, for a pctile-100 leader).
    _rk_band, rk_val, rk_sub = _roce_sector_label(_g(stock, "sector_roce_pct_rank", None))
    rk_clr = {"none": MUTE, "leader": GREEN, "above": GOLD, "below": RED}[_rk_band]

    # ── Tile 2: Sector ROE moat (EMC — 17th WCS sector-relative persistence) ────
    emc_on   = int(_g(stock, "emc_flag", 0)) == 1
    emc_beat = int(_g(stock, "emc_sector_beat_count", 0))
    emc_clr  = GREEN if emc_on else MUTE
    emc_val  = f"{emc_beat}/5"
    emc_sub  = "Beats sector ROE (EMC moat)" if emc_on else "Lags sector ROE"

    # ── Tile 3: Sector capital cycle (Chancellor — Capital Returns) ─────────────
    phase = str(_g(stock, "sector_capital_phase", "⚖️ Neutral") or "⚖️ Neutral")
    _phase_map = {
        "🔥 Hot Capital (caution)":        (GOLD,  "🔥 Hot",     "Sector over-investing — mean-reversion risk"),
        "❄️ Capital Starved (opportunity)": (GREEN, "❄️ Starved", "Under-invested sector — supply opportunity"),
        "⚖️ Neutral":                      (MUTE,  "⚖️ Neutral", "Balanced sector capital cycle"),
    }
    cap_clr, cap_val, cap_sub = _phase_map.get(
        phase, (MUTE, "⚖️ Neutral", "Balanced sector capital cycle")
    )

    sector = _esc(_g(stock, "sector", "—") or "—")

    # ── Lead tile: rank within the sector cohort (the named-cohort position) ────
    # Reads the engine's sector_composite_rank / sector_peer_count (#X of N by post-penalty
    # composite). _g → None on NaN/missing; a sole-listed peer (count < 2) has no cohort.
    # Sub avoids the sector name (the header already shows it) — _tile re-escapes sub, so a
    # name with '&' would double-escape; keeping it name-free sidesteps that entirely.
    s_rank = _g(stock, "sector_composite_rank", None)
    s_cnt  = _g(stock, "sector_peer_count", None)
    if s_rank is None or s_cnt is None or int(s_cnt) < 2:
        sr_clr, sr_val, sr_sub = MUTE, "—", "No sector cohort"
    else:
        s_rank, s_cnt = int(s_rank), int(s_cnt)
        pos = s_rank / s_cnt                       # 0 = top of sector
        sr_clr = GREEN if pos <= 0.25 else GOLD if pos <= 0.50 else MUTE
        sr_val = f"#{s_rank} of {s_cnt}"
        sr_sub = f"Top {max(1, round(pos * 100))}% by composite"

    def _tile(color, value, label, sub):
        return (
            f"<div style='flex:1;flex-shrink:0;min-width:150px;"
            f"background:{color}12;border:1px solid {color}40;border-radius:10px;padding:11px 14px;'>"
            f"<div style='font-size:0.6rem;font-weight:700;color:{MUTE};text-transform:uppercase;"
            f"letter-spacing:0.6px;white-space:nowrap;'>{_esc(label)}</div>"
            f"<div style='font-size:1.3rem;font-weight:900;color:{color};line-height:1.2;"
            f"margin-top:3px;white-space:nowrap;'>{_esc(value)}</div>"
            f"<div style='font-size:0.62rem;color:{color};margin-top:2px;'>{_esc(sub)}</div>"
            f"</div>"
        )

    header = (
        f"<div style='font-size:0.62rem;font-weight:800;color:{MUTE};text-transform:uppercase;"
        f"letter-spacing:0.8px;margin:16px 0 6px 0;'>📊 vs Sector Peers · {sector}</div>"
    )
    tiles = (
        _tile(sr_clr,  sr_val,  "🏅 Rank in Sector",            sr_sub) +
        _tile(rk_clr,  rk_val,  "🛡️ ROCE Percentile in Sector", rk_sub) +
        _tile(emc_clr, emc_val, "📈 Sector ROE Moat",           emc_sub) +
        _tile(cap_clr, cap_val, "🔄 Sector Capital Cycle",       cap_sub)
    )
    return (
        header
        + f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:6px;'>{tiles}</div>"
    )


# Trajectory card — the SECOND derivative. Every other Overview panel answers "where does this
# business stand"; none answers "which way is it moving, and is the move speeding up".
#
# WHY THE TWO SECTIONS ARE SEPARATED AND LABELLED WITH THEIR OWN BASES. The engine carries seven
# columns whose names all end in _acceleration, and they are THREE unrelated things:
#     pat/rev/ebitda_acceleration = 3Y CAGR - 5Y CAGR   → a true acceleration (rate of a rate)
#     npm/opm/gpm_acceleration    = latest QUARTER - THE SAME QUARTER ONE YEAR AGO (*_pyq)
#         REBASED 2026-09-18. These were three DIFFERENT bases and two of them were scored:
#         npm/opm measured the latest quarter against the ANNUAL figure 1Y back, and gpm against
#         the 5Y MEDIAN. A quarter minus an annual number is not a delta, and it folded seasonality
#         into a signal read as structural — measurably: the old GPM form had a universe median of
#         +6.07pp, claiming the typical company's gross margin was improving six points. On the
#         same-quarter basis that median is +0.00pp. See core/data_engine.py for the full evidence.
#     ebit_acceleration           = EBIT growth - EBITDA growth, SAME window — a D&A-intensity
#                                   spread, not a time comparison at all; deliberately NOT shown
#                                   here, because it does not answer this card's question.
# Rendering all seven under one "Acceleration" heading would repeat the All Data "PAT YoY" bug
# (tests/test_all_data_yoy_labels.py): different bases stacked in one column invite a comparison
# that is not valid. So growth and margin sit in separate blocks, and every margin row prints the
# base it was measured against instead of leaving the reader to assume they match.
_ACCEL_GROWTH = [
    ("PAT",     "pat_acceleration",    "pat_gr_3y",    "pat_gr_5y"),
    ("Revenue", "rev_acceleration",    "rev_gr_3y",    "rev_gr_5y"),
    ("EBITDA",  "ebitda_acceleration", "ebitda_gr_3y", "ebitda_gr_5y"),
]
_ACCEL_MARGIN = [
    # The base column and its on-screen label must track core/data_engine.py's formula exactly —
    # a stale label beside a corrected number is worse than the original defect, because it reads
    # as authoritative. Pinned by tests/test_pyq_margin_basis.py.
    ("NPM", "npm_acceleration", "npm_latest_q", "npm_pyq", "same qtr LY"),
    ("OPM", "opm_acceleration", "opm_latest_q", "opm_pyq", "same qtr LY"),
    ("GPM", "gpm_acceleration", "gpm_latest_q", "gpm_pyq", "same qtr LY"),
]


# Below this, a move is noise and the card says "flat". NOT a taste call: measured across all
# 5,421 growth-acceleration values on the live universe, |delta| has a median of 10.5pp and a p10
# of exactly 2.0pp — so this band marks the smallest decile of real moves as no-move. The first
# cut used 0.5pp, which called only 2.7% flat and let a 0.5pp wobble outvote a 49pp collapse:
# Igarashi Motors read "Decelerating — 1 up · 2 down" on the strength of Revenue 3Y 9.7% vs 5Y
# 10.2%, a non-event. At 2.0pp the same stock reads "Mixed — 1 up · 1 down · 1 flat", which is
# what its rows actually say. Display-only — this band never reaches a score.
_ACCEL_FLAT_PP = 2.0
_ACCEL_DP = 1          # decimal places shown; classification rounds to this FIRST (see _accel_move)


def _accel_move(delta):
    """Classify a delta as +1 / -1 / 0, judging the number the READER CAN SEE.

    Rounds to the displayed precision BEFORE comparing. Without that, the page and its own
    colours disagree: at the old 0.5pp cutoff, 30 deltas across the universe rendered as exactly
    "0.5pp" and 12 of them were grey while 18 were coloured and counted — identical text, opposite
    treatment, with nothing on screen to explain the difference. Rounding first makes the rule
    inspectable: what you read is what was counted."""
    if delta is None:
        return None
    shown = round(delta, _ACCEL_DP)
    return 1 if shown > _ACCEL_FLAT_PP else -1 if shown < -_ACCEL_FLAT_PP else 0


def _accel_val(stock: pd.Series, key: str):
    """Read a delta off the ENGINE column. Never re-derived from its two inputs here: a display
    that recomputes an engine number drifts from it the moment one side changes (the Fisher
    module/engine-pill lesson). The inputs are shown as evidence, not used as the source."""
    v = stock.get(key, None)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def _accel_row(label: str, delta, lhs, rhs, lhs_tag: str, rhs_tag: str) -> str:
    """One row: the move, then the two numbers it was computed from.

    Missing data renders as an em-dash in muted grey, never as 0.0. A zero here would read as
    "flat" — a confident claim — when the truth is that the figure is absent (CLAUDE.md's semantic
    truth principle, at the display boundary)."""
    move = _accel_move(delta)
    if move is None:
        val_html = f'<span style="font-size:0.82rem;font-weight:800;color:{COLORS["text_muted"]};">—</span>'
        sub = "no data"
    else:
        # Colour and count come from ONE classifier, so a grey row and an uncounted row are always
        # the same row. Deriving the colour separately is how they drift apart.
        clr = (COLORS["green"] if move > 0 else
               COLORS["red"]   if move < 0 else COLORS["text_secondary"])
        val_html = (f'<span style="font-size:0.82rem;font-weight:800;color:{clr};">'
                    f'{delta:+.{_ACCEL_DP}f}<span style="font-size:0.6rem;font-weight:600;'
                    f'color:{COLORS["text_muted"]};">pp</span></span>')
        _f = lambda x: "—" if x is None else f"{x:.1f}%"
        sub = f"{lhs_tag} {_f(lhs)} vs {rhs_tag} {_f(rhs)}"
    # Label and its evidence sit TOGETHER on the left and read as one phrase ("PAT · 3Y 19.6% vs
    # 5Y 69.0%"); only the number is pushed right, so it forms a scannable column. The first cut
    # pinned the evidence to the right edge too, which on a full-width tab left ~1000px of dead
    # space between a row's name and everything that explains it.
    return (
        f'<div style="display:flex;align-items:baseline;gap:10px;'
        f'padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
        f'<span style="font-size:0.72rem;color:{COLORS["text_secondary"]};min-width:64px;">{_esc(label)}</span>'
        f'<span style="font-size:0.6rem;color:{COLORS["text_muted"]};'
        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{_esc(sub)}</span>'
        f'<span style="flex:1;"></span>{val_html}</div>'
    )


def render_trajectory_card(stock: pd.Series):
    """Overview: is the business speeding up or slowing down? Stateless, display-only."""
    growth, margin = [], []
    up = down = known = 0
    for label, key, a, b in _ACCEL_GROWTH:
        d = _accel_val(stock, key)
        m = _accel_move(d)
        if m is not None:
            known += 1
            up += m > 0
            down += m < 0
        growth.append(_accel_row(label, d, _accel_val(stock, a), _accel_val(stock, b), "3Y", "5Y"))
    for label, key, a, b, base_tag in _ACCEL_MARGIN:
        d = _accel_val(stock, key)
        margin.append(_accel_row(label, d, _accel_val(stock, a), _accel_val(stock, b), "Q", base_tag))

    # Headline first (the app's inverted pyramid). Two rules it has to obey:
    #   1. count only what could be MEASURED — "2 of 3" where the third input is absent would
    #      overstate the evidence;
    #   2. account for EVERY measured leg. The first cut read "Mixed — 1 up, 1 down of 3" on a row
    #      set of -1.9 / +1.2 / -0.1: truthful, but it left the reader subtracting to discover the
    #      third leg was flat. A leg inside the +/-0.5pp dead band is a real answer, not a gap, so
    #      it is named rather than implied.
    if known == 0:
        head_txt, head_clr = "Not enough growth history to judge", COLORS["text_muted"]
    else:
        flat = known - up - down
        detail = " · ".join(f"{n} {w}" for n, w in ((up, "up"), (down, "down"), (flat, "flat")) if n)
        if up > down:
            head_txt, head_clr = f"Accelerating — {detail}", COLORS["green"]
        elif down > up:
            head_txt, head_clr = f"Decelerating — {detail}", COLORS["red"]
        elif up == 0 and down == 0:
            head_txt, head_clr = f"Flat — no material move in {known}", COLORS["text_secondary"]
        else:
            head_txt, head_clr = f"Mixed — {detail}", COLORS["gold"]

    def _sub(title: str, tip: str) -> str:
        return (f'<div style="font-size:0.58rem;font-weight:800;color:{COLORS["text_muted"]};'
                f'text-transform:uppercase;letter-spacing:0.7px;margin:10px 0 3px 0;">'
                f'{title}{help_chip("", tip)}</div>')

    st.markdown(
        f'<div style="font-size:0.62rem;font-weight:800;color:{COLORS["text_muted"]};'
        f'text-transform:uppercase;letter-spacing:0.8px;margin:13px 0 4px 0;">Trajectory'
        f'{help_chip("", "Is the business speeding up or slowing down? Growth acceleration compares the 3-year CAGR against the 5-year CAGR — positive means the recent stretch outgrew the longer record. Margin trend compares the latest quarter against its own base. Measured in percentage points (pp). A move smaller than ±2.0pp counts as flat — that is the smallest tenth of all moves in this universe, so it is noise rather than a change of direction.")}'
        f'</div>'
        f'<div style="font-size:0.78rem;font-weight:800;color:{head_clr};margin-bottom:2px;">{_esc(head_txt)}</div>'
        # Two columns, not two stacked blocks. Full-width rows on this tab left ~900px of dead gap
        # between a row's evidence and its number; side by side, each column is about half as wide,
        # the gap closes, and the card is three rows tall instead of six. auto-fit/minmax means it
        # collapses back to one column on a narrow viewport rather than crushing the text.
        # CSS grid rather than st.columns: ui_tearsheet is bound by the no-layout-widgets rule
        # (CLAUDE.md §5, pinned by test_tearsheet_stateless_contract.py).
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));'
        f'gap:0 26px;align-items:start;">'
        f'<div>'
        f'{_sub("Growth Acceleration · 3Y CAGR vs 5Y CAGR", "The second derivative: 3-year CAGR minus 5-year CAGR. Positive = the last three years grew faster than the five-year record, so growth is picking up. Negative = the longer record was stronger and growth is fading.")}'
        f'{"".join(growth)}'
        f'</div><div>'
        f'{_sub("Margin Trend · latest quarter vs its own base", "The latest quarter's margin against the base named on each row — NPM and OPM against the figure one year back, GPM against its 5-year median. The bases differ, so each row states its own; these are margin CHANGES, not accelerations.")}'
        f'{"".join(margin)}'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def render_sector_peer_strip(stock: pd.Series):
    """Mount the 'vs Sector Peers' strip (Overview tab). Stateless — single st.markdown."""
    st.markdown(_sector_peer_strip_html(stock), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MOSL WEALTH CREATION MATRIX — single-card summary of the 30-study signals
# ═══════════════════════════════════════════════════════════════

def render_mosl_wealth_matrix(stock: pd.Series):
    """
    Renders the MOSL Wealth Creation Matrix — a compact, space-optimised card summarising
    the highest-conviction valuation signals from the 30 Annual Wealth Creation Studies.
    PURE DISPLAY — reads pre-materialized columns only; zero math, zero sorting, zero apply().
    Mounted beneath the Mauboussin radar. Defensive: all values via _g()/_esc(), float-guarded.
    Shows: 5-Yr Payback · P/E-vs-RoE margin of safety · Absolute Economic Profit · Atoms/Bits design.
    """
    def _num(key, default=0.0):
        """Float-safe numeric extraction — guards None / NaN / inf into a clean float."""
        v = stock.get(key, default)
        try:
            f = float(v)
            if f != f or f in (float("inf"), float("-inf")):   # NaN or inf
                return default
            return f
        except (TypeError, ValueError):
            return default

    # ── Pull pre-computed metrics defensively ───────────────────────────────
    payback   = _num("payback_trailing_5y", default=float("nan"))
    payback_g = _num("payback_ratio", default=float("nan"))       # growth-adjusted fallback
    pay_show  = payback if payback == payback else payback_g       # prefer trailing; else growth
    pe_to_roe = _num("pe_to_roe_ratio", default=float("nan"))     # PE / sustainable ROE; <1 = MoS
    pe_below  = int(_num("pe_below_roe", 0))
    ep_abs    = _num("economic_profit", default=float("nan"))     # ₹ Cr/yr (reserves × (RoE−CoE))
    ep_pos    = int(_num("economic_profit_positive", 0))
    atb       = str(_g(stock, "atoms_to_bits_label", "Hybrid") or "Hybrid")

    _GOLD, _GREEN, _RED, _BLUE, _MUTE = (
        COLORS["gold"], COLORS["green"], COLORS["red"], COLORS["blue"], COLORS["text_muted"]
    )

    # ── Tile 1: 5-Year Payback ──────────────────────────────────────────────
    if pay_show == pay_show and pay_show > 0:
        pay_clr   = _GREEN if pay_show < 1.0 else (_GOLD if pay_show < 2.0 else _MUTE)
        pay_val   = f"{pay_show:.2f}x"
        pay_badge = "5-YR PAYBACK CLEAR" if pay_show < 1.0 else (
            "Attractive (<2x)" if pay_show < 2.0 else "Full valuation")
    else:
        pay_clr, pay_val, pay_badge = _MUTE, "—", "No earnings basis"

    # ── Tile 2: P/E vs Sustainable RoE (Motilal's original margin of safety) ─
    if pe_to_roe == pe_to_roe and pe_to_roe > 0:
        roe_disc  = (1.0 - pe_to_roe) * 100.0   # positive = PE below ROE = margin of safety
        mos_clr   = _GREEN if pe_below == 1 else _RED
        mos_val   = f"{roe_disc:+.0f}%"
        mos_badge = "P/E BELOW SUSTAINABLE RoE" if pe_below == 1 else "Premium to RoE"
    else:
        mos_clr, mos_val, mos_badge = _MUTE, "—", "RoE basis unavailable"

    # ── Tile 3: Absolute Economic Profit (28th WCS, exact book-value math) ───
    if ep_abs == ep_abs:
        ep_clr   = _GREEN if ep_pos == 1 else _RED
        ep_val   = f"₹{ep_abs:,.0f} Cr"
        ep_badge = "VALUE CREATOR (RoE > CoE)" if ep_pos == 1 else "Value Destroyer (RoE < CoE)"
    else:
        ep_clr, ep_val, ep_badge = _MUTE, "—", "Equity or RoE unavailable"

    # ── Tile 4: Business Design (26th WCS Atoms→Bits) ───────────────────────
    _atb_map = {
        "Bits":   (_BLUE,  "💡", "Asset-light · network-effect scale"),
        "Atoms":  (_GOLD,  "🏭", "Capital-intensive · linear scale"),
        "Hybrid": (_MUTE,  "⚙️", "Mixed physical + digital model"),
    }
    atb_clr, atb_icon, atb_desc = _atb_map.get(atb, (_MUTE, "⚙️", "Mixed model"))

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='sec-head'>🏛️ MOSL Wealth Creation Matrix — 30-Study Signal Summary</div>",
        unsafe_allow_html=True,
    )

    def _tile(color, value, label, sub):
        return (
            f"<div style='flex:1;flex-shrink:0;min-width:150px;white-space:nowrap;"
            f"background:{color}12;border:1px solid {color}40;border-radius:10px;padding:12px 14px;'>"
            f"<div style='font-size:0.6rem;font-weight:700;color:{_MUTE};text-transform:uppercase;"
            f"letter-spacing:0.6px;white-space:nowrap;'>{_esc(label)}</div>"
            f"<div style='font-size:1.35rem;font-weight:900;color:{color};line-height:1.2;"
            f"margin-top:3px;white-space:nowrap;'>{_esc(value)}</div>"
            f"<div style='font-size:0.62rem;color:{color};margin-top:2px;white-space:nowrap;'>"
            f"{_esc(sub)}</div>"
            f"</div>"
        )

    tiles = (
        _tile(pay_clr, pay_val, "⏳ 5-Yr Payback", pay_badge) +
        _tile(mos_clr, mos_val, "📉 P/E vs RoE", mos_badge) +
        _tile(ep_clr, ep_val, "📊 Economic Profit / yr", ep_badge) +
        _tile(atb_clr, f"{atb_icon} {atb}", "🌐 Business Design", atb_desc)
    )

    st.markdown(
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;'>{tiles}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-size:0.6rem;color:{_MUTE};margin-bottom:14px;'>"
        f"Payback = MktCap ÷ trailing-5Y PAT · P/E-vs-RoE = Motilal 1st-Study margin of safety · "
        f"Economic Profit = equity (reserves) × (RoE − {12.0:.0f}% cost of equity)</div>",
        unsafe_allow_html=True,
    )