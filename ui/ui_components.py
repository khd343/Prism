"""
PRISM — UI Components
=====================
Reusable Streamlit UI widgets, cards, and charts.
Premium dark-mode design system.
"""

import base64
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import html as _html
from config import COLORS, TIER_COLORS, CONVICTION_TIERS, UI, FRAMEWORK_CATEGORIES
from ui.shariah_view import SHARIAH_COLORS, shariah_label


# ── PRISM brand mark ──────────────────────────────────────────────────────────
# A geometric triangular prism: a single white beam enters the left face and refracts
# out the right into the product's OWN 6-axis spectrum (purple Moat / green Growth /
# blue Cash / orange Momentum / gold Governance). Embedded as a base64 data-URI <img>
# so it survives Streamlit's HTML sanitizer and renders crisp at any size (16px → header).
_PRISM_SVG = (
    '<svg viewBox="0 0 72 56" xmlns="http://www.w3.org/2000/svg" fill="none">'
    '<line x1="3" y1="29" x2="19" y2="29" stroke="#e6edf3" stroke-width="2.4" stroke-linecap="round"/>'
    '<path d="M28 7 L11 47 L45 47 Z" stroke="#e6edf3" stroke-width="2.4" stroke-linejoin="round" fill="#e6edf3" fill-opacity="0.05"/>'
    '<line x1="39" y1="33" x2="69" y2="21" stroke="#a371f7" stroke-width="2.3" stroke-linecap="round"/>'
    '<line x1="39" y1="33" x2="69" y2="28" stroke="#3fb950" stroke-width="2.3" stroke-linecap="round"/>'
    '<line x1="39" y1="33" x2="69" y2="34" stroke="#58a6ff" stroke-width="2.3" stroke-linecap="round"/>'
    '<line x1="39" y1="33" x2="69" y2="40" stroke="#f0883e" stroke-width="2.3" stroke-linecap="round"/>'
    '<line x1="39" y1="33" x2="69" y2="47" stroke="#d29922" stroke-width="2.3" stroke-linecap="round"/>'
    '</svg>'
)
PRISM_LOGO_URI = "data:image/svg+xml;base64," + base64.b64encode(_PRISM_SVG.encode("utf-8")).decode("ascii")


def prism_mark(px: int = 48) -> str:
    """Inline PRISM logo at the given pixel width (height auto-scaled to the 72×56 viewBox)."""
    h = round(px * 56 / 72)
    return (f'<img src="{PRISM_LOGO_URI}" alt="PRISM" '
            f'style="width:{px}px;height:{h}px;display:block;margin:0 auto;" />')


def inject_css():
    """Inject the premium dark-mode CSS design system."""
    st.markdown(f"""
    <style>
    @import url('{UI["font_url"]}');

    /* ── Global ── */
    html, body, [data-testid="stAppViewContainer"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* ── Responsive Layout ── */
    /* padding-top must clear Streamlit's fixed ~3.75rem header bar, otherwise the first
       content row (the Mandate selector buttons) is hidden underneath it. */
    section[data-testid="stMain"] > div.block-container {{
        max-width: 100%; overflow-x: hidden; box-sizing: border-box;
        padding-top: 3.5rem;
    }}
    /* Blend Streamlit's header bar into the dark theme (keeps the toolbar usable, no white bar) */
    [data-testid="stHeader"] {{
        background: transparent;
    }}
    /* Tighten the default top-of-app gap so the extra header clearance doesn't feel empty */
    section[data-testid="stMain"] > div.block-container > div:first-child {{
        padding-top: 0;
    }}

    /* ── Hero Banner ── */
    .hero-banner {{
        text-align: center; padding: 2rem 1.5rem 1.8rem;
        background: linear-gradient(135deg, {COLORS['gradient_start']} 0%,
                    {COLORS['gradient_mid']} 40%, {COLORS['gradient_end']} 100%);
        border: 1px solid rgba(88,166,255,0.15);
        border-radius: 16px; margin-bottom: 1.5rem;
        position: relative; overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
    }}
    .hero-banner::before {{
        content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(ellipse at 30% 20%, rgba(228,179,65,0.08) 0%, transparent 50%),
                    radial-gradient(ellipse at 70% 80%, rgba(139,92,246,0.06) 0%, transparent 50%);
        pointer-events: none;
    }}
    .hero-icon {{ font-size: 3rem; line-height: 1; position: relative;
        filter: drop-shadow(0 0 14px rgba(255,215,0,0.5)); margin-bottom: 6px; }}
    .hero-title {{
        font-size: 2.4rem; font-weight: 900; position: relative;
        background: linear-gradient(120deg, {COLORS['gold']} 0%, {COLORS['blue']} 50%, {COLORS['green']} 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: 1.5px; line-height: 1.15; margin: 0;
    }}
    .hero-sub {{
        font-size: 0.85rem; font-weight: 500; color: {COLORS['text_secondary']};
        letter-spacing: 2.5px; text-transform: uppercase;
        position: relative; margin-top: 8px;
    }}
    .hero-badge {{
        display: inline-block; font-size: 0.65rem; font-weight: 700;
        color: {COLORS['gold']}; background: rgba(228,179,65,0.10);
        border: 1px solid rgba(228,179,65,0.25); padding: 4px 16px;
        border-radius: 12px; margin-top: 12px; position: relative;
        letter-spacing: 1px;
    }}

    /* ── Metric Strip ── */
    .m-strip {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }}
    .m-chip {{
        background: {COLORS['bg_secondary']}; border: 1px solid {COLORS['border']};
        border-radius: 12px; padding: 14px 0; text-align: center; flex: 1; min-width: 100px;
        transition: all 0.2s ease;
    }}
    .m-chip:hover {{ border-color: {COLORS['border_hover']}; transform: translateY(-1px); }}
    .m-val {{ font-size: 1.5rem; font-weight: 700; color: {COLORS['text_primary']}; line-height: 1; }}
    .m-lbl {{ font-size: 0.65rem; color: {COLORS['text_secondary']}; text-transform: uppercase;
              letter-spacing: 0.6px; margin-top: 4px; }}
    .m-green .m-val {{ color: {COLORS['green']}; }}
    .m-red .m-val {{ color: {COLORS['red']}; }}
    .m-gold .m-val {{ color: {COLORS['gold']}; }}
    .m-blue .m-val {{ color: {COLORS['blue']}; }}
    .m-purple .m-val {{ color: {COLORS['purple']}; }}

    /* ── Stock Cards ── */
    .stock-card {{
        background: {COLORS['bg_secondary']}; border: 1px solid {COLORS['border']};
        border-radius: 14px; padding: 18px 20px; margin-bottom: 10px;
        transition: all 0.2s ease; cursor: default;
    }}
    .stock-card:hover {{
        border-color: {COLORS['border_hover']};
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }}

    /* ── Pill Tags ── */
    .pill {{
        display: inline-block; padding: 3px 10px; border-radius: 10px;
        font-size: 0.7rem; font-weight: 600; margin: 2px 3px; border: 1px solid;
    }}
    .pill-green {{ color: {COLORS['green']}; border-color: rgba(63,185,80,0.3);
                   background: rgba(63,185,80,0.08); }}
    .pill-red {{ color: {COLORS['red']}; border-color: rgba(248,81,73,0.3);
                 background: rgba(248,81,73,0.08); }}
    .pill-gold {{ color: {COLORS['gold']}; border-color: rgba(228,179,65,0.3);
                  background: rgba(228,179,65,0.08); }}
    .pill-blue {{ color: {COLORS['blue']}; border-color: rgba(88,166,255,0.3);
                  background: rgba(88,166,255,0.08); }}
    .pill-purple {{ color: {COLORS['purple']}; border-color: rgba(139,92,246,0.3);
                    background: rgba(139,92,246,0.08); }}

    /* ── Tier Card ── */
    .tier-card {{
        border-radius: 12px; padding: 16px 20px; margin-bottom: 10px;
        transition: all 0.2s ease;
    }}
    .tier-card:hover {{ transform: translateY(-1px); }}
    .tier-header {{
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 8px;
    }}
    .tier-name {{ font-weight: 800; font-size: 1rem; }}
    .tier-count {{
        font-size: 0.75rem; font-weight: 600; padding: 3px 10px;
        border-radius: 8px; background: rgba(255,255,255,0.08);
    }}

    /* ── Section Headers ── */
    .sec-head {{
        font-size: 0.9rem; font-weight: 700; color: {COLORS['text_primary']};
        letter-spacing: 0.3px; margin: 24px 0 10px 0;
        display: flex; align-items: center; gap: 8px;
    }}
    .sec-cap {{
        font-size: 0.72rem; color: {COLORS['text_muted']};
        margin-top: -6px; margin-bottom: 12px;
    }}

    /* ── DataFrames ── */
    div[data-testid="stDataFrame"] > div {{ border-radius: 10px; overflow: hidden; }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: {COLORS['bg_secondary']};
        border-radius: 12px 12px 0 0;
        padding: 6px 6px 0 6px;
        border-bottom: 2px solid {COLORS['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 10px 20px;
        font-weight: 600;
        border-radius: 10px 10px 0 0;
        font-size: 0.85rem;
        color: {COLORS['text_secondary']} !important;
        background: transparent;
        border: 1px solid transparent;
        transition: all 0.2s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {COLORS['text_primary']} !important;
        background: rgba(255,255,255,0.05);
    }}
    .stTabs [aria-selected="true"] {{
        color: {COLORS['gold']} !important;
        background: rgba(228,179,65,0.08) !important;
        border-color: rgba(228,179,65,0.3) !important;
        border-bottom-color: transparent !important;
    }}

    /* ── Sidebar ── */
    .sb-brand {{
        background: linear-gradient(135deg, rgba(139,92,246,0.08) 0%,
                    rgba(228,179,65,0.10) 50%, rgba(63,185,80,0.08) 100%);
        border: 1px solid rgba(228,179,65,0.3); border-radius: 16px;
        padding: 20px 14px 14px; text-align: center; margin-bottom: 16px;
        position: relative; overflow: hidden;
    }}
    .sb-brand::before {{
        content: ''; position: absolute; top: -40%; left: -40%;
        width: 180%; height: 180%;
        background: radial-gradient(circle, rgba(228,179,65,0.08) 0%, transparent 70%);
        animation: sb-pulse 6s ease-in-out infinite;
    }}
    @keyframes sb-pulse {{ 0%,100% {{ opacity: 0.4; }} 50% {{ opacity: 1; }} }}
    .sb-brand-icon {{ font-size: 2.2rem; position: relative; line-height: 1;
        filter: drop-shadow(0 0 8px rgba(228,179,65,0.4)); margin-bottom: 4px; }}
    .sb-brand-title {{
        font-size: 1.15rem; font-weight: 800; position: relative;
        background: linear-gradient(120deg, {COLORS['gold']} 0%, {COLORS['purple']} 50%, {COLORS['green']} 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .sb-brand-ver {{
        display: inline-block; font-size: 0.6rem; font-weight: 700;
        color: {COLORS['gold']}; background: rgba(228,179,65,0.10);
        border: 1px solid rgba(228,179,65,0.3); padding: 2px 10px;
        border-radius: 10px; margin-top: 6px; position: relative;
    }}

    /* ══════════════════════════════════════════════
       TEARSHEET — Premium section styles
       ══════════════════════════════════════════════ */

    /* Hero header card */
    .ts-hero {{
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        border: 1px solid {COLORS['border']};
        border-radius: 20px; padding: 28px 30px; margin-bottom: 14px;
        position: relative; overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.04);
    }}
    .ts-hero::before {{
        content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(ellipse at 85% 50%, rgba(88,166,255,0.06) 0%, transparent 55%),
                    radial-gradient(ellipse at 10% 80%, rgba(139,92,246,0.04) 0%, transparent 50%);
        pointer-events: none;
    }}
    .ts-score-ring {{
        width: 110px; height: 110px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        flex-direction: column; flex-shrink: 0;
        border-width: 3px; border-style: solid;
        box-shadow: 0 0 28px rgba(0,0,0,0.5);
        position: relative;
    }}
    .ts-score-val {{
        font-size: 2.6rem; font-weight: 900; line-height: 1;
    }}
    .ts-score-lbl {{
        font-size: 0.5rem; letter-spacing: 1px; text-transform: uppercase;
        margin-top: 2px; opacity: 0.7;
    }}

    /* Score strip grid */
    .ts-score-strip {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 8px; margin: 10px 0 16px 0;
    }}
    .ts-score-cell {{
        background: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px; padding: 12px 10px;
        text-align: center; transition: border-color 0.2s ease;
    }}
    .ts-score-cell:hover {{ border-color: {COLORS['border_hover']}; }}
    .ts-score-cell-lbl {{
        font-size: 0.58rem; color: {COLORS['text_muted']};
        text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px;
    }}
    .ts-score-cell-val {{
        font-size: 1.9rem; font-weight: 900; line-height: 1;
    }}
    .ts-score-bar-bg {{
        height: 3px; background: {COLORS['bg_tertiary']};
        border-radius: 2px; margin-top: 7px; overflow: hidden;
    }}
    .ts-score-bar-fill {{
        height: 3px; border-radius: 2px;
    }}

    /* Framework grid cards */
    .ts-fw-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
        gap: 8px; margin-top: 8px;
    }}
    .ts-fw-card {{
        border-radius: 10px; padding: 11px 13px;
        border-width: 1px; border-style: solid; transition: transform 0.15s ease;
    }}
    .ts-fw-card:hover {{ transform: translateY(-2px); }}
    .ts-fw-card-head {{
        display: flex; align-items: center; gap: 8px; margin-bottom: 5px;
    }}
    .ts-fw-card-name {{ font-weight: 700; font-size: 0.78rem; }}
    .ts-fw-card-desc {{ font-size: 0.65rem; color: {COLORS['text_muted']}; line-height: 1.4; }}

    /* Flag rows in forensics */
    .ts-flag-row {{
        display: flex; align-items: flex-start; gap: 12px;
        padding: 9px 14px; border-radius: 6px; margin: 4px 0;
        border-left-width: 3px; border-left-style: solid;
    }}
    .ts-flag-sev {{ font-size: 1rem; flex-shrink: 0; margin-top: 1px; }}
    .ts-flag-title {{ font-size: 0.78rem; font-weight: 600; line-height: 1.3; }}

    /* Fisher proxy cards */
    .ts-fisher-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 8px; margin-top: 10px;
    }}
    .ts-fisher-card {{
        border-radius: 9px; padding: 10px 12px;
        border-width: 1px; border-style: solid;
    }}
    .ts-fisher-head {{
        display: flex; align-items: center; gap: 7px; margin-bottom: 4px;
    }}
    .ts-fisher-key {{ font-size: 0.68rem; font-weight: 700; }}
    .ts-fisher-val {{ font-size: 0.85rem; font-weight: 800; }}

    /* Sell alert banners */
    .ts-sell-banner {{
        border-radius: 12px; padding: 14px 18px; margin: 6px 0;
        border-width: 1px; border-style: solid;
        display: flex; align-items: flex-start; gap: 12px;
    }}
    .ts-sell-icon {{ font-size: 1.4rem; flex-shrink: 0; }}
    .ts-sell-title {{ font-size: 0.88rem; font-weight: 800; margin-bottom: 2px; }}
    .ts-sell-body {{ font-size: 0.75rem; }}

    /* Raw data metric cell */
    .ts-raw-cell {{
        background: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px; padding: 10px 12px; text-align: center;
    }}
    .ts-raw-lbl {{
        font-size: 0.6rem; color: {COLORS['text_muted']};
        text-transform: uppercase; letter-spacing: 0.5px;
    }}
    .ts-raw-val {{
        font-size: 1.1rem; font-weight: 800;
        color: {COLORS['text_primary']}; margin-top: 3px;
    }}
    /* Plain-language "?" help affordance + pure-CSS hover tooltip (no widget, no JS, no state) */
    .ts-help {{
        display: inline-flex; align-items: center; justify-content: center;
        width: 13px; height: 13px; margin-left: 4px; vertical-align: middle;
        border: 1px solid {COLORS['text_muted']}; border-radius: 50%;
        font-size: 9px; font-weight: 700; line-height: 1; cursor: help;
        color: {COLORS['text_muted']}; text-transform: none; position: relative;
    }}
    .ts-help:hover {{ border-color: {COLORS['blue']}; color: {COLORS['blue']}; }}
    .ts-help::after {{
        content: attr(data-tip);
        position: absolute; bottom: 160%; left: 50%; transform: translateX(-50%);
        width: 200px; background: {COLORS['bg_primary']};
        border: 1px solid {COLORS['border_hover']}; border-radius: 8px; padding: 9px 11px;
        font-size: 0.72rem; font-weight: 400; line-height: 1.45; letter-spacing: normal;
        text-transform: none; text-align: left; color: {COLORS['text_secondary']};
        white-space: normal; z-index: 9999; pointer-events: none;
        box-shadow: 0 8px 24px rgba(0,0,0,0.55);
        opacity: 0; visibility: hidden; transition: opacity 0.13s ease;
    }}
    .ts-help:hover::after {{ opacity: 1; visibility: visible; }}

    /* KPI strip in forensics */
    .ts-kpi-strip {{
        display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px;
    }}
    .ts-kpi-cell {{
        background: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px; padding: 10px 16px;
        min-width: 100px; text-align: center; flex: 1;
    }}
    .ts-kpi-val {{ font-size: 1.5rem; font-weight: 900; line-height: 1; }}
    .ts-kpi-lbl {{
        font-size: 0.58rem; color: {COLORS['text_muted']};
        text-transform: uppercase; letter-spacing: 0.6px; margin-top: 3px;
    }}
    </style>
    """, unsafe_allow_html=True)


def render_hero_banner(compact: bool = False):
    """Render the main hero banner. compact=True renders a slim one-line brand strip for the
    page top (the sidebar already carries the full PRISM identity, so a second tall hero is
    redundant chrome the daily user scrolls past). Display-only — no widgets, no state."""
    if compact:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px; padding:6px 4px 10px 4px;
                    border-bottom:1px solid {COLORS['border']}; margin-bottom:10px;">
            <span style="display:flex; align-items:center;">{prism_mark(26)}</span>
            <span style="font-size:1.05rem; font-weight:800; letter-spacing:0.5px;
                         color:{COLORS['text_primary']};">{UI['app_title']}</span>
            <span style="font-size:0.78rem; color:{COLORS['text_secondary']};">·&nbsp;{UI['app_subtitle']}</span>
            <span style="margin-left:auto; font-size:0.68rem; color:{COLORS['text_muted']};
                         text-transform:uppercase; letter-spacing:1px;">v{UI['version']} · QUANTAMENTAL INTELLIGENCE</span>
        </div>
        """, unsafe_allow_html=True)
        return
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-icon">{prism_mark(60)}</div>
        <h1 class="hero-title">{UI['app_title']}</h1>
        <p class="hero-sub">{UI['app_subtitle']}</p>
        <div class="hero-badge">v{UI['version']} · QUANTAMENTAL INTELLIGENCE</div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_strip(metrics: list):
    """Render a horizontal metric strip. Each metric: (value, label, color_class)."""
    chips = ""
    for val, label, cls in metrics:
        chips += f'<div class="m-chip {cls}"><div class="m-val">{val}</div><div class="m-lbl">{label}</div></div>'
    st.markdown(f'<div class="m-strip">{chips}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MARKET-STATE "PULSE BAND" — the Market Pulse tab's headline
# ═══════════════════════════════════════════════════════════════
# The tab is named Market Pulse but historically showed only screen counts; the market's actual
# state (breadth / regime / conviction distribution / capital rotation) lived only in the page-top
# banner. This band surfaces it here — breadth-led (the honest, always-moving signal), regime as a
# small derived chip (complementary to the page-top banner, not a duplicate headline). Display-only.

def _pulse_stats(df) -> dict:
    """Pure, vectorized market-state aggregates for the Pulse band. Every field degrades gracefully
    (None / empty) when its column is absent or all-NaN — NEVER raises. No row iteration / no apply."""
    n = int(len(df))

    # ── Breadth: Weinstein stage distribution. Match the unambiguous 'Stage N' substring (robust to
    #    emoji drift); Unknown is the RESIDUAL so the five buckets always sum to n (bar fills 100%). ──
    breadth = None
    if n > 0 and "weinstein_stage" in df.columns:
        s = df["weinstein_stage"].astype("string").fillna("")
        adv  = int(s.str.contains("Stage 2", na=False).sum())
        base = int(s.str.contains("Stage 1", na=False).sum())
        top  = int(s.str.contains("Stage 3", na=False).sum())
        decl = int(s.str.contains("Stage 4", na=False).sum())
        if adv + base + top + decl > 0:                       # present-but-unclassified → unavailable
            breadth = {"Advancing": adv, "Basing": base, "Topping": top,
                       "Declining": decl, "Unknown": max(0, n - (adv + base + top + decl))}

    # ── Regime: same source as the page-top banner; rendered as a small DERIVED chip, not a headline ──
    regime = str(df.attrs.get("detected_market_regime", "SIDEWAYS"))

    # ── Median distance off the 52-week high (the robust, always-moving headline) ──
    off_high = None
    if "dist_52wh" in df.columns:
        _m = df["dist_52wh"].median()                          # skipna by default
        off_high = None if pd.isna(_m) else float(_m)

    # ── Conviction ladder: counts per tier, keyed off the config ladder (DRY, never hardcoded) ──
    ladder = None
    if "conviction_tier" in df.columns:
        _vc = pd.to_numeric(df["conviction_tier"], errors="coerce").value_counts()
        _counts = {int(k): int(v) for k, v in _vc.items() if pd.notna(k)}
        ladder = [(t["tier"], t["emoji"], t["label"], _counts.get(t["tier"], 0))
                  for t in CONVICTION_TIERS]

    # ── Capital rotation: Hot vs Starved sectors ──
    capital = None
    if "sector_capital_phase" in df.columns:
        s = df["sector_capital_phase"].astype("string").fillna("")
        capital = {"hot":     int(s.str.contains("Hot", na=False).sum()),
                   "starved": int(s.str.contains("Starved", na=False).sum())}

    # ── Valuation temperature ──
    med_composite = None
    if "composite_score" in df.columns:
        _m = df["composite_score"].median()
        med_composite = None if pd.isna(_m) else float(_m)
    tailwind_pct = None
    if n > 0 and "sector_tailwind" in df.columns:
        tailwind_pct = float(100.0 * df["sector_tailwind"].fillna(0).astype(float).mean())

    return {"n": n, "breadth": breadth, "regime": regime, "off_high": off_high,
            "ladder": ladder, "capital": capital,
            "med_composite": med_composite, "tailwind_pct": tailwind_pct}


def _pulse_card(title: str, badge_html: str, body_html: str, C: dict, flex: str = "1",
                help_html: str = "") -> str:
    return (
        f'<div style="flex:{flex};min-width:150px;background:{C["bg_secondary"]};'
        f'border:1px solid {C["border"]};border-radius:10px;padding:10px 13px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;gap:6px;">'
        f'<span style="font-size:0.62rem;font-weight:800;letter-spacing:0.05em;'
        f'color:{C["text_muted"]};">{title}{help_html}</span>{badge_html}</div>'
        f'{body_html}</div>')


def _pulse_band_html(stats: dict) -> str:
    """Pure → compact inline-HTML Pulse band. NEVER emits a literal nan/None (missing → '—')."""
    C = COLORS

    def _fmt(v, suf=""):
        return "—" if v is None or pd.isna(v) else f"{v:.0f}{suf}"

    # ── Card 1 · Breadth (headline) + regime chip + off-high caption ──
    regime = _html.escape(str(stats.get("regime", "SIDEWAYS")))
    _r_clr = C["green"] if regime == "BULL" else C["red"] if regime == "BEAR" else C["gold"]
    _r_emo = "🟢" if regime == "BULL" else "🔴" if regime == "BEAR" else "🟡"
    b = stats.get("breadth")
    if b:
        total = sum(b.values()) or 1
        _segs = [("Advancing", C["green"]), ("Basing", C["text_muted"]),
                 ("Topping", C["gold"]), ("Declining", C["red"]), ("Unknown", C["border"])]
        bar = "".join(
            f'<div style="width:{100.0 * b[name] / total:.2f}%;background:{clr};height:100%;"></div>'
            for name, clr in _segs)
        _p = lambda c: f"{100.0 * c / total:.0f}%"
        breadth_inner = (
            f'<div style="display:flex;height:8px;border-radius:4px;overflow:hidden;'
            f'background:{C["bg_tertiary"]};margin:7px 0 5px 0;">{bar}</div>'
            f'<div style="font-size:0.72rem;color:{C["text_secondary"]};">'
            f'{_p(b["Advancing"])} Adv · {_p(b["Declining"])} Decl · {_p(b["Topping"])} Top</div>'
            f'<div style="font-size:0.7rem;color:{C["text_muted"]};margin-top:2px;">'
            f'median stock {_fmt(stats.get("off_high"))}% off 52w-high</div>')
    else:
        breadth_inner = (f'<div style="font-size:0.74rem;color:{C["text_muted"]};margin-top:8px;">'
                         f'Breadth unavailable</div>')
    card_breadth = _pulse_card(
        "MARKET BREADTH",
        f'<span style="font-size:0.72rem;font-weight:800;color:{_r_clr};white-space:nowrap;">'
        f'{_r_emo} {regime}</span>',
        breadth_inner, C, flex="1.7", help_html=help_chip("Market Breadth"))

    # ── Card 2 · Conviction ladder (tiers 1–4; tier 5 'Not Ready' is the noisy majority, omitted) ──
    ladder = stats.get("ladder")
    if ladder:
        chips = " ".join(
            f'<span style="font-size:0.82rem;font-weight:700;color:{C["text_primary"]};">'
            f'{_html.escape(emo)}{cnt}</span>'
            for tier, emo, lbl, cnt in ladder if tier <= 4)
        invest = sum(cnt for tier, emo, lbl, cnt in ladder if tier <= 3)
        ladder_inner = (
            f'<div style="margin:7px 0 4px 0;display:flex;gap:10px;flex-wrap:wrap;">{chips}</div>'
            f'<div style="font-size:0.7rem;color:{C["text_muted"]};">T1–T3 investable: '
            f'<strong style="color:{C["text_secondary"]};">{invest}</strong></div>')
    else:
        ladder_inner = f'<div style="font-size:0.74rem;color:{C["text_muted"]};margin-top:8px;">—</div>'
    card_conv = _pulse_card("CONVICTION", "", ladder_inner, C, help_html=help_chip("Conviction"))

    # ── Card 3 · Capital rotation ──
    cap = stats.get("capital")
    if cap:
        cap_inner = (
            f'<div style="margin-top:7px;font-size:0.78rem;">'
            f'<div style="color:{C["blue"]};">❄️ <strong>{cap["starved"]}</strong> Starved</div>'
            f'<div style="color:{C["orange"]};margin-top:3px;">🔥 <strong>{cap["hot"]}</strong> Hot</div>'
            f'</div>'
            f'<div style="font-size:0.66rem;color:{C["text_muted"]};margin-top:3px;">'
            f'opportunity / caution</div>')
    else:
        cap_inner = f'<div style="font-size:0.74rem;color:{C["text_muted"]};margin-top:8px;">—</div>'
    card_cap = _pulse_card("CAPITAL ROTATION", "", cap_inner, C, help_html=help_chip("Capital Rotation"))

    # ── Card 4 · Valuation temperature ──
    val_inner = (
        f'<div style="margin-top:7px;font-size:1.3rem;font-weight:800;color:{C["text_primary"]};">'
        f'{_fmt(stats.get("med_composite"))}</div>'
        f'<div style="font-size:0.66rem;color:{C["text_muted"]};">median composite</div>'
        f'<div style="font-size:0.7rem;color:{C["text_secondary"]};margin-top:4px;">'
        f'{_fmt(stats.get("tailwind_pct"))}% in tailwind sectors</div>')
    card_val = _pulse_card("VALUATION", "", val_inner, C, help_html=help_chip("Valuation"))

    return (f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;">'
            f'{card_breadth}{card_conv}{card_cap}{card_val}</div>')


def render_pulse_band(df) -> None:
    """Market-state Pulse band for the Market Pulse tab — breadth-led, display-only. Only st.* call."""
    st.markdown(_pulse_band_html(_pulse_stats(df)), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PLAIN-LANGUAGE "?" HELP — single home of the .ts-help chip + glossary
# ═══════════════════════════════════════════════════════════════
# This module owns the .ts-help CSS, so the chip's renderer + glossary live here too (the single
# source of truth). ui_tearsheet.py and ui_scanner.py import help_chip / _RAW_GLOSSARY from here.
# The glossary explains the TERM for a beginner; it NEVER judges the value (good/bad needs thresholds
# = engine drift). A label with no entry simply gets no "?" — the completeness net.
_RAW_GLOSSARY = {
    "Wealth Tier": "The wealth-engine tier — which way is this business's value creation moving? Three clocks: EP% (excess return above the cost of equity), Vel% (this year's change in it), and the 5-year margin tau. CLIMBING★ = earning, improving, confirmed; RECOVERING = not earning yet, improving, confirmed; UNEVEN = the clocks disagree; STALLED = not improving. Price-blind and forensics-blind (the ⚠ marks 8+ red flags or a Schilit fail and never alters the tier); a description, not a recommendation.",
    "Wealth EP%": "Excess return: economic profit ÷ reserves × 100 = ROE minus the 12% cost of equity, exactly — both are built on the same equity base. Scale-free, so a ₹200 Cr and a ₹2,000 Cr business compare fairly. The universe median is NEGATIVE — above zero already beats the median listed company.",
    "Wealth Vel%": "This year's CHANGE in the excess return, in points of the reserves base. The 28th Wealth Creation study's finding is that DIRECTION beats level — a business climbing from a low base has historically outperformed a high earner sliding backwards.",
    # ── All-Data grid: curated residual orphans (coverage audit 2026-06-17) ──
    "Mcap Tier":          "Market-cap tier: Mega (≥₹2L Cr) / Large (≥₹20k) / Mid (≥₹5k) / Small (≥₹500) / Micro (≥₹100) / Nano (<₹100 Cr).",
    "Cyclicality Tier":   "The kind of business by industry — A Deep-Cyclical/Commodity, B Cyclical, C Defensive, D Sensitive/Structural-Growth, E Financials, F Catch-all. A holding-regime hint: cyclicals (A/B) tend to be timing/overlay trades while defensives and structural growers (C/D) are steadier to hold through the cycle. Display-only context, never scored.",
    "Earn Drawdown 5Y":   "The deepest peak-to-trough fall in the company's annual net profit over the last 6 years (current plus 5 prior). Near 0 means earnings only ever rose (a steady compounder); a large value means profits collapsed at some point (a cyclical signature); above 100% means the trough year went into a loss. Capped at 300% (beyond that the figure is dominated by a near-zero peak and isn't meaningful). Display-only context, never scored.",
    "ROE Turnaround":     "ROE is still below 15% but has turned up above its 5-year trend — an early-innings quality inflection (a turnaround bargain).",
    "Category Winner":    "14th-WCS sector Category Winner: top-30% capital efficiency (ROCE) within its sector AND above-market 5-year revenue growth — the sector leader.",
    "Enduring VC":        "18th-WCS Enduring Value Creator: positive economic profit + consistent earnings + a decade of ROE ≥ 15% (above cost of equity) — durable, proven compounding.",
    "Compound Power":     "Sustained compounding power: profit growth clears 15% (3Y), 12% (5Y) and 10% (10Y) — earnings compounding across every horizon.",
    "Steady in Volatile": "A consistent earner (no PAT crashes) operating inside a structurally volatile sector — a steady compounder where its peers are erratic.",
    "QMOM Quality":       "Gray Quantitative-Momentum quality (0–1): the average percentile rank across ROCE, low debt/equity, operating cash flow and gross profitability.",
    "EPS Accelerating":   "EPS growth is ACCELERATING — the rate of earnings growth is itself speeding up (the O'Neil / CAN SLIM earnings-acceleration core).",
    "UU Setup":           "Unknown-&-Unknowable setup (15th WCS): a small/mid-cap (<₹20k Cr) at a payback ratio under 1 with ROE turning up — a rare high-conviction early-multibagger profile.",
    "Fast Creator":       "Fastest-Wealth-Creator setup (MOSL): a small base (<₹4k Cr market cap) at a single-digit P/E with >35% PAT CAGR — the rare profile of the fastest historical multibaggers.",
    "Cyclical Mirage":    "A recent revenue-growth surge masking weak 10-year ROCE — the growth is not backed by durable returns on capital (a low-quality / cyclical mirage).",
    "Dilution Vampire":   "Fast revenue growth (≥30%) but sub-cost-of-capital ROE (<12%) funded by equity dilution — growth that erodes per-share value.",
    "Pledge Re-rate":     "Promoter de-pledging catalyst: the stock was meaningfully pledged (>10% a year ago), has cut pledge by >30% and is now near-clean (<5%) — a classic re-rating trigger.",
    # ── Market Pulse band (the four market-state cards) ──
    "Market Breadth":     "How widely the market is rising or falling — the share of stocks in each Weinstein stage (advancing, basing, topping, declining), plus how far the median stock sits below its 52-week high. Broad advancing is healthy; broad declining is a weak, risk-off tape.",
    "Conviction":         "How the whole universe splits across the conviction tiers right now — 🏆 Crown Jewels, 🥇 Strong Compounders, 🥈 Emerging Quality, 🥉 On Radar. 'T1–T3 investable' sums the top three tiers: the names worth deep work today.",
    "Capital Rotation":   "Where capital is flooding in versus draining out, by sector. 🔥 Hot = sectors whose asset base is growing fastest (heavy investment tends to pressure future returns); ❄️ Starved = low or negative asset growth (classic recovery set-ups).",
    "Valuation":          "A quick read on how richly the market is priced — the median composite score across all stocks, plus the share of stocks sitting in structurally-tailwind sectors. A lower median points to a cheaper or weaker overall tape.",
    # ── Business Quality ──
    "ROCE Current":  "Return on Capital Employed — out of every ₹100 the business puts to work (its own money + debt), how much yearly profit it earns. Higher means a better money-making machine.",
    "ROCE 10Y Med":  "The middle (median) ROCE over the last 10 years — shows whether high returns are durable, not a one-year fluke.",
    "ROCE 5Y Med":   "The middle (median) ROCE over the last 5 years — a more recent read on return quality.",
    "ROCE Δ 10Y":    "Change in ROCE, in percentage points: the median of the last 3 years minus the median of the last 10. Positive means the business earns more on each rupee of capital than it used to — the direction, where ROCE itself is only the level. Blank when the company lacks enough history for the two windows to differ.",
    "ROCE Δ 2Y":     "Change in ROCE, in percentage points, over a SHORT window: the latest year minus the average of the two years before it. Where ROCE Δ 10Y asks whether the business has re-rated over a decade, this asks whether it turned recently — the two agree only loosely, so read them together. Positive means the most recent year beat its own two-year base. Blank when any of the three years is missing, or when ROCE is above 100% — capital employed is then near zero and the ratio carries no information.",
    "ROE Current":   "Return on Equity — yearly profit earned on the shareholders' own money only (excludes debt). How hard your equity is working.",
    "ROE 10Y Med":   "The middle (median) ROE over 10 years — shows if equity returns have stayed strong over a full cycle.",
    "NPM":           "Net Profit Margin — of every ₹100 of sales, how much is left as final profit after all costs and taxes.",
    "NPM 5Y Med":    "The middle (median) net profit margin over 5 years — shows if margins are stable, not a one-off.",
    "OPM":           "Operating Profit Margin — profit from the core business (before interest and tax) per ₹100 of sales.",
    "OPM Δ YoY-Q":   "Change in operating margin, in percentage points: the latest quarter minus the same quarter one year earlier. Like-for-like quarters cancel seasonality — a quarter measured against an annual figure folds the festive-season swing into a number read as structural, which is what this column did before 2026-09-18. Positive means margins are expanding right now. Scored (0.10 of the margin facet). Blank when the year-ago quarter is missing; never filled from a stale basis.",
    "Malik Score":   "Sanjay Bakshi / Malik quality checklist — how many of 5 financial-strength tests the company passes.",
    "Malik Pass":    "Whether the company clears the full Malik quality checklist.",
    "Malik Label":   "A one-word verdict (Strong / Average / Weak) summarising the Malik quality checklist result.",
    "Piotroski":     "Piotroski F-Score — a 0-to-9 financial-health checklist covering profitability, debt and efficiency. 8-9 is very healthy books; under 5 is weak.",
    "Fisher Scal. Score": "Phil Fisher scalability score (0-4) — can this business keep growing without its economics breaking down.",
    "Fisher Quadrant":    "Where the company sits on Fisher's growth-vs-quality map (e.g. Catalyst Play, Laggard).",
    "IBAS Architecture":  "One of Mukherjea's 4 moat sources — durable advantage from how the business is built/organised (its 'architecture').",
    "IBAS Innovation":    "One of Mukherjea's 4 moat sources — advantage from genuine, hard-to-copy innovation.",
    "IBAS Reputation":    "One of Mukherjea's 4 moat sources — advantage from brand trust and reputation customers pay up for.",
    "IBAS Strategic":     "One of Mukherjea's 4 moat sources — advantage from strategic assets (licences, networks, locations). The four IBAS scores average into the overall moat number.",
    "Moat Endurance":     "Whether the company's competitive advantage (moat) is widening, holding, eroding or degrading over time — durability, not just how big the moat is today.",
    # ── Growth ──
    "PAT 5Y CAGR":   "Net profit's smoothed yearly growth rate over 5 years (CAGR = compound annual growth rate).",
    "PAT 3Y CAGR":   "Net profit's smoothed yearly growth rate over the last 3 years.",
    "Rev 10Y CAGR":  "Sales' smoothed yearly growth rate over 10 years — the long-run top-line trend.",
    "Rev 5Y CAGR":   "Sales' smoothed yearly growth rate over 5 years.",
    "Q Rev YoY":     "Sales in the latest quarter versus the same quarter a year earlier. This is a "
                     "QUARTERLY figure despite the source column's annual-sounding name — do not read "
                     "it against the 5Y/3Y CAGRs as if it were the latest full year.",
    "EPS 5Y CAGR":   "Earnings-per-share smoothed yearly growth over 5 years — profit growth on a per-share basis (accounts for dilution).",
    "Q EPS YoY":     "Earnings per share in the latest quarter versus the same quarter a year "
                     "earlier — a QUARTERLY figure, not the latest full year.",
    "Q PAT YoY":     "Latest quarter's net profit versus the same quarter last year — the most recent profit trend.",
    "Op Leverage":   "Operating leverage — whether profit is growing faster than sales (a sign fixed costs are being spread over more revenue).",
    "Op Lev (3Y)":   "Operating leverage over 3 years — how much faster (or slower) profit grew than sales. Positive means profits are scaling faster than revenue.",
    "Lynch Category":"Peter Lynch's stock type — Fast Grower, Stalwart, Slow Grower or Turnaround — which sets how to judge it.",
    # ── Cash & Debt ──
    "CFO/PAT":       "Cash from operations divided by reported profit. Near or above 100% means profits are backed by real cash; well below is a warning.",
    "FCF Yield":     "Free cash flow (cash left after running and maintaining the business) as a % of the company's market value.",
    "FCF/CFO":       "How much of the operating cash survives as free cash after capital spending.",
    "FCF/PAT":       "Free cash flow versus reported profit — another check that profit becomes spendable cash. Both from the same year: while the cash-flow statement is a year behind the latest profit figures, last year's profit is used.",
    "FCF Imputed":   "'Yes' means free cash flow wasn't reported directly and we estimated it from operating cash — treat it as approximate.",
    "FCF Reconstructed":"'Yes' means free cash flow was rebuilt from its parts rather than taken raw — so the FCF figure here is not a direct report.",
    "SSGR":          "Self-Sustainable Growth Rate — the fastest the company can grow using only its own profits, with no new debt or share sales.",
    "SSGR Cushion":  "How much head-room there is between what the company can self-fund (SSGR) and how fast it's actually growing. Negative means it must borrow or dilute to grow.",
    "D/E Ratio":     "Debt-to-Equity — total borrowings versus shareholders' money. Higher means more financial risk.",
    "Int Coverage":  "Interest coverage — how many times operating profit covers the interest bill. Higher is safer.",
    "Current Ratio": "Short-term assets divided by short-term dues — above 1 means it can cover near-term bills.",
    "Tax Rate Est":  "An estimate of the tax the company effectively pays on its profit.",
    "Asset Growth":  "How fast the balance sheet (total assets) is expanding. Very fast asset growth can signal over-investment.",
    "CFROIC":        "Cash Return on Invested Capital — like ROCE but counts only real cash profit, a check that returns are backed by actual cash.",
    "Ext Financing": "External financing to assets — how much the company is raising from outside (debt + equity). Negative means it's returning cash to investors.",
    "Capital Alloc": "A label for how management deploys cash (e.g. returning capital, reinvesting, raising).",
    "Sector Capital":"Whether the company's sector is flooded with new capital (often bad for future returns) or starved of it (often good).",
    # ── Valuation ──
    "PE":            "Price-to-Earnings — the share price divided by yearly profit per share; roughly how many years of current profit you pay for the stock.",
    "Fair PE (QGLP)":"An estimated 'fair' PE for this company based on its quality and growth — compare it to the actual PE.",
    "Industry PE":   "The typical PE of this company's industry — context for whether it's cheap or dear versus peers.",
    "P/B":           "Price-to-Book — share price versus the company's net asset (book) value per share.",
    "P/S":           "Price-to-Sales — the share price versus the company's yearly sales per share. Useful when profits are thin, lumpy or negative and PE becomes meaningless.",
    "FGV":           "Future Growth Value — the slice of today's share price that rests on FUTURE growth rather than the profits the company already makes. High means a lot of growth is already priced in.",
    "PEG":           "PE divided by growth — a way to judge if the PE is justified by growth. Around 1 is often considered fair.",
    "PEG Zone":      "A simple band (cheap / fair / stretched) based on the PEG ratio.",
    "Earnings Yield":"Profit per share divided by price — the flip side of PE, shown as a % (like an interest rate the earnings 'pay').",
    "Dividend Yield":"Dividends paid per share over the last 12 months divided by the share price, as a %. A 0 means the company paid no dividend in that window — a statement, not missing data. Read it beside Earnings Yield: a dividend yield at or above the 10-year G-Sec rate (~7%) is rare and is MOSL Study 16's buy signal.",
    "PE vs 10Y Med": "Today's PE versus the stock's own 10-year average PE — is it expensive or cheap against its own history.",
    "EV/EBITDA Dir": "Enterprise value to EBITDA — a debt-aware valuation multiple ('Dir' = taken directly, not proxied).",
    "Payback Ratio": "MOSL payback — the price you pay (market cap) divided by the company's expected cumulative profit over the next 5 years. Below 1× means under ~5 years of earnings covers the price; lower is a faster payback.",
    "P/E vs ROE MoS":"Margin of safety from comparing the PE you pay against the quality (ROE) you get.",
    "Valuation Scr": "The engine's overall valuation score (0-100). Higher = cheaper for the quality.",
    "O'Shaughnessy VC":"O'Shaughnessy Value Composite — a combined cheapness rank across several value measures (PE, PB, PS, EV/EBITDA, cash flow).",
    "Trending Value":"O'Shaughnessy's signal: statistically cheap AND already showing price momentum.",
    "Buy Zone":      "A timing label for whether the current price sits in a sensible entry area.",
    # ── Ownership & Governance ──
    "Promoter %":    "How much of the company the founders/controlling owners (promoters) hold. High skin-in-the-game is usually reassuring.",
    "Pledge %":      "How much of the promoters' shares are pledged as loan collateral. High pledging is a governance risk.",
    "FII %":         "Stake held by Foreign Institutional Investors.",
    "DII %":         "Stake held by Domestic Institutional Investors (Indian mutual funds, insurers).",
    "Promoter Chg":  "Change in promoter holding in the latest quarter. Buying is a positive sign; selling can be a warning.",
    "Promoter 1Y Δ": "Change in promoter holding over the last year, in percentage points. Promoters rarely move in a single quarter, so a year shows the direction: steady buying is a positive sign; steady selling can be a warning.",
    "Promoter 3Y Δ": "Change in promoter holding over 3 years — the longer-term ownership trend.",
    "FII Chg":       "Change in foreign-institution holding in the latest quarter.",
    "DII Chg":       "Change in domestic-institution holding in the latest quarter.",
    "Smart Money":   "An accumulation read that blends recent trading volume, institutional (FII/DII) flow, and price strength. Volume is the heaviest input, so it reflects buying interest broadly — not purely institutional money.",
    "Gov Bonus":     "Governance bonus — a score rewarding clean ownership signals (high promoter skin, insiders buying, no pledging, no dilution).",
    "Mgmt Integrity":"A 0-3 read on management trustworthiness from accounting and ownership behaviour.",
    "Dilution Flag": "Flags whether the company has been issuing lots of new shares (diluting existing holders). Two cases are deliberately NOT treated as dilution: a share count that jumps 1.5x or more in a year (a bonus, split, rights issue or IPO re-basing — the share row shows the raw counts and 'restructuring'), and a company listed within the last ~2 years, whose prior-year share count predates listing so the comparison is not a measurement (shown as 'pre-listing baseline'). Both land on the Watch tier: the count moved, so they are never certified zero-dilution, but they are not hard-rejected on a number that is not dilution. Below 1.5x, a jump of more than 10% counts as dilution even when it is a bonus, a rights issue or shares held by an employee trust that only this year's count includes: PRISM's data can't tell those apart from a placement.",
    "Liquidity": "How much of the stock changes hands a day: the average value traded over the last 20 days (average volume × price), in ₹ crore. It decides whether you can buy in your size — keep one order under about 5% of a day's traded value, so the market can take it without moving the price much (₹1 Cr a day → orders up to about ₹5 lakh). A buyability check, not a score: liquid and illiquid stocks have done about as well on median, and nothing in PRISM's scores reads it. A small or BSE-only stock may also be under exchange surveillance (ASM/GSM), which traded value cannot show.",
    "Recently Listed": "Listed within roughly the last two years. Shown only for young listings, on purpose: the source's listing-age figure is capped for older companies (over 1,100 stocks read the same maximum), so it cannot be used as a general company age. Two things follow. There is little public track record behind the scores. And the prior-year share count predates listing, so the dilution comparison is not a measurement — the share row reads 'pre-listing baseline' instead of a percentage, and the dilution tier is set to Watch, never Clean and never Hard Reject.",
    # ── Technical & Momentum ──
    "CRS 50D":       "Comparative Relative Strength over ~50 days — how the stock's price is doing versus the market, recently.",
    "CRS 26W":       "Relative strength versus the market over ~26 weeks (about 6 months).",
    "CRS 52W":       "Relative strength versus the market over ~52 weeks (about a year).",
    "RS Composite":  "A blended relative-strength rank — leadership versus the whole market (higher = stronger leader).",
    "RSI 14D":       "Relative Strength Index (14-day) — a 0-100 momentum gauge; very high can mean overbought, very low oversold.",
    "Vol Ratio":     "Recent trading volume versus its own average — above 1 means unusually heavy activity.",
    "Dist 52WH":     "How far below its 52-week high the price is. Near the high (small number) shows strength.",
    "VSTOP Green":   "Whether the price is above its volatility-stop trend line (a simple 'trend is up' check).",
    "Breakout Scr":  "A score for how close the stock is to breaking out of a price base to new highs.",
    "Off ATH":       "How far the price is below its all-time high, in % (0% = at the all-time high). At the high, no earlier buyer is sitting on a loss and waiting to sell at breakeven. A stock can make a 52-week high and still sit far below an older peak, which is why this reads beside Breakout. For a recent listing it is the high since listing. It is also one of the inputs of the Breakout score.",
    "Momentum Scr":  "The engine's overall price-momentum score (0-100), blending relative strength, trend quality, breakout proximity, volume confirmation and sector leadership.",
    "Weinstein Stage":"Stan Weinstein's stage of the price cycle — Stage 2 (advancing) is the buy zone, Stage 4 (declining) is avoid.",
    # ── Forensic Summary ──
    "Red Flags":     "How many accounting/governance warning signs fired, out of all the forensic checks. Fewer is better.",
    "Forensic Scr":  "An overall accounting-cleanliness score (0-100). Higher means fewer warning signs.",
    "Forensic Mult": "The penalty multiplier the forensic flags apply to the final score — 100% means no penalty, lower means the score was cut for risk.",
    "Schilit Pass":  "Passes Howard Schilit's Financial Shenanigans screen (✓): at most 2 of its 4 accounting checks fired — earnings manipulation, cash-flow distortion, a leverage trap on the balance sheet, and inventory or receivables bloat. A separate screen from the red-flag count: a fail keeps Soundness below SOUND, turns its Forensics reading red and puts a ⚠ beside the Wealth tier.",
    "Accruals Ratio":"The gap between reported profit and real cash. A big positive gap warns profit isn't turning into cash; negative (like -0.10) is healthy/conservative. Profit and cash are taken from the same year: while the cash-flow statement is a year behind the latest profit figures, last year's profit is used.",
    "Econ Profit":   "Economic profit — profit left after charging for the cost of all the capital used. Positive means genuine value creation.",
    "EP Spread":     "Economic-profit spread — the return on capital minus the cost of capital (ROIC − WACC). Positive means it earns more than its capital costs.",
    "Earnings Power":"Heiserman's earnings-power box — whether profits are backed by both strong economics and real cash.",
    "QGLP Score":    "Motilal Oswal's QGLP score — Quality, Growth, Longevity, Price combined into one number.",
    "QGLP Pass":     "Whether the stock clears the full QGLP quality-growth-price screen.",
    "Composite Scr": "The final blended score (0-100) after quality, momentum and the forensic penalty — the engine's headline number.",
    "Conviction Tier":"The final conviction bucket (Tier 1 = highest conviction, Tier 5 = lowest), set by the composite score after penalties.",
    # ── MOSL Wealth Creation Signals ──
    "Corporate Class":"Motilal Oswal's Great / Good / Gruesome label for capital-allocation quality (Great creates value, Gruesome destroys it).",
    "EMC Sector-Beat":"Economic-moat check — in how many of 5 timeframes the company's returns beat its sector.",
    "EMC Flag":      "Whether the company shows a durable economic moat versus its sector.",
    "CAP Years":     "Competitive Advantage Period — for how many years (of 5 checked) returns stayed above the cost of capital.",
    "GAP Years":     "Growth Advantage Period — for how many years (of 3 checked) growth stayed above a high bar.",
    "CAP-GAP Score": "A combined score for how long the company has sustained both high returns and high growth.",
    "Consistency Champ":"Whether profits have grown steadily (a 'consistent' compounder) rather than lumpily.",
    "PAT Falls >10%":"In how many of the last 5 years profit fell more than 10% — a volatility/consistency check.",
    "Volatile Flag": "Flags an earnings profile that swings a lot year to year.",
    "EP Quintile":   "Which fifth (1 = best) the company falls into on economic-profit power versus the universe.",
    "EP Top Q1/Q2":  "Whether the company is in the top two-fifths for economic-profit creation.",
    "Winner Category":"Whether it sits in a sector enjoying a structural tailwind.",
    "Sector Leader":  "How strong a leader the company is inside its own sector (0-100), judged against sector peers rather than the whole market.",
    "Winning Invest.":"Whether it's a category leader inside a winning sector.",
    "100x Candidate":"Passes Motilal Oswal's tough small-cap screen for stocks that could compound enormously over the long run.",
    "Mid→Mega":      "A mid-cap candidate with the profile to grow into a mega-cap.",
    "Bruised Blue Chip":"A high-quality company trading unusually cheap (below 2x book) after a setback.",
    "Growth-Value Trap":"Warns of a company that grows but earns less than its cost of equity — growth that destroys value.",
    "Cyclical Peak Trap":"Warns of a commodity/cyclical stock that looks cheap at peak-cycle profits but is actually expensive.",
    "Atoms/Bits":    "Whether the business is physical-goods ('Atoms') or asset-light/digital ('Bits'), which changes how to value it.",
    "PSG":           "Price-to-Sales-to-Growth — a sales-based valuation lens useful for fast growers whose PE/PEG can mislead.",
    # ── Layer-1 hero + Layer-2 scorecard jargon (the terms the casual reader meets FIRST) ──
    # The 6 verdict axes (verdict_axis_*) — explain what each axis WEIGHS, never the value.
    "Moat Axis":       "The Soundness scorecard's competitive-advantage axis — how durable and wide the company's moat is (returns on capital + IBAS moat sources). One of the 6 axes the soundness verdict weighs.",
    "Growth Axis":     "The Soundness scorecard's growth axis — how fast and how durably sales and earnings are compounding. One of the 6 axes the soundness verdict weighs.",
    "Valuation Axis":  "The Soundness scorecard's valuation axis — how the price compares to the quality and earnings you get (PE vs fair PE, earnings yield, payback). One of the 6 axes.",
    "Balance Axis":    "The Soundness scorecard's balance-sheet axis — financial strength and debt safety (debt-to-equity, interest cover, net cash). One of the 6 axes.",
    "Governance Axis": "The Soundness scorecard's governance axis — ownership quality and management trust (promoter skin-in-the-game, pledging, dilution). One of the 6 axes.",
    "Forensics Axis":  "The Soundness scorecard's forensics axis — accounting cleanliness and warning-sign count (Piotroski, red flags, balance-sheet bloat). One of the 6 axes.",
    # Deep Signals strip — cross-cutting synthesis metrics.
    "WCS":            "Wealth-Creation Score — a 0-to-10 composite of the Motilal Oswal wealth-creation tests (quality, growth and longevity together). Higher means more of those traits are present.",
    "Econ-Profit":    "Economic profit — the profit left after charging for the cost of ALL the capital the business uses, shown here in ₹ crore. Positive means it earns more than its capital costs.",
    "VCR":            "Value-Creation Ratio — the company's return on capital divided by its ~12% cost of equity. Above 1× means it earns more than its capital costs (genuine value creation); around 1× is break-even.",
    "Terms-of-Trade": "Working-capital terms-of-trade — the gap in days between how fast the company collects from customers and pays its suppliers. Positive means suppliers fund its working capital.",
    "Cash-Machine":   "Cash-Machine score (0, 50 or 100) — how reliably the business turns reported profit into operating cash: 100 = all profit converts to cash with surplus free cash, 50 = solid conversion, 0 = paper profits.",
    # Entry Timing strip — momentum reads (the WHEN, not the WHAT).
    "RS":             "Relative Strength — how the stock's price has performed against the whole market (0-100). Higher means it is leading the market.",
    "Traj":           "Price trajectory — the slope/shape of the recent price trend; a positive value means the trend is pointing up.",
    "EPS-Accel":      "Earnings acceleration — whether profit growth is speeding up (▲) or slowing down (▼) versus the prior period, not merely growing.",
    "Vol":            "Volume score (0-100) — whether recent trading volume is confirming the price move; higher means heavier, conviction-backed activity.",
    # Hero headline numbers.
    "Composite Score":   "The engine's headline 0-100 score — quality, momentum and the forensic penalty blended into one number. It sets the conviction tier shown beside it.",
    "Evidence Coverage": "How much real data the score rests on — the % of the core ranked inputs that were actually present (not estimated). A high score on low coverage is less reliable.",
    # The 5 quality/momentum sub-scores in the score strip (the building blocks of the composite —
    # distinct from the 6 verdict axes above, which are the verdict's display lenses).
    "Moat Score":        "PRISM's 0-100 sub-score for the durability of the competitive advantage (returns on capital + IBAS moat). A building block of the composite.",
    "Growth Score":      "PRISM's 0-100 sub-score for the strength and durability of earnings and revenue growth. A building block of the composite.",
    "Improvement Score": "PRISM's 0-100 sub-score for whether the business got BETTER over the last year: profit growth, revenue growth and the change in operating margin, each versus the year before and ranked against the whole market, in equal thirds. 20% of the quality score. Added because a study of the next three years' winners found recent improvement predicted returns where margin and valuation LEVELS did not, and the most recent year predicted best. A loss year gives no profit-growth figure, and a stock with no evidence at all scores a neutral 50.",
    "Cash Score":        "PRISM's 0-100 sub-score for how strongly the business turns profit into real cash (CFO/PAT, free cash flow). A building block of the composite.",
    "Momentum Score":    "PRISM's 0-100 sub-score for price strength and trend (relative strength, breakout proximity, volume). A building block of the composite.",
    "Governance Score":  "PRISM's governance bonus — points for clean ownership (high promoter skin-in-the-game, insiders buying, no pledging, no dilution). Added to the composite, not multiplied.",
    # ── All Data tab — pass-3 surfaced orphans (verified-alive 2026-06-17) ──
    "Moat Endur ×":      "Current ROCE divided by its own 10-year median — above 1.0× means returns on capital are running ABOVE the company's own history (moat widening); below 1.0× means eroding.",
    "Elite ROE":         "Fires when Return on Equity is ≥ 35% — Motilal Oswal's 6th-study 'elite' bar, a rare tier of ultra-high-return franchises.",
    "ROE Rising":        "Fires when current ROE is above BOTH its 5-year and 10-year medians — return on equity on an improving multi-year trajectory.",
    "PAT 1Y Δ %":        "The one-year change in net profit, in % — positive means profit grew, negative means it fell year-on-year.",
    "Value Migration":   "Fires when the company is in the top quartile of its sector by revenue growth — a sign that value (demand/share) is migrating toward this business within its sector.",
    "CWIP/FA %":         "Capital-Work-in-Progress as a % of fixed assets — how much new capacity (plants/projects) is under construction versus assets already in place. Higher = more aggressive expansion.",
    "EBITDA→PAT Gap":    "The drag between operating profit and net profit — depreciation + interest + tax as a % of EBITDA. It is NOT a tax rate; a high gap means a heavy depreciation/interest/tax load.",
    "Supplier Float":    "A 0-100 score of how much suppliers and customers fund the company's growth — a negative cash-conversion cycle (collect before you pay) scores high (≈-120 days ≈ 100). Zero for financials.",
    "Negative WC":       "Fires when the cash-conversion cycle is negative — the business collects from customers before paying suppliers, so growth is self-funded (a working-capital float advantage).",
    "Payoff Ratio":      "Mauboussin payoff multiple (≈1-4×) — for undervalued names, fair PE ÷ current PE: the implied upside-to-downside reward. Around 1.5× is neutral; higher means more asymmetric upside.",
    "Exp Gap Rank":      "The stock's 0-100 percentile rank on the market-implied expectations gap — how the growth already priced in compares, across the universe, to what the business is likely to deliver.",
    "Trend Score":       "A 0-100 technical trend-quality score blending 200-day moving-average direction, volatility-stop, trend strength (ADX), RSI zone and golden-cross. Higher = a cleaner established uptrend.",
    "Diamond Flags":     "How many of Mukherjea's 'Diamonds in the Dust' forensic checks fired — a second accounting-quality lens distinct from the main red-flag count. Fewer is cleaner.",
    "SQGLP Score":       "Motilal Oswal's SQGLP score (0-5) — how many of the five pillars (Size, Quality, Growth, Longevity, Price) the stock clears. The strictest QGLP variant, adding a small-Size requirement.",
    "QV Score":          "A quantitative value-and-quality composite (Wesley Gray style) blending liquidity, institutional smart-money flow, relative-strength consistency and efficiency. Higher = a stronger quant profile.",
    "Sector Type":       "Whether the company's sector is structurally Consistent (steady compounders) or Volatile (cyclical) — context for how much to trust its earnings stability.",
}


def help_chip(label: str = "", tip: str = "") -> str:
    """Pure-CSS "?" hover affordance — the SINGLE source of the ``.ts-help`` chip markup.

    Looks up the plain-language explanation in ``_RAW_GLOSSARY`` by ``label`` unless an explicit
    ``tip`` is given (explicit wins, mirroring ``_cell``'s ``help=`` override). Returns ``""`` when
    there is nothing to explain — so a term with no glossary entry gets no chip rather than a bare
    "?". No widget, no JS, no state: safe anywhere in the stateless UI layer (CLAUDE.md §5).
    """
    text = tip or _RAW_GLOSSARY.get(label, "")
    if not text:
        return ""
    return f'<span class="ts-help" data-tip="{_html.escape(text)}">?</span>'


def render_stock_card(row: pd.Series, show_scores: bool = True):
    """Render a premium stock card."""
    tier = int(row.get("conviction_tier", 5))
    tc = TIER_COLORS.get(tier, TIER_COLORS[5])

    pills = ""
    # Catalysts — one pill per engine catalyst, named as the sidebar filter names it
    # (pinned by tests/test_catalyst_surfaces.py)
    if row.get("cat_capacity", 0) == 1:
        pills += '<span class="pill pill-blue">🔥 Capacity Explosion</span>'
    if row.get("cat_oplev", 0) == 1:
        pills += '<span class="pill pill-green">🔥 OpLev Inflection</span>'
    if row.get("cat_inst_discovery", 0) == 1:
        pills += '<span class="pill pill-purple">🔥 Inst Discovery</span>'
    if row.get("cat_deleveraging", 0) == 1:
        pills += '<span class="pill pill-gold">🔥 Deleveraging</span>'
    if row.get("cat_lynch_dream", 0) == 1:
        pills += '<span class="pill pill-green">🔥 Lynch Dream</span>'
    if row.get("cat_strong_quarter", 0) == 1:
        pills += '<span class="pill pill-green">🔥 Strong Quarter</span>'
        
    # Frameworks — compact CATEGORY-COUNT chips (the 5 §7 groups) instead of a flat pill list.
    # Reveals the stock's conviction CHARACTER at a glance (quality-moat vs momentum vs value play),
    # and declutters: ~9 framework pills → up to 5 category chips. Full grouped list is on the tearsheet.
    fw_str = row.get("frameworks_passed", "None")
    _passed = set(fw_str.split(", ")) if (fw_str and fw_str != "None") else set()
    if _passed:
        for _cemoji, _clbl, _cclr, _cfws in FRAMEWORK_CATEGORIES:
            _cn = sum(1 for f in _cfws if f in _passed)
            if _cn:
                pills += (
                    f'<span class="pill" style="border-color:{_cclr}66;color:{_cclr};'
                    f'background:{_cclr}14;font-weight:700;">{_cemoji} {_clbl} {_cn}</span>'
                )
            
    # Legacy specific tags
    if row.get("tsunami_signal", 0) == 1:
        pills += '<span class="pill pill-gold">🌊 Tsunami</span>'
    if row.get("net_debt_negative", 0) == 1:
        pills += '<span class="pill pill-green">Net Cash</span>'
        
    # ── ALPHA VECTORS ──
    # 1. Moat Growth Matrix
    mg_quad = row.get("moat_growth_quad", "")
    if "Wealth Creator" in mg_quad:
        pills += f'<span class="pill pill-green">{mg_quad}</span>'
    elif "Quality Trap" in mg_quad:
        pills += f'<span class="pill pill-gold">{mg_quad}</span>'
    elif "Growth Trap" in mg_quad:
        pills += f'<span class="pill pill-blue">{mg_quad}</span>'
    elif "Destroyer" in mg_quad:
        pills += f'<span class="pill pill-red">{mg_quad}</span>'
        
    # 2. Cash Machine (Accrual Anomaly)
    cm_label = row.get("cash_machine_label", "")
    if "Cash Machine" in cm_label:
        pills += f'<span class="pill pill-green">{cm_label}</span>'
    elif "Paper Profits" in cm_label:
        pills += f'<span class="pill pill-red">{cm_label}</span>'
        
    # 3. Buy Zone (Actionability)
    bz_label = row.get("buy_zone_label", "")
    if "Perfect Entry" in bz_label:
        pills += f'<span class="pill pill-green">{bz_label}</span>'
    elif "Extended" in bz_label:
        pills += f'<span class="pill pill-red">{bz_label}</span>'

    # 4. Bruised Blue Chip (WCS 29): use the engine-computed flag (P/B ≤2.0 + sector tailwind)
    #    Emoji 🏛️ matches the _FW_META "Bruised Blue Chip 29" entry (zero-duplicate matrix).
    if row.get("bruised_blue_chip_29", 0) == 1:
        pills += '<span class="pill pill-blue">🏛️ Bruised Blue Chip</span>'

    # 5. 100x Candidate (17th WCS Mouse-to-Elephant) — rare, high-conviction alpha tag
    #    Emoji 🐘 (literal "Mouse-to-Elephant") matches the _FW_META "100x Candidate" entry.
    if row.get("mosl_100x_candidate", 0) == 1:
        pills += '<span class="pill pill-gold">🐘 100x Candidate</span>'

    # 6. Atoms→Bits business design (26th WCS) — surface asset-light "Bits" vs capital-heavy "Atoms"
    _atb = str(row.get("atoms_to_bits_label", "") or "")
    if _atb == "Bits":
        pills += '<span class="pill pill-blue">💡 Bits</span>'
    elif _atb == "Atoms":
        pills += ('<span class="pill" style="border-color:rgba(228,179,65,0.3);'
                  'color:#e3b341;background:rgba(228,179,65,0.06);">🏭 Atoms</span>')

    # ── Verdict strip: gate + forensic + moat quadrant in one scannable line ──
    _gate_ok = row.get("gate_pass", 0) == 1
    _failed_esc = _html.escape(str(row.get("failed_gates", "") or ""))
    if _gate_ok:
        _gate_v = f'<span style="color:{COLORS["green"]};font-size:0.72rem;font-weight:700;">✅ All Gates</span>'
    else:
        _short = (_failed_esc[:38] + "…") if len(_failed_esc) > 38 else _failed_esc
        _gate_v = f'<span style="color:{COLORS["red"]};font-size:0.72rem;font-weight:700;">❌ {_short}</span>'

    _f_lbl = str(row.get("forensic_label", "🟢 Clean") or "🟢 Clean")
    _f_cnt = int(row.get("red_flag_count", 0) or 0)
    if "Clean" in _f_lbl:
        _forensic_v = f'<span style="color:{COLORS["green"]};font-size:0.72rem;">🟢 Clean</span>'
    else:
        _forensic_v = f'<span style="color:{COLORS["red"]};font-size:0.72rem;">🔴 Risk — {_f_cnt}⚑</span>'

    # The moat-growth quadrant is already a coloured pill in the tag row — don't duplicate it here.
    # The status strip is gate + forensic, complementing (not repeating) the verdict chip — and since 2026-10-16
    # the Shariah screen's word, in its colour (ui/shariah_view.py; nothing over a stock it did not screen).
    _dot = '<span style="color:#555;font-size:0.65rem;margin:0 3px;">·</span>'
    _sh = row.get("shariah_status")
    _shariah_v = (f'<span style="color:{SHARIAH_COLORS[_sh]};font-size:0.72rem;font-weight:700;">'
                  f'{shariah_label(_sh)}</span>') if _sh in SHARIAH_COLORS else ""
    _verdict_strip = _dot.join([v for v in (_gate_v, _forensic_v, _shariah_v) if v])

    if show_scores:
        _bars = [
            ("Moat",       row.get("moat_score",      0), COLORS["purple"]),
            ("Growth",     row.get("growth_score",     0), COLORS["green"]),
            ("Cash",       row.get("cash_score",       0), COLORS["blue"]),
            ("Momentum",   row.get("momentum_score",   0), COLORS["orange"]),
            ("Governance", row.get("governance_bonus", 0), COLORS["gold"]),
        ]
        _bar_items = ""
        for lbl, sc, clr in _bars:
            _w = min(100, max(0, float(sc or 0)))
            _bar_items += (
                f'<div style="flex:1;">'
                f'<div style="font-size:0.57rem;color:{COLORS["text_secondary"]};margin-bottom:3px;'
                f'text-transform:uppercase;letter-spacing:0.5px;">{lbl}</div>'
                f'<div style="background:{COLORS["bg_tertiary"]};border-radius:4px;height:5px;overflow:hidden;">'
                f'<div style="width:{_w:.0f}%;height:5px;border-radius:4px;background:{clr};"></div>'
                f'</div>'
                f'</div>'
            )
        _score_bars_html = (
            f'<div style="display:flex;gap:6px;margin-top:11px;padding-top:10px;'
            f'border-top:1px solid rgba(255,255,255,0.05);">{_bar_items}</div>'
        )
    else:
        _score_bars_html = ""

    _esc_name     = _html.escape(str(row.get('name', 'N/A') or 'N/A'))
    _esc_sector   = _html.escape(str(row.get('sector', '') or ''))
    _esc_industry = _html.escape(str(row.get('industry', '') or ''))
    _esc_mcat     = _html.escape(str(row.get('market_category', '') or ''))

    # ── Engine verdict chip: the card LEADS with the decision (scan by SOUND/MIXED/FLAWED) ──
    _vdir   = str(row.get("verdict_direction", "") or "")
    _vemoji = str(row.get("verdict_emoji", "") or "")
    _vclr = {"SOUND": COLORS["green"], "MIXED": COLORS["gold"], "FLAWED": COLORS["text_muted"]}.get(
        _vdir, COLORS["text_muted"])
    _verdict_chip = (
        f'<div style="display:inline-block;font-size:0.78rem;font-weight:900;color:{_vclr};'
        f'letter-spacing:0.6px;white-space:nowrap;margin-bottom:5px;padding:3px 11px;border-radius:11px;'
        f'background:{_vclr}1f;border:1px solid {_vclr}66;">{_vemoji} {_html.escape(_vdir)}</div>'
    ) if _vdir else ""

    # ── 💹 Wealth-tier chip beside the Soundness chip: the two verdicts at browse level.
    # Filter-result coherence — if you filtered Wealth = RECOVERING, the cards SHOW it. N/A and
    # missing render nothing (no chip over no data); ⚠ rides with the tier, never blended.
    _wt = str(row.get("wealth_tier", "") or "")
    _wt_c = {"CLIMBING★": COLORS["green"], "CLIMBING": COLORS["green"],
             "RECOVERING": COLORS["gold"], "UNEVEN": COLORS["gold"],
             "STALLED": COLORS["text_muted"]}.get(_wt)
    _wt_w = " ⚠" if int(row.get("wealth_warn", 0) or 0) == 1 else ""
    _wealth_chip = (
        f'<div style="display:inline-block;font-size:0.78rem;font-weight:900;color:{_wt_c};'
        f'letter-spacing:0.6px;white-space:nowrap;margin:0 0 5px 6px;padding:3px 11px;border-radius:11px;'
        f'background:{_wt_c}1f;border:1px solid {_wt_c}66;">💹 {_html.escape(_wt)}{_wt_w}</div>'
    ) if _wt_c else ""

    card_html = f"""
    <div class="stock-card" style="border-left: 3px solid {tc['border']};">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div style="flex:1;min-width:0;">
                <div style="font-weight:800; font-size:1.05rem; color:{COLORS['text_primary']};">
                    {row.get('tier_emoji', '')} #{int(row.get('rank', 0))} · {_esc_name}
                </div>
                <div style="font-size:0.75rem; color:{COLORS['text_secondary']}; margin-top:2px;">
                    {_esc_sector} · {_esc_industry} · ₹{row.get('market_cap', 0):,.0f} Cr · {_esc_mcat}
                </div>
                <div style="margin-top:5px;">{_verdict_strip}</div>
            </div>
            <div style="text-align:right;flex-shrink:0;margin-left:12px;">
                {_verdict_chip}{_wealth_chip}
                <div style="font-size:1.8rem; font-weight:900; color:{tc['text']};">{row.get('composite_score', 0):.0f}</div>
                <div style="font-size:0.65rem; color:{COLORS['text_muted']};">COMPOSITE</div>
            </div>
        </div>
        <div style="margin-top:8px;">{pills}</div>
        {_score_bars_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_radar_chart(row: pd.Series, title: str = "Quality Radar") -> go.Figure:
    """Create a radar chart for a stock's quality sub-scores."""
    categories = ['Moat', 'Growth', 'Improvement', 'Cash Quality', 'Margins', 'Balance Sheet']
    values = [
        row.get("moat_score", 0),
        row.get("growth_score", 0),
        row.get("improvement_score", 0),
        row.get("cash_score", 0),
        row.get("margin_score", 0),
        row.get("balance_sheet_score", 0),
    ]
    values += [values[0]]  # close the polygon
    categories += [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories,
        fill='toself',
        fillcolor='rgba(139,92,246,0.15)',
        line=dict(color=COLORS['purple'], width=2),
        marker=dict(size=6, color=COLORS['purple']),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=COLORS['bg_secondary'],
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=True,
                          tickfont=dict(size=9, color=COLORS['text_muted']),
                          gridcolor=COLORS['border']),
            angularaxis=dict(tickfont=dict(size=11, color=COLORS['text_primary']),
                           gridcolor=COLORS['border']),
        ),
        showlegend=False,
        title=dict(text=title, font=dict(size=14, color=COLORS['text_primary'])),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=30, l=60, r=60),
        height=350,
    )
    return fig


def render_tier_summary(df: pd.DataFrame):
    """Render conviction tier summary cards."""
    for tier_cfg in CONVICTION_TIERS:
        tier_num = tier_cfg["tier"]
        count = (df["conviction_tier"] == tier_num).sum()
        gate_passed = ((df["conviction_tier"] == tier_num) & (df["gate_pass"] == 1)).sum()
        tc = TIER_COLORS[tier_num]

        st.markdown(f"""
        <div class="tier-card" style="background:{tc['bg']}; border: 1px solid {tc['border']};">
            <div class="tier-header">
                <span class="tier-name" style="color:{tc['text']};">
                    {tier_cfg['emoji']} Tier {tier_num} — {tier_cfg['label']}
                </span>
                <span class="tier-count" style="color:{tc['text']};">{count} stocks</span>
            </div>
            <div style="font-size:0.75rem; color:{COLORS['text_secondary']};">
                {tier_cfg['description']} · {gate_passed} gate-qualified
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_sidebar_brand():
    """Render the sidebar brand card."""
    st.markdown(f"""
    <div class="sb-brand">
        <div class="sb-brand-icon">{prism_mark(42)}</div>
        <div class="sb-brand-title">PRISM</div>
        <div class="sb-brand-ver">v{UI['version']} · QUANTAMENTAL INTELLIGENCE</div>
    </div>
    """, unsafe_allow_html=True)
