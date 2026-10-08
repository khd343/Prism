"""
PRISM — Quantamental Intelligence
=================================
Every lens. One verdict. — Regime-Aware, Master-Driven
Dr. Malik + Raamdeo Agrawal + O'Neil + Mukherjea + Marks + Fisher + Lynch
"""
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import html as _html

import streamlit as st


def _prism_favicon(size: int = 128):
    """Browser-tab favicon = the PRISM refracting-prism mark on a dark app-icon TILE (mirrors
    _PRISM_SVG in ui/ui_components.py). Drawn BOLD on a dark rounded background so it stays legible
    at 16px tab size — thin white strokes on transparency were invisible. PIL-only + inline so it
    runs BEFORE set_page_config without importing the (st-touching) ui package; page_icon takes a
    PIL.Image reliably (an SVG data-URI favicon is flaky across browsers). Strokes are sized as a %
    of the canvas so they survive the browser's downscale to 16/32px."""
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # Dark rounded tile — makes the mark pop on ANY browser tab (light or dark chrome).
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=round(size * 0.22),
                        fill=(13, 17, 23, 255), outline=(48, 54, 61, 255),
                        width=max(1, round(size * 0.015)))
    pad = size * 0.16
    s = (size - 2 * pad) / 72.0          # _PRISM_SVG viewBox is 72×56
    oy = (size - 56 * s) / 2.0
    P = lambda x, y: (pad + x * s, oy + y * s)
    WHITE = (230, 237, 243, 255)

    def _rline(p1, p2, rgb, w):          # round-capped line (emulates the SVG stroke-linecap)
        d.line([p1, p2], fill=rgb, width=w)
        r = w / 2.0
        for cx, cy in (p1, p2):
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb)

    tri = [P(28, 7), P(11, 47), P(45, 47)]
    d.polygon(tri, fill=WHITE)                                       # SOLID prism (legible at 16px)
    _rline(P(1, 29), P(20, 29), WHITE, max(2, round(size * 0.045)))  # bold incoming light beam
    ws = max(2, round(size * 0.058))                                 # bold refracted 5-axis spectrum
    for y, rgb in ((21, (163, 113, 247, 255)), (28, (63, 185, 80, 255)), (34, (88, 166, 255, 255)),
                   (40, (240, 136, 62, 255)), (47, (210, 153, 34, 255))):
        _rline(P(40, 33), P(70, y), rgb, ws)
    return img


st.set_page_config(page_title="PRISM — Quantamental Intelligence", page_icon=_prism_favicon(),
                   layout="wide", initial_sidebar_state="expanded")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import re
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core import (fetch_and_clean_data, run_full_scoring, compute_forensic_signals,
                  apply_forensic_penalty, compute_verdict, run_scoring_pipeline)
from core.data_engine import extract_spreadsheet_id
from ui import (render_moat_growth_matrix, render_fisher_module,
                render_ep_power_curve_module, render_bruised_blue_chip_badge,
                render_multitrillioncap_card, render_forensic_perimeter, render_guru_frameworks,
                render_financial_insights, render_stock_hero, render_verdict_scorecard, render_score_strip,
                render_sell_alerts_panel, render_raw_signals,
                render_canslim_radar, render_sepa_radar, render_schilit_shield, render_dorsey_radar,
                render_outsider_radar, render_marks_radar, render_malik_radar,
                render_lynch_radar, render_mauboussin_radar, render_qglp_radar, render_mosl_wealth_matrix,
                render_piotroski_checklist,
                render_sector_peer_strip,
                render_trajectory_card,
                render_valuation_inversion_and_sizing_cockpit,
                inject_css, render_hero_banner, render_metric_strip, render_pulse_band,
                render_stock_card, help_chip,
                render_radar_chart, render_sidebar_brand,
                render_reference, render_concepts, render_flags, render_frameworks,
                render_wcs_studies, build_reference_markdown)
from ui.ui_discovery import render_discovery_sidebar, clear_all_filters, keep_selected
from ui.ui_scanner import _SCANNER_HEADER_TIPS, ds_view_columns
from ui.ui_components import _RAW_GLOSSARY
from ui.ui_group_views import (MEASURES, OWNERSHIP, OWNERSHIP_GROUP_NOTE, OWNERSHIP_NOTE, SOURCE_COLUMNS,
                               VIEW_MEASURES, VIEWS, group_measures)
from ui.ui_reference_data import CONCEPT_REFERENCE, WCS_STUDIES
from ui.ui_tearsheet import _FLAG_DISPLAY, _FW_META
from ui.ui_scans import (SCANS, CANDIDATES as SCAN_CANDIDATES, LIQUIDITY_STEPS as SCAN_LIQUIDITY_STEPS,
                         liquidity_help as scan_liquidity_help, left_scan as scan_left,
                         left_summary as scan_left_summary, left_legend as scan_left_legend,
                         table_columns as scan_table_columns, card_html as scan_card_html,
                         card_summary_html as scan_card_summary_html,
                         latest_label as scan_latest_label, load_records as load_scan_records,
                         members_label as scan_members_label,
                         record_state as scan_record_state, scan_table, table_config as scan_table_config)
from config import (COLORS, TIER_COLORS, CONVICTION_TIERS, UI, HARD_GATES,
                    QUALITY_WEIGHTS, MOMENTUM_WEIGHTS, COMPOSITE_WEIGHTS,
                    VALUATION_SIGNALS,
                    BAID_SELL_TRIGGERS, MEAN_REVERSION, PEG_ZONES,
                    ANALYSIS_MODES, DEFAULT_ANALYSIS_MODE, FORENSIC_MAX_FLAGS,
                    FORENSIC_PENALTY_TIERS, GOVERNANCE_RISK_MULTIPLIERS)


# ═══════════════════════════════════════════════════════════════
# 3-TIER CACHE SPLIT
# Tier 1: fetch_and_clean_data — CACHED. Only reruns on Clear Cache or new sheet.
# Tier 2: run_full_scoring     — NOT cached. Instant on dropdown change.
# Tier 3: run_forensic_analysis— NOT cached. Instant.
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def get_clean_data(data_source, file_signature: str, sheet_id, _uploaded_dict=None):
    """Tier-1: Expensive data fetch + clean. Heavily cached.
    file_signature (stable string: name+size per file) is NOT underscored, so Streamlit HASHES it —
    it is the real cache key that busts the cache when a different file is uploaded.
    _uploaded_dict IS underscored so Streamlit skips hashing the raw, unhashable stream objects.
    """
    t0 = time.time()
    df = fetch_and_clean_data(data_source, _uploaded_dict, sheet_id)
    elapsed = time.time() - t0
    return df, elapsed

@st.cache_data(show_spinner=False, ttl=900)
def get_data_freshness(data_source: str, sheet_id, file_signature: str):
    """How old is the loaded data? Read from the sheet's own name.

    The ingestion pipeline renames the sheet to the session its numbers came
    from ("PRISM 2026-08-28 Fri"), and Google returns that in the export
    endpoint's Content-Disposition header — so freshness costs one lightweight
    request, no Drive API and no extra column.

    Cached for 15 minutes: the answer only changes once a day, and Streamlit
    reruns the script on every widget interaction. file_signature participates
    in the key so switching sheets re-reads immediately.
    """
    from core.sheet_meta import describe

    return describe(extract_spreadsheet_id(sheet_id) if sheet_id else None, data_source)


def _freshness_color(tone: str) -> str:
    """sheet_meta's tone -> this app's palette."""
    return {"green": COLORS["green"], "gold": COLORS["gold"],
            "red": COLORS["red"]}.get(tone, COLORS["text_muted"])


def get_scored_data(clean_df: pd.DataFrame, analysis_mode: str, scoring_profile: str) -> pd.DataFrame:
    """Tier-2+3: Instant scoring + forensic pass. NOT cached — runs in <0.5s on dropdown change.

    3-step sequencing contract (non-negotiable order):
      1. compute_forensic_signals : Piotroski F-Score → 27 red flags → Schilit 4-checkers →
                                    Cashflow Triangle. Writes forensic_score, forensic_label,
                                    red_flag_count, piotroski_fscore, schilit_forensic_score.
                                    MUST run first: 5 framework gates read these columns.
      2. run_full_scoring         : Hard gates → Quality → Momentum → Governance → Composite →
                                    Framework flags (Diamond, Dhandho, SQGLP Engine, Schilit,
                                    Fisher all read forensic columns from step 1). → Tsunami.
      3. apply_forensic_penalty   : Cascading multiplier on composite_score → conviction tier
                                    reassignment. MUST run last among scoring steps: composite_score
                                    only exists after step 2.
      4. compute_verdict          : Display-only decision-synthesis. Reads the POST-penalty
                                    composite_score / conviction_tier (consistent after step 3) +
                                    the 6 axes → verdict_direction / strength / narrative / risk.
                                    Adds ZERO scoring; only verdict_* label columns.
    """
    return run_scoring_pipeline(clean_df, analysis_mode, scoring_profile)


@st.cache_data(show_spinner=False, max_entries=3)
def _load_vintage_clean(copy_id: str):
    """🔁 Movers, phase 1: an ARCHIVED vintage (a Drive copy of the data sheet, by id) through
    the IDENTICAL loader the live sheet takes — XLSX download, six tabs by name, merge, coerce,
    derive. Cached on the copy id alone: the clean frame does not depend on mode or profile, so
    switching profiles re-scores (phase 2) without re-downloading. max_entries=3 — each entry is
    a full 2,100 × ~700 frame."""
    return fetch_and_clean_data("sheet", None, copy_id)


@st.cache_data(show_spinner=False, max_entries=3)
def _score_vintage(copy_id: str, engine: str, analysis_mode: str, scoring_profile: str, _clean):
    """🔁 Movers, phase 2: the canonical 4-step pipeline with the SAME mode and profile as the
    live frame — so both sides of the diff are scored by the same engine moments apart, and a
    difference can only be the company changing. A scored file from the past would confound
    "the company moved" with "we fixed a bug"; this never can.

    Cached on (copy id, engine hash, mode, profile); `_clean` is underscored so Streamlit does
    not hash the frame on every call — the four strings ARE the identity. The regime is returned
    beside the frame because a cached DataFrame's `.attrs` is not a contract worth relying on
    across pickling."""
    scored = run_scoring_pipeline(_clean, analysis_mode, scoring_profile)
    return scored, str(scored.attrs.get("detected_market_regime", "SIDEWAYS"))


@st.cache_data(show_spinner=False, ttl=900, max_entries=4)
def _load_archive_index(index_id: str):
    """The PRISM Archive Index (dates · quarter labels · copy ids · status) — one small sheet
    the Apps Script archiver maintains. 15-minute TTL: it changes at most once a day."""
    from ui.ui_movers import load_index
    return load_index(index_id)

inject_css()

# Data Source UI
if "data_source" not in st.session_state:
    st.session_state.data_source = "sheet"

with st.sidebar:
    render_sidebar_brand()

    st.markdown("### 📂 Data Source")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Google Sheets", type="primary" if st.session_state.data_source == "sheet" else "secondary", width="stretch"):
            st.session_state.data_source = "sheet"
            st.rerun()
    with col2:
        if st.button("📁 Upload CSV", type="primary" if st.session_state.data_source == "upload" else "secondary", width="stretch"):
            st.session_state.data_source = "upload"
            st.rerun()

    if st.button("🔄 Clear Cache & Reload", width="stretch"):
        # Full refresh: clear the Tier-1 data cache AND the Tier-2 scored-df session cache,
        # so a re-score runs from scratch (picks up engine code changes, not stale labels).
        st.cache_data.clear()
        st.session_state.pop("_scored_df", None)
        st.session_state.pop("_score_key", None)
        # Flag for the post-load confirmation toast — before this, the only evidence a reload
        # actually happened was the Load Time metric silently changing (user-reported 2026-08-29).
        st.session_state["_cache_cleared"] = True
        st.rerun()

    sheet_id = None
    uploaded_dict = None
    data_ready = False

    if st.session_state.data_source == "sheet":
        # DEV CONVENIENCE: PRISM_SHEET_ID env var pre-fills the box so a local dev server boots
        # WITH data (no manual sidebar entry) → fast Playwright/visual-check loop. Unset in prod
        # (Streamlit Cloud never sets it) → identical behaviour to before. The legacy
        # STOCKSCAN_SHEET_ID is still honored (backward-compat) so existing dev/deploy envs keep working.
        _default_sheet = os.environ.get("PRISM_SHEET_ID") or os.environ.get("STOCKSCAN_SHEET_ID", "")
        # CLOUD-RESTART PERSISTENCE (2026-08-29): ?sheet=<id> in the page URL. Community Cloud
        # sleeps/restarts the server, wiping session_state while the BROWSER keeps showing the old
        # page — the link then sits visibly in the box but the new server session never received
        # it, so the first click (Clear Cache & Reload included) dropped to the welcome screen,
        # and Enter on the UNCHANGED text re-submits nothing (the frontend thinks it's already
        # committed). Reproduced live on the deployed app 2026-08-29. The query param survives the
        # restart inside the URL itself: it re-seeds the box (seed-before-instantiate — the same
        # widget-state law as ui_discovery) and data auto-reloads with zero retyping. NOTHING is
        # hardcoded: the param holds whatever the user last typed. The key is deliberately NOT
        # sb_-prefixed — the sidebar's clear-all callback wipes every sb_* key.
        if not _default_sheet:
            _default_sheet = st.query_params.get("sheet", "")
        if "data_sheet_box" not in st.session_state:
            st.session_state.data_sheet_box = _default_sheet
        sheet_id = st.text_input("Google Sheets URL or ID", key="data_sheet_box",
                                 placeholder="Enter Google Sheet ID...")
        if sheet_id:
            data_ready = True
            _sid = extract_spreadsheet_id(sheet_id)
            if _sid and st.query_params.get("sheet") != _sid:
                st.query_params["sheet"] = _sid
    elif st.session_state.data_source == "upload":
        uploaded_files = st.file_uploader(
            "Upload ONE workbook (.xlsx with the 6 tabs) — or all 6 CSV files",
            type=["xlsx", "csv"], accept_multiple_files=True)
        # WORKBOOK PATH (2026-08-30): a single .xlsx (e.g. "PRISM 2026-08-28 Fri.xlsx") carries
        # all six tabs read BY NAME (§0 contract) through the same parse path as Google Sheets.
        # Takes precedence over any CSVs uploaded alongside it; the 6-CSV path below is unchanged.
        _xlsx_uploads = [f for f in (uploaded_files or []) if f.name.lower().endswith(".xlsx")]
        if _xlsx_uploads:
            uploaded_dict = {"workbook": _xlsx_uploads[0]}
            st.markdown(f"✅ **workbook** ← `{_xlsx_uploads[0].name}` — all 6 tabs read by name")
            if len(_xlsx_uploads) > 1:
                st.warning(f"Multiple workbooks uploaded — using `{_xlsx_uploads[0].name}`.")
            data_ready = True
        elif uploaded_files and len(uploaded_files) > 0:
            uploaded_dict = {}
            _unmatched = []
            for f in uploaded_files:
                fname = f.name.lower()
                # Most-specific keywords first — prevents "cashflow_ratios.csv" misrouting to "ratio"
                if   "shareholding" in fname: uploaded_dict["shareholding"] = f
                elif "technical"    in fname: uploaded_dict["technical"]    = f
                elif "cashflow"     in fname or "cash_flow" in fname: uploaded_dict["cashflow"] = f
                elif "balance"      in fname: uploaded_dict["balance"]      = f
                elif "income"       in fname: uploaded_dict["income"]       = f
                elif "ratio"        in fname: uploaded_dict["ratio"]        = f
                else: _unmatched.append(f.name)
            # Show slot-by-slot match status so user sees exactly what mapped where
            _slots = ["ratio", "income", "balance", "cashflow", "shareholding", "technical"]
            _status_lines = []
            for _s in _slots:
                if _s in uploaded_dict:
                    _status_lines.append(f"✅ **{_s}** ← `{uploaded_dict[_s].name}`")
                else:
                    _status_lines.append(f"❌ **{_s}** — not matched")
            if _unmatched:
                for _u in _unmatched:
                    _status_lines.append(f"⚠️ `{_u}` — unrecognized (rename to include the sheet type)")
            st.markdown("\n".join(_status_lines))
            # All 6 required — load_all_csvs raises FileNotFoundError on any missing slot
            if all(_s in uploaded_dict for _s in _slots):
                data_ready = True
            else:
                _missing = [s for s in _slots if s not in uploaded_dict]
                st.warning(f"Missing sheets: {', '.join(_missing)}. Upload all 6 to proceed.")

    # ══ Sidebar Data Source Ends Here ══
    # (Analysis Mode lives in the ⚙️ Config tab; the Command Center it once named was
    #  removed 2026-08-24, and the Scoring Profile beside it 2026-09-20.)

if not data_ready:
    st.info("👋 Welcome! Please select a data source from the sidebar (Google Sheets or Upload CSV) to begin scanning.")
    st.stop()

with st.spinner("🔄 Loading data..."):
    try:
        if uploaded_dict:
            file_sig = "|".join(
                f"{k}:{v.name}:{v.size}"
                for k, v in uploaded_dict.items()
                if v is not None
            )
        else:
            file_sig = f"local_{sheet_id or 'default'}"
        clean_df, load_time = get_clean_data(
            st.session_state.data_source, file_sig, sheet_id, _uploaded_dict=uploaded_dict
        )
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

if st.session_state.pop("_cache_cleared", False):
    st.toast(f"✅ Cache cleared — data reloaded fresh from source in {load_time:.1f}s", icon="✅")

# ═══════════════════════════════════════════════════════════════
# BRAND — compact strip at page top (identity → control → context)
# ═══════════════════════════════════════════════════════════════
render_hero_banner(compact=True)

# ═══════════════════════════════════════════════════════════════
# SCORING CONTROL — one plain widget, living in the ⚙️ Config tab
# ═══════════════════════════════════════════════════════════════
# The old Command Center (six mandate buttons + weights strip + Advanced Override) was REMOVED
# 2026-08-24 after measurement proved it a false promise: three of six mandates were ranking-
# identical (the profile feeds ONLY the QGLP screen — qglp_score/qglp_pass — never the
# composite), and the prominent Q/G/L/P weights strip implied engine re-weighting that never
# happened. The ONE REAL knob remains as a plain selectbox in ⚙️ Config (widget-owned key, no
# callbacks, no canonical/mirror dance — the machinery that caused the prod KeyError crash).
# The second knob — "QGLP Screen Profile" — was REMOVED 2026-09-20 on the same measurement,
# carried one step further: it re-ranked nothing AND rewrote itself on a mode change. The full
# account is in config.MASTER_PROFILES; the pins are tests/test_qglp_profile_fixed.py.
#
# Reading the widget keys HERE — before the Config tab renders them — is correct and current:
# Streamlit commits a changed widget's value to session_state BEFORE the rerun starts. Fresh
# cfg_* keys on purpose: resurrected sessions carrying the old adv_*/_w_* keys are ignored.
st.session_state.setdefault("cfg_mode", DEFAULT_ANALYSIS_MODE)   # config owns the default (2026-09-19)

analysis_mode   = st.session_state["cfg_mode"]
# The QGLP screen is a CONSTANT, not a control (2026-09-20). What stood here was a snap of
# cfg_profile into the active mode's permitted set — and because "Technical Only" permitted
# neither Balanced nor anything the other modes defaulted to, the round trip
# Breakout -> Technical Only -> Breakout left the screen on Momentum and it STUCK: QGLP passers
# 413 -> 631 and 218 stocks' MOSL lens count moved, with composite, rank and every header tile
# unchanged, so nothing on screen contradicted it. Named rather than inlined so the radar, the
# Config readout and get_adaptive_weights all resolve one thing.
scoring_profile = "Balanced"

# ── Scoring ────────────────────────────────────────────────────
_score_key = f"{file_sig}::{analysis_mode}::{scoring_profile}"
if st.session_state.get("_score_key") != _score_key or "_scored_df" not in st.session_state:
    with st.spinner(f"🧭 Scoring — {analysis_mode}..."):
        try:
            _df_scored = get_scored_data(clean_df, analysis_mode, scoring_profile)
            st.session_state["_scored_df"] = _df_scored
            st.session_state["_score_key"] = _score_key
        except Exception as e:
            st.error(f"❌ Scoring error: {e}")
            st.stop()
df = st.session_state["_scored_df"]

if df is None or df.empty:
    st.warning("⚠️ No data returned after scoring. Check your data source or filters.")
    st.stop()

adaptive_w = df.attrs.get("adaptive_weights", {})
# Key metrics
total = len(df)
gate_passed = int(df["gate_pass"].sum())
tier1 = int((df["conviction_tier"] == 1).sum())
tier2 = int((df["conviction_tier"] == 2).sum())
tsunami_count = int(df["tsunami_signal"].sum())
avg_quality = df["quality_score"].mean()
qualified = df[df["gate_pass"] == 1]


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    # Data vintage leads the card: every number below it is only as good as this
    # date, so it is read first and set apart from the counts by a rule.
    _fresh = get_data_freshness(st.session_state.data_source, sheet_id, file_sig)
    _f_clr = _freshness_color(_fresh.tone)
    _f_note = f"{_fresh.status}" if _fresh.is_known else "date not in the sheet name"
    _f_tip = (f"Sheet name: {_fresh.title}" if _fresh.title
              else "Named by the ingestion pipeline on every run")
    st.markdown(f"""
    <div style="background:{COLORS['bg_secondary']}; border:1px solid {COLORS['border']};
                border-radius:12px; padding:12px 14px; margin:10px 0;">
        <div title="{_f_tip}" style="display:flex; align-items:baseline; gap:7px;
                    padding:1px 0 9px 0; margin-bottom:8px;
                    border-bottom:1px solid {COLORS['border']};">
            <span style="color:{_f_clr}; font-size:0.6rem; line-height:1;">●</span>
            <span style="font-size:0.82rem; font-weight:700; color:{COLORS['text_primary']};
                         letter-spacing:0.01em;">{_fresh.label}</span>
            <span style="margin-left:auto; font-size:0.66rem; color:{_f_clr};
                         text-transform:uppercase; letter-spacing:0.05em;
                         font-weight:600;">{_f_note}</span>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:{COLORS['text_primary']}; padding:3px 0;">
            <span>📊 Universe</span><span style="font-weight:700;">{total}</span>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:{COLORS['green']}; padding:3px 0;">
            <span>✅ Gate Passed</span><span style="font-weight:700;">{gate_passed}</span>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:{COLORS['gold']}; padding:3px 0;">
            <span>🏆 Crown Jewels</span><span style="font-weight:700;">{tier1}</span>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:{COLORS['purple']}; padding:3px 0;">
            <span>🌊 Tsunami</span><span style="font-weight:700;">{tsunami_count}</span>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:{COLORS['text_muted']}; padding:3px 0;">
            <span>⏱️ Load Time</span><span style="font-weight:700;">{load_time:.1f}s</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Scored-data export — placeholder here (prominent, in the data panel); FILLED after the filter
    # cascade runs (below) so it reflects the LIVE filter: it downloads exactly the stocks surviving
    # your sidebar filters (every column), or the whole universe when nothing is filtered. Distinct
    # from the Deep Scanner's curated (~40-col) export and the All-Data single-row export.
    from datetime import date as _date
    from ui.ui_export import engine_version, scored_universe_csv, universe_signature
    _scored_dl_ph = st.empty()

    regime = df.attrs.get("detected_market_regime", "SIDEWAYS")
    regime_color = COLORS['green'] if regime == "BULL" else COLORS['red'] if regime == "BEAR" else COLORS['gold']
    st.markdown(f"""
    <div style="background:{COLORS['bg_tertiary']}; border-left:4px solid {regime_color}; padding:8px 12px; margin-bottom:15px; border-radius:4px;">
        <div style="font-size:0.75rem; color:{COLORS['text_muted']}; text-transform:uppercase; letter-spacing:1px;">Detected Regime</div>
        <div style="font-size:1.1rem; font-weight:800; color:{regime_color};">{regime} MARKET</div>
    </div>
    """, unsafe_allow_html=True)


# Discovery filter cascade — built in ui/ui_discovery.py (stateful counterpart to the
# stateless ui_tearsheet). Returns the fully-filtered frame the tabs render.
filt = render_discovery_sidebar(df)

# Fill the scored-data download now that the filtered frame exists. Filter-aware: exports the surviving
# rows (all columns), cached on the scoring key + an EXACT row-identity digest so it re-serializes
# ONLY when the filter result actually changes — not on every rerun (the full-frame to_csv is the
# expensive part). Exact matters: the old count+composite-sum signature collided on live data (two
# different one-stock filter states both hashed to "1|90.00" and the second download served the
# first stock's CSV — 2026-08-29 audit; pinned in tests/test_export.py). No filter → whole universe.
with _scored_dl_ph.container():
    # NAMED BY THE DATA VINTAGE, NOT THE CLICK DATE (2026-09-02). This download is the Cloud-side
    # snapshot: the sheet's own date is the identity of the data (a download of 28-Aug data taken
    # on the 15th is 28-Aug data), so the file is named by it and stamped with it. Only when the
    # sheet carries no date does the click date stand in. The vintage and the engine hash enter
    # the cache key so a refreshed sheet or a redeployed engine can never be served stale bytes.
    _vintage = _fresh.data_date.isoformat() if _fresh.is_known else None
    _fname_date = _vintage or _date.today().isoformat()
    _dl_sig = (f"{_score_key}|{universe_signature(filt['name'])}|{_vintage}|{engine_version()}"
               f"|{_date.today().isoformat()}")
    st.download_button(
        f"📥 Download {len(filt):,} stocks · all {df.shape[1]} cols",
        data=scored_universe_csv(_dl_sig, filt, _vintage, st.session_state.data_source),
        file_name=f"prism_scored_{_fname_date}_{len(filt)}stocks.csv",
        mime="text/csv",
        width="stretch",
        help="Downloads the CURRENTLY FILTERED stocks (every column) as Excel-safe CSV — reflects your "
             "sidebar filters (no filter = the full universe). For a curated column set, use the Deep "
             "Scanner's export.",
    )


# ═══════════════════════════════════════════════════════════════
# STATS STRIP (above tabs)
# ═══════════════════════════════════════════════════════════════
render_metric_strip([
    (f"{total}", "Universe", "m-blue"),
    (f"{gate_passed}", "Gate Passed", "m-green"),
    (f"{tier1}", "Crown Jewels", "m-gold"),
    (f"{tier2}", "Strong", "m-green"),
    (f"{tsunami_count}", "Tsunami", "m-purple"),
    (f"{avg_quality:.0f}", "Avg Quality", "m-blue"),
])

# (The Q/G/L/P weights strip was removed with the Command Center — those weights only ever
# drove the QGLP screen, not the composite; the screen line now lives beside its knobs in ⚙️ Config.)

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tabs = st.tabs(["🏠 Discovery", "🎯 Scans", "🔍 Deep Scanner", "🔬 The Tear-Sheet", "🌊 Market Pulse", "⚙️ Config", "📖 Reference"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: DISCOVERY DASHBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[0]:

    # ── Compact tier strip (replaces 5 stacked tier cards) ────────
    _tier_strip_html = ""
    for _tc in CONVICTION_TIERS:
        _tn   = _tc["tier"]
        _fcnt = int((filt["conviction_tier"] == _tn).sum())
        _acnt = int((df["conviction_tier"] == _tn).sum())
        if _acnt == 0:
            continue
        _ts = TIER_COLORS.get(_tn, TIER_COLORS[5])
        _tier_strip_html += (
            f'<div style="flex:1;min-width:90px;background:{_ts["bg"]};border:1px solid {_ts["border"]};'
            f'border-radius:10px;padding:11px 8px;text-align:center;">'
            f'<div style="font-size:1.5rem;font-weight:900;color:{_ts["text"]};line-height:1;">{_fcnt}</div>'
            f'<div style="font-size:0.67rem;font-weight:700;color:{_ts["text"]};margin-top:3px;'
            f'text-transform:uppercase;letter-spacing:0.4px;">{_tc["emoji"]} {_tc["label"]}</div>'
            f'<div style="font-size:0.57rem;color:{COLORS["text_muted"]};margin-top:2px;">of {_acnt} total</div>'
            f'</div>'
        )
    st.markdown(
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;">{_tier_strip_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Control: how many cards to show ────────────────────────────
    # The "Sort by" pills (Quality / Momentum / PEG) were REMOVED 2026-08-24: they reordered the
    # cards by numbers the cards never display (only the composite is shown, in large type), which
    # also produced the confusing "#37 above #5" rank jumble. The Deep Scanner sorts by those
    # metrics AND shows each as a visible column — that is the view built for numeric comparison.
    # Discovery now always presents the engine's own ranking.
    _, _dc2 = st.columns([6, 2])
    with _dc2:
        _disc_n = st.selectbox(
            "Show", [10, 20, 30, 50], index=1, key="disc_n",
        )

    # LOAD-BEARING sort — not cosmetic. run_full_scoring sorts by the PRE-penalty composite and
    # resets the index; apply_forensic_penalty then multiplies composite_score and re-derives
    # `rank` WITHOUT re-sorting the frame. So `filt` arrives in stale pre-penalty ROW order while
    # its `rank` column is post-penalty (verified live 2026-08-24: rank is non-monotonic in frame
    # order, though it matches the composite ranking exactly). Dropping this sort would render
    # cards in an order contradicting their own #rank badge. Pinned by tests/test_page_order.py.
    _disc_df = (filt.sort_values("composite_score", ascending=False)
                if "composite_score" in filt.columns else filt.copy())
    _shown_n = int(_disc_n or 20)

    # ── No-match dead-end → actionable empty-state (filters can narrow to zero; the engine and
    # the non-empty path below are untouched — this only ADDS the empty branch) ──
    if _disc_df.empty:
        # Name the filter that emptied the list — the sidebar cascade records it (the first filter
        # to take a non-empty frame to zero) and publishes it on filt.attrs. Read from `filt`, not
        # `_disc_df`: attrs need not survive sort_values, but `filt` is the object it was set on.
        _culprit = str(filt.attrs.get("zero_culprit", "") or "")
        _culprit_line = (
            f'<div style="font-size:0.75rem;color:{COLORS["red"]};margin-top:8px;">'
            f'⚠️ <strong>{_html.escape(_culprit)}</strong> removed the last stocks — loosen that one first.'
            f'</div>'
        ) if _culprit else ""
        st.markdown(
            f'<div style="text-align:center;background:{COLORS["bg_secondary"]};'
            f'border:1px dashed {COLORS["border"]};border-radius:12px;padding:28px 18px;margin-top:6px;">'
            f'<div style="font-size:1.4rem;margin-bottom:6px;">🔍</div>'
            f'<div style="font-size:0.95rem;font-weight:800;color:{COLORS["text_primary"]};">'
            f'No stocks match these filters</div>'
            f'<div style="font-size:0.72rem;color:{COLORS["text_muted"]};margin-top:4px;">'
            f'Your active filters narrowed all {len(df):,} stocks out. Loosen one — or clear everything '
            f'and start fresh.</div>{_culprit_line}</div>',
            unsafe_allow_html=True,
        )
        _, _ec, _ = st.columns([3, 2, 3])
        with _ec:
            st.button("🧹 Clear all filters", key="disc_clear", width="stretch",
                      on_click=clear_all_filters)
    else:
        st.markdown(
            f'<div class="sec-head">🏆 Top Picks — {len(_disc_df)} stocks</div>',
            unsafe_allow_html=True,
        )

        # One-time legend for the cards' sub-score bars — explained ONCE here (the scan-friendly
        # alternative to repeating ~100 identical "?" chips, one on every card). Reuses the shared
        # glossary via help_chip, so these definitions never drift from the tearsheet's.
        _SS_LABELS = ("Moat", "Growth", "Cash", "Momentum", "Governance")
        _ss_legend = " &nbsp;·&nbsp; ".join(_l + help_chip(_l + " Score") for _l in _SS_LABELS)
        st.markdown(
            f'<div style="font-size:0.62rem;color:{COLORS["text_muted"]};margin:0 0 10px 2px;">'
            f'Card score bars &nbsp;—&nbsp; {_ss_legend}</div>',
            unsafe_allow_html=True,
        )

        # ── Stock cards with tearsheet shortcut ────────────────────────
        _disc_slice = _disc_df.head(_shown_n)
        for _di in range(len(_disc_slice)):
            _drow = _disc_slice.iloc[_di]
            render_stock_card(_drow, show_scores=True)
            _, _btn_c = st.columns([8, 2])
            with _btn_c:
                if st.button(
                    "🔬 Open Analysis →",
                    key=f"disc_ts_{_di}",
                    width="stretch",
                    type="secondary",
                    help=f"View full tearsheet for {_drow.get('name', '')}",
                ):
                    st.session_state["xray_stock"] = _drow.get("name", "")
                    st.toast(f"🔬 {_drow.get('name', '')} ready — click The Tear-Sheet tab")

        if len(_disc_df) > _shown_n:
            st.markdown(
                f'<div style="text-align:center;padding:12px 0 4px;font-size:0.73rem;'
                f'color:{COLORS["text_muted"]};">'
                f'{len(_disc_df) - _shown_n} more stocks — increase "Show" above</div>',
                unsafe_allow_html=True,
            )


# ── Shared filter row (Market Pulse lenses AND 🎯 Scans) ─────────────────────────────────────────
# Hoisted out of _render_market_pulse on 2026-10-06, unchanged but for the indent, so 🎯 Scans calls
# the SAME cascade-safe row the QGLP / MOSL / Wealth / Movers lenses use: one dialect, one definition.
# Called inside the Market Pulse fragment its widgets still belong to that fragment.
_WT_ORDER = ["BUY★", "BUY", "WATCH★", "WATCH", "AVOID", "N/A"]
_MP_CAP_ORDER = ["Mega Cap", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap", "Nano Cap"]
_MP_CATALYSTS = {   # MUST mirror ui_discovery's _CATALYSTS — pinned by test_market_pulse_tabs
    "🔥 Capacity Explosion": "cat_capacity",
    "🔥 OpLev Inflection":    "cat_oplev",
    "🔥 Deleveraging":         "cat_deleveraging",
    "🔥 Lynch Dream":          "cat_lynch_dream",
    "🔥 Inst Discovery":       "cat_inst_discovery",
    "🔥 Strong Quarter":       "cat_strong_quarter",
}

def _mp_clear_lens(defaults):
    """Reset controls by SETTING each key to ITS OWN default — never `del`: deleting an
    instantiated widget's key lets the frontend resurrect the stale value on the next rerun
    (the Steel bug class; see ui_discovery.clear_all_filters). A MAPPING, not a key list:
    "All" is not a universal default (the Sectors size dial defaults to 5)."""
    for k, v in defaults.items():
        st.session_state[k] = v

def _mp_ms(slot, label, options, key, help_text, counts):
    """THE ONE cascade-safe multiselect used by every Market Pulse filter row (lens rows,
    Sectors, Industry) — one dialect, not three. Mirrors ui_discovery._ms_cascade: state is
    managed here (no `default=` arg, which would trigger Streamlit's default-plus-state
    warning) and a stored pick the cascade has narrowed out is KEPT in the option list with its
    honest count of 0 (ui_discovery.keep_selected — ONE rule for both filter surfaces). It was
    PRUNED until 2026-09-02, which switched the filter off and WIDENED the result: Wealth
    Tier=BUY★ + a sector holding no BUY★ showed the whole sector instead of 0. Keeping the
    value in the options is what stops a keyed widget raising; applying it is what keeps the
    answer honest.

    `counts` maps option -> live count in the CURRENT (already narrowed) frame. DISPLAY only:
    never bake a volatile number into an option VALUE or the pruning stops matching. The UNIT
    of that count differs by filter kind and the caller decides it — a re-aggregating filter
    counts STOCKS (they are what gets re-averaged), a row filter counts the ROWS it hides
    (sectors / industries). Mixing the two silently would be the units trap this row layout
    was deliberately held back for."""
    stored = list(st.session_state.get(key, []))
    st.session_state[key] = stored
    options = keep_selected(options, stored)
    with slot:
        return st.multiselect(
            label, options, key=key, help=help_text,
            format_func=lambda v, _c=counts: f"{v}  ·  {_c.get(v, 0)}",
        )

def _mp_lens_row(frame, prefix, extra_keys=(), with_tier=True):
    """The lens-filter row: Sector · [Wealth Tier ·] Market Cap · Catalyst · Clear.

    MULTI-SELECT AND CASCADING (2026-08-30). Each control takes SEVERAL values (OR within a
    control, AND across controls), and every control's options AND live counts are computed
    from the frame already narrowed by the controls to its LEFT — so picking a sector
    immediately reshapes the tier list and its numbers, and a dead-end combination cannot be
    selected. Mirrors ui_discovery._ms_cascade rather than inventing a second dialect:
    session_state is managed here (never a `default=` arg, which triggers Streamlit's
    default-plus-state warning) and any stored selection is PRUNED to the current options
    every run — mandatory, not optional, because cascading removes options and a keyed widget
    whose stored value is absent from its options raises.

    Measured before building (local CSVs): cascading narrows materially on the small cohorts —
    QGLP 325 stocks, Sector=Auto Ancillaries takes tiers 6 to 4 and catalysts 5 to 4;
    IT-Software takes catalysts 5 to 3 — moderately on MOSL, and barely on Wealth (the whole
    universe, so most sectors still hold every tier). The counts earn their place on all three.

    Returns (filtered_frame, n_active). The 🧹 button materializes ONLY when a filter is
    active, in a FIXED last slot — zero furniture when idle, zero layout jump when it appears.
    """
    # `extra_keys` are controls the TAB owns OUTSIDE this row (the Wealth tab keeps its own
    # Tier control above the table). They filter nothing here — the tab already applied them —
    # but they must count as active and must be reset by the 🧹, or a Clear sitting on screen
    # would visibly clear the row and silently leave another filter running.
    keys = ([f"mp_{prefix}_sec"] + ([f"mp_{prefix}_wt"] if with_tier else [])
            + [f"mp_{prefix}_cap", f"mp_{prefix}_cat"] + list(extra_keys))
    slots = st.columns([2.6, 1.9, 1.9, 2.0, 1.0] if with_tier
                       else [3.0, 2.2, 2.4, 1.1])
    _cf = frame                      # progressively narrowed cascade frame

    # 1 ── Sector (widest funnel first, so every later count reflects it)
    _sec_opts = (sorted(_cf["sector"].dropna().astype(str).unique().tolist())
                 if "sector" in _cf.columns else [])
    _sec_n = _cf["sector"].astype(str).value_counts().to_dict() if _sec_opts else {}
    sel_sec = _mp_ms(slots[0], "Sector", _sec_opts, f"mp_{prefix}_sec",
                    "Show only these sectors. Pick several — a stock matching ANY of them "
                    "qualifies. Counts are live in this lens's cohort.", _sec_n)
    if sel_sec:
        _cf = _cf[_cf["sector"].astype(str).isin(sel_sec)]

    _i = 1
    # 2 ── Wealth tier (counted AFTER the sector narrowing — the cascade)
    if with_tier:
        _wt_opts = [t for t in _WT_ORDER
                    if "wealth_tier" in _cf.columns and (_cf["wealth_tier"] == t).any()]
        _wt_n = _cf["wealth_tier"].astype(str).value_counts().to_dict() if _wt_opts else {}
        sel_wt = _mp_ms(slots[_i], "Wealth tier", _wt_opts, f"mp_{prefix}_wt",
                       "Cross-lens: this framework's passers x the wealth engine's tier. Pick "
                       "several (e.g. BUY★ and BUY together). Counts reflect the sector "
                       "filter above.", _wt_n)
        if sel_wt:
            _cf = _cf[_cf["wealth_tier"].isin(sel_wt)]
        _i += 1

    # 3 ── Market cap
    _cap_opts = [c for c in _MP_CAP_ORDER
                 if "market_category" in _cf.columns and (_cf["market_category"] == c).any()]
    _cap_n = _cf["market_category"].astype(str).value_counts().to_dict() if _cap_opts else {}
    sel_cap = _mp_ms(slots[_i], "Market cap", _cap_opts, f"mp_{prefix}_cap",
                    "Market-cap tiers (the sheet's Market Category). Pick several; counts "
                    "reflect every filter to the left.", _cap_n)
    if sel_cap:
        _cf = _cf[_cf["market_category"].isin(sel_cap)]

    # 4 ── Catalyst (OR across the flags — a stock firing ANY selected catalyst qualifies)
    _cat_n = {l: int(_cf[c].fillna(0).sum()) for l, c in _MP_CATALYSTS.items() if c in _cf.columns}
    _cat_opts = [l for l in _MP_CATALYSTS if _cat_n.get(l, 0) > 0]
    sel_cat = _mp_ms(slots[_i + 1], "Catalyst", _cat_opts, f"mp_{prefix}_cat",
                    "Fast-moving inflections firing right now. Pick several — ANY of them "
                    "qualifies. Only catalysts alive in this cohort are offered.", _cat_n)
    if sel_cat:
        _hit = None
        for _l in sel_cat:
            _m = _cf[_MP_CATALYSTS[_l]].fillna(0) == 1
            _hit = _m if _hit is None else (_hit | _m)
        _cf = _cf[_hit]

    # Reset value for a multiselect is [] — never "All", and never `del` (deleting an
    # instantiated widget's key lets the frontend resurrect the stale value: the Steel bug).
    _defaults = {k: [] for k in keys}
    _n_active = sum(1 for k in keys if st.session_state.get(k))
    with slots[-1]:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if _n_active:
            st.button("🧹 Clear", key=f"mp_{prefix}_clear", width="stretch",
                      on_click=_mp_clear_lens, args=(_defaults,))
    return _cf, _n_active


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: SCANS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ── 🎯 Scans (2026-10-06) ─────────────────────────────────────────────────────────────────────────
# Measured stock lists, one definition each (ui/ui_scans.py): each card shows its own record from the
# saved snapshots (ui/scan_records.json, written by tools/scan_records.py), and the list reads the
# WHOLE market — never the sidebar's `filt` — so it is exactly what that record measured. Display and
# recipes live in the module; app.py owns the scan choice, the liquidity floor and the row click
# (tests/test_scans.py).
def _pick_scan(key):
    st.session_state["scan_sel"] = key


with tabs[1]:
    _sc_recs = load_scan_records()
    _sc_keys = [s["key"] for s in SCANS]
    if st.session_state.get("scan_sel") not in _sc_keys:
        st.session_state["scan_sel"] = _sc_keys[0]
    _sc_sel = next(s for s in SCANS if s["key"] == st.session_state["scan_sel"])
    _sc_latest = scan_latest_label(_sc_recs)       # the design-period record's last snapshot
    _sc_new_label = scan_members_label(_sc_recs)   # the month-end copy 🆕 compares against
    _sc_counts = {s["key"]: int(s["mask"](df).sum()) for s in SCANS}
    st.markdown(
        "<div class='sec-cap'>Stock lists PRISM has measured, each with its record — in-sample means "
        "studied while choosing the recipe. A list narrows the market; the decision stays yours.</div>",
        unsafe_allow_html=True)
    for _sc_col, _sc in zip(st.columns(len(SCANS)), SCANS):
        with _sc_col:
            _sc_state, _sc_rec = scan_record_state(_sc_recs, _sc)
            _sc_on = _sc["key"] == _sc_sel["key"]
            st.markdown(scan_card_summary_html(_sc, _sc_counts[_sc["key"]], _sc_state, _sc_rec, _sc_on),
                        unsafe_allow_html=True)
            st.button("✓ Showing" if _sc_on else "Show list", key=f"scan_open_{_sc['key']}",
                      on_click=_pick_scan, args=(_sc["key"],),
                      type="primary" if _sc_on else "secondary", width="stretch")
    # THE FULL RECORD, ONE CLICK AWAY (2026-10-08). Three full cards side by side were 406 / 408 / 596px
    # tall and pushed the list ~240px below a laptop's first screen, so the cards above carry one honest
    # headline each and the SELECTED scan's whole record — every window, after costs, what was left
    # out, the caveats — sits here at full width, where each line fits on one line.
    _sc_state_sel, _sc_rec_sel = scan_record_state(_sc_recs, _sc_sel)
    with st.expander(f"📊 Full record — {_sc_sel['emoji']} {_sc_sel['name']}: every window or year, after "
                     f"costs, what was left out and the caveats", expanded=False):
        st.markdown(scan_card_html(_sc_sel, _sc_counts[_sc_sel["key"]], _sc_state_sel, _sc_rec_sel, True,
                                   _sc_latest, with_head=False), unsafe_allow_html=True)
    # ONE FUNNEL, TOP DOWN (2026-10-06). The card's recipe ranks the WHOLE market, so every listed stock
    # is exactly the match its record measured. Then the traded-value floor and — only while the
    # sidebar actually narrows the market — the same 🎯 opt-in Market Pulse uses; then the shared lens
    # row (Sector · Wealth tier · Market cap · Catalyst, live counts, 🧹 Clear). Every control narrows
    # the FINISHED list, never the recipe: ranking inside a filtered slice would be a different scan
    # with no record (tests/test_scans.py).
    # ONE ROW of filters (2026-10-07): Liquidity, then the lens row — its 🧹 slot sits empty when idle —
    # and the 🎯 line only while the sidebar narrows, so the default view spends no row on it.
    _sc_narrowed = 0 < len(filt) < len(df)
    if _sc_narrowed:
        st.checkbox(
            f"🎯 Show only my filtered stocks — {len(filt):,} of {len(df):,}", key="scan_use_sidebar",
            help="Narrows this list to the stocks your sidebar filters kept. The recipe still ranks the "
                 "whole market, so every stock shown is the same match it would be unfiltered, and the "
                 "record still describes the whole scan.")
    _sc_scoped = _sc_narrowed and bool(st.session_state.get("scan_use_sidebar", False))
    _sc_lc, _sc_rc = st.columns([1.5, 8.5])
    with _sc_lc:
        _sc_min = st.selectbox(
            "Liquidity", SCAN_LIQUIDITY_STEPS, key="scan_liq",
            format_func=lambda v: "Any" if v is None else f"₹{v:g} Cr/day",
            help=scan_liquidity_help())
    _sc_members = _sc_rec_sel.get("members_at_latest") if _sc_state_sel == "measured" else None
    # 📄 window = the days from the 🆕 baseline to the data's own date (the Movers rule): a result
    # declared inside it arrived after the baseline copy was taken.
    _sc_base_day = _sc_recs.get("members_date") if _sc_recs else None
    _sc_data_day = _fresh.data_date if _fresh.is_known else _date.today()
    _sc_window = (_sc_data_day - _date.fromisoformat(_sc_base_day)).days if _sc_base_day else None
    _sc_base = scan_table(df, _sc_sel, min_cr=_sc_min,
                          keep_ids=set(filt['company_id'].astype(str)) if _sc_scoped else None)
    with _sc_rc:
        _sc_lens, _sc_lens_n = _mp_lens_row(
            df[df["company_id"].astype(str).isin(set(_sc_base["company_id"]))], "scan")
    _sc_tbl = scan_table(df, _sc_sel, members_at_latest=_sc_members, min_cr=_sc_min,
                         keep_ids=set(_sc_lens["company_id"].astype(str)), result_window_days=_sc_window)
    _sc_total = _sc_counts[_sc_sel["key"]]
    if _sc_tbl.empty:
        if _sc_total == 0:
            st.info("No stock matches this scan today.")
        elif _sc_base.empty and _sc_scoped:
            st.info(f"None of the {_sc_total:,} matches is among your {len(filt):,} filtered stocks"
                    f"{' at this traded-value floor' if _sc_min else ''} — untick 🎯 to see the whole scan.")
        elif _sc_base.empty:
            st.info(f"None of the {_sc_total:,} matches clears the traded-value floor — choose a lower one.")
        else:
            st.info(f"None of these {len(_sc_base):,} stocks passes the filters in the row above — "
                    f"🧹 Clear resets them.")
    else:
        _sc_why = " · ".join(w for w in ("your filters" if _sc_scoped else "",
                                         f"₹{_sc_min:g} Cr+ a day" if _sc_min else "",
                                         f"{_sc_lens_n} filter{'s' if _sc_lens_n > 1 else ''} above"
                                         if _sc_lens_n else "") if w)
        _sc_of = (f" of {_sc_total:,} stocks" + (f" ({_sc_why})" if _sc_why else "")
                  if len(_sc_tbl) < _sc_total else " stocks")
        _sc_new = (f" · 🆕 new since {_sc_new_label} · 📄 result declared since then"
                   if _sc_members is not None and _sc_new_label else "")
        # "load it into", not "open": Streamlit cannot switch tabs from code, so the reader clicks the tab.
        st.caption(f"{len(_sc_tbl):,}{_sc_of} · sorted by Rank{_sc_new} · click a row to load it into the "
                   f"Tear-Sheet tab")
        # The key follows the VISIBLE list: when any control changes it, the table is a new widget and
        # an old row pick cannot point at a different stock.
        _sc_ids = universe_signature(_sc_tbl["company_id"])
        _sc_pick = st.dataframe(
            _sc_tbl, column_config=scan_table_config(_sc_new_label, _sc_sel),
            column_order=list(scan_table_columns(_sc_sel)),
            hide_index=True, width="stretch", height=min(560, 38 + 35 * len(_sc_tbl)),
            on_select="rerun", selection_mode="single-row",
            key=f"scan_tbl_{_sc_sel['key']}_{_sc_ids}")
        _sc_rows = _sc_pick.selection.rows if _sc_pick and hasattr(_sc_pick, "selection") else []
        if _sc_rows:
            _sc_name = _sc_tbl.iloc[_sc_rows[0]]["name"]
            # This tab renders BEFORE the Tear-Sheet's stock selectbox, so the key is set directly (the
            # Deep Scanner pattern). The staged _pending_xray + st.rerun() of QGLP/Tsunami is for tabs
            # rendered AFTER that widget; here the staged key is consumed further down the script, so it
            # would rerun forever. Set only when the pick CHANGES, so a row left selected never overrides
            # a stock chosen later in the Tear-Sheet itself (tests/test_scans.py).
            _sc_sig = f"{_sc_sel['key']}|{_sc_ids}|{_sc_name}"
            if st.session_state.get("_scan_last_pick") != _sc_sig:
                st.session_state["_scan_last_pick"] = _sc_sig
                st.session_state["xray_stock"] = _sc_name
            st.markdown(f"""
            <div style="padding:9px 14px;margin-top:8px;background:rgba(228,179,65,0.07);
                 border:1px solid rgba(228,179,65,0.3);border-radius:8px;font-size:0.8rem;">
              🔬 <strong style="color:{COLORS['text_primary']};">{_html.escape(str(_sc_name))}</strong>
              set — <strong style="color:{COLORS['blue']};">click The Tear-Sheet tab</strong> for full analysis.
            </div>
            """, unsafe_allow_html=True)
    # "Left 🌱" (2026-10-07): the exit side. Only a scan with measured exit reasons shows it, on the same
    # baseline and the same rule as 🆕 (members only while the record is current). The 🎯 sidebar scope
    # narrows it; the liquidity floor and the lens row do not (a stock does not leave by trading less).
    if "exit_reason" in _sc_sel and _sc_state_sel == "measured":
        if _sc_members is None or not _sc_new_label:
            st.caption(f"Left {_sc_sel['emoji']}: shown once the weekly record has saved a month-end list "
                       f"to compare against.")
        else:
            _sc_left = scan_left(df, _sc_sel, _sc_members,
                                 keep_ids=set(filt['company_id'].astype(str)) if _sc_scoped else None)
            if _sc_left.empty:
                st.caption(f"None left {_sc_sel['emoji']} since {_sc_new_label}.")
            else:
                with st.expander(f"Left {_sc_sel['emoji']} since {_sc_new_label}: {len(_sc_left):,} — "
                                 f"{scan_left_summary(_sc_left)}", expanded=False):
                    st.markdown(scan_left_legend())
                    st.dataframe(
                        _sc_left, hide_index=True, width="stretch",
                        height=min(420, 38 + 35 * len(_sc_left)),
                        column_order=["name", "_why", "sector", "_sales_chg", "_profit_chg"],
                        column_config={
                            "name": st.column_config.TextColumn("Stock"),
                            "_why": st.column_config.TextColumn("Why it left"),
                            "sector": st.column_config.TextColumn("Sector"),
                            "_sales_chg": st.column_config.NumberColumn(
                                "Sales", format="%+.0f%%", help="Sales over the last year against the year before."),
                            "_profit_chg": st.column_config.NumberColumn(
                                "Profit", format="%+.0f%%", help="Profit over the last year against the year "
                                "before (blank when the year before was a loss)."),
                        })
    # A footnote, so it sits after the list: candidates are not scans yet and carry no record.
    with st.expander(f"🔬 Being measured ({len(SCAN_CANDIDATES)}) — not scans yet; December's review decides",
                     expanded=False):
        st.markdown("\n".join(f"- **{_n}** — {_r}" for _n, _r in SCAN_CANDIDATES))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: DEEP SCANNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Deep Scanner column widths, px (2026-10-05). EVERY column a view can show has one, so a view's width
# is arithmetic the tests can check instead of whatever Streamlit's auto-size picks — auto-sized text
# columns grew to their longest cell (Which Flags reached ~500px). Long text truncates; the full text
# is a click away. Kept OUTSIDE the tab block on purpose: the label-map scans there read dict keys
# as header labels, and a width table must not count as a label.
_DS_W = {
    "rank": 55, "name": 180,
    # 🏆 Core
    "verdict_direction": 90, "wealth_tier": 75, "sector": 125, "market_category": 85,
    "composite_score": 105, "data_coverage_pct": 75, "conviction_tier": 50, "gate_pass": 60,
    "moat_growth_quad": 125,
    # 📊 Quality — its parts read as numbers beside the Quality bar
    "quality_score": 105, "moat_score": 60, "growth_score": 65, "improvement_score": 95,
    "cash_score": 55, "roce": 60, "roce_expansion": 95,
    "roce_inflection": 90, "opm": 55, "opm_acceleration": 105,
    # 💰 Valuation
    "close_price": 80, "fair_value_qglp": 70, "valuation_score": 105, "expected_excess_return": 70,
    "pe": 60, "pb_ratio": 55, "peg": 60, "dividend_yield": 80, "fcf_yield": 70, "buy_zone_label": 120,
    "market_cap": 90,   # in no view: shown while the MCap sort is active
    # 🔬 Forensic
    "red_flag_count": 75, "red_flag_list": 190, "piotroski_fscore": 70, "forensic_multiplier": 90,
    "cfo_to_pat": 75, "accruals_ratio": 80, "debt_to_equity": 55, "promoter_holdings": 80,
    "pledged_percentage": 75,
    # 📈 Technical
    "dist_to_vstop": 60, "momentum_score": 105, "d49_momentum_quality": 110, "rsi_14d": 50,
    "breakout_score": 105, "d48_breakout_readiness": 90, "dist_ath": 65, "crs_52w": 60, "weinstein_stage": 150,
    # 👥 Ownership
    "governance_bonus": 90, "change_promoter_1y": 110, "fii_holdings": 55, "change_fii_lq": 80,
    "dii_holdings": 55, "change_dii_lq": 80, "smart_money_flow": 165,
    # shown only while its sort is active
    "result_when": 95,
}

with tabs[2]:

    # ── Column view presets ────────────────────────────────────────
    # 📌 Rebuilt 2026-10-05 after measuring them at a 1536px laptop width: every view LEADS with
    # Rank · Stock (configured below), so Rank — the Score order exactly — shows where a stock stands
    # in every view. Every column has a set width (_DS_W)
    # and every view fits the screen with the sidebar open. Taken out as measured repeats: Forensic
    # Score (an exact copy of the red-flag count, -1.00), Piotroski and CFO/PAT from Quality (both
    # live in Forensic), the 52-week-high distance (-0.95 with Breakout), VSTOP (identical to
    # Stop Δ > 0 on 100% of stocks) and E.Yield (P/E inverted). Moved, as the user approved to fit
    # the screen: Smart Money and Governance (a composite leg built from ownership signals) to
    # 👥 Ownership, Price to Valuation only; MCap (size, not value) shows while sorted by and stays
    # in the export; Tsunami (3 of 2,717 stocks) keeps its own Market Pulse tab.
    # tests/test_deep_scanner_views.py
    _DS_VIEWS = {
        "🏆 Core":      ["rank","name","verdict_direction","wealth_tier","sector","market_category","composite_score",
                         "data_coverage_pct","conviction_tier","gate_pass","moat_growth_quad"],
        "📊 Quality":   ["rank","name","quality_score","moat_score","growth_score","improvement_score","cash_score",
                         "roce","roce_expansion",
                         "roce_inflection","opm","opm_acceleration"],
        "💰 Valuation": ["rank","name","close_price","fair_value_qglp","valuation_score","expected_excess_return",
                         "pe","pb_ratio","peg","dividend_yield","fcf_yield","buy_zone_label"],
        "🔬 Forensic":  ["rank","name","red_flag_count","red_flag_list","piotroski_fscore",
                         "forensic_multiplier","cfo_to_pat","accruals_ratio","debt_to_equity",
                         "promoter_holdings","pledged_percentage"],
        # Technical view: each categorical VERDICT sits directly AFTER the number it interprets
        # (2026-08-30 surfacing — d49 reads momentum/RSI, d48 reads breakout_score's distance
        # inputs), so the table teaches itself: 82 and 🎯 IMMINENT land in the same glance.
        # Off ATH (2026-10-05, from ValuePickr's 52-week / all-time-high thread) follows the breakout
        # pair: a stock can make a 52-week high and still sit far below an older peak. It already
        # feeds the Breakout score (config BREAKOUT_SIGNALS["ath_distance"]); on its own it added
        # +0.04 to +0.05 beyond Breakout and Momentum in all four test windows, so its WEIGHT is
        # December's question (docs/december-2026-plan.md rule 15; tests/test_ath_and_new_highs.py).
        "📈 Technical": ["rank","name","dist_to_vstop","momentum_score","d49_momentum_quality",
                         "rsi_14d","breakout_score","d48_breakout_readiness","dist_ath",
                         "crs_52w","weinstein_stage"],
        # 👥 Ownership (2026-10-05): each stake sits directly BEFORE its change, so level and move read
        # in one glance. Promoters rarely move in a quarter (68% flat), so theirs is the 1-year change.
        # Governance leads it: the composite's governance leg is built from these ownership signals.
        "👥 Ownership": ["rank","name","governance_bonus","promoter_holdings","change_promoter_1y","pledged_percentage",
                         "fii_holdings","change_fii_lq","dii_holdings","change_dii_lq","smart_money_flow"],
    }
    _DS_SORTS = {
        "Score ↓":    ("composite_score", False),
        "Quality ↓":  ("quality_score",   False),
        "Momentum ↓": ("momentum_score",  False),
        "PEG ↑":      ("peg",             True),
        "MCap ↓":     ("market_cap",      False),
        # 🆕 Results ↑ (2026-08-30): freshness as an ORDERING, deliberately NOT a filter/tier —
        # measured live, "reported ≤7d" holds 47.5% of the universe right after earnings season
        # and ~5% mid-quarter (seasonal noise a gate can't survive; a sort always works).
        # Ascending puts NEGATIVE ages first = results DUE soon (scheduled, not yet declared —
        # the sheet's days_from_result flipped), then the freshest actual reporters. NaN last.
        "🆕 Results ↑": ("result_age_days", True),
        # Breakout ↓ (2026-09-19): ADMITTED ON FORWARD EVIDENCE, not taste. A full-column IC sweep
        # over the two available windows (2026-06-17→08-22, 08-22→09-19) ranked breakout_score the
        # most CONSISTENT predictor in the engine — +0.150 then +0.172 rank-IC, the best "weaker
        # window" figure of 400 numeric columns, holding through a regime change (W1 flat, W2 a
        # −2.0% median tape). composite_score by the same measure ranks 88th (+0.134 → +0.034).
        #
        # THE RANK CORRELATION SAYS REDUNDANT AND IS WRONG HERE, which is why it was checked the
        # other way. Spearman(breakout_score, momentum_score) is +0.88 — but a sort is consumed at
        # the HEAD, and the heads barely intersect: top-25 overlap 5/25 (20%), top-50 15/50 (30%),
        # against Score ↓ just 1/25 (4%). This is the ⚖️ Fair Value trap inverted — there Pearson
        # said "orthogonal" while the selected SETS overlapped 86%; here the correlation says
        # "duplicate" while the visible head is 80% different. Correlation of columns is not
        # overlap of the rows a user actually reads.
        #
        # A SORT, NOT A WEIGHT: two windows (one only 28 days) is nowhere near enough to move a
        # score, and the momentum columns are structurally favoured at this horizon because
        # fundamentals barely change inside a month. Ordering what is already on screen is the
        # proportionate use of that evidence. breakout_score is already a Technical-view column,
        # so this satisfies the sort-by-visible doctrine without adding an orphan.
        "🚀 Breakout ↓": ("breakout_score", False),
    }

    # ── Control bar ────────────────────────────────────────────────
    st.markdown(
        f'<div style="font-size:0.7rem;font-weight:700;color:{COLORS["text_muted"]};'
        f'text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">'
        f'🔍 Deep Scanner</div>',
        unsafe_allow_html=True,
    )
    _ds_c1, _ds_c2, _ds_c3 = st.columns([1.5, 5.5, 2])
    with _ds_c1:
        ds_search = st.text_input(
            "Search", placeholder="Search stock name…",
            key="ds_search", label_visibility="collapsed",
        )
    with _ds_c2:
        # A segmented control, not pills: the widget for 'pick one view' — pills read as tags.
        ds_view = st.segmented_control(
            "Column View", list(_DS_VIEWS.keys()),
            default="🏆 Core", key="ds_view",
        )
        if not ds_view:
            ds_view = "🏆 Core"
    with _ds_c3:
        ds_sort_label = st.selectbox(
            "Sort", list(_DS_SORTS.keys()),
            key="ds_sort", label_visibility="collapsed",
        )
    if ds_view == OWNERSHIP:
        st.caption(OWNERSHIP_NOTE)

    # ── Filter + sort ──────────────────────────────────────────────
    ds_df = filt.copy()
    if ds_search and ds_search.strip():
        ds_df = ds_df[ds_df["name"].str.contains(ds_search.strip(), case=False, na=False)]
    _sort_col, _sort_asc = _DS_SORTS[ds_sort_label]
    if _sort_col in ds_df.columns:
        ds_df = ds_df.sort_values(_sort_col, ascending=_sort_asc)

    # ── Stats strip ────────────────────────────────────────────────
    _ds_t1   = int((ds_df["conviction_tier"] == 1).sum()) if "conviction_tier" in ds_df.columns else 0
    _ds_tsun = int(ds_df["tsunami_signal"].sum()) if "tsunami_signal" in ds_df.columns else 0
    _ds_avg  = ds_df["composite_score"].mean() if "composite_score" in ds_df.columns and len(ds_df) else 0
    _ds_gate = int(ds_df["gate_pass"].sum()) if "gate_pass" in ds_df.columns else len(ds_df)
    st.markdown(f"""
    <div style="display:flex;gap:20px;padding:8px 2px 12px 2px;
         border-bottom:1px solid {COLORS['border']};margin-bottom:10px;flex-wrap:wrap;
         align-items:center;">
      <span style="font-size:0.82rem;font-weight:800;color:{COLORS['text_primary']};">
        {len(ds_df)} stocks
      </span>
      <span style="font-size:0.78rem;color:{COLORS['text_muted']};">
        Avg&nbsp;<strong style="color:{COLORS['blue']};font-size:0.86rem;">{_ds_avg:.0f}</strong>
      </span>
      <span style="font-size:0.78rem;color:{COLORS['green']};">
        ✅ {_ds_gate} gate&nbsp;passed
      </span>
      <span style="font-size:0.78rem;color:{COLORS['gold']};">
        🏆 {_ds_t1} Crown&nbsp;Jewels
      </span>
      <span style="font-size:0.78rem;color:{COLORS['purple']};">
        🌊 {_ds_tsun} Tsunami
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Column selection ───────────────────────────────────────────
    # Sort-by-visible doctrine (2026-08-30): the 🆕 Results sort orders by a column no view
    # carries, so when it is active, materialize the age as a READABLE text column —
    # "📅 due 4d" (scheduled, not yet declared) vs "8d ago" (reported) — an ordering the
    # table cannot explain is the same rank-jumble that got Discovery's sort pills removed.
    if ds_sort_label == "🆕 Results ↑" and "result_age_days" in ds_df.columns:
        _ra = pd.to_numeric(ds_df["result_age_days"], errors="coerce")
        _days = _ra.abs().round().astype("Int64").astype(str)
        ds_df = ds_df.assign(result_when=np.select(
            [_ra < 0, _ra == 0, _ra > 0],
            ["📅 due " + _days + "d", "today", _days + "d ago"],
            default=""))
    # …and made GENERAL 2026-10-05: whatever the table is sorted by is on screen. A view that lacks
    # the sorted column gets it right after Stock (Rank stands for Score, result_when for Results);
    # a view that carries it keeps its curated order. Measured before: in 5 of 6 views the rows were
    # ordered by a Score no column showed.
    _view_cols = ds_view_columns(_DS_VIEWS.get(ds_view, _DS_VIEWS["🏆 Core"]), _sort_col, ds_df.columns)
    _display_df = ds_df[_view_cols].reset_index(drop=True)

    # ── Column config ──────────────────────────────────────────────
    _CC: dict = {}
    # 📌 THE SPINE (2026-10-05): Rank · Stock lead every view. NOT pinned (2026-10-06): Streamlit draws
    # a pinned column faded, which made the stock name the dimmest text in the table, and every view
    # fits the screen, so the spine stays in view anyway (tests/test_no_faded_columns.py). Configured
    # ONLY here — a later label map would overwrite it (tests/test_deep_scanner_views.py).
    _DS_SPINE = {"rank": "Rank", "name": "Stock"}
    for _sp, _sl in _DS_SPINE.items():
        if _sp in _display_df.columns:
            _CC[_sp] = (st.column_config.NumberColumn(_sl, help=_SCANNER_HEADER_TIPS.get(_sp), format="%.0f",
                                                      width=_DS_W.get(_sp))
                        if _sp == "rank" else
                        st.column_config.TextColumn(_sl, help=_SCANNER_HEADER_TIPS.get(_sp),
                                                    width=_DS_W.get(_sp)))
    # Every column takes its width from _DS_W (above the tab block): Streamlit's defaults — ~150px a
    # score bar, text sized to its longest cell — pushed the views off a 1536px laptop screen.
    # Score BARS for the headline scores only — Quality's parts and Governance read as numbers
    # (_num_fmt), which is part of what lets every view fit a laptop screen (2026-10-05).
    for _sc, _sl in {
        "composite_score": "Score", "quality_score": "Quality", "momentum_score": "Momentum",
        "breakout_score": "Breakout", "valuation_score": "Valuation",
    }.items():
        if _sc in _display_df.columns:
            _CC[_sc] = st.column_config.ProgressColumn(
                _sl, help=_SCANNER_HEADER_TIPS.get(_sc), min_value=0, max_value=100, format="%.0f",
                width=_DS_W.get(_sc))
    for _bc, _lbl in {"gate_pass": "✅ Gate"}.items():
        if _bc in _display_df.columns:
            _CC[_bc] = st.column_config.CheckboxColumn(_lbl, help=_SCANNER_HEADER_TIPS.get(_bc), width=_DS_W.get(_bc))
    _num_fmt = {
        "conviction_tier": ("Tier",     "T%.0f"),
        "moat_score":      ("Moat",     "%.0f"),   # Quality's parts, as numbers beside its bar (2026-10-05)
        "growth_score":    ("Growth",   "%.0f"),
        "improvement_score": ("Improvement", "%.0f"),
        "cash_score":      ("Cash",     "%.0f"),
        "governance_bonus": ("Governance", "%.0f"),   # 👥 Ownership: a composite leg built from ownership signals
        "piotroski_fscore":("F-Score",  "%.0f/9"),
        "peg":             ("PEG",      "%.2f×"),
        "pe":              ("P/E",      "%.1f×"),
        "pb_ratio":        ("P/B",      "%.1f×"),
        "cfo_to_pat":      ("CFO/PAT",  "%.0f%%"),
        "opm":             ("OPM",      "%.1f%%"),
        "opm_acceleration": ("OPM Δ YoY-Q", "%+.1fpp"),
        "roce":            ("ROCE",     "%.1f%%"),
        "roce_expansion":  ("ROCE Δ 10Y", "%+.1fpp"),
        "roce_inflection": ("ROCE Δ 2Y",  "%+.1fpp"),
        "debt_to_equity":  ("D/E",      "%.2f"),
        "promoter_holdings":("Promoter","%.1f%%"),
        "pledged_percentage":("Pledged","%.1f%%"),
        "change_promoter_1y": ("Promoter Δ 1Y", "%+.2fpp"),   # 👥 Ownership: each stake beside its change
        "fii_holdings":    ("FII",      "%.1f%%"),
        "change_fii_lq":   ("FII Δ qtr", "%+.2fpp"),
        "dii_holdings":    ("DII",      "%.1f%%"),
        "change_dii_lq":   ("DII Δ qtr", "%+.2fpp"),
        "rsi_14d":         ("RSI",      "%.0f"),
        "dividend_yield":  ("Div Yield", "%.2f%%"),   # Valuation: vendor TTM yield beside E.Yield; 0.00% = does not pay (2026-09-19)
        "fcf_yield":       ("FCF Yld",  "%.1f%%"),
        "market_cap":      ("MCap ₹Cr", "%.0f"),
        "red_flag_count":  ("🚩 Flags","%.0f"),
        "accruals_ratio":  ("Accruals", "%.2f"),
        "crs_52w":         ("RS 52W",   "%.0f"),
        "expected_excess_return": ("Edge %", "%.1f%%"),
        "close_price":     ("Price ₹",  "%.2f"),     # Valuation+Technical: the number every ₹ column is measured against
        "fair_value_qglp": ("Fair ₹",   "%.0f"),     # Valuation: QGLP fair PE × EPS (blank = loss-maker, undefined)
        "dist_to_vstop":   ("Stop Δ",   "%.1f%%"),   # Technical: % above(+)/below(−) the Volatility Stop
        "dist_ath":        ("Off ATH",  "%.1f%%"),   # Technical: % below the all-time high (0 = at it)
        "data_coverage_pct":      ("Evidence",   "%.0f%%"),   # Core: score-confidence % (high score on thin data = trap)
        "forensic_multiplier":    ("Forensic ×", "%.2f"),     # Forensic: the penalty cutting composite (1.00 clean → 0.50 high-risk)
    }
    for _nc, (_nl, _nf) in _num_fmt.items():
        if _nc in _display_df.columns:
            _CC[_nc] = st.column_config.NumberColumn(_nl, help=_SCANNER_HEADER_TIPS.get(_nc), format=_nf,
                                                     width=_DS_W.get(_nc))
    # String decision-signal + identity columns get clean headers (else they show raw snake_case).
    for _tc, _tl in {
        "sector": "Sector", "market_category": "Market Cap",
        "verdict_direction": "Soundness", "wealth_tier": "Wealth", "weinstein_stage": "Trend",
        "moat_growth_quad": "Moat·Growth", "smart_money_flow": "Smart Money",
        "buy_zone_label": "Buy Zone",
        "red_flag_list": "Which Flags",
        "result_when": "🆕 Results",
        "d48_breakout_readiness": "Readiness",
        "d49_momentum_quality": "Mom. Quality",
    }.items():
        if _tc in _display_df.columns:
            _CC[_tc] = st.column_config.TextColumn(_tl, help=_SCANNER_HEADER_TIPS.get(_tc), width=_DS_W.get(_tc))
    # Safety net: a future _DS_VIEWS column with a tip but no typed config above still gets its
    # hover tooltip (raw header). NOTE: Streamlit issue #10841 — header tooltips don't render in
    # the dataframe's FULL-SCREEN mode; they work in the normal embedded view.
    for _col in _display_df.columns:
        if _col not in _CC and _SCANNER_HEADER_TIPS.get(_col):
            _CC[_col] = st.column_config.Column(help=_SCANNER_HEADER_TIPS[_col], width=_DS_W.get(_col))

    # ── Render table — or a smart, cause-specific empty-state ──────
    if filt.empty:
        # Sidebar filters narrowed everything out → the fix is Clear all filters.
        st.markdown(
            f'<div style="text-align:center;background:{COLORS["bg_secondary"]};'
            f'border:1px dashed {COLORS["border"]};border-radius:12px;padding:26px 18px;margin-top:6px;">'
            f'<div style="font-size:1.3rem;margin-bottom:6px;">🔍</div>'
            f'<div style="font-size:0.95rem;font-weight:800;color:{COLORS["text_primary"]};">'
            f'No stocks match your filters</div>'
            f'<div style="font-size:0.72rem;color:{COLORS["text_muted"]};margin-top:4px;">'
            f'Your sidebar filters narrowed all {len(df):,} stocks out — loosen them or clear everything.'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        _, _ec, _ = st.columns([3, 2, 3])
        with _ec:
            st.button("🧹 Clear all filters", key="ds_clear", width="stretch",
                      on_click=clear_all_filters)
    elif ds_df.empty:
        # Filters DO match stocks; the search box killed them → clear the search, not the filters.
        st.info(f"🔍 No stock matches “{ds_search}” among the {len(filt):,} filtered stocks — "
                f"clear the search box above to see them all.")
    else:
        _sel = st.dataframe(
            _display_df,
            column_config=_CC,
            width="stretch",
            height=580,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        # ── Tearsheet shortcut on row select ──────────────────────────
        _sel_rows = _sel.selection.rows if _sel and hasattr(_sel, "selection") else []
        if _sel_rows:
            _picked = ds_df.iloc[_sel_rows[0]]["name"]
            st.session_state["xray_stock"] = _picked
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;padding:10px 14px;
                 background:rgba(88,166,255,0.06);border:1px solid rgba(88,166,255,0.25);
                 border-radius:8px;margin-top:8px;">
              <span style="font-size:1rem;">🔬</span>
              <span style="font-size:0.8rem;color:{COLORS['text_secondary']};">
                <strong style="color:{COLORS['text_primary']};">{_picked}</strong>
                set as active stock —
                <strong style="color:{COLORS['blue']};">click The Tear-Sheet tab</strong> to view full analysis.
              </span>
            </div>
            """, unsafe_allow_html=True)

        # ── Export — the CURATED columns (the deduped union of every view preset, ~50 meaningful
        # cols) instead of the ~500 raw internal columns (rf_/cat_/vqs_/proxies). Rows are the
        # searched/sorted ds_df; the column set is auto-derived from _DS_VIEWS so it never drifts,
        # and it's ~10x smaller to serialize on every rerun. ──
        # …plus every SORT column (2026-10-05): MCap left the views as size, not value, and the file
        # should carry whatever the table can be ordered by.
        _export_cols = [c for c in dict.fromkeys([*(_c for _v in _DS_VIEWS.values() for _c in _v),
                                                  *(_s for _s, _ in _DS_SORTS.values())])
                        if c in ds_df.columns]
        _safe_mode = analysis_mode.replace(" ", "_").lower()
        # Encode via the shared _to_csv_bytes (UTF-8-with-BOM) — the SAME Excel-safe path the sidebar
        # full-dump uses — so the export's emoji decision-columns (moat_growth_quad ⭐💀, smart_money_flow
        # ⚪✅❌, weinstein_stage, buy_zone_label) render in Excel instead of mojibaking on a BOM-less file.
        from ui.ui_export import _to_csv_bytes
        st.download_button(
            f"📥 Export {len(ds_df)} stocks · {len(_export_cols)} columns — {analysis_mode}",
            data=_to_csv_bytes(ds_df[_export_cols]),
            file_name=f"scan_{_safe_mode}.csv",
            mime="text/csv",
            width="stretch",
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4: THE TEAR-SHEET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[3]:
    all_stock_names = df["name"].dropna().tolist()
    if not all_stock_names:
        st.info("No stocks available. Check your data source.")
    else:
        # ── Stock selector: search + dropdown ────────────────────────
        _ts_c1, _ts_c2 = st.columns([2, 5])
        with _ts_c1:
            search_ticker = st.text_input(
                "Search", placeholder="🔍  HDFC, Infosys, TATA…",
                key="search_ticker", label_visibility="collapsed",
            )
        _term  = (search_ticker or "").strip().upper()
        _names = [n for n in all_stock_names if _term in n.upper()] if _term else all_stock_names
        if not _names:
            _names = all_stock_names
        # Cross-tab handoff: tabs that render AFTER this one (Market Pulse) cannot assign the
        # xray_stock widget key directly — Streamlit raises StreamlitAPIException (set-after-
        # instantiation). They stage a transient _pending_xray + st.rerun() instead; consume it
        # HERE, before the selectbox below is instantiated, so its index reflects the jumped stock.
        if "_pending_xray" in st.session_state:
            st.session_state["xray_stock"] = st.session_state.pop("_pending_xray")
        _prev_sel = st.session_state.get("xray_stock")
        _ts_idx   = _names.index(_prev_sel) if _prev_sel in _names else 0
        with _ts_c2:
            selected = st.selectbox(
                "Stock", _names, index=_ts_idx, key="xray_stock",
                label_visibility="collapsed",
            )

        stock   = df[df["name"] == selected].iloc[0]
        _regime = df.attrs.get("detected_market_regime", "SIDEWAYS")

        # ── Null-safe getter — available to ALL inner tabs ─────────────────
        def _sg(k, d=0):
            v = stock.get(k, d)
            return d if (v is None or (isinstance(v, float) and np.isnan(v))) else v

        # Pre-compute verdict inputs once — reused across tabs
        _gate_ok  = stock.get("gate_pass", 0) == 1
        _sell_any = stock.get("sell_alert_any", 0) == 1
        _mr_risk  = stock.get("mean_reversion_risk", 0) == 1
        _tier_num = int(_sg("conviction_tier", 5))
        _tc       = TIER_COLORS.get(_tier_num, TIER_COLORS[5])
        _tier_cfg = next((t for t in CONVICTION_TIERS if t["tier"] == _tier_num), CONVICTION_TIERS[-1])
        _comp_sc  = float(_sg("composite_score", 0))

        # ── Verdict header: reads the pre-computed verdict_* columns (core/verdict_engine.py) ──
        # Hard overrides (gate fail / sell alert) take precedence; otherwise the engine's veto-aware
        # verdict drives the band. No verdict logic is computed here — single source of truth is the engine.
        _vdir  = str(_sg("verdict_direction", "FLAWED") or "FLAWED")
        # verdict_strength is deliberately NOT read here (removed 2026-08-27). It is a measured
        # 1:1 rename of conviction_tier, and the hero already carries the tier badge and the
        # score — so "🟡 MIXED · HIGH CONVICTION · Score 90/100" stated the same fact three ways
        # on one screen. The COLUMN still exists (snapshot-schema stability, orphan principle:
        # an unsurfaced column harms nobody); only the display is retired. Pinned by
        # tests/test_verdict_vocabulary.py.
        _vconf = str(stock.get("verdict_confidence", "") or "")
        _vnarr = str(stock.get("verdict_narrative", "") or "")
        _vrisk = str(stock.get("verdict_top_risk", "") or "")
        _vemoji = str(stock.get("verdict_emoji", "") or "")

        if not _gate_ok:
            _verdict, _verdict_clr, _verdict_bg = "SYSTEM REJECTED", COLORS["red"], "rgba(248,81,73,0.09)"
            _verdict_reason = f"Hard Gate Failure — {stock.get('failed_gates', 'Unknown')}"
        elif _sell_any:
            _verdict, _verdict_clr, _verdict_bg = "SELL ALERT", COLORS["red"], "rgba(248,81,73,0.07)"
            _verdict_reason = "One or more Baid sell triggers have fired — review Forensics tab."
        else:
            _dir_map = {
                "SOUND":  (COLORS["green"],      "rgba(63,185,80,0.08)"),
                "MIXED":  (COLORS["gold"],       "rgba(228,179,65,0.07)"),
                "FLAWED": (COLORS["text_muted"], "rgba(110,118,129,0.06)"),
            }
            _verdict_clr, _verdict_bg = _dir_map.get(_vdir, _dir_map["FLAWED"])
            _verdict = f"{_vemoji} {_vdir}".strip()
            _verdict_reason = _vnarr or f"Tier {_tier_num} · Score {_comp_sc:.0f}/100"

        # Score · Confidence subline (engine path only)
        _meta_bits = []
        if _gate_ok and not _sell_any:
            _meta_bits.append(f"Score {_comp_sc:.0f}/100")
            if _vconf:
                _meta_bits.append(f"🔍 {_vconf} data")
        _meta_line = " · ".join(_meta_bits)

        _pill_css = ("font-size:0.67rem;font-weight:700;padding:2px 10px;border-radius:12px;"
                     "white-space:nowrap;")
        _risk_pill = (
            f'<span style="{_pill_css}background:rgba(248,81,73,0.13);color:{COLORS["red"]};'
            f'border:1px solid rgba(248,81,73,0.4);">{_vrisk}</span>'
        ) if (_vrisk and _gate_ok and not _sell_any) else ""
        _mr_pill = (
            f'<span style="{_pill_css}background:rgba(228,179,65,0.15);color:{COLORS["gold"]};'
            f'border:1px solid rgba(228,179,65,0.4);">⚠️ Mean Reversion</span>'
        ) if _mr_risk else ""

        # ── WHAT-vs-WHEN reconciliation: the verdict is a FUNDAMENTAL call (own this business?);
        # Weinstein stage is the TECHNICAL trend (is the trend with you?). They're orthogonal and
        # can disagree — a BUY/WATCH on a stock below its falling 30-week MA (Stage 3/4) is a
        # watchlist candidate, not a buy-now. Surface that tension (display only — the verdict
        # engine is untouched; this never changes the direction). Fires only on real conflict.
        _wstage = str(stock.get("weinstein_stage", "") or "")
        _trend_conflict = (_gate_ok and not _sell_any and _vdir in ("SOUND", "MIXED")
                           and ("Stage 4" in _wstage or "Stage 3" in _wstage))
        _trend_pill = (
            f'<span style="{_pill_css}background:rgba(228,179,65,0.15);color:{COLORS["gold"]};'
            f'border:1px solid rgba(228,179,65,0.4);">⚠️ Against 30-wk trend</span>'
        ) if _trend_conflict else ""
        if _trend_conflict and "Stage 4" in _wstage:
            _trend_msg = ("📉 Strong business, weak trend — price is below a falling 30-week MA "
                          "(Stage 4). A watchlist candidate; wait for a Stage-2 base before buying.")
        elif _trend_conflict:  # Stage 3 Top
            _trend_msg = ("⚠️ Strong business, topping trend — price has slipped below its 30-week MA "
                          "(Stage 3). Don't chase; wait for the trend to reset.")
        else:
            _trend_msg = ""
        _trend_action = (
            f'<div style="font-size:0.72rem;color:{COLORS["gold"]};margin-top:6px;line-height:1.4;'
            f'background:rgba(228,179,65,0.08);border:1px solid rgba(228,179,65,0.25);'
            f'border-radius:7px;padding:6px 10px;">{_trend_msg}</div>'
        ) if _trend_conflict else ""

        # ── 💹 Wealth-tier pill: the third layer of the grammar, on the reading surface ──
        # A LABELED PILL, never a competing banner — the band is Soundness's home. Unlocked by
        # the vocabulary rename: "FLAWED · 💹 WATCH★" reads as two layers disagreeing openly
        # (the feature), where the old "AVOID · BUY★" read as a contradiction (the cf_triangle
        # defect). ⚠ rides with the tier per the wealth_warn contract — never blended.
        _wtier = str(stock.get("wealth_tier", "") or "")
        _wwarn = " ⚠" if int(stock.get("wealth_warn", 0) or 0) == 1 else ""
        _WT_PILL_CLR = {
            "BUY★":   ("rgba(63,185,80,0.13)",   COLORS["green"], "rgba(63,185,80,0.4)"),
            "BUY":    ("rgba(63,185,80,0.10)",   COLORS["green"], "rgba(63,185,80,0.3)"),
            "WATCH★": ("rgba(228,179,65,0.15)",  COLORS["gold"],  "rgba(228,179,65,0.4)"),
            "WATCH":  ("rgba(228,179,65,0.10)",  COLORS["gold"],  "rgba(228,179,65,0.3)"),
            "AVOID":  ("rgba(110,118,129,0.12)", COLORS["text_secondary"], "rgba(110,118,129,0.35)"),
        }
        _wealth_pill = ""
        if _wtier in _WT_PILL_CLR:          # N/A and missing render nothing — no pill over no data
            _bg, _fg, _bd = _WT_PILL_CLR[_wtier]
            _wealth_pill = (
                f'<span style="{_pill_css}background:{_bg};color:{_fg};'
                f'border:1px solid {_bd};">💹 {_wtier}{_wwarn}</span>'
            )

        st.markdown(f"""
        <div style="background:{_verdict_bg};border:1px solid {_verdict_clr}55;
             border-left:4px solid {_verdict_clr};border-radius:10px;
             padding:11px 16px;margin:6px 0 10px 0;">
          <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
            <span style="font-size:0.85rem;font-weight:900;color:{_verdict_clr};
                 letter-spacing:1.1px;white-space:nowrap;">{_verdict}</span>
            <span style="font-size:0.7rem;color:{COLORS['text_secondary']};
                 white-space:nowrap;">{_meta_line}</span>
            {_wealth_pill}{_risk_pill}{_mr_pill}{_trend_pill}
          </div>
          <div style="font-size:0.75rem;color:{COLORS['text_secondary']};margin-top:5px;">
            {_verdict_reason}</div>
          {_trend_action}
        </div>
        """, unsafe_allow_html=True)

        # ── Verdict scorecard: the 6-axis evidence grid (Layer 2, directly under the verdict) ──
        render_verdict_scorecard(stock)

        # Sell alerts panel — only rendered when active
        if _sell_any:
            render_sell_alerts_panel(stock)

        # ── Hero + score strip ────────────────────────────────────────────
        render_stock_hero(stock, regime=_regime)
        render_score_strip(stock)

        # ── Inner tabs ────────────────────────────────────────────────────
        _itabs = st.tabs([
            "📋 Overview",
            "🔬 Forensics",
            "🏛️ Frameworks",
            "📈 Matrix & WCS",
            "📊 All Data",
        ])

        # ══ Tab A: Overview ════════════════════════════════════════════════
        # Visual quality profile (radar) + signal badges → the deep financial breakdown.
        # The old 7-KPI buy-checklist was REMOVED (2026-06-14): every one of its metrics is shown,
        # with more depth, in the Business & Financial Analysis below, and its tier/score header
        # duplicated the verdict header above. The verdict + 6-axis scorecard are now the at-a-glance.
        with _itabs[0]:
            _ov1, _ov2 = st.columns([3, 2])

            with _ov1:
                fig = render_radar_chart(stock, f"{selected} — Quality Radar")
                st.plotly_chart(fig, width="stretch")

            with _ov2:
                # Quality facets — the radar's LEGEND (the polygon shows shape; these are the exact
                # scores). Cash + Margin are unique here — the orthogonal scorecard omits them.
                def _qfrow(lbl, key):
                    sc = _sg(key, None)
                    if sc is None:
                        clr, vs = COLORS["text_muted"], "—"
                    else:
                        sc = float(sc)
                        clr = (COLORS["green"] if sc >= 60 else
                               COLORS["gold"]  if sc >= 40 else COLORS["red"])
                        vs = f"{sc:.0f}"
                    return (
                        f'<div style="display:flex;justify-content:space-between;align-items:center;'
                        f'padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
                        f'<span style="font-size:0.72rem;color:{COLORS["text_secondary"]};">{lbl}</span>'
                        f'<span style="font-size:0.82rem;font-weight:800;color:{clr};">{vs}'
                        f'<span style="font-size:0.6rem;color:{COLORS["text_muted"]};">/100</span></span></div>'
                    )
                _facets = (
                    _qfrow("🛡️ Moat",   "moat_score")          +
                    _qfrow("📈 Growth", "growth_score")        +
                    _qfrow("🚀 Improvement", "improvement_score") +
                    _qfrow("💰 Cash",   "cash_score")          +
                    _qfrow("📊 Margin", "margin_score")        +
                    _qfrow("⚖️ Balance","balance_sheet_score")
                )
                st.markdown(
                    f'<div style="font-size:0.62rem;font-weight:800;color:{COLORS["text_muted"]};'
                    f'text-transform:uppercase;letter-spacing:0.8px;margin:2px 0 4px 0;">Quality Facets</div>'
                    f'{_facets}',
                    unsafe_allow_html=True,
                )

                # Signal badges
                pio_raw = stock.get("piotroski_fscore", None)
                pio_val = None
                if pio_raw is not None and not (isinstance(pio_raw, float) and np.isnan(pio_raw)):
                    try:
                        pio_val = int(float(pio_raw))
                    except Exception:
                        pio_val = None
                pio_str = f"{pio_val}/9" if pio_val is not None else "N/A"
                pio_clr = (COLORS["green"] if pio_val is not None and pio_val >= 7 else
                           COLORS["gold"]  if pio_val is not None and pio_val >= 5 else
                           COLORS["text_muted"] if pio_val is None else COLORS["red"])
                smart  = str(stock.get("smart_money_flow", "⚪ Neutral") or "⚪ Neutral")
                cf_tri = str(stock.get("cf_triangle", "") or "")
                quad   = str(stock.get("moat_growth_quad", "") or "")
                badge_items = [(f"F-Score {pio_str}", pio_clr), (smart, COLORS["purple"])]
                if cf_tri:
                    badge_items.append((cf_tri, COLORS["blue"]))
                if quad:
                    badge_items.append((quad, _tc["text"]))
                bdgs = "".join(
                    f'<span style="display:inline-block;padding:3px 9px;border-radius:6px;'
                    f'font-size:0.68rem;font-weight:700;margin:2px 3px 2px 0;'
                    f'background:{c}18;border:1px solid {c}40;color:{c};">{lbl}</span>'
                    for lbl, c in badge_items
                )
                st.markdown(
                    f'<div style="font-size:0.62rem;font-weight:800;color:{COLORS["text_muted"]};'
                    f'text-transform:uppercase;letter-spacing:0.8px;margin:13px 0 8px 0;">Signals</div>'
                    f'{bdgs}',
                    unsafe_allow_html=True,
                )

            # vs Sector Peers — contextualizes the at-a-glance quality (radar + facets) against
            # the stock's OWN sector before the absolute financials below: the value-trap guard
            # (a high absolute score that is bottom-quartile for its sector, or vice-versa).
            render_sector_peer_strip(stock)

            # Trajectory — the second derivative, and the one question the rest of Overview does
            # not ask: not where this business stands, but which way it is moving and whether the
            # move is speeding up. Sits after the peer strip (absolute -> relative -> directional)
            # and before the detailed financials it summarises.
            render_trajectory_card(stock)

            st.markdown(
                f"<div class='sec-head'>📊 Business & Financial Analysis</div>",
                unsafe_allow_html=True,
            )
            render_financial_insights(stock)

        # ══ Tab B: Forensics ═══════════════════════════════════════════════
        with _itabs[1]:
            # The Fraud Perimeter renders its own richer KPI row (Red Flags · Forensic Score ·
            # Score Multiplier · Piotroski · Mgmt Integrity); a separate strip here just duplicated
            # F-Score/Red Flags/Forensic. CF Triangle still shows in the Overview "Signals" strip.
            st.markdown(
                f"<div class='sec-head'>🔬 Forensic Fraud Perimeter ({FORENSIC_MAX_FLAGS}-Flag Cascade)</div>",
                unsafe_allow_html=True,
            )
            render_forensic_perimeter(stock)

            # The F-Score is already shown as a NUMBER in the perimeter KPI row; this is the
            # checklist behind it. All nine components are computed at 100% coverage and none of
            # them reached the screen until now. Ordered weakest-to-strongest evidence:
            # red flags -> Piotroski -> Fisher.
            st.markdown("<br>", unsafe_allow_html=True)
            render_piotroski_checklist(stock)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='sec-head'>🧠 Systematic Fisher Proxy — 7 Automated Checks</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='sec-cap'>Phil Fisher's 15 qualitative points translated into strict "
                f"quantitative proxies using pre-derived CSV columns. "
                f"100% automated — zero manual input.</div>",
                unsafe_allow_html=True,
            )
            render_fisher_module(stock)

            st.markdown("<br>", unsafe_allow_html=True)
            render_schilit_shield(stock)

        # ══ Tab C: Guru Frameworks ═════════════════════════════════════════
        with _itabs[2]:
            st.markdown(
                f"<div class='sec-head'>🏛️ Guru Framework Alignment — {len(_FW_META)} Frameworks</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='sec-cap'>Pre-computed framework badges from scoring engine. "
                f"Each represents a complete quantamental screen from a master investor's methodology.</div>",
                unsafe_allow_html=True,
            )
            render_guru_frameworks(stock)

            # Deep-dive guru radars — collapsed by default (Layer 3 evidence). The verdict header,
            # 6-axis scorecard and categorized frameworks above already SUMMARIZE these same
            # dimensions; expand a radar only to audit one methodology's detail. Nothing removed —
            # just decluttered (9 stacked radars → 9 collapsed expanders). Calls kept explicit so the
            # app-wiring contract tests (canslim→sepa→dorsey order) still hold.
            st.markdown(
                "<div class='sec-cap' style='margin-top:16px;'>🔬 Deep-dive radars — expand to audit "
                "a specific methodology (its signals are already summarized in the scorecard above).</div>",
                unsafe_allow_html=True,
            )
            with st.expander("👑 QGLP — Raamdeo's Process (Q·G·L·P)", expanded=False):
                render_qglp_radar(stock, scoring_profile)
            with st.expander("📊 CAN SLIM — Tactical Momentum (O'Neil)", expanded=False):
                render_canslim_radar(stock)
            with st.expander("⚡ Minervini SEPA — Momentum & VCP", expanded=False):
                render_sepa_radar(stock)
            with st.expander("🌊 Dorsey — Wide-Moat Pillars", expanded=False):
                render_dorsey_radar(stock)
            with st.expander("🎯 Outsider CEO — Capital Allocation", expanded=False):
                render_outsider_radar(stock)
            with st.expander("🛡️ Marks — Cycle Position", expanded=False):
                render_marks_radar(stock)
            with st.expander("📚 Malik — Quality Checklist", expanded=False):
                render_malik_radar(stock)
            with st.expander("👓 Lynch — Category & PEG", expanded=False):
                render_lynch_radar(stock)
            with st.expander("🔮 Mauboussin — Expectations & Payoff", expanded=False):
                render_mauboussin_radar(stock)
            with st.expander("🏛️ MOSL Wealth-Creation Matrix", expanded=False):
                render_mosl_wealth_matrix(stock)

        # ══ Tab D: Matrix & WCS ════════════════════════════════════════════
        with _itabs[3]:
            render_moat_growth_matrix(filt, highlight_stock=selected)
            st.markdown("<br>", unsafe_allow_html=True)
            render_ep_power_curve_module(stock)
            st.markdown("<br>", unsafe_allow_html=True)
            render_valuation_inversion_and_sizing_cockpit(stock)
            render_bruised_blue_chip_badge(stock)
            render_multitrillioncap_card(stock)

        # ══ Tab E: All Data ════════════════════════════════════════════════
        with _itabs[4]:
            st.markdown(
                f"<div class='sec-head'>📊 Raw Signal Data — Full Universe Output</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='sec-cap'>Every final, decision-grade signal the engine computes, grouped by "
                f"category — intermediate working columns are omitted here (the Export below carries the "
                f"complete machine-readable row). Engine-computed; nothing is re-calculated on this tab.</div>",
                unsafe_allow_html=True,
            )
            # The search box lives HERE, not in ui_tearsheet: that module is bound by the
            # stateless contract (app.py owns session_state). Same split as ds_search/ref_search.
            _ad_q = st.text_input(
                "Filter signals", value="", key="ad_search", label_visibility="collapsed",
                placeholder="🔎 Filter signals — by name (roce) or by meaning (cost of equity)…",
                help="Matches the signal's label AND its description, so you can search for what a "
                     "number MEANS, not only what it is called. Every word must appear. Blank = all.",
            )
            render_raw_signals(stock, query=_ad_q)
            # Breathing room before the Export so it doesn't crowd the last data section.
            st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
            # `stock` IS df[df["name"] == selected].iloc[0] (assigned once at the top of this tab) and
            # is the very row the grid above just rendered. Reuse it instead of re-running the same
            # lookup twice more: one derivation cannot drift from what the tab displayed, two can.
            _stock_export = pd.DataFrame({"Signal": stock.index, "Value": stock.values})
            # Excel-safe UTF-8-with-BOM encode (the SAME path as the Deep Scanner + sidebar exports) —
            # this row's Value column is full of emoji decision-strings (corporate_class 🏆, smart_money
            # ⚪/✅/❌, weinstein_stage, verdict emojis) + Indian names that mojibake under a bare to_csv.
            from ui.ui_export import _to_csv_bytes
            st.download_button(
                f"📥 Export {selected} — full row · all {df.shape[1]} signals",
                data=_to_csv_bytes(_stock_export),
                file_name=f"{re.sub(r'[^A-Za-z0-9._-]+', '_', selected).lower()}_signals.csv",
                mime="text/csv",
                width="stretch",
                # Both sibling data exports state their column count in the label; this one said only
                # "(all columns)". The help names the one thing measurement showed a user WILL hit:
                # the CSV is keyed by ENGINE column name, and NONE of the 154 display labels the grid
                # above just taught them appear in it (0 of 154 — verified on live data).
                help=f"The complete machine-readable row — all {df.shape[1]} engine columns, including "
                     f"the intermediate working columns the grid above omits. Rows are keyed by engine "
                     f"column name (roce_med_10y), NOT the display labels shown above (ROCE 10Y Med). "
                     f"Excel-safe UTF-8. The filter box does not narrow this export.",
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5: MARKET PULSE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FRAGMENT (2026-08-29): Market Pulse is market-wide BY DEFAULT — it reads the module-level `df`
# unless the 🎯 scope checkbox is ticked, which points every lens (never the vitals band) at
# `filt` instead. CORRECTED 2026-09-20: this comment used to say "verified: zero `filt` reads",
# which the scope control makes false — the read is deliberate, guarded by `0 < len(filt) <
# len(df)`, and pinned by tests/test_market_pulse_scope.py. The performance claim below is
# UNAFFECTED: `filt` is computed in the sidebar during the full run, so the fragment only reads
# an already-built frame and a scope toggle still costs one scoped rerun, not a full one.
# (mp_* keys are consumed nowhere else, and no variable assigned here is referenced after the
# block.) Its own controls (mp_sec_*, mp_ind_*,
# mp_wealth_tier) therefore need no full-script rerun: measured pre-fragment, one in-tab
# selectbox change cost 951 ms of which ~97% was re-rendering every OTHER tab. @st.fragment
# scopes those interactions to this tab. Sidebar changes still rerun the full app (fragments
# re-execute during full reruns), so nothing filter-related changes behavior.
@st.fragment
def _render_market_pulse():

    # ══ SCOPE (2026-09-20) — the ONE control that points every lens at the sidebar cohort ══════
    # Market Pulse has always been market-wide. This makes that a CHOICE instead of a property,
    # because "which sectors dominate the list I am actually considering" is a real question the
    # tab could not answer. Default OFF: breadth is what the tab is for, and an opt-in keeps
    # every existing caption and number byte-identical until asked.
    #
    # READ FROM SESSION_STATE BEFORE ANY WIDGET RENDERS — the cfg_mode/archive-id pattern.
    # Streamlit commits a widget's new value before the rerun, so the checkbox can sit BELOW the
    # pulse band (where a reader expects a scope control) while the frames above it already
    # reflect it. Layout never dictates data flow.
    #
    # THE GUARD IS `0 < len(filt) < len(df)`, both ends load-bearing: with no sidebar filter
    # there is nothing to limit to and the control would be furniture (the 🧹 Clear rule), and
    # with a filter that matched NOTHING "show only my filtered stocks" is a dead screen — the
    # sidebar already says so in its own words.
    _mp_narrowed = 0 < len(filt) < len(df)
    _mp_scoped   = _mp_narrowed and bool(st.session_state.get("mp_use_sidebar", False))
    _mp_df       = filt if _mp_scoped else df

    def _mp_scope_cap():
        """The ONE scope sentence every lens caption ends with — one definition, five surfaces.
        Four captions used to hardcode "Market-wide (ignores sidebar filters)"; with scoping on
        that sentence contradicts the checkbox directly above it, which is the caption-drift
        class this app has shipped twice. Derived, never retyped."""
        return (f"Your filters: {len(_mp_df):,} of {len(df):,} stocks."
                if _mp_scoped else "Market-wide (ignores sidebar filters).")

    def _mp_empty(market, scoped):
        """Pick the empty-state sentence that is TRUE of the cohort actually searched.

        Found in review 2026-09-20. "No stocks currently pass the strict QGLP gates" is a
        statement about the MARKET, and with the 🎯 scope on it is simply false — 413 stocks pass
        it; none of YOUR 876 do. Same for the Tsunami line. An empty state is the one place a
        reader has nothing else to go on, so it is the worst place to overstate. Both sentences
        are written out at each site rather than assembled, because a suffix bolted onto a
        market-wide claim still reads as a market-wide claim."""
        return scoped if _mp_scoped else market

    # THE TWO AGGREGATING TABS NEED ONE NUMBER THE OTHERS DO NOT. 📈 Sectors and 🏭 Industry do
    # not list stocks, they AVERAGE them, and an average over one stock is that stock wearing a
    # group's name. Measured on the live universe: filtering to BUY★ leaves 15 of 59 sectors
    # whose "average" IS a single company (Ceramic Products, 68.2, n=1) — and the 🏭 tab has no
    # min-stocks dial at all, where 32% of industries sit at <=2 stocks even UNFILTERED, rising
    # to 78% of survivors under BUY★. So the honesty number is printed rather than assumed:
    # how many groups are shown, and how many of them rest on too few stocks to be a group.
    # Only when scoped — an unfiltered tab is unchanged, down to the byte.
    _MP_THIN_N     = 2      # an average over <=2 stocks is a stock, not a group average
    _MP_MIN_GROUPS = 3      # fewer surviving groups than this is not a ranking, and says so

    def _mp_group_scope(n_shown, n_thin, unit, hint):
        """Render the scope line for an AGGREGATING tab. One definition, two surfaces.

        Two states, one line: normally a muted caption naming the cohort and the group count;
        below _MP_MIN_GROUPS an st.info that says it is not a ranking and names BOTH exits.
        The thin rows are still SHOWN — hiding them would answer a question the reader did not
        ask; the warning is what stops a 1-row table reading like a leaderboard."""
        if not _mp_scoped:
            return
        # ZERO IS NOT OURS TO REPORT. Both tabs already own an empty state, and each is MORE
        # specific than anything this helper could say: 📈 Sectors names the min-stocks dial, and
        # 🏭 Industry's drill-down names the sector that emptied it. Firing as well would stack
        # two info boxes on one condition — and in the drill-down case the second would blame the
        # SIDEBAR for a narrowing the sector picker caused, which is worse than redundant.
        if n_shown == 0:
            return
        # THE BAR IS USABLE GROUPS, NOT ROWS — found in the browser 2026-09-20. The first
        # version warned on `n_shown < 3`, so filtering to 3 stocks in 3 different industries
        # printed the ordinary caption over a table whose every "industry average" was one
        # company. Row count is not the question; a ranking needs groups that are actually
        # groups. For 📈 Sectors the min-stocks dial already removes the thin ones (n_thin is
        # 0 at any dial >2), so this reduces to the old rule there and changes nothing.
        _usable = n_shown - n_thin
        if _usable < _MP_MIN_GROUPS:
            _detail = (f"only **{n_shown}** {unit}" if not n_thin else
                       f"**{n_shown}** {unit}, **{n_thin}** of them resting on ≤{_MP_THIN_N} stocks")
            st.info(f"⚠️ Your filters leave **{len(_mp_df):,}** stocks and {_detail} — "
                    f"too few to rank. {hint}")
            return
        _t = (f" · <b>{n_thin}</b> of them rest on ≤{_MP_THIN_N} stocks"
              if n_thin else "")
        st.markdown(
            f"<div class='sec-cap'>🎯 Your filters: <b>{len(_mp_df):,}</b> of {len(df):,} stocks "
            f"· <b>{n_shown}</b> {unit}{_t}</div>", unsafe_allow_html=True)

    # 🚀 ONE TOOLTIP, TWO TABLES (2026-10-03) — the Improvement column on 📈 Sectors and 🏭 Industry.
    # The definition IS the glossary entry, the text the Deep Scanner's Improvement header and the
    # Reference tab already show, so a redesign of the facet reaches every surface at once. Only
    # how to read a GROUP is written here. Pinned in tests/test_market_pulse_improvement.py.
    _MP_IMPROVEMENT_HELP = (
        "The average Improvement Score of the row's stocks. "
        + _RAW_GLOSSARY["Improvement Score"]
        + " A group above 50 is improving more than the market's typical company. It describes "
          "the businesses, not the price and not every stock in the group: read it beside "
          "Momentum and against Count.")

    # 🔭 COLUMN VIEWS (2026-10-05) — 📈 Sectors and 🏭 Industry show one view at a time, the Deep
    # Scanner's own. What each measure is and how a group is summarised lives in ONE place,
    # ui/ui_group_views.py; these two only lay it out, once, for both tables.
    # Pinned in tests/test_market_pulse_views.py.
    # ONE bar width for every 0-100 bar in both tables. Streamlit's default (~150px a bar) pushed the
    # 7th column off a 1536px laptop screen; at 105px every view of both tables fits (browser-measured).
    _MP_BAR_W = 105
    _MP_VIEW_CFG = {
        k: (st.column_config.ProgressColumn(m["header"], min_value=0, max_value=100,
                                            format=m["fmt"], help=m["help"], width=_MP_BAR_W)
            if m["bar"] else
            st.column_config.NumberColumn(m["header"], format=m["fmt"], help=m["help"], width=m["width"]))
        for k, m in MEASURES.items()
    }

    def _mp_view_columns(view, core, lead, tail, stats, skip=()):
        """The columns a view shows: the tab's own Core list, or Count + the tab's ranking column,
        the view's measures, and any trailing context column. Switching views changes the COLUMNS,
        never the ranking — the ranking column is in every view so the order always reads. A
        cleared selection (None) falls back to Core. Only columns the table actually carries, minus
        any the table skips: a measure that earned its place on one table only."""
        cols = [*lead, *VIEW_MEASURES[view], *tail] if view in VIEW_MEASURES else core
        return [c for c in cols if c in stats.columns and c not in skip]

    # ── Pre-compute section datasets ───────────────────────────────
    _mp_ts   = (_mp_df[_mp_df["tsunami_signal"] == 1].sort_values("composite_score", ascending=False)
                if "tsunami_signal" in _mp_df.columns else _mp_df.iloc[:0])
    # QGLP decides WHO is listed; PRISM's rank decides the ORDER (2026-10-06). Among QGLP passers the
    # QGLP score did not predict returns in any of the test windows (rank-IC -0.14 to +0.01) while
    # PRISM's score did in every one (+0.03 to +0.12); sorted by QGLP score, the tab opened on stocks
    # ranked ~1,000-2,300 (tests/test_market_pulse_lens_order.py). The QGLP column still sorts on click.
    _mp_qglp = (_mp_df[_mp_df["qglp_pass"] == 1].sort_values("rank")
                if "qglp_pass" in _mp_df.columns else _mp_df.iloc[:0])

    # ── Market-state Pulse band (breadth-led market vitals — what the tab's name promises) ──────
    # NEVER SCOPED, and that is a hard rule rather than an oversight: breadth of a shortlist is
    # not breadth. "87% of my 20 filtered stocks are above their 200DMA" is a statement about the
    # shortlist, printed in the place a reader goes for the market. The vitals are always the
    # market; only the lenses below can be narrowed. Pinned.
    render_pulse_band(df)

    # The scope control itself — materialises ONLY when the sidebar is actually narrowing, so an
    # unfiltered session sees exactly the screen it saw before this existed. st.checkbox, not
    # st.toggle: ui_discovery already uses three checkboxes for on/off filters and none of the
    # latter, and a second widget vocabulary for the same job is the "second dialect" the lens
    # row's own docstring warns against. The COUNT lives in the label — scope a reader cannot
    # see is scope they will misread.
    if _mp_narrowed:
        st.checkbox(
            f"🎯 Show only my filtered stocks — {len(filt):,} of {len(df):,}",
            key="mp_use_sidebar",
            help="Points every lens below at the stocks your sidebar filters kept, instead of "
                 "the whole market. The vitals band above always stays market-wide — breadth of "
                 "a shortlist is not breadth. Each tab's own filter row still applies on top.",
        )

    # ── Shared lens filters (2026-08-30): Sector · Wealth Tier · Market Cap · Catalyst ─────
    # LOCAL to each lens tab (mp_* keys) — these slice the LENS's own cohort and are NOT the
    # sidebar cascade; they apply on top of whatever frame the tab was handed. CORRECTED
    # 2026-09-20: this used to add that the "Market-wide (ignores sidebar filters)" captions
    # "stay true", which the 🎯 scope checkbox makes conditional — the captions now derive
    # that sentence from _mp_scope_cap(). The fragment scopes every control to this tab, so
    # each rerun costs ~nothing.

    # ── Inner navigation tabs ──────────────────────────────────────
    _mp_tabs = st.tabs([
        "🌊 Tsunami",
        "🏛️ QGLP",
        "🔭 MOSL",
        "💹 Wealth",
        "📈 Sectors",
        "🏭 Industry",
        "🔁 Movers",
    ])   # Stage 3: dropped "💙 Blue Chips" + brittle "🚀 Tipping Points" (folded into Sectors).
         # Blue Chips' reason CORRECTED 2026-09-10: not dead (bruised_blue_chip fires 47/2717 = 1.7%) but
         # REDUNDANT — Framework 7, the sidebar's 💎 Bruised Blue Chip flag and Movers all reach it already.
         # 🔁 Movers APPENDED 2026-09-04 (index 6) — same rule: appended, never inserted.
         # 🏭 Industry APPENDED 2026-08-28 — appended, never inserted: each `with _mp_tabs[i]` body
         # binds by index, so inserting anywhere earlier renders existing content into a new tab.

    # ══ Tsunami ════════════════════════════════════════════════════
    with _mp_tabs[0]:
        st.markdown(
            f"<div class='sec-cap'>All 7 conviction conditions fire together: Quality + Momentum + "
            f"Governance + Technical. Rare by design.</div>",
            unsafe_allow_html=True,
        )
        if len(_mp_ts) == 0:
            st.info(_mp_empty(
                "🌊 No tsunami signals in current conditions — all 7 gates must fire simultaneously.",
                f"🌊 None of your {len(_mp_df):,} filtered stocks is a Tsunami — all 7 gates must fire "
                f"simultaneously. Untick 🎯 above to scan the whole market."))
        else:
            _ts_undi = int(_mp_ts["tsunami_undiscovered"].sum()) if "tsunami_undiscovered" in _mp_ts.columns else 0
            _ts_avg  = float(_mp_ts["composite_score"].mean())   if "composite_score"      in _mp_ts.columns else 0
            st.markdown(f"""
            <div style="display:flex;gap:20px;padding:8px 2px 12px 2px;
                 border-bottom:1px solid {COLORS['border']};margin-bottom:10px;flex-wrap:wrap;">
              <span style="font-size:0.82rem;font-weight:800;color:{COLORS['purple']};">
                🌊 {len(_mp_ts)} Tsunami signals
              </span>
              <span style="font-size:0.78rem;color:{COLORS['gold']};">
                🏆 {_ts_undi} undiscovered
              </span>
              <span style="font-size:0.78rem;color:{COLORS['text_muted']};">
                Avg score <strong style="color:{COLORS['green']}">{_ts_avg:.0f}</strong>
              </span>
            </div>
            """, unsafe_allow_html=True)

            # Same ordering fix as QGLP below: 12 columns, ~8 fit. The Tsunami claim is that all 7
            # conviction conditions fire at once, so the EVIDENCE for that (scores, F-score, entry
            # zone) leads and the wide context columns follow.
            _ts_cols = [c for c in ["rank","name","verdict_direction",
                                    "composite_score","quality_score","momentum_score",
                                    "piotroski_fscore","buy_zone_label",
                                    "sector","market_category","market_cap","smart_money_flow"]
                        if c in _mp_ts.columns]
            _ts_sel = st.dataframe(
                _mp_ts[_ts_cols].reset_index(drop=True),
                column_config={
                    "verdict_direction": st.column_config.TextColumn("Soundness", help="The engine's overall SOUND / MIXED / FLAWED gate — a Tsunami setup can still be MIXED/FLAWED on valuation or entry timing."),
                    "composite_score": st.column_config.ProgressColumn("Score",    min_value=0, max_value=100, format="%.0f"),
                    "quality_score":   st.column_config.ProgressColumn("Quality",  min_value=0, max_value=100, format="%.0f"),
                    "momentum_score":  st.column_config.ProgressColumn("Momentum", min_value=0, max_value=100, format="%.0f"),
                    "piotroski_fscore": st.column_config.NumberColumn("F-Score",   format="%.0f/9"),
                    "market_cap":      st.column_config.NumberColumn("MCap ₹Cr",   format="%.0f"),
                    "rank":            st.column_config.NumberColumn("Rank",        format="%.0f"),
                    # 2026-08-30: these five rendered as RAW snake_case headers ("name", "sector",
                    # "smart_money_flow"…) — the dataframe's column names leaking into shipped UI.
                    # Labels match the Deep Scanner's vocabulary exactly (one column, one name).
                    "name":            st.column_config.TextColumn("Stock", width="medium"),
                    "sector":          st.column_config.TextColumn("Sector"),
                    "market_category": st.column_config.TextColumn("Market Cap"),
                    "smart_money_flow": st.column_config.TextColumn("Smart Money"),
                    "buy_zone_label":  st.column_config.TextColumn("Buy Zone"),
                },
                width="stretch",
                height=min(480, 80 + len(_mp_ts) * 35 + 40),
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
            )
            _ts_rows = _ts_sel.selection.rows if _ts_sel and hasattr(_ts_sel, "selection") else []
            if _ts_rows:
                _ts_pick = _mp_ts.iloc[_ts_rows[0]]["name"]
                # Stage a transient key + rerun (NOT a direct widget-key set — this tab renders
                # after the Tear-Sheet selectbox). The change-guard is essential: st.dataframe's
                # selection persists across reruns, so an unguarded set+rerun would loop forever.
                if _ts_pick != st.session_state.get("xray_stock"):
                    st.session_state["_pending_xray"] = _ts_pick
                    st.rerun()
                st.markdown(f"""
                <div style="padding:9px 14px;margin-top:8px;background:rgba(139,92,246,0.07);
                     border:1px solid rgba(139,92,246,0.3);border-radius:8px;font-size:0.8rem;">
                  🔬 <strong style="color:{COLORS['text_primary']};">{_ts_pick}</strong>
                  set — <strong style="color:{COLORS['blue']};">click The Tear-Sheet tab</strong> for full analysis.
                </div>
                """, unsafe_allow_html=True)

    # ══ QGLP ═══════════════════════════════════════════════════════
    with _mp_tabs[1]:
        st.markdown(
            f"<div class='sec-cap'>Raamdeo Agrawal's framework: ROCE>15%, PAT growth>15%, "
            f"Promoter>50%, reasonable valuation. Strict gates. {_mp_scope_cap()}</div>",
            unsafe_allow_html=True,
        )
        if len(_mp_qglp) == 0:
            st.info(_mp_empty(
                "No stocks currently pass the strict QGLP gates.",
                f"None of your {len(_mp_df):,} filtered stocks passes the strict QGLP gates. "
                f"Untick 🎯 above to scan the whole market."))
        else:
            _q_total = len(_mp_qglp)
            _mp_qglp, _q_act = _mp_lens_row(_mp_qglp, "qglp")
            if _mp_qglp.empty:
                st.info("No QGLP passer matches these lens filters — 🧹 Clear resets them.")
            else:
                _q_avg = float(_mp_qglp["qglp_score"].mean()) if "qglp_score" in _mp_qglp.columns else 0
                st.markdown(f"""
                <div style="display:flex;gap:20px;padding:8px 2px 12px 2px;
                     border-bottom:1px solid {COLORS['border']};margin-bottom:10px;flex-wrap:wrap;">
                  <span style="font-size:0.82rem;font-weight:800;color:{COLORS['gold']};">
                    🏛️ {len(_mp_qglp)}{f' of {_q_total}' if _q_act else ''} QGLP compounders
                  </span>
                  <span style="font-size:0.78rem;color:{COLORS['text_muted']};">
                    Avg QGLP score <strong style="color:{COLORS['blue']}">{_q_avg:.0f}</strong>
                  </span>
                </div>
                """, unsafe_allow_html=True)

                # COLUMN ORDER MATTERS HERE, and it is the fix for a real defect (2026-08-27): 13
                # columns are defined but only ~8 fit the container, and `sector` — the widest column
                # in the frame ("Infrastructure Developers & Operators") — sat at position 5. It shoved
                # qglp_price, the "P" in QGLP, off-screen entirely. A tab showcasing a four-leg
                # framework was showing one and a half legs.
                # FITS A LAPTOP SCREEN since 2026-10-06 (it overflowed 1536px by ~400px): every column
                # has a set width summing to 1,030px, the four legs read as NUMBERS beside the QGLP bar
                # (the Deep Scanner's Quality-view rule), Buy Zone (is it a buy now?) precedes Sector,
                # and MCap and Smart Money left with the user's approval — both stay on the tear-sheet,
                # Smart Money also in the Deep Scanner's Ownership view (tests/test_market_pulse_columns.py).
                _q_cols = [c for c in ["rank","name","verdict_direction","red_flag_count",
                                       "qglp_score","qglp_quality","qglp_growth","qglp_longevity","qglp_price",
                                       "buy_zone_label","sector"]
                           if c in _mp_qglp.columns]
                _q_sel = st.dataframe(
                    _mp_qglp[_q_cols].reset_index(drop=True),
                    column_config={
                        "verdict_direction": st.column_config.TextColumn("Soundness", width=90, help="The engine's overall SOUND / MIXED / FLAWED gate — most QGLP passers are MIXED/FLAWED on valuation, so this surfaces the few that are buyable now."),
                        # Set widths, 1,030px in all (the grid is 1,074px at 1536px, less 44px for
                        # the row-selection column); text wider than its column is cut, as in the
                        # Deep Scanner.
                        "name":           st.column_config.TextColumn("Stock", width=180),
                        "sector":         st.column_config.TextColumn("Sector", width=105),
                        "buy_zone_label": st.column_config.TextColumn("Buy Zone", width=120),
                        "qglp_score":     st.column_config.ProgressColumn("QGLP",      min_value=0, max_value=100, format="%.0f", width=105),
                        "qglp_quality":   st.column_config.NumberColumn("Quality",   format="%.0f", width=65),
                        "qglp_growth":    st.column_config.NumberColumn("Growth",    format="%.0f", width=65),
                        "qglp_longevity": st.column_config.NumberColumn("Longevity", format="%.0f", width=85),
                        "qglp_price":     st.column_config.NumberColumn("Price/PEG", format="%.0f", width=85),
                        "red_flag_count": st.column_config.NumberColumn("🚩 Flags",    format="%.0f", width=75, help="Forensic red flags raised (0 = clean). QGLP gates on quality/growth, NOT forensics — so this is the risk check the screen itself doesn't do."),
                        "rank":           st.column_config.NumberColumn("Rank",         format="%.0f", width=55,
                                            help="PRISM's overall rank, and the table's order: QGLP decides "
                                                 "who is listed, PRISM's score decides the order. Among QGLP "
                                                 "passers the QGLP score did not predict returns in the test "
                                                 "windows; PRISM's score did. Click the QGLP header to sort by it."),
                    },
                    width="stretch",
                    height=min(500, 80 + len(_mp_qglp) * 35 + 40),
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                )
                _q_rows = _q_sel.selection.rows if _q_sel and hasattr(_q_sel, "selection") else []
                if _q_rows:
                    _q_pick = _mp_qglp.iloc[_q_rows[0]]["name"]
                    # Transient key + rerun + change-guard (see Tsunami above — same set-after-widget rule).
                    if _q_pick != st.session_state.get("xray_stock"):
                        st.session_state["_pending_xray"] = _q_pick
                        st.rerun()
                    st.markdown(f"""
                    <div style="padding:9px 14px;margin-top:8px;background:rgba(228,179,65,0.07);
                         border:1px solid rgba(228,179,65,0.3);border-radius:8px;font-size:0.8rem;">
                      🔬 <strong style="color:{COLORS['text_primary']};">{_q_pick}</strong>
                      set — <strong style="color:{COLORS['blue']};">click The Tear-Sheet tab</strong> for full analysis.
                    </div>
                    """, unsafe_allow_html=True)

    # ══ MOSL — convergence across the Wealth Creation Study family ═════
    with _mp_tabs[2]:
        # EXACT TOKENS, NEVER SUBSTRINGS. `frameworks_passed` joins names with ", ", and "QGLP" is
        # a SUBSTRING of "SQGLP Century Stock" — matching by substring inflated a first measurement
        # of this very cohort by 37 stocks. Splitting on the ", " boundary yields whole tokens only
        # (the same discipline ui_tearsheet._parse_frameworks uses). Cross-checked against the
        # authoritative qglp_pass column: both give 328.
        #
        # WHY A VIEW AND NOT NEW ENGINE COLUMNS: the fw_* booleans are LOCALS inside
        # scoring_engine.run_full_scoring and are never persisted, so frameworks_passed is the only
        # surviving record. Persisting them would be the cleaner data model but it is an engine
        # change, and this is a display feature.
        _MOSL_LENSES = ["QGLP", "Economic Moat", "Consistent in Volatile", "EP Hockey Stick",
                        "CAP-GAP Compounder", "SQGLP Century Stock", "100x Candidate",
                        "Blue Chip Quality", "MOSL Wealth Creator", "Bruised Blue Chip 29"]
        _tok = _mp_df.get("frameworks_passed", pd.Series("", index=_mp_df.index)).fillna("").astype(str).map(
            lambda _s: {t.strip() for t in re.split(r"\s*,\s*", _s) if t.strip()})
        # RAREST LENS FIRST (2026-10-06): the Lenses cell is cut off at the table's edge on ~30% of
        # rows, and in the list's own order the common lenses (Economic Moat, Consistent, QGLP) led
        # while the rare ones that say the most (Blue Chip Quality, MOSL Wealth Creator) fell off it.
        # Rarity is counted over the stocks being scanned; sorted() is stable, so a tie keeps the
        # list's order. The count column says how many (tests/test_market_pulse_lens_order.py).
        _mosl_order = sorted(_MOSL_LENSES, key=lambda m: int(_tok.map(lambda t: m in t).sum()))
        _mosl = _mp_df.copy()
        _mosl["mosl_n"] = _tok.map(lambda t: sum(1 for m in _MOSL_LENSES if m in t))
        _mosl["mosl_hits"] = _tok.map(lambda t: " · ".join(m for m in _mosl_order if m in t))
        # >=2 because ONE lens is not convergence -- the tab's whole claim is that independent
        # studies from the same house agree.
        _mosl = _mosl[_mosl["mosl_n"] >= 2].sort_values(
            ["mosl_n", "composite_score"], ascending=False)

        st.markdown(
            f"<div class='sec-cap'>How many of the <b>{len(_MOSL_LENSES)} Motilal Oswal Wealth "
            f"Creation lenses</b> (studies 16–30) a stock clears at once. Unlike a count across all "
            f"37 frameworks — where gate strictness varies by design and the numbers are not "
            f"comparable — these come from one research programme, so agreement between them means "
            f"something. Showing stocks that clear <b>2 or more</b>; one lens is not convergence."
            f"</div>",
            unsafe_allow_html=True,
        )
        # THE CAVEAT IS IN THE CAPTION, NOT A TOOLTIP. Measured 2026-08-27 with EXACT-TOKEN
        # matching: the 4-or-more cohort (175 stocks) carries a median of 6 red flags against the
        # universe's 5 — slightly WORSE, not better — and 80.6% of it is AVOID. These lenses gate
        # quality, growth and longevity; none of them reads the forensics.
        # (An earlier substring-matched pass reported 5 flags and 83.6%. It was wrong: "QGLP" is a
        # substring of "SQGLP Century Stock", which pulled 37 extra stocks into the cohort.)
        st.markdown(
            f"<div style='font-size:0.72rem;color:{COLORS['gold']};margin:-4px 0 10px 0;'>"
            f"⚠️ Convergence is <b>agreement, not safety</b>. These lenses test quality, growth and "
            f"longevity — none of them reads the forensics, so a high count carries no clean-books "
            f"claim. Check <b>Verdict</b> and <b>🚩 Flags</b> on every row. Unvalidated against "
            f"forward returns: read it as convergence, never as conviction.</div>",
            unsafe_allow_html=True,
        )

        _m_total = len(_mosl)
        _mosl, _m_act = _mp_lens_row(_mosl, "mosl")
        if _mosl.empty:
            st.info(_mp_empty(
                "No convergence stock matches — loosen the lens filters (🧹 Clear resets them)."
                if _m_act else "No stock clears 2 or more MOSL lenses in this universe.",
                f"No stock clears 2 or more MOSL lenses among your {len(_mp_df):,} filtered stocks"
                + (" with these lens filters (🧹 Clear resets them)" if _m_act else "")
                + ". Untick 🎯 above to scan the whole market."))
        else:
            _n4 = int((_mosl["mosl_n"] >= 4).sum())
            st.markdown(f"""
            <div style="display:flex;gap:20px;align-items:center;margin-bottom:6px;">
              <span style="font-size:1.05rem;font-weight:800;color:{COLORS['gold']};">
                🔭 {len(_mosl)}{f' of {_m_total}' if _m_act else ''} stocks clear 2+ lenses
              </span>
              <span style="font-size:0.8rem;color:{COLORS['text_secondary']};">
                {_n4} clear 4+ &nbsp;·&nbsp; deepest agreement: {int(_mosl['mosl_n'].max())} of {len(_MOSL_LENSES)}
              </span>
            </div>
            """, unsafe_allow_html=True)

            _m_cols = [c for c in ["rank", "name", "verdict_direction", "red_flag_count",
                                   "mosl_n", "composite_score", "mosl_hits"]
                       if c in _mosl.columns]
            st.dataframe(
                _mosl[_m_cols].reset_index(drop=True),
                column_config={
                    "rank":             st.column_config.NumberColumn("Rank", format="%.0f"),
                    "name":             st.column_config.TextColumn("Stock", width="medium"),
                    "verdict_direction": st.column_config.TextColumn("Soundness", help="The engine's overall SOUND / MIXED / FLAWED gate. Most high-convergence names are FLAWED — the MOSL lenses do not read forensics or entry timing, and the soundness gate does."),
                    "red_flag_count":   st.column_config.NumberColumn("🚩 Flags", format="%.0f", width="small", help="Forensic red flags. The MOSL lenses gate quality/growth/longevity and NOT forensics, so this is the risk check the convergence count itself does not do."),
                    "mosl_n":           st.column_config.ProgressColumn("MOSL", min_value=0, max_value=len(_MOSL_LENSES), format="%.0f", width="small", help="How many of the 10 Wealth Creation lenses this stock clears."),
                    "composite_score":  st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f", width="small"),
                    "mosl_hits":        st.column_config.TextColumn("Lenses cleared", width="large"),
                },
                width="stretch",
                height=min(500, 80 + len(_mosl) * 35 + 40),
                hide_index=True,
            )

    # ══ Wealth — the change-lens tiers (engine columns from verdict_engine) ═════
    with _mp_tabs[3]:
        # Everything here READS the wealth_* columns compute_verdict materialized — zero logic
        # lives in this tab, so the tier a snapshot captures is byte-identical to the tier shown.
        # Grammar, provenance and the four rules: core/verdict_engine.py + tests/test_wealth_tier.py.
        # _WT_ORDER hoisted above the tabs (shared with the lens-filter row).
        _wt_counts = _mp_df["wealth_tier"].value_counts() if "wealth_tier" in _mp_df.columns else {}
        st.markdown(
            f"<div class='sec-cap'>The <b>wealth-engine tier</b> — three clocks, nothing else: "
            f"<b>EP%</b> (economic profit ÷ reserves = ROE − cost of equity, so a ₹200 Cr and a "
            f"₹2,000 Cr business compare fairly) · <b>Vel%</b> (this year's change in that excess "
            f"return) · <b>tau</b> (the 5-year margin spine). BUY★ = earning, improving, confirmed; "
            f"WATCH★ = the confirmed turnaround (not earning yet, improving with a spine); AVOID = "
            f"nothing improving — the LEVEL may be fine. {_mp_scope_cap()}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:0.72rem;color:{COLORS['gold']};margin:-4px 0 10px 0;'>"
            f"⚠️ <b>Price-blind and forensics-blind by design.</b> BUY means the wealth engine is "
            f"buy-grade — not that the price is right (check Valuation) and not that the books are "
            f"clean (the ⚠ marker and 🚩 count carry that; they never alter the tier, so the "
            f"tension stays visible). A description to decide WITH, not a recommendation.</div>",
            unsafe_allow_html=True,
        )
        if "wealth_tier" not in df.columns:
            st.info("wealth_tier not present — re-run the pipeline.")
        else:
            st.markdown(
                "<div style='font-size:0.78rem;font-weight:700;margin-bottom:6px;'>"
                + " &nbsp;·&nbsp; ".join(
                    f"{t} <span style='color:{COLORS['text_secondary']};'>{int(_wt_counts.get(t, 0))}</span>"
                    for t in _WT_ORDER)
                + "</div>",
                unsafe_allow_html=True,
            )
            # MULTI-SELECT 2026-08-30 (user request): several tiers at once — "BUY★ and BUY" and
            # "WATCH★ and WATCH" are the two readings this tab is actually used for, and neither
            # was expressible one tier at a time. Same shared _mp_ms helper as every other Market
            # Pulse filter, so the counts-on-options grammar is identical. The label is VISIBLE
            # here (it was collapsed for the selectbox): an empty multiselect shows only a generic
            # placeholder, and a collapsed label takes its help tooltip down with it.
            _wt_n = {str(k): int(v) for k, v in _wt_counts.items()} if len(_wt_counts) else {}
            _wt_pick = _mp_ms(st.container(), "Tier", [t for t in _WT_ORDER if _wt_n.get(t, 0)],
                              "mp_wealth_tier",
                              "Filter to these tiers — pick several (BUY★ + BUY, or the two "
                              "WATCH grades together); a stock in ANY of them qualifies. The "
                              "table always sorts strongest tier first, then by Vel%.", _wt_n)
            _wl = _mp_df.copy()
            _wl["_wt_ord"] = _wl["wealth_tier"].map({t: i for i, t in enumerate(_WT_ORDER)})
            _wl["_warn_txt"] = np.where(_wl["wealth_warn"].fillna(0) == 1, "⚠", "")
            if _wt_pick:
                _wl = _wl[_wl["wealth_tier"].isin(_wt_pick)]
            _w_total = len(_wl)
            # Tier stays the tab's own control above — but it is declared to the lens row as an
            # extra key so the row's 🧹 Clear resets it along with Sector/Cap/Catalyst.
            _wl, _w_act = _mp_lens_row(_wl, "w", extra_keys=("mp_wealth_tier",), with_tier=False)
            if _wl.empty:
                st.info(_mp_empty(
                    "No stock in this tier matches the lens filters — 🧹 Clear resets them.",
                    f"No stock in this tier matches, among your {len(_mp_df):,} filtered stocks — "
                    f"🧹 Clear resets the lens row, or untick 🎯 above to scan the whole market."))
            _wl = _wl.sort_values(["_wt_ord", "wealth_vel_pct"], ascending=[True, False])
            _wt_cols = [c for c in ["wealth_tier", "name", "_warn_txt", "wealth_ep_pct",
                                    "wealth_vel_pct", "moat_tau", "moat_score", "growth_score",
                                    "red_flag_count", "verdict_direction", "reserves"]
                        if c in _wl.columns]
            st.dataframe(
                _wl[_wt_cols].reset_index(drop=True),
                column_config={
                    "wealth_tier":       st.column_config.TextColumn("Tier", width="small"),
                    "name":              st.column_config.TextColumn("Stock", width="medium"),
                    "_warn_txt":         st.column_config.TextColumn("⚠", width="small", help="Forensic caution: 8+ red flags or a Schilit checker fail. Never changes the tier — it rides beside it."),
                    "wealth_ep_pct":     st.column_config.NumberColumn("EP%",  format="%+.1f", width="small", help="Excess return: ROE − cost of equity, in percentage points. The universe median is NEGATIVE — being above zero already beats the median listed company."),
                    "wealth_vel_pct":    st.column_config.NumberColumn("Vel%", format="%+.1f", width="small", help="This year's CHANGE in the excess return, in points of the reserves base. Direction beats level — the 28th WCS's own finding."),
                    "moat_tau":          st.column_config.NumberColumn("Margin τ", format="%+.2f", width="small", help="5-year margin trend (rank correlation, −1…+1). ≥ +0.25 confirms; ≤ −0.25 caps the tier at WATCH."),
                    "moat_score":        st.column_config.NumberColumn("Moat", format="%.0f", width="small"),
                    "growth_score":      st.column_config.NumberColumn("Growth", format="%.0f", width="small"),
                    "red_flag_count":    st.column_config.NumberColumn("🚩 Flags", format="%.0f", width="small"),
                    "verdict_direction": st.column_config.TextColumn("Soundness", width="small", help="The soundness gate (level + forensics + valuation) beside the wealth tier (pure change). They answer DIFFERENT questions and disagreeing openly is the point: the engine's rare SOUND names include stocks this lens reads as decaying, and its FLAWED pile hides confirmed turnarounds."),
                    "reserves":          st.column_config.NumberColumn("Reserves ₹Cr", format="%.0f", width="small", help="Reserves — the equity base behind EP% and Vel% (the same base economic profit is computed on). A tiny base can make the percentages explode — check this before believing an extreme EP%."),
                },
                width="stretch",
                height=min(520, 80 + len(_wl) * 35 + 40),
                hide_index=True,
            )

    # ══ Sectors ════════════════════════════════════════════════════
    with _mp_tabs[4]:
        # The size floor is a CONTROL now, so this caption cannot hardcode it. It is rendered into
        # a placeholder AFTER the filter row has produced _min_n — the same set-after-the-widget
        # pattern the sidebar funnel uses. The first cut left "≥5 stocks" literal here and it went
        # stale the moment the dial moved to 15: a caption contradicting its own control, which is
        # the exact defect class this session has been removing.
        _sec_cap_ph = st.empty()
        # Cap-tier filter — Market Pulse is market-wide by design (ignores the sidebar filter), so this
        # slices the WHOLE-sector aggregation by size. selectbox (not pills): cleaner for 7 options +
        # always returns a value; format_func adds the tier emoji while the option value stays the exact
        # market_category string (zero-mapping filter). Guarded if the column is absent.
        # TWO KINDS OF CONTROL, and the caption says so because they look identical and are not:
        #   RE-AGGREGATING (market-cap, cyclicality) filter the STOCKS, so every average and the
        #     % Qualify are recomputed over the surviving subset.
        #   ROW filters (capital phase, minimum size) hide whole sectors and leave the numbers
        #     of the survivors untouched.
        # sector_capital_phase is CONSTANT within a sector today (measured: 0 of 81 vary), so
        # filtering stocks by it would be equivalent — but it is applied AFTER aggregation anyway,
        # so it stays correct if that ever stops being true rather than silently part-filtering a
        # sector and skewing its averages.
        _sec_src = _mp_df
        # MULTI-SELECT + CASCADE (2026-08-30 Phase 2). Each control takes several values (OR
        # within, AND across) and every option list + count is computed from the frame already
        # narrowed by the controls to its left. THE UNITS DIFFER BY FILTER KIND, which is why this
        # tab was held back a phase: the three RE-AGGREGATING filters (market-cap, wealth,
        # cyclicality) count STOCKS -- those are what gets re-averaged -- while the capital-phase
        # ROW filter counts SECTORS, because sectors are the rows it hides. Labelling a sector-row
        # filter with a stock count would put two units side by side, the "100% trap" class.
        _c1, _c2, _c3, _c4, _c5, _c6 = st.columns([2.1, 2.4, 1.8, 1.9, 1.5, 1.0])
        _scf = _mp_df                              # progressively narrowed cascade frame (STOCKS)

        # 1 -- Market-cap tier (re-aggregating; counts = stocks)
        from config import MCAP_TIERS
        if "market_category" in _scf.columns:
            _cap_n = _scf["market_category"].astype(str).value_counts().to_dict()
            _cap_opts = [t for t in MCAP_TIERS if _cap_n.get(t, 0) > 0]
            _cap = _mp_ms(_c1, "Market-cap tier", _cap_opts, "mp_sec_cap",
                          "Re-aggregates: keeps only stocks in these tiers, then recomputes every "
                          "sector average. Counts are STOCKS. Pick several - any of them qualifies.",
                          _cap_n)
            if _cap:
                _scf = _scf[_scf["market_category"].isin(_cap)]
        else:
            _cap = []

        # 2 -- Cyclicality tier (re-aggregating; counts = stocks). Canonical order, not alphabetical.
        _CYC_SEC = ["Defensive", "Sensitive / Structural-Growth", "Cyclical",
                    "Deep Cyclical / Commodity", "Financials", "Catch-all"]
        if "cyclicality_tier" in _scf.columns:
            _cyc_n = _scf["cyclicality_tier"].astype(str).value_counts().to_dict()
            _cyc_opts = [t for t in _CYC_SEC if _cyc_n.get(t, 0) > 0]
            _cyc_sec = _mp_ms(_c2, "Cyclicality tier", _cyc_opts, "mp_sec_cyc",
                              "Re-aggregates: tiers cross sector lines (42 of 81 sectors hold more "
                              "than one), so this is a STOCK filter - hiding whole sectors could not "
                              "answer 'how do sectors rank among Defensive stocks only'. Counts are "
                              "STOCKS and reflect the market-cap filter to the left.", _cyc_n)
            if _cyc_sec:
                _scf = _scf[_scf["cyclicality_tier"].isin(_cyc_sec)]
        else:
            _cyc_sec = []

        # 3 -- Wealth tier (re-aggregating; counts = stocks, after cap AND cyclicality)
        _WT_SEC = ["BUY★", "BUY", "WATCH★", "WATCH", "AVOID", "N/A"]
        if "wealth_tier" in _scf.columns:
            _wt_n = _scf["wealth_tier"].astype(str).value_counts().to_dict()
            _wt_opts = [t for t in _WT_SEC if _wt_n.get(t, 0) > 0]
            _wt_sec = _mp_ms(_c3, "Wealth tier", _wt_opts, "mp_sec_wealth",
                             "Re-aggregates over the survivors. 'BUY★ and BUY, grouped by sector' "
                             "shows where the improving-wealth names concentrate. Counts are STOCKS "
                             "and reflect every filter to the left.", _wt_n)
            if _wt_sec:
                _scf = _scf[_scf["wealth_tier"].isin(_wt_sec)]
        else:
            _wt_sec = []

        # 4 -- Capital phase (ROW filter; counts = SECTORS, the rows it hides)
        if "sector_capital_phase" in _scf.columns:
            _ph_pairs = _scf[["sector", "sector_capital_phase"]].dropna().drop_duplicates()
            _ph_n = _ph_pairs["sector_capital_phase"].value_counts().to_dict()
            _ph_opts = sorted(_ph_n)
            _phase = _mp_ms(_c4, "Capital phase", _ph_opts, "mp_sec_phase",
                            "Hides rows. The phase is a SECTOR attribute (constant within a sector), "
                            "so this shows or hides whole sectors and changes no average - the counts "
                            "are therefore SECTORS, not stocks.", _ph_n)
        else:
            _phase = []

        with _c5:
            # 1 ADDED 2026-08-30 (user request): shows EVERY sector, even single-stock ones.
            # index=1 keeps the DEFAULT at 5 - the untouched tab is unchanged (pinned). This stays a
            # SELECTBOX by design: it is a numeric THRESHOLD, not a set-membership filter.
            _min_n = st.selectbox(
                "Min stocks / sector", [1, 5, 10, 15, 20, 30], index=1, key="mp_sec_minn",
                help="Hides rows. A sector's % Qualify is only as trustworthy as its sample: at n=7 one "
                     "company moves it 14 points, at n=96 it moves it 1. Raise this to see only sectors "
                     "big enough for the percentage to mean something - or drop to 1 to see them all.",
            )

        with _c6:
            # Default-aware conditional Clear: each control resets to ITS OWN default - [] for the
            # four multiselects, 5 for the size dial. Fixed slot, button only when something differs.
            _SEC_DEFAULTS = {"mp_sec_cap": [], "mp_sec_wealth": [], "mp_sec_cyc": [],
                             "mp_sec_phase": [], "mp_sec_minn": 5}
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            if any(st.session_state.get(k, d) != d for k, d in _SEC_DEFAULTS.items()):
                st.button("🧹 Clear", key="mp_sec_clear", width="stretch",
                          on_click=_mp_clear_lens, args=(_SEC_DEFAULTS,))

        _sec_cap_ph.markdown(
            f"<div class='sec-cap'>Every sector with <strong>≥{_min_n} stocks</strong> — "
            f"Quality / Improvement / Momentum / Valuation / Score averaged across <strong>all</strong> its stocks "
            f"(sample-robust, not just the gate-passers). <strong>% Qualify</strong> = the share "
            f"clearing the hard gates (the sector's quality breadth). Ranked by % Qualify "
            f"(most-investable first). Capital-cycle phase is named below: 🔥 hot (over-investing — "
            f"caution) · ❄️ starved (under-invested — opportunity). A sector average can hide up to "
            f"<strong>50 points</strong> of industry dispersion — see 🏭 Industry for the split. "
            f"<strong>Strong qtr</strong> is the share of its companies whose latest quarter beat the same "
            f"quarter last year on revenue, profit and margin. "
            f"Switch the <strong>Column View</strong> to see the sectors from another side — raw ratios "
            f"as medians, yes/no measures as shares of the sector's stocks.</div>",
            unsafe_allow_html=True,
        )

        if _cap:
            _sec_src = _sec_src[_sec_src["market_category"].isin(_cap)]
        if _cyc_sec and "cyclicality_tier" in _sec_src.columns:
            _sec_src = _sec_src[_sec_src["cyclicality_tier"].isin(_cyc_sec)]
        # 💹 TIER-SHARE BASE — captured AFTER the cap and cyclicality filters, BEFORE the wealth
        # filter, so Defensive × BUY★ reads "BUY★ share among the sector's Defensive stocks". The share
        # column answers "how much of this group's FULL roster is tier X?"; computed after the
        # wealth filter it would read 100% everywhere the moment a tier is selected (the filter
        # keeps only that tier — the trap the design review caught). Denominator = the roster
        # under every OTHER filter. The tier shown follows the filter selection; All → BUY★,
        # the top of the forward-validated monotonic ladder.
        _sec_share_base = _sec_src
        # The tier-share column now follows the SET of selected tiers (share of the roster in ANY
        # of them); nothing selected keeps the BUY★ default - the top of the validated ladder.
        _sec_share_tiers = list(_wt_sec) if _wt_sec else ["BUY★"]
        _sec_share_tier = (_sec_share_tiers[0] if len(_sec_share_tiers) == 1
                           else ("+".join(_sec_share_tiers) if len(_sec_share_tiers) <= 3
                                 else f"{len(_sec_share_tiers)} tiers"))
        if _wt_sec:
            _sec_src = _sec_src[_sec_src["wealth_tier"].isin(_wt_sec)]
        if _cap or _wt_sec or _cyc_sec:
            _bits = " · ".join(b for b in [", ".join(_cap), ", ".join(_cyc_sec),
                                           ", ".join(_wt_sec)] if b)
            st.caption(f"📊 {len(_sec_src):,} stocks ({_bits}) across "
                       f"{_sec_src['sector'].nunique()} sectors — averages recomputed on this subset.")

        # WHOLE-sector aggregation over ALL stocks — bigger samples = robust averages (the fix for
        # comparing a 3-stock sector to a 50-stock one). % Qualify = gate-pass rate, the sample-size-
        # immune breadth signal. The >=5-stock floor reuses the engine's own sector_capital_phase guard
        # ("median unstable below 5"). No top-N cap — every reliable sector is shown, sorted by Score.
        # 🚀 avg_improvement (2026-10-03): are the group's businesses getting BETTER over the last
        # year? The strongest group-level signal measured — 3Y rank-IC +0.44 for sectors and +0.32
        # for industries, still +0.40 / +0.28 after the group's prior price run — but MIXED on
        # PRISM's two short windows (66d ~0, 28d +0.27 / +0.23), so it is a column, never the sort.
        # Averaged in THIS call so every filter above, and the 🎯 scope, reach it.
        _sec_stats = _sec_src.groupby("sector").agg(
            stocks=("name", "count"),
            pct_qualify=("gate_pass", lambda s: 100.0 * s.mean()),
            avg_quality=("quality_score",    "mean"),
            avg_improvement=("improvement_score", "mean"),
            avg_momentum=("momentum_score",  "mean"),
            avg_valuation=("valuation_score","mean"),
            avg_composite=("composite_score","mean"),
        )
        # 🔭 The view measures (ui/ui_group_views.py), from the SAME filtered stocks as every column above.
        _sec_stats = _sec_stats.join(group_measures(_sec_src, "sector"))
        # 👑 T1 REMOVED 2026-08-28 (user call, sparsity-backed): nonzero in 7 of 81 sectors (9%)
        # and 7 of 355 industries (2%) — the same 7 names live in Discovery's tier filter.
        # 💹 tier share — over _sec_share_base (the pre-wealth-filter roster; see above). Exact
        # equality, never a contains match: "BUY" is a substring of "BUY★" (the QGLP⊂SQGLP class).
        if "wealth_tier" in _sec_share_base.columns:
            _sec_stats["pct_tier"] = (
                _sec_share_base.groupby("sector")["wealth_tier"]
                .apply(lambda s: 100.0 * s.isin(_sec_share_tiers).mean())
                .reindex(_sec_stats.index)
            )
        # Sort by % Qualify (breadth), then Score — so the most-INVESTABLE sectors lead. Sorting by
        # Score alone would rank a 0%-qualify sector #1 (e.g. Financial Services scores high on
        # fundamentals but every stock fails a hard gate), which misleads at a glance.
        _sec_stats = (_sec_stats[_sec_stats["stocks"] >= _min_n]
                      .sort_values(["pct_qualify", "avg_composite"], ascending=False))
        # Phase applied AFTER aggregation — a row filter, so the survivors' averages are untouched.
        if _phase and "sector_capital_phase" in df.columns:
            _keep = set(df.loc[df["sector_capital_phase"].isin(_phase), "sector"].dropna().unique())
            _sec_stats = _sec_stats[_sec_stats.index.isin(_keep)]

        _mp_group_scope(len(_sec_stats),
                        int((_sec_stats["stocks"] <= _MP_THIN_N).sum()),
                        f"sectors with ≥{_min_n} stocks",
                        "Drop **Min stocks / sector** to 1 to see them all, or widen your sidebar filters.")
        if _sec_stats.empty:
            st.info(f"No sector clears these filters at ≥{_min_n} stocks — widen the selection or "
                    f"lower the minimum.")
        else:
            # Score (avg_composite) sat second-to-last and rendered as a bar plus a single
            # truncated digit. The three figures a reader scans first — how many, what share
            # qualifies, and how they score — now lead; the component averages follow.
            # 🏆 Core is the overview; Quality and Valuation moved to their own views (2026-10-05).
            # Improvement sits BESIDE Momentum: its tooltip tells the reader to read the pair together.
            # 🔥 Strong qtr (2026-10-05) — the share of the sector's companies with a strong latest
            # quarter. It sits BEFORE Improvement so Improvement stays beside Momentum. SECTORS ONLY:
            # it passed the pre-declared bar here (+0.15 to +0.31 beyond Momentum) and FAILED it for
            # industries (-0.02 / +0.03 in Jan-Mar) — December re-tests (tests/test_strong_quarter.py).
            _sec_order = [c for c in ["stocks", "pct_qualify", "avg_composite", "pct_tier",
                                      "grp_strong_quarter", "avg_improvement", "avg_momentum"]
                          if c in _sec_stats.columns]
            # 🔭 One view at a time — the Deep Scanner's own. A DISPLAY choice, not a filter: the 🧹
            # Clear above never resets it, and it never re-ranks (the sort is fixed above).
            _sec_view = st.segmented_control("Column View", list(VIEWS), default="🏆 Core", key="mp_sec_view")
            if _sec_view == OWNERSHIP:
                st.caption(OWNERSHIP_GROUP_NOTE)
            _sec_show = _mp_view_columns(_sec_view, _sec_order, ["stocks", "pct_qualify"], [], _sec_stats)
            st.dataframe(
                _sec_stats[_sec_show].reset_index(),
                column_config={
                    # reset_index() materializes the groupby key as a COLUMN, and a column with no
                    # config entry renders under its raw snake_case name -- this table showed
                    # "sector" on screen. The 2026-08-30 header-vocabulary pass missed it because
                    # its scan only inspects columns that HAVE a column_config entry, so a column
                    # with none was invisible to it. Found in the browser 2026-08-31.
                    "sector":        st.column_config.TextColumn("Sector", width=180),
                    "stocks":        st.column_config.NumberColumn("Count", format="%.0f"),
                    "pct_qualify":   st.column_config.ProgressColumn("% Qualify", min_value=0, max_value=100, format="%.0f%%", width=_MP_BAR_W,
                                       help="Share of the sector's stocks that clear all hard gates — its quality breadth. "
                                            "SCALE-FREE, not statistically robust: a percentage stops big sectors "
                                            "dominating, but small ones then reach extremes easily. Measured "
                                            "2026-08-27: 8 of the top 10 sectors hold fewer than 12 stocks (median 9 "
                                            "vs 19 universe-wide), and at n=7 a single stock moves this 14 points. "
                                            "Read it alongside Count."),
                    "pct_tier":      st.column_config.ProgressColumn(f"💹 {_sec_share_tier} %", min_value=0, max_value=100, format="%.0f%%", width=_MP_BAR_W,
                                       help=f"Share of the sector's FULL roster in the {_sec_share_tier} wealth tier. The tier "
                                            f"follows the Wealth-tier filter (All → BUY★, the top of the ladder); the "
                                            f"denominator deliberately IGNORES that filter — computed after it, this column "
                                            f"would read 100% everywhere. Unverifiable (N/A) stocks stay in the denominator "
                                            f"and dilute the share. Universe BUY★ base rate ≈ 12%. Price-blind and "
                                            f"forensics-blind, like the tier itself; read against Count."),
                    "avg_quality":   st.column_config.ProgressColumn("Quality",  min_value=0, max_value=100, format="%.0f", width=_MP_BAR_W),
                    "avg_improvement": st.column_config.ProgressColumn("Improvement", min_value=0, max_value=100, format="%.0f", width=_MP_BAR_W,
                                       help=_MP_IMPROVEMENT_HELP),
                    "avg_momentum":  st.column_config.ProgressColumn("Momentum", min_value=0, max_value=100, format="%.0f", width=_MP_BAR_W),
                    "avg_valuation": st.column_config.ProgressColumn("Valuation",min_value=0, max_value=100, format="%.0f", width=_MP_BAR_W),
                    "avg_composite": st.column_config.ProgressColumn("Score",    min_value=0, max_value=100, format="%.0f", width=_MP_BAR_W),
                    **_MP_VIEW_CFG,
                },
                width="stretch",
                height=min(700, 80 + len(_sec_stats) * 35),
                hide_index=True,
            )

        # Capital-cycle phase — NAMES the Hot/Starved sectors (the Pulse band only COUNTS them),
        # computed universe-wide; always shown, independent of the cap filter / floor above.
        if "sector_capital_phase" in df.columns:
            import html as _html
            _phase_by_sec = df.groupby("sector")["sector_capital_phase"].first().fillna("")
            _hot     = sorted(_phase_by_sec[_phase_by_sec.str.contains("Hot", na=False)].index)
            _starved = sorted(_phase_by_sec[_phase_by_sec.str.contains("Starved", na=False)].index)
            _join = lambda xs: " · ".join(_html.escape(str(s)) for s in xs) if xs else "—"
            st.markdown(
                f'<div style="font-size:0.72rem;line-height:1.7;margin-top:12px;'
                f'border-top:1px solid {COLORS["border"]};padding-top:10px;">'
                f'<span style="color:{COLORS["orange"]};font-weight:700;">🔥 Hot capital '
                f'({len(_hot)})</span>'
                f'<span style="color:{COLORS["text_muted"]};"> — over-investing, caution: </span>'
                f'<span style="color:{COLORS["text_secondary"]};">{_join(_hot)}</span><br>'
                f'<span style="color:{COLORS["blue"]};font-weight:700;">❄️ Capital-starved '
                f'({len(_starved)})</span>'
                f'<span style="color:{COLORS["text_muted"]};"> — under-invested, opportunity: </span>'
                f'<span style="color:{COLORS["text_secondary"]};">{_join(_starved)}</span></div>',
                unsafe_allow_html=True,
            )

    # ══ Industry ═══════════════════════════════════════════════════
    with _mp_tabs[5]:
        # WHY THIS IS NOT THE SECTORS TAB WITH A DIFFERENT GROUPBY KEY. `sector` has 81 values,
        # `industry` has 355 — and averaging up to the sector destroys real dispersion. Measured
        # 2026-08-28: the six sizeable industries inside Pharmaceuticals run 18.1 → 51.3 on average
        # composite, a 33-point spread the Sectors tab reports as one number (FMCG 22.9, Auto
        # Ancillaries 22.6). 20 of the 76 industries holding ≥8 stocks sit more than 5 points from
        # their parent sector's average. That gap is this tab's whole subject, so it is a COLUMN
        # (Δ vs Sector) and the table sorts by it rather than by % Qualify.
        #
        # SORTING BY THE DELTA DOES NOT FIX THE SMALL-SAMPLE PROBLEM — an earlier version of this
        # comment claimed it did, and measurement contradicted it. Δ is MORE small-sample sensitive
        # than % Qualify, not less: % Qualify is bounded 0–100 while Δ is unbounded, so a one-stock
        # industry sitting 38 points off its sector average tops the entire table. Measured with no
        # floor at all: 5 of the top 10 rows were single-stock industries, and the top 10 overlapped
        # a floored table by 1 of 10. There is deliberately NO floor here — see the block below for
        # the trade that was made and the mitigation (Count, first column).
        if "industry" not in df.columns:
            st.info("🏭 No `industry` column in the loaded frame — re-run the pipeline.")
        else:
            # NO SIZE FLOOR — every industry is shown, down to the ones holding a single stock.
            # This went dial (5..30) → fixed 8 → none, each step on the user's explicit call.
            #
            # THE COST IS REAL AND ACCEPTED, recorded here so nobody "fixes" it back. Sorting by Δ
            # does not neutralise small samples — Δ is MORE exposed to them than % Qualify, since
            # % Qualify is bounded 0–100 while Δ is not. Measured 2026-08-28 with no floor: 5 of the
            # top 10 rows are single-stock industries, the leader is "Auto Ancillaries - Seats"
            # (n=1, +32) — one company's score minus a sector average — and the unfloored top 10
            # shares 1 row with the floored one. The floor did not trim the tail; it decided who led.
            # The mitigation is Count, on every row, second from the left: a reader can see n=1 and
            # discount it. That is the trade the user chose, deliberately, twice.

            # Same two-kinds architecture as Sectors, minus one control:
            #   RE-AGGREGATING  Market-cap tier, Wealth tier — filter the STOCKS, every average and
            #                   the sector BASELINE are recomputed over the survivors.
            #   ROW FILTER      Min stocks/industry — hides rows, touches no number.
            # Capital phase is deliberately absent: sector_capital_phase is a SECTOR attribute with
            # no industry-level analogue, so carrying it over would attach a sector's phase to an
            # industry that only partly lives in it.
            # A PROJECTION: a column averaged below must be listed here or the tab dies with a
            # KeyError — improvement_score was missed on 2026-10-03 and the browser caught it.
            # …and every input the 🔭 view measures read (ui_group_views.SOURCE_COLUMNS), de-duplicated.
            _IND_KEEP = [c for c in dict.fromkeys([
                             "industry", "sector", "name", "composite_score",
                             "quality_score", "improvement_score", "momentum_score",
                             "valuation_score", "gate_pass", "conviction_tier",
                             "market_category", "wealth_tier", *SOURCE_COLUMNS])
                         if c in _mp_df.columns]
            _ind_src = _mp_df[_IND_KEEP].copy()
            _ind_src["industry"] = _ind_src["industry"].astype(str).str.strip()
            _ind_src = _ind_src[~_ind_src["industry"].isin(["", "nan", "None"])]

            def _ind_sector_pairs(frame):
                """industry x sector stock counts, ranked (count desc, then sector ASC).

                THE one source the three views below derive from, so they cannot drift apart.
                The explicit sort is the determinism mandate: an unsorted mode reorders between
                processes (PYTHONHASHSEED) and the displayed parent sector would flicker.
                """
                return (frame.groupby(["industry", "sector"]).size().rename("n")
                        .reset_index()
                        .sort_values(["industry", "n", "sector"], ascending=[True, False, True]))

            def _ind_dominant(frame):
                """industry -> its ONE dominant sector (+ that sector's stock count `n`).

                DISPLAY AND THE PEER BASELINE ONLY. Industry is NOT nested inside sector (136 of
                355 span more than one), so "the" sector of an industry is a modal choice. This
                stays single-valued on purpose: `_dom_sec` drives the leave-one-out Δ vs Sector
                arithmetic below, and "which sector's peers?" has no defined answer for a 50/50
                industry — one row must contribute to exactly one baseline. Where that choice is
                weak the label says so, with its share.
                """
                return _ind_sector_pairs(frame).drop_duplicates("industry").set_index("industry")

            def _ind_homes(frame):
                """industry -> EVERY sector TIED for its maximum. What the drill-down matches on.

                ADDED 2026-09-21 after the user picked Steel and "Mining/Minerals - Iron Ore" did
                not appear: its 6 stocks are exactly 3 Mining & Mineral products / 3 Steel, and
                the alphabetical tie-break filed it under Mining because "M" precedes "S". Half
                that industry is Steel and it was unreachable from Steel. 24 of 371 industries are
                exact ties, so 24 rows were filed by SPELLING — deterministic, but carrying no
                information about the business.

                THRESHOLD-FREE BY CONSTRUCTION. The rejected alternative was "any sector holding
                >= 40%", which is the invented-threshold anti-pattern §5 exists to stop. "Tied for
                the maximum" needs no number: a tie earns every home, a clear winner keeps exactly
                one, and "merely touches the sector" never qualifies (Trading is 62% Trading and
                4% Chemicals — it must not surface under Chemicals).
                """
                pair = _ind_sector_pairs(frame)
                return pair[pair["n"] == pair.groupby("industry")["n"].transform("max")]

            # MULTI-SELECT + CASCADE (2026-08-30 Phase 2) — same two-kinds/two-units grammar as
            # Sectors: the RE-AGGREGATING filters (market-cap, wealth) count STOCKS, the ROW filter
            # (sector drill-down) counts INDUSTRIES, because industries are the rows it hides.
            _i1, _i2, _i3, _i4 = st.columns([2.2, 2.0, 2.4, 1.0])
            _icf = _ind_src                            # cascade frame, narrowed left to right

            # 1 -- Market-cap tier (re-aggregating; counts = stocks)
            from config import MCAP_TIERS
            if "market_category" in _icf.columns:
                _ind_cap_n = _icf["market_category"].astype(str).value_counts().to_dict()
                _ind_cap_opts = [t for t in MCAP_TIERS if _ind_cap_n.get(t, 0) > 0]
                _ind_cap = _mp_ms(_i1, "Market-cap tier", _ind_cap_opts, "mp_ind_cap",
                                  "Re-aggregates: keeps only stocks in these tiers, then recomputes "
                                  "every industry average AND its sector baseline over the same "
                                  "survivors. Counts are STOCKS.", _ind_cap_n)
                if _ind_cap:
                    _icf = _icf[_icf["market_category"].isin(_ind_cap)]
            else:
                _ind_cap = []

            # 2 -- Wealth tier (re-aggregating; counts = stocks, after the cap narrowing)
            if "wealth_tier" in _icf.columns:
                _ind_wt_n = _icf["wealth_tier"].astype(str).value_counts().to_dict()
                _ind_wt_opts = [t for t in ["BUY★", "BUY", "WATCH★", "WATCH", "AVOID", "N/A"]
                                if _ind_wt_n.get(t, 0) > 0]
                _ind_wt = _mp_ms(_i2, "Wealth tier", _ind_wt_opts, "mp_ind_wealth",
                                 "Re-aggregates. 'BUY★ and BUY, grouped by industry' shows where the "
                                 "improving-wealth names actually concentrate — a far sharper answer "
                                 "than the same question asked of an 81-value sector. Counts are "
                                 "STOCKS and reflect the market-cap filter to the left.", _ind_wt_n)
                if _ind_wt:
                    _icf = _icf[_icf["wealth_tier"].isin(_ind_wt)]
            else:
                _ind_wt = []

            # 3 -- SECTOR DRILL-DOWN (2026-08-28) — the navigation the tab pair implies: spot a
            # sector on 📈 Sectors, open 🏭 Industry, see its internal dispersion. A ROW FILTER
            # applied AFTER aggregation (matches on the DOMINANT sector), so every number —
            # averages, Δ, 💹 share — is exactly what the unfiltered table shows. Sectors hold a
            # median of 3 industries (max 38), so this turns 355 rows into a focused split.
            # The counts are INDUSTRIES-BY-HOME, computed off `_icf` — the frame the table itself
            # will aggregate — so the number beside an option is exactly the number of rows it
            # leaves. IT MUST READ THE SAME MAP THE FILTER DOES (`_ind_homes`, not `_ind_dominant`):
            # if the counts kept the single-home map while the filter multi-homed, every tied
            # industry would make the option under-report by one and nothing would fail loudly —
            # the table would just show more rows than the label promised. Pinned behaviourally.
            # A tied industry counts under each sector it ties in, so the column sums above the
            # industry total; each INDIVIDUAL option still states exactly what it yields.
            _ind_sec_n = (_ind_homes(_icf).groupby("sector")["industry"].nunique().to_dict()
                          if not _icf.empty else {})
            _ind_sec_opts = sorted(_ind_sec_n)         # sorted() — determinism mandate
            _ind_sec = _mp_ms(_i3, "Sector (drill-down)", _ind_sec_opts, "mp_ind_sec",
                              "Hides rows: shows only industries whose LARGEST SHARE of stocks "
                              "sits in these sectors. An industry that TIES for its top sector is "
                              "reachable from every sector it ties in — a 50/50 industry has no "
                              "single home. Applied after aggregation — no average, Δ or 💹 share "
                              "changes. Counts are INDUSTRIES, not stocks; a '~' row names the "
                              "share actually sitting in the sector beside it.", _ind_sec_n)

            with _i4:
                # Default-aware conditional Clear (same grammar as Sectors + the lens rows).
                _IND_DEFAULTS = {"mp_ind_cap": [], "mp_ind_wealth": [], "mp_ind_sec": []}
                st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                if any(st.session_state.get(k, d) != d for k, d in _IND_DEFAULTS.items()):
                    st.button("🧹 Clear", key="mp_ind_clear", width="stretch",
                              on_click=_mp_clear_lens, args=(_IND_DEFAULTS,))

            if _ind_cap:
                _ind_src = _ind_src[_ind_src["market_category"].isin(_ind_cap)]
            # 💹 TIER-SHARE BASE — same design as the Sectors tab: captured AFTER the cap filter,
            # BEFORE the wealth filter, so the share column keeps the FULL roster as denominator
            # (computed after the filter it would read 100% everywhere). All → BUY★.
            _ind_share_base = _ind_src
            _ind_share_tiers = list(_ind_wt) if _ind_wt else ["BUY★"]
            _ind_share_tier = (_ind_share_tiers[0] if len(_ind_share_tiers) == 1
                               else ("+".join(_ind_share_tiers) if len(_ind_share_tiers) <= 3
                                     else f"{len(_ind_share_tiers)} tiers"))
            if _ind_wt:
                _ind_src = _ind_src[_ind_src["wealth_tier"].isin(_ind_wt)]

            st.markdown(
                f"<div class='sec-cap'>All <strong>{_ind_src['industry'].nunique()} industries</strong>, "
                f"ranked by <strong>Δ vs Sector</strong> — how far its average score sits above or "
                f"below its sector <strong>peers</strong> (the industry's own stocks are excluded "
                f"from the baseline, so a sector-dominating industry cannot damp its own gap). That "
                f"is the one thing the Sectors tab cannot show: inside Pharmaceuticals alone, "
                f"industry averages run from 18 to 51. Positive = this industry outscores the rest "
                f"of its sector; negative = the rest of the sector is carrying it.</div>",
                unsafe_allow_html=True,
            )

            # DOMINANT SECTOR, not parent — industry is NOT nested inside sector. 136 of the 355
            # industries span more than one (dominant-sector share across ALL 355: median 1.00,
            # minimum 0.33 — the 136 multi-sector ones are the impure tail). The modal sector is
            # picked with an explicit (count desc, sector asc) tie-break: an unsorted mode is
            # non-deterministic across processes (PYTHONHASHSEED), which would make the displayed
            # parent sector flicker between runs.
            _ind_stats = _ind_src.groupby("industry").agg(
                stocks=("name", "count"),
                pct_qualify=("gate_pass", lambda s: 100.0 * s.mean()),
                avg_composite=("composite_score", "mean"),
                avg_quality=("quality_score", "mean"),
                avg_improvement=("improvement_score", "mean"),   # 🚀 see the Sectors note
                avg_momentum=("momentum_score", "mean"),
                avg_valuation=("valuation_score", "mean"),
            )
            # 🔭 The view measures, from the SAME filtered stocks (see the Sectors note).
            _ind_stats = _ind_stats.join(group_measures(_ind_src, "industry"))
            # 💹 tier share — over _ind_share_base (pre-wealth-filter roster; see above). Exact
            # equality, never contains: "BUY" ⊂ "BUY★".
            if "wealth_tier" in _ind_share_base.columns and not _ind_stats.empty:
                _ind_stats["pct_tier"] = (
                    _ind_share_base.groupby("industry")["wealth_tier"]
                    .apply(lambda s: 100.0 * s.isin(_ind_share_tiers).mean())
                    .reindex(_ind_stats.index)
                )
            if _ind_stats.empty:
                st.info("No stocks match these filters — widen the selection.")
            else:
                _ind_dom  = _ind_dominant(_ind_src)
                _dom_sec  = _ind_dom["sector"].reindex(_ind_stats.index)
                # pd.Series (not a bare array): the drill-down below row-filters _ind_stats, and
                # positional alignment would silently pair shares with the wrong industries.
                _dom_share = pd.Series(
                    np.where(_ind_stats["stocks"] > 0,
                             _ind_dom["n"].reindex(_ind_stats.index) / _ind_stats["stocks"],
                             np.nan),
                    index=_ind_stats.index)

                # BOTH TERMS OF THE DIFFERENCE COME FROM ONE POPULATION. The baseline is grouped off
                # `_ind_src` — the FILTERED frame — not off `df`. Comparing a Small-Cap-only industry
                # average against an all-cap sector average would report the cap effect as though it
                # were an industry effect: the cross-year-basis defect in different clothes.
                #
                # LEAVE-ONE-OUT BASELINE (2026-08-28; was the plain sector mean). Including the
                # industry's own stocks damps the delta by exactly (1 − its share of the sector) —
                # measured: QSR displayed −7.0 against a true peer gap of −48.1 (it IS 83% of its
                # sector, so the baseline was mostly itself), Paints +4.2 vs +33.5, max distortion
                # 41 points — and the damping factor appears nowhere on screen, so a reader could
                # neither see nor undo it. The baseline now excludes the industry's own in-sector
                # stocks: Δ is the gap to its sector PEERS. The aggregate sort barely moves (rank
                # corr 0.971 vs the old math); the muted tail rows were the point.
                #
                # THE DEGENERATE CASE NOW FALLS OUT OF THE MATH. A sector holding only this
                # industry leaves zero peers, the count guard emits np.nan, and the row renders
                # blank — never a 0.0 sentinel that reads "perfectly average" when the truth is
                # "nothing to compare against". The old explicit ≥2-industries test is gone because
                # it became REDUNDANT, not because the rule changed (live: 8 of 355 industries,
                # counted dynamically in the footer).
                _sec_agg = _ind_src.groupby("sector")["composite_score"].agg(["sum", "count"])
                _ind_dom_rows = _ind_src[_ind_src["sector"].values ==
                                         _ind_src["industry"].map(_dom_sec).values]
                _ind_own = (_ind_dom_rows.groupby("industry")["composite_score"]
                            .agg(["sum", "count"]).reindex(_ind_stats.index))
                _peer_sum = _dom_sec.map(_sec_agg["sum"])   - _ind_own["sum"].fillna(0.0)
                _peer_cnt = _dom_sec.map(_sec_agg["count"]) - _ind_own["count"].fillna(0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    _ind_stats["delta_vs_sector"] = np.where(
                        _peer_cnt > 0,
                        _ind_stats["avg_composite"] - _peer_sum / _peer_cnt, np.nan)
                # "~" flags an industry whose stocks are NOT mostly in the sector named beside it,
                # AND NAMES THE SHARE (2026-09-21): the bare tilde said "loosely" without saying how
                # loosely, so a 50% row and a 79% row read identically — and a row the drill-down
                # reached from a TIED sector had nothing on screen explaining why it sits under a
                # different name. np.where evaluates both branches, so the concat is NaN-safe by
                # construction: a NaN share fails `< 0.8` and takes the plain branch anyway.
                _ind_dom_pct = _dom_share.mul(100).round(0).fillna(0).astype(int).astype(str)
                _ind_stats["dom_sector"] = np.where(_dom_share < 0.8,
                                                    "~ " + _dom_sec.astype(str)
                                                    + " · " + _ind_dom_pct + "%",
                                                    _dom_sec.astype(str))

                if _ind_cap or _ind_wt:
                    _ind_bits = " · ".join(b for b in [", ".join(_ind_cap),
                                                       ", ".join(_ind_wt)] if b)
                    st.caption(f"🏭 {len(_ind_src):,} stocks ({_ind_bits}) across "
                               f"{_ind_src['industry'].nunique()} industries — averages and the "
                               f"sector baseline both recomputed on this subset.")

                # Incomparable rows carry no signal, so they sink rather than heading a descending
                # sort; avg_composite breaks ties and orders that trailing group sensibly.
                _ind_stats = _ind_stats.sort_values(["delta_vs_sector", "avg_composite"],
                                                    ascending=[False, False], na_position="last")

                # SECTOR DRILL-DOWN — row filter, applied AFTER every number is computed, so no
                # average, Δ or 💹 share moves (pinned). Matches on HOMES (every sector tied for the
                # industry's maximum), not on the single display sector: a 3-3 industry belongs to
                # both and must be reachable from both.
                if _ind_sec:
                    _ind_home_map = _ind_homes(_ind_src)
                    _ind_reach = set(_ind_home_map.loc[_ind_home_map["sector"].isin(_ind_sec),
                                                       "industry"])
                    _ind_stats = _ind_stats[_ind_stats.index.isin(_ind_reach)]
                    if _ind_stats.empty:
                        st.info(f"No industry is at home in {', '.join(_ind_sec)} under these "
                                f"filters — their stocks sit inside industries whose largest "
                                f"share lives elsewhere (the ~ rows of those sectors).")

                # Placed AFTER the sector drill-down, which row-filters _ind_stats above: a count
                # taken before it would overstate what the table actually shows.
                _mp_group_scope(len(_ind_stats),
                                int((_ind_stats["stocks"] <= _MP_THIN_N).sum()),
                                "industries",
                                "🏭 Industry has no minimum-stocks dial — widen your sidebar filters "
                                "or read 📈 Sectors instead, which does.")

                # Signal before context — the same invariant tests/test_market_pulse_columns.py pins
                # for the other tables. Sector names are the widest strings in the frame
                # ("Infrastructure Developers & Operators"), so the sector goes last.
                _ind_order = [c for c in ["stocks", "pct_qualify", "avg_composite",
                                          "delta_vs_sector", "pct_tier",
                                          "avg_improvement", "avg_momentum",
                                          "dom_sector"]
                              if c in _ind_stats.columns]
                # 🔭 The same views as 📈 Sectors. Every view keeps Count and Δ vs Sector — the
                # ranking — up front, and the sector last.
                _ind_view = st.segmented_control("Column View", list(VIEWS), default="🏆 Core", key="mp_ind_view")
                if _ind_view == OWNERSHIP:
                    st.caption(OWNERSHIP_GROUP_NOTE)
                # 📈 Near 52w high is SECTORS ONLY (2026-10-05): industry breadth failed the pre-declared
                # bar in one window (-0.01 beyond Momentum, 06-17 start) — December re-tests it
                # (docs/december-2026-plan.md rule 16; tests/test_market_pulse_views.py).
                _ind_show = _mp_view_columns(_ind_view, _ind_order, ["stocks", "delta_vs_sector"],
                                             ["dom_sector"], _ind_stats, skip=("grp_near_high",))
                st.dataframe(
                    _ind_stats[_ind_show].reset_index(),
                    column_config={
                        "industry":       st.column_config.TextColumn("Industry", width=180),
                        "stocks":         st.column_config.NumberColumn("Count", format="%.0f",
                                            help="Read every percentage on this row against this number first."),
                        "pct_qualify":    st.column_config.ProgressColumn("% Qualify", min_value=0, max_value=100, format="%.0f%%", width=_MP_BAR_W,
                                            help="Share of the industry's stocks clearing all hard gates. SCALE-FREE, "
                                                 "not statistically robust — and much less robust here than on the "
                                                 "Sectors tab: the median industry holds 3 stocks against 19 for "
                                                 "sectors. This is a column, not the sort key, for exactly that reason."),
                        "avg_composite":  st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f", width=_MP_BAR_W),
                        "delta_vs_sector": st.column_config.NumberColumn("Δ vs Sector", format="%+.1f", width="small",
                                            help="Average score minus the average of its sector PEERS — the OTHER stocks "
                                                 "in the sector this industry mostly sits in; its own stocks are excluded "
                                                 "from the baseline (including them shrinks a dominant industry's gap "
                                                 "toward zero by its own weight). Both terms are computed over the SAME "
                                                 "filtered stocks. BLANK means incomparable, not zero: that sector holds "
                                                 "no other industry, so no peers exist."),
                        "pct_tier":       st.column_config.ProgressColumn(f"💹 {_ind_share_tier} %", min_value=0, max_value=100, format="%.0f%%", width=_MP_BAR_W,
                                            help=f"Share of the industry's FULL roster in the {_ind_share_tier} wealth tier. "
                                                 f"Follows the Wealth-tier filter (All → BUY★); the denominator deliberately "
                                                 f"IGNORES that filter — computed after it, the column would read 100% "
                                                 f"everywhere. N/A stocks dilute the share. Universe BUY★ base rate ≈ 12%. "
                                                 f"Read against Count — even more so here than on Sectors (median industry "
                                                 f"holds 3 stocks)."),
                        "avg_quality":    st.column_config.ProgressColumn("Quality",   min_value=0, max_value=100, format="%.0f", width=_MP_BAR_W),
                        "avg_improvement": st.column_config.ProgressColumn("Improvement", min_value=0, max_value=100, format="%.0f", width=_MP_BAR_W,
                                            help=_MP_IMPROVEMENT_HELP),
                        "avg_momentum":   st.column_config.ProgressColumn("Momentum",  min_value=0, max_value=100, format="%.0f", width=_MP_BAR_W),
                        "avg_valuation":  st.column_config.ProgressColumn("Valuation", min_value=0, max_value=100, format="%.0f", width=_MP_BAR_W),
                        "dom_sector":     st.column_config.TextColumn("Sector (dominant)", width=160,
                                            help="Industry is NOT nested inside sector — 136 of 355 span more than one. "
                                                 "This is the sector holding the LARGEST SHARE of the industry's stocks; a "
                                                 "leading '~' means under 80% sit there and the figure beside it is that "
                                                 "share, so read the Δ for that row loosely. An industry can TIE for two "
                                                 "sectors — the drill-down then reaches it from either, while this column "
                                                 "keeps one name because the Δ baseline needs exactly one peer group."),
                        **_MP_VIEW_CFG,
                    },
                    width="stretch",
                    height=min(700, 80 + len(_ind_stats) * 35),
                    hide_index=True,
                )

                _ind_blank = int(_ind_stats["delta_vs_sector"].isna().sum())
                _ind_tilde = int((_dom_share.reindex(_ind_stats.index) < 0.8).sum())
                st.markdown(
                    f'<div style="font-size:0.72rem;line-height:1.7;margin-top:12px;'
                    f'border-top:1px solid {COLORS["border"]};padding-top:10px;'
                    f'color:{COLORS["text_muted"]};">'
                    f'<span style="color:{COLORS["text_secondary"]};font-weight:700;">Reading this '
                    f'table.</span> {len(_ind_stats)} industries, no size floor. Check '
                    f'<strong>Count</strong> before trusting a Δ — a one-stock industry is just '
                    f'that single stock measured against its sector average. '
                    f'<strong>{_ind_blank}</strong> show a blank Δ — their sector contains no other '
                    f'industry, so there is nothing to compare them against (blank, deliberately, '
                    f'rather than a 0.0 that would read as "average"). '
                    f'<strong>{_ind_tilde}</strong> carry a <strong>~</strong> — fewer than 80% of '
                    f'their stocks sit in the sector named beside them, because industry is not a '
                    f'clean subdivision of sector.</div>',
                    unsafe_allow_html=True,
                )

    # ══ Movers ═════════════════════════════════════════════════════
    # APPENDED 2026-09-04 at index 6. What changed since the previous DATA VINTAGE — the one
    # question no other surface answers, and the engine's own thesis (direction beats level)
    # applied to itself. The previous side is an ARCHIVED Drive copy of the data sheet,
    # re-scored with the RUNNING engine (_load_vintage): same engine on both sides by
    # construction, so a move is the company changing, never PRISM changing. The diff and the
    # page are pure/stateless in ui/ui_movers.py; every widget (index id, picker, button, the
    # lens row) lives here because this file owns session state.
    with _mp_tabs[6]:
        from datetime import date as _mv_today
        from core.sheet_meta import fetch_sheet_title, parse_data_date
        from ui.ui_export import engine_version as _mv_engine
        from ui.ui_movers import (JOIN_KEY as _MV_KEY, compute_movers, default_vintage, fy_quarter,
                                  quarter_end_vintages, quarter_label, reason_counts, render_movers,
                                  restrict)

        # ONE LINE. Measured before this pass: seven controls and ~280 words sat above the first
        # data row and ⭐ What matters began ~900px down a 732px viewport. The how-and-why now
        # lives in the ⓘ tooltip on the result header, not in permanent prose.
        st.markdown(
            f"<div class='sec-cap'>What changed since the previous <b>data vintage</b> — the archived copy "
            f"re-scored by this engine, so a move is the company changing, never PRISM. "
            f"{_mp_scope_cap()} Filters narrow the current side.</div>", unsafe_allow_html=True)

        # ── 1. Archive id: secrets → session (seed-before-instantiate), then READ FROM SESSION
        # STATE before any widget renders. Streamlit commits a changed widget's value to
        # session_state before the rerun starts (the fact cfg_mode relies on), so the box can sit
        # wherever the layout wants — the compact row when unconfigured, a ⚙️ popover once secrets
        # supply it — without the data flow depending on widget order.
        try:
            _mv_default_id = str(st.secrets.get("ARCHIVE_INDEX_SHEET_ID", "") or "")
        except Exception:                    # no secrets file at all (local dev without one)
            _mv_default_id = ""
        if "mp_mv_index" not in st.session_state:
            st.session_state["mp_mv_index"] = _mv_default_id
        _mv_id = str(st.session_state["mp_mv_index"]).strip()
        _mv_configured = bool(_mv_default_id)

        def _mv_id_box():
            st.text_input(
                "Archive index sheet ID — or one archived copy's ID", key="mp_mv_index",
                placeholder="ID of 'PRISM Archive Index' (or set ARCHIVE_INDEX_SHEET_ID in secrets)",
                help="The Apps Script archiver keeps an index sheet of every archived vintage. Paste its "
                     "ID once here (or set ARCHIVE_INDEX_SHEET_ID in Streamlit secrets). A single archived "
                     "copy's ID also works — its vintage is read from the sheet's own name.")

        # ── 2. ONE compact setup row: [id box · picker · button] unconfigured, [picker · button · ⚙️]
        # once secrets carry the id. Slots are filled as the data becomes known — a columns row does
        # not care about fill order.
        _mv_row = st.columns([3.0, 2.6, 1.4] if not _mv_configured else [3.4, 1.9, 0.7])
        if _mv_configured:
            with _mv_row[2]:
                st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                with st.popover("⚙️", help="Archive index id — prefilled from secrets"):
                    _mv_id_box()
        else:
            with _mv_row[0]:
                _mv_id_box()
        _mv_pk, _mv_bt = (_mv_row[1], _mv_row[2]) if not _mv_configured else (_mv_row[0], _mv_row[1])

        if not _mv_id:
            st.info("No archive configured. Paste the PRISM Archive Index sheet ID above (or set "
                    "ARCHIVE_INDEX_SHEET_ID in secrets). Nothing to compare until then.")
        else:
            # ── 3. Resolve: an index (many vintages) or a single copy (one vintage from its name) ──
            _mv_ok, _mv_err, _mv_quarterly = None, None, False
            try:
                _mv_ok = _load_archive_index(_mv_id)
                # One copy per quarter: the calendar quarter-end copy, the cleanest state of a results
                # quarter (ui_movers.QUARTER_END_MONTHS). Until the archive holds one, every copy.
                _mv_qe = quarter_end_vintages(_mv_ok)
                if not _mv_qe.empty:
                    _mv_ok, _mv_quarterly = _mv_qe, True
            except ValueError:               # not an index → maybe a copy: read its own name
                _title = fetch_sheet_title(extract_spreadsheet_id(_mv_id))
                _d = parse_data_date(_title)
                if _d is None:
                    _mv_err = (f"'{_title or _mv_id}' is neither a PRISM Archive Index nor a "
                               f"'PRISM YYYY-MM-DD' copy — nothing to compare against.")
                else:
                    _mv_ok = pd.DataFrame({"vintage_date": [_d.isoformat()],
                                           "fy_quarter": [fy_quarter(_d.isoformat())],
                                           "spreadsheet_id": [extract_spreadsheet_id(_mv_id)]})
            except Exception as _e:
                _mv_err = (f"Could not read the archive: {_e}. Check the ID and that the sheet is "
                           f"shared (Anyone with the link → Viewer).")

            if _mv_err:
                st.error(_mv_err)
            elif _mv_ok is None or _mv_ok.empty:
                st.info("The archive index has no usable ('ok') vintage yet. The first archived "
                        "quarter appears there after the ingest archives it.")
            else:
                # ── 4. Pick the previous vintage. Options are index ROWS, not cascade facets: a
                # stored id no longer in the index cannot be loaded at all, so snapping to the
                # default here is honest, not the silent-widening class.
                _mv_ids = list(_mv_ok["spreadsheet_id"])
                # _mv_lab (short) names the copy on the button and in the notes; the picker reads the
                # results each quarter-end copy holds.
                _mv_lab = dict(zip(_mv_ok["spreadsheet_id"],
                                   _mv_ok["fy_quarter"] + " · " + _mv_ok["vintage_date"]))
                _mv_opt = (dict(zip(_mv_ok["spreadsheet_id"], _mv_ok["vintage_date"].map(quarter_label)))
                           if _mv_quarterly else _mv_lab)
                _mv_def = default_vintage(_mv_ok)["spreadsheet_id"]
                if st.session_state.get("mp_mv_pick") not in _mv_ids:
                    st.session_state["mp_mv_pick"] = _mv_def
                with _mv_pk:
                    _mv_pick = st.selectbox(
                        "Previous quarter" if _mv_quarterly else "Previous vintage", _mv_ids, key="mp_mv_pick",
                        format_func=lambda i: _mv_opt.get(i, i),
                        help=("One copy per quarter: the copy at the calendar quarter-end (30 Jun, 30 Sep, "
                              "31 Dec, 31 Mar). Every company has filed that quarter's results by then and the "
                              "next season has not begun, so a move is the business changing, not a company "
                              "that had not reported yet. To compare with any other archived copy, paste that "
                              "copy's ID where the archive index ID goes." if _mv_quarterly else
                              "No quarter-end copy is archived yet, so every copy is listed. Default = the most "
                              "recent on-cycle quarter; off-cycle rows are real data captured mid-quarter."))
                _mv_prev_v = str(_mv_ok.loc[_mv_ok["spreadsheet_id"] == _mv_pick, "vintage_date"].iloc[0])
                _mv_prev_q = str(_mv_ok.loc[_mv_ok["spreadsheet_id"] == _mv_pick, "fy_quarter"].iloc[0])
                _mv_cur_v = _fresh.data_date.isoformat() if _fresh.is_known else _mv_today.today().isoformat()
                _mv_cur_q = fy_quarter(_mv_cur_v)

                # ── 5. The re-score runs ONLY behind an explicit click (Market Pulse is a fragment;
                # every inner-tab body renders on every run). Once a compare has run for THIS pick
                # the label says "Re-compare" — a re-run is what it does (instant from cache unless
                # the engine or profile changed). The rerun after staging recomputes that label: on
                # the click run the button is instantiated BEFORE its own click is known.
                _mv_done = st.session_state.get("mp_mv_loaded") == _mv_pick
                _mv_btn = (f"↻ Re-compare with {_mv_lab[_mv_pick]}" if _mv_done
                           else f"🔁 Compare with {_mv_lab[_mv_pick]}")
                with _mv_bt:
                    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                    if st.button(_mv_btn, key="mp_mv_go"):
                        st.session_state["mp_mv_loaded"] = _mv_pick
                        st.rerun()
                if st.session_state.get("mp_mv_loaded") != _mv_pick:
                    st.info(f"Click **Compare** to re-score the {_mv_prev_q} copy ({_mv_prev_v}) with "
                            f"the current engine and diff it against today's data ({_mv_cur_v}). "
                            f"About ten seconds the first time; instant after.")
                else:
                    # PHASED, TIMED PROGRESS in a PLACEHOLDER: the phases are for the wait
                    # (download+derive cached on the copy → score cached on copy+engine+mode+profile
                    # → diff); on success the row is cleared and the total folds into the result
                    # header, so no dead row sits above the fold. On error the status stays, in red.
                    _mv_res, _mv_elapsed = None, None
                    _mv_ph = st.empty()
                    with _mv_ph.status(f"Comparing with {_mv_prev_q} ({_mv_prev_v})…", expanded=False) as _mv_st:
                        try:
                            _t0 = time.perf_counter()
                            _mv_st.update(label="① Downloading the archived copy and deriving signals…")
                            _mv_clean = _load_vintage_clean(_mv_pick)
                            _t1 = time.perf_counter()
                            _mv_st.write(f"① Downloaded + derived: {len(_mv_clean):,} stocks · {_t1 - _t0:.0f}s")
                            _mv_st.update(label=f"② Scoring with engine {_mv_engine()} ({analysis_mode})…")
                            _mv_prev_df, _mv_prev_regime = _score_vintage(
                                _mv_pick, _mv_engine(), analysis_mode, scoring_profile, _mv_clean)
                            _t2 = time.perf_counter()
                            _mv_st.write(f"② Scored · regime {_mv_prev_regime} · {_t2 - _t1:.0f}s")
                            _mv_st.update(label="③ Diffing against today's data…")
                            # The calendar gap makes 'fresh results' exact (reported AFTER the
                            # previous vintage) — see compute_movers(days_between=...).
                            _mv_gap = (_mv_today.fromisoformat(_mv_cur_v) - _mv_today.fromisoformat(_mv_prev_v)).days
                            _mv_res = compute_movers(_mv_prev_df, df, days_between=_mv_gap)
                            _t3 = time.perf_counter()
                            _mv_elapsed = _t3 - _t0
                            _mv_st.write(f"③ Diffed: {_mv_res['n_both']:,} stocks on both sides · {_t3 - _t2:.2f}s")
                            _mv_st.update(label=f"Compared with {_mv_prev_q} ({_mv_prev_v}) in {_mv_elapsed:.0f}s",
                                          state="complete")
                        except Exception as _e:
                            _mv_st.update(label=f"Could not compare with the {_mv_prev_q} copy", state="error")
                            st.error(f"Could not load or diff the {_mv_prev_q} copy: {_e}")
                            _mv_res = None
                    if _mv_res is not None:
                        _mv_ph.empty()       # the timing folds into the result header
                        if _mv_prev_v == _mv_cur_v:
                            st.warning("The previous copy carries the SAME vintage date as today's "
                                       "data — nothing can have moved. Pick an older vintage.")
                        # ── 6. FILTERS — one group, one 🧹. The lens row narrows the CURRENT side,
                        # applied AFTER the diff (never before: a filtered-out stock would read as
                        # 'dropped') and only when it actually narrowed the frame — chips alone must
                        # never stamp '(whole universe)' on an unrestricted page. The reason chips
                        # choose which material movers to show (inside material, before its cap;
                        # a kept stock keeps all its reasons); they are registered with the lens
                        # row's 🧹 via extra_keys so one Clear resets both.
                        # SCOPED FRAME IN, restrict() OUT — the sidebar cohort narrows the lens
                        # row's input, but the RESULT is still filtered after the diff by
                        # restrict(). Pre-filtering the frames compute_movers() diffs would turn
                        # every excluded stock into a fake "dropped" row (guarded in ui_movers).
                        _mv_cur_f, _mv_act = _mp_lens_row(_mp_df, "mv", extra_keys=("mp_mv_why",))
                        if len(_mv_cur_f) < len(df):
                            _mv_res = restrict(_mv_res, _mv_cur_f[_MV_KEY])
                        _mv_rc = reason_counts(_mv_res)
                        _mv_why = _mp_ms(st.container(), "What matters — reasons", list(_mv_rc),
                                         "mp_mv_why",
                                         "Keep only stocks carrying ANY selected reason (they keep all "
                                         "their reasons). Empty = every material mover. Counts are stocks.",
                                         _mv_rc)
                        _mv_picked = render_movers(_mv_res, {
                            "prev_vintage": _mv_prev_v, "cur_vintage": _mv_cur_v,
                            "prev_label": _mv_prev_q, "cur_label": _mv_cur_q,
                            "engine": _mv_engine(), "prev_engine": _mv_engine(),
                            "prev_regime": _mv_prev_regime,
                            "cur_regime": str(df.attrs.get("detected_market_regime", "SIDEWAYS")),
                            "mode": analysis_mode,
                            "reasons": _mv_why, "elapsed": _mv_elapsed,
                        })
                        # CLICK A MOVER, READ ITS TEAR-SHEET — the same handoff Tsunami and QGLP
                        # use. Stage a transient key + rerun rather than setting the xray_stock
                        # widget key (this tab renders AFTER the Tear-Sheet selectbox, so a direct
                        # set raises set-after-instantiation). The change-guard is essential:
                        # st.dataframe's selection persists across reruns, so an unguarded
                        # set+rerun would loop forever.
                        if _mv_picked and _mv_picked != st.session_state.get("xray_stock"):
                            st.session_state["_pending_xray"] = _mv_picked
                            st.rerun()

with tabs[4]:
    _render_market_pulse()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6: CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NEVER FRAGMENT THIS TAB. cfg_mode is "the one control that re-ranks the universe": the TOP of
# this script reads st.session_state["cfg_mode"] to build the scored frame. Inside a fragment,
# an Analysis-Mode change would rerun only the fragment — the frame would never recompute and
# every tab would show STALE RANKINGS under a control that claims to re-rank. A state-of-the-art
# audit proposed fragmenting this tab as a speedup on 2026-08-29; rejected for exactly this
# reason (pinned by test_market_pulse_tabs).
with tabs[5]:
    st.markdown(f"<div class='sec-head'>⚙️ System Configuration — The Engine Rulebook</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='sec-cap'>The live scoring control, then a read-only view of the "
        f"deterministic weights and hard gates every stock is measured against. To change the "
        f"constants, edit <code>config.py</code> — the single source of truth.</div>",
        unsafe_allow_html=True,
    )

    # ── The live scoring control (moved from the front-page Command Center, 2026-08-24) ──
    # ONE control since 2026-09-20. A "QGLP Screen Profile" selectbox sat beside this one and was
    # removed: across all eight profiles it left composite_score, rank and the top-50 identical
    # while swinging qglp_pass 4x, and a mode change could rewrite it silently. The half of it
    # that was ever informative — the gate thresholds — is printed as a FACT directly below.
    # Kept at half width: a lone full-width selectbox reads as a form rather than a dial.
    _cfg_c1, _ = st.columns(2)
    with _cfg_c1:
        st.selectbox(
            "Analysis Mode", options=list(ANALYSIS_MODES.keys()),
            # the default is MARKED, not just pre-selected: a reader who changes it should be able
            # to see what they moved away from without opening config.py
            format_func=lambda k: (ANALYSIS_MODES[k]["label"]
                                   + (" · default" if k == DEFAULT_ANALYSIS_MODE else "")),
            key="cfg_mode",
            help="Blend of the composite — the one control that re-ranks the universe "
                 "(Breakout 33/33/33 quality·momentum·breakout, the default since 2026-09-19 · "
                 "Hybrid 70/30 · Fundamental 100/0 · Technical 10/90). December's 6-month window "
                 "reviews the default; see the note in config.ANALYSIS_MODES.",
        )
    # The description renders OUTSIDE the column, at full width. Browser-checked 2026-09-20: with
    # the profile selectbox gone the right column is empty, and a caption left inside wrapped to
    # FOUR lines against that dead space. At full width it reads in two. The selectbox itself stays
    # half width — a full-width dropdown reads as a form field rather than a dial.
    st.caption(ANALYSIS_MODES[analysis_mode]["description"])
    _fit_cfg = int(((df["gate_pass"] == 1)
                    & (df.get("qglp_pass", pd.Series(0, index=df.index)) == 1)).sum())
    st.markdown(
        f'<div style="font-size:0.72rem;color:{COLORS["text_muted"]};margin:2px 0 14px 2px;">'
        f'🎯 QGLP screen — ROCE≥{adaptive_w.get("roce_gate", 15):.0f}% · '
        f'Growth≥{adaptive_w.get("growth_gate", 15):.0f}% · PEG≤{adaptive_w.get("peg_gate", 1.5):.1f} '
        f'&nbsp;→&nbsp;<span style="color:{COLORS["gold"]};font-weight:700;">{_fit_cfg} fit</span> '
        f'(of {gate_passed} gate-passed)</div>',
        unsafe_allow_html=True,
    )

    # ── Presentation helpers (pure display — no data mutation) ──────────────
    def _cfg_wbar(label: str, frac: float, color: str, note: str = "") -> str:
        """A labelled horizontal weight bar, clamped to [0,100]%."""
        w = max(0.0, min(100.0, float(frac) * 100.0))
        _note = (f'<span style="color:{COLORS["text_muted"]};font-weight:400;"> · {note}</span>'
                 if note else "")
        return (
            f'<div style="margin-bottom:9px;">'
            f'<div style="display:flex;justify-content:space-between;font-size:0.72rem;margin-bottom:3px;">'
            f'<span style="color:{COLORS["text_secondary"]};font-weight:600;">{label}{_note}</span>'
            f'<span style="color:{color};font-weight:800;">{frac*100:.0f}%</span></div>'
            f'<div style="background:{COLORS["bg_tertiary"]};border-radius:4px;height:6px;overflow:hidden;">'
            f'<div style="width:{w:.0f}%;height:6px;border-radius:4px;background:{color};"></div></div>'
            f'</div>'
        )

    def _cfg_card(title: str, icon: str, body_html: str, accent: str) -> str:
        return (
            f'<div style="background:{COLORS["bg_secondary"]};border:1px solid {COLORS["border"]};'
            f'border-left:3px solid {accent};border-radius:10px;padding:14px 16px;margin-bottom:12px;">'
            f'<div style="font-size:0.66rem;font-weight:800;color:{accent};text-transform:uppercase;'
            f'letter-spacing:1.2px;margin-bottom:10px;">{icon} &nbsp;{title}</div>{body_html}</div>'
        )

    _q_src = {"moat": "SQGLP", "growth": "SQGLP", "improvement": "3Y winners study", "cash": "Coffee Can",
              "margin": "Fisher", "balance_sheet": "Baid", "valuation": "Marks+Baid"}
    _q_clr = {"moat": COLORS["purple"], "growth": COLORS["green"], "improvement": COLORS["green"],
              "cash": COLORS["blue"],
              "margin": COLORS["orange"], "balance_sheet": COLORS["gold"], "valuation": COLORS["cyan"]}

    # ── Composite Score Formula — the master blend the sub-weights below feed into ──
    # Mirrors scoring_engine: composite = quality·(F) + momentum·(M) + governance·gov_w, where
    # governance is fixed and F+M fill (1-gov_w), split by analysis mode (from ANALYSIS_MODES — DRY).
    _gov_w  = COMPOSITE_WEIGHTS.get("governance", 0.15)
    _scale  = 1.0 - _gov_w
    _mode_icon = {"Hybrid": "🧭", "Fundamental": "📊", "Technical": "📈", "Breakout": "⚡"}
    _mode_rows = "".join(
        f'<div style="display:flex;justify-content:space-between;font-size:0.72rem;padding:3px 0;'
        f'border-bottom:1px solid rgba(255,255,255,0.04);">'
        f'<span style="color:{COLORS["text_secondary"]};">{_mode_icon.get(_m, "•")} {_m}</span>'
        f'<span style="color:{COLORS["text_muted"]};">Quality '
        f'<strong style="color:{COLORS["purple"]};">{_v["fundamental_w"]*100:.0f}%</strong> : '
        f'Momentum <strong style="color:{COLORS["orange"]};">{_v["momentum_w"]*100:.0f}%</strong>'
        # the third leg prints ONLY where a mode carries it — a 0% would read as a leg the other
        # modes have and set to nothing, when in fact they do not have it at all
        + (f' : Breakout <strong style="color:{COLORS["green"]};">{_v["breakout_w"]*100:.0f}%</strong>'
           if _v.get("breakout_w", 0.0) > 0.0 else "")
        + '</span></div>'
        for _m, _v in ANALYSIS_MODES.items()
    )
    _comp_body = (
        f'<div style="font-size:0.82rem;color:{COLORS["text_primary"]};font-weight:700;margin-bottom:4px;">'
        f'Composite = Quality × F &nbsp;+&nbsp; Momentum × M &nbsp;(+&nbsp;Breakout × B)&nbsp;+&nbsp; '
        f'Governance × <span style="color:{COLORS["gold"]};">{_gov_w*100:.0f}%</span></div>'
        f'<div style="font-size:0.68rem;color:{COLORS["text_muted"]};margin-bottom:8px;">'
        f'Governance is fixed at {_gov_w*100:.0f}%; F, M and B (B in the Breakout mode only) split the remaining {_scale*100:.0f}% by analysis mode:</div>'
        f'{_mode_rows}'
        f'<div style="font-size:0.68rem;color:{COLORS["text_muted"]};margin-top:8px;'
        f'border-top:1px solid {COLORS["border"]};padding-top:8px;">'
        f'Then: <strong style="color:{COLORS["text_secondary"]};">+ framework boosts</strong> (e.g. SQGLP +15) '
        f'→ <strong style="color:{COLORS["text_secondary"]};">× forensic penalty</strong> multiplier '
        f'→ clamped to a final 0–100 score.</div>'
    )
    st.markdown(_cfg_card("Composite Score Formula — How the Final Score is Built", "🧮",
                          _comp_body, COLORS["green"]), unsafe_allow_html=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        _qbody = "".join(
            _cfg_wbar(k.replace("_", " ").title(), v, _q_clr.get(k, COLORS["blue"]), _q_src.get(k, ""))
            for k, v in QUALITY_WEIGHTS.items()
        )
        st.markdown(_cfg_card(f"Quality Sub-Weights · {len(QUALITY_WEIGHTS)} Layers", "🏭", _qbody, COLORS["purple"]),
                    unsafe_allow_html=True)
    with cc2:
        _mbody = "".join(
            _cfg_wbar(k.replace("_", " ").title(), v, COLORS["orange"])
            for k, v in MOMENTUM_WEIGHTS.items()
        )
        _mbody += (
            f'<div style="border-top:1px solid {COLORS["border"]};margin-top:8px;padding-top:10px;">'
            + _cfg_wbar("Governance Blend (composite)", COMPOSITE_WEIGHTS["governance"], COLORS["gold"])
            + '</div>'
        )
        st.markdown(_cfg_card("Momentum Sub-Weights · CAN-SLIM", "⚡", _mbody, COLORS["orange"]),
                    unsafe_allow_html=True)

    # Hard gates — clean grid of pass-criteria chips
    _gate_cells = "".join(
        f'<div style="flex:1;min-width:210px;background:{COLORS["bg_tertiary"]};'
        f'border:1px solid {COLORS["border"]};border-radius:8px;padding:9px 12px;">'
        f'<span style="color:{COLORS["green"]};font-size:0.82rem;font-weight:800;">✓</span> '
        f'<span style="color:{COLORS["text_secondary"]};font-size:0.71rem;">{cfg["description"]}</span></div>'
        for _name, cfg in HARD_GATES.items()
    )
    st.markdown(
        _cfg_card(f"Hard Gates · {len(HARD_GATES)} Criteria — Every Stock Must Pass ALL", "🚨",
                  f'<div style="display:flex;gap:6px;flex-wrap:wrap;">{_gate_cells}</div>', COLORS["red"]),
        unsafe_allow_html=True,
    )

    # ── Conviction Tiers — the post-penalty composite_score → tier mapping (from CONVICTION_TIERS) ──
    _tier_rows = "".join(
        f'<div style="display:flex;align-items:center;gap:10px;padding:5px 0;'
        f'border-bottom:1px solid rgba(255,255,255,0.04);flex-wrap:wrap;">'
        f'<span style="font-size:0.78rem;font-weight:800;color:{t["color"]};min-width:150px;">'
        f'{t["emoji"]} {t["label"]}</span>'
        f'<span style="font-size:0.68rem;font-weight:700;color:{t["color"]};background:{t["color"]}1a;'
        f'border:1px solid {t["color"]}44;border-radius:5px;padding:1px 8px;white-space:nowrap;">'
        f'score ≥ {t["min"]}</span>'
        f'<span style="font-size:0.7rem;color:{COLORS["text_secondary"]};flex:1;min-width:200px;">'
        f'{t["description"]}</span></div>'
        for t in CONVICTION_TIERS
    )
    st.markdown(
        _cfg_card(f"Conviction Tiers · {len(CONVICTION_TIERS)} Bands — Score → Tier Mapping", "🏆",
                  _tier_rows, COLORS["gold"]),
        unsafe_allow_html=True,
    )

    # ── Asymmetric Penalty Multipliers — the two "× penalty" levers the formula card references ──
    # Both schedules render live from config (FORENSIC_PENALTY_TIERS + GOVERNANCE_RISK_MULTIPLIERS),
    # the SAME constants the engine applies — so this card can never drift from the real penalty.
    st.markdown(
        f'<div style="font-size:0.72rem;color:{COLORS["text_secondary"]};margin:2px 0 8px 2px;">'
        f'🔻 <strong>Negative signals don\'t subtract points — they MULTIPLY the composite down</strong>, '
        f'so the penalty scales with conviction (a 90-score loses more absolute points than a 20). '
        f'Forensic flags are <em>evidence</em> (harsher, ×0.50 floor); ownership signals are '
        f'<em>warnings</em> (milder, ×0.70 floor).</div>',
        unsafe_allow_html=True,
    )

    def _pen_color(m: float) -> str:
        """Severity tint for a penalty multiplier (display-only)."""
        return (COLORS["green"] if m >= 0.999 else COLORS["gold"] if m >= 0.85
                else COLORS["orange"] if m >= 0.70 else COLORS["red"])

    def _pen_row(left: str, mult: float, right: str = "") -> str:
        c = _pen_color(mult)
        _r = (f'<span style="font-size:0.66rem;color:{COLORS["text_muted"]};flex:1;">{right}</span>'
              if right else '<span style="flex:1;"></span>')
        return (
            f'<div style="display:flex;align-items:center;gap:10px;padding:4px 0;'
            f'border-bottom:1px solid rgba(255,255,255,0.04);">'
            f'<span style="font-size:0.72rem;color:{COLORS["text_secondary"]};min-width:92px;">{left}</span>'
            f'<span style="font-size:0.74rem;font-weight:800;color:{c};min-width:54px;">× {mult:.2f}</span>'
            f'{_r}</div>'
        )

    # Forensic cascade rows — derive the count RANGE from the ascending max_flags upper bounds.
    _fc_rows, _prev = "", -1
    for _t in FORENSIC_PENALTY_TIERS:
        _mx = _t["max_flags"]
        if _mx is None:
            _rng = f"{_prev + 1}+ flags"
        elif _mx == _prev + 1:
            _rng = f"{_mx} flag" + ("" if _mx == 1 else "s")
        else:
            _rng = f"{_prev + 1}–{_mx} flags"
        _fc_rows += _pen_row(_rng, _t["multiplier"], _t["label"])
        _prev = _mx if _mx is not None else _prev

    # Governance shield rows — exact count → multiplier; the highest key is the "N+" bucket.
    _gk = sorted(GOVERNANCE_RISK_MULTIPLIERS)
    _gmax = max(_gk)
    _g_lbl = {0: "clean", 1: "caution", 2: "structural concern", 3: "promoter signal"}
    _gov_rows = ""
    for _k in _gk:
        _lab = (f"{_k}+ signals" if (_k == _gmax and _k > 0)
                else "no signals" if _k == 0 else f"{_k} signal" + ("" if _k == 1 else "s"))
        _gov_rows += _pen_row(_lab, GOVERNANCE_RISK_MULTIPLIERS[_k], _g_lbl.get(_k, ""))

    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown(
            _cfg_card("Forensic Red-Flag Cascade — Evidence", "🔬",
                      f'<div style="font-size:0.64rem;color:{COLORS["text_muted"]};margin-bottom:6px;">'
                      f'red_flag_count → multiplier on composite_score</div>{_fc_rows}', COLORS["red"]),
            unsafe_allow_html=True,
        )
    with pc2:
        st.markdown(
            _cfg_card("Governance Risk Shield — Warnings", "🛡️",
                      f'<div style="font-size:0.64rem;color:{COLORS["text_muted"]};margin-bottom:6px;">'
                      f'hard ownership-risk signals → multiplier on composite_score</div>{_gov_rows}',
                      COLORS["gold"]),
            unsafe_allow_html=True,
        )

    # ── SYSTEM RISK MONITORS (Baid Sell Triggers + Mean Reversion) ──────────
    st.markdown("---")
    st.markdown(f"<div class='sec-head'>🛡️ System Risk Monitors</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='sec-cap'>Live, universe-wide risk counts computed by the engine this run.</div>",
        unsafe_allow_html=True,
    )
    _sell_cnt = int(df.get("sell_alert_any", pd.Series(0, dtype=int)).fillna(0).sum())
    _mr_cnt   = int(df.get("mean_reversion_risk", pd.Series(0, dtype=int)).fillna(0).sum())

    rm1, rm2 = st.columns(2)
    with rm1:
        _baid_clr  = COLORS["red"] if _sell_cnt else COLORS["green"]
        _baid_body = (
            f'<div style="font-size:1.7rem;font-weight:900;color:{_baid_clr};line-height:1;">'
            f'{_sell_cnt}<span style="font-size:0.7rem;color:{COLORS["text_muted"]};font-weight:600;">'
            f'&nbsp;stocks flagged</span></div>'
            f'<div style="margin-top:8px;">'
            + "".join(
                f'<div style="font-size:0.7rem;color:{COLORS["text_secondary"]};padding:3px 0;'
                f'border-bottom:1px solid rgba(255,255,255,0.04);">'
                f'<strong style="color:{COLORS["text_primary"]};">{n.replace("_"," ").title()}</strong> — '
                f'{c["description"]}</div>'
                for n, c in BAID_SELL_TRIGGERS.items()
            )
            + '</div>'
        )
        st.markdown(_cfg_card("Baid Sell Triggers", "📉", _baid_body, COLORS["red"]),
                    unsafe_allow_html=True)
    with rm2:
        _mr_clr  = COLORS["gold"] if _mr_cnt else COLORS["green"]
        _mr_body = (
            f'<div style="font-size:1.7rem;font-weight:900;color:{_mr_clr};line-height:1;">'
            f'{_mr_cnt}<span style="font-size:0.7rem;color:{COLORS["text_muted"]};font-weight:600;">'
            f'&nbsp;at cyclical peak</span></div>'
            f'<div style="font-size:0.72rem;color:{COLORS["text_secondary"]};margin-top:8px;">'
            f'OPM or NPM &gt; {MEAN_REVERSION["opm_spike_threshold"]}× their 5Y median — current '
            f'margins are likely unsustainable (Marks: extremes revert).</div>'
            f'<div style="font-size:0.72rem;color:{COLORS["text_muted"]};margin-top:8px;'
            f'border-top:1px solid {COLORS["border"]};padding-top:8px;">Quality-score penalty applied: '
            f'<strong style="color:{COLORS["gold"]};">−{(1-MEAN_REVERSION["penalty_factor"])*100:.0f}%</strong> '
            f'for each flagged stock.</div>'
        )
        st.markdown(_cfg_card("Mean Reversion Risk (Marks)", "🌡️", _mr_body, COLORS["gold"]),
                    unsafe_allow_html=True)

    # ── 🩺 DATA HEALTH — the source-sheet gaps the engine works around (LIVE, never hardcoded) ──
    # The engine rulebook's missing chapter: the two known sheet defects degrade real signals
    # (DPR → fabricated full retention; CR-1YB → dead Piotroski F6), and until now they lived only
    # in session memories. Every figure below is COMPUTED THIS RUN, so the day the sheet is fixed
    # the rows turn green by themselves — the card self-resolves, it cannot go stale.
    st.markdown("---")
    st.markdown(f"<div class='sec-head'>🩺 Data Health</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='sec-cap'>Known source-sheet gaps, measured live on this run's data. These are "
        f"fixed in the <strong>Google Sheet / CSVs</strong>, not in code — each row shows exactly "
        f"what it degrades and turns green on its own once the sheet carries the real figure.</div>",
        unsafe_allow_html=True,
    )

    def _dh_row(dot_clr: str, label: str, value: str, consequence: str) -> str:
        return (
            f'<div style="display:flex;align-items:baseline;gap:10px;padding:6px 0;'
            f'border-bottom:1px solid rgba(255,255,255,0.04);flex-wrap:wrap;">'
            f'<span style="color:{dot_clr};font-size:0.9rem;line-height:1;">●</span>'
            f'<span style="font-size:0.74rem;font-weight:700;color:{COLORS["text_primary"]};'
            f'min-width:150px;">{label}</span>'
            f'<span style="font-size:0.74rem;font-weight:800;color:{dot_clr};min-width:110px;">{value}</span>'
            f'<span style="font-size:0.68rem;color:{COLORS["text_muted"]};flex:1;min-width:260px;">'
            f'{consequence}</span></div>'
        )

    # 1. DPR coverage — missing payout is read as "retains everything" (RR fabricated at 1.0).
    _dpr_s = df.get("dividend_payout_ratio", pd.Series(np.nan, index=df.index))
    _dpr_cov = float(_dpr_s.notna().mean()) * 100.0
    _dpr_clr = (COLORS["green"] if _dpr_cov >= 90 else
                COLORS["gold"] if _dpr_cov >= 60 else COLORS["red"])
    _dh = _dh_row(
        _dpr_clr, "Dividend Payout (DPR)", f"{_dpr_cov:.0f}% populated",
        "Missing rows are read as full retention (RR = 1.0) — inflates Value Creation Velocity, "
        "g★ and the misallocation flag. Fix: the DPR column in the source sheet.",
    )

    # 2. CR one-year-back — the known copy bug: identical to current CR for every row.
    _cr0 = df.get("current_ratio", pd.Series(np.nan, index=df.index))
    _cr1 = df.get("current_ratio_1yb", pd.Series(np.nan, index=df.index))
    _cr_both = _cr0.notna() & _cr1.notna()
    _cr_same = float((_cr0[_cr_both] == _cr1[_cr_both]).mean()) * 100.0 if _cr_both.any() else np.nan
    if pd.isna(_cr_same):
        _dh += _dh_row(COLORS["text_muted"], "Current Ratio 1Y-back", "not reported",
                       "No prior-year liquidity figure at all — Piotroski F6 and the "
                       "liquidity-improving check cannot run.")
    else:
        _cr_clr = (COLORS["green"] if _cr_same < 50 else
                   COLORS["gold"] if _cr_same < 95 else COLORS["red"])
        _dh += _dh_row(
            _cr_clr, "Current Ratio 1Y-back", f"{_cr_same:.0f}% identical to CR",
            "A copy of the current figure carries no year-over-year information — Piotroski F6 "
            "and the liquidity-improving check stay dead until the sheet holds the real prior year.",
        )

    # 3. Overall evidence coverage — context, from the engine's own confidence input.
    _cov_s = df.get("data_coverage_pct", pd.Series(np.nan, index=df.index))
    if _cov_s.notna().any():
        _cov_med, _cov_p10 = float(_cov_s.median()), float(_cov_s.quantile(0.10))
        _cov_clr = COLORS["green"] if _cov_p10 >= 60 else COLORS["gold"]
        _dh += _dh_row(_cov_clr, "Evidence coverage", f"median {_cov_med:.0f}%",
                       f"The 44-input coverage behind the 🔍 confidence badge; the thinnest tenth "
                       f"of the universe sits at {_cov_p10:.0f}% or less.")

    # 4. Data vintage — every row above is only as current as the sheet itself.
    #    Read live from the spreadsheet's name, which the ingestion pipeline sets
    #    to the trading session the numbers came from ("PRISM 2026-08-28 Fri").
    #    Graded in SESSIONS, not calendar days: Friday's data on a Monday is the
    #    freshest that exists, and day-counting would call it three days stale.
    _vin = get_data_freshness(st.session_state.data_source, sheet_id, file_sig)
    if _vin.is_known:
        _dh += _dh_row(_freshness_color(_vin.tone), "Data as of",
                       f"{_vin.label} · {_vin.status}",
                       f"Taken from the spreadsheet name \"{_vin.title}\". "
                       f"The pipeline renames the sheet on every run, so this is the "
                       f"session these numbers actually describe — not when they were loaded."
                       if _vin.title else
                       "Most recently written of the six source CSVs.")
    else:
        _dh += _dh_row(COLORS["text_muted"], "Data as of", _vin.label,
                       "No date in the spreadsheet name, so freshness cannot be verified. "
                       "The ingestion pipeline names the sheet \"PRISM <YYYY-MM-DD> <Day>\"; "
                       "a hand-renamed sheet loses this check.")

    st.markdown(_cfg_card("Source-Sheet Gaps & Validation Cadence — Live", "🩺", _dh, COLORS["cyan"]),
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; padding:20px; color:{COLORS['text_muted']}; font-size:0.75rem;">
        PRISM v{UI['version']} · Quantamental Intelligence · Every lens, one verdict
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 7: REFERENCE — searchable glossary (renders the _RAW_GLOSSARY single source, count shown live)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FRAGMENT (2026-08-29): fully self-contained — static dicts + its own search box; a glossary
# search previously cost a full-app rerun per submit.
@st.fragment
def _render_reference():
    st.markdown(
        f'<div style="font-size:0.7rem;font-weight:700;color:{COLORS["text_muted"]};'
        f'text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">'
        f'📖 Reference — Glossary</div>',
        unsafe_allow_html=True,
    )
    _ref_q = st.text_input(
        "Search the glossary", key="ref_search",
        placeholder="Search any term or label (e.g. PEG, Wealth Creator, Stage 2)…",
        label_visibility="collapsed",
    )
    # Offline copy of the ENTIRE reference (ignores the search filter) — one generator, same
    # single-source dicts as the on-screen render, so the download can never drift from the app.
    # The 37-framework registry rides along: _FW_META (tuple (color, emoji, desc)) adapted to the
    # {emoji,name,desc} shape the builder emits — same single source the tearsheet renders.
    _fw_md = {name: {"emoji": meta[1], "name": name, "desc": meta[2]}
              for name, meta in _FW_META.items()}
    st.download_button(
        "📥 Download Reference (Markdown)",
        data=build_reference_markdown(_RAW_GLOSSARY, CONCEPT_REFERENCE, _FLAG_DISPLAY, frameworks=_fw_md,
                                      studies=WCS_STUDIES),
        file_name="prism_reference.md", mime="text/markdown",
        width="stretch",
    )
    # ── TWO-MODE LAYOUT (2026-08-29): BROWSE vs SEARCH ─────────────────────────────────────
    # The tab had become five stacked corpora (~70KB of text) — the app's worst scroll. BROWSE
    # (empty query) now shows five INNER TABS, one corpus each (the Market Pulse inner-tab
    # pattern). SEARCH (any query) hides the tab bar and renders UNIFIED results from all five
    # corpora — preserving the one-box-searches-everything power; splitting search per-tab would
    # recreate the export/search asymmetry class (frameworks, 2026-08-28). The five pure render
    # helpers are reused identically in both branches, so the modes can never drift.
    def _sec_head(text_html, color, top=6):
        st.markdown(
            f'<div style="font-size:0.72rem;font-weight:800;color:{color};'
            f'text-transform:uppercase;letter-spacing:1px;margin:{top}px 0 2px 0;">'
            f'{text_html}</div>', unsafe_allow_html=True)

    def _show_concepts(q):
        h = render_concepts(CONCEPT_REFERENCE, q)
        if h:
            _sec_head("Labels &amp; Verdicts — what each value means", COLORS["text_secondary"])
            st.markdown(h, unsafe_allow_html=True)

    def _show_glossary(q):
        _sec_head("Glossary — terms", COLORS["text_secondary"], top=20)
        st.markdown(render_reference(_RAW_GLOSSARY, q), unsafe_allow_html=True)

    def _show_flags(q):
        # Rendered straight from the engine's single-source _FLAG_DISPLAY (no copy).
        h = render_flags(_FLAG_DISPLAY, q)
        if h:
            _sec_head("Forensic Red Flags — what each warning means", COLORS["red"], top=20)
            st.markdown(h, unsafe_allow_html=True)

    def _show_frameworks(q):
        # The same _fw_md the download consumes — export and search can never disagree (2026-08-28).
        h = render_frameworks(_fw_md, q)
        if h:
            _sec_head(f"Framework Registry — the {len(_fw_md)} lenses", COLORS["purple"], top=20)
            st.markdown(h, unsafe_allow_html=True)

    def _show_wcs(q):
        # One entry per COMPLETELY-READ study (the WCS_STUDIES honesty contract); grows with the
        # early-era reading program; the same list rides into the Markdown download above.
        h = render_wcs_studies(WCS_STUDIES, q)
        if h:
            _sec_head(f"📚 Wealth Creation Studies — {len(WCS_STUDIES)} of 30 read &amp; verified",
                      COLORS["gold"], top=20)
            st.markdown(h, unsafe_allow_html=True)

    if _ref_q.strip():
        st.markdown(
            f'<div style="font-size:0.68rem;color:{COLORS["text_muted"]};margin:2px 0 8px 0;">'
            f'🔎 Searching across all five sections — clear the box to browse by tab.</div>',
            unsafe_allow_html=True)
        _show_concepts(_ref_q)
        _show_glossary(_ref_q)
        _show_flags(_ref_q)
        _show_frameworks(_ref_q)
        _show_wcs(_ref_q)
    else:
        _ref_tabs = st.tabs(["🏷️ Labels & Verdicts", "📖 Glossary", "🚩 Red Flags",
                             "🏛️ Frameworks", "📚 WCS Studies"])
        with _ref_tabs[0]:
            _show_concepts("")
        with _ref_tabs[1]:
            _show_glossary("")
        with _ref_tabs[2]:
            _show_flags("")
        with _ref_tabs[3]:
            _show_frameworks("")
        with _ref_tabs[4]:
            _show_wcs("")


with tabs[6]:
    _render_reference()
