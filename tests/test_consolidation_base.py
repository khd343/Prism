"""Contract: ⏳ Consolidation Base (2026-10-15) — one label for the engine, the sidebar filter, the
Tear-Sheet card and the Reference tab.

WHY. The Tear-Sheet card called every 10-day volume dry-up a "Consolidation Base" ("VCP Firing").
Measured 2026-10-15 on the live sheet (14 Oct data, 2,695 stocks) it fired for 1,218 (45%), 42-50% of
every trend stage alike — 377 of them Stage 4 declines — and its small print still quoted the pre-June
rule ("10D volume < 50D") after the engine moved to 70%. The label now keeps the dry-up a VCP needs apart from the ones that are not bases,
using only rules PRISM already applies (no new number):

  ⭐ VCP Setup      vcp_volume_dryup (10-day volume below 70% of the 50-day) + 📈 Stage 2 + within 25% of
                    the 52-week high (Trend Template criterion 7) + not below the Volatility Stop
  💤 Drying Up      the same dry-up anywhere else
  ⚪ Normal Volume   no dry-up
  ""               the volume averages are missing — the sidebar's ❔ Unknown, never "no dry-up"

⭐ + 💤 together are exactly the old "VCP Firing" set. Display only: no score reads the label.
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import pytest

import data_engine as DE
import ui.ui_tearsheet as T
from config import COLORS
from test_cockpit_display_truth import _Recorder, _render

_ROOT = Path(__file__).resolve().parent.parent

_MAPPED = set()
for _m in (DE.COMMON_COLS, DE.RATIO_COLS, DE.INCOME_COLS, DE.BALANCE_COLS,
           DE.CASHFLOW_COLS, DE.SHAREHOLDING_COLS, DE.TECHNICAL_COLS):
    _MAPPED.update(_m.values())

# Every leg of the setup met: Stage 2 (close 100 > 30-week 90 > 200-day 80), 10% below the 52-week
# high, the stop at 90 (the price 11% above it), the 10-day volume at 60% of the 50-day.
_SETUP = dict(close_price=100.0, sma_30w=90.0, sma_200d=80.0, dist_52wh=10.0, vstop_value=90.0,
              vol_sma_10d=60.0, vol_sma_50d=100.0)


def _derive(*rows):
    """compute_derived_signals over one row per override of _SETUP — every other mapped column NaN
    (the §6 synthetic-frame pattern)."""
    df = pd.DataFrame([{**_SETUP, **r} for r in rows])
    df.insert(0, "company_id", [f"NSE:T{i}" for i in range(len(df))])
    df.insert(1, "name", [f"T{i}" for i in range(len(df))])
    missing = sorted(c for c in _MAPPED if c not in df.columns)
    if missing:
        df = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=missing)], axis=1)
    return DE.compute_derived_signals(df)


def _labels(*rows):
    return _derive(*rows)["consolidation_base"].tolist()


# ── 1. The engine's label ─────────────────────────────────────────────────────────────────────────
def test_a_dry_up_in_a_stage_2_stock_near_its_high_and_above_its_stop_is_a_vcp_setup():
    assert _labels({}) == ["⭐ VCP Setup"]


@pytest.mark.parametrize("trend", [
    dict(sma_30w=90.0, sma_200d=95.0),     # 🔄 Stage 1: above a 30-week that sits below the 200-day
    dict(sma_30w=105.0, sma_200d=95.0),    # ⚠️ Stage 3: below a 30-week that sits above the 200-day
    dict(sma_30w=105.0, sma_200d=110.0),   # 📉 Stage 4: below a 30-week below the 200-day
    dict(sma_200d=np.nan),                 # ❔ no stage: unverifiable is not passed
], ids=["stage1", "stage3", "stage4", "stage_unknown"])
def test_a_dry_up_outside_stage_2_is_drying_up_not_a_base(trend):
    assert _labels(trend) == ["💤 Drying Up"]


def test_within_25_percent_of_the_high_is_inclusive_like_the_sepa_gate():
    assert _labels(dict(dist_52wh=25.0), dict(dist_52wh=25.01), dict(dist_52wh=np.nan)) == [
        "⭐ VCP Setup", "💤 Drying Up", "💤 Drying Up"]


def test_a_stock_below_its_stop_is_never_a_setup():
    """The Buy Zone calls a price under its Volatility Stop 'Trend Broken' — a broken trend never
    wears the good label (the Marks rule). AT the stop is not below it."""
    assert _labels(dict(vstop_value=105.0), dict(vstop_value=100.0), dict(vstop_value=np.nan)) == [
        "💤 Drying Up", "⭐ VCP Setup", "💤 Drying Up"]


def test_the_dry_up_line_is_the_engines_strict_70_percent():
    assert _labels(dict(vol_sma_10d=69.9), dict(vol_sma_10d=70.0), dict(vol_sma_10d=120.0)) == [
        "⭐ VCP Setup", "⚪ Normal Volume", "⚪ Normal Volume"]


@pytest.mark.parametrize("hole", [dict(vol_sma_10d=np.nan), dict(vol_sma_50d=np.nan),
                                  dict(vol_sma_50d=0.0)], ids=["no_10d", "no_50d", "zero_50d"])
def test_missing_volume_averages_are_unknown_never_normal_volume(hole):
    out = _derive(hole)
    assert out["consolidation_base"].iloc[0] == "", "a data hole is ❔ Unknown, not a measurement"
    assert np.isnan(out["vcp_volume_ratio"].iloc[0])
    assert out["vcp_volume_dryup"].iloc[0] == 0, "the SEPA bonus keeps reading the hole as no dry-up"


def test_the_ratio_is_the_10_day_average_over_the_50_day():
    assert _derive({}, dict(vol_sma_10d=150.0))["vcp_volume_ratio"].tolist() == pytest.approx([0.6, 1.5])


def test_setup_and_drying_up_together_are_exactly_the_old_vcp_firing_set():
    """⭐ + 💤 must be vcp_volume_dryup == 1 row for row, so picking both in the sidebar gives back the
    card's old "VCP Firing" list — nothing more, nothing less — and every state is reachable."""
    out = _derive(*[dict(vol_sma_10d=v10, sma_30w=s30, dist_52wh=d, vstop_value=vs)
                    for v10 in (50.0, 69.9, 70.0, 120.0, np.nan)
                    for s30 in (90.0, 105.0)
                    for d in (10.0, 30.0)
                    for vs in (90.0, 105.0)])
    dry_states = out["consolidation_base"].isin(["⭐ VCP Setup", "💤 Drying Up"])
    assert dry_states.tolist() == (out["vcp_volume_dryup"] == 1).tolist()
    assert sorted(set(out["consolidation_base"])) == sorted(
        ["⭐ VCP Setup", "💤 Drying Up", "⚪ Normal Volume", ""])


def test_no_score_reads_the_label_or_the_ratio():
    """Display only: written in data_engine.py, read by the UI. A scorer, gate or ranker that read them
    would move ranks silently — re-measure before ever wiring one in."""
    for f in sorted((_ROOT / "core").glob("*.py")):
        if f.name == "data_engine.py":
            continue
        src = f.read_text(encoding="utf-8")
        for col in ("consolidation_base", "vcp_volume_ratio"):
            assert col not in src, f"core/{f.name} reads {col} — the label is display-only"


# ── 2. The sidebar filter ─────────────────────────────────────────────────────────────────────────
def _sidebar_app():
    """Drives the REAL sidebar over four stocks, one per state: A ⭐ (Stage 2), B 💤 (Stage 4),
    C ⚪ (Stage 2), D a hole (Stage 2) — each with its own Lynch type, so the filter below can show
    what this one kept."""
    import pandas as _pd
    import streamlit as _st

    from ui.ui_discovery import render_discovery_sidebar

    df = _pd.DataFrame({
        "name":               ["A", "B", "C", "D"],
        "sector":             ["X"] * 4,
        "industry":           ["Y"] * 4,
        "weinstein_stage":    ["📈 Stage 2 Advancing", "📉 Stage 4 Declining",
                               "📈 Stage 2 Advancing", "📈 Stage 2 Advancing"],
        "consolidation_base": ["⭐ VCP Setup", "💤 Drying Up", "⚪ Normal Volume", ""],
        "lynch_category":     ["Fast Grower", "Declining", "Stalwart", "Slow Grower"],
        "conviction_tier":    [1, 2, 3, 4],
        "piotroski_fscore":   [5, 6, 7, 8],
        "red_flag_count":     [0, 1, 0, 1],      # a 0..0 range would break the Max-red-flags slider
        "quality_score":      [10.0] * 4,
        "composite_score":    [10.0] * 4,
    })
    filt = render_discovery_sidebar(df)
    _st.text(f"KEPT={sorted(filt['name'].tolist())}")
    _st.text(f"CULPRIT={filt.attrs.get('zero_culprit', '')}")


def _sidebar(**state):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_sidebar_app)
    for k, v in state.items():
        at.session_state[k] = v
    at.run(timeout=30)          # the whole sidebar renders; 3s (the default) overruns under a loaded suite
    assert not at.exception, f"sidebar raised: {at.exception}"
    return at


def _out(at, key):
    return next(t.value for t in at.text if t.value.startswith(key + "=")).split("=", 1)[1]


@pytest.mark.parametrize("pick,kept", [
    (["⭐ VCP Setup"], "['A']"),
    (["💤 Drying Up"], "['B']"),
    (["⭐ VCP Setup", "💤 Drying Up"], "['A', 'B']"),     # every dry-up: the card's old "VCP Firing"
    (["⚪ Normal Volume"], "['C']"),
    (["❔ Unknown"], "['D']"),                             # the honest hole is selectable
], ids=["setup", "drying", "every_dry_up", "normal", "unknown"])
def test_the_filter_keeps_exactly_its_picks(pick, kept):
    assert _out(_sidebar(sb_consolbase=pick), "KEPT") == kept


def test_its_live_counts_follow_the_stage_picked_above_it():
    at = _sidebar(sb_weinstein=["📈 Stage 2 Advancing"])
    assert at.multiselect(key="sb_consolbase").options == [
        "⭐ VCP Setup  ·  1", "⚪ Normal Volume  ·  1", "❔ Unknown  ·  1"]


def test_a_pick_here_narrows_every_filter_below_it():
    at = _sidebar(sb_consolbase=["⭐ VCP Setup"])
    assert at.multiselect(key="sb_lynchcat").options == ["Fast Grower  ·  1"]


def test_a_pick_that_empties_the_list_is_named_as_the_culprit():
    at = _sidebar(sb_weinstein=["📉 Stage 4 Declining"], sb_consolbase=["⭐ VCP Setup"])
    assert _out(at, "KEPT") == "[]"
    assert _out(at, "CULPRIT") == "Consolidation Base"


def test_an_active_pick_shows_its_chip_group_badge_and_funnel_count():
    at = _sidebar(sb_consolbase=["⭐ VCP Setup"])
    assert at.button(key="chip_sb_consolbase").label == "✕  Consolidation Base: ⭐ VCP Setup"
    assert "📈 Trend · Style · Flow  ·  1" in [e.label for e in at.expander]
    assert any("1 filter active" in m.value for m in at.markdown)


# ── 3. The Tear-Sheet card ────────────────────────────────────────────────────────────────────────
_STAGE2 = dict(weinstein_stage="📈 Stage 2 Advancing", dist_52wh=13.2, dist_to_vstop=8.0)


def _card(**over):
    """The ⏳ Consolidation Base card's own html — the last card of the cockpit's structural row."""
    html = _render(**over)
    return html[html.index("Consolidation Base"):]


def _seen(card):
    """The card as read on screen: its hover tip names every label, so a value check must skip it."""
    return re.sub(r'data-tip="[^"]*"', "", card)


def test_the_card_shows_the_engines_label_and_the_facts_it_was_decided_on():
    card = _card(consolidation_base="⭐ VCP Setup", vcp_volume_ratio=0.49, **_STAGE2)
    assert "⭐ VCP Setup" in _seen(card)
    assert "10-day volume at 49% of the 50-day · Stage 2 · 13% below its 52-week high" in card
    assert COLORS["gold"] in card


def test_a_drying_up_card_names_the_stage_and_the_stop_that_keep_it_from_a_setup():
    card = _card(consolidation_base="💤 Drying Up", vcp_volume_ratio=0.55,
                 weinstein_stage="📉 Stage 4 Declining", dist_52wh=38.0, dist_to_vstop=-6.0)
    assert "💤 Drying Up" in _seen(card)
    assert "10-day volume at 55% of the 50-day · Stage 4 · 38% below its 52-week high · below its stop" in card
    assert COLORS["gold"] not in card


def test_normal_volume_states_the_ratio():
    card = _card(consolidation_base="⚪ Normal Volume", vcp_volume_ratio=0.92, **_STAGE2)
    assert "⚪ Normal Volume" in _seen(card) and "10-day volume at 92% of the 50-day" in card


@pytest.mark.parametrize("hole", ["", np.nan], ids=["blank", "nan"])
def test_an_unknown_label_reads_not_available_never_normal(hole):
    seen = _seen(_card(consolidation_base=hole, vcp_volume_ratio=np.nan))
    assert "N/A" in seen and "volume averages not reported" in seen
    assert "Normal Volume" not in seen and "Active" not in seen


@pytest.mark.parametrize("ratio,label,shown", [
    (0.697, "💤 Drying Up", "69.7%"),        # a dry-up never prints as "70%": its line is BELOW 70%
    (0.6996, "⭐ VCP Setup", "69.96%"),
    (0.7004, "⚪ Normal Volume", "70%"),
], ids=["dry_rounds_up", "dry_two_places", "normal_at_line"])
def test_a_printed_ratio_never_contradicts_its_label(ratio, label, shown):
    card = _card(consolidation_base=label, vcp_volume_ratio=ratio, **_STAGE2)
    assert f"10-day volume at {shown} of the 50-day" in card


@pytest.mark.parametrize("dist,label,shown", [
    (25.3, "💤 Drying Up", "25.3% below its 52-week high"),   # failed the 25% line: never "25% below"
    (24.96, "⭐ VCP Setup", "25% below its 52-week high"),
    (0.2, "⭐ VCP Setup", "at its 52-week high"),
], ids=["over_the_line", "on_the_line", "at_the_high"])
def test_a_printed_distance_never_contradicts_its_label(dist, label, shown):
    card = _card(consolidation_base=label, vcp_volume_ratio=0.5,
                 weinstein_stage="📈 Stage 2 Advancing", dist_52wh=dist, dist_to_vstop=8.0)
    assert shown in card


# ── 4. The SEPA panel's V box (the same dry-up, scored as SEPA's bonus) ──────────────────────────────
def _sepa_html(**over):
    rec, real = _Recorder(), T.st
    try:
        T.st = rec
        T.render_sepa_radar(pd.Series({"sepa_pass": 0, "sepa_score": 3, **over}))
    finally:
        T.st = real
    return " ".join(rec.out)


def test_the_sepa_v_box_states_the_engines_dry_up_line_and_names_a_hole():
    assert "below 70% of the 50-day" in _sepa_html(sepa_vcp_dryup=1, vcp_volume_ratio=0.5)
    assert "at or above 70% of the 50-day" in _sepa_html(sepa_vcp_dryup=0, vcp_volume_ratio=0.9)
    assert "Volume averages not reported" in _sepa_html(sepa_vcp_dryup=0, vcp_volume_ratio=np.nan)


# ── 5. One vocabulary, one set of lines ───────────────────────────────────────────────────────────
def test_the_reference_tab_explains_each_label_the_filter_offers():
    from ui.ui_reference_data import CONCEPT_REFERENCE
    assert [lab for lab, _ in CONCEPT_REFERENCE["⏳ Consolidation Base"]] == list(DE.CONSOLIDATION_BASE_LABELS)


def test_searching_the_reference_for_the_filters_name_finds_every_state():
    """A reader who meets the ⏳ Consolidation Base filter and searches the Reference for its name must land
    on all three states. The search reads labels and explanations, never the category title (seen in the
    browser 2026-10-15: "Consolidation" found nothing)."""
    from ui.ui_reference import render_concepts
    from ui.ui_reference_data import CONCEPT_REFERENCE
    found = render_concepts(CONCEPT_REFERENCE, "consolidation base")
    for label in ("⭐ VCP Setup", "💤 Drying Up", "⚪ Normal Volume"):
        assert label in found, f"searching 'consolidation base' misses {label}"


def test_every_explanation_quotes_the_engines_own_lines():
    """The card's old small print said "10D < 50D" months after the engine moved to 70% — the drift this
    pins shut: the sidebar help, the card's tip and the Reference entry must state the engine's lines."""
    from ui.ui_reference_data import CONCEPT_REFERENCE
    from config import TREND_TEMPLATE_MAX_BELOW_HIGH        # one line for criterion 7, shared with SEPA
    dry, high = f"{DE.VCP_DRYUP_SHARE:.0%}", f"{TREND_TEMPLATE_MAX_BELOW_HIGH}%"
    texts = {
        "sidebar help": _sidebar().multiselect(key="sb_consolbase").help,
        "card tip": _card(consolidation_base="⭐ VCP Setup", vcp_volume_ratio=0.49, **_STAGE2),
        "Reference": dict(CONCEPT_REFERENCE["⏳ Consolidation Base"])["⭐ VCP Setup"],
    }
    for where, text in texts.items():
        assert dry in text and high in text, f"{where} does not state the engine's {dry} / {high} lines"
