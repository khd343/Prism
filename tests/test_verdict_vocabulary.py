"""
test_verdict_vocabulary.py
==========================
Contract for the engine verdict's vocabulary: 🟢 SOUND · 🟡 MIXED · 🔴 FLAWED.

RENAMED 2026-08-27 from BUY / WATCH / AVOID, for two reasons this file exists to keep true:

  1. NO WORD IN TWO LAYERS. The wealth tier then spoke action words (BUY★ / BUY / WATCH★ /
     WATCH / AVOID / N/A — the user's "calculated verdict, my decision"). The engine verdict
     speaking the SAME words meant "AVOID" carried two different meanings in adjacent columns —
     the cf_triangle "Perfect — Buy Zone" defect class rebuilt at vocabulary level. The engine
     now speaks CONDITION words only.

     And since 2026-10-15 the wealth tier speaks DIRECTION words (CLIMBING★ / CLIMBING /
     RECOVERING / UNEVEN / STALLED / N/A): measured on the live sheet, its BUY sat on 518 stocks
     — 85% of BUY★ FLAWED here — and its AVOID on businesses earning far above their cost of
     equity. Neither layer issues an action word now; the rule below now spans every label column
     on screen, not just these two (RISING was rejected for nesting inside the EP Power Curve's
     "Not Rising", IMPROVING for being that curve's own label).

  2. HONESTY. With 18 affirmatives against 93% of the universe failing, the engine verdict was
     never a buy call — it is a qualification gate (quality + forensics + valuation + timing).
     Condition words say what it measures; the action stays with the user.

THE COLLISION AUDIT THAT CHOSE THESE WORDS (each alternative rejected for a concrete reason):
     PASS / FAIL           conflates with gate_pass / schilit_pass / "Gate Passed"
     QUALIFIED / REJECTED  conflates with the Sectors tab's "% Qualify" (= gate-pass rate)
     STRONG / WEAK         already verdict_strength's vocabulary AND a conviction-tier label
     NEUTRAL               already smart_money_flow ⚪ and sector_capital_phase ⚖️
     UNSOUND               SOUND ⊂ UNSOUND — substring nesting, the bug class hit four times
                           on 2026-08-27 alone (QGLP⊂SQGLP, "sis ltd"⊂Mphasis, …)

THE THREE-LAYER ARCHITECTURE THIS PINS:
     Conviction Tier (1–5, Crown Jewel…)  how good, per the composite      — graded
     Verdict (SOUND/MIXED/FLAWED)         does it qualify, with the vetoes — gate
     Wealth Tier (CLIMBING★…N/A)          which direction                  — change

Old snapshots keep the recorded strings of both renames; tools/validate.py maps each column's own
old words on load so longitudinal comparisons survive them.

Run with: pytest tests/test_verdict_vocabulary.py -v
"""

import contextlib
import io as _io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd
import pytest

from core.verdict_engine import WEALTH_NA, WEALTH_TIERS, compute_verdict

ENGINE_WORDS = {"SOUND", "MIXED", "FLAWED"}
WEALTH_WORDS = {*WEALTH_TIERS, WEALTH_NA}
OLD_WORDS = {"BUY", "WATCH", "AVOID"}            # the verdict's own words until 2026-08-27
ACTION_WORDS = {"BUY★", "BUY", "WATCH★", "WATCH", "AVOID", "SELL", "HOLD", "ACCUMULATE"}

_BASE = {
    "conviction_tier": 1, "composite_score": 90.0, "forensic_score": 90.0,
    "schilit_pass": 1, "red_flag_count": 1, "corporate_class": "🏆 GREAT",
    "quality_score": 90.0, "growth_score": 80.0, "expected_excess_return": 20.0,
    "pe": 20.0, "fair_pe_qglp": 30.0, "buy_zone_label": "Accumulate",
    "governance_risk_multiplier": 1.0, "data_coverage_pct": 90.0,
    "net_worth": 1000.0, "economic_profit": 100.0,
    "economic_profit_velocity": 50.0, "moat_tau": 1.0,
}


def _verdict(**o):
    return compute_verdict(pd.DataFrame([{**_BASE, **o}]))


# -- 1. The vocabulary itself ----------------------------------------------------------------
def test_the_three_states_and_their_emoji():
    for o, word, emoji in [({}, "SOUND", "🟢"),
                           ({"conviction_tier": 3}, "MIXED", "🟡"),
                           ({"conviction_tier": 5}, "FLAWED", "🔴")]:
        d = _verdict(**o)
        assert d["verdict_direction"].iloc[0] == word
        assert d["verdict_emoji"].iloc[0] == emoji, f"{word} lost its {emoji}"


def test_the_vetoes_speak_the_new_words():
    assert _verdict(forensic_score=40.0)["verdict_direction"].iloc[0] == "FLAWED"
    assert _verdict(schilit_pass=0)["verdict_direction"].iloc[0] == "MIXED"   # soft downgrade


def test_old_action_words_never_appear():
    for o in ({}, {"conviction_tier": 3}, {"conviction_tier": 5}, {"forensic_score": 40.0}):
        v = _verdict(**o)["verdict_direction"].iloc[0]
        assert v not in OLD_WORDS, f"the old vocabulary is back: {v!r}"
        assert v in ENGINE_WORDS


# -- 2. THE RULE: no word in two layers ------------------------------------------------------
def test_engine_and_wealth_vocabularies_are_disjoint():
    """The reason for the rename. If these sets ever intersect, one word means two things in
    adjacent columns again."""
    assert ENGINE_WORDS & WEALTH_WORDS == set(), (
        f"vocabulary collision between the verdict and the wealth tier: "
        f"{ENGINE_WORDS & WEALTH_WORDS}"
    )


def test_no_engine_word_nests_inside_another():
    """UNSOUND was rejected because SOUND ⊂ UNSOUND. Keep the set nesting-free so substring
    matching can never silently cross-match (the QGLP⊂SQGLP class)."""
    for a in ENGINE_WORDS:
        for b in ENGINE_WORDS:
            if a != b:
                assert a not in b, f"{a!r} is a substring of {b!r}"


# -- 3. On the live universe -----------------------------------------------------------------
@pytest.fixture(scope="module")
def live():
    from data_engine import (coerce_numeric_columns, compute_derived_signals, load_all_csvs,
                             merge_datasets)
    from core import run_scoring_pipeline
    with contextlib.redirect_stdout(_io.StringIO()):
        return run_scoring_pipeline(
            compute_derived_signals(coerce_numeric_columns(merge_datasets(load_all_csvs("local")))))


def test_live_verdicts_use_exactly_the_new_vocabulary(live):
    got = set(live["verdict_direction"].dropna().astype(str))
    assert got <= ENGINE_WORDS, f"unexpected verdict values on live data: {got - ENGINE_WORDS}"
    assert got == ENGINE_WORDS, f"a verdict state died on live data — only {got} present"


def test_the_gate_stays_a_gate(live):
    """The calibration that justified condition words: an affirmative-rare, reject-heavy gate.
    If SOUND ever became common, the 'this is a gate, not advice' framing needs revisiting."""
    share = live["verdict_direction"].value_counts(normalize=True)
    assert share.get("SOUND", 0) < 0.10, f"SOUND at {share.get('SOUND', 0):.1%} — no longer rare"
    assert share.get("FLAWED", 0) > 0.50, "FLAWED no longer dominates — the gate's shape changed"


def test_neither_layer_speaks_an_action_word(live):
    """2026-10-15: the last action words left the screen with the wealth tier's rename."""
    for col in ("verdict_direction", "wealth_tier"):
        got = set(live[col].dropna().astype(str))
        assert not got & ACTION_WORDS, f"{col} speaks action words again: {sorted(got & ACTION_WORDS)}"
    assert set(live["wealth_tier"].dropna().astype(str)) == WEALTH_WORDS, "a wealth word died on live data"


def _label_columns(live):
    """Every column that shows a LABEL — a short categorical string — plus sector and industry, whose
    names sit beside the tier too (BUILDING was rejected for 'Building Materials')."""
    cols = []
    for c in live.columns:
        if c == "wealth_tier" or live[c].dtype != object:
            continue
        s = live[c].dropna().astype(str)
        if 0 < s.nunique() <= 80 and s.str.len().mean() <= 40 or c in ("sector", "industry"):
            cols.append(c)
    return cols


def test_no_wealth_word_appears_in_another_layers_labels(live):
    """NO WORD IN TWO LAYERS, across every label on screen and case-blind (RISING ⊂ "Not Rising" is
    exactly what a substring match would catch, and why RISING lost). A wealth word inside another
    label, or another label inside a wealth word, lets one filter or search cross layers."""
    cols = _label_columns(live)
    assert {"verdict_direction", "ep_power_curve", "moat_growth_quad", "buy_zone_label",
            "weinstein_stage", "sector", "industry"} <= set(cols), f"the label scan lost a column: {cols}"
    clashes = []
    for c in cols:
        for v in sorted(set(live[c].dropna().astype(str))):
            for w in sorted(WEALTH_TIERS):
                bare = w.rstrip("★").lower()
                if bare in v.lower() or (len(v) >= 4 and v.lower() in bare):
                    clashes.append(f"{w} ~ {c}={v!r}")
    assert not clashes, "a wealth word collides with another layer's label:\n  " + "\n  ".join(clashes)


def test_no_wealth_word_nests_inside_another_wealth_word_but_its_star():
    """Within the ladder only CLIMBING ⊂ CLIMBING★ nests — by design, and every reader matches the tier
    by equality (never a contains match), which tests/test_sector_filters.py pins for the share column."""
    nested = sorted((a, b) for a in WEALTH_TIERS for b in WEALTH_TIERS if a != b and a in b)
    assert nested == [("CLIMBING", "CLIMBING★")], nested


def test_the_glossary_explains_all_three(live):
    from ui.ui_reference_data import CONCEPT_REFERENCE
    section = next((v for k, v in CONCEPT_REFERENCE.items() if "Soundness" in k), None)
    assert section is not None, "the Soundness glossary section is gone"
    documented = {lab for lab, _ in section}
    assert ENGINE_WORDS <= documented, f"glossary missing: {ENGINE_WORDS - documented}"
    assert not (OLD_WORDS & documented), "the glossary still documents the old action words"


# -- 4. The strength word stays off the screen -----------------------------------------------
def test_verdict_strength_is_never_rendered():
    """DISPLAY-RETIRED 2026-08-27. verdict_strength is a measured 1:1 map of conviction_tier --
    zero added information -- and the hero showed it beside the tier badge AND the score: the
    same fact three times on one screen ("MIXED / HIGH CONVICTION / Score 90/100"). The COLUMN
    remains (snapshot-schema stability; the orphan principle -- an unsurfaced column harms
    nobody); no UI surface may READ it for display again. A surface that wants a strength word
    should render the tier, which is the same fact with its own established vocabulary.

    Matches the READ pattern, not the word -- app.py's explanatory comment legitimately names
    the column, and prose that names a banned thing is not the banned thing (the substring-scan
    lesson, learned four times on 2026-08-27)."""
    import glob
    import re
    root = os.path.join(os.path.dirname(__file__), "..")
    read_pat = re.compile(r"""(?:\.get\(\s*|_sg\(\s*|_g\(\s*\w+,\s*)["']verdict_strength["']""")
    offenders = []
    for f in [os.path.join(root, "app.py")] + glob.glob(os.path.join(root, "ui", "*.py")):
        if read_pat.search(_io.open(f, encoding="utf-8").read()):
            offenders.append(os.path.basename(f))
    assert not offenders, (
        f"these UI files read verdict_strength for display again: {offenders} -- it is a 1:1 "
        f"duplicate of the tier; render the tier instead"
    )
