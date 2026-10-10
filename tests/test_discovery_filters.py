"""Contract: the two new Discovery filters (Cyclicality Tier + Sector Capital Phase) are wired
correctly into the cascade. Static source check — no Streamlit runtime, no data dir.

Pins exactly the two genuinely-new risk points (everything else rides the battle-tested
_ms_cascade / _ordered_present machinery the other 11 categorical filters already use):

  1. REGISTRATION — each new sb_* key must appear in BOTH the 🏢 Universe _grp() call (the group
     badge) AND the bottom _active_n() total (the funnel count). Miss either and the active-filter
     count silently under-reports (no crash, a real bug).
  2. LABEL DRIFT — the filter's option labels must match what the engine actually emits. Cyclicality
     is import-pinned to TIER_LABELS (single source of truth). Sector-capital-phase has no exported
     constant (the 3 strings are inline literals in data_engine + the tearsheet tile), so we cross-file
     pin: the literals must appear in BOTH the filter AND the engine emitter — a one-sided rename fails.

NOT here by design: a column-existence test — cyclicality_tier is already runtime-guarded by
test_cyclicality.py and sector_capital_phase by test_capital_cycle_sector.py (re-testing violates §2).
"""
from pathlib import Path

import pytest

from config import FRAMEWORK_CATEGORIES
from core.cyclicality_map import TIER_LABELS
from ui.ui_discovery import _CHIP_META, _compute_active_chips, _family_count, keep_selected

_ROOT = Path(__file__).resolve().parent.parent
_DISC = (_ROOT / "ui" / "ui_discovery.py").read_text(encoding="utf-8")
_DENG = (_ROOT / "core" / "data_engine.py").read_text(encoding="utf-8")
_CAP_PHASES = ["🔥 Hot Capital (caution)", "❄️ Capital Starved (opportunity)", "⚖️ Neutral"]


def test_new_keys_registered_in_both_spots():
    grp_line = next(l for l in _DISC.splitlines() if '_grp("🏢 Universe"' in l)
    # rindex → the FINAL _active_n( call (the funnel TOTAL); .index would grab the `def _active_n(`
    # definition / the in-_grp `a = _active_n()` call, whose block holds only *keys, not the sb_* list.
    i = _DISC.rindex("_active_n(")
    active_block = _DISC[i:_DISC.index(")", i)]
    for k in ("sb_cyc", "sb_capphase"):
        assert k in grp_line,     f"{k} missing from the 🏢 Universe _grp() — group badge will undercount"
        assert k in active_block, f"{k} missing from _active_n() — the funnel total will undercount"
    # GENERALIZED 2026-08-28: this pin used to check only the two keys it was written for, which is
    # how sb_wealthtier shipped registered in the chips and the group badge but MISSING from the
    # funnel total — the headline read "No filters — full universe" over a filtered count. Every
    # chip-registered key must appear in the funnel total, so the NEXT filter cannot repeat it.
    from ui.ui_discovery import _CHIP_META
    missing = [k for k, _h, _kind in _CHIP_META if k not in active_block]
    assert not missing, (
        f"chip-registered filter keys missing from the funnel _active_n() total (the headline "
        f"will undercount active filters): {missing}"
    )


def test_clear_all_is_always_an_on_click_callback():
    """THE PARKED CLEAR-ALL BUG's call-site discipline (mechanised 2026-08-28): an inline
    `if st.button(...): clear_all_filters()` from the MAIN BODY runs after the sidebar widgets
    instantiated, and deleting an instantiated widget's key lets the frontend resurrect its value
    on the rerun — filters come back, intermittently, and AppTest (no frontend) can never catch
    it. So the discipline is pinned structurally: every clear button wires on_click, no call site
    invokes clear_all_filters() inline, and the function itself contains no st.rerun() (a no-op
    inside callbacks, and the tell-tale of the old inline pattern)."""
    app_src = (_ROOT / "app.py").read_text(encoding="utf-8")
    for src, where in ((_DISC, "ui_discovery.py"), (app_src, "app.py")):
        assert "clear_all_filters()" not in src.replace("def clear_all_filters()", ""), (
            f"an INLINE clear_all_filters() call is back in {where} — the resurrection bug returns"
        )
    assert app_src.count("on_click=clear_all_filters") == 2, "the two empty-state buttons must wire on_click"
    assert _DISC.count("on_click=clear_all_filters") == 1, "the sidebar button must wire on_click"
    fn = _DISC[_DISC.index("def clear_all_filters"):_DISC.index("def render_discovery_sidebar")]
    # Match the STATEMENT, not the docstring's prose mention of it (the substring-scan-over-source
    # lesson, learned repeatedly: prose naming a banned thing is not the banned thing).
    import re as _re
    assert not _re.search(r"^\s+st\.rerun\(\)\s*$", fn, _re.M), (
        "clear_all_filters must not call st.rerun() (no-op in callbacks)"
    )


def test_active_groups_stay_open_across_reruns():
    """st.expander has no persistent open-state (no key in Streamlit 1.54) → it re-renders at its
    static `expanded=` default every rerun, so selecting a filter / clicking Clear-all would snap
    the box shut. _grp must keep a group OPEN when it holds an active filter (expanded or a > 0).
    Pin it so the fix can't silently revert to a static `expanded=expanded`."""
    # Assert the actual st.expander KWARG (`expanded=(...)`) — unique to the call. The docstring's
    # prose says "(expanded or a > 0)" WITHOUT the `expanded=` prefix, so this can't be satisfied by
    # the comment alone: a revert to a static `expanded=expanded` genuinely RED-fails.
    assert "expanded=(expanded or a > 0)" in _DISC, (
        "_grp's st.expander() must pass expanded=(expanded or a > 0) — otherwise every rerun "
        "(filter select / Clear-all) collapses the box the user is working in"
    )


def test_filter_labels_match_the_engine_verbatim():
    # cyclicality: import-pinned to the engine's source of truth
    for v in TIER_LABELS.values():
        assert v in _DISC, f"cyclicality label drifted from TIER_LABELS: {v!r}"
    # capital-phase: no exported constant, so cross-file pin (emitter + filter must agree)
    for p in _CAP_PHASES:
        assert p in _DISC, f"capital-phase label missing from ui_discovery: {p!r}"
        assert p in _DENG, f"capital-phase label drifted in data_engine (emitter): {p!r}"


# ── applied-filter chip strip ─────────────────────────────────────────────────
def test_compute_active_chips_per_kind():
    """The pure chip detector handles every filter shape: ms (first value +N), ms_count (count, no
    raw cat_* leak), sel ('All'=off), slider-max (active when < ceiling), slider-min (active when >0),
    bool. Mirrors the funnel's active-logic so chips and badges agree."""
    state = {"sb_cyc": ["Defensive", "Financials"], "sb_sector": "Steel", "sb_industry": "All",
             "sb_maxrf": 3, "sb_mincov": 0, "sb_minscore": 70, "sb_gate": True,
             "sb_catalyst": ["cat_capacity", "cat_oplev"]}
    chips = dict(_compute_active_chips(state, rf_max=28))
    assert chips["sb_cyc"] == "Cyclicality: Defensive +1"   # ms: first value + N
    assert chips["sb_sector"] == "Sector: Steel"            # sel active
    assert "sb_industry" not in chips                       # "All" = off
    assert chips["sb_maxrf"] == "Max Red Flags: ≤3"         # slider-max active (3 < 28)
    assert "sb_mincov" not in chips                         # 0 = off
    assert chips["sb_minscore"] == "Min Score: ≥70"         # slider-min active
    assert chips["sb_gate"] == "Gate-passed"               # bool active
    assert chips["sb_catalyst"] == "Catalyst: 2 selected"  # ms_count — no raw cat_* leak


def test_no_active_filters_yields_no_chips():
    """All-default state → empty chip list (so the strip renders nothing)."""
    assert _compute_active_chips({}, rf_max=28) == []
    assert _compute_active_chips({"sb_cyc": [], "sb_sector": "All", "sb_maxrf": 28, "sb_minq": 0,
                                  "sb_gate": False}, rf_max=28) == []


def test_every_funnel_key_has_a_chip_entry():
    """Every sb_* key counted in the funnel total MUST have a _CHIP_META entry — else a filter can be
    active yet produce no removable chip (the registration-drift guard, like the _grp/_active_n pair)."""
    import re
    i = _DISC.rindex("_active_n(")
    total_keys = set(re.findall(r'"(sb_\w+)"', _DISC[i:_DISC.index(")", i)]))
    missing = total_keys - {k for k, _, _ in _CHIP_META}
    assert not missing, f"filters counted in the funnel total but with no chip entry: {missing}"


# ── 🎭 Framework Family filter (min-count per family, AND across families) ──────
def test_family_filter_registered_in_group_and_funnel():
    """sb_fwfam must be wired into BOTH the 🧬 Frameworks _grp() (group badge) AND the final
    _active_n() funnel total — the same registration contract as every other filter. Its min-count
    PARAMETER (sb_fwfam_min) is NOT an independent filter and must NOT be counted in the funnel."""
    grp_line = next(l for l in _DISC.splitlines() if '_grp("🧬 Frameworks"' in l)
    i = _DISC.rindex("_active_n(")
    active_block = _DISC[i:_DISC.index(")", i)]
    assert "sb_fwfam" in grp_line, "sb_fwfam missing from the 🧬 Frameworks _grp() — group badge undercounts"
    assert "sb_fwfam" in active_block, "sb_fwfam missing from _active_n() — funnel total undercounts"
    assert "sb_fwfam_min" not in active_block, "sb_fwfam_min is a parameter, not a counted filter"


def test_family_filter_sourced_from_framework_categories():
    """The family options are built from the live FRAMEWORK_CATEGORIES taxonomy (single source), so a
    renamed/added family can never silently desync the filter from the cards' category chips."""
    assert "FRAMEWORK_CATEGORIES" in _DISC, "family filter must read FRAMEWORK_CATEGORIES, not a hardcoded list"


def test_family_count_boundary_safe_and_correct():
    """_family_count counts only EXACT framework tokens (boundary-safe) — 'Bruised Blue Chip' must
    NOT match 'Bruised Blue Chip 29' — and sums across the family's member frameworks."""
    import pandas as pd
    df = pd.DataFrame({"frameworks_passed": [
        "Coffee Can, Diamond, Quality Compounder",   # 3 Moats frameworks
        "QGLP, CAN SLIM",                            # 0 Moats
        "Bruised Blue Chip",                         # boundary: NOT the '…29' framework
    ]})
    moats = next(fws for (_e, lbl, _c, fws) in FRAMEWORK_CATEGORIES if lbl == "Moats")
    assert list(_family_count(df, moats)) == [3, 0, 0]
    mosl = next(fws for (_e, lbl, _c, fws) in FRAMEWORK_CATEGORIES if lbl == "MOSL")
    assert list(_family_count(df, mosl)) == [0, 1, 0]  # row C ('Bruised Blue Chip') must score 0


def test_family_count_handles_missing_and_empty():
    """No frameworks_passed column, an empty fw list, or NaN cells → all-zero (never crashes)."""
    import pandas as pd
    assert list(_family_count(pd.DataFrame({"x": [1, 2]}), ["Coffee Can"])) == [0, 0]
    df = pd.DataFrame({"frameworks_passed": ["Coffee Can", None]})
    assert list(_family_count(df, [])) == [0, 0]
    assert list(_family_count(df, ["Coffee Can"])) == [1, 0]


# ══════════════════════════════════════════════════════════════════════
# Blank labels must never become dropdown options (added 2026-08-23)
# ══════════════════════════════════════════════════════════════════════
# `_ordered_present` built options with frame[col].dropna() — which strips NaN but NOT the
# empty string. The "no verdict from a data hole" pass makes an unknown label "" instead of
# fabricating a verdict, so 7 dropdowns began offering an unlabelled, unusable checkbox
# (peg_zone, mef_label, moat_growth_quad, cash_machine_label, d49_momentum_quality, plus
# ep_power_curve / earnings_power_box which carried default="" all along). Selecting it
# silently narrowed the universe to rows the engine deliberately declined to judge.


def _ui_discovery_source() -> str:
    import io as _io
    import os as _os
    return _io.open(_os.path.join(_os.path.dirname(__file__), "..", "ui", "ui_discovery.py"),
                    encoding="utf-8").read()


def test_option_builder_excludes_blank_labels():
    src = _ui_discovery_source()
    i = src.find("def _ordered_present")
    assert i > 0, "helper not found"
    block = src[i:i + 900]
    assert "dropna()" in block, "helper no longer drops NaN"
    assert "if v.strip()" in block, "helper does not exclude blank labels"


def test_blank_never_offered_across_label_columns():
    """Exercise the REAL helper (module-level since the ❔ Unknown pass, so import it directly)."""
    import pandas as _pd

    from ui.ui_discovery import _ordered_present as fn

    frame = _pd.DataFrame({
        "peg_zone":  ["🟢 Fair PEG", "", "🔴 Overpriced", None],
        "mef_label": ["", "   ", "✅ Intact", "🔴 Degrading"],
    })
    for col, order in [("peg_zone",  ["🟢 Fair PEG", "🔴 Overpriced"]),
                       ("mef_label", ["✅ Intact", "🔴 Degrading"])]:
        opts = fn(frame, col, order)
        assert "" not in opts, f"{col}: blank option offered -> {opts}"
        assert not any(o.strip() == "" for o in opts), f"{col}: whitespace option -> {opts}"
        for real in order:
            assert real in opts, f"{col}: lost a real label {real!r}"


# ══════════════════════════════════════════════════════════════════════
# ❔ Unknown option + zero-results culprit (added 2026-08-23)
# ══════════════════════════════════════════════════════════════════════
# Two approved UX upgrades, one shared substrate:
#   1. The honest blanks the "no verdict from a data hole" pass produces (255 peg_zone,
#      177 mef_label, 133 ep_power_curve, …) must be SELECTABLE — a "❔ Unknown" option that
#      matches NaN/blank rows — not just hidden. Weinstein already emits a LITERAL "❔ Unknown"
#      label, so the sentinel must match by value AND union the holes (never replace value-match).
#   2. When the cascade hits 0 results the funnel must NAME the filter that emptied it, which
#      requires every one of the ~32 filter applications to route through one choke point.


def test_unknown_option_appended_when_blanks_present():
    """_ordered_present appends ❔ Unknown (last) iff the cascade frame holds NaN/blank rows —
    and never duplicates it when the engine itself emits the literal label (Weinstein)."""
    import pandas as _pd

    from ui.ui_discovery import _UNKNOWN, _ordered_present

    frame = _pd.DataFrame({
        "peg_zone":  ["🟢 Fair PEG", "", "🔴 Overpriced", None],       # holes -> sentinel offered
        "mef_label": ["✅ Intact", "🔴 Degrading", "✅ Intact", "✅ Intact"],  # fully labelled -> no sentinel
        "weinstein_stage": ["📈 Stage 2 Advancing", "❔ Unknown", "❔ Unknown", None],  # literal + hole
    })
    opts = _ordered_present(frame, "peg_zone", ["🟢 Fair PEG", "🔴 Overpriced"])
    assert opts[-1] == _UNKNOWN, f"sentinel missing/not last -> {opts}"
    assert _UNKNOWN not in _ordered_present(frame, "mef_label", ["✅ Intact", "🔴 Degrading"]), \
        "sentinel offered for a fully-labelled column"
    wein = _ordered_present(frame, "weinstein_stage", ["📈 Stage 2 Advancing", "❔ Unknown"])
    assert wein.count(_UNKNOWN) == 1, f"literal engine label duplicated the sentinel -> {wein}"


def test_label_mask_sentinel_matches_holes_and_literal():
    """_label_mask: real labels match by value; the ❔ Unknown sentinel additionally claims the
    honest holes (NaN / '' / whitespace) AND still matches a literal '❔ Unknown' the engine
    emits — so 'Unknown' always means the full 'engine declined to judge' set."""
    import pandas as _pd

    from ui.ui_discovery import _UNKNOWN, _label_mask

    s = _pd.Series(["🟢 Fair PEG", "", "   ", None, "❔ Unknown", "🔴 Overpriced"])
    assert list(_label_mask(s, ["🟢 Fair PEG"])) == [True, False, False, False, False, False]
    assert list(_label_mask(s, [_UNKNOWN])) == [False, True, True, True, True, False]
    assert list(_label_mask(s, ["🔴 Overpriced", _UNKNOWN])) == [False, True, True, True, True, True]
    assert list(_label_mask(s, [])) == [False] * 6, "empty selection must match nothing"


def test_first_zero_filter_names_first_culprit():
    """The choke point records the FIRST filter that empties a previously non-empty cascade —
    never a later filter applied to an already-empty frame, never a filter that leaves rows."""
    import pandas as _pd

    from ui.ui_discovery import _first_zero_filter

    df = _pd.DataFrame({"x": [1, 2, 3]})
    log = []
    out = _first_zero_filter(df, df["x"] > 1, "Keeps Rows", log)
    assert len(out) == 2 and log == [], "a filter that leaves rows must not be logged"
    out = _first_zero_filter(out, out["x"] > 99, "The Culprit", log)
    assert len(out) == 0 and log == ["The Culprit"]
    out = _first_zero_filter(out, out["x"] > 0, "Too Late", log)
    assert log == ["The Culprit"], "an already-empty frame must not overwrite the culprit"


def test_zero_culprit_reaches_the_main_panel_empty_state():
    """Two places report "no stocks match": the sidebar funnel and the Discovery main-panel card.
    They must name the SAME culprit — a funnel that says "PEG Zone removed the last stocks" beside
    a main panel that says only "loosen one" makes the user hunt for what the code already knows.

    The sidebar publishes the culprit on `filt.attrs` (the same channel the pipeline already uses
    for `detected_market_regime`); app.py's empty branch reads it. Set on `filt` AFTER the .copy(),
    so this never depends on pandas propagating attrs through an operation — which differs between
    the local pandas 3 and prod's pandas 2.
    """
    disc = _ui_discovery_source()
    app_src = (_ROOT / "app.py").read_text(encoding="utf-8")
    assert 'attrs["zero_culprit"]' in disc, \
        "render_discovery_sidebar must publish the culprit on filt.attrs"
    assert "zero_culprit" in app_src, \
        "the Discovery main-panel empty state must read filt.attrs['zero_culprit']"
    # the publish must happen on `filt` (post-copy), never on the cascade frame `_cf`
    assert '_cf.attrs["zero_culprit"]' not in disc, \
        "publish on filt (post-copy), not on _cf — attrs propagation through .copy() is not a contract"


def _zero_culprit_app():
    """Mini-app for AppTest: drives the REAL sidebar over a 2-row synthetic frame (no CSVs needed,
    fully deterministic) and prints what the caller receives."""
    import pandas as _pd
    import streamlit as _st

    from ui.ui_discovery import render_discovery_sidebar

    df = _pd.DataFrame({
        "name":             ["A", "B"],
        "sector":           ["X", "X"],
        "industry":         ["Y", "Y"],
        "conviction_tier":  [1, 2],
        "piotroski_fscore": [5, 6],
        "red_flag_count":   [0, 1],
        "quality_score":    [10.0, 12.0],
        "composite_score":  [10.0, 12.0],
    })
    filt = render_discovery_sidebar(df)
    _st.text(f"N={len(filt)}")
    _st.text(f"CULPRIT={filt.attrs.get('zero_culprit', '<MISSING>')}")


def test_sidebar_publishes_the_culprit_through_the_real_widget_machinery():
    """End-to-end on the ONE link the pure unit tests can't reach: that the culprit survives out of
    render_discovery_sidebar to the caller, through real Streamlit widgets. Both synthetic rows score
    10-12, so a Min-Composite of 50 empties the frame and 'Min Score' must be named."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_zero_culprit_app)
    at.session_state["sb_minscore"] = 50          # above both composites -> guaranteed zero
    # EXPLICIT TIMEOUT. AppTest defaults to 3s, and this script renders the whole discovery
    # sidebar; under a loaded machine (a parallel full-suite run) it overran and the failure looked
    # exactly like a real regression -- it cost three verification runs on 2026-08-31 before being
    # identified as a timeout. Nothing about the assertions changed.
    at.run(timeout=30)
    assert not at.exception, f"sidebar raised: {at.exception}"
    out = [t.value for t in at.text]
    assert "N=0" in out, f"filter did not empty the frame: {out}"
    assert "CULPRIT=Min Score" in out, f"culprit not published to the caller: {out}"


def test_no_culprit_published_when_results_remain():
    """The inverse: with no filter active the frame is full, so the culprit must be EMPTY — never a
    stale name that would make the main panel accuse an innocent filter."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_zero_culprit_app)
    at.run(timeout=30)                            # see the sibling test: 3s default is too tight
    assert not at.exception, f"sidebar raised: {at.exception}"
    out = [t.value for t in at.text]
    assert "N=2" in out, f"unfiltered frame should keep both rows: {out}"
    assert "CULPRIT=" in out, f"culprit key must exist and be empty: {out}"


# ── Never silently drop what the user chose (2026-09-02) ──────────────────────────────────────
def test_keep_selected_appends_dead_picks_last_and_never_duplicates():
    """THE ONE RULE both filter surfaces share (ui_discovery._ms_cascade + both selectboxes, and
    app.py's _mp_ms imports it — one definition). A stored pick that the cascade has narrowed
    out of the live options is KEPT — appended after the live options, so the canonical order is
    untouched and Streamlit's value-not-in-options crash cannot occur — and it renders with its
    honest count of 0. The old rule PRUNED it, which switched the filter off and showed MORE
    stocks: Wealth Tier=BUY★ then Sector=Air Transport Service returned the sector's 4 non-BUY★
    names where the honest answer is 0 (measured 2026-09-02, 106 cascade pairs affected)."""
    assert keep_selected(["a", "b"], ["b", "z"]) == ["a", "b", "z"]
    assert keep_selected(["a", "b"], []) == ["a", "b"]
    assert keep_selected([], ["z"]) == ["z"]
    assert keep_selected(["a"], ["z", "z"]) == ["a", "z"], "a repeated stored value must not duplicate"
    assert keep_selected(["a", "b"], ["b", "a"]) == ["a", "b"], "live order wins over stored order"
    assert keep_selected(["a"], "All") == ["a", "All"], "a scalar (selectbox) value is kept too"
    assert keep_selected(["All", "a"], "All") == ["All", "a"], "a live scalar is not appended again"


def _widening_app():
    """Mini-app for AppTest: a 2-row frame whose two rows disagree on EVERY facet, so a downstream
    pick can always be narrowed out by an upstream one."""
    import pandas as _pd
    import streamlit as _st

    from ui.ui_discovery import render_discovery_sidebar

    df = _pd.DataFrame({
        "name":             ["A", "B"],
        "sector":           ["P", "Q"],
        "industry":         ["P-ind", "Q-ind"],
        "market_category":  ["Large Cap", "Nano Cap"],
        "wealth_tier":      ["BUY★", "AVOID"],
        "conviction_tier":  [1, 2],
        "piotroski_fscore": [5, 6],
        "red_flag_count":   [0, 1],
        "quality_score":    [10.0, 12.0],
        "composite_score":  [10.0, 12.0],
    })
    filt = render_discovery_sidebar(df)
    _st.text(f"N={len(filt)}")
    _st.text(f"CULPRIT={filt.attrs.get('zero_culprit', '<MISSING>')}")
    _st.text(f"WT={_st.session_state.get('sb_wealthtier')}")
    _st.text(f"SECTOR={_st.session_state.get('sb_sector')}")


def _run(at):
    at.run(timeout=30)
    assert not at.exception, f"sidebar raised: {at.exception}"
    return {t.value.split("=", 1)[0]: t.value.split("=", 1)[1] for t in at.text}


def test_upstream_pick_cannot_switch_off_a_downstream_multiselect():
    """Wealth Tier=BUY★ (downstream) then Sector=Q, which holds no BUY★ (upstream). The honest
    result is ZERO and the culprit is the tier filter — not the whole of sector Q with the tier
    quietly dropped. The stored selection must survive the run intact."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_widening_app)
    at.session_state["sb_wealthtier"] = ["BUY★"]
    r = _run(at)
    assert r["N"] == "1", f"tier filter alone should keep exactly A: {r}"
    at.session_state["sb_sector"] = "Q"
    r = _run(at)
    assert r["N"] == "0", f"the tier filter was silently dropped — the widening is back: {r}"
    assert r["CULPRIT"] == "Wealth Tier", f"the culprit must name the filter that emptied it: {r}"
    assert r["WT"] == "['BUY★']", f"the user's pick was pruned from session state: {r}"


def test_upstream_pick_cannot_switch_off_a_downstream_selectbox():
    """Same law for the two selectboxes: Sector=P then Market Category=Nano Cap (which P lacks).
    The old seed-before-instantiate reset the sector to 'All' whenever it fell out of the live
    list — the same silent widening in selectbox clothing."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_widening_app)
    at.session_state["sb_sector"] = "P"
    r = _run(at)
    assert r["N"] == "1", f"sector filter alone should keep exactly A: {r}"
    at.session_state["sb_mcap"] = ["Nano Cap"]
    r = _run(at)
    assert r["N"] == "0", f"the sector filter reset itself to All — the widening is back: {r}"
    assert r["CULPRIT"] == "Sector", f"the culprit must name the sector filter: {r}"
    assert r["SECTOR"] == "P", f"the user's sector pick was reset: {r}"


def test_no_filter_site_prunes_a_stored_selection_any_more():
    """Structural: the prune comprehension is gone from _ms_cascade, the 'reset to All when absent'
    seeds are gone from both selectboxes, and all three route through keep_selected."""
    i = _DISC.index("def _ms_cascade(")
    body = _DISC[i:_DISC.index("_cf = df", i)]
    assert "if v in options]" not in body, "_ms_cascade still prunes the stored selection"
    assert "keep_selected(" in body, "_ms_cascade does not route through keep_selected"
    for opts in ("_sector_opts", "_industry_opts"):
        assert f"not in {opts}" not in _DISC, f"a selectbox still resets to All when its value leaves {opts}"
    assert _DISC.count("keep_selected(") >= 3, "expected _ms_cascade + both selectboxes to use keep_selected"


# ── Two filters admitted 2026-09-10 (All-Data gap sweep) ─────────────────────────────────────
def test_capital_allocation_filter_is_registered_and_cascades():
    """💰 Capital Allocation — the OUTSIDERS axis (buybacks/dividends vs dilution), computed by the
    engine and shown in All Data but unscreenable until now. MEASURED on live data: 3 well-spread
    values (Returning 22% · Neutral 70% · Raising 8%) and ORTHOGONAL — max overlap with ANY existing
    filter is 34% (vs 💰 Cash Machine). 'Returning capital AND BUY★' was previously inexpressible."""
    grp = next(l for l in _DISC.splitlines() if '_grp("🎯 Decision & Class"' in l)
    assert "sb_capalloc" in grp, "not registered in the 🎯 Decision & Class group badge"
    i = _DISC.rindex("_active_n(")
    assert "sb_capalloc" in _DISC[i:_DISC.index(")", i)], "missing from the funnel total"
    assert '_ms_cascade("Capital Allocation"' in _DISC, "must be the shared cascade multiselect"
    assert 'count_col="capital_allocation_signal"' in _DISC, "no live facet counts"
    assert '_label_mask(_cf["capital_allocation_signal"], sel_capalloc)' in _DISC, (
        "must use _label_mask so the ❔ Unknown sentinel claims the honest holes")


def test_float_funded_filter_is_registered_and_seeds_before_instantiating():
    """💧 Float-funded (negative working capital) — customers fund the business (the Dhandho/float
    read). MEASURED: fires 11.8%, and only 13% overlap with the nearest existing filter — the most
    orthogonal candidate in the whole All-Data gap. Crucially it is an INCLUSION screen that
    Max-red-flags cannot express: P(rf>=3 | flag) is 80%, EXACTLY the universe baseline, so it is
    independent of forensic severity rather than a slice of it."""
    grp = next(l for l in _DISC.splitlines() if '_grp("💰 Moat · Value · Entry"' in l)
    assert "sb_floatfunded" in grp, "not registered in the 💰 Moat · Value · Entry group badge"
    i = _DISC.rindex("_active_n(")
    assert "sb_floatfunded" in _DISC[i:_DISC.index(")", i)], "missing from the funnel total"
    assert 'st.session_state.setdefault("sb_floatfunded", False)' in _DISC, (
        "the Steel repro: every non-multiselect must seed before instantiating")
    j = _DISC.index('st.session_state.setdefault("sb_floatfunded"')
    assert j < _DISC.index('key="sb_floatfunded"'), "seeded AFTER the widget — resurrection class"


def test_the_two_new_filters_have_chips_and_narrow_through_the_choke_point():
    from ui.ui_discovery import _CHIP_META
    kinds = {k: kind for k, _h, kind in _CHIP_META}
    assert kinds.get("sb_capalloc") == "ms", "Capital Allocation chip missing/wrong kind"
    assert kinds.get("sb_floatfunded") == "bool", "Float-funded chip missing/wrong kind"
    for label in ('"Capital Allocation")', '"Float-Funded")'):
        assert label in _DISC, f"{label} never reaches _narrow — the culprit log cannot name it"


def _new_filters_app():
    """Mini-app for AppTest: drives the REAL sidebar over a 3-row frame built so every new filter
    discriminates — including against its own inversion."""
    import pandas as _pd
    import streamlit as _st

    from ui.ui_discovery import render_discovery_sidebar

    df = _pd.DataFrame({
        "name":                      ["A", "B", "C"],
        "sector":                    ["X", "X", "X"],
        "industry":                  ["Y", "Y", "Y"],
        "conviction_tier":           [1, 2, 3],
        "piotroski_fscore":          [5, 6, 7],
        "red_flag_count":            [0, 1, 2],
        "quality_score":             [10.0, 12.0, 14.0],
        "composite_score":           [10.0, 12.0, 14.0],
        "capital_allocation_signal": ["💰 Returning Capital", "⚠️ Raising Capital", "💰 Returning Capital"],
        "negative_wc_flag":          [1, 0, 0],
    })
    filt = render_discovery_sidebar(df)
    _st.text(f"N={len(filt)}")
    _st.text(f"KEPT={sorted(filt['name'].tolist())}")


def _run_new(**state):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_new_filters_app)
    for k, v in state.items():
        at.session_state[k] = v
    at.run(timeout=30)
    assert not at.exception, f"sidebar raised: {at.exception}"
    out = {t.value.split("=", 1)[0]: t.value.split("=", 1)[1] for t in at.text}
    return out


def test_the_two_new_filters_actually_filter_through_the_real_widget_machinery():
    """BEHAVIOURAL, because the structural pins above are not enough — a mutation run proved they
    stay green when a filter is rendered but NEVER APPLIED (`if False and ...`) and when the
    float-funded mask is INVERTED. Source text is not behaviour; this runs the real cascade."""
    assert _run_new()["KEPT"] == "['A', 'B', 'C']", "unfiltered baseline is wrong"
    # capital allocation selects by label, and is not inverted
    assert _run_new(sb_capalloc=["💰 Returning Capital"])["KEPT"] == "['A', 'C']"
    assert _run_new(sb_capalloc=["⚠️ Raising Capital"])["KEPT"] == "['B']"
    assert _run_new(sb_capalloc=["💰 Returning Capital", "⚠️ Raising Capital"])["N"] == "3", "OR within the control"
    # float-funded keeps ONLY negative working capital (inversion would keep B and C)
    assert _run_new(sb_floatfunded=True)["KEPT"] == "['A']"
    assert _run_new(sb_floatfunded=False)["N"] == "3", "unticked must filter nothing"
    # AND across controls, and the pair composes
    assert _run_new(sb_capalloc=["💰 Returning Capital"], sb_floatfunded=True)["KEPT"] == "['A']"
    assert _run_new(sb_capalloc=["⚠️ Raising Capital"], sb_floatfunded=True)["N"] == "0", (
        "an empty intersection must be empty, not silently widened")


def test_the_new_filters_name_themselves_as_the_zero_results_culprit():
    """The funnel's zero-state must name the dial to loosen — so both must apply through _narrow,
    not merely appear in the source."""
    from streamlit.testing.v1 import AppTest
    for state, culprit in [({"sb_capalloc": ["⚠️ Raising Capital"], "sb_floatfunded": True}, "Float-Funded"),
                           ({"sb_floatfunded": True, "sb_minscore": 50}, "Min Score")]:
        at = AppTest.from_function(_new_filters_app)
        for k, v in state.items():
            at.session_state[k] = v
        at.run(timeout=30)
        assert not at.exception, f"sidebar raised: {at.exception}"
        assert "N=0" in [t.value for t in at.text], f"expected an empty frame for {state}"


# ── EVERY filter must actually filter (2026-09-10) ───────────────────────────────────────────
# MEASURED before this existed: mutating each filter's guard to `if False and …` — rendered,
# counted, chipped, applying NOTHING — left 27 of 31 filters UNDETECTED by the whole suite. Only
# Market Cap and Sector (the widening tests) and the two 2026-09-10 admissions were protected.
# A broken filter never crashes; it returns MORE stocks than it should, and every number on
# screen stays plausible. This table closes that gap for the whole sidebar.
#
# THE FRAME: 3 rows. On every LABEL column A and C share a value and B differs, so selecting
# A's value must yield [A, C] and selecting B's must yield [B] — which fails an inverted mask
# as well as a dead one. On every FLAG column only A fires, so a flag filter must yield [A].
def _all_filters_app():
    import pandas as _pd
    import streamlit as _st

    from ui.ui_discovery import render_discovery_sidebar

    lab = lambda a, b: [a, b, a]          # A and C share; B differs
    flag = lambda: [1, 0, 0]              # only A fires
    df = _pd.DataFrame({
        "name": ["A", "B", "C"],
        # ── 🏢 Universe
        "market_category":   lab("Mega Cap", "Nano Cap"),
        "vol_sma_20d":       [1e5, 1e3, 1e5],                 # × ₹1,000 = ₹10 / ₹0.1 / ₹10 Cr a day
        "close_price":       [1000.0, 1000.0, 1000.0],
        "sector":            lab("P", "Q"),
        "industry":          lab("P-ind", "Q-ind"),
        "cyclicality_tier":  lab("Defensive", "Cyclical"),
        "sector_capital_phase": lab("⚖️ Neutral", "🔥 Hot Capital (caution)"),
        # ── 🎯 Decision & Class
        "conviction_tier":   [1, 2, 1],
        "verdict_direction": lab("SOUND", "FLAWED"),
        "wealth_tier":       lab("BUY★", "AVOID"),
        "corporate_class":   lab("🏆 GREAT", "💀 GRUESOME"),
        "capital_allocation_signal": lab("💰 Returning Capital", "⚠️ Raising Capital"),
        # ── 🛡️ Safety
        "red_flag_count":    [0, 9, 0],
        "piotroski_fscore":  [8, 2, 8],                       # -> 💪 Strong / ⚠️ Weak / 💪 Strong
        "cash_machine_label": lab("💰 Cash Machine", "📄 Paper Profits"),
        "data_coverage_pct": [95.0, 10.0, 95.0],
        "result_stale_flag": [0, 1, 0],                       # Hide-stale drops B
        # ── 🧬 Frameworks
        "frameworks_passed": ["Diamond, QGLP", "Fisher Quality", "Diamond"],
        # ── 💰 Moat · Value · Entry
        "moat_growth_quad":  lab("⭐ Wealth Creator", "💀 Wealth Destroyer"),
        "peg_zone":          lab("💎 Deep Value", "🔴 Overpriced"),
        "buy_zone_label":    lab("🟢 Perfect Entry (Low Risk)", "🔻 Below Stop (Trend Broken)"),
        "negative_wc_flag":  flag(),
        # ── 📈 Trend · Style · Flow
        "weinstein_stage":   lab("📈 Stage 2 Advancing", "📉 Stage 4 Declining"),
        "lynch_category":    lab("Fast Grower", "Declining"),
        "mef_label":         lab("🟢 Expanding", "🔴 Degrading"),
        "cf_triangle":       lab("✅ Self-Funding", "🚨 Debt Trap — Avoid"),
        "smart_money_flow":  lab("🌊💎 Elite Accumulation", "❌ Distribution"),
        # ── 🔥 Catalysts & Alerts · 🚀 Multibagger
        "ep_power_curve":    lab("🚀 Hockey Stick", "📉 Value Trap"),
        "earnings_power_box": lab("📦 Earnings Power", "⚠️ Weakest"),
        "cat_capacity":      flag(),
        "sell_alert_cash_collapse": flag(),
        "mosl_100x_candidate": flag(),
        # ── 🌊 Refine
        "gate_pass":         [1, 0, 1],
        "quality_score":     [90.0, 5.0, 90.0],
        "composite_score":   [90.0, 5.0, 90.0],
    })
    filt = render_discovery_sidebar(df)
    _st.text(f"KEPT={sorted(filt['name'].tolist())}")


# (session_state to set, expected survivors). Each case sets exactly ONE filter; every other
# control sits at its show-all default, so a case measures that filter alone.
_FILTER_CASES = [
    ("sb_mcap",        {"sb_mcap": ["Mega Cap"]},                    ["A", "C"]),
    ("sb_liq",         {"sb_liq": 1.0},                              ["A", "C"]),
    ("sb_sector",      {"sb_sector": "P"},                           ["A", "C"]),
    ("sb_industry",    {"sb_industry": "Q-ind"},                     ["B"]),
    ("sb_cyc",         {"sb_cyc": ["Defensive"]},                    ["A", "C"]),
    ("sb_capphase",    {"sb_capphase": ["🔥 Hot Capital (caution)"]}, ["B"]),
    ("sb_tier",        {"sb_tier": [1]},                             ["A", "C"]),
    ("sb_verdict",     {"sb_verdict": ["SOUND"]},                    ["A", "C"]),
    ("sb_wealthtier",  {"sb_wealthtier": ["AVOID"]},                 ["B"]),
    ("sb_corpclass",   {"sb_corpclass": ["🏆 GREAT"]},               ["A", "C"]),
    ("sb_capalloc",    {"sb_capalloc": ["💰 Returning Capital"]},    ["A", "C"]),
    ("sb_maxrf",       {"sb_maxrf": 2},                              ["A", "C"]),
    ("sb_piotier",     {"sb_piotier": ["💪 Strong (≥7)"]},           ["A", "C"]),
    ("sb_cashmach",    {"sb_cashmach": ["📄 Paper Profits"]},        ["B"]),
    ("sb_mincov",      {"sb_mincov": 50},                            ["A", "C"]),
    ("sb_hidestale",   {"sb_hidestale": True},                       ["A", "C"]),
    ("sb_fw_include",  {"sb_fw_include": ["QGLP"]},                  ["A"]),
    ("sb_fw_exclude",  {"sb_fw_exclude": ["Diamond"]},               ["B"]),
    ("sb_fw_combine",  {"sb_fw_combine": ["Diamond", "QGLP"]},       ["A"]),
    ("sb_fwfam",       {"sb_fwfam": ["Fisher"]},                     ["B"]),
    ("sb_moat",        {"sb_moat": ["⭐ Wealth Creator"]},           ["A", "C"]),
    ("sb_peg_zone",    {"sb_peg_zone": ["🔴 Overpriced"]},           ["B"]),
    ("sb_buy_zone",    {"sb_buy_zone": ["🟢 Perfect Entry (Low Risk)"]}, ["A", "C"]),
    ("sb_floatfunded", {"sb_floatfunded": True},                     ["A"]),
    ("sb_weinstein",   {"sb_weinstein": ["📈 Stage 2 Advancing"]},   ["A", "C"]),
    ("sb_lynchcat",    {"sb_lynchcat": ["Declining"]},               ["B"]),
    ("sb_mef",         {"sb_mef": ["🟢 Expanding"]},                 ["A", "C"]),
    ("sb_cftri",       {"sb_cftri": ["🚨 Debt Trap — Avoid"]},       ["B"]),
    ("sb_smartflow",   {"sb_smartflow": ["🌊💎 Elite Accumulation"]}, ["A", "C"]),
    ("sb_eppc",        {"sb_eppc": ["🚀 Hockey Stick"]},             ["A", "C"]),
    ("sb_epbox",       {"sb_epbox": ["⚠️ Weakest"]},                 ["B"]),
    ("sb_catalyst",    {"sb_catalyst": ["cat_capacity"]},            ["A"]),
    ("sb_sellalert",   {"sb_sellalert": ["sell_alert_cash_collapse"]}, ["A"]),
    ("sb_mbsetup",     {"sb_mbsetup": ["mosl_100x_candidate"]},      ["A"]),
    ("sb_gate",        {"sb_gate": True},                            ["A", "C"]),
    ("sb_minq",        {"sb_minq": 50},                              ["A", "C"]),
    ("sb_minscore",    {"sb_minscore": 50},                          ["A", "C"]),
]


@pytest.mark.parametrize("key,state,expected", _FILTER_CASES, ids=[c[0] for c in _FILTER_CASES])
def test_every_sidebar_filter_actually_filters(key, state, expected):
    """One case per filter, driven through the REAL widget machinery. Catches a filter that is
    rendered but never applied, one whose mask is inverted, and one that honours only its first
    pick — none of which a source-text scan can see."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_all_filters_app)
    for k, v in state.items():
        at.session_state[k] = v
    at.run(timeout=30)
    assert not at.exception, f"{key} raised: {at.exception}"
    kept = next(t.value for t in at.text if t.value.startswith("KEPT="))
    assert kept == f"KEPT={expected}", f"{key}: expected {expected}, got {kept.split('=', 1)[1]}"


def test_the_filter_case_table_covers_every_registered_filter():
    """Teeth: the table above must not silently fall behind the sidebar. Every chip-registered
    key needs a case, or the next filter ships unprotected exactly as these 27 did."""
    from ui.ui_discovery import _CHIP_META
    covered = {k for k, _s, _e in _FILTER_CASES}
    registered = {k for k, _h, _kind in _CHIP_META}
    missing = sorted(registered - covered)
    assert not missing, f"registered filters with no behavioural case: {missing}"
    assert not sorted(covered - registered), "a case names a key that is not a registered filter"


def test_the_rejected_candidates_stay_out():
    """Two candidates from the same sweep were REJECTED by measurement and must not drift in.
    dilution_vampire_flag: P(rf>=3 | flag) = 99% — reachable by lowering Max red flags, and an
    exclusion-only use case. cyclical_mirage_flag: 91%, same. Their absence is a decision, not an
    oversight (the Safety group's own comment recorded the original census)."""
    for col in ("dilution_vampire_flag", "cyclical_mirage_flag"):
        assert f'_cf["{col}"]' not in _DISC, (
            f"{col} became a filter, but it is subsumed by red_flag_count>=3 — re-measure first")


def test_all_filter_sites_route_through_choke_point():
    """Static pin: NO filter application may bypass the _narrow choke point (`_cf = _cf[` == 0),
    and the funnel's zero-state must actually read the culprit log."""
    src = _ui_discovery_source()
    n_direct = src.count("_cf = _cf[")
    assert n_direct == 0, (
        f"{n_direct} filter site(s) bypass the choke point — every application must be "
        "`_cf = _narrow(_cf, <mask>, <label>)` so the funnel can name a zero-results culprit"
    )
    assert "_first_zero_filter" in src, "choke-point helper not wired"
    assert src.count("_zero_at") >= 2, "funnel zero-state does not read the culprit log"


def test_every_non_multiselect_widget_seeds_its_key_before_instantiating():
    """THE STEEL REPRO (2026-08-29, user-reported the day after the on_click fix): deleting a
    widget key does NOT clear the widget's frontend state. A selectbox/slider/checkbox that
    renders with no session value asks the FRONTEND, which still remembers the old pick — so a
    cleared Sector=Steel resurrected on the next rerun. _ms_cascade multiselects were always
    immune because they unconditionally re-seed their key pre-instantiation; this pins the same
    discipline onto every non-multiselect sb_ widget, which is what makes delete-based Clear-All
    authoritative for ALL widget kinds. AppTest cannot catch a regression here (no frontend to
    resurrect from), so the seed is pinned structurally."""
    # 2026-09-02: the two selectboxes seed with setdefault like every other non-multiselect —
    # the old `not in st.session_state or not in <opts>` form ALSO reset a stored pick to "All"
    # whenever the cascade narrowed it out (the selectbox form of the silent widening).
    for key, seed in [
        ("sb_liq",       'st.session_state.setdefault("sb_liq"'),
        ("sb_sector",    'st.session_state.setdefault("sb_sector"'),
        ("sb_industry",  'st.session_state.setdefault("sb_industry"'),
        ("sb_maxrf",     'st.session_state.setdefault("sb_maxrf"'),
        ("sb_mincov",    'st.session_state.setdefault("sb_mincov"'),
        ("sb_hidestale", 'st.session_state.setdefault("sb_hidestale"'),
        ("sb_gate",      'st.session_state.setdefault("sb_gate"'),
        ("sb_minq",      'st.session_state.setdefault("sb_minq"'),
        ("sb_minscore",  'st.session_state.setdefault("sb_minscore"'),
        ("sb_fwfam_min", '"sb_fwfam_min" not in st.session_state'),
    ]:
        assert seed in _DISC, (
            f"{key} lost its seed-before-instantiate guard — a cleared value will resurrect "
            f"from frontend widget state (the Steel repro)"
        )
        # the seed must appear BEFORE the widget that uses the key
        assert _DISC.index(seed) < _DISC.index(f'key="{key}"'), (
            f"{key}'s seed sits after its widget — it must run before instantiation"
        )


def test_cash_machine_filter_labels_match_the_engine():
    """The 34th filter (2026-08-29): cash_machine_label options must match the emitter verbatim
    (data_engine:1105) — the same cross-file pin the capital-phase filter carries."""
    for lbl in ["💰 Cash Machine", "✅ Solid", "📄 Paper Profits"]:
        assert lbl in _DISC, f"cash-machine label missing from ui_discovery: {lbl!r}"
        assert lbl in _DENG, f"cash-machine label drifted in data_engine (emitter): {lbl!r}"


# ── THE LABEL ROUND-TRIP (2026-09-21) ──────────────────────────────────────────────────────
# THE USER FOUND THIS AND THE ORDER MATTERED: pick Sector = Steel FIRST, then add Market
# Category = Small Cap + Mid Cap, and the Deep Scanner read "No stocks match your filters" with
# the funnel blaming Sector — while 60 Steel small/mid-caps sat in the frame. Picking the market
# caps FIRST worked fine.
#
# MEASURED AT THE BOUNDARY with a temporary probe in the live browser (AppTest cannot see this
# class at all — it assigns session_state directly and never exercises the widget registry, and
# it returned the correct 60 for BOTH orders):
#
#     STORED_BEFORE_WIDGET = 'Steel  ·  68'      <- already corrupt before the widget rendered
#     opts_tail            = [..., 'Steel  ·  68']
#     steel_rows_in_frame  = 39   ->   rows_after_sector = 0
#
# Nothing in the app writes sb_sector except the widget itself, so Streamlit round-tripped the
# selectbox's FORMATTED LABEL back into session_state when the cascade changed that label between
# reruns (Steel · 68 -> Steel · 29 -> Steel · 39). keep_selected then APPENDED the corrupt string
# as a legitimate option — so the widget never raised — and the filter compared a label against raw
# sector values and matched nothing. The safety net that exists to prevent silent WIDENING was
# quietly preserving a silent ZEROING.
#
# The multiselects were immune (sel_mcap stayed ['Small Cap', 'Mid Cap']); this is a selectbox-only
# failure, which is why only Sector and Industry carry the recovery.

def test_canonical_pick_returns_a_real_value_untouched():
    from ui.ui_discovery import canonical_pick
    dom = {"All", "Steel", "Trading"}
    assert canonical_pick("Steel", dom) == "Steel"
    assert canonical_pick("All", dom) == "All"


def test_canonical_pick_recovers_a_formatted_label():
    """The exact corruption measured in the browser."""
    from ui.ui_discovery import canonical_pick
    dom = {"All", "Steel"}
    assert canonical_pick("Steel  ·  68", dom) == "Steel", (
        "the round-tripped label must resolve back to the raw sector value"
    )


def test_canonical_pick_keeps_a_value_the_cascade_narrowed_out():
    """THE REGRESSION THIS MUST NOT CAUSE. keep_selected exists because pruning a stored pick
    switches the filter OFF and shows MORE stocks (measured 2026-09-02). So recovery validates
    against the FULL domain, never the narrowed cascade frame: a real sector with zero rows under
    the current filters stays selected, reads `· 0`, applies, and names itself as culprit."""
    from ui.ui_discovery import canonical_pick
    full_domain = {"All", "Steel", "Air Transport Service"}
    assert canonical_pick("Air Transport Service", full_domain) == "Air Transport Service"


def test_a_real_value_containing_the_separator_is_never_split():
    """FOUND BY A MISSED MUTATION. Dropping the `if stored in domain: return stored` early-return
    is INERT for every ordinary name — "Air Transport Service" has no separator, so splitting it
    returns itself — so nothing failed. The guard only shows its teeth on a legitimate value that
    happens to contain the separator, which is exactly the case that would be silently mangled.
    """
    from ui.ui_discovery import canonical_pick, _PICK_SEP
    odd = f"Iron{_PICK_SEP}Steel"          # a real domain value that looks like a label
    assert canonical_pick(odd, {"All", odd}) == odd, (
        "a value present in the domain must be returned untouched — splitting it first would "
        "mangle any name containing the separator"
    )


def test_canonical_pick_falls_back_when_the_value_is_unrecoverable():
    from ui.ui_discovery import canonical_pick
    assert canonical_pick("Nonexistent Sector", {"All", "Steel"}) == "All"
    assert canonical_pick(None, {"All", "Steel"}) == "All"


def test_the_formatter_and_the_recoverer_share_one_separator():
    """A silent drift here re-opens the bug: change the format_func's separator alone and the
    recoverer stops splitting on it, so the label sails through as a value again."""
    from ui.ui_discovery import _PICK_SEP, _fmt_pick, canonical_pick
    label = _fmt_pick("Steel", 68)
    assert _PICK_SEP in label
    assert canonical_pick(label, {"All", "Steel"}) == "Steel", (
        "the recoverer cannot split what the formatter produced — they have drifted apart"
    )


def test_both_cascade_selectboxes_recover_their_stored_value():
    """Structural, because the failure is invisible to AppTest. Sector and Industry are the only
    selectboxes in the cascade and both showed the same corruption shape."""
    src = _DISC
    for key in ("sb_sector", "sb_industry"):
        i = src.index(f'st.session_state.setdefault("{key}"')
        seg = src[i:i + 700]
        assert "canonical_pick(" in seg, (
            f"{key} does not recover its stored value — a round-tripped label would be appended by "
            f"keep_selected as a legitimate option and silently match zero rows"
        )


def test_the_recovery_validates_against_the_full_frame_not_the_cascade_frame():
    """If recovery checked `_cf` (already narrowed), a legitimately narrowed-out pick would reset
    to All — the silent WIDENING keep_selected was built to stop. It must read the unfiltered df."""
    src = _DISC
    for key in ("sb_sector", "sb_industry"):
        col = "sector" if key == "sb_sector" else "industry"
        i = src.index(f'st.session_state.setdefault("{key}"')
        # Wide enough to span the recovery block, its provenance comment and the options line
        # below it — a window that stops short would pass on the recovery alone and never check
        # that the OPTIONS still come from the narrowed frame, which is the other half of the rule.
        seg = src[i:i + 1400]
        assert f'df["{col}"]' in seg, (
            f"{key}'s recovery domain is not built from the FULL df — validating against the "
            f"narrowed cascade frame would silently widen the filter"
        )
        assert f'_cf["{col}"].dropna().unique().tolist()' in seg, (
            f"{key} must still build its OPTIONS from the narrowed cascade frame"
        )


# ── HARDENING THE WHOLE CLASS (2026-09-21) ─────────────────────────────────────────────────
# The 2026-09-21 label round-trip was fixed on the TWO selectboxes because a probe showed the
# multiselects clean in ONE run. That is one observation, not a proof: the bug needed a specific
# ORDER to appear, so another order could corrupt a different widget. Measured, 19 of the 28
# cascade widgets carry a count-bearing label — the exact shape that failed — and only 2 were
# defended.
#
# The answer is NOT more reachability tests: 31 of those already exist, they all stayed green
# while the bug was live, and they test the LOGIC, which was never wrong. Instead:
#   1. recover at ONE choke point for every count-bearing widget, so it does not matter which
#      widget Streamlit corrupts or in what order;
#   2. pin the separator invariant that makes the recovery provably unambiguous;
#   3. INJECT the corruption in AppTest — it cannot PRODUCE this bug (no widget registry) but it
#      can REPRODUCE it, which is the regression test the first fix lacked;
#   4. check the sidebar against an independent pandas intersection (an oracle, not a restatement).


def _ROUNDTRIP_FRAME():
    """A frame where several filters compose non-trivially, so an oracle over it means something."""
    import pandas as _pd
    return _pd.DataFrame({
        "name":             list("abcdefgh"),
        "sector":           ["Steel", "Steel", "Steel", "Trading",
                             "Trading", "Chemicals", "Chemicals", "Steel"],
        "industry":         ["Tubes", "Wires", "Tubes", "Retail",
                             "Retail", "Dyes", "Dyes", "Wires"],
        "market_category":  ["Small Cap", "Mid Cap", "Large Cap", "Small Cap",
                             "Mid Cap", "Small Cap", "Large Cap", "Mid Cap"],
        "conviction_tier":  [1, 2, 3, 4, 5, 1, 2, 3],
        "piotroski_fscore": [5, 6, 7, 8, 5, 6, 7, 8],
        "red_flag_count":   [0, 1, 2, 3, 0, 1, 2, 3],
        "quality_score":    [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0],
        "composite_score":  [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0],
    })


def _roundtrip_app():
    """Mini-app: drives the REAL sidebar and reports what the caller gets.

    The frame arrives through session_state because AppTest.from_function runs the callable in an
    ISOLATED namespace — module-level helpers are invisible to it (NameError on the first run).
    Passing the spec keeps ONE definition of the fixture instead of a second inline copy that
    could drift away from the oracle's pandas side.
    """
    import pandas as _pd
    import streamlit as _st
    from ui.ui_discovery import render_discovery_sidebar
    filt = render_discovery_sidebar(_pd.DataFrame(_st.session_state["_FRAME_SPEC"]))
    _st.text("N=%d" % len(filt))
    _st.text("SECTORS=%s" % sorted(set(filt["sector"])))
    _st.text("CAPS=%s" % sorted(set(filt["market_category"])))


def _run_app(state, frame=None):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_roundtrip_app)
    at.session_state["_FRAME_SPEC"] = {c: list(v) for c, v
                                       in (_ROUNDTRIP_FRAME() if frame is None else frame).items()}
    for k, v in state.items():
        at.session_state[k] = v
    at.run(timeout=30)          # explicit: the 3s default overran here on 2026-08-31
    assert not at.exception, "sidebar raised: %s" % at.exception
    return {t.value.split("=", 1)[0]: t.value.split("=", 1)[1] for t in at.text}


# ── 1. the separator invariant that makes recovery unambiguous ──────────────────────────

def test_no_filterable_value_contains_the_label_separator():
    """RECOVERY SPLITS ON `_PICK_SEP`, so it is only provably correct while no real value contains
    it. Measured 2026-09-21 across all 21 filterable columns on live data: 0 violations, and the
    middot does not appear in any value at all. Pinned rather than assumed — the day a vendor ships
    a sector named with a middot, this fails instead of a filter silently matching the wrong rows."""
    import contextlib as _c, io as _i
    from ui.ui_discovery import _PICK_SEP
    import re as _re
    cols = sorted(set(_re.findall(r'count_col="(\w+)"', _DISC))) + ["sector", "industry"]
    with _c.redirect_stdout(_i.StringIO()):
        from core import fetch_and_clean_data, run_scoring_pipeline
        live = run_scoring_pipeline(fetch_and_clean_data("local"))
    present = [c for c in sorted(set(cols)) if c in live.columns]
    assert len(present) >= 15, "expected the full filterable column set, found %d" % len(present)
    bad = {c: [v for v in live[c].dropna().astype(str).unique() if _PICK_SEP in v]
           for c in present}
    bad = {c: v for c, v in bad.items() if v}
    assert not bad, (
        "these filterable values contain the option-label separator, so canonical_pick could "
        "split a REAL value: %s" % bad
    )


# ── 2. the list form, for multiselects ──────────────────────────────────────────────────

def test_canonical_picks_recovers_every_label_in_a_list():
    from ui.ui_discovery import canonical_picks, _fmt_pick
    dom = {"Small Cap", "Mid Cap"}
    got = canonical_picks([_fmt_pick("Small Cap", 1048), _fmt_pick("Mid Cap", 427)], dom)
    assert got == ["Small Cap", "Mid Cap"], got


def test_canonical_picks_drops_what_it_cannot_recover_without_emptying_the_rest():
    """A multiselect has no 'All' sentinel, so an unrecoverable entry is DROPPED, not turned into
    a fallback — and it must not take its valid siblings with it."""
    from ui.ui_discovery import canonical_picks
    assert canonical_picks(["Small Cap", "Nonsense"], {"Small Cap"}) == ["Small Cap"]
    assert canonical_picks([], {"Small Cap"}) == []


def test_canonical_picks_preserves_order_and_deduplicates():
    from ui.ui_discovery import canonical_picks, _fmt_pick
    dom = {"A", "B"}
    assert canonical_picks(["B", "A", _fmt_pick("B", 9)], dom) == ["B", "A"], (
        "recovery must not reorder a selection, and a label whose raw value is already selected "
        "must not be added twice"
    )


def test_every_count_bearing_widget_recovers_its_stored_value():
    """ONE CHOKE POINT. 19 of 28 cascade widgets carry a count-bearing label; defending only the
    two that were observed to fail would leave 17 relying on a single lucky observation."""
    i = _DISC.index("def _ms_cascade(")
    body = _DISC[i:_DISC.index("_cf = df ", i)]
    assert "canonical_picks(" in body, (
        "_ms_cascade does not recover its stored selection — every count_col multiselect is "
        "exposed to the same label round-trip that hit the Sector selectbox"
    )
    assert "count_col" in body.split("canonical_picks(")[0][-400:], (
        "recovery must be tied to the count-bearing case (count_col is what makes the label "
        "volatile, and it also names the column the domain comes from)"
    )


# ── 3. INJECTION: AppTest cannot cause this bug, but it can reproduce it ─────────────────

def test_a_round_tripped_selectbox_label_still_filters_correctly():
    """THE REGRESSION TEST THE FIRST FIX LACKED. Streamlit writes the DISPLAY string back into
    session_state; AppTest never does that on its own, so inject it directly. Before the fix this
    returns 0 with the funnel blaming Sector."""
    out = _run_app({"sb_sector": "Steel  \u00b7  99"})
    assert out["N"] == "4", "a round-tripped sector label matched %s rows, expected all 4 Steel" % out["N"]
    assert out["SECTORS"] == "['Steel']", out["SECTORS"]


def test_a_round_tripped_multiselect_label_still_filters_correctly():
    """The half that was never proven — only observed clean once, in one order."""
    out = _run_app({"sb_mcap": ["Small Cap  \u00b7  42"]})
    assert out["N"] == "3", "a round-tripped cap label matched %s rows, expected 3 Small Caps" % out["N"]
    assert out["CAPS"] == "['Small Cap']", out["CAPS"]


def test_the_users_exact_repro_survives_both_labels_being_round_tripped():
    """Sector = Steel + Market Category = [Small, Mid], both stored as labels. The live shape of
    the 2026-09-21 report, which read 'No stocks match your filters' over 60 real rows."""
    out = _run_app({"sb_sector": "Steel  \u00b7  99",
                    "sb_mcap": ["Small Cap  \u00b7  42", "Mid Cap  \u00b7  7"]})
    assert out["N"] == "3", (
        "the user's exact repro returned %s rows; the frame holds 3 Steel small/mid-caps" % out["N"]
    )
    assert out["SECTORS"] == "['Steel']", out["SECTORS"]


def test_the_unknown_sentinel_survives_the_recovery():
    """FOUND BY A MISSED MUTATION (U3). `❔ Unknown` is a legitimate pick that appears in NO
    column — it claims the honest holes (NaN/blank) that _label_mask unions in. So a domain built
    from column values ALONE silently drops it, the filter switches off, and the result gets BIGGER
    rather than smaller: the silent-widening class keep_selected exists to prevent.

    Unit-testing canonical_picks would not catch it — the defect is in how _ms_cascade BUILDS the
    domain, so this drives the real widget with a frame that has an honest hole in it.
    """
    from ui.ui_discovery import _UNKNOWN
    f = _ROUNDTRIP_FRAME()
    f.loc[0, "market_category"] = None                     # exactly one honest hole
    out = _run_app({"sb_mcap": [_UNKNOWN]}, frame=f)
    assert out["N"] == "1", (
        "selecting ❔ Unknown matched %s rows, expected the 1 blank — a dropped sentinel would "
        "show the whole frame instead" % out["N"]
    )


# ── 3b. THE format_func WIDGETS — the half the 2026-09-21 fix missed (a PRODUCTION CRASH) ──
# Reported from Streamlit Cloud 2026-09-22: `KeyError` at ui_discovery `_cat_name[c]`, inside the
# 🔥 Catalyst format_func, which takes down render_discovery_sidebar and with it THE WHOLE APP.
#
# The 2026-09-21 commit was titled "close the label round-trip for the whole cascade, not two
# widgets", and it did not: recovery was keyed on `count_col`, but EIGHT widgets carry their live
# count through an explicit `format_func` instead, so none of them was ever covered. For four of
# them (catalyst, sell alert, candidate flags, framework family) the format_func does a HARD dict
# lookup, so a round-tripped display label is not a silent zero — it is a crash. The other four
# (`.get`-based) silently zero or, for Exclude, silently WIDEN. Reproduced by injection before any
# fix was written: all four hard-lookup widgets raised; all four .get widgets returned wrong rows.
#
# The assertion is deliberately the STRONG one: a round-tripped label must filter EXACTLY like the
# value it came from. "Does not crash" alone would pass a fix that drops the pick — which switches
# the filter off and widens the result, the 2026-09-02 bug keep_selected exists to prevent.

def _fam_label(family, count):
    """The framework-family label in its real shape, with config's own emoji."""
    emoji = next(e for (e, lbl, _c, _f) in FRAMEWORK_CATEGORIES if lbl == family)
    return f"{emoji} {family} ({count})"


# (widget key, the legitimate stored value, the DISPLAY label Streamlit can write back — built in
# the widget's real label shape with a deliberately STALE count, since the count is what moves)
_FORMAT_FUNC_CASES = [
    ("sb_catalyst",   ["cat_capacity"],             ["🔥 Capacity Explosion (99)"]),
    ("sb_sellalert",  ["sell_alert_cash_collapse"], ["🚨 Cash Collapse (99)"]),
    ("sb_mbsetup",    ["mosl_100x_candidate"],      ["🐘 100x Candidate (99)"]),
    ("sb_fwfam",      ["Fisher"],                   [_fam_label("Fisher", 99)]),
    ("sb_fw_include", ["QGLP"],                     ["QGLP  ·  99"]),
    ("sb_fw_exclude", ["Diamond"],                  ["Diamond  ·  99"]),
    ("sb_fw_combine", ["Diamond", "QGLP"],          ["Diamond  ·  99", "QGLP  ·  99"]),
    ("sb_piotier",    ["💪 Strong (≥7)"],            ["💪 Strong (≥7)  ·  99"]),
]


@pytest.mark.parametrize("key,good,label", _FORMAT_FUNC_CASES, ids=[c[0] for c in _FORMAT_FUNC_CASES])
def test_a_round_tripped_format_func_label_filters_exactly_like_its_value(key, good, label):
    from streamlit.testing.v1 import AppTest

    def kept(stored):
        at = AppTest.from_function(_all_filters_app)
        at.session_state[key] = stored
        at.run(timeout=30)
        assert not at.exception, (
            f"{key} CRASHED the whole sidebar on a round-tripped label {stored!r} — the production "
            f"KeyError of 2026-09-22: {at.exception}"
        )
        return next(t.value for t in at.text if t.value.startswith("KEPT="))

    want = kept(good)
    # vacuity guard: the valid pick must actually narrow, or 'same rows' proves nothing
    assert want != "KEPT=['A', 'B', 'C']", f"{key}: the control pick does not narrow the frame"
    got = kept(label)
    assert got == want, (
        f"{key}: a round-tripped label {label!r} kept {got.split('=', 1)[1]} but the value it came "
        f"from keeps {want.split('=', 1)[1]} — the pick was dropped or mis-recovered, so the filter "
        f"silently changed meaning"
    )


def test_recover_labels_is_order_preserving_deduplicating_and_drops_only_the_unrecoverable():
    """The pure function, pinned on its four properties. Both label shapes must reverse; a
    legitimate value must pass untouched (including one whose own text ends in brackets); a label
    whose value is already selected must not duplicate it; and an entry that maps to nothing is
    dropped WITHOUT taking its valid siblings with it."""
    from ui.ui_discovery import recover_labels, _fmt_pick
    names = {"cat_a": "🔥 Alpha", "cat_b": "🔥 Beta"}
    paren = lambda c: f"{names[c]} ({7})"                       # the flag OR-group shape
    assert recover_labels(["🔥 Beta (99)", "🔥 Alpha (1)"], list(names), paren) == ["cat_b", "cat_a"]
    assert recover_labels(["cat_a", "🔥 Alpha (3)"], list(names), paren) == ["cat_a"]   # no dup
    assert recover_labels(["garbage (1)", "cat_b"], list(names), paren) == ["cat_b"]    # siblings kept
    assert recover_labels([], list(names), paren) == []
    tiers = ["💪 Strong (≥7)", "⚠️ Weak (≤3)"]
    dot = lambda v: _fmt_pick(v, 5)                              # the _fmt_pick shape
    assert recover_labels([_fmt_pick("⚠️ Weak (≤3)", 99)], tiers, dot) == ["⚠️ Weak (≤3)"]
    assert recover_labels(["💪 Strong (≥7)"], tiers, dot) == ["💪 Strong (≥7)"]          # untouched


@pytest.mark.parametrize("key,label", [
    ("sb_catalyst",   ["🔥 Capacity Explosion (99)"]),
    ("sb_fw_include", ["QGLP  ·  99"]),
], ids=["sb_catalyst", "sb_fw_include"])
def test_a_round_tripped_label_survives_being_narrowed_out_upstream(key, label):
    """THE FULL-FRAME RULE, for format_func widgets. Wealth Tier = AVOID (upstream) keeps only row
    B, which has neither cat_capacity nor QGLP — so both values are narrowed OUT of the live
    options by the time their widget renders. The honest answer is 0 rows with the filter still
    applied. A recovery domain built from the narrowed frame (`_cf`) cannot recognise the label,
    drops the pick, switches the filter off and returns row B: the 2026-09-02 widening bug.
    The single-filter cases above cannot see this — nothing upstream narrows there."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_all_filters_app)
    at.session_state["sb_wealthtier"] = ["AVOID"]
    at.session_state[key] = label
    at.run(timeout=30)
    assert not at.exception, f"{key} raised: {at.exception}"
    kept = next(t.value for t in at.text if t.value.startswith("KEPT="))
    assert kept == "KEPT=[]", (
        f"{key}: a round-tripped label narrowed out upstream kept {kept.split('=', 1)[1]}; the "
        f"filter must still apply and read 0 — a dropped pick widens the result instead"
    )


def test_every_format_func_widget_passes_its_recovery_domain():
    """THE STRUCTURAL HALF, so the next widget cannot repeat the gap. Recovery for a format_func
    widget needs the FULL set of legitimate values (the same full-frame rule as count_col — a value
    the cascade narrowed out must survive), and only the call site knows it. AST, not text: a
    keyword inside a comment or an `if False:` branch must not satisfy this."""
    import ast
    calls, bad = 0, []
    for node in ast.walk(ast.parse(_DISC)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_ms_cascade":
            kws = {k.arg for k in node.keywords}
            if "format_func" in kws:
                calls += 1
                if "domain" not in kws and "count_col" not in kws:
                    bad.append(node.lineno)
    assert calls >= 7, f"teeth: expected >= 7 format_func cascade widgets, found {calls}"
    assert not bad, (
        f"_ms_cascade widgets at lines {bad} carry a live count via format_func but pass no "
        f"recovery `domain=` — a round-tripped label will crash or silently re-filter them"
    )


# ── 4. ORACLE: the sidebar against an independent intersection ───────────────────────────

def test_the_sidebar_equals_an_independent_pandas_intersection():
    """AN ORACLE, NOT A RESTATEMENT. The sidebar's whole contract is
    `filt == df[mask_a & mask_b & ...]`, so the honest check is to compute that separately and
    demand agreement. Catches any filter that silently matches nothing or too much, whatever the
    cause — a dead guard, an inverted mask, or a corrupted stored value."""
    frame = _ROUNDTRIP_FRAME()
    cases = [
        ({}, lambda d: d),
        ({"sb_sector": "Steel"}, lambda d: d[d["sector"] == "Steel"]),
        ({"sb_mcap": ["Small Cap"]}, lambda d: d[d["market_category"].isin(["Small Cap"])]),
        ({"sb_mcap": ["Small Cap", "Mid Cap"]},
         lambda d: d[d["market_category"].isin(["Small Cap", "Mid Cap"])]),
        ({"sb_sector": "Steel", "sb_mcap": ["Small Cap", "Mid Cap"]},
         lambda d: d[(d["sector"] == "Steel") & d["market_category"].isin(["Small Cap", "Mid Cap"])]),
        ({"sb_sector": "Chemicals", "sb_mcap": ["Large Cap"]},
         lambda d: d[(d["sector"] == "Chemicals") & d["market_category"].isin(["Large Cap"])]),
        ({"sb_sector": "Steel", "sb_industry": "Wires"},
         lambda d: d[(d["sector"] == "Steel") & (d["industry"] == "Wires")]),
    ]
    for state, oracle in cases:
        want = len(oracle(frame))
        got = int(_run_app(state)["N"])
        assert got == want, (
            "sidebar and pandas disagree for %r: sidebar %d, intersection %d" % (state, got, want)
        )


def test_the_oracle_is_not_vacuous():
    """Teeth: the cases above must actually discriminate, or the oracle passes on anything."""
    frame = _ROUNDTRIP_FRAME()
    counts = {
        len(frame),
        len(frame[frame["sector"] == "Steel"]),
        len(frame[frame["market_category"].isin(["Small Cap"])]),
        len(frame[(frame["sector"] == "Steel") & (frame["industry"] == "Wires")]),
    }
    assert len(counts) >= 4, "the oracle cases collapse to %s — they cannot discriminate" % counts
