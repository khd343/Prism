"""Contract: the Market Pulse inner-tab set, pinned so a tab cannot be re-added — or a removed
renderer re-wired — without a conscious change.

STAGE 3 (2026-06-18) cut the set to {Tsunami, QGLP, Sectors}, dropping 💙 Blue Chips and
🚀 Tipping Points (brittle; folded into an enhanced Sectors view).
Those two stay out, and the renderer checks below are what keep them out.

💙 BLUE CHIPS — REASON CORRECTED 2026-09-10. Stage 3 removed it as DEAD ("fired on 0% of the
universe"). That claim is now measurably FALSE: on the 2,717-stock universe `bruised_blue_chip` fires
47 times (1.7%) — alive, and in the census's healthy band. The tab still stays out, for the reason
that IS still true: it is REDUNDANT THREE WAYS — (1) the Discovery sidebar's Candidate Flags filter
offers 💎 Bruised Blue Chip directly, (2) Framework 7 renders the same gate on every tearsheet, and
(3) the Movers multibagger group already surfaces stocks entering the state. A whole inner tab to
re-list 47 names three other surfaces already reach is a fourth door onto one room. Keeping the dead
justification would have hidden a live signal behind a stale number — the record is corrected here so
a future reader re-litigates the RIGHT question (redundancy), not a refuted one (liveness).

🔭 MOSL ADDED 2026-08-27, and this file did its job: the change failed here first and had to be
justified rather than slipped in. Neither Stage-3 removal reason applies to it —

    not dead      586 stocks clear 2+ of the 10 Wealth Creation lenses; a clean pyramid from
                  999 at zero down to 1 stock at eight; deepest agreement 8 of 10
    not brittle   a VIEW over frameworks already implemented and audited, with no new gate and no
                  engine change; exact-token parsed and cross-checked against the authoritative
                  qglp_pass column (both 328)
    not redundant corr(convergence, composite_score) = 0.479 — it carries information the score
                  does not

💹 WEALTH ADDED 2026-08-27 (same session, later), and again this file forced the justification:
    not dead      every one of its six tiers is populated on live data (BUY★ 12% · BUY 5% ·
                  WATCH★ 9% · WATCH 32% · AVOID 34% · N/A 8%) — pinned by tests/test_wealth_tier.py
    not brittle   a pure READ of the wealth_* columns compute_verdict materializes; the tab holds
                  zero logic, so the tier a snapshot captures equals the tier displayed
    not redundant it answers the question no other surface does — "is it becoming more valuable?"
                  — and provably does not collapse into verdict_direction (the engine's complete
                  18-stock BUY list splits across four wealth tiers)

🏭 INDUSTRY ADDED 2026-08-28, and it clears the same three bars:
    not dead      76 industries hold ≥8 stocks (54 at ≥10) out of 355 — against 81 sectors, so the
                  view is 4.4× finer and still well populated at its default floor
    not brittle   a pure groupby over columns that already exist (industry, sector,
                  composite_score, gate_pass). No new gate, no engine change, nothing to calibrate
    not redundant THE STRONGEST CASE OF THE FOUR. Sector averaging destroys real dispersion: the
                  six sizeable industries inside Pharmaceuticals run 18.1 → 51.3 on average
                  composite — a 33-point spread the Sectors tab reports as ONE number (FMCG 22.9,
                  Auto Ancillaries 22.6). 20 of those 76 industries sit more than 5 points from
                  their parent sector's average, from Pharma - MNC bulk Drugs at +22.7 to Auto
                  Ancillaries - Gears at −12.9. The tab makes that gap its sort key.
                  Note it is NOT a drill-down: 136 of 355 industries span more than one sector, so
                  the hierarchy the name suggests does not exist — see tests/test_industry_tab.py.

The behaviour of the tabs themselves is pinned by tests/test_mosl_convergence_tab.py and
tests/test_wealth_tier.py. This file pins only the SET, which is the thing a future edit is most
likely to change carelessly.
"""
import ast
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent / "app.py"


def _mp_tab_labels():
    """The string-literal labels of the `_mp_tabs = st.tabs([...])` assignment."""
    tree = ast.parse(_APP.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if (isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_mp_tabs" for t in n.targets)
                and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "tabs"
                and n.value.args and isinstance(n.value.args[0], ast.List)):
            return [e.value for e in n.value.args[0].elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return None


def test_market_pulse_tab_set_is_exact():
    """Order matters as much as membership: every `with _mp_tabs[i]` body is bound by index, so a
    reordering silently renders the wrong content into the wrong tab."""
    labels = _mp_tab_labels()
    # 🔁 Movers APPENDED 2026-09-04 (index 6): what changed since the previous data vintage —
    # not dead (every column it diffs moves between refreshes), not brittle (a join + a diff,
    # zero thresholds, zero scoring, pure + stateless in ui/ui_movers.py), not redundant (no
    # surface shows change over time; Vel% is a financial delta, not a scoring delta).
    assert labels == ["🌊 Tsunami", "🏛️ QGLP", "🔭 MOSL", "💹 Wealth", "📈 Sectors",
                      "🏭 Industry", "🔁 Movers"], labels


def test_removed_renderers_are_not_called():
    src = _APP.read_text(encoding="utf-8")
    assert "render_bruised_blue_chips(" not in src, "Blue Chips list renderer must be gone"
    assert "render_multi_trillion_tipping_points(" not in src, "Tipping Points renderer must be gone"


def test_removed_renderers_are_not_imported():
    """No dangling import of the deleted list renderers (would crash app boot)."""
    src = _APP.read_text(encoding="utf-8")
    assert "render_bruised_blue_chips," not in src and "render_multi_trillion_tipping_points," not in src


def test_tab_extractor_has_teeth():
    labels = _mp_tab_labels()
    assert labels is not None and len(labels) == 7


def test_the_stage_3_removals_were_not_quietly_restored():
    """The two tabs Stage 3 deleted must stay deleted -- adding MOSL is not licence to bring back
    the redundant one or the brittle one that was folded into Sectors."""
    labels = _mp_tab_labels() or []
    joined = " ".join(labels)
    assert "Blue Chip" not in joined, (
        "💙 Blue Chips is redundant with Framework 7 + the sidebar's Candidate Flags filter; it stays out"
    )
    assert "Tipping" not in joined, "🚀 Tipping Points was folded into Sectors; it stays out"


# ── Fragment boundaries (2026-08-29) ─────────────────────────────────────────────────────────
def _app_src():
    import io as _io, os
    return _io.open(os.path.join(os.path.dirname(__file__), "..", "app.py"), encoding="utf-8").read()


def test_market_pulse_is_a_fragment_called_in_its_tab():
    """Market Pulse is market-wide by design (reads module `df` only — verified: zero filt/attrs
    reads, mp_* keys consumed nowhere else). Its in-tab controls cost 951 ms pre-fragment, ~97%
    of it re-rendering the OTHER tabs. The fragment scopes them to this tab."""
    src = _app_src()
    i = src.index("def _render_market_pulse():")
    deco = src[:i].rstrip().splitlines()[-1]
    assert deco.strip() == "@st.fragment", "Market Pulse lost its @st.fragment decorator"
    # Positions shifted by one on 2026-10-06 when 🎯 Scans became tab 2.
    assert "with tabs[4]:" + chr(10) + "    _render_market_pulse()" in src, (
        "the fragment is no longer called in its tab")


def test_reference_is_a_fragment_called_in_its_tab():
    src = _app_src()
    i = src.index("def _render_reference():")
    deco = src[:i].rstrip().splitlines()[-1]
    assert deco.strip() == "@st.fragment", "Reference lost its @st.fragment decorator"
    assert "with tabs[6]:" + chr(10) + "    _render_reference()" in src


def test_config_tab_is_never_fragmented():
    """THE CORRECTNESS TOMBSTONE. cfg_mode re-ranks the universe: the top of the script reads it
    to build the scored frame. Fragmented, an Analysis-Mode change would rerun only the fragment
    and every tab would show STALE RANKINGS. A 2026-08-29 audit proposed fragmenting Config as a
    speedup; rejected for exactly this reason — this pin keeps it rejected."""
    src = _app_src()
    cfg = src[src.index("# TAB 6: CONFIGURATION"):src.index("# TAB 7: REFERENCE")]
    assert "@st.fragment" not in cfg, (
        "Config was fragmented — cfg_mode changes will no longer recompute the scored frame"
    )
    assert "NEVER FRAGMENT THIS TAB" in cfg, "the tombstone comment explaining WHY is gone"
    assert 'key="cfg_mode"' in cfg, "cfg_mode moved out of Config — re-verify the fragment safety story"


# ── 🔁 Movers wiring (2026-09-04) — the rules app.py must keep, pinned where they live ───────
def _movers_block():
    src = _app_src()
    i = src.index("with _mp_tabs[6]:")
    return src, src[i:src.index("with tabs[4]:", i)]


def test_movers_is_the_appended_seventh_tab_and_stateless_where_it_should_be():
    """Body bound to index 6 (appended, never inserted); the diff + page live in ui_movers,
    every widget lives here."""
    src, blk = _movers_block()
    assert "from ui.ui_movers import" in blk and "render_movers(" in blk and "compute_movers(" in blk
    assert "def compute_movers" not in src and "def render_movers" not in src, (
        "the diff/page must live in ui/ui_movers.py, not be re-implemented in app.py")


def test_movers_previous_side_is_rescored_by_the_same_engine_behind_a_click():
    """SAME ENGINE BY CONSTRUCTION: the archived copy goes through _load_vintage with the engine
    hash, mode and profile of the live frame — never a stored score. And that ~1-minute re-score
    runs ONLY after the Compare button: Market Pulse is a fragment whose every inner-tab body
    renders on every run."""
    src, blk = _movers_block()
    # PHASED (2026-09-04): download+derive is cached on the copy alone, scoring on
    # copy+engine+mode+profile — so a profile switch re-scores without re-downloading, and the
    # st.status shows where the minute goes. The same-engine rule is unchanged.
    assert "_score_vintage(" in blk and "_mv_pick, _mv_engine(), analysis_mode, scoring_profile, _mv_clean)" in blk, (
        "the previous vintage is not re-scored with the live engine/mode/profile")
    assert "_mv_clean = _load_vintage_clean(_mv_pick)" in blk, "the archived copy no longer takes the live loader"
    assert blk.index('key="mp_mv_go"') < blk.index("_load_vintage_clean("), "the load is not gated behind the button"
    assert "_mv_ph.status(" in blk and blk.index("_mv_ph.status(") < blk.index("_load_vintage_clean("), (
        "the phased progress container is gone — a bare minute reads as a hang")
    for fn in ("def _load_vintage_clean(", "def _score_vintage("):
        i = src.index(fn)
        deco = src[src.rindex("@st.cache_data", 0, i):i]
        assert "max_entries" in deco, f"{fn} must be cached with a bound — each entry is a full frame"
    i = src.index("def _load_vintage_clean(")
    assert 'fetch_and_clean_data("sheet", None, copy_id)' in src[i:i + 900], "the copy must take the live loader path"
    j = src.index("def _score_vintage(")
    assert "run_scoring_pipeline(_clean, analysis_mode, scoring_profile)" in src[j:j + 1200], (
        "the archived copy must take the canonical pipeline with the same mode/profile")


def test_movers_hands_a_clicked_row_to_the_tearsheet():
    """THE LOOP. Tsunami, QGLP and Discovery all hand a clicked row to the Tear-Sheet; Movers named
    the candidates and could not pass them on, so the reader had to retype a name in another tab.
    render_movers RETURNS the pick (the module never touches session_state) and app.py stages the
    transient `_pending_xray` + reruns — never a direct xray_stock set, because this tab renders
    AFTER the Tear-Sheet selectbox. The change-guard is load-bearing: st.dataframe's selection
    persists across reruns, so an unguarded set+rerun loops forever."""
    _, blk = _movers_block()
    assert "_mv_picked = render_movers(" in blk, "the clicked stock is not captured from the page"
    guard = 'if _mv_picked and _mv_picked != st.session_state.get("xray_stock"):'
    assert guard in blk, "the change-guard is missing — an unguarded set+rerun is an infinite loop"
    stage = blk[blk.index(guard):]
    assert 'st.session_state["_pending_xray"] = _mv_picked' in stage, "the pick is never staged"
    assert "st.rerun()" in stage, "nothing reruns, so the Tear-Sheet never consumes the staged pick"
    assert 'st.session_state["xray_stock"] = _mv_picked' not in blk, (
        "a direct widget-key set raises set-after-instantiation from this tab")


def test_movers_reason_chips_use_the_shared_cascade_helper_and_the_button_relabels():
    """The reason chips on ⭐ What matters are the ONE Market Pulse multiselect dialect (_mp_ms:
    keep_selected rule, live counts = stocks per reason) and reach the stateless module through
    meta["reasons"] — never a second widget dialect, never session_state read inside ui_movers.
    And once a compare has run for the picked vintage the button says "↻ Re-compare", because a
    button that still says "Compare" invites a click that appears to do nothing."""
    _, blk = _movers_block()
    assert "_mv_rc = reason_counts(_mv_res)" in blk, "the chip counts must come from reason_counts (same tagging as material)"
    assert '_mp_ms(st.container(), "What matters — reasons", list(_mv_rc),' in blk, (
        "the reason chips are not the shared _mp_ms cascade helper")
    assert '"mp_mv_why",' in blk and '"reasons": _mv_why,' in blk, "the selection does not reach render_movers via meta"
    assert blk.index("_mv_rc = reason_counts(") < blk.index("_mv_picked = render_movers("), (
        "chips must render (and be read) before the page that filters by them")
    assert '_mv_done = st.session_state.get("mp_mv_loaded") == _mv_pick' in blk
    assert 'f"↻ Re-compare with {_mv_lab[_mv_pick]}" if _mv_done' in blk, "the button does not relabel after a compare"
    # Seen live: on the click run the button is instantiated BEFORE the click is known, so without
    # a rerun it still reads "Compare" over a page that has just compared.
    i = blk.index('if st.button(_mv_btn, key="mp_mv_go"):')
    click_body = blk[i:blk.index('if st.session_state.get("mp_mv_loaded") != _mv_pick', i)]
    assert 'st.session_state["mp_mv_loaded"] = _mv_pick' in click_body and "st.rerun()" in click_body, (
        "the click must stage the pick AND rerun, or the label lags one run behind")


def test_movers_setup_is_one_compact_row_and_the_payoff_is_above_the_fold():
    """THE FOLD, pinned structurally (a pixel cannot be unit-tested; the browser measured it).
    Measured live before the fix: seven controls and ~280 words above ⭐ What matters, which
    began ~900px down a 732px viewport. Now: a one-line caption (≤ 40 words); the archive id is
    READ from session_state before any widget renders, so layout never dictates data flow; the
    id box, picker and button share ONE st.columns row (the box moves into a ⚙️ popover once
    secrets configure it); the phased status lives in a placeholder that is EMPTIED on success
    with the elapsed time folded into the result header; the reason chips are registered with
    the lens row's 🧹 via extra_keys; and restrict runs only when the lens actually narrowed the
    frame, so chips alone can never stamp '(whole universe)' on an unrestricted page."""
    _, blk = _movers_block()
    cap = blk[blk.index("sec-cap'>") + len("sec-cap'>"):blk.index("</div>", blk.index("sec-cap'>"))]
    assert len(cap.split()) <= 40, f"the caption is a paragraph again ({len(cap.split())} words)"
    read = '_mv_id = str(st.session_state["mp_mv_index"]).strip()'
    assert read in blk and blk.index(read) < blk.index("_load_archive_index("), (
        "the archive id must be read from session_state before resolving, so layout is free")
    assert blk.index(read) < blk.index('key="mp_mv_index"'), "the id is read after the widget instantiates — layout is coupled again"
    assert "_mv_row = st.columns(" in blk, "setup controls are not on one compact row"
    assert 'st.popover("⚙️' in blk, "no ⚙️ popover for the archive id once it is configured"
    assert "_mv_ph = st.empty()" in blk and "_mv_ph.status(" in blk, "the phased status is not in a placeholder"
    assert "_mv_ph.empty()" in blk, "the status row is never cleared after success — a dead row above the fold"
    assert '"elapsed": _mv_elapsed' in blk, "the compare time is not handed to the header"
    # REPOINTED 2026-09-20 from the literal `_mp_lens_row(df, "mv", ...)`: the Market Pulse
    # scope checkbox feeds the lens row `_mp_df` (df, or the sidebar cohort when ticked), so
    # the old string pinned the FRAME ARGUMENT — which was never this test's subject. The
    # invariant is that the reason chips ride in extra_keys so the row's 🧹 resets them; a
    # Clear that visibly clears the row and silently leaves another filter running is the
    # defect this line exists to prevent (§6 stale-test updating).
    import re as _re
    _lens = _re.search(r'_mp_lens_row\(\s*(\w+)\s*,\s*"mv"\s*,\s*extra_keys=\(("[^"]+",?\s*)+\)\)', blk)
    assert _lens, "the Movers lens row is gone, or no longer registers extra_keys"
    assert "mp_mv_why" in _lens.group(0), "the reason chips are not registered with the lens 🧹"
    assert _lens.group(1) in ("df", "_mp_df"), (
        f"the Movers lens row is fed {_lens.group(1)!r} — it must be the full universe or the "
        f"scope-aware frame, never a frame pre-filtered some other way"
    )
    assert "if len(_mv_cur_f) < len(df):" in blk, "restrict must run only when the lens narrowed the frame"
    assert "Click **Compare**" in blk, "sanity: the pre-compare hint still exists"


def test_movers_diffs_whole_frames_and_applies_the_lens_afterwards():
    """The lens narrows the CURRENT side AFTER the diff. Filtering the current frame first would
    turn every filtered-out stock into a fake 'dropped' row."""
    _, blk = _movers_block()
    call = "compute_movers(_mv_prev_df, df, days_between=_mv_gap)"
    assert call in blk, "the diff must run on the whole live frame, with the vintage gap for exact fresh-results"
    assert blk.index(call) < blk.index("restrict(_mv_res"), "restrict must follow the diff, never precede it"
    assert 'compute_movers(_mv_prev_df, _mv_cur_f' not in blk
    assert "_mv_gap = (_mv_today.fromisoformat(_mv_cur_v) - _mv_today.fromisoformat(_mv_prev_v)).days" in blk, (
        "the gap must be the calendar days between the two vintage dates")


# ── Lens-filter rows on QGLP / MOSL / Wealth (2026-08-30) ────────────────────────────────────
def test_lens_rows_wired_into_the_three_tabs():
    """QGLP and MOSL had ZERO controls over 328/586-row cohorts; the lens row (Sector · Wealth
    Tier · Market Cap · Catalyst · conditional Clear) fixes that. Wealth keeps its own Tier
    control above the table, so its row is with_tier=False — passing True there would render a
    SECOND tier control beside the first.

    That own control went MULTI-SELECT 2026-08-30 on the user's call, and it is declared to the
    row via `extra_keys` — it filters nothing inside the row (the tab has already applied it) but
    it counts as active and the row's 🧹 resets it. Without that, a Clear sitting on screen would
    visibly clear Sector/Cap/Catalyst and silently leave the tier filter running."""
    src = _app_src()
    assert '_mp_lens_row(_mp_qglp, "qglp")' in src, "QGLP tab lost its lens row"
    assert '_mp_lens_row(_mosl, "mosl")' in src, "MOSL tab lost its lens row"
    assert '_mp_lens_row(_wl, "w", extra_keys=("mp_wealth_tier",), with_tier=False)' in src, (
        "Wealth tab's lens row must stay with_tier=False (its own Tier control exists) AND declare "
        "that control via extra_keys, or the 🧹 Clear leaves it running")


def test_the_wealth_tabs_own_tier_control_is_multi_select():
    """The user's request, and the two readings it unblocks: "BUY★ and BUY" (the buy-grade half of
    the ladder) and "WATCH★ and WATCH" (both turnaround grades). Neither was expressible one tier
    at a time — the control was a selectbox."""
    import ast as _ast
    src = _app_src()
    call = next((n for n in _ast.walk(_ast.parse(src))
                 if isinstance(n, _ast.Call) and getattr(n.func, "id", "") == "_mp_ms"
                 and isinstance(n.args[3], _ast.Constant) and n.args[3].value == "mp_wealth_tier"),
                None)
    assert call is not None, "the Tier control is no longer a cascade multiselect"
    assert '"Tier", ["All"] + _WT_ORDER' not in src, "the single-pick Tier selectbox is back"
    assert '_wl[_wl["wealth_tier"].isin(_wt_pick)]' in src, (
        "the tier selection is not applied as a set — only the first pick would survive")
    assert '_wl["wealth_tier"] == _wt_pick' not in src, "a scalar tier comparison is still there"


def test_the_tier_options_offer_no_empty_tier():
    """Counts ride on the options, so offering a 0-stock tier would be a visible dead end."""
    src = _app_src()
    i = src.index('"mp_wealth_tier"')
    seg = src[i - 400:i]
    assert "[t for t in _WT_ORDER if _wt_n.get(t, 0)]" in seg, (
        "the Tier options no longer drop tiers that hold nothing")


def test_lens_clear_sets_all_and_never_deletes():
    """The reset callback must SET each key back to "All" — `del` on an instantiated widget's
    key lets the frontend resurrect the stale value on the next rerun (the Steel bug class).
    And the 🧹 button must be CONDITIONAL: rendered only when a filter is active, in a fixed
    slot (zero furniture idle, zero layout jump)."""
    src = _app_src()
    i = src.index("def _mp_clear_lens(")
    fn = src[i:src.index("def _mp_lens_row(")]
    assert "st.session_state[k] = v" in fn, "the reset no longer SETS keys to their defaults"
    assert "del " not in fn, "del in the lens reset — the resurrection class returns"
    j = src.index("if _n_active:")
    assert "st.button" in src[j:j + 250] and "_mp_clear_lens" in src[j:j + 250], (
        "the Clear button is no longer conditional on an active filter")


def test_lens_row_seeds_and_stale_guards_every_key():
    """Widget-state law — UPDATED 2026-09-02: the stored selection is KEPT, never pruned.
    `keep_selected` (ONE definition, imported from ui_discovery so both filter surfaces obey one
    rule) appends any stored pick the cascade narrowed out to the option list BEFORE the widget
    instantiates — so Streamlit's value-not-in-options crash cannot occur AND the filter stays
    applied with an honest count of 0. Pruning did the opposite: it switched the filter off and
    WIDENED the result (Wealth Tier=BUY★ + Sector=Air Transport Service showed the sector's 4
    non-BUY★ names instead of 0). Seeding by assignment (not `del`) keeps the Steel fix intact."""
    src = _app_src()
    # the cascade mechanics live in the hoisted _mp_ms helper (shared by every Market Pulse row
    # since Phase 2 — one dialect, not three); the lens row is one of its callers.
    i = src.index("def _mp_ms(")
    # module level since 2026-10-06 (shared with 🎯 Scans)
    fn = src[i:src.index("def _mp_lens_row(")]
    assert "if v in options]" not in fn, "the cascade prunes stored selections again — the widening class"
    keep = "options = keep_selected(options, stored)"
    assert keep in fn, "_mp_ms does not route through the shared keep_selected rule"
    assert fn.index(keep) < fn.index("st.multiselect"), "keeping must precede instantiation"
    seed = "st.session_state[key] = stored"
    assert seed in fn and fn.index(seed) < fn.index("st.multiselect"), "the key is no longer seeded by assignment"
    j = src.index("from ui.ui_discovery import")
    assert "keep_selected" in src[j:src.index(chr(10), j)], (
        "app.py must import keep_selected from ui_discovery — one definition, two surfaces")
    # Scan CODE only — the helper's own docstring explains the ban, and a naive substring
    # search matches that explanation (the prose-vs-code trap this project has hit repeatedly).
    body = fn.split(chr(34) * 3, 2)[2]
    assert "default=" not in body, (
        "passing default= alongside a key triggers Streamlit's default-plus-session-state warning "
        "— ui_discovery._ms_cascade manages state itself and this must too"
    )


def test_lens_row_is_multiselect_and_cascades():
    """THE CASCADE CONTRACT. Every control is a multi-select (OR within, AND across), and each
    one's options AND counts come from `_cf` — the frame already narrowed by the controls to its
    left. Measured: on the QGLP cohort, Sector=Auto Ancillaries takes the tier list 6 -> 4 and
    catalysts 5 -> 4. A row that computed all four option lists from the unfiltered frame (the
    pre-2026-08-30 behaviour) would show tiers that return zero rows."""
    src = _app_src()
    # the function body alone — since 2026-10-06 it is module level, so a slice to the next
    # Market Pulse landmark would also swallow the 🎯 Scans tab between them
    fn = next(ast.get_source_segment(src, n) for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "_mp_lens_row")
    # module level since 2026-10-06 (shared with 🎯 Scans)
    helper = src[src.index("def _mp_ms("):src.index("def _mp_lens_row(")]
    assert "st.selectbox" not in fn, "a single-select control is back in the lens row"
    assert "st.multiselect" in helper and "_mp_ms(" in fn, (
        "the lens row no longer routes through the shared _mp_ms cascade helper")
    # each stage must narrow _cf before the next stage reads it
    for sel, col in [("sel_sec", "sector"), ("sel_wt", "wealth_tier"), ("sel_cap", "market_category")]:
        assert f"if {sel}:" in fn, f"{sel} never narrows the cascade frame"
    assert fn.index("sel_sec") < fn.index("_wt_opts") < fn.index("_cap_opts"), (
        "the cascade order broke — later controls must be computed AFTER earlier narrowing"
    )
    # counts are display-only: baked into format_func, never into the option values
    assert "format_func=lambda v, _c=counts" in helper, "facet counts are no longer display-only"


def test_lens_reset_value_is_empty_list_not_all():
    """A multi-select resets to [] — "All" was the SINGLE-select sentinel and would be a stale
    value the pruning immediately strips, leaving the Clear button unable to clear."""
    src = _app_src()
    # the function body alone — since 2026-10-06 it is module level, so a slice to the next
    # Market Pulse landmark would also swallow the 🎯 Scans tab between them
    fn = next(ast.get_source_segment(src, n) for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "_mp_lens_row")
    assert "_defaults = {k: [] for k in keys}" in fn, "the lens reset no longer resets to []"
    assert '{k: "All" for k in keys}' not in fn, "the single-select sentinel is back"


def test_mp_catalysts_mirrors_ui_discovery():
    """One catalyst vocabulary: app.py's _MP_CATALYSTS literal must equal ui_discovery's
    _CATALYSTS literal (label -> flag column). A drifted copy silently filters on flags the
    sidebar no longer means."""
    import ast, io as _io, os
    root = os.path.join(os.path.dirname(__file__), "..")

    def _dict_literal(path, name):
        for n in ast.walk(ast.parse(_io.open(path, encoding="utf-8").read())):
            if (isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
                    and isinstance(n.value, ast.Dict)):
                return {k.value: v.value for k, v in zip(n.value.keys, n.value.values)}
        return None

    app_map = _dict_literal(os.path.join(root, "app.py"), "_MP_CATALYSTS")
    disc_map = _dict_literal(os.path.join(root, "ui", "ui_discovery.py"), "_CATALYSTS")
    assert app_map and disc_map, "one of the catalyst dicts vanished"
    assert app_map == disc_map, f"catalyst vocabularies drifted: app={app_map} vs discovery={disc_map}"


def test_sectors_and_industry_clear_are_default_aware_and_complete():
    """The Sectors/Industry Clear resets each control to ITS OWN default — one value is not
    universal: the four set-membership controls reset to [] (multi-select, 2026-08-30) while the
    size dial resets to 5, and resetting THAT to [] would corrupt a numeric selectbox.
    COMPLETENESS is the contract: every mp_sec_*/mp_ind_* widget key must appear in its defaults
    map, so a future sixth control cannot ship without declaring its default."""
    import re
    src = _app_src()
    i = src.index("_SEC_DEFAULTS = {")
    sec_map = src[i:src.index("}", i)]
    for k in ("mp_sec_cap", "mp_sec_wealth", "mp_sec_cyc", "mp_sec_phase"):
        assert f'"{k}": []' in sec_map, f"{k} missing from _SEC_DEFAULTS (or not reset to empty)"
    assert '"mp_sec_minn": 5' in sec_map, "the size dial must reset to 5, never to a selection"
    j = src.index("_IND_DEFAULTS = {")
    ind_map = src[j:src.index("}", j)]
    for k in ("mp_ind_cap", "mp_ind_wealth", "mp_ind_sec"):
        assert f'"{k}": []' in ind_map, f"{k} missing from _IND_DEFAULTS (or not reset to empty)"
    # completeness: no mp_sec_/mp_ind_ WIDGET key exists outside its defaults map (clear buttons
    # excluded). Both calling conventions counted — the multiselects pass their key positionally
    # to the shared _mp_ms helper, so a `key="..."` scan alone would see almost nothing.
    widget_keys = (set(re.findall(r'key="(mp_(?:sec|ind)_[a-z_]+)"', src))
                   | set(re.findall(r'_mp_ms\([^,]+,[^,]+,[^,]+,\s*"(mp_(?:sec|ind)_[a-z_]+)"', src)))
    assert len(widget_keys) >= 8, f"the completeness scan found only {widget_keys} — it has lost its teeth"
    declared = set(re.findall(r'"(mp_(?:sec|ind)_[a-z_]+)":', sec_map + ind_map))
    # Exempt: the Clear buttons themselves, and the 🔭 Column View switchers (2026-10-05) — a
    # DISPLAY choice, not a filter. Resetting the view on Clear would make the Clear button appear
    # the moment someone picks a view; tests/test_market_pulse_views.py pins that Clear never does.
    missing = sorted(widget_keys - declared - {"mp_sec_clear", "mp_ind_clear", "mp_sec_view", "mp_ind_view"})
    assert not missing, f"controls with NO declared reset default: {missing}"
    # both buttons are conditional + wired to the default-aware reset
    for anchor in ("_SEC_DEFAULTS.items())", "_IND_DEFAULTS.items())"):
        k = src.index(anchor)
        assert "st.button" in src[k:k + 300] and "_mp_clear_lens" in src[k:k + 300]


# ── Column-header vocabulary (2026-08-30 final-build pass) ───────────────────────────────────
def _configured_headers():
    """Every ("column", "Header") pair from a st.column_config.*("Header"...) call in app.py."""
    import re
    src = _app_src()
    pat = re.compile(r'"(?P<col>[a-z_0-9]+)":\s*st\.column_config\.\w+\(\s*"(?P<label>[^"]*)"')
    out = {}
    for m in pat.finditer(src):
        out.setdefault(m.group("col"), set()).add(m.group("label"))
    return out


def test_no_dataframe_header_is_a_raw_column_name():
    """Shipped UI must never show the dataframe's own snake_case column name. The Tsunami tab
    displayed "name", "sector", "smart_money_flow", "market_category" and "buy_zone_label" raw,
    and three tabs headed the stock column "name" while the Deep Scanner called it "Stock"."""
    bad = sorted(f"{c} -> {l!r}" for c, labels in _configured_headers().items()
                 for l in labels if l == c or (l.islower() and "_" in l))
    assert not bad, "raw column names used as display headers: " + " | ".join(bad)


def test_one_column_has_exactly_one_header_everywhere():
    """THE CONSISTENCY PIN. The same column must carry the same name in every table it appears
    in. red_flag_count once had THREE names across the app ("🚩" in Wealth, "🚩 Flags" in
    QGLP/MOSL, "Red Flags" in the Deep Scanner) — a reader cannot learn a vocabulary that
    changes per tab."""
    clashes = {c: sorted(l) for c, l in _configured_headers().items() if len(l) > 1}
    assert not clashes, f"columns headed differently in different tabs: {clashes}"


def test_the_header_scan_has_teeth():
    """Guard against the scan silently matching nothing (a vacuous pass — the failure mode that
    bit the framework-pairing pin earlier the same day)."""
    h = _configured_headers()
    assert len(h) >= 20, f"header scan found only {len(h)} configured columns — the regex broke"
    assert h.get("name") == {"Stock"}, f"the stock column is not uniformly 'Stock': {h.get('name')}"
