"""
test_sector_filters.py
======================
Contract for the Market Pulse → Sectors filter row.

FIVE CONTROLS, TWO KINDS — and the difference is invisible on screen, which is why it is pinned
here rather than left to whoever edits this next:

    RE-AGGREGATING   Market-cap tier, Wealth Tier, Cyclicality tier (returned 2026-08-28 —
                     measured: 42 of 81 sectors hold >1 tier, so only a stock filter can answer
                     "sector rankings among Defensive stocks"; the 2026-08-27 swap had quietly
                     lost that view)
                     filter the STOCKS, so every average and % Qualify is recomputed
    ROW FILTERS      Capital phase, Min stocks/sector
                     hide whole sectors and leave the survivors' numbers untouched

REPLACED 2026-08-27 (user request): the re-aggregating slot held Cyclicality tier; it now holds
the Wealth Tier. Same semantics, same slot, so every behavioural contract below carries over.
Cyclicality filtering still exists in the Discovery sidebar (sb_cyc) — no capability was lost.

Four identical-looking controls that behave in two different ways is a trap. The tests below
assert the behaviour, not the label.

MULTI-SELECT + CASCADE 2026-08-30 (user request). The four set-membership controls became
multiselects driven by the shared `_mp_ms` helper: several values each (OR within a control, AND
across), options and live counts computed from the frame already narrowed by the controls to the
left. Every semantic above survives the change — what moved is `== scalar` → `.isin(list)` and
`"All"` → `[]` as the neutral state. THE UNITS ARE THE NEW HAZARD, and they are pinned below: a
re-aggregating filter's count is STOCKS (what gets re-averaged), a row filter's count is SECTORS
(the rows it hides). One number, two possible meanings, is the "100% trap" class again.

The size dial stays a SELECTBOX: it is a numeric threshold, not a set.

WHY THE SIZE DIAL EXISTS, and why it is the important one. The floor was hardcoded at 5, and that
is precisely why the ranking was dominated by tiny sectors: an extreme % Qualify is easy at n=7
and near-impossible at n=96. Measured 2026-08-27 — 8 of the top 10 sectors held fewer than 12
stocks (median 9 against 19 universe-wide), and Glass & Glass Products ranked #1 at 88% on 8
stocks while carrying a below-average composite of 27.9. Raising the floor to 15 changes the top
six COMPLETELY: 0 of 6 in common. The tab's own tooltip already warned about this; now it can be
acted on.

DEFAULT STAYS 5, so the tab is byte-identical for anyone who does not touch the control.

Run with: pytest tests/test_sector_filters.py -v
"""

import contextlib
import io as _io
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import pandas as pd
import pytest

from data_engine import (coerce_numeric_columns, compute_derived_signals, load_all_csvs,
                         merge_datasets)

_APP = os.path.join(os.path.dirname(__file__), "..", "app.py")


def _scf_start(src):
    """Index of the Sectors cascade-frame assignment, found by a NAME-AGNOSTIC regex.

    Two scans below slice the cascade block starting from that line, and both used its exact text
    as a landmark. On 2026-09-20 the Market Pulse scope checkbox changed the frame it starts from
    (the full universe, or the sidebar cohort when ticked) and both tests died on 'substring not
    found' -- while their actual subjects, the units rule and the cascade ORDER, were untouched.
    The name of the source frame was never what they pinned, so the landmark stopped naming it.
    """
    m = re.search(r"^\s*_scf = \w+", src, re.M)
    assert m, "the Sectors cascade-frame assignment is gone -- these scans lost their anchor"
    return m.start()


@pytest.fixture(scope="module")
def src():
    return _io.open(_APP, encoding="utf-8").read()


@pytest.fixture(scope="module")
def ms_calls(src):
    """Every `_mp_ms(...)` call in app.py, read from the AST.

    A plain substring scan for a key would ALSO match the `_SEC_DEFAULTS` dict and stay green with
    the control itself deleted — a vacuous pass. The AST asks the real question: is this key wired
    to the cascade multiselect helper?
    """
    import ast as _ast
    calls = [n for n in _ast.walk(_ast.parse(src))
             if isinstance(n, _ast.Call) and getattr(n.func, "id", "") == "_mp_ms"]
    assert len(calls) >= 7, f"only {len(calls)} _mp_ms calls found — this scan has lost its teeth"
    return calls


@pytest.fixture(scope="module")
def ms_help(ms_calls):
    """key -> help text, taken from the AST so implicitly-concatenated literals arrive JOINED.
    Scanning the source text for a sentence that the line-wrapper split across two literals is the
    substring-over-prose trap this suite has fallen into before."""
    import ast as _ast
    return {c.args[3].value: c.args[4].value for c in ms_calls
            if isinstance(c.args[3], _ast.Constant) and isinstance(c.args[4], _ast.Constant)}


@pytest.fixture(scope="module")
def ms_keys(ms_calls):
    """Literal keys only: the lens rows build theirs as f-strings (`f"mp_{prefix}_sec"`), which is
    correct there and simply not what this file is about."""
    import ast as _ast
    return {c.args[3].value for c in ms_calls if isinstance(c.args[3], _ast.Constant)}


@pytest.fixture(scope="module")
def live():
    from core import run_scoring_pipeline
    with contextlib.redirect_stdout(_io.StringIO()):
        return run_scoring_pipeline(
            compute_derived_signals(coerce_numeric_columns(merge_datasets(load_all_csvs("local")))))


def _agg(d, min_n=5):
    """Mirror of the tab's aggregation, so these tests measure what it measures."""
    g = d.groupby("sector").agg(
        stocks=("name", "count"),
        pct_qualify=("gate_pass", lambda s: 100.0 * s.mean()),
        avg_composite=("composite_score", "mean"),
    )
    return (g[g["stocks"] >= min_n]
            .sort_values(["pct_qualify", "avg_composite"], ascending=False))


# -- 1. The controls exist, and the default changes nothing ---------------------------------
@pytest.mark.parametrize("key", ["mp_sec_cap", "mp_sec_wealth", "mp_sec_cyc", "mp_sec_phase"])
def test_each_control_is_present_with_its_own_key(ms_keys, key):
    assert key in ms_keys, f"the {key} control is gone, or is no longer a cascade multiselect"


def test_the_size_dial_stays_a_selectbox(src):
    """A numeric THRESHOLD, not a set: 'at least 5 and at least 15' is not a question anyone asks,
    and a multiselect would offer it."""
    assert 'key="mp_sec_minn"' in src, "the mp_sec_minn control is gone"
    i = src.index('key="mp_sec_minn"')
    assert "st.selectbox" in src[i - 300:i], "the size dial stopped being a selectbox"


def test_the_size_dial_defaults_to_the_old_hardcoded_floor(src):
    """Anyone who never touches it must see exactly the tab they saw before."""
    i = src.index('key="mp_sec_minn"')
    block = src[i - 400:i + 200]
    # 1 ADDED 2026-08-30 (user request: a show-everything floor). The REAL invariant is that the
    # DEFAULT still resolves to 5 -- anyone who never touches the dial sees the tab unchanged.
    assert "[1, 5, 10, 15, 20, 30]" in block, f"the size options changed: {block[:200]}"
    assert "index=1" in block, "the default no longer resolves to 5 -- untouched behaviour changed"
    assert [1, 5, 10, 15, 20, 30][1] == 5   # the index-1 option IS 5 (self-check of this pin)


def test_the_hardcoded_floor_is_gone(src):
    assert '_sec_stats["stocks"] >= _min_n' in src, "the floor is not driven by the control"
    assert '_sec_stats["stocks"] >= 5]' not in src, "the hardcoded 5 is still there"


# -- 2. THE SEMANTICS: which filters move the numbers, and which only hide rows --------------
def test_wealth_tier_is_a_stock_filter_that_re_aggregates(live):
    """The premise for the slot it inherited from cyclicality: wealth_tier varies WITHIN sectors,
    so slicing by it genuinely changes what a sector's average is computed over."""
    v = live.groupby("sector")["wealth_tier"].nunique()
    assert (v > 1).sum() > 10, (
        f"wealth_tier now varies within only {int((v > 1).sum())} sectors -- if it became a "
        f"sector-constant attribute it should be a ROW filter like capital phase, not a stock one"
    )
    base = _agg(live)
    sliced = _agg(live[live["wealth_tier"] == "CLIMBING★"])
    common = base.index.intersection(sliced.index)
    assert len(common) > 5, "not enough overlap to compare"
    moved = (base.loc[common, "pct_qualify"] - sliced.loc[common, "pct_qualify"]).abs() > 0.01
    assert moved.any(), "slicing by wealth tier changed no sector's % Qualify -- it is not re-aggregating"


def test_the_wealth_filter_is_actually_applied_before_aggregation(src):
    """STRUCTURAL, because the data-property test above is blind to app.py: it slices the live
    frame itself, so gutting the filter application in the app leaves it green (a mutation run
    proved exactly that). This pins that the selection reaches _sec_src BEFORE the groupby."""
    i = src.index('"mp_sec_wealth"')
    j = src.index("_sec_stats = _sec_src.groupby", i)
    between = src[i:j]
    assert '_sec_src[_sec_src["wealth_tier"].isin(_wt_sec)]' in between, (
        "the Wealth Tier selection is no longer applied to _sec_src before aggregation -- the "
        "control renders but filters nothing"
    )


def test_capital_phase_is_a_row_filter_that_moves_no_number(live):
    """It is applied AFTER aggregation, so a surviving sector's figures must be identical."""
    v = live.groupby("sector")["sector_capital_phase"].nunique()
    assert (v > 1).sum() == 0, (
        f"sector_capital_phase now varies within {int((v > 1).sum())} sectors. The tab applies it "
        f"after aggregation precisely so this stays correct -- but the help text calling it a "
        f"sector attribute needs revisiting."
    )
    base = _agg(live)
    keep = set(live.loc[live["sector_capital_phase"].astype(str).str.contains("Hot"), "sector"])
    filtered = base[base.index.isin(keep)]
    assert len(filtered) > 0, "no Hot sector survives -- probe is stale"
    pd.testing.assert_frame_equal(filtered, base.loc[filtered.index], check_like=True)


def test_the_phase_filter_is_applied_after_aggregation(src):
    """Structural: filtering the stocks instead would part-filter a sector and skew its averages
    the moment phase stops being sector-constant."""
    i = src.index("_sec_stats = (_sec_stats[")
    block = src[i:i + 900]
    assert "sector_capital_phase" in block, "the phase filter moved out of the post-aggregation block"
    assert "_sec_stats.index.isin" in block, "the phase filter no longer selects whole sectors by index"


# -- 3. The size dial must actually change the answer ----------------------------------------
def test_raising_the_minimum_changes_which_sectors_lead(live):
    """The whole reason the control exists. If this stops being true the small-sample bias has
    gone and the dial is merely decorative."""
    top5 = list(_agg(live, 5).head(6).index)
    top15 = list(_agg(live, 15).head(6).index)
    overlap = len(set(top5) & set(top15))
    assert overlap <= 3, (
        f"raising the floor from 5 to 15 leaves {overlap} of the top 6 unchanged; the small-sample "
        f"dominance this control addresses appears to have gone -- re-measure before trusting the "
        f"help text, which claims the ranking changes completely"
    )


def test_small_sectors_really_do_dominate_the_default_view(live):
    """The claim in the help text, checked against live data."""
    top10 = _agg(live, 5).head(10)
    assert top10["stocks"].median() < _agg(live, 5)["stocks"].median(), (
        "the top-ranked sectors are no longer smaller than typical -- the help text's premise is stale"
    )


def test_the_floor_never_admits_a_sector_below_it(live):
    for n in (1, 5, 10, 15, 20, 30):
        g = _agg(live, n)
        if g.empty:
            continue
        assert g["stocks"].min() >= n, f"a sector below {n} stocks survived the {n} floor"


# -- 4. The two kinds are explained to the reader --------------------------------------------
def test_the_help_text_distinguishes_the_two_behaviours(src):
    """Four identical-looking selectboxes behaving in two ways needs saying, not guessing."""
    i = src.index('"mp_sec_cap"')
    block = src[i - 900:i + 5200]      # widened 2026-08-28: the cyclicality control joined the row
    assert block.lower().count("re-aggregat") >= 3, (
        "the three re-aggregating controls no longer all say that they recompute the averages"
    )
    assert block.lower().count("hides rows") >= 2, (
        "the row filters no longer say that they only hide rows"
    )

def test_the_caption_follows_the_size_dial(src):
    """The caption hardcoded "≥5 stocks" while the floor became a control, so it contradicted its
    own dial the moment that moved to 15. Verified in the browser before it was fixed."""
    # SCOPED TO THE RENDERED STRING, not the whole file. The first version scanned all of app.py
    # for the literal and matched the CODE COMMENT that explains this very bug -- the fourth
    # substring-scan-over-source mistake of the session. Prose naming a banned string is not the
    # banned string.
    i = src.index("_sec_cap_ph.markdown(")
    block = src[i:src.index("unsafe_allow_html=True,", i)]
    assert "{_min_n}" in block, "the caption no longer interpolates the chosen minimum"
    assert "≥5" not in block, f"the caption hardcodes the old floor: {block[:160]}"


# ── 5. 💹 tier-share column (2026-08-28) — one column, follows the filter ────────────────────
def test_tier_share_base_is_captured_before_the_wealth_filter(src):
    """THE 100% TRAP, pinned structurally: the share column's denominator must be the
    pre-wealth-filter roster. Captured after the filter, selecting any tier makes every
    surviving row read a meaningless 100% — the design review caught this before it shipped."""
    i = src.index("_sec_share_base = _sec_src")
    j = src.index('_sec_src = _sec_src[_sec_src["wealth_tier"].isin(_wt_sec)]')
    assert i < j, "the share base is captured AFTER the wealth filter — the 100% trap is live"
    assert '_sec_share_base.groupby("sector")["wealth_tier"]' in src, (
        "the share aggregation no longer reads the pre-filter base — capturing it means nothing "
        "if the groupby runs on the filtered frame"
    )


def test_the_100_percent_trap_is_real(live):
    """Self-verifying: the naive post-filter share IS 100% everywhere, and the pre-filter one
    genuinely varies — so the structural pin above defends a measured hazard, not a style."""
    filt = live[live["wealth_tier"] == "STALLED"]
    trap = filt.groupby("sector")["wealth_tier"].apply(lambda s: 100.0 * (s == "STALLED").mean())
    assert (trap == 100.0).all(), "the trap probe went stale — post-filter share is no longer trivial"
    full = live.groupby("sector")["wealth_tier"].apply(lambda s: 100.0 * (s == "STALLED").mean())
    assert full.std() > 1.0, "the full-roster share no longer varies across sectors"


def test_tier_share_is_exact_match_never_contains(src, live):
    """'CLIMBING' ⊂ 'CLIMBING★' (the QGLP⊂SQGLP class). Structural: exact MEMBERSHIP in the app — `.isin`
    over the selected tier set since the control went multi-select, which is still exact-match
    per value and still not a substring test. Live: the two matchers genuinely disagree, so the
    pin has teeth."""
    # ANCHORED STRUCTURALLY (2026-10-03). The window was a fixed 2,500 characters from the first
    # mention of the tier, and it broke when the 🚀 Improvement column added lines ABOVE the share
    # computation — a change this test has no opinion about. Its subject is the computation, so the
    # window now runs from the tier's definition to the END of that computation, however much code
    # sits between them.
    i = src.index("_sec_share_tier")
    seg = src[i:src.index(".reindex(_sec_stats.index)", src.index('_sec_stats["pct_tier"]', i))]
    assert "s.isin(_sec_share_tiers)" in seg, "the share is no longer an exact-membership match"
    assert ".str.contains" not in seg, "a contains-match crept into the tier share"
    star = live.groupby("sector")["wealth_tier"].apply(lambda s: 100.0 * (s == "CLIMBING★").mean())
    loose = live.groupby("sector")["wealth_tier"].apply(
        lambda s: 100.0 * s.astype(str).str.contains("CLIMBING", regex=False).mean())
    assert (loose - star).max() > 1.0, "CLIMBING and CLIMBING★ no longer diverge — re-check before relaxing"


def test_tier_share_defaults_to_the_top_tier_and_the_label_follows(src):
    """Nothing selected → the top of the ladder (CLIMBING★ — its order is not yet tested forward:
    December plan, rule 2); a selection renames the column so its semantics are self-announcing.
    Multi-select 2026-08-30: the share follows the SET, and the label degrades gracefully (one tier →
    its name, more → a count): since the 2026-10-15 rename two words already overfill a header
    ('CLIMBING★+CLIMBING' is 18 characters), so the tooltip and the caption above name the set."""
    assert '_sec_share_tiers = list(_wt_sec) if _wt_sec else [_WT_ORDER[0]]' in src
    assert 'f"💹 {_sec_share_tier} %"' in src, "the column label no longer follows the filter"


def test_tier_share_base_rate_is_measured_not_typed(src):
    """The tooltip quoted 'Universe BUY★ base rate ≈ 12%' — typed once, true once (13.3% on the
    14 Oct sheet), and the same 12% whichever tier the column showed. Both tabs now compute it from
    the whole market, for the tier set the column actually shows."""
    assert "≈ 12%" not in src, "a typed base rate is back"
    for tab in ("_sec", "_ind"):
        assert f'{tab}_share_rate = (100.0 * df["wealth_tier"].isin({tab}_share_tiers).mean()' in src
        assert f"{{{tab}_share_rate:.0f}}%" in src, f"the {tab} tooltip no longer quotes the measured rate"


def test_the_tier_share_label_never_outgrows_a_column_header(src):
    """Behavioural mirror of the label expression above, so the degradation is checked and not just
    read: the ladder is 6 tiers deep and naming all six would produce a 40-character header."""
    i = src.index("_sec_share_tiers = list(_wt_sec)")
    expr = src[i:src.index("if _wt_sec:", i)]
    ns = {}
    for sel, want in [(["CLIMBING★"], "CLIMBING★"), (["RECOVERING"], "RECOVERING"),
                      (["CLIMBING★", "CLIMBING"], "2 tiers"),
                      (["CLIMBING★", "CLIMBING", "RECOVERING", "UNEVEN"], "4 tiers"), ([], "CLIMBING★")]:
        ns["_wt_sec"] = sel
        ns["_WT_ORDER"] = ["CLIMBING★", "CLIMBING", "RECOVERING", "UNEVEN", "STALLED", "N/A"]
        exec(expr.replace("\n" + " " * 8, "\n"), {}, ns)
        assert ns["_sec_share_tier"] == want, f"{sel} labelled {ns['_sec_share_tier']!r}, want {want!r}"
        assert len(ns["_sec_share_tier"]) <= 12, "the header label got long enough to be clipped"


def test_tier_share_is_not_a_sort_key(src):
    """A column, not the ranking: Sectors keeps % Qualify → Score."""
    i = src.index('_sec_stats = (_sec_stats[_sec_stats["stocks"] >= _min_n]')
    assert "pct_tier" not in src[i:i + 300], "pct_tier became a sort key"


# ── 6. Cyclicality tier — the returned third re-aggregator (2026-08-28) ──────────────────────
def test_cyclicality_really_does_vary_within_sectors(live):
    """The premise that makes this a STOCK filter (and made the 2026-08-27 swap a silent
    capability loss): tiers cross sector lines. Measured at build: 42 of 81. If this collapses,
    the filter should become a row filter and this file rewritten."""
    v = live.groupby("sector")["cyclicality_tier"].nunique()
    assert (v > 1).sum() > 10, (
        f"cyclicality varies within only {int((v > 1).sum())} sectors — the stock-filter premise died"
    )


def test_cyclicality_filter_is_applied_before_the_share_base_and_the_groupby(src):
    """Ordering is the whole contract: cap → cyclicality → [💹 share base] → wealth → groupby.
    Applied after the share base, Defensive × CLIMBING★ would silently show the ALL-stock share."""
    i = src.index('_sec_src[_sec_src["cyclicality_tier"].isin(_cyc_sec)]')
    j = src.index("_sec_share_base = _sec_src")
    k = src.index('_sec_src = _sec_src[_sec_src["wealth_tier"].isin(_wt_sec)]')
    m = src.index("_sec_stats = _sec_src.groupby")
    assert i < j < k < m, "the cyclicality filter is out of order in the filter chain"


def test_cyclicality_slice_re_aggregates(live):
    """Behavioural: slicing to one tier genuinely changes sector averages (it is not a row hide)."""
    base = _agg(live)
    sliced = _agg(live[live["cyclicality_tier"] == "Defensive"])
    common = base.index.intersection(sliced.index)
    assert len(common) > 5, "not enough overlap to compare"
    moved = (base.loc[common, "avg_composite"] - sliced.loc[common, "avg_composite"]).abs() > 0.01
    assert moved.any(), "slicing by cyclicality changed no sector average — it is not re-aggregating"


# ── 7. Multi-select + cascade (2026-08-30) — the units rule, pinned ──────────────────────────
def test_every_set_control_takes_several_values(src):
    """The user's actual request. `.isin` on all four, nowhere a scalar `== _sel` that would
    silently accept only the first pick."""
    for var, col in [("_cap", "market_category"), ("_wt_sec", "wealth_tier"),
                     ("_cyc_sec", "cyclicality_tier"), ("_phase", "sector_capital_phase")]:
        assert f'["{col}"].isin({var})' in src, f"{var} is no longer applied as a set membership"
        assert f'["{col}"] == {var}' not in src, f"{var} still has a scalar comparison somewhere"


def test_the_re_aggregating_counts_are_stocks_and_the_row_filter_counts_sectors(src):
    """THE UNITS RULE, and the reason this tab was held back a phase. The three re-aggregating
    filters count STOCKS — value_counts over the stock frame. The capital-phase ROW filter counts
    SECTORS, so its counts come from DE-DUPLICATED (sector, phase) pairs: counting its stocks
    would put '412' beside an option that hides 9 rows."""
    i = src.index('"mp_sec_phase"')
    block = src[_scf_start(src):i]             # from the cascade frame, so the counts are inside
    for col in ["market_category", "cyclicality_tier", "wealth_tier"]:
        assert f'_scf["{col}"].astype(str).value_counts()' in block, (
            f"the {col} facet count is no longer a straight stock count"
        )
    ph = src[src.index("_ph_pairs"):i]
    assert 'drop_duplicates()' in ph, "the capital-phase counts are no longer de-duplicated to sectors"
    assert '["sector", "sector_capital_phase"]' in ph, "the phase count lost its sector pairing"


def test_the_help_text_states_which_unit_each_count_is_in(ms_help):
    """A bare number beside an option means nothing until you know what it counts, and the two
    kinds sit in the same row."""
    for k in ["mp_sec_cap", "mp_sec_wealth", "mp_sec_cyc"]:
        assert "Counts are STOCKS" in ms_help[k], f"{k} stopped naming its unit"
    assert "SECTORS, not stocks" in ms_help["mp_sec_phase"], (
        "the row filter no longer warns that its unit differs from its neighbours'"
    )


def test_the_phase_count_really_would_mislead_as_stocks(live):
    """Teeth for the rule above: the two units are far apart on live data, so labelling the row
    filter with a stock count would be a real mislead, not a pedantic one."""
    pairs = live[["sector", "sector_capital_phase"]].dropna().drop_duplicates()
    sectors = pairs["sector_capital_phase"].value_counts()
    stocks = live["sector_capital_phase"].dropna().value_counts()
    assert not sectors.empty, "no capital phases on live data — probe is stale"
    common = sectors.index.intersection(stocks.index)
    assert (stocks[common] / sectors[common]).max() > 3, (
        "stock and sector counts per phase are now within 3x of each other — re-measure before "
        "relaxing the units rule"
    )


def test_the_cascade_narrows_left_to_right(src):
    """Each control's options come from `_scf`, the frame already narrowed by the ones to its left
    — never from `df`. A stale option list is how a cascade quietly starts lying: it keeps offering
    a combination that yields nothing.

    ONE ORDER, THREE TIMES OVER (fixed 2026-08-30 while pinning this). The row is laid out in the
    order the filters are actually APPLIED — cap → cyclicality → wealth → phase — because the
    application order is not free: the 💹 share base must be captured after cyclicality and before
    wealth (test_cyclicality_filter_is_applied_before_the_share_base_and_the_groupby). The first
    build put wealth second on screen but narrowed neither by the other, so 'Defensive × CLIMBING★'
    could be offered as a live pair when the intersection was empty. Column order, cascade order
    and application order are now the same order.
    """
    block = src[_scf_start(src):src.index('key="mp_sec_minn"')]
    assert block.count("_scf[_scf[") == 3, "the three re-aggregating filters no longer all narrow _scf"
    # GUARDED, not merely present: a mutation run put `if False:` above each narrowing line and an
    # existence check stayed green. Whitespace-collapsed so the pin is indentation-agnostic.
    flat = re.sub(r"\s+", " ", block)
    for var, col in [("_cap", "market_category"), ("_cyc_sec", "cyclicality_tier"),
                     ("_wt_sec", "wealth_tier")]:
        assert f'if {var}: _scf = _scf[_scf["{col}"].isin({var})]' in flat, (
            f"the {col} narrowing is no longer reached by its own selection"
        )
    for col in ["market_category", "cyclicality_tier", "wealth_tier", "sector_capital_phase"]:
        assert f'df["{col}"]' not in block.replace("_scf", ""), (
            f"the {col} facet reads the unfiltered universe again — the cascade is broken"
        )
    assert (block.index('_scf[_scf["market_category"]') < block.index('_scf["cyclicality_tier"]')
            < block.index('_scf[_scf["cyclicality_tier"]') < block.index('_scf["wealth_tier"]')
            < block.index('_scf[_scf["wealth_tier"]') < block.index("_ph_pairs")), (
        "the cascade order broke — each control must be COUNTED before the next one narrows"
    )


def test_the_clear_resets_every_set_control_to_empty(src):
    """[] is the neutral state now, not "All" — and a reset that SETS (never `del`) is the law
    this whole app learned the hard way (the Steel bug: `del` lets the frontend resurrect the
    stale value on the next run)."""
    i = src.index("_SEC_DEFAULTS = {")
    d = src[i:src.index("}", i) + 1]
    for k in ["mp_sec_cap", "mp_sec_wealth", "mp_sec_cyc", "mp_sec_phase"]:
        assert f'"{k}": []' in d, f"{k} does not reset to the empty selection"
    assert '"mp_sec_minn": 5' in d, "the size dial no longer resets to its own default of 5"


def test_the_sector_column_is_not_shown_under_its_raw_name(src):
    """`reset_index()` turns the groupby key into a COLUMN, and a column with no config entry
    renders under its raw snake_case name — this table displayed "sector". The app-wide header pin
    (test_market_pulse_tabs::test_no_dataframe_header_is_a_raw_column_name) cannot catch it: that
    scan reads st.column_config pairs, so a column with NO pair is invisible to it. Caught in the
    browser instead, and pinned here where the table lives."""
    # Repointed 2026-10-05: the table now displays the selected view's list (_sec_show); the
    # subject — the sector column carries a config — is unchanged.
    i = src.index("_sec_stats[_sec_show].reset_index()")
    block = src[i:src.index("hide_index=True", i)]
    assert '"sector":' in block, "the sector column has no column_config — it renders raw"
    assert 'st.column_config.TextColumn("Sector"' in block, (
        "the sector column is no longer given a display name")


def test_the_caption_points_to_the_industry_tab(src):
    """The dispersion pointer (measured: up to 50 points of industry spread inside one sector):
    the Sectors caption must hand the reader to 🏭 Industry."""
    i = src.index("_sec_cap_ph.markdown(")
    block = src[i:src.index("unsafe_allow_html=True,", i)]
    assert "🏭 Industry" in block, "the Sectors caption no longer points to the Industry tab"
    assert "50 points" in block, "the dispersion magnitude vanished from the caption"
