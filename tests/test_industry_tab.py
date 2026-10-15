"""
test_industry_tab.py
====================
Contract for the Market Pulse → 🏭 Industry tab (added 2026-08-28).

WHY THE TAB EXISTS — the one number that justifies it. `sector` has 81 values; `industry` has 355.
Averaging to the sector hides real dispersion: the six sizeable industries inside Pharmaceuticals
run 18.1 → 51.3 on average composite, a 33-point spread that the Sectors tab reports as a single
number. Of the 76 industries holding ≥8 stocks, 20 sit more than 5 points from their parent
sector's average (Pharma - MNC bulk Drugs +22.7; Auto Ancillaries - Gears −12.9). That gap IS the
tab's subject, so it is a column — `delta_vs_sector` — and the table sorts by it.

THREE HAZARDS MEASUREMENT CAUGHT, each pinned below because each is invisible on screen:

  1. INDUSTRY IS NOT NESTED INSIDE SECTOR. 136 of 355 industries span more than one sector. A
     drill-down that assumes a tree is wrong. The tab shows the DOMINANT sector (modal, with a
     deterministic tie-break) and marks the impure ones with "~". Median purity is 0.91 but the
     minimum is 0.38, so the marker is not decorative.

  2. A DEGENERATE DELTA MUST BE NaN, NEVER 0.0. Three of the 76 industries are the only industry
     in their dominant sector, so industry-average and sector-average are the SAME NUMBER and the
     delta is exactly zero by construction. Rendering that as 0.0 says "perfectly average"; the
     truth is "there is nothing to compare against". This is CLAUDE.md's semantic-truth principle
     (never inject a sentinel where the honest answer is missing), and it is the single most
     likely thing a future edit breaks — `a - b` is the obvious implementation and it is wrong.

  3. `ret_vs_industry_1y` IS REJECTED, deliberately. It looks like the perfect column for this tab
     (an industry-relative return the engine already computes), but 7 of the 76 industries have
     ZERO non-null coverage of it and it ranges −1307…+585. Aggregating it would produce a
     confident-looking number backed by nothing. The test below re-measures the coverage hole so
     the rejection cannot quietly stop being true.

THERE IS NO SIZE FLOOR — every industry shows, down to n=1. It went dial (5..30) → fixed 8 → none,
each step on the user's explicit call, the last made twice after the cost was measured and put in
front of them. The cost is real: the median industry holds 3 stocks against 19 for sectors, and
sorting by Δ does not neutralise that — Δ is MORE exposed than % Qualify, being unbounded where
% Qualify is capped at 0–100. With no floor, 5 of the top 10 rows are single-stock industries and
the leader is Auto Ancillaries - Seats (n=1, +32). Count sits first in the table as the mitigation.
That trade is pinned by test_small_samples_lead_the_table_by_design, which RECORDS it rather than
objecting to it — a floor coming back is a reversal to argue for, not a fix.

BASIS RULE. When a re-aggregating filter narrows the stocks, the sector baseline must be recomputed
over the SAME survivors. Comparing a Smallcap-only industry average against an all-cap sector
average is the cross-year-basis defect wearing different clothes: both terms of a difference must
come from one source. Pinned behaviourally below.

Run with: pytest tests/test_industry_tab.py -v
"""

import ast
import contextlib
import io as _io
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd
import pytest

from data_engine import (coerce_numeric_columns, compute_derived_signals, load_all_csvs,
                         merge_datasets)

_APP = os.path.join(os.path.dirname(__file__), "..", "app.py")

# THE TAB HAS NO SIZE FLOOR — every industry shows, down to n=1. It went dial (5..30) → fixed 8 →
# none, each step on the user's explicit call. NO_FLOOR is what the table actually applies;
# SIZEABLE is only an analysis threshold these tests use to reason about well-sampled industries.
NO_FLOOR = 1
SIZEABLE = 8
# Below this share of its stocks sitting in one sector, an industry is labelled impure.
PURITY_MARK = 0.8


@pytest.fixture(scope="module")
def src():
    return _io.open(_APP, encoding="utf-8").read()


@pytest.fixture(scope="module")
def block(src):
    """Just the Industry tab's own source — so a check cannot accidentally pass on the Sectors
    tab's very similar code sitting directly above it."""
    start = src.index("# ══ Industry ══")
    # The block ends at the NEXT inner-tab header or the section divider, whichever comes first.
    # 🔁 Movers was appended after Industry on 2026-09-04; slicing to the divider alone would
    # pull its widgets into these pins (found the moment it landed).
    ends = [i for i in (src.find("# ══ ", start + 20), src.find("# ━━━", start)) if i != -1]
    return src[start:min(ends)]


@pytest.fixture(scope="module")
def ms_help(src):
    """key -> help text for this tab's cascade multiselects, read from the AST so that implicitly
    concatenated literals arrive JOINED. Scanning the source text for a sentence the line-wrapper
    split across two literals is the substring-over-prose trap this suite has hit before."""
    out = {}
    for n in ast.walk(ast.parse(src)):
        if (isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_mp_ms"
                and isinstance(n.args[3], ast.Constant)
                and str(n.args[3].value).startswith("mp_ind_")):
            out[n.args[3].value] = n.args[4].value
    assert len(out) == 3, f"expected 3 Industry multiselects, found {sorted(out)}"
    return out


@pytest.fixture(scope="module")
def live():
    from core import run_scoring_pipeline
    with contextlib.redirect_stdout(_io.StringIO()):
        return run_scoring_pipeline(
            compute_derived_signals(coerce_numeric_columns(merge_datasets(load_all_csvs("local")))))


def _agg(df, floor=NO_FLOOR):
    """An INDEPENDENT re-implementation of the tab's aggregation, written from the spec rather
    than copied from app.py — so it can disagree with the app and catch it."""
    d = df[["industry", "sector", "name", "composite_score", "gate_pass"]].copy()
    d["industry"] = d["industry"].astype(str).str.strip()
    g = d.groupby("industry").agg(stocks=("name", "count"),
                                  avg_composite=("composite_score", "mean"))
    g = g[g["stocks"] >= floor]
    pair = (d.groupby(["industry", "sector"]).size().rename("n").reset_index()
            .sort_values(["industry", "n", "sector"], ascending=[True, False, True]))
    dom = pair.drop_duplicates("industry").set_index("industry")
    g["dom_sector"] = dom["sector"].reindex(g.index)
    g["purity"] = dom["n"].reindex(g.index) / g["stocks"]
    # LEAVE-ONE-OUT baseline (v2 of the tab, 2026-08-28): the peers are the dominant sector's
    # OTHER stocks — this industry's own in-sector stocks are subtracted out. Including them
    # damped the delta by (1 − the industry's share of the sector): QSR showed −7.0 against a
    # true peer gap of −48.1. Zero peers (a sector holding only this industry) → NaN falls out
    # of the count guard — the degenerate case needs no separate rule any more.
    sec_sum = d.groupby("sector")["composite_score"].sum()
    sec_cnt = d.groupby("sector")["composite_score"].count()
    in_dom = d[d["sector"].values == d["industry"].map(dom["sector"]).values]
    own = (in_dom.groupby("industry")["composite_score"].agg(["sum", "count"])
           .reindex(g.index))
    peer_sum = g["dom_sector"].map(sec_sum) - own["sum"].fillna(0.0)
    peer_cnt = g["dom_sector"].map(sec_cnt) - own["count"].fillna(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        g["delta"] = np.where(peer_cnt > 0, g["avg_composite"] - peer_sum / peer_cnt, np.nan)
    return g


# ── 1. The tab is wired, and appended rather than inserted ──────────────────────────────────
def test_industry_tab_is_the_sixth_tab(src):
    """Appended AFTER Sectors on purpose: every `with _mp_tabs[i]` body binds by index, so
    inserting anywhere else silently renders existing content into the wrong tab."""
    tree = ast.parse(src)
    labels = None
    for n in ast.walk(tree):
        if (isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_mp_tabs" for t in n.targets)
                and isinstance(n.value, ast.Call)
                and getattr(n.value.func, "attr", "") == "tabs"):
            labels = [e.value for e in n.value.args[0].elts if isinstance(e, ast.Constant)]
    assert labels is not None, "_mp_tabs assignment not found"
    assert labels[4] == "📈 Sectors", f"Sectors moved off index 4: {labels}"
    assert labels[5] == "🏭 Industry", f"Industry is not at index 5: {labels}"
    assert "with _mp_tabs[5]:" in src, "the Industry tab has no renderer body"


def test_industry_tab_reads_the_industry_column(block):
    assert '"industry"' in block, "the tab does not reference the industry column at all"


# ── 2. The degenerate delta: NaN, never 0.0 ─────────────────────────────────────────────────
def test_a_degenerate_delta_case_actually_exists_live(live):
    """SELF-VERIFYING. If no industry is alone in its sector any more, the guard below is dead
    weight and this test says so instead of silently passing forever."""
    g = _agg(live)
    d = live.copy()
    d["industry"] = d["industry"].astype(str).str.strip()
    n_ind = d.groupby("sector")["industry"].nunique()
    solo = g["dom_sector"].map(n_ind) == 1
    assert int(solo.sum()) >= 1, (
        "no industry is the sole industry of its sector any more — re-measure before deleting the "
        "guard, and delete the guard's justification with it"
    )


def test_degenerate_delta_is_nan_not_zero(live):
    """The defect this pins: `avg_industry - avg_sector` is the obvious implementation, and for a
    sector holding exactly one industry it returns 0.0 — which reads as 'perfectly average' when
    the truth is 'no comparison exists'. A sentinel standing in for missing evidence."""
    g = _agg(live)
    d = live.copy()
    d["industry"] = d["industry"].astype(str).str.strip()
    n_ind = d.groupby("sector")["industry"].nunique()
    solo = g["dom_sector"].map(n_ind) == 1
    assert g.loc[solo, "delta"].isna().all(), (
        f"these industries are alone in their sector and must show no delta at all: "
        f"{sorted(g.index[solo & g['delta'].notna()])}"
    )
    assert not (g.loc[solo, "delta"] == 0.0).any(), "a 0.0 sentinel leaked into a degenerate row"


def test_the_app_guards_the_degenerate_case(block):
    """The behavioural tests above run against this file's own re-implementation. This one checks
    app.py actually carries the guard, so the two cannot drift apart. Since the LOO baseline
    (2026-08-28) the guard IS the peer-count denominator: zero peers → np.nan falls out of the
    math instead of needing a separate ≥2-industries test."""
    assert "_peer_cnt > 0" in block, (
        "the peer-count guard is gone — a zero-peer division would emit inf/0.0 instead of the "
        "honest NaN for industries alone in their sector"
    )
    assert "np.nan" in block, "no NaN is ever emitted for the incomparable rows"


def test_the_app_baseline_excludes_the_industrys_own_stocks(block):
    """THE LOO PIN (2026-08-28). Including the industry's own stocks in the sector baseline
    damps its delta by exactly (1 − its share of the sector) — invisible on screen, and it hit
    hardest precisely on the sector-dominating rows the tab exists to expose (QSR: −7.0 shown
    vs −48.1 true). The app must subtract the industry's own in-sector sum and count."""
    assert '- _ind_own["sum"]' in block.replace("  ", " ") or '_ind_own["sum"]' in block, (
        "the baseline no longer subtracts the industry's own score sum — self-inclusion is back"
    )
    assert '_ind_own["count"]' in block, (
        "the baseline no longer subtracts the industry's own stock count"
    )


def test_delta_is_not_uniformly_nan(live):
    """The mirror failure: guarding so hard that every row loses its delta."""
    g = _agg(live)
    assert g["delta"].notna().sum() >= 20, "the delta column is mostly empty — the guard is too broad"


# ── 3. The basis rule: baseline recomputed over the same survivors ──────────────────────────
def test_delta_baseline_uses_the_filtered_subset(live):
    """Both terms of the difference must come from one population. Filtering the stocks to one
    cap tier and comparing against an ALL-cap sector average is the cross-year-basis defect in a
    different costume: the gap it reports is mostly the cap effect, not the industry."""
    sub = live[live["market_category"] == "Small Cap"] if "market_category" in live.columns else live
    assert len(sub) >= 200, f"only {len(sub)} Small Cap rows — the cap tier labels changed"
    g_sub = _agg(sub, floor=5)
    d = sub.copy()
    d["industry"] = d["industry"].astype(str).str.strip()
    base_sub = d.groupby("sector")["composite_score"].mean()
    base_all = live.groupby("sector")["composite_score"].mean()
    shared = [s for s in base_sub.index if s in base_all.index]
    assert (base_sub[shared] - base_all[shared]).abs().max() > 0.5, (
        "the filtered and unfiltered sector baselines are identical, so this test proves nothing"
    )
    # LOO expectation, recomputed here by MASKING (a different route than the mirror's
    # sum-subtraction, so the two implementations can disagree and catch each other):
    row = g_sub[g_sub["delta"].notna()].index[0]
    _dsec = g_sub.loc[row, "dom_sector"]
    _d2 = sub.copy()
    _d2["industry"] = _d2["industry"].astype(str).str.strip()
    _peers = _d2[(_d2["sector"] == _dsec) & (_d2["industry"] != row)]["composite_score"]
    expect = g_sub.loc[row, "avg_composite"] - _peers.mean()
    assert abs(g_sub.loc[row, "delta"] - expect) < 1e-9


def test_dominant_industries_are_no_longer_self_damped(live):
    """The defect the LOO baseline closed (2026-08-28), measured live: including an industry's
    own stocks damped its delta by (1 − its sector share) — QSR displayed −7.0 against a true
    peer gap of −48.1. The LOO delta must therefore differ MATERIALLY from the include-self
    delta somewhere in the table. If this ever fails, either self-inclusion crept back into
    the mirror, or the universe lost its sector-dominating industries — re-measure before
    touching anything."""
    g = _agg(live)
    d = live.copy()
    d["industry"] = d["industry"].astype(str).str.strip()
    base_incl = d.groupby("sector")["composite_score"].mean()
    incl = g["avg_composite"] - g["dom_sector"].map(base_incl)
    both = g["delta"].notna()
    gap = (g.loc[both, "delta"] - incl[both]).abs()
    assert gap.max() > 5.0, (
        f"LOO and include-self deltas agree within {gap.max():.2f} pts everywhere — the "
        f"self-damping this baseline exists to remove has vanished"
    )
    # And the two must agree where self-inclusion is immaterial: tiny industries in big sectors.
    small = both & (g["stocks"] <= 2) & (g["purity"] >= 0.99)
    if small.any():
        small_gap = (g.loc[small, "delta"] - incl[small]).abs()
        assert small_gap.median() < 2.0, (
            "even negligible-share industries diverge between LOO and include-self — the "
            "baseline arithmetic is off, not just de-damped"
        )


def test_app_recomputes_the_baseline_after_filtering(block):
    """Source-level: the sector baseline must be grouped off the FILTERED frame, not off `df`."""
    m = re.search(r'(\w+)\.groupby\("sector"\)\["composite_score"\]', block)
    assert m, "no per-sector composite baseline is computed in the Industry tab"
    assert m.group(1) != "df", (
        "the sector baseline is computed from the unfiltered `df` while the industry averages come "
        "from the filtered frame — the two terms of the delta no longer share a population"
    )


# ── 4. Non-nesting: dominant sector, deterministic, marked when impure ──────────────────────
def test_industries_really_do_span_multiple_sectors(live):
    """SELF-VERIFYING: the whole dominant-sector apparatus exists because of this fact."""
    d = live.copy()
    d["industry"] = d["industry"].astype(str).str.strip()
    spanning = (d.groupby("industry")["sector"].nunique() > 1).sum()
    assert spanning >= 50, (
        f"only {spanning} industries span >1 sector; if industry has become a clean child of "
        f"sector, the '~' marker and the dominant-sector language should be simplified away"
    )


def test_impure_industries_exist_and_are_marked(live, block):
    g = _agg(live)
    impure = g[g["purity"] < PURITY_MARK]
    assert len(impure) >= 5, (
        f"only {len(impure)} industries fall below {PURITY_MARK} purity — re-measure before "
        f"keeping the marker"
    )
    assert '"~' in block or "'~" in block, (
        "no impurity marker in the source: an industry whose stocks sit 38% in its labelled sector "
        "would be displayed as though it belonged there cleanly"
    )


def test_dominant_sector_tie_break_is_deterministic(block):
    """CLAUDE.md determinism mandate. A modal pick with an unsorted tie-break reorders between
    processes (PYTHONHASHSEED), so the displayed parent sector would flicker."""
    assert "drop_duplicates" in block, "no single-row-per-industry reduction found"
    assert "ascending=[True, False, True]" in block, (
        "the dominant-sector pick has no deterministic tie-break — sort by (industry, count desc, "
        "sector asc) so an exact tie always resolves the same way"
    )


# ── 5. The rejected column ─────────────────────────────────────────────────────────────────
def test_ret_vs_industry_is_not_aggregated(block):
    assert "ret_vs_industry" not in block, (
        "ret_vs_industry_* was aggregated into the industry table. It is the most tempting column "
        "for this tab and the least trustworthy: see the coverage test below."
    )


def test_the_reason_ret_vs_industry_was_rejected_is_still_true(live):
    """SELF-VERIFYING: if the coverage hole is ever filled, this fails and the column becomes fair
    game — the rejection is evidence-based, not a taste."""
    d = live.copy()
    d["industry"] = d["industry"].astype(str).str.strip()
    big = d["industry"].value_counts()
    big = big[big >= SIZEABLE].index
    cov = d[d["industry"].isin(big)].groupby("industry")["ret_vs_industry_1y"].apply(
        lambda s: s.notna().mean())
    assert int((cov == 0).sum()) >= 1, (
        "every sizeable industry now has some ret_vs_industry_1y coverage — remeasure; the column "
        "may now be safe to aggregate"
    )


# ── 6. The size floor ──────────────────────────────────────────────────────────────────────
def test_there_is_no_size_floor(block):
    """WHAT THE TAB DOES NOW, and the history so nobody re-derives it. It shipped with a
    5..30 selectbox mirroring the Sectors tab's; that became a fixed 8; the floor then went
    entirely. Every step was the user's explicit call, the last one made twice.

    So a floor reappearing is not a bug fix — it is a reversal, and it should have to argue with
    test_small_samples_lead_the_table_by_design below, which records exactly what the floor bought
    and what it cost."""
    assert "_IND_MIN_N" not in block, "a size-floor constant is back"
    assert "Min stocks / industry" not in block, "the retired dial is back"
    assert "mp_ind_minn" not in block, "the retired dial's widget key is back"
    assert not re.search(r'\["stocks"\]\s*>=', block), "a >= stocks row filter is back"


def test_every_industry_survives_to_the_table(live):
    """No floor means no industry is hidden -- including the single-stock ones."""
    d = live.copy()
    d["industry"] = d["industry"].astype(str).str.strip()
    d = d[~d["industry"].isin(["", "nan", "None"])]
    assert len(_agg(live)) == d["industry"].nunique(), "some industries are being dropped"
    assert int((_agg(live)["stocks"] == 1).sum()) >= 1, (
        "no single-stock industry survives -- something is still filtering by size"
    )


def test_small_samples_lead_the_table_by_design(live):
    """THE ACCEPTED COST, pinned so it stays a known trade rather than becoming a surprise.

    Sorting by Δ does NOT neutralise small samples -- an earlier comment in app.py claimed it did
    and measurement refuted it. Δ is MORE exposed than % Qualify, because % Qualify is bounded
    0-100 while Δ is unbounded. With no floor, the leaders are one- and two-stock industries whose
    Δ is one company's score minus a sector average.

    This test does not object to that. It records it, and it fails if the shape changes enough
    that the reasoning behind the decision no longer holds -- at which point the choice deserves
    re-making rather than inheriting."""
    t = _agg(live).sort_values("delta", ascending=False, na_position="last")
    top = t.head(10)
    assert int((top["stocks"] <= 2).sum()) >= 3, (
        f"only {int((top['stocks'] <= 2).sum())} of the top 10 are 1-2 stock industries. The "
        f"small-sample exposure this tab knowingly accepts has changed shape -- remeasure before "
        f"trusting either this comment or app.py's."
    )
    floored_top = set(_agg(live, floor=SIZEABLE)
                      .sort_values("delta", ascending=False, na_position="last").head(10).index)
    assert len(set(top.index) & floored_top) <= 3, (
        "the unfloored and well-sampled leaders now largely agree, so the no-floor choice costs "
        "little -- worth recording, but check the numbers in app.py's comment still hold"
    )


def test_count_is_the_mitigation_and_sits_up_front(src):
    """With no floor, Count is the ONLY thing standing between a reader and a one-stock Δ. It has
    to be visible without scrolling, which means near the left edge."""
    i = src.index("_ind_order = [c for c in [")
    cols = re.findall(r'"([a-z_]+)"', src[i:src.index("]", src.index("[c for c in [", i) + 14) + 1])
    assert cols.index("stocks") == 0, f"Count moved to position {cols.index('stocks')}"


def test_industries_are_smaller_than_sectors(live):
    """SELF-VERIFYING: the fact underneath every small-sample caveat on this tab."""
    d = live.copy()
    d["industry"] = d["industry"].astype(str).str.strip()
    assert d["industry"].value_counts().median() < d["sector"].value_counts().median(), (
        "industries are no longer smaller than sectors -- the caveats need re-justifying"
    )
    assert len(_agg(live)) >= 25, "too few industries for a useful tab"


# ── 7. Layout + house rules ────────────────────────────────────────────────────────────────
def test_the_wide_sector_column_comes_last(src):
    """Same invariant tests/test_market_pulse_columns.py pins for the other tables: the columns
    that make the tab's point precede the wide context one. Sector names here run to
    'Infrastructure Developers & Operators'."""
    i = src.index("_ind_order = [c for c in [")
    cols = re.findall(r'"([a-z_]+)"', src[i:src.index("]", src.index("[c for c in [", i) + 14) + 1])
    assert "delta_vs_sector" in cols, "the tab's headline column is not in the table"
    assert cols.index("delta_vs_sector") <= 3, (
        f"Δ vs Sector sits at position {cols.index('delta_vs_sector')} and will be clipped at the "
        f"right edge — it is the reason the tab exists"
    )
    assert cols.index("dom_sector") == len(cols) - 1, "the wide sector column must come last"


def test_table_sorts_by_delta_with_missing_last(block):
    """The user's choice: rank by distance from the parent sector. Rows with no comparison carry
    no signal, so they sink rather than floating to the top of a descending sort."""
    calls = re.findall(r"sort_values\((.{0,200}?)\)", block, re.S)
    assert calls, "no sort_values call in the Industry tab"
    ranked = [c for c in calls if "delta_vs_sector" in c]
    assert ranked, f"nothing sorts by the delta; sorts found: {calls}"
    assert all("na_position=\"last\"" in c for c in ranked), (
        "incomparable rows must sort LAST. In a descending sort pandas puts NaN first by default, "
        "so the rows with no comparison would head the table."
    )


def test_no_rowwise_iteration(block):
    """CLAUDE.md absolute vectorization mandate."""
    for banned in ("apply(axis=1)", "iterrows", "itertuples", "for _, row in"):
        assert banned not in block, f"row-wise iteration ({banned}) in the Industry tab"


def test_no_state_mutating_widgets_beyond_selectboxes(block):
    """Market Pulse owns no session state beyond its own widget keys. st.button/slider would
    write state app.py owns. ONE examined exception (2026-08-30, user-requested): the 🧹 Clear
    button — a default-aware reset that only SETS this tab's own mp_ind_* keys via the shared
    _mp_clear_lens callback (the app-wide conditional-Clear grammar). Any OTHER button stays
    banned."""
    buttons = re.findall(r"st\.button\([^)]*key=\"(\w+)\"", block)
    assert buttons in ([], ["mp_ind_clear"]), f"unexamined button(s) in the Industry tab: {buttons}"
    if buttons:
        i = block.index('key="mp_ind_clear"')
        assert "_mp_clear_lens" in block[i - 100:i + 200], (
            "the Industry Clear must wire the shared default-aware reset, nothing else")
    for banned in ("st.slider(", "st.number_input("):
        assert banned not in block, f"{banned} in the Industry tab"


def test_widget_keys_are_unique_to_this_tab(src, block):
    """A key collision with the Sectors tab would make one tab's control silently drive the
    other's — both tabs render on every run."""
    # MULTI-SELECT 2026-08-30: the three filters pass their key POSITIONALLY to the shared
    # _mp_ms cascade helper, so a `key="..."` scan alone now sees only the Clear button. Both
    # forms are collected — the pin is about the key SET, not the calling convention.
    keys = (re.findall(r'key="(\w+)"', block)
            + re.findall(r'_mp_ms\([^,]+,[^,]+,[^,]+,\s*"(\w+)"', block))
    assert sorted(keys) == ["mp_ind_cap", "mp_ind_clear", "mp_ind_sec", "mp_ind_view", "mp_ind_wealth"], (
        f"expected the two RE-AGGREGATING filters (market-cap, wealth tier), the sector "
        f"DRILL-DOWN row filter (added 2026-08-28, examined), the 🧹 Clear (added "
        f"2026-08-30, examined — default-aware reset), and the 🔭 Column View switcher (added "
        f"2026-10-05, examined — a display choice, never a filter; tests/test_market_pulse_views.py), "
        f"found {keys}. The min-stocks dial stays retired; any other widget arrived unexamined."
    )
    for k in keys:
        assert k.startswith("mp_ind_"), f"key {k!r} is not namespaced to the Industry tab"
        bindings = (src.count(f'key="{k}"')
                    + len(re.findall(r'_mp_ms\([^,]+,[^,]+,[^,]+,\s*"' + k + '"', src)))
        assert bindings == 1, f"key {k!r} drives {bindings} widgets in app.py, expected exactly 1"


def test_missing_industry_column_is_handled(block):
    """Every other Market Pulse section degrades rather than crashing when a column is absent."""
    assert 'if "industry" not in' in block or '"industry" in df.columns' in block, (
        "no guard for a frame without an industry column"
    )


# ── 7. 💹 tier-share column (2026-08-28) — same design as the Sectors tab ────────────────────
def test_tier_share_base_is_captured_before_the_wealth_filter(block):
    """The 100% trap (see test_sector_filters for the live proof): the denominator must be the
    pre-wealth-filter roster."""
    i = block.index("_ind_share_base = _ind_src")
    j = block.index('_ind_src = _ind_src[_ind_src["wealth_tier"].isin(_ind_wt)]')
    assert i < j, "the share base is captured AFTER the wealth filter — the 100% trap is live"
    assert '_ind_share_base.groupby("industry")["wealth_tier"]' in block, (
        "the share aggregation no longer reads the pre-filter base"
    )


def test_tier_share_is_exact_match_never_contains(block):
    """Exact MEMBERSHIP since the control went multi-select (2026-08-30): still one exact match per
    selected tier, still never a substring test — "CLIMBING" is a prefix of "CLIMBING★"."""
    # ANCHORED STRUCTURALLY (2026-10-03): a fixed 2,600-character window broke when the 🚀
    # Improvement column added a line above the share computation. The window now ends where the
    # computation does — same subject, no dependence on what sits between. See the Sectors twin in
    # tests/test_sector_filters.py.
    i = block.index("_ind_share_tier")
    seg = block[i:block.index(".reindex(_ind_stats.index)", block.index('_ind_stats["pct_tier"]', i))]
    assert "s.isin(_ind_share_tiers)" in seg, "the share is no longer an exact-membership match"
    assert ".str.contains" not in seg, "a contains-match crept into the tier share"


def test_tier_share_defaults_to_the_top_tier_and_the_label_follows(block):
    """Nothing selected → the top of the ladder (CLIMBING★); a selection renames the column.
    Multi-select: the share follows the SET and the label degrades (one tier → its name, more → a
    count) so it still fits a column header — the same grammar as the Sectors tab, pinned
    behaviourally there."""
    assert '_ind_share_tiers = list(_ind_wt) if _ind_wt else [_WT_ORDER[0]]' in block
    i = block.index("_ind_share_tier = (")
    assert 'else f"{len(_ind_share_tiers)} tiers")' in block[i:i + 200], "the Industry label no longer degrades like Sectors"
    assert 'f"💹 {_ind_share_tier} %"' in block, "the column label no longer follows the filter"


def test_tier_share_is_not_the_sort_key(block):
    """Industry keeps its Δ-vs-Sector sort — the tab's whole subject."""
    i = block.index("_ind_stats.sort_values")
    assert "pct_tier" not in block[i:i + 250], "pct_tier became a sort key"


def test_industry_tier_share_discriminates(live):
    """Liveness: among industries with n≥5, the CLIMBING★ share must genuinely spread (measured at
    build: sector-level 0%→50% against a 12% base rate). A near-constant column is dead weight."""
    d = live.copy()
    d["industry"] = d["industry"].astype(str).str.strip()
    n = d.groupby("industry")["name"].count()
    share = d.groupby("industry")["wealth_tier"].apply(lambda s: 100.0 * (s == "CLIMBING★").mean())
    big = share[n >= 5]
    assert len(big) >= 30, f"only {len(big)} industries hold ≥5 stocks — threshold stale"
    assert big.max() - big.min() > 20.0, (
        f"CLIMBING★ share barely varies across industries ({big.min():.0f}–{big.max():.0f}%) — "
        f"the column no longer discriminates; re-measure before keeping it"
    )


# ── 8. Sector drill-down — a row filter, applied after every number (2026-08-28) ─────────────
def test_drill_down_is_applied_after_aggregation_and_sort(block):
    """Row-filter semantics: the drill must touch no average, Δ or 💹 share — so it must sit
    AFTER the aggregation, the delta, the share and the sort."""
    i = block.index('_ind_stats = _ind_stats[_ind_stats.index.isin(_ind_reach)]')
    assert block.index("_ind_stats = _ind_src.groupby") < i
    assert block.index('_ind_stats["delta_vs_sector"]') < i
    assert block.index('_ind_stats["pct_tier"]') < i
    assert block.index("_ind_stats.sort_values") < i


def test_drill_down_matches_every_home_not_only_the_displayed_sector(block):
    """UPDATED 2026-09-21 (§6). This pinned `_dom_sec...isin(_ind_sec)` — matching the ONE sector
    the table displays — and its docstring claimed a row "appears under its dominant home, never
    under a minority sector".

    The second half still holds and is pinned below. The first half was the defect: an industry
    split EXACTLY 3-3 has no single dominant home, and the alphabetical tie-break handed
    "Mining/Minerals - Iron Ore" to Mining & Mineral products, making it unreachable from Steel
    while half its stocks are Steel. The drill now matches `_ind_homes` — every sector tied for
    the maximum — so a tie is reachable from each, and a MINORITY sector still never matches.
    """
    assert "_ind_stats = _ind_stats[_ind_stats.index.isin(_ind_reach)]" in block, (
        "the drill-down no longer filters on the home map"
    )
    assert "_ind_home_map = _ind_homes(_ind_src)" in block, (
        "the drill-down's home map is not built from the frame the table aggregates"
    )
    assert "_dom_sec.reindex(_ind_stats.index).isin(_ind_sec)" not in block, (
        "the drill reverted to the single displayed sector — a tied industry becomes unreachable "
        "from half of its own stocks' sector again"
    )


def test_dom_share_is_index_aligned_for_the_drill(block):
    """_dom_share was a bare positional array; the drill row-filters _ind_stats, so the footer's
    ~ count would silently pair shares with the wrong industries. It must be an index-aligned
    Series, and the footer must reindex it to the drilled view."""
    assert "_dom_share = pd.Series(" in block, "_dom_share reverted to a positional array"
    assert "_dom_share.reindex(_ind_stats.index)" in block, "the footer no longer follows the drilled view"


def test_drill_options_are_sorted(block):
    """Determinism mandate: an unsorted unique() would reorder the dropdown per process."""
    i = block.index("_ind_sec_opts")
    assert "sorted(" in block[i:i + 140], "the drill-down options are not sorted"


# ── Multi-select + cascade (2026-08-30) ──────────────────────────────────────────────────────
def test_every_set_control_takes_several_values(block):
    """The user's request. `.isin` everywhere, and nowhere a scalar `==` that would silently
    honour only the first pick."""
    for var, col in [("_ind_cap", "market_category"), ("_ind_wt", "wealth_tier")]:
        assert f'["{col}"].isin({var})' in block, f"{var} is no longer applied as a set membership"
        assert f'["{col}"] == {var}' not in block, f"{var} still has a scalar comparison"
    assert ".isin(_ind_sec)" in block, "the drill-down takes only one sector again"


def test_the_drill_down_counts_industries_not_stocks(block, ms_help):
    """THE UNITS RULE. Market-cap and wealth are re-aggregating filters, so their facet counts are
    STOCKS; the drill-down hides INDUSTRY ROWS, so its count must be industries-by-dominant-sector.
    A stock count there would promise 412 rows and deliver 9."""
    i = block.index("_ind_sec_n =")
    seg = block[i:i + 200]
    assert "_ind_homes(" in seg, (
        "the drill-down count no longer reads the HOME map — and it must be the same map the "
        "filter uses, or a tied industry makes its option under-report with nothing failing"
    )
    assert "nunique()" in seg, "counting rows, not distinct industries, double-counts a tie"
    assert '_icf["sector"].astype(str).value_counts()' not in block, (
        "the drill-down count reverted to a straight stock count — wrong unit"
    )
    for k, unit in [("mp_ind_cap", "Counts are STOCKS"), ("mp_ind_wealth", "Counts are STOCKS"),
                    ("mp_ind_sec", "INDUSTRIES, not stocks")]:
        assert unit in ms_help[k], f"the {k} help text no longer names its unit ({unit})"


def test_the_two_units_really_are_far_apart(live):
    """Teeth: on live data a sector holds a median of 3 industries but tens of stocks, so
    mislabelling the drill-down count would be a large, not pedantic, error."""
    d = live.dropna(subset=["industry", "sector"])
    pair = (d.groupby(["industry", "sector"]).size().rename("n").reset_index()
            .sort_values(["industry", "n", "sector"], ascending=[True, False, True]))
    dom = pair.drop_duplicates("industry")
    inds = dom["sector"].value_counts()
    stocks = d["sector"].value_counts()
    common = inds.index.intersection(stocks.index)
    assert (stocks[common] / inds[common]).median() > 3, (
        "stocks-per-sector and industries-per-sector have converged — re-measure the units rule"
    )


def test_the_sector_map_is_computed_once_and_derived_three_ways(block):
    """UPDATED 2026-09-21 (§6). Was: one `_ind_dominant` helper, two call sites. The split into
    display (`_ind_dominant`) and reachability (`_ind_homes`) keeps the ORIGINAL subject — no two
    inline copies that can drift — by deriving BOTH from one `_ind_sector_pairs` table. If they
    ever came from separate groupbys, an option could promise rows the table does not show.
    """
    for fn in ("_ind_sector_pairs", "_ind_dominant", "_ind_homes"):
        assert block.count(f"def {fn}(") == 1, f"{fn} is missing or duplicated"
    assert block.count("_ind_sector_pairs(") == 3, (
        "expected ONE pair table feeding exactly the two derived views — if a view builds its own "
        "groupby the two can disagree"
    )
    i = block.index("def _ind_sector_pairs(")
    body = block[i:block.index("_i1, _i2, _i3, _i4", i)]
    assert "ascending=[True, False, True]" in body, "the deterministic tie-break left the helper"
    assert 'drop_duplicates("industry")' in body, (
        "the single-home reduction is gone — display and the Δ baseline need exactly one sector"
    )
    assert 'transform("max")' in body, (
        "the home map no longer selects every sector TIED for the maximum — that tie-free rule is "
        "what replaced an invented share threshold"
    )


def test_the_industry_cascade_narrows_left_to_right(block):
    """Options and counts come from `_icf` — the frame already narrowed by the controls to the
    left — so a dead-end combination cannot be offered."""
    i = block.index("_icf = _ind_src")
    seg = block[i:block.index('"mp_ind_sec"', i)]
    # GUARDED, not merely present: a mutation run put `if False:` above each narrowing line and an
    # existence check stayed green. Whitespace-collapsed so the pin is indentation-agnostic.
    flat = re.sub(r"\s+", " ", seg)
    for var, col in [("_ind_cap", "market_category"), ("_ind_wt", "wealth_tier")]:
        assert f'if {var}: _icf = _icf[_icf["{col}"].isin({var})]' in flat, (
            f"the {col} filter no longer narrows the cascade under its own selection"
        )
    assert seg.index('_icf[_icf["market_category"]') < seg.index('_icf["wealth_tier"].astype(str)'), (
        "wealth facets are counted before the market-cap narrowing — the cascade order broke"
    )
    assert "_ind_homes(_icf)" in block, "the drill-down counts read an un-narrowed frame"


# ── 6. TIED HOMES — an industry with no single dominant sector (2026-09-21) ────────────────
# THE USER FOUND THIS BY USING THE TAB: they picked Steel in the drill-down and
# "Mining/Minerals - Iron Ore" did not appear. Its 6 stocks split EXACTLY 3 Mining & Mineral
# products / 3 Steel, and the (count desc, sector ASC) tie-break handed it to Mining because
# "M" precedes "S". Half that industry is Steel and it was unreachable from Steel.
#
# 24 of 371 industries are exact ties, so 24 rows were filed by SPELLING — a deterministic key
# carrying no information. Worse, both help texts promised "the MAJORITY of stocks", which is
# false for the 41 industries whose dominant share is <= 50%.
#
# THE FIX IS THRESHOLD-FREE ON PURPOSE. An earlier draft of mine proposed "appear under any
# sector holding >= 40%" and that was the invented-threshold anti-pattern CLAUDE.md §5 exists to
# stop. "Every sector TIED FOR THE MAXIMUM" needs no number: a tie gets every home it earned, a
# clear winner keeps exactly one, and nothing has to be calibrated.
#
# DISPLAY AND THE PEER BASELINE STAY SINGLE-HOMED. _dom_sec also drives the leave-one-out
# Δ vs Sector arithmetic, and "which sector's peers?" has no defined answer for a 50/50 industry.
# Multi-homing the FILTER is cheap; multi-homing a computed number is not. The `~ <share>%` label
# is what makes the pair legible: a row reached from Steel but measured against Mining peers says
# so on its face.


def _extract_fn(block, name):
    """Lift a nested helper out of the tab source and make it CALLABLE.

    The Industry helpers live inside a `with` block, so they cannot be imported. Scanning their
    source for a substring is the trap this suite has paid for repeatedly (`if False and ...`
    satisfies any presence scan). Executing the real text runs the real code.
    """
    import textwrap
    lines = block.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(f"def {name}("):
            indent = len(line) - len(line.lstrip())
            body = [line]
            for nxt in lines[i + 1:]:
                if nxt.strip() and (len(nxt) - len(nxt.lstrip())) <= indent:
                    break
                body.append(nxt)
            ns = {"pd": pd, "np": np}
            # _ind_homes/_ind_dominant both call _ind_sector_pairs — pull the dependency in first
            # or the extracted function raises NameError at call time (it did, on the first run).
            if name != "_ind_sector_pairs":
                ns["_ind_sector_pairs"] = _extract_fn(block, "_ind_sector_pairs")
            exec(compile(textwrap.dedent("\n".join(body)), f"<{name}>", "exec"), ns)
            return ns[name]
    raise AssertionError(
        f"{name}() is not defined in the Industry block — the drill-down cannot multi-home a "
        f"tied industry without it"
    )


def _homes(df):
    """INDEPENDENT re-implementation (spec, not a copy): industry -> every sector tied for max."""
    d = df[["industry", "sector", "name"]].copy()
    d["industry"] = d["industry"].astype(str).str.strip()
    pair = d.groupby(["industry", "sector"]).size().rename("n").reset_index()
    return pair[pair["n"] == pair.groupby("industry")["n"].transform("max")]


def test_exact_ties_exist_so_this_section_is_not_vacuous(live):
    """If industry ever becomes a clean child of sector these pins stop meaning anything."""
    h = _homes(live)
    tied = h.groupby("industry").size()
    assert (tied > 1).sum() >= 5, (
        f"only {(tied > 1).sum()} industries tie for their top sector — re-measure before keeping "
        f"the multi-home rule"
    )


def test_the_real_helper_multi_homes_a_tie_and_leaves_a_clear_winner_alone(block):
    """BEHAVIOURAL, on the shipped code: a tie gets every home, a plurality gets exactly one."""
    homes = _extract_fn(block, "_ind_homes")
    frame = pd.DataFrame({
        "name":     [f"s{i}" for i in range(8)],
        # TieCo: 2 Alpha / 2 Zeta  ->  BOTH (and note Alpha would win an alphabetical tie-break)
        # ClearCo: 3 Alpha / 1 Zeta ->  Alpha only
        "industry": ["TieCo"] * 4 + ["ClearCo"] * 4,
        "sector":   ["Alpha", "Alpha", "Zeta", "Zeta", "Alpha", "Alpha", "Alpha", "Zeta"],
    })
    out = homes(frame)
    got = {i: sorted(g["sector"]) for i, g in out.groupby("industry")}
    assert got["TieCo"] == ["Alpha", "Zeta"], (
        f"a 2-2 tie must be reachable from BOTH sectors, got {got.get('TieCo')}"
    )
    assert got["ClearCo"] == ["Alpha"], (
        f"a 3-1 plurality must keep ONE home, got {got.get('ClearCo')} — this is the guard against "
        f"the rule degenerating into 'any sector it touches at all'"
    )


def test_a_tied_industry_is_reachable_from_every_sector_it_ties_in(live, block):
    """The user's case, generalised over live data — the app's helper must agree with the spec."""
    homes = _extract_fn(block, "_ind_homes")
    d = live.copy()
    d["industry"] = d["industry"].astype(str).str.strip()
    app = {i: set(g["sector"]) for i, g in homes(d).groupby("industry")}
    ref = {i: set(g["sector"]) for i, g in _homes(d).groupby("industry")}
    assert app == ref, "the tab's home map disagrees with an independent re-implementation"
    multi = {i: s for i, s in app.items() if len(s) > 1}
    assert multi, "no multi-homed industry on live data"
    for ind, secs in list(multi.items())[:50]:
        counts = d[d["industry"] == ind]["sector"].value_counts()
        assert len({counts[s] for s in secs}) == 1, (
            f"{ind} is multi-homed across {secs} whose counts differ — only an exact tie may "
            f"produce more than one home"
        )


def test_the_drill_down_count_equals_the_rows_the_pick_yields(live, block):
    """LOCKSTEP — the file's own promise: 'the number beside an option is exactly the number of
    rows it leaves'. Multi-homing breaks that the moment the counts and the filter disagree, and
    an off-by-one there is invisible (the table just shows a different number of rows)."""
    homes = _extract_fn(block, "_ind_homes")
    d = live.copy()
    d["industry"] = d["industry"].astype(str).str.strip()
    h = homes(d)
    counts = h.groupby("sector")["industry"].nunique().to_dict()
    for sec, n in sorted(counts.items())[:25]:
        yielded = h[h["sector"] == sec]["industry"].nunique()
        assert yielded == n, f"{sec}: option says {n} industries, the pick yields {yielded}"


def test_no_help_text_claims_a_majority_it_cannot_deliver(ms_help, block):
    """41 of 371 industries have a dominant share <= 50%, so 'MAJORITY' was simply false there.

    Two-sided: the word must be gone AND the accurate one present, or a future edit could drop
    both and leave the control undescribed.
    """
    drill = ms_help.get("mp_ind_sec", "")
    assert drill, "the sector drill-down has no help text"
    assert "majority" not in drill.lower(), (
        f"the drill-down still promises a MAJORITY: {drill!r}"
    )
    assert "largest share" in drill.lower(), "the drill-down must say what it actually matches on"
    col = block[block.index('"dom_sector"'):] if '"dom_sector"' in block else ""
    assert "MAJORITY of the industry" not in col, (
        "the Sector (dominant) column help still claims the majority sits there — false for the "
        "41 industries at or below a 50% dominant share"
    )


def test_an_ambiguous_row_prints_its_share_not_just_a_tilde(block):
    """The '~' said 'loosely'; a number says HOW loosely. Without it a row reached from Steel but
    labelled 'Mining & Mineral products' has nothing on screen explaining the pair."""
    i = block.index('_ind_stats["dom_sector"]')
    seg = block[i:i + 600]
    assert "_dom_share" in seg, (
        "the dominant-sector label is built without reference to the share — an industry at 50% "
        "and one at 79% render identically"
    )
    assert "%" in seg, "the ambiguous label carries no percentage"


def test_display_and_the_peer_baseline_stay_single_homed(block):
    """Multi-homing the FILTER must not leak into the computed Δ. 'Which sector's peers?' has no
    answer for a 50/50 industry, so _dom_sec stays one value per row and keeps driving it."""
    assert "drop_duplicates" in block, "the single-home reduction for display/baseline is gone"
    assert "_peer_sum = _dom_sec.map(" in block, (
        "the leave-one-out baseline no longer reads the single dominant sector — if it now reads "
        "the multi-home map, one industry contributes to several sector baselines"
    )
