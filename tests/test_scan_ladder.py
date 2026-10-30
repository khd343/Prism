"""
test_scan_ladder.py — the 🚀 ladder: two wider lists of the Playbook's own recipe, shown as plain lists (2026-10-30,
the operator's call).

On the 🎯 Scans tab the 🚀 Playbook Shortlist gains a "List" control with three layers of its recipe: 🚀 Playbook (all
six conditions of ui_scans.playbook_conditions, unchanged), B · without 🔥 (every condition but the 🔥 Strong Quarter)
and C · price only (Momentum and Breakout in the top third, the price above its stop). They are exactly the L0/L1/L2
lists of Kite's frozen ladder test (Kite v2 addendum 36), so B and C carry NO record: a record would publicly score
the comparison that test must judge blind until May 2027. They are no scan, so nothing that records, publishes or
fingerprints a scan can see them.

Pinned here: the 🚀 layer IS the shortlist; 🚀 ⊆ B ⊆ C; each layer read from playbook_conditions itself; the List tag
is the tightest layer, ranked on the whole market; the ladder is no scan; the B and C tables (List after Rank in both
views, no 🆕, 🔥 only where it says something, 🚀's 1,030px Core budget and its order, a Rank tip that claims no record);
🚀's table and 🌱's exactly as they were; where app.py draws the control; and the 📖 guide's one paragraph on it.
tests/test_scans_tab_runs.py runs it in the real tab.

Run with: pytest tests/test_scan_ladder.py -v
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.test_scans import _universe  # noqa: E402
from ui import ui_scans as S  # noqa: E402

CORE, TECH = S.SCAN_VIEWS
PLAY = next(s for s in S.SCANS if s["key"] == "playbook")
SEED = next(s for s in S.SCANS if s["key"] == "compounder")
WIDE = ("no_quarter", "price_only")                  # B and C: the layers with no record
_APP = (ROOT / "app.py").read_text(encoding="utf-8")


def _market(seed: int, n: int = 400) -> pd.DataFrame:
    """A seeded random market with holes: every input of the six conditions varies, about 8% of the recipe's numbers
    are missing (a missing value never passes a condition), and 5% of the stocks have no Improvement leg at all (no
    evidence: never in 🚀 or B, possible in C). Breakout follows momentum, as it does in the engine (a fifth of
    momentum is how close the price is to a breakout), so every rung of the ladder holds stocks."""
    rng = np.random.default_rng(seed)
    momentum, rev, pat_ly = rng.uniform(0, 100, n), rng.uniform(100, 160, n), rng.uniform(-1, 12, n)
    df = pd.DataFrame({
        "company_id": [f"NSE:R{k:03d}" for k in range(n)], "name": [f"Random {k:03d}" for k in range(n)],
        "sector": rng.choice(["Alpha", "Beta", "Gamma", "Delta"], n), "rank": rng.permutation(n) + 1,
        "composite_score": rng.uniform(20, 90, n), "market_category": rng.choice(["Small Cap", "Mid Cap"], n),
        "momentum_score": momentum, "breakout_score": 0.6 * momentum + rng.uniform(0, 40, n),
        "improvement_score": rng.uniform(0, 100, n), "dist_to_vstop": rng.normal(3, 6, n),
        "dist_52wh": rng.uniform(0, 40, n),
        "rev_pyq": 100.0, "rev_lq": rev, "pat_pyq": pat_ly, "pat_lq": pat_ly * rng.uniform(0.8, 1.8, n),
        "ebitda_pyq": 15.0, "ebitda_lq": rev * rng.uniform(0.12, 0.22, n),
        "pat_1yb": rng.uniform(-5, 50, n), "pat": rng.uniform(-5, 80, n), "opm_1yb": rng.uniform(5, 25, n),
        "opm": rng.uniform(5, 30, n), "revenue_1yb": rng.uniform(50, 500, n), "revenue": rng.uniform(50, 650, n),
        "vol_sma_20d": rng.uniform(1e4, 1e6, n), "close_price": rng.uniform(20, 900, n),
        "rf_wc_double_squeeze": 0, "rf_receivables_bloat": 0, "rf_negative_fcf": 0,
        "result_age_days": rng.integers(-5, 80, n), "weinstein_stage": "📈 Stage 2 Advancing",
    })
    for col in ("momentum_score", "breakout_score", "improvement_score", "dist_to_vstop", "rev_lq", "pat_lq"):
        df.loc[rng.random(n) < 0.08, col] = np.nan
    df.loc[rng.random(n) < 0.05, ["pat", "pat_1yb", "revenue", "revenue_1yb", "opm", "opm_1yb"]] = np.nan
    return df


def _no_strong_quarter():
    """tests/test_scans.py's universe with its one match's quarter made ordinary (revenue up 5%): out of 🚀, still B."""
    df = _universe()
    df.loc[df.index[-1], "rev_lq"] = 105.0
    return df


def _no_evidence():
    """tests/test_scans.py's universe with its one match's Improvement legs blank, as its
    test_no_improvement_evidence_never_passes has them: the score still ranks top, but on no evidence the stock is in
    no layer that reads Improvement (🚀, B) — and still in C, which does not."""
    df = _universe()
    df.loc[df.index[-1], ["pat", "pat_1yb", "revenue", "revenue_1yb", "opm", "opm_1yb"]] = np.nan
    return df


# The frames the recipe tests use (tests/test_scans.py::_universe) and two seeded random markets with holes.
_FRAMES = {"universe": _universe, "universe, no strong quarter": _no_strong_quarter,
           "universe, no Improvement evidence": _no_evidence,
           "random 7": lambda: _market(7), "random 20261030": lambda: _market(20261030)}


def _masks(df):
    return [S.playbook_layer_mask(df, layer["key"]) for layer in S.PLAYBOOK_LAYERS]


# ── 1. The three layers, each its conditions read from the recipe ────────────────────────────────────────────
def test_the_ladder_is_three_layers_tightest_first():
    assert [layer["key"] for layer in S.PLAYBOOK_LAYERS] == ["playbook", "no_quarter", "price_only"]
    assert [layer["tag"] for layer in S.PLAYBOOK_LAYERS] == ["🚀", "B", "C"]
    assert [layer["label"] for layer in S.PLAYBOOK_LAYERS] == ["🚀 Playbook", "B · without 🔥", "C · price only"]
    recipe = tuple(S.playbook_conditions(_universe()).columns)
    assert S.PLAYBOOK_LAYERS[0]["conditions"] == recipe, "🚀 is every condition of the recipe"
    assert S.PLAYBOOK_LAYERS[1]["conditions"] == tuple(c for c in recipe if c != "quarter")
    assert S.PLAYBOOK_LAYERS[2]["conditions"] == ("momentum", "breakout", "stop")


@pytest.mark.parametrize("frame", sorted(_FRAMES))
def test_the_rocket_layer_is_the_shortlist_itself(frame):
    df = _FRAMES[frame]()
    pd.testing.assert_series_equal(S.playbook_layer_mask(df, "playbook"), S.playbook_shortlist(df))


@pytest.mark.parametrize("frame", sorted(_FRAMES))
def test_each_list_holds_the_one_before_it(frame):
    rocket, b, c = _masks(_FRAMES[frame]())
    assert not (rocket & ~b).any(), "every 🚀 stock is in B"
    assert not (b & ~c).any(), "every B stock is in C"


@pytest.mark.parametrize("frame", sorted(_FRAMES))
def test_b_drops_only_the_quarter_and_c_reads_the_price_alone(frame):
    """Checked against playbook_conditions directly, never against the layer table the code reads."""
    df = _FRAMES[frame]()
    met = S.playbook_conditions(df)
    b = met["momentum"] & met["breakout"] & met["improvement"] & met["evidence"] & met["stop"]
    c = met["momentum"] & met["breakout"] & met["stop"]
    assert S.playbook_layer_mask(df, "no_quarter").tolist() == b.tolist()
    assert S.playbook_layer_mask(df, "no_quarter").tolist() == met.drop(columns="quarter").all(axis=1).tolist()
    assert S.playbook_layer_mask(df, "price_only").tolist() == c.tolist()


def test_the_test_markets_reach_every_rung():
    """Not vacuous: each random market holds stocks in 🚀, in B but not 🚀, in C but not B, and outside C; the
    universe without its strong quarter moves its one match from 🚀 to B, and without Improvement evidence to C."""
    for seed in (7, 20261030):
        rocket, b, c = _masks(_market(seed))
        assert rocket.any() and (b & ~rocket).any() and (c & ~b).any() and (~c).any(), seed
    rocket, b, c = _masks(_no_strong_quarter())
    assert not rocket.any() and b.tolist() == [False] * 11 + [True]
    rocket, b, c = _masks(_no_evidence())
    assert not rocket.any() and not b.any() and c.tolist() == [False] * 11 + [True]
    assert S.playbook_conditions(_no_evidence()).drop(columns="evidence").iloc[-1].all(), (
        "the match fails on evidence alone")


def test_a_missing_input_passes_no_layer():
    """Unverifiable is not passed: without a Breakout column no layer can be judged, so none matches."""
    df = _universe().drop(columns=["breakout_score"])
    for layer in S.PLAYBOOK_LAYERS:
        assert not S.playbook_layer_mask(df, layer["key"]).any(), layer["key"]


# ── 2. The List tag: the tightest layer, ranked on the whole market ───────────────────────────────────────────
def test_the_list_tag_is_the_tightest_layer_and_blank_outside_c():
    df = _universe()
    df.loc[10, "dist_to_vstop"] = 5.0                                  # every leg but the quarter: B
    df.loc[9, ["dist_to_vstop", "improvement_score"]] = [5.0, 0.0]     # momentum, breakout and the stop only: C
    tag = S.playbook_layer_tag(df)
    assert tag.tolist() == [""] * 9 + ["C", "B", "🚀"]
    assert tag.index.equals(df.index)


@pytest.mark.parametrize("frame", sorted(_FRAMES))
def test_the_tag_is_the_first_layer_met_tightest_first(frame):
    df = _FRAMES[frame]()
    want = np.select([m.to_numpy() for m in _masks(df)], ["🚀", "B", "C"], default="")
    assert S.playbook_layer_tag(df).tolist() == want.tolist()


def test_the_list_column_is_ranked_on_the_whole_market_never_the_narrowed_list():
    """The recipe's own rule (tests/test_scans.py): ranking inside a narrowed list re-ranks every percentile — of
    three B stocks kept, the one with the lowest momentum would fall out of 'the top third' and lose its tag."""
    df = _market(7)
    whole = dict(zip(df["company_id"], S.playbook_layer_tag(df)))
    b_ids = sorted(df.loc[S.playbook_layer_mask(df, "no_quarter"), "company_id"])
    keep = set(b_ids[:3])
    t = S.scan_table(df, S.playbook_layer_scan(PLAY, "no_quarter"), keep_ids=keep)
    assert dict(zip(t["company_id"], t["_layer"])) == {i: whole[i] for i in keep}
    t = S.scan_table(df, S.playbook_layer_scan(PLAY, "price_only"), min_cr=2.0)
    assert 0 < len(t) < int(S.playbook_layer_mask(df, "price_only").sum()), "the floor must really narrow"
    assert dict(zip(t["company_id"], t["_layer"])) == {i: whole[i] for i in t["company_id"]}


# ── 3. No record can see the ladder ───────────────────────────────────────────────────────────────────────────
# Every reader of a scan's record goes through SCANS: the weekly record job, the local record tool (local-only, so
# checked where it exists), the Playbook feed and Kite's v2 lists.
_RECORD_READERS = ("ui/scan_records_job.py", "ui/playbook_feed_job.py", "ui/v2_lists_job.py", "tools/scan_records.py")


def test_the_ladder_is_no_scan_so_no_record_can_see_it():
    """B and C are Kite's ladder-test lists (addendum 36): a record would publicly score the comparison that test
    must judge blind."""
    assert [s["key"] for s in S.SCANS] == ["playbook", "leaders", "compounder"], "the scans, unchanged"
    assert not set(WIDE) & {s["key"] for s in S.SCANS}
    assert not set(WIDE) & set(S.load_records().get("scans", {})), "scan_records.json holds no B or C record"
    for key in WIDE:
        assert all(S.playbook_layer_scan(PLAY, key) is not s for s in S.SCANS)
    read = 0
    for rel in _RECORD_READERS:
        path = ROOT / rel
        if not path.exists():
            continue
        read += 1
        src = path.read_text(encoding="utf-8")
        for word in ("PLAYBOOK_LAYERS", "playbook_layer", *WIDE):
            assert word not in src, f"{rel} reads the ladder ({word})"
    assert read >= 3, "the tracked record readers were read"


def test_the_rocket_layer_is_the_scan_itself():
    assert S.playbook_layer_scan(PLAY, "playbook") is PLAY
    keys = [S.playbook_layer_scan(PLAY, layer["key"])["key"] for layer in S.PLAYBOOK_LAYERS]
    assert keys == ["playbook", *WIDE], "each list is its own table, and 🚀's is the scan's own"


# ── 4. The B and C tables ─────────────────────────────────────────────────────────────────────────────────────
_WIDE_CORE = ("rank", "_layer", "name", "market_category", "sector", "composite_score", "momentum_score",
              "improvement_score", "_strong_q", "_measured_flags", "weinstein_stage", "_liq_cr")
_WIDE_TECH = ("rank", "_layer", "name", "close_price", "_liq_cr", "dist_to_vstop", "momentum_score",
              "d49_momentum_quality", "rsi_14d", "breakout_score", "crs_52w", "weinstein_stage")


@pytest.mark.parametrize("key", WIDE)
def test_b_and_c_show_the_list_after_rank_and_no_new_mark(key):
    wide = S.playbook_layer_scan(PLAY, key)
    want = _WIDE_CORE if key == "price_only" else tuple(c for c in _WIDE_CORE if c != "_strong_q")
    assert S.table_columns(wide, CORE) == want
    assert S.table_columns(wide, TECH) == _WIDE_TECH
    for view in S.SCAN_VIEWS:
        cols = S.table_columns(wide, view)
        assert cols[cols.index("rank") + 1] == "_layer" and "_new" not in cols, view


def test_the_quarter_column_shows_only_where_it_says_something():
    """B lacks only 🚀's 🔥 Strong Quarter, so on B its ✓ falls exactly on the rows List marks 🚀: the column says
    nothing new, and on screen (2026-10-30) its 40px cut B's stock and sector names. So B hides it, as 🚀 does, and
    takes 🚀's widths; on C it is information (test_the_quarter_column_varies_on_the_price_only_list)."""
    b = S.playbook_layer_scan(PLAY, "no_quarter")
    t = S.scan_table(_market(7), b)
    assert len(t) > 0 and (t["_strong_q"] == "✓").tolist() == (t["_layer"] == "🚀").tolist()
    assert "_strong_q" not in S.table_columns(b, CORE)
    assert "_strong_q" in S.table_columns(S.playbook_layer_scan(PLAY, "price_only"), CORE)


@pytest.mark.parametrize("key", WIDE)
def test_b_and_c_keep_the_rockets_core_budget(key):
    """List takes New's 55px. On B, 🔥 is hidden and its 40px go to Stock and Sector with 🚀's own widths; on C, 🔥
    keeps them and Stock and Sector take the shared widths."""
    wide = S.playbook_layer_scan(PLAY, key)
    w = S.table_widths(wide)
    rocket = sum(S.table_widths(PLAY)[c] for c in S.table_columns(PLAY, CORE))
    assert sum(w[c] for c in S.table_columns(wide, CORE)) == rocket == 1030
    assert w["_layer"] == S.TABLE_WIDTHS["_new"] == 55
    own = S.table_widths(PLAY if key == "no_quarter" else {})
    assert (w["name"], w["sector"]) == (own["name"], own["sector"]), "B as wide as 🚀's names; C the shared widths"
    assert sum(w[c] for c in S.table_columns(wide, TECH)) + 44 <= 1074, "📈 Technical fits a 1536px laptop"
    cfg = S.table_config("30 Sep", wide)
    for view in S.SCAN_VIEWS:
        for c in S.table_columns(wide, view):
            assert c in cfg and cfg[c]["width"] == w[c], (view, c)


def test_the_note_says_what_the_list_is_and_that_it_has_no_record():
    """The line under the control while B or C is chosen (app.py draws it): what the list is, that it has no record of
    its own (so the 📊 record above is 🚀's), and which test judges it. No internal addendum number and no promise of
    a record after May (the review of 2026-10-30). ONE line on a 1536px laptop: B's 148 characters fit there and C's
    200 wrapped (measured 2026-10-30), so each note keeps within 148."""
    notes = {layer["key"]: layer["note"] for layer in S.PLAYBOOK_LAYERS}
    assert notes["playbook"] == "", "🚀 is the scan its record measured: no note"
    assert notes["no_quarter"].startswith("B — ") and "without the 🔥 Strong Quarter" in notes["no_quarter"]
    assert notes["price_only"].startswith("C — ") and "Momentum and Breakout in the top third" in notes["price_only"]
    assert "above its stop" in notes["price_only"]
    for key in WIDE:
        for words in ("No record of its own", "Kite's ladder test", "May 2027"):
            assert words in notes[key], (key, words)
        assert "(36)" not in notes[key] and "until" not in notes[key], f"{key}: no number, no promise of a record"
        assert len(notes[key]) <= 148, f"{key}: one line under the control ({len(notes[key])} characters)"


def test_the_list_header_explains_every_letter():
    cfg = S.table_config("30 Sep", S.playbook_layer_scan(PLAY, "no_quarter"))["_layer"]
    assert cfg["label"] == "List"
    for layer in S.PLAYBOOK_LAYERS:
        assert f"{layer['tag']} = " in cfg["help"], layer["tag"]
    for words in ("Strong Quarter", "top third", "stop", "no record", "May 2027", "blind"):
        assert words in cfg["help"], words
    assert "(36)" not in cfg["help"], "the screen names no internal addendum number"


@pytest.mark.parametrize("key", WIDE)
def test_the_rank_tip_on_b_and_c_claims_no_record_of_its_own(key):
    """B and C take 🚀's order but not its record: their Rank tip names no order check and no 2026-10-06 test (both
    are 🚀's alone; the review of 2026-10-30 found 🚀's tip on them), while 🚀's own tip keeps both."""
    tip = S.table_config("30 Sep", S.playbook_layer_scan(PLAY, key))["rank"]["help"]
    assert "order check" not in tip and "2026-10-06" not in tip, tip
    assert "no record" in tip and "Momentum, high to low" in tip, tip
    rocket = S.table_config("30 Sep", PLAY)["rank"]["help"]
    assert "order check" in rocket and "2026-10-06" in rocket


@pytest.mark.parametrize("key,tags", [("no_quarter", {"🚀", "B"}), ("price_only", {"🚀", "B", "C"})])
def test_b_and_c_list_every_match_by_momentum_and_never_mark_new(key, tags):
    df = _market(7)
    wide = S.playbook_layer_scan(PLAY, key)
    t = S.scan_table(df, wide, members_at_latest=None, result_window_days=30)
    assert sorted(t["company_id"]) == sorted(df.loc[S.playbook_layer_mask(df, key), "company_id"])
    assert not t["_new"].str.contains("🆕").any(), "no record, so no baseline: nothing is called new"
    assert S.momentum_ordered(wide)
    assert list(t["momentum_score"]) == sorted(t["momentum_score"], reverse=True), "🚀's order: momentum, high to low"
    assert set(t["_layer"]) == tags, "each row shows its tightest list"
    assert set(t.loc[t["_layer"] == "🚀", "_strong_q"]) == {"✓"}


def test_the_quarter_column_varies_on_the_price_only_list():
    t = S.scan_table(_market(7), S.playbook_layer_scan(PLAY, "price_only"))
    assert set(t["_strong_q"]) == {"✓", ""}, "on C a 🔥 Strong Quarter is information, not a given"


# ── 5. 🚀's table and 🌱's, exactly as they were ───────────────────────────────────────────────────────────────
_ROCKET_CORE = ("rank", "_new", "name", "market_category", "sector", "composite_score", "momentum_score",
                "improvement_score", "_measured_flags", "weinstein_stage", "_liq_cr")
_TECH = ("rank", "_new", "name", "close_price", "_liq_cr", "dist_to_vstop", "momentum_score", "d49_momentum_quality",
         "rsi_14d", "breakout_score", "crs_52w", "weinstein_stage")
_ROCKET_WIDTHS = {"rank": 50, "_new": 55, "name": 185, "market_category": 75, "sector": 125, "composite_score": 75,
                  "momentum_score": 85, "improvement_score": 100, "_measured_flags": 125, "weinstein_stage": 90,
                  "_liq_cr": 65, "close_price": 80, "dist_to_vstop": 60, "d49_momentum_quality": 110, "rsi_14d": 50,
                  "breakout_score": 85, "crs_52w": 60}
# Every column scan_table hands the grid for a scan (each view picks from them), as before the ladder.
_FRAME_COLUMNS = ("rank", "_new", "name", "market_category", "sector", "composite_score", "momentum_score",
                  "improvement_score", "_strong_q", "_measured_flags", "weinstein_stage", "_liq_cr", "close_price",
                  "dist_to_vstop", "d49_momentum_quality", "rsi_14d", "breakout_score", "crs_52w", "_cycle",
                  "company_id")


def test_the_rockets_table_is_the_table_it_was():
    assert S.table_columns(PLAY, CORE) == _ROCKET_CORE and S.table_columns(PLAY, TECH) == _TECH
    w = S.table_widths(PLAY)
    assert {c: w[c] for c in (*_ROCKET_CORE, *_TECH)} == _ROCKET_WIDTHS
    df = _market(7)
    members = set(df["company_id"][::2])
    for scan in S.SCANS:
        t = S.scan_table(df, scan, members_at_latest=members, result_window_days=30)
        assert tuple(t.columns) == _FRAME_COLUMNS, f"{scan['key']}: the table carries no List column"
    t = S.scan_table(df, PLAY, members_at_latest=members, result_window_days=30)
    want = df[S.playbook_shortlist(df)].sort_values(["momentum_score", "rank"], ascending=[False, True],
                                                    kind="mergesort")
    assert t["company_id"].tolist() == want["company_id"].tolist() and len(t) > 0
    assert t["_new"].str.startswith("🆕").tolist() == (~want["company_id"].isin(members)).tolist()


def test_the_compounder_columns_are_unchanged_in_both_views():
    assert S.table_columns(SEED, CORE) == ("rank", "_new", "name", "market_category", "sector", "composite_score",
                                           "momentum_score", "improvement_score", "_strong_q", "_cycle",
                                           "_measured_flags", "weinstein_stage", "_liq_cr")
    assert S.table_columns(SEED, TECH) == _TECH, (
        "Cycle ↓ stays out of 📈 Technical: it sits after 🔥, which Technical does not show")


# ── 6. Where app.py draws it ──────────────────────────────────────────────────────────────────────────────────
def _scans_block():
    return next(n for n in ast.parse(_APP).body if isinstance(n, ast.With)
                and ast.unparse(n.items[0].context_expr) == "tabs[1]")


def test_the_list_sits_between_the_scan_and_its_filters_on_the_rocket_only():
    """scan → list → filters → columns → table: after 📖 How it works, before the 🎯 opt-in, the Shariah line and
    the filter row; drawn only on 🚀, with its counts read on the whole market."""
    block = _scans_block()
    calls = [n for n in ast.walk(block) if isinstance(n, ast.Call)]
    seg = [n for n in calls if ast.unparse(n.func) == "st.segmented_control" and ast.unparse(n.args[0]) == "'List'"]
    assert len(seg) == 1
    kw = {k.arg: ast.unparse(k.value) for k in seg[0].keywords}
    assert kw["key"] == "'scan_layer'" and "format_func" in kw
    guide = next(n for n in calls if ast.unparse(n.func) == "st.expander" and "How it works" in ast.unparse(n))
    filters = [n for n in calls if ast.unparse(n.func) in ("st.checkbox", "st.selectbox", "_scan_filter_row")
               or ast.unparse(n) == "st.columns([2.1, 7.9])"]
    assert len(filters) == 4
    assert guide.lineno < seg[0].lineno < min(n.lineno for n in filters)
    guard = next(n for n in ast.walk(block) if isinstance(n, ast.If) and seg[0] in ast.walk(n))
    assert ast.unparse(guard.test) == "_sc_sel['key'] == 'playbook'"
    src = ast.unparse(block)
    assert "playbook_layer_mask(df, " in src and "playbook_layer_mask(filt" not in src, "counts on the whole market"
    assert "_sc_tbl_scan = playbook_layer_scan(_sc_sel, _sc_layer)" in src
    assert "scan_tbl_{_sc_tbl_scan['key']}_{_sc_view}_{_sc_ids}" in src.replace('"', "'"), (
        "the widget key follows the list, so a row pick on one list never carries into another")
    assert "_sc_sig = f'{_sc_tbl_scan['key']}|{_sc_ids}|{_sc_name}'" in src.replace('"', "'"), (
        "a row pick is keyed on the list shown: keyed on 🚀's, the same stock picked again on B after a filter left B "
        "showing 🚀's stocks repeats the signature, and the click silently does nothing (review 2026-10-30, M5)")


def test_the_list_is_a_choice_the_clear_never_resets():
    """The tab's 🧹 Clear empties the filters, never the list; and nothing in app.py writes it. (Leaving 🚀 does reset
    it: tests/test_scans_tab_runs.py.)"""
    tree = ast.parse(_APP)
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == "_SCAN_FILTER_DEFAULTS")
    assert "scan_layer" not in ast.literal_eval(node.value)
    writes = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Assign)
              and any("scan_layer" in ast.unparse(t) for t in n.targets)]
    assert not writes, writes


# ── 7. The 📖 guide stays 🚀's, and says so ────────────────────────────────────────────────────────────────────
def test_the_rocket_guide_names_the_wider_lists_and_says_they_have_no_record():
    """The 🚀 guide (ui/scan_guide.py) gains one dated paragraph in its `changes`. Prose is not in the guide's stamp
    (tests/test_scan_guide.py), so it is pinned here to the labels the List control shows and to the words it defines
    them with: rename a layer and this fails; redefine one and the definition tests above fail — re-read the paragraph
    either way."""
    from ui import scan_guide as G
    para = [p for p in G.GUIDES["playbook"]["changes"] if "List control" in p]
    assert len(para) == 1, "one paragraph on the ladder"
    for layer in S.PLAYBOOK_LAYERS[1:]:
        assert f"**{layer['label']}**" in para[0], layer["label"]
    for words in ("every condition but the 🔥 Strong Quarter", "Momentum and Breakout in the top third",
                  "above its stop", "Neither has a record", "Kite's ladder test (36)", "May 2027"):
        assert words in para[0], words
    shown = G.guide_html(PLAY)
    assert all(layer["label"] in shown for layer in S.PLAYBOOK_LAYERS[1:]), "the paragraph reaches the 📖 panel"
