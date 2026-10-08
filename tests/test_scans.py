"""
test_scans.py
=============
Contract for the 🎯 Scans tab (2026-10-06): measured stock lists, one definition each.

WHY IT EXISTS. The 2026-10-06 measurement sprint (docs/december-2026-plan.md section 8) found two
recipes that beat the market in all four windows on the saved snapshots: the playbook shortlist and
the stocks of the leading-breadth sectors. A screener usually shows a list with no evidence; this
tab shows each list beside ITS OWN record. Three rules carry that promise, and each is pinned here:

1. ONE DEFINITION. Each scan's recipe is one function in ui/ui_scans.py. The tab calls it and the
   local record tool (tools/scan_records.py) imports the same function, so the record always
   describes exactly the stocks on screen.
2. A RECORD NEVER OUTLIVES ITS RECIPE. ui/scan_records.json stores a fingerprint of the recipe it
   measured. If the recipe changes, the card shows no numbers until it is re-measured, and the
   committed file must match the current recipes or this suite fails.
3. NOTHING HIDDEN, NOTHING PROMISED. Flagged stocks stay in the list with their flags shown; no
   'buy' wording; a missing value never passes a condition (unverifiable is not passed).

MEASURED CHOICES: the combined 'shortlist + leading sectors' scan was tested and failed (14–29
stocks, weaker than the shortlist alone), so it has no card; inside the shortlist the top 25 by Rank
did not beat the rest (+0.6 / +2.3 / −1.1 / −1.7), so the list shows every match, sorted by Rank for
reading only.
"""
import ast
import inspect
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from ui import ui_scans as S  # noqa: E402


# ── Toy universe ──────────────────────────────────────────────────────────────────────────────────
def _universe(n=12):
    """n stocks; the LAST one is strong on every leg of the playbook recipe, so it is the one match.
    Quarterly levels make it a 🔥 Strong Quarter (revenue +30%, profit +50%, margin 15% -> 20%)."""
    i = np.arange(n, dtype=float)
    df = pd.DataFrame({
        "company_id": [f"NSE:S{k:02d}" for k in range(n)],
        "name": [f"Stock {k:02d}" for k in range(n)],
        "sector": ["Alpha"] * n,
        "rank": np.arange(n, 0, -1),                      # the best stock ranks 1
        "composite_score": 40 + i,
        "momentum_score": i, "breakout_score": i, "improvement_score": i,
        "dist_to_vstop": np.where(i == n - 1, 5.0, -1.0),
        "rev_pyq": 100.0, "rev_lq": np.where(i == n - 1, 130.0, 101.0),
        "pat_pyq": 10.0, "pat_lq": np.where(i == n - 1, 15.0, 10.0),
        "ebitda_pyq": 15.0, "ebitda_lq": np.where(i == n - 1, 26.0, 15.0),
        "weinstein_stage": "📈 Stage 2 Advancing",
        "vol_sma_20d": 200_000.0, "close_price": 500.0,   # ₹10 Cr a day
        "rf_wc_double_squeeze": 0, "rf_receivables_bloat": 0, "rf_negative_fcf": 0,
    })
    return df


# ── 1. The playbook recipe: every leg is required, nothing passes on a missing value ───────────────
def test_the_strong_stock_matches_and_no_other_does():
    m = S.playbook_shortlist(_universe())
    assert m.tolist() == [False] * 11 + [True]


@pytest.mark.parametrize("col,value", [
    ("momentum_score", -1.0), ("breakout_score", -1.0), ("improvement_score", -1.0),
    ("dist_to_vstop", -0.1),
    ("rev_lq", 105.0),                       # revenue up only 5%: not a strong quarter
    ("pat_pyq", -5.0),                       # a loss a year ago: cannot be judged, never strong
    ("momentum_score", np.nan), ("breakout_score", np.nan), ("improvement_score", np.nan),
    ("dist_to_vstop", np.nan), ("rev_lq", np.nan),
])
def test_each_leg_is_required(col, value):
    df = _universe()
    df.loc[df.index[-1], col] = value
    assert not S.playbook_shortlist(df).any(), f"{col}={value} must take the stock out of the scan"


def test_a_missing_column_matches_nothing():
    """Unverifiable is not passed: an archived vintage without a column yields an empty scan."""
    df = _universe().drop(columns=["improvement_score"])
    assert not S.playbook_shortlist(df).any()


def test_the_improvement_leg_is_the_top_half_not_the_top_third():
    df = _universe()
    df["improvement_score"] = np.where(df.index == df.index[-1], 6.0, df["improvement_score"])
    # 6 of 0..11 ranks 7th of 12 (pct 0.58): top half, not top third
    assert S.playbook_shortlist(df).iloc[-1]


# ── 2. Leading sectors: the Sectors tab's own breadth, sectors of 5+ stocks ───────────────────────
def _sectors():
    """Five sectors of 5 stocks with descending breadth, plus a 4-stock sector that would top both
    shares — too small to rank. Breadth = share within 5% of the 52-week high and share with a
    strong quarter (the Sectors tab's grp_near_high / grp_strong_quarter)."""
    rows = []
    near = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0, "Tiny": 4}
    strong = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0, "Tiny": 4}
    for sec, k in near.items():
        n = 4 if sec == "Tiny" else 5
        for j in range(n):
            s = j < strong[sec]
            rows.append(dict(company_id=f"{sec}{j}", name=f"{sec}{j}", sector=sec,
                             dist_52wh=2.0 if j < k else 30.0,
                             rev_pyq=100.0, rev_lq=130.0 if s else 101.0,
                             pat_pyq=10.0, pat_lq=15.0 if s else 10.0,
                             ebitda_pyq=15.0, ebitda_lq=26.0 if s else 15.0))
    return pd.DataFrame(rows)


def test_leading_sectors_ranks_breadth_and_skips_small_sectors():
    assert S.leading_sectors(_sectors()) == ["A", "B", "C", "D", "E"]


def test_the_leading_sector_scan_is_every_stock_in_those_sectors():
    df = _sectors()
    m = S.leading_sector_stocks(df)
    assert set(df.loc[m, "sector"]) == {"A", "B", "C", "D", "E"}
    assert int(m.sum()) == 25, "every stock of a leading sector belongs to the scan, not only its leaders"


def test_leading_sectors_is_deterministic_on_ties():
    df = _sectors()
    tie = ["dist_52wh", "rev_lq", "pat_lq", "ebitda_lq"]          # E copies D: an exact tie on both shares
    df.loc[df["sector"] == "E", tie] = df.loc[df["sector"] == "D", tie].values
    first = S.leading_sectors(df)
    assert first == S.leading_sectors(df.sample(frac=1.0, random_state=7))
    assert first[:3] == ["A", "B", "C"] and set(first[3:]) == {"D", "E"}


# ── 3. A record never outlives its recipe ─────────────────────────────────────────────────────────
def _recipe_a(df):
    """Docstring A."""
    return df["x"] > 1


def _recipe_b(df):
    """Docstring B — different words, same logic."""
    return df["x"] > 1


def _recipe_c(df):
    """Docstring A."""
    return df["x"] > 2


def test_the_fingerprint_follows_the_logic_not_the_words():
    fp = S._fingerprint_of
    assert fp([_recipe_a]) == fp([_recipe_b]), "a docstring edit must not invalidate a record"
    assert fp([_recipe_a]) != fp([_recipe_c]), "a threshold change must invalidate the record"


def test_every_scan_fingerprints_the_engine_pieces_it_reads():
    names = {s["key"]: {f.__name__ for f in S._recipe_functions(s)} for s in S.SCANS}
    assert {"playbook_shortlist", "strong_quarter", "_pct", "_num"} <= names["playbook"]
    assert {"leading_sector_stocks", "leading_sectors", "group_measures", "strong_quarter"} <= names["leaders"]


# ── 3b. The fingerprint follows the engine behind the columns (2026-10-07) ───────────────────────
# Until 2026-10-07 the fingerprint covered the recipe's own code only: re-defining how momentum_score
# or improvement_score is computed changed the list on screen while the card still read "measured".
def _deps(scan):
    return set(S._dependencies([*S._recipe_functions(scan), *scan.get("inputs", ())],
                               scan.get("values"), scan.get("skip", ())))


def _scan(key):
    return next(s for s in S.SCANS if s["key"] == key)


def test_the_playbook_fingerprint_reaches_the_scores_it_reads():
    d = _deps(_scan("playbook"))
    for piece in ("fn core.scoring_engine.compute_momentum_score", "fn core.scoring_engine._compute_breakout_score",
                  "fn core.scoring_engine._compute_rs_score", "fn core.scoring_engine._compute_improvement_score",
                  "fn core.scoring_engine._pct_rank", "value MOMENTUM_WEIGHTS", "value BREAKOUT_SIGNALS"):
        assert piece in d, f"the Playbook record does not follow {piece}"
    assert not any(k.startswith("fn core.data_engine") for k in d), (
        "compute_derived_signals (dist_to_vstop) is deliberately not followed — see SCANS")


def test_a_changed_score_weight_or_helper_stales_the_playbook_record(monkeypatch):
    from core import scoring_engine as E
    play = _scan("playbook")
    before = S.recipe_fingerprint(play)
    monkeypatch.setattr(E, "BREAKOUT_SIGNALS", {**E.BREAKOUT_SIGNALS, "ath_distance": 0.5})
    assert S.recipe_fingerprint(play) != before, "a re-weighted breakout score must stale the record"
    monkeypatch.undo()
    assert S.recipe_fingerprint(play) == before
    monkeypatch.setattr(E, "_pct_rank", lambda series, ascending=True: series.rank(pct=True) * 99)
    assert S.recipe_fingerprint(play) != before, "a changed ranking helper must stale the record"


def test_the_leaders_fingerprint_follows_its_two_measures_not_the_whole_table(monkeypatch):
    from ui.ui_group_views import MEASURES, _share
    lead = _scan("leaders")
    before = S.recipe_fingerprint(lead)
    monkeypatch.setitem(MEASURES, "grp_new_column", dict(source="x", stat="share", value=lambda f: f["x"] > 0))
    assert S.recipe_fingerprint(lead) == before, "a new Sectors column must not stale this record"
    monkeypatch.setitem(MEASURES["grp_near_high"], "value",
                        lambda f: _share(f["dist_52wh"] <= 10, f["dist_52wh"].notna()))
    assert S.recipe_fingerprint(lead) != before, "a changed near-high threshold must stale it"
    monkeypatch.undo()
    # Same module, same name ("<lambda>"): only their logic tells the two measures apart.
    monkeypatch.setitem(MEASURES["grp_near_high"], "value", MEASURES["grp_strong_quarter"]["value"])
    assert S.recipe_fingerprint(lead) != before, "two Sectors measures must not hash as one"


def test_the_fingerprint_is_the_same_on_every_run():
    """Sets and dicts are sorted before hashing: the weekly job and the local tool must agree."""
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); from ui import ui_scans as S; "
            "print(','.join(S.recipe_fingerprint(s) for s in S.SCANS))" % ROOT)
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": seed}).stdout.strip().splitlines()[-1]
            for seed in ("1", "2", "3")}
    assert len(outs) == 1, outs


def test_record_states(tmp_path):
    scan = S.SCANS[0]
    assert S.record_state({}, scan) == ("unmeasured", None)
    good = {"scans": {scan["key"]: {"fingerprint": S.recipe_fingerprint(scan), "windows": []}}}
    assert S.record_state(good, scan)[0] == "measured"
    stale = {"scans": {scan["key"]: {"fingerprint": "0" * 16, "windows": []}}}
    assert S.record_state(stale, scan)[0] == "stale"


def test_a_bad_records_file_loads_as_no_records(tmp_path):
    p = tmp_path / "r.json"
    assert S.load_records(p) == {}
    p.write_text("{not json", encoding="utf-8")
    assert S.load_records(p) == {}
    p.write_text(json.dumps({"version": S.RECORD_VERSION + 1, "scans": {}}), encoding="utf-8")
    assert S.load_records(p) == {}


def test_the_committed_records_match_the_current_recipes():
    """The file the live app reads. Regenerate with tools/scan_records.py after a recipe change or a
    new snapshot; a stale file fails here rather than showing an old record beside a new list."""
    rec = S.load_records()
    assert rec, f"{S.RECORDS_PATH.name} is missing or unreadable"
    assert rec["version"] == S.RECORD_VERSION
    for scan in S.SCANS:
        r = rec["scans"].get(scan["key"])
        assert r, f"no record for {scan['key']}"
        assert r["fingerprint"] == S.recipe_fingerprint(scan), (
            f"{scan['key']}: the recipe changed since it was measured — run tools/scan_records.py")
        if "study" in r:                       # 🌱: a study, not saved windows (2026-10-07)
            st_ = r["study"]
            assert r["windows"] == [], f"{scan['key']}: a study record carries no saved windows"
            assert r["designed_on"] == S.scan_designed_on(scan) and r["horizon_days"] == S.scan_horizon(scan)
            assert st_["n_match"] >= 30 and st_["n_rest"] >= 30
            assert 0 < st_["match_doubled_pct"] < 100 and 0 < st_["rest_doubled_pct"] < 100
            assert st_["caveats"] and st_["definition_note"] and st_["source"], (
                "a study record states its caveats, how it differs from the live recipe, and its source")
        else:
            assert len(r["windows"]) >= 2
            for w in r["windows"]:
                assert w["days"] >= 25 and w["start"] < w["end"], w
                assert w["n"] >= 30, f"{scan['key']} {w['start']}->{w['end']}: n={w['n']} is too few to show"
                assert 0 <= w["beat_pct"] <= 100
            assert r["members_at_latest"], f"{scan['key']}: no 🆕 baseline"
        # 🌱's baseline arrives with the weekly job's first run after it was added; when present it is
        # the same shape as every other scan's.
        members = r.get("members_at_latest", [])
        assert members == sorted(set(members)) and all(isinstance(x, str) for x in members)
    assert rec["latest_snapshot"] >= max(w["end"] for r in rec["scans"].values() for w in r["windows"])


# ── 4. The list: sorted by Rank, nothing hidden, new names marked, liquidity is a visible choice ───
def test_the_list_keeps_flagged_stocks_and_names_their_flags():
    df = _universe()
    for c in ("rf_wc_double_squeeze", "rf_negative_fcf"):
        df.loc[df.index[-1], c] = 1
    t = S.scan_table(df, S.SCANS[0])
    assert len(t) == 1, "a flagged stock is shown, never removed"
    assert t["_measured_flags"].iloc[0] == "Squeeze · Neg FCF"


def test_the_list_is_sorted_by_rank_and_marks_new_names():
    df = pd.concat([_universe()] * 1, ignore_index=True)
    df["rank"] = [5, 9, 1, 7, 3, 12, 11, 2, 10, 6, 8, 4]
    t = S.scan_table(df, {"key": "all", "mask": lambda d: pd.Series(True, index=d.index)},
                     members_at_latest={"NSE:S00", "NSE:S01"})
    assert t["rank"].tolist() == sorted(t["rank"].tolist())
    new = dict(zip(t["company_id"], t["_new"]))
    assert new["NSE:S00"] == "" and new["NSE:S01"] == "" and new["NSE:S02"] == "🆕"


def test_no_new_marks_without_a_record():
    t = S.scan_table(_universe(), {"key": "all", "mask": lambda d: pd.Series(True, index=d.index)},
                     members_at_latest=None)
    assert (t["_new"] == "").all(), "with no record there is no baseline, so nothing is called new"


def test_the_liquidity_floor_is_applied_after_the_recipe_and_drops_unknowns():
    df = _universe()
    every = {"key": "all", "mask": lambda d: pd.Series(True, index=d.index)}
    df.loc[0, "vol_sma_20d"] = 1_000.0          # ₹0.05 Cr a day
    df.loc[1, "vol_sma_20d"] = np.nan           # unknown
    assert len(S.scan_table(df, every)) == 12, "no floor = every match"
    t = S.scan_table(df, every, min_cr=1.0)
    assert set(t["company_id"]) == set(df["company_id"]) - {"NSE:S00", "NSE:S01"}
    assert t["_liq_cr"].min() >= 1.0


def test_the_recipe_sees_the_whole_market_not_the_liquid_slice():
    """Percentiles are taken over the universe; the floor only narrows the finished list. Stock 8 is
    top third of the MARKET on momentum and breakout, but the bottom of the liquid slice (8..11), so
    ranking inside the slice would lose it."""
    df = _universe()
    strong = {"rev_lq": 130.0, "pat_lq": 15.0, "ebitda_lq": 26.0, "dist_to_vstop": 5.0}
    weak = {"rev_lq": 101.0, "pat_lq": 10.0, "ebitda_lq": 15.0, "dist_to_vstop": -1.0}
    for c in strong:
        df.loc[8, c], df.loc[11, c] = strong[c], weak[c]
    df.loc[:7, "vol_sma_20d"] = 1.0                       # stocks 0..7 trade almost nothing
    t = S.scan_table(df, S.SCANS[0], min_cr=1.0)
    assert t["company_id"].tolist() == ["NSE:S08"]


def test_the_liquidity_guide_turns_every_step_into_an_order_size():
    """Which floor to pick depends on the buyer's order (2026-10-07): the help names, for every step, the
    largest order that stays under ORDER_SHARE of a day's traded value — derived from LIQUIDITY_STEPS, so
    a new step brings its own line and a changed share moves every number."""
    assert S.ORDER_SHARE == 0.05
    assert [S.order_size_lakh(s) for s in (0.5, 1.0, 2.0, 5.0)] == [2.5, 5.0, 10.0, 25.0]
    text = S.liquidity_help()
    steps = [s for s in S.LIQUIDITY_STEPS if s is not None]
    lines = [ln for ln in text.splitlines() if ln.startswith("- ")]
    assert lines == [f"- ₹{s:g} Cr/day → orders up to about ₹{S.order_size_lakh(s):g} lakh" for s in steps]
    assert "- ₹0.5 Cr/day → orders up to about ₹2.5 lakh" in lines         # 0.5 Cr × 5% = ₹2.5 lakh
    assert "not part of the record" in text and "every match" in text
    assert "about 5% of a day's traded value" in text and "ASM/GSM" in text
    assert "about ₹10 lakh" in S.table_config("x")["_liq_cr"]["help"]       # the column says it too


def test_sidebar_filters_narrow_the_finished_list_not_the_recipe():
    """🎯 Show only my filtered stocks (2026-10-06): the sidebar narrows the LIST; the recipe still
    ranks the whole market. Stock 8 is a whole-market match; kept alone with stock 9, re-ranking inside
    those two would drop it below the top third — it must stay a match."""
    df = _universe()
    strong = {"rev_lq": 130.0, "pat_lq": 15.0, "ebitda_lq": 26.0, "dist_to_vstop": 5.0}
    for c, v in strong.items():
        df.loc[8, c] = v
    t = S.scan_table(df, S.SCANS[0], keep_ids={"NSE:S08", "NSE:S09"})
    assert t["company_id"].tolist() == ["NSE:S08"]
    assert len(S.scan_table(df, S.SCANS[0], keep_ids=set())) == 0
    assert sorted(S.scan_table(df, S.SCANS[0])["company_id"]) == ["NSE:S08", "NSE:S11"], (
        "no keep_ids = every match")


def test_the_scope_checkbox_mirrors_market_pulse_and_never_feeds_the_recipe():
    tree = ast.parse(_APP)
    block = next(n for n in tree.body if isinstance(n, ast.With)
                 and ast.unparse(n.items[0].context_expr) == "tabs[1]")
    src = ast.unparse(block)
    assert "_sc_narrowed = 0 < len(filt) < len(df)" in src, (
        "the checkbox appears only while the sidebar actually narrows the market (the Market Pulse rule)")
    boxes = [n for n in ast.walk(block) if isinstance(n, ast.Call) and ast.unparse(n.func) == "st.checkbox"]
    assert len(boxes) == 1 and any(k.arg == "key" and ast.unparse(k.value) == "'scan_use_sidebar'"
                                   for k in boxes[0].keywords)
    assert "🎯 Show only my filtered stocks" in ast.unparse(boxes[0])
    guard = next(n for n in ast.walk(block) if isinstance(n, ast.If) and boxes[0] in ast.walk(n))
    assert ast.unparse(guard.test) == "_sc_narrowed"
    assert "keep_ids=set(filt['company_id'].astype(str)) if _sc_scoped else None" in src
    key = next(k for n in ast.walk(block) if isinstance(n, ast.Call) and ast.unparse(n.func) == "st.dataframe"
               for k in n.keywords if k.arg == "key")
    assert "_sc_ids" in ast.unparse(key.value), (
        "the table key follows the visible list: a narrowed list is a new table, so an old row pick "
        "cannot point at a different stock")
    assert "_sc_ids = universe_signature(_sc_tbl['company_id'])" in src


def test_the_tab_reuses_the_market_pulse_filter_row():
    """One dialect: the Scans filters ARE the QGLP / MOSL / Wealth lens row (Sector · Wealth tier ·
    Market cap · Catalyst · 🧹), hoisted to module level on 2026-10-06 rather than copied."""
    tree = ast.parse(_APP)
    defs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert {"_mp_lens_row", "_mp_ms", "_mp_clear_lens"} <= set(defs), "the lens row must be module level"
    block = next(n for n in tree.body if isinstance(n, ast.With)
                 and ast.unparse(n.items[0].context_expr) == "tabs[1]")
    calls = [n for n in ast.walk(block) if isinstance(n, ast.Call) and ast.unparse(n.func) == "_mp_lens_row"]
    assert len(calls) == 1 and ast.unparse(calls[0].args[1]) == "'scan'"
    assert "_mp_lens_row(_mp_qglp, \"qglp\")" in _APP, "Market Pulse still calls the same row"


# ── 📄 the fundamental clock ─────────────────────────────────────────────────────────────────────────
def test_a_result_declared_since_the_baseline_is_marked():
    """📄 = a quarterly result declared inside the window (days since the 🆕 baseline). result_age_days
    is negative for a result only SCHEDULED, which is not news yet."""
    df = _universe()
    df["result_age_days"] = [3, 5, 40, -2, np.nan, 0, 6, 6, 6, 6, 6, 6]
    every = {"key": "all", "mask": lambda d: pd.Series(True, index=d.index)}
    t = S.scan_table(df, every, members_at_latest={"NSE:S01", "NSE:S02", "NSE:S03", "NSE:S04", "NSE:S05"},
                     result_window_days=6)
    mark = dict(zip(t["company_id"], t["_new"]))
    assert mark["NSE:S00"] == "🆕 📄", "new AND a fresh result: it entered on the result"
    assert mark["NSE:S01"] == "📄", "already in the scan, result fresh"
    assert mark["NSE:S02"] == "", "result older than the baseline"
    assert mark["NSE:S03"] == "", "a scheduled (not declared) result is not news"
    assert mark["NSE:S04"] == "", "no result date: not judged"
    assert mark["NSE:S05"] == "📄", "declared on the data's own day"
    assert mark["NSE:S06"] == "🆕", "age 6 is outside a 6-day window: new on price, not on a result"


def test_no_result_mark_without_a_window():
    df = _universe()
    df["result_age_days"] = 1
    t = S.scan_table(df, {"key": "all", "mask": lambda d: pd.Series(True, index=d.index)})
    assert (t["_new"] == "").all()


@pytest.mark.parametrize("scan", S.SCANS, ids=lambda s: s["key"])
def test_the_table_fits_a_laptop_screen_with_one_header_each(scan):
    """Per scan (2026-10-07): every shown column has a set width, the total fits the 1,030px a 1536px
    laptop leaves, and a scan that hides a column gives its width to the others (no dead strip)."""
    cols, w = S.table_columns(scan), S.table_widths(scan)
    assert sum(w[c] for c in cols) + 44 <= 1074
    assert sum(w[c] for c in cols) == sum(S.TABLE_WIDTHS[c] for c in S.TABLE_COLUMNS), (
        f"{scan['key']}: a hidden column's width must go to the columns still shown")
    cfg = S.table_config("03 Oct", scan)
    for c in cols:
        assert c in cfg, f"{c} has no column config"
        assert cfg[c]["width"] == w[c], f"{c}: the config must use this scan's width"
        label = cfg[c]["label"]
        assert label and label != c and not (label.islower() and "_" in label), (c, label)
    assert {cfg[c]["label"] for c in ("rank", "name", "composite_score", "momentum_score",
                                       "improvement_score", "sector", "weinstein_stage")} == {
        "Rank", "Stock", "Score", "Momentum", "Improvement", "Sector", "Trend"}, (
        "the same column keeps the header it has in the Deep Scanner")


def test_a_column_is_hidden_only_when_the_recipe_fixes_its_value():
    """2026-10-07: on the Playbook the 🔥 column read ✓ on 100% of 275 stocks, because the recipe
    requires a Strong Quarter. A hidden column must be constant BY CONSTRUCTION: here the one match is
    strong, and making it not strong takes it out of the scan."""
    play = next(s for s in S.SCANS if s["key"] == "playbook")
    assert play["fixed"] == ("_strong_q",)
    assert "_strong_q" not in S.table_columns(play)
    t = S.scan_table(_universe(), play)
    assert len(t) == 1 and set(t["_strong_q"]) == {"✓"}
    df = _universe()
    df.loc[df.index[-1], "pat_lq"] = 10.0                 # profit flat: not a strong quarter
    assert S.scan_table(df, play).empty, "a stock without the fixed value cannot be in the scan"
    leaders = next(s for s in S.SCANS if s["key"] == "leaders")
    assert S.table_columns(leaders) == S.TABLE_COLUMNS, "Leading-Sector Stocks varies on every column"


def test_trend_stays_on_the_playbook_because_the_recipe_allows_a_stage_3_or_4():
    """'Above its stop' is not 'above its 30-week average': a stock in Stage 4 can pass every leg, so
    hiding Trend would hide a warning (it read 0% Stage 3/4 on 2026-10-07 — by data, not by design)."""
    df = _universe()
    df.loc[df.index[-1], "weinstein_stage"] = "📉 Stage 4 Declining"
    play = next(s for s in S.SCANS if s["key"] == "playbook")
    t = S.scan_table(df, play)
    assert t["weinstein_stage"].tolist() == ["📉 Stage 4"]
    assert "weinstein_stage" in S.table_columns(play)


def test_every_engine_trend_label_has_a_short_form():
    """The short labels must cover every label the engine writes; an unknown one passes unchanged."""
    src = open(os.path.join(ROOT, "core", "data_engine.py"), encoding="utf-8").read()
    # The rule lives in data_engine.weinstein_stage since 2026-10-08 (one rule for stocks and indices).
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "weinstein_stage")
    call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func) == "np.select")
    labels = [e.value for e in call.args[1].elts]
    assert len(labels) == 4 and set(labels) == set(S.TREND_SHORT), labels
    df = _universe()
    df["weinstein_stage"] = "❔ Unknown"
    t = S.scan_table(df, {"key": "all", "mask": lambda d: pd.Series(True, index=d.index)})
    assert set(t["weinstein_stage"]) == {"❔ Unknown"}
    cfg = S.table_config("")
    for short in [*S.TREND_SHORT.values(), "❔"]:
        assert short.split()[0] in cfg["weinstein_stage"]["help"], f"the tip does not explain {short}"
    assert not __import__("re").search(r"\bbuys?\b", cfg["weinstein_stage"]["help"], __import__("re").I)


# ── 5. The card says what was measured, and only that ─────────────────────────────────────────────
def _record(vals=(7.1, 10.7, 3.3, 2.9)):
    w = [dict(start="2026-06-17", end="2026-08-22", days=66), dict(start="2026-06-17", end="2026-10-03", days=108),
         dict(start="2026-08-22", end="2026-09-19", days=28), dict(start="2026-08-22", end="2026-10-03", days=42)]
    for d, v, b, n in zip(w, vals, (69, 69, 63, 61), (150, 150, 237, 235)):
        d.update(median_excess_pp=v, beat_pct=b, n=n)
    return {"windows": w}


def test_a_measured_card_shows_every_window_and_says_provisional():
    h = S.card_html(S.SCANS[0], 235, "measured", _record(), selected=True, latest="03 Oct")
    for s in ("+7.1", "+10.7", "+3.3", "+2.9", "61–69%", "150–237", "provisional", "235"):
        assert s in h, s


def test_the_card_says_when_its_first_unseen_window_is_due():
    h = S.card_html(S.SCANS[0], 235, "measured", _record(), selected=False, latest="03 Oct")
    assert "DESIGN-PERIOD RECORD" in h and "none yet" in h and S._first_unseen_due() in h
    assert "NEVER SAW" not in h


def test_the_card_shows_unseen_windows_beside_the_design_period_record():
    rec = _record()
    rec["out_of_sample"] = [dict(start="2026-10-31", end="2026-12-04", days=34, n=120,
                                 median_excess_pp=-1.4, beat_pct=47)]
    h = S.card_html(S.SCANS[0], 235, "measured", rec, selected=False, latest="03 Oct")
    assert "NEVER SAW" in h and "-1.4" in h and "47%" in h and "none yet" not in h
    assert h.index("DESIGN-PERIOD RECORD") < h.index("NEVER SAW"), "the design period reads first"


def test_an_unmeasured_or_stale_card_shows_no_numbers():
    h = S.card_html(S.SCANS[0], 235, "unmeasured", None, selected=False, latest="")
    assert "Not yet measured" in h and "+7.1" not in h
    h = S.card_html(S.SCANS[0], 235, "stale", _record(), selected=False, latest="03 Oct")
    assert "changed since" in h and "+7.1" not in h, "a stale record must not sit beside a new list"


def test_the_card_names_the_record_in_sample_and_out_of_sample_pending():
    """2026-10-07: "provisional" alone was read as "proven, just early". The design-period record is
    in-sample (the recipe was chosen on it), its windows share start dates, and nothing out of sample
    exists until the first unseen window — the card says all three in words."""
    h = S.card_html(S.SCANS[0], 235, "measured", _record(), selected=True, latest="03 Oct")
    for s in ("IN-SAMPLE · DESIGN-PERIOD RECORD", "chosen on these windows", "not proven", "before costs",
              "4 short windows from 2 start dates", "Out-of-sample: pending", S._first_unseen_due()):
        assert s in h, s


def test_a_record_without_net_or_left_out_counts_says_so():
    h = S.card_html(S.SCANS[0], 235, "measured", _record(), selected=True, latest="03 Oct")
    assert "not measured for these windows yet" in h, "absent is not zero"
    assert "Left out" not in h


def test_the_card_shows_the_net_record_and_what_was_left_out():
    rec = _record()
    for w, v in zip(rec["windows"], (4.0, 6.2, None, 0.8)):
        w.update(dropped_no_return=3, dropped_corp_action=1,
                 net=dict(min_cr=2.0, cost_pp=1.0, n=40) if v is None else
                 dict(min_cr=2.0, cost_pp=1.0, n=60, median_excess_pp=v, beat_pct=55))
    h = S.card_html(S.SCANS[0], 235, "measured", rec, selected=True, latest="03 Oct")
    for s in ("Net of a 1.0pp round trip", "₹2 Cr+/day", "+4.0", "+6.2", "+0.8", "–", "55% beat",
              "60 stocks", "3 with no price at the end", "1 corporate actions"):
        assert s in h, s
    assert "not measured for these windows yet" not in h


def test_the_recipe_docstrings_quote_the_committed_record():
    """2026-10-07: the docstrings had drifted from scan_records.json (+10.7 for +10.3; the leaders'
    +2.9 / +8.6 for +3.9 / +11.2). One definition covers the prose too: any figure a recipe's
    docstring quotes is the in-sample record, window by window."""
    import re
    rec = S.load_records()
    for scan in S.SCANS:
        quoted = [float(x) for x in re.findall(r"[+-]\d+\.\d", scan["mask"].__doc__ or "")]
        assert quoted, f"{scan['key']}: its docstring quotes no record"
        st_ = rec["scans"][scan["key"]].get("study")
        if st_:                                # a study's figures, in the order the docstring gives them
            want = [st_["match_cagr_pct"], st_["rest_cagr_pct"]]
            for y in st_.get("years", []):
                want += [y["match_excess_pp"], y["rest_excess_pp"]]
            assert quoted == want, (scan["key"], quoted, want)
            continue
        ws = rec["scans"][scan["key"]]["windows"]
        assert quoted == [w["median_excess_pp"] for w in ws], (scan["key"], quoted)
        beat = re.search(r"(\d+)-(\d+)% of matches", scan["mask"].__doc__)
        assert beat and (int(beat[1]), int(beat[2])) == (min(w["beat_pct"] for w in ws),
                                                         max(w["beat_pct"] for w in ws)), scan["key"]


def test_no_buy_wording():
    """Matched / record / provisional — PRISM narrows, the user decides."""
    tree = ast.parse(inspect.getsource(S))
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef)) and n.body
            and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    import re
    bad = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
           and id(n) not in docs and re.search(r"\bbuys?\b", n.value, re.I)]
    assert not bad, bad


# ── 6. The app wires it as designed ───────────────────────────────────────────────────────────────
_APP = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()


def test_scans_is_the_second_tab():
    tree = ast.parse(_APP)
    call = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == "tabs")
    labels = [e.value for e in call.args[0].elts]
    assert labels[:2] == ["🏠 Discovery", "🎯 Scans"]


def test_a_row_click_sets_the_stock_directly_and_never_reruns():
    """Found in the browser on 2026-10-06: the first build copied QGLP's 'stage _pending_xray, then
    st.rerun()'. That works for tabs rendered AFTER the Tear-Sheet's stock selectbox, where the staged
    key is consumed earlier in the same run; Scans renders BEFORE it, so every run would stage and
    rerun again. The stock is set directly (the Deep Scanner pattern), and only when the pick changes,
    so a row left selected never overrides a stock chosen later in the Tear-Sheet."""
    tree = ast.parse(_APP)
    block = next(n for n in tree.body if isinstance(n, ast.With)
                 and ast.unparse(n.items[0].context_expr) == "tabs[1]")
    src = ast.unparse(block)
    assert "st.rerun" not in src and "_pending_xray" not in src
    assert "st.session_state['xray_stock'] = _sc_name" in src
    assert "_scan_last_pick" in src, "the stock must be set only when the pick changes"
    widget_at = _APP.index('key="xray_stock"')
    assert block.lineno < _APP[:widget_at].count("\n") + 1, (
        "the direct set is only legal while Scans renders before the Tear-Sheet's stock selectbox")


def test_the_list_comes_before_the_footnotes_and_the_caption_does_not_overpromise():
    """2026-10-07: the list is the tab's payoff, so 'Being measured' (no record yet) sits after it. A
    row click loads the stock — Streamlit cannot switch tabs from code — so the caption says so."""
    tree = ast.parse(_APP)
    block = next(n for n in tree.body if isinstance(n, ast.With)
                 and ast.unparse(n.items[0].context_expr) == "tabs[1]")
    calls = [n for n in ast.walk(block) if isinstance(n, ast.Call)]
    def line(func, text):
        return next(n.lineno for n in calls if ast.unparse(n.func) == func and text in ast.unparse(n))
    table = line("st.dataframe", "scan_tbl_")                  # the scan's own list
    left = line("st.expander", "Left ")                        # Left 🌱, the exit side
    measured = line("st.expander", "Being measured")
    assert table < left < measured, "the list, then who left it, then the footnote"
    src = ast.unparse(block)
    assert "load it into the Tear-Sheet tab" in src and "open its Tear-Sheet" not in src
    assert "column_order=list(scan_table_columns(_sc_sel))" in src, "the table shows this scan's columns"
    assert "scan_table_config(_sc_new_label, _sc_sel)" in src, "and this scan's widths"


def test_the_tab_reads_the_whole_market_and_never_the_sidebar_frame():
    tree = ast.parse(_APP)
    block = next(n for n in tree.body if isinstance(n, ast.With)
                 and ast.unparse(n.items[0].context_expr) == "tabs[1]")
    src = ast.unparse(block)
    assert "scan_table(df," in src, "the scan must run on the whole market"
    assert "scan_table(filt" not in src


# ── 7. 🌱 Compounder Candidates (2026-10-07) ─────────────────────────────────────────────────────────
def _growers(n=10):
    """n stocks; profit growth and margin change both rise with the index, so the top 40% on each leg
    is stocks 6..9 (percentile rank >= 0.6 of 10)."""
    i = np.arange(n, dtype=float)
    return pd.DataFrame({
        "company_id": [f"NSE:G{k:02d}" for k in range(n)], "name": [f"G{k}" for k in range(n)],
        "rank": np.arange(1, n + 1),
        "pat_1yb": 100.0, "pat": 100.0 + 10 * i,                 # profit growth 0% .. 90%
        "opm_1yb": 10.0, "opm": 10.0 + i,                        # margin change 0 .. +9 points
        "revenue_1yb": 100.0, "revenue": 100.0 + 2 * i,          # sales growth 0% .. 18%
    })


def test_compounder_takes_the_top_40pct_on_both_legs_across_the_market():
    assert S.compounder_candidates(_growers()).tolist() == [False] * 6 + [True] * 4


@pytest.mark.parametrize("col,value,why", [
    ("pat_1yb", -5.0, "a loss a year ago has no growth rate"),
    ("pat", -1.0, "a loss now has no growth rate"),
    ("pat", np.nan, "missing profit passes nothing"),
    ("opm", np.nan, "missing margin passes nothing"),
    ("opm", 10.0, "margin flat is not in the top 40%"),
    ("revenue", 105.0, "sales up only 5%: not revenue-led"),
    ("revenue_1yb", -1.0, "no sales base: no growth rate, not revenue-led"),
    ("revenue", np.nan, "missing sales passes nothing"),
])
def test_compounder_needs_every_leg(col, value, why):
    df = _growers()
    df.loc[df.index[-1], col] = value
    assert not S.compounder_candidates(df).iloc[-1], why


def test_compounder_ranks_the_whole_market_not_a_slice():
    """Percentiles are taken over the frame given, which the tab passes whole: the same stock can be
    top 40% of the market and bottom of a slice."""
    df = _growers()
    assert S.compounder_candidates(df).iloc[6]
    assert not S.compounder_candidates(df.iloc[6:].reset_index(drop=True)).iloc[0]


def test_revenue_led_needs_sales_up_10pct_and_profit_outrunning_them():
    df = _growers()
    # stock 9: sales +18%, profit +90% -> led; stock 4: sales +8% -> not more than 10%
    led = S.revenue_led(df)
    assert led.iloc[9] and not led.iloc[4]
    df.loc[9, "pat"] = 110.0                                     # profit +10% < sales +18%
    assert not S.revenue_led(df).iloc[9], "profit must grow faster than sales"
    df.loc[8, "revenue_1yb"] = -1.0
    assert not S.revenue_led(df).iloc[8], "no sales base, no growth rate"


def test_compounder_requires_revenue_led_and_ranks_the_market_first():
    """Revenue-led is part of the recipe (2026-10-07: without it the backtest edge all but vanished).
    The top-40% cuts rank the WHOLE market before it narrows, as the backtest did: narrowing first
    would re-rank a smaller set and change which stocks count as top 40%."""
    df = _growers()
    df.loc[7, "revenue"] = 101.0          # a top-40% stock whose sales grew 1%: out
    df.loc[0:5, "revenue"] = 100.0        # sales flat for the bottom six: they still count in the ranking
    assert S.compounder_candidates(df).tolist() == [False] * 6 + [True, False, True, True]
    df.loc[9, "revenue"] = 195.0          # still top on profit (+90%) and margin, but sales +95% outran
    assert not S.compounder_candidates(df).iloc[9], "profit must grow faster than sales"


def _study_record():
    return json.loads(S.RECORDS_PATH.read_text(encoding="utf-8"))["scans"]["compounder"]


def test_the_compounder_card_shows_its_study_named_in_sample_with_its_caveats():
    import html as _html
    comp = _scan("compounder")
    h = _html.unescape(S.card_html(comp, 320, "measured", _study_record(), selected=True, latest="03 Oct"))
    for s in ("3-YEAR BACKTEST · IN-SAMPLE", "Oct 2023", "+20.4%/yr", "+4.4%/yr", "Doubled: 40.7% vs 20.8%",
              "182 vs 1,325", "+31.2", "+13.5", "+23.6", "23→24", "24→25 ✦", "25→26",
              "Ahead in all 3 years — inside every size group, against its own industry, with last year's "
              "momentum held", "Beat the market: 71% / 64% / 68% of matches, against 44% / 46% / 47% of the rest",
              "✦ Cleanest test: 24→25", "net margin",
              "Better odds, not a promise: 59% of matches did not double",
              "Survivors only", "before costs", "FY21 → FY23", "Out-of-sample: pending",
              "one 1-year window a year, each starting at the end of October",
              "31 Oct 2027"):
        assert s in h, s
    assert "DESIGN-PERIOD" not in h and "short windows" not in h, "a study is not saved windows"
    assert h.index("+31.2") < h.index("+13.5") < h.index("+23.6"), "the years read in order"


def test_the_compounder_card_shows_unseen_windows_once_they_exist():
    rec = _study_record()
    rec["out_of_sample"] = [dict(start="2026-10-31", end="2027-10-31", days=365, n=300,
                                 median_excess_pp=4.2, beat_pct=58)]
    h = S.card_html(_scan("compounder"), 320, "measured", rec, selected=False, latest="03 Oct")
    assert "NEVER SAW" in h and "+4.2" in h and "58%" in h and "pending" not in h
    assert h.index("IN-SAMPLE") < h.index("NEVER SAW")


def test_a_stale_compounder_record_shows_no_study():
    h = S.card_html(_scan("compounder"), 320, "stale", _study_record(), selected=False, latest="")
    assert "changed since" in h and "+20.4" not in h


def test_each_scan_is_judged_on_its_own_clock():
    assert S._first_unseen_due(_scan("playbook")) == "25 Nov"
    assert S._first_unseen_due(_scan("compounder")) == "31 Oct 2027"
    assert S.scan_horizon(_scan("compounder")) == 365 and S.scan_horizon(_scan("playbook")) == 25


def test_the_compounder_fingerprint_follows_its_helpers():
    d = _deps(_scan("compounder"))
    assert {"fn ui.ui_scans._growth", "fn ui.ui_scans._num", "fn ui.ui_scans.revenue_led"} <= d, (
        "revenue-led is part of the recipe, so its record follows it")


def test_no_scan_carries_an_optional_narrowing():
    """The 🌱 revenue-led switch became part of the recipe (2026-10-07): no scan has a `refine`, and the
    tab has no toggle, so nothing on screen narrows a list in a way its record does not describe."""
    assert not any("refine" in s for s in S.SCANS)
    tree = ast.parse(_APP)
    block = next(n for n in tree.body if isinstance(n, ast.With)
                 and ast.unparse(n.items[0].context_expr) == "tabs[1]")
    assert "st.toggle" not in ast.unparse(block)


# ── 8. 🌱's year-by-year record (2026-10-07) ─────────────────────────────────────────────────────────
def _years():
    return _study_record()["study"]["years"]


def test_the_yearly_record_is_three_separate_years_in_order():
    ys = _years()
    assert [y["label"] for y in ys] == ["23→24", "24→25", "25→26"]
    assert [y["known"] for y in ys] == ["FY23", "FY24", "FY25"], "each year judged on the results public at its start"
    for y in ys:
        assert y["n_match"] >= 30 and y["n_rest"] >= 30
        assert 0 <= y["match_beat_pct"] <= 100 and 0 <= y["rest_beat_pct"] <= 100
        assert set(y["size_pp"]) == {"small", "mid", "large"}
    assert ys[0]["momentum_held_pp"] is None, "the first window has no previous year to hold"
    st_ = _study_record()["study"]
    assert st_["cleanest"]["label"] in [y["label"] for y in ys] and st_["years_note"] and st_["years_source"]
    assert "second" not in st_, "the yearly record replaces the profit-and-sales-only falling-year reading"


def test_the_card_claims_only_what_every_year_measured():
    """The 'ahead in all N years — controls' line is computed from the record, never written by hand:
    flip one year and the card must say so, and drop each control that no longer holds every year."""
    ys = _years()
    assert S.year_claims(ys) == (3, 3, ["inside every size group", "against its own industry",
                                       "with last year's momentum held"])
    import copy
    flipped = copy.deepcopy(ys)
    flipped[1]["match_excess_pp"], flipped[1]["rest_excess_pp"] = -3.0, 1.0
    flipped[1]["size_pp"]["mid"] = [-2.0, 1.0]
    flipped[2]["momentum_held_pp"] = -0.5
    assert S.year_claims(flipped) == (2, 3, ["against its own industry"])
    rec = copy.deepcopy(_study_record())
    rec["study"]["years"] = flipped
    h = S.card_html(_scan("compounder"), 255, "measured", rec, selected=False, latest="")
    assert "Ahead in 2 of 3 years — against its own industry." in h and "Ahead in all" not in h
    assert "-3.0" in h, "a losing year shows as it is"


def test_a_study_without_yearly_record_still_renders_its_two_bars():
    import copy
    rec = copy.deepcopy(_study_record())
    for k in ("years", "cleanest", "years_note"):
        rec["study"].pop(k, None)
    h = S.card_html(_scan("compounder"), 255, "measured", rec, selected=False, latest="")
    assert "Matches" in h and "Rest" in h and "+20.4%" in h and "Year by year" not in h and "✦" not in h


def test_the_yearly_bars_compare_by_height():
    import re
    h = S._year_bars(_years(), "24→25", S.COLORS, __import__("html").escape)
    heights = [int(x) for x in re.findall(r"width:14px;height:(\d+)px", h)]
    assert heights == [34, 6 + round(28 * 13.5 / 31.2), 6 + round(28 * 23.6 / 31.2)], heights
    assert heights[1] < heights[2] < heights[0], "the tallest bar is the largest year"


# ── 9. 🌱 "Cycle ↓" — supply-chain velocity as a reading aid (2026-10-07) ───────────────────────────
def _cycles(rows):
    """rows: (cf_year_lag, ccc, ccc_1yb, revenue, revenue_1yb, revenue_2yb)."""
    cols = ["cf_year_lag", "ccc", "ccc_1yb", "revenue", "revenue_1yb", "revenue_2yb"]
    return pd.DataFrame(rows, columns=cols)


def test_cycle_needs_15_days_shorter_and_sales_up_more_than_15pct():
    df = _cycles([                         # every row: balance sheet on the P&L's year (cf_year_lag 0)
        (0, 85, 100, 120, 100, 90),       # -15 days, sales +20%            -> ✓ (15 days is enough)
        (0, 86, 100, 120, 100, 90),       # -14 days                        -> no
        (0, 85, 100, 115, 100, 90),       # sales +15% exactly              -> no (must be MORE than 15%)
        (0, 85, 100, 116, 100, 90),       # sales +16%                      -> ✓
        (0, np.nan, 100, 130, 100, 90),   # no cash cycle                   -> no
        (0, 85, 100, 130, 0, 90),         # no sales base                   -> no
        (0, 85, 100, 130, -5, 90),        # negative base                   -> no
    ])
    assert S.cycle_shortened(df).tolist() == [True, False, False, True, False, False, False]


def test_cycle_needs_the_balance_sheet_on_the_same_year_as_the_pl():
    """Measured 2026-10-07: a YEAR-OLD cash-cycle improvement did not help 🌱 (+9.4 vs +15.9, +32.0 vs
    +32.0); only the same-year one did. The vendor's balance sheet lags its P&L for most stocks in
    October (cf_year_lag == 1), so those get no ✓ until it rolls — whatever their numbers say."""
    df = _cycles([
        (0, 80, 100, 130, 100, 90),       # balance sheet on the P&L's year, -20 days, sales +30% -> ✓
        (1, 80, 100, 130, 100, 90),       # balance sheet a year behind: same numbers           -> no
        (1, 80, 100, 101, 100, 80),       # behind, last year's sales +25%                       -> no
        (np.nan, 80, 100, 130, 100, 90),  # year cannot be told                                  -> no
        (0, 80, 100, 101, 100, 80),       # same year, this year's sales +1%                     -> no
    ])
    assert S.cycle_shortened(df).tolist() == [True, False, False, False, False]


def test_cycle_is_a_column_on_the_compounder_table_only():
    comp, play, lead = _scan("compounder"), _scan("playbook"), _scan("leaders")
    cols = S.table_columns(comp)
    assert cols[cols.index("_strong_q") + 1] == "_cycle", "it reads beside 🔥"
    assert "_cycle" not in S.table_columns(play) and "_cycle" not in S.table_columns(lead)
    assert S.table_columns(lead) == S.TABLE_COLUMNS
    cfg = S.table_config("", comp)["_cycle"]
    assert cfg["label"] == "Cycle ↓" and "not part of 🌱's recipe or record" in cfg["help"]


def test_cycle_is_a_reading_aid_not_part_of_the_record():
    assert "fn ui.ui_scans.cycle_shortened" not in _deps(_scan("compounder")), (
        "the ✓ must not reach the fingerprint: changing it cannot stale the record")


def test_the_compounder_table_marks_its_cycle_rows():
    df = _growers()
    df["cf_year_lag"] = 0.0
    df["ccc_1yb"], df["ccc"] = 100.0, 100.0
    df.loc[[6, 9], "ccc"] = 70.0                               # G06 sales +12%, G09 +18%
    t = S.scan_table(df, _scan("compounder"))
    marks = dict(zip(t["company_id"], t["_cycle"]))
    assert marks == {"NSE:G06": "", "NSE:G07": "", "NSE:G08": "", "NSE:G09": "✓"}, (
        "G06's cycle shortened but sales grew only 12%; G09's shortened with sales +18%")


def test_the_first_due_date_follows_the_start_month():
    assert S._first_unseen_due(_scan("compounder")) == "31 Oct 2027"
    assert S._first_unseen_due({"designed_on": "2026-03-15", "horizon_days": 365, "start_month": 10}) == "31 Oct 2027", (
        "designed in March: the first start is that October's month-end, a year later the first result")
    assert S._first_unseen_due({"designed_on": "2026-11-02", "horizon_days": 365, "start_month": 10}) == "30 Oct 2028", (
        "designed after October: the first start is next October (31 Oct 2027 + 365 days; 2028 is a leap year)")
    assert S._first_unseen_due(_scan("playbook")) == "25 Nov", "a scan with no start month is unchanged"


# ── 8. Left 🌱 — the exit side (2026-10-07) ───────────────────────────────────────────────────────────
def _leavers():
    """Five non-matches, one per reason, and one 🌱 match. Sales/profit as (year before, last year)."""
    rows = [  # company, revenue_1yb, revenue, pat_1yb, pat, opm_1yb, opm
        ("NSE:FELL", 100, 130, 10, 8, 10, 12),       # profit fell (sales still up 30%)
        ("NSE:LOSS", 100, 120, -5, -9, 5, 4),        # loss widened: profit fell
        ("NSE:SLOW", 100, 105, 10, 12, 10, 11),      # profit up, sales only +5%: not revenue-led
        ("NSE:RANK", 100, 120, 10, 13, 10, 10.1),    # revenue-led, profit up: others grew faster
        ("NSE:MISS", np.nan, 120, 10, 14, 10, 11),   # a figure missing
        ("NSE:STAY", 100, 150, 10, 30, 10, 20),      # still a 🌱 match
    ]
    df = pd.DataFrame(rows, columns=["company_id", "revenue_1yb", "revenue", "pat_1yb", "pat",
                                     "opm_1yb", "opm"])
    df["name"] = df["company_id"].str[4:].str.title()
    df["sector"] = "S"
    return df


_SEEDLING = next(s for s in S.SCANS if s["key"] == "compounder")
_STAY_ONLY = {**_SEEDLING, "mask": lambda d: d["company_id"] == "NSE:STAY"}


def test_each_exit_gets_the_reason_the_evidence_ranks_worst():
    df = _leavers()
    why = dict(zip(df["company_id"], S.compounder_exit_reason(df)))
    assert why["NSE:FELL"] == "profit_fell" and why["NSE:LOSS"] == "profit_fell"
    assert why["NSE:SLOW"] == "not_revenue_led"
    assert why["NSE:RANK"] == "rank_slipped"
    assert why["NSE:MISS"] == "unjudged"
    assert list(S.EXIT_REASONS) == ["profit_fell", "not_revenue_led", "rank_slipped", "unjudged", "gone"]


def test_left_lists_baseline_members_not_in_the_scan_now_worst_first():
    df = _leavers()
    base = list(df["company_id"]) + ["NSE:GONE"]
    t = S.left_scan(df, _STAY_ONLY, base)
    assert "NSE:STAY" not in set(t["company_id"]), "a stock still in the scan did not leave"
    assert t["company_id"].tolist() == ["NSE:FELL", "NSE:LOSS", "NSE:SLOW", "NSE:RANK", "NSE:MISS", "NSE:GONE"]
    assert t["_why"].tolist()[0] == "🔴 profit fell" and t["_why"].tolist()[-1] == "⚪ not in today's data"
    row = t.set_index("company_id")
    assert row.loc["NSE:FELL", "_sales_chg"] == pytest.approx(30.0)
    assert row.loc["NSE:FELL", "_profit_chg"] == pytest.approx(-20.0)
    assert pd.isna(row.loc["NSE:LOSS", "_profit_chg"]), "no growth rate from a loss year"
    assert row.loc["NSE:GONE", "sector"] == "", "a stock gone from the data shows no sector, not 'None'"
    assert S.left_summary(t) == ("🔴 2 profit fell · 🟠 1 no longer revenue-led · ⚪ 1 others grew faster · "
                                 "⚪ 1 a figure is missing · ⚪ 1 not in today's data")


def test_left_needs_a_baseline_and_measured_exit_reasons_and_obeys_the_sidebar_scope():
    df = _leavers()
    assert S.left_scan(df, _STAY_ONLY, None).empty, "no baseline, no exits"
    no_reasons = {k: v for k, v in _STAY_ONLY.items() if k != "exit_reason"}
    assert S.left_scan(df, no_reasons, list(df["company_id"])).empty, "only measured exits are shown"
    t = S.left_scan(df, _STAY_ONLY, list(df["company_id"]) + ["NSE:GONE"], keep_ids={"NSE:FELL", "NSE:STAY"})
    assert t["company_id"].tolist() == ["NSE:FELL"], "scoped: only kept stocks, and no gone ones"
    assert [s["key"] for s in S.SCANS if "exit_reason" in s] == ["compounder"]


def test_left_runs_the_recipe_on_the_whole_market():
    """A stock leaves when the RECIPE drops it on the whole market — the same mask the record measures."""
    df = _leavers()
    seen = []
    spy = {**_STAY_ONLY, "mask": lambda d: (seen.append(len(d)), d["company_id"] == "NSE:STAY")[1]}
    S.left_scan(df, spy, list(df["company_id"]), keep_ids={"NSE:FELL"})
    assert seen == [len(df)]


def test_the_exit_reasons_are_not_part_of_the_recipe_fingerprint():
    bare = {k: v for k, v in _SEEDLING.items() if k != "exit_reason"}
    assert S.recipe_fingerprint(_SEEDLING) == S.recipe_fingerprint(bare)


def test_the_legend_says_which_reasons_are_measured():
    lines = S.left_legend().splitlines()
    assert len(lines) == len(S.EXIT_REASONS)
    measured = [ln for ln in lines if "— measured:" in ln]
    assert [ln.split("**")[1] for ln in measured] == ["profit fell", "no longer revenue-led"]
    assert all("not measured" in ln for ln in lines if ln not in measured)
    assert "8 to 11 points" in lines[0] and "+9.8 and +26.6" in lines[1]


# ── The summary card you choose from (2026-10-08) ────────────────────────────────────────────────────
def _with_net(rec, nets):
    for w, v in zip(rec["windows"], nets):
        w["net"] = {"min_cr": 2.0, "cost_pp": 1.0, "n": 40}
        if v is not None:
            w["net"].update(median_excess_pp=v, beat_pct=55)
    return rec


def test_the_summary_names_in_sample_and_gives_both_sides_of_costs():
    """Three full cards side by side were 406 / 408 / 596px tall; the summary carries one honest
    headline: in-sample, how many windows it was ahead in, before AND after costs, and when the first
    out-of-sample test is due."""
    play = S.SCANS[0]
    h = S.card_summary_html(play, 235, "measured", _with_net(_record(), [6.1, 8.9, 1.9, 1.4]), True)
    for s in ("IN-SAMPLE", "4 of 4", "+2.9 to +10.7 pts before costs", "+1.4 to +8.9 after costs",
              f"first test due about {S._first_unseen_due(play)}", "235"):
        assert s in h, s
    assert f"min-height:{S.SUMMARY_MIN_HEIGHT}px" in h, "every summary card is one height, so the buttons align"


def test_the_summary_counts_a_losing_window_and_says_when_costs_are_partial():
    rec = _with_net(_record((7.1, -1.0, 3.3, 2.9)), [6.1, None, 1.9, None])
    h = S.card_summary_html(S.SCANS[0], 235, "measured", rec, False)
    assert "3 of 4" in h and "-1.0 to +7.1 pts before costs" in h
    assert "+1.9 to +6.1 after costs (2 of 4 windows)" in h, "partial net must say how many windows it covers"
    h = S.card_summary_html(S.SCANS[0], 235, "measured", _record(), False)
    assert "after costs" not in h, "no net record, no after-costs claim"


def test_the_summary_of_a_study_says_how_many_did_not_double():
    seed = next(s for s in S.SCANS if s["key"] == "compounder")
    state, rec = S.record_state(S.load_records(), seed)
    h = S.card_summary_html(seed, 255, state, rec, False)
    st_ = rec["study"]
    ahead, total, _ = S.year_claims(st_["years"])
    for s in ("IN-SAMPLE", st_["title"].upper(), "BEFORE COSTS", f'{st_["match_cagr_pct"]:+.1f}%/yr',
              f"Ahead in {ahead} of {total} years", f'{100 - st_["match_doubled_pct"]:.0f}% of matches did not double',
              S._first_unseen_due(seed)):
        assert s in h, s


def test_the_summary_reports_unseen_windows_once_they_exist():
    rec = _record()
    rec["out_of_sample"] = [dict(start="2026-10-31", end="2026-12-04", days=34, n=120, median_excess_pp=-1.4,
                                 beat_pct=47),
                            dict(start="2026-10-31", end="2027-01-08", days=69, n=118, median_excess_pp=2.0,
                                 beat_pct=56)]
    h = S.card_summary_html(S.SCANS[0], 235, "measured", rec, False)
    assert "ahead in 1 of 2 unseen windows (-1.4 to +2.0 pts)" in h and "pending" not in h


def test_a_stale_or_unmeasured_summary_shows_no_numbers():
    h = S.card_summary_html(S.SCANS[0], 235, "stale", _record(), False)
    assert "changed since" in h and "+7.1" not in h and "+10.7" not in h
    assert "Not yet measured" in S.card_summary_html(S.SCANS[0], 235, "unmeasured", None, False)


def test_the_cards_are_summaries_and_the_full_record_sits_under_them():
    """app.py: the card row uses the summary; ONE full-record panel renders card_html for the SELECTED
    scan without repeating its head; it sits between the cards and the filters."""
    tree = ast.parse(_APP)
    block = next(n for n in tree.body if isinstance(n, ast.With)
                 and ast.unparse(n.items[0].context_expr) == "tabs[1]")
    calls = [n for n in ast.walk(block) if isinstance(n, ast.Call)]
    summ = [n for n in calls if ast.unparse(n.func) == "scan_card_summary_html"]
    full = [n for n in calls if ast.unparse(n.func) == "scan_card_html"]
    assert len(summ) == 1 and ast.unparse(summ[0].args[0]) == "_sc", "the card row shows the summary"
    assert len(full) == 1 and ast.unparse(full[0].args[0]) == "_sc_sel", "the full record is the selected scan's"
    assert any(k.arg == "with_head" and ast.unparse(k.value) == "False" for k in full[0].keywords)
    panel = next(n for n in calls if ast.unparse(n.func) == "st.expander" and "Full record" in ast.unparse(n))
    liq = next(n for n in calls if ast.unparse(n.func) == "st.selectbox" and "scan_liq" in ast.unparse(n))
    assert summ[0].lineno < panel.lineno < full[0].lineno < liq.lineno


def test_the_full_record_without_its_head_is_the_same_record():
    rec = _with_net(_record(), [6.1, 8.9, 1.9, 1.4])
    with_head = S.card_html(S.SCANS[0], 235, "measured", rec, True, "03 Oct")
    bare = S.card_html(S.SCANS[0], 235, "measured", rec, True, "03 Oct", with_head=False)
    assert S.SCANS[0]["recipe"] not in bare and S.SCANS[0]["recipe"] in with_head
    assert with_head.replace(S._card_head(S.SCANS[0], 235, S.COLORS, __import__("html").escape), "") == bare


# ── The local record tool never overwrites a study (2026-10-08) ─────────────────────────────────────
def test_only_the_short_horizon_scans_are_measured_on_the_snapshots():
    """tools/scan_records.py measures 25-day snapshot windows. 🌱's evidence horizon is a year and its
    record is a 3-year study, so it must never be measured there."""
    flags = {s["key"]: S.measured_on_snapshots(s) for s in S.SCANS}
    assert flags == {"playbook": True, "leaders": True, "compounder": False}, flags
    recs = S.load_records()["scans"]
    for s in S.SCANS:
        if recs.get(s["key"], {}).get("study"):
            assert not S.measured_on_snapshots(s), f"{s['key']} has a study, so snapshots must not touch it"


def test_the_local_record_tool_leaves_a_study_record_untouched(monkeypatch):
    """Runs the REAL local tool with the measurement faked to always return a window: without the guard
    it would write that window into 🌱's record. Skips where the gitignored tool is absent."""
    import importlib.util
    path = os.path.join(ROOT, "tools", "scan_records.py")
    if not os.path.exists(path):
        pytest.skip("tools/scan_records.py is local-only")
    spec = importlib.util.spec_from_file_location("_scan_records_tool", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    frame = pd.DataFrame({"company_id": ["NSE:A"], "close_price": [1.0]})
    fake = {"start": "2026-01-01", "end": "2026-02-20", "days": 50, "median_excess_pp": 1.0,
            "beat_pct": 50, "n": 40}
    written = {}
    monkeypatch.setattr(tool.V, "_load_snapshots", lambda: [("2026-01-01", frame), ("2026-02-20", frame)])
    monkeypatch.setattr(tool.J, "measure_window", lambda *a, **k: dict(fake))
    monkeypatch.setattr(tool.J, "engine_version", lambda: "test")
    monkeypatch.setattr(tool.J, "write", lambda new, *a, **k: written.update(new))
    before = S.load_records()["scans"]["compounder"]
    tool.main()
    assert written, "the faked windows must change the short scans, so the tool writes"
    assert written["scans"]["compounder"] == before, "the tool overwrote 🌱's study record"
    assert written["scans"]["playbook"]["windows"] == [fake], "the short scans are still measured"

