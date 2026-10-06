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
        assert len(r["windows"]) >= 2
        for w in r["windows"]:
            assert w["days"] >= 25 and w["start"] < w["end"], w
            assert w["n"] >= 30, f"{scan['key']} {w['start']}->{w['end']}: n={w['n']} is too few to show"
            assert 0 <= w["beat_pct"] <= 100
        members = r["members_at_latest"]
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


def test_the_table_fits_a_laptop_screen_with_one_header_each():
    cols = S.TABLE_COLUMNS
    assert sum(S.TABLE_WIDTHS[c] for c in cols) + 44 <= 1074
    cfg = S.table_config("03 Oct")
    for c in cols:
        assert c in cfg, f"{c} has no column config"
        label = cfg[c]["label"]
        assert label and label != c and not (label.islower() and "_" in label), (c, label)
    assert {cfg[c]["label"] for c in ("rank", "name", "composite_score", "momentum_score",
                                       "improvement_score", "sector", "weinstein_stage")} == {
        "Rank", "Stock", "Score", "Momentum", "Improvement", "Sector", "Trend"}, (
        "the same column keeps the header it has in the Deep Scanner")


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


def test_an_unmeasured_or_stale_card_shows_no_numbers():
    h = S.card_html(S.SCANS[0], 235, "unmeasured", None, selected=False, latest="")
    assert "Not yet measured" in h and "+7.1" not in h
    h = S.card_html(S.SCANS[0], 235, "stale", _record(), selected=False, latest="03 Oct")
    assert "changed since" in h and "+7.1" not in h, "a stale record must not sit beside a new list"


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


def test_the_tab_reads_the_whole_market_and_never_the_sidebar_frame():
    tree = ast.parse(_APP)
    block = next(n for n in tree.body if isinstance(n, ast.With)
                 and ast.unparse(n.items[0].context_expr) == "tabs[1]")
    src = ast.unparse(block)
    assert "scan_table(df," in src, "the scan must run on the whole market"
    assert "scan_table(filt" not in src
