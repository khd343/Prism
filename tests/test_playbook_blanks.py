"""
test_playbook_blanks.py
=======================
Contract: what a 🚀 Playbook list could not judge (2026-10-16).

THE RISK. Kite sells a held stock once it is out of the list 3 sessions running (decay). A stock can be out of the
list because an input of the recipe came in BLANK — the vendor sent no CRS for it that day, a quarter half-filled in
results season — and then the recipe never judged it out. Measured day to day (10 Sep - 9 Oct 2026): 14 times in 6,017
member-days a list member lost an input it had the day before, and 2 of those left the list that day; on the 15 Oct
copy one blank quarterly level or stop distance would knock any of the 267 members off.

THE FIX — PRISM states the facts, Kite decides with history (its tests/test_playbook_rules.py):
* playbook_conditions: the recipe's six conditions, one definition; the shortlist is the stocks meeting all six.
* RECIPE_INPUTS: what each condition reads, pinned two ways — to every column the recipe's code names (the
  fingerprint's own walk) and on a real copy (blanking an input moves only the conditions that list it).
  BLANK_IS_A_FACT: the two columns whose blank is the vendor's answer, never a gap.
* blanks(scored): each manifest says which inputs came in blank — for a member every one (what it passed without),
  and for a stock off the list whose EVERY failed condition has a blank input, those inputs per failed condition.

Run with: pytest tests/test_playbook_blanks.py -v
"""
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import data_engine as de  # noqa: E402
from ui import playbook_feed_job as F  # noqa: E402
from ui import scan_records_job as J  # noqa: E402
from ui import ui_scans as S  # noqa: E402

CONDITIONS = ["momentum", "breakout", "quarter", "improvement", "evidence", "stop"]
INPUTS = sorted({c for v in F.RECIPE_INPUTS.values() for c in v})
_ARCHIVE = ROOT / "Other Resources" / "archive-cache" / "PRISM 2026-10-09.xlsx"


def _frame(n=30, members=10):
    """n stocks; the last `members` pass all six conditions, the rest fail the stop and the quarter on complete data.
    Every recipe input is present; the two blank-is-a-fact columns are blank, as they often are."""
    i = np.arange(n, dtype=float)
    m = i >= n - members
    df = pd.DataFrame({
        "company_id": [f"NSE:S{k:02d}" for k in range(n)],
        "momentum_score": i, "breakout_score": i, "improvement_score": i,
        "dist_to_vstop": np.where(m, 5.0, -1.0),
        "rev_pyq": 100.0, "rev_lq": np.where(m, 130.0, 101.0), "pat_pyq": 10.0, "pat_lq": np.where(m, 15.0, 10.0),
        "ebitda_pyq": 15.0, "ebitda_lq": np.where(m, 26.0, 15.0),
        "pat": 12.0, "pat_1yb": 10.0, "revenue": 110.0, "revenue_1yb": 100.0, "opm": 15.0, "opm_1yb": 14.0,
        "sector": "Alpha", "breakout_window": np.nan, "golden_cross_days": np.nan,
    })
    for c in sorted(set(INPUTS) - set(df.columns)):
        df[c] = 1.0
    return df


# ── 1. The recipe is six named conditions, one definition ─────────────────────────────────────────────────────────
def test_the_recipe_is_six_named_conditions():
    assert list(S.playbook_conditions(_frame()).columns) == CONDITIONS


def test_the_shortlist_is_the_stocks_meeting_all_six():
    df = _frame()
    cond = S.playbook_conditions(df)
    assert cond.dtypes.eq(bool).all()
    assert S.playbook_shortlist(df).equals(cond.all(axis=1))
    assert S.playbook_shortlist(df).sum() == 10


@pytest.mark.parametrize("condition,breaks", [
    ("momentum", {"momentum_score": -1.0}), ("breakout", {"breakout_score": -1.0}),
    ("quarter", {"rev_lq": 100.0}), ("improvement", {"improvement_score": -1.0}),
    ("evidence", {c: np.nan for c in ("pat", "pat_1yb", "revenue", "revenue_1yb", "opm", "opm_1yb")}),
    ("stop", {"dist_to_vstop": -0.5}),
])
def test_missing_any_one_condition_leaves_the_list(condition, breaks):
    df = _frame()
    for col, value in breaks.items():
        df.loc[25, col] = value
    cond = S.playbook_conditions(df)
    assert not cond.loc[25, condition] and cond.loc[25].drop(condition).all()
    assert not S.playbook_shortlist(df).loc[25]


def test_the_conditions_never_align_on_the_index():
    """The shortlist was one expression on the frame's own rows: a repeated index label must not matter."""
    df = _frame()
    df.index = [0] * len(df)
    assert S.playbook_shortlist(df).tolist() == S.playbook_shortlist(_frame()).tolist()


# ── 2. What each condition reads, pinned to the code ──────────────────────────────────────────────────────────────
# Columns the recipe's code names that are NOT inputs a blank could cost a stock its place on — each with its reason.
_NOT_INPUTS = {
    "sma_50d": "only the golden cross's history test reads it, and a blank there lifts that leg to neutral",
    "vstop_value": "read through dist_to_vstop, which is blank when the stop is blank or off-scale",
    "above_sma200": "derived by price_trend_inputs from close_price and sma_200d",
    "vstop_green": "derived by price_trend_inputs from the stop (dist_to_vstop) and close_price",
    "vstop_fresh": "derived by price_trend_inputs from last_vstop_change, the stop and close_price",
    "vol_ratio": "derived by price_trend_inputs from volume and vol_sma_20d",
}


def _strings_named_by(funcs) -> set:
    """Every string the given functions' code holds — constants, default arguments, and the keys and values of every
    settings table read — through every PRISM function they reach: the walk the recipe fingerprint makes."""
    seen, out, todo = set(), set(), list(funcs)

    def take(v):
        if isinstance(v, str):
            out.add(v)
        elif isinstance(v, dict):
            for k, x in v.items():
                take(k)
                take(x)
        elif isinstance(v, (list, tuple, set, frozenset)):
            for x in v:
                take(x)

    while todo:
        fn = todo.pop()
        if fn in seen:
            continue
        seen.add(fn)
        codes = [fn.__code__]
        while codes:
            for c in codes.pop().co_consts:
                codes.append(c) if inspect.iscode(c) else take(c)
        take(fn.__defaults__ or ())
        for name in sorted(S._referenced_globals(fn)):
            obj = fn.__globals__.get(name)
            if inspect.isfunction(obj) and S._is_project(obj):
                todo.append(obj)
            elif isinstance(obj, S._PLAIN) and not name.startswith("__"):
                take(obj)
    return out


def test_every_condition_says_what_it_reads():
    assert list(F.RECIPE_INPUTS) == CONDITIONS
    assert all(v == tuple(sorted(set(v))) and v for v in F.RECIPE_INPUTS.values())
    assert not set(F.BLANK_IS_A_FACT) & set(INPUTS)


def test_every_column_the_recipe_reads_is_classified():
    """A column added to the recipe's code (a new signal in a score, a new condition's input) fails here until it is
    named: an input of the conditions that read it, a blank-is-a-fact column, or one of _NOT_INPUTS with its reason."""
    scan = F.playbook_scan()
    named = _strings_named_by([*S._recipe_functions(scan), *scan.get("inputs", ())])
    vendor = {v for m in (de.COMMON_COLS, de.RATIO_COLS, de.INCOME_COLS, de.BALANCE_COLS, de.CASHFLOW_COLS,
                          de.SHAREHOLDING_COLS, de.TECHNICAL_COLS) for v in m.values()}
    derived = set(de.price_trend_inputs(pd.DataFrame(index=[0])).columns)
    read = named & (vendor | derived)
    assert read == set(INPUTS) | set(F.BLANK_IS_A_FACT) | set(_NOT_INPUTS), (
        f"unclassified: {sorted(read - set(INPUTS) - set(F.BLANK_IS_A_FACT) - set(_NOT_INPUTS))}; "
        f"classified but no longer read: {sorted(set(INPUTS) | set(F.BLANK_IS_A_FACT) | set(_NOT_INPUTS) - read)}")


@pytest.fixture(scope="module")
def real():
    if not _ARCHIVE.exists():
        pytest.skip("the archive copies are local only (gitignored)")
    return F.score(_ARCHIVE.read_bytes())[3]


_SOURCE = {"dist_to_vstop": "vstop_value"}            # a derived reading is blanked at its vendor source


def test_blanking_an_input_moves_only_the_conditions_that_read_it(real):
    """Every stock's value blanked at once and the copy re-scored with the current engine (rescore_snapshot): a
    condition that does not list the input must not move for a single stock."""
    base = S.playbook_conditions(J.rescore_snapshot(real))
    assert base.all(axis=1).equals(S.playbook_shortlist(real)), "re-scoring the copy changed its list"
    moved_elsewhere = {}
    for c in INPUTS:
        d2 = real.copy()
        d2[_SOURCE.get(c, c)] = np.nan
        now = S.playbook_conditions(J.rescore_snapshot(d2))
        moved = {k for k in CONDITIONS if not now[k].equals(base[k])}
        unlisted = sorted(moved - {k for k in CONDITIONS if c in F.RECIPE_INPUTS[k]})
        if unlisted:
            moved_elsewhere[c] = unlisted
    assert moved_elsewhere == {}


# ── 3. Each list says which inputs came in blank ──────────────────────────────────────────────────────────────────
def test_a_complete_universe_has_no_blanks():
    assert F.blanks(_frame()) == {"in": {}, "out": {}}


def test_a_member_lists_every_input_it_passed_without():
    df = _frame()
    df.loc[25, ["crs_50d", "ret_vs_industry_3m"]] = np.nan
    assert F.blanks(df) == {"in": {"NSE:S25": ["crs_50d", "ret_vs_industry_3m"]}, "out": {}}


def test_a_stock_out_on_a_blank_names_it_under_the_condition_it_failed():
    df = _frame()
    df.loc[25, "dist_to_vstop"] = np.nan
    b = F.blanks(df)
    assert b == {"in": {}, "out": {"NSE:S25": {"stop": ["dist_to_vstop"]}}}
    assert not S.playbook_shortlist(df).loc[25]


def test_every_failed_condition_is_named_with_its_own_blanks():
    df = _frame()
    df.loc[25, ["dist_to_vstop", "rev_lq"]] = np.nan                # out on the stop and on the quarter
    assert F.blanks(df)["out"] == {"NSE:S25": {"quarter": ["rev_lq"], "stop": ["dist_to_vstop"]}}


def test_one_failure_on_complete_data_judges_a_stock_out():
    """A condition failed on the data it has is a judgement, whatever else came in blank: the stock is not listed."""
    df = _frame()
    df.loc[25, "dist_to_vstop"] = np.nan
    df.loc[25, "rev_lq"] = 101.0                                     # the quarter is judged not strong
    assert F.blanks(df) == {"in": {}, "out": {}}


def test_a_blank_that_is_a_fact_is_never_listed():
    df = _frame()
    assert df["breakout_window"].isna().all() and df["golden_cross_days"].isna().all()
    assert F.blanks(df) == {"in": {}, "out": {}}


def test_a_blank_sector_is_blank_text():
    df = _frame()
    df.loc[25, "sector"] = " "
    df.loc[26, "sector"] = np.nan
    assert F.blanks(df)["in"] == {"NSE:S25": ["sector"], "NSE:S26": ["sector"]}


def test_an_absent_input_column_is_blank_for_every_stock():
    """Unverifiable is not passed: a column the copy lacks is a blank on every stock, never skipped."""
    df = _frame().drop(columns="crs_26w")
    b = F.blanks(df)
    assert set(b["in"]) == {f"NSE:S{k}" for k in range(20, 30)} and all(v == ["crs_26w"] for v in b["in"].values())


def test_a_symbol_on_the_list_is_never_named_off_it():
    """The loader merges on company_id without dropping a repeated row, so one id can be a member on one row and fail on
    another. A symbol on the list is not off it: `out` never names a member, or Kite would refuse the whole list."""
    df = pd.concat([_frame(), _frame().iloc[[25]]], ignore_index=True)   # member NSE:S25 again ...
    df.loc[len(df) - 1, "dist_to_vstop"] = np.nan                        # ... out on a blank stop alone
    cond = S.playbook_conditions(df)
    assert cond.iloc[25].all() and not cond.iloc[-1]["stop"] and cond.iloc[-1].drop("stop").all()
    assert "NSE:S25" not in F.blanks(df)["out"]


def test_the_map_is_sorted_and_the_same_whatever_the_row_order():
    df = _frame()
    df.loc[[21, 27], "crs_52w"] = np.nan
    df.loc[22, "adx_14w"] = np.nan
    df.loc[3, "dist_to_vstop"] = np.nan                              # out on the stop, but judged on the quarter
    a = F.blanks(df)
    b = F.blanks(df.sample(frac=1, random_state=7).reset_index(drop=True))
    assert json.dumps(a) == json.dumps(b)
    assert list(a["in"]) == sorted(a["in"]) and all(v == sorted(v) for v in a["in"].values())


def test_a_real_copy_states_what_it_could_not_judge(real):
    b = F.blanks(real)
    cond = S.playbook_conditions(real)
    ids = real["company_id"].astype(str)
    members = set(ids[cond.all(axis=1)])
    assert set(b["in"]) <= members and not set(b["out"]) & members
    assert b["in"] and b["out"]                                      # 53 and 64 on the 9 Oct copy
    by_id = dict(zip(ids, range(len(real))))
    for sym, failed in b["out"].items():
        row = by_id[sym]
        assert set(failed) == set(cond.columns[~cond.iloc[row].to_numpy()])
        for c, names in failed.items():
            assert names and set(names) <= set(F.RECIPE_INPUTS[c])
            assert all(not F.filled(real, n).iloc[row] for n in names)
    for sym, names in b["in"].items():
        assert all(not F.filled(real, n).iloc[by_id[sym]] for n in names)
    text = json.dumps(b, indent=2)
    assert not any(f'"{c}"' in text for c in F.BLANK_IS_A_FACT)
