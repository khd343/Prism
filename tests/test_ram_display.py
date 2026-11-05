"""
test_ram_display.py — RAM on PRISM's screens (2026-11-05).

The Technicals tab now carries four columns another job fills each night: RAM, RAM Rank, RAM Pct and RAM Date. PRISM
shows them, and only the records job's order check reads one, RAM Rank. Pinned here, each against the break it exists to catch:

  1. The loader: each header lands on its own column, a blank stays blank (never 0), and RAM Date survives whatever the
     XLSX hands it (text, a real date cell, a blank) where the numeric coercion would have blanked it. An old copy
     without the headers loads, and the engine's schema guard gives every reader the four columns as blanks.
  2. The daily Data check: a date in a "number column" would have been a false red every night.
  3. The tips: one source (the glossary), read by the Tear-Sheet's ?, the Reference, the Deep Scanner and the 🎯 Scans.
  4. The 🔢 RAM view, first and default on every list (the 🚀 Playbook, B and C, then 🌬️ and 🌱): the table the operator reads,
     by RAM Rank with blanks last. 🏆 Core and 📈 Technical, on every list, are exactly main's.
  5. The Deep Scanner: main's views, and RAM Rank in its sort dropdown, low numbers first, blanks last, on screen when sorted.
  6. The Tear-Sheet: its own RAM RANK row under the ⏱️ strip, in the strip's look, never a chip on it, and no nan.
  7. Nothing scores on them: every score, tier, rank, verdict and flag identical with and without RAM values; every recipe's
     fingerprint unmoved; only the loader, the Data check, the three screens and the records job's order check, which reads
     RAM Rank and decides nothing (tests/test_scan_order_reading.py), name the columns at all.
  8. No word of action, no rule and no person in their labels and tips.

tests/test_scans_tab_runs.py runs the view in the real tab.

Run with: pytest tests/test_ram_display.py -v
"""
import ast
import contextlib
import html
import io
import json
import re
import sys
import textwrap
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import COLORS, SHEET_TAB_NAMES  # noqa: E402
from core import data_engine as de  # noqa: E402
from core import run_scoring_pipeline  # noqa: E402
from core.cyclicality_map import INDUSTRY_TIER  # noqa: E402
from ui import data_check_job as D  # noqa: E402
from ui import ui_scans as S  # noqa: E402
from ui.ui_components import _RAW_GLOSSARY  # noqa: E402
from ui.ui_scanner import _SCANNER_HEADER_TIPS, ds_view_columns  # noqa: E402


# ── 1. The loader ────────────────────────────────────────────────────────────────────────────────────────────────────
def test_each_sheet_header_lands_on_its_own_column():
    """Four different values, so a swapped pair cannot pass. The header is `RAM Pct`, not `RAM %`: the sheet's own
    matcher folds names to letters and digits, and RAM and RAM % would both read as `ram`."""
    raw = pd.DataFrame({"companyId": ["NSE:A"], "Close Price": [10.0], "RAM": [0.123456], "RAM Rank": [312.0],
                        "RAM Pct": [78.44], "RAM Date": ["2026-11-03"]})
    out = de._apply_column_mapping(raw, de.TECHNICAL_COLS, "technical").iloc[0]
    assert (out["ram_score"], out["ram_rank"], out["ram_pct"], out["ram_date"]) == (0.123456, 312.0, 78.44, "2026-11-03")


def test_no_two_technicals_headers_fold_alike():
    """The sheet's own matcher (Prism.gs _norm) folds a header to lower-case letters and digits. A header reading `RAM %`
    would fold to `ram` and take RAM's values; `RAM Pct` keeps the four apart."""
    folded = [re.sub(r"[^a-z0-9]", "", h.lower()) for h in de.TECHNICAL_COLS]
    assert len(folded) == len(set(folded)), sorted(f for f in set(folded) if folded.count(f) > 1)
    assert {"ram", "ramrank", "rampct", "ramdate"} <= set(folded)


def test_the_wrong_tab_guard_still_fires_with_the_ram_headers_in_the_map():
    """CLAUDE.md §0: never weaken it. Another tab handed over as Technicals holds none of its headers."""
    other_tab = pd.DataFrame({"companyId": ["NSE:A"], "ROCE Median 10 Years": [25.0]})
    with pytest.raises(ValueError, match="wrong sheet/tab"):
        de._apply_column_mapping(other_tab, de.TECHNICAL_COLS, "technical")


def _coerced(**cols) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    frame = pd.DataFrame({"company_id": [f"NSE:S{i}" for i in range(n)], "name": [f"S{i}" for i in range(n)]})
    for col, values in cols.items():
        frame[col] = pd.Series(values, index=frame.index)
    return de.coerce_numeric_columns(frame)


@pytest.mark.parametrize("col", ["ram_score", "ram_rank", "ram_pct"])
def test_a_blank_number_stays_blank_never_zero(col):
    out = _coerced(**{col: [12.5, np.nan, None, "", "not a number"]})[col]
    assert out.iloc[0] == 12.5
    assert out.iloc[1:].isna().all(), f"{col}: a blank reads as {out.iloc[1:].tolist()}, never a number"


# What the XLSX can hand for one RAM Date cell, and what the loader must keep. Google Sheets may turn the sync's
# 2026-11-03 into a real date cell; openpyxl reads that as a datetime and calamine as a Timestamp.
_ISO = "2026-11-03"
_DATE_CELLS = [("text", "2026-11-03", _ISO), ("datetime", datetime(2026, 11, 3), _ISO),
               ("timestamp", pd.Timestamp("2026-11-03"), _ISO), ("date", date(2026, 11, 3), _ISO),
               # A session is the day the cell WRITES, whatever follows it, and is never converted to another zone.
               ("text_with_a_time", "2026-11-03 23:10:00", _ISO), ("zoned_utc", "2026-11-03T17:40:00Z", _ISO),
               ("zoned_just_after_midnight_in_india", "2026-11-03T01:00+05:30", _ISO),
               ("zoned_date_cell", pd.Timestamp("2026-11-03T01:00+05:30"), _ISO)]
_NOT_A_SESSION = [("nan", np.nan), ("nat", pd.NaT), ("none", None), ("empty", ""),
                  ("words", "not a date"),
                  ("impossible", "2026-13-45"),   # never crash on a stranger's cell (CLAUDE.md §5)
                  ("either_month", "03/11/2026"),  # 3 Nov or 11 Mar: never guessed
                  ("serial", 46329.0),             # to_datetime would call it nanoseconds from 1970: "1 Jan"
                  ("integer_serial", 46329), ("year_as_an_integer", 2026),
                  ("year", "2026"), ("year_and_month", "2026-11"),  # pandas' ISO reading makes these 1 Jan and 1 Nov
                  ("compact", "20261103"), ("a_digit_too_many", "2026-11-035")]


@pytest.mark.parametrize("cell,want", [c[1:] for c in _DATE_CELLS], ids=[c[0] for c in _DATE_CELLS])
def test_a_ram_date_survives_the_numeric_coercion_in_every_form(cell, want):
    out = _coerced(ram_date=[cell, cell])["ram_date"]
    assert out.tolist() == [want, want]


@pytest.mark.parametrize("cell", [c[1] for c in _NOT_A_SESSION], ids=[c[0] for c in _NOT_A_SESSION])
def test_what_is_not_a_session_reads_blank(cell):
    out = _coerced(ram_date=["2026-11-03", cell])["ram_date"]
    assert out.iloc[0] == _ISO, "a good cell beside it survives"
    assert pd.isna(out.iloc[1]), f"{cell!r} read as {out.iloc[1]!r}"


@pytest.mark.parametrize("cell", [c[1] for c in _NOT_A_SESSION], ids=[c[0] for c in _NOT_A_SESSION])
def test_with_no_good_cell_beside_it_what_is_not_a_session_still_reads_blank(cell):
    """A good ISO cell first would let pandas infer ISO from it, so here only the reading's own guard can say no."""
    out = _coerced(ram_date=[cell, cell])["ram_date"]
    assert out.isna().all(), f"{cell!r} read as {out.tolist()}"


def test_a_zoned_cell_beside_plain_dates_reads_its_written_day_and_never_raises():
    """pandas hands back an object column for this mix, and .dt then raised: one odd cell in a display-only column took
    down the whole app and two scheduled jobs (CLAUDE.md §5: nothing from outside may crash a reader)."""
    out = de.ram_date_text(pd.Series(["2026-11-04", "2026-11-04T17:40:00Z"], dtype=object))
    assert out.tolist() == ["2026-11-04", "2026-11-04"]


def test_two_different_offsets_read_their_written_days_never_another_zones():
    out = de.ram_date_text(pd.Series(["2026-11-04T01:00+05:30", "2026-11-03T23:00-04:00"], dtype=object))
    assert out.tolist() == ["2026-11-04", "2026-11-03"], "in UTC these two would swap: 3 Nov and 4 Nov"


def test_a_whole_column_of_real_dates_reads_the_same():
    """No emoji row above the names: pandas then types the column itself, as datetime64."""
    frame = pd.DataFrame({"company_id": ["NSE:A", "NSE:B"], "ram_date": pd.to_datetime(["2026-11-03", None])})
    out = de.coerce_numeric_columns(frame)["ram_date"]
    assert out.iloc[0] == _ISO and pd.isna(out.iloc[1])


def _workbook(ram_cells: list, headers: tuple = ("RAM", "RAM Rank", "RAM Pct", "RAM Date")) -> io.BytesIO:
    """An XLSX in the live sheet's own layout: the emoji section row, then the names, then one row per stock in
    `ram_cells` (each a tuple under `headers`). A date cell is written as a real date cell, as Google exports it."""
    from openpyxl import Workbook
    mapped = {"ratio": "ROCE Median 10 Years", "income": "PAT Growth 5 Years", "balance": "Debt",
              "cashflow": "Operating Cash Flow", "shareholding": "Promoter Holdings", "technical": "Close Price"}
    book = Workbook()
    book.remove(book.active)
    for key, tab in SHEET_TAB_NAMES.items():
        sheet = book.create_sheet(tab)
        extra = headers if key == "technical" else ()
        sheet.append(["📋 Details", None, "📊 Section", *(None for _ in extra)])
        sheet.append(["companyId", "Name", mapped[key], *extra])
        for i, cells in enumerate(ram_cells):
            sheet.append([f"NSE:S{i}", f"S{i} Ltd", 10.0 + i, *(cells if key == "technical" else ())])
    buf = io.BytesIO()
    book.save(buf)
    buf.seek(0)
    return buf


def _load(buf: io.BytesIO, engine: str, monkeypatch) -> dict:
    monkeypatch.setattr(de, "_xlsx_engine", lambda: engine)
    with contextlib.redirect_stdout(io.StringIO()):
        return de.load_all_csvs("upload", uploaded_files={"workbook": buf})


@pytest.mark.parametrize("engine", ["openpyxl", "calamine"])
def test_a_workbook_hands_over_text_date_cells_and_blanks_alike(engine, monkeypatch):
    """The real parse: two rows of headers, one date cell, one text date, one row RAM did not rank, one it did not reach."""
    pytest.importorskip("python_calamine" if engine == "calamine" else "openpyxl")
    buf = _workbook([(0.5, 312, 78.44, datetime(2026, 11, 3)),
                     (0.4, 313, 78.20, "2026-11-03"),
                     (None, None, None, "2026-11-03"),
                     (None, None, None, None)])
    with contextlib.redirect_stdout(io.StringIO()):
        out = de.coerce_numeric_columns(de.merge_datasets(_load(buf, engine, monkeypatch)))
    assert out["ram_date"].tolist()[:3] == [_ISO, _ISO, _ISO] and pd.isna(out["ram_date"].iloc[3])
    assert out["ram_rank"].tolist()[:2] == [312.0, 313.0] and out["ram_rank"].iloc[2:].isna().all()
    assert out["ram_pct"].tolist()[:2] == [78.44, 78.20] and out["ram_pct"].iloc[2:].isna().all()
    assert out["ram_score"].tolist()[:2] == [0.5, 0.4] and out["ram_score"].iloc[2:].isna().all()


@pytest.mark.parametrize("engine", ["openpyxl", "calamine"])
def test_one_zoned_cell_in_a_workbook_does_not_stop_the_load(engine, monkeypatch):
    """The real parse, as the review's copy of the preview workbook took it: a zoned cell among real date cells and text."""
    pytest.importorskip("python_calamine" if engine == "calamine" else "openpyxl")
    buf = _workbook([(0.5, 312, 78.44, datetime(2026, 11, 3)), (0.4, 313, 78.20, "2026-11-03"),
                     (0.3, 314, 78.00, "2026-11-03T17:40:00Z"), (None, None, None, "2026-11-03T01:00+05:30")])
    with contextlib.redirect_stdout(io.StringIO()):
        out = de.coerce_numeric_columns(de.merge_datasets(_load(buf, engine, monkeypatch)))
    assert out["ram_date"].tolist() == [_ISO] * 4
    assert out["ram_rank"].tolist()[:3] == [312.0, 313.0, 314.0], "and the rest of the tab loads as ever"


RAM_COLUMNS = ("ram_score", "ram_rank", "ram_pct", "ram_date")
_MAPPED = sorted({v for m in (de.COMMON_COLS, de.RATIO_COLS, de.INCOME_COLS, de.BALANCE_COLS, de.CASHFLOW_COLS,
                              de.SHAREHOLDING_COLS, de.TECHNICAL_COLS) for v in m.values()})


def _old_copy(n: int = 6) -> pd.DataFrame:
    """A merged frame from before the RAM columns (CLAUDE.md §6: every other mapped column materialized as NaN)."""
    frame = pd.DataFrame({c: [np.nan] * n for c in _MAPPED if c not in RAM_COLUMNS})
    frame["company_id"] = [f"NSE:S{i}" for i in range(n)]
    frame["name"] = [f"S{i} Ltd" for i in range(n)]
    return frame


def test_an_old_copy_loads_and_every_reader_finds_the_columns_blank():
    """Archived vintages, the scan-records job, Movers, an old local CSV: none has the headers. The engine's schema guard
    materializes them as blanks, so every surface reads one shape and none needs an `if`."""
    with contextlib.redirect_stdout(io.StringIO()):
        out = de.compute_derived_signals(de.coerce_numeric_columns(_old_copy()))
    for col in RAM_COLUMNS:
        assert col in out.columns and out[col].isna().all(), col


def test_an_old_workbook_without_the_headers_still_loads(monkeypatch):
    buf = _workbook([(), ()], headers=())
    with contextlib.redirect_stdout(io.StringIO()):
        out = de.coerce_numeric_columns(de.merge_datasets(_load(buf, "openpyxl", monkeypatch)))
    assert len(out) == 2 and "ram_rank" not in out.columns, "the loader invents nothing; the engine's guard fills it"


def test_tonights_sheet_with_all_four_columns_blank_loads_blank(monkeypatch):
    buf = _workbook([(None, None, None, None)] * 3)
    with contextlib.redirect_stdout(io.StringIO()):
        out = de.coerce_numeric_columns(de.merge_datasets(_load(buf, "openpyxl", monkeypatch)))
    for col in RAM_COLUMNS:
        assert out[col].isna().all(), col


# ── 2. The daily Data check (ui/data_check_job.py): its red run emails the operator, who is away for six months ─────────
def _reading(results, name):
    return next(r for r in results if r.name == name)


@pytest.mark.parametrize("engine", ["openpyxl", "calamine"])
def test_the_first_ram_night_does_not_turn_the_data_check_red(engine, monkeypatch):
    """The type check reads the raw tabs and calls anything to_numeric cannot read 'text in a number column': a date, as
    text or as a date cell, would be 100% text and a false red every night."""
    pytest.importorskip("python_calamine" if engine == "calamine" else "openpyxl")
    buf = _workbook([(0.5, 312, 78.44, datetime(2026, 11, 3)), (0.4, 313, 78.2, datetime(2026, 11, 3)),
                     (0.3, 314, 78.0, "2026-11-03"), (None, None, None, "2026-11-03")])
    reading = _reading(D.check_types(_load(buf, engine, monkeypatch)), "Number columns")
    assert reading.status == "ok", reading.detail


def test_tonights_blank_columns_do_not_turn_the_data_check_red(monkeypatch):
    buf = _workbook([(None, None, None, None)] * 3)
    reading = _reading(D.check_types(_load(buf, "openpyxl", monkeypatch)), "Number columns")
    assert reading.status == "ok", reading.detail


def test_only_the_date_is_excused_a_ram_number_holding_words_is_still_a_break():
    raw = {"technical": pd.DataFrame({"company_id": list("abcd"), "ram_rank": ["high", "low", "mid", "top"]})}
    reading = _reading(D.check_types(raw), "Number columns")
    assert reading.status == "fail" and "ram_rank" in reading.detail


@pytest.mark.parametrize("blank", ["all blank", "absent"])
def test_ram_is_no_key_column_so_its_blanks_fail_no_coverage_check(blank):
    stocks = pd.DataFrame({c: np.arange(1.0, 51.0) for c in D.CORE_COLUMNS + D.OTHER_COLUMNS})
    if blank == "all blank":
        stocks = stocks.assign(**{c: np.nan for c in RAM_COLUMNS})
    reading = _reading(D.check_coverage(stocks), "Key columns")
    assert reading.status == "ok", reading.detail


# ── 3. The tips: one source, plain words ─────────────────────────────────────────────────────────────────────────────
TIP_RANK = ("RAM's rank among about 1,400 liquid NSE main-board stocks: 1 is the strongest. Momentum over 1 to 12 "
            "months, risk-adjusted, from NSE's own closes. Blank: not ranked (BSE, thin, under a year of prices, or a "
            "check doubted it). Display only: it changes no score or list; the order check on the 🎯 Scans cards measures it.")
TIP_PCT = "RAM's score as a percentile, 100 the strongest. Display only: it changes no score, list or record."
CORE, TECH = S.SCAN_VIEWS
RAM = "🔢 RAM"
_PLAY = next(s for s in S.SCANS if s["key"] == "playbook")
_LEAD = next(s for s in S.SCANS if s["key"] == "leaders")
_SEED = next(s for s in S.SCANS if s["key"] == "compounder")
_B, _C = S.playbook_layer_scan(_PLAY, "no_quarter"), S.playbook_layer_scan(_PLAY, "price_only")
LISTS = (_PLAY, _B, _C, _LEAD, _SEED)
_LIST_IDS = [s["key"] for s in LISTS]


def test_the_tips_have_one_source_read_by_every_surface():
    assert _RAW_GLOSSARY["RAM Rank"] == TIP_RANK and _RAW_GLOSSARY["RAM %"] == TIP_PCT
    assert _SCANNER_HEADER_TIPS["ram_rank"] == TIP_RANK, "the Deep Scanner"
    cfg = S.table_config("03 Oct", _PLAY)
    assert cfg["ram_rank"]["help"] == TIP_RANK and cfg["ram_pct"]["help"] == TIP_PCT, "the 🎯 Scans table"
    assert cfg["ram_score"]["help"] == _RAW_GLOSSARY["RAM"], "the raw RAM, in plain words"
    assert "z-score" not in _RAW_GLOSSARY["RAM"] and "Display only" in _RAW_GLOSSARY["RAM"]
    assert cfg["#"]["help"] and cfg["symbol"]["help"]


# ── 4. The 🔢 RAM view on every list: Core and Technical are main's ───────────────────────────────────────────────────
@pytest.mark.parametrize("scan", LISTS, ids=_LIST_IDS)
def test_the_view_is_on_every_list_and_opens_first(scan):
    assert S.RAM_VIEW == RAM and S.TABLE_VIEWS == (RAM, CORE, TECH) and S.SCAN_VIEWS == (CORE, TECH)
    assert S.table_columns(scan, S.TABLE_VIEWS[0]) == S.RAM_VIEW_COLUMNS


@pytest.mark.parametrize("scan", (_LEAD, _SEED), ids=lambda s: s["key"])
def test_the_view_leaves_out_what_a_list_adds_to_core_and_core_keeps_it(scan):
    """🌬️'s and 🌱's New, 🌱's Cycle ↓ and its narrow Stock are Core's: the view reads the same on every list (2026-11-05)."""
    view, core = S.table_columns(scan, RAM), S.table_columns(scan, CORE)
    assert "_new" not in view and "_cycle" not in view
    assert "_new" in core and ("_cycle" in core) == (scan is _SEED)
    drawn = S.table_widths(S.ram_view_scan(scan))
    assert (drawn["name"], drawn["sector"]) == (312, 170)
    assert S.table_widths(scan) == {**S.TABLE_WIDTHS, **scan.get("widths", {})}, "Core's widths stay the list's own"


_VIEW_HEADERS = {"#": "#", "ram_rank": "RAM Rank", "symbol": "Symbol", "name": "Stock", "ram_pct": "RAM %", "ram_score": "RAM",
                 "close_price": "Price ₹", "market_category": "Category", "sector": "Sector", "_liq_cr": "₹ Cr/day"}


@pytest.mark.parametrize("scan", LISTS, ids=_LIST_IDS)
def test_the_views_columns_and_headers_in_their_order(scan):
    cols = S.table_columns(scan, RAM)
    assert cols == tuple(_VIEW_HEADERS)
    cfg = S.table_config("03 Oct", S.ram_view_scan(scan))
    assert {c: cfg[c]["label"] for c in cols} == _VIEW_HEADERS


def test_the_numbers_read_as_the_brief_says():
    cfg = S.table_config("03 Oct", _PLAY)
    formats = {c: (cfg[c]["type_config"]["type"], cfg[c]["type_config"].get("format")) for c in ("ram_rank", "ram_pct", "ram_score")}
    assert formats == {"ram_rank": ("number", "%.0f"), "ram_pct": ("number", "%.1f"), "ram_score": ("number", "%.2f")}
    assert cfg["_liq_cr"]["type_config"]["format"] == "%.2f", "the same reading and format as the other views"


@pytest.mark.parametrize("scan", LISTS, ids=_LIST_IDS)
def test_the_view_fits_the_budget_shrinking_nothing_and_stock_takes_the_rest(scan):
    w = S.table_widths(S.ram_view_scan(scan))
    cols = S.table_columns(scan, RAM)
    assert sum(w[c] for c in cols) == 1030
    # Each column's header and widest value, measured 2026-11-05 in the grid's font (test_scan_views' method) on the 4 Nov copy,
    # and that evening on the live sheet with 🌬️ and 🌱: 🌱's longest name needs 310, so Stock keeps 312 and Sector its 170.
    floors = {"#": 44, "ram_rank": 75, "symbol": 103, "ram_pct": 57, "ram_score": 45, "name": 310}
    assert {c: w[c] for c in floors} == {c: max(w[c], floors[c]) for c in floors}
    assert (w["name"], w["sector"]) == (312, 170)
    main = {"close_price": 80, "market_category": 75, "_liq_cr": 65}
    assert {c: w[c] for c in main} == main, "main's widths"


# Main's (9284e96) columns and widths, typed out: the old tests hold them too, and this says it of every list at once.
_MAIN_CORE = ("rank", "_new", "name", "market_category", "sector", "composite_score", "momentum_score", "improvement_score",
              "_strong_q", "_measured_flags", "weinstein_stage", "_liq_cr")
_MAIN_TECH = ("rank", "_new", "name", "close_price", "_liq_cr", "dist_to_vstop", "momentum_score", "d49_momentum_quality",
              "rsi_14d", "breakout_score", "crs_52w", "weinstein_stage")
_MAIN_WIDTHS = {"rank": 50, "_new": 55, "name": 175, "market_category": 75, "sector": 95, "composite_score": 75,
                "momentum_score": 85, "improvement_score": 100, "_strong_q": 40, "_measured_flags": 125,
                "weinstein_stage": 90, "_liq_cr": 65, "_cycle": 60, "_layer": 55, "close_price": 80, "dist_to_vstop": 60,
                "d49_momentum_quality": 110, "rsi_14d": 50, "breakout_score": 85, "crs_52w": 60}


def _core(*out, after_rank=None, after_quarter=None):
    cols = [c for c in _MAIN_CORE if c not in out]
    if after_rank:
        cols.insert(1, after_rank)
    if after_quarter:
        cols.insert(cols.index("_strong_q") + 1, after_quarter)
    return tuple(cols)


_MAIN_VIEWS = {"playbook": (_core("_strong_q"), _MAIN_TECH, {"name": 185, "sector": 125}),
               "leaders": (_MAIN_CORE, _MAIN_TECH, {}),
               "compounder": (_core(after_quarter="_cycle"), _MAIN_TECH,
                              {"name": 150, "sector": 80, "composite_score": 65, "_measured_flags": 120, "weinstein_stage": 85}),
               "no_quarter": (_core("_new", "_strong_q", after_rank="_layer"),
                              tuple(c for c in (*_MAIN_TECH[:1], "_layer", *_MAIN_TECH[1:]) if c != "_new"),
                              {"name": 185, "sector": 125}),
               "price_only": (_core("_new", after_rank="_layer"),
                              tuple(c for c in (*_MAIN_TECH[:1], "_layer", *_MAIN_TECH[1:]) if c != "_new"), {})}


@pytest.mark.parametrize("scan", LISTS, ids=lambda s: s["key"])
def test_core_and_technical_are_mains_on_every_list(scan):
    core, tech, own = _MAIN_VIEWS[scan["key"]]
    assert S.table_columns(scan, CORE) == core and S.table_columns(scan) == core and S.table_columns(scan, TECH) == tech
    w = S.table_widths(scan)
    assert {c: w[c] for c in (*core, *tech)} == {c: {**_MAIN_WIDTHS, **own}[c] for c in (*core, *tech)}


def _order_market(**over) -> pd.DataFrame:
    """Eight stocks. RAM Rank 1, a tie at 2, then 5 and 9, and three with no RAM. Inside the tie the stronger momentum goes
    first; among the blanks S7 and S1 tie on momentum, so the better PRISM Rank goes first (today's order)."""
    frame = pd.DataFrame({"company_id": [f"NSE:SYM{i}" for i in range(8)], "name": [f"S{i} Ltd" for i in range(8)],
                          "momentum_score": [90.0, 80.0, 70.0, 75.0, 60.0, 50.0, 40.0, 80.0],
                          "rank": [1.0, 6.0, 3.0, 4.0, 5.0, 7.0, 8.0, 2.0],
                          "ram_rank": [5.0, np.nan, 2.0, 2.0, np.nan, 1.0, 9.0, np.nan],
                          "ram_pct": [80.0, np.nan, 95.0, 95.0, np.nan, 99.9, 70.0, np.nan],
                          "ram_score": [1.5, np.nan, 2.0, 2.0, np.nan, 2.9, 0.5, np.nan],
                          "ram_date": ["2026-11-04"] * 8})
    return frame.assign(**over)


_EVERY_STOCK = {"key": "playbook", "order": "momentum", "mask": lambda d: pd.Series(True, index=d.index)}


def _names(table) -> list:
    return [n.split()[0] for n in table["name"]]


def test_the_view_lists_by_ram_rank_with_blanks_last_and_todays_order_inside_ties_and_blanks():
    market = _order_market()
    table = S.ram_view_table(S.scan_table(market, _EVERY_STOCK), market)
    assert _names(table) == ["S5", "S3", "S2", "S0", "S6", "S7", "S1", "S4"]
    assert table["ram_rank"].iloc[:5].tolist() == [1.0, 2.0, 2.0, 5.0, 9.0] and table["ram_rank"].iloc[5:].isna().all()


def test_ties_keep_todays_order_on_a_list_as_long_as_the_real_ones():
    """pandas' default sort keeps ties on short lists only: from about 20 rows it does not, and the real lists run to hundreds."""
    n = 200
    market = pd.DataFrame({"company_id": [f"NSE:SYM{i}" for i in range(n)], "name": [f"S{i} Ltd" for i in range(n)],
                           "momentum_score": np.arange(n, 0, -1, dtype=float), "rank": np.arange(1.0, n + 1.0),
                           "ram_rank": (np.arange(n) % 3 + 1).astype(float)})
    table = S.ram_view_table(S.scan_table(market, _EVERY_STOCK), market)
    assert table["name"].tolist() == [f"S{i} Ltd" for tie in range(3) for i in range(tie, n, 3)]


@pytest.mark.parametrize("case", ["blank", "absent"])
def test_a_list_with_no_ram_is_todays_order_byte_for_byte(case):
    market = (_order_market(ram_rank=np.nan, ram_pct=np.nan, ram_score=np.nan, ram_date=np.nan) if case == "blank"
              else _order_market().drop(columns=list(RAM_COLUMNS)))
    today = S.scan_table(market, _EVERY_STOCK)
    view = S.ram_view_table(today, market)
    pd.testing.assert_frame_equal(view[list(today.columns)], today)
    assert view["ram_rank"].isna().all() and view["ram_score"].isna().all(), "blank, never 0"


def test_the_place_is_the_list_as_shown_and_a_filter_renumbers_it():
    market = _order_market()
    table = S.ram_view_table(S.scan_table(market, _EVERY_STOCK), market)
    assert table["#"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    kept = S.scan_table(market, _EVERY_STOCK, keep_ids={"NSE:SYM0", "NSE:SYM4", "NSE:SYM6"})
    narrow = S.ram_view_table(kept, market)
    assert _names(narrow) == ["S0", "S6", "S4"] and narrow["#"].tolist() == [1, 2, 3]


def test_the_symbol_is_the_vendors_without_its_exchange():
    market = _order_market(company_id=["NSE:LAURUSLABS", "BSE:500325", "NSE:M&M", "NSE:A", "NSE:B", "NSE:C", "NSE:D", "NSE:E"])
    table = S.ram_view_table(S.scan_table(market, _EVERY_STOCK), market).set_index("name")
    assert table.loc["S0 Ltd", "symbol"] == "LAURUSLABS" and table.loc["S1 Ltd", "symbol"] == "500325"
    assert table.loc["S2 Ltd", "symbol"] == "M&M"


def _caption(market: pd.DataFrame, keep_ids=None) -> str:
    table = S.ram_view_table(S.scan_table(market, _EVERY_STOCK, keep_ids=keep_ids), market)
    return S.ram_view_order_words(table, "Momentum, high to low", market)


def test_the_caption_says_what_the_view_is_sorted_by_and_how_old_its_ranks_are():
    assert _caption(_order_market()) == "RAM Rank, strongest first · no RAM last · RAM of 4 Nov"


@pytest.mark.parametrize("dates,want", [
    (["2026-11-03"] * 4 + ["2026-11-04"] * 4, "RAM of 4 Nov"),                # the newest of the rows shown
    (["2026-11-04"] * 4 + ["2026-11-03"] * 4, "RAM of 4 Nov"),                # wherever it sits
    (["2026-09-30"] * 4 + ["2026-10-01"] * 4, "RAM of 1 Oct"),                # across a month end, not by the day's digits
    ([np.nan] * 4 + ["2026-12-01"] * 4, "RAM of 1 Dec"),                      # a blank date among them, and no zero-padding
], ids=["newest_last", "newest_first", "across_a_month_end", "some_blank"])
def test_the_caption_reads_the_newest_ram_date_when_the_rows_dates_differ(dates, want):
    assert _caption(_order_market(ram_date=dates)).endswith(f"· no RAM last · {want}")


def test_the_newest_is_of_the_rows_shown_not_of_the_copy():
    dates = ["2026-11-03"] * 7 + ["2026-11-04"]
    assert _caption(_order_market(ram_date=dates), keep_ids={"NSE:SYM0", "NSE:SYM2"}).endswith("· RAM of 3 Nov")


def test_ranks_with_no_date_at_all_say_nothing_of_their_age():
    assert _caption(_order_market(ram_date=np.nan)) == "RAM Rank, strongest first · no RAM last"


def test_a_list_with_no_ram_on_a_copy_that_has_it_says_this_list_not_this_copy():
    """A BSE-only list on a copy where 1,244 stocks have RAM: 'no RAM on this copy' would be false."""
    bse_unranked = _order_market(company_id=[f"BSE:{i}" if i in (1, 4, 7) else f"NSE:SYM{i}" for i in range(8)])
    assert _caption(bse_unranked, keep_ids={"BSE:1", "BSE:4", "BSE:7"}) == "Momentum, high to low · no RAM in this list"


@pytest.mark.parametrize("case", ["blank", "absent"])
def test_a_copy_with_no_ram_at_all_says_so(case):
    market = (_order_market(ram_rank=np.nan, ram_pct=np.nan, ram_score=np.nan, ram_date=np.nan) if case == "blank"
              else _order_market().drop(columns=list(RAM_COLUMNS)))
    assert _caption(market) == "Momentum, high to low · no RAM on this copy"


@pytest.mark.parametrize("scan", LISTS, ids=lambda s: s["key"])
def test_the_lists_own_frame_carries_no_ram_so_core_and_technical_cannot_show_it(scan):
    frame = S.scan_table(_order_market(), dict(scan, mask=_EVERY_STOCK["mask"]))
    assert not set(RAM_COLUMNS) & set(frame.columns), "only the view brings RAM in (ram_view_table)"


def _render_view(table: pd.DataFrame, scan: dict):
    """The view's table through Streamlit's own harness, with the app's own column config and order."""
    from streamlit.testing.v1 import AppTest

    def _app():
        import streamlit as st
        st.dataframe(st.session_state["table"], column_config=st.session_state["config"],
                     column_order=st.session_state["order"], hide_index=True)

    at = AppTest.from_function(_app)
    at.session_state["table"] = table
    at.session_state["config"] = S.table_config("03 Oct", S.ram_view_scan(scan))
    at.session_state["order"] = list(S.table_columns(scan, RAM))
    at.run(timeout=90)
    return at


@pytest.mark.parametrize("case", ["filled", "tonight's blanks", "an old copy"])
def test_the_view_renders_in_every_case_with_its_columns_in_order(case):
    market = _order_market()
    if case == "tonight's blanks":
        market = market.assign(ram_score=np.nan, ram_rank=np.nan, ram_pct=np.nan, ram_date=np.nan)
    elif case == "an old copy":
        market = market.drop(columns=list(RAM_COLUMNS))
    at = _render_view(S.ram_view_table(S.scan_table(market, _EVERY_STOCK), market), _PLAY)
    assert not at.exception, [str(e.value) for e in at.exception]
    grid = at.dataframe[0]
    assert list(grid.proto.column_order) == list(_VIEW_HEADERS)
    got = grid.value["ram_rank"].iloc[:2].tolist()
    assert got == [1.0, 2.0] if case == "filled" else pd.isna(got).all(), got


# ── 5. The Deep Scanner: main's views, and RAM Rank in its sort dropdown ──────────────────────────────────────────────
_TREE = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
_SORT = "RAM Rank ↑"


def _app_dict(name: str) -> dict:
    node = next(n.value for n in ast.walk(_TREE) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return ast.literal_eval(node)


# Main's (9284e96) Deep Scanner tables, whole.
_MAIN_DS_VIEWS = {
    "🏆 Core": ["rank", "name", "verdict_direction", "wealth_tier", "sector", "market_category", "traded_value_cr",
        "composite_score", "gate_pass", "improvement_score", "momentum_score"],
    "📊 Quality": ["rank", "name", "quality_score", "moat_score", "growth_score", "improvement_score", "cash_score", "roce",
        "roce_expansion", "roce_inflection", "opm", "opm_acceleration"],
    "💰 Valuation": ["rank", "name", "close_price", "fair_value_qglp", "valuation_score", "expected_excess_return", "pe",
        "pb_ratio", "peg", "dividend_yield", "fcf_yield", "buy_zone_label"],
    "🔬 Forensic": ["rank", "name", "red_flag_count", "red_flag_list", "piotroski_fscore", "schilit_pass",
        "forensic_multiplier", "cfo_to_pat", "accruals_ratio", "debt_to_equity", "data_coverage_pct"],
    "📈 Technical": ["rank", "name", "dist_to_vstop", "momentum_score", "d49_momentum_quality", "rsi_14d", "breakout_score",
        "d48_breakout_readiness", "dist_ath", "crs_52w", "weinstein_stage"],
    "👥 Ownership": ["rank", "name", "governance_bonus", "promoter_holdings", "change_promoter_1y", "pledged_percentage",
        "fii_holdings", "change_fii_lq", "dii_holdings", "change_dii_lq", "smart_money_flow"],
}
_MAIN_DS_W = {
    "rank": 50, "name": 180, "verdict_direction": 80, "wealth_tier": 93, "sector": 115, "market_category": 80,
    "traded_value_cr": 65, "composite_score": 97, "gate_pass": 60, "quality_score": 105, "moat_score": 60,
    "growth_score": 65, "improvement_score": 95, "cash_score": 55, "roce": 60, "roce_expansion": 95,
    "roce_inflection": 90, "opm": 55, "opm_acceleration": 105, "close_price": 80, "fair_value_qglp": 70,
    "valuation_score": 105, "expected_excess_return": 70, "pe": 60, "pb_ratio": 55, "peg": 60, "dividend_yield": 80,
    "fcf_yield": 70, "buy_zone_label": 120, "market_cap": 90, "red_flag_count": 75, "red_flag_list": 190,
    "piotroski_fscore": 70, "schilit_pass": 70, "forensic_multiplier": 90, "cfo_to_pat": 75, "accruals_ratio": 80,
    "debt_to_equity": 55, "data_coverage_pct": 75, "dist_to_vstop": 60, "momentum_score": 105,
    "d49_momentum_quality": 110, "rsi_14d": 50, "breakout_score": 105, "d48_breakout_readiness": 90, "dist_ath": 65,
    "crs_52w": 60, "weinstein_stage": 140, "governance_bonus": 90, "promoter_holdings": 80, "change_promoter_1y":
    110, "pledged_percentage": 75, "fii_holdings": 55, "change_fii_lq": 80, "dii_holdings": 55, "change_dii_lq": 80,
    "smart_money_flow": 160, "result_when": 95}


def test_every_deep_scanner_view_is_mains():
    assert _app_dict("_DS_VIEWS") == _MAIN_DS_VIEWS, "RAM Rank shows only while sorted by it"
    assert {c: w for c, w in _app_dict("_DS_W").items() if c != "ram_rank"} == _MAIN_DS_W


def test_the_dropdown_offers_ram_rank_low_numbers_first():
    assert _app_dict("_DS_SORTS")[_SORT] == ("ram_rank", True)
    assert _app_dict("_num_fmt")["ram_rank"] == ("RAM Rank", "%.0f")
    assert _app_dict("_DS_W")["ram_rank"] >= 75, "its width entry: the header in full"


def test_a_view_on_an_old_frame_without_the_column_simply_leaves_it_out():
    view = _app_dict("_DS_VIEWS")["📈 Technical"]
    assert ds_view_columns(view, "ram_rank", set(view)) == view


_DS_HARNESS = textwrap.dedent("""
    import sys
    sys.path.insert(0, {root!r})
    import numpy as np
    import pandas as pd
    import streamlit as _st
    from config import COLORS
    from ui.ui_group_views import OWNERSHIP, OWNERSHIP_NOTE
    from ui.ui_scanner import _SCANNER_HEADER_TIPS, ds_view_columns


    class _Keeps:                # st, except that it keeps the bytes the export hands its button (AppTest cannot read them)
        def __getattr__(self, name):
            return getattr(_st, name)

        def download_button(self, label, data=None, **kw):
            _st.session_state["export"] = data
            return _st.download_button(label, data=data, **kw)


    st = _Keeps()
    clear_all_filters = lambda: None
    analysis_mode = "Hybrid"
    tabs = [None, None, _st.container()]
    df = _st.session_state["market"]
    filt = df
""")


def _run_deep_scanner(tmp_path, market: pd.DataFrame, view: str = "📈 Technical", sort: str = _SORT):
    """The Deep Scanner exactly as app.py has it, its width table and its tab block, on a small market (the way
    tests/test_scans_tab_runs.py runs the 🎯 Scans tab)."""
    from streamlit.testing.v1 import AppTest
    parts = [n for n in _TREE.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "_DS_W"]
    parts += [n for n in _TREE.body if isinstance(n, ast.With) and ast.unparse(n.items[0].context_expr) == "tabs[2]"]
    assert len(parts) == 2, "app.py's Deep Scanner: its width table and one block"
    script = tmp_path / "deep_scanner.py"
    script.write_text(_DS_HARNESS.format(root=str(ROOT)) + "\n".join(ast.unparse(p) for p in parts), encoding="utf-8")
    at = AppTest.from_file(str(script), default_timeout=60)
    at.session_state["market"] = market
    at.session_state["ds_view"] = view
    at.session_state["ds_sort"] = sort
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


@pytest.mark.parametrize("view", list(_app_dict("_DS_VIEWS")))
def test_the_sort_lists_low_numbers_first_with_blanks_last_and_the_column_on_screen_in_every_view(tmp_path, view):
    grid = _run_deep_scanner(tmp_path, _order_market(), view=view).dataframe[0]
    shown = grid.value
    assert list(shown.columns[:3]) == ["rank", "name", "ram_rank"], "right after Stock: the rule whatever is sorted is on screen"
    assert shown["ram_rank"].iloc[:5].tolist() == [1.0, 2.0, 2.0, 5.0, 9.0] and shown["ram_rank"].iloc[5:].isna().all()
    config = json.loads(grid.proto.columns)["ram_rank"]
    assert (config["label"], config["help"], config["width"]) == ("RAM Rank", TIP_RANK, 75)
    assert config["type_config"]["type"] == "number"


def test_without_the_sort_no_view_shows_it(tmp_path):
    assert "ram_rank" not in _run_deep_scanner(tmp_path, _order_market(), sort="Score ↓").dataframe[0].value.columns


def _export(at) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(at.session_state["export"]), encoding="utf-8-sig")


def test_the_exports_rows_follow_the_sort(tmp_path):
    """The column is in the file under any sort (main's rule: the export carries every sort column); its ROWS follow this one."""
    ram = _export(_run_deep_scanner(tmp_path, _order_market()))["ram_rank"]
    assert ram.iloc[:5].tolist() == [1.0, 2.0, 2.0, 5.0, 9.0] and ram.iloc[5:].isna().all()
    other = _export(_run_deep_scanner(tmp_path, _order_market(), sort="Score ↓"))["ram_rank"]
    assert other.iloc[0] == 5.0, "under another sort the same column stands in that sort's order"


def test_the_deep_scanners_grid_keeps_streamlits_own_blank(tmp_path):
    """Only the 🔢 RAM view draws a dash: here it would also change every P/E and PEG blank."""
    assert _run_deep_scanner(tmp_path, _order_market()).dataframe[0].proto.placeholder == ""


@pytest.mark.parametrize("case", ["tonight's blanks", "an old copy"])
def test_the_sort_on_a_list_with_no_ram_runs_and_shows_no_number(tmp_path, case):
    market = (_order_market(ram_rank=np.nan, ram_pct=np.nan, ram_score=np.nan, ram_date=np.nan) if case == "tonight's blanks"
              else _order_market().drop(columns=list(RAM_COLUMNS)))
    shown = _run_deep_scanner(tmp_path, market).dataframe[0].value
    assert "ram_rank" not in shown.columns or shown["ram_rank"].isna().all(), "blank, never 0"


# ── 6. The Tear-Sheet's RAM row, through the real render (AppTest; a synthetic row needs no data folder) ──────────────
_BLANK = {"ram_score": np.nan, "ram_rank": np.nan, "ram_pct": np.nan, "ram_date": np.nan}
_RANKED = {"ram_score": 2.5, "ram_rank": 4.0, "ram_pct": 99.84, "ram_date": "2026-11-04"}
_LINES = [
    ("ranked", _RANKED, "#4 · 99.8% · 4 Nov"),
    ("the_top_of_the_list", {**_RANKED, "ram_rank": 1.0, "ram_pct": 100.0}, "#1 · 100.0% · 4 Nov"),
    ("a_date_cell_timestamp", {**_RANKED, "ram_date": pd.Timestamp("2026-11-04")}, "#4 · 99.8% · 4 Nov"),
    ("a_date_cell_datetime", {**_RANKED, "ram_date": datetime(2026, 11, 4)}, "#4 · 99.8% · 4 Nov"),
    ("another_month", {**_RANKED, "ram_date": "2026-12-01"}, "#4 · 99.8% · 1 Dec"),
    ("a_zoned_date", {**_RANKED, "ram_date": "2026-11-04T01:00+05:30"}, "#4 · 99.8% · 4 Nov"),
    ("a_rank_without_a_date", {**_RANKED, "ram_date": np.nan}, "#4 · 99.8%"),
    ("a_day_first_date", {**_RANKED, "ram_date": "03/11/2026"}, "#4 · 99.8%"),
    ("a_year_for_a_date", {**_RANKED, "ram_date": "2026"}, "#4 · 99.8%"),
    ("a_year_and_month_for_a_date", {**_RANKED, "ram_date": "2026-11"}, "#4 · 99.8%"),
    ("a_serial_for_a_date", {**_RANKED, "ram_date": 46329.0}, "#4 · 99.8%"),
    ("an_impossible_date", {**_RANKED, "ram_date": "2026-13-45"}, "#4 · 99.8%"),
    ("not_ranked_on_a_malformed_date", {**_BLANK, "ram_date": "03/11/2026"}, "—"),
    ("a_rank_without_a_percentile", {**_RANKED, "ram_pct": np.nan}, "#4 · 4 Nov"),
    ("not_ranked", {**_BLANK, "ram_date": "2026-11-04"}, "— · not ranked on 4 Nov"),
    ("not_ranked_on_a_date_cell", {**_BLANK, "ram_date": datetime(2026, 11, 4)}, "— · not ranked on 4 Nov"),
    ("all_four_blank", _BLANK, "—"),
    ("blank_date_as_nat", {**_BLANK, "ram_date": pd.NaT}, "—"),
    ("blank_date_as_none", {**_BLANK, "ram_date": None}, "—"),
    ("an_old_copy_without_the_columns", {}, "—"),
]


def _scorecard_blocks(stock: pd.Series) -> list:
    from streamlit.testing.v1 import AppTest

    def _app():
        import streamlit as st
        from ui.ui_tearsheet import render_verdict_scorecard
        render_verdict_scorecard(st.session_state["stock_row"])

    at = AppTest.from_function(_app)
    at.session_state["stock_row"] = stock
    at.run(timeout=90)
    assert not at.exception, [str(e.value) for e in at.exception]
    return [str(md.value) for md in at.markdown]


def _visible(block: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", block)).split())


def _ram_blocks(blocks: list) -> list:
    return [b for b in blocks if _visible(b).startswith("RAM RANK")]


@pytest.mark.parametrize("cells,want", [c[1:] for c in _LINES], ids=[c[0] for c in _LINES])
def test_the_tear_sheet_reads_its_row_in_each_form(cells, want):
    found = _ram_blocks(_scorecard_blocks(pd.Series({"rs_score": 85.0, **cells})))
    assert len(found) == 1, "one row"
    assert _visible(found[0]) == f"RAM RANK ? {want}"


@pytest.mark.parametrize("cells", [_RANKED, _BLANK, {}], ids=["ranked", "blank", "absent"])
def test_the_whole_scorecard_leaks_no_nan(cells):
    blocks = _scorecard_blocks(pd.Series(cells))
    text = " ".join(blocks).lower()
    assert not re.findall(r"[+\-₹ >#]nan[%<· ]", text), "no nan, no nan%, no #nan"
    assert "nat" not in _visible(_ram_blocks(blocks)[0]).lower().split()


def test_it_is_a_row_of_its_own_in_the_strips_look_not_a_chip_on_the_entry_timing_strip():
    """That strip's admission is closed by a live rule (tests/test_deep_signal_chips.py)."""
    blocks = _scorecard_blocks(pd.Series(_RANKED))
    timing, row = next(b for b in blocks if "ENTRY TIMING" in b), _ram_blocks(blocks)[0]
    assert "RAM" not in timing and "#4" not in timing
    assert blocks.index(row) == blocks.index(timing) + 1, "right under the strip"
    label = 'font-size:0.6rem;font-weight:800;color:' + COLORS["text_muted"] + ';letter-spacing:0.5px;">'
    assert label + "RAM RANK<" in row and label + "⏱️ ENTRY TIMING<" in timing, "the strip's own small label"
    chip = "font-size:0.62rem;font-weight:700;padding:2px 8px;border-radius:10px;"
    assert chip in row and chip in timing, "one pill in the strip's own chip style"
    assert f'color:{COLORS["text_secondary"]};white-space:nowrap;' in row, "neutral: nothing here grades the stock"


def test_the_row_carries_the_one_tip_on_its_question_mark():
    block = _ram_blocks(_scorecard_blocks(pd.Series(_RANKED)))[0]
    assert html.escape(TIP_RANK) in block, "the same words as the table"


# ── 7. Nothing scores on them ────────────────────────────────────────────────────────────────────────────────────────────
# The columns the comparison below must never be vacuous about: a score, a tier, a rank, a verdict and a flag of each kind.
_MUST_COMPARE = {"composite_score", "quality_score", "momentum_score", "breakout_score", "improvement_score", "rank",
                 "conviction_tier", "tier_label", "verdict_direction", "verdict_strength", "wealth_tier", "gate_pass",
                 "forensic_score", "forensic_multiplier", "red_flag_count", "frameworks_passed", "rf_snoa"}


def _ram_filled(ids) -> dict:
    """Values with no relation to any score, and a real date: a reader of any of them would move something."""
    n = len(ids)
    rank = np.arange(n, 0, -1, dtype=float)
    rank[::7] = np.nan                                  # not ranked, as a seventh of them would be
    return {"ram_score": rank / 1000.0, "ram_rank": rank, "ram_pct": 100.0 * rank / n,
            "ram_date": np.where(np.isnan(rank), None, "2026-11-03")}


def _synthetic_market(n: int = 300) -> pd.DataFrame:
    """Every mapped column filled from a seeded generator (no data folder needed: this half also runs in CI)."""
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({c: rng.uniform(-20.0, 120.0, n) for c in _MAPPED if c not in RAM_COLUMNS})
    frame["company_id"] = [f"NSE:S{i}" for i in range(n)]
    frame["name"] = [f"S{i} Ltd" for i in range(n)]
    industries = sorted(INDUSTRY_TIER)[:12]
    frame["industry"] = [industries[i % len(industries)] for i in range(n)]
    frame["sector"] = [f"Sector {i % 6}" for i in range(n)]
    frame["market_category"] = [("Large Cap", "Mid Cap", "Small Cap", "Micro Cap")[i % 4] for i in range(n)]
    frame["market_cap"] = rng.uniform(100.0, 90000.0, n)
    frame["close_price"] = rng.uniform(5.0, 3000.0, n)
    return frame


def _scored_from_merged(merged: pd.DataFrame) -> pd.DataFrame:
    with contextlib.redirect_stdout(io.StringIO()):
        return run_scoring_pipeline(de.compute_derived_signals(de.coerce_numeric_columns(merged.copy())))


def _assert_same_but_ram(a: pd.DataFrame, b: pd.DataFrame):
    cols = [c for c in a.columns if c not in RAM_COLUMNS]
    assert _MUST_COMPARE <= set(cols), sorted(_MUST_COMPARE - set(cols))
    assert sorted(cols) == sorted(c for c in b.columns if c not in RAM_COLUMNS), "RAM added or lost a column downstream"
    pd.testing.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True), check_exact=True)


@pytest.fixture(scope="module")
def synthetic():
    merged = _synthetic_market()
    return {"absent": _scored_from_merged(merged),
            "filled": _scored_from_merged(merged.assign(**_ram_filled(merged["company_id"]))),
            "blank": _scored_from_merged(merged.assign(**{c: np.nan for c in RAM_COLUMNS}))}


@pytest.mark.parametrize("case", ["filled", "blank"])
def test_ram_moves_no_score_tier_rank_verdict_or_flag(synthetic, case):
    _assert_same_but_ram(synthetic["absent"], synthetic[case])


def test_the_values_reach_the_scored_frame_untouched(synthetic):
    """So the identity above is no accident of the columns having been dropped."""
    filled, absent = synthetic["filled"], synthetic["absent"]
    want = _ram_filled(filled["company_id"])["ram_rank"]
    assert filled["ram_rank"].notna().sum() == np.isfinite(want).sum() > 0
    assert (filled["ram_date"].dropna() == "2026-11-03").all()
    assert absent["ram_rank"].isna().all(), "the old copy reads blank through the whole pipeline"


@pytest.fixture(scope="module")
def real():
    """The same comparison on the newest local copy through the real loader, RAM added to its raw Technicals tab."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            datasets = de.load_all_csvs("local")
    except Exception as exc:                                     # the data is gitignored
        pytest.skip(f"local data not available: {exc}")
    with contextlib.redirect_stdout(io.StringIO()):
        plain = run_scoring_pipeline(de._clean_from({k: v.copy() for k, v in datasets.items()}))
        tech = datasets["technical"]
        datasets["technical"] = tech.assign(**_ram_filled(tech["company_id"]))
        with_ram = run_scoring_pipeline(de._clean_from(datasets))
    return plain, with_ram


def test_ram_moves_nothing_on_the_real_market(real):
    _assert_same_but_ram(*real)


@pytest.mark.parametrize("key", ["playbook", "no_quarter", "price_only", "leaders", "compounder"])
def test_every_list_holds_the_same_stocks_in_the_same_order_on_the_real_market(real, key):
    plain, with_ram = real
    scan = next(s for s in LISTS if s["key"] == key)
    a, b = S.scan_table(plain, scan), S.scan_table(with_ram, scan)
    assert len(a) > 0 and a["company_id"].tolist() == b["company_id"].tolist()


def test_the_order_reading_runs_on_the_real_market_with_ram_and_without(real):
    """Real labels through the real pipeline: the filters leave some of 🚀's matches, RAM or no RAM. The return is a fixture."""
    from ui import scan_records_job as J
    _, with_ram = real
    match = _PLAY["mask"](with_ram)
    ret = pd.Series(np.arange(len(with_ram), dtype=float), index=with_ram.index)
    a, b = J.order_reading(with_ram, ret, match), J.order_reading(with_ram.drop(columns="ram_rank"), ret, match)
    assert 0 < a["n"] == b["n"] < int(match.sum()), "the filters are alive: not none of 🚀's matches, and not all of them"
    assert b.get("ram_pct") is None


# The fingerprints every record and the ladder were measured with, read on origin/main (9284e96) before RAM. 🚀, 🌬️ and 🌱
# are also pinned to their records (tests/test_scans.py); B and C have none. A recipe's real change re-pins these on purpose.
_BEFORE_RAM = {"playbook": "80251b8b4da6d635", "leaders": "a5fa9c66bfd431e0", "compounder": "0e2a24062b1f2ea8",
               "no_quarter": "1f17120242b5640f", "price_only": "1f17120242b5640f"}


@pytest.mark.parametrize("scan", LISTS, ids=lambda s: s["key"])
def test_no_recipes_fingerprint_moved(scan):
    assert S.recipe_fingerprint(scan) == _BEFORE_RAM[scan["key"]]


_RAM_NAME = re.compile(r"\bram_(score|rank|pct|date)\b")


@pytest.mark.parametrize("scan", LISTS, ids=lambda s: s["key"])
def test_nothing_a_recipe_reaches_names_a_ram_column(scan):
    """The lasting half of the pin above: it survives a real recipe change and still fails on a recipe that reads RAM."""
    reached = S._dependencies([*S._recipe_functions(scan), *scan.get("inputs", ())], scan.get("values"), scan.get("skip", ()))
    named = sorted(k for k, v in reached.items() if _RAM_NAME.search(f"{k} {v}"))
    assert not named, named


# Who may name a RAM column at all: the loader, the Data check's text list, the three screens and, since the order check
# moved to RAM Rank, the records job (2026-11-05). Not the scoring engines, the verdict, the Shariah screen, the feed, the
# v2 lists or the archive.
_MAY_NAME_RAM = {"app.py", "core/data_engine.py", "ui/data_check_job.py", "ui/scan_records_job.py", "ui/ui_scanner.py",
                 "ui/ui_scans.py", "ui/ui_tearsheet.py"}


def test_only_the_loader_the_data_check_the_screens_and_the_order_check_name_a_ram_column():
    files = [ROOT / "app.py", ROOT / "config.py", *sorted((ROOT / "core").glob("*.py")), *sorted((ROOT / "ui").glob("*.py"))]
    naming = {p.relative_to(ROOT).as_posix() for p in files if _RAM_NAME.search(p.read_text(encoding="utf-8"))}
    assert naming == _MAY_NAME_RAM, f"unexpected: {sorted(naming - _MAY_NAME_RAM)}; gone: {sorted(_MAY_NAME_RAM - naming)}"


# ── 8. The words ─────────────────────────────────────────────────────────────────────────────────────────────────────
_ACTION_WORDS = re.compile(r"\b(buy|buys|buying|sell|sells|selling|hold|avoid|kite|operator)\b", re.I)


def test_no_word_of_action_and_no_rule_or_person_anywhere_on_screen():
    from ui.ui_tearsheet import ram_line
    cfg = S.table_config("03 Oct", _PLAY)
    words = [_RAW_GLOSSARY["RAM Rank"], _RAW_GLOSSARY["RAM %"], _RAW_GLOSSARY["RAM"], S.RAM_VIEW,
             S.ram_view_order_words(pd.DataFrame({"ram_rank": [1.0], "ram_date": ["2026-11-04"]}), "x", _order_market()),
             S.ram_view_order_words(pd.DataFrame({"ram_rank": [np.nan], "ram_date": [np.nan]}), "x", _order_market()),
             *(cfg[c][k] for c in ("#", "ram_rank", "symbol", "ram_pct", "ram_score") for k in ("label", "help")),
             *(ram_line(pd.Series(line[1])) for line in _LINES)]
    assert all(words)
    for text in words:
        assert not _ACTION_WORDS.search(text), text


def test_the_guides_stamps_are_mains():
    """A view explains nothing in a guide's stamp, and RAM moved no scan setting, so no prose went stale. 🌱's stamp moved once
    since, on purpose: its record date and its recipe's words, re-read and re-stamped on 2026-11-05."""
    from ui import scan_guide as G
    assert {s["key"]: G.guide_stamp(s) for s in S.SCANS} == {
        "playbook": "135cfc40d93df721", "leaders": "c5b0c3512a06fdd2", "compounder": "d3e637eb0be7a2e1"}


def test_the_guide_says_how_every_list_opens_and_what_the_order_check_reads():
    from ui import scan_guide as G
    t = " ".join(map(str, dict(G.general_sections())["How to read the table"]))
    for words in ("every list opens", "🔢 RAM", "RAM Rank", "🏆 Core and 📈 Technical", "as before", "order check", "what Kite can trade"):
        assert words in t, words
    assert "🚀 lists" not in t and "Momentum's top" not in t, "the order check moved to RAM Rank (2026-11-05)"
    for scan in S.SCANS:                # 🌱 too, whose order section reads PRISM Rank, not Momentum
        assert "The table opens on the 🔢 RAM view" in " ".join(G._order(scan)), scan["key"]


def _source_files() -> list:
    return [ROOT / "app.py", ROOT / "config.py", *sorted((ROOT / "core").glob("*.py")), *sorted((ROOT / "ui").glob("*.py"))]


def test_nothing_else_reads_the_display_order():
    """The view's order is made by one function the tab alone calls: the records job, the feed, the v2 lists and the order
    check read their own frames (scan_records_job.order_reading sorts RAM Rank and Rank itself), and no job calls scan_table."""
    readers = {name: {p.relative_to(ROOT).as_posix() for p in _source_files() if re.search(rf"\b{name}\b", p.read_text(encoding="utf-8"))}
               for name in ("ram_view_table", "scan_table", "TABLE_VIEWS")}
    assert readers == {"ram_view_table": {"app.py", "ui/ui_scans.py"}, "scan_table": {"app.py", "ui/ui_scans.py"},
                       "TABLE_VIEWS": {"app.py", "ui/ui_scans.py"}}


def test_the_view_moves_no_record_or_recipe():
    """The order is the view's: the scan's `order`, which the card and guide speak of, is still main's Momentum."""
    assert S.momentum_ordered(_PLAY) and S.momentum_ordered(_B) and S.momentum_ordered(_C) and S.momentum_ordered(_LEAD)
    assert sorted(S.ORDER_RULE["scans"]) == ["leaders", "playbook"]
    for scan in LISTS:
        assert S.recipe_fingerprint(S.ram_view_scan(scan)) == _BEFORE_RAM[scan["key"]], scan["key"]
