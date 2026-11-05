"""
test_ram_display.py — RAM on PRISM's screens (2026-11-05).

The Technicals tab now carries four columns another job fills each night: RAM, RAM Rank, RAM Pct and RAM Date. PRISM
shows them and computes nothing from them. Pinned here, each against the break it exists to catch:

  1. The loader: each header lands on its own column, a blank stays blank (never 0), and RAM Date survives whatever the
     XLSX hands it (text, a real date cell, a blank) where the numeric coercion would have blanked it. An old copy
     without the headers loads, and the engine's schema guard gives every reader the four columns as blanks.
  2. The daily Data check: a date in a "number column" would have been a false red every night.
  3. The tips: one source (the glossary), read by the Tear-Sheet's ?, the Reference, the Deep Scanner and the 🎯 Scans.
  4. The 🎯 Scans: RAM Rank right after Momentum in both views, RAM % after it in 📈 Technical; no row and no order moves.
  5. The Deep Scanner: both columns in its 📈 Technical view, one home, numbers with a set width, the export following.
  6. The Tear-Sheet: one line in each of its three forms under the ⏱️ strip, never a chip on it, and no nan.
  7. Nothing reads them: every score, tier, rank, verdict and flag identical with and without RAM values; every recipe's
     fingerprint unmoved; only the loader, the Data check and the three screens name the columns at all.
  8. No word of action, no rule and no person in their labels and tips.

The widths they cost are pinned beside the others (tests/test_scan_views.py, tests/test_deep_scanner_views.py).

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

from config import SHEET_TAB_NAMES  # noqa: E402
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
               ("timestamp", pd.Timestamp("2026-11-03"), _ISO), ("date", date(2026, 11, 3), _ISO)]
_NOT_A_SESSION = [("nan", np.nan), ("nat", pd.NaT), ("none", None), ("empty", ""),
                  ("words", "not a date"),
                  ("impossible", "2026-13-45"),   # never crash on a stranger's cell (CLAUDE.md §5)
                  ("either_month", "03/11/2026"),  # 3 Nov or 11 Mar: never guessed
                  ("serial", 46329.0)]             # to_datetime would call it nanoseconds from 1970: "1 Jan"


@pytest.mark.parametrize("cell,want", [c[1:] for c in _DATE_CELLS], ids=[c[0] for c in _DATE_CELLS])
def test_a_ram_date_survives_the_numeric_coercion_in_every_form(cell, want):
    out = _coerced(ram_date=[cell, cell])["ram_date"]
    assert out.tolist() == [want, want]


@pytest.mark.parametrize("cell", [c[1] for c in _NOT_A_SESSION], ids=[c[0] for c in _NOT_A_SESSION])
def test_what_is_not_a_session_reads_blank(cell):
    out = _coerced(ram_date=["2026-11-03", cell])["ram_date"]
    assert out.iloc[0] == _ISO, "a good cell beside it survives"
    assert pd.isna(out.iloc[1]), f"{cell!r} read as {out.iloc[1]!r}"


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
            "check doubted it). Display only: it changes no score, list or record.")
TIP_PCT = "RAM's score as a percentile, 100 the strongest. Display only: it changes no score, list or record."
CORE, TECH = S.SCAN_VIEWS
_PLAY = next(s for s in S.SCANS if s["key"] == "playbook")
LISTS = (*S.SCANS, S.playbook_layer_scan(_PLAY, "no_quarter"), S.playbook_layer_scan(_PLAY, "price_only"))
_LIST_IDS = [s["key"] for s in LISTS]


def test_both_tips_have_one_source_read_by_every_surface():
    assert _RAW_GLOSSARY["RAM Rank"] == TIP_RANK and _RAW_GLOSSARY["RAM %"] == TIP_PCT, "the tear-sheet's ? and the Reference"
    assert _SCANNER_HEADER_TIPS["ram_rank"] == TIP_RANK and _SCANNER_HEADER_TIPS["ram_pct"] == TIP_PCT, "the Deep Scanner"
    cfg = S.table_config("03 Oct", _PLAY)
    assert cfg["ram_rank"]["help"] == TIP_RANK and cfg["ram_pct"]["help"] == TIP_PCT, "the 🎯 Scans table"


# ── 4. The 🎯 Scans tab ──────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("scan", LISTS, ids=_LIST_IDS)
def test_ram_rank_sits_right_after_momentum_in_both_views(scan):
    for view in S.SCAN_VIEWS:
        cols = S.table_columns(scan, view)
        assert cols[cols.index("momentum_score") + 1] == "ram_rank", view


@pytest.mark.parametrize("scan", LISTS, ids=_LIST_IDS)
def test_ram_pct_follows_it_in_technical_only(scan):
    tech = S.table_columns(scan, TECH)
    assert tech[tech.index("ram_rank") + 1] == "ram_pct"
    assert "ram_pct" not in S.table_columns(scan, CORE), "Core's width budget is full: one column there"


def test_the_columns_are_labelled_and_read_as_sortable_numbers():
    cfg = S.table_config("03 Oct", _PLAY)
    assert (cfg["ram_rank"]["label"], cfg["ram_pct"]["label"]) == ("RAM Rank", "RAM %")
    rank, pct = cfg["ram_rank"]["type_config"], cfg["ram_pct"]["type_config"]
    assert (rank["type"], rank["format"]) == ("number", "%.0f"), "a whole rank, sorted as a number"
    assert (pct["type"], pct["format"]) == ("number", "%.1f"), "the % is in the header"


def test_mom_quality_no_longer_claims_both_its_numbers_sit_beside_it():
    """RAM's two columns now stand between Momentum and Mom. Quality on the Scans table, so the one tip it shares with the
    Deep Scanner says which numbers it reads and no longer where they sit."""
    tip = S.table_config("03 Oct", _PLAY)["d49_momentum_quality"]["help"]
    assert "Reads the Momentum and RSI columns" in tip and "beside it" not in tip


def _market() -> pd.DataFrame:
    """Six stocks whose RAM order is the reverse of their momentum and rank, so a sort that read RAM would show."""
    return pd.DataFrame({"company_id": [f"NSE:S{i}" for i in range(6)], "name": [f"S{i} Ltd" for i in range(6)],
                         "rank": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "momentum_score": [90.0, 80.0, 70.0, 60.0, 50.0, 40.0],
                         "ram_score": [0.1, 0.2, 0.3, 0.4, 0.5, np.nan],
                         "ram_rank": [500.0, 400.0, 300.0, 200.0, 100.0, np.nan],
                         "ram_pct": [10.0, 20.0, 30.0, 40.0, 50.5, np.nan],
                         "ram_date": ["2026-11-03"] * 5 + [np.nan]})


_EVERY_STOCK = {"key": "all", "mask": lambda d: pd.Series(True, index=d.index)}


def test_the_table_carries_the_values_and_a_blank_stays_blank():
    t = S.scan_table(_market(), _EVERY_STOCK).set_index("name")
    assert t.loc["S4 Ltd", "ram_rank"] == 100.0 and t.loc["S4 Ltd", "ram_pct"] == 50.5
    assert pd.isna(t.loc["S5 Ltd", "ram_rank"]) and pd.isna(t.loc["S5 Ltd", "ram_pct"]), "blank, never 0"


@pytest.mark.parametrize("scan", S.SCANS, ids=lambda s: s["key"])
def test_an_old_copy_without_the_columns_shows_blank_columns_never_zero(scan):
    """Every view's column must exist, or the grid's column_order drops it; and a missing column shows no number."""
    old = _market().drop(columns=list(RAM_COLUMNS))
    t = S.scan_table(old, dict(scan, mask=_EVERY_STOCK["mask"]))
    assert t["ram_rank"].isna().all() and t["ram_pct"].isna().all()
    assert t["ram_rank"].dtype == float, "a numeric column, so the header still sorts"


@pytest.mark.parametrize("key", ["all", "momentum"])
def test_ram_moves_no_row_and_no_order(key):
    """The list's rows and default order are the recipe's own, with or without RAM values."""
    scan = dict(_EVERY_STOCK, order="momentum") if key == "momentum" else _EVERY_STOCK
    filled, old = _market(), _market().drop(columns=list(RAM_COLUMNS))
    assert S.scan_table(filled, scan)["name"].tolist() == S.scan_table(old, scan)["name"].tolist()
    assert S.scan_table(filled, scan)["name"].tolist() == [f"S{i} Ltd" for i in range(6)], "by Rank or Momentum, not RAM"


def _render_table(table: pd.DataFrame, scan: dict, view: str):
    """The Scans table through Streamlit's own harness, with the app's own column config and order."""
    from streamlit.testing.v1 import AppTest

    def _app():
        import streamlit as st
        st.dataframe(st.session_state["table"], column_config=st.session_state["config"],
                     column_order=st.session_state["order"], hide_index=True)

    at = AppTest.from_function(_app)
    at.session_state["table"] = table
    at.session_state["config"] = S.table_config("03 Oct", scan)
    at.session_state["order"] = list(S.table_columns(scan, view))
    at.run(timeout=30)
    return at


@pytest.mark.parametrize("view", [CORE, TECH])
@pytest.mark.parametrize("case", ["filled", "tonight's blanks", "an old copy"])
def test_both_views_render_in_every_case(case, view):
    market = _market()
    if case == "tonight's blanks":
        market = market.assign(ram_score=np.nan, ram_rank=np.nan, ram_pct=np.nan, ram_date=np.nan)
    elif case == "an old copy":
        market = market.drop(columns=list(RAM_COLUMNS))
    table = S.scan_table(market, _EVERY_STOCK)
    at = _render_table(table, _PLAY, view)
    assert not at.exception, [str(e.value) for e in at.exception]
    shown = at.dataframe[0].value
    assert "ram_rank" in shown.columns
    want = [500.0, 400.0] if case == "filled" else [None, None]
    got = shown["ram_rank"].iloc[:2].tolist()
    assert (got == want) if case == "filled" else pd.isna(got).all(), got


# ── 5. The Deep Scanner: app.py's own tables, and its own tab block run under AppTest ─────────────────────────────────
_TREE = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))


def _app_dict(name: str) -> dict:
    node = next(n.value for n in ast.walk(_TREE) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return ast.literal_eval(node)


def test_the_deep_scanners_technical_view_shows_both_columns_between_momentum_and_breakout():
    """After the momentum group and before the breakout group: each verdict keeps the number it reads beside it."""
    view = _app_dict("_DS_VIEWS")["📈 Technical"]
    i = view.index("ram_rank")
    assert view[i - 3:i + 3] == ["momentum_score", "d49_momentum_quality", "rsi_14d", "ram_rank", "ram_pct", "breakout_score"]


def test_they_appear_in_no_other_view():
    homes = {v for v, cols in _app_dict("_DS_VIEWS").items() if "ram_rank" in cols or "ram_pct" in cols}
    assert homes == {"📈 Technical"}, "one home per column (tests/test_deep_scanner_views.py)"


def test_they_read_as_sortable_numbers_with_a_set_width():
    fmt = _app_dict("_num_fmt")
    assert fmt["ram_rank"] == ("RAM Rank", "%.0f") and fmt["ram_pct"] == ("RAM %", "%.1f")
    width = _app_dict("_DS_W")
    assert width["ram_rank"] >= 75 and width["ram_pct"] >= 57, "the header in full: RAM Rank needs 75, RAM % 57"


def test_a_view_on_an_old_frame_without_the_columns_simply_leaves_them_out():
    view = _app_dict("_DS_VIEWS")["📈 Technical"]
    available = set(view) - {"ram_rank", "ram_pct"}
    assert ds_view_columns(view, "composite_score", available) == [c for c in view if c in available]


_DS_HARNESS = textwrap.dedent('''
    import sys
    sys.path.insert(0, {root!r})
    import numpy as np
    import pandas as pd
    import streamlit as st
    from config import COLORS
    from ui.ui_group_views import OWNERSHIP, OWNERSHIP_NOTE
    from ui.ui_scanner import _SCANNER_HEADER_TIPS, ds_view_columns

    clear_all_filters = lambda: None
    analysis_mode = "Hybrid"
    tabs = [None, None, st.container()]
    df = st.session_state["market"]
    filt = df
''')


def _run_deep_scanner(tmp_path, market: pd.DataFrame, view: str = "📈 Technical"):
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
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _export_columns(at) -> int:
    label = at.get("download_button")[0].proto.label
    return int(re.search(r"(\d+) columns", label).group(1))


def test_the_real_tab_shows_both_columns_with_their_header_tip_and_width(tmp_path):
    at = _run_deep_scanner(tmp_path, _market())
    shown = at.dataframe[0].value
    assert shown["ram_rank"].iloc[:2].tolist() == [500.0, 400.0] and shown["ram_pct"].iloc[4] == 50.5
    config = json.loads(at.dataframe[0].proto.columns)
    assert config["ram_rank"]["label"] == "RAM Rank" and config["ram_pct"]["label"] == "RAM %"
    assert config["ram_rank"]["help"] == TIP_RANK and config["ram_pct"]["help"] == TIP_PCT
    assert config["ram_rank"]["width"] == 75 and config["ram_pct"]["width"] == 57
    assert config["ram_rank"]["type_config"]["type"] == "number", "a number column: its header sorts"


def test_no_other_view_of_the_real_tab_shows_them(tmp_path):
    shown = _run_deep_scanner(tmp_path, _market(), view="🏆 Core").dataframe[0].value
    assert "ram_rank" not in shown.columns and "ram_pct" not in shown.columns


def test_the_real_tabs_export_follows_the_view(tmp_path):
    """Its columns are the views', so the two go along where the frame holds them, and the file invents no column."""
    with_ram = _export_columns(_run_deep_scanner(tmp_path, _market()))
    without = _export_columns(_run_deep_scanner(tmp_path, _market().drop(columns=list(RAM_COLUMNS))))
    assert with_ram == without + 2


def test_tonights_blanks_show_blank_on_the_real_tab(tmp_path):
    market = _market().assign(ram_score=np.nan, ram_rank=np.nan, ram_pct=np.nan, ram_date=np.nan)
    shown = _run_deep_scanner(tmp_path, market).dataframe[0].value
    assert shown["ram_rank"].isna().all() and shown["ram_pct"].isna().all(), "blank, never 0"


def test_an_old_copy_runs_the_real_tab_without_the_columns(tmp_path):
    shown = _run_deep_scanner(tmp_path, _market().drop(columns=list(RAM_COLUMNS))).dataframe[0].value
    assert "ram_rank" not in shown.columns and shown["momentum_score"].notna().all()


# ── 6. The Tear-Sheet's RAM line, through the real render (AppTest; a synthetic row needs no data folder) ─────────────
_BLANK = {"ram_score": np.nan, "ram_rank": np.nan, "ram_pct": np.nan, "ram_date": np.nan}
_RANKED = {"ram_score": 0.5, "ram_rank": 312.0, "ram_pct": 78.44, "ram_date": "2026-11-03"}
_LINES = [
    ("ranked", _RANKED, "RAM 78.4% · #312 · 3 Nov"),
    ("the_top_of_the_list", {**_RANKED, "ram_rank": 1.0, "ram_pct": 100.0}, "RAM 100.0% · #1 · 3 Nov"),
    ("a_date_cell_timestamp", {**_RANKED, "ram_date": pd.Timestamp("2026-11-03")}, "RAM 78.4% · #312 · 3 Nov"),
    ("a_date_cell_datetime", {**_RANKED, "ram_date": datetime(2026, 11, 3)}, "RAM 78.4% · #312 · 3 Nov"),
    ("another_month", {**_RANKED, "ram_date": "2026-12-01"}, "RAM 78.4% · #312 · 1 Dec"),
    ("a_rank_without_a_date", {**_RANKED, "ram_date": np.nan}, "RAM 78.4% · #312"),
    ("a_rank_without_a_percentile", {**_RANKED, "ram_pct": np.nan}, "RAM — · #312 · 3 Nov"),
    ("not_ranked", {**_BLANK, "ram_date": "2026-11-03"}, "RAM — · not ranked on 3 Nov"),
    ("not_ranked_on_a_date_cell", {**_BLANK, "ram_date": datetime(2026, 11, 3)}, "RAM — · not ranked on 3 Nov"),
    ("all_four_blank", _BLANK, "RAM —"),
    ("blank_date_as_nat", {**_BLANK, "ram_date": pd.NaT}, "RAM —"),
    ("blank_date_as_none", {**_BLANK, "ram_date": None}, "RAM —"),
    ("an_old_copy_without_the_columns", {}, "RAM —"),
]


def _scorecard_blocks(stock: pd.Series) -> list:
    from streamlit.testing.v1 import AppTest

    def _app():
        import streamlit as st
        from ui.ui_tearsheet import render_verdict_scorecard
        render_verdict_scorecard(st.session_state["stock_row"])

    at = AppTest.from_function(_app)
    at.session_state["stock_row"] = stock
    at.run(timeout=30)
    assert not at.exception, [str(e.value) for e in at.exception]
    return [str(md.value) for md in at.markdown]


def _visible(block: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", block)).split())


def _ram_blocks(blocks: list) -> list:
    return [b for b in blocks if _visible(b).startswith("RAM ")]


@pytest.mark.parametrize("cells,want", [c[1:] for c in _LINES], ids=[c[0] for c in _LINES])
def test_the_tear_sheet_reads_one_line_in_each_form(cells, want):
    found = _ram_blocks(_scorecard_blocks(pd.Series({"rs_score": 85.0, **cells})))
    assert len(found) == 1, "one line"
    assert _visible(found[0]).removesuffix("?").strip() == want


@pytest.mark.parametrize("cells", [_RANKED, _BLANK, {}], ids=["ranked", "blank", "absent"])
def test_the_whole_scorecard_leaks_no_nan(cells):
    text = " ".join(_scorecard_blocks(pd.Series(cells))).lower()
    assert not re.findall(r"[+\-₹ >#]nan[%<· ]", text), "no nan, no nan%, no #nan"
    assert "nat" not in _visible(" ".join(_ram_blocks(_scorecard_blocks(pd.Series(cells))))).lower().split()


def test_it_is_a_line_of_its_own_not_a_chip_on_the_entry_timing_strip():
    """That strip's admission is closed by a live rule (tests/test_deep_signal_chips.py)."""
    blocks = _scorecard_blocks(pd.Series(_RANKED))
    timing = next(b for b in blocks if "ENTRY TIMING" in b)
    assert "RAM" not in timing and "#312" not in timing
    assert blocks.index(_ram_blocks(blocks)[0]) == blocks.index(timing) + 1, "right under the strip, in the momentum area"


def test_the_line_carries_the_one_tip_on_its_question_mark():
    block = _ram_blocks(_scorecard_blocks(pd.Series(_RANKED)))[0]
    assert html.escape(TIP_RANK) in block, "the same words as the two tables"


# ── 7. Nothing reads them ────────────────────────────────────────────────────────────────────────────────────────────
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


# The fingerprints every record and the ladder were measured with, read on origin/main (9284e96) before RAM. 🚀, 🌬️ and 🌱
# are also pinned to their records (tests/test_scans.py); B and C have none. A recipe's real change re-pins these on purpose.
_BEFORE_RAM = {"playbook": "80251b8b4da6d635", "leaders": "a5fa9c66bfd431e0", "compounder": "0e2a24062b1f2ea8",
               "no_quarter": "1f17120242b5640f", "price_only": "1f17120242b5640f"}


@pytest.mark.parametrize("scan", LISTS, ids=_LIST_IDS)
def test_no_recipes_fingerprint_moved(scan):
    assert S.recipe_fingerprint(scan) == _BEFORE_RAM[scan["key"]]


_RAM_NAME = re.compile(r"\bram_(score|rank|pct|date)\b")


@pytest.mark.parametrize("scan", LISTS, ids=_LIST_IDS)
def test_nothing_a_recipe_reaches_names_a_ram_column(scan):
    """The lasting half of the pin above: it survives a real recipe change and still fails on a recipe that reads RAM."""
    reached = S._dependencies([*S._recipe_functions(scan), *scan.get("inputs", ())], scan.get("values"), scan.get("skip", ()))
    named = sorted(k for k, v in reached.items() if _RAM_NAME.search(f"{k} {v}"))
    assert not named, named


# Who may name a RAM column at all: the loader, the Data check's text list and the three screens. Not the scoring engines,
# the verdict, the Shariah screen, the feed, the v2 lists, the scan records or the archive.
_MAY_NAME_RAM = {"app.py", "core/data_engine.py", "ui/data_check_job.py", "ui/ui_scanner.py", "ui/ui_scans.py",
                 "ui/ui_tearsheet.py"}


def test_only_the_loader_the_data_check_and_the_screens_name_a_ram_column():
    files = [ROOT / "app.py", ROOT / "config.py", *sorted((ROOT / "core").glob("*.py")), *sorted((ROOT / "ui").glob("*.py"))]
    naming = {p.relative_to(ROOT).as_posix() for p in files if _RAM_NAME.search(p.read_text(encoding="utf-8"))}
    assert naming == _MAY_NAME_RAM, f"unexpected: {sorted(naming - _MAY_NAME_RAM)}; gone: {sorted(_MAY_NAME_RAM - naming)}"


# ── 8. The words ─────────────────────────────────────────────────────────────────────────────────────────────────────
_ACTION_WORDS = re.compile(r"\b(buy|buys|buying|sell|sells|selling|hold|avoid|kite|operator)\b", re.I)


def test_no_word_of_action_and_no_rule_or_person_anywhere_on_screen():
    from ui.ui_tearsheet import ram_line
    cfg = S.table_config("03 Oct", _PLAY)
    words = [_RAW_GLOSSARY["RAM Rank"], _RAW_GLOSSARY["RAM %"], *(cfg[c][k] for c in ("ram_rank", "ram_pct")
                                                                  for k in ("label", "help")),
             *(ram_line(pd.Series(line[1])) for line in _LINES)]
    assert all(words)
    for text in words:
        assert not _ACTION_WORDS.search(text), text


def test_the_guides_stamps_are_untouched():
    """A column explains nothing in a guide's stamp (test_a_guide_never_outlives_its_scan), and no scan setting moved."""
    from ui import scan_guide as G
    for scan in S.SCANS:
        assert G.GUIDES[scan["key"]]["written_for"] == G.guide_stamp(scan), scan["key"]
