"""
test_market_box.py
==================
The market in the sidebar, and the regime kept out of the scores (2026-10-08).

WHY. The detected regime votes on the share of stocks BEATING the Nifty 500 — near 50% by
construction. On 2026-10-08 it read SIDEWAYS, half a point from BULL ("Offence Mode"), while the Nifty
500 itself was in a Weinstein Stage-4 decline and the small-cap indices in Stage 2. So:
  * config.REGIME_DRIVES_SCORING is off: scoring uses SIDEWAYS (tests/test_regime_detection.py pins
    that a forced BULL/BEAR then moves no score, and that the machinery still works switched on);
  * the sidebar box leads with the benchmark indices' own stages — the SAME rule every stock is staged
    with (data_engine.weinstein_stage) — and labels the breadth regime "shown, not applied".
Pinned here: the one stage rule, the index table, the one-load plumbing, and the box itself.
"""
import ast
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

import core.data_engine as de  # noqa: E402
import ui.ui_market as M  # noqa: E402
from config import BENCHMARK_INDICES  # noqa: E402

_APP = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()


# ── The one stage rule ───────────────────────────────────────────────────────────────────────────────
def test_one_weinstein_rule_for_stocks_and_indices():
    close = pd.Series([110.0, 110.0, 90.0, 90.0, np.nan])
    s30 = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0])
    s200 = pd.Series([95.0, 105.0, 95.0, 105.0, 95.0])
    assert de.weinstein_stage(close, s30, s200).tolist() == [
        "📈 Stage 2 Advancing", "🔄 Stage 1 Basing", "⚠️ Stage 3 Top", "📉 Stage 4 Declining", "❔ Unknown"]
    src = open(os.path.join(ROOT, "core", "data_engine.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    # Exactly one place PRODUCES stage labels (an np.select over them), and it is weinstein_stage;
    # other code may READ a label (the trend modifiers compare against one).
    producers = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == "np.select"
                 and "📉 Stage 4 Declining" in ast.unparse(n)]
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "weinstein_stage")
    assert len(producers) == 1 and producers[0] in list(ast.walk(fn)), "one stage rule, in weinstein_stage"
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "weinstein_stage"]
    assert len(calls) >= 2, "both the stocks and the index table must use the one rule"


# ── CAN SLIM's M reads the scoring regime ────────────────────────────────────────────────────────────
def test_can_slim_m_reads_the_scoring_regime_before_the_detected_one():
    """The third path from the regime into the scores (2026-10-08). No stock in the local data passes
    CAN SLIM's other legs, so a forced BEAR cannot show M's source behaviourally there — the lookup
    order is pinned instead: the scoring regime first (attrs, then its column), the detected one only
    as a fallback for frames scored outside run_full_scoring."""
    src = open(os.path.join(ROOT, "core", "scoring_engine.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    assigns = {ast.unparse(n.targets[0]): ast.unparse(n.value) for n in ast.walk(tree)
               if isinstance(n, ast.Assign) and len(n.targets) == 1}
    assert assigns["_regime_cs"].startswith("df.attrs.get('scoring_regime', df.attrs.get('detected_market_regime'")
    assert assigns["_regime_col"].startswith("next((c for c in ('_scoring_regime', '_detected_market_regime')")
    assert assigns["market_ok_cs"] == "_regime_cs != 'BEAR'"


# ── The index table ──────────────────────────────────────────────────────────────────────────────────
def _tech_rows():
    """The Technicals tab as the loader hands it over: a stock, every declared index in a scrambled
    order, and an undeclared index-like row (no market cap / sector / industry). One price per declared
    index, alternating above and below its averages, so the fixture follows config (ten since
    2026-11-01) instead of a hardcoded count."""
    ids = ["NSE:TCS", *reversed(list(BENCHMARK_INDICES)), "NSE:NEWINDEX"]
    n = len(ids)
    close = [4000.0] + [110.0 if i % 2 == 0 else 90.0 for i in range(len(BENCHMARK_INDICES))] + [50.0]
    return pd.DataFrame({"company_id": ids, "close_price": close, "sma_30w": [100.0] * n,
                         "sma_200d": [95.0] * n, "sma_50d": [101.0] * n, "rsi_14d": [40.0] * n,
                         "dist_52wh": [8.0] * n})


def test_the_index_table_holds_the_declared_indices_in_order_with_their_stages():
    t = de.benchmark_table({"technical": _tech_rows()})
    assert t["company_id"].tolist() == list(BENCHMARK_INDICES), "config order, the market first"
    assert t["name"].tolist() == list(BENCHMARK_INDICES.values())
    expect = de.weinstein_stage(t["close_price"], t["sma_30w"], t["sma_200d"])
    assert t["stage"].tolist() == list(expect)
    assert set(t["stage"]) == {"📈 Stage 2 Advancing", "⚠️ Stage 3 Top"}
    assert "NSE:NEWINDEX" not in set(t["company_id"]), "only the declared indices are shown"


def test_the_index_table_is_empty_without_index_rows():
    stocks_only = _tech_rows().iloc[[0]]
    assert de.benchmark_table({"technical": stocks_only}).empty
    assert de.benchmark_table({}).empty and "stage" in de.benchmark_table({}).columns


def test_one_load_gives_the_stocks_and_the_index_table(monkeypatch):
    """load_market reads the data ONCE and returns exactly what fetch_and_clean_data returns, plus
    the index table; the app's cached loader uses it, never a second fetch."""
    calls = []
    real = de.load_all_csvs

    def counting(*a, **k):
        calls.append(k.get("data_source", a[0] if a else "local"))
        return real(*a, **k)
    monkeypatch.setattr(de, "load_all_csvs", counting)
    try:
        stocks, bench = de.load_market("local")
    except Exception as exc:                                       # the CSVs are gitignored
        pytest.skip(f"local data not available: {exc}")
    # One load per call: the local branch may hand a workbook to the upload parser internally
    # ("upload"), but the source itself is read once.
    assert calls.count("local") == 1, f"one load, two results: {calls}"
    assert isinstance(bench, pd.DataFrame) and "stage" in bench.columns
    pd.testing.assert_frame_equal(stocks, de.fetch_and_clean_data("local"))
    tree = ast.parse(_APP)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_clean_data")
    called = {ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)}
    assert "load_market" in called and "fetch_and_clean_data" not in called


# ── The box ──────────────────────────────────────────────────────────────────────────────────────────
def test_short_names_cover_exactly_the_declared_indices_in_order():
    assert list(M.SHORT_NAMES) == list(BENCHMARK_INDICES)
    assert M.MARKET_ID == next(iter(BENCHMARK_INDICES)), "the market leads"


def _bench():
    return de.benchmark_table({"technical": _tech_rows()})


def test_the_box_leads_with_the_market_and_shows_every_index_stage():
    h = M.market_box_html(_bench(), "SIDEWAYS")
    names = [M.SHORT_NAMES[i] for i in BENCHMARK_INDICES]
    pos = [h.index(f">{n}<") for n in names]
    assert pos == sorted(pos), "the indices in config order, the Nifty 500 first"
    assert "Market · Weinstein stage" in h and "📈 Stage 2" in h and "⚠️ Stage 3" in h
    assert "font-weight:800" in h.split(">Nifty 500<")[0][-200:], "the market row is the bold one"
    nifty = _bench().set_index("company_id").loc[M.MARKET_ID, "stage"]
    assert f"border-left:4px solid {M._stage_colour(nifty, M.COLORS)}" in h, "the edge follows the Nifty 500"


# ── Nifty500 Shariah (2026-11-01): Kite trades only Shariah stocks, so this is its real market ───────────────
SH = "NSE:NIFTY500SH"
_NAN = re.compile(r"\b(nan|None)\b", re.I)


def test_the_shariah_index_has_its_short_name():
    assert M.SHORT_NAMES[SH] == "Shariah 500"


def test_shariah_500_renders_second_with_its_full_name_as_the_tooltip():
    h = M.market_box_html(_bench(), "SIDEWAYS")
    pos = [h.index(f">{n}<") for n in ("Nifty 500", "Shariah 500", "Smallcap 250")]
    assert pos == sorted(pos), "the market, then the Shariah 500, then the rest as before"
    assert 'title="Nifty500 Shariah: ' in h


def test_a_missing_shariah_row_is_left_out_and_nothing_reads_nan():
    """Copies before 2 Nov 2026 have no Nifty500 Shariah row: the box leaves it out, exactly as it left out
    the two MidSmall indices on copies before 13 Oct, and every other row renders as before."""
    tech = _tech_rows()
    h = M.market_box_html(de.benchmark_table({"technical": tech[tech["company_id"] != SH]}), "SIDEWAYS")
    assert ">Shariah 500<" not in h and "Nifty500 Shariah" not in h
    pos = [h.index(f">{M.SHORT_NAMES[i]}<") for i in BENCHMARK_INDICES if i != SH]
    assert pos == sorted(pos)
    assert not _NAN.search(h)


@pytest.mark.parametrize("blank", ["close_price", "sma_30w", "sma_200d"])
def test_a_blank_shariah_input_reads_unknown_never_nan(blank):
    """The row is there but a number the stage needs is blank: the ❔ Unknown every stock and index reads,
    muted, never 'nan', 'None', 'Stage nan' or an empty badge."""
    tech = _tech_rows()
    tech.loc[tech["company_id"] == SH, blank] = np.nan
    bench = de.benchmark_table({"technical": tech})
    assert bench.set_index("company_id").loc[SH, "stage"] == "❔ Unknown"
    h = M.market_box_html(bench, "SIDEWAYS")
    row = h.split('title="Nifty500 Shariah: ')[1].split("</div>")[0]
    assert ">Shariah 500<" in row and ">❔ Unknown<" in row and M.COLORS["text_muted"] in row
    assert not _NAN.search(h)


def test_the_box_labels_the_regime_as_shown_not_applied(monkeypatch):
    h = M.market_box_html(_bench(), "BULL")
    assert "Breadth regime: <b" in h and "Bull" in h and "shown, not applied to scores" in h
    monkeypatch.setattr(M, "REGIME_DRIVES_SCORING", True)
    assert "· applied to scores" in M.market_box_html(_bench(), "BULL")


def test_without_index_data_the_box_shows_the_regime_honestly():
    for empty in (pd.DataFrame(), None, de.benchmark_table({})):
        h = M.market_box_html(empty, "SIDEWAYS")
        assert "Detected Regime" in h and "SIDEWAYS MARKET" in h
        assert "shown, not applied to scores" in h and "Index stages appear once" in h


def test_the_sidebar_renders_the_box_from_the_one_load():
    """The sheet's table from the one load, and NSE's numbers (2026-11-04) on the sheet's own date when it has one."""
    src = _APP
    assert "clean_df, bench_df, load_time = get_clean_data(" in src
    assert ("st.markdown(market_box_html(bench_df, regime, get_index_view(_as_of), ref_day=_ref_day), "
            "unsafe_allow_html=True)") in src
    assert "_as_of = _fresh.data_date if _fresh.is_known else None" in src
    assert ">Detected Regime</div>" not in src, "the old inline box is gone; ui_market renders it"
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_index_view")
    deco = [ast.unparse(d) for d in fn.decorator_list]
    assert any(d.startswith("st.cache_data(") and "ttl=3600" in d for d in deco), deco
    body = ast.unparse(fn)
    assert "read_history(INDEX_CLOSES_URL)" in body and "box_view(" in body and "as_of=as_of" in body


# ── NSE's own numbers and the 200-week line (2026-11-04) ─────────────────────────────────────────────────────────
import html as _html  # noqa: E402
from datetime import date as _date  # noqa: E402

import core.index_closes as X  # noqa: E402

D = _date(2026, 11, 4)


def _view(over=None, day=D, drop=()):
    """A box_view for every declared index: a 205-week history, 10% above its 200-week average, Stage 2."""
    base = {cid: dict(close=110.0, sma_30w=100.0, sma_200d=95.0, sma_200w=100.0, weeks=205, vs_200w=10.0,
                      stage="📈 Stage 2 Advancing", last_date=day) for cid in BENCHMARK_INDICES if cid not in drop}
    for cid, kw in (over or {}).items():
        base[cid].update(kw)
    return pd.DataFrame([dict(company_id=cid, name=BENCHMARK_INDICES[cid], **v) for cid, v in base.items()])[
        X.VIEW_COLUMNS]


def _sheet(stage="📉 Stage 4 Declining"):
    b = _bench()
    b["stage"] = stage
    return b


def _cell_tip(h, full):
    return h.split(f'title="{full}: ')[1].split('"')[0]


def test_with_nse_numbers_each_row_shows_its_stage_and_its_200_week_line():
    h = M.market_box_html(_sheet(), "SIDEWAYS", _view(), ref_day=D)
    pos = [h.index(f">{M.SHORT_NAMES[i]}<") for i in BENCHMARK_INDICES]
    assert pos == sorted(pos), "config order, the Nifty 500 first"
    assert ">vs 200W<" in h and ">Stage<" in h and h.count(">+10.0%<") == len(BENCHMARK_INDICES)
    assert "NSE closes · 4 Nov 2026" in h and "vs 200W unavailable" not in h
    assert "font-weight:800" in h.split(">Nifty 500<")[0][-300:], "the market row is the bold one"
    assert h.count("grid-template-columns:minmax(0,1fr) auto auto") == 1, "one grid: the columns line up"
    assert h.count("grid-column:1 / -1") == 1, "one rule under the Nifty 500"


def test_nses_stage_leads_over_the_sheets():
    """3 Nov 2026: the vendor's history put Nifty500 Shariah in Stage 4; NSE's own numbers give Stage 3."""
    view = _view({"NSE:NIFTY500SH": dict(stage="⚠️ Stage 3 Top", close=6796.3, sma_200w=6621.4, vs_200w=2.64)})
    h = M.market_box_html(_sheet("📉 Stage 4 Declining"), "SIDEWAYS", view, ref_day=D)
    tip = _cell_tip(h, "Nifty500 Shariah")
    assert tip == "⚠️ Stage 3 Top · 2.6% above its 200-week average (6,621.4) · NSE close 6,796.3 on 4 Nov 2026"
    pct, stage = h.split(">Shariah 500<")[1].split("</span>")[0:2]
    assert pct.endswith(">+2.6%") and stage.endswith(">⚠️ Stage 3")


@pytest.mark.parametrize("v,text,colour", [(14.94, "+14.9%", "green"), (-1.62, "−1.6%", "red"),
                                           (0.04, "0.0%", "text_muted"), (-0.04, "0.0%", "text_muted")])
def test_vs_200w_is_coloured_on_the_figure_as_printed(v, text, colour):
    h = M.market_box_html(_sheet(), "SIDEWAYS", _view({"NSE:MIDSMAL400": dict(vs_200w=v)}), ref_day=D)
    cell = h.split(">MidSmall 400<")[1].split("</span>")[0]
    assert cell.endswith(f">{text}") and f"color:{M.COLORS[colour]};" in cell


def test_a_young_index_shows_a_dash_and_says_why():
    view = _view({"NSE:NIF500MO50": dict(sma_200w=np.nan, vs_200w=np.nan, weeks=127)})
    h = M.market_box_html(_sheet(), "SIDEWAYS", view, ref_day=D)
    assert h.split(">Momentum 50<")[1].split("</span>")[0].endswith(">—")
    assert "127 weeks of NSE closes, fewer than the 200 a 200-week average needs" in _cell_tip(h, "Nifty500 Momentum 50")


def test_an_index_nse_lacks_keeps_the_sheets_stage_and_says_so():
    h = M.market_box_html(_sheet("⚠️ Stage 3 Top"), "SIDEWAYS", _view(drop=("NSE:MIDSMAL400",)), ref_day=D)
    assert h.split(">MidSmall 400<")[1].split("</span>")[0].endswith(">—")
    assert _cell_tip(h, "Nifty MidSmallcap 400") == "⚠️ Stage 3 Top · from the sheet: NSE had no close for it"


def test_an_index_whose_nse_closes_stopped_keeps_the_sheets_stage_and_says_so():
    """Review, 4 Nov 2026: an index NSE renames or drops keeps its last close on the branch for ever, so its numbers
    would have sat under "NSE closes · <today>" for months. More than INDEX_CLOSES_STALE_DAYS before the sheet's date,
    a row is the sheet's again and says why; with no sheet row it is left out."""
    old = {"NSE:MIDSMAL400": dict(last_date=_date(2026, 10, 27), vs_200w=33.3)}
    h = M.market_box_html(_sheet("⚠️ Stage 3 Top"), "SIDEWAYS", _view(old), ref_day=D)
    assert h.split(">MidSmall 400<")[1].split("</span>")[0].endswith(">—") and ">+33.3%<" not in h
    assert _html.unescape(_cell_tip(h, "Nifty MidSmallcap 400")) == (
        "⚠️ Stage 3 Top · from the sheet: NSE's closes for it stop at 27 Oct 2026")
    assert ">MidSmall 400<" not in M.market_box_html(pd.DataFrame(), "SIDEWAYS", _view(old), ref_day=D)
    week = {"NSE:MIDSMAL400": dict(last_date=_date(2026, 10, 28), vs_200w=33.3)}
    assert ">+33.3%<" in M.market_box_html(_sheet(), "SIDEWAYS", _view(week), ref_day=D), "a week behind is NSE's"


def test_the_source_line_says_it_reads_on_or_before_the_sheets_date():
    """Review, 4 Nov 2026: in the evening the sheet is often a session ahead of NSE's newest kept close; the line under
    the rows gives that close's date, and its tooltip says how the date is chosen."""
    h = _html.unescape(M.market_box_html(_sheet(), "SIDEWAYS", _view(day=_date(2026, 11, 3)), ref_day=D))
    assert "NSE closes · 3 Nov 2026" in h and "on or before the sheet's date" in h


def test_no_text_in_the_box_is_smaller_than_the_design_systems_sub_label():
    """docs/ui_design_system.md: the scale's smallest size is the 0.62rem sub-label (review, 4 Nov 2026: the column
    labels were 0.58)."""
    for nse in (_view(), None):
        h = M.market_box_html(_sheet(), "SIDEWAYS", nse, ref_day=D)
        assert min(float(s) for s in re.findall(r"font-size:([\d.]+)rem", h)) >= 0.62


@pytest.mark.parametrize("nse,why", [(None, "NSE's index closes could not be read"),
                                     (pd.DataFrame(columns=X.VIEW_COLUMNS), "NSE's index closes could not be read"),
                                     ("stale", "NSE's index closes stop at 21 Oct 2026")])
def test_without_fresh_nse_numbers_the_box_is_the_sheets_and_says_why(nse, why):
    nse = _view(day=_date(2026, 10, 21)) if isinstance(nse, str) else nse
    h = _html.unescape(M.market_box_html(_sheet("⚠️ Stage 3 Top"), "SIDEWAYS", nse, ref_day=D))
    assert h.count(f"vs 200W unavailable: {why} — stages from the sheet.") == 1
    assert ">vs 200W<" not in h and "NSE closes ·" not in h and "grid-template-columns" not in h
    assert h.count("display:flex;justify-content:space-between") == len(BENCHMARK_INDICES), "the sheet's rows as before"
    assert h.count(">⚠️ Stage 3<") == len(BENCHMARK_INDICES) and ">📉 Stage 4<" not in h, "every row the sheet's stage"


def test_a_week_old_view_is_still_fresh_and_a_day_more_is_not():
    fresh = M.market_box_html(_sheet(), "SIDEWAYS", _view(day=_date(2026, 10, 28)), ref_day=D)
    stale = M.market_box_html(_sheet(), "SIDEWAYS", _view(day=_date(2026, 10, 27)), ref_day=D)
    assert "NSE closes · 28 Oct 2026" in fresh and "vs 200W unavailable" in stale


def test_without_the_sheets_index_rows_nses_numbers_still_fill_the_box():
    h = M.market_box_html(pd.DataFrame(), "SIDEWAYS", _view(), ref_day=D)
    assert ">vs 200W<" in h and "Detected Regime" not in h and ">Shariah 500<" in h
    assert "Detected Regime" in M.market_box_html(pd.DataFrame(), "SIDEWAYS", None, ref_day=D)


def test_the_edge_follows_the_nifty_500s_nse_stage():
    h = M.market_box_html(_sheet("📈 Stage 2 Advancing"), "SIDEWAYS",
                          _view({"NSE:CNX500": dict(stage="📉 Stage 4 Declining")}), ref_day=D)
    assert f"border-left:4px solid {M.COLORS['red']}" in h


def test_nothing_reads_nan_or_none_in_any_mode():
    views = [_view(), _view({"NSE:NIF500MO50": dict(sma_200w=np.nan, vs_200w=np.nan, weeks=127)}),
             _view(drop=("NSE:CNX500",)), None, _view(day=_date(2026, 9, 1))]
    for v in views:
        for sheet in (_sheet(), pd.DataFrame()):
            h = M.market_box_html(sheet, "SIDEWAYS", v, ref_day=D)
            assert not _NAN.search(h), h[:300]
