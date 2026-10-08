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
    """The Technicals tab as the loader hands it over: a stock, the seven indices in a scrambled
    order, and an undeclared index-like row (no market cap / sector / industry)."""
    ids = ["NSE:TCS", *reversed(list(BENCHMARK_INDICES)), "NSE:NEWINDEX"]
    n = len(ids)
    close = [4000.0] + [110.0, 90.0, 110.0, 90.0, 110.0, 90.0, 110.0] + [50.0]
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
    src = _APP
    assert "clean_df, bench_df, load_time = get_clean_data(" in src
    assert "st.markdown(market_box_html(bench_df, regime), unsafe_allow_html=True)" in src
    assert ">Detected Regime</div>" not in src, "the old inline box is gone; ui_market renders it"
