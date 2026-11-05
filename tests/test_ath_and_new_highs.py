"""Off ATH (🔎 Deep Scanner, 📈 Technical): the distance to the all-time high, beside the breakout pair.

From ValuePickr's 52-week / all-time-high thread. Measured 2026-10-05 on the four saved snapshots
(corporate actions excluded), with the bar declared before measuring (>= +0.03 beyond Breakout and
Momentum in every window): the all-time-high distance added +0.051 / +0.049 / +0.037 / +0.054 to
forward rank-IC beyond the Breakout AND Momentum scores, more than any column the Technical view
already showed (RSI, Stop and 52-week RS added -0.03 to +0.07). It already feeds the Breakout score
(BREAKOUT_SIGNALS["ath_distance"]), so it is UNDER-weighted there, not missing: the column shows it,
and its weight is December's question (docs/december-2026-plan.md rule 15).

The sector half (📈 Sectors' Near 52w high) is pinned in tests/test_market_pulse_views.py.
"""
import ast
import contextlib
import io
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))

TREE = ast.parse(io.open(ROOT / "app.py", encoding="utf-8").read())


def _dict(name):
    node = next(n.value for n in ast.walk(TREE) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return dict(zip([k.value for k in node.keys], node.values))


def test_off_ath_left_the_technical_view_for_ram_rank_and_improvement():
    """The operator, 2026-11-05: it feeds the Breakout score already, and the table had no room."""
    tech = [e.value for e in _dict("_DS_VIEWS")["📈 Technical"].elts]
    assert "dist_ath" not in tech and "ram_rank" in tech and "improvement_score" in tech


def test_off_ath_stays_in_the_csv_download():
    """The export is built from the views and the sorts, so a column that leaves every view must be named, or it vanishes quietly."""
    src = io.open(ROOT / "app.py", encoding="utf-8").read()
    assert '"dist_ath"' in src[src.index("_export_cols = ["):src.index("_safe_mode = ")]


def test_the_tooltip_comes_from_the_glossary_and_says_what_it_is():
    from ui.ui_components import _RAW_GLOSSARY
    from ui.ui_scanner import _SCANNER_HEADER_TIPS
    tip = _RAW_GLOSSARY["Off ATH"]
    assert _SCANNER_HEADER_TIPS["dist_ath"] == tip, "one text, so the header and the Reference cannot drift"
    assert "0%" in tip and "all-time high" in tip and "52-week high" in tip
    assert "Breakout score" in tip, "the reader must learn that it already feeds Breakout (pinned below)"


def test_the_breakout_score_still_reads_the_all_time_high():
    """The tooltip says Off ATH is an input of the Breakout score. If the engine ever stops reading it,
    this fails and the tooltip must change with it: the words are tied to the engine."""
    src = io.open(ROOT / "core" / "scoring_engine.py", encoding="utf-8").read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_compute_breakout_score")
    assert any(isinstance(n, ast.Constant) and n.value == "dist_ath" for n in ast.walk(fn)), (
        "_compute_breakout_score no longer reads dist_ath, but the Off ATH tooltip says it does")
    from config import BREAKOUT_SIGNALS
    assert BREAKOUT_SIGNALS.get("ath_distance", 0) > 0


_DATA = ROOT / "Other Resources" / "CSV Data"


@pytest.mark.skipif(not _DATA.is_dir(), reason="local data not present (code-only checkout)")
def test_the_vendor_column_is_a_distance_below_the_all_time_high():
    from data_engine import coerce_numeric_columns, load_all_csvs, merge_datasets
    with contextlib.redirect_stdout(io.StringIO()):
        live = coerce_numeric_columns(merge_datasets(load_all_csvs("local")))
    ath = pd.to_numeric(live["dist_ath"], errors="coerce")
    high = pd.to_numeric(live["dist_52wh"], errors="coerce")
    assert ath.notna().mean() >= 0.95
    assert ath.min() >= 0 and ath.max() < 100, "a % below the high: 0 at the high, never 100 or more"
    both = ath.notna() & high.notna()
    assert (ath[both] >= high[both] - 0.5).mean() >= 0.99, (
        "the all-time high is at least the 52-week high, so the price can never sit closer to it")
