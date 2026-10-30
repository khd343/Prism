"""
test_scan_views.py — the Scans table gains the app's 📈 Technical view (2026-10-13).

The user asked for these columns on every scan: Stock, Sector, Score, RS 52W, Stop Δ, Momentum,
Breakout, ₹ Cr/day, RSI, Trend, Price and Mom. Quality. Six were already on the table; the six
missing ones are the Deep Scanner's 📈 Technical columns plus Price. Adding them to the one table
would have pushed it ~450px past a 1536px laptop (it already used 1,025 of 1,030px) and mixed two
questions — "why is it in the list" with "is now a good moment, how far is the stop, can I buy one
share" — so they arrive as a SECOND VIEW under the app-wide names, the way the Deep Scanner and the
Sectors / Industry tables already work.

Pinned: the two views by their app-wide names; 🏆 Core unchanged and the default; the 📈 Technical
columns, with Mom. Quality between the two numbers it reads (Momentum and RSI); every column of
every view configured with the Deep Scanner's header and tip, inside the screen; the table carries
every column a view can show; the view is a display choice the 🧹 Clear never resets.

Run with: pytest tests/test_scan_views.py -v
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ui import ui_scans as S  # noqa: E402
from ui.ui_group_views import VIEWS  # noqa: E402
from ui.ui_scanner import _SCANNER_HEADER_TIPS  # noqa: E402

_APP = (ROOT / "app.py").read_text(encoding="utf-8")
CORE, TECH = "🏆 Core", "📈 Technical"
TECH_COLS = ("rank", "_new", "name", "close_price", "_liq_cr", "dist_to_vstop", "momentum_score",
             "d49_momentum_quality", "rsi_14d", "breakout_score", "crs_52w", "weinstein_stage")


def test_the_scans_offer_two_views_by_their_app_wide_names():
    assert S.SCAN_VIEWS == (CORE, TECH)
    assert set(S.SCAN_VIEWS) <= set(VIEWS), "one vocabulary: the Deep Scanner's and Market Pulse's view names"


@pytest.mark.parametrize("scan", S.SCANS, ids=lambda s: s["key"])
def test_core_is_the_table_it_was_and_the_default(scan):
    assert S.table_columns(scan) == S.table_columns(scan, CORE)
    assert "dist_to_vstop" not in S.table_columns(scan, CORE), "Core keeps its columns"


@pytest.mark.parametrize("scan", S.SCANS, ids=lambda s: s["key"])
def test_the_technical_view_holds_the_entry_columns(scan):
    cols = S.table_columns(scan, TECH)
    assert cols == TECH_COLS
    i = cols.index("d49_momentum_quality")
    assert cols[i - 1] == "momentum_score" and cols[i + 1] == "rsi_14d", (
        "Mom. Quality sits between the two numbers it reads, as in the Deep Scanner")


@pytest.mark.parametrize("view", (CORE, TECH))
@pytest.mark.parametrize("scan", S.SCANS, ids=lambda s: s["key"])
def test_every_view_fits_a_laptop_screen_with_the_deep_scanners_headers(scan, view):
    cols, w = S.table_columns(scan, view), S.table_widths(scan)
    assert sum(w[c] for c in cols) + 44 <= 1074, f"{scan['key']} {view}: wider than a 1536px laptop"
    cfg = S.table_config("03 Oct", scan)
    for c in cols:
        assert c in cfg and cfg[c]["width"] == w[c], c
        label = cfg[c]["label"]
        assert label and label != c and not (label.islower() and "_" in label), (c, label)


# ── The Category column (2026-10-14, the user's ask: "in Core, before the Sector column") ──────────────
@pytest.mark.parametrize("scan", S.SCANS, ids=lambda s: s["key"])
def test_core_shows_the_category_right_before_the_sector(scan):
    cols = S.table_columns(scan, CORE)
    assert cols[cols.index("sector") - 1] == "market_category", "Category sits right before Sector"
    assert "market_category" not in S.table_columns(scan, TECH), "asked for in Core only"
    cfg = S.table_config("03 Oct", scan)
    assert cfg["market_category"]["label"] == "Category", "the user's word, and the vendor's (Market Category)"
    assert "Mega, Large, Mid, Small, Micro or Nano Cap" in cfg["market_category"]["help"]


def test_the_table_carries_each_stocks_category():
    df = pd.DataFrame({"company_id": ["NSE:A", "NSE:B"], "name": ["A", "B"], "rank": [1, 2],
                       "market_category": ["Small Cap", np.nan]})
    t = S.scan_table(df, {"key": "all", "mask": lambda d: pd.Series(True, index=d.index)})
    assert t.loc[0, "market_category"] == "Small Cap"
    assert pd.isna(t.loc[1, "market_category"]), "a missing category shows blank, never a guess"


# The narrowest each column can be and still show its header and its widest common value in full —
# measured 2026-10-14 in the grid's own font (14px Source Sans + 16px padding), read from the live grid:
# headers Rank 46, Momentum 84, Improvement 96, ⚠ Measured flags 119, ₹ Cr/day 64, Cycle ↓ 59, 🔥 36,
# Category 69; values "2717" 44, "1234.56" 62, "📈 Stage 2" 81, "Small Cap" / "Large Cap" 75. A width
# below its floor cuts the header or the common value.
_FLOOR = {"rank": 46, "momentum_score": 84, "improvement_score": 96, "_measured_flags": 119, "_liq_cr": 64,
          "_cycle": 59, "_strong_q": 36, "weinstein_stage": 81, "market_category": 75}


@pytest.mark.parametrize("scan", S.SCANS, ids=lambda s: s["key"])
def test_no_column_is_narrower_than_its_header_and_common_values(scan):
    w = S.table_widths(scan)
    for view in S.SCAN_VIEWS:
        for c in S.table_columns(scan, view):
            if c in _FLOOR:
                assert w[c] >= _FLOOR[c], f"{scan['key']} {view}: {c} at {w[c]}px is under its {_FLOOR[c]}px"


def test_the_new_columns_carry_the_deep_scanners_headers_and_tips():
    cfg = S.table_config("03 Oct", S.SCANS[0])
    assert {c: cfg[c]["label"] for c in ("close_price", "dist_to_vstop", "d49_momentum_quality", "rsi_14d",
                                         "breakout_score", "crs_52w")} == {
        "close_price": "Price ₹", "dist_to_vstop": "Stop Δ", "d49_momentum_quality": "Mom. Quality",
        "rsi_14d": "RSI", "breakout_score": "Breakout", "crs_52w": "RS 52W"}
    for c in ("close_price", "dist_to_vstop", "d49_momentum_quality", "rsi_14d", "breakout_score", "crs_52w"):
        assert cfg[c]["help"] == _SCANNER_HEADER_TIPS[c], f"{c}: one tip, the Deep Scanner's"


def test_the_table_carries_every_column_a_view_can_show():
    df = pd.DataFrame({"company_id": ["NSE:A"], "name": ["A"], "rank": [1], "close_price": [250.0],
                       "rsi_14d": [61.0], "crs_52w": [80.0], "d49_momentum_quality": ["⚡ HIGH"]})
    t = S.scan_table(df, {"key": "all", "mask": lambda d: pd.Series(True, index=d.index)})
    for view in S.SCAN_VIEWS:
        for c in S.table_columns(S.SCANS[0], view):
            assert c in t.columns, f"{view} shows {c}, the table must carry it"
    assert t.loc[0, "close_price"] == 250.0 and t.loc[0, "d49_momentum_quality"] == "⚡ HIGH"
    assert np.isnan(t.loc[0, "dist_to_vstop"]), "a missing input shows blank, never a made-up 0"


def _scans_block():
    tree = ast.parse(_APP)
    return next(n for n in tree.body if isinstance(n, ast.With) and ast.unparse(n.items[0].context_expr) == "tabs[1]")


def test_the_tab_lets_the_reader_switch_views_and_shows_the_chosen_one():
    src = ast.unparse(_scans_block())
    assert "st.segmented_control('Column View', list(SCAN_VIEWS), default='🏆 Core', key='scan_view')" in src
    # Repointed 2026-10-30, not loosened: the view applies to the chosen LIST of the selected scan (_sc_tbl_scan is
    # _sc_sel itself, or 🚀's B or C — tests/test_scan_ladder.py), whose key the widget key now carries.
    assert "_sc_tbl_scan = _sc_sel" in src and "_sc_tbl_scan = playbook_layer_scan(_sc_sel, _sc_layer)" in src
    assert "column_order=list(scan_table_columns(_sc_tbl_scan, _sc_view))" in src, "the table shows the chosen view"
    assert "scan_tbl_{_sc_tbl_scan['key']}_{_sc_view}_{_sc_ids}" in src.replace('"', "'"), (
        "a new view is a new widget, so a row pick from one view never carries into the other")


def test_the_view_is_a_display_choice_the_clear_never_resets():
    """The Market Pulse rule (2026-10-05): the view is never a filter — no 🧹 Clear resets it. REPOINTED
    2026-10-13 from the lens row (gone from this tab) to the tab's own 🧹, whose reset map is
    _SCAN_FILTER_DEFAULTS; and nothing in app.py writes the view's key."""
    tree = ast.parse(_APP)
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == "_SCAN_FILTER_DEFAULTS")
    assert "scan_view" not in ast.literal_eval(node.value), "the tab's 🧹 Clear must not reset the view"
    writes = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Assign)
              and any("scan_view" in ast.unparse(t) for t in n.targets)]
    assert not writes, writes
