"""
test_scan_filters.py — the 🎯 Scans tab's own filters: Liquidity · Shariah · Exchange · Market cap · Weinstein Stage ·
Buy Zone (2026-10-13, the user's choice; Shariah joined 2026-10-16 and Exchange, on Shariah's line, 2026-11-03 — the
operator's briefs), every one opening empty — Shariah and Exchange on All, which is no filter.

Until today the tab carried the Market Pulse lens row (Sector · Wealth tier · Market cap · Catalyst) beside
Liquidity. All of those also live in the sidebar, which reaches this list through the 🎯 opt-in, so the
user kept only the four that answer "can I buy it, and is this a moment to": Liquidity, Market cap,
Weinstein Stage and Buy Zone. It first shipped opening on ₹1 Cr a day and Stage 1 + 2; the user removed
both defaults the same day, so every control opens EMPTY and the list opens on the whole scan. Every
control narrows the FINISHED list (the recipe already ran on the whole market); the caption names what
narrows it; 🧹 Clear empties the row.

Pinned here from the source (AST); tests/test_scans_tab_runs.py runs the real tab and checks behaviour.

Run with: pytest tests/test_scan_filters.py -v
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_APP = (ROOT / "app.py").read_text(encoding="utf-8")
_TREE = ast.parse(_APP)
FIRST_STAGES = ["📈 Stage 2 Advancing", "🔄 Stage 1 Basing"]     # the order the sidebar lists them in


def _scans_block():
    return next(n for n in _TREE.body if isinstance(n, ast.With)
                and ast.unparse(n.items[0].context_expr) == "tabs[1]")


def _fn(name):
    return next(n for n in _TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _labels(node):
    """The literal label of every select widget the tab block draws itself (st.selectbox /
    st.multiselect, first argument)."""
    return [c.args[0].value for c in ast.walk(node) if isinstance(c, ast.Call)
            and ast.unparse(c.func) in ("st.selectbox", "st.multiselect")
            and c.args and isinstance(c.args[0], ast.Constant)]


def _row_labels():
    """The labels _scan_filter_row draws, in order: the first field of each entry of its `spec`."""
    spec = next(n for n in ast.walk(_fn("_scan_filter_row")) if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == "spec")
    calls = [n for n in ast.walk(_fn("_scan_filter_row")) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "_mp_ms"]
    assert len(calls) == 1 and ast.unparse(calls[0].args[1]) == "label", "every row control goes through _mp_ms"
    return [e.elts[0].value for e in spec.value.elts]


def _shariah_labels():
    """The label of the Shariah control the row draws — the sidebar's own (ui_discovery.shariah_filter)."""
    return [c.args[0].value for c in ast.walk(_fn("_scan_filter_row")) if isinstance(c, ast.Call)
            and ast.unparse(c.func) == "shariah_filter"]


def _exchange_labels():
    """The label of the Exchange control the row draws — the sidebar's own (ui_discovery.exchange_filter)."""
    return [c.args[0].value for c in ast.walk(_fn("_scan_filter_row")) if isinstance(c, ast.Call)
            and ast.unparse(c.func) == "exchange_filter"]


def test_the_tab_offers_exactly_six_filters_of_its_own():
    """Four of the user's (2026-10-13), Shariah (2026-10-16) and Exchange (2026-11-03), the last two drawn by the
    controls the sidebar uses. REPOINTED 2026-11-03 from five: Exchange joined."""
    labels = _labels(_scans_block()) + _shariah_labels() + _exchange_labels() + _row_labels()
    assert sorted(labels) == sorted(["Liquidity", "Shariah", "Exchange", "Market cap", "Weinstein Stage",
                                     "Buy Zone"]), labels
    calls = {ast.unparse(n.func) for n in ast.walk(_scans_block()) if isinstance(n, ast.Call)}
    assert "_mp_lens_row" not in calls, "Sector, Wealth tier and Catalyst live in the sidebar (via 🎯)"


def test_the_filters_read_left_to_right_liquidity_first():
    order = _row_labels()
    assert order == ["Market cap", "Weinstein Stage", "Buy Zone"], "the cascade's order is the row's order"
    src = ast.unparse(_scans_block())
    assert src.index("'Liquidity'") < src.index("_scan_filter_row("), "Liquidity is the first control"
    row = ast.unparse(_fn("_scan_filter_row"))
    assert row.index("shariah_filter(") < row.index("for i, (label"), "Shariah opens the row, after Liquidity"
    # 2026-11-03: Shariah → Exchange → Market cap → Stage → Buy Zone. Exchange is drawn on the stocks Shariah kept and
    # applied before the multiselects are counted.
    assert (row.index("shariah_filter(") < row.index("shariah_mask(") < row.index("exchange_filter(")
            < row.index("exchange_mask(") < row.index("for i, (label")), "Exchange comes right after Shariah"


def test_every_control_opens_empty():
    """REPOINTED 2026-10-13, the user's call: the ₹1 Cr a day / Stage 1 + 2 defaults were removed. 2026-10-16:
    Shariah joined, opening on All — no filter — so nobody's view changed (the operator's brief). 2026-11-03: Exchange
    joined the same way."""
    from ui.shariah_view import SHARIAH_ALL
    from ui.ui_discovery import EXCHANGE_ALL
    node = next(n for n in _TREE.body if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == "_SCAN_FILTER_DEFAULTS")
    assert ast.literal_eval(node.value) == {"scan_liq": None, "scan_shariah": SHARIAH_ALL,
                                            "scan_exchange": EXCHANGE_ALL, "scan_cap": [], "scan_stage": [],
                                            "scan_zone": []}


def test_the_options_are_the_sidebars_one_definition():
    from ui import ui_discovery as D
    assert D._WEIN_ORDER[:2] == FIRST_STAGES and D._WEIN_ORDER[-1] == D._UNKNOWN
    assert D._BUY_ZONE_ORDER[0] == "🟢 Perfect Entry (Low Risk)" and len(D._BUY_ZONE_ORDER) == 6
    imp = next(n for n in _TREE.body if isinstance(n, ast.ImportFrom) and n.module == "ui.ui_discovery")
    assert {"_BUY_ZONE_ORDER", "_WEIN_ORDER"} <= {a.name for a in imp.names}, "imported, never retyped"
    src = ast.unparse(_fn("_scan_filter_row"))
    assert "_WEIN_ORDER" in src and "_BUY_ZONE_ORDER" in src and "_MP_CAP_ORDER" in src


def test_the_clear_returns_each_control_to_its_default_and_leaves_the_view_alone():
    src = ast.unparse(_fn("_scan_filter_row"))
    assert "on_click=_mp_clear_lens" in src and "_SCAN_FILTER_DEFAULTS" in src
    defaults = ast.literal_eval(next(n for n in _TREE.body if isinstance(n, ast.Assign)
                                     and ast.unparse(n.targets[0]) == "_SCAN_FILTER_DEFAULTS").value)
    assert "scan_view" not in defaults, "the Column View is a display choice, not a filter"


def test_exchange_shares_the_shariah_line_and_adds_no_row():
    """2026-11-03: Exchange sits on Shariah's line, right of it (the one line split into two slots), so the table
    starts where it did: the tab still draws three rows of columns — the cards, the Shariah line, the filter row."""
    block = _scans_block()
    cols = [n for n in ast.walk(block) if isinstance(n, ast.Call) and ast.unparse(n.func) == "st.columns"]
    assert len(cols) == 3, [ast.unparse(c) for c in cols]
    line = [ast.unparse(n) for n in ast.walk(block) if isinstance(n, ast.Assign)
            and ast.unparse(n).startswith("_sc_sh, _sc_ex = st.columns(")]
    assert len(line) == 1, "Shariah and Exchange take the two slots of one line"
    call = next(n for n in ast.walk(block) if isinstance(n, ast.Call) and ast.unparse(n.func) == "_scan_filter_row")
    assert [ast.unparse(a) for a in call.args[1:]] == ["_sc_slots", "_sc_sh", "_sc_ex"]


def test_the_exchange_reaches_only_the_finished_list():
    """The recipe ranks the whole market and the cards count it, so Exchange is drawn and applied inside the row
    alone, which is handed the finished list (tests/test_scans_tab_runs.py runs it)."""
    row = ast.unparse(_fn("_scan_filter_row"))
    for name in ("exchange_filter(", "exchange_mask("):
        assert _APP.count(name) == 1 and name in row, name
