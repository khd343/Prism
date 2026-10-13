"""
test_scan_filters.py — the 🎯 Scans tab's own filters: Liquidity · Market cap · Weinstein Stage · Buy Zone
(2026-10-13, the user's choice), opening on ₹1 Cr a day and Stage 1 + 2.

Until today the tab carried the Market Pulse lens row (Sector · Wealth tier · Market cap · Catalyst) beside
Liquidity. All of those also live in the sidebar, which reaches this list through the 🎯 opt-in, so the
user kept only the four that answer "can I buy it, and is this a moment to": Liquidity, Market cap,
Weinstein Stage and Buy Zone — and set two defaults: Liquidity ₹1 Cr a day, Stage 1 + Stage 2. Every
control narrows the FINISHED list (the recipe already ran on the whole market); the caption names what
narrows it; 🧹 Clear returns every control to ITS default, never to "everything".

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
DEFAULT_STAGES = ["📈 Stage 2 Advancing", "🔄 Stage 1 Basing"]


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


def test_the_tab_offers_exactly_four_filters_of_its_own():
    labels = _labels(_scans_block()) + _row_labels()
    assert sorted(labels) == sorted(["Liquidity", "Market cap", "Weinstein Stage", "Buy Zone"]), labels
    calls = {ast.unparse(n.func) for n in ast.walk(_scans_block()) if isinstance(n, ast.Call)}
    assert "_mp_lens_row" not in calls, "Sector, Wealth tier and Catalyst live in the sidebar (via 🎯)"


def test_the_filters_read_left_to_right_liquidity_first():
    order = _row_labels()
    assert order == ["Market cap", "Weinstein Stage", "Buy Zone"], "the cascade's order is the row's order"
    src = ast.unparse(_scans_block())
    assert src.index("'Liquidity'") < src.index("_scan_filter_row("), "Liquidity is the first control"


def test_the_list_opens_on_one_crore_a_day_and_stage_1_and_2():
    node = next(n for n in _TREE.body if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == "_SCAN_FILTER_DEFAULTS")
    assert ast.literal_eval(node.value) == {"scan_liq": 1.0, "scan_cap": [], "scan_stage": DEFAULT_STAGES,
                                            "scan_zone": []}


def test_the_options_are_the_sidebars_one_definition():
    from ui import ui_discovery as D
    assert D._WEIN_ORDER[:2] == DEFAULT_STAGES and D._WEIN_ORDER[-1] == D._UNKNOWN
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
