"""
test_scans_tab_runs.py
======================
The 🎯 Scans tab, RUN — not read (2026-10-07).

Every other Scans test reads app.py's tab block as source (AST). This one executes it: the real
`with tabs[1]:` block and `_pick_scan` are lifted out of app.py and run under Streamlit's AppTest on a
synthetic market, so a wrong name, a widget that raises, or a switch that does not reach the table
fails here instead of on the live app. Only the parts that need the full app are stubbed: the data
load (`df` / `filt`), the shared lens row (returns its input unfiltered) and the freshness probe.

What it pins: all three cards render, there is no toggle (2026-10-07: 🌱's revenue-led leg is part of
its recipe, not a switch), and picking 🌱 shows exactly the recipe's matches.
"""
import ast
import os
import sys
import textwrap

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

_HARNESS = textwrap.dedent('''
    import sys
    sys.path.insert(0, {root!r})
    import html as _html
    from datetime import date as _date
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd
    import streamlit as st

    from config import COLORS

    def universe_signature(ids):
        return str(abs(hash(tuple(ids))))

    def _mp_lens_row(frame, prefix, extra_keys=(), with_tier=True):
        return frame, 0

    _fresh = SimpleNamespace(is_known=False, data_date=None)

    rng = np.random.default_rng(7)
    n = 300
    df = pd.DataFrame({{
        "company_id": [f"NSE:T{{k:03d}}" for k in range(n)], "name": [f"Test {{k:03d}}" for k in range(n)],
        "sector": rng.choice(["Alpha", "Beta", "Gamma", "Delta", "Eps", "Zeta", "Eta"], n),
        "rank": rng.permutation(n) + 1, "composite_score": rng.uniform(20, 90, n),
        "momentum_score": rng.uniform(0, 100, n), "breakout_score": rng.uniform(0, 100, n),
        "improvement_score": rng.uniform(0, 100, n), "dist_to_vstop": rng.normal(3, 6, n),
        "dist_52wh": rng.uniform(0, 40, n), "weinstein_stage": "📈 Stage 2 Advancing",
        "rev_pyq": 100.0, "rev_lq": rng.uniform(90, 140, n), "pat_pyq": 10.0, "pat_lq": rng.uniform(8, 16, n),
        "ebitda_pyq": 15.0, "ebitda_lq": rng.uniform(13, 24, n),
        "pat_1yb": rng.uniform(-5, 50, n), "pat": rng.uniform(-5, 80, n),
        "opm_1yb": rng.uniform(5, 25, n), "opm": rng.uniform(5, 30, n),
        "revenue_1yb": rng.uniform(50, 500, n), "revenue": rng.uniform(50, 650, n),
        "vol_sma_20d": rng.uniform(1e4, 1e6, n), "close_price": rng.uniform(20, 900, n),
        "rf_wc_double_squeeze": 0, "rf_receivables_bloat": 0, "rf_negative_fcf": 0,
        "result_age_days": rng.integers(-5, 80, n),
    }})
    filt = df
    tabs = [st.container(), st.container()]
''')


def _app_source() -> str:
    """The Scans tab exactly as app.py has it: its ui.ui_scans import, _pick_scan, the tab block."""
    src = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    parts = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "ui.ui_scans"]
    parts += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_pick_scan"]
    parts += [n for n in tree.body if isinstance(n, ast.With)
              and ast.unparse(n.items[0].context_expr) == "tabs[1]"]
    assert len(parts) == 3, "app.py's Scans tab is no longer one import, one callback and one block"
    return _HARNESS.format(root=ROOT) + "\n".join(ast.unparse(p) for p in parts)


@pytest.fixture
def tab(tmp_path):
    from streamlit.testing.v1 import AppTest
    script = tmp_path / "scans_tab.py"
    script.write_text(_app_source(), encoding="utf-8")
    at = AppTest.from_file(str(script), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _frame():
    """The harness's synthetic market, rebuilt here to compute the expected lists."""
    ns = {}
    exec(_HARNESS.format(root=ROOT).replace("tabs = [st.container(), st.container()]", ""), ns)
    return ns["df"]


def test_every_card_renders_and_the_compounder_list_opens(tab):
    from ui import ui_scans as S
    html = "".join(m.value for m in tab.markdown)
    for s in S.SCANS:
        assert s["name"] in html, f"{s['name']} card missing"
    assert not tab.toggle, "no scan narrows its list with a switch its record does not describe"
    tab.button(key="scan_open_compounder").click().run()
    assert not tab.exception, [e.value for e in tab.exception]
    expected = S.scan_table(_frame(), next(s for s in S.SCANS if s["key"] == "compounder"))
    assert len(tab.dataframe) == 1 and len(expected) > 0
    assert sorted(tab.dataframe[0].value["name"]) == sorted(expected["name"])


def test_the_cycle_column_shows_on_the_compounder_table_only(tab):
    """'Cycle ↓' (2026-10-07) reaches the screen through app.py's column_order: shown, beside 🔥, on 🌱;
    absent on 🚀 — read from the rendered table, not from the module."""
    first = list(tab.dataframe[0].proto.column_order)
    assert "_cycle" not in first, "the first scan (🚀 Playbook) shows the shared table"
    tab.button(key="scan_open_compounder").click().run()
    assert not tab.exception, [e.value for e in tab.exception]
    order = list(tab.dataframe[0].proto.column_order)
    assert order[order.index("_strong_q") + 1] == "_cycle"
