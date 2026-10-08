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
its recipe, not a switch), and picking 🌱 shows exactly the recipe's matches; the Liquidity guide; and
"Left 🌱" — who left since the baseline and why, a note while there is no baseline, nothing on a stale
record or on scans without measured exits.
"""
import ast
import json
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


def _app_source(records: dict = None) -> str:
    """The Scans tab exactly as app.py has it: its ui.ui_scans import, _pick_scan, the tab block.
    `records` replaces the committed records (a baseline the committed file does not have yet)."""
    src = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    parts = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "ui.ui_scans"]
    parts += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_pick_scan"]
    parts += [n for n in tree.body if isinstance(n, ast.With)
              and ast.unparse(n.items[0].context_expr) == "tabs[1]"]
    assert len(parts) == 3, "app.py's Scans tab is no longer one import, one callback and one block"
    if records is not None:
        parts.insert(1, ast.parse(f"load_scan_records = lambda: __import__('json').loads({json.dumps(records)!r})"))
    return _HARNESS.format(root=ROOT) + "\n".join(ast.unparse(p) for p in parts)


def _records_without_baselines() -> dict:
    """The committed records with every 🆕 baseline removed. The weekly job rewrites the baselines with
    REAL company ids, which a synthetic market does not hold, so a test that read them as committed
    would change with every job run (2026-10-08: the first 🌱 baseline put a 'Left 🌱' table of 254
    'gone' stocks under this market). Tests about baselines inject their own (the Left 🌱 tests below)."""
    from ui import ui_scans as S
    records = json.loads(json.dumps(S.load_records()))
    for rec in records["scans"].values():
        rec.pop("members_at_latest", None)
    return records


@pytest.fixture
def tab(tmp_path):
    from streamlit.testing.v1 import AppTest
    script = tmp_path / "scans_tab.py"
    script.write_text(_app_source(_records_without_baselines()), encoding="utf-8")
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


def test_the_liquidity_selector_shows_the_order_size_guide_and_still_starts_at_any(tab):
    """The guide reaches the screen through app.py (read from the rendered selector), and it changes
    nothing else: same steps, default Any, and a floor still narrows only the finished list."""
    from ui import ui_scans as S
    sel = tab.selectbox(key="scan_liq")
    assert sel.help == S.liquidity_help()
    assert list(sel.options) == ["Any"] + [f"₹{s:g} Cr/day" for s in S.LIQUIDITY_STEPS if s is not None]
    assert sel.value is None
    sel.set_value(5.0).run()
    assert not tab.exception, [e.value for e in tab.exception]
    expected = S.scan_table(_frame(), S.SCANS[0], min_cr=5.0)
    assert 0 < len(expected) < len(S.scan_table(_frame(), S.SCANS[0])), "the floor must really narrow"
    assert sorted(tab.dataframe[0].value["name"]) == sorted(expected["name"])


def _run(tmp_path, records):
    from streamlit.testing.v1 import AppTest
    script = tmp_path / "scans_tab_left.py"
    script.write_text(_app_source(records), encoding="utf-8")
    at = AppTest.from_file(str(script), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.button(key="scan_open_compounder").click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_left_seedling_shows_who_left_since_the_baseline_and_why(tmp_path):
    """The exit side, RUN: a baseline holding today's 🌱 stocks plus three that are not 🌱 now and one
    gone from the data -> one collapsed 'Left 🌱 since 30 Sep: 4' line whose table is left_scan's."""
    from ui import ui_scans as S
    df = _frame()
    seedling = next(s for s in S.SCANS if s["key"] == "compounder")
    now = df.loc[seedling["mask"](df), "company_id"].tolist()
    out = df.loc[~seedling["mask"](df), "company_id"].tolist()[:3]
    records = json.loads(json.dumps(S.load_records()))
    records["scans"]["compounder"]["members_at_latest"] = now + out + ["NSE:GONE"]
    records["members_date"] = "2026-09-30"
    at = _run(tmp_path, records)
    expected = S.left_scan(df, seedling, now + out + ["NSE:GONE"])
    assert len(expected) == 4
    label = next(e.label for e in at.expander if e.label.startswith("Left 🌱"))
    assert label == f"Left 🌱 since 30 Sep: 4 — {S.left_summary(expected)}"
    left_tbl = at.dataframe[1].value
    assert left_tbl["company_id"].tolist() == expected["company_id"].tolist()
    assert list(at.dataframe[1].proto.column_order) == ["name", "_why", "sector", "_sales_chg", "_profit_chg"]


def test_left_seedling_waits_for_a_baseline_and_says_so(tmp_path):
    """Before the weekly job has saved a 🌱 baseline: one caption, no empty table."""
    from ui import ui_scans as S
    records = json.loads(json.dumps(S.load_records()))
    records["scans"]["compounder"].pop("members_at_latest", None)
    at = _run(tmp_path, records)
    assert not any(e.label.startswith("Left") for e in at.expander)
    assert any("shown once the weekly record has saved a month-end list" in c.value for c in at.caption)


def test_left_seedling_says_none_when_nobody_left(tmp_path):
    from ui import ui_scans as S
    df = _frame()
    seedling = next(s for s in S.SCANS if s["key"] == "compounder")
    records = json.loads(json.dumps(S.load_records()))
    records["scans"]["compounder"]["members_at_latest"] = df.loc[seedling["mask"](df), "company_id"].tolist()
    records["members_date"] = "2026-09-30"
    at = _run(tmp_path, records)
    assert any(c.value == "None left 🌱 since 30 Sep." for c in at.caption)


def test_no_left_line_on_scans_without_measured_exits(tmp_path):
    """🚀 and 🌬️ have no measured exit reasons, so even with a baseline they show no Left line."""
    from streamlit.testing.v1 import AppTest
    from ui import ui_scans as S
    records = json.loads(json.dumps(S.load_records()))
    for key in ("playbook", "leaders"):
        records["scans"][key]["members_at_latest"] = ["NSE:T000", "NSE:GONE"]
    records["members_date"] = "2026-09-30"
    script = tmp_path / "scans_tab_other.py"
    script.write_text(_app_source(records), encoding="utf-8")
    at = AppTest.from_file(str(script), default_timeout=60)
    at.run()
    for key in ("playbook", "leaders"):
        at.button(key=f"scan_open_{key}").click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert not any(e.label.startswith("Left") for e in at.expander), key
        assert not any(c.value.startswith(("Left", "None left")) for c in at.caption), key


def test_no_left_line_when_the_record_is_stale(tmp_path):
    """A changed recipe makes the record stale: its baseline describes a different scan, so no exits
    and no 'waiting' note either — the card already says the record is stale."""
    from ui import ui_scans as S
    records = json.loads(json.dumps(S.load_records()))
    records["scans"]["compounder"]["fingerprint"] = "0000000000000000"
    records["scans"]["compounder"]["members_at_latest"] = ["NSE:T000", "NSE:GONE"]
    records["members_date"] = "2026-09-30"
    at = _run(tmp_path, records)
    assert not any(e.label.startswith("Left") for e in at.expander)
    assert not any(c.value.startswith(("Left", "None left")) for c in at.caption)

