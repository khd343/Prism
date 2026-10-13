"""
test_scans_tab_runs.py
======================
The 🎯 Scans tab, RUN — not read (2026-10-07).

Every other Scans test reads app.py's tab block as source (AST). This one executes it: the real
`with tabs[1]:` block and `_pick_scan` are lifted out of app.py and run under Streamlit's AppTest on a
synthetic market, so a wrong name, a widget that raises, or a switch that does not reach the table
fails here instead of on the live app. Only the parts that need the full app are stubbed: the data
load (`df` / `filt`) and the freshness probe. The tab's own filter row (2026-10-13: Liquidity · Market cap ·
Weinstein Stage · Buy Zone) runs for real — its helper, the shared multiselect and the 🧹 Clear are lifted
from app.py with the rest.

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

    _fresh = SimpleNamespace(is_known=False, data_date=None)

    rng = np.random.default_rng(7)
    n = 300
    df = pd.DataFrame({{
        "company_id": [f"NSE:T{{k:03d}}" for k in range(n)], "name": [f"Test {{k:03d}}" for k in range(n)],
        "sector": rng.choice(["Alpha", "Beta", "Gamma", "Delta", "Eps", "Zeta", "Eta"], n),
        "rank": rng.permutation(n) + 1, "composite_score": rng.uniform(20, 90, n),
        "momentum_score": rng.uniform(0, 100, n), "breakout_score": rng.uniform(0, 100, n),
        "improvement_score": rng.uniform(0, 100, n), "dist_to_vstop": rng.normal(3, 6, n),
        "dist_52wh": rng.uniform(0, 40, n),
        "weinstein_stage": np.array(["📈 Stage 2 Advancing", "🔄 Stage 1 Basing", "⚠️ Stage 3 Top",
                                     "📉 Stage 4 Declining", "❔ Unknown"])[np.arange(n) % 5],
        "market_category": np.array(["Large Cap", "Mid Cap", "Small Cap", "Micro Cap"])[(np.arange(n) // 3) % 4],
        "buy_zone_label": np.array(["🟢 Perfect Entry (Low Risk)", "🟡 Standard Zone", "🟠 Loose Entry Zone",
                                    "🔴 Extended (Wait for Pullback)"])[(np.arange(n) // 2) % 4],
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
    parts = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module in ("ui.ui_scans", "ui.ui_discovery")]
    parts += [n for n in tree.body if isinstance(n, ast.Assign)
              and ast.unparse(n.targets[0]) in ("_MP_CAP_ORDER", "_SCAN_FILTER_DEFAULTS")]
    parts += [n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name in ("_pick_scan", "_mp_clear_lens", "_mp_ms", "_scan_filter_row")]
    parts += [n for n in tree.body if isinstance(n, ast.With)
              and ast.unparse(n.items[0].context_expr) == "tabs[1]"]
    assert len(parts) == 9, "app.py's Scans tab: two imports, two constants, four functions and one block"
    if records is not None:     # after BOTH imports, or the ui_scans import re-binds load_scan_records
        parts.insert(2, ast.parse(f"load_scan_records = lambda: __import__('json').loads({json.dumps(records)!r})"))
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


# AppTest (Streamlit 1.54) cannot serialise a SINGLE-choice segmented control (2026-10-13, when the
# Scans table gained its 🏆 Core / 📈 Technical switch): ButtonGroup.indices loops `for v in self.value`,
# and for single selection the value is the option itself, a str — so it walks "🏆 Core" letter by
# letter and fails on the first, "🏆". The browser is unaffected (checked 2026-10-13 at 1536px: both
# views on all three scans, a row click loads the Tear-Sheet). The harness wraps a single value in a
# list, so the real widget still runs; the tripwire below fails the day Streamlit fixes it, and then
# this goes.
from streamlit.testing.v1 import element_tree as _et  # noqa: E402

_ORIG_INDICES = _et.ButtonGroup.indices


def _indices(self):
    v = self.value
    vals = v if isinstance(v, (list, tuple)) else ([] if v is None else [v])
    return [self.options.index(self.format_func(x)) for x in vals]


@pytest.fixture(autouse=True)
def _single_choice_button_groups(monkeypatch):
    monkeypatch.setattr(_et.ButtonGroup, "indices", property(_indices))


def test_apptest_still_needs_the_single_choice_workaround(tmp_path):
    """TRIPWIRE: when this fails, Streamlit's own ButtonGroup.indices handles a single-choice value —
    delete _indices and the autouse fixture above (and this test)."""
    from streamlit.testing.v1 import AppTest
    script = tmp_path / "seg.py"
    script.write_text('import streamlit as st\n'
                      'st.segmented_control("View", ["🏆 Core", "📈 Technical"], default="🏆 Core", key="v")\n',
                      encoding="utf-8")
    at = AppTest.from_file(str(script), default_timeout=30)
    at.run()
    el = at.get("button_group")[0]
    with pytest.raises(ValueError):
        _ORIG_INDICES.fget(el)
    assert _indices(el) == [0], "the workaround reads the chosen option"


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


DEFAULT_STAGES = ("📈 Stage 2 Advancing", "🔄 Stage 1 Basing")


def _expected(scan, min_cr=1.0, stages=DEFAULT_STAGES, caps=(), zones=()):
    """The names the tab should list: the recipe on the whole market, the liquidity floor, then the
    tab's filters — at their defaults (2026-10-13: ₹1 Cr a day, Stage 1 + 2) unless told otherwise."""
    from ui import ui_scans as S
    df = _frame()
    kept = df[df["company_id"].isin(set(S.scan_table(df, scan, min_cr=min_cr)["company_id"]))]
    for col, picks in (("weinstein_stage", stages), ("market_category", caps), ("buy_zone_label", zones)):
        if picks:
            kept = kept[kept[col].isin(picks)]
    return sorted(kept["name"])


def test_every_card_renders_and_the_compounder_list_opens(tab):
    from ui import ui_scans as S
    html = "".join(m.value for m in tab.markdown)
    for s in S.SCANS:
        assert s["name"] in html, f"{s['name']} card missing"
    assert not tab.toggle, "no scan narrows its list with a switch its record does not describe"
    tab.button(key="scan_open_compounder").click().run()
    assert not tab.exception, [e.value for e in tab.exception]
    expected = _expected(next(s for s in S.SCANS if s["key"] == "compounder"))
    assert len(tab.dataframe) == 1 and len(expected) > 0
    assert sorted(tab.dataframe[0].value["name"]) == expected


def test_the_view_switch_reaches_the_table(tab):
    """RUN (2026-10-13): the table opens on 🏆 Core; choosing 📈 Technical shows the Technical columns
    on the same stocks — the switch reaches the table, not just the source (tests/test_scan_views.py)."""
    from ui import ui_scans as S
    play = S.SCANS[0]
    view = tab.get("button_group")[0]
    assert view.value == "🏆 Core"
    assert list(tab.dataframe[0].proto.column_order) == list(S.table_columns(play, "🏆 Core"))
    core_names = list(tab.dataframe[0].value["name"])
    view.set_value("📈 Technical").run()
    assert not tab.exception, [e.value for e in tab.exception]
    assert list(tab.dataframe[0].proto.column_order) == list(S.table_columns(play, "📈 Technical"))
    assert list(tab.dataframe[0].value["name"]) == core_names, "a view changes columns, never the list"


def test_the_full_record_panel_follows_the_chosen_scan(tab):
    """RUN: the one 'Full record' panel under the cards names the selected scan and holds its record —
    the windows for 🚀, the study for 🌱 (2026-10-08)."""
    from ui import ui_scans as S
    panels = [e for e in tab.expander if e.label.startswith("📊 Full record")]
    assert len(panels) == 1 and S.SCANS[0]["name"] in panels[0].label
    assert "DESIGN-PERIOD RECORD" in "".join(m.value for m in panels[0].markdown)
    tab.button(key="scan_open_compounder").click().run()
    assert not tab.exception, [e.value for e in tab.exception]
    panels = [e for e in tab.expander if e.label.startswith("📊 Full record")]
    seed = next(s for s in S.SCANS if s["key"] == "compounder")
    assert len(panels) == 1 and seed["name"] in panels[0].label
    assert "BACKTEST · IN-SAMPLE" in "".join(m.value for m in panels[0].markdown)


def test_the_cycle_column_shows_on_the_compounder_table_only(tab):
    """'Cycle ↓' (2026-10-07) reaches the screen through app.py's column_order: shown, beside 🔥, on 🌱;
    absent on 🚀 — read from the rendered table, not from the module."""
    first = list(tab.dataframe[0].proto.column_order)
    assert "_cycle" not in first, "the first scan (🚀 Playbook) shows the shared table"
    tab.button(key="scan_open_compounder").click().run()
    assert not tab.exception, [e.value for e in tab.exception]
    order = list(tab.dataframe[0].proto.column_order)
    assert order[order.index("_strong_q") + 1] == "_cycle"


def test_the_liquidity_selector_shows_the_order_size_guide_and_starts_at_one_crore(tab):
    """The guide reaches the screen through app.py (read from the rendered selector): same steps, and —
    the user's default since 2026-10-13 — ₹1 Cr a day (it started at Any before); a floor still narrows
    only the finished list."""
    from ui import ui_scans as S
    comp = next(s for s in S.SCANS if s["key"] == "compounder")
    tab.button(key="scan_open_compounder").click().run()        # the synthetic market's larger list
    sel = tab.selectbox(key="scan_liq")
    assert sel.help == S.liquidity_help()
    assert list(sel.options) == ["Any"] + [f"₹{s:g} Cr/day" for s in S.LIQUIDITY_STEPS if s is not None]
    assert sel.value == 1.0
    sel.set_value(5.0).run()
    assert not tab.exception, [e.value for e in tab.exception]
    expected = _expected(comp, min_cr=5.0)
    assert 0 < len(expected) < len(_expected(comp, min_cr=None)), "the floor must really narrow"
    assert sorted(tab.dataframe[0].value["name"]) == expected


def test_the_list_opens_on_one_crore_and_stage_1_and_2_and_says_so(tab):
    """RUN (2026-10-13): the user's defaults reach the table — liquid stocks in Stage 1 or 2 — and the
    caption names both, so a narrowed list never reads as the whole scan."""
    from ui import ui_scans as S
    assert tab.multiselect(key="scan_stage").value == list(DEFAULT_STAGES)
    assert tab.multiselect(key="scan_cap").value == [] and tab.multiselect(key="scan_zone").value == []
    shown = sorted(tab.dataframe[0].value["name"])
    assert shown == _expected(S.SCANS[0]) and 0 < len(shown) < len(_expected(S.SCANS[0], min_cr=None, stages=()))
    cap = next(c.value for c in tab.caption if "sorted by Rank" in c.value)
    assert "₹1 Cr+ a day" in cap and "Stage 2" in cap and "Stage 1" in cap, cap
    assert not [b for b in tab.button if b.key == "scan_clear"], "no 🧹 while every control is at its default"


def test_the_pills_show_the_tables_short_words_and_filter_by_the_engines_labels(tab):
    """Two full Stage labels wrapped the row onto a second line at 1536px (seen 2026-10-13), so the
    pills show the table's short words — while the VALUES stay the engine's labels, so filtering,
    stored picks and the 🧹 defaults are unchanged."""
    stage = tab.multiselect(key="scan_stage")
    assert stage.value == list(DEFAULT_STAGES), "the values are the engine's own labels"
    shown = " | ".join(stage.options)
    assert "📈 Stage 2  ·  " in shown and "Advancing" not in shown, shown
    tab.button(key="scan_open_compounder").click().run()
    zone = tab.multiselect(key="scan_zone")
    assert any(o.startswith("🟢 Perfect Entry  ·  ") for o in zone.options), zone.options
    assert not any("(Low Risk)" in o for o in zone.options)


def test_each_filter_narrows_the_finished_list(tab):
    """On 🌱, the synthetic market's larger list (12 at the defaults, 25 across every stage)."""
    from ui import ui_scans as S
    play = next(s for s in S.SCANS if s["key"] == "compounder")
    tab.button(key="scan_open_compounder").click().run()
    tab.multiselect(key="scan_stage").set_value(["⚠️ Stage 3 Top"]).run()
    assert sorted(tab.dataframe[0].value["name"]) == _expected(play, stages=("⚠️ Stage 3 Top",))
    tab.multiselect(key="scan_cap").set_value(["Small Cap"]).run()
    assert sorted(tab.dataframe[0].value["name"]) == _expected(play, stages=("⚠️ Stage 3 Top",), caps=("Small Cap",))
    tab.multiselect(key="scan_stage").set_value([]).run()
    tab.multiselect(key="scan_zone").set_value(["🟢 Perfect Entry (Low Risk)"]).run()
    assert not tab.exception, [e.value for e in tab.exception]
    want = _expected(play, stages=(), caps=("Small Cap",), zones=("🟢 Perfect Entry (Low Risk)",))
    assert sorted(tab.dataframe[0].value["name"]) == want and len(want) > 0


def test_clear_returns_every_control_to_its_default(tab):
    tab.selectbox(key="scan_liq").set_value(None).run()
    tab.multiselect(key="scan_stage").set_value(["📉 Stage 4 Declining"]).run()
    tab.multiselect(key="scan_cap").set_value(["Mid Cap"]).run()
    clear = [b for b in tab.button if b.key == "scan_clear"]
    assert clear, "🧹 shows once a control leaves its default"
    clear[0].click().run()
    assert not tab.exception, [e.value for e in tab.exception]
    assert tab.selectbox(key="scan_liq").value == 1.0, "back to ₹1 Cr a day, not to Any"
    assert tab.multiselect(key="scan_stage").value == list(DEFAULT_STAGES)
    assert tab.multiselect(key="scan_cap").value == [] and tab.multiselect(key="scan_zone").value == []
    assert tab.get("button_group")[0].value == "🏆 Core", "the view is not a filter"


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

