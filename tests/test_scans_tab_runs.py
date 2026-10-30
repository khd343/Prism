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
record or on scans without measured exits. And the 🚀 ladder (2026-10-30, tests/test_scan_ladder.py): the List
control on 🚀 only, its live counts, B and C listed with their note and no record marks, the 🧹 leaving it alone.
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
        "shariah_status": np.array(["Shariah", "Non-Shariah", "Doubtful"])[np.arange(n) % 3],
    }})
    filt = df
    tabs = [st.container(), st.container()]
''')


def _app_source(records: dict = None) -> str:
    """The Scans tab exactly as app.py has it: its ui.ui_scans import, _pick_scan, the tab block.
    `records` replaces the committed records (a baseline the committed file does not have yet)."""
    src = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom)
               and n.module in ("ui.ui_scans", "ui.ui_discovery", "ui.scan_guide")]
    parts = list(imports)
    parts += [n for n in tree.body if isinstance(n, ast.Assign)
              and ast.unparse(n.targets[0]) in ("_MP_CAP_ORDER", "_SCAN_FILTER_DEFAULTS")]
    parts += [n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name in ("_pick_scan", "_mp_clear_lens", "_mp_ms", "_scan_filter_row")]
    parts += [n for n in tree.body if isinstance(n, ast.With)
              and ast.unparse(n.items[0].context_expr) == "tabs[1]"]
    assert len(parts) == 10, "app.py's Scans tab: three imports, two constants, four functions and one block"
    if records is not None:     # after EVERY import, or the ui_scans import re-binds load_scan_records
        parts.insert(len(imports), ast.parse(
            f"load_scan_records = lambda: __import__('json').loads({json.dumps(records)!r})"))
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


def _control(at, key):
    """The segmented control drawn under `key`, or None when the tab does not draw it — found by its key, never by
    its place: since 2026-10-30 🚀 draws two (List above the filters, Column View above the table)."""
    found = [b for b in at.get("button_group") if b.key == key]
    return found[0] if found else None


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


def _expected(scan, min_cr=None, stages=(), caps=(), zones=(), shariah=None):
    """The names the tab should list: the recipe on the whole market, the liquidity floor, then the
    tab's filters — all empty unless told otherwise (2026-10-13: the user removed the ₹1 Cr a day and
    Stage 1 + 2 defaults the day they shipped). Shariah (2026-10-16) is matched by equality."""
    from ui import ui_scans as S
    df = _frame()
    kept = df[df["company_id"].isin(set(S.scan_table(df, scan, min_cr=min_cr)["company_id"]))]
    if shariah:
        kept = kept[kept["shariah_status"] == shariah]
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
    view = _control(tab, "scan_view")             # by key since 2026-10-30: 🚀 also draws the List control
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


def test_how_it_works_follows_the_chosen_scan(tab):
    """RUN (2026-10-14): the 📖 How it works panel beside 📊 Full record holds the SELECTED scan's guide —
    the same ui/scan_guide.py text the Reference tab's 🎯 Scans section shows."""
    from ui import ui_scans as S

    def panel():
        found = [e for e in tab.expander if e.label.startswith("📖 How it works")]
        assert len(found) == 1, [e.label for e in tab.expander]
        return found[0].label, "".join(m.value for m in found[0].markdown)

    label, body = panel()
    assert S.SCANS[0]["name"] in label and "top third" in body and "top 40%" not in body
    tab.button(key="scan_open_compounder").click().run()
    assert not tab.exception, [e.value for e in tab.exception]
    label, body = panel()
    comp = next(s for s in S.SCANS if s["key"] == "compounder")
    assert comp["name"] in label and "top 40%" in body and "top third" not in body


def test_the_cycle_column_shows_on_the_compounder_table_only(tab):
    """'Cycle ↓' (2026-10-07) reaches the screen through app.py's column_order: shown, beside 🔥, on 🌱;
    absent on 🚀 — read from the rendered table, not from the module."""
    first = list(tab.dataframe[0].proto.column_order)
    assert "_cycle" not in first, "the first scan (🚀 Playbook) shows the shared table"
    tab.button(key="scan_open_compounder").click().run()
    assert not tab.exception, [e.value for e in tab.exception]
    order = list(tab.dataframe[0].proto.column_order)
    assert order[order.index("_strong_q") + 1] == "_cycle"


def test_the_liquidity_selector_shows_the_order_size_guide_and_starts_at_any(tab):
    """The guide reaches the screen through app.py (read from the rendered selector): same steps, default
    Any (a ₹1 Cr default shipped and was removed on 2026-10-13, the user's call), and a floor still
    narrows only the finished list."""
    from ui import ui_scans as S
    comp = next(s for s in S.SCANS if s["key"] == "compounder")
    tab.button(key="scan_open_compounder").click().run()        # the synthetic market's larger list
    sel = tab.selectbox(key="scan_liq")
    assert sel.help == S.liquidity_help()
    assert list(sel.options) == ["Any"] + [f"₹{s:g} Cr/day" for s in S.LIQUIDITY_STEPS if s is not None]
    assert sel.value is None
    assert sel.proto.placeholder == "Any", "None also means 'nothing chosen' to Streamlit: it must read Any"
    sel.set_value(5.0).run()
    assert not tab.exception, [e.value for e in tab.exception]
    expected = _expected(comp, min_cr=5.0)
    assert 0 < len(expected) < len(_expected(comp, min_cr=None)), "the floor must really narrow"
    assert sorted(tab.dataframe[0].value["name"]) == expected


def test_the_list_opens_on_the_whole_scan_and_a_filter_is_named_in_the_caption(tab):
    """RUN (2026-10-13, repointed the same day): every control opens empty, so the table is the whole scan
    and the caption claims no narrowing; once a filter narrows it, the caption names it."""
    from ui import ui_scans as S
    assert tab.selectbox(key="scan_liq").value is None
    for key in ("scan_cap", "scan_stage", "scan_zone"):
        assert tab.multiselect(key=key).value == [], key
    assert sorted(tab.dataframe[0].value["name"]) == _expected(S.SCANS[0])
    cap = next(c.value for c in tab.caption if "sorted by Momentum" in c.value)
    head = cap.split("·")[0]
    assert " of " not in head and "(" not in head, cap
    assert not [b for b in tab.button if b.key == "scan_clear"], "no 🧹 while every control is empty"
    tab.button(key="scan_open_compounder").click().run()      # 🌱 holds Stage 2 stocks (the 🚀 four do not)
    tab.multiselect(key="scan_stage").set_value(["📈 Stage 2 Advancing"]).run()
    cap = next(c.value for c in tab.caption if "sorted by Rank" in c.value)
    assert "(📈 Stage 2)" in cap, cap


def test_the_table_order_follows_the_scan(tab):
    """RUN (2026-10-13, the operator's call): 🚀 opens by momentum, high to low, and its caption says so;
    🌱 lists by Rank and says that (tests/test_scan_order.py)."""
    play = tab.dataframe[0].value
    assert len(play) > 2
    assert list(play["momentum_score"]) == sorted(play["momentum_score"], reverse=True)
    assert any("sorted by Momentum, high to low" in c.value for c in tab.caption)
    tab.button(key="scan_open_compounder").click().run()
    assert not tab.exception, [e.value for e in tab.exception]
    comp = tab.dataframe[0].value
    assert len(comp) > 2 and list(comp["rank"]) == sorted(comp["rank"])
    assert any("sorted by Rank" in c.value for c in tab.caption)
    assert not any("sorted by Momentum" in c.value for c in tab.caption)


def test_the_pills_show_the_tables_short_words_and_filter_by_the_engines_labels(tab):
    """Two full Stage labels wrapped the row onto a second line at 1536px (seen 2026-10-13), so the
    pills show the table's short words — while the VALUES stay the engine's labels, so filtering and
    stored picks are unchanged. On 🌱, which holds both stages (an option a list does not offer cannot be
    picked in the browser either)."""
    tab.button(key="scan_open_compounder").click().run()
    tab.multiselect(key="scan_stage").set_value(["📈 Stage 2 Advancing", "🔄 Stage 1 Basing"]).run()
    stage = tab.multiselect(key="scan_stage")
    assert stage.value == ["📈 Stage 2 Advancing", "🔄 Stage 1 Basing"], "the values are the engine's labels"
    shown = " | ".join(stage.options)
    assert "📈 Stage 2  ·  " in shown and "Advancing" not in shown, shown
    zone = tab.multiselect(key="scan_zone")
    assert any(o.startswith("🟢 Perfect Entry  ·  ") for o in zone.options), zone.options
    assert not any("(Low Risk)" in o for o in zone.options)


def test_each_filter_narrows_the_finished_list(tab):
    """On 🌱, the synthetic market's larger list (25 matches)."""
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


def test_the_shariah_filter_keeps_only_its_status_and_opens_on_all(tab):
    """RUN (2026-10-16, the operator's brief): the Shariah control opens on All — the whole scan, every status —
    with live counts; a pick keeps exactly that status ('Shariah' does not keep 'Non-Shariah': equality, never
    contains), the caption names it, and 🧹 returns it to All."""
    from ui import ui_scans as S
    comp = next(s for s in S.SCANS if s["key"] == "compounder")
    tab.button(key="scan_open_compounder").click().run()
    sel = tab.selectbox(key="scan_shariah")
    assert sel.value == "All"
    assert [o.split("  ·  ")[0] for o in sel.options] == ["All", "✅ Shariah", "⛔ Non-Shariah", "📝 Review",
                                                          "❔ Doubtful"]
    sel.set_value("Shariah").run()
    assert not tab.exception, [e.value for e in tab.exception]
    want = _expected(comp, shariah="Shariah")
    assert 0 < len(want) < len(_expected(comp)), "the pick must really narrow"
    assert sorted(tab.dataframe[0].value["name"]) == want
    cap = next(c.value for c in tab.caption if "sorted by Rank" in c.value)
    assert "(✅ Shariah)" in cap, cap
    clear = [b for b in tab.button if b.key == "scan_clear"]
    assert clear, "🧹 shows once Shariah leaves All"
    clear[0].click().run()
    assert tab.selectbox(key="scan_shariah").value == "All"


def test_clear_returns_every_control_to_its_default(tab):
    """Every control's default is empty (2026-10-13), so 🧹 empties the row — and leaves the view alone."""
    tab.selectbox(key="scan_liq").set_value(5.0).run()
    tab.multiselect(key="scan_stage").set_value(["📉 Stage 4 Declining"]).run()
    tab.multiselect(key="scan_cap").set_value(["Mid Cap"]).run()
    clear = [b for b in tab.button if b.key == "scan_clear"]
    assert clear, "🧹 shows once a control leaves its default"
    clear[0].click().run()
    assert not tab.exception, [e.value for e in tab.exception]
    assert tab.selectbox(key="scan_liq").value is None, "back to Any"
    assert tab.multiselect(key="scan_stage").value == []
    assert tab.multiselect(key="scan_cap").value == [] and tab.multiselect(key="scan_zone").value == []
    assert _control(tab, "scan_view").value == "🏆 Core", "the view is not a filter"


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


# ── The 🚀 ladder (2026-10-30, the operator's call; tests/test_scan_ladder.py) ─────────────────────────────────
def _layer_counts():
    """Each layer's whole-market count on the harness's market."""
    from ui import ui_scans as S
    df = _frame()
    return {layer["key"]: int(S.playbook_layer_mask(df, layer["key"]).sum()) for layer in S.PLAYBOOK_LAYERS}


def _notes(at):
    from ui import ui_scans as S
    notes = {layer["note"] for layer in S.PLAYBOOK_LAYERS if layer["note"]}
    return [c.value for c in at.caption if c.value in notes]


def test_the_list_control_opens_on_the_rocket_with_each_lists_live_count(tab):
    """RUN: on 🚀 the List control shows the three layers with their whole-market counts and opens on 🚀's own list,
    which is the table — with no note, because 🚀 is the scan its record measured."""
    from ui import ui_scans as S
    n = _layer_counts()
    assert 0 < n["playbook"] < n["no_quarter"] < n["price_only"], "the harness market reaches every rung"
    ctl = _control(tab, "scan_layer")
    assert ctl is not None and ctl.proto.label == "List"
    assert ctl.value == "playbook", "🚀 opens on its own list"
    # A label's leading emoji travels as the option's icon (Streamlit's button group), drawn before its words.
    shown = [" ".join(p for p in (o.content_icon, o.content) if p) for o in ctl.options]
    assert shown == [f"{layer['label']} · {n[layer['key']]:,}" for layer in S.PLAYBOOK_LAYERS]
    assert sorted(tab.dataframe[0].value["name"]) == _expected(S.SCANS[0]) and len(_expected(S.SCANS[0])) == n["playbook"]
    assert not _notes(tab)
    # Deselected (AppTest deselects with [], never None), the List falls back to 🚀's own list, as Column View does.
    _control(tab, "scan_layer").set_value("no_quarter").run()
    _control(tab, "scan_layer").set_value([]).run()
    assert not tab.exception, [e.value for e in tab.exception]
    assert _control(tab, "scan_layer").value is None
    assert sorted(tab.dataframe[0].value["name"]) == _expected(S.SCANS[0]) and not _notes(tab)


@pytest.mark.parametrize("key", ["no_quarter", "price_only"])
def test_b_and_c_list_their_stocks_with_one_note_and_no_record_marks(tmp_path, key):
    """RUN, with a 🚀 baseline on file (what the 🆕 marks read): 🚀 marks its new names, while B and C mark none, name
    no baseline and show no Left — they have no record. Their table, its columns and its widget key are the list's;
    the 📊 record and the 📖 guide stay 🚀's."""
    from streamlit.testing.v1 import AppTest
    from ui import ui_scans as S
    df = _frame()
    records = _records_without_baselines()
    records["scans"]["playbook"]["members_at_latest"] = df["company_id"].tolist()[:150]
    records["members_date"] = "2026-09-30"
    script = tmp_path / "scans_tab_ladder.py"
    script.write_text(_app_source(records), encoding="utf-8")
    at = AppTest.from_file(str(script), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("🆕 new since 30 Sep" in c.value for c in at.caption), "the control: 🚀 reads its baseline"
    _control(at, "scan_layer").set_value(key).run()
    assert not at.exception, [e.value for e in at.exception]
    n, layer = _layer_counts()[key], next(lay for lay in S.PLAYBOOK_LAYERS if lay["key"] == key)
    table = at.dataframe[0]
    assert sorted(table.value["name"]) == sorted(df.loc[S.playbook_layer_mask(df, key), "name"]) and len(table.value) == n
    assert list(table.proto.column_order) == list(S.table_columns(S.playbook_layer_scan(S.SCANS[0], key), "🏆 Core"))
    assert f"-scan_tbl_{key}_" in table.proto.id, "each list is its own table widget"
    assert not table.value["_new"].astype(str).str.contains("🆕").any()
    cap = next(c.value for c in at.caption if "sorted by Momentum" in c.value)
    assert cap.startswith(f"{n:,} stocks · ") and "🆕" not in cap, cap
    assert _notes(at) == [layer["note"]], "one note, under the control"
    assert not any(e.label.startswith("Left") for e in at.expander)
    assert not any(c.value.startswith(("Left", "None left")) for c in at.caption)
    panels = [e.label for e in at.expander if e.label.startswith(("📊 Full record", "📖 How it works"))]
    assert len(panels) == 2 and all("🚀 Playbook Shortlist" in p for p in panels), panels
    # Narrowed, the caption counts against the LIST's own matches, never 🚀's.
    at.selectbox(key="scan_liq").set_value(5.0).run()
    assert not at.exception, [e.value for e in at.exception]
    k = len(S.scan_table(df, S.playbook_layer_scan(S.SCANS[0], key), min_cr=5.0))
    assert 0 < k < n and len(at.dataframe[0].value) == k, "the floor must really narrow"
    cap = next(c.value for c in at.caption if "sorted by Momentum" in c.value)
    assert cap.startswith(f"{k:,} of {n:,} stocks (₹5 Cr+ a day) · "), cap


def test_the_list_control_is_the_rockets_alone_and_never_reaches_another_scan(tab):
    """RUN: on 🌱 there is no List control and 🌱's table is its own whatever was chosen on 🚀; back on 🚀 the control
    returns on 🚀's own list (Streamlit drops the state of a control it did not draw: 🚀 reopens on the list its record
    measures) and the table is the list it shows."""
    from ui import ui_scans as S
    _control(tab, "scan_layer").set_value("price_only").run()
    tab.button(key="scan_open_compounder").click().run()
    assert not tab.exception, [e.value for e in tab.exception]
    assert _control(tab, "scan_layer") is None, "the List control is 🚀's alone"
    comp = next(s for s in S.SCANS if s["key"] == "compounder")
    assert sorted(tab.dataframe[0].value["name"]) == _expected(comp)
    assert list(tab.dataframe[0].proto.column_order) == list(S.table_columns(comp, "🏆 Core"))
    assert not _notes(tab)
    tab.button(key="scan_open_playbook").click().run()
    assert not tab.exception, [e.value for e in tab.exception]
    ctl, df = _control(tab, "scan_layer"), _frame()
    assert ctl is not None
    assert ctl.value == "playbook", "leaving 🚀 resets the list; back on 🚀 it reopens on 🚀's own"
    assert sorted(tab.dataframe[0].value["name"]) == sorted(df.loc[S.playbook_layer_mask(df, ctl.value), "name"])


def test_the_clear_leaves_the_list_alone(tab):
    """The list is a choice, like the scan: 🧹 empties the filter row and keeps it."""
    _control(tab, "scan_layer").set_value("price_only").run()
    tab.selectbox(key="scan_liq").set_value(5.0).run()
    clear = [b for b in tab.button if b.key == "scan_clear"]
    assert clear, "🧹 shows once a control leaves its default"
    clear[0].click().run()
    assert not tab.exception, [e.value for e in tab.exception]
    assert tab.selectbox(key="scan_liq").value is None
    assert _control(tab, "scan_layer").value == "price_only"

