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
And Exchange (2026-11-03), right of Shariah: the market holds BSE ids, so a pick narrows the list, never the cards.
"""
import ast
import html
import json
import os
import re
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
        "company_id": [f"{{'BSE' if k % 5 == 2 else 'NSE'}}:T{{k:03d}}" for k in range(n)],   # 60 on BSE (2026-11-03)
        "name": [f"Test {{k:03d}}" for k in range(n)],
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
    'gone' stocks under this market). Tests about baselines inject their own (the Left 🌱 tests below).
    Its unseen windows go too, for the same reason: the job adds them from January 2027."""
    from ui import ui_scans as S
    records = json.loads(json.dumps(S.load_records()))
    for rec in records["scans"].values():
        rec.pop("members_at_latest", None)
        rec["out_of_sample"] = []
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


def _expected(scan, min_cr=None, stages=(), caps=(), zones=(), shariah=None, exchange=None):
    """The names the tab should list: the recipe on the whole market, the liquidity floor, then the
    tab's filters — all empty unless told otherwise (2026-10-13: the user removed the ₹1 Cr a day and
    Stage 1 + 2 defaults the day they shipped). Shariah (2026-10-16) is matched by equality, Exchange
    (2026-11-03) by the company id's prefix."""
    from ui import ui_scans as S
    df = _frame()
    kept = df[df["company_id"].isin(set(S.scan_table(df, scan, min_cr=min_cr)["company_id"]))]
    if shariah:
        kept = kept[kept["shariah_status"] == shariah]
    if exchange:
        kept = kept[kept["company_id"].str.startswith(f"{exchange}:")]
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
    """RUN (2026-10-13): choosing 🏆 Core, then 📈 Technical, shows each one's columns on the same stocks — the switch reaches
    the table, not just the source (tests/test_scan_views.py). Every list opens on 🔢 RAM since 2026-11-05."""
    from ui import ui_scans as S
    play = S.SCANS[1]
    tab.button(key=f"scan_open_{play['key']}").click().run()
    opened = list(tab.dataframe[0].value["name"])
    _control(tab, "scan_view").set_value("🏆 Core").run()   # by key since 2026-10-30: 🚀 also draws the List control
    assert not tab.exception, [e.value for e in tab.exception]
    assert list(tab.dataframe[0].proto.column_order) == list(S.table_columns(play, "🏆 Core"))
    core_names = list(tab.dataframe[0].value["name"])
    assert sorted(core_names) == sorted(opened), "the view holds its own order, never its own list"
    _control(tab, "scan_view").set_value("📈 Technical").run()
    assert not tab.exception, [e.value for e in tab.exception]
    assert list(tab.dataframe[0].proto.column_order) == list(S.table_columns(play, "📈 Technical"))
    assert list(tab.dataframe[0].value["name"]) == core_names, "a view changes columns, never the list"


def test_the_full_record_panel_follows_the_chosen_scan(tab):
    """RUN: the one 'Full record' panel under the cards names the selected scan and holds its record —
    before any window, 🚀's own first date (2026-11-05); the study for 🌱 (2026-10-08), which its label names."""
    from ui import ui_scans as S
    panels = [e for e in tab.expander if e.label.startswith("📊 Full record")]
    assert len(panels) == 1 and S.SCANS[0]["name"] in panels[0].label
    assert panels[0].label.endswith(": every window, after costs, and what was left out")
    assert "No record yet. The first window starts on the 30 Nov copy." in "".join(m.value for m in panels[0].markdown)
    tab.button(key="scan_open_compounder").click().run()
    assert not tab.exception, [e.value for e in tab.exception]
    panels = [e for e in tab.expander if e.label.startswith("📊 Full record")]
    seed = next(s for s in S.SCANS if s["key"] == "compounder")
    assert len(panels) == 1 and seed["name"] in panels[0].label
    assert panels[0].label.endswith(": every window or year, after costs, what was left out and the caveats")
    assert "BACKTEST · IN-SAMPLE" in "".join(m.value for m in panels[0].markdown)


def test_the_short_lists_say_no_record_yet_and_only_the_compounder_names_an_old_record(tab):
    """RUN (2026-11-05): 🚀's and 🌬️'s cards say they have no record and when their first window starts, and only 🌱's
    calls a record in-sample, its study; nothing leaks a blank as nan or draws an empty record table. The line above the
    cards says what in-sample means, and no longer that every record is of windows nobody saw."""
    from ui import ui_scans as S
    page = "".join(m.value for m in tab.markdown)
    for scan in S.SCANS:
        if scan["key"] not in S.STUDY_SCANS:
            assert S.no_record_words(scan) in page, scan["key"]
    assert "No record yet" in page and S.no_record_words(next(s for s in S.SCANS if s["key"] == "compounder")) not in page
    assert page.count("IN-SAMPLE · 3-YEAR BACKTEST · BEFORE COSTS") == 1 and page.count("IN-SAMPLE") == 1, "🌱's summary alone"
    assert "in-sample means studied while choosing the recipe" in page and "of windows nobody saw" not in page
    for word in ("In-sample", "DESIGN-PERIOD", "backtest"):
        assert word not in page, word
    assert not re.search(r"\bnan\b", page, re.I)
    record = "".join(m.value for e in tab.expander if e.label.startswith("📊") for m in e.markdown)
    assert "<table" not in record, "the guide beside it has its own table of ways out; a record with no window has none"


def _kept_fields_only(members=None) -> dict:
    """A record file holding only what stays since 2026-11-05: the fingerprints, the unseen windows and the 🆕 baseline."""
    from ui import ui_scans as S
    return {"version": S.RECORD_VERSION, "designed_on": S.DESIGNED_ON, "members_date": "2026-10-30",
            "scans": {s["key"]: {"fingerprint": S.recipe_fingerprint(s), "windows": [], "out_of_sample": [],
                                 "members_at_latest": members(s) if members else []} for s in S.SCANS}}


def _scans_tab(tmp_path, records):
    from streamlit.testing.v1 import AppTest
    script = tmp_path / "scans_tab_kept.py"
    script.write_text(_app_source(records), encoding="utf-8")
    at = AppTest.from_file(str(script), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("key", ["playbook", "leaders", "compounder"])
def test_the_tab_runs_on_a_file_with_only_the_kept_fields(tmp_path, key):
    """RUN: every scan's list, panels and marks, on the file as it now is: a stock on the baseline is not new and a stock off
    it is, so the baseline still does its one job, and no record is needed for a list to show."""
    from ui import ui_scans as S
    df = _frame()
    scan = next(s for s in S.SCANS if s["key"] == key)
    matches = sorted(df.loc[scan["mask"](df), "company_id"].astype(str))
    assert len(matches) > 2, "the harness market reaches every scan"
    at = _scans_tab(tmp_path, _kept_fields_only(lambda s: matches[1:] if s["key"] == key else []))
    at.button(key=f"scan_open_{key}").click().run()
    assert not at.exception, [e.value for e in at.exception]
    if key == "playbook":                                        # 🚀 opens on 🔢 RAM, which has no New column
        _control(at, "scan_view").set_value("🏆 Core").run()
        assert not at.exception, [e.value for e in at.exception]
    table = at.dataframe[0].value
    new = dict(zip(table["company_id"], table["_new"]))
    assert set(new) == set(matches), "the scan lists every match, record or none"
    assert "🆕" in new[matches[0]] and "🆕" not in new[matches[1]]
    panels = {e.label[0]: "".join(m.value for m in e.markdown) for e in at.expander if e.label.startswith(("📊", "📖"))}
    assert S.no_record_words(scan) in panels["📊"] and "None has finished yet" in panels["📖"]


def test_the_first_window_on_file_reaches_the_tab(tmp_path):
    """RUN: what the job writes reaches both cards with no design-period record beside it."""
    from ui import ui_scans as S
    play = S.SCANS[0]
    records = _kept_fields_only()
    records["scans"]["playbook"]["out_of_sample"] = [
        dict(start="2026-11-30", end="2026-12-31", days=31, n=120, median_excess_pp=1.4, beat_pct=55, final=True,
             fingerprint=S.recipe_fingerprint(play))]
    at = _scans_tab(tmp_path, records)
    page = html.unescape("".join(m.value for m in at.markdown))
    assert "ahead in 1 of 1 unseen window: +1.4 pts before costs" in page and "NEVER SAW" in page
    assert page.count(S.no_record_words(S.SCANS[1])) == 1, "🌬️'s card still has no record; 🚀's has one now"
    assert page.count(S.no_record_words(S.SCANS[2])) == 1, "and 🌱's, in this file with no study, says the same on its own date"


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
    """'Cycle ↓' (2026-10-07) reaches the screen through app.py's column_order: shown, beside 🔥, on 🌱's 🏆 Core;
    absent on 🚀 — read from the rendered table, not from the module. Every list opens on 🔢 RAM since 2026-11-05."""
    _control(tab, "scan_view").set_value("🏆 Core").run()
    first = list(tab.dataframe[0].proto.column_order)
    assert "_cycle" not in first, "the first scan (🚀 Playbook) shows the shared table"
    tab.button(key="scan_open_compounder").click().run()
    assert not tab.exception, [e.value for e in tab.exception]
    order = list(tab.dataframe[0].proto.column_order)
    assert order[order.index("_strong_q") + 1] == "_cycle"
    _control(tab, "scan_view").set_value("🔢 RAM").run()
    assert "_cycle" not in tab.dataframe[0].proto.column_order, "the view reads the same on every list"


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


# ── Exchange (2026-11-03, the operator's brief): right of Shariah, NSE or BSE from the company id ──────────────
def _card_counts(at):
    """Each scan card's count, as the card shows it ('<n> now')."""
    from ui import ui_scans as S
    out = {}
    for s in S.SCANS:
        card = next(m.value for m in at.markdown if f'{s["emoji"]} {html.escape(s["name"])}</span>' in m.value)
        out[s["key"]] = int(re.search(r">([\d,]+)<span[^>]*> now</span>", card).group(1).replace(",", ""))
    return out


def test_the_exchange_filter_opens_on_all_and_keeps_only_its_exchange(tab):
    """RUN: Exchange opens on All — the whole scan — with each option's live count in Shariah's form and the sidebar's
    help; a pick keeps exactly the ids with its prefix, the caption names it in plain words, and 🧹 returns it to All."""
    from ui import ui_scans as S
    from ui.ui_discovery import _EXCHANGE_HELP
    comp = next(s for s in S.SCANS if s["key"] == "compounder")
    tab.button(key="scan_open_compounder").click().run()
    every, nse, bse = _expected(comp), _expected(comp, exchange="NSE"), _expected(comp, exchange="BSE")
    assert nse and bse and len(nse) + len(bse) == len(every), "the list holds both exchanges"
    sel = tab.selectbox(key="scan_exchange")
    assert sel.value == "All" and sel.label == "Exchange" and sel.help == _EXCHANGE_HELP
    assert list(sel.options) == [f"All  ·  {len(every)}", f"NSE  ·  {len(nse)}", f"BSE  ·  {len(bse)}"]
    assert not [b for b in tab.button if b.key == "scan_clear"], "All is no filter"
    sel.set_value("NSE").run()
    assert not tab.exception, [e.value for e in tab.exception]
    assert sorted(tab.dataframe[0].value["name"]) == nse
    cap = next(c.value for c in tab.caption if "sorted by Rank" in c.value)
    assert cap.startswith(f"{len(nse)} of {len(every)} stocks (NSE only) · "), cap
    tab.selectbox(key="scan_exchange").set_value("BSE").run()
    assert sorted(tab.dataframe[0].value["name"]) == bse
    clear = [b for b in tab.button if b.key == "scan_clear"]
    assert clear, "🧹 shows once Exchange leaves All"
    clear[0].click().run()
    assert not tab.exception, [e.value for e in tab.exception]
    assert tab.selectbox(key="scan_exchange").value == "All"
    assert sorted(tab.dataframe[0].value["name"]) == every


def test_the_exchange_narrows_the_list_never_the_cards_or_the_recipe(tab):
    """The recipe ranks the WHOLE market (the code's own rule). With Exchange on NSE the table is the whole market's
    matches that are on NSE — not the recipe re-run on NSE's stocks alone, which lists other names in this market — and
    the cards and 🚀's List keep their whole-market counts, which the caption counts against."""
    from ui import ui_scans as S
    df = _frame()
    nse = df["company_id"].str.startswith("NSE:")
    whole, layers = {s["key"]: int(s["mask"](df).sum()) for s in S.SCANS}, _layer_counts()
    assert whole != {s["key"]: int((s["mask"](df) & nse).sum()) for s in S.SCANS}, "teeth: NSE alone counts less"
    assert layers != {k: int((S.playbook_layer_mask(df, k) & nse).sum()) for k in layers}, "teeth, for the List"
    tab.selectbox(key="scan_exchange").set_value("NSE").run()
    assert not tab.exception, [e.value for e in tab.exception]
    assert _card_counts(tab) == whole, "a filter on the list never reaches the cards"
    shown = [" ".join(p for p in (o.content_icon, o.content) if p) for o in _control(tab, "scan_layer").options]
    assert shown == [f"{layer['label']} · {layers[layer['key']]:,}" for layer in S.PLAYBOOK_LAYERS], "nor 🚀's List"
    comp = next(s for s in S.SCANS if s["key"] == "compounder")
    tab.button(key="scan_open_compounder").click().run()
    assert not tab.exception, [e.value for e in tab.exception]
    want = _expected(comp, exchange="NSE")
    resliced = sorted(df.loc[df["company_id"].isin(set(S.scan_table(df[nse], comp)["company_id"])), "name"])
    assert resliced != want, "teeth: the recipe re-run on NSE alone lists other stocks in this market"
    assert sorted(tab.dataframe[0].value["name"]) == want
    assert _card_counts(tab) == whole
    cap = next(c.value for c in tab.caption if "sorted by Rank" in c.value)
    assert cap.startswith(f"{len(want)} of {whole['compounder']} stocks (NSE only) · "), cap


def test_the_exchange_counts_what_shariah_kept_and_shariah_counts_the_whole_list(tab):
    """The cascade (2026-11-03): Shariah → Exchange → Market cap → Stage → Buy Zone. Exchange's counts are the stocks
    Shariah kept; Shariah, before it, still counts the whole list whatever Exchange holds."""
    from ui import ui_scans as S
    comp = next(s for s in S.SCANS if s["key"] == "compounder")
    tab.button(key="scan_open_compounder").click().run()
    tab.selectbox(key="scan_shariah").set_value("Shariah").run()
    kept = _expected(comp, shariah="Shariah")
    nse, bse = _expected(comp, shariah="Shariah", exchange="NSE"), _expected(comp, shariah="Shariah", exchange="BSE")
    assert 0 < len(bse) < len(_expected(comp, exchange="BSE")), "teeth: Shariah really narrows the BSE stocks"
    assert list(tab.selectbox(key="scan_exchange").options) == [f"All  ·  {len(kept)}", f"NSE  ·  {len(nse)}",
                                                                 f"BSE  ·  {len(bse)}"]
    tab.selectbox(key="scan_shariah").set_value("All").run()
    tab.selectbox(key="scan_exchange").set_value("BSE").run()
    assert not tab.exception, [e.value for e in tab.exception]
    assert len(_expected(comp, shariah="Shariah", exchange="BSE")) < len(kept), "teeth"
    assert f"✅ Shariah  ·  {len(kept)}" in tab.selectbox(key="scan_shariah").options, "Shariah counts the whole list"


def test_an_exchange_with_no_stock_reads_zero_and_shows_the_tabs_empty_message(tab):
    """Empty states read plainly. 🚀's matches here are all on NSE, so its option reads 'BSE  ·  0'; on 🌱, Shariah's
    Doubtful pick leaves no BSE stock. Choosing it shows the tab's own empty-list message, never an error or a table."""
    from ui import ui_scans as S
    play = S.SCANS[0]
    assert _expected(play) and not _expected(play, exchange="BSE")
    ex = tab.selectbox(key="scan_exchange")
    assert list(ex.options) == [f"All  ·  {len(_expected(play))}", f"NSE  ·  {len(_expected(play))}", "BSE  ·  0"]
    ex.set_value("BSE").run()
    assert not tab.exception and not tab.error, [e.value for e in tab.exception]
    assert not tab.dataframe
    assert [i.value for i in tab.info] == [f"None of these {len(_expected(play))} stocks passes BSE only — widen one of "
                                           f"the picks in this row."]
    comp = next(s for s in S.SCANS if s["key"] == "compounder")
    tab.button(key="scan_open_compounder").click().run()
    tab.selectbox(key="scan_shariah").set_value("Doubtful").run()
    assert not tab.exception, [e.value for e in tab.exception]
    doubtful = _expected(comp, shariah="Doubtful")
    assert doubtful and not _expected(comp, shariah="Doubtful", exchange="BSE"), "Shariah leaves no BSE stock"
    assert tab.selectbox(key="scan_exchange").options[-1] == "BSE  ·  0"
    assert not tab.dataframe and not tab.error
    assert [i.value for i in tab.info] == [f"None of these {len(_expected(comp))} stocks passes ❔ Doubtful + BSE only "
                                           f"— widen one of the picks in this row."]


def test_clear_returns_every_control_to_its_default(tab):
    """Every control's default is empty (2026-10-13), so 🧹 empties the row — and leaves the view alone. Exchange
    (2026-11-03) goes back to All, exactly as the others go back to theirs."""
    tab.selectbox(key="scan_liq").set_value(5.0).run()
    tab.selectbox(key="scan_exchange").set_value("NSE").run()
    tab.multiselect(key="scan_stage").set_value(["📉 Stage 4 Declining"]).run()
    tab.multiselect(key="scan_cap").set_value(["Mid Cap"]).run()
    clear = [b for b in tab.button if b.key == "scan_clear"]
    assert clear, "🧹 shows once a control leaves its default"
    clear[0].click().run()
    assert not tab.exception, [e.value for e in tab.exception]
    assert tab.selectbox(key="scan_liq").value is None, "back to Any"
    assert tab.selectbox(key="scan_exchange").value == "All"
    assert tab.multiselect(key="scan_stage").value == []
    assert tab.multiselect(key="scan_cap").value == [] and tab.multiselect(key="scan_zone").value == []
    assert _control(tab, "scan_view").value == "🔢 RAM", "the view is not a filter"


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
    _control(at, "scan_view").set_value("🏆 Core").run()   # 2026-11-05: 🚀 opens on 🔢 RAM, which has no New column to read
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
    assert list(tab.dataframe[0].proto.column_order) == list(S.RAM_VIEW_COLUMNS)
    _control(tab, "scan_view").set_value("🏆 Core").run()   # 2026-11-05: C's List column would show here, had the choice leaked
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


# ── The 🔢 RAM view, RUN (2026-11-05): tests/test_ram_display.py holds its pieces, this its place in the real tab ─────────
_RAM_FILLED = ("_nse = df['company_id'].str.startswith('NSE:')\n"
               "df = df.assign(ram_rank=df['momentum_score'].rank().where((np.arange(n) % 3 != 0) & _nse),\n"
               "               ram_pct=100.0 - df['momentum_score'], ram_score=df['momentum_score'] / 50.0,\n"
               "               ram_date=np.where(_nse, '2026-11-04', None))\n")
# st, except that its table reports the first row picked, as a click does (AppTest cannot click a row).
_PICKS_THE_FIRST_ROW = """
_st = st
class _Picks:
    def __getattr__(self, name):
        return getattr(_st, name)
    def dataframe(self, *args, **kw):
        _st.dataframe(*args, **kw)
        return SimpleNamespace(selection=SimpleNamespace(rows=[0]))
st = _Picks()
"""


def _ram_tab(tmp_path, extra=""):
    """The real tab on a market whose RAM runs against its Momentum, with every third stock and every BSE id unranked."""
    from streamlit.testing.v1 import AppTest
    source = _app_source(_records_without_baselines()).replace("filt = df\n", _RAM_FILLED + "filt = df\n" + extra, 1)
    script = tmp_path / "scans_tab.py"
    script.write_text(source, encoding="utf-8")
    at = AppTest.from_file(str(script), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_the_rocket_lists_open_on_the_ram_view_by_ram_rank_with_blanks_last(tmp_path):
    from ui import ui_scans as S
    at = _ram_tab(tmp_path)
    for key in ("playbook", "no_quarter", "price_only"):
        if key != "playbook":
            _control(at, "scan_layer").set_value(key).run()
            assert not at.exception, [e.value for e in at.exception]
        view = _control(at, "scan_view")
        assert view.value == S.RAM_VIEW and [o.content for o in view.options] == ["RAM", "Core", "Technical"], key
        grid = at.dataframe[0]
        assert list(grid.proto.column_order) == list(S.RAM_VIEW_COLUMNS), key
        t = grid.value
        ranked = t["ram_rank"].notna().sum()
        assert t["ram_rank"].iloc[:ranked].is_monotonic_increasing and t["ram_rank"].iloc[ranked:].isna().all(), key
        assert t["momentum_score"].iloc[ranked:].is_monotonic_decreasing, f"{key}: the blanks keep today's order"
        assert t["#"].tolist() == list(range(1, len(t) + 1)), key
        assert any("sorted by RAM Rank, strongest first · no RAM last" in c.value for c in at.caption), key
    assert t["ram_rank"].isna().any(), "C holds blanks to put last"


def test_flipping_to_core_or_technical_is_mains_table_and_mains_caption(tmp_path):
    from ui import ui_scans as S
    at = _ram_tab(tmp_path)
    for view in S.SCAN_VIEWS:
        _control(at, "scan_view").set_value(view).run()
        assert not at.exception, [e.value for e in at.exception]
        grid = at.dataframe[0]
        assert list(grid.proto.column_order) == list(S.table_columns(S.SCANS[0], view)), view
        assert grid.value["momentum_score"].is_monotonic_decreasing, f"{view}: the list's own order"
        assert ("ram_rank" in grid.value.columns) == (view == "📈 Technical"), "2026-11-05: RAM Rank joins 📈 Technical alone"
        cap = next(c.value for c in at.caption if "sorted by" in c.value)
        assert "sorted by Momentum, high to low · click a row" in cap, cap


def test_every_scan_opens_on_the_ram_view_and_a_choice_stays_when_the_scan_changes(tab):
    """2026-11-05: the same three options on every list are one widget, so 📈 Technical chosen on 🚀 is 📈 Technical on 🌬️ and 🌱."""
    from ui import ui_scans as S
    assert _control(tab, "scan_view").value == S.RAM_VIEW
    for scan in S.SCANS[1:]:
        tab.button(key=f"scan_open_{scan['key']}").click().run()
        assert not tab.exception, [e.value for e in tab.exception]
        view = _control(tab, "scan_view")
        assert view.value == S.RAM_VIEW and [o.content for o in view.options] == ["RAM", "Core", "Technical"], scan["key"]
        assert list(tab.dataframe[0].proto.column_order) == list(S.RAM_VIEW_COLUMNS), scan["key"]
    _control(tab, "scan_view").set_value("📈 Technical").run()
    for scan in S.SCANS:
        tab.button(key=f"scan_open_{scan['key']}").click().run()
        assert not tab.exception, [e.value for e in tab.exception]
        assert _control(tab, "scan_view").value == "📈 Technical", scan["key"]
        assert list(tab.dataframe[0].proto.column_order) == list(S.table_columns(scan, "📈 Technical")), scan["key"]


@pytest.mark.parametrize("key, own_column, own_order", [("leaders", "momentum_score", "is_monotonic_decreasing"),
                                                        ("compounder", "rank", "is_monotonic_increasing")])
def test_leaders_and_compounder_read_by_ram_rank_with_blanks_last_in_their_own_order(tmp_path, key, own_column, own_order):
    from ui import ui_scans as S
    at = _ram_tab(tmp_path)
    at.button(key=f"scan_open_{key}").click().run()
    assert not at.exception, [e.value for e in at.exception]
    grid = at.dataframe[0]
    assert list(grid.proto.column_order) == list(S.RAM_VIEW_COLUMNS)
    t = grid.value
    ranked = t["ram_rank"].notna().sum()
    assert 0 < ranked < len(t), "the synthetic market gives the list both, to order"
    assert t["ram_rank"].iloc[:ranked].is_monotonic_increasing and t["ram_rank"].iloc[ranked:].isna().all()
    assert getattr(t[own_column].iloc[ranked:], own_order), "the blanks keep the list's own order"
    assert t["#"].tolist() == list(range(1, len(t) + 1))
    assert any("sorted by RAM Rank, strongest first · no RAM last · RAM of 4 Nov · click a row" in c.value for c in at.caption)


def test_a_copy_with_no_ram_reads_todays_order_and_says_so(tab):
    t = tab.dataframe[0].value
    assert t["ram_rank"].isna().all() and t["momentum_score"].is_monotonic_decreasing
    assert any("sorted by Momentum, high to low · no RAM on this copy" in c.value for c in tab.caption)


@pytest.mark.parametrize("key, words", [("leaders", "Momentum, high to low"), ("compounder", "Rank")])
def test_a_copy_with_no_ram_reads_leaders_and_compounder_in_their_own_order_and_says_so(tab, key, words):
    tab.button(key=f"scan_open_{key}").click().run()
    assert not tab.exception, [e.value for e in tab.exception]
    opened = list(tab.dataframe[0].value["name"])
    assert any(f"sorted by {words} · no RAM on this copy" in c.value for c in tab.caption), key
    _control(tab, "scan_view").set_value("🏆 Core").run()
    assert list(tab.dataframe[0].value["name"]) == opened, "with no RAM the view keeps the list's order, row for row"


@pytest.mark.parametrize("key", ["playbook", "no_quarter", "price_only", "leaders", "compounder"])
def test_technical_shows_ram_rank_and_improvement_on_every_list_in_the_real_tab(tmp_path, key):
    """2026-11-05, the operator's ask. It keeps each list's own order, New's legend stays with Core, and B and C keep List."""
    from ui import ui_scans as S
    at = _ram_tab(tmp_path)
    if key in ("leaders", "compounder"):
        at.button(key=f"scan_open_{key}").click().run()
    elif key != "playbook":
        _control(at, "scan_layer").set_value(key).run()
    _control(at, "scan_view").set_value("📈 Technical").run()
    assert not at.exception, [e.value for e in at.exception]
    grid = at.dataframe[0]
    order = list(grid.proto.column_order)
    assert "ram_rank" in order and "improvement_score" in order and "_new" not in order, key
    assert ("_layer" in order) == (key in ("no_quarter", "price_only")), key
    config = json.loads(grid.proto.columns)
    assert config["ram_rank"]["label"] == "RAM Rank" and config["improvement_score"]["label"] == "Improvement", key
    assert config["name"]["width"] == (130 if key in ("no_quarter", "price_only") else 170), key
    t = grid.value
    assert t["ram_rank"].notna().any() and t["ram_rank"].isna().any(), f"{key}: a rank here, a dash there"
    own = t["rank"].is_monotonic_increasing if key == "compounder" else t["momentum_score"].is_monotonic_decreasing
    assert own, f"{key}: the list's own order, not RAM Rank's"
    cap = next(c.value for c in at.caption if "sorted by" in c.value)
    assert "🆕" not in cap and "RAM of" not in cap, cap


def test_core_keeps_new_and_its_legend_while_technical_does_not(tmp_path):
    from streamlit.testing.v1 import AppTest
    records = _records_without_baselines()
    records["scans"]["playbook"]["members_at_latest"] = _frame()["company_id"].tolist()[:150]
    records["members_date"] = "2026-09-30"
    script = tmp_path / "scans_tab_new.py"
    script.write_text(_app_source(records), encoding="utf-8")
    at = AppTest.from_file(str(script), default_timeout=60)
    at.run()
    _control(at, "scan_view").set_value("🏆 Core").run()
    assert "_new" in at.dataframe[0].proto.column_order
    assert any("🆕 new since 30 Sep" in c.value for c in at.caption), "Core reads the baseline"
    _control(at, "scan_view").set_value("📈 Technical").run()
    assert "_new" not in at.dataframe[0].proto.column_order
    assert not any("🆕" in c.value for c in at.caption), "the legend stays with its column"


def test_a_row_click_in_technical_loads_the_row_on_screen(tmp_path):
    at = _ram_tab(tmp_path, _PICKS_THE_FIRST_ROW)
    _control(at, "scan_view").set_value("📈 Technical").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["xray_stock"] == at.dataframe[0].value["name"].iloc[0]


def test_a_row_click_after_the_sort_loads_that_stock(tmp_path):
    at = _ram_tab(tmp_path, _PICKS_THE_FIRST_ROW)
    t = at.dataframe[0].value
    assert len(t) > 1 and t["ram_rank"].iloc[0] == t["ram_rank"].min() and t["momentum_score"].iloc[0] != t["momentum_score"].max()
    assert at.session_state["xray_stock"] == t["name"].iloc[0], "the row on screen, not the row of the list's own order"


def test_the_caption_says_how_old_the_ranks_are(tmp_path):
    at = _ram_tab(tmp_path)
    assert any("sorted by RAM Rank, strongest first · no RAM last · RAM of 4 Nov · click a row" in c.value for c in at.caption)


def test_a_bse_only_list_on_a_copy_with_ram_says_this_list_and_keeps_todays_order(tmp_path):
    at = _ram_tab(tmp_path)
    _control(at, "scan_layer").set_value("price_only").run()
    at.selectbox(key="scan_exchange").set_value("BSE").run()
    assert not at.exception, [e.value for e in at.exception]
    t = at.dataframe[0].value
    assert len(t) > 1 and t["ram_rank"].isna().all() and t["momentum_score"].is_monotonic_decreasing
    cap = next(c.value for c in at.caption if "sorted by" in c.value)
    assert "sorted by Momentum, high to low · no RAM in this list" in cap and "on this copy" not in cap, cap
    assert "RAM of" not in cap, cap


def test_the_view_shows_a_dash_for_a_blank_and_every_other_view_keeps_streamlits_own(tmp_path):
    from ui import ui_scans as S
    at = _ram_tab(tmp_path)
    assert at.dataframe[0].proto.placeholder == "—", "not the word None, under three columns on half of each list"
    for view in S.SCAN_VIEWS:
        _control(at, "scan_view").set_value(view).run()
        assert not at.exception, [e.value for e in at.exception]
        assert at.dataframe[0].proto.placeholder == ("—" if view == "📈 Technical" else ""), view   # it carries RAM Rank (2026-11-05)
    at.button(key="scan_open_leaders").click().run()
    assert at.dataframe[0].proto.placeholder == "—", "📈 Technical stays chosen on 🌬️"
    _control(at, "scan_view").set_value("🏆 Core").run()
    assert at.dataframe[0].proto.placeholder == "", "Core keeps Streamlit's own"
    _control(at, "scan_view").set_value(S.RAM_VIEW).run()
    assert at.dataframe[0].proto.placeholder == "—", "🌬️ draws the view as 🚀 does"


def test_the_tab_draws_the_view_at_its_own_widths_on_every_list(tmp_path):
    """Not 🚀's own Stock 185 and Sector 125, which the view's would silently fall back to if the tab skipped ram_view_scan."""
    at = _ram_tab(tmp_path)
    for key in ("playbook", "no_quarter", "price_only"):
        if key != "playbook":
            _control(at, "scan_layer").set_value(key).run()
            assert not at.exception, [e.value for e in at.exception]
        config = json.loads(at.dataframe[0].proto.columns)
        assert (config["name"]["width"], config["sector"]["width"]) == (312, 170), key
    for key in ("leaders", "compounder"):           # 🌱's own Stock 150 and Sector 80 are Core's, not the view's
        at.button(key=f"scan_open_{key}").click().run()
        assert not at.exception, [e.value for e in at.exception]
        config = json.loads(at.dataframe[0].proto.columns)
        assert (config["name"]["width"], config["sector"]["width"]) == (312, 170), key


def test_deselecting_the_active_segment_keeps_the_lists_first_view(tmp_path):
    from ui import ui_scans as S
    at = _ram_tab(tmp_path)
    _control(at, "scan_view").set_value([]).run()                     # a click on the active segment empties the control
    assert not at.exception, [e.value for e in at.exception]
    assert list(at.dataframe[0].proto.column_order) == list(S.RAM_VIEW_COLUMNS), "the 🚀 list's first view, not Core"
    assert any("sorted by RAM Rank" in c.value for c in at.caption)
    at.button(key="scan_open_leaders").click().run()
    _control(at, "scan_view").set_value([]).run()
    assert not at.exception, [e.value for e in at.exception]
    assert list(at.dataframe[0].proto.column_order) == list(S.RAM_VIEW_COLUMNS), "🌬️'s first view, the same"


def test_a_filter_renumbers_the_place_in_the_real_tab(tmp_path):
    at = _ram_tab(tmp_path)
    _control(at, "scan_layer").set_value("price_only").run()
    whole = at.dataframe[0].value
    at.selectbox(key="scan_liq").set_value(5.0).run()
    assert not at.exception, [e.value for e in at.exception]
    narrow = at.dataframe[0].value
    assert 0 < len(narrow) < len(whole), "the floor must really narrow"
    assert narrow["#"].tolist() == list(range(1, len(narrow) + 1))


def test_the_views_caption_names_no_new_column_it_does_not_show(tmp_path):
    from streamlit.testing.v1 import AppTest
    records = _records_without_baselines()
    records["scans"]["playbook"]["members_at_latest"] = _frame()["company_id"].tolist()[:150]
    records["members_date"] = "2026-09-30"
    script = tmp_path / "scans_tab_new.py"
    script.write_text(_app_source(records), encoding="utf-8")
    at = AppTest.from_file(str(script), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    cap = next(c.value for c in at.caption if "sorted by" in c.value)
    assert "🆕" not in cap and "📄" not in cap, cap
