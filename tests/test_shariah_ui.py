"""
test_shariah_ui.py
==================
The Shariah screen on screen (2026-10-16): ONE vocabulary and ONE filter on every surface.

  ✅ Shariah (green) · ⛔ Non-Shariah (red) · 📝 Review (blue) · ❔ Doubtful (amber) — none but Shariah is ever
  green; the words from config.SHARIAH_STATUSES, the icons and colours from ui/shariah_view.py, nowhere else.
  Colour always comes with an icon and a word. The icons were checked against PRISM's own: ✅ already means
  "passed" and ❔ "no data"; 📝 was unused; 🟢🟡🔴 belong to the Soundness verdict and are not used. No religious
  symbols.

  The filter (All · Shariah · Non-Shariah · Review · Doubtful, live counts, default All) is ONE control —
  ui_discovery.shariah_filter — drawn by the Discovery sidebar and the 🎯 Scans row. Readers match by
  EQUALITY: "Shariah" sits inside "Non-Shariah".

  The Tear-Sheet: a chip in the verdict band, and a "Shariah screen" card in Overview — status, reason, six
  rows (Rule · This company · Limit · Result), what to check for every rule PRISM cannot decide, a stateless
  <details> "How this is checked" and the footnote. The Reference tab explains it, every number from config.
  Every CSV download carries shariah_status and shariah_reason.

Run with: pytest tests/test_shariah_ui.py -v
"""
import ast
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _ROOT)

from config import COLORS, SHARIAH_RULES, SHARIAH_STATUSES
from ui.shariah_view import (SHARIAH_ALL, SHARIAH_COLORS, SHARIAH_FILTER_OPTIONS, SHARIAH_FOOTNOTE, SHARIAH_ICONS,
                             SHARIAH_RESULTS, rule_limit_text, shariah_counts, shariah_label, shariah_mask,
                             shariah_pill_html)


def _src(rel):
    return open(os.path.join(_ROOT, rel), encoding="utf-8").read()


# ── 1. One vocabulary ─────────────────────────────────────────────────────────────────────────────────
def test_one_icon_and_one_colour_per_word_keyed_from_config():
    assert tuple(SHARIAH_ICONS) == tuple(SHARIAH_COLORS) == SHARIAH_STATUSES
    assert SHARIAH_ICONS == {"Shariah": "✅", "Non-Shariah": "⛔", "Review": "📝", "Doubtful": "❔"}
    assert SHARIAH_COLORS == {"Shariah": COLORS["green"], "Non-Shariah": COLORS["red"], "Review": COLORS["blue"],
                              "Doubtful": COLORS["gold"]}


def test_doubtful_is_never_green_and_no_icon_is_borrowed():
    assert SHARIAH_COLORS["Doubtful"] != COLORS["green"] and SHARIAH_COLORS["Review"] != COLORS["green"]
    assert len(set(SHARIAH_COLORS.values())) == 4, "four words, four colours"
    icons = set(SHARIAH_ICONS.values())
    assert len(icons) == 4
    assert not icons & {"🟢", "🟡", "🔴"}, "those are the Soundness verdict's own emoji"
    assert not icons & {"☪️", "☪", "🕌", "📿", "🕋"}, "no religious symbols"


def test_a_label_is_icon_plus_word():
    assert [shariah_label(s) for s in SHARIAH_STATUSES] == ["✅ Shariah", "⛔ Non-Shariah", "📝 Review",
                                                            "❔ Doubtful"]
    assert shariah_label(SHARIAH_ALL) == "All"
    for missing in (None, np.nan, "", "Halal"):
        assert shariah_label(missing) == "", repr(missing)


def test_a_rule_result_is_icon_plus_word_in_the_same_colours():
    assert SHARIAH_RESULTS == {"pass": "✅ Pass", "fail": "⛔ Fail", "review": "📝 Review", "unknown": "❔ Unknown"}


def test_the_pill_carries_icon_word_and_colour():
    for s in SHARIAH_STATUSES:
        html = shariah_pill_html(s)
        assert shariah_label(s) in html and SHARIAH_COLORS[s] in html, s
    assert shariah_pill_html(np.nan) == "" and shariah_pill_html(None) == ""


def test_every_limit_reads_config(monkeypatch):
    assert [rule_limit_text(k) for k in SHARIAH_RULES] == [
        "not a prohibited business", "at most 5% of revenue", "at most 33% of total assets",
        "at least 20% of total assets", "at most the market cap", "below 33% of total assets"]
    monkeypatch.setitem(SHARIAH_RULES["debt"], "limit", 0.30)
    assert rule_limit_text("debt") == "at most 30% of total assets"


# ── 2. The filter: equality, live counts, default All ───────────────────────────────────────────────
def _statuses():
    return pd.DataFrame({"name": list("ABCDEF"),
                         "shariah_status": ["Shariah", "Non-Shariah", "Doubtful", "Non-Shariah", np.nan, "Review"]})


def test_the_filter_matches_by_equality_never_contains():
    f = _statuses()
    assert f.loc[shariah_mask(f, "Shariah"), "name"].tolist() == ["A"], "'Shariah' sits inside 'Non-Shariah'"
    assert f.loc[shariah_mask(f, "Non-Shariah"), "name"].tolist() == ["B", "D"]
    assert f.loc[shariah_mask(f, "Review"), "name"].tolist() == ["F"]
    assert f.loc[shariah_mask(f, "Doubtful"), "name"].tolist() == ["C"]
    assert shariah_mask(f, SHARIAH_ALL).all()


def test_the_counts_are_live_and_by_equality():
    assert shariah_counts(_statuses()) == {"All": 6, "Shariah": 1, "Non-Shariah": 2, "Review": 1, "Doubtful": 1}
    assert shariah_counts(_statuses().iloc[:0]) == {"All": 0, "Shariah": 0, "Non-Shariah": 0, "Review": 0,
                                                    "Doubtful": 0}


def test_the_options_open_on_all_and_list_every_status():
    assert SHARIAH_FILTER_OPTIONS == ("All", *SHARIAH_STATUSES)
    assert SHARIAH_ALL == SHARIAH_FILTER_OPTIONS[0]


def test_a_frame_without_the_column_keeps_nothing_but_all():
    f = pd.DataFrame({"name": ["A", "B"]})
    assert shariah_mask(f, SHARIAH_ALL).all() and not shariah_mask(f, "Shariah").any()
    assert shariah_counts(f) == {"All": 2, "Shariah": 0, "Non-Shariah": 0, "Review": 0, "Doubtful": 0}


# ── 3. The one shared control, drawn by both tabs ───────────────────────────────────────────────────
def _widget_app():
    import pandas as _pd
    import streamlit as _st

    from ui.shariah_view import shariah_mask as _mask
    from ui.ui_discovery import shariah_filter

    df = _pd.DataFrame({"name": list("ABCD"),
                        "shariah_status": ["Non-Shariah", "Shariah", "Non-Shariah", "Doubtful"]})
    pick = shariah_filter("Shariah screen", "k", df, help="the Shariah screen")
    _st.text(f"PICK={pick}")
    _st.text(f"KEPT={sorted(df.loc[_mask(df, pick), 'name'])}")


def _run_widget(state=None):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_widget_app)
    for k, v in (state or {}).items():
        at.session_state[k] = v
    at.run(timeout=30)
    assert not at.exception, at.exception
    return at, {t.value.split("=", 1)[0]: t.value.split("=", 1)[1] for t in at.text}


def test_the_control_opens_on_all_with_live_counts():
    at, out = _run_widget()
    assert out == {"PICK": "All", "KEPT": "['A', 'B', 'C', 'D']"}
    assert at.selectbox(key="k").options == ["All  ·  4", "✅ Shariah  ·  1", "⛔ Non-Shariah  ·  2",
                                             "📝 Review  ·  0", "❔ Doubtful  ·  1"]


def test_a_pick_keeps_only_its_own_status():
    _, out = _run_widget({"k": "Shariah"})
    assert out == {"PICK": "Shariah", "KEPT": "['B']"}


def test_a_round_tripped_label_is_recovered_to_its_value():
    """The 2026-09-21 class: Streamlit can write the DISPLAY label back into session_state when the count in it
    changes. Recovered, the filter still applies; unrecovered, it would match nothing or fall back to All."""
    _, out = _run_widget({"k": "✅ Shariah  ·  462"})
    assert out == {"PICK": "Shariah", "KEPT": "['B']"}
    _, out = _run_widget({"k": "⛔ Non-Shariah  ·  0"})
    assert out == {"PICK": "Non-Shariah", "KEPT": "['A', 'C']"}


def test_the_control_seeds_its_key_before_the_widget():
    tree = ast.parse(_src("ui/ui_discovery.py"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "shariah_filter")
    src = ast.unparse(fn)
    assert src.index("st.session_state.setdefault(key") < src.index("st.selectbox("), "the Steel repro"
    assert "recover_labels(" in src, "a round-tripped label must be recovered before it is read"


def test_both_tabs_draw_the_one_control_and_no_copy_of_it():
    disc, app = _src("ui/ui_discovery.py"), _src("app.py")
    assert 'shariah_filter("Shariah screen", "sb_shariah", _cf' in disc
    assert 'shariah_filter("Shariah", "scan_shariah", _cf' in app
    assert "SHARIAH_FILTER_OPTIONS" not in app, "the Scans row builds no option list of its own"
    assert app.count("shariah_filter(") == 1 and disc.count("shariah_filter(") == 2   # the def + the call


# ── 4. Discovery: registered like every other filter ───────────────────────────────────────────────
def test_the_sidebar_registers_the_filter_like_every_other():
    from ui.ui_discovery import _CHIP_META, _compute_active_chips
    assert ("sb_shariah", "Shariah", "sel") in _CHIP_META
    assert _compute_active_chips({"sb_shariah": "Doubtful"}, 28) == [("sb_shariah", "Shariah: Doubtful")]
    assert _compute_active_chips({"sb_shariah": "All"}, 28) == []
    disc = _src("ui/ui_discovery.py")
    grp = next(l for l in disc.splitlines() if '_grp("🏢 Universe"' in l)
    assert "sb_shariah" in grp
    i = disc.rindex("_active_n(")
    assert "sb_shariah" in disc[i:disc.index(")", i)], "missing from the funnel total"
    assert '"sb_sector", "sb_industry", "sb_shariah"' in disc, "its 'All' must not count as an active filter"
    assert '_narrow(_cf, shariah_mask(_cf, sel_shariah), "Shariah")' in disc, "through the choke point"


def test_the_discovery_card_names_the_status(monkeypatch):
    import ui.ui_components as C
    shown = []
    monkeypatch.setattr(C.st, "markdown", lambda html, **k: shown.append(html))
    row = pd.Series({"name": "Test Co", "rank": 1, "conviction_tier": 2, "composite_score": 70.0,
                     "market_cap": 1000.0, "shariah_status": "Non-Shariah"})
    C.render_stock_card(row)
    assert "⛔ Non-Shariah" in shown[-1] and SHARIAH_COLORS["Non-Shariah"] in shown[-1]
    C.render_stock_card(row.drop("shariah_status"))
    assert "Shariah" not in shown[-1], "no label over no data"


# ── 5. The Tear-Sheet ────────────────────────────────────────────────────────────────────────────────
def _stock(**over):
    from core.shariah import compute_shariah
    base = dict(company_id="NSE:TEST", name="Test Co", industry="Speciality Chemicals", sector="Chemicals",
                cyclicality_tier_code="A", total_assets=1000.0, debt=100.0, fixed_assets=500.0, cwip=0.0,
                inventory=100.0, cash_equivalents=50.0, market_cap=2000.0, price_to_book=2.0, revenue=1000.0,
                revenue_1yb=1000.0, ebitda=200.0, ebit=180.0, days_receivable=73.0, cf_year_lag=0.0)
    return compute_shariah(pd.DataFrame([{**base, **over}])).iloc[0]


def test_the_card_leads_with_the_answer_then_six_rows():
    from ui.ui_tearsheet import shariah_card_html
    s = _stock(cash_equivalents=900.0)                                     # Rule 6 cannot be proven: for review
    html = shariah_card_html(s)
    assert "📝 Review" in html and s["shariah_reason"] in html
    assert html.index("📝 Review") < html.index("Rule</th>"), "answer first"
    for k, r in SHARIAH_RULES.items():
        assert f"{r['n']} · {r['name']}" in html, k
        assert rule_limit_text(k) in html, k
    for h in ("Rule", "This company", "Limit", "Result"):
        assert f">{h}</th>" in html
    assert "✅ Pass" in html and html.count("📝 Review") == 2, "the status and Rule 6's result"
    assert "❔ Doubtful" in shariah_card_html(_stock(fixed_assets=np.nan)), "a missing figure reads Doubtful"
    assert "Check: " + s["shariah_investments_check"] in html, "what to look up, with its rupee limit"


def test_the_card_strips_the_limit_from_this_company_and_shows_it_once():
    from ui.ui_tearsheet import shariah_card_html
    html = shariah_card_html(_stock(debt=410.0))
    assert ">debt is 41% of assets<" in html and "(limit 33%)<" not in html.split("Rule</th>")[1]


def test_the_card_explains_itself_and_carries_the_footnote():
    from ui.ui_tearsheet import shariah_card_html
    html = shariah_card_html(_stock())
    assert "<details" in html and "<summary" in html and "How this is checked" in html
    assert SHARIAH_FOOTNOTE in html
    assert SHARIAH_FOOTNOTE == ("A screen from financial data, not a fatwa. Interest income is not in the data: "
                                "check the annual report and purify dividends and gains. Confirm with your scholar.")


def test_the_card_shows_a_ruling_and_a_lapse():
    from ui.ui_tearsheet import shariah_card_html
    s = _stock()
    s["shariah_ruling"] = "your figures: Annual report FY26, notes 6 and 12, checked 2026-10-20"
    assert "✍ your figures: Annual report FY26" in shariah_card_html(s)
    s["shariah_ruling"] = ""
    assert "✍" not in shariah_card_html(s)


def test_the_card_without_a_screen_says_so():
    from ui.ui_tearsheet import shariah_card_html
    assert "not screened" in shariah_card_html(pd.Series({"name": "X"}))


def test_the_verdict_band_carries_the_chip():
    app = _src("app.py")
    assert "_shariah_pill = shariah_pill_html(stock.get(\"shariah_status\"))" in app
    band = app[app.index("_wealth_pill}{_risk_pill}") - 40: app.index("_wealth_pill}{_risk_pill}") + 40]
    assert "{_shariah_pill}" in band, "the chip sits in the always-visible header row"


def test_the_overview_renders_the_card_after_the_trajectory():
    app = _src("app.py")
    assert app.index("render_trajectory_card(stock)") < app.index("render_shariah_card(stock)") < app.index(
        "render_financial_insights(stock)")


# ── 6. The Reference tab ─────────────────────────────────────────────────────────────────────────────
def test_the_reference_explains_labels_rules_sight_purification_and_source():
    from ui.ui_reference_data import CONCEPT_REFERENCE
    section = CONCEPT_REFERENCE["📜 Shariah screen"]
    labels = [l for l, _ in section]
    assert labels[:4] == list(SHARIAH_STATUSES), "the four labels, exactly as the column holds them"
    for r in SHARIAH_RULES.values():
        assert f"Rule {r['n']} — {r['name']}" in labels
    text = " ".join(e for _, e in section)
    for needle in ("KMI-30", "AAOIFI", "purif", "cannot see", "annual report", "SHARIAH_RULINGS"):
        assert needle in text, needle


def test_the_reference_reads_every_number_from_config(monkeypatch):
    from ui import ui_reference_data as R
    monkeypatch.setitem(SHARIAH_RULES["debt"], "limit", 0.30)
    monkeypatch.setitem(SHARIAH_RULES["illiquid"], "limit", 0.25)
    text = " ".join(e for _, e in R.shariah_concepts())
    assert "at most 30% of total assets" in text and "at least 25% of total assets" in text
    block = _src("ui/ui_reference_data.py").split("def shariah_concepts")[1].split('CONCEPT_REFERENCE["📜')[0]
    assert not re.search(r"\b(33|20|5)%", block), "a number typed into the explainer instead of read from config"


# ── 7. Every CSV download ────────────────────────────────────────────────────────────────────────────
def test_the_deep_scanner_export_carries_status_and_reason():
    tree = ast.parse(_src("app.py"))
    exp = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
           and any(isinstance(t, ast.Name) and t.id == "_export_cols" for t in n.targets)]
    assert len(exp) == 1
    assert "'shariah_status'" in ast.unparse(exp[0].value) and "'shariah_reason'" in ast.unparse(exp[0].value)


def test_the_movers_download_carries_status_and_reason():
    from ui.ui_movers import with_shariah
    mat = pd.DataFrame({"company_id": ["A", "B"], "name": ["a", "b"]})
    sh = pd.DataFrame({"company_id": ["B", "A"], "shariah_status": ["Doubtful", "Shariah"],
                       "shariah_reason": ["r2", "r1"], "other": [1, 2]})
    out = with_shariah(mat, sh)
    assert out.columns.tolist() == ["company_id", "name", "shariah_status", "shariah_reason"]
    assert out["shariah_status"].tolist() == ["Shariah", "Doubtful"]
    pd.testing.assert_frame_equal(with_shariah(mat, None), mat)
    assert "with_shariah(mat, meta.get(\"shariah\"))" in _src("ui/ui_movers.py")
    assert '"shariah": df[["company_id", "shariah_status", "shariah_reason"]]' in _src("app.py")


def test_the_full_downloads_keep_every_column():
    from ui.ui_export import stamp_snapshot
    f = pd.DataFrame({"name": ["A"], "shariah_status": ["Shariah"], "shariah_reason": ["r"]})
    assert {"shariah_status", "shariah_reason"} <= set(stamp_snapshot(f, "2026-10-15", "sheet").columns)
