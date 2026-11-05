"""Contract for the Deep Scanner CSV export — the 📥 download button in app.py's Deep Scanner tab.
Pins two regressions the 2026-06-20 audit surfaced:

  1. The export must encode via ui_export._to_csv_bytes (UTF-8 BOM) so its emoji decision-columns
     (moat_growth_quad ⭐💀, smart_money_flow ⚪✅❌, weinstein_stage, buy_zone_label) render in Excel
     instead of mojibake — the sibling sidebar full-dump export already does exactly this.
  2. Every column the _DS_VIEWS presets surface (== the export's deduped union) must exist on the real
     scored frame, or it silently drops from BOTH the on-screen table and the export, with the button's
     "{n} columns" count quietly wrong and no error raised.

app.py's tab body is Streamlit runtime code (not unit-renderable), so finding #1 is pinned by a precise
AST check on the Deep Scanner block — the same house style as test_app_imports / test_tooltip_coverage.
"""
import ast
import contextlib
import io
from pathlib import Path

import pytest

_APP = Path(__file__).resolve().parent.parent / "app.py"
_DATA_DIR = Path(__file__).resolve().parent.parent / "Other Resources" / "CSV Data"


def _tab_position(tree, label):
    """The tab's position in app.py's own st.tabs([...]) list, found by its LABEL — so a tab inserted
    before it (🎯 Scans became tab 2 on 2026-10-06) cannot leave this test reading the wrong body."""
    call = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "tabs")
    return [e.value for e in call.args[0].elts].index(label)


def _deep_scanner_block(tree):
    """The Deep Scanner's `with tabs[i]:` With-node (i found by its label) — scopes the search to this tab so it can't pick
    up a download_button from an unrelated tab (the Tear-Sheet and sidebar each have their own)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                ctx = item.context_expr
                if (isinstance(ctx, ast.Subscript) and isinstance(ctx.value, ast.Name)
                        and ctx.value.id == "tabs"
                        and isinstance(ctx.slice, ast.Constant)
                        and ctx.slice.value == _tab_position(tree, "🔍 Deep Scanner")):
                    return node
    return None


def _download_button_data_kw(block):
    """The `data=` keyword value-node of the Deep Scanner's st.download_button call (None if absent)."""
    for node in ast.walk(block):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "download_button"):
            for kw in node.keywords:
                if kw.arg == "data":
                    return kw.value
    return None


def test_export_encodes_through_bom_helper():
    """The Deep Scanner export must hand download_button bytes from _to_csv_bytes (UTF-8 BOM), not a
    bare DataFrame.to_csv() — otherwise its emoji decision-columns mojibake when opened in Excel."""
    block = _deep_scanner_block(ast.parse(_APP.read_text(encoding="utf-8"), filename="app.py"))
    assert block is not None, "could not locate the Deep Scanner's tab block in app.py"
    data = _download_button_data_kw(block)
    assert data is not None, "Deep Scanner download_button has no data= keyword"
    assert (isinstance(data, ast.Call) and isinstance(data.func, ast.Name)
            and data.func.id == "_to_csv_bytes"), (
        "Deep Scanner export must encode via _to_csv_bytes(...) for an Excel-safe UTF-8 BOM "
        "(consistency with the sidebar full-dump export); found a different data= expression."
    )


def _ds_view_union(tree):
    """The deduped union of every _DS_VIEWS preset column plus every _DS_SORTS column — exactly the
    export's column set (the sort columns joined 2026-10-05, when MCap left the views), AST-parsed."""
    union: list = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_DS_VIEWS" for t in node.targets)
                and isinstance(node.value, ast.Dict)):
            for preset in node.value.values:
                for elt in getattr(preset, "elts", []):
                    if isinstance(elt, ast.Constant) and elt.value not in union:
                        union.append(elt.value)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_DS_SORTS" for t in node.targets)
                and isinstance(node.value, ast.Dict)):
            for sort in node.value.values:
                col = sort.elts[0].value
                if col not in union:
                    union.append(col)
    return union


@pytest.mark.skipif(not _DATA_DIR.is_dir(),
                    reason="Local CSV data not present (code-only checkout) — needs the real scored frame")
def test_every_export_column_exists_on_scored_frame():
    """Every _DS_VIEWS preset column (== the export's deduped union) must resolve on the real scored
    frame. The export filters with `if _c in ds_df.columns`, so a renamed/typo'd engine column silently
    vanishes from BOTH the on-screen table and the CSV — the button's '{n} columns' count quietly drops
    and nothing raises. This locks the union against the live pipeline so that drift goes red."""
    union = _ds_view_union(ast.parse(_APP.read_text(encoding="utf-8"), filename="app.py"))
    assert len(union) >= 40, f"expected ~44 export columns parsed from _DS_VIEWS, got {len(union)}"

    from core.data_engine import (load_all_csvs, merge_datasets,
                                  coerce_numeric_columns, compute_derived_signals)
    from core import run_scoring_pipeline
    with contextlib.redirect_stdout(io.StringIO()):
        df = load_all_csvs("local")
        df = merge_datasets(df)
        df = coerce_numeric_columns(df)
        df = compute_derived_signals(df)
        df = run_scoring_pipeline(df)

    missing = [c for c in union if c not in df.columns]
    assert not missing, f"Deep Scanner export columns absent from the scored frame: {missing}"


def _download_button_by_label(tree, label_substr):
    """The st.download_button Call whose label (first positional arg, str or f-string) contains
    label_substr — targets one button by its visible text without depending on tab nesting."""
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "download_button" and node.args):
            label = node.args[0]
            if isinstance(label, ast.JoinedStr):
                text = "".join(v.value for v in label.values if isinstance(v, ast.Constant))
            elif isinstance(label, ast.Constant) and isinstance(label.value, str):
                text = label.value
            else:
                text = ""
            if label_substr in text:
                return node
    return None


def test_all_data_export_encodes_through_bom_helper():
    """The All Data tab's single-row export (📥 ... Full Data Row) must ALSO encode via _to_csv_bytes
    (UTF-8 BOM) — its Value column dumps every engine column, which includes emoji decision-strings
    (corporate_class 🏆, smart_money_flow ⚪/✅/❌, weinstein_stage, verdict emojis) + Indian names that
    mojibake in Excel under a bare DataFrame.to_csv(). Same regression class as the Deep Scanner export,
    on the third (previously unguarded) download button.

    LOCATED BY ITS DATA ARGUMENT, NOT ITS LABEL. This originally searched for the literal label
    "Full Data Row" and broke when that label was reworded to state the signal count. The label is
    copy and will change again; `data=...(_stock_export)` is the thing this test is actually about.
    """
    tree = ast.parse(_APP.read_text(encoding="utf-8"), filename="app.py")
    btn = next((n for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "download_button"
                and any(kw.arg == "data" and "_stock_export" in ast.unparse(kw.value)
                        for kw in n.keywords)), None)
    assert btn is not None, "could not locate the All Data single-row export button in app.py"
    data = next((kw.value for kw in btn.keywords if kw.arg == "data"), None)
    assert data is not None, "All Data export download_button has no data= keyword"
    assert (isinstance(data, ast.Call) and isinstance(data.func, ast.Name)
            and data.func.id == "_to_csv_bytes"), (
        "All Data single-row export must encode via _to_csv_bytes(...) for an Excel-safe UTF-8 BOM "
        "(consistency with the Deep Scanner + sidebar exports); found a different data= expression."
    )


# ── 🆕 Results sort (2026-08-30) — freshness as ORDERING, never a gate ─────────────────────────
def _app_src():
    return _APP.read_text(encoding="utf-8")


def test_results_sort_is_ascending_on_result_age_days():
    """Freshness ships as a SORT because a filter measurably fails liveness: 'reported <=7d' held
    47.5% of the universe right after earnings season and ~5% mid-quarter — seasonal noise no gate
    survives, while an ordering always works. Ascending is load-bearing: negative ages (a result
    SCHEDULED but not yet declared) must lead, then the freshest reporters; descending would lead
    with the stalest."""
    src = _app_src()
    i = src.index("_DS_SORTS = {")
    block = src[i:src.index("}", i)]
    assert '"🆕 Results ↑"'.encode().decode("unicode_escape") in block or "🆕 Results ↑" in block, (
        "the 🆕 Results sort option vanished from _DS_SORTS")
    line = next(l for l in block.splitlines() if "result_age_days" in l)
    assert "True" in line, "🆕 Results must sort ASCENDING (due-soon first, freshest next)"


def test_results_sort_materializes_a_readable_column():
    """Sort-by-visible doctrine: no view carries result_age_days, so the active sort must
    materialize the readable result_when column ('📅 due 4d' = scheduled/not yet declared vs
    '8d ago' = reported) and place it beside the name — an ordering the table cannot explain is
    the rank-jumble that got Discovery's sort pills removed (2026-08-24)."""
    src = _app_src()
    i = src.index('ds_sort_label == "🆕 Results ↑"')
    block = src[i:i + 900]
    assert "result_when" in block, "the readable result_when column is no longer materialized"
    for piece in ('"📅 due "', '"today"', '"d ago"'):
        assert piece in block, f"result_when lost its {piece} state — a negative age would read as reported"
    # Placed beside the name by the SHARED rule since 2026-10-05 (ui_scanner.ds_view_columns maps
    # the Results sort to result_when) — it was an inline _view_cols.insert. Behavioural now: the
    # rule must put it right after Stock, and the app must build its columns AFTER materializing it.
    from ui.ui_scanner import ds_view_columns
    cols = ds_view_columns(["rank", "name", "quality_score"], "result_age_days",
                           {"rank", "name", "quality_score", "result_age_days", "result_when"})
    assert cols[:3] == ["rank", "name", "result_when"], "result_when is computed but never placed beside the name"
    assert "ds_view_columns(" in src[i:], "the columns are built before result_when exists, so it can never show"
    # And it must render with a clean header + a tooltip that explains the due-vs-ago distinction.
    assert '"result_when": "🆕 Results"' in src
    from ui.ui_scanner import _SCANNER_HEADER_TIPS
    assert "not yet declared" in _SCANNER_HEADER_TIPS.get("result_when", ""), (
        "the result_when tip must explain that 'due' means the result is NOT yet declared")


# ── D48/D49 surfaced in the Technical view (2026-08-30) ──────────────────────────────────────
def test_technical_view_pairs_each_verdict_with_its_number():
    """Breakout Readiness and Momentum Quality were computed, Reference-documented, and shown
    NOWHERE (measured live: 🎯 IMMINENT 11.4%, 🔥 OVERHEATED 6.9% — a warning nothing displayed).
    The design contract: each categorical VERDICT sits directly AFTER the number it interprets —
    d49 after momentum_score, d48 after breakout_score — so 82 and 🎯 IMMINENT land in one glance.
    Order is the UX; a reshuffle that keeps both columns but breaks adjacency fails here."""
    src = _app_src()
    i = src.index('"📈 Technical": ["rank"')
    import ast
    view = ast.literal_eval(src[src.index("[", i):src.index("]", i) + 1])
    for verdict, number in [("d49_momentum_quality", "momentum_score"),
                            ("d48_breakout_readiness", "breakout_score")]:
        assert verdict in view, f"{verdict} vanished from the Technical view — the orphan returns"
        assert view.index(verdict) == view.index(number) + 1, (
            f"{verdict} must sit directly after {number} (verdict-beside-its-number)")
    # Clean headers + tooltips: both render as TextColumns with scanner tips.
    assert '"d48_breakout_readiness": "Readiness"' in src
    assert '"d49_momentum_quality": "Mom. Quality"' in src
    from ui.ui_scanner import _SCANNER_HEADER_TIPS
    for col in ("d48_breakout_readiness", "d49_momentum_quality"):
        assert len(_SCANNER_HEADER_TIPS.get(col, "")) >= 60, f"{col} header tooltip missing/thin"


def test_d48_d49_actionable_states_carry_their_glyphs():
    """The rare-glyph rule: emoji ONLY on the states a scanner acts on (🎯 IMMINENT, 🔥 OVERHEATED,
    ⚡ HIGH — 7-25%% each), plain text on the common rest — so the glyphs stay signals, not static.
    Engine labels and Reference entries must agree exactly."""
    import io as _io, os
    eng = _io.open(os.path.join(os.path.dirname(__file__), "..", "core", "data_engine.py"),
                   encoding="utf-8").read()
    assert '["🎯 IMMINENT", "NEAR", "FAR"]' in eng
    assert '["🔥 OVERHEATED", "⚡ HIGH", "WEAK"]' in eng
    from ui.ui_reference_data import CONCEPT_REFERENCE
    assert [l for l, _ in CONCEPT_REFERENCE["🎯 Breakout Readiness"]] == ["🎯 IMMINENT", "NEAR", "FAR"]
    assert [l for l, _ in CONCEPT_REFERENCE["⚡ Momentum Quality"]] == ["🔥 OVERHEATED", "⚡ HIGH", "WEAK"]


def test_breakout_sort_exists_and_its_column_is_visible():
    """Breakout ↓ was admitted on FORWARD EVIDENCE (2026-09-19), not taste: a full-column IC sweep
    over both available windows ranked breakout_score the most consistent predictor in the engine
    (+0.150 then +0.172, best "weaker window" figure of 400 numeric columns, holding through a
    regime change), against composite_score at rank 88 (+0.134 -> +0.034).

    It also has to satisfy the SORT-BY-VISIBLE doctrine — you cannot order a table by a column the
    reader cannot see — so this pins both halves together. The rank correlation with
    momentum_score is +0.88, which looks redundant and is not: the heads barely intersect
    (top-25 overlap 5/25), the mirror of the Fair Value trap where Pearson said orthogonal while
    the selected sets overlapped 86%.
    """
    import os
    src = open(os.path.join(os.path.dirname(__file__), "..", "app.py"), encoding="utf-8").read()
    i = src.index("_DS_SORTS = {")
    block = src[i:i + 2600]
    assert "Breakout ↓" in block, "the Breakout sort option vanished from _DS_SORTS"
    assert '("breakout_score", False)' in block, (
        "Breakout ↓ must map to breakout_score DESCENDING — higher score = closer to breakout, "
        "and the measured rank-IC is positive in both windows"
    )
    # sort-by-visible: the column must appear in a Deep Scanner preset, or the sort is unreachable
    j = src.index('"📈 Technical":')
    view = src[j:j + 500]
    assert "breakout_score" in view, (
        "breakout_score is sortable but not shown in the Technical view — the sort-by-visible "
        "doctrine: a reader cannot order a table by a column that is not on it"
    )
