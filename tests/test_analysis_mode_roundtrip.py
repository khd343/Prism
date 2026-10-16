"""
test_analysis_mode_roundtrip.py — the Analysis Mode selectbox survives a tab that outlived a deploy (2026-10-16).

PRODUCTION CRASH, 2026-10-16: "KeyError: '⚡ Quantamental + Breakout · default'" at app.py's mode caption, in browser
tabs opened before the 15 Oct deploy that moved the default from Breakout to Momentum + Breakout. Streamlit 1.54 syncs a
selectbox by its DISPLAY label and hands back a label it cannot match as it is (SelectboxSerde.deserialize); Breakout's
label had lost its " · default" mark, so the tab's re-sent label matched no option and reached session_state raw.
app.py reads cfg_mode at the TOP of the script, before the selectbox can check it, scored with it, and indexed
ANALYSIS_MODES with it. A fresh tab was never affected (checked on the live app the same day).

The fix is the label round-trip rule the sidebar already follows (ui_discovery.recover_labels, 2026-09-22;
ui_liquidity.step_value, 2026-09-21): config.mode_value turns whatever session state holds into a real mode key before
anything reads it — a key passes through, the label of any mode, with or without the default mark, recovers its key
(recover, do not drop), and anything else reads as the default. config.mode_label is the selectbox's ONE label
function, so the recovery cannot drift from the label it reverses.

Run with: pytest tests/test_analysis_mode_roundtrip.py -v
"""
import ast
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

import config  # noqa: E402

_APP = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
_TREE = ast.parse(_APP)
_STALE = "⚡ Quantamental + Breakout · default"     # the exact value in the 16 Oct traceback


# ── 1. The resolver ──────────────────────────────────────────────────────────────────────────────────
def test_a_mode_key_passes_through():
    for k in config.ANALYSIS_MODES:
        assert config.mode_value(k) == k


def test_any_modes_label_recovers_its_key_with_or_without_the_default_mark():
    for k, m in config.ANALYSIS_MODES.items():
        assert config.mode_value(m["label"]) == k
        assert config.mode_value(m["label"] + " · default") == k
        assert config.mode_value(config.mode_label(k)) == k


def test_the_label_in_the_16_oct_traceback_recovers_breakout():
    assert config.ANALYSIS_MODES["Breakout"]["label"] + " · default" == _STALE, "the traceback's label, as it was"
    assert config.mode_value(_STALE) == "Breakout"


@pytest.mark.parametrize("stored", [None, 42, [], ["Breakout"], "", "Nope", "⚡ Quantamental + Breakout · old"])
def test_anything_else_reads_as_the_default_and_never_raises(stored):
    assert config.mode_value(stored) == config.DEFAULT_ANALYSIS_MODE


def test_only_the_default_carries_the_mark_and_every_label_is_distinct():
    assert [k for k in config.ANALYSIS_MODES if config.mode_label(k).endswith(" · default")] == \
        [config.DEFAULT_ANALYSIS_MODE]
    labels = [m["label"] for m in config.ANALYSIS_MODES.values()]
    assert len(set(labels)) == len(labels), "two modes sharing a label would make the recovery ambiguous"


# ── 2. app.py resolves the stored mode before anything reads it ──────────────────────────────────────
def _top_level(pred) -> int:
    hits = [i for i, n in enumerate(_TREE.body) if pred(n)]
    assert len(hits) == 1, hits
    return hits[0]


def _mode_read_block() -> list:
    """app.py's top-level statements from seeding cfg_mode to reading it into analysis_mode, as written."""
    start = _top_level(lambda n: 'st.session_state.setdefault("cfg_mode"' in ast.unparse(n).replace("'", '"'))
    end = _top_level(lambda n: isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "analysis_mode")
    assert start < end
    return _TREE.body[start:end + 1]


def _mode_selectbox() -> ast.Call:
    hits = [c for c in ast.walk(_TREE) if isinstance(c, ast.Call) and ast.unparse(c.func) == "st.selectbox"
            and any(k.arg == "key" and isinstance(k.value, ast.Constant) and k.value.value == "cfg_mode"
                    for k in c.keywords)]
    assert len(hits) == 1, "app.py's Analysis Mode selectbox"
    return hits[0]


def test_the_selectbox_label_is_the_one_mode_label():
    fmt = next(k.value for k in _mode_selectbox().keywords if k.arg == "format_func")
    assert ast.unparse(fmt) == "mode_label"


def test_app_resolves_the_stored_mode_before_reading_it():
    block = [ast.unparse(n).replace("'", '"') for n in _mode_read_block()]
    assert 'st.session_state["cfg_mode"] = mode_value(st.session_state["cfg_mode"])' in block


# ── 3. The crash itself, from app.py's own lines ─────────────────────────────────────────────────────
def _harness() -> str:
    """app.py's mode lines run in order: the seed-to-read block, the selectbox, the caption at its old line 3346."""
    caption = [n for n in ast.walk(_TREE) if isinstance(n, ast.Expr)
               and ast.unparse(n) == "st.caption(ANALYSIS_MODES[analysis_mode]['description'])"]
    assert len(caption) == 1
    body = _mode_read_block() + [ast.Expr(_mode_selectbox()), caption[0]]
    return "import streamlit as st\nfrom config import *\n" + "\n".join(ast.unparse(n) for n in body)


def _run(tmp_path, stored=None):
    from streamlit.testing.v1 import AppTest
    script = tmp_path / "mode_app.py"
    script.write_text(_harness(), encoding="utf-8")
    at = AppTest.from_file(str(script), default_timeout=30)
    if stored is not None:
        at.session_state["cfg_mode"] = stored
    at.run()
    return at


def test_a_tab_that_outlived_the_deploy_keeps_its_mode_instead_of_crashing(tmp_path):
    at = _run(tmp_path, _STALE)
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["cfg_mode"] == "Breakout"
    assert at.caption[0].value == config.ANALYSIS_MODES["Breakout"]["description"]


def test_a_fresh_tab_opens_on_the_default(tmp_path):
    at = _run(tmp_path)
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["cfg_mode"] == config.DEFAULT_ANALYSIS_MODE
