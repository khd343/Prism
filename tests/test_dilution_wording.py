"""
test_dilution_wording.py — the screen says only what PRISM knows about a share-count jump (2026-10-09).

Measured that day (docs/known-issues.md, the dilution entry): PRISM's six tabs cannot tell a bonus or a
rights issue from a placement. Restated EPS and "clean" ratios both failed against public records:
CONCOR's 25% jump is a real 1:4 bonus, Swiss Military's and MITCON's are rights issues, two more have no
corporate action on record. So the penalty stays — a jump of more than 10% below 1.5x is Tier 3 — but the
WORDS called every such jump "predatory" and "a QIP", which is a claim the data cannot make. These pins
keep the wording to what is known: the size of the jump, and the three things it could be.

Run with: pytest tests/test_dilution_wording.py -v
"""
import ast
import glob
import os
import re
import sys

import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

_POSSIBILITIES = ("placement", "rights issue", "bonus")
_CLAIMS = re.compile(r"\bpredatory\b|\bQIP\b", re.I)


def test_the_tier_3_flag_line_names_every_possibility_and_claims_none():
    from ui.ui_tearsheet import _get_flag_context
    for row in (pd.Series({"dilution_flag": 3, "shares_gr_yoy": 25.0}),
                pd.Series({"dilution_flag": 3, "shares_gr_yoy": float("nan")})):
        text = _get_flag_context(row, "rf_dilution")
        assert not _CLAIMS.search(text), text
        assert all(p in text.lower() for p in _POSSIBILITIES), text
    assert "25.0%" in _get_flag_context(pd.Series({"dilution_flag": 3, "shares_gr_yoy": 25.0}), "rf_dilution"), (
        "the measured jump stays on screen — only its interpretation changed")


def test_the_tier_2_line_is_unchanged():
    from ui.ui_tearsheet import _get_flag_context
    assert "Tier 2 (3–10%)" in _get_flag_context(pd.Series({"dilution_flag": 2, "shares_gr_yoy": 5.0}), "rf_dilution")


def test_the_reference_gate_entry_names_every_possibility_and_claims_none():
    from ui.ui_reference_data import CONCEPT_REFERENCE
    entries = [t for rows in CONCEPT_REFERENCE.values() for k, t in rows if k == "No Dilution (gate)"]
    assert len(entries) == 1
    text = entries[0]
    assert not _CLAIMS.search(text), text
    assert all(p in text.lower() for p in _POSSIBILITIES), text
    assert "10%" in text and "1.5" in text, "the two lines the engine actually draws"


def test_the_glossary_says_a_bonus_below_1_5x_still_counts():
    from ui.ui_components import _RAW_GLOSSARY
    text = _RAW_GLOSSARY["Dilution Flag"].lower()
    assert "below 1.5x" in text and "bonus" in text and "rights issue" in text and "can't tell" in text


def _screen_strings(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
            and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docs]


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(ROOT, "ui", "*.py")))
                         + [os.path.join(ROOT, "app.py")], ids=os.path.basename)
def test_no_screen_text_calls_a_share_jump_predatory_or_a_qip(path):
    bad = [s for s in _screen_strings(path) if _CLAIMS.search(s)]
    assert not bad, f"{os.path.basename(path)}: {bad}"
