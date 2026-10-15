"""Docs drift from the engine SILENTLY — pin the vocabulary the same way the code is pinned.

Measured 2026-09-09: docs/handbook/ had drifted from the engine in thirteen days and nothing
said so. The COUNTS were all correct (28 red flags, 37 frameworks); the NAMING had rotted —
ch07 documented the headline call as BUY/WATCH/AVOID after the 2026-08-27 rename to
SOUND/MIXED/FLAWED (itself pinned, for code, by tests/test_verdict_vocabulary.py), and the
whole wealth ladder was undocumented while ch07 used that layer's action words for a different
layer. Three defects, none caught by 2731 green tests, because no test read the prose.

Scope is docs/handbook/ deliberately. docs/user-guide/ carries the same drift (02-discovery-tab
and 10-faq still present BUY/WATCH/AVOID as current; two files describe the display-retired
verdict_strength as live) but that prose is the user's to edit — logged in docs/known-issues.md
instead of enforced here.

docs/ is gitignored (private handover), so these skip when it is absent — same pattern as the
stockscans_sync boundary pin.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import config

HANDBOOK = Path(__file__).resolve().parent.parent / "docs" / "handbook"

pytestmark = pytest.mark.skipif(
    not HANDBOOK.is_dir(), reason="docs/ is gitignored — private handover, not in a fresh clone"
)

# Words the verdict layer STOPPED using on 2026-08-27. The wealth ladder then spoke them, until its own
# rename on 2026-10-15; no layer uses them now, so presenting them as current is drift either way.
RETIRED_VERDICT = re.compile(r"BUY\s*/\s*WATCH\s*/\s*AVOID")
# The wealth ladder's words until 2026-10-15 (now CLIMBING★ … STALLED). The starred two are unambiguous.
RETIRED_WEALTH = re.compile(r"BUY★|WATCH★")
# A line may name the old words if it is plainly marking them as history.
HISTORICAL = re.compile(r"until 2026|were called|previously|renamed|old (?:words|strings)", re.I)


def _lines():
    for md in sorted(HANDBOOK.glob("*.md")):
        for n, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            yield md.name, n, line


def _text():
    return "\n".join(line for _f, _n, line in _lines())


def test_retired_verdict_vocabulary_is_never_presented_as_current():
    offenders = [f"{f}:{n}  {line.strip()[:90]}"
                 for f, n, line in _lines()
                 if RETIRED_VERDICT.search(line) and not HISTORICAL.search(line)]
    assert not offenders, (
        "the verdict reads SOUND/MIXED/FLAWED since 2026-08-27; these lines still teach the "
        "old words as current:\n  " + "\n  ".join(offenders))


def test_retired_wealth_vocabulary_is_never_presented_as_current():
    offenders = [f"{f}:{n}  {line.strip()[:90]}"
                 for f, n, line in _lines()
                 if RETIRED_WEALTH.search(line) and not HISTORICAL.search(line)]
    assert not offenders, (
        "the wealth ladder reads CLIMBING★ … STALLED since 2026-10-15; these lines still teach the "
        "old words as current:\n  " + "\n  ".join(offenders))


def test_current_verdict_vocabulary_is_documented_where_the_verdict_lives():
    """The rename is only half done if the new words never appear — and "somewhere in the
    handbook" is too low a bar: they must be in the chapter that DEFINES the verdict."""
    ch = (HANDBOOK / "07-the-verdict.md").read_text(encoding="utf-8")
    for word in ("SOUND", "MIXED", "FLAWED"):
        assert re.search(rf"\b{word}\b", ch), f"{word} is undocumented in 07-the-verdict.md"


def test_documented_red_flag_count_matches_the_engine():
    """FORENSIC_MAX_FLAGS is bumped by hand; the prose quoting it must move with it."""
    quoted = {int(m) for _f, _n, line in _lines()
              for m in re.findall(r"(\d+)\s+red[- ]flags?\b", line, re.I)}
    wrong = quoted - {config.FORENSIC_MAX_FLAGS}
    assert not wrong, (
        f"handbook quotes {sorted(wrong)} red flags; config.FORENSIC_MAX_FLAGS is "
        f"{config.FORENSIC_MAX_FLAGS}")


def test_wealth_ladder_is_documented():
    """A whole scoring layer went undocumented for two weeks — pin that it stays covered.
    Matched as a SECTION HEADING, not a substring: "Wealth Ladderr" contains "Wealth Ladder"."""
    text = _text()
    assert re.search(r"^##\s+Wealth Ladder\b", text, re.M), (
        "the wealth ladder has no section heading in the handbook")
    from core.verdict_engine import WEALTH_NA, WEALTH_TIERS
    for tier in (*WEALTH_TIERS, WEALTH_NA):
        assert tier in text, f"wealth tier {tier!r} is undocumented"
