"""
Shariah screen — ONE display vocabulary for every surface (2026-10-16).

The three words come from config.SHARIAH_STATUSES (core/shariah.py writes them); this module gives each its icon
and colour once, holds the equality filter and its live counts, and words each rule's limit from config. Pure —
no Streamlit — so the stateless Tear-Sheet can use it. The one WIDGET both tabs draw lives in ui_discovery
(shariah_filter), beside the label-recovery machinery it needs.

    ✅ Shariah       green   passes every rule PRISM can check
    ⛔ Non-Shariah   red     a prohibited business, or a proven fail on a rule
    📝 Review        blue    nothing visible fails; your annual-report check or your scholar settles it
    ❔ Doubtful      amber   a figure is missing — PRISM cannot screen it fully
Only Shariah is ever green.

Icons, checked against what PRISM already says with them (2026-10-16): ✅ means "passed" (74 uses) and ❔ means
"unknown, no data" in the Discovery filters — the same meanings here; ⛔ appeared once, for "excluded"; 📝 was
unused. 🟢🟡🔴 are the Soundness verdict's own emoji, so none of them is used. No religious symbols.

EQUALITY, NEVER CONTAINS: "Shariah" sits inside "Non-Shariah", so a substring match would return both
(tests/test_shariah_ui.py, tests/test_shariah_screen.py).
"""
import html

import pandas as pd

from config import COLORS, SHARIAH_RULE_STATES, SHARIAH_RULES, SHARIAH_STATUSES

SHARIAH_ALL = "All"
SHARIAH_FILTER_OPTIONS = (SHARIAH_ALL, *SHARIAH_STATUSES)
SHARIAH_ICONS = dict(zip(SHARIAH_STATUSES, ("✅", "⛔", "📝", "❔")))
SHARIAH_COLORS = dict(zip(SHARIAH_STATUSES, (COLORS["green"], COLORS["red"], COLORS["blue"], COLORS["gold"])))
SHARIAH_RESULTS = dict(zip(SHARIAH_RULE_STATES, ("✅ Pass", "⛔ Fail", "📝 Review", "❔ Unknown")))
SHARIAH_RESULT_COLORS = dict(zip(SHARIAH_RULE_STATES, (COLORS["green"], COLORS["red"], COLORS["blue"],
                                                       COLORS["gold"])))
SHARIAH_FOOTNOTE = ("A screen from financial data, not a fatwa. Interest income is not in the data: check the "
                    "annual report and purify dividends and gains. Confirm with your scholar.")


def shariah_label(status) -> str:
    """'✅ Shariah' — icon and word. 'All' stays 'All'; anything else (blank, unknown) is ''."""
    if status == SHARIAH_ALL:
        return SHARIAH_ALL
    return f"{SHARIAH_ICONS[status]} {status}" if status in SHARIAH_ICONS else ""


def shariah_counts(frame: pd.DataFrame) -> dict:
    """Live count per filter option, by EQUALITY; 'All' counts every row."""
    vc = frame["shariah_status"].value_counts() if "shariah_status" in frame.columns else pd.Series(dtype=int)
    return {SHARIAH_ALL: len(frame), **{s: int(vc.get(s, 0)) for s in SHARIAH_STATUSES}}


def shariah_mask(frame: pd.DataFrame, pick) -> pd.Series:
    """The rows a pick keeps, by EQUALITY: 'All' keeps every row; a frame without the screen keeps none."""
    if pick == SHARIAH_ALL:
        return pd.Series(True, index=frame.index)
    if "shariah_status" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame["shariah_status"].eq(pick)


def shariah_pill_html(status) -> str:
    """The status as a small pill — colour WITH icon and word; '' when there is no status."""
    if status not in SHARIAH_ICONS:
        return ""
    c = SHARIAH_COLORS[status]
    return (f'<span style="font-size:0.67rem;font-weight:700;padding:2px 10px;border-radius:12px;'
            f'white-space:nowrap;background:{c}1f;color:{c};border:1px solid {c}66;">'
            f'{html.escape(shariah_label(status))}</span>')


def rule_limit_text(key: str) -> str:
    """A rule's limit in words, its number read from config.SHARIAH_RULES at call time."""
    limit = SHARIAH_RULES[key]["limit"]
    pct = f"{limit * 100:.0f}%" if limit is not None else ""
    return {"business": "not a prohibited business",
            "income": f"at most {pct} of revenue",
            "debt": f"at most {pct} of total assets",
            "illiquid": f"at least {pct} of total assets",
            "nla": "at most the market cap",
            "investments": f"below {pct} of total assets"}[key]
