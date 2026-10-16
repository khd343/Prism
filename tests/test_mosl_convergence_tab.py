"""
test_mosl_convergence_tab.py
============================
Contract for the Market Pulse → 🔭 MOSL tab: how many of the 10 Motilal Oswal Wealth Creation
lenses (studies 16–30) a stock clears at once.

WHY THIS VIEW IS LEGITIMATE WHERE A GENERIC ONE IS NOT. A count across all 37 frameworks would be
meaningless: gate strictness varies BY DESIGN and the code says so — Dhandho fires on 1 stock
because Pabrai Ch.10 mandates "Few Bets, Big Bets, Infrequent Bets", CAN SLIM on 2 because
O'Neil's all-seven rule is that strict, while Schilit Clean fires on 62.5% because it is a
cleanliness check rather than a screen. Putting those side by side invites the reader to compare
numbers that are not comparable. The MOSL lenses come from ONE research programme, so agreement
between them means something.

THE TRAP THIS FILE EXISTS FOR — and it caught the author first. `frameworks_passed` is a
", "-joined string, and **"QGLP" is a SUBSTRING of "SQGLP Century Stock"**. A first measurement
used `.str.contains` and pulled 37 extra stocks into the cohort, which then produced WRONG caveat
figures (5 median flags and 83.6% AVOID instead of the true 6 and 80.6%). Exact-token splitting is
not a style preference here; it is the difference between right and wrong numbers.

THE SIGNAL IS REAL, and was checked before the tab was built:
    convergence distribution is a clean pyramid  — 0:999 … 4:99 … 8:1
    corr(convergence, composite_score) = 0.479   — informative, nowhere near redundant

THE CAVEAT IS LOAD-BEARING. The 4+ cohort carries a MEDIAN OF 6 RED FLAGS against the universe's
5 — slightly worse, not better — and 80.6% of it is AVOID. These lenses gate quality, growth and
longevity; none reads the forensics. The tab says so in its caption rather than a tooltip, and
these tests pin that it keeps saying so.

Run with: pytest tests/test_mosl_convergence_tab.py -v
"""

import contextlib
import io as _io
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import pandas as pd
import pytest

from data_engine import (coerce_numeric_columns, compute_derived_signals, load_all_csvs,
                         merge_datasets)

_APP = os.path.join(os.path.dirname(__file__), "..", "app.py")


@pytest.fixture(scope="module")
def src():
    return _io.open(_APP, encoding="utf-8").read()


@pytest.fixture(scope="module")
def live():
    from core import run_scoring_pipeline
    with contextlib.redirect_stdout(_io.StringIO()):
        return run_scoring_pipeline(
            compute_derived_signals(coerce_numeric_columns(merge_datasets(load_all_csvs("local")))))


@pytest.fixture(scope="module")
def lenses(src):
    m = re.search(r"_MOSL_LENSES = \[(.*?)\]", src, re.S)
    assert m, "the MOSL lens list is gone"
    return [x.strip().strip('"') for x in m.group(1).replace("\n", " ").split(",") if x.strip()]


def _tokens(live):
    return (live.get("frameworks_passed", pd.Series("", index=live.index))
            .fillna("").astype(str)
            .map(lambda s: {t.strip() for t in re.split(r"\s*,\s*", s) if t.strip()}))


def _convergence(live, lenses):
    return _tokens(live).map(lambda t: sum(1 for m in lenses if m in t))


# -- 1. THE SUBSTRING TRAP ------------------------------------------------------------------
def test_exact_token_matching_not_substring(live, lenses):
    """The defect that produced wrong caveat numbers on the first pass."""
    tok = _tokens(live)
    fp = live["frameworks_passed"].fillna("").astype(str)
    exact = int(tok.map(lambda t: "QGLP" in t).sum())
    substr = int(fp.str.contains("QGLP", regex=False).sum())
    assert substr > exact, (
        "the substring/token gap has closed, so this probe no longer demonstrates the hazard -- "
        "check whether SQGLP Century Stock still exists before relaxing anything"
    )
    assert exact == int((live["qglp_pass"] == 1).sum()), (
        f"exact-token QGLP ({exact}) disagrees with the authoritative qglp_pass column "
        f"({int((live['qglp_pass'] == 1).sum())}) -- the parsing is wrong"
    )


def test_the_tab_splits_on_the_comma_boundary(src):
    """Structural: the tab must parse tokens, never test substrings."""
    i = src.index("_MOSL_LENSES")
    block = src[i:i + 1400]
    assert "re.split" in block, "the MOSL tab no longer splits frameworks_passed into tokens"
    assert ".str.contains" not in block, (
        "the MOSL tab is substring-matching again. 'QGLP' is a substring of 'SQGLP Century Stock' "
        "and this silently inflates the cohort by 37 stocks."
    )


# -- 2. The signal that justified building it ------------------------------------------------
def test_convergence_still_discriminates(live, lenses):
    n = _convergence(live, lenses)
    assert n.max() >= 5, f"deepest agreement is only {n.max()} lenses -- the view has lost its point"
    share_multi = (n >= 2).mean()
    assert 0.10 < share_multi < 0.60, (
        f"{share_multi:.1%} of the universe clears 2+ lenses; outside this band the tab is either "
        f"empty or indiscriminate"
    )


def test_convergence_is_not_just_the_composite_score(live, lenses):
    """If it tracked the score it would add nothing over what the tearsheet already shows; if it tracked
    nothing it would be noise. Since 2026-10-15 the two halves are judged against different things: the
    default composite (Momentum + Breakout) leaves quality out of its blend, so corr with it fell to 0.126
    on the local data (0.479 under Hybrid when the tab was built, 0.294 under thirds) -- LESS redundant,
    not noisier. Noise is judged against quality_score, which the lenses gate: 0.589, the same under
    every mode."""
    conv = _convergence(live, lenses)
    c = conv.corr(live["composite_score"])
    assert c < 0.85, f"corr(convergence, composite) = {c:.3f}: near 1 means the tab repeats the score"
    q = conv.corr(live["quality_score"])
    assert q > 0.15, f"corr(convergence, quality_score) = {q:.3f}: near 0 means the lenses measure noise"


def test_every_listed_lens_actually_fires(live, lenses):
    """A lens that never appears would silently cap the achievable count."""
    tok = _tokens(live)
    dead = [m for m in lenses if not tok.map(lambda t, m=m: m in t).any()]
    assert not dead, f"these MOSL lenses never fire, so they only dilute the count: {dead}"


# -- 3. The caveat must survive, and stay true -----------------------------------------------
def test_the_caption_warns_that_convergence_is_not_safety(src):
    i = src.index("_MOSL_LENSES")
    block = src[i - 2500:i + 4000]
    low = block.lower()
    assert "agreement, not safety" in low, "the caveat headline is gone"
    assert "forensic" in low, "the caveat no longer says these lenses skip the forensics"
    assert "conviction" in low, "the caveat no longer distinguishes convergence from conviction"


def test_the_caveats_numbers_are_still_true(live, lenses):
    """SELF-VERIFYING, the pattern used elsewhere in this suite: the claim is that convergence does
    NOT imply clean books. If the high-convergence cohort ever became materially cleaner than the
    universe, the warning would be misleading and must be rewritten."""
    n = _convergence(live, lenses)
    hi = live[n >= 4]
    assert len(hi) > 30, f"only {len(hi)} stocks clear 4+ lenses -- too few to judge"
    assert hi["red_flag_count"].median() >= live["red_flag_count"].median(), (
        f"the 4+ cohort now carries FEWER red flags ({hi['red_flag_count'].median():.0f}) than the "
        f"universe ({live['red_flag_count'].median():.0f}). The caption warns the opposite; "
        f"remeasure and rewrite it."
    )
    avoid = (hi["verdict_direction"].astype(str) == "FLAWED").mean()
    assert avoid > 0.5, (
        f"only {avoid:.1%} of the high-convergence cohort is AVOID; the caveat's premise has "
        f"changed and the wording needs revisiting"
    )


# -- 4. Layout: do not repeat the clipping defect --------------------------------------------
def test_the_table_stays_narrow(src):
    """The QGLP tab shipped 13 columns and showed ~8, hiding its own price leg. This tab was built
    deliberately narrow; a contract keeps it that way."""
    i = src.index("_m_cols = [c for c in [")
    block = src[i:src.index("]", i + 20) + 1]
    cols = re.findall(r'"([a-z_]+)"', block)
    assert len(cols) <= 8, f"the MOSL table has grown to {len(cols)} columns: {cols}"
    assert "mosl_n" in cols and "verdict_direction" in cols and "red_flag_count" in cols, (
        f"the convergence count and its two risk columns are the point of this table: {cols}"
    )


def test_sectors_tab_still_has_its_own_index(src):
    """Adding a tab shifts every index after it -- an off-by-one here silently renders the wrong
    body into the wrong tab.

    The label count is read from the st.tabs([...]) list itself rather than matched against a
    hardcoded set of tab NAMES. The first version alternated on Tsunami|QGLP|MOSL|Wealth|Sectors
    inside a 220-char window, which meant it could not see a newly added tab at all: appending
    🏭 Industry (2026-08-28) left it counting 5 declared tabs against 6 bodies and failing on a
    change that was correct. A guard that cannot survive the event it guards against is worse than
    no guard -- it trains you to edit it."""
    assert "_mp_tabs[3]" in src, "the Sectors tab was not reindexed after MOSL was inserted"
    _i = src.index("_mp_tabs = st.tabs([")
    n_tabs = len(re.findall(r'"[^"]+"', src[_i:src.index("])", _i)]))
    assert n_tabs >= 5, f"only {n_tabs} tab labels parsed -- the extractor lost its teeth"
    used = {int(x) for x in re.findall(r"_mp_tabs\[(\d)\]", src)}
    assert used == set(range(n_tabs)), f"tab bodies {sorted(used)} do not cover the {n_tabs} tabs declared"
