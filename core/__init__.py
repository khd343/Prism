"""
Core Execution Engines
======================
Exposes the primary data, scoring, and forensic engines.
"""

from .data_engine import fetch_and_clean_data, load_market
from .forensic_engine import run_forensic_analysis, compute_forensic_signals, apply_forensic_penalty
from .scoring_engine import run_full_scoring
from .verdict_engine import compute_verdict


from config import DEFAULT_ANALYSIS_MODE


def run_scoring_pipeline(clean_df, analysis_mode: str = DEFAULT_ANALYSIS_MODE,
                         scoring_profile: str = "Balanced"):
    """Canonical 4-step scoring sequence — the ONE locked order (CLAUDE.md §5), never reorder:

        1. compute_forensic_signals  → Piotroski + red flags + Schilit (5 framework gates read these)
        2. run_full_scoring          → quality/momentum/composite + all framework flags
        3. apply_forensic_penalty    → cascading multiplier on composite_score + tier reassignment
        4. compute_verdict           → display-only decision synthesis (verdict_* columns)

    Single source of truth: app.get_scored_data and tools/snapshot.py both call this so the
    sequence can never silently diverge or be reordered. Order + data-threading are pinned by
    tests/test_pipeline_order.py.
    """
    df = compute_forensic_signals(clean_df)
    df = run_full_scoring(df, analysis_mode, scoring_profile)
    df = apply_forensic_penalty(df)
    df = compute_verdict(df)
    return df


__all__ = [
    "fetch_and_clean_data",
    "load_market",
    "run_forensic_analysis",
    "compute_forensic_signals",
    "apply_forensic_penalty",
    "run_full_scoring",
    "compute_verdict",
    "run_scoring_pipeline",
]
