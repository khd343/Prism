"""
test_price_trend_inputs.py
==========================
Contract for the derived price inputs the 🚀 Playbook and the Momentum score read (2026-10-15).

WHY. They used to be computed inline in core/data_engine.compute_derived_signals, which the 🚀 recipe fingerprint
does not follow (it computes hundreds of columns). So the V-Stop unit fix of 2026-10-15 changed the list while the
fingerprint stayed da869c1c0a1c43d7, and neither Kite's check nor the 🚀 card could see it. Now they come from ONE
function, core.data_engine.price_trend_inputs, assigned once and listed in the 🚀 scan's fingerprint `inputs`.

Run with: pytest tests/test_price_trend_inputs.py -v
"""
import ast
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from core import data_engine as de  # noqa: E402
from ui import ui_scans as S  # noqa: E402

FIVE = ("vol_ratio", "vstop_fresh", "above_sma200", "vstop_green", "dist_to_vstop")


def _raw(**cols):
    base = dict(volume=[100.0], vol_sma_20d=[200.0], close_price=[110.0], sma_200d=[100.0],
                vstop_value=[100.0], last_vstop_change=[2.0])
    base.update(cols)
    return pd.DataFrame(base)


def test_each_input_from_the_raw_columns():
    out = de.price_trend_inputs(_raw())
    assert out.columns.tolist() == list(FIVE)
    assert out.loc[0, "vol_ratio"] == 0.5
    assert out.loc[0, "above_sma200"] == 1 and out.loc[0, "vstop_green"] == 1
    assert out.loc[0, "dist_to_vstop"] == 10.0


def test_a_missing_average_volume_gives_no_ratio():
    for v in (0.0, np.nan):
        assert np.isnan(de.price_trend_inputs(_raw(vol_sma_20d=[v])).loc[0, "vol_ratio"])


def test_an_off_scale_stop_counts_as_missing():
    """MOTHERSON once carried a stop of 7684 against a 130 close (paise): no green, no distance."""
    out = de.price_trend_inputs(_raw(close_price=[130.0], vstop_value=[7684.0]))
    assert out.loc[0, "vstop_green"] == 0 and np.isnan(out.loc[0, "dist_to_vstop"])


def test_the_five_are_assigned_once_and_only_from_price_trend_inputs():
    """A later `df["vstop_fresh"] = ...` elsewhere would change the list without changing the fingerprint."""
    seen = {k: [] for k in FIVE}
    for folder in ("core", "ui"):
        for name in sorted(os.listdir(os.path.join(ROOT, folder))):
            if not name.endswith(".py"):
                continue
            tree = ast.parse(open(os.path.join(ROOT, folder, name), encoding="utf-8").read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == "df"
                                and isinstance(t.slice, ast.Constant) and t.slice.value in FIVE):
                            seen[t.slice.value].append((name, ast.unparse(node.value)))
    for k in FIVE:
        assert seen[k] == [("data_engine.py", f"_pti['{k}']")], (k, seen[k])


def test_the_playbook_fingerprint_follows_price_trend_inputs(monkeypatch):
    pb = next(s for s in S.SCANS if s["key"] == "playbook")
    assert de.price_trend_inputs in pb["inputs"]
    before = S.recipe_fingerprint(pb)
    monkeypatch.setattr(de, "VSTOP_FRESH_MAX_WEEKS", de.VSTOP_FRESH_MAX_WEEKS + 1)
    assert S.recipe_fingerprint(pb) != before
