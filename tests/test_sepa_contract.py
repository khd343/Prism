"""
Contract Tests — Mark Minervini SEPA Momentum Framework (Framework 21)
======================================================================
Automated verification that docs/sepa_momentum_specs.json, core/data_engine.py,
core/scoring_engine.py, and ui/ui_tearsheet.py are in perfect alignment.

7-pillar architecture (decoupled like Malik / Lynch / Mauboussin / CAN SLIM):
  T — sepa_trend_template  (d45_trend_structure >= 4)          HARD_GATE
  A — sepa_adx_confirmed   (adx_14w >= 20)                     HARD_GATE
  L — sepa_low_base        (dist_52wl >= 30)                   HARD_GATE
  R — sepa_rs_confirmed    (crs_aligned == 1)                  HARD_GATE
  E — sepa_earnings_fuel   (eps>=25 & rev>=20 & roe>=17)       HARD_GATE
  I — sepa_institutional   (change_fii_lq>0 OR change_dii_lq>0) HARD_GATE
  V — sepa_vcp_dryup       (vol_sma_10d < 70% of vol_sma_50d)  SCORE_BONUS (not in pass)

Key invariants:
  • sepa_pass = 6 hard gates AND sepa_near_high (Trend Template C7) AND sepa_mcap_ok (>= ₹500 Cr)
  • sepa_score = 7 pillars (incl. VCP bonus); range 0-7
  • sepa_pass==1  ⇒  sepa_score in {6, 7}   (VCP is the only optional point)
  • sepa_score==7 does NOT guarantee sepa_pass (mcap gate may fail)
  • VCP mirrors can_slim_vcp: score bonus only, NEVER inside the boolean pass chain
"""

import json
import re
import os
import sys
import pytest
import pandas as pd
import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SPEC_PATH = os.path.join(REPO_ROOT, "docs", "sepa_momentum_specs.json")
SE_PATH   = os.path.join(REPO_ROOT, "core", "scoring_engine.py")
DE_PATH   = os.path.join(REPO_ROOT, "core", "data_engine.py")
UI_PATH   = os.path.join(REPO_ROOT, "ui",   "ui_tearsheet.py")
APP_PATH  = os.path.join(REPO_ROOT, "app.py")
TOKEN_TEST_PATH = os.path.join(REPO_ROOT, "tests", "test_framework_token_boundary.py")


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def spec() -> dict:
    with open(SPEC_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def se_source() -> str:
    with open(SE_PATH, encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def de_source() -> str:
    with open(DE_PATH, encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def ui_source() -> str:
    with open(UI_PATH, encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def app_source() -> str:
    with open(APP_PATH, encoding="utf-8") as f:
        return f.read()


# ── Source-block extractors ───────────────────────────────────────────────────

def _fw_sepa_block(src: str) -> str:
    """Extract the fw_sepa block from scoring_engine.py source."""
    start = src.find("# ── Framework 21: fw_sepa")
    assert start != -1, "Cannot find fw_sepa block anchor (Framework 21) in scoring_engine.py"
    end = src.find("# ── Framework 22", start)
    assert end != -1, "Cannot find end boundary (Framework 22) of fw_sepa block"
    return src[start:end]


def _d45_block(src: str) -> str:
    """Extract the d45_trend_structure block from data_engine.py source."""
    start = src.find("# ── D45: Trend Structure Score")
    assert start != -1, "Cannot find d45 block anchor in data_engine.py"
    end = src.find("# ── D47:", start)
    assert end != -1, "Cannot find end boundary (D47) of d45 block"
    return src[start:end]


def _render_sepa_block(src: str) -> str:
    """Extract the render_sepa_radar function body from ui_tearsheet.py."""
    start = src.find("def render_sepa_radar")
    assert start != -1, "Cannot find render_sepa_radar in ui_tearsheet.py"
    nxt = src.find("\ndef ", start + 10)
    return src[start:nxt] if nxt != -1 else src[start:]


def _render_raw_block(src: str) -> str:
    start = src.find("def render_raw_signals")
    assert start != -1, "Cannot find render_raw_signals in ui_tearsheet.py"
    nxt = src.find("\ndef ", start + 10)
    return src[start:nxt] if nxt != -1 else src[start:]


# ── Mock data helpers ─────────────────────────────────────────────────────────

def _build_mock_sepa_row(**overrides) -> dict:
    """Fully-passing SEPA row: all 6 hard gates green + VCP bonus → pass=1, score=7.

    SEPA pillars read pre-computed inputs (materialized in data_engine), so the mock
    supplies them directly — identical pattern to the Lynch / CAN SLIM contract tests.
    """
    base = {
        "d45_trend_structure":  5,     # T: >= 4
        "adx_14w":             25.0,   # A: >= 20
        "dist_52wl":           40.0,   # L: >= 30
        "dist_52wh":           10.0,   # C7: <= 25 (within 25% of 52-week high)
        "crs_aligned":          1,     # R: == 1
        "eps_gr_yoy":          30.0,   # E: >= 25
        "rev_gr_yoy":          25.0,   # E: >= 20
        "roe":                 20.0,   # E: >= 17
        "change_fii_lq":        1.0,   # I: > 0
        "change_dii_lq":        0.0,   # I: dii component
        "vcp_volume_dryup":     1,     # V: bonus active
        "market_cap":        1000.0,   # pass: >= 500
        # scaffold
        "close_price":        250.0,
        "name":               "TestSepaStock",
        "sector":             "Consumer",
        "is_financial":       False,
    }
    base.update(overrides)
    return base


def _run_sepa(rows: list, regime: str = "SIDEWAYS") -> pd.DataFrame:
    from config import MASTER_PROFILES
    from core.scoring_engine import compute_qglp_score
    df = pd.DataFrame(rows)
    df.attrs["detected_market_regime"] = regime
    profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
    return compute_qglp_score(df, profile)


# ═══════════════════════════════════════════════════════════════════════════════
# TestSepaSpecLedger
# ═══════════════════════════════════════════════════════════════════════════════

class TestSepaSpecLedger:
    def test_spec_file_exists(self):
        assert os.path.exists(SPEC_PATH), f"Spec file not found: {SPEC_PATH}"

    def test_spec_is_valid_json(self, spec):
        assert isinstance(spec, dict)

    def test_all_seven_pillar_keys_present(self, spec):
        expected = {
            "sepa_trend_template", "sepa_adx_confirmed", "sepa_low_base",
            "sepa_rs_confirmed", "sepa_earnings_fuel", "sepa_institutional",
            "sepa_vcp_dryup",
        }
        assert expected == set(spec["pillars"].keys()), \
            f"Pillar keys mismatch: {set(spec['pillars'].keys())}"

    def test_vcp_pillar_is_score_bonus(self, spec):
        assert spec["pillars"]["sepa_vcp_dryup"]["gate_type"] == "SCORE_BONUS", \
            "sepa_vcp_dryup must be SCORE_BONUS, not a hard gate"

    def test_six_pillars_are_hard_gates(self, spec):
        for k in ("sepa_trend_template", "sepa_adx_confirmed", "sepa_low_base",
                  "sepa_rs_confirmed", "sepa_earnings_fuel", "sepa_institutional"):
            assert spec["pillars"][k]["gate_type"] == "HARD_GATE", \
                f"{k} must be a HARD_GATE"

    def test_score_range_and_max_pass(self, spec):
        assert spec["_meta"]["score_range"] == "0-7"
        assert spec["_meta"]["score_max_pass"] == 6

    def test_version_string(self, spec):
        assert "sepa-momentum-codex" in spec["_meta"]["version"]

    def test_framework_label(self, spec):
        assert spec["_meta"]["frameworks_passed_label"] == "SEPA Momentum"


# ═══════════════════════════════════════════════════════════════════════════════
# TestSepaTrendTemplateRebuild
# ═══════════════════════════════════════════════════════════════════════════════

class TestSepaTrendTemplateRebuild:
    def test_d45_contains_all_five_components(self, de_source):
        block = _d45_block(de_source)
        for token in ("above_sma200", "sma_30w", "sma_200d", "sma_50d", "vstop_green"):
            assert token in block, f"d45_trend_structure must reference {token}"

    def test_d45_does_not_use_adx(self, de_source):
        block = _d45_block(de_source)
        assert "adx_14w" not in block, \
            "ADX must be removed from d45 (it becomes sepa_adx_confirmed in scoring_engine)"

    def test_d45_is_sum_of_five_terms(self, de_source):
        block = _d45_block(de_source)
        # The assignment sums 5 components → 4 '+' operators inside the parenthesised sum.
        assign = block[block.find('df["d45_trend_structure"]'):]
        assign = assign[:assign.find("\n\n")] if "\n\n" in assign else assign
        assert assign.count("+") >= 4, "d45 must sum 5 components (>= 4 '+' operators)"

    def test_d45_spec_declares_zero_to_five_range(self, spec):
        # The rebuilt Trend Template spans 0-5 (was 0-3); pass threshold is 4 of 5.
        assert spec["trend_template_rebuild"]["new_range"] == "0-5"
        assert spec["trend_template_rebuild"]["pass_threshold"] == 4
        assert set(spec["trend_template_rebuild"]["components"].keys()) == {
            "C1", "C2", "C3", "C5", "VSTOP"
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TestSepaVCPNotInHardGate  (mirrors TestSBonusVCPVolume in test_canslim_contract.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSepaVCPNotInHardGate:
    def test_vcp_not_in_fw_sepa_boolean_chain(self, se_source):
        block = _fw_sepa_block(se_source)
        fw_def = re.search(r"fw_sepa\s*=\s*\((.*?)\)\s*\n\s*df\[\"sepa_pass\"\]",
                           block, re.DOTALL)
        assert fw_def is not None, "fw_sepa = (...) boolean chain not found"
        assert "sepa_vcp_dryup" not in fw_def.group(1), \
            "sepa_vcp_dryup must NOT appear inside fw_sepa = (...) — it is a SCORE bonus"

    def test_vcp_in_score_summation(self, se_source):
        block = _fw_sepa_block(se_source)
        score_def = re.search(r'df\["sepa_score"\]\s*=\s*\((.*?)\)', block, re.DOTALL)
        assert score_def is not None, "sepa_score summation not found"
        assert "sepa_vcp_dryup" in score_def.group(1), \
            "sepa_vcp_dryup MUST appear in the sepa_score summation"

    def test_pass_chain_has_six_hard_pillars(self, se_source):
        block = _fw_sepa_block(se_source)
        fw_def = re.search(r"fw_sepa\s*=\s*\((.*?)\)\s*\n\s*df\[\"sepa_pass\"\]",
                           block, re.DOTALL).group(1)
        for k in ("sepa_trend_template", "sepa_adx_confirmed", "sepa_low_base",
                  "sepa_rs_confirmed", "sepa_earnings_fuel", "sepa_institutional"):
            assert k in fw_def, f"{k} must be in the fw_sepa hard-gate chain"


# ═══════════════════════════════════════════════════════════════════════════════
# TestSepaPillarArithmetic
# ═══════════════════════════════════════════════════════════════════════════════

class TestSepaPillarArithmetic:
    def test_full_pass_scores_seven(self):
        df = _run_sepa([_build_mock_sepa_row()])
        assert int(df["sepa_score"].iloc[0]) == 7
        assert int(df["sepa_pass"].iloc[0]) == 1

    def test_trend_template_threshold(self):
        df = _run_sepa([
            _build_mock_sepa_row(d45_trend_structure=3),   # < 4 fails
            _build_mock_sepa_row(d45_trend_structure=4),   # == 4 passes
        ])
        assert int(df["sepa_trend_template"].iloc[0]) == 0
        assert int(df["sepa_trend_template"].iloc[1]) == 1

    def test_adx_boundary(self):
        df = _run_sepa([
            _build_mock_sepa_row(adx_14w=19.9),
            _build_mock_sepa_row(adx_14w=20.0),
        ])
        assert int(df["sepa_adx_confirmed"].iloc[0]) == 0
        assert int(df["sepa_adx_confirmed"].iloc[1]) == 1

    def test_low_base_boundary(self):
        df = _run_sepa([
            _build_mock_sepa_row(dist_52wl=29.9),
            _build_mock_sepa_row(dist_52wl=30.0),
        ])
        assert int(df["sepa_low_base"].iloc[0]) == 0
        assert int(df["sepa_low_base"].iloc[1]) == 1

    def test_near_high_criterion7_gates_pass(self):
        """Trend Template Criterion 7 (SEPA Codex Ch.2, mandatory all-8): price must be
        within 25% of the 52-week high (Close >= 0.75 x 52wk high). Added 2026-06-12 —
        previously a stock 40% below its high could pass SEPA. 'Almost passing = FAIL.'"""
        df = _run_sepa([
            _build_mock_sepa_row(dist_52wh=25.1),   # 25.1% below high → C7 fails → no pass
            _build_mock_sepa_row(dist_52wh=25.0),   # exactly at the book line → passes
        ])
        assert int(df["sepa_pass"].iloc[0]) == 0, \
            "Stock >25% below its 52-week high must fail SEPA (Trend Template C7)"
        assert int(df["sepa_pass"].iloc[1]) == 1

    def test_near_high_missing_data_fails_conservatively(self):
        import numpy as np
        df = _run_sepa([_build_mock_sepa_row(dist_52wh=np.nan)])
        assert int(df["sepa_pass"].iloc[0]) == 0, \
            "Missing 52WH distance → C7 unverifiable → conservative fail"

    def test_earnings_fuel_eps_boundary(self):
        df = _run_sepa([
            _build_mock_sepa_row(eps_gr_yoy=24.9),                 # EPS fails
            _build_mock_sepa_row(eps_gr_yoy=25.0, rev_gr_yoy=20.0, roe=17.0),  # all pass
        ])
        assert int(df["sepa_earnings_fuel"].iloc[0]) == 0
        assert int(df["sepa_earnings_fuel"].iloc[1]) == 1

    def test_institutional_or_logic(self):
        df = _run_sepa([
            _build_mock_sepa_row(change_fii_lq=0.0, change_dii_lq=0.0),   # both zero → fail
            _build_mock_sepa_row(change_fii_lq=0.0, change_dii_lq=0.1),   # dii > 0 → pass
        ])
        assert int(df["sepa_institutional"].iloc[0]) == 0
        assert int(df["sepa_institutional"].iloc[1]) == 1

    def test_vcp_dryup_passthrough(self):
        df = _run_sepa([
            _build_mock_sepa_row(vcp_volume_dryup=0),
            _build_mock_sepa_row(vcp_volume_dryup=1),
        ])
        assert int(df["sepa_vcp_dryup"].iloc[0]) == 0
        assert int(df["sepa_vcp_dryup"].iloc[1]) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# TestSepaNaNConservative
# ═══════════════════════════════════════════════════════════════════════════════

class TestSepaNaNConservative:
    def test_all_hard_pillars_fail_on_nan(self):
        nan = float("nan")
        row = {
            "d45_trend_structure": nan, "adx_14w": nan, "dist_52wl": nan,
            "crs_aligned": nan, "eps_gr_yoy": nan, "rev_gr_yoy": nan, "roe": nan,
            "change_fii_lq": nan, "change_dii_lq": nan, "vcp_volume_dryup": nan,
            "market_cap": nan, "close_price": 100.0, "name": "NaNStock", "sector": "X",
            "is_financial": False,
        }
        df = _run_sepa([row])
        for col in ("sepa_trend_template", "sepa_adx_confirmed", "sepa_low_base",
                    "sepa_rs_confirmed", "sepa_earnings_fuel", "sepa_institutional",
                    "sepa_vcp_dryup"):
            assert int(df[col].iloc[0]) == 0, f"{col} must be 0 on NaN input (conservative)"
        assert int(df["sepa_pass"].iloc[0]) == 0
        assert int(df["sepa_score"].iloc[0]) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# TestSepaScoreInvariant
# ═══════════════════════════════════════════════════════════════════════════════

class TestSepaScoreInvariant:
    def test_pass_implies_score_six_or_seven(self):
        df = _run_sepa([
            _build_mock_sepa_row(vcp_volume_dryup=1),   # pass, score 7
            _build_mock_sepa_row(vcp_volume_dryup=0),   # pass, score 6
        ])
        for i in (0, 1):
            if int(df["sepa_pass"].iloc[i]) == 1:
                assert int(df["sepa_score"].iloc[i]) in (6, 7)

    def test_score_seven_does_not_guarantee_pass(self):
        # All 7 pillars fire but mcap below 500 → score 7, pass 0.
        df = _run_sepa([_build_mock_sepa_row(market_cap=100.0)])
        assert int(df["sepa_score"].iloc[0]) == 7
        assert int(df["sepa_pass"].iloc[0]) == 0

    def test_score_six_with_pass_one(self):
        df = _run_sepa([_build_mock_sepa_row(vcp_volume_dryup=0)])
        assert int(df["sepa_pass"].iloc[0]) == 1
        assert int(df["sepa_score"].iloc[0]) == 6


# ═══════════════════════════════════════════════════════════════════════════════
# TestSepaIndexAlignment
# ═══════════════════════════════════════════════════════════════════════════════

class TestSepaIndexAlignment:
    def test_shuffled_integer_index(self):
        rows = [_build_mock_sepa_row(), _build_mock_sepa_row(adx_14w=10.0)]
        df = pd.DataFrame(rows, index=[7, 3])
        df.attrs["detected_market_regime"] = "SIDEWAYS"
        from config import MASTER_PROFILES
        from core.scoring_engine import compute_qglp_score
        profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
        out = compute_qglp_score(df, profile)
        assert int(out.loc[7, "sepa_pass"]) == 1
        assert int(out.loc[3, "sepa_adx_confirmed"]) == 0

    def test_string_index(self):
        rows = [_build_mock_sepa_row(), _build_mock_sepa_row(dist_52wl=10.0)]
        df = pd.DataFrame(rows, index=["alpha", "beta"])
        df.attrs["detected_market_regime"] = "SIDEWAYS"
        from config import MASTER_PROFILES
        from core.scoring_engine import compute_qglp_score
        profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
        out = compute_qglp_score(df, profile)
        assert int(out.loc["alpha", "sepa_pass"]) == 1
        assert int(out.loc["beta", "sepa_low_base"]) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# TestSepaUIContract
# ═══════════════════════════════════════════════════════════════════════════════

class TestSepaUIContract:
    def test_render_function_importable(self):
        from ui.ui_tearsheet import render_sepa_radar
        assert callable(render_sepa_radar)

    def test_render_function_defined(self, ui_source):
        assert "def render_sepa_radar" in ui_source

    def test_render_pure_display_docstring(self, ui_source):
        block = _render_sepa_block(ui_source)
        assert "PURE DISPLAY" in block, "render_sepa_radar must declare PURE DISPLAY"

    def test_render_uses_blue_theme(self, ui_source):
        block = _render_sepa_block(ui_source)
        assert "#58a6ff" in block, "render_sepa_radar must use the #58a6ff blue theme"

    def test_render_shows_score_out_of_seven(self, ui_source):
        block = _render_sepa_block(ui_source)
        assert "/ 7" in block or "/7" in block, "render_sepa_radar must display score out of 7"

    def test_render_reads_all_seven_pillar_columns(self, ui_source):
        block = _render_sepa_block(ui_source)
        for col in ("sepa_trend_template", "sepa_adx_confirmed", "sepa_low_base",
                    "sepa_rs_confirmed", "sepa_earnings_fuel", "sepa_institutional",
                    "sepa_vcp_dryup"):
            assert col in block, f"render_sepa_radar must read {col}"

    def test_vcp_pillar_uses_amber_not_red(self, ui_source):
        block = _render_sepa_block(ui_source)
        # The VCP bonus pillar must use the amber watch colour (#e3b341) when not active,
        # never the red fail colour exclusively. Presence of amber proves non-red styling.
        assert "#e3b341" in block, \
            "VCP bonus pillar must show amber/watch styling (#e3b341) when not active, not red"

    def test_render_no_threshold_recomputation(self, ui_source):
        block = _render_sepa_block(ui_source)
        # Pure display must not recompute engine thresholds (no raw numeric comparisons
        # on the underlying input columns). Reading pre-materialized == 1 flags is fine.
        assert "adx_14w" not in block, "UI must not read raw adx_14w (pure display only)"
        assert "vol_sma_50d" not in block, "UI must not read raw vol_sma_50d (pure display only)"


# ═══════════════════════════════════════════════════════════════════════════════
# TestSepaRawSignals
# ═══════════════════════════════════════════════════════════════════════════════

def _render_sepa_radar_block(src: str) -> str:
    """The SEPA pillar grid was removed from the All Data tab (it duplicated the Frameworks-tab
    radar); render_sepa_radar is now the single on-screen home for these cells. Export still
    carries every column."""
    start = src.find("def render_sepa_radar")
    assert start != -1, "Cannot find render_sepa_radar in ui_tearsheet.py"
    nxt = src.find("\ndef ", start + 10)
    return src[start:nxt] if nxt != -1 else src[start:]


class TestSepaRawSignals:
    def test_all_pillar_columns_in_radar(self, ui_source):
        block = _render_sepa_radar_block(ui_source)
        for col in ("sepa_trend_template", "sepa_adx_confirmed", "sepa_low_base",
                    "sepa_rs_confirmed", "sepa_earnings_fuel", "sepa_institutional",
                    "sepa_vcp_dryup"):
            assert col in block, f"render_sepa_radar must display {col}"

    def test_score_and_pass_in_radar(self, ui_source):
        block = _render_sepa_radar_block(ui_source)
        assert "sepa_score" in block
        assert "sepa_pass" in block


# ═══════════════════════════════════════════════════════════════════════════════
# TestSepaAppWiring
# ═══════════════════════════════════════════════════════════════════════════════

class TestSepaAppWiring:
    def test_app_imports_render_sepa_radar(self, app_source):
        assert "render_sepa_radar" in app_source

    def test_app_calls_render_sepa_radar(self, app_source):
        assert re.search(r"render_sepa_radar\s*\(\s*stock\s*\)", app_source), \
            "app.py must call render_sepa_radar(stock)"

    def test_call_order_after_canslim_before_dorsey(self, app_source):
        cs  = app_source.find("render_canslim_radar(stock)")
        sepa = app_source.find("render_sepa_radar(stock)")
        dor = app_source.find("render_dorsey_radar(stock)")
        assert cs != -1 and sepa != -1 and dor != -1
        assert cs < sepa < dor, "render_sepa_radar(stock) must be AFTER canslim, BEFORE dorsey"

    def test_init_exports_render_sepa_radar(self):
        init_src = open(os.path.join(REPO_ROOT, "ui", "__init__.py"), encoding="utf-8").read()
        assert '"render_sepa_radar"' in init_src, "__all__ must include render_sepa_radar"
        assert "render_sepa_radar = _stub" in init_src, "stub fallback must cover render_sepa_radar"


# ═══════════════════════════════════════════════════════════════════════════════
# TestSepaTokenBoundary
# ═══════════════════════════════════════════════════════════════════════════════

class TestSepaTokenBoundary:
    def test_token_written_by_scoring_engine(self, se_source):
        assert '"SEPA Momentum|"' in se_source, \
            "scoring_engine must write the exact 'SEPA Momentum|' token into fw_str"

    def test_token_in_boundary_test_lists(self):
        token_src = open(TOKEN_TEST_PATH, encoding="utf-8").read()
        assert "SEPA Momentum" in token_src, \
            "'SEPA Momentum' must be present in test_framework_token_boundary.py token lists"

    def test_no_substring_collision(self, se_source):
        # No other framework token may contain 'SEPA Momentum' as a partial fragment.
        tokens = re.findall(r'np\.where\([^,]+,\s*"([^"|]+)\|"', se_source)
        sepa_like = [t for t in tokens if "SEPA Momentum" in t and t != "SEPA Momentum"]
        assert not sepa_like, f"Token collision with 'SEPA Momentum': {sepa_like}"


# ═══════════════════════════════════════════════════════════════════════════════
# TestSepaNonPillarGates (2026-10-15) — certification's gates that have no pillar box
# ═══════════════════════════════════════════════════════════════════════════════

class TestSepaNonPillarGates:
    """Certification also needs Trend Template criterion 7 (within 25% of the 52-week high) and the
    ₹500 Cr floor, and no pillar box shows either: on the 14 Oct sheet 9 stocks showed six green boxes
    and "Criteria Not Met" with nothing saying why. The engine writes each as a flag the SEPA panel
    reads — never re-judges."""

    _SIX = ("sepa_trend_template", "sepa_adx_confirmed", "sepa_low_base",
            "sepa_rs_confirmed", "sepa_earnings_fuel", "sepa_institutional")

    def test_near_high_flag_is_criterion_7_inclusive(self):
        df = _run_sepa([_build_mock_sepa_row(dist_52wh=d) for d in (25.0, 25.01, np.nan)])
        assert df["sepa_near_high"].tolist() == [1, 0, 0], "<= 25% passes; a missing distance fails"

    def test_market_cap_flag_is_the_floor_inclusive(self):
        df = _run_sepa([_build_mock_sepa_row(market_cap=m) for m in (500.0, 499.9, np.nan)])
        assert df["sepa_mcap_ok"].tolist() == [1, 0, 0], ">= ₹500 Cr passes; a missing cap fails"

    def test_certification_is_the_six_pillars_and_both_flags(self):
        df = _run_sepa([_build_mock_sepa_row(d45_trend_structure=t, dist_52wh=d, market_cap=m)
                        for t in (3, 5) for d in (10.0, 30.0) for m in (300.0, 900.0)])
        six = df[list(self._SIX)].eq(1).all(axis=1)
        gates = df["sepa_near_high"].eq(1) & df["sepa_mcap_ok"].eq(1)
        assert df["sepa_pass"].tolist() == (six & gates).astype(int).tolist()
        assert int(df["sepa_pass"].sum()) == 1, "only trend 5 + 10% below the high + ₹900 Cr passes"
