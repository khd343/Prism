"""Contract: the concept reference — EVERY categorical value-label PRISM produces is explained, and
the explanations are not shallow. The completeness guarantee is now CODE-DRIVEN, not hand-curated:
the universe of labels is extracted from the code that produces them (np.select / np.where string
choices + a few importable label constants + the catalyst/alert dicts), so a category can never be
silently forgotten — the test RED-fails listing the exact missing labels.

Guards:
  1. COVERAGE — every code-produced categorical label has a CONCEPT_REFERENCE explanation (or is
     consciously grandfathered in _INTENTIONAL_INTERNAL with a reason).
  2. QUALITY — every explanation clears a 40-char floor (catches "there but not properly explained").
  3. PURITY  — ui_reference_data.py is pure data (no imports / no Streamlit).
"""
import ast
import re
from pathlib import Path

from ui.ui_reference_data import CONCEPT_REFERENCE

_ROOT = Path(__file__).resolve().parent.parent

# Files that PRODUCE categorical display labels (np.select choice-lists, etc.).
_ENGINE_FILES = ["core/data_engine.py", "core/forensic_engine.py",
                 "core/scoring_engine.py", "core/verdict_engine.py"]


def _np_call_str_labels(src, attr):
    """Every STRING-literal choice/branch of every `np.<attr>(...)` call — the label VALUES.
    np.select: args[1] (choicelist List) + default= / args[2].  np.where: the branch args.
    Numeric-choice calls (scores) yield no strings and are naturally excluded."""
    out = set()
    for n in ast.walk(ast.parse(src)):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == attr
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "np"):
            continue
        cands = []
        if attr == "select":
            if len(n.args) >= 2 and isinstance(n.args[1], ast.List):
                cands += n.args[1].elts
            cands += [kw.value for kw in n.keywords if kw.arg == "default"]
            if len(n.args) >= 3:
                cands.append(n.args[2])
        else:  # where — the true/false branch values
            cands += list(n.args[1:])
        out |= {c.value for c in cands
                if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    return out


def _dict_keys(src, var):
    """String keys of the `var = {…}` dict literal (e.g. _CATALYSTS / _SELL_ALERTS)."""
    for n in ast.walk(ast.parse(src)):
        if (isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == var for t in n.targets)
                and isinstance(n.value, ast.Dict)):
            return {k.value for k in n.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return set()


def _fw_meta_keys():
    """The 37 framework names — AST-parsed from the function-local _FW_META in ui_tearsheet.py (no
    import / no refactor of that 3k-line, parallel-owned module; same AST approach as the emoji test)."""
    src = (_ROOT / "ui" / "ui_tearsheet.py").read_text(encoding="utf-8")
    for n in ast.walk(ast.parse(src)):
        if (isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_FW_META" for t in n.targets)
                and isinstance(n.value, ast.Dict)):
            return {k.value for k in n.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return set()


def _universe():
    """The full categorical-label universe, extracted FROM THE CODE that produces it."""
    u = set()
    for f in _ENGINE_FILES:
        u |= _np_call_str_labels((_ROOT / f).read_text(encoding="utf-8"), "select")
    disc = (_ROOT / "ui" / "ui_discovery.py").read_text(encoding="utf-8")
    u |= _np_call_str_labels(disc, "where")                  # piotroski strength tiers
    u |= _dict_keys(disc, "_CATALYSTS") | _dict_keys(disc, "_SELL_ALERTS")
    u |= _fw_meta_keys()                                      # the 37 frameworks
    from config import CONVICTION_TIERS
    u |= {t["label"] for t in CONVICTION_TIERS}
    from core.cyclicality_map import TIER_LABELS
    u |= set(TIER_LABELS.values())
    from core.data_engine import CONSOLIDATION_BASE_LABELS     # np.select reads them from the constant
    u |= set(CONSOLIDATION_BASE_LABELS)
    sc = (_ROOT / "core" / "scoring_engine.py").read_text(encoding="utf-8")
    u |= set(re.findall(r'return "(BULL|BEAR|SIDEWAYS)"', sc))   # market regime tags (Market Pulse)
    # Genuine display LABELS only — drop: format/template strings; and NARRATIVE sentences (the verdict
    # synthesis produces full self-explanatory sentences via np.select — they end with '.' / run long /
    # contain mid-sentence '. '; those are explanations already, not labels to define).
    def _is_label(x):
        x = x.strip()
        return (re.search(r"[A-Za-z]", x) and "{" not in x and "\n" not in x and "%" not in x
                and not x.endswith(".") and ". " not in x and len(x) <= 48)
    return {x for x in u if _is_label(x)}


def _clean(s):
    """Emoji/punct-insensitive key — sources differ (emoji labels vs plain 'label' fields)."""
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def _clean_key(s):
    return " ".join(_clean(s))


# Labels the enumerator pulls that are genuinely INTERNAL / never shown to a user — excused with a
# REASON only after review. Anything NOT here MUST be explained. (Mirrors the glossary grandfather.)
_INTENTIONAL_INTERNAL = {}


def test_every_categorical_label_is_explained():
    explained = {_clean_key(l) for cat in CONCEPT_REFERENCE.values() for l, _ in cat}
    missing = sorted(u for u in _universe()
                     if _clean_key(u) not in explained and u not in _INTENTIONAL_INTERNAL)
    assert not missing, (
        f"{len(missing)} categorical labels the code PRODUCES but CONCEPT_REFERENCE never explains "
        f"(explain each from its source, or grandfather with a reason):\n  " + "\n  ".join(missing))


def test_no_shallow_explanation():
    thin = [(lbl, len(exp.strip())) for entries in CONCEPT_REFERENCE.values()
            for lbl, exp in entries if len(exp.strip()) < 40]
    assert not thin, f"explanations under the 40-char quality floor: {thin}"


def test_reference_data_is_pure():
    src = (_ROOT / "ui" / "ui_reference_data.py").read_text(encoding="utf-8")
    assert "import streamlit" not in src and "\nimport " not in src, \
        "ui_reference_data.py must be pure data — no imports / no Streamlit"


def test_every_forensic_flag_has_a_display_description():
    """Every forensic rf_ flag the engine computes MUST have a _FLAG_DISPLAY description — else it
    fires on a Tear-Sheet with no explanation AND is absent from the Reference tab (which renders
    _FLAG_DISPLAY directly). This is the forensic-flag completeness guarantee, drift-proof."""
    from ui.ui_tearsheet import _FLAG_DISPLAY
    from config import FORENSIC_MAX_FLAGS
    src = (_ROOT / "core" / "forensic_engine.py").read_text(encoding="utf-8")
    rf_cols = set(re.findall(r'df\["(rf_[^"]+)"\]\s*=', src))
    missing = sorted(rf_cols - set(_FLAG_DISPLAY))
    assert not missing, f"forensic rf_ flags with NO _FLAG_DISPLAY description: {missing}"
    assert len(_FLAG_DISPLAY) == FORENSIC_MAX_FLAGS, \
        f"_FLAG_DISPLAY has {len(_FLAG_DISPLAY)} entries; FORENSIC_MAX_FLAGS = {FORENSIC_MAX_FLAGS}"


# ═══════════════════════════════════════════════════════════════════════════
# THE ALL-LABEL COVERAGE SWEEP (2026-08-28) — closing the June blind spot
# ═══════════════════════════════════════════════════════════════════════════
def test_every_label_emitting_column_is_fully_documented():
    """The filterable-set coverage above has a structural blind spot this file's own history
    recorded ("green concept-coverage tests have blind spots — always content-check"): a label
    column that is NOT a Discovery filter could ship undocumented and every test stayed green.
    It happened — tier_label (the app's most visible vocabulary: Crown Jewels…Not Ready) and
    atoms_to_bits_label were absent from the reference until a manual sweep found them on
    2026-08-28. This test IS that sweep, permanent: every live value of every label-emitting
    column must be a documented concept label. A new label column goes in this list the day it
    is born, and cannot ship undocumented without turning this red."""
    import contextlib, io as _io, sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
    from data_engine import (coerce_numeric_columns, compute_derived_signals, load_all_csvs,
                             merge_datasets)
    from core import run_scoring_pipeline
    from ui.ui_reference_data import CONCEPT_REFERENCE
    with contextlib.redirect_stdout(_io.StringIO()):
        df = run_scoring_pipeline(
            compute_derived_signals(coerce_numeric_columns(merge_datasets(load_all_csvs("local")))))
    documented = set()
    for _cat, entries in CONCEPT_REFERENCE.items():
        documented |= {lbl for lbl, _ in entries}
    LABEL_COLS = ["buy_zone_label", "weinstein_stage", "cf_triangle", "lynch_category",
                  "mef_label", "smart_money_flow", "wealth_tier", "ep_power_curve",
                  "corporate_class", "tier_label", "moat_growth_quad", "peg_zone",
                  "earnings_power_box", "cyclicality_tier", "sector_capital_phase", "cash_machine_label",
                  "verdict_direction", "atoms_to_bits_label", "forensic_label",
                  "shariah_status"]          # born 2026-10-16 (core/shariah.py) — in the list the day it was born
    problems = []
    for col in sorted(LABEL_COLS):
        assert col in df.columns, f"{col} vanished from the frame — update the sweep list"
        vals = {str(v) for v in df[col].dropna().unique() if str(v).strip() and str(v) != "nan"}
        missing = sorted(v for v in vals if v not in documented)
        if missing:
            problems.append(f"{col}: {missing}")
    assert not problems, (
        "these live label values have NO entry in the searchable reference "
        "(add them to CONCEPT_REFERENCE, written from the code that emits them):  "
        + " | ".join(problems)
    )


def test_soundness_owns_the_layer_name_in_category_headers():
    """The 2026-08-27 rename rule: the display label for the engine-verdict layer is SOUNDNESS
    on every surface. Two reference category headers ("Verdict — Evidence Confidence" /
    "Verdict — Top Risk") missed the rename until the 2026-08-28 content audit. Category
    headers naming this layer must say Soundness; lowercase generic prose is fine, and the
    Mauboussin "EV Verdict" entry is a different layer's own term."""
    from ui.ui_reference_data import CONCEPT_REFERENCE
    offenders = [k for k in CONCEPT_REFERENCE if "Verdict" in k]
    assert not offenders, (
        f"category headers still name the layer 'Verdict' (it is Soundness on every surface): {offenders}"
    )


# ── Analysis Mode ↔ config parity (added 2026-09-20) ─────────────────────────────
# WHY THESE EXIST: the corpus carried three Analysis Mode entries and said Hybrid was "The
# all-round default" — false since 2026-09-19, when DEFAULT_ANALYSIS_MODE became "Breakout",
# which had no entry at all. So the searchable Reference contradicted the ⚙️ Config tab two
# tabs away, which prints "· default" on the real one. Nothing could see it: the corpus is
# hand-written prose and nothing compared it to config.
#
# WHY NOT JUST RENDER config's OWN TEXT HERE, which would remove the copy entirely: two
# reasons, and the second is structural. (1) REGISTER — config's description is a CAPTION
# carrying measurement provenance ("...which rank Hybrid #56/#60/#60 of 66 weightings.
# December's 6-month window reviews it"); a glossary entry answers "what is this?", and an IC
# sweep ranking does not belong in one. The same conclusion the 2026-09-20 empty states
# reached: text written for one surface does not transplant to another. (2) ui_reference_data
# is PINNED AS PURE DATA with no imports (test_reference_data_is_pure, below), so it cannot
# read config even if we wanted it to.
#
# So the prose stays hand-written and the FACTS are pinned against config instead: every mode
# is present, and exactly one entry claims default status — the one that holds it.

_MODE_SECTION = "🎛️ Analysis Mode"


def _mode_entries():
    """The (title, definition) pairs of the Analysis Mode section, as data not source text."""
    assert _MODE_SECTION in CONCEPT_REFERENCE, (
        "the %r section is gone from CONCEPT_REFERENCE" % _MODE_SECTION)
    return list(CONCEPT_REFERENCE[_MODE_SECTION])


def test_every_analysis_mode_has_a_reference_entry():
    """A mode a reader can SELECT must be a mode they can LOOK UP.

    The join key is config's own `label`, which the corpus titles already match exactly — so a
    renamed or added mode fails here instead of silently going undocumented (⚡ Breakout did,
    for a day, while being the default).
    """
    from config import ANALYSIS_MODES
    titles = {t for t, _ in _mode_entries()}
    missing = sorted(m["label"] for m in ANALYSIS_MODES.values() if m["label"] not in titles)
    assert not missing, (
        "these Analysis Modes are selectable but absent from the Reference: %s" % missing)


def test_exactly_one_mode_entry_claims_default_and_it_is_the_real_default():
    """The claim must track config, in BOTH directions.

    Two-sided on purpose: the failure this pins was not a missing word, it was the word sitting
    on the WRONG mode. Asserting only "the default mode says default" would still pass with
    Hybrid's stale "The all-round default." left in place beside it.
    """
    from config import ANALYSIS_MODES, DEFAULT_ANALYSIS_MODE
    want = ANALYSIS_MODES[DEFAULT_ANALYSIS_MODE]["label"]
    claim = [t for t, d in _mode_entries() if "default" in d.lower()]
    assert claim == [want], (
        "exactly one Analysis Mode entry may claim default status and it must be %r "
        "(config.DEFAULT_ANALYSIS_MODE = %r); entries claiming it: %s"
        % (want, DEFAULT_ANALYSIS_MODE, claim))
