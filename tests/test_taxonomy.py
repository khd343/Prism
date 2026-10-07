"""Contract: core/taxonomy.py — the vendor's industry/sector names reach PRISM's code in PRISM's names.

WHY (2026-10-07). Between the 2026-09-19 and 2026-10-06 vintages the vendor re-spelled its taxonomy
("IT - Software" → "IT Software", "Auto Ancillaries - Gears" → "Auto Gears"). Every name lookup in
PRISM falls back without an error, so nothing failed: 896 of 2,712 stocks silently took a wrong
cyclicality tier (it caps fair_pe_qglp), 19 financials lost is_financial, 29 utilities is_utility,
and IT companies were judged on the default receivables threshold. These tests run on the names the
vendor actually sent (LIVE_2026_10_06 below — names only, no data) so they need no private file, and
they fail the day a code literal, a map regeneration or a new rename breaks the translation again.
"""
import ast
import contextlib
import glob
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd
import pytest

import config
from core import taxonomy as T
from core.cyclicality_map import INDUSTRY_TIER, SECTOR_TIER_FALLBACK
from data_engine import (compute_derived_signals, merge_datasets, COMMON_COLS, RATIO_COLS, INCOME_COLS,
                         BALANCE_COLS, CASHFLOW_COLS, SHAREHOLDING_COLS, TECHNICAL_COLS)

_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a)


def _live():
    """The live vintage as (sector, industry, stocks) rows: as sent, and translated."""
    sent = pd.DataFrame([(s, i, n) for (s, i), n in LIVE_2026_10_06.items()],
                        columns=["sector", "industry", "stocks"])
    return sent, _quiet(T.canonicalize_taxonomy, sent.copy())


def _tier(industry, sector):
    """The engine's chain (data_engine.compute_derived_signals): industry → sector → "F"."""
    return INDUSTRY_TIER.get(industry) or SECTOR_TIER_FALLBACK.get(sector) or "F"


# ── 1. the translation itself is sound ─────────────────────────────────────────────────────
@pytest.mark.parametrize("column", ["industry", "sector"])
def test_no_two_prism_names_normalize_alike(column):
    """Spelling matching is 1:1 only if no two PRISM names share a normalized key. A regeneration
    that added the vendor's spelling beside the old name ("Diamond Gems and Jewellery" next to
    "Diamond, Gems & Jewellery") fails here instead of silently dropping the key."""
    seen = {}
    for name in sorted(T.VOCABULARY[column]):
        seen.setdefault(T.normalize(name), []).append(name)
    clashes = {k: v for k, v in seen.items() if len(v) > 1}
    assert not clashes, f"{column}: PRISM names that normalize alike: {clashes}"


@pytest.mark.parametrize("column", ["industry", "sector"])
def test_every_rename_is_live_reachable_and_lands_on_a_prism_name(column):
    renames = T.VENDOR_RENAMES[column]
    live = {i if column == "industry" else s for (s, i) in LIVE_2026_10_06}
    assert list(renames) == sorted(renames), "VENDOR_RENAMES must be key-sorted (determinism)"
    for vendor, prism in renames.items():
        assert prism in T.VOCABULARY[column], f"{vendor!r} → {prism!r}: target is not a PRISM name"
        assert vendor not in T.VOCABULARY[column], f"{vendor!r} is already a PRISM name: the rename is dead"
        assert T.normalize(vendor) not in T._SPELLING[column], (
            f"{vendor!r} already matches by spelling, so the rename can never fire")
        assert vendor in live, f"{vendor!r} is not a name the vendor sent: a guess, not a translation"


def test_the_moved_industry_returns_to_a_sector_prism_names():
    for industry, sector in T.SECTOR_OF_INDUSTRY.items():
        assert industry in T.VOCABULARY["industry"] and sector in T.VOCABULARY["sector"]
    assert "Quick Service Restaurant" in config.EPOCH5_MODERN["tipping_sectors"]


@pytest.mark.parametrize("column", ["industry", "sector"])
def test_the_translation_never_merges_two_names_the_vendor_keeps_apart(column):
    """Two vendor industries folded into one PRISM name would merge two peer groups."""
    sent, out = _live()
    pairs = pd.DataFrame({"sent": sent[column], "prism": out[column]}).drop_duplicates()
    if column == "sector":   # the one deliberate move: QSR chains back to their own sector
        pairs = pairs[~sent.loc[pairs.index, "industry"].map(
            lambda i: T.canonical_name(i, "industry")).isin(T.SECTOR_OF_INDUSTRY)]
    merged = pairs.groupby("prism")["sent"].nunique()
    assert (merged == 1).all(), f"merged: {merged[merged > 1].to_dict()}"


# ── 2. on the names the vendor actually sent ─────────────────────────────────────────────────
def test_every_live_name_is_a_prism_name_or_a_known_new_one():
    """Kept as the vendor wrote them: industries the vendor regrouped or created (their tier comes
    from the translated sector), and two sectors PRISM never had. A new name here is a decision —
    add a rename only when it is a plain relabel."""
    _, out = _live()
    assert sorted(set(out["sector"]) - T.VOCABULARY["sector"]) == ["Alcoholic Beverages", "Insurance"]
    assert sorted(set(out["industry"]) - T.VOCABULARY["industry"]) == [
        "Airport Restaurants and Lounges", "Auto Parts Distribution", "Biotechnology",
        "Cryogenic Equipment", "Cutting Tools and Machine Tools", "Engineering and Infrastructure",
        "FMCG and Soap Ingredients", "General Electric Equipment", "IT Services", "Insurance Distribution",
        "Machine Tools", "Medical Equipment and Diagnostic Technology",
        "Natural Extracts and Food Ingredients", "Pharma Distribution", "Pharma Formulations",
        "Pharma Gelatin Capsules", "Pharma IV Fluids", "Pharma Integrated",
        "Publishing and Learning Services", "Railway Equipment", "Restaurants",
        "Secure Printing and Payment Cards", "Solar Power and EPC", "Technology Distribution",
        "Telecommunications Equipment and Network EPC",
    ]


def test_only_one_live_stock_reaches_the_catch_all_without_a_tier():
    """Before: 1,043 stocks ended in "F", 758 of them only because their names had changed. Now
    one does: the vendor's first insurance distributor, in a sector PRISM has never had."""
    _, out = _live()
    untiered = out[[i not in INDUSTRY_TIER and s not in SECTOR_TIER_FALLBACK
                    for i, s in zip(out["industry"], out["sector"])]]
    assert untiered[["sector", "industry", "stocks"]].values.tolist() == [
        ["Insurance", "Insurance Distribution", 1]]


def test_tiers_on_the_live_vintage():
    sent, out = _live()
    out["tier"] = [_tier(i, s) for i, s in zip(out["industry"], out["sector"])]
    assert out.groupby("tier")["stocks"].sum().to_dict() == {
        "A": 363, "B": 1285, "C": 394, "D": 340, "E": 45, "F": 285}
    tier = {(s, i): t for (s, i), t in zip(zip(sent["sector"], sent["industry"]), out["tier"])}
    assert tier[("Chemicals", "Specialty Chemicals")] == "D"           # was A: P/E capped by mistake
    assert tier[("IT Software", "IT Services")] == "D"                  # was F
    assert tier[("Castings Forgings and Fasteners", "Castings Forgings and Fasteners")] == "A"  # was F
    assert tier[("Diamond Gems and Jewellery", "Diamond Gems and Jewellery")] == "B"
    assert tier[("Aerospace and Defence", "Aerospace and Defence Equipment")] == "D"


def test_financials_utilities_and_sector_sets_see_the_live_names():
    sent, out = _live()
    fin = out["industry"].isin(config.FINANCIAL_SECTORS) | out["sector"].isin(config.FINANCIAL_SECTOR_NAMES)
    assert int(out.loc[fin, "stocks"].sum()) == 62          # 43 before the translation
    assert int(out.loc[out["sector"].isin(config.UTILITY_SECTOR_NAMES), "stocks"].sum()) == 41   # 12 before
    for ind in ("Finance Holding Company", "NBFC Others", "Capital Markets RTA",
                "Finance Capital Markets Brokers", "Infra and Real Estate Investment Trust"):
        assert T.canonical_name(ind, "industry") in config.FINANCIAL_SECTORS, ind
    assert T.canonical_name("Stock and Commodity Brokers", "sector") in config.FINANCIAL_SECTOR_NAMES
    assert T.canonical_name("Power Generation and Distribution", "sector") in config.UTILITY_SECTOR_NAMES
    assert T.canonical_name("IT Software", "sector") in config.CONSISTENT_SECTORS


def test_every_name_written_in_prisms_code_matches_the_translated_live_data():
    """THE GUARD FOR THE ~50 SITES. A string in PRISM's code that names a live industry or sector
    (in any spelling) must be spelled as the translation spells it — otherwise that lookup has
    silently stopped matching, which is exactly how this broke."""
    sent, out = _live()
    live = {}                     # column → normalized name → the spellings the translated data carries
    for c in ("industry", "sector"):
        live[c] = {}
        for v, p in [*zip(sent[c], out[c]), *zip(out[c], out[c])]:
            live[c].setdefault(T.normalize(v), set()).add(p)
    files = (glob.glob(os.path.join(_ROOT, "core", "*.py")) + glob.glob(os.path.join(_ROOT, "ui", "*.py"))
             + [os.path.join(_ROOT, "config.py"), os.path.join(_ROOT, "app.py")])
    stale = []
    for f in sorted(files):
        if os.path.basename(f) in ("taxonomy.py", "cyclicality_map.py"):
            continue
        for node in ast.walk(ast.parse(open(f, encoding="utf-8").read())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) < 80:
                key = T.normalize(node.value)
                spelled = set().union(*(live[c][key] for c in live if key in live[c]))
                if spelled and node.value not in spelled:
                    stale.append(f"{os.path.relpath(f, _ROOT)}:{node.lineno} {node.value!r} → data says {sorted(spelled)}")
    assert not stale, "names in code the data no longer carries:\n" + "\n".join(stale)


# ── 3. the mechanics, and that the engine receives PRISM names ─────────────────────────────────
def test_canonicalize_leaves_everything_else_alone_and_is_idempotent():
    df = pd.DataFrame({"company_id": ["NSE:A", "NSE:B", "NSE:C", "NSE:D"],
                       "industry": ["Auto Gears", np.nan, "IT Services", "Speciality Chemicals"],
                       "sector": ["Auto Ancillaries", "IT Software", None, "Chemicals"],
                       "pe": [10.0, 20.0, 30.0, 40.0]}, index=[7, 8, 9, 10])
    once = _quiet(T.canonicalize_taxonomy, df.copy())
    assert once["industry"].tolist()[0] == "Auto Ancillaries - Gears"
    assert pd.isna(once["industry"].iloc[1]) and once["sector"].iloc[2] is None
    assert once["industry"].iloc[2] == "IT Services"                   # regrouped name: kept
    assert once["sector"].iloc[1] == "IT - Software"
    assert once["industry"].iloc[3] == "Speciality Chemicals"           # already PRISM's: unchanged
    assert once.index.tolist() == [7, 8, 9, 10] and once["pe"].tolist() == df["pe"].tolist()
    pd.testing.assert_frame_equal(_quiet(T.canonicalize_taxonomy, once.copy()), once)
    _quiet(T.canonicalize_taxonomy, pd.DataFrame({"pe": [1.0]}))       # no taxonomy columns: no error


def test_merge_datasets_hands_prism_names_to_the_engine():
    ids = ["NSE:IT1", "NSE:FIN1", "NSE:QSR1", "NSE:CH1"]
    ratio = pd.DataFrame({"company_id": ids, "name": ["It", "Fin", "Qsr", "Chem"],
                          "industry": ["IT Services", "Finance Holding Company", "Quick Service Restaurant",
                                       "Specialty Chemicals"],
                          "sector": ["IT Software", "Finance", "Hotels and Restaurants", "Chemicals"]})
    other = pd.DataFrame({"company_id": ids})
    master = _quiet(merge_datasets, {"ratio": ratio, **{k: other for k in
                                     ("income", "balance", "cashflow", "shareholding", "technical")}})
    assert master["sector"].tolist() == ["IT - Software", "Finance", "Quick Service Restaurant", "Chemicals"]
    assert master["industry"].tolist() == ["IT Services", "Finance - Holding Company",
                                           "Quick Service Restaurant - QSR", "Speciality Chemicals"]
    mapped = set()
    for m in (COMMON_COLS, RATIO_COLS, INCOME_COLS, BALANCE_COLS, CASHFLOW_COLS, SHAREHOLDING_COLS,
              TECHNICAL_COLS):
        mapped.update(m.values())
    full = pd.concat([master, pd.DataFrame(np.nan, index=master.index,
                                           columns=sorted(mapped - set(master.columns)))], axis=1)
    out = _quiet(compute_derived_signals, full)
    assert out["cyclicality_tier_code"].tolist() == ["D", "E", "B", "D"]
    assert out["is_financial"].tolist() == [False, True, False, False]


# (sector, industry) → stocks, as the vendor wrote them in the 2026-10-06 vintage (2,712 stocks).
LIVE_2026_10_06 = {
    ('Aerospace and Defence', 'Aerospace and Defence Equipment'): 24,
    ('Aerospace and Defence', 'Aviation'): 1,
    ('Aerospace and Defence', 'EMS'): 2,
    ('Aerospace and Defence', 'Electronics Others'): 1,
    ('Aerospace and Defence', 'Engineering Heavy General'): 3,
    ('Aerospace and Defence', 'Engineering Light General'): 3,
    ('Aerospace and Defence', 'Industrial Explosives'): 2,
    ('Aerospace and Defence', 'Metal Copper and Copper Alloy Products'): 1,
    ('Aerospace and Defence', 'Miscellaneous'): 1,
    ('Aerospace and Defence', 'Shipbuilding and Repairs'): 3,
    ('Aerospace and Defence', 'Shipping Proxy'): 2,
    ('Aerospace and Defence', 'Telecommunications Equipment'): 1,
    ('Agro Chemicals', 'Pesticides and Agrochemicals'): 25,
    ('Air Transport Service', 'Air Transport Service'): 2,
    ('Air Transport Service', 'Realty Construction and Contracting'): 1,
    ('Air Transport Service', 'Transport Airlines'): 3,
    ('Alcoholic Beverages', 'Alcoholic Beverages'): 1,
    ('Auto Ancillaries', 'Auto AC'): 1,
    ('Auto Ancillaries', 'Auto Axle and Brakes'): 3,
    ('Auto Ancillaries', 'Auto Axles'): 4,
    ('Auto Ancillaries', 'Auto Batteries'): 5,
    ('Auto Ancillaries', 'Auto Bus and LCVs'): 1,
    ('Auto Ancillaries', 'Auto Clutches'): 1,
    ('Auto Ancillaries', 'Auto Electric Utility'): 1,
    ('Auto Ancillaries', 'Auto Engine Parts'): 8,
    ('Auto Ancillaries', 'Auto Gears'): 10,
    ('Auto Ancillaries', 'Auto Lighting'): 3,
    ('Auto Ancillaries', 'Auto Others'): 14,
    ('Auto Ancillaries', 'Auto Parts Distribution'): 1,
    ('Auto Ancillaries', 'Auto Plastic Parts'): 1,
    ('Auto Ancillaries', 'Auto Seats'): 1,
    ('Auto Ancillaries', 'Auto Sheet Metal'): 2,
    ('Auto Ancillaries', 'Auto Shock Absorbers'): 1,
    ('Auto Ancillaries', 'Auto Spare Parts Accessories'): 4,
    ('Auto Ancillaries', 'Auto Springs'): 2,
    ('Auto Ancillaries', 'Auto Tractors'): 1,
    ('Auto Ancillaries', 'Auto Transmission'): 2,
    ('Auto Ancillaries', 'Auto Two Wheeler Parts'): 4,
    ('Auto Ancillaries', 'Auto Two and Three Wheeler Parts'): 2,
    ('Auto Ancillaries', 'Auto Two and Three Wheelers'): 1,
    ('Auto Ancillaries', 'Auto Wheels'): 3,
    ('Auto Ancillaries', 'Bearings'): 1,
    ('Auto Ancillaries', 'Castings Forgings and Fasteners'): 1,
    ('Auto Ancillaries', 'Diversified Auto Components'): 21,
    ('Auto Ancillaries', 'Engines'): 1,
    ('Auto Ancillaries', 'Railways Kavach and Springs'): 2,
    ('Automobile', 'Auto Bus and LCVs'): 3,
    ('Automobile', 'Auto Four Wheelers'): 5,
    ('Automobile', 'Auto Tractors'): 2,
    ('Automobile', 'Auto Two and Three Wheelers'): 7,
    ('Automobile', 'Automobiles LCVs and HCVs'): 3,
    ('Automobile', 'Automobiles Scooters and Three Wheelers'): 1,
    ('Automobile', 'Diversified Auto Components'): 1,
    ('Automobile', 'General Electric Equipment'): 1,
    ('Bearings', 'Bearings'): 10,
    ('Bearings', 'Trading'): 1,
    ('Cables', 'Cables Power'): 15,
    ('Cables', 'Cables Telecom'): 4,
    ('Cables', 'EPC'): 1,
    ('Cables', 'Metals'): 1,
    ('Cables', 'Railways Kavach and Springs'): 1,
    ('Capital Goods Electrical Equipment', 'CNC Machines'): 1,
    ('Capital Goods Electrical Equipment', 'Cables Power'): 1,
    ('Capital Goods Electrical Equipment', 'Capital Goods Electric General'): 21,
    ('Capital Goods Electrical Equipment', 'Capital Goods Electrical Equipment'): 1,
    ('Capital Goods Electrical Equipment', 'Capital Goods Engineering General'): 1,
    ('Capital Goods Electrical Equipment', 'Capital Goods Engineering Heavy'): 4,
    ('Capital Goods Electrical Equipment', 'Capital Goods Gensets and Turbines'): 4,
    ('Capital Goods Electrical Equipment', 'Capital Goods Solar'): 5,
    ('Capital Goods Electrical Equipment', 'Capital Goods Switchgear'): 2,
    ('Capital Goods Electrical Equipment', 'Capital Goods Transformers'): 14,
    ('Capital Goods Electrical Equipment', 'Electric Equipment General'): 21,
    ('Capital Goods Electrical Equipment', 'Electrical Equipment and HVDC'): 3,
    ('Capital Goods Electrical Equipment', 'Electrical Switchgear Relays and Circuits'): 3,
    ('Capital Goods Electrical Equipment', 'Engineering Light General'): 1,
    ('Capital Goods Electrical Equipment', 'Engineering Turnkey Services'): 1,
    ('Capital Goods Electrical Equipment', 'Infra Engineering General'): 1,
    ('Capital Goods Electrical Equipment', 'Machine Tools Others'): 1,
    ('Capital Goods Electrical Equipment', 'Shipping Proxy'): 1,
    ('Capital Goods Electrical Equipment', 'Shunt Resistors'): 1,
    ('Capital Goods Electrical Equipment', 'Trading'): 2,
    ('Capital Goods Electrical Equipment', 'Transmission Line Towers and Equipment'): 1,
    ('Capital Goods Non Electrical Equipment', 'Abrasives and Grinding Wheels'): 3,
    ('Capital Goods Non Electrical Equipment', 'Auto Tractors'): 1,
    ('Capital Goods Non Electrical Equipment', 'Bearings'): 1,
    ('Capital Goods Non Electrical Equipment', 'CNC Machines'): 2,
    ('Capital Goods Non Electrical Equipment', 'Capital Goods EPC and Cranes'): 3,
    ('Capital Goods Non Electrical Equipment', 'Capital Goods Electrical Equipment'): 1,
    ('Capital Goods Non Electrical Equipment', 'Capital Goods Engineering General'): 4,
    ('Capital Goods Non Electrical Equipment', 'Capital Goods Engineering Heavy'): 11,
    ('Capital Goods Non Electrical Equipment', 'Capital Goods Mining Equipment'): 2,
    ('Capital Goods Non Electrical Equipment', 'Compressors'): 4,
    ('Capital Goods Non Electrical Equipment', 'Cryogenic Equipment'): 1,
    ('Capital Goods Non Electrical Equipment', 'Cutting Tools and Machine Tools'): 1,
    ('Capital Goods Non Electrical Equipment', 'DI Pipes and Saw Pipes'): 2,
    ('Capital Goods Non Electrical Equipment', 'Diversified'): 1,
    ('Capital Goods Non Electrical Equipment', 'Electric Equipment General'): 1,
    ('Capital Goods Non Electrical Equipment', 'Electrodes Graphites'): 1,
    ('Capital Goods Non Electrical Equipment', 'Electrodes Welding Equipment'): 3,
    ('Capital Goods Non Electrical Equipment', 'Engineering General'): 1,
    ('Capital Goods Non Electrical Equipment', 'Engineering Heavy General'): 4,
    ('Capital Goods Non Electrical Equipment', 'Engineering Heavy Material Handling'): 4,
    ('Capital Goods Non Electrical Equipment', 'Engineering Heavy Plastic Machinery'): 1,
    ('Capital Goods Non Electrical Equipment', 'Engineering Light Gears'): 1,
    ('Capital Goods Non Electrical Equipment', 'Engineering Light General'): 21,
    ('Capital Goods Non Electrical Equipment', 'Engines'): 2,
    ('Capital Goods Non Electrical Equipment', 'Gensets'): 2,
    ('Capital Goods Non Electrical Equipment', 'Hydraulics'): 3,
    ('Capital Goods Non Electrical Equipment', 'Machine Tools'): 1,
    ('Capital Goods Non Electrical Equipment', 'Machine Tools Others'): 3,
    ('Capital Goods Non Electrical Equipment', 'Oil Drilling and Exploration'): 1,
    ('Capital Goods Non Electrical Equipment', 'Pollution Control Equipment'): 1,
    ('Capital Goods Non Electrical Equipment', 'Pumps'): 3,
    ('Capital Goods Non Electrical Equipment', 'Railways'): 3,
    ('Capital Goods Non Electrical Equipment', 'Solar EPC'): 1,
    ('Capital Goods Non Electrical Equipment', 'Solar Pumps'): 2,
    ('Capital Goods Non Electrical Equipment', 'Stainless Steel'): 1,
    ('Capital Goods Non Electrical Equipment', 'Steel Tubes and Pipes'): 2,
    ('Capital Goods Non Electrical Equipment', 'Textile Machinery'): 1,
    ('Capital Goods Non Electrical Equipment', 'Water Treatment'): 1,
    ('Capital Goods Non Electrical Equipment', 'Welding Equipment'): 4,
    ('Castings Forgings and Fasteners', 'Capital Goods Mining Equipment'): 1,
    ('Castings Forgings and Fasteners', 'Castings Forgings and Fasteners'): 17,
    ('Castings Forgings and Fasteners', 'Castings Grey Iron'): 2,
    ('Castings Forgings and Fasteners', 'Castings Steel and Alloy'): 9,
    ('Castings Forgings and Fasteners', 'DI Pipes and Saw Pipes'): 1,
    ('Castings Forgings and Fasteners', 'Fasteners'): 1,
    ('Castings Forgings and Fasteners', 'Forgings'): 8,
    ('Cement', 'Cement'): 29,
    ('Cement', 'Cement Products'): 1,
    ('Cement Products', 'Cement'): 1,
    ('Cement Products', 'Cement Products'): 4,
    ('Cement Products', 'Pre Engineering Buildings'): 1,
    ('Ceramic Products', 'Ceramics Sanitaryware and Others'): 1,
    ('Ceramic Products', 'Ceramics Tiles'): 4,
    ('Ceramic Products', 'Ceramics Tiles and Sanitaryware'): 5,
    ('Ceramic Products', 'Trading'): 1,
    ('Chemicals', 'Aluminium'): 1,
    ('Chemicals', 'Carbon Black'): 4,
    ('Chemicals', 'Chemicals Fluorine'): 3,
    ('Chemicals', 'Chemicals Gelatine'): 2,
    ('Chemicals', 'Chemicals Inorganic'): 9,
    ('Chemicals', 'Chemicals Organic'): 22,
    ('Chemicals', 'Chemicals Organic Alcohol Based'): 1,
    ('Chemicals', 'Chemicals Organic Maleic Anhydride'): 1,
    ('Chemicals', 'Chemicals Others'): 6,
    ('Chemicals', 'Dyes Intermediate'): 2,
    ('Chemicals', 'Dyes and Pigments'): 22,
    ('Chemicals', 'Edible Oils Agro Processing'): 1,
    ('Chemicals', 'Fertilisers'): 2,
    ('Chemicals', 'Finance Holding Company'): 1,
    ('Chemicals', 'Industrial Explosives'): 2,
    ('Chemicals', 'Industrial Gas'): 3,
    ('Chemicals', 'Inorganic Chemicals Caustic Soda and Soda Ash'): 8,
    ('Chemicals', 'Lubricants'): 4,
    ('Chemicals', 'Pesticides and Agrochemicals'): 1,
    ('Chemicals', 'Petrochemicals Polymers'): 2,
    ('Chemicals', 'Plastics Plastic and Plastic Products'): 2,
    ('Chemicals', 'Rubber Processing and Rubber Products'): 1,
    ('Chemicals', 'Specialty Chemicals'): 55,
    ('Chemicals', 'Sugar'): 1,
    ('Chemicals', 'Trading'): 7,
    ('Co Working', 'Realty CoWorking'): 3,
    ('Computer Education', 'Computer Education'): 3,
    ('Computer Education', 'IT Education'): 1,
    ('Construction', 'Construction Civil and Turnkey'): 5,
    ('Construction', 'Construction Factories and Offices and Commercial'): 7,
    ('Construction', 'Construction Housing'): 3,
    ('Construction', 'Construction and Contracting'): 5,
    ('Construction', 'Infra General'): 1,
    ('Construction', 'Pre Engineering Buildings'): 1,
    ('Consumer Durables', 'Building Material USA'): 2,
    ('Consumer Durables', 'CCTV Camera'): 1,
    ('Consumer Durables', 'Capital Goods Electric General'): 1,
    ('Consumer Durables', 'Consumer Electronics'): 13,
    ('Consumer Durables', 'Domestic Appliances'): 7,
    ('Consumer Durables', 'EMS'): 7,
    ('Consumer Durables', 'Electronics Equipment and Components'): 5,
    ('Consumer Durables', 'Electronics Others'): 3,
    ('Consumer Durables', 'Electronics Power Devices and Equipment'): 2,
    ('Consumer Durables', 'Electronics Soft Ferrites'): 1,
    ('Consumer Durables', 'Electronics TV and Audio and VCR and VCP'): 3,
    ('Consumer Durables', 'Granite and Marble'): 5,
    ('Consumer Durables', 'Mattress'): 1,
    ('Consumer Durables', 'Miscellaneous'): 5,
    ('Consumer Durables', 'Opalware'): 1,
    ('Consumer Durables', 'Photographic and Allied Products'): 1,
    ('Consumer Durables', 'Trading'): 1,
    ('Consumer Durables', 'Watches'): 2,
    ('Consumer Durables', 'Watches and Accessories'): 1,
    ('Credit Rating Agencies', 'Credit Rating Agencies'): 2,
    ('Crude Oil and Natural Gas', 'Oil Drilling and Exploration'): 6,
    ('Crude Oil and Natural Gas', 'Oil Exploration and Allied Services'): 3,
    ('Diamond Gems and Jewellery', 'Diamond Gems and Jewellery'): 37,
    ('Diamond Gems and Jewellery', 'Lab Grown Diamonds'): 2,
    ('Diamond Gems and Jewellery', 'Trading'): 4,
    ('Diversified', 'Capital Goods EPC and Cranes'): 1,
    ('Diversified', 'Diversified'): 13,
    ('Diversified', 'Facility Management'): 1,
    ('Diversified', 'Glass and Glass Products'): 1,
    ('Diversified', 'Lubricants'): 1,
    ('Diversified', 'Pharma Others'): 1,
    ('Diversified', 'Printing Publishing and Stationery'): 1,
    ('Diversified', 'Sugar'): 1,
    ('Diversified', 'Textiles Composite Mills'): 1,
    ('Diversified', 'Textiles Spinning'): 1,
    ('Dry Cells', 'Dry Cells'): 2,
    ('Dry Cells', 'Miscellaneous'): 1,
    ('E Commerce and App based Aggregator', 'E Commerce Platform Food'): 2,
    ('E Commerce and App based Aggregator', 'E Commerce Platform Travel'): 3,
    ('E Commerce and App based Aggregator', 'E Commerce Platform Utility'): 11,
    ('E Commerce and App based Aggregator', 'E Services'): 1,
    ('E Commerce and App based Aggregator', 'Entertainment and Media'): 1,
    ('E Commerce and App based Aggregator', 'Finance Investment and Others'): 1,
    ('E Commerce and App based Aggregator', 'Finance and Investments Others'): 1,
    ('E Commerce and App based Aggregator', 'IT Services'): 1,
    ('E Commerce and App based Aggregator', 'Insurance Proxy'): 1,
    ('E Commerce and App based Aggregator', 'Logistics'): 1,
    ('E Commerce and App based Aggregator', 'Medical Equipment'): 1,
    ('E Commerce and App based Aggregator', 'New age Platform E Retail'): 1,
    ('E Commerce and App based Aggregator', 'Textiles Readymade Apparel'): 1,
    ('E Commerce and App based Aggregator', 'Trading'): 4,
    ('E Commerce and App based Aggregator', 'Transport Road'): 1,
    ('E Commerce and App based Aggregator', 'Travel Agencies'): 1,
    ('Edible Oil', 'Edible Oils Agro Processing'): 5,
    ('Edible Oil', 'Solvent Extraction'): 4,
    ('Edible Oil', 'Trading'): 1,
    ('Education', 'Computer Education'): 3,
    ('Education', 'Miscellaneous'): 7,
    ('Education', 'Platform Education'): 3,
    ('Electronics', 'Aerospace and Defence Equipment'): 1,
    ('Electronics', 'CCTV Camera'): 1,
    ('Electronics', 'EMS'): 3,
    ('Electronics', 'Electronics Equipment and Components'): 1,
    ('Electronics', 'Electronics Others'): 6,
    ('Electronics', 'Electronics Power Devices and Equipment'): 1,
    ('Electronics', 'IT Services'): 2,
    ('Electronics', 'Railways Kavach and Springs'): 1,
    ('Engineering', 'Capital Goods Others'): 1,
    ('Engineering', 'Castings Forgings and Fasteners'): 1,
    ('Engineering', 'Electrical Equipment and HVDC'): 1,
    ('Engineering', 'Engineering General'): 4,
    ('Engineering', 'Engineering Heavy General'): 4,
    ('Engineering', 'Engineering Light General'): 4,
    ('Engineering', 'Engineering Light Tools and Moulds'): 1,
    ('Engineering', 'Engineering R and D'): 1,
    ('Engineering', 'Engineering and Infrastructure'): 1,
    ('Engineering', 'Hydraulics'): 1,
    ('Engineering', 'Machine Tools Others'): 1,
    ('Engineering', 'Refrigeration'): 1,
    ('Engineering', 'Services Others'): 1,
    ('Engineering', 'Textile Machinery'): 2,
    ('Entertainment', 'Amusement Parks'): 3,
    ('FMCG', 'Agricultural Processing Maize'): 3,
    ('FMCG', 'Aquaculture'): 4,
    ('FMCG', 'Cigarettes and Tobacco Products'): 1,
    ('FMCG', 'Contraceptives and Protectives'): 1,
    ('FMCG', 'Detergents and Intermediates'): 1,
    ('FMCG', 'Edible Oils Agro Processing'): 2,
    ('FMCG', 'FMCG Animal and Poultry'): 5,
    ('FMCG', 'FMCG Chocolate'): 2,
    ('FMCG', 'FMCG Contract Mfg'): 4,
    ('FMCG', 'FMCG Dairy Products'): 8,
    ('FMCG', 'FMCG Dry Fruits'): 2,
    ('FMCG', 'FMCG Foods'): 6,
    ('FMCG', 'FMCG Personal Care'): 11,
    ('FMCG', 'FMCG Processing Other'): 3,
    ('FMCG', 'FMCG Shrimp'): 2,
    ('FMCG', 'FMCG Snacks'): 4,
    ('FMCG', 'FMCG and Soap Ingredients'): 1,
    ('FMCG', 'Floriculture and Tissue Culture'): 2,
    ('FMCG', 'Food Processing Atta and Rava and Sooji'): 4,
    ('FMCG', 'Food Processing Egg Powder'): 1,
    ('FMCG', 'Food Processing Others'): 18,
    ('FMCG', 'Food Processing Rice'): 7,
    ('FMCG', 'Food Processing Spices and Pickles'): 3,
    ('FMCG', 'Food and Dairy Products'): 4,
    ('FMCG', 'Miscellaneous'): 2,
    ('FMCG', 'Natural Extracts and Food Ingredients'): 1,
    ('FMCG', 'New age Platform E Retail'): 1,
    ('FMCG', 'Personal Care'): 2,
    ('FMCG', 'Seeds and Tissue Culture and Bio Technology'): 2,
    ('FMCG', 'Starch'): 2,
    ('FMCG', 'Sugar'): 1,
    ('FMCG', 'Trading'): 4,
    ('Ferro Alloys', 'Castings Steel and Alloy'): 1,
    ('Ferro Alloys', 'Ferro Alloys'): 5,
    ('Ferro Alloys', 'Mining and Minerals'): 1,
    ('Fertilizers', 'Fertilisers'): 14,
    ('Fertilizers', 'Fertilizers Nitrogenous and Phosphatic'): 1,
    ('Fertilizers', 'Phosphatic Fertilizers Single Super Phosphate'): 2,
    ('Finance', 'Computer Education'): 1,
    ('Finance', 'Finance Holding Company'): 10,
    ('Finance', 'Finance Housing'): 1,
    ('Finance', 'Finance Investment and Others'): 5,
    ('Finance', 'Finance PSU Lending'): 1,
    ('Finance', 'Finance and Investments Others'): 4,
    ('Finance', 'Miscellaneous'): 2,
    ('Finance', 'NBFC Holding Companies'): 1,
    ('Finance', 'NBFC Others'): 3,
    ('Finance', 'Realty Construction and Contracting'): 1,
    ('Financial Services', 'Capital Markets RTA'): 2,
    ('Financial Services', 'Exchanges'): 3,
    ('Financial Services', 'Finance Capital Markets'): 1,
    ('Gas Distribution', 'Gas Distribution'): 8,
    ('Glass and Glass Products', 'Glass Others'): 1,
    ('Glass and Glass Products', 'Glass and Glass Products'): 5,
    ('Glass and Glass Products', 'Glass lining Equipment'): 3,
    ('Glass and Glass Products', 'Opalware'): 2,
    ('Glass and Glass Products', 'Packaging FMCG and Consumers'): 1,
    ('Healthcare', 'Diagnostic Services'): 11,
    ('Healthcare', 'Electronics Others'): 1,
    ('Healthcare', 'Hospitals'): 19,
    ('Healthcare', 'Hospitals and Medical Services'): 14,
    ('Healthcare', 'Medical Equipment'): 7,
    ('Healthcare', 'Medical Equipment and Diagnostic Technology'): 1,
    ('Healthcare', 'Miscellaneous'): 3,
    ('Healthcare', 'Pharma Formulations'): 1,
    ('Healthcare', 'Services Others'): 1,
    ('Healthcare', 'Trading'): 1,
    ('Hotels and Restaurants', 'Airport Restaurants and Lounges'): 1,
    ('Hotels and Restaurants', 'Hotels'): 33,
    ('Hotels and Restaurants', 'Hotels Resorts'): 3,
    ('Hotels and Restaurants', 'Quick Service Restaurant'): 8,
    ('Hotels and Restaurants', 'Resorts'): 3,
    ('Hotels and Restaurants', 'Restaurants'): 1,
    ('IT Hardware', 'Computer Hardware'): 4,
    ('IT Hardware', 'Computer Peripherals and Accessories'): 1,
    ('IT Hardware', 'Data Centre'): 1,
    ('IT Hardware', 'IT Networking Equipment'): 3,
    ('IT Hardware', 'IT Services'): 1,
    ('IT Hardware', 'Technology Distribution'): 1,
    ('IT Hardware', 'Trading'): 5,
    ('IT Software', 'Advertisement'): 1,
    ('IT Software', 'Data Centre'): 2,
    ('IT Software', 'Engineering R and D'): 7,
    ('IT Software', 'Geospatial'): 3,
    ('IT Software', 'IT Enabled Services'): 12,
    ('IT Software', 'IT Enabled Services and BPO'): 11,
    ('IT Software', 'IT Networking Equipment'): 1,
    ('IT Software', 'IT Product Companies'): 10,
    ('IT Software', 'IT Services'): 110,
    ('IT Software', 'Miscellaneous'): 1,
    ('IT Software', 'Publishing and Learning Services'): 1,
    ('IT Software', 'Services Others'): 2,
    ('Infrastructure Developers and Operators', 'Cables Telecom'): 1,
    ('Infrastructure Developers and Operators', 'Capital Goods Engineering Heavy'): 1,
    ('Infrastructure Developers and Operators', 'Cement Products'): 2,
    ('Infrastructure Developers and Operators', 'Construction Civil and Turnkey'): 33,
    ('Infrastructure Developers and Operators', 'Construction and Contracting'): 4,
    ('Infrastructure Developers and Operators', 'Data Centre'): 1,
    ('Infrastructure Developers and Operators', 'EPC'): 4,
    ('Infrastructure Developers and Operators', 'Engineering Turnkey Services'): 17,
    ('Infrastructure Developers and Operators', 'General Electric Equipment'): 1,
    ('Infrastructure Developers and Operators', 'Infra Construction and Contracting'): 20,
    ('Infrastructure Developers and Operators', 'Infra Engineering General'): 8,
    ('Infrastructure Developers and Operators', 'Infra General'): 2,
    ('Infrastructure Developers and Operators', 'Infra Power Generation and Distribution'): 4,
    ('Infrastructure Developers and Operators', 'Mining and Minerals'): 1,
    ('Infrastructure Developers and Operators', 'Power Transmission and Equipment'): 1,
    ('Infrastructure Developers and Operators', 'Project Consultancy and Turnkey'): 5,
    ('Infrastructure Developers and Operators', 'Pumps'): 1,
    ('Infrastructure Developers and Operators', 'Realty Commercial'): 1,
    ('Infrastructure Developers and Operators', 'Realty Construction and Contracting'): 1,
    ('Infrastructure Developers and Operators', 'Solar EPC'): 2,
    ('Infrastructure Developers and Operators', 'Solar Pumps'): 1,
    ('Infrastructure Developers and Operators', 'Transmission Line Towers and Equipment'): 2,
    ('Infrastructure Developers and Operators', 'Water Treatment'): 3,
    ('Infrastructure Investment Trusts', 'Infra and Real Estate Investment Trust'): 2,
    ('Infrastructure Investment Trusts', 'Infrastructure Investment Trusts'): 2,
    ('Insurance', 'Insurance Distribution'): 1,
    ('Leather', 'Footwear'): 8,
    ('Leather', 'Leather Products Others'): 1,
    ('Leather', 'Plastics Plastic and Plastic Products'): 1,
    ('Leather', 'Textiles'): 1,
    ('Logistics', 'Air Transport Service'): 1,
    ('Logistics', 'Logistics'): 15,
    ('Logistics', 'Logistics Warehousing and Supply Chain'): 10,
    ('Logistics', 'Miscellaneous'): 13,
    ('Logistics', 'New age Platform E Retail'): 1,
    ('Logistics', 'Transport Road'): 6,
    ('Marine Port and Services', 'Logistics'): 1,
    ('Marine Port and Services', 'Marine Port and Services'): 3,
    ('Media Print and Television and Radio', 'Advertisement'): 1,
    ('Media Print and Television and Radio', 'Entertainment and Media'): 2,
    ('Media Print and Television and Radio', 'Miscellaneous'): 2,
    ('Media Print and Television and Radio', 'Newspaper'): 3,
    ('Mining and Mineral Products', 'Metal Copper and Copper Alloy Products'): 1,
    ('Mining and Mineral Products', 'Metal Others'): 1,
    ('Mining and Mineral Products', 'Mining and Minerals'): 14,
    ('Mining and Mineral Products', 'Mining and Minerals Iron Ore'): 3,
    ('Miscellaneous', 'Advertisement'): 1,
    ('Miscellaneous', 'Capital Goods EPC and Cranes'): 1,
    ('Miscellaneous', 'Cement'): 1,
    ('Miscellaneous', 'Chemicals Inorganic'): 1,
    ('Miscellaneous', 'Contraceptives and Protectives'): 1,
    ('Miscellaneous', 'E Commerce Platform Travel'): 1,
    ('Miscellaneous', 'Entertainment and Media'): 1,
    ('Miscellaneous', 'Facility Management'): 3,
    ('Miscellaneous', 'Finance and Investments Others'): 2,
    ('Miscellaneous', 'Infotech and Database'): 1,
    ('Miscellaneous', 'Infra Construction and Contracting'): 1,
    ('Miscellaneous', 'Insurance Proxy'): 1,
    ('Miscellaneous', 'LPG Bottling'): 1,
    ('Miscellaneous', 'Lab Grown Diamonds'): 1,
    ('Miscellaneous', 'Logistics'): 1,
    ('Miscellaneous', 'Miscellaneous'): 52,
    ('Miscellaneous', 'Platform Matrimony'): 1,
    ('Miscellaneous', 'Railways'): 2,
    ('Miscellaneous', 'Recycling'): 3,
    ('Miscellaneous', 'Rubber Products'): 1,
    ('Miscellaneous', 'Services Others'): 8,
    ('Miscellaneous', 'Shipping'): 2,
    ('Miscellaneous', 'Starch'): 1,
    ('Miscellaneous', 'Transport Road'): 1,
    ('Miscellaneous', 'Travel Agencies'): 1,
    ('Miscellaneous', 'Tyres and Tubes'): 1,
    ('Non Ferrous Metals', 'Aluminium'): 4,
    ('Non Ferrous Metals', 'Aluminium Products'): 3,
    ('Non Ferrous Metals', 'Aluminium Sheets and Coils and Wires and Others'): 6,
    ('Non Ferrous Metals', 'Chemicals Inorganic'): 1,
    ('Non Ferrous Metals', 'Copper Wires'): 2,
    ('Non Ferrous Metals', 'Electrical Equipment and HVDC'): 1,
    ('Non Ferrous Metals', 'Metal Copper and Copper Alloy Products'): 5,
    ('Non Ferrous Metals', 'Metal Others'): 2,
    ('Non Ferrous Metals', 'Metals'): 2,
    ('Non Ferrous Metals', 'Mining and Minerals'): 1,
    ('Non Ferrous Metals', 'Recycling'): 4,
    ('Non Ferrous Metals', 'Shunt Resistors'): 1,
    ('Non Ferrous Metals', 'Steel Wires'): 1,
    ('Oil Drill and Allied', 'Oil Drilling and Exploration'): 2,
    ('Packaging', 'Cylinder'): 1,
    ('Packaging', 'Packaging BOPP'): 4,
    ('Packaging', 'Packaging BOPP Self adhesive Tape'): 1,
    ('Packaging', 'Packaging FMCG and Consumers'): 6,
    ('Packaging', 'Packaging Films'): 3,
    ('Packaging', 'Packaging Laminates'): 2,
    ('Packaging', 'Packaging Metallic'): 4,
    ('Packaging', 'Packaging Others'): 9,
    ('Packaging', 'Packaging Plastic Containers'): 3,
    ('Packaging', 'Packaging Polysacks'): 4,
    ('Packaging', 'Packaging and Containers'): 3,
    ('Packaging', 'Plastics Plastic Containers'): 1,
    ('Packaging', 'Textiles Technical Textile'): 1,
    ('Paints and Varnish', 'Building Materials Paints'): 7,
    ('Paints and Varnish', 'Paints and Varnishes'): 1,
    ('Paper', 'Paper'): 20,
    ('Paper', 'Paper Packaging'): 1,
    ('Petrochemicals', 'Petrochemicals Others'): 6,
    ('Petrochemicals', 'Petrochemicals Polymers'): 5,
    ('Petrochemicals', 'Specialty Chemicals'): 1,
    ('Pharmaceuticals', 'Ayurvedic'): 2,
    ('Pharmaceuticals', 'Biotechnology'): 1,
    ('Pharmaceuticals', 'Chemicals Organic'): 2,
    ('Pharmaceuticals', 'Pharma API'): 25,
    ('Pharmaceuticals', 'Pharma Animal'): 3,
    ('Pharmaceuticals', 'Pharma Bulk Drugs'): 11,
    ('Pharmaceuticals', 'Pharma CRAMS'): 15,
    ('Pharmaceuticals', 'Pharma Distribution'): 4,
    ('Pharmaceuticals', 'Pharma Formulations'): 66,
    ('Pharmaceuticals', 'Pharma Gelatin Capsules'): 1,
    ('Pharmaceuticals', 'Pharma IV Fluids'): 3,
    ('Pharmaceuticals', 'Pharma Integrated'): 3,
    ('Pharmaceuticals', 'Pharma MNC Bulk Drugs'): 8,
    ('Pharmaceuticals', 'Pharma Others'): 7,
    ('Pharmaceuticals', 'Seeds and Tissue Culture and Bio Technology'): 1,
    ('Pharmaceuticals', 'Trading'): 4,
    ('Plantation and Plantation Products', 'FMCG Coffee'): 1,
    ('Plantation and Plantation Products', 'Miscellaneous'): 1,
    ('Plantation and Plantation Products', 'Plantations Tea and Coffee'): 2,
    ('Plantation and Plantation Products', 'Rubber Processing and Rubber Products'): 6,
    ('Plantation and Plantation Products', 'Rubber Products'): 5,
    ('Plantation and Plantation Products', 'Tea'): 11,
    ('Plastic Products', 'Building Materials Plastic Pipes'): 5,
    ('Plastic Products', 'Cylinder'): 1,
    ('Plastic Products', 'Irrigation System'): 1,
    ('Plastic Products', 'Miscellaneous'): 1,
    ('Plastic Products', 'Plastics Drip Irrigation'): 3,
    ('Plastic Products', 'Plastics Furniture'): 1,
    ('Plastic Products', 'Plastics Others'): 15,
    ('Plastic Products', 'Plastics Pipes and Fittings'): 4,
    ('Plastic Products', 'Plastics Plastic and Plastic Products'): 7,
    ('Plastic Products', 'Plastics Sheets and Films'): 4,
    ('Plastic Products', 'Plastics Thermoware'): 1,
    ('Plastic Products', 'Travel Bags'): 2,
    ('Plywood Boards and Laminates', 'Decoratives Wood based'): 3,
    ('Plywood Boards and Laminates', 'Laminates'): 1,
    ('Plywood Boards and Laminates', 'Plywood Boards and Laminates'): 6,
    ('Plywood Boards and Laminates', 'Trading'): 1,
    ('Power Generation and Distribution', 'Electrical Equipment and HVDC'): 1,
    ('Power Generation and Distribution', 'Power Generation and Distribution'): 21,
    ('Power Generation and Distribution', 'Power Generation and Supply'): 2,
    ('Power Generation and Distribution', 'Power Transmission and Equipment'): 2,
    ('Power Generation and Distribution', 'Recycling'): 1,
    ('Power Generation and Distribution', 'Solar Power and EPC'): 1,
    ('Power Generation and Distribution', 'Trading'): 1,
    ('Power Infrastructure', 'Engineering Turnkey Services'): 1,
    ('Power Infrastructure', 'Solar EPC'): 3,
    ('Printing and Stationery', 'Printing Publishing and Stationery'): 2,
    ('Printing and Stationery', 'Printing and Stationery'): 5,
    ('Printing and Stationery', 'Secure Printing and Payment Cards'): 1,
    ('Railways', 'Railway Equipment'): 1,
    ('Railways', 'Railways'): 2,
    ('Readymade Garments and Apparel', 'Textiles Hosiery and Knitwear'): 2,
    ('Readymade Garments and Apparel', 'Textiles Readymade Apparel'): 13,
    ('Readymade Garments and Apparel', 'Textiles Spinning'): 1,
    ('Real Estate Investment Trusts', 'Infra and Real Estate Investment Trust'): 5,
    ('Real Estate Investment Trusts', 'Real Estate Investment Trusts'): 4,
    ('Realty', 'Construction Factories and Offices and Commercial'): 21,
    ('Realty', 'Construction Housing'): 7,
    ('Realty', 'Construction and Contracting'): 3,
    ('Realty', 'Data Centre'): 1,
    ('Realty', 'Diversified'): 2,
    ('Realty', 'EPC'): 1,
    ('Realty', 'Electronics Equipment and Components'): 1,
    ('Realty', 'Engineering General'): 1,
    ('Realty', 'Entertainment and Media'): 2,
    ('Realty', 'Finance Capital Markets Brokers'): 1,
    ('Realty', 'Finance Holding Company'): 1,
    ('Realty', 'Finance and Investments Others'): 1,
    ('Realty', 'Miscellaneous'): 3,
    ('Realty', 'Power Generation and Distribution'): 1,
    ('Realty', 'Realty CoWorking'): 1,
    ('Realty', 'Realty Commercial'): 2,
    ('Realty', 'Realty Construction and Contracting'): 23,
    ('Realty', 'Realty National'): 7,
    ('Realty', 'Realty Regional'): 8,
    ('Refineries', 'Biofuel'): 1,
    ('Refineries', 'Refineries'): 8,
    ('Refractories', 'Refractories'): 7,
    ('Retail', 'Ceramics Tiles and Sanitaryware'): 1,
    ('Retail', 'FMCG Coffee'): 1,
    ('Retail', 'Footwear'): 1,
    ('Retail', 'New age Platform E Retail'): 1,
    ('Retail', 'Pharma Distribution'): 1,
    ('Retail', 'Platform Others'): 1,
    ('Retail', 'Retail Departmental Stores'): 6,
    ('Retail', 'Retail Electronics'): 3,
    ('Retail', 'Retail Vehicles'): 2,
    ('Retail', 'Textiles Readymade Apparel'): 9,
    ('Retail', 'Trading'): 15,
    ('Retail', 'Travel Bags'): 1,
    ('Retail', 'Watches'): 2,
    ('Ship Building', 'Shipbuilding and Repairs'): 1,
    ('Ship Building', 'Shipping'): 1,
    ('Ship Building', 'Shipyards Dockrepairing'): 1,
    ('Shipping', 'Shipping'): 4,
    ('Shipping', 'Shipping and Dredging'): 3,
    ('Steel', 'CRGO Pipe'): 1,
    ('Steel', 'Capital Goods Transformers'): 1,
    ('Steel', 'DI Pipes and Saw Pipes'): 1,
    ('Steel', 'Mining and Minerals Iron Ore'): 3,
    ('Steel', 'Miscellaneous'): 1,
    ('Steel', 'Pre Engineering Buildings'): 3,
    ('Steel', 'Railways'): 1,
    ('Steel', 'Scaffolding'): 1,
    ('Steel', 'Stainless Steel'): 3,
    ('Steel', 'Steel'): 25,
    ('Steel', 'Steel Pig Iron'): 2,
    ('Steel', 'Steel Products'): 16,
    ('Steel', 'Steel Rolling'): 5,
    ('Steel', 'Steel Sponge Iron'): 7,
    ('Steel', 'Steel Tubes and Pipes'): 20,
    ('Steel', 'Steel Wires'): 6,
    ('Steel', 'Trading'): 1,
    ('Stock and Commodity Brokers', 'Finance Capital Markets Brokers'): 1,
    ('Stock and Commodity Brokers', 'Finance Capital Markets Wealth Management'): 1,
    ('Sugar', 'Sugar'): 19,
    ('Sugar', 'Sugar Integrated'): 9,
    ('Sugar', 'Sugar Others'): 1,
    ('Telecom Equipment and Infra Services', 'Engineering Turnkey Services'): 1,
    ('Telecom Equipment and Infra Services', 'Telecom Services'): 7,
    ('Telecom Equipment and Infra Services', 'Telecommunications Equipment'): 5,
    ('Telecom Equipment and Infra Services', 'Telecommunications Equipment and Network EPC'): 1,
    ('Telecom Service', 'Entertainment and Media'): 1,
    ('Telecom Service', 'Telecom Services'): 9,
    ('Textiles', 'Diversified'): 2,
    ('Textiles', 'Leather Products Others'): 1,
    ('Textiles', 'Mattress'): 1,
    ('Textiles', 'Packaging BOPP'): 1,
    ('Textiles', 'Recycling'): 1,
    ('Textiles', 'Services Others'): 1,
    ('Textiles', 'Textiles'): 1,
    ('Textiles', 'Textiles Acrylic Fibre'): 2,
    ('Textiles', 'Textiles Composite Cotton and Blended Fabric'): 1,
    ('Textiles', 'Textiles Composite Mills'): 3,
    ('Textiles', 'Textiles Cotton Yarn EOUs'): 3,
    ('Textiles', 'Textiles Cotton Yarn Open End Spinning'): 1,
    ('Textiles', 'Textiles Denim Fabric'): 4,
    ('Textiles', 'Textiles General'): 4,
    ('Textiles', 'Textiles Home Textile'): 5,
    ('Textiles', 'Textiles Hosiery and Knitwear'): 4,
    ('Textiles', 'Textiles Jute Yarn and Products'): 1,
    ('Textiles', 'Textiles Jute and Jute Products'): 3,
    ('Textiles', 'Textiles Manmade PFY and PSF'): 5,
    ('Textiles', 'Textiles Others'): 19,
    ('Textiles', 'Textiles Processing'): 1,
    ('Textiles', 'Textiles Processing and Texturising'): 5,
    ('Textiles', 'Textiles Rayon'): 1,
    ('Textiles', 'Textiles Readymade Apparel'): 8,
    ('Textiles', 'Textiles Socks'): 1,
    ('Textiles', 'Textiles Spinning'): 15,
    ('Textiles', 'Textiles Spinning Synthetic and Blended'): 5,
    ('Textiles', 'Textiles Spinning and Cotton and Blended'): 12,
    ('Textiles', 'Textiles Technical Textile'): 2,
    ('Textiles', 'Textiles Texturising'): 2,
    ('Textiles', 'Textiles Weaving'): 9,
    ('Textiles', 'Textiles Worsted Fabric'): 2,
    ('Textiles', 'Trading'): 7,
    ('Tobacco Products', 'Cigarettes and Tobacco Products'): 3,
    ('Trading', 'Capital Goods Solar'): 1,
    ('Trading', 'Cigarettes and Tobacco Products'): 1,
    ('Trading', 'Computer Hardware'): 1,
    ('Trading', 'Diversified'): 1,
    ('Trading', 'Engineering General'): 1,
    ('Trading', 'FMCG Personal Care'): 1,
    ('Trading', 'Logistics'): 1,
    ('Trading', 'New age Platform E Retail'): 2,
    ('Trading', 'Packaging and Containers'): 1,
    ('Trading', 'Refrigeration'): 1,
    ('Trading', 'Textiles Readymade Apparel'): 1,
    ('Trading', 'Trading'): 97,
    ('Tyres', 'Tyres'): 1,
    ('Tyres', 'Tyres and Tubes'): 8,
}
