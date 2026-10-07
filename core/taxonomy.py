"""The vendor's industry and sector names → the names PRISM's code is written in (2026-10-07).

WHY. Between the 2026-09-19 and 2026-10-06 vintages the data vendor re-spelled its taxonomy:
"IT - Software" → "IT Software", "Diamond, Gems & Jewellery" → "Diamond Gems and Jewellery",
"Auto Ancillaries - Gears" → "Auto Gears". PRISM keys on these names in ~50 places: the cyclicality
map (core/cyclicality_map.py, which caps fair_pe_qglp for tier A), config.FINANCIAL_SECTORS (which
picks the financial gate set), the Atoms/Bits and commodity-peak sector sets, the forensic sector
thresholds. On the 2026-10-06 vintage 283 of 376 industries and 31 of 81 sectors no longer matched,
and silently, because every lookup falls back without an error. Measured on that vintage (the full
pipeline run twice, vendor names vs these): 896 of 2,712 stocks carried a wrong cyclicality tier,
19 financials (REITs, InvITs, brokers, holding companies) had lost is_financial, 29 utilities
is_utility, 63 the right fair_pe_qglp; IT companies were judged on the default receivables
threshold instead of their own. composite_score moved on 423 stocks.

WHAT. canonicalize_taxonomy() runs once, at the end of merge_datasets, and rewrites each vendor
name to the PRISM name it is the same as:
  1. exact — already a PRISM name: unchanged.
  2. spelling — equal after normalize() (case, "&" vs "and", punctuation): 1:1 by construction,
     and test_taxonomy proves no two PRISM names normalize alike.
  3. renamed — VENDOR_RENAMES, a reviewed table of plain relabels ("Auto Gears" was "Auto
     Ancillaries - Gears"; the vendor also corrected typos PRISM's names keep: "Fastners", "Polutry").
  4. anything else is kept as the vendor wrote it. A genuinely new industry then takes its tier
     from its (translated) sector, exactly as before the rename.
Then SECTOR_OF_INDUSTRY puts back the one industry the vendor moved to another sector.
Only relabels whose meaning is unmistakable are in VENDOR_RENAMES. Where the vendor regrouped
("IT Services", "Pharma Formulations") or a name could fit two old ones, the vendor's name is kept
and the sector decides — so nothing here invents a classification.

Because the map generator reads the same merge_datasets frame, it sees PRISM names too, and a
regeneration cannot add the vendor's new spellings beside the old ones.
"""
import re

import pandas as pd

from config import (CONSISTENT_SECTORS, EPOCH5_MODERN, FINANCIAL_SECTOR_NAMES, FINANCIAL_SECTORS,
                    UTILITY_SECTOR_NAMES)
from core.cyclicality_map import INDUSTRY_TIER, SECTOR_TIER_FALLBACK

# vendor name (2026-10-06 vintage) → the PRISM name it relabels. Reviewed one by one; every
# target must exist in the vocabulary (test_taxonomy) so a map regeneration cannot orphan one.
VENDOR_RENAMES = {
    "industry": {
        "Aerospace and Defence Equipment": "Aerospace & Defence - Equipments",
        "Auto AC": "Auto Ancillaries - AC",
        "Auto Axle and Brakes": "Auto Ancillaries - Axle & Brakes",
        "Auto Axles": "Auto Ancillaries - Axle shafts",
        "Auto Batteries": "Auto Ancillaries - Batteries",
        "Auto Clutches": "Auto Ancillaries - Clutches",
        "Auto Electric Utility": "Auto Ancillaries - Electric Utility",
        "Auto Engine Parts": "Auto Ancillaries - Engine Parts",
        "Auto Four Wheelers": "Auto - 4 Wheelers",
        "Auto Gears": "Auto Ancillaries - Gears",
        "Auto Lighting": "Auto Ancillaries - Head lamps lights",
        "Auto Others": "Auto Ancillaries - Others",
        "Auto Plastic Parts": "Auto Ancillaries - Plastic Mouldings",
        "Auto Seats": "Auto Ancillaries - Seats",
        "Auto Sheet Metal": "Auto Ancillaries - Sheet Metal",
        "Auto Shock Absorbers": "Auto Ancillaries - Shock Absorber",
        "Auto Spare Parts Accessories": "Auto Ancillaries - Spare Parts Accessories",
        "Auto Springs": "Auto Ancillaries - Springs",
        "Auto Transmission": "Auto Ancillaries - Transmission",
        "Auto Two Wheeler Parts": "Auto Ancillaries - 2 Wheelers",
        "Auto Two and Three Wheeler Parts": "Auto Ancillaries - 2&3 Wheelers",
        "Auto Two and Three Wheelers": "Auto - 2 & 3 Wheelers",
        "Auto Wheels": "Auto Ancillaries - Wheels",
        "Automobiles Scooters and Three Wheelers": "Automobiles - Scooters & 3 - Wheelers",
        "Biofuel": "Bio-fuel",
        "Capital Goods Mining Equipment": "Capital Goods - Mining Equipement",
        "Capital Markets RTA": "Finance - Capital Markets - RTA",
        "Castings Forgings and Fasteners": "Castings, Forgings & Fastners",
        "Chemicals Fluorine": "Chemicals - Flourine",
        "Diversified Auto Components": "Auto Ancillaries - Diversified",
        "Electrical Equipment and HVDC": "Electrical Equipments/HVDC",
        "Electrical Switchgear Relays and Circuits": "Electric Equipment - Switchgears/Relays/Circuits",
        "Engineering R and D": "IT - ER&D",
        "FMCG Animal and Poultry": "FMCG - Animal/Polutry",
        "Glass lining Equipment": "Engineering - Heavy - Glass - lined Equipment",
        "IT Enabled Services and BPO": "IT Enabled Services/Business Process Outsourcing",
        "Inorganic Chemicals Caustic Soda and Soda Ash": "Chemicals - Inorganic - Caustic Soda/Soda Ash",
        "Petrochemicals Polymers": "Petrochem - Polymers",
        "Pharma Bulk Drugs": "Pharmaceuticals Bulk Drugs",
        "Pharma CRAMS": "Pharma - API & CRAMS",
        "Phosphatic Fertilizers Single Super Phosphate": "Fertilizers - Phosphatic - Single Super Phosphate",
        "Quick Service Restaurant": "Quick Service Restaurant - QSR",
        "Shipbuilding and Repairs": "Ship - Docks/Breaking/Repairs",
        "Specialty Chemicals": "Speciality Chemicals",
        "Textiles Cotton Yarn Open End Spinning": "Textiles - Cotton Yarn - Open - Ended Spinning",
        "Welding Equipment": "Welding Equipments",
    },
    "sector": {
        "Castings Forgings and Fasteners": "Castings, Forgings & Fastners",
        "Readymade Garments and Apparel": "Readymade Garments/ Apparells",
    },
}

# Where the vendor MOVED an industry to another sector (not a re-spelling): PRISM keeps the sector it
# had. QSR chains went from "Quick Service Restaurant" into "Hotels and Restaurants" — the sector
# config.EPOCH5_MODERN["tipping_sectors"] and the forensic lease check name, verified against the
# CSV's sector column at the time ("CSV has no trailing 's'").
SECTOR_OF_INDUSTRY = {
    "Quick Service Restaurant - QSR": "Quick Service Restaurant",
}

# The names PRISM's code is written in. Every config name set that keys on these columns is
# included, so a name used only there (never tiered) is still a translation target.
VOCABULARY = {
    "industry": frozenset(INDUSTRY_TIER) | FINANCIAL_SECTORS,
    "sector": (frozenset(SECTOR_TIER_FALLBACK) | FINANCIAL_SECTOR_NAMES | UTILITY_SECTOR_NAMES
               | CONSISTENT_SECTORS | frozenset(EPOCH5_MODERN["tipping_sectors"])),
}


def normalize(name) -> str:
    """Spelling-blind key: lower case, every non-alphanumeric ("&", "-", "/", ",") read as a space,
    the word "and" dropped — "A & B", "A/B", "A - B" and "A and B" all become "a b"."""
    words = re.sub(r"[^a-z0-9]+", " ", str(name).lower()).split()
    return " ".join(w for w in words if w != "and")


def _spelling_index(names) -> dict:
    index = {}
    for n in sorted(names):
        index.setdefault(normalize(n), []).append(n)
    # two PRISM names that normalize alike would make the key ambiguous: leave such a key out
    return {k: v[0] for k, v in index.items() if len(v) == 1}


_SPELLING = {col: _spelling_index(names) for col, names in VOCABULARY.items()}


def canonical_name(name, column: str):
    """One vendor name → the PRISM name (exact, spelling, renamed), or the name itself."""
    if not isinstance(name, str):
        return name
    vocab = VOCABULARY[column]
    if name in vocab:
        return name
    hit = _SPELLING[column].get(normalize(name))
    if hit is not None:
        return hit
    return VENDOR_RENAMES[column].get(name.strip(), name)


def canonicalize_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    """Rewrite df["industry"] and df["sector"] in PRISM's names (in place; returns df)."""
    for col in ("industry", "sector"):
        if col not in df.columns:
            continue
        values = df[col].dropna().unique()
        mapping = {v: canonical_name(v, col) for v in values}
        changed = {v: c for v, c in mapping.items() if c != v}
        if changed:
            df[col] = df[col].map(lambda v: changed.get(v, v))
        unknown = sorted(str(c) for c in set(mapping.values()) if c not in VOCABULARY[col])
        print(f"  🏷️ {col}: {len(changed)} of {len(values)} vendor names translated to PRISM names"
              + (f"; {len(unknown)} new to PRISM, kept: {', '.join(unknown[:8])}"
                 + (" …" if len(unknown) > 8 else "") if unknown else ""))
    if "industry" in df.columns and "sector" in df.columns:
        moved = df["industry"].isin(SECTOR_OF_INDUSTRY)
        if moved.any():
            df.loc[moved, "sector"] = df.loc[moved, "industry"].map(SECTOR_OF_INDUSTRY)
    return df
