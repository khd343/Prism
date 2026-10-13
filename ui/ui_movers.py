"""PRISM — 🔁 Movers: what changed between two data vintages.

The engine's whole thesis is that DIRECTION beats level (Vel%, the 28th WCS), yet every other
surface shows one instant. This module diffs two vintages of the scored frame — the live sheet
against an archived Drive copy — and reports the moves: wealth-tier upgrades, soundness flips,
rank jumps, new gate passers, red-flag rises, fresh results.

THREE RULES THIS MODULE IS BUILT ON
  1. Same engine on both sides, by construction. The caller re-scores the archived RAW copy with
     the running engine (app.py `_load_vintage`), so a difference can only be "the company
     changed" — never "PRISM changed". A scored file from the past would confound the two.
  2. Never a fabricated delta. A stock present on one side only is NEW or DROPPED; a NaN on
     either side of a numeric column drops that row from that section. No sentinels, no fillna
     into a delta (CLAUDE.md semantic-truth mandate). `.eq(1)` treats NaN as "not a move".
  3. Zero scoring logic and zero mutation. Reads columns compute_verdict / run_full_scoring
     already produce; works on column subsets; both inputs come back byte-identical.

STATELESS, like ui_tearsheet: no st.button/slider/number_input, no st.columns/st.metric — the
header strip is inline flex. Widgets (the vintage picker, the compare button) live in app.py.
"""
import re

import numpy as np
import pandas as pd
import streamlit as st

from config import COLORS
from ui.ui_export import _to_csv_bytes

JOIN_KEY = "company_id"
# The wealth ladder, best first. N/A means "an input is missing — never condemned on absent
# evidence"; it sits OUTSIDE the ladder so a move to or from it is reported as unverifiable,
# never as an upgrade or a downgrade.
WEALTH_LADDER = ["BUY★", "BUY", "WATCH★", "WATCH", "AVOID"]
UNVERIFIABLE = "N/A"
SOUND_LADDER = ["SOUND", "MIXED", "FLAWED"]

# Every column the diff reads. Only those present on BOTH sides are diffed; a column missing from
# one vintage simply yields an empty section rather than a crash — an older engine may not have
# emitted it, and honesty beats coverage.
_COLS = ["name", "sector", "market_category", "wealth_tier", "verdict_direction",
         "conviction_tier", "rank", "composite_score", "gate_pass", "tsunami_signal",
         "red_flag_count", "result_age_days"]


# ── Vintage bookkeeping (pure) ────────────────────────────────────────────────
def fy_quarter(iso: str) -> str:
    """Indian-FY quarter label for a vintage date, by RESULT-FILING window — a byte-for-byte
    mirror of Prism.gs::_fyQuarter (the archiver), pinned by the same five cases so the label
    PRISM computes for the live side can never disagree with the label the index carries for
    the archived side. A vintage belongs to a quarter once that quarter's deadline has passed
    (30 May Q4 · 14 Aug Q1 · 14 Nov Q2 · 14 Feb Q3); captured in the deadline's month or the one
    after = on-cycle, later = "(off-cycle)"."""
    y, mo, _ = (int(p) for p in iso.split("-"))
    md = iso[5:]
    gates = [("02-14", 2, "Q3", y), ("05-30", 5, "Q4", y), ("08-14", 8, "Q1", y + 1),
             ("11-14", 11, "Q2", y + 1)]
    passed = None
    for g in gates:
        if md >= g[0]:
            passed = g
    if passed is None:
        passed = ("", -1, "Q2", y)
    label = f"FY{str(passed[3])[2:]}{passed[2]}"
    months_since = 99 if passed[1] < 0 else mo - passed[1]
    return label if months_since <= 1 else f"{label} (off-cycle)"


def usable_vintages(index_df: pd.DataFrame) -> pd.DataFrame:
    """The archive index → the rows Movers may load: status 'ok' only, newest first, dates as
    'YYYY-MM-DD' text. Raises if the sheet is not a PRISM Archive Index — a wrong id must fail
    loud, not render an empty picker."""
    need = {"vintage_date", "fy_quarter", "spreadsheet_id", "status"}
    missing = sorted(need - set(map(str, index_df.columns)))
    if missing:
        raise ValueError(f"not a PRISM Archive Index — missing columns {missing}")
    ok = index_df[index_df["status"].astype(str).str.strip() == "ok"].copy()
    ok["vintage_date"] = ok["vintage_date"].astype(str).str[:10]
    ok["fy_quarter"] = ok["fy_quarter"].astype(str).str.strip()
    ok["spreadsheet_id"] = ok["spreadsheet_id"].astype(str).str.strip()
    ok = ok.sort_values("vintage_date", ascending=False, kind="mergesort").reset_index(drop=True)
    return ok[["vintage_date", "fy_quarter", "spreadsheet_id"]]


def default_vintage(ok: pd.DataFrame):
    """The row to compare against by default: the most recent ON-CYCLE quarter (a clean
    quarter boundary), else the most recent row of any kind, else None."""
    if ok.empty:
        return None
    on_cycle = ok[~ok["fy_quarter"].str.contains("off-cycle", regex=False)]
    return (on_cycle if not on_cycle.empty else ok).iloc[0]


# The Movers picker offers ONE copy per quarter (2026-10-07): the copy at a CALENDAR quarter-end. It
# is the cleanest state of a results quarter — every company has filed (the deadline has passed) and
# the next season has not begun:
#   30 Jun: Jan-Mar results with the annual (deadline 30 May; next season mid-July)
#   30 Sep: Apr-Jun results (deadline 14 Aug; next season October)
#   31 Dec: Jul-Sep results, SME half-years too (deadline 14 Nov; next season January)
#   31 Mar: Oct-Dec results (deadline 14 Feb; next season April)
# A copy taken mid-season mixes "the company changed" with "the company had not reported yet". The
# archive still keeps every month-end copy (the scan records and the 🆕 baseline read them); any one
# of them can still be compared by pasting its own id.
QUARTER_END_MONTHS = (3, 6, 9, 12)
RESULTS_OF = {"Q1": "Apr–Jun", "Q2": "Jul–Sep", "Q3": "Oct–Dec", "Q4": "Jan–Mar + annual"}


def quarter_end_vintages(ok: pd.DataFrame) -> pd.DataFrame:
    """usable_vintages rows → one per quarter: the newest copy dated in a quarter-end month (Mar,
    Jun, Sep, Dec), newest quarter first. Empty when the archive holds none yet."""
    month = pd.to_datetime(ok["vintage_date"], errors="coerce").dt.month
    q = ok[month.isin(QUARTER_END_MONTHS)]
    # the month key from the FILTERED rows: assigning a Series of the unfiltered frame to an empty
    # selection makes pandas invent a row of NaN — a blank entry in the picker
    q = q.assign(_ym=q["vintage_date"].str[:7])
    q = q.sort_values("vintage_date", ascending=False, kind="mergesort").drop_duplicates("_ym")
    return q.drop(columns="_ym").reset_index(drop=True)


def quarter_label(iso: str) -> str:
    """'Apr–Jun results (FY27 Q1) · 30 Sep 2026' — which results a quarter-end copy holds, from the
    same fy_quarter the archiver writes (always on-cycle in a quarter-end month)."""
    code = fy_quarter(iso).split(" ")[0]                     # 'FY27Q1'
    fy, q = code[:4], code[4:]
    day = pd.Timestamp(iso)
    return f"{RESULTS_OF[q]} results ({fy} {q}) · {day.day} {day:%b %Y}"


_SHEETS_URL_ID = re.compile(r"(/spreadsheets/d/)[A-Za-z0-9_-]+")


def redact_ids(text, *ids) -> str:
    """An error the Movers tab prints, with every sheet id taken out: each known id, and any id inside a
    Google Sheets URL, reads '[sheet id]'. The app is public (2026-10-13), and the archive's id is a secret,
    logged by shape, never by value. The longest id goes first, so no shorter one leaves a fragment."""
    out = _SHEETS_URL_ID.sub(r"\1[sheet id]", str(text))
    for known in sorted({str(i).strip() for i in ids if i and str(i).strip()}, key=lambda s: (-len(s), s)):
        out = out.replace(known, "[sheet id]")
    return out


def load_index(sheet_id: str) -> pd.DataFrame:
    """Fetch the archive index by id (the same XLSX export path the data loader uses — never
    per-tab CSV, never gviz) and return usable_vintages of its first tab."""
    from core.data_engine import _xlsx_engine, extract_spreadsheet_id
    xid = extract_spreadsheet_id(sheet_id)
    wb = pd.ExcelFile(f"https://docs.google.com/spreadsheets/d/{xid}/export?format=xlsx",
                      engine=_xlsx_engine())
    return usable_vintages(wb.parse(wb.sheet_names[0]))


# ── The diff (pure) ───────────────────────────────────────────────────────────
def _pos(series: pd.Series, ladder) -> pd.Series:
    """Ladder position (0 = best); NaN for anything not on the ladder (N/A, blank, unknown)."""
    return series.map({v: i for i, v in enumerate(ladder)}).astype(float)


def _ladder_moves(both: pd.DataFrame, col: str, ladder, keep):
    """(improved, worsened, unverifiable) for a labelled ladder column. `steps` = rungs climbed
    (positive = improved). A change touching a label off the ladder is unverifiable."""
    now, was = both[col], both[f"{col}_prev"]
    changed = now.notna() & was.notna() & (now.astype(str) != was.astype(str))
    p_now, p_was = _pos(now, ladder), _pos(was, ladder)
    on_ladder = p_now.notna() & p_was.notna()
    steps = (p_was - p_now).where(changed & on_ladder)
    cols = keep + [f"{col}_prev", col]
    up = both.loc[changed & on_ladder & (steps > 0), cols].assign(steps=steps[changed & on_ladder & (steps > 0)].astype(int))
    dn = both.loc[changed & on_ladder & (steps < 0), cols].assign(steps=steps[changed & on_ladder & (steps < 0)].astype(int))
    un = both.loc[changed & ~on_ladder, cols]
    up = up.sort_values(["steps", "rank", "name"], ascending=[False, True, True], kind="mergesort")
    dn = dn.sort_values(["steps", "rank", "name"], ascending=[True, True, True], kind="mergesort")
    un = un.sort_values(["rank", "name"], kind="mergesort")
    return up, dn, un


def compute_movers(prev: pd.DataFrame, cur: pd.DataFrame, days_between=None) -> dict:
    """Diff two scored vintages joined on company_id. Pure; vectorized; never mutates inputs.

    `days_between` — the calendar gap between the two vintage dates, when the caller knows it.
    It makes `fresh` exact ("the result date is after the previous vintage": 0 ≤ age < gap);
    without it the diff falls back to the age-reset rule, which cannot see a result that landed
    between two vintages whose ages happen to be similar — the quarterly case.

    Returns a dict of frames, every one sorted deterministically (tie-broken by name):
      new, dropped                       stocks on one side only (no deltas — there are none)
      wealth_up, wealth_down             ladder moves with `steps` (rungs)
      wealth_unverifiable                a move touching N/A / an unknown label
      sound_up, sound_down               SOUND/MIXED/FLAWED moves
      rank                               every stock with rank on both sides: rank_delta
                                         (positive = climbed), composite_delta, tier_delta
      gate_new, gate_lost, tsunami_new   0→1 / 1→0 flag transitions (NaN = not a move)
      flags                              red_flag_count changes, rises first
      fresh                              a result landed between the vintages
    plus `counts` (ints) and `n_both`.
    """
    for side, f in (("previous", prev), ("current", cur)):
        if JOIN_KEY not in f.columns:
            raise ValueError(f"the {side} vintage has no {JOIN_KEY} column — cannot join")
        d = int(f[JOIN_KEY].duplicated().sum())
        if d:
            raise ValueError(f"the {side} vintage has {d} duplicate {JOIN_KEY} rows — the join would fan out")

    cols = [c for c in _COLS if c in cur.columns and c in prev.columns]
    a = cur[[JOIN_KEY] + cols]
    b = prev[[JOIN_KEY] + cols]
    m = a.merge(b, on=JOIN_KEY, how="outer", suffixes=("", "_prev"), indicator=True, sort=True)

    ident = [c for c in ["name", "sector", "market_category"] if c in cols]
    # `rank` and `name` are ALWAYS present in the working frame — every section sorts on them —
    # materialized as NaN / the key when a vintage lacks them, so a thin frame degrades to an
    # unsorted-by-rank view instead of a KeyError.
    if "name" not in cols:
        m["name"] = m[JOIN_KEY].astype(str)
        ident.append("name")
    if "rank" not in cols:
        m["rank"] = np.nan
    keep = [JOIN_KEY] + ident + ["rank"] + (["composite_score"] if "composite_score" in cols else [])

    both = m[m["_merge"] == "both"].drop(columns="_merge")
    new = m.loc[m["_merge"] == "left_only", [JOIN_KEY] + cols].sort_values(["rank", "name"] if "rank" in cols else ["name"], kind="mergesort")
    dropped = (m.loc[m["_merge"] == "right_only", [JOIN_KEY] + [f"{c}_prev" for c in cols]]
                .rename(columns={f"{c}_prev": c for c in cols}))
    dropped = dropped.sort_values(["rank", "name"] if "rank" in cols else ["name"], kind="mergesort")

    out = {"new": new.reset_index(drop=True), "dropped": dropped.reset_index(drop=True),
           "n_both": int(len(both))}
    empty = both.iloc[0:0]

    if "wealth_tier" in cols:
        up, dn, un = _ladder_moves(both, "wealth_tier", WEALTH_LADDER, keep)
    else:
        up = dn = un = empty
    out["wealth_up"], out["wealth_down"], out["wealth_unverifiable"] = up, dn, un

    if "verdict_direction" in cols:
        su, sd, _ = _ladder_moves(both, "verdict_direction", SOUND_LADDER, keep)
    else:
        su = sd = empty
    out["sound_up"], out["sound_down"] = su, sd

    if "rank" in cols:
        r = both[keep + ["rank_prev"] + [c for c in ["composite_score_prev", "conviction_tier",
                                                     "conviction_tier_prev"] if c in both.columns]].copy()
        r["rank_delta"] = r["rank_prev"] - r["rank"]                 # positive = climbed
        if "composite_score" in cols:
            r["composite_delta"] = r["composite_score"] - r["composite_score_prev"]
        if "conviction_tier" in cols:
            r["tier_delta"] = r["conviction_tier_prev"] - r["conviction_tier"]   # positive = better tier
        r = r[r["rank_delta"].notna()]
        # Climbers first, biggest climb on top; the renderer reverses the negative tail so the
        # biggest FALL leads its own section. One sort, one order, tie-broken by name.
        r = r.sort_values(["rank_delta", "name"], ascending=[False, True], kind="mergesort")
        out["rank"] = r.reset_index(drop=True)
    else:
        out["rank"] = empty

    def _flag_move(col, frm, to):
        if col not in cols:
            return empty
        mask = both[col].eq(to) & both[f"{col}_prev"].eq(frm)
        return both.loc[mask, keep].sort_values(["rank", "name"], kind="mergesort").reset_index(drop=True)

    out["gate_new"] = _flag_move("gate_pass", 0, 1)
    out["gate_lost"] = _flag_move("gate_pass", 1, 0)
    out["tsunami_new"] = _flag_move("tsunami_signal", 0, 1)

    if "red_flag_count" in cols:
        fl = both[keep + ["red_flag_count_prev", "red_flag_count"]].copy()
        fl["flag_delta"] = fl["red_flag_count"] - fl["red_flag_count_prev"]
        fl = fl[fl["flag_delta"].notna() & (fl["flag_delta"] != 0)]
        fl = fl.sort_values(["flag_delta", "rank", "name"], ascending=[False, True, True], kind="mergesort")
        out["flags"] = fl.reset_index(drop=True)
    else:
        out["flags"] = empty

    if "result_age_days" in cols:
        age, age_prev = both["result_age_days"], both["result_age_days_prev"]
        # A result landed iff the age RESET — smaller now than it was — OR the previous side was a
        # SCHEDULED result (negative age) and now it is declared (non-negative). The second clause
        # was missing at first: the June-29 vintage carried a scheduled Q1 date for most of the
        # universe, so `cur < prev` alone counted 193 of ~2,000 fresh results. A negative age NOW
        # is still never fresh — nothing has landed yet.
        if days_between is not None:
            # EXACT, given the gap: the result date (today − age) is after the previous vintage
            # iff age < gap. Quarterly vintages both sit ~20-40 days after a deadline, so the
            # age-reset rule below cannot see most of them — live it counted 877 of ~2,000.
            fresh_mask = age.notna() & (age >= 0) & (age < float(days_between))
        else:
            fresh_mask = (age.notna() & age_prev.notna() & (age >= 0)
                          & ((age_prev < 0) | (age < age_prev)))
        fr = both.loc[fresh_mask, keep + ["result_age_days"]]
        out["fresh"] = fr.sort_values(["result_age_days", "rank", "name"], kind="mergesort").reset_index(drop=True)
    else:
        out["fresh"] = empty

    # CHURN — the headline. Share of COMPARABLE stocks (column present on both sides) whose label
    # or count changed. Measured live on the first quarter-over-quarter run: 47% of wealth tiers,
    # 80% of flag counts. A fact about the engine's label stability that nothing else can see.
    churn = {}
    for col in ("wealth_tier", "verdict_direction", "red_flag_count", "gate_pass"):
        if col not in cols:
            continue
        now, was = both[col], both[f"{col}_prev"]
        comparable = now.notna() & was.notna()
        n = int(comparable.sum())
        changed = comparable & ((now.astype(str) != was.astype(str))
                                if col in ("wealth_tier", "verdict_direction") else (now != was))
        churn[col] = float(changed.sum() / n) if n else float("nan")
    out["churn"] = churn

    # IDENTITY for every both-side stock — what `material` joins its reasons onto.
    ident = both[[JOIN_KEY] + [c for c in ("name", "sector", "wealth_tier", "rank") if c in both.columns]].copy()
    ident["rank_delta"] = (both["rank_prev"] - both["rank"]) if "rank_prev" in both.columns else np.nan
    out["ident"] = ident.reset_index(drop=True)

    out["counts"] = {k: int(len(v)) for k, v in out.items() if isinstance(v, pd.DataFrame)}
    return out


_REASON_ORDER = {"→": 0, "BUY★": 1, "gate": 2, "🌊": 3, "rank": 4, "flags": 5}   # by first token


def _signed(s: pd.Series) -> pd.Series:
    """+220 / −680 as text, vectorized (a true minus sign, so it cannot be misread as a hyphen)."""
    v = s.astype(int)
    return pd.Series(np.where(v > 0, "+" + v.astype(str), "−" + (-v).astype(str)), index=s.index)


GOOD_REASONS = ("→ BUY★", "gate ✓", "🌊 new", "rank +", "flags −")   # a move that improves the stock
REASON_TOKENS = ("→ BUY★", "BUY★ →", "gate ✓", "gate ✗", "🌊 new", "rank", "flags")


def flag_threshold(res: dict, floor: int = 3, pct: float = 0.90, min_n: int = 50) -> int:
    """How big a red-flag change has to be to count as MATERIAL this quarter: the top decile of
    non-zero |Δ flags|, never below `floor`.

    MEASURED (June-29 → Sep-03): the median |Δ flags| was 2 and 80% of stocks changed, so a fixed
    ≥ 3 admitted 553 stocks — the bulk, not the tail. The 90th percentile was 5 (181 stocks). A
    fixed number is wrong in both directions: too loose in a noisy quarter, and in a quiet one a
    percentile alone would crown a Δ of 1 "material" — hence the floor. Below `min_n` changed
    stocks a percentile is noise, so the floor stands alone. Pure; the render states the value."""
    fl = res.get("flags")
    if fl is None or fl.empty or "flag_delta" not in fl.columns:
        return int(floor)
    a = fl["flag_delta"].abs()
    a = a[a > 0]
    if len(a) < min_n:
        return int(floor)
    return int(max(floor, int(np.ceil(a.quantile(pct)))))


def _tag_reasons(res: dict, top_rank: int, min_flags: int) -> pd.DataFrame:
    """One row per (stock, reason): company_id · why · tok · good. THE single place the material
    reasons are defined, so material() and reason_counts() can never disagree about what counts.
    `tok` is the filter token: the whole reason for label moves ("→ BUY★", "gate ✓" …), the first
    word for the numeric ones ("rank", "flags") so a chip covers every magnitude."""
    def tag(frame, why):
        return frame[[JOIN_KEY]].assign(why=why)

    up, dn, r, fl = res["wealth_up"], res["wealth_down"], res["rank"], res["flags"]
    parts = [tag(res["gate_new"], "gate ✓"), tag(res["gate_lost"], "gate ✗"), tag(res["tsunami_new"], "🌊 new")]
    if not up.empty:
        parts.append(tag(up[up["wealth_tier"] == WEALTH_LADDER[0]], "→ BUY★"))
    if not dn.empty:
        parts.append(tag(dn[dn["wealth_tier_prev"] == WEALTH_LADDER[0]], "BUY★ →"))
    if not r.empty and top_rank > 0:
        top = r.reindex(r["rank_delta"].abs().sort_values(ascending=False, kind="mergesort").index).head(top_rank)
        parts.append(top[[JOIN_KEY]].assign(why="rank " + _signed(top["rank_delta"])))
    if not fl.empty:
        big = fl[fl["flag_delta"].abs() >= flag_threshold(res, floor=min_flags)]
        parts.append(big[[JOIN_KEY]].assign(why="flags " + _signed(big["flag_delta"])))
    tagged = pd.concat(parts, ignore_index=True)
    if tagged.empty:
        return pd.DataFrame(columns=[JOIN_KEY, "why", "tok", "good"])
    first = tagged["why"].str.split(" ").str[0]
    tagged["tok"] = np.where(first.isin(["rank", "flags"]), first, tagged["why"])
    tagged["good"] = tagged["why"].str.startswith(GOOD_REASONS)
    return tagged


def reason_counts(res: dict, top_rank: int = 25, min_flags: int = 3) -> dict:
    """{reason token: number of distinct stocks carrying it} — the live counts on the reason
    chips, from the SAME tagging material() uses. Canonical token order, present tokens only."""
    t = _tag_reasons(res, top_rank, min_flags)
    if t.empty:
        return {}
    n = t.groupby("tok")[JOIN_KEY].nunique()
    return {k: int(n[k]) for k in REASON_TOKENS if k in n.index}


def material(res: dict, top_rank: int = 25, min_flags: int = 3, cap=40, reasons=None) -> pd.DataFrame:
    """'What matters' — one row per stock with a MATERIAL move, reasons joined in a fixed order:
    into / out of BUY★ (ladder moves only — an N/A transition is unverifiable, not a verdict),
    crossed the gate either way, a new Tsunami setup, the `top_rank` biggest |Δ rank|, and
    |Δ flags| ≥ `min_flags`. Sorted by number of reasons (desc), then rank; capped at `cap`
    (None = everything, for the download).

    `reasons` (tokens from REASON_TOKENS) keeps the stocks carrying ANY selected reason — applied
    to the FULL set BEFORE the cap, because filtering the visible 40 would silently miss every
    match that ranked 41st or lower. A kept stock shows ALL its reasons: the filter chooses the
    stocks, not the words. `direction` is ↑ when every reason improves, ↓ when every reason
    deteriorates, ↕ when mixed. Pure."""
    cols_out = [JOIN_KEY, "direction", "name", "sector", "why", "wealth_tier", "rank", "rank_delta"]
    tagged = _tag_reasons(res, top_rank, min_flags)
    if tagged.empty:
        return pd.DataFrame(columns=cols_out)
    if reasons:
        keep_ids = tagged.loc[tagged["tok"].isin(list(reasons)), JOIN_KEY].unique()
        tagged = tagged[tagged[JOIN_KEY].isin(keep_ids)]
        if tagged.empty:
            return pd.DataFrame(columns=cols_out)
    tagged = tagged.assign(_o=tagged["why"].str.split(" ").str[0].map(_REASON_ORDER))
    tagged = tagged.sort_values([JOIN_KEY, "_o"], kind="mergesort")
    agg = (tagged.groupby(JOIN_KEY, sort=True)
           .agg(why=("why", " · ".join), n=("why", "size"), good=("good", "sum"))
           .reset_index())
    agg["direction"] = np.select([agg["good"] == agg["n"], agg["good"] == 0], ["↑", "↓"], "↕")
    ident = res["ident"]
    for c in ("sector", "wealth_tier"):
        if c not in ident.columns:
            ident = ident.assign(**{c: np.nan})
    m = agg.merge(ident, on=JOIN_KEY, how="left")
    m = m.sort_values(["n", "rank", "name"], ascending=[False, True, True], kind="mergesort")
    m = m[cols_out].reset_index(drop=True)
    return m if cap is None else m.head(cap)


def restrict(res: dict, ids) -> dict:
    """Apply the lens row to a computed result: keep only stocks whose CURRENT-side company_id is
    in `ids`. Computed AFTER the diff, never before — filtering the current frame first would
    turn every filtered-out stock into a fake 'dropped' row. `dropped` has no current side to
    filter on, so it is left whole (its note says so). Counts are recomputed. Pure."""
    keep = set(map(str, ids))
    out = {}
    for k, v in res.items():
        if isinstance(v, pd.DataFrame) and k != "dropped":
            out[k] = v[v[JOIN_KEY].astype(str).isin(keep)].reset_index(drop=True)
        else:
            out[k] = v
    # SCOPE HONESTY (found in the browser, 2026-09-04). `ident` holds one row per both-side stock
    # and IS filtered above, so the header's stock count must be re-derived from it — it read
    # "2,101 stocks" over tables showing 69 Steel names. CHURN cannot be recomputed here (it needs
    # the before/after label pairs the merge held, which no section frame carries), so it is
    # MARKED instead: the number stays true of the whole universe and the header says so. A
    # universe rate printed beside a filtered table, unlabelled, is the scope-mislabel class.
    out["n_both"] = int(len(out["ident"])) if isinstance(out.get("ident"), pd.DataFrame) else res["n_both"]
    out["restricted"] = True
    out["counts"] = {k: int(len(v)) for k, v in out.items() if isinstance(v, pd.DataFrame)}
    return out


# ── Rendering (stateless) ─────────────────────────────────────────────────────
# None = HIDDEN. Four kinds of column earn their width and nothing else does: identity (Stock,
# Sector), the thing that moved, the delta, and where the stock now stands. Hidden deliberately:
# the join key; market_category (constant noise repeated in every table — market cap is one click
# away on the tear-sheet); gate_pass / tsunami_signal (raw 0/1, and the sections they belong to
# are defined BY those transitions); and every NUMERIC `_prev` twin, because the delta beside the
# current value already carries it. The LABEL `_prev` twins stay — "was AVOID, now BUY★" is the
# whole content of a ladder move.
_HDR = {
    JOIN_KEY: None, "market_category": None, "gate_pass": None, "tsunami_signal": None,
    "rank_prev": None, "composite_score_prev": None, "conviction_tier_prev": None,
    "red_flag_count_prev": None,
    "name": "Stock", "sector": "Sector", "why": "Why", "direction": "Dir",
    "rank": "Rank", "rank_delta": "Δ Rank",
    "composite_score": "Score", "composite_delta": "Δ Score",
    "conviction_tier": "Tier", "tier_delta": "Δ Tier",
    "wealth_tier": "Wealth tier", "wealth_tier_prev": "was", "steps": "Rungs",
    "verdict_direction": "Soundness", "verdict_direction_prev": "was",
    "red_flag_count": "🚩 Flags", "flag_delta": "Δ Flags",
    "result_age_days": "Days since result",
}


def _table(df: pd.DataFrame, limit: int = 40, select: bool = False, cap_px: int = 420):
    """One section table. Column names are never shown raw: every visible column has a header in
    _HDR, and the hidden ones are declared there as None. Capped at `limit` rows — the section
    header states "showing 40 of 1,992" whenever that bites, because a count the table cannot
    deliver is the same quiet mislead as a mislabelled unit.

    `select=True` turns on single-row selection and RETURNS the chosen stock name (or None). The
    module stays stateless: it reads the selection Streamlit hands back and returns it; app.py
    owns the session_state write that stages the tear-sheet."""
    if df.empty:
        st.markdown(f"<div style='font-size:0.74rem;color:{COLORS['text_muted']};"
                    f"padding:2px 0 10px 2px;'>none</div>", unsafe_allow_html=True)
        return None
    show = [c for c in df.columns if _HDR.get(c, c) is not None]
    cfg = {}
    for c in show:
        label = _HDR.get(c, c.replace("_", " ").title())
        if c in ("composite_score", "composite_delta"):
            cfg[c] = st.column_config.NumberColumn(label, format="%+.1f" if "delta" in c else "%.1f", width="small")
        elif c in ("rank", "rank_delta", "steps", "tier_delta", "flag_delta",
                   "red_flag_count", "result_age_days", "conviction_tier"):
            cfg[c] = st.column_config.NumberColumn(label, format="%+d" if ("delta" in c or c == "steps") else "%d", width="small")
        else:
            cfg[c] = st.column_config.TextColumn(
                label, width="large" if c == "why" else ("medium" if c == "name" else "small"))
    head = df[show].head(limit).reset_index(drop=True)
    kw = dict(column_config=cfg, width="stretch", hide_index=True,
              height=min(cap_px, 60 + len(head) * 35))
    if not select:
        st.dataframe(head, **kw)
        return None
    sel = st.dataframe(head, on_select="rerun", selection_mode="single-row", **kw)
    rows = sel.selection.rows if sel is not None and hasattr(sel, "selection") else []
    return str(df.iloc[rows[0]]["name"]) if rows and "name" in df.columns else None


def _section(title: str, n: int, note: str = "", shown: int = None) -> None:
    """Section header: title, true count, and — when the table is truncated — exactly how much of
    it you are looking at."""
    trunc = ("" if shown is None or shown >= n else
             f"<span style='font-size:0.68rem;color:{COLORS['gold']};'>showing {shown} of {n:,}</span>")
    st.markdown(
        f"<div style='display:flex;align-items:baseline;gap:8px;margin:14px 0 4px 0;flex-wrap:wrap;'>"
        f"<span style='font-size:0.9rem;font-weight:800;color:{COLORS['text_primary']};'>{title}</span>"
        f"<span style='font-size:0.72rem;font-weight:700;color:{COLORS['purple']};'>{n:,}</span>"
        + trunc
        + (f"<span style='font-size:0.68rem;color:{COLORS['text_muted']};'>{note}</span>" if note else "")
        + "</div>", unsafe_allow_html=True)


def _block(title: str, df: pd.DataFrame, note: str = "", limit: int = 40) -> None:
    """Header + table, with the truncation stated. The pair is always written together, so it
    cannot drift into a count the table does not honour."""
    _section(title, len(df), note, shown=min(len(df), limit))
    _table(df, limit=limit)


def _churn_bars(churn: dict, restricted: bool = False) -> str:
    """Churn as four small BARS, not a sentence: what share of comparable stocks changed, per
    label. It is the page's most important number — 47% of wealth tiers flipping in one quarter
    is a fact about the ENGINE's label stability — and a bar reads in a glance where
    "47% · 7% · 80% · 17%" needs a sentence. When the page is lens-restricted the rate is still
    the whole universe's (it cannot be recomputed from section frames) and says so."""
    if not churn:
        return ""
    names = {"wealth_tier": "wealth tier", "verdict_direction": "soundness",
             "red_flag_count": "red-flag count", "gate_pass": "gate"}
    cells = []
    for k, v in churn.items():
        if k not in names or v != v:
            continue
        pct = max(0.0, min(100.0, float(v) * 100.0))
        cells.append(
            f"<div style='min-width:150px;'>"
            f"<div style='display:flex;justify-content:space-between;font-size:0.66rem;color:{COLORS['text_muted']};'>"
            f"<span>{names[k]}</span><b style='color:{COLORS['text_primary']};'>{v:.0%}</b></div>"
            f"<div style='height:6px;border-radius:3px;background:{COLORS['bg_tertiary']};overflow:hidden;'>"
            f"<div style='width:{pct:.0f}%;height:6px;background:{COLORS['purple']};'></div></div></div>")
    scope = (f" <span style='font-weight:400;color:{COLORS['gold']};'>(whole universe — the lens filter "
             f"below does not narrow this)</span>" if restricted else "")
    return (f"<div style='margin-top:8px;'><div style='font-size:0.7rem;font-weight:700;"
            f"color:{COLORS['text_primary']};margin-bottom:4px;'>Churn — share of stocks whose label changed"
            f"{scope}</div><div style='display:flex;gap:16px;flex-wrap:wrap;'>" + "".join(cells) + "</div></div>")


def render_movers(res: dict, meta: dict):
    """The Movers page below the picker. `meta` carries what only the caller knows:
    prev_vintage, cur_vintage (ISO), prev_label, cur_label (FY quarter), engine, prev_regime,
    cur_regime, mode. ("profile" left this list 2026-09-20 with the selector — the QGLP screen is
    a constant, so printing it named a control the reader cannot see.)

    RETURNS the stock name clicked in ⭐ What matters, or None — the module never writes
    session_state (app.py owns it and stages the tear-sheet), which is what keeps this file
    stateless while still closing the loop from a mover to its full analysis."""
    c = res["counts"]
    same_engine = meta.get("prev_engine", meta.get("engine")) == meta.get("engine")
    regime_changed = meta.get("prev_regime") != meta.get("cur_regime")
    # THE EXPLAINER IS A TOOLTIP. It was two lines of permanent prose on every render; measured,
    # ~280 words sat above the first data row. The ⓘ carries it for whoever wants it.
    tip = ("Both sides were scored by the same engine, moments apart, so every move below is the "
           "company changing, never PRISM changing."
           + (" The regime changed between the vintages: the regime-adaptive weights shift every "
              "composite with it, so read rank and label moves first." if regime_changed else ""))
    elapsed = meta.get("elapsed")
    muted, strong = COLORS["text_muted"], COLORS["text_primary"]

    def cell(label, value, tone=None):
        return (f"<span style='white-space:nowrap;'><span style='font-size:0.62rem;color:{muted};"
                f"text-transform:uppercase;letter-spacing:0.5px;'>{label}</span> "
                f"<b style='color:{tone or strong};'>{value}</b></span>")

    st.markdown(
        f"<div style='background:{COLORS['bg_secondary']};border:1px solid {COLORS['border']};"
        f"border-radius:10px;padding:8px 14px;margin:6px 0 8px 0;font-size:0.76rem;line-height:1.55;'>"
        f"<div style='display:flex;flex-wrap:wrap;gap:4px 20px;align-items:baseline;'>"
        f"<span style='font-size:1.05rem;font-weight:800;color:{strong};'>{meta['prev_label']} → {meta['cur_label']}</span>"
        + cell("vintages", f"{meta['prev_vintage']} → {meta['cur_vintage']}")
        + cell("comparable", f"{res['n_both']:,}") + cell("new", c["new"]) + cell("dropped", c["dropped"])
        + cell("engine", f"{meta.get('engine', 'unknown')}{'' if same_engine else ' ⚠ differs'}")
        + cell("mode", f"{meta.get('mode', '')}")
        + cell("regime", f"{meta.get('prev_regime', '?')} → {meta.get('cur_regime', '?')}"
                         + (" ⚠ regime changed" if regime_changed else ""),
               COLORS["gold"] if regime_changed else None)
        + (cell("compared in", f"{elapsed:.0f}s") if elapsed is not None else "")
        + f"<span title='{tip}' style='cursor:help;color:{COLORS['blue']};font-weight:700;white-space:nowrap;'>ⓘ same engine</span>"
        f"</div>"
        + _churn_bars(res.get("churn", {}), bool(res.get("restricted")))
        + (f"<div style='color:{COLORS['gold']};margin-top:6px;font-size:0.7rem;'>⚠️ <b>Cash-flow caveat.</b> "
           f"Cash-flow statements are filed half-yearly, so cash-driven signals refresh only in the June and "
           f"December vintages — a change in them here is rare and worth a second look.</div>"
           if any(q in meta.get("cur_label", "") for q in ("Q1", "Q3")) else "")
        + "</div>", unsafe_allow_html=True)

    # THE PAGE IS ONE TABLE PLUS ITS EVIDENCE. Measured on the first live run, rendering all
    # sixteen sections flat produced a 7,820px page — 10.7 screens of scrolling, thirteen tables
    # pinned at their height cap, and ⭐ What matters (the entire point of the materiality pass)
    # sitting as one of sixteen equals. The star stays open and selectable; the fifteen evidence
    # sections go behind five sub-tabs, which is the st.tabs language Market Pulse already speaks
    # and which — unlike st.expander on Streamlit 1.54 — holds its selection across a rerun.
    # UNCAPPED here: the table shows the top 40 and states it; the 📥 below carries the rest.
    # The reason chips (app.py, meta["reasons"]) filter INSIDE material, before the cap.
    mat = material(res, reasons=meta.get("reasons"), cap=None)
    _section("⭐ What matters", len(mat),
             f"into / out of BUY★ · crossed the gate · new Tsunami · top-25 rank jumps · |Δ flags| ≥ "
             f"{flag_threshold(res)} (this quarter's top decile, floor 3) — one row per stock. "
             f"↑ improving · ↓ deteriorating · ↕ mixed. Click a row to load it into the Tear-Sheet.",
             shown=min(len(mat), 40))
    picked = _table(mat, limit=40, select=True, cap_px=560)
    if not mat.empty:
        # A download is the honest answer to "showing 40 of 63": the other 23 are one click away.
        # st.download_button writes no session_state, so it belongs here beside its table.
        st.download_button(
            f"📥 Download all {len(mat):,} material movers (CSV)",
            data=_to_csv_bytes(mat.drop(columns=[JOIN_KEY])),
            file_name=f"prism_movers_{meta.get('prev_vintage', 'prev')}_to_{meta.get('cur_vintage', 'cur')}.csv",
            mime="text/csv", key="mp_mv_dl",
            help="Every stock the chips currently select, not just the 40 on screen. Excel-safe UTF-8.",
        )
    if picked:
        st.markdown(
            f"<div style='padding:9px 14px;margin:6px 0 2px 0;background:rgba(139,92,246,0.07);"
            f"border:1px solid rgba(139,92,246,0.3);border-radius:8px;font-size:0.8rem;'>"
            f"🔬 <strong style='color:{COLORS['text_primary']};'>{picked}</strong> set — "
            f"<strong style='color:{COLORS['blue']};'>click The Tear-Sheet tab</strong> for full analysis."
            f"</div>", unsafe_allow_html=True)

    r, fl = res["rank"], res["flags"]
    r_up = r[r["rank_delta"] > 0] if not r.empty else r
    r_dn = r[r["rank_delta"] < 0].iloc[::-1] if not r.empty else r
    fl_up = fl[fl["flag_delta"] > 0] if not fl.empty else fl
    fl_dn = fl[fl["flag_delta"] < 0].iloc[::-1] if not fl.empty else fl
    _ev = st.tabs([
        f"💹 Wealth · {c['wealth_up'] + c['wealth_down'] + c['wealth_unverifiable']:,}",
        f"🧭 Soundness · {c['sound_up'] + c['sound_down']:,}",
        f"📈 Rank · {len(r_up) + len(r_dn):,}",
        f"🚦 Gates & flags · {c['gate_new'] + c['gate_lost'] + c['tsunami_new'] + c['flags']:,}",
        f"🆕 Universe · {c['fresh'] + c['new'] + c['dropped']:,}",
    ])
    with _ev[0]:
        _block("Upgrades", res["wealth_up"], "rungs climbed on BUY★ › BUY › WATCH★ › WATCH › AVOID")
        _block("Downgrades", res["wealth_down"])
        _block("Unverifiable", res["wealth_unverifiable"],
               "moved to or from N/A: an input went missing or came back — not a verdict")
    with _ev[1]:
        _block("Improved", res["sound_up"], "FLAWED › MIXED › SOUND")
        _block("Worsened", res["sound_down"])
    with _ev[2]:
        _block("Climbers", r_up, "Δ Rank positive = climbed")
        _block("Fallers", r_dn)
    with _ev[3]:
        _block("New gate passers", res["gate_new"])
        _block("Lost the gate", res["gate_lost"])
        _block("New Tsunami setups", res["tsunami_new"])
        _block("Red flags — rises", fl_up, "forensics deteriorating")
        _block("Red flags — falls", fl_dn)
    with _ev[4]:
        _block("Fresh results", res["fresh"], "a result landed between the two vintages")
        _block("New to the universe", res["new"], "no deltas — there is no previous side")
        _block("Dropped from the universe", res["dropped"])
    return picked
