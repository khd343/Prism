"""
🧭 The market box in the sidebar (2026-10-08) — display only.

It shows the benchmark indices' Weinstein stages (the same rule PRISM stages every stock with:
data_engine.weinstein_stage) and, under them, the breadth regime the engine detects — labelled as
shown, not applied, while config.REGIME_DRIVES_SCORING is off. Nothing here reaches a score.

Why the indices lead: the detected regime votes on the share of stocks BEATING the Nifty 500, which
sits near 50% by construction; on 2026-10-08 it read SIDEWAYS while the Nifty 500 itself was in a
Stage-4 decline and the small-cap indices in Stage 2. The index's own trend is the market; the
breadth vote is context.

NSE's own numbers (2026-11-04): each row reads NSE's daily index file (core/index_closes.py), and a
second column, vs 200W, gives the close against the simple average of its last 200 weekly closes —
the line Kite's shadow book gates on for MidSmall 400 (addendum 37, rule 8). The vendor's history for
Nifty500 Shariah was measured wrong on 3 Nov 2026; NSE's is the index's own. When NSE's closes cannot
be read, or stop more than config.INDEX_CLOSES_STALE_DAYS before the sheet's date, the box shows the
sheet's stages exactly as before and says why the column is missing; an index whose own closes stop
that long before (NSE renamed or dropped it) keeps the sheet's stage, and says so.
"""
import html
from datetime import datetime

import numpy as np
import pandas as pd

from config import BENCHMARK_INDICES, COLORS, INDEX_CLOSES_STALE_DAYS, INDEX_WEEKS_LONG, REGIME_DRIVES_SCORING
from core.sheet_meta import IST

# Short names for a ~270px sidebar; the full index name is the row's tooltip. Keyed by the vendor id
# (config.BENCHMARK_INDICES is the one list; a test keeps the two in step).
SHORT_NAMES = {
    "NSE:CNX500": "Nifty 500",
    "NSE:NIFTY500SH": "Shariah 500",
    "NSE:SMALLCA250": "Smallcap 250",
    "NSE:NFMICRO250": "Microcap 250",
    "NSE:MIDSMAL400": "MidSmall 400",
    "NSE:NIF500MO50": "Momentum 50",
    "NSE:NI200MOM30": "Momentum 30",
    "NSE:NQUALITY30": "Quality 30",
    "NSE:NS250MQ100": "Smallcap Mom-Quality",
    "NSE:NMS400MQ10": "MidSmall Mom-Quality",
}
MARKET_ID = "NSE:CNX500"

_STAGE_TIP = ("Weinstein stage — the same rule PRISM stages every stock with: the index against its "
              "30-week average, and that average against its 200-day. 📈 Stage 2 advancing · 🔄 Stage 1 "
              "basing · ⚠️ Stage 3 topping · 📉 Stage 4 declining. Display only: it changes no score.")
_VS_TIP = (f"vs 200W: how far the index's NSE close sits above (green) or below (red) the simple average of its "
           f"last {INDEX_WEEKS_LONG} weekly closes — a week's close is its last session's, and the current week "
           "counts with its latest close. The line Kite's shadow book gates on for MidSmall 400. An index with "
           f"fewer than {INDEX_WEEKS_LONG} weeks of NSE closes shows —. Display only: it changes no score.")
_SRC_TIP = ("Every number in this box is NSE's own, from its daily index file, which PRISM's Index closes job "
            "keeps each night: each index's newest close on or before the sheet's date (the newest kept when the "
            "sheet has no date). The date here is that close's — in the evening often a session behind the sheet.")
_NEEDED = {"company_id", "close", "stage", "vs_200w", "sma_200w", "weeks", "last_date"}


def _stage_colour(stage: str, c: dict) -> str:
    if "Stage 2" in stage:
        return c["green"]
    if "Stage 4" in stage:
        return c["red"]
    if "Stage 3" in stage:
        return c["gold"]
    return c["text_muted"]


def _regime_line(regime: str, c: dict) -> str:
    applied = "applied to scores" if REGIME_DRIVES_SCORING else "shown, not applied to scores"
    tip = ("The share of stocks beating the Nifty 500 (50 days, 26 weeks) and above their 200-day "
           "average, 2 of 3 votes. It sits near 50% by construction, so it reads SIDEWAYS most of the "
           "time; until December's review it does not change any score.")
    return (f'<div style="font-size:0.66rem;color:{c["text_muted"]};margin-top:7px;line-height:1.35;" '
            f'title="{html.escape(tip)}">Breadth regime: <b style="color:{c["text_secondary"]};">'
            f'{html.escape(regime.title())}</b> · {applied}</div>')


def _day(d) -> str:
    return f"{d.day} {d:%b %Y}"


def _usable(nse, ref):
    """(NSE's view, None) when it can lead the box, else (None, why not)."""
    if nse is None or nse.empty or not _NEEDED <= set(nse.columns):
        return None, "NSE's index closes could not be read"
    newest = max(nse["last_date"])
    if (ref - newest).days > INDEX_CLOSES_STALE_DAYS:
        return None, f"NSE's index closes stop at {_day(newest)}"
    return nse, None


def _vs(v, c: dict) -> tuple:
    """vs 200W as printed, and its colour — classified on the figure as printed, so a 0.0 is never red."""
    if v is None or not np.isfinite(v):
        return "—", c["text_muted"]
    shown = round(float(v), 1)
    if shown == 0:
        return "0.0%", c["text_muted"]
    return f"{shown:+.1f}%".replace("-", "−"), (c["green"] if shown > 0 else c["red"])


def _frame(edge: str, body: str, regime: str, c: dict) -> str:
    return (f'<div style="background:{c["bg_tertiary"]};border-left:4px solid {edge};padding:8px 12px;'
            f'margin-bottom:15px;border-radius:4px;">'
            f'<div style="font-size:0.7rem;color:{c["text_muted"]};text-transform:uppercase;letter-spacing:1px;'
            f'margin-bottom:5px;" title="{html.escape(_STAGE_TIP)}">Market · Weinstein stage</div>'
            f'{body}{_regime_line(regime, c)}</div>')


def _sheet_box(rows: pd.DataFrame, regime: str, why: str, c: dict) -> str:
    """The sheet's stages, exactly as the box showed them before NSE's numbers, and why the column is missing."""
    esc = html.escape
    market = rows[rows["company_id"] == MARKET_ID]
    edge = _stage_colour(str(market["stage"].iloc[0]), c) if len(market) else c["border"]
    body = ""
    for cid, stage in zip(rows["company_id"], rows["stage"]):
        stage = str(stage)
        lead = cid == MARKET_ID
        short = " ".join(stage.split()[:3])
        body += (f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px;'
                 f'padding:{"1px 0 4px 0" if lead else "1px 0"};'
                 f'{"border-bottom:1px solid " + c["border"] + ";margin-bottom:3px;" if lead else ""}" '
                 f'title="{esc(BENCHMARK_INDICES.get(cid, cid))}: {esc(stage)}">'
                 f'<span style="font-size:{"0.82rem" if lead else "0.74rem"};'
                 f'font-weight:{"800" if lead else "500"};color:{c["text_primary"] if lead else c["text_secondary"]};">'
                 f'{esc(SHORT_NAMES.get(cid, cid))}</span>'
                 f'<span style="font-size:{"0.8rem" if lead else "0.72rem"};font-weight:700;'
                 f'color:{_stage_colour(stage, c)};white-space:nowrap;">{esc(short)}</span></div>')
    body += (f'<div style="font-size:0.62rem;color:{c["text_muted"]};margin-top:5px;line-height:1.35;" '
             f'title="{esc(_VS_TIP)}">vs 200W unavailable: {esc(why)} — stages from the sheet.</div>')
    return _frame(edge, body, regime, c)


def _nse_box(view: pd.DataFrame, sheet, regime: str, c: dict, ref) -> str:
    """Each index's stage and vs 200W from NSE's numbers, in one grid so the columns line up. An index NSE has no
    close for, or whose closes stop more than INDEX_CLOSES_STALE_DAYS before `ref`, keeps the sheet's stage, says why,
    and shows —; with no sheet row it is left out, so a stale number never sits under NSE's date."""
    esc = html.escape
    n_by = view.set_index("company_id")
    s_by = sheet.drop_duplicates("company_id").set_index("company_id") if sheet is not None else None
    rows = []                                          # (id, stage, vs text, vs colour, tooltip)
    for cid, full in BENCHMARK_INDICES.items():        # the ten declared indices, not the stock matrix
        last = n_by.at[cid, "last_date"] if cid in n_by.index else None
        if last is not None and (ref - last).days <= INDEX_CLOSES_STALE_DAYS:
            r = n_by.loc[cid]
            stage = str(r["stage"])
            txt, col = _vs(r["vs_200w"], c)
            where = f"NSE close {r['close']:,.1f} on {_day(r['last_date'])}"
            if txt == "—":
                line = (f"{int(r['weeks'])} weeks of NSE closes, fewer than the {INDEX_WEEKS_LONG} a 200-week "
                        f"average needs")
            else:
                side = "above" if txt.startswith("+") else "below" if txt.startswith("−") else "at"
                line = f"{txt.lstrip('+−')} {side} its 200-week average ({r['sma_200w']:,.1f})"
            rows.append((cid, stage, txt, col, f"{full}: {stage} · {line} · {where}"))
        elif s_by is not None and cid in s_by.index:
            stage = str(s_by.at[cid, "stage"])
            why = f"NSE's closes for it stop at {_day(last)}" if last is not None else "NSE had no close for it"
            rows.append((cid, stage, "—", c["text_muted"], f"{full}: {stage} · from the sheet: {why}"))
    lead_stage = next((s for cid, s, *_ in rows if cid == MARKET_ID), None)
    edge = _stage_colour(lead_stage, c) if lead_stage else c["border"]
    label = (f'font-size:0.62rem;color:{c["text_muted"]};text-transform:uppercase;letter-spacing:0.6px;'
             f'white-space:nowrap;')
    cells = (f'<span></span><span style="{label}text-align:right;" title="{esc(_VS_TIP)}">vs 200W</span>'
             f'<span style="{label}" title="{esc(_STAGE_TIP)}">Stage</span>')
    for cid, stage, txt, col, tip in rows:
        lead = cid == MARKET_ID
        t = esc(tip)
        cells += (f'<span title="{t}" style="padding:1px 0;min-width:0;overflow:hidden;text-overflow:ellipsis;'
                  f'white-space:nowrap;font-size:{"0.82rem" if lead else "0.74rem"};'
                  f'font-weight:{"800" if lead else "500"};'
                  f'color:{c["text_primary"] if lead else c["text_secondary"]};">{esc(SHORT_NAMES.get(cid, cid))}</span>'
                  f'<span title="{t}" style="padding:1px 0;text-align:right;white-space:nowrap;'
                  f'font-variant-numeric:tabular-nums;font-size:{"0.78rem" if lead else "0.72rem"};font-weight:700;'
                  f'color:{col};">{esc(txt)}</span>'
                  f'<span title="{t}" style="padding:1px 0;white-space:nowrap;font-size:{"0.8rem" if lead else "0.72rem"};'
                  f'font-weight:700;color:{_stage_colour(stage, c)};">{esc(" ".join(stage.split()[:3]))}</span>')
        if lead:
            cells += f'<div style="grid-column:1 / -1;border-bottom:1px solid {c["border"]};margin:2px 0 3px 0;"></div>'
    newest = max(view["last_date"])
    # A 6px gap: at 10px the two longest names (Smallcap / MidSmall Mom-Quality) lacked 1-2px of the 300px sidebar
    # (measured 4 Nov 2026: 114 and 115px needed, 113 given); a narrower box still ellipsises, never wraps.
    body = (f'<div style="display:grid;grid-template-columns:minmax(0,1fr) auto auto;column-gap:6px;'
            f'align-items:baseline;">{cells}</div>'
            f'<div style="font-size:0.62rem;color:{c["text_muted"]};margin-top:5px;" title="{esc(_SRC_TIP)}">'
            f'NSE closes · {esc(_day(newest))}</div>')
    return _frame(edge, body, regime, c)


def market_box_html(bench: pd.DataFrame, regime: str, nse: pd.DataFrame = None, ref_day=None) -> str:
    """The sidebar box. With NSE's numbers (core/index_closes.box_view, no more than INDEX_CLOSES_STALE_DAYS before
    `ref_day`, the sheet's date): each index's stage and its line against the 200-week average, the Nifty 500 first
    and in bold, the date of NSE's closes under them. Without them: the sheet's stages exactly as before, and a line
    saying why the column is missing. With neither (data from before the indices were added): the regime alone,
    labelled the same honest way. The box's edge follows the Nifty 500's stage; the breadth regime closes it."""
    c = COLORS
    esc = html.escape
    sheet = bench if bench is not None and not bench.empty and "stage" in bench.columns else None
    ref = ref_day or datetime.now(IST).date()
    view, why = _usable(nse, ref)
    if view is not None:
        return _nse_box(view, sheet, regime, c, ref)
    if sheet is not None:
        return _sheet_box(sheet, regime, why, c)
    colour = c["green"] if regime == "BULL" else c["red"] if regime == "BEAR" else c["gold"]
    return (f'<div style="background:{c["bg_tertiary"]};border-left:4px solid {colour};padding:8px 12px;'
            f'margin-bottom:15px;border-radius:4px;">'
            f'<div style="font-size:0.75rem;color:{c["text_muted"]};text-transform:uppercase;'
            f'letter-spacing:1px;">Detected Regime</div>'
            f'<div style="font-size:1.1rem;font-weight:800;color:{colour};">{esc(regime)} MARKET</div>'
            + _regime_line(regime, c).replace("Breadth regime: ", "Breadth vote: ", 1)
            + f'<div style="font-size:0.62rem;color:{c["text_muted"]};margin-top:3px;">Index stages '
              f'appear once the benchmark indices are in the sheet.</div></div>')
