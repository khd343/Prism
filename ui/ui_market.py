"""
🧭 The market box in the sidebar (2026-10-08) — display only.

It shows the benchmark indices' Weinstein stages (the same rule PRISM stages every stock with:
data_engine.weinstein_stage) and, under them, the breadth regime the engine detects — labelled as
shown, not applied, while config.REGIME_DRIVES_SCORING is off. Nothing here reaches a score.

Why the indices lead: the detected regime votes on the share of stocks BEATING the Nifty 500, which
sits near 50% by construction; on 2026-10-08 it read SIDEWAYS while the Nifty 500 itself was in a
Stage-4 decline and the small-cap indices in Stage 2. The index's own trend is the market; the
breadth vote is context.
"""
import html

import pandas as pd

from config import BENCHMARK_INDICES, COLORS, REGIME_DRIVES_SCORING

# Short names for a ~270px sidebar; the full index name is the row's tooltip. Keyed by the vendor id
# (config.BENCHMARK_INDICES is the one list; a test keeps the two in step).
SHORT_NAMES = {
    "NSE:CNX500": "Nifty 500",
    "NSE:SMALLCA250": "Smallcap 250",
    "NSE:NFMICRO250": "Microcap 250",
    "NSE:NIF500MO50": "Momentum 50",
    "NSE:NI200MOM30": "Momentum 30",
    "NSE:NQUALITY30": "Quality 30",
    "NSE:NS250MQ100": "Smallcap Mom-Quality",
}
MARKET_ID = "NSE:CNX500"

_STAGE_TIP = ("Weinstein stage — the same rule PRISM stages every stock with: the index against its "
              "30-week average, and that average against its 200-day. 📈 Stage 2 advancing · 🔄 Stage 1 "
              "basing · ⚠️ Stage 3 topping · 📉 Stage 4 declining. Display only: it changes no score.")


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


def market_box_html(bench: pd.DataFrame, regime: str) -> str:
    """The sidebar box. With the benchmark table: each index and its stage, the Nifty 500 first and in
    bold, the box's edge coloured by the Nifty 500's stage; the breadth regime under them. Without it
    (data from before the indices were added): the regime alone, labelled the same honest way."""
    c = COLORS
    esc = html.escape
    rows = bench if bench is not None else pd.DataFrame()
    if rows.empty or "stage" not in rows.columns:
        colour = c["green"] if regime == "BULL" else c["red"] if regime == "BEAR" else c["gold"]
        return (f'<div style="background:{c["bg_tertiary"]};border-left:4px solid {colour};padding:8px 12px;'
                f'margin-bottom:15px;border-radius:4px;">'
                f'<div style="font-size:0.75rem;color:{c["text_muted"]};text-transform:uppercase;'
                f'letter-spacing:1px;">Detected Regime</div>'
                f'<div style="font-size:1.1rem;font-weight:800;color:{colour};">{esc(regime)} MARKET</div>'
                + _regime_line(regime, c).replace("Breadth regime: ", "Breadth vote: ", 1)
                + f'<div style="font-size:0.62rem;color:{c["text_muted"]};margin-top:3px;">Index stages '
                  f'appear once the benchmark indices are in the sheet.</div></div>')
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
    return (f'<div style="background:{c["bg_tertiary"]};border-left:4px solid {edge};padding:8px 12px;'
            f'margin-bottom:15px;border-radius:4px;">'
            f'<div style="font-size:0.7rem;color:{c["text_muted"]};text-transform:uppercase;letter-spacing:1px;'
            f'margin-bottom:5px;" title="{esc(_STAGE_TIP)}">Market · Weinstein stage</div>'
            f'{body}{_regime_line(regime, c)}</div>')
