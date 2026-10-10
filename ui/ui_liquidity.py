"""PRISM — liquidity: one vocabulary for every surface (2026-10-09).

How much of a stock changes hands a day decides whether you can buy it in your size. The NUMBER is the
engine's (core.data_engine.traded_value_cr: 20-day average volume × price, ₹ crore a day, materialised
as `traded_value_cr`); this module holds the words around it, so the Scans floor, the sidebar filter,
the Deep Scanner column and the Tear-Sheet line say the same thing. Pure, no widgets.

Measured 2026-10-09 on the four saved snapshots: 42% of stocks trade under ₹1 Cr a day, and so do 16
of the top 50 by rank. Liquidity tracks market cap (rank correlation 0.88), yet a Small-cap-and-above
filter keeps 308 illiquid stocks and drops 85 liquid micro caps. Liquid and illiquid stocks did about as
well on median, so this is a buyability check, never a score input (tests/test_liquidity.py).
"""

LIQUIDITY_STEPS = (None, 0.5, 1.0, 2.0, 5.0)    # ₹ crore a day; None = any
HEADER = "₹ Cr/day"                              # the one table header for the number
# Which step to pick depends on the buyer's order, not on the stock (2026-10-07): keep one order under
# ORDER_SHARE of a day's traded value — a common, cautious rule of thumb for an order the market takes
# without moving the price much. Guidance only: it filters nothing and touches no record.
ORDER_SHARE = 0.05
SURVEILLANCE_NOTE = ("Small or BSE-only stock: check the exchange's surveillance status (ASM/GSM) before "
                     "buying — traded value cannot show it.")


def step_label(step) -> str:
    """A floor as words: 'Any', or '₹1 Cr/day'. A label passes through unchanged: Streamlit's test
    runner hands a select_slider's format_func its own labels on a rerun, and a round-tripped display
    string must never crash the sidebar (the 2026-09-21 label round-trip class)."""
    if isinstance(step, str):
        return step
    return "Any" if step is None else f"₹{step:g} Cr/day"


def step_value(stored):
    """Recover a floor from whatever session state holds: a step, or its label (round-tripped). Anything
    else is not a floor PRISM offers and reads as Any — the selectboxes' fallback rule (canonical_pick)."""
    by_label = {step_label(s): s for s in LIQUIDITY_STEPS}
    if isinstance(stored, str):
        return by_label.get(stored)
    return stored if stored in LIQUIDITY_STEPS else None


def order_size_lakh(cr_per_day: float) -> float:
    """The largest order (₹ lakh) that stays under ORDER_SHARE of cr_per_day (₹ crore = 100 lakh)."""
    return round(cr_per_day * ORDER_SHARE * 100, 2)


def order_size_text(cr_per_day: float) -> str:
    """That order as words, in the unit an investor would say: '₹5,000', '₹1.4 lakh', '₹2.5 Cr'."""
    rupees = cr_per_day * 1e7 * ORDER_SHARE
    if rupees >= 1e7:
        return f"₹{round(rupees / 1e7, 1):,g} Cr"
    if rupees >= 1e5:
        return f"₹{round(rupees / 1e5, 1):g} lakh"
    return f"₹{round(rupees, -2):,.0f}"


def per_day_text(cr_per_day: float) -> str:
    """The traded value as words, precise where it is small: '₹0.28 Cr/day', '₹18.0 Cr/day', '₹1,234 Cr/day'.
    Under ₹0.01 Cr it says so rather than print a '₹0.00' that reads as no trading at all."""
    if cr_per_day < 0.01:
        return "under ₹0.01 Cr/day"
    if round(cr_per_day, 2) < 1:
        return f"₹{cr_per_day:.2f} Cr/day"
    if round(cr_per_day, 1) < 100:
        return f"₹{cr_per_day:.1f} Cr/day"
    return f"₹{cr_per_day:,.0f} Cr/day"


def trade_line(cr_per_day) -> str:
    """The Tear-Sheet's one line: how much trades a day and which order fits — '' when unknown."""
    try:
        v = float(cr_per_day)
    except (TypeError, ValueError):
        return ""
    if v != v or v < 0:                      # NaN or impossible: unknown, said nothing
        return ""
    if v == 0:
        return "no trades in the last 20 days"
    return f"trades {per_day_text(v)} · orders up to about {order_size_text(v)}"


def order_guide() -> str:
    """Which floor fits which order size — derived from LIQUIDITY_STEPS, so it cannot drift from them."""
    return "\n".join(f"- ₹{s:g} Cr/day → orders up to about ₹{order_size_lakh(s):g} lakh"
                     for s in LIQUIDITY_STEPS if s is not None)
