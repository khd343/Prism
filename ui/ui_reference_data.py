"""CONCEPT_REFERENCE — plain-language meaning of every CATEGORICAL VALUE-LABEL PRISM shows.

The glossary (_RAW_GLOSSARY) explains the column NAMES (e.g. "Moat", "PEG Zone"); this explains the
VALUE OUTCOMES those columns take (Wealth Creator, Deep Value, Stage 2, …) — the labels you see on a
Tear-Sheet / All-Data cell that the glossary alone never defines. Grouped by category.

ACCURACY CONTRACT: every explanation is DERIVED FROM THE CODE that produces the label — each category
block cites its `file:line (np.select)` source. Do not edit a meaning without re-reading that source.
Coverage of the filterable sets is pinned by tests/test_concept_reference.py (cross-checked to the
live ui_discovery filters), and a 40-char quality floor guards against shallow entries.

Pure data — zero Streamlit. Rendered by ui_reference.render_concepts(); surfaced in the 📖 Reference tab.
"""
from config import STALE_RESULT_DAYS


CONCEPT_REFERENCE = {
    # ── moat_growth_quad — core/data_engine.py:1725 (has_moat = ROCE 5y-med ≥15; has_growth = PAT 5y ≥15)
    "🧭 Moat × Growth Quadrant": [
        ("⭐ Wealth Creator", "High returns AND high growth — 5-year-median ROCE ≥15% and 5-year profit growth ≥15%. The best quadrant: a durable money-machine that is also compounding."),
        ("🛡️ Quality Trap", "High returns but low growth — ROCE ≥15% yet profit growth below 15%. A quality business whose engine has stopped compounding."),
        ("⚡ Growth Trap", "High growth but low returns — profit growing ≥15% while ROCE is below 15%. Growth that may DESTROY value because it earns below its cost of capital — expansion without economics."),
        ("💀 Wealth Destroyer", "Neither durable returns nor growth — ROCE below 15% and growth below 15%. The weakest position on the moat-vs-growth map."),
    ],
    # ── peg_zone — core/data_engine.py:1815 (PEG thresholds; negative PEG = falling earnings)
    "💰 Valuation — PEG Zone": [
        ("💎 Deep Value", "PEG of 0.5 or less — earnings growth is priced very cheaply relative to the growth rate."),
        ("🟢 Fair PEG", "PEG between 0.5 and 1.0 — Peter Lynch's sweet spot: paying roughly fairly for the growth."),
        ("🟡 Stretched", "PEG between 1.0 and 1.5 — paying a premium to the growth rate; valuation getting fuller."),
        ("🟠 Expensive", "PEG between 1.5 and 2.0 — the price runs well ahead of the earnings-growth rate."),
        ("🔴 Overpriced", "PEG above 2.0 — the valuation is far ahead of the earnings growth it rests on."),
        ("🔴 Declining", "PEG is negative because earnings are falling, so the ratio is meaningless — a warning, not a bargain."),
    ],
    # ── buy_zone_label — core/data_engine.py:1267 (distance above the Volatility Stop)
    "🎯 Entry — Buy Zone": [
        ("🟢 Perfect Entry (Low Risk)", "Price sits within 5% above its Volatility Stop — the tightest, lowest-risk entry: little downside to the stop if the trend breaks."),
        ("🟡 Standard Zone", "Price is 5–12% above the Volatility Stop — a normal volatility buffer; an ordinary entry, not extended."),
        ("🔴 Extended (Wait for Pullback)", "Price is more than 25% above the Volatility Stop — stretched far from support; waiting for a pullback lowers entry risk."),
        ("🔻 Below Stop (Trend Broken)", "Price has fallen BELOW its Volatility Stop — the trend is broken. The most dangerous state, not an entry however cheap it looks."),
        ("🟠 Loose Entry Zone", "12–25% above the Volatility Stop — still in the uptrend, but entering here means accepting a 12–25% loss if the stop is honored (~3× Minervini's max-loss rule). Fine to hold; mediocre to initiate."),
        ("⚪ Uncharted", "No valid Volatility Stop (missing price/volatility data) — entry timing can't be judged."),
    ],
    # ── weinstein_stage — core/data_engine.py:2204 (price vs rising/falling 30-week MA + MA stacking)
    "📈 Trend — Weinstein Stage": [
        ("📈 Stage 2 Advancing", "Price above a rising 30-week moving average with the moving averages stacked up — Weinstein's confirmed uptrend, the buy stage."),
        ("🔄 Stage 1 Basing", "Bottoming/sideways after a decline — building a base before a possible advance; accumulate, don't chase."),
        ("⚠️ Stage 3 Top", "Topping after an advance — momentum stalling and distribution risk rising; tighten up."),
        ("📉 Stage 4 Declining", "Price below a falling 30-week moving average — Weinstein's downtrend, the avoid stage."),
        ("❔ Unknown", "Not enough price / moving-average history to place the stock in a Weinstein stage."),
    ],
    # ── lynch_category — core/data_engine.py:3005 (5-year revenue growth bands)
    "🚀 Style — Lynch Type": [
        ("Fast Grower", "Revenue growing 20%+ a year (5-year) — Lynch's high-growth archetype; judge it on whether the growth can last."),
        ("Stalwart", "Revenue growing 10–20% a year — large, steady compounders; reliable but rarely explosive."),
        ("Slow Grower", "Revenue growing 0–10% a year — mature and low-growth; usually held for dividends/value, not growth."),
        ("Declining", "Revenue shrinking — the 5-year top-line growth is below zero."),
    ],
    # ── mef_label — core/data_engine.py:3219 (moat_endurance_factor = current ÷ 10y-median ROCE)
    "🏰 Moat Endurance": [
        ("🟢 Expanding", "The moat is widening — current ROCE is 1.2× or more of its 10-year median; returns improving over time."),
        ("✅ Intact", "The moat is holding — current ROCE is at or above its 10-year median (about 1.0–1.2×)."),
        ("🟡 Eroding", "The moat is weakening — current ROCE has slipped to 0.80–1.0× its 10-year median."),
        ("🔴 Degrading", "The moat is breaking down — current ROCE is below 0.80× its 10-year median."),
    ],
    # ── smart_money_flow — core/data_engine.py:1234 (volume-quality + FII/DII flow + price confirmation)
    "🌊 Smart-Money Flow": [
        ("🌊💎 Elite Accumulation", "The strongest institutional-buying signal — top volume-quality (80+) with FII and DII flows converging in."),
        ("🎯 Strong Accumulation", "High volume-quality (60+) with FIIs or DIIs net buying — clear accumulation."),
        ("✅ Moderate Interest", "Decent volume-quality (40+) AND the price confirming (not lagging the market) — mild interest."),
        ("⚪ Neutral", "No clear institutional accumulation or distribution signal."),
        ("❌ Distribution", "Both FIIs and DIIs net selling while the price falls — Wyckoff distribution (institutions exiting), not interest."),
    ],
    # ── wealth_tier — core/verdict_engine.py (the change-lens verdict; price- and forensics-blind) ──
    "💹 Wealth Tier": [
        ("CLIMBING★", "All three wealth clocks agree: earning above the cost of equity (EP% > 0), that excess return materially improving this year (Vel% ≥ +0.5), and the 5-year margin trend confirming it (tau ≥ +0.25). Describes the business, never the price: whether the price is right is Valuation's question. Called BUY★ until 2026-10-15."),
        ("CLIMBING", "Earning above the cost of equity and improving, but the 5-year margin trend is flat — the engine is running, its durability is unproven. Called BUY until 2026-10-15."),
        ("RECOVERING", "The confirmed turnaround: NOT yet earning above the cost of equity, but the excess return is improving and the 5-year margins confirm the turn. Soundness's FLAWED pile is where these live. Called WATCH★ until 2026-10-15."),
        ("UNEVEN", "The clocks disagree: improving this year without earning the cost of equity, improving while the 5-year margins fade, or margins confirming without this year's improvement. Fading margins (tau ≤ −0.25) cap even a strong year here. Called WATCH until 2026-10-15."),
        ("STALLED", "Not improving this year (most of these are falling) and no confirming margin trend. The LEVEL may be excellent — a business can earn far above its cost of equity and still land here because its excess return stopped rising. Called AVOID until 2026-10-15."),
        ("N/A", "An input the tier needs (the reserves equity base, economic profit, its velocity, or the margin tau) is missing or unusable — unverifiable is neither passed nor condemned, so no tier is assigned. The row still shows whatever numbers exist."),
    ],
    # ── cf_triangle — core/forensic_engine.py:1006 (signs of operating/investing/financing cash flow)
    "💵 Cash-Flow Triangle": [
        ("✅ Self-Funding", "Operating cash positive, investing in the business, AND paying down financing — the healthiest cash pattern."),
        ("⚠️ Growth Phase — Watch D/E", "Operating cash is positive but the company is also BORROWING to invest — growth funded by debt; watch leverage."),
        ("🚨 Debt Trap — Avoid", "Burning operating cash, still spending, and borrowing to stay afloat — the dangerous cash pattern."),
        ("⚪ Mixed Pattern", "A cash-flow mix that doesn't fit the clean Perfect / Growth / Debt-trap patterns."),
    ],
    # ── corporate_class — core/scoring_engine.py:3142 (MOSL 13th-study Great/Good/Gruesome)
    "🏛️ Corporate Class (MOSL)": [
        ("🏆 GREAT", "MOSL's top capital-allocation class — 10-year-median ROCE ≥25%, strong free-cash conversion, and still high today. A proven, cash-generative compounder."),
        ("👍 GOOD", "Solid but not elite — 10-year-median ROCE ≥12% with weaker cash conversion. Decent economics, not best-in-class."),
        ("💀 GRUESOME", "Destroying economic value — 10-year-median ROCE below 12%, under the cost of capital; earns less than it costs to fund."),
    ],
    # ── capital_allocation_signal — core/data_engine.py:1977 (external_financing_to_assets)
    "🏦 Capital Allocation": [
        ("💰 Returning Capital", "Net capital FLOWS OUT to owners — buybacks, debt repayment and dividends exceed new raising (external financing below −5% of assets). Disciplined."),
        ("⚠️ Raising Capital", "Net capital is being RAISED — new equity/debt exceeds what's returned by more than 15% of assets; dilutive or leveraging."),
        ("⚖️ Neutral", "Capital raised and returned roughly balance (between −5% and +15% of assets)."),
    ],
    # ── cyclicality_tier — core/cyclicality_map.py TIER_LABELS (industry → behavioral tier)
    "🔄 Cyclicality Tier": [
        ("Deep Cyclical / Commodity", "A price-taking commodity business (metals, sugar, refining) — earnings swing hard with the commodity cycle; trade the cycle, don't marry it."),
        ("Cyclical", "Demand/capex/discretionary cyclical — sensitive to the economic cycle, but less extreme than a pure commodity."),
        ("Defensive", "Stable, non-cyclical demand (FMCG, pharma, utilities) — earnings hold up through the cycle; suited to holding."),
        ("Sensitive / Structural-Growth", "A secular, structural grower — driven more by a long-run growth theme than by the macro cycle."),
        ("Financials", "Banks, NBFCs and insurers — they ride their own credit and interest-rate cycle, judged differently from operating companies."),
        ("Catch-all", "Heterogeneous or hard-to-classify (conglomerates, trading, diversified) — no single cyclicality label fits."),
        ("earn-DD (Earnings Drawdown)", "The number beside the tier badge: the deepest peak-to-trough fall in annual net profit over the last 6 years — a drawdown applied to PROFITS instead of price. ~0% = profits only ever rose (steady compounder); ~50% = profits once halved (cyclical signature); above 100% = the trough year was an outright loss (capped at 300% where the math turns to noise). The tier is the business's REPUTATION by industry; earn-DD is its RECORD — a 'Defensive' name carrying a big earn-DD is behaving cyclically despite its label. Needs 4 of 6 years reported; display-only, never scored."),
    ],
    # ── sector_capital_phase — core/data_engine.py:2004 (Chancellor capital cycle, sector asset growth)
    "❄️ Sector Capital Phase": [
        ("🔥 Hot Capital (caution)", "The sector is over-investing — capital is flooding in (high sector asset growth), which historically pressures future returns. Mean-reversion risk."),
        ("❄️ Capital Starved (opportunity)", "The sector is under-invested — little new capital coming in, which historically sets up supply tightness and recovery. Opportunity."),
        ("⚖️ Neutral", "Sector capital investment is neither unusually hot nor starved."),
    ],
    # ── verdict_direction — core/verdict_engine.py:72 (conviction tier + forensic/governance vetoes)
    "⚖️ Soundness": [
        ("SOUND", "The engine's affirmative gate — quality, clean books and a reasonable price all clear (tier 1–2, no forensic or governance veto). Renamed from BUY on 2026-08-27: with 18 passers against 93% of the universe failing, this was always a qualification gate, not a buy call. Condition words here, direction words on the wealth tier, no word in both — and no action words on either."),
        ("MIXED", "The middle of the gate — promising but with a caveat: mid conviction, or a soft forensic/timing/governance downgrade from SOUND. Renamed from WATCH on 2026-08-27."),
        ("FLAWED", "The gate found flaws — low conviction, OR a hard veto from severe forensic red flags / value-destroying capital allocation. Says nothing about direction: a FLAWED stock can be a confirmed turnaround on the wealth tier (RECOVERING), and that pairing is exactly what the two lenses exist to surface. Renamed from AVOID."),
    ],
    # ── conviction_tier / tier_label — config.CONVICTION_TIERS (composite-score bands)
    "🏆 Conviction Tier": [
        ("Crown Jewels", "Tier 1 (composite ≥85) — the highest-conviction compounders; deep-dive and build a position."),
        ("Strong Compounders", "Tier 2 (composite ≥70) — quality with momentum confirmation; watchlist priority."),
        ("Emerging Quality", "Tier 3 (composite ≥55) — quality building and momentum developing; monitor for an upgrade."),
        ("On Radar", "Tier 4 (composite ≥40) — on the radar but not yet qualified; keep watching."),
        ("Not Ready", "Tier 5 (composite below 40) — does not yet clear the bar; pass for now."),
    ],
    # ── mcap_tier / market_category — config.MCAP_TIERS (market-cap bands, ₹ Cr)
    "🏢 Market Cap Tier": [
        ("Mega Cap", "Market capitalisation of ₹2,00,000 Cr or more — the very largest, most-liquid companies."),
        ("Large Cap", "Market capitalisation of ₹20,000–2,00,000 Cr — large, well-established companies."),
        ("Mid Cap", "Market capitalisation of ₹5,000–20,000 Cr — mid-sized companies."),
        ("Small Cap", "Market capitalisation of ₹500–5,000 Cr — smaller companies, more volatile."),
        ("Micro Cap", "Market capitalisation of ₹100–500 Cr — very small companies, often thinly traded."),
        ("Nano Cap", "Market capitalisation below ₹100 Cr — the smallest and least-liquid companies."),
    ],
    # ── cash_machine_label — core/data_engine.py:1046 (cash_machine_score from CFO/PAT)
    "💵 Cash Machine": [
        ("💰 Cash Machine", "Profits are fully cash-backed — operating cash flow comfortably exceeds reported profit. The strongest cash-quality signal."),
        ("✅ Solid", "Operating cash flow is above 80% of reported profit — acceptable cash conversion."),
        ("📄 Paper Profits", "Operating cash flow lags reported profit — earnings aren't turning into cash. A quality warning."),
    ],
    # ── ep_power_curve — core/data_engine.py:1506 (economic profit level × its velocity)
    "📊 Economic-Profit Power Curve": [
        ("🚀 Hockey Stick", "Earns above its cost of equity AND that economic profit is larger than a year ago — climbing the Power Curve. The best state."),
        ("➖ EP Positive, Not Rising", "Earns above its cost of equity, but the profit is not larger than last year — or the prior year isn't reported. Value-creating, not yet climbing."),
        ("📈 Improving", "Economic profit still negative but larger than a year ago — below its cost of equity, yet the trend is turning up."),
        ("📉 Value Trap", "Negative economic profit and no larger than a year ago — earns below its cost of equity with no upturn."),
    ],
    # ── earnings_power_box — core/data_engine.py:1547 (Heiserman defensive × enterprising)
    "📦 Earnings Power Box": [
        ("📦 Earnings Power", "Strong on BOTH Heiserman tests — defensive (stable, profitable) and enterprising (reinvesting for growth). The top box."),
        ("💰 Cash Cow", "Defensive but not enterprising — stable and profitable, but reinvesting little for growth."),
        ("🚀 Cash-Hungry Grower", "Enterprising but not defensive — reinvesting hard for growth, but the base economics aren't yet stable."),
        ("⚠️ Weakest", "Weak on both Heiserman tests — neither stable economics nor productive reinvestment."),
    ],
    # ── 🚀 Multibagger Setups (Candidate Flags) — mirrors the Discovery filter group's OR-list
    # (ui_discovery.py:644 _MULTIBAGGER). Exact dropdown labels so the filter and Reference read 1:1;
    # the EP Power Curve + Earnings Power Box dropdowns in the same group are the two categories above. ──
    "🚀 Multibagger Setups": [
        ("🐘 100x Candidate", "MOSL's Mouse-to-Elephant screen: an early, small-base business with high returns reinvested at scale and a long runway — the raw ingredients for a ~100× multibagger. The rarest, highest-ceiling setup (same flag as the 🐘 entry under MOSL Frameworks)."),
        ("🏅 Category Winner", "The sector leader on the WCS screen: top-30% capital efficiency (ROCE) within its OWN sector AND above-market 5-year revenue growth — winning its category on both quality and growth."),
        ("📈 Compound Growth", "Sustained compounding power — profit growth clears 15% (3Y), 12% (5Y) and 10% (10Y); earnings compound across every horizon, not just one good stretch."),
        ("🛡️ Consistency Champion", "Profits have grown steadily and durably — a 'consistent' compounder whose earnings rise smoothly rather than lumpily. Low-volatility compounding is the coffee-can ideal."),
        ("🔄 ROE Turnaround", "ROE is still below 15% but has turned UP above its 5-year trend — an early-innings quality inflection (a turnaround bargain), caught before the re-rating."),
        ("🧲 Value Migration", "The company is in the top quartile of its sector by revenue growth — a sign that value (demand and market share) is migrating TOWARD this business within its sector."),
        ("💎 Bruised Blue Chip", "A high-quality company trading unusually cheap — below 2× book value — after a setback. Quality on sale: a strong franchise the market has temporarily marked down."),
        ("🚀 Mid→Mega", "A mid-cap with the financial profile to grow into a mega-cap — the size-migration thesis (small/mid → large) that produces the biggest multibaggers."),
    ],
    # ── fisher_lifecycle_quadrant — core/scoring_engine.py:2787 (Fisher quality_pass × scalability pass)
    "🧬 Fisher Lifecycle Quadrant": [
        ("👑 Apex Winner", "Elite quality business at its operating-leverage peak — passes both the Fisher quality and scalability screens. Prime entry."),
        ("🐢 Steady Compounder", "Proven quality with no current inflection — a durable steady-state hold past its fastest phase."),
        ("⚡ Catalyst Play", "A scalability inflection firing but structural quality not yet proven — an earlier, trading-style candidate."),
        ("⚪ Laggard", "Neither the Fisher quality nor the scalability screen passes — no current edge on this map."),
    ],
    # ── malik_label / forensic_label — data_engine.py:1705 / forensic_engine.py:166 (checklist strength)
    "📋 Quality / Forensic Strength": [
        ("🟢 Strong", "A strong rating on the relevant checklist (Malik quality or forensic accounting-quality) — passes most or all of its tests."),
        ("🟡 Moderate", "A middling rating — passes some of the checklist's tests but not most."),
        ("🟠 Weak", "A weak rating — fails most of the Malik quality checklist's tests."),
        ("🔴 Poor", "The lowest Malik-quality rating — fails nearly all of the financial-strength tests."),
        ("🔴 Weak", "A weak forensic rating — the accounting-quality checks raise concern."),
    ],
    # ── Piotroski Strength — ui/ui_discovery.py np.where (F-Score band)
    "🛡️ Piotroski Strength": [
        ("💪 Strong (≥7)", "Piotroski F-Score of 7–9 — very healthy books across profitability, leverage and efficiency."),
        ("➖ Moderate (4–6)", "Piotroski F-Score of 4–6 — middling financial health on the nine-point checklist."),
        ("⚠️ Weak (≤3)", "Piotroski F-Score of 3 or below — weak financial health; treat with caution."),
    ],
    # ── trend_modifier — core/data_engine.py:2230 (Weinstein stage × Grimes path event)
    "↩️ Trend Modifier": [
        ("↩️ Pullback", "Trend continuation — the higher-edge setup in Grimes's research: a Stage 2 dip below the 50-day line, still above the 30-week average, on lighter volume (the 10-day average below the 50-day) — a buy-the-dip setup."),
        ("🚀 Breakout", "With-trend edge: a Stage-2 stock within 3% of its 52-week high, not extended, on volume expansion — a breakout setup."),
        ("⚠️ Bounce", "Low-confidence counter-trend: a Stage-4 rally back up to the falling 30-week MA — Weinstein's 'don't chase' zone."),
        ("⚠️ Extended", "Low-confidence caution: stretched far above the 30-week MA and overbought (RSI > 70) — termination risk."),
    ],
    # ── consolidation_base — core/data_engine.py (Minervini's VCP, as far as one day's data can see it)
    "⏳ Consolidation Base": [
        ("⭐ VCP Setup", "A consolidation base as far as one day's data can check Minervini's VCP: volume has dried up — the 10-day average below 70% of the 50-day — in a Stage 2 uptrend within 25% of its 52-week high and not below its stop. The narrowing price swings need a chart."),
        ("💤 Drying Up", "The same 10-day volume dry-up outside that setup — a stock not in Stage 2 (or with no stage yet), one more than 25% below its 52-week high, or one below its stop. Quiet alone is not a consolidation base."),
        ("⚪ Normal Volume", "No dry-up, so no consolidation base by this measure: the 10-day average volume is at least 70% of the 50-day average."),
    ],
    # ── d48_breakout_readiness — core/data_engine.py:2280 (distance to 52w/13w high)
    "🎯 Breakout Readiness": [
        ("🎯 IMMINENT", "Within 10% of the 52-week high AND within 5% of the 13-week high — a breakout looks imminent."),
        ("NEAR", "Within 20% of the 52-week high — approaching breakout territory."),
        ("FAR", "More than 20% below the 52-week high — not near a breakout."),
    ],
    # ── d49_momentum_quality — core/data_engine.py:2289 (RSI + ADX)
    "⚡ Momentum Quality": [
        ("🔥 OVERHEATED", "RSI above 70 — momentum is strong but overbought; pullback risk is elevated."),
        ("⚡ HIGH", "RSI in the healthy 50–70 zone with a strong trend (ADX > 20) — high-quality momentum."),
        ("WEAK", "Momentum is weak — neither overbought nor in a confirmed strong trend."),
    ],
    # ── verdict coverage confidence — core/verdict_engine.py:92 (evidence coverage %)
    "🔍 Soundness — Evidence Confidence": [
        ("High", "The soundness verdict rests on 80%+ evidence coverage — most ranked inputs reported; trust it more."),
        ("Medium", "60–80% evidence coverage — a fair amount of the inputs reported."),
        ("Low", "40–60% evidence coverage — a meaningful share of inputs are missing; treat with care."),
        ("Very Low", "Under 40% evidence coverage — the soundness verdict rests on thin data; tentative."),
    ],
    # ── result_age_days / result_stale_flag — core/data_engine.py (sign-flipped days_from_result;
    # stale past config.STALE_RESULT_DAYS). Shown as the tearsheet '⏳ Stale Nd' badge + the Discovery
    # 'Hide Stale' filter.
    # A recency signal (how OLD the data is) — sibling to, and distinct from, the Evidence badge above. ──
    "⏳ Result Recency": [
        ("⏳ Stale", f"The tearsheet badge shown when the company has filed no result for more than {STALE_RESULT_DAYS} days — past every listing deadline, whether it reports quarterly or, on the SME platform, half-yearly. A missed deadline often comes before distress, and the financials may predate recent events, so treat the numbers as out of date. The Discovery tab's 'Hide Stale' filter drops these."),
        ("Result Age (days)", "How many days since the company last reported financial results — higher means staler numbers. The recency sibling to the 🔍 Evidence badge: coverage measures how MUCH of the data reported, this measures how OLD it is."),
    ],
    # ── verdict_axis_governance — core/verdict_engine.py:116 (governance multiplier)
    "🛡️ Soundness — Governance Axis": [
        ("Govern 🟢 Safe", "No governance penalty — promoter pledge, dilution and related-party signals are clean."),
        ("Govern 🟡 Caution", "A mild governance penalty — one or more governance signals warrant caution."),
        ("Govern 🔴 Risk", "A heavy governance penalty — serious pledge/dilution/related-party risk drags the score."),
    ],
    # ── verdict_axis_forensics — core/verdict_engine.py:111 (nested np.where): the Forensics pill in
    # the 6-axis scorecard, parallel to the Governance axis above. np.where in an engine file isn't
    # enumerated by the categorical-label net, so these need explicit entries. ──
    "🔬 Soundness — Forensics Axis": [
        ("Forensics 🟢 Clean", "The scorecard's Forensics pill when accounting signals look clean — fewer than 5 red flags and no severe forensic or Schilit veto."),
        ("Forensics 🟡 Watch", "The Forensics pill when 5 or more red flags fire but no hard veto — some accounting-quality caution; read the specific flags before acting."),
        ("Forensics 🔴 Flagged", "The Forensics pill when a severe forensic veto fires — forensic score below 50, ten or more red flags, or a Schilit checker hard-fail. A serious accounting-quality concern."),
    ],
    # ── verdict_top_risk — core/verdict_engine.py:121 (the single most important risk)
    "⚠️ Soundness — Top Risk": [
        ("🚨 Severe forensic / accounting-quality flags", "The dominant risk: severe forensic red flags veto the thesis — verify the accounts before anything."),
        ("💀 Value-destroying capital allocation", "The dominant risk: a Gruesome capital allocator earning below its cost of capital — destroys value."),
        ("🕵️ Schilit forensic checker flags", "The dominant risk: Schilit accounting-quality checkers fire — the reported numbers may be aggressive."),
        ("⚠️ Governance risk (pledge/dilution)", "The dominant risk: governance — meaningful promoter pledging or dilution."),
        ("⏳ Poor entry timing — wait for a base", "The dominant risk is timing, not quality — the chart is poorly placed; wait for a base."),
        ("🔍 Thin data — verdict tentative", "The dominant caveat: evidence coverage is thin, so the soundness verdict is tentative until more inputs report."),
    ],
    # ── Catalysts — ui/ui_discovery.py _CATALYSTS (fast-moving change triggers)
    "🔥 Catalysts": [
        ("🔥 Capacity Explosion", "A capacity-expansion catalyst — the fixed-asset block grew this year AND has compounded faster than 15% a year over 3 years. It reads fixed assets only, not CWIP, so it confirms capacity already on the books rather than capacity still under construction."),
        ("🔥 OpLev Inflection", "An operating-leverage inflection — profit growth pulling decisively ahead of revenue growth."),
        ("🔥 Deleveraging", "A deleveraging catalyst — debt being repaid materially, easing the balance sheet and interest drag."),
        ("🔥 Lynch Dream", "A Lynch GARP setup — fast growth available at a reasonable PEG; the classic Peter Lynch profile."),
        ("🔥 Inst Discovery", "Early institutional discovery — accumulation signs while the stock is still under-owned."),
        ("🔥 Strong Quarter", "A results-season catalyst — the latest reported quarter beat the same quarter last year on all three counts: revenue up more than 15%, profit up more than 20%, and EBITDA margin wider. Growth is only measured off a positive base, so a company that made a loss, or had no revenue, a year ago cannot be judged and is never flagged. It moves through each results season. Shown, not scored."),
    ],
    # ── Sell Alerts — ui/ui_discovery.py _SELL_ALERTS (Baid sell triggers)
    "🚨 Sell Alerts": [
        ("🚨 Cash Collapse", "Operating cash flow has collapsed relative to profit — the cash engine is breaking down."),
        ("🚨 Overvalued", "The valuation has run far ahead of the fundamentals — priced for perfection."),
        ("🚨 Thesis Broken", "A core pillar of the bull thesis (growth/returns) has broken — re-underwrite or exit."),
        ("🚨 Treadmill", "Running to stand still — heavy reinvestment producing little incremental return."),
        ("🚨 Sequential Decline", "Sequential (quarter-on-quarter) deterioration — the recent trend is turning down."),
        ("🚨 Mgmt Deteriorated", "Management-quality / governance signals have deteriorated — a red flag for owners."),
    ],
    # ── Frameworks — _FW_META (ui_tearsheet.py:853); explanations condensed from docs/handbook/08 ──
    # Each is a famous investor's/book's pass-fail screen; a Tear-Sheet shows which a stock passes.
    "🏛️ Frameworks — MOSL Wealth Creation": [
        ("🥇 QGLP", "Raamdeo/MOSL flagship — Quality + Growth + Longevity at a reasonable Price: ROCE ≥~15%, 5-yr profit growth ≥~15%, PEG ≤~1.5 with positive earnings."),
        ("🌟 MOSL Wealth Creator", "The proven long-run wealth-creator profile — profit growth consistent across 3/5/10-yr horizons, a low payback ratio, and a wide economic-profit spread."),
        ("👑 SQGLP Century Stock", "The strictest bar — QGLP plus Size: the full quality-growth-price test on a small/mid base that can still multiply many times. Passes are very rare."),
        ("🐘 100x Candidate", "MOSL Mouse-to-Elephant — an early, small-base business with high returns reinvested at scale and a long runway: the raw ingredients for a ~100× multibagger."),
        ("🩹 Fallen Quality", "A genuinely high-quality business temporarily beaten down — strong long-run quality plus a sharp recent price/valuation fall. Quality on sale."),
        ("📐 CAP-GAP Compounder", "MOSL 22nd study: RoE of 15%+ sustained across the decade, five years and today (a long Competitive Advantage Period) AND profit growth of 15%+ across all three windows (a long Growth Advantage Period). Longevity proven on both axes."),
        ("🏰 Economic Moat", "MOSL 17th study: return on equity above the SECTOR average in at least 4 of 5 time windows — a durable edge over direct peers. Sector-relative by design: a 20% RoE bank can have a moat a 50% RoE consumer company lacks."),
        ("💙 Blue Chip Quality", "An established, high-quality large-cap — large size with strong, stable quality metrics. A proven, lower-risk compounder."),
        ("🌪️ Consistent in Volatile", "A steady performer through volatile markets — low earnings/return volatility alongside solid quality. Holds up when the market doesn't."),
        ("🏒 EP Hockey Stick", "MOSL's 28th study: earns above its cost of equity, that profit is growing, AND the shares still trade at a P/E of 20 or less — the study's full TEMP setup (value creation bought cheap)."),
        ("🏛️ Bruised Blue Chip 29", "MOSL's 29th study — a quality large-cap fallen hard, trading at a steep discount (P/B ≤~2×) to its own history. Temporarily punished by the market."),
        ("🌐 Multi-Trillion Cap", "MOSL's 30th study — the very largest, most-proven compounders: mega-cap size with elite, durable quality. Battleship-grade."),
    ],
    "📚 Frameworks — Fundamental & Cash Quality": [
        ("☕ Coffee Can", "Mukherjea — own clean, consistent compounders and forget them: ROCE ≥15% over 10 & 5 yrs, revenue up every year, CFO/EBITDA ≥90%, D/E <1, pledge <10%."),
        ("💎 Diamond", "Mukherjea's 'Diamonds in the Dust' — forensic-verified compounders: high consistent returns, low debt, and clean accounting that survives the forensic screen."),
        ("🕊️ Peaceful Investing", "Vijay Malik's systematic forensic filter — margins, cash conversion, low debt (interest cover ≥3×), self-funded growth, clean forensics. Sleep-at-night investing."),
        ("💰 Unusual Billionaires", "Mukherjea's 'Greatness Formula' — sustained high returns on capital AND consistent revenue growth reinvested over a long period. A great franchise, both engines firing."),
        ("⏳ Long Game Quality", "Khandelwal's fort-like businesses — the strictest balance-sheet bar (interest cover ≥5×) plus strong free cash flow after capex (FCF ≥60% of profit). Funds its own future."),
        ("📚 Baid Compounder", "Gautam Baid's steady compounding — solid quality with balance-sheet strength and consistent growth. A dependable, disciplined compounder."),
        ("🏅 Basant 30% Club", "Basant Maheshwari — a fast grower at a reasonable price: high sustained earnings growth (book bar ~20%+) at a valuation that still leaves room."),
        ("⭐ Quality Compounder", "The 'Quality Investing' three-circle compounder — low capital intensity (asset turnover >4), a free-cash-flow floor (yield ≥2%), and high returns on capital. Asset-light, cash-generative."),
    ],
    "⚡ Frameworks — Momentum & Growth": [
        ("📡 CAN SLIM", "O'Neil — elite growth at a confirmed breakout: current EPS & sales each +25%, 5-yr EPS ≥25%, ROE ≥17%, within 15% of the 52-wk high, volume ≥1.5×, top-20% RS, institutions buying, market not bearish. Very rare."),
        ("⚡ SEPA Momentum", "Minervini's Specific Entry Point Analysis, as PRISM checks it: 4 of 5 trend checks (price above its 30-week and 200-day averages, the averages stacked, price above its stop), the price above its 50-day average, a strong trend (ADX 20+), at least 30% above the 52-week low and within 25% of the high, relative strength beating the Nifty 500 on all three timeframes, institutions adding (FII or DII stake up last quarter) and earnings growth (EPS 25%+, revenue 20%+, ROE 17%+), on a ₹500 Cr+ company. A volume dry-up (10-day volume below 70% of the 50-day) adds to the score but is not required."),
        ("🚀 Quality Momentum", "Wesley Gray's Quantitative Momentum — the top 10% of ₹500 Cr+ stocks by their 12-month return excluding the last month (Gray's 2-12 momentum), with all three relative-strength readings positive, a governance guard (pledge ≤30%) and quality. Durable price momentum in a quality name."),
        # Name matches _FW_META / frameworks_passed exactly ("Lynch Dream") — the trailing
        # "Framework" made this the one entry a pill-to-reference lookup could not resolve.
        ("👓 Lynch Dream", "Peter Lynch's growth-at-a-reasonable-price — strong EPS growth at PEG ≤~1, modest institutional ownership (room to be discovered), real free cash flow, and no inventory surge."),
        ("📈 EP Improver", "MOSL 28th study's turnaround stage: economic profit still NEGATIVE but climbing, with returns, capital efficiency and margins all turning at once. The study calls this stage speculative — completed turnarounds earned its best returns, at lower odds. The step before 🏒 EP Hockey Stick."),
        ("😊 SMILE", "Vijay Kedia — Small size, Medium experience, Large aspiration, Extra-large potential, with Integrity: a ₹100–2,000 Cr small-cap, 5-yr growth ≥20%, ROCE ≥20%, honest management."),
    ],
    "🛡️ Frameworks — Valuation & Capital Allocation": [
        ("🧮 Magic Formula", "Greenblatt — cheap AND good: high earnings yield (EBIT/EV ≥8%) and high return on capital (ROCE ≥20%), excluding financials/utilities. A high-return business at a cheap enterprise price."),
        ("🎲 Dhandho Asymmetry", "Pabrai — heads-I-win, tails-I-don't-lose-much: fallen ≥30% from the 52-wk high (perceived risk) yet a high FCF yield ≥8% (actual low risk), with clean forensics."),
        ("🔄 Parikh Contrarian", "Parag Parikh — out-of-favour but sound: low valuation (P/E <20), strong liquidity (current ratio >1.5), and decent returns (5-yr ROCE ≥12%). A sensible contrarian value candidate."),
        ("🌊 Wide Moat", "Pat Dorsey — structural, durable moats: high returns (ROCE ≥20% both windows), a healthy FCF yield ≥5%, and a moat that isn't eroding. Wide and holding."),
        ("🎯 Outsider CEO", "Thorndike's 'Outsiders' — elite capital allocators: disciplined buybacks without dilution, strong cash generation, debt discipline. Compounds per-share value."),
        ("🔮 Expectations Matrix", "Mauboussin — the price embeds expectations: judge the growth/returns the price implies versus what the business can deliver. A pass = the market implies less than the business can."),
        ("🕵️ Schilit Clean", "Howard Schilit's Financial Shenanigans screen, passed — the stock clears the four-checker accounting-manipulation perimeter (at most two checkers firing)."),
        ("🛡️ Marks Cycle Shield", "Howard Marks — respect the cycle: price-vs-value and cycle position are favourable, not late-cycle euphoric. A defensive overlay."),
    ],
    "🎣 Frameworks — Fisher & Mayer": [
        ("🎣 Fisher Quality", "Philip Fisher's 15 qualitative points, as automated proxies — margins, growth durability, R&D/efficiency, management quality. Scores like a Fisher 'uncommon' franchise."),
        ("📶 Fisher Scalability", "Does the business still have room to grow — a revenue runway, operating leverage, pricing power, and no dilution. The growth story isn't finished."),
        ("💯 100-Bagger", "Phelps/Mayer '100 Baggers' — the long-compounding, small-base setup: growth consistent across horizons, a low payback ratio, a wide economic-profit spread, a small base, and low pledging."),
    ],
    # ── Market regime (scoring_engine detect_market_regime) ──
    # Market-wide readings shown on the Market Pulse tab + the banner — never traits of one stock.
    "🌊 Market & Regime": [
        ("🐂 BULL", "The whole market is in a healthy uptrend — strong breadth (most stocks above their long-term averages). PRISM trusts momentum signals more in a Bull regime."),
        ("🐻 BEAR", "The whole market is weak — poor breadth. PRISM gets stricter and BLOCKS new momentum (CAN SLIM) entries, since most stocks fall in a falling market. Weight quality and caution."),
        ("➡️ SIDEWAYS", "The market is range-bound — breadth is mixed, neither clearly rising nor falling. The default, in-between regime."),
        ("🌊 Tsunami", "The rarest, highest-conviction setup — all SEVEN conviction conditions (quality + momentum + governance + technical) fire at once. On most days none qualify: it also needs a brand-new uptrend (the volatility stop turned green within the last four weeks)."),
        ("🚀 Tipping Points", "A Market Pulse watch-list of stocks at a potential inflection — where a change in the business may be about to accelerate. Context, not a verdict."),
    ],
    # ── Sizing-cockpit cards — ui_tearsheet.render_valuation_inversion_and_sizing_cockpit (~2999) ──
    # Deep Layer-3 metric cards on the Tear-Sheet's Matrix & WCS tab; value + status only, so the
    # term itself needs defining here (read from the cockpit's labels/thresholds).
    "🔮 Returns & Mispricing Cockpit": [
        ("👑 Expected Return Estimate", "A three-part estimate of the annual return the business itself could deliver: sustainable growth + free-cash-flow yield + a gradual re-rating toward a quality-justified P/E. Each part is capped at sensible limits and the card shows the split, so one broken input can never fabricate a wild number. A model estimate from fundamentals — not a price forecast."),
        ("⏳ Margin Trend (5Y Tau)", "Whether OPERATING MARGINS have been trending up or fading across roughly the last five years (a rank-correlation over the margin history). Positive (above ~0.25) = margins strengthening, often a widening moat; negative = margin pressure. Direction only, not level."),
        ("📊 Price vs Fundamentals", "Where this stock's price sits versus every other stock AFTER adjusting for fundamentals (profitability, growth, size). 'Cheaper than 80%' means only one in five stocks is priced lower for similar quality. A relative-value rank — remember cheap can be cheap for a reason."),
        ("🚨 Hard Volatility Stop-Loss Level", "The price at which a volatility-based trailing stop would trigger — a risk-management exit reference for sizing a position, never a price target or a forecast."),
        ("🎯 Executable Capital Weight", "The position size the sizing engine can justify TODAY as a % of the portfolio — a fractional-Kelly heuristic on proxy odds, capped by the 1%-risk volatility-stop rule, per-stock (not portfolio-normalized). Read it beside the EV thesis band: a 0% here with a high band means 'target — but no entry right now'."),
        ("💰 Capital Deployment (10L Base)", "The rupee amount the Executable Capital Weight implies on an illustrative ₹10-lakh portfolio — the same weight expressed in rupees, to make the sizing concrete."),
        ("Value Creation Velocity", "Reinvestment rate × capital spread (the ROCE earned above the cost of capital) — how fast the company compounds intrinsic value by reinvesting at high returns. Higher = faster wealth creation."),
        ("Market-Implied Expectations Gap", "The gap between the growth the current share price already implies and the growth the business actually needs to justify it (Mauboussin). Positive = the market expects MORE than the fundamentals require — a high bar to clear."),
    ],
    # ── Mauboussin Expectations-Investing Radar — ui_tearsheet.render_mauboussin_radar (~2817): the
    # PIE audit (T/O/C pillars) + per-stock Payoff Framework. These are numeric / UI-composed card
    # signals (not np.select categoricals), so the label-coverage net doesn't enumerate them — grounded
    # in mauboussin_expectations_specs.json. The framework one-liner lives under "Frameworks" above. ──
    "🔮 Mauboussin — Expectations Investing": [
        ("Price-Implied Expectations (PIE) Audit", "Mauboussin & Rappaport's core move, read off the price: instead of guessing a fair value, judge the expectations the current price already bakes in — and score how many of three gates (Gap / Treadmill / CAP) the stock clears, shown out of 3. Repointed 2026-08-28 (v2.0): the certifying gate is now the book's own ch.7 buy standard — the expectations gap — where the old gate reduced to a single operational flag (the other two legs cleared 95%+ of stocks) and certified 45.8% of the universe."),
        ("G · Expectations Discount", "PIE pillar 1, the CERTIFYING gate — a green check means the growth the price implies (inverted from the P/B–Gordon identity) sits at least 5 points BELOW the growth the business can sustainably fund: a margin of safety in expectations, the book's ch.7 buy standard. Strict: if any input is missing the pillar fails — unverifiable is never certified. The 5-point bar is PRISM's census-calibrated choice; the book deliberately prescribes no universal cutoff."),
        ("T · Treadmill Safety", "PIE pillar 2, a DISQUALIFIER — a green check means the stock is NOT priced for indefinite perfection: it doesn't need a continuous stream of positive surprises just to hold today's price. Red means the price already assumes relentless out-performance."),
        ("C · CAP Trap Clear", "PIE pillar 3, a DISQUALIFIER — a green check means there's no dangerous pairing of a long competitive-advantage-period expectation with DECELERATING returns on capital. The trap: the price assumes a durable moat while ROCE is actually sliding."),
        ("Implied CAP Proxy", "A UNITLESS proxy (P/E × NOPAT margin × retention) for how long the price assumes the business keeps earning excess returns — higher = a longer implied competitive-advantage period, but NOT literal years. Very high values are a caution: the market is paying for a moat that must last improbably long."),
        ("NOPAT Margin", "Net Operating Profit After Tax as a share of sales — the clean, capital-structure-neutral operating profitability that Mauboussin's value-driver math runs on (it strips out financing effects)."),
        ("🧮 Payoff Framework — Expected Excess Return", "The per-stock expected value of the trade: P(Upside) × Upside% − P(Downside) × Downside%, where P(Upside) is the trajectory-calibrated win probability. The book's bar to act is a minimum 5% edge."),
        ("EV Upside %", "The reward leg of the payoff: how far the price could rise to reach the P/E its quality justifies (the gap to a quality-fair multiple)."),
        ("EV Downside %", "The risk leg of the payoff: the distance from today's price down to the volatility stop-loss level. Paired with EV Upside % to size the bet honestly."),
        ("EV Verdict & Position Sizing", "Translates the Expected Excess Return into PRISM's own THESIS sizing bands — roughly: 15%+ edge → High Conviction (8–12% weight); 10%+ → Moderate-High (5–8%); 5%+ → Moderate (3–5%); below 5% → Insufficient Edge, no position. Attribution corrected 2026-08-27: the expected-value method is Mauboussin & Rappaport (Expectations Investing ch.7, 'Buy, Sell, or Hold?'), but the BANDS ARE OURS — the book has 12 chapters (no ch.13) and prescribes no position sizes, no hurdle rate and no Kelly (0 occurrences of each). It gives price-to-expected-value and years-to-convergence, and leaves sizing to the investor. This is a thesis size, not an executable one — see the cockpit's thesis-vs-executable strip."),
    ],
    # ── Schilit Accounting Anomaly Shield — ui_tearsheet.render_schilit_shield (~2162); the four
    # Schilit checkers (schilit_ems/cfs/kms_* flags). Wording follows each checker's own description. ──
    "🛡️ Schilit Anomaly Shield": [
        ("EMS Anomaly Gimmick", "Schilit Earnings-Manipulation check — flags income-statement gimmicks such as aggressive revenue recognition or capitalising expenses to inflate reported profit."),
        ("CFS Cash Flow Trap", "Schilit Cash-Flow-Shenanigans check — flags operating-cash divergence and paper-profit shifts, where reported earnings aren't backed by real operating cash."),
        ("KMS Leverage Mirage", "Schilit Key-Metrics check (leverage) — flags off-balance-sheet guarantees and pledged-cash mismatches that disguise a company's true leverage."),
        ("KMS Operational Bloat", "Schilit Key-Metrics check (operations) — flags channel-stuffing and asset/inventory aging that bloats the balance sheet ahead of trouble."),
    ],
    # ── Economic-profit dynamics + tax — ui_tearsheet EP power-curve strip (~305) + tax_rate_est ──
    "📈 Economic-Profit Dynamics & Tax": [
        ("EP Velocity (YoY)", "The year-on-year change in economic profit (₹ Cr) — how fast the company's profit ABOVE its cost of capital is rising or falling. Rising velocity means value creation is accelerating."),
        ("EP Trajectory", "The company's position on Motilal Oswal's economic-profit power curve (28th study) — where it sits on the create → sustain → erode arc of economic value over time."),
        ("Tax Rate (Est.)", "The estimated effective tax rate, (PBT − PAT) ÷ PBT. A profitable company paying under ~10% can signal deferred-tax exhaustion, tax-holiday reliance, or opaque structuring — a forensic caution, not a positive."),
    ],
    # ── Systematic Fisher Proxy — ui_tearsheet.render_fisher_module (~1033); Fisher's 15 qualitative
    # points, the 7 quantifiable from CSV data. Each entry pairs the Fisher point with its proxy. ──
    "🧠 Systematic Fisher Proxy": [
        ("P1: Market Potential", "Fisher Point 1 — does the business have products or services with enough market room for years of sales growth? Proxy: 5-year revenue growth of 15% or more."),
        ("P4: Sales Org Efficiency", "Fisher Point 4 — an above-average sales and distribution organisation. Proxy: profit growing faster than sales (operating leverage is working)."),
        ("P5: Worthwhile Margins", "Fisher Point 5 — does the business earn a worthwhile profit margin? Proxy: a net profit margin above 10%."),
        ("P6: Margin Trajectory", "Fisher Point 6 — is the company doing what it needs to maintain or improve margins? Proxy: net margin at least as high as last year."),
        ("P10: Accounting Controls", "Fisher Point 10 — sound cost analysis and accounting controls. Proxy: operating cash flow at least 70% of reported profit, so earnings are backed by real cash."),
        ("P13: No Equity Dilution", "Fisher Point 13 — will growth force equity raises that dilute existing holders? Proxy: a stable share count, with no meaningful dilution."),
        ("P15: Accounting Integrity", "Fisher Point 15 — management of unquestionable integrity. Proxy: a clean forensic verdict — a high forensic score with few red flags."),
    ],
    # ── Hard gates + verdict-band states — config.HARD_GATES (descriptions verbatim) + the verdict
    # header's SYSTEM-REJECTED / SELL-ALERT branches in app.py. Every stock must pass ALL gates. ──
    "🚨 Hard Gates & Rejection (Pass ALL)": [
        ("Gate-Passed", "The stock cleared EVERY hard safety gate below — the universal floor a stock must pass before PRISM scores it seriously. 'Gate-passed' means safe and eligible, not 'buy'."),
        ("SYSTEM REJECTED", "The Tear-Sheet Soundness-band state when a stock FAILS any one hard gate — it is eliminated regardless of its other scores. The band names the gate that failed."),
        ("SELL ALERT", "The Soundness-band state when a Baid sell-trigger has fired (e.g. cash collapse, thesis broken) — a held or candidate stock flashing risk; review the Forensics tab before acting."),
        ("Debt Safety (gate)", "Hard gate: debt-to-equity ≤ 1.0 — caps balance-sheet risk before a stock can score (Baid prefers ≤ 0.5)."),
        ("Current Ratio (gate)", "Hard gate: current ratio ≥ 1.0 — a basic liquidity floor, so current assets at least cover current liabilities."),
        ("Pledge Safety (gate)", "Hard gate: promoter shares pledged ≤ 20% — limits the forced-selling risk that comes from promoters pledging stock as collateral."),
        ("Pledge Direction (gate)", "Hard gate: promoter pledging is NOT rising quarter-on-quarter — a rising pledge is an early governance warning."),
        ("Promoter Alignment (gate)", "Hard gate: promoter holding ≥ 30% — the founders must keep meaningful skin in the game."),
        ("Cash Quality (gate)", "Hard gate: operating cash flow ≥ 70% of reported profit — earnings must be backed by real cash, not just accounting accruals."),
        ("No Dilution (gate)", "Hard gate: the share count may not grow more than 10% in a year — small ESOP-level issues pass. A bigger jump fails whether it was a placement, a rights issue, a bonus or shares held by an employee trust that only this year's count includes: PRISM's data cannot tell them apart. A jump of 1.5x or more is treated as a bonus or split and passes."),
        ("Positive OCF (gate)", "Hard gate: operating cash flow must be positive — the business has to actually generate cash from its operations."),
        ("Positive PAT (gate)", "Hard gate: annual profit after tax above zero — loss-making companies do not pass the screen."),
        ("Revenue Floor (gate)", "Hard gate: revenue growth of at least −20% year-on-year — excludes businesses in revenue freefall."),
        ("Mandate Screen (ROCE · Growth · PEG)", "On top of the universal safety gates, each mandate adds its own thesis screen — a minimum ROCE, a minimum growth rate and a PEG ceiling (shown in the banner). 'Mandate fit' = passes both the safety gates and this screen."),
    ],
    # ── Forensic integrity verdicts — forensic_engine.py forensic_label (np.where, ~656) + the
    # Schilit shield pass/fail banner (schilit_pass, score ≥ 70). Binary verdicts shown on the UI. ──
    "🕵️ Forensic Integrity (Schilit & Purity Gates)": [
        ("🟢 Clean", "The forensic-integrity state 'Clean' — the stock clears a strict four-part hard gate: operating cash flow ≥ 80% of profit, promoter pledge under 10%, no share dilution, AND zero red flags. The binary integrity stamp the SQGLP gate relies on."),
        ("⚠️ Integrity Gates Open", "The forensic-integrity state when any one of those four purity conditions is not met — most of the universe sits here, since a single red flag of 28 opens a gate. It means 'not surgically clean', NOT detected manipulation: a stock can hold a high forensic score and a Schilit pass and still show this. Treat it as a prompt to read WHICH flags fired, not as an accusation."),
        ("Perimeter Secure (Schilit)", "The Schilit Anomaly Shield's PASS state — at most two of the four Schilit checkers fired (a Schilit score of 70 or more). The accounting clears the manipulation screen."),
        ("Shenanigan Alert (Schilit)", "The Schilit Anomaly Shield's FAIL state — three or more of the four checkers fired (Schilit score below 70). The accounting raises manipulation concerns; investigate before trusting the reported numbers."),
    ],
    # ── Analysis Mode selector — config.ANALYSIS_MODES (label + description). ──
    "🎛️ Analysis Mode": [
        ("🔀 Hybrid (Quantamental)", "Analysis Mode — scores on BOTH fundamentals and technicals: a great business that institutions are also buying now. Weights quality 70% to momentum 30%."),
        ("📚 Fundamental Only", "Analysis Mode — pure business quality, setting price action aside. For long-term, buy-and-hold Coffee Can investors."),
        ("📈 Technical Only", "Analysis Mode — pure price action and institutional money flow (O'Neil rules), with fundamentals set aside."),
        ("⚡ Quantamental + Breakout", "Analysis Mode — quality, momentum and breakout in equal thirds, which puts price strength at roughly two thirds of the composite (breakout is already part of momentum). It out-predicted the Hybrid blend on every forward window measured so far, and the first six-month window in December compares it with Momentum + Breakout."),
        ("🚀 Momentum + Breakout", "Analysis Mode — momentum and breakout in halves, with quality left out of the blend; the forensic penalty, hard gates and framework boosts still act on the score. The default: it ranked the stocks that went on to do best more accurately than the equal-thirds blend on three of the four forward windows measured so far — all flat or falling markets, the longest about three and a half months — and the six-month window in December reviews that."),
    ],
    # ── The QGLP screen — config.MASTER_PROFILES, now a CONSTANT not a selector. ──
    # The eight-profile "Scoring Profile" control was removed 2026-09-20: it re-ranked nothing
    # (composite, rank and the top-50 were identical across all eight) while swinging qglp_pass
    # 4x, and a mode change could rewrite it silently. Seven entries went with it; a reference
    # that still described them would send a reader looking for a control that is not there.
    "🎚️ QGLP Screen": [
        ("Balanced (QGLP)", "The fixed QGLP screen — Raamdeo Agrawal's framework as the 25th Wealth Creation Study states it: a balanced weighting of Quality, Growth, Longevity and Price, gated at ROCE ≥ 15%, PAT 5Y CAGR ≥ 15% and PEG ≤ 1.5. It is a FACT about the framework, not a setting: there is no profile selector, and the same gates apply to every stock on every surface (the 🏛️ QGLP tab, the tearsheet QGLP card and radar, and the QGLP lens in the 🔭 MOSL convergence count). The market regime is the one thing that moves them — a detected BEAR tape raises the ROCE gate to 20%."),
    ],
    # ── tier_label — config.CONVICTION_TIERS via apply_forensic_penalty (post-penalty score bands:
    # ≥85 / ≥70 / ≥55 / ≥40 / rest). Added 2026-08-28: the app's most visible labels were absent
    # from the searchable reference (found by the all-label coverage sweep). ──
    "🏆 Conviction Tier": [
        ("🏆 Crown Jewels", "Post-penalty composite score of 85 or more — the highest-conviction compounders; deep-dive and build a position. The rarest band (single digits of 2,100+ stocks)."),
        ("🥇 Strong Compounders", "Score 70–84 — quality with momentum confirmation; watchlist priority."),
        ("🥈 Emerging Quality", "Score 55–69 — quality building and momentum developing; monitor for an upgrade."),
        ("🥉 On Radar", "Score 40–54 — some quality signals but they need time; the early watchlist."),
        ("❌ Not Ready", "Score below 40 — insufficient quality or momentum for the engine to carry a thesis; ignore for now."),
    ],
    # ── atoms_to_bits_label — core/data_engine.py:1419 (sector-mapped, 26th WCS Atoms→Bits;
    # unmapped sectors default to Hybrid). ──
    "🌐 Business Design (Atoms → Bits)": [
        ("Bits", "An asset-light, digital-first business (26th Wealth Creation Study taxonomy) — scales through networks and software rather than plants; sector-mapped."),
        ("Atoms", "A capital-intensive physical business — growth needs plants, inventory and working capital, so scaling is linear with capital."),
        ("Hybrid", "A mixed physical-plus-digital model — also the default for sectors the Atoms→Bits map does not classify either way."),
    ],

}


# ═══════════════════════════════════════════════════════════════════════════════════════════
# WCS_STUDIES — Raamdeo Agrawal's Annual Wealth Creation Studies, explained simply.
#
# HONESTY CONTRACT: an entry ships ONLY after the study has been read COMPLETELY and its
# findings verified against the actual text (the early-era reading program, 2026-08-29 →).
# Nothing here is summarized from second-hand notes — every claim traces to the study itself.
# Entries are appended in the same change as each study's read completes (docs-as-code).
# Keys: study · years · pub · theme · says (plain-language findings) · prism (what the engine took).
# Pure data — zero Streamlit. Rendered by ui_reference.render_wcs_studies(); also emitted into the
# Markdown download. Pinned by tests/test_reference_tab.py.
# ═══════════════════════════════════════════════════════════════════════════════════════════

WCS_STUDIES = [
    {
        "study": "1st Study", "years": "1991–1996", "pub": "June 1996",
        "theme": "The Inquire 100 — where it all began",
        "says": (
            "The founding study. It picks the 100 companies that at least QUADRUPLED their market "
            "value in five years (32%+ a year) and asks what they have in common. The answers set "
            "the template for the next thirty years: speed comes from SMALL — 80 of the 100 started "
            "under ₹150 crore, while the giants gave safety but never topped the speed charts. "
            "87% stuck to ONE business instead of diversifying. Moderate sales growth (median just "
            "23%) was enough, because margins and asset productivity did the compounding. 68 of 100 "
            "shrank their debt or barely grew it. High tax-payers earned HIGHER P/Es — honest "
            "earnings get rewarded. And the crown jewel: in 1991, 63 of the 100 traded at a P/E far "
            "BELOW their ROE — by 1996 the P/E had caught up. The rule: a stock's fair P/E sits "
            "near its sustainable ROE, so buying well below that line carries a built-in margin of "
            "safety."
        ),
        "prism": (
            "The P/E-below-ROE margin-of-safety rule is implemented directly (the pe-vs-ROE spread "
            "and ratio signals). The small-cap-speed finding feeds the mid/small-cap compounder "
            "boost. ROCE-and-ROE rising over time — stated here first — became the ROCE-expansion "
            "thesis behind Economic Profit."
        ),
    },
    {
        "study": "2nd Study", "years": "1992–1997", "pub": "February 1998",
        "theme": "Swimming against the tide — wealth creation in a falling market",
        "says": (
            "The bear-market edition: the Sensex FELL 22% over this window, and only 45 companies "
            "qualified (versus ~300 the year before). The survivors shared three things. LEADERSHIP: "
            "30 of the 45 were #1 or #2 in their business — 'winners are passionate leaders'. "
            "IMPROVING ECONOMICS: average ROCE rose from 20.4% to 24.5% — a right business is not "
            "just superior, it 'gets better with time'. SELF-FUNDING: the first DuPont analysis in "
            "the series showed margins rising, asset turnover improving, and leverage FALLING every "
            "year — growth paid for internally. Blue chips jumped from 40% to 69% of the list: in "
            "bad markets, proven quality wins. The valuation lesson cuts the other way from the 1st "
            "study: 1992's P/Es sat far ABOVE ROE (wealth creators at 44× vs 22% ROE) and were "
            "crushed back to parity — 'prices accorded relatively higher in comparison to long-term "
            "ROE are clearly unsustainable'. Its first-ever Wealth DILUTORS table shows the wreckage: "
            "companies at P/E 100+ with single-digit ROE destroyed thousands of crores."
        ),
        "prism": (
            "The Wealth Dilutors table is the empirical ancestor of the valuation-multiple trap "
            "(high P/E on low ROE gets a valuation-score penalty). The DuPont decomposition is "
            "implemented as the ROE-attribution signals. Blue-chip resilience in downturns echoes "
            "in the cyclicality tiers and regime detection."
        ),
    },
    {
        "study": "3rd Study", "years": "1993–1998", "pub": "January 1999",
        "theme": "Creators and Destroyers — the first two-sided study",
        "says": (
            "Two firsts. FIRST: information technology enters the wealth-creator list (Satyam #1 at "
            "87% a year, Wipro #2) — the study that caught the IT wave before the boom. SECOND: it "
            "ranks the top 50 Wealth DESTROYERS alongside the 100 creators, and the contrast is the "
            "lesson. Destroyers are asset-heavy commodity businesses (76% of them) making "
            "undifferentiated intermediate products with no pricing power — and they destroyed not "
            "just existing capital but ₹127 billion of FRESH capital raised on the way down. "
            "Both groups were focused, so focus alone is not enough: the NATURE of the business "
            "decides. Creators' average ROCE rose again (16% to 20% — third study in a row), and "
            "their earnings were remarkably steady (ROCE varied only 6% around its mean). The "
            "valuation chapter states the identity Price/Book = ROE × P/E — 'P/E multiples are a "
            "function of ROE' — and introduces the first ROE/P-E matrix: companies moving from "
            "high-ROE-low-P/E to high-ROE-high-P/E returned 49% a year, 'the safest investment "
            "strategy', while good businesses that stayed high-ROE 'tend to get better over time'."
        ),
        "prism": (
            "Third independent validation of the P/E-below-ROE margin-of-safety signals (the "
            "high-ROE-low-P/E quadrant was the safest 49%-a-year cohort). The destroyer profile — "
            "asset-heavy, undifferentiated, fed by fresh capital — is what the Atoms/Bits design "
            "labels, the external-financing red flag, and the dilution checks screen against. The "
            "earnings-steadiness finding is the earliest ancestor of the Consistents framework."
        ),
    },
    {
        "study": "4th Study", "years": "1994–1999", "pub": "February 2000",
        "theme": "The birth of PEG — growth, bought at the right price",
        "says": (
            "The study where GROWTH takes the throne, published one month before the dot-com peak. "
            "Nine of the ten fastest wealth creators are IT companies (Satyam at 137% a year, "
            "Wipro at 123%). Companies growing earnings above 25% a year produced 76% of all "
            "wealth created — and of only 117 companies in the whole market that grew that fast, "
            "65 made the wealth-creator list. But the deeper lesson is about PRICE: this is the "
            "series' first PEG study. Infosys bought at PEG 0.4 returned 102% a year; Hindustan "
            "Lever — a superb business — bought at PEG 1.46 returned only 24%. Splitting the 100 "
            "at the median PEG of 0.83: low-PEG stocks returned 45% a year with a 68% hit rate, "
            "high-PEG stocks just 13%. 'No P/E multiple can be regarded as high or low unless it "
            "is measured relative to the stock's underlying earnings growth.' The study ends with "
            "the series' first explicit screen: earnings growth above 25%, ROE/ROCE above 25%, "
            "PEG well below 1.0 — and a prescient warning that New-Economy valuations were "
            "'questionable'. Also the fourth straight ROCE expansion, and 61% of high-ROCE "
            "companies stayed high: good economics 'tend to get better with time'."
        ),
        "prism": (
            "The origin of every PEG signal — the PEG zones, the QGLP price gate, and the PEG≤1 "
            "anchor all trace here. Its growth-quality-price trinity (growth + ROE + PEG) is the "
            "embryo of QGLP itself, and 'sustainability of growth' foreshadows the CAP/GAP "
            "longevity framework. The Wealth Creator badge's ancestry runs straight through this "
            "study's 25/25/1 screen."
        ),
    },
    {
        "study": "5th Study", "years": "1995–2000", "pub": "February 2001",
        "theme": "Multi-baggers and the payback ratio — anatomy of a 100%-a-year stock",
        "says": (
            "The bubble-era study, written just after the dot-com top. All ten of the fastest "
            "creators compounded above 100% a year (SSI 195%, Wipro 194% — Wipro alone made a "
            "quarter of ALL wealth created), and 80% of the group's five-year wealth arrived in "
            "the final year — after LOSING money for the first two ('it is easier to tell what "
            "will happen to the price of a stock than how much time will elapse', Fisher). The "
            "study dissects what a multi-bagger looks like BEFORE it runs: a genuine growth story "
            "(top-10 profit growth ~63% a year); a long competitive advantage period (the series' "
            "first Mauboussin citation); a very large opportunity relative to a SMALL starting "
            "size (mean entry value just ₹233 crore — 'focus on the price of companies, not the "
            "price of shares'); bought out of favour, with low trading volume and low "
            "institutional holding; run by outstanding, honest management. Then the sober math: "
            "86% of the top-10 wealth came from P/E re-rating, only 14% from earnings. And its "
            "signature tool — the PAYBACK RATIO: market value today divided by the next five "
            "years' profits. Stocks bought below 1× payback returned 73% a year on average and "
            "produced 6 of the 10 multi-baggers; every higher band earned progressively less."
        ),
        "prism": (
            "The verified origin of the payback-ratio family — the payback signals and their "
            "bands (≤1× → 73.1% mean return) implement this study's exact definition and table. "
            "The multi-bagger checklist is the earliest blueprint of SQGLP: small and unknown, "
            "quality management, growth, long CAP, price with re-rating room. The low-institutional-"
            "ownership entry condition survives in the SEPA institutional signal and SQGLP's "
            "Size pillar."
        ),
    },
    {
        "study": "6th Study", "years": "1996–2001", "pub": "January 2002",
        "theme": "The Five Forces of Wealth Creation — high ROE meets margin of safety",
        "says": (
            "The post-crash study: only 71 companies qualified, and while the group fell 71% from "
            "the bubble peak it still beat the flat Sensex by 427%. Its backdrop is stark — India "
            "Inc's ROE collapsed from 14% to 7% over the five years while the creators held ~22%. "
            "The study builds its Five Forces: (1) HIGH ROE well above the cost of capital — the "
            "famous table shows just 12 of 71 companies with ROE above 35% created 50% of ALL the "
            "wealth ('they are few, but necessary in the portfolio'); (2) high ROE with INCREASING "
            "capital employed — the 'lethal combination', because big wealth needs big capital "
            "deployed at high incremental returns; (3) GROWTH, but only in the same franchise — "
            "the value formula C × (RoC − G)/(R − G) says growth where returns sit below the cost "
            "of capital destroys value; (4) the COST OF CAPITAL itself — falling interest rates "
            "lift all valuations (the Dow's 1964-81 vs 1981-94 story); (5) MARGIN OF SAFETY via "
            "the payback ratio, now refined with the ROE cross-table: payback under 1× with ROE "
            "above 35% returned 74% a year (Infosys, Wipro and Satyam all sat exactly there), and "
            "'higher the ROE, longer can be the payback'. The study also splits intrinsic value "
            "into asset replacement value, earning power value, and growth value — growth counting "
            "only where a competitive advantage already exists."
        ),
        "prism": (
            "The verified source of the ROE-elite signal (the 12-companies-50%-of-wealth table is "
            "real). The value formula seeds the growth-value trap and Economic Profit doctrine — "
            "growth below the cost of capital destroys value — sixteen years before the 23rd "
            "formalized it. The payback-under-2 conclusion and the ROE-scaled payback tolerance "
            "inform the payback tier signals; the three-components-of-value split parallels the "
            "Earnings Power Box."
        ),
    },
    {
        "study": "7th Study", "years": "1997–2002", "pub": "January 2003",
        "theme": "Interest rates rule everything — India 2002 as America 1981",
        "says": (
            "The macro study. Its centrepiece retells Buffett's famous Fortune analysis — the Dow "
            "at 874 in 1964, 875 in 1981, then 9,181 by 1998, explained almost entirely by "
            "interest rates tripling and then collapsing, with investors 'habitually guided by "
            "the rear view mirror' — and argues India in 2002 IS America in 1981: rates falling "
            "from 14% to 6%, interest the second-largest cost in corporate India (bigger than "
            "labour), and a 2% rate drop worth ~20% on India Inc's profits. A stock is a "
            "'disguised bond' whose coupon is retained earnings reinvested at the ROE — which is "
            "why equities beat bonds over long tenures. The business analysis adds two new "
            "lenses: a CAPITAL ALLOCATION table (Wipro created ₹19 of market value per rupee of "
            "capital employed; across four studies, companies that RAISED their ROCE made 56-91% "
            "of all wealth — 'better capital allocation has been consistently rewarded'), and an "
            "AGE analysis (most wealth comes from companies under 30 years old). Speed still "
            "lives in small size: sub-₹250-crore companies compounded at 49% versus 15% for the "
            "billion-club, and Wipro itself started the period at just ₹573 crore. Commodity "
            "share hit its highest ever (27%) — even the focused-business rule gets its caveat: "
            "'total focus may not be a sufficient condition, but is a necessary one'."
        ),
        "prism": (
            "The ROCE-improvement cross-table (55 of 85 raising ROCE made 84% of the wealth) is "
            "the strongest multi-study formalization of the ROCE-expansion doctrine behind the "
            "ROCE-trajectory and Economic Profit signals. The capital-allocation lens feeds the "
            "capital-misallocation and Outsiders-style checks. Its appendix also settled an "
            "engine citation: the claim that the top-7 fastest all had payback below 0.5× is "
            "false (only 4 of 7 did) — the payback tiers now rest on the 5th and 6th studies' "
            "verified band tables instead."
        ),
    },
    {
        "study": "8th Study", "years": "1998–2003", "pub": "January 2004",
        "theme": "Transitory vs enduring multibaggers — and when the market breeds them",
        "says": (
            "The study that separates the two kinds of multibagger. TRANSITORY ones ride a cycle "
            "or a fad with questionable management, attract the crowd, and then correct almost "
            "100% — the 115 stocks that trebled in the 1998–2000 boom mostly destroyed as much "
            "wealth coming down as they made going up (Pentamedia fell from ₹65 billion to ₹2.5 "
            "billion). ENDURING ones — Infosys, HDFC Bank, Cipla — are run by quality "
            "managements, look expensive at purchase, and keep their gains. The recipe is "
            "Buffett's trinity: a good BUSINESS whose improvement is permanent, not cyclical; "
            "good MANAGEMENT (judged, via Mauboussin, on leadership, drive, and capital "
            "allocation — HDFC Bank grew net worth 687% while Global Trust Bank destroyed 99%); "
            "and an underpriced ENTRY ('overpriced stocks have no chance of becoming "
            "multibaggers'). Its timing tool: the ratio of the market's EARNINGS YIELD to the "
            "G-Sec yield — at 0.13 in 1992 (Sensex P/E 70×) only 3 stocks trebled in two years; "
            "at 1.53 in March 2003 (P/E under 10× with rates at 6%), 734 did. Tweedy Browne's "
            "value pointers checked out in India too: the cheapest P/B and P/E quartiles beat "
            "the falling market, the dearest lost most. Commodity share of the creators hit 46% "
            "— the cyclical wave arriving. 'Market folly contributes enormously to the creation "
            "of mega multibaggers.'"
        ),
        "prism": (
            "The verified source of the earnings-yield-versus-G-Sec signal (the ratio that "
            "flagged March 2003 as the multibagger nursery). The transitory-vs-enduring split is "
            "the doctrine behind the quality gates, cyclicality tiers, and consistency checks — "
            "why the engine scores durable economics over cyclical bounce. The Tweedy Browne "
            "quartile evidence validates the low-P/E and low-P/B legs of the value signals, and "
            "the management test feeds the capital-allocation checks."
        ),
    },
    {
        "study": "9th Study", "years": "1999–2004", "pub": "January 2005",
        "theme": "Back to basics — the era of commodities",
        "says": (
            "The commodities study. For the first time, the biggest wealth creators are commodity "
            "companies (ONGC, Reliance, IOC top the size list), and the study explains why: after "
            "years of starved capacity, utilisation tightened, prices squeezed upward, and profits "
            "went exponential. The durable findings restate the series' core laws with fresh "
            "numbers: ROCE EXPANSION is the number-one wealth driver — 66 of the 100 creators "
            "increased ROCE, accounting for 75% of the wealth created. Earnings growth above 25% "
            "drove about two-thirds of the wealth — while two-thirds of the wealth came from "
            "companies growing SALES below 20%, proving margins and capital productivity, not "
            "revenue, do the compounding. Small caps were three times faster than large (mean "
            "speed 63% under ₹2.5 billion versus 28% above ₹10 billion). And the creators entered "
            "the period at a P/E near 11 versus the Sensex at 27 — the margin-of-safety entry, "
            "again. Focused companies were 96% of the group. The macro half — the commodity-cycle "
            "timing call — proved right for the 2003-07 bull run."
        ),
        "prism": (
            "Pure validation of the engine's core: the ROCE-expansion finding is the doctrine "
            "behind the ROCE-trajectory and Economic Profit signals, the moderate-sales/high-"
            "earnings result supports scoring earnings quality over revenue growth, and the "
            "small-cap speed tables feed the size tiers. The cycle-timing half is deliberately "
            "NOT screened per-stock — sector capital-phase and regime detection carry its spirit."
        ),
    },
    {
        "study": "10th Study", "years": "2000–2005", "pub": "December 2005",
        "theme": "Consistent Wealth Creators — the resilient few, bought below median value",
        "says": (
            "The anniversary study, and the revenge of the old economy: exactly ONE technology "
            "company made the creators' list (fad investing at 2000's prices was 'the single "
            "biggest source of wealth destruction' — Wipro was simultaneously the #1 wealth "
            "DESTROYER of 2000-05 and a top-3 wealth creator of the full 15 years, bought-price "
            "being the whole difference). Commodities peaked at 51% of the creators, and the "
            "study made the contrarian call that technology 'could rule the roost during the "
            "next five years'. Its famous screen: for a doubler every two years, look for "
            "P/B below 1, P/E below 10, or P/Sales below 0.5 — the cheapest buckets compounded "
            "37-44% a year. The heart is the new 15-year lens, THE MOST CONSISTENT: Hero Honda "
            "and Ranbaxy appeared in all ten studies (40.8% and 34.7% a year with dividends "
            "reinvested). Nine of the top-10 consistents are consumer-facing, five are pharma, "
            "ALL are non-cyclical segment leaders with high returns on net worth. Their "
            "rolling-return record: ~90% odds of positive 5-year returns, 87% odds of beating "
            "the Sensex — and bought BELOW their own long-term median P/E, the win rate goes to "
            "nearly 100% with average absolute returns of 45.7%. 'Price is what you pay, value "
            "is what you get.'"
        ),
        "prism": (
            "The direct ancestor of the Consistents framework (the 27th study turned this "
            "theme's appearance-count into testable profit criteria the engine implements). The "
            "doubler screen is the origin of the deep-value formulas the WCS score carries "
            "(P/E<10, P/B<1 legs verified against the 13th). The buy-below-median-valuation "
            "entry rule lives on in the valuation-percentile and mean-reversion signals; "
            "'leaders in non-cyclical businesses with high return on net worth' is the "
            "cyclicality-tier plus quality-gate combination in one sentence."
        ),
    },
    {
        "study": "11th Study", "years": "2001–2006", "pub": "January 2007",
        "theme": "Terms of Trade — bargaining power you can read off the balance sheet",
        "says": (
            "TERMS OF TRADE is the ratio of a company's DEBTORS to its CREDITORS — customers "
            "versus suppliers, expressed as a percentage, and LOWER is better. Under 100% your "
            "suppliers are financing you; over it, you are financing your customers. It decides "
            "returns because capital employed is fixed plus working capital and almost nothing "
            "can be done about the fixed half: favourable terms can push working capital to zero "
            "or negative, lifting RoCE without one extra rupee of profit. Adverse terms do the "
            "reverse, and a company that cannot fund itself from operations is funded by banks "
            "and short-term lenders instead — who then set the terms. THE EVIDENCE: of the 85 "
            "non-banking wealth creators, 70 had favourable terms and carried 81% of the wealth "
            "created at 33.2% CAGR, against 29.3% for the 15 adverse names. The mirror is "
            "sharper — 37 of the 50 wealth DESTROYERS had adverse terms, accounting for 84.8% of "
            "all value destroyed. And 76 of the 85 converted profit into positive operating "
            "cash: earning it and collecting it are two different tests. Eleven creators broke "
            "the rule, and every one except Ranbaxy earned a high RoCE — software exporters "
            "collect slowly but carry margins big enough to pay for it. Ranbaxy is the "
            "counter-case: RoCE collapsed by the terminal year and its wealth creation slowed to "
            "11% CAGR. Adverse terms are survivable ONLY when margins cover them. Finally the "
            "trait is STICKY — across 429 BSE-500 companies, 205 of 243 favourable names stayed "
            "favourable and 124 of 186 adverse stayed adverse; only ~100 ever switched sides."
        ),
        "prism": (
            "Implemented as `terms_of_trade_spread` (days-payable − days-receivable) with "
            "`favorable_terms_flag` firing when DSO < DPO. A DOCUMENTED DEVIATION: the study's "
            "exact metric is the debtors/creditors ratio; the engine uses the directionally-"
            "equivalent days form, which the source CSVs supply directly (high DPO + low DSO = "
            "favourable terms). Both go NaN for financials, where days-payable is nulled "
            "upstream — the study excluded banks from this cut too. This read also settled a "
            "provenance question: the Buffett value-creation-ratio concept the engine uses is "
            "Buffett's own, not from this study — the false stamp was corrected."
        ),
    },
    {
        "study": "12th Study", "years": "2002–2007", "pub": "December 2007",
        "theme": "The Next Trillion Dollar opportunity — a macro interlude",
        "says": (
            "Where the Next-Trillion-Dollar thesis is born — the 14th study later builds its "
            "Winner Categories on this. India had just crossed US$1 trillion of GDP in 2007, and "
            "the study reads the shape of that journey rather than just the milestone. The first "
            "25 years to 2002 compounded nominal GDP at only 6.2% a year, reaching just under "
            "US$0.5 trillion. The following FIVE years more than doubled it to US$1 trillion, a "
            "16% CAGR. So: thirty years for the first trillion, and a projected five for the "
            "second. The arithmetic of a large base is the point — at roughly the same growth "
            "RATE, the GDP added over the next five years exceeds everything added in the "
            "previous THIRTY, and is double the last five. Per capita follows: US$897 in 2007 to "
            "a projected US$1,657 by 2012, a 12.8% CAGR, against just 2.4% through the earlier "
            "decades. The consequences the study draws for investors: the NTD era should bring "
            "distinctly buoyant corporate profits and a boom in savings and investment. Its "
            "valuation stance is deliberately two-sided and worth keeping — at prevailing prices "
            "the margin of safety was LOW, yet very high liquidity can hold markets at rich "
            "valuations for quite some time, so being right about value does not make you right "
            "about timing. And the line that generalises past the macro call: BARGAINS ARE FOUND "
            "WHERE MARKETS ARE BLIND to a large business opportunity, a positive change, or "
            "sustained growth — while LOSSES ARE GUARANTEED when one grossly overpays. The first "
            "half is a search instruction; the second is the only certainty in the sentence."
        ),
        "prism": (
            "Substrate only — no framework to implement. Its main engine relevance is negative: "
            "a compounder flag's claimed 12th-study origin was checked against the full text "
            "and found fabricated (the study says nothing about reinvestment rates), so the "
            "attribution was corrected while the logic — a sound quantitative construct — "
            "stayed."
        ),
    },
    {
        "study": "13th Study", "years": "2003–2008", "pub": "December 2008",
        "theme": "The Great, the Good and the Gruesome",
        "says": (
            "Buffett's three-way sort of all businesses, from his 2007 letter, applied to India "
            "with hard thresholds. The measure is ADJUSTED RoE — strip income from cash "
            "equivalents out of PAT, strip excess cash out of net worth, then divide — so a "
            "cash-heavy balance sheet cannot flatter the ratio. GREAT: a 10-year average "
            "adjusted RoE above 25%, never below 15% in ANY of those ten years, and a RISING "
            "trend. GRUESOME: a 10-year average below 10%. GOOD: everything in between, "
            "typically 15-25% RoE on a 20-30% payout — banking, steel and engineering, which "
            "must plough back earnings for fixed and working capital. Great companies need an "
            "enduring moat, and the study allows only two sources: lowest-cost production, or "
            "powerful brands. They grow SLOWER than the other two types but consume little extra "
            "capital, becoming cash machines with rising RoE and high payouts — Buffett's point "
            "being that even without organic growth such a business is rewarding, since you take "
            "its lush earnings and buy similar businesses with them. GOOD companies must "
            "reinvest heavily to keep growing ('put-up-more-to-earn-more'), which caps both "
            "return ratios and payouts. GRUESOME is the trap, and the trap is GROWTH: these "
            "often post the HIGHEST growth rates, swallow enormous capital, and earn nothing. "
            "Buffett's airlines — durable advantage elusive since the Wright brothers, capital "
            "demand insatiable ever since, 'investors have poured money into a bottomless pit, "
            "ATTRACTED BY GROWTH WHEN THEY SHOULD HAVE BEEN REPELLED BY IT.' Two subtleties "
            "worth keeping: Great businesses have LOW dependence on management greatness while "
            "Gruesome ones have high — the moat, not the manager, does the work; and RoE is a "
            "LAG indicator, the outcome of qualitative traits rather than their cause, so it "
            "tells you which bucket a company is in, not which it is heading toward."
        ),
        "prism": (
            "Implemented twice: a book-faithful Great/Gruesome screen using the study's exact "
            "ROE thresholds, and the Corporate Class taxonomy that drives a 50% score haircut "
            "for Gruesome and a boost for Great — the latter deliberately uses ROCE with "
            "reasoned thresholds, a documented deviation. The sure-shot formulas are the WCS "
            "deep-value score. The 25%-Great bar was tightened to the book's own number after "
            "this study's audit."
        ),
    },
    {
        "study": "14th Study", "years": "2004–2009", "pub": "December 2009",
        "theme": "Winner Categories, Category Winners — the three-layer funnel",
        "says": (
            "A three-step funnel, stated by the study as an equation chain: WINNER CATEGORIES = "
            "India's Next-Trillion-Dollar opportunity + scalability; CATEGORY WINNERS = winner "
            "categories + entry barriers + great management; WINNING INVESTMENTS = category "
            "winners + a reasonable valuation. Each step adds one filter, and the last one is "
            "price — a great company in a great category is still not an investment at any "
            "price. The NTD framing is the study's engine. India took more than 60 years after "
            "independence to reach its first US$1 trillion of GDP in FY08; growing nominally at "
            "12-15% it adds the SECOND trillion in five to six years — FY10 nominal GDP of "
            "US$1.3 trillion compounding at 12.4% (7% real plus 5% inflation) reaches US$2.3 "
            "trillion by FY15. China is the precedent and the warning about pace: its first "
            "trillion took until 1998, the second about six years to 2005, the THIRD just three "
            "years, and the fourth arrived in a single year — 2008, at US$4 trillion. India is "
            "read as mimicking that curve with roughly a decade's lag. The definition that makes "
            "'winner category' testable rather than rhetorical: a category expected to grow 18%+ "
            "a year, i.e. at least 1.5x the ~12% nominal GDP assumption. On the demand side the "
            "study leans on McKinsey's 'The Bird of Gold': India's aggregate consumption "
            "QUADRUPLES between 2005 and 2025, lifting it from the world's 12th largest consuming "
            "market to its 5th. The investing consequence is that category selection comes first "
            "— being right about the company inside a category growing below GDP is not enough."
        ),
        "prism": (
            "Implemented as the category-winner flag (top-tier ROCE within the sector plus "
            "sustained revenue growth — the leadership proxy), the winner-category flag "
            "(sector revenue growth at least 1.5× the universe median, the study's own "
            "GDP-multiple rule made regime-robust), and their intersection — the rare 'winning "
            "investment' cell that fires on about 4-5% of the universe."
        ),
    },
    {
        "study": "15th Study", "years": "2005–2010", "pub": "December 2010",
        "theme": "UU Investing — profiting from the Unknown and Unknowable",
        "says": (
            "The epistemology study. It takes Ralph Gomory's 1995 split of knowledge into three "
            "states and maps each onto investing. KNOWN, or RISK — outcomes AND probabilities "
            "both specified, which is Frank Knight's 1921 definition; an insurer's claims "
            "distribution. UNKNOWN, or UNCERTAINTY — events specified, probabilities not: "
            "terrorist attacks, systemic financial risk, and STOCK PRICES. UNKNOWABLE, or "
            "IGNORANCE — even the events cannot be identified in advance, being discontinuous "
            "with no precedent and no model, so a new model must be conceived after the fact; "
            "the long-term future of a sunrise industry or a start-up. The conclusion is "
            "uncomfortable: RISK — what every textbook and optimiser is built for — is 'not "
            "very relevant' to stock markets, since almost no outcome has a known probability. "
            "UNCERTAINTY is what every investor actually faces. The UNKNOWABLE is the domain "
            "'profitably exploited by intelligent investors' — the edge lives exactly where the "
            "distribution cannot be computed, which is why it stays available. The "
            "practical consequence: portfolio optimisation alone is insufficient equipment, and "
            "decision theory plus a conjectured distribution has to carry the weight. Its worked "
            "case was Indian banking projected to 2020. The study also carries the PAYBACK RATIO "
            "— current market cap divided by the estimated profits of the next five years — and "
            "reports it with unusual candour. Sorted by 2005 payback, the seven companies under "
            "0.5x went on to compound at 61% a year and the 28 between 0.5x and 1x at 36%, "
            "against 23%, 28% and 21% for the bands above 1x. Its verdict: a payback under 1x "
            "'continues to remain a reliable indicator of significantly superior returns' — WHILE "
            "THE OTHER THREE ELEMENTS of its own multi-bagger formula FAILED the test in this "
            "study. A series willing to publish which of its own rules stopped working is worth "
            "reading closely."
        ),
        "prism": (
            "The payback-ratio family implements this study's exact formula (market cap over "
            "five years of growth-projected profits), with the tier flags grounded in its "
            "verified band ladder. The UU setup flag combines the study's ingredients — small "
            "size, sub-1 payback, and an ROE turnaround — into one screen for the "
            "unknowable-but-cheap cell."
        ),
    },
    {
        "study": "16th Study", "years": "2006–2011", "pub": "December 2011",
        "theme": "Blue Chip investing — dividends as the quality and timing signal",
        "says": (
            "Blue Chip Investing, and the study's real subject is the DIVIDEND PAYOUT RATIO — "
            "not the dividend yield. That distinction is the finding: yields are almost "
            "homogenous across the market, with 66% of the top 100 wealth creators sitting "
            "below a 1.5% base yield, so yield cannot separate anything. Payout can, and it "
            "carries two strong correlations the study measures across roughly 2,100 listed "
            "companies — payout correlates highly and positively with RoE, and P/E correlates "
            "highly and positively with payout, rising from about 10x in the lowest payout band "
            "to 28x in the highest. Why a high payout signals quality rather than a lack of "
            "ideas: a company able to pay most of its profit out is (1) intrinsically profitable "
            "enough that it need retain very little to fund growth, (2) run by managers willing "
            "to share economic benefit with minority shareholders, and (3) far less exposed to "
            "MISALLOCATION of retained earnings into unrelated diversification or risky overseas "
            "acquisitions. The combination that matters is high payout WITH growth — either "
            "alone is ordinary. The study also flags a structural rise in payout ratios as a "
            "coming source of P/E re-rating. On consistency, this was the first study in which "
            "more than ten companies appeared in the top 100 across ten consecutive studies; "
            "Kotak Mahindra Bank led on the 47% ten-year price CAGR tie-break, and HDFC Bank had "
            "by then delivered 30% PAT growth for 38 consecutive quarters. Consumer-facing "
            "businesses dominate the consistent list — the same conclusion the series reaches "
            "from several directions."
        ),
        "prism": (
            "Implemented as the blue-chip quality flag covering five of Weiss's six screens "
            "(the institutional-investor COUNT is not in the data — a documented omission) "
            "plus the synthetic dividend-yield buy signal. The whole framework is dividend-"
            "fed, which is why it stayed dark until the payout-ratio column was fixed at the "
            "source — it self-revived to 90 passers the day the data healed."
        ),
    },
    {
        "study": "17th Study", "years": "2007–2012", "pub": "December 2012",
        "theme": "Economic Moat — the fountainhead of wealth creation",
        "says": (
            "The moat study. An Economic Moat protects profits from competitive attack, and the "
            "reason one is REQUIRED is mechanical: capital always chases returns, so a highly "
            "profitable business without a deep enough moat will see rivals drag its returns "
            "down to the cost of capital or below. Buffett's warning from his 2007 letter frames "
            "it — business history is full of 'Roman Candles', companies whose moats proved "
            "illusory and were soon crossed. The study widens the field beyond rivals, citing "
            "Joan Magretta on Porter: a company competes for profit with its CUSTOMERS (who want "
            "more for less), its SUPPLIERS (who want more for delivering less), SUBSTITUTES, and "
            "potential entrants whose mere threat caps what it can charge — the Five Forces. "
            "THE MOST USEFUL RESULT is what the backtest did NOT need. Testing Economic Moat "
            "Companies over 2002-2012, the study applied no criterion — past, present or future "
            "— other than the moat itself: no earnings forecast, no valuation call. EMCs still "
            "outperformed the benchmark. It calls this being 'earnings and valuations agnostic' "
            "and 'one of the most liberating conclusions' available to an investor, because "
            "judging staying power is a far easier call than forecasting growth and the multiple "
            "the market will pay for it. Two honest caveats ride with it: the claim is about a "
            "PORTFOLIO, not any single name — the study pre-empts critics by naming Hindustan "
            "Unilever underperforming for nearly 11 years from 1994 and Infosys for 10 years "
            "from its 2000 peak — and the reverse case is brutal. A BREACHED moat destroys "
            "wealth on a scale: Telecom was the third-largest wealth creator and four years "
            "later led the destruction table, supplying 4 of the top 10 destroyers, RCom alone "
            "losing ₹677bn at −28% CAGR. The clean contrast is Hero MotoCorp against TVS — same "
            "industry, same era, 56% average RoE against 14%, and 363% outperformance over ten "
            "years."
        ),
        "prism": (
            "Implemented as the Economic Moat Company flag — beating the sector average ROE "
            "across at least four of five available time windows, a faithful proxy for the "
            "study's six-of-eight-years persistence (with the study's own rule that peerless "
            "companies qualify on high absolute ROE). Feeds the Wide Moat framework and the "
            "moat axis throughout the app."
        ),
    },
    {
        "study": "18th Study", "years": "2008–2013", "pub": "December 2013",
        "theme": "Uncommon Profits — the emergence and endurance of Value Creators",
        "says": (
            "UNCOMMON PROFIT is return on capital held above the cost of capital, and the word "
            "carries two meanings at once: it DEFIES the economic law that competition drags "
            "returns down to the cost of capital, and the companies that manage it are simply "
            "not common. The study then measures exactly how uncommon, and this is the finding "
            "to remember: of the 2,200 companies listed in 2004, 568 earned an RoE above 15% — "
            "and only 86 sustained it for each of the next ten years. The attrition is visible "
            "year by year (568 → 433 → 357 → 305 → 259 → 171 → 143 → 125 → 99 → 86): high "
            "returns are not held, they LEAK. A single year's snapshot is just as skewed — "
            "across 3,300 listed companies in FY13, 2,697 (82%) earned under 15% RoE, and only "
            "76 cleared 25-30%. The study splits the problem in two. EMERGENCE — becoming a "
            "value creator at all — is rare, and its two probability-raisers are a strong "
            "corporate parent and a NON-cyclical business. ENDURANCE — staying one — is "
            "threatened by three things: disruptive innovation or competition, major regulatory "
            "change, and capital misallocation. Note that two of those three are self-inflicted "
            "or structural rather than bad luck. A separate finding lands hard on ownership: "
            "state-owned companies' share of wealth creation COLLAPSED from 51% in 2005 to 9% "
            "in 2013. The Munger line the study opens with is the whole thesis compressed — over "
            "20 to 30 years a business earning 18% on capital hands you a fine result even at an "
            "expensive-looking entry price, while a 6% business held 40 years returns about 6% "
            "no matter how large the discount you bought it at."
        ),
        "prism": (
            "Implemented as the emerging-value-creator flag: current ROE at or above 15% while "
            "the five-year median sits below it — the first-time crossing, exactly as defined, "
            "deliberately WAITING for the actual crossing per the study's own pitfall warning. "
            "The 15% hurdle here is the study's own number, distinct from the 12% cost of "
            "equity used system-wide."
        ),
    },
    {
        "study": "19th Study", "years": "2009–2014", "pub": "December 2014",
        "theme": "100x — the power of growth, and the birth of SQGLP",
        "says": (
            "The birth of SQGLP, and the series' 100-bagger study. It takes its frame from "
            "Thomas Phelps' 1972 book '100 To 1 In The Stock Market', which found 365 US stocks "
            "that rose 100-fold or more across the 40 years ending 1971. Both halves of '100x "
            "over time' matter: the precise multiple is less the point than what it forces you "
            "to confront about compounding — 7% post-tax fixed income reinvested for 20 years "
            "grows 4x, but at 7% inflation that is ZERO gain in purchasing power. THE FIVE "
            "ELEMENTS, stated as a checklist: S — SIZE: small and relatively "
            "unknown, on sales AND market cap, with low analyst coverage, low institutional "
            "holding and low traded volume. Q — QUALITY: of business (a large existing or "
            "potential profit pool, a favourable competitive landscape, scope for "
            "above-cost-of-capital returns) and of management (unquestionable integrity, "
            "demonstrable competence, a growth mindset). G — GROWTH: the MULTIPLICATIVE "
            "interplay of volume, selling price and margin. L — LONGEVITY: assess the "
            "competitive advantage period and test whether growth is reverting to mean. P — "
            "PRICE: enough room left for a re-rating. Size is not sentiment — sorted by base "
            "market cap, the smallest bucket compounded fastest, and the study's summary is "
            "'small is big in Wealth Creation', with only two routes to above-average returns: "
            "large but UNPOPULAR (rare, needing a bear market or a temporary earnings dip) and "
            "small but HIGH-GROWTH (common, and the growth investor's hunting ground). Growth is "
            "likewise measured, not asserted: price CAGR climbs monotonically with earnings — "
            "25%, 28%, 38%, 42%, 46% across PAT-growth bands from under 10% to over 40%. Two "
            "warnings close it. Phil Fisher's weighting — management 90%, industry 9%, all else "
            "1%. And the study's own caution that QUALITY DOES NOT GUARANTEE GROWTH. Phelps has "
            "the last word: vision to see them, courage to buy them, patience to hold them — "
            "and patience is the rarest of the three."
        ),
        "prism": (
            "Implemented as the SQGLP score (numeric proxies for all five pillars), the "
            "Century Stock flag (4+ of 5 pillars), the SQGLP engine bonus, and the 100x "
            "candidate screen. This study's audit also corrected a wrong-study attribution: "
            "a secondary 100x screen citing the 17th actually belongs here, and its numeric "
            "gates are engine constructions, now labeled as such."
        ),
    },
    {
        "study": "20th Study", "years": "2010–2015", "pub": "December 2015",
        "theme": "Mid-to-Mega — the MQGLP journey up the market-cap ranks",
        "says": (
            "The study that makes MARKET-CAP RANK the unit of analysis rather than absolute size. "
            "It sorts the market into three categories by rank — MEGA is a rank inside the top "
            "100, MID is 101-300, MINI is beyond 300 — which yields nine possible crossovers, of "
            "which only three matter to a buyer: Mini-to-Mega, Mini-to-Mid and MID-TO-MEGA. The "
            "study's answer is that Mid-to-Mega is the most profitable-cum-plausible crossover "
            "once adjusted for risk, and it reaches that from three directions: the performance "
            "of Mid-to-Mega portfolios from 2000 to 2015, stock-specific cases, and a 3x3 matrix "
            "of crossover returns against their probabilities. The resulting profile is unusually "
            "concrete — in any given year it should be possible to build a Mid-to-Mega portfolio "
            "of 9 to 12 stocks, bought at about 20% portfolio RoE and a 15-23x P/E, that "
            "delivers 20-35% PAT CAGR and 29-46% return CAGR over the following five years, for "
            "21-28% of alpha over the Sensex. Two structural observations support the thesis: "
            "dropouts from the Mega category have been FALLING over time (from the low 40s "
            "toward the mid-20s), and those that do drop out are increasingly replaced by "
            "Mid-to-Mega risers rather than by new listings — so the escalator is real and "
            "crowded from below. The framework is MQGLP: Mid-size, Quality, Growth, Longevity "
            "and Price — SQGLP with the size pillar re-aimed from 'small and unknown' to "
            "'mid-ranked and climbing'. Munger supplies the logic for stacking five conditions "
            "at once: really big effects, lollapalooza effects, come only from large "
            "combinations of factors. And one hard gate: INDUSTRY LEADERSHIP is a necessary "
            "pre-requisite to becoming a megacorp — no laggard makes the crossing."
        ),
        "prism": (
            "Implemented as the mid-to-mega candidate flag using ACTUAL market-cap ranks "
            "101-300 (an earlier absolute-rupee band proxy was measurably wrong — only 1 of "
            "34 candidates was truly Mid — and was fixed to the study's own rank definition), "
            "plus the value-migration flag for within-sector share capture."
        ),
    },
    {
        "study": "21st Study", "years": "2011–2016", "pub": "December 2016",
        "theme": "Focused investing — the power of allocation",
        "says": (
            "The series' first study on HOW MUCH to buy — the previous twenty were all about "
            "what to buy. Its opening exhibit makes the case in one table: take the SAME ten "
            "stocks, with returns running from +50% down to −40%, and vary only the weights. "
            "Three portfolios result — +5.0%, +18.5% and −8.5%. A 27-point spread produced by "
            "allocation alone, with stock selection held identical. The study notes how little "
            "literature exists on this, since allocation is usually waved away as art rather "
            "than science. Its framework is KELLY, reduced to three usable insights: (1) look "
            "for an ASYMMETRIC payoff, (2) create EDGE — Kelly pays you for the edge, not the "
            "excitement, and (3) when both are present, BET BIG. Because chances that satisfy "
            "all three come seldom, the strategy that follows is Focused Investing: few "
            "positions, sized properly, held. The four keys are named explicitly — a clear "
            "portfolio goal, superior stock selection, rational allocation, and ACTIVE "
            "MONITORING (concentration without monitoring is just exposure). The governing "
            "quote, from Concentrated Investing by Benello, Van Biema and Carlisle: 'bet "
            "seldom, and only when the odds are strongly in your favor, but when you do, bet "
            "big, hold for the long term, and control your downside risk.' The study's own "
            "framing of the prize is worth keeping: disciplined practice should produce "
            "EXCEPTIONAL returns rather than merely acceptable ones — allocation is what "
            "separates the two, because good selection alone lands you in the middle portfolio."
        ),
        "prism": (
            "Implemented as the Kelly-Minervini sizing module behind the SEPA Risk & "
            "Allocation cockpit: quarter-Kelly fractional sizing capped by the 1%-stop-risk "
            "rule, computing an executable capital weight and rupee deployment per stock. "
            "The study's 'bet big on asymmetry' is exactly why sizing scales with edge "
            "rather than being equal-weighted."
        ),
    },
    {
        "study": "22nd Study", "years": "2012–2017", "pub": "December 2017",
        "theme": "CAP & GAP — the power of longevity",
        "says": (
            "Longevity gets split into TWO clocks, and the study defines both against the same "
            "15% cost of equity. CAP — Competitive Advantage Period — is how long a company earns "
            "returns above its cost of capital, i.e. RoE > 15%; competitors are drawn by excess "
            "returns and eventually compete them away, so CAP measures the longevity of the MOAT. "
            "GAP — Growth Advantage Period — is how long profits grow faster than the benchmark, "
            "i.e. earnings growth > 15%. The study operationalises GAP strictly: successive years "
            "with a rolling 5-year PAT CAGR of 15% and no year's profit below the previous one. "
            "The two-line summary is the study's own: MOAT WITHOUT GROWTH WILL UNDERPERFORM; "
            "GROWTH WITHOUT MOAT WILL END SOON. How rare this is, measured: of the 613 companies "
            "listed throughout 1997-2017, only 87 achieved a Price CAGR above 25% and only 89 a "
            "Profit CAGR above 25% — while 292 of them (48%) compounded between 0 and 15%, and "
            "167 lost profit outright. Screening 223 companies with at least ₹5bn of PAT left 145 "
            "with a GAP of five years or more. The sharpest finding is a trade-off: LENGTH and "
            "HEIGHT of growth are INVERSELY correlated. Supernormal growth sustains only 5-6 "
            "years and belongs mostly to cyclicals, or to secular businesses early on when the "
            "base is low — high-CAP names are the secular compounders, high-GAP names the "
            "cyclical bursts. So catching height means catching it early: 13 of the top 15 "
            "high-GAP companies started with PAT under ₹1bn and 10 with market cap under ₹10bn. "
            "The three shared traits are a clear strategy, a high growth mindset and a "
            "high-growth industry — and two of those three are properties of management."
        ),
        "prism": (
            "Implemented as the CAP and GAP extended flags (returns above the hurdle across "
            "all windows; profit growth above 15% across all windows), the 0-4 CAP-GAP "
            "score, the longevity framework pill, and the Moat×Growth quadrant label the "
            "app shows everywhere — the study's own four names, verbatim."
        ),
    },
    {
        "study": "23rd Study", "years": "2013–2018", "pub": "November 2018",
        "theme": "Valuation insights — what works, what doesn't",
        "says": (
            "The study that goes after the P in QGLP. Intrinsic value has exactly two drivers — "
            "RoE and earnings growth — and value is created ONLY when RoE exceeds the cost of "
            "equity; below that line growth destroys value, because each extra rupee employed "
            "earns less than it costs. Hence a prescription that splits by starting point: "
            "LOW-RoE companies must raise RoE, HIGH-RoE companies must raise growth. It names "
            "the numerical triad of QGLP — RoE, PAT growth, P/E — notes these had been judged "
            "largely INDEPENDENTLY until now, and integrates them through a simplified "
            "Discounted Free Cash Flow to Equity model. THE HEADLINE RESULT is the PEG table. "
            "Sorting the 2013 wealth creators by PEG and measuring what they returned to 2018: "
            "the 23 companies under 0.5x compounded at 38% a year, the 26 between 0.5x and 1x "
            "at 28%, while the 1-1.5x, 1.5-2x and >3x buckets delivered 20%, 16% and 20%. "
            "Nearly HALF the wealth creators were under 1x in 2013, and they produced the "
            "highest returns — hence the study's claim that PEG below 1x is 'a near-infallible "
            "formula for healthy outperformance'. READ THE METHOD BEFORE TRUSTING IT: the PEG "
            "is trailing P/E divided by the NEXT five years' earnings CAGR, computed with "
            "perfect foresight (a 2013 P/E of 20x against 25% realised growth gives 0.8x), so "
            "it grades entry prices in hindsight rather than handing you a live screen. The "
            "study also traces valuation's five eras via Hagstrom — book value in the 1930s-40s, "
            "dividend yield in the 1950s, earnings growth in the 1960s, RoE and cash flow in "
            "the 1980s, and cash return on invested capital emerging — concluding there will "
            "never be a final word. Its own last word is blunter: OVERPAYING DOESN'T PAY."
        ),
        "prism": (
            "Implemented as the growth-value trap: growth at 15%+ while ROE sits below the "
            "12% cost of equity draws a direct composite penalty — the exact profile the "
            "study's table shows destroying value. The audit found this study's engine "
            "implementation completely clean: thresholds, worked example, and citations all "
            "verified exact."
        ),
    },
    {
        "study": "24th Study", "years": "2014–2019", "pub": "December 2019",
        "theme": "Management integrity — understanding sharp practices",
        "says": (
            "The forensic study. It opens on Phil Fisher's weighting — quoted in this series "
            "since the 19th study: 'in evaluating a common stock, the management is 90%, "
            "industry is 9%, and all other factors 1%' — so getting management integrity right "
            "is the critical FIRST step. Its companion line: "
            "there is only one way of writing honest accounts and infinite ways of manipulating "
            "them. The core mechanism is 'Credit P&L, Debit Balance Sheet' — inflate reported "
            "profit and stuff the resulting 'financial trash' into balance-sheet corners — which "
            "is why the study insists you juxtapose the CASH FLOW statement against the P&L: the "
            "P&L is the easier of the two to manipulate. Its two policy asks follow from that: "
            "managements should be STATUTORILY required to publish a simplified free-cash-flow "
            "statement, and auditors should be made accountable to minority shareholders. The "
            "stakes, measured: of 3,440 companies listed in Dec-2014, 594 were no longer listed "
            "by Dec-2019 and another 510 had fallen 70-100% — 1,104 in all, close to a third of "
            "the market. The Satyam case supplies the one checkable tell: the fraud was so well "
            "managed that almost no evidence trail existed, and in hindsight the single weak "
            "point was that OTHER INCOME was far too low for the cash balance claimed. Raju's "
            "confession put non-existent cash at ₹5,040 crore, and a quarter reported at ₹2,700 "
            "crore of revenue and a 24% margin was actually ₹2,112 crore at 3%. The closing "
            "instruction is behavioural, quoting Thomas Phelps: with tens of thousands of stocks "
            "available it is 'downright stupid' to buy one run by men of doubtful integrity — "
            "run at the FIRST hint. Talk to customers, employees, suppliers and competitors "
            "until you reach what The Investment Checklist calls the moment of integrity."
        ),
        "prism": (
            "The doctrine behind the forensic layer: the accrual red flags (profit not backed "
            "by cash), the CFO-to-profit checks, and the Schilit-style perimeter all screen "
            "for exactly the credit-P&L-debit-balance-sheet pattern. The SQGLP quality "
            "pillar's integrity gate (a red-flag ceiling) draws directly on this study."
        ),
    },
    {
        "study": "25th Study", "years": "2015–2020", "pub": "December 2020",
        "theme": "The QGLP checklist — 25 questions, 25 years",
        "says": (
            "The silver-jubilee study distils twenty-five years into the master framework, QGLP: "
            "Quality of business times quality of management; Growth in earnings; Longevity of "
            "both; and a reasonable Price. It is built as 25 QUESTIONS backed by 25 frameworks — "
            "is the business profitable, with favourable terms of trade and healthy margins? Does "
            "the DuPont decomposition show real return quality? Is there a moat, and will growth "
            "happen inside it? Is management able, honest, with skin in the game and sensible "
            "capital allocation? Is the price backed by a margin of safety? QGL tells you WHAT to "
            "buy; P tells you WHETHER to buy it now. The 25-year lookback is what gives the "
            "checklist its teeth. Of the top 500 companies listed in 1995, only 100 outperformed "
            "the benchmark over the following 25 years — sustained wealth creation is the "
            "exception, not the rule, and four in five of the era's leaders failed at it. What "
            "separated the survivors: 'stock returns are slaves of earnings power and growth — in "
            "the very long run, valuations matter LESS.' Infosys, the fastest, compounded 30% a "
            "year for 25 years into a 688x price multiple on 33% PAT growth; Reliance created the "
            "most absolute wealth at ₹6,307bn; Kotak Mahindra was the most consistent, appearing "
            "in 21 of the studies at 21% CAGR. Two closing warnings sit against each other: time "
            "is a friend of good companies and the ENEMY of bad ones, so holding is only a virtue "
            "if the business is; and over 50% of today's market cap comes from companies listed "
            "AFTER 1995, so the next twenty-five years will mostly be made by names not yet on "
            "the list."
        ),
        "prism": (
            "The QGLP framework implements the checklist's quantifiable core: Q from "
            "ROCE-rank plus promoter conduct, G from profit and EPS growth, L from ten-year "
            "ROE persistence, P from the PEG zone — with hard gates on each. Nearly every "
            "checklist question that data can answer maps to a live engine signal, from "
            "DuPont attribution to the red-flag risk ceiling; the purely qualitative "
            "questions (culture, succession) are the documented residue."
        ),
    },
    {
        "study": "26th Study", "years": "2016–2021", "pub": "December 2021",
        "theme": "Atoms to Bits — wealth creation in the digital era",
        "says": (
            "Value MIGRATES from ATOMS — physical, capital-heavy businesses that scale linearly "
            "with plants and inventory — to BITS: asset-light, near-zero-marginal-cost, "
            "network-effect businesses that replicate almost friction-free. The study calls the "
            "migration inevitable and shows it: of the current US top-10 by market cap, 7 are "
            "Bits. In its micro-economic model the Bits business earns a contribution of 780 "
            "against 150 for the Atoms one on the same demand curve. THE ACCOUNTING TRAP, and "
            "the study proves it with a worked example rather than asserting it: two startups, "
            "identical sales of 1,000 and identical opening equity of 500. The Atoms company "
            "capitalises 500 of capex and depreciates it at 50 a year, and reports a PROFIT of "
            "250. The Bits company expenses its entire 450 of software development immediately "
            "and reports a LOSS of 155 — yet both finish holding the same 300 of cash. The loss "
            "is OPTICAL: a self-generated intangible charged straight to the income statement. "
            "Hence 'cash flow is the leveler between Atoms and Bits financials', and P/E is "
            "meaningless for Bits. The study's answer is to average THREE valuations — DCF "
            "(hard, because cash flows stay negative for years), comparables with emphasis on "
            "PSG (Price/Sales to Growth), and the last private-equity round. Hyper-growth needs "
            "five things together: a large opportunity, genuine product-market fit, wide "
            "distribution, network effects and favourable unit economics. The call: expect "
            "Financials and Technology to lead wealth creation, and India is at the cusp of it."
        ),
        "prism": (
            "Implemented as the Atoms/Bits/Hybrid business-design label (a full sector "
            "mapping shown through the app) and the PSG ratio surfaced beside it on the "
            "tear-sheet — kept as a raw peer-relative number, deliberately without a "
            "threshold, because the study itself sets none."
        ),
    },
    {
        "study": "27th Study", "years": "2017–2022", "pub": "December 2022",
        "theme": "Consistents and Volatiles — execution is non-negotiable",
        "says": (
            "The study divides the ENTIRE corporate sector into exactly two types — CONSISTENTS "
            "and VOLATILES — and makes the split concrete with arithmetic instead of adjectives. "
            "A company is a Consistent only if it meets all THREE criteria on its PAT trend: "
            "(1) over 15 years, annual PAT must not fall by more than 10% on more than three "
            "occasions (twice if the window is 10 years); (2) NO single fall in PAT may exceed "
            "50%; and (3) the terminal year's PAT must not be lower than the initial year's. "
            "Everything that fails any of the three is a Volatile — there is no third bucket, "
            "and no benefit of the doubt. The verdict the study draws is blunt: CONSISTENCY IS "
            "THE SOURCE OF OUTPERFORMANCE, VOLATILITY THE SOURCE OF UNDERPERFORMANCE, and "
            "excellence in execution is NON-NEGOTIABLE for consistency — a Consistent is not a "
            "company in a kind industry, it is one that executed for fifteen years without a "
            "profit collapse. The method came from marrying two ideas: that the only defence "
            "against an uncertain future is a deeper reading of the certain past, and Howard "
            "Marks's point in Mastering the Market Cycle that while you cannot know where you "
            "are going, you can always know where you stand today. Two further conclusions: "
            "buying below MEDIAN valuations tilts the odds in the investor's favour, and the "
            "Financial sector, emerging from Covid, was expected to dominate wealth creation "
            "ahead. The study is candid that its own thresholds are 'not cast in stone' — "
            "readers may substitute their own, provided there is logic and math behind them."
        ),
        "prism": (
            "Implemented as the consistency champion flag with the study's three criteria "
            "proportionally scaled to the available five-year window, plus the volatile "
            "flag as its near-inverse. A substrate audit later hardened it with the "
            "'unverifiable is not passed' rule — a profit-to-loss collapse now counts as "
            "the greater-than-50% fall it is, instead of being skipped."
        ),
    },
    {
        "study": "28th Study", "years": "2018–2023", "pub": "December 2023",
        "theme": "Hockey-stick returns — the power of Economic Profit",
        "says": (
            "The study PRISM's wealth engine is built on. It starts from the identity PAT × P/E "
            "= market cap, so a hockey-stick share price must come from hockey-stick EARNINGS, a "
            "hockey-stick MULTIPLE, or both — Tata Elxsi compounded 65% a year over 2013-22 on "
            "37% profit growth and a P/E that went from 19x to 100x. It then argues accounting "
            "profit is the wrong yardstick because it ignores the equity employed to earn it. "
            "Exhibit 3 is the proof: Indian Oil earned 4x Nestle's profit on 56x the net worth — "
            "7% RoE against 96% — and carried the SMALLER market cap. ECONOMIC PROFIT fixes "
            "that. EP = Accounting Profit − an equity charge of Net Worth × cost of equity "
            "(the study uses a uniform 10%), which rearranges via RoE = PAT ÷ Net Worth into the "
            "form the engine uses: EP = NET WORTH × (RoE − CoE). Two determinants, nothing else. "
            "Mapping every company by EP produces the POWER CURVE, and its shape is the finding: "
            "in 2013 the top quintile of India Inc earned ₹1,394bn of economic profit while the "
            "bottom destroyed ₹830bn — and the middle three quintiles sat at just −41bn, 22bn "
            "and 106bn. Almost nothing happens in the middle; the tails are everything. What "
            "pays is ENDING in the top: measured by end-quintile across six ten-year windows, "
            "CAGRs run 24% / 21% / 10% / 8% / 4% against 11% for the Nifty 500 — and, in the "
            "study's own words, that holds 'no matter what the starting Quintile is'. Hence the "
            "hockey stick: the returns come from MOVING UP the curve, not from sitting still at "
            "the top. The framing is borrowed from McKinsey's 'Strategy Beyond the Hockey Stick' "
            "— its three transferable claims being that economic profit beats accounting profit, "
            "that every company maps somewhere on the curve, and that climbing it pays."
        ),
        "prism": (
            "Implemented end to end: the Economic Profit family on a consistent reserves "
            "basis, EP quintiles, the top-quintile quality bonus, the Hockey Stick pill "
            "(with the study's own P gate — calibrated to its 10.8% base rate), and the "
            "Approaching stage for negative-EP companies climbing with confirmation. This "
            "study's audits were the program's hardest teachers: a sign-flip on negative "
            "equity, sentinel fills, and a mixed cross-year basis were all found here and "
            "became standing engine-wide rules."
        ),
    },
    {
        "study": "29th Study", "years": "2019–2024", "pub": "December 2024",
        "theme": "Bruised Blue Chips — quality, fallen, bought near the lows",
        "says": (
            "Blue Chips are aspirational but usually richly valued, so the study asks when you "
            "actually get to own one — and answers: after it has been BRUISED. Noting that no "
            "textbook definition of a Blue Chip exists, it builds one: a listing history of at "
            "least 10 years, plus EITHER a top-50 market cap OR a top-250 market cap carrying a "
            "10-year average RoE of at least 20%. That yields 68 Blue Chips for FY2023-24. It is "
            "equally candid about the other half of the name: the phrase is borrowed from a "
            "booklet, 'How To Create Wealth "
            "Investing In Turnaround Stocks', which never defines 'bruised', so the threshold "
            "here is MOSL's own construction. A Bruised Blue Chip is one whose price at any "
            "point over a ten-year window fell 50% or more below its prior five-year high — "
            "operationally, take the low across FY15-24 and test it against the FY10-14 peak. "
            "FOUR reasons to want them: a 50%-plus correction is a golden opportunity to build "
            "LARGE positions in companies normally too expensive to accumulate; the returns are "
            "attractive; the payoff is asymmetric; and mortality is very low — the study's "
            "separate finding that the probability of PERMANENT loss of capital is low for this "
            "cohort is what makes the asymmetry real rather than hopeful. The caveat is stated "
            "in the same breath: this holds assuming no STRUCTURAL decline in fundamentals, "
            "which is the whole judgement. The process is three steps and deliberately patient: "
            "(1) build a watchlist first, before anything is cheap; (2) WAIT for a buying "
            "trigger, principally a sector tailwind or a change of management — a fallen price "
            "alone is not a trigger; and (3) buy at an attractive valuation, typically a "
            "Price/Book under 2x. Two findings from the window itself: wealth creation over "
            "these five years was the HIGHEST ever recorded by the series while destruction was "
            "among the lowest, and PSUs regained prominence in wealth creation after years of "
            "marginalisation."
        ),
        "prism": (
            "Implemented with the study's verbatim criteria: the blue-chip definition (top-50 "
            "or top-250 with ten-year ROE ≥20%), a bruised proxy from the 52-week-high "
            "distance, and the P/B<2 entry gate. It fires on almost nothing near market "
            "highs — verified as correct rarity, not a bug — and is built to activate in "
            "drawdowns, exactly as a contrarian watchlist should."
        ),
    },
    {
        "study": "30th Study", "years": "2020–2025", "pub": "December 2025",
        "theme": "The Multi-Trillion Dollar opportunity — sectors at the tipping point",
        "says": (
            "The series' thirtieth study, and the direct sequel to the 12th's Next-Trillion-Dollar "
            "thesis — the NTD era is declared over and the MULTI-TRILLION DOLLAR era, 2025 to "
            "2042, begins. The symmetry is the argument: India's GDP QUADRUPLED to USD 4 trillion "
            "over the last 17 years, and on a 17-year CAGR of 9% (which already embeds about 3% "
            "of rupee depreciation) it quadruples again to USD 16+ trillion by 2042. The study "
            "anticipates the objection — 9% sounds modest — and answers with the base: seventeen "
            "years out, GDP is USD 12 trillion HIGHER than today, because the same rate applied "
            "to a far larger base adds vastly more. Per capita follows, doubling roughly every "
            "nine years after allowing 1% population growth: USD 2,600 today, USD 5,200 by 2034, "
            "USD 10,400 by 2043. The consequence the study cares most about is the savings "
            "tsunami — cumulative gross domestic saving rising from USD 13.5 trillion across the "
            "NTD era to USD 47 trillion across the MTD era, capital that has to go somewhere. "
            "Where it goes is the investable call: FINANCIALS, explicitly including capital "
            "markets, and CONSUMER DISCRETIONARIES are expected to see explosive expansion as "
            "they hit their tipping point. Two framing claims close it — that there is no "
            "absolute upper limit to financial wealth, 'albeit with potholes on the way', and "
            "the more tactical view that LARGE CAPS are likely to outperform in the medium term, "
            "a notable inversion of the series' long-standing small-is-fast finding."
        ),
        "prism": (
            "Implemented as the sector-tailwind flag (the study's sector set, with the "
            "pattern fixed against the real data after generic tokens silently excluded BSE "
            "— the fastest wealth creator — from its own headline sector) and the Multi-"
            "Trillion tipping-point flag, which narrows to the specific sectors AND requires "
            "live inflection evidence: volume surge, earnings momentum, or a near-breakout "
            "price."
        ),
    },
]
