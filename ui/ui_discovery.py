"""PRISM — discovery filter cascade (the universe-level 'find stocks' UI).

STATEFUL BY DESIGN: this module OWNS the sidebar filter widgets and their session_state — it is
the deliberate counterpart to the STATELESS ui_tearsheet.py (which renders one stock and must
never hold widgets). The CLAUDE.md §5 stateless mandate binds ui_tearsheet, NOT this file. Do
not 'harmonize' this module to the stateless rule.
"""
import re

import numpy as np
import pandas as pd
import streamlit as st

from config import COLORS, FRAMEWORK_CATEGORIES, STALE_RESULT_DAYS


# ── Active-filter chips: registry + pure detector (unit-tested) ───────────────
# (sb_key, chip label, kind). kind drives BOTH the active-check and the label. MUST cover every key
# in the funnel's _active_n total (pinned by test) so no filter is ever active-but-chipless.
_CHIP_META = [
    ("sb_mcap", "Market Cap", "ms"), ("sb_sector", "Sector", "sel"), ("sb_industry", "Industry", "sel"),
    ("sb_cyc", "Cyclicality", "ms"), ("sb_capphase", "Capital Phase", "ms"), ("sb_tier", "Tier", "ms"),
    ("sb_verdict", "Soundness", "ms"), ("sb_wealthtier", "Wealth Tier", "ms"), ("sb_corpclass", "Corp Class", "ms"),
    ("sb_capalloc", "Capital Allocation", "ms"), ("sb_floatfunded", "Float-Funded", "bool"), ("sb_maxrf", "Max Red Flags", "max"),
    ("sb_piotier", "Piotroski", "ms"), ("sb_cashmach", "Cash Machine", "ms"), ("sb_mincov", "Min Coverage", "min"), ("sb_hidestale", "Hide Stale", "bool"),
    ("sb_fwfam", "FW Family", "ms"),
    ("sb_fw_exclude", "Exclude FW", "ms"), ("sb_fw_include", "Include FW", "ms"), ("sb_fw_combine", "Combine FW", "ms"),
    ("sb_moat", "Moat", "ms"), ("sb_peg_zone", "PEG Zone", "ms"), ("sb_buy_zone", "Buy Zone", "ms"),
    ("sb_weinstein", "Weinstein", "ms"), ("sb_lynchcat", "Lynch", "ms"), ("sb_mef", "Moat Endurance", "ms"),
    ("sb_cftri", "Cash-Flow Tri", "ms"), ("sb_smartflow", "Smart Money", "ms"),
    ("sb_catalyst", "Catalyst", "ms_count"), ("sb_sellalert", "Sell Alert", "ms_count"),
    ("sb_eppc", "EP Curve", "ms"), ("sb_epbox", "Earnings Box", "ms"), ("sb_mbsetup", "Multibagger", "ms_count"),
    ("sb_gate", "Gate-passed", "bool"), ("sb_minq", "Min Quality", "min"), ("sb_minscore", "Min Score", "min"),
]


def _compute_active_chips(state, rf_max):
    """Pure: (session_state mapping, red-flag ceiling) -> ordered [(key, label)] for each ACTIVE
    filter. Per-kind active-logic mirrors the funnel's _active_n so chips and badges always agree."""
    out = []
    for key, human, kind in _CHIP_META:
        v = state.get(key)
        if kind == "sel":
            if v not in (None, "All"):
                out.append((key, f"{human}: {v}"))
        elif kind == "max":
            if v is not None and v < rf_max:
                out.append((key, f"{human}: ≤{v}"))
        elif kind == "min":
            if v and v > 0:
                out.append((key, f"{human}: ≥{v}"))
        elif kind == "bool":
            if v:
                out.append((key, human))
        elif kind == "ms_count":
            if v:
                out.append((key, f"{human}: {len(v)} selected"))
        else:  # "ms" — human-readable values
            if v:
                out.append((key, f"{human}: {v[0]}" + (f" +{len(v) - 1}" if len(v) > 1 else "")))
    return out


def _family_count(frame: pd.DataFrame, fw_names) -> pd.Series:
    """Vectorized per-row count of how many of `fw_names` each stock passes, read from the
    comma-joined `frameworks_passed` string. Boundary-safe (the same anchored pattern as the
    Discovery framework filter) so 'Bruised Blue Chip' never substring-matches 'Bruised Blue Chip
    29'. The loop is over a fixed ~8-12 framework names building array ops — NOT a row loop.
    Module-level + pure so it is unit-tested (like _compute_active_chips). Returns an int Series."""
    if "frameworks_passed" not in frame.columns or not fw_names:
        return pd.Series(0, index=frame.index)
    s = frame["frameworks_passed"].fillna("")
    cnt = pd.Series(0, index=frame.index)
    for fw in fw_names:
        _pat = r"(?:^|, )" + re.escape(fw) + r"(?:,|$)"
        cnt = cnt + s.str.contains(_pat, regex=True, na=False).astype(int)
    return cnt


# ── ❔ Unknown option + zero-results culprit: pure substrate (unit-tested) ────
_UNKNOWN = "❔ Unknown"


def _blank_mask(series: pd.Series) -> pd.Series:
    """Rows where the engine deliberately declined to judge — NaN or blank/whitespace label.
    The SINGLE definition of 'unknown' shared by the option builder (_ordered_present), the
    facet counts (_ms_cascade) and the filter mask (_label_mask), so the three can never
    disagree on what ❔ Unknown means."""
    return series.isna() | (series.astype(str).str.strip() == "")


def _label_mask(series: pd.Series, sel) -> pd.Series:
    """Mask for a label multiselect whose options may include the ❔ Unknown sentinel. Real
    labels match by value — INCLUDING a literal '❔ Unknown' the engine itself emits (Weinstein
    Stage) — and the sentinel ADDITIONALLY claims the honest holes (NaN/blank), so 'Unknown'
    always means the full 'engine declined to judge' set regardless of how a column spells it."""
    m = series.isin(list(sel))
    if _UNKNOWN in sel:
        m = m | _blank_mask(series)
    return m


def _first_zero_filter(frame: pd.DataFrame, mask, label: str, zero_log: list) -> pd.DataFrame:
    """Apply one filter mask to the cascade frame, recording in `zero_log` the FIRST filter that
    empties a previously non-empty cascade — the funnel's zero-results state names it so the
    user knows exactly which dial to loosen. Pure + module-level so it is unit-tested; the
    sidebar binds it to its local log via the nested `_narrow` choke point."""
    out = frame[mask]
    if len(out) == 0 and len(frame) > 0 and not zero_log:
        zero_log.append(label)
    return out


def _ordered_present(frame: pd.DataFrame, col: str, order) -> list:
    """Labels present in frame[col], in canonical `order`; unknown labels appended last.

    BLANKS ARE NEVER OFFERED AS THEMSELVES (fixed 2026-08-23): the "no verdict from a data
    hole" pass makes an unknown label "" instead of fabricating a verdict, and dropna() does
    not strip the empty string — so 7 dropdowns briefly offered an unlabelled checkbox. Those
    honest holes now surface as a selectable ❔ Unknown option (appended LAST, and never
    duplicated when the engine already emits the literal label, e.g. Weinstein Stage)."""
    if col not in frame.columns:
        return []
    present = {v for v in frame[col].dropna().astype(str).unique() if v.strip()}
    opts = [v for v in order if v in present]
    opts += [v for v in sorted(present) if v not in opts]
    if _UNKNOWN not in opts and bool(_blank_mask(frame[col]).any()):
        opts.append(_UNKNOWN)
    return opts


_PICK_SEP = "  ·  "


def _fmt_pick(value, count) -> str:
    """The ONE label form for a cascade selectbox option: "<value>  ·  <live count>".

    Shared with canonical_pick below so the two cannot drift — change the separator here and the
    recoverer follows, which is the whole point of it being a constant rather than an f-string
    repeated at each call site.
    """
    return f"{value}{_PICK_SEP}{count}"


def canonical_pick(stored, domain, fallback="All"):
    """Recover a selectbox value that came back as its FORMATTED LABEL.

    MEASURED IN THE BROWSER 2026-09-21, from a user report. Picking Sector = Steel and THEN adding
    Market Category = Small Cap + Mid Cap returned zero stocks with the funnel blaming Sector,
    while 39 Steel small/mid-caps sat in the frame. A probe at the boundary showed
    st.session_state["sb_sector"] held 'Steel  ·  68' — the label Streamlit round-tripped back when
    the cascade changed it between reruns (68 -> 29 -> 39). Nothing in this module writes that key
    except the widget. keep_selected then appended the corrupt string as a legitimate option, so
    the widget never raised, and `frame["sector"] == "Steel  ·  68"` matched nothing: the guard
    built to stop silent WIDENING was preserving a silent ZEROING.

    THE DOMAIN MUST BE THE FULL, UNFILTERED SET — never the narrowed cascade frame. A real sector
    the cascade has narrowed out has to SURVIVE (it reads `· 0`, applies, and names itself as the
    culprit); validating against the narrowed frame would reset it to "All" and re-introduce the
    2026-09-02 widening bug this codebase already paid for once.

    Order matters: a legitimate value is returned untouched, so a sector whose real name somehow
    contained the separator is never mangled. Only an unrecognised string is split.
    """
    if stored in domain:
        return stored
    head = str(stored).split(_PICK_SEP)[0]
    return head if head in domain else fallback


def canonical_picks(stored, domain) -> list:
    """List form of canonical_pick, for the cascade's multiselects.

    An unrecoverable entry is DROPPED rather than replaced: a multiselect has no "All" sentinel,
    and empty already means "no filter", so substituting a fallback would invent a selection the
    user never made. Order is preserved (the chip row reads in pick order) and duplicates are
    collapsed, so a label whose raw value is already selected cannot appear twice.

    SCOPE — CORRECTED 2026-09-22. This covers the 19 widgets whose live count arrives via
    `count_col`, and this docstring originally claimed that closed the class. It did not: 8 more
    widgets carry their count through an explicit `format_func`, were never covered, and four of
    them crashed the app in production. Those are recovered by `recover_labels`; a structural test
    now requires every format_func cascade widget to pass its recovery domain.
    """
    out = []
    for v in (stored or []):
        r = canonical_pick(v, domain, fallback=None)
        if r is not None and r not in out:
            out.append(r)
    return out


# A cascade label's ONLY volatile part is its live count, in one of the two house shapes:
# "<value>  ·  <n>" (_fmt_pick / the count_col widgets) or "<name> (<n>)" (the flag OR-groups).
_COUNT_TAIL = re.compile(r"(?:\s*\(\d+\)|" + re.escape(_PICK_SEP) + r"\d+)$")


def _label_stem(label) -> str:
    """A display label minus its trailing live count — the part that is stable across reruns.
    Anchored and applied once, so a value like "💪 Strong (≥7)" keeps its own brackets."""
    return _COUNT_TAIL.sub("", str(label), count=1)


def recover_labels(stored, domain, format_func) -> list:
    """Map every stored entry back to a legitimate VALUE, for widgets whose live count arrives
    through an explicit `format_func` rather than `count_col`.

    PRODUCTION CRASH, 2026-09-22. canonical_picks covered the 19 count_col widgets and its
    docstring claimed that closed the class; it did not. Eight widgets carry their count through
    format_func instead — 🔥 Catalyst, 🚨 Sell Alerts, 🚀 Candidate Flags, 🎭 Framework Family, the
    three framework set-algebra pickers and Piotroski Strength — and none was covered. When Streamlit
    wrote a display label back into session_state, keep_selected offered it as an option and the
    four hard-lookup format_funcs (`_cat_name[c]` …) raised KeyError inside
    render_discovery_sidebar, taking the WHOLE APP down. The other four silently zeroed, and
    Exclude silently widened.

    The reverse map is built FROM format_func itself, so the recoverer cannot drift from the label
    it reverses (the same reason _fmt_pick and canonical_pick share _PICK_SEP). The domain must be
    the FULL set of legitimate values, never the narrowed live options: a value the cascade
    narrowed out has to survive, read 0, apply and name itself as culprit.

    RECOVER, DO NOT DROP, wherever a match exists — dropping a pick switches its filter OFF and
    WIDENS the result, the 2026-09-02 bug keep_selected exists to prevent. Only a genuinely
    unrecoverable entry is dropped (it cannot be applied either: there is no column to filter by).
    Order preserved, duplicates collapsed."""
    dom = list(domain)
    legit = set(dom)
    by_stem = {_label_stem(format_func(v)): v for v in dom}
    out = []
    for s in (stored or []):
        v = s if s in legit else by_stem.get(_label_stem(s))
        if v is not None and v not in out:
            out.append(v)
    return out


def keep_selected(options, stored) -> list:
    """Live option list + every stored pick the cascade has narrowed OUT of it, appended last.

    THE ONE RULE for every cascading filter on BOTH surfaces (the sidebar's _ms_cascade and its
    two selectboxes here; app.py's Market Pulse `_mp_ms` imports it): never silently drop what the
    user chose. A keyed Streamlit widget raises when its stored value is absent from `options`, and
    the first design answered that by PRUNING the stored selection to the live list — which
    switched the filter OFF and showed MORE stocks, not zero. Measured 2026-09-02: Wealth Tier=BUY★
    then Sector=Air Transport Service (a sector with no BUY★) returned the sector's 4 non-BUY★
    names where the honest answer is 0, the culprit banner never fired because the filter was
    never applied, and 106 of the cascade's ordered pairs could do it. Keeping the pick in the
    option list — after the live options, so canonical order is untouched — lets every count
    lookup render its honest 0, the filter apply, the result read 0, and the culprit name it.

    `stored` may be a list (multiselect) or a scalar (selectbox). Pure; unit-tested."""
    if stored is None:
        extra = []
    elif isinstance(stored, str) or not hasattr(stored, "__iter__"):
        extra = [stored]
    else:
        extra = list(stored)
    out = list(options)
    for v in extra:
        if v not in out:
            out.append(v)
    return out


def _remove_one_filter(key) -> None:
    """Chip ✕ callback — drop one filter (re-inits to its show-all default on the auto-rerun the
    button fires). Runs BEFORE any widget instantiates, so popping a widget-key is safe (no
    'cannot modify state after widget instantiated' error). Same delete-key mechanism as clear_all."""
    st.session_state.pop(key, None)


def clear_all_filters() -> None:
    """Delete every `sb_*` filter selection so the cascade resets to its show-all defaults.
    SINGLE SOURCE — the sidebar 'Clear all' button and both empty-state buttons all wire it via
    on_click, so the three can never diverge. Only `sb_*` keys are removed, so any other widget
    key (e.g. the Discovery tab's `disc_clear` button) is untouched.

    MUST BE AN on_click CALLBACK, NEVER an inline `if st.button(...)` call (fixed 2026-08-28 —
    the parked "Clear All sometimes doesn't clear" bug, finally mechanised): an inline call from
    a MAIN-BODY button runs AFTER the sidebar widgets instantiated that run, and deleting an
    instantiated widget's key lets the frontend RESURRECT its value on the rerun — the filters
    come back. Callbacks run before any widget instantiates, so the deletes stick. Invisible to
    AppTest (no frontend state to resurrect), which is why the bug never reproduced in tests —
    the call-site discipline is pinned structurally instead. No st.rerun() here: callbacks
    trigger their own rerun, and st.rerun() inside one is a no-op."""
    for _k in [k for k in st.session_state if k.startswith("sb_") and k != "sb_clear"]:
        del st.session_state[_k]


def render_discovery_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Build the cascade sidebar + the two Refine toggles, fill the live funnel, and return the
    fully-filtered frame `filt`. Self-contained: the funnel placeholder is created and filled
    inside this function (the old app.py main-body coupling is gone). The only input it reads is
    `df`; the only name it produces for the caller is the returned `filt`."""
    with st.sidebar:
        st.markdown(f"<div class='sec-head'>🎯 Filters</div>", unsafe_allow_html=True)

        # Live results funnel (filled in the main body once the final filtered count is known) +
        # one-click reset. The funnel is the heart of the "interconnected" feel — the count drops
        # in real time as filters narrow the universe.
        _funnel = st.empty()
        _chips_ph = st.empty()   # applied-filter chips; filled at END (post-prune, like _funnel)
        st.button("🧹 Clear all filters", key="sb_clear", width="stretch",
                  on_click=clear_all_filters)

        def _active_n(*keys):
            """Count filters in a group that ACTUALLY narrow. Every filter now defaults to its
            'show all' state — blank for multiselects, 'All' for the sector/industry selectboxes,
            unticked/0 for the toggles — so a filter is active iff it holds a non-default value."""
            n = 0
            for k in keys:
                v = st.session_state.get(k)
                if k in ("sb_sector", "sb_industry"):
                    n += 1 if (v and v != "All") else 0
                elif k == "sb_maxrf":
                    n += 1 if (v is not None and v < _RF_MAX) else 0   # show-all value is the max
                else:
                    n += 1 if v else 0          # blank multiselect / unticked / slider-0 = inactive
            return n

        def _grp(title, *keys, expanded=False):
            """An expander whose header carries a live active-filter count — and that stays OPEN
            whenever the group holds an active filter (expanded or a > 0).

            WHY: st.expander has no persistent open-state (no `key` in Streamlit 1.54), so it
            re-renders at its STATIC `expanded=` default on every rerun. Without this, selecting a
            filter (a rerun) or clicking Clear-all (st.rerun) snapped every expanded=False box shut —
            the box you were working in collapsed under you. Reading `_active_n` here (which sees the
            PRIOR run's persisted sb_* selections, before the widgets re-render) keeps any group you're
            actively using open across reruns. The body always renders either way (collapse is CSS),
            so filters keep applying regardless — this only governs the visual open/closed default."""
            a = _active_n(*keys)
            return st.expander(title + (f"  ·  {a}" if a else ""), expanded=(expanded or a > 0))

        # ── SMART INTERCONNECTED FILTER CASCADE ──────────────────────────────────
        # Every option-based filter narrows the OPTIONS of every filter BELOW it. A single
        # progressively-narrowed frame (_cf) drives all option lists AND the final filtered
        # dataframe — so what a dropdown shows is exactly what survives the filter (zero
        # drift between options and results).
        # Defensive: stored multiselect selections are pruned to the current valid options
        # before each widget renders, preventing Streamlit's "value not in options" crash.
        def _ms_cascade(label, options, key, default, help=None, format_func=None, count_col=None,
                        domain=None):
            """Cascade-safe multiselect. Fully manages session_state (no `default=` arg, which
            avoids Streamlit's default-plus-session-state warning). A stored pick the cascade has
            narrowed out is KEPT in the option list with its honest count of 0 (keep_selected) —
            never pruned, which would switch the filter off and widen the result. Empty selection
            = no filter.
            format_func lets the OPTION VALUES stay stable (so pruning holds) while the DISPLAY
            can vary per run (e.g. a live count) — never bake volatile text into the values.
            count_col: when given (and no explicit format_func), each option is auto-annotated with
            its LIVE count in the cascade frame `_cf` at THIS point — so the number reflects every
            filter above it (the 'smart faceted' feel). Counts live in the display only.
            domain: REQUIRED with an explicit format_func — the FULL set of legitimate values, used
            to recover a round-tripped display label (recover_labels). Pinned structurally."""
            # Seed by assignment BEFORE instantiating (the Steel resurrection fix), and keep every
            # stored pick — a value the cascade narrowed out is appended to the options so the
            # widget cannot raise and the filter still applies (reads 0, names itself as culprit).
            stored = list(st.session_state.get(key, default))
            # RECOVER A ROUND-TRIPPED LABEL before the selection is seeded, offered or applied.
            # The at-risk marker is a LIVE COUNT IN THE LABEL — a count moves as the cascade
            # narrows, which is what lets Streamlit write the DISPLAY string back into
            # session_state. It arrives two ways, and the 2026-09-21 fix covered only the first
            # (count_col) while claiming the whole class; the second (an explicit format_func)
            # crashed the app in production on 2026-09-22. Both branches below are required.
            # The domain is ALWAYS the full set, never `_cf` — a value the cascade has narrowed out
            # must SURVIVE (it reads 0, applies, and names itself as culprit); validating against
            # the narrowed frame would silently drop it and widen the result, the 2026-09-02 bug.
            # _UNKNOWN is a legitimate pick that appears in no column, so it joins the domain.
            if count_col is not None and count_col in df.columns:
                stored = canonical_picks(stored, {_UNKNOWN, *df[count_col].dropna().unique()})
            elif format_func is not None and domain is not None:
                stored = recover_labels(stored, domain, format_func)
            st.session_state[key] = stored
            options = keep_selected(options, stored)
            if format_func is None and count_col is not None and count_col in _cf.columns:
                _vc = _cf[count_col].astype(str).value_counts().to_dict()
                # ❔ Unknown facet count = the honest holes (NaN/blank) PLUS any literal
                # "❔ Unknown" the engine emits — the same union _label_mask filters by.
                _nb = int(_blank_mask(_cf[count_col]).sum())
                if _nb:
                    _vc[_UNKNOWN] = _vc.get(_UNKNOWN, 0) + _nb
                format_func = lambda v, _c=_vc: f"{v}  ·  {_c.get(str(v), 0)}"
            _kw = {"key": key, "help": help}
            if format_func is not None:
                _kw["format_func"] = format_func
            return st.multiselect(label, options, **_kw)

        _cf = df   # progressively-narrowed cascade frame — drives every option list below
        # Zero-results culprit log + the choke point EVERY filter application must go through
        # (pinned by test: no direct-subscript filter may appear). One funnel-point means the Unknown-mask
        # logic, the culprit naming, and any future per-filter behavior live in ONE place.
        _zero_at = []

        def _narrow(frame, mask, label):
            return _first_zero_filter(frame, mask, label, _zero_at)

        # Fixed slider ceiling for the 🛡️ Safety "Max red flags" dial (computed on the FULL df so the
        # slider range is stable across the cascade); _active_n treats this value as the show-all state.
        _RF_MAX = int(df["red_flag_count"].max()) if "red_flag_count" in df.columns else 28

        with _grp("🏢 Universe", "sb_mcap", "sb_sector", "sb_industry", "sb_cyc", "sb_capphase", expanded=False):
            # 1. Market Category — cascade root (only categories present in the data)
            _ALL_MCAPS = ["Mega Cap", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap", "Nano Cap"]
            _mcap_opts = _ordered_present(_cf, "market_category", _ALL_MCAPS)
            sel_mcap = _ms_cascade("Market Category", _mcap_opts, "sb_mcap", default=[],
                                   help="Blank = all market-cap tiers. Pick one or more to narrow.",
                                   count_col="market_category")
            if sel_mcap:
                _cf = _narrow(_cf, _label_mask(_cf["market_category"], sel_mcap), "Market Cap")

            # 2. Sector — only sectors within the chosen market categories
            # SEED-BEFORE-INSTANTIATE (2026-08-29, the user's Steel repro): deleting a widget
            # key does NOT clear the widget's frontend state — a selectbox with no session value
            # asks the frontend, which still remembers the old pick, and the cleared filter
            # RESURRECTS on the next rerun. _ms_cascade multiselects were always immune because
            # they unconditionally re-seed the key pre-instantiation; every non-multiselect
            # widget now does the same, which is what makes delete-based Clear-All authoritative.
            # KEPT, NOT RESET (2026-09-02): a stored sector the Market-Cap filter narrowed out used
            # to snap back to "All" — the selectbox form of the silent widening. It now stays in
            # the list (reading `· 0`), the filter applies, and the culprit names it.
            st.session_state.setdefault("sb_sector", "All")
            # RECOVER THE ROUND-TRIPPED LABEL (2026-09-21) before the value is read for anything.
            # Streamlit can write this selectbox's DISPLAY string back into session_state when the
            # cascade changes the label between reruns; keep_selected would then append the corrupt
            # string as a valid option and the filter would match zero rows while the sector sat
            # right there. Domain = the FULL df, so a legitimately narrowed-out pick still survives.
            st.session_state["sb_sector"] = canonical_pick(
                st.session_state["sb_sector"],
                {"All", *df["sector"].dropna().astype(str)})
            _sector_opts = keep_selected(["All"] + sorted(_cf["sector"].dropna().unique().tolist()),
                                         st.session_state["sb_sector"])
            _sec_vc = _cf["sector"].value_counts().to_dict()
            sel_sector = st.selectbox(
                "Sector", _sector_opts, key="sb_sector",
                format_func=lambda v: "All" if v == "All" else _fmt_pick(v, _sec_vc.get(v, 0)),
                help="Filter to one sector. 'All' = every sector; also narrows the Industry list below.",
            )
            if sel_sector != "All":
                _cf = _narrow(_cf, _cf["sector"] == sel_sector, "Sector")

            # 3. Industry — only industries within the chosen categories AND sector
            # Same keep-not-reset rule as Sector. Consequence worth knowing: after CHANGING the
            # sector, a previously chosen industry stays selected (reading `· 0`) and the result
            # is 0 with the culprit banner pointing at it — one ✕ click, instead of the old
            # silent snap to "All". One rule everywhere beats a special case here: 136 of 355
            # industries span more than one sector, so Industry is a facet, not a strict child.
            st.session_state.setdefault("sb_industry", "All")
            # Same recovery as Sector above — these two are the cascade's only selectboxes, and the
            # multiselects were measured immune (sel_mcap stayed clean through the same repro).
            st.session_state["sb_industry"] = canonical_pick(
                st.session_state["sb_industry"],
                {"All", *df["industry"].dropna().astype(str)})
            _industry_opts = keep_selected(["All"] + sorted(_cf["industry"].dropna().unique().tolist()),
                                           st.session_state["sb_industry"])
            _ind_vc = _cf["industry"].value_counts().to_dict()
            sel_industry = st.selectbox(
                "Industry", _industry_opts, key="sb_industry",
                format_func=lambda v: "All" if v == "All" else _fmt_pick(v, _ind_vc.get(v, 0)),
                help="Granular industry within the selected sector. Narrows with Sector above.",
            )
            if sel_industry != "All":
                _cf = _narrow(_cf, _cf["industry"] == sel_industry, "Industry")

            # 3b. Cyclicality Tier — structural business-type roll-up of industry (hold-vs-trade
            # "what": cyclicals are timing trades; defensives/structural-growers hold through cycle)
            _CYC_ORDER = ["Deep Cyclical / Commodity", "Cyclical", "Defensive",
                          "Sensitive / Structural-Growth", "Financials", "Catch-all"]
            _cyc_opts = _ordered_present(_cf, "cyclicality_tier", _CYC_ORDER)
            sel_cyc = _ms_cascade("Cyclicality Tier", _cyc_opts, "sb_cyc", default=[],
                                  help="Business-type by industry. Deep-Cyclical/Cyclical = timing "
                                       "trades; Defensive/Structural-Growth = hold through the cycle. Empty = all.",
                                  count_col="cyclicality_tier")
            if sel_cyc and "cyclicality_tier" in _cf.columns:
                _cf = _narrow(_cf, _label_mask(_cf["cyclicality_tier"], sel_cyc), "Cyclicality")

            # 3c. Sector Capital Phase — Chancellor capital cycle ("when": Starved = under-invested
            # supply opportunity; Hot = over-investing, mean-reversion risk)
            _CAPPHASE_ORDER = ["❄️ Capital Starved (opportunity)", "⚖️ Neutral", "🔥 Hot Capital (caution)"]
            _cap_opts = _ordered_present(_cf, "sector_capital_phase", _CAPPHASE_ORDER)
            sel_cap = _ms_cascade("Sector Capital Phase", _cap_opts, "sb_capphase", default=[],
                                  help="The sector's capital cycle. Starved = under-invested (supply "
                                       "opportunity); Hot = over-investing (mean-reversion risk). Empty = all.",
                                  count_col="sector_capital_phase")
            if sel_cap and "sector_capital_phase" in _cf.columns:
                _cf = _narrow(_cf, _label_mask(_cf["sector_capital_phase"], sel_cap), "Capital Phase")
            st.caption(f"→ {len(_cf):,} remaining")

        with _grp("🎯 Decision & Class", "sb_tier", "sb_verdict", "sb_wealthtier", "sb_corpclass", "sb_capalloc", expanded=True):
            # 4. Conviction Tier — only tiers present in the remaining stocks
            _tier_opts = sorted(int(t) for t in _cf["conviction_tier"].dropna().unique())
            sel_tier = _ms_cascade("Conviction Tier", _tier_opts, "sb_tier", default=[],
                                   help="Blank = all tiers. 1 = Crown Jewels, 2 = Strong … pick to narrow.",
                                   count_col="conviction_tier")
            if sel_tier:
                _cf = _narrow(_cf, _label_mask(_cf["conviction_tier"], sel_tier), "Tier")

            # 4b. Verdict — the engine's SOUND/MIXED/FLAWED gate (filter to the rare SOUNDs)
            _verdict_opts = _ordered_present(_cf, "verdict_direction", ["SOUND", "MIXED", "FLAWED"])
            sel_verdict = _ms_cascade("Soundness", _verdict_opts, "sb_verdict", default=[],
                                      help="The engine's top-line decision. Empty = all stocks.",
                                      count_col="verdict_direction")
            if sel_verdict and "verdict_direction" in _cf.columns:
                _cf = _narrow(_cf, _label_mask(_cf["verdict_direction"], sel_verdict), "Soundness")

            # 4b2. Wealth Tier — the change-lens verdict (verdict_engine.wealth_tier): is the
            # business BECOMING more valuable? Orthogonal to Verdict above by construction (the
            # engine's complete 18-stock BUY list splits across four wealth tiers) — so composing
            # the two is the point: Verdict=AVOID + Wealth=WATCH★ is the confirmed-turnaround
            # screen the engine cannot express alone. Price- and forensics-blind by design; pair
            # with Max Red Flags below for the clean version.
            _wt_opts = _ordered_present(_cf, "wealth_tier",
                                        ["BUY★", "BUY", "WATCH★", "WATCH", "AVOID", "N/A"])
            sel_wtier = _ms_cascade("Wealth Tier", _wt_opts, "sb_wealthtier", default=[],
                                    help="The wealth-engine tier (EP% · Vel% · margin tau): is it becoming "
                                         "more valuable? BUY★ = earning above cost of equity, improving, "
                                         "margins confirming; WATCH★ = confirmed turnaround. Price-blind — "
                                         "not a recommendation. Empty = all.",
                                    count_col="wealth_tier")
            if sel_wtier and "wealth_tier" in _cf.columns:
                _cf = _narrow(_cf, _label_mask(_cf["wealth_tier"], sel_wtier), "Wealth Tier")

            # 4c. Corporate Class — Motilal Oswal capital-allocation quality (Great / Good / Gruesome)
            _corp_opts = _ordered_present(_cf, "corporate_class", ["🏆 GREAT", "👍 GOOD", "💀 GRUESOME"])
            sel_corp = _ms_cascade("Corporate Class", _corp_opts, "sb_corpclass", default=[],
                                   help="Capital-allocation quality. 'Only Great', or exclude Gruesome.",
                                   count_col="corporate_class")
            if sel_corp and "corporate_class" in _cf.columns:
                _cf = _narrow(_cf, _label_mask(_cf["corporate_class"], sel_corp), "Corp Class")

            # 4d. Capital Allocation — the OUTSIDERS axis: is management RETURNING capital
            # (buybacks/dividends) or RAISING it (dilution)? Computed by the engine and shown in
            # All Data, but unscreenable until 2026-09-10. Admitted on measurement: 3 well-spread
            # values (Returning 22% · Neutral 70% · Raising 8%) and genuinely orthogonal — its
            # largest overlap with ANY existing filter is 34% (💰 Cash Machine). Sits beside
            # Corporate Class because both read management, but they are NOT the same read:
            # corporate_class grades capital-allocation QUALITY, this names its DIRECTION.
            _CAPALLOC_ORDER = ["💰 Returning Capital", "⚖️ Neutral", "⚠️ Raising Capital"]
            _capalloc_opts = _ordered_present(_cf, "capital_allocation_signal", _CAPALLOC_ORDER)
            sel_capalloc = _ms_cascade("Capital Allocation", _capalloc_opts, "sb_capalloc", default=[],
                                       help="What management does with the cash. 💰 Returning = buybacks/"
                                            "dividends (the Outsiders pattern); ⚠️ Raising = dilution. "
                                            "Empty = all.",
                                       count_col="capital_allocation_signal")
            if sel_capalloc and "capital_allocation_signal" in _cf.columns:
                _cf = _narrow(_cf, _label_mask(_cf["capital_allocation_signal"], sel_capalloc),
                              "Capital Allocation")
            st.caption(f"→ {len(_cf):,} remaining")

        with _grp("🛡️ Safety", "sb_maxrf", "sb_piotier", "sb_cashmach", "sb_mincov", "sb_hidestale", expanded=False):
            # Risk-control screen — the system's most defensible, validation-INDEPENDENT edge: avoiding
            # the zeros (Gensol et al.). Deliberately NOT here: a forensic_label dropdown (98.6% one value
            # → degenerate; red_flag_count is its live form) and an "exclude risk-flags" group
            # (pledge_rising / dilution_vampire / debt_restatement are 84–99% subsumed by red_flag_count≥3
            # — redundant per the orthogonality census).
            # 4d. Max red flags — cap forensic severity (0 = pristine; _RF_MAX = show all)
            st.session_state.setdefault("sb_maxrf", _RF_MAX)     # seed-before-instantiate
            sel_maxrf = st.slider("Max red flags", 0, _RF_MAX, _RF_MAX, key="sb_maxrf",
                                  help="Cap how many of the 28 forensic red flags a stock may carry. Max = all.")
            if sel_maxrf < _RF_MAX and "red_flag_count" in _cf.columns:
                _cf = _narrow(_cf, _cf["red_flag_count"] <= sel_maxrf, "Max Red Flags")

            # 4e. Piotroski Strength — financial-strength tier from the F-Score (derived inline, vectorized)
            _pf = _cf["piotroski_fscore"]
            _pio_tier = np.where(_pf >= 7, "💪 Strong (≥7)",
                                 np.where(_pf >= 4, "➖ Moderate (4–6)", "⚠️ Weak (≤3)"))
            _PIO_TIERS = ["💪 Strong (≥7)", "➖ Moderate (4–6)", "⚠️ Weak (≤3)"]
            _pio_opts = [t for t in _PIO_TIERS if t in set(_pio_tier)]
            _pio_vc = pd.Series(_pio_tier).value_counts().to_dict()   # derived array → explicit count
            sel_pio = _ms_cascade("Piotroski Strength", _pio_opts, "sb_piotier", default=[],
                                  help="Financial-strength tier (Piotroski F-Score). Empty = all.",
                                  format_func=lambda v: f"{v}  ·  {_pio_vc.get(v, 0)}",
                                  domain=_PIO_TIERS)
            if sel_pio:
                _cf = _narrow(_cf, pd.Series(_pio_tier, index=_cf.index).isin(sel_pio), "Piotroski")

            # 4e2. Cash Machine — accrual quality: are reported profits turning into CASH?
            # (data_engine:1105 — the O'Glove/accrual-anomaly tiering; the 34th filter, added
            # 2026-08-29 after the gap audit: balanced buckets 43/29/27%, max redundancy vs any
            # existing filter 38% Jaccard, and 📄 Paper Profits carries a median 8 red flags
            # against 💰 Cash Machine's 3 — the one axis no other filter names by CAUSE.)
            _CASHM_ORDER = ["💰 Cash Machine", "✅ Solid", "📄 Paper Profits"]
            _cashm_opts = _ordered_present(_cf, "cash_machine_label", _CASHM_ORDER)
            sel_cashm = _ms_cascade("Cash Machine", _cashm_opts, "sb_cashmach", default=[],
                                    help="Accrual quality — is the profit turning into cash? "
                                         "💰 = CFO backs the earnings; 📄 Paper Profits = it "
                                         "doesn't (median 8 red flags vs 3). Empty = all.",
                                    count_col="cash_machine_label")
            if sel_cashm and "cash_machine_label" in _cf.columns:
                _cf = _narrow(_cf, _label_mask(_cf["cash_machine_label"], sel_cashm), "Cash Machine")

            # 4f. Min data coverage % — don't trust a high score built on thin data
            st.session_state.setdefault("sb_mincov", 0)          # seed-before-instantiate
            sel_mincov = st.slider("Min data coverage %", 0, 100, 0, key="sb_mincov",
                                   help="Hide stocks whose score rests on below-this-% evidence coverage.")
            if sel_mincov > 0 and "data_coverage_pct" in _cf.columns:
                _cf = _narrow(_cf, _cf["data_coverage_pct"] >= sel_mincov, "Min Coverage")

            # 4g. Hide stale results — drop frozen filers (no result for STALE_RESULT_DAYS: a missed listing
            # deadline under any calendar, so an SME on its half-yearly calendar is never dropped)
            st.session_state.setdefault("sb_hidestale", False)   # seed-before-instantiate
            if st.checkbox("Hide stale results (no result in 8 months)", value=False, key="sb_hidestale",
                           help=f"Drops companies that have filed no result for over {STALE_RESULT_DAYS} days — "
                                f"past every listing deadline, quarterly or the SME half-yearly calendar — "
                                f"the filing freeze that often comes before distress.") \
                    and "result_stale_flag" in _cf.columns:
                _cf = _narrow(_cf, _cf["result_stale_flag"] == 0, "Hide Stale")
            st.caption(f"→ {len(_cf):,} remaining")

        # ── 3-TIER FRAMEWORK FILTER ENGINE ───────────────────────────────────────
        # Set Algebra: (Universe − Excluded) ∩ (Included ∪ ∅) ∩ (∀ Combined)
        # Evaluation order: Exclude first → Include second → Combination last.
        # Each panel's options cascade from the panel above — zero impossible combos.
        #
        # Panel 1 — Exclude (NOT): remove stocks passing ANY selected framework
        # Panel 2 — Include (OR) : show stocks passing ANY selected framework (current behavior)
        # Panel 3 — Combination (AND): keep only stocks passing ALL selected simultaneously

        def _extract_frameworks(frame):
            """Extract sorted unique framework names from the cascade frame.
            Splitter uses ', ' — the exact separator written by scoring_engine fw_str builder."""
            if "frameworks_passed" not in frame.columns:
                return []
            return sorted(set(
                fw.strip()
                for cell in frame["frameworks_passed"].dropna()
                if cell != "None"
                for fw in cell.split(", ")
                if fw.strip()
            ))

        def _fw_match_mask(frame, fw_list, logic="or"):
            """Build a boolean mask for stocks matching framework list.
            logic='or'  → stock passes ANY of the selected frameworks
            logic='and' → stock passes ALL of the selected frameworks simultaneously
            Regex is boundary-safe: prevents 'Bruised Blue Chip' from substring-matching
            'Bruised Blue Chip 29' via anchored exact-token pattern."""
            if not fw_list or "frameworks_passed" not in frame.columns:
                return pd.Series(True, index=frame.index)
            if logic == "or":
                mask = pd.Series(False, index=frame.index)
                for fw in fw_list:
                    _pat = r"(?:^|, )" + re.escape(fw) + r"(?:,|$)"
                    mask = mask | frame["frameworks_passed"].str.contains(_pat, regex=True, na=False)
                return mask
            else:  # logic == "and"
                mask = pd.Series(True, index=frame.index)
                for fw in fw_list:
                    _pat = r"(?:^|, )" + re.escape(fw) + r"(?:,|$)"
                    mask = mask & frame["frameworks_passed"].str.contains(_pat, regex=True, na=False)
                return mask

        def _fw_counts(frame):
            """name -> # stocks passing it (boundary-safe), to annotate each framework option with a
            live count. Loops the ~37 framework names building array ops — NOT a row loop."""
            if "frameworks_passed" not in frame.columns:
                return {}
            s = frame["frameworks_passed"].fillna("")
            return {fw: int(s.str.contains(r"(?:^|, )" + re.escape(fw) + r"(?:,|$)",
                                           regex=True, na=False).sum())
                    for fw in _extract_frameworks(frame)}

        with _grp("🧬 Frameworks", "sb_fwfam", "sb_fw_exclude", "sb_fw_include", "sb_fw_combine", expanded=False):
            st.caption("Set algebra: (Universe − Exclude) ∩ Include ∩ Combination")

            # 5·0 FRAMEWORK FAMILY — coarse conviction-CHARACTER filter (the 5 §7 families), applied
            # before the fine name-level set algebra below. Mirrors the category-count chips on the
            # cards: a stock must pass ≥ the chosen minimum frameworks in EACH selected family (AND
            # across families). Built on FRAMEWORK_CATEGORIES — the live taxonomy, not the dead reverse-map.
            _avail_fw = set(_extract_frameworks(_cf))
            _fam_present = [(e, lbl, clr, fws) for (e, lbl, clr, fws) in FRAMEWORK_CATEGORIES
                            if any(f in _avail_fw for f in fws)]
            # Over ALL families, not just the present ones: keep_selected can hand the widget a
            # family the cascade narrowed out, and its format_func / mask must still resolve it.
            _fam_meta = {lbl: (e, clr, fws) for (e, lbl, clr, fws) in FRAMEWORK_CATEGORIES}
            _fam_avail_n = {lbl: sum(1 for f in fws if f in _avail_fw)
                            for (e, lbl, clr, fws) in _fam_present}
            sel_fam = _ms_cascade(
                "🎭 Framework Family", [lbl for (e, lbl, clr, fws) in _fam_present], "sb_fwfam", default=[],
                help="Filter by conviction CHARACTER (Quality-Moat / Momentum / Value / MOSL / Fisher). "
                     "A stock must pass at least the minimum number of frameworks in EACH selected "
                     "family. The number in brackets is how many of that family's frameworks are still "
                     "available. Empty = all stocks.",
                format_func=lambda l: f"{_fam_meta[l][0]} {l} ({_fam_avail_n.get(l, 0)})",
                domain=list(_fam_meta),
            )
            if sel_fam:
                _fam_cap = max((_fam_avail_n.get(l, 1) for l in sel_fam), default=1)
                # Pre-seed + clamp the stored min BEFORE the widget instantiates (avoids the
                # value-and-session-state warning, and an out-of-range stored value when the cap shrinks).
                if "sb_fwfam_min" not in st.session_state:
                    st.session_state["sb_fwfam_min"] = 1
                st.session_state["sb_fwfam_min"] = min(max(1, int(st.session_state["sb_fwfam_min"])),
                                                       max(1, _fam_cap))
                _fam_min = st.number_input(
                    "Min frameworks per selected family", min_value=1, max_value=max(1, _fam_cap),
                    step=1, key="sb_fwfam_min",
                    help="Each selected family must contribute at least this many passed frameworks. "
                         "1 = passes any framework in the family; higher = stronger conviction in it.",
                )
                _fam_mask = pd.Series(True, index=_cf.index)
                for l in sel_fam:
                    _fam_mask = _fam_mask & (_family_count(_cf, _fam_meta[l][2]) >= int(_fam_min))
                _cf = _narrow(_cf, _fam_mask, "FW Family")
                st.markdown(
                    f'<div style="font-size:0.6rem;color:{COLORS["purple"]};padding:0 0 6px 2px;">'
                    f'🎭 {len(sel_fam)} famil{"y" if len(sel_fam) == 1 else "ies"} · ≥{int(_fam_min)} each '
                    f'· {len(_cf)} stocks remaining</div>',
                    unsafe_allow_html=True,
                )

            # Recovery domain for the three pickers below: every framework in the FULL frame, never
            # `_cf` — a framework the cascade narrowed out must survive recovery (recover_labels).
            _fw_domain = _extract_frameworks(df)

            # 5a. EXCLUDE Framework — NOT logic (applied first, narrows universe)
            _all_fw_excl = _extract_frameworks(_cf)
            _fwc_excl = _fw_counts(_cf)
            sel_fw_exclude = _ms_cascade(
                "🚫 Exclude Framework", _all_fw_excl, "sb_fw_exclude", default=[],
                help="Remove stocks passing ANY of these frameworks. Applied first.",
                format_func=lambda v, _c=_fwc_excl: f"{v}  ·  {_c.get(v, 0)}",
                domain=_fw_domain,
            )
            if sel_fw_exclude:
                _excl_mask = _fw_match_mask(_cf, sel_fw_exclude, logic="or")
                _cf = _narrow(_cf, ~_excl_mask, "Exclude FW")
            # Exclude badge
            _n_excl_fw = len(sel_fw_exclude)
            if _n_excl_fw > 0:
                st.markdown(
                    f'<div style="font-size:0.6rem;color:{COLORS["red"]};padding:0 0 6px 2px;"'
                    f'>⛔ {_n_excl_fw} excluded · {len(_cf)} stocks remaining</div>',
                    unsafe_allow_html=True,
                )

            # 5b. INCLUDE Framework — OR logic (show stocks passing ANY selected)
            _all_fw_incl = _extract_frameworks(_cf)
            _fwc_incl = _fw_counts(_cf)
            sel_fw_include = _ms_cascade(
                "✅ Include Framework", _all_fw_incl, "sb_fw_include", default=[],
                help="Show stocks passing ANY of these. Empty = all remaining stocks.",
                format_func=lambda v, _c=_fwc_incl: f"{v}  ·  {_c.get(v, 0)}",
                domain=_fw_domain,
            )
            if sel_fw_include:
                _incl_mask = _fw_match_mask(_cf, sel_fw_include, logic="or")
                _cf = _narrow(_cf, _incl_mask, "Include FW")
            # Include badge
            _n_incl_fw = len(sel_fw_include)
            if _n_incl_fw > 0:
                st.markdown(
                    f'<div style="font-size:0.6rem;color:{COLORS["green"]};padding:0 0 6px 2px;"'
                    f'>✅ {_n_incl_fw} included (OR) · {len(_cf)} stocks remaining</div>',
                    unsafe_allow_html=True,
                )

            # 5c. COMBINATION Framework — AND logic (stock must pass ALL selected)
            _all_fw_comb = _extract_frameworks(_cf)
            _fwc_comb = _fw_counts(_cf)
            sel_fw_combine = _ms_cascade(
                "🔗 Combination Framework", _all_fw_comb, "sb_fw_combine", default=[],
                help="Stock must pass ALL of these simultaneously. AND logic.",
                format_func=lambda v, _c=_fwc_comb: f"{v}  ·  {_c.get(v, 0)}",
                domain=_fw_domain,
            )
            if sel_fw_combine:
                _comb_mask = _fw_match_mask(_cf, sel_fw_combine, logic="and")
                _cf = _narrow(_cf, _comb_mask, "Combine FW")
            # Combination badge
            _n_comb_fw = len(sel_fw_combine)
            if _n_comb_fw > 0:
                st.markdown(
                    f'<div style="font-size:0.6rem;color:{COLORS["blue"]};padding:0 0 6px 2px;"'
                    f'>🔗 {_n_comb_fw} required (AND) · {len(_cf)} stocks remaining</div>',
                    unsafe_allow_html=True,
                )
            st.caption(f"→ {len(_cf):,} remaining")

        with _grp("💰 Moat · Value · Entry", "sb_moat", "sb_peg_zone", "sb_buy_zone", "sb_floatfunded", expanded=False):
            # 6. Moat-Growth quadrant — only quadrants present in the remaining stocks
            _MOAT_ORDER = ["⭐ Wealth Creator", "🛡️ Quality Trap", "⚡ Growth Trap", "💀 Wealth Destroyer"]
            _moat_opts = _ordered_present(_cf, "moat_growth_quad", _MOAT_ORDER)
            sel_moat = _ms_cascade("Moat", _moat_opts, "sb_moat", default=[],
                                   help="Moat-growth quadrant: Wealth Creator / Quality Trap / "
                                        "Growth Trap / Wealth Destroyer. Empty = all stocks.",
                                   count_col="moat_growth_quad")
            if sel_moat:
                _cf = _narrow(_cf, _label_mask(_cf["moat_growth_quad"], sel_moat), "Moat")

            # 7. PEG Zone — valuation tier (options present in remaining stocks, canonical order)
            _PEG_ZONE_ORDER = [
                "💎 Deep Value", "🟢 Fair PEG", "🟡 Stretched",
                "🟠 Expensive", "🔴 Overpriced", "🔴 Declining",
            ]
            _peg_opts = _ordered_present(_cf, "peg_zone", _PEG_ZONE_ORDER)
            sel_peg_zone = _ms_cascade("PEG Zone", _peg_opts, "sb_peg_zone", default=[],
                                       help="Valuation tier from the PEG ratio. Empty = all stocks.",
                                       count_col="peg_zone")
            if sel_peg_zone and "peg_zone" in _cf.columns:
                _cf = _narrow(_cf, _label_mask(_cf["peg_zone"], sel_peg_zone), "PEG Zone")

            # 8. Buy Zone — entry timing vs Volatility Stop (options present in remaining stocks)
            _BUY_ZONE_ORDER = [
                "🟢 Perfect Entry (Low Risk)", "🟡 Standard Zone",
                "🟠 Loose Entry Zone", "🔴 Extended (Wait for Pullback)",
                "🔻 Below Stop (Trend Broken)", "⚪ Uncharted",
            ]
            _buy_opts = _ordered_present(_cf, "buy_zone_label", _BUY_ZONE_ORDER)
            sel_buy_zone = _ms_cascade("Buy Zone", _buy_opts, "sb_buy_zone", default=[],
                                       help="Entry timing vs the Volatility Stop. Empty = all stocks.",
                                       count_col="buy_zone_label")
            if sel_buy_zone and "buy_zone_label" in _cf.columns:
                _cf = _narrow(_cf, _label_mask(_cf["buy_zone_label"], sel_buy_zone), "Buy Zone")

            # 8a. Float-funded — NEGATIVE working capital: customers and suppliers fund the
            # business, so growth needs no capital of its own (the Dhandho/float read, and why it
            # belongs with Moat rather than Safety — it is pricing power, not risk).
            # Admitted 2026-09-10 as the MOST orthogonal candidate in the All-Data gap sweep:
            # fires 11.8%, only 13% overlap with the nearest existing filter, and — decisively —
            # P(red_flag_count>=3 | flag) = 80%, EXACTLY the universe baseline. It is independent
            # of forensic severity, so Max-red-flags cannot express it; and it is an INCLUSION
            # screen, which a `<=` threshold could never be. (Its sibling candidates
            # dilution_vampire_flag and cyclical_mirage_flag were REJECTED on the same test at
            # 99% and 91% — those ARE slices of red_flag_count>=3.)
            st.session_state.setdefault("sb_floatfunded", False)   # seed-before-instantiate
            if st.checkbox("💧 Float-funded only (negative working capital)", value=False,
                           key="sb_floatfunded",
                           help="Show only businesses funded by their own customers/suppliers — "
                                "negative working capital. Growth costs them no capital.") \
                    and "negative_wc_flag" in _cf.columns:
                _cf = _narrow(_cf, _cf["negative_wc_flag"].fillna(0) == 1, "Float-Funded")
            st.caption(f"→ {len(_cf):,} remaining")

        with _grp("📈 Trend · Style · Flow", "sb_weinstein", "sb_lynchcat", "sb_mef",
                  "sb_cftri", "sb_smartflow", expanded=False):
            # 8b. Weinstein Stage — the 30-week-trend stage (Stage 2 = uptrend buy, Stage 4 = avoid)
            _WEIN_ORDER = ["📈 Stage 2 Advancing", "🔄 Stage 1 Basing", "⚠️ Stage 3 Top",
                           "📉 Stage 4 Declining", "❔ Unknown"]
            _wein_opts = _ordered_present(_cf, "weinstein_stage", _WEIN_ORDER)
            sel_wein = _ms_cascade("Weinstein Stage", _wein_opts, "sb_weinstein", default=[],
                                   help="Long-term price trend stage. Empty = all stocks.",
                                   count_col="weinstein_stage")
            if sel_wein and "weinstein_stage" in _cf.columns:
                _cf = _narrow(_cf, _label_mask(_cf["weinstein_stage"], sel_wein), "Weinstein")

            # 8c. Lynch Type — Peter Lynch's stock archetype
            _lynch_opts = _ordered_present(_cf, "lynch_category",
                                           ["Fast Grower", "Stalwart", "Slow Grower", "Declining"])
            sel_lynch = _ms_cascade("Lynch Type", _lynch_opts, "sb_lynchcat", default=[],
                                    help="Lynch's classification (Fast Grower / Stalwart / …). Empty = all.",
                                    count_col="lynch_category")
            if sel_lynch and "lynch_category" in _cf.columns:
                _cf = _narrow(_cf, _label_mask(_cf["lynch_category"], sel_lynch), "Lynch")

            # 8d. Moat Endurance — is the competitive advantage widening or eroding
            _mef_opts = _ordered_present(_cf, "mef_label",
                                         ["🟢 Expanding", "✅ Intact", "🟡 Eroding", "🔴 Degrading"])
            sel_mef = _ms_cascade("Moat Endurance", _mef_opts, "sb_mef", default=[],
                                  help="Whether the moat is widening or eroding over time. Empty = all.",
                                  count_col="mef_label")
            if sel_mef and "mef_label" in _cf.columns:
                _cf = _narrow(_cf, _label_mask(_cf["mef_label"], sel_mef), "Moat Endurance")

            # 8e. Cash-Flow Triangle — cash-flow quality pattern
            _cftri_opts = _ordered_present(_cf, "cf_triangle",
                                           ["✅ Self-Funding", "⚪ Mixed Pattern",
                                            "⚠️ Growth Phase — Watch D/E", "🚨 Debt Trap — Avoid"])
            sel_cftri = _ms_cascade("Cash-Flow Triangle", _cftri_opts, "sb_cftri", default=[],
                                    help="Operating/investing/financing cash-flow quality. Empty = all.",
                                    count_col="cf_triangle")
            if sel_cftri and "cf_triangle" in _cf.columns:
                _cf = _narrow(_cf, _label_mask(_cf["cf_triangle"], sel_cftri), "Cash-Flow Tri")

            # 8f. Smart-Money Flow — the 5-level institutional-flow read
            _smf_opts = _ordered_present(_cf, "smart_money_flow",
                                         ["🌊💎 Elite Accumulation", "🎯 Strong Accumulation",
                                          "✅ Moderate Interest", "⚪ Neutral", "❌ Distribution"])
            sel_smf = _ms_cascade("Smart-Money Flow", _smf_opts, "sb_smartflow", default=[],
                                  help="Institutional accumulation/distribution level. Empty = all.",
                                  count_col="smart_money_flow")
            if sel_smf and "smart_money_flow" in _cf.columns:
                _cf = _narrow(_cf, _label_mask(_cf["smart_money_flow"], sel_smf), "Smart Money")
            st.caption(f"→ {len(_cf):,} remaining")

        with _grp("🔥 Catalysts & Alerts", "sb_catalyst", "sb_sellalert", expanded=False):
            # 9. Catalyst — fast-moving EVENT triggers (debt repair, new capacity, margin inflection,
            # early institutional discovery, Lynch GARP). OR logic: show stocks with ANY selected catalyst.
            # Option VALUES are the stable cat_* column names (so the cascade pruning holds); the live
            # count is shown only via format_func. Only catalysts present in the remaining frame are listed.
            _CATALYSTS = {
                "🔥 Capacity Explosion": "cat_capacity",
                "🔥 OpLev Inflection":    "cat_oplev",
                "🔥 Deleveraging":         "cat_deleveraging",
                "🔥 Lynch Dream":          "cat_lynch_dream",
                "🔥 Inst Discovery":       "cat_inst_discovery",
                "🔥 Strong Quarter":       "cat_strong_quarter",
            }
            _cat_name = {v: k for k, v in _CATALYSTS.items()}
            _cat_opts = [c for c in _CATALYSTS.values() if c in _cf.columns and int(_cf[c].sum()) > 0]
            sel_catalyst = _ms_cascade(
                "🔥 Catalyst", _cat_opts, "sb_catalyst", default=[],
                help="Show stocks where ANY selected catalyst (a fast-moving change) is firing. OR logic. "
                     "Empty = all stocks. Count = stocks with that catalyst in the current filtered set.",
                format_func=lambda c: f"{_cat_name[c]} ({int(_cf[c].sum())})",
                # only columns that EXIST: recovery renders every domain value through format_func
                domain=[c for c in _CATALYSTS.values() if c in df.columns],
            )
            if sel_catalyst:
                _cat_mask = pd.Series(False, index=_cf.index)
                for _c in sel_catalyst:
                    _cat_mask = _cat_mask | (_cf[_c] == 1)
                _cf = _narrow(_cf, _cat_mask, "Catalyst")
                st.markdown(
                    f'<div style="font-size:0.6rem;color:{COLORS["orange"]};padding:0 0 6px 2px;"'
                    f'>🔥 {len(sel_catalyst)} catalyst(s) (OR) · {len(_cf)} stocks remaining</div>',
                    unsafe_allow_html=True,
                )

            # 10. Sell Alerts — Baid sell triggers (separate 0/1 cols). OR logic: show stocks with ANY
            # selected alert (surface risk to review). Same stable-column-name + format_func pattern as 🔥.
            _SELL_ALERTS = {
                "🚨 Cash Collapse":      "sell_alert_cash_collapse",
                "🚨 Overvalued":         "sell_alert_overvalued",
                "🚨 Thesis Broken":      "sell_alert_thesis_broken",
                "🚨 Treadmill":          "sell_alert_treadmill",
                "🚨 Sequential Decline": "sell_alert_sequential_decline",
                "🚨 Mgmt Deteriorated":  "sell_alert_mgmt_deteriorated",
            }
            _sa_name = {v: k for k, v in _SELL_ALERTS.items()}
            _sa_opts = [c for c in _SELL_ALERTS.values() if c in _cf.columns and int(_cf[c].sum()) > 0]
            sel_alert = _ms_cascade(
                "🚨 Sell Alerts", _sa_opts, "sb_sellalert", default=[],
                help="Show stocks where ANY selected Baid sell trigger is active. OR logic. Empty = all.",
                format_func=lambda c: f"{_sa_name[c]} ({int(_cf[c].sum())})",
                domain=[c for c in _SELL_ALERTS.values() if c in df.columns],
            )
            if sel_alert:
                _sa_mask = pd.Series(False, index=_cf.index)
                for _c in sel_alert:
                    _sa_mask = _sa_mask | (_cf[_c] == 1)
                _cf = _narrow(_cf, _sa_mask, "Sell Alert")
                st.markdown(
                    f'<div style="font-size:0.6rem;color:{COLORS["red"]};padding:0 0 6px 2px;"'
                    f'>🚨 {len(sel_alert)} alert(s) (OR) · {len(_cf)} stocks remaining</div>',
                    unsafe_allow_html=True,
                )
            st.caption(f"→ {len(_cf):,} remaining")

        # ── 🚀 Multibagger Setups — the engine's multibagger-thesis reads, finally filterable ──
        # A tool named for multibaggers should let you isolate its OWN candidates: the deepest
        # "creates economic value / is a 100x candidate" signals were computed + shown on the tearsheet
        # but never screenable. EP curve/box are categoricals (like cf_triangle); the candidate flags
        # are an OR-group (the same stable-column-name + live-count mechanic as 🔥 Catalysts).
        with _grp("🚀 Multibagger Setups", "sb_eppc", "sb_epbox", "sb_mbsetup", expanded=False):
            # EP Power Curve — economic-profit trajectory (the deepest "creates value" read)
            _EPPC_ORDER = ["🚀 Hockey Stick", "➖ EP Positive, Not Rising", "📈 Improving", "📉 Value Trap"]
            _eppc_opts = _ordered_present(_cf, "ep_power_curve", _EPPC_ORDER)
            sel_eppc = _ms_cascade(
                "EP Power Curve", _eppc_opts, "sb_eppc", default=[],
                help="Economic-profit trajectory. 🚀 Hockey Stick = positive AND accelerating economic "
                     "profit (the multibagger fundamental); 📉 Value Trap = destroying value. Empty = all.",
                count_col="ep_power_curve")
            if sel_eppc and "ep_power_curve" in _cf.columns:
                _cf = _narrow(_cf, _label_mask(_cf["ep_power_curve"], sel_eppc), "EP Curve")

            # Earnings Power Box — Heiserman quadrant (economic profit × free cash flow); the
            # standard builder so its honest holes surface as ❔ Unknown like every sibling
            _EPBOX_ORDER = ["📦 Earnings Power", "💰 Cash Cow", "🚀 Cash-Hungry Grower", "⚠️ Weakest"]
            _epbox_opts = _ordered_present(_cf, "earnings_power_box", _EPBOX_ORDER)
            sel_epbox = _ms_cascade(
                "Earnings Power Box", _epbox_opts, "sb_epbox", default=[],
                help="Heiserman's economic-profit × free-cash-flow quadrant. 📦 Earnings Power = both "
                     "positive (the elite); ⚠️ Weakest = neither. Empty = all.",
                count_col="earnings_power_box")
            if sel_epbox and "earnings_power_box" in _cf.columns:
                _cf = _narrow(_cf, _label_mask(_cf["earnings_power_box"], sel_epbox), "Earnings Box")

            # Multibagger-candidate flags — OR-group (stable column-name values + live-count format_func,
            # exactly like 🔥 Catalysts). Show stocks firing ANY selected candidate flag.
            _MULTIBAGGER = {
                "🐘 100x Candidate":       "mosl_100x_candidate",
                "🏅 Category Winner":      "category_winner_flag",
                "📈 Compound Growth":      "compound_growth_power_flag",
                "🛡️ Consistency Champion": "consistency_champion",
                "🔄 ROE Turnaround":       "roe_turnaround_flag",
                "🧲 Value Migration":      "value_migration_flag",
                "💎 Bruised Blue Chip":    "bruised_blue_chip",
                "🚀 Mid→Mega":             "mid_to_mega_candidate",
            }
            _mb_name = {v: k for k, v in _MULTIBAGGER.items()}
            _mb_opts = [c for c in _MULTIBAGGER.values()
                        if c in _cf.columns and int(_cf[c].fillna(0).sum()) > 0]
            sel_mb = _ms_cascade(
                "🚀 Candidate Flags", _mb_opts, "sb_mbsetup", default=[],
                help="Show stocks firing ANY selected multibagger-candidate flag (OR). e.g. 100x "
                     "Candidate (MOSL screen), Category Winner (best-in-industry), Compound Growth. "
                     "Empty = all. Count = stocks with that flag in the current filtered set.",
                format_func=lambda c: f"{_mb_name[c]} ({int(_cf[c].fillna(0).sum())})",
                domain=[c for c in _MULTIBAGGER.values() if c in df.columns])
            if sel_mb:
                _mb_mask = pd.Series(False, index=_cf.index)
                for _c in sel_mb:
                    _mb_mask = _mb_mask | (_cf[_c].fillna(0) == 1)
                _cf = _narrow(_cf, _mb_mask, "Multibagger")
                st.markdown(
                    f'<div style="font-size:0.6rem;color:{COLORS["purple"]};padding:0 0 6px 2px;">'
                    f'🚀 {len(sel_mb)} setup(s) (OR) · {len(_cf)} stocks remaining</div>',
                    unsafe_allow_html=True,
                )
            st.caption(f"→ {len(_cf):,} remaining")

        with _grp("🌊 Refine", "sb_gate", "sb_minq", "sb_minscore", expanded=False):
            # Final power-user knobs — now applied to the cascade frame (_cf) IN-GROUP, exactly like
            # every other group, so the funnel, this group's "·N" badge, and its "→ remaining"
            # caption all agree (no more last-caption-vs-funnel drift). All default OFF.
            st.session_state.setdefault("sb_gate", False)        # seed-before-instantiate
            gate_only = st.checkbox("Gate-passed only", value=False, key="sb_gate",
                                    help="Show only stocks that clear the engine's quality gate.")
            if gate_only and "gate_pass" in _cf.columns:
                _cf = _narrow(_cf, _cf["gate_pass"] == 1, "Gate-passed")
            st.session_state.setdefault("sb_minq", 0)            # seed-before-instantiate
            min_quality = st.slider("Min Quality Score", 0, 100, 0, key="sb_minq",
                                    help="Min fundamental quality score (PRE-forensic-penalty — moat + "
                                         "growth + cash + governance, before red-flag cuts).")
            if min_quality > 0 and "quality_score" in _cf.columns:
                _cf = _narrow(_cf, _cf["quality_score"] >= min_quality, "Min Quality")
            st.session_state.setdefault("sb_minscore", 0)        # seed-before-instantiate
            min_score = st.slider("Min Composite Score", 0, 100, 0, key="sb_minscore",
                                  help="Min headline composite_score (post-forensic-penalty — the score "
                                       "your tiers are built on; stronger than Min Quality, which is pre-penalty).")
            if min_score > 0 and "composite_score" in _cf.columns:
                _cf = _narrow(_cf, _cf["composite_score"] >= min_score, "Min Score")
            st.caption(f"→ {len(_cf):,} remaining")

    # The cascade frame (_cf) now encodes EVERY filter — the categorical groups AND the Refine
    # thresholds — so the dropdown options and the final result are genuinely identical and `filt`
    # is just `_cf`. The .copy() is REQUIRED: _cf is `df` itself when nothing is filtered, so the
    # copy guarantees the caller can never mutate the source frame.
    filt = _cf.copy()
    # Publish the zero-results culprit to the caller so the Discovery main-panel empty state names
    # the SAME filter this sidebar's funnel does (two disagreeing zero-states make the user hunt for
    # what the code already knows). `.attrs` is the channel the pipeline already uses for
    # `detected_market_regime`. Set on `filt` AFTER the .copy() — never rely on pandas propagating
    # attrs through an operation, which differs between local pandas 3 and prod's pandas 2.
    filt.attrs["zero_culprit"] = _zero_at[0] if _zero_at else ""

    # Applied-filter chips — computed AFTER the cascade so a value pruned by an upper filter can't
    # linger as a stale chip (same end-fill rationale as the funnel). Click ✕ to drop one filter.
    _chips = _compute_active_chips(st.session_state, _RF_MAX)
    with _chips_ph.container():
        if _chips:
            st.caption(f"🎯 {len(_chips)} active · click to remove")
            for _ck, _clabel in _chips:
                st.button(f"✕  {_clabel}", key=f"chip_{_ck}", on_click=_remove_one_filter,
                          args=(_ck,), width="stretch")

    # Fill the live results funnel (placeholder created at the TOP of the sidebar filter panel).
    # `filt` == `_cf` now, so the count reflects the WHOLE cascade — every group, Refine included.
    # The headline number sits above every filter while measuring all of them; the funnel is what
    # makes the panel feel like one interconnected system: change any filter, watch this number move.
    _uni_n, _fin_n = len(df), len(filt)
    _pct = (_fin_n / _uni_n) if _uni_n else 0.0
    _active_total = _active_n(
        "sb_mcap", "sb_sector", "sb_industry", "sb_cyc", "sb_capphase", "sb_tier", "sb_verdict",
        "sb_wealthtier", "sb_corpclass", "sb_capalloc",
        "sb_maxrf", "sb_piotier", "sb_cashmach", "sb_mincov", "sb_hidestale",
        "sb_fwfam", "sb_fw_exclude", "sb_fw_include", "sb_fw_combine", "sb_moat", "sb_peg_zone", "sb_buy_zone",
        "sb_floatfunded",
        "sb_weinstein", "sb_lynchcat", "sb_mef", "sb_cftri", "sb_smartflow",
        "sb_catalyst", "sb_sellalert", "sb_eppc", "sb_epbox", "sb_mbsetup",
        "sb_gate", "sb_minq", "sb_minscore",
    )
    # Every filter now defaults to its show-all state, so zero active filters == the full universe.
    _flabel = (f"🎯 {_active_total} filter{'s' if _active_total != 1 else ''} active"
               if _active_total else "○ No filters — full universe")
    # Zero-results culprit — when the cascade is empty, name the filter that emptied it (recorded
    # by the _narrow choke point) so the user knows exactly which dial to loosen or ✕-chip away.
    _culprit = ""
    if _fin_n == 0 and _zero_at:
        _culprit = (
            f'<div style="font-size:0.62rem;color:{COLORS["red"]};margin-top:5px;">'
            f'⚠️ <strong>{_zero_at[0]}</strong> removed the last stocks — loosen it or click its ✕ chip</div>'
        )
    _funnel.markdown(
        f'''<div style="background:linear-gradient(135deg,{COLORS['bg_secondary']},{COLORS['bg_tertiary']});
             border:1px solid {COLORS['border']};border-radius:10px;padding:10px 12px;margin:2px 0 10px 0;">
          <div style="display:flex;align-items:baseline;justify-content:space-between;">
            <span style="font-size:1.5rem;font-weight:800;color:{COLORS['text_primary']};line-height:1;">{_fin_n:,}</span>
            <span style="font-size:0.62rem;color:{COLORS['text_muted']};">of {_uni_n:,} · {_pct:.0%}</span>
          </div>
          <div style="background:{COLORS['bg_tertiary']};border-radius:4px;height:5px;overflow:hidden;margin:6px 0;">
            <div style="width:{max(_pct * 100, 1):.0f}%;height:5px;border-radius:4px;background:{COLORS['purple']};"></div>
          </div>
          <div style="font-size:0.6rem;color:{COLORS['text_muted']};text-transform:uppercase;letter-spacing:0.6px;">
            {_flabel}
          </div>{_culprit}
        </div>''',
        unsafe_allow_html=True,
    )
    return filt
