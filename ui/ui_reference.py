"""In-app searchable glossary for the 📖 Reference tab.

Pure render helper — returns inline dark-theme HTML and makes ZERO Streamlit calls.
The st.text_input search widget lives in app.py (which owns session_state); this module
only formats. The glossary (the `_RAW_GLOSSARY` single source of truth in
ui_components) is INJECTED so the coverage test can assert on the returned string.

The Markdown export (build_reference_markdown) also emits the 37-framework registry: `_FW_META`
now lives at module scope in ui_tearsheet.py, imported by app.py and passed through here.
"""
import html

from config import COLORS


def render_reference(glossary: dict, query: str = "") -> str:
    """Inline dark-theme HTML for the Reference tab: a short stable intro + the glossary
    as definition rows, filtered case-insensitively by `query` over term AND definition.
    Pure: no Streamlit, no I/O, no global state. Terms sorted case-insensitively (deterministic),
    so acronyms (CFO, CRS) interleave naturally with words (Capital, Cash) instead of ASCII-clumping."""
    # Token-AND match: split the query on whitespace and require EVERY token to appear (in the term
    # OR its definition), order-independent — so "vampire dilution" still finds "Dilution Vampire" and
    # "roce coffee" finds entries mentioning both. Empty query → no tokens → all rows. Per-token plain
    # substring (NOT regex), so query chars like '(' can never raise.
    tokens = (query or "").lower().split()
    items = [
        (term, defn) for term, defn in sorted(glossary.items(), key=lambda kv: kv[0].lower())
        if all(tok in f"{term}\n{defn}".lower() for tok in tokens)
    ]

    intro = (
        f'<div style="color:{COLORS["text_secondary"]};font-size:0.8rem;line-height:1.5;'
        f'margin:4px 0 14px 0;">PRISM scores ~2,100 stocks through many framework lenses and '
        f'synthesises them into one verdict per stock. This is the searchable glossary — type to '
        f'find any term you see on screen. Each entry explains what the term <em>means</em>; it '
        f'never says whether a value is good or bad.</div>'
    )

    if not items:
        return (
            intro
            + f'<div style="color:{COLORS["text_muted"]};font-size:0.85rem;padding:10px 0;">'
            + f'No terms match “{html.escape(query or "")}”.</div>'
        )

    count = (
        f'<div style="font-size:0.66rem;color:{COLORS["text_muted"]};text-transform:uppercase;'
        f'letter-spacing:0.5px;margin-bottom:6px;">{len(items)} of {len(glossary)} terms</div>'
    )
    rows = "".join(
        f'<div style="padding:9px 0;border-bottom:1px solid {COLORS["border"]};">'
        f'<div style="font-size:0.82rem;font-weight:700;color:{COLORS["text_primary"]};">'
        f'{html.escape(term)}</div>'
        f'<div style="font-size:0.74rem;color:{COLORS["text_secondary"]};line-height:1.45;'
        f'margin-top:2px;">{html.escape(str(defn))}</div></div>'
        for term, defn in items
    )
    return intro + count + rows


def render_concepts(concept_ref: dict, query: str = "") -> str:
    """Inline dark-theme HTML for the categorical VALUE-labels (Wealth Creator, Deep Value, Stage 2…),
    grouped by category, filtered case-insensitively by `query` over label AND explanation. Pure: no
    Streamlit, no I/O. Returns "" when nothing matches (the glossary section shows its own state)."""
    tokens = (query or "").lower().split()   # token-AND: every word must appear (see render_reference)
    blocks = []
    for category, entries in concept_ref.items():
        rows = [(lbl, exp) for lbl, exp in entries
                if all(tok in f"{lbl}\n{exp}".lower() for tok in tokens)]
        if not rows:
            continue
        head = (f'<div style="font-size:0.7rem;font-weight:800;color:{COLORS["purple"]};'
                f'text-transform:uppercase;letter-spacing:0.8px;margin:16px 0 6px 0;">'
                f'{html.escape(category)}</div>')
        body = "".join(
            f'<div style="padding:7px 0;border-bottom:1px solid {COLORS["border"]};">'
            f'<div style="font-size:0.8rem;font-weight:700;color:{COLORS["text_primary"]};">{html.escape(lbl)}</div>'
            f'<div style="font-size:0.74rem;color:{COLORS["text_secondary"]};line-height:1.45;margin-top:2px;">'
            f'{html.escape(str(exp))}</div></div>'
            for lbl, exp in rows
        )
        blocks.append(head + body)
    return "".join(blocks)


def render_flags(flag_display: dict, query: str = "") -> str:
    """Inline dark-theme HTML for the forensic RED FLAGS — each shown as the engine's OWN display
    description + severity dot, severity-sorted (🔴→🟠→🟡). Pure: no Streamlit. Single source =
    `_FLAG_DISPLAY` in ui_tearsheet (rf_col → (description, severity)); rendered, never duplicated."""
    tokens = (query or "").lower().split()   # token-AND: every word must appear (see render_reference)
    _sev = {"🔴": 0, "🟠": 1, "🟡": 2}
    rows = [(desc, sev) for rf, (desc, sev) in flag_display.items()
            if all(tok in f"{desc}\n{rf}".lower() for tok in tokens)]
    if not rows:
        return ""
    rows.sort(key=lambda r: (_sev.get(r[1], 9), r[0].lower()))
    return "".join(
        f'<div style="padding:7px 0;border-bottom:1px solid {COLORS["border"]};">'
        f'<span style="font-size:0.8rem;">{html.escape(sev)} </span>'
        f'<span style="font-size:0.78rem;color:{COLORS["text_secondary"]};line-height:1.45;">'
        f'{html.escape(desc)}</span></div>'
        for desc, sev in rows
    )


def render_frameworks(fw_meta: dict, query: str = "") -> str:
    """Inline dark-theme HTML for the 37-framework registry — the SAME `_FW_META`-derived dict
    the Markdown export consumes (app.py adapts it to {emoji,name,desc}), so the on-screen
    section and the download can never disagree. Added 2026-08-28: the export documented all 37
    while on-screen search could find only 6 — "what is Dhandho Asymmetry?" had an answer in the
    file but not in the search box. Pure: no Streamlit, no I/O. Sorted by name (deterministic);
    token-AND filtered over name AND description; returns "" when nothing matches."""
    tokens = (query or "").lower().split()
    rows = []
    for _key, meta in sorted(fw_meta.items(), key=lambda kv: kv[0].lower()):
        emoji = meta.get("emoji", "") if isinstance(meta, dict) else ""
        name = meta.get("name", _key) if isinstance(meta, dict) else str(meta)
        desc = meta.get("desc", "") if isinstance(meta, dict) else ""
        if all(tok in f"{name}\n{desc}".lower() for tok in tokens):
            rows.append((emoji, name, desc))
    if not rows:
        return ""
    return "".join(
        f'<div style="padding:7px 0;border-bottom:1px solid {COLORS["border"]};">'
        f'<div style="font-size:0.8rem;font-weight:700;color:{COLORS["text_primary"]};">'
        f'{html.escape(emoji)} {html.escape(name)}</div>'
        + (f'<div style="font-size:0.74rem;color:{COLORS["text_secondary"]};line-height:1.45;'
           f'margin-top:2px;">{html.escape(str(desc))}</div>' if desc else "")
        + "</div>"
        for emoji, name, desc in rows
    )


def render_wcs_studies(studies: list, query: str = "") -> str:
    """Inline dark-theme HTML for the Wealth Creation Studies corpus — one entry per COMPLETELY-READ
    study (the WCS_STUDIES honesty contract: nothing summarized second-hand). Same grammar as the
    other corpora: token-AND filtered over every text field, deterministic order (list order = study
    number), returns "" when nothing matches. Pure: no Streamlit, no I/O."""
    tokens = (query or "").lower().split()
    rows = []
    for s in studies:
        blob = "\n".join(str(s.get(k, "")) for k in ("study", "years", "pub", "theme", "says", "prism"))
        if all(tok in blob.lower() for tok in tokens):
            rows.append(s)
    if not rows:
        return ""
    return "".join(
        f'<div style="padding:10px 0;border-bottom:1px solid {COLORS["border"]};">'
        f'<div style="font-size:0.82rem;font-weight:700;color:{COLORS["text_primary"]};">'
        f'📕 {html.escape(str(s.get("study", "")))} ({html.escape(str(s.get("years", "")))}) — '
        f'{html.escape(str(s.get("theme", "")))}</div>'
        f'<div style="font-size:0.66rem;color:{COLORS["text_muted"]};margin:1px 0 4px 0;">'
        f'published {html.escape(str(s.get("pub", "")))}</div>'
        f'<div style="font-size:0.74rem;color:{COLORS["text_secondary"]};line-height:1.5;">'
        f'{html.escape(str(s.get("says", "")))}</div>'
        f'<div style="font-size:0.72rem;color:{COLORS["text_muted"]};line-height:1.45;margin-top:5px;">'
        f'<strong>In PRISM:</strong> {html.escape(str(s.get("prism", "")))}</div></div>'
        for s in rows
    )


def build_reference_markdown(glossary: dict, concept_ref: dict, flag_display: dict,
                             frameworks: dict = None, studies: list = None, as_of: str = None,
                             scans: str = None) -> str:
    """The FULL Reference as one Markdown string, assembled from the SAME single-source dicts the
    render_* functions consume (NOT scraped from their HTML) — so the download, the on-screen
    Reference, and any future docs/reference.md can never drift. Exports EVERYTHING (no query filter).
    Deterministic ordering, matching the renderers. PURE: no Streamlit, no I/O.

    `frameworks` is optional: app.py passes the tearsheet's _FW_META (adapted to {emoji,name,desc}) to
    emit a Frameworks section; None -> the section is omitted (no crash). Each value may be an
    {emoji,name[,desc]} dict or a bare string. `scans` is the 🎯 Scans guide already written as Markdown
    by ui/scan_guide.guide_markdown (2026-10-14) — the same sections the screen shows."""
    import datetime as _dt
    as_of = as_of or _dt.date.today().isoformat()

    def _clean(s):
        # One line, whitespace-collapsed. Bullets (not tables) need no MD-escaping; keeping the text
        # verbatim lets the coverage test match terms exactly after the same collapse.
        return " ".join(str(s).split())

    n_concepts = sum(len(v) for v in concept_ref.values())
    out = [
        "# PRISM Reference", "",
        f"_Generated {as_of} · {len(glossary)} glossary terms · {n_concepts} concepts · "
        f"{len(flag_display)} red flags_", "",
    ]

    out += ["## Glossary", ""]
    for term, defn in sorted(glossary.items(), key=lambda kv: kv[0].lower()):
        out.append(f"- **{_clean(term)}** — {_clean(defn)}")
    out.append("")

    out += ["## Concepts", ""]
    for category, entries in concept_ref.items():            # dict order = deterministic
        out += [f"### {_clean(category)}", ""]
        for lbl, exp in entries:
            out.append(f"- **{_clean(lbl)}** — {_clean(exp)}")
        out.append("")

    out += ["## Forensic Red Flags", ""]
    _sev = {"🔴": 0, "🟠": 1, "🟡": 2}
    for desc, sev in sorted(((d, s) for d, s in flag_display.values()),
                            key=lambda r: (_sev.get(r[1], 9), r[0].lower())):
        out.append(f"- {sev} {_clean(desc)}")
    out.append("")

    if frameworks:
        out += ["## Frameworks", ""]
        for _key, meta in frameworks.items():
            if isinstance(meta, dict):
                emoji, name, desc = meta.get("emoji", ""), meta.get("name", _key), meta.get("desc", "")
            else:
                emoji, name, desc = "", str(meta), ""
            line = f"- {emoji} **{_clean(name)}**"
            if desc:
                line += f" — {_clean(desc)}"
            out.append(line)
        out.append("")

    if studies:
        out += ["## Wealth Creation Studies", "",
                "_One entry per completely-read study — every claim verified against the text._", ""]
        for s in studies:
            out += [f"### 📕 {_clean(s.get('study', ''))} ({_clean(s.get('years', ''))}) — "
                    f"{_clean(s.get('theme', ''))}", "",
                    f"_Published {_clean(s.get('pub', ''))}_", "",
                    _clean(s.get("says", "")), "",
                    f"**In PRISM:** {_clean(s.get('prism', ''))}", ""]

    if scans:
        out += [scans.rstrip(), ""]

    return "\n".join(out).rstrip() + "\n"
