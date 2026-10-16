"""
UI Rendering Components
=======================
Exposes all frontend visualization and layout rendering modules.
"""

try:
    from .ui_tearsheet import (
        render_moat_growth_matrix,
        render_fisher_module,
        render_ep_power_curve_module,
        render_bruised_blue_chip_badge,
        render_multitrillioncap_card,
        render_forensic_perimeter,
        render_guru_frameworks,
        render_financial_insights,
        render_stock_hero,
        render_verdict_scorecard,
        render_score_strip,
        render_sell_alerts_panel,
        render_raw_signals,
        render_canslim_radar,
        render_sepa_radar,
        render_schilit_shield,
        render_dorsey_radar,
        render_outsider_radar,
        render_marks_radar,
        render_malik_radar,
        render_lynch_radar,
        render_mauboussin_radar,
        render_qglp_radar,
        render_mosl_wealth_matrix,
        render_piotroski_checklist,
        render_sector_peer_strip,
        render_trajectory_card,
        render_shariah_card,
        render_valuation_inversion_and_sizing_cockpit,
    )
except ImportError as e:
    import traceback
    traceback.print_exc()
    # Soft-fail: create stub functions so app.py doesn't crash
    def _stub(*args, **kwargs):
        import streamlit as st
        st.warning(f"⚠️ Tearsheet module unavailable: {e}")
    render_moat_growth_matrix = _stub
    render_fisher_module = _stub
    render_ep_power_curve_module = _stub
    render_bruised_blue_chip_badge = _stub
    render_multitrillioncap_card = _stub
    render_forensic_perimeter = _stub
    render_guru_frameworks = _stub
    render_financial_insights = _stub
    render_stock_hero = _stub
    render_verdict_scorecard = _stub
    render_score_strip = _stub
    render_sell_alerts_panel = _stub
    render_raw_signals = _stub
    render_canslim_radar = _stub
    render_sepa_radar = _stub
    render_schilit_shield = _stub
    render_dorsey_radar = _stub
    render_outsider_radar = _stub
    render_marks_radar       = _stub
    render_malik_radar       = _stub
    render_lynch_radar       = _stub
    render_mauboussin_radar  = _stub
    render_qglp_radar = _stub
    render_mosl_wealth_matrix = _stub
    render_piotroski_checklist = _stub
    render_sector_peer_strip = _stub
    render_trajectory_card = _stub
    render_shariah_card = _stub
    render_valuation_inversion_and_sizing_cockpit = _stub

from .ui_components import (
    inject_css,
    render_hero_banner,
    render_metric_strip,
    render_pulse_band,
    render_stock_card,
    render_radar_chart,
    render_tier_summary,
    render_sidebar_brand,
    help_chip,
)

from .ui_reference import (render_reference, render_concepts, render_flags,
                           render_frameworks, render_wcs_studies, build_reference_markdown)

__all__ = [
    "render_moat_growth_matrix",
    "render_fisher_module",
    "render_ep_power_curve_module",
    "render_bruised_blue_chip_badge",
    "render_multitrillioncap_card",
    "render_forensic_perimeter",
    "render_guru_frameworks",
    "render_financial_insights",
    "render_stock_hero",
    "render_verdict_scorecard",
    "render_score_strip",
    "render_sell_alerts_panel",
    "render_raw_signals",
    "render_canslim_radar",
    "render_sepa_radar",
    "render_schilit_shield",
    "render_dorsey_radar",
    "render_outsider_radar",
    "render_marks_radar",
    "render_malik_radar",
    "render_lynch_radar",
    "render_mauboussin_radar",
    "render_qglp_radar",
    "render_mosl_wealth_matrix",
    "render_piotroski_checklist",
    "render_sector_peer_strip",
    "render_trajectory_card",
    "render_shariah_card",
    "render_valuation_inversion_and_sizing_cockpit",
    "inject_css",
    "render_hero_banner",
    "render_metric_strip",
    "render_pulse_band",
    "render_stock_card",
    "render_radar_chart",
    "render_tier_summary",
    "render_sidebar_brand",
    "help_chip",
    "render_reference",
    "render_concepts",
    "render_flags",
    "render_frameworks",
    "render_wcs_studies",
    "build_reference_markdown",
]
