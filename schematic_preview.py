"""Compact vector well schematic used by the main T&D Streamlit application."""
from html import escape

def build_well_schematic_svg(
    td=2500.0, section_count=4, casing_shoe=550.0,
    liner1_top=500.0, liner1_shoe=1300.0,
    liner2_top=1250.0, liner2_shoe=2100.0,
    kop=1600.0, eob=2000.0, final_inc=30.0,
    operation="Drilling", dp_length=1600.0, hwdp_length=400.0,
    dc_length=500.0, running_tubular_length=0.0,
    casing_label='20 in × 94 ppf | ID 18.730 in',
    liner1_label='13-3/8 in × 72 ppf | ID 12.347 in',
    liner2_label='9-5/8 in × 53.5 ppf | ID 8.835 in',
    open_hole_label='Open hole | 8.75 in',
):
    W, H = 760, 900
    # Fixed size locked for consistent display
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" preserveAspectRatio="xMidYMid meet">', '</svg>']
    return ''.join(parts)
