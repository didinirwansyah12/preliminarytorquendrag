import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as st_components
from schematic_preview import build_well_schematic_svg

from td_hybrid_prototype import Survey, Component, TDConfig, minimum_curvature, solve_depth_sweep, mud_top_md_for_fill_fraction
from report import generate_tnd_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_PATH = os.path.join(BASE_DIR, 'TnD Icon 2.png')
RIGSIS_LOGO_PATH = os.path.join(BASE_DIR, 'RIGSIS Logo.png')

st.set_page_config(
    page_title='Preliminary Torque & Drag Analysis',
    page_icon=ICON_PATH if os.path.exists(ICON_PATH) else '🔩',
    layout='wide',
)

# ============================================================
# DATABASES
# ============================================================

# Common API 5CT casing sizes. Display includes OD, nominal weight and calculated/standard ID.
# The selected item is used to populate the casing ID automatically.
CASING_DATABASE = {
    # Requested priority sizes appear first: 30, 20, 13-3/8, 10-3/4, 8-5/8, 7 in.
    '30.000 × 202.50 lb/ft | ID 29.000 in': dict(od=30.000, wt=202.50, id=29.000),
    '20.000 × 94.00 lb/ft | ID 18.730 in': dict(od=20.000, wt=94.00, id=18.730),
    '20.000 × 106.00 lb/ft | ID 18.560 in': dict(od=20.000, wt=106.00, id=18.560),
    '20.000 × 133.00 lb/ft | ID 18.376 in': dict(od=20.000, wt=133.00, id=18.376),
    '13.375 × 48.00 lb/ft | ID 12.615 in': dict(od=13.375, wt=48.00, id=12.615),
    '13.375 × 54.50 lb/ft | ID 12.459 in': dict(od=13.375, wt=54.50, id=12.459),
    '13.375 × 61.00 lb/ft | ID 12.347 in': dict(od=13.375, wt=61.00, id=12.347),
    '13.375 × 68.00 lb/ft | ID 12.259 in': dict(od=13.375, wt=68.00, id=12.259),
    '13.375 × 72.00 lb/ft | ID 12.191 in': dict(od=13.375, wt=72.00, id=12.191),
    '13.375 × 86.00 lb/ft | ID 12.000 in': dict(od=13.375, wt=86.00, id=12.000),
    '10.750 × 40.50 lb/ft | ID 9.950 in': dict(od=10.750, wt=40.50, id=9.950),
    '10.750 × 45.50 lb/ft | ID 9.850 in': dict(od=10.750, wt=45.50, id=9.850),
    '10.750 × 51.00 lb/ft | ID 9.750 in': dict(od=10.750, wt=51.00, id=9.750),
    '8.625 × 24.00 lb/ft | ID 7.921 in': dict(od=8.625, wt=24.00, id=7.921),
    '8.625 × 28.00 lb/ft | ID 7.825 in': dict(od=8.625, wt=28.00, id=7.825),
    '8.625 × 32.00 lb/ft | ID 7.725 in': dict(od=8.625, wt=32.00, id=7.725),
    '11.750 × 38.00 lb/ft | ID 11.000 in': dict(od=11.750, wt=38.00, id=11.000),
    '11.750 × 47.00 lb/ft | ID 10.880 in': dict(od=11.750, wt=47.00, id=10.880),
    '9.625 × 36.00 lb/ft | ID 8.921 in': dict(od=9.625, wt=36.00, id=8.921),
    '9.625 × 40.00 lb/ft | ID 8.835 in': dict(od=9.625, wt=40.00, id=8.835),
    '9.625 × 43.50 lb/ft | ID 8.755 in': dict(od=9.625, wt=43.50, id=8.755),
    '9.625 × 47.00 lb/ft | ID 8.681 in': dict(od=9.625, wt=47.00, id=8.681),
    '9.625 × 53.50 lb/ft | ID 8.535 in': dict(od=9.625, wt=53.50, id=8.535),
    '7.625 × 26.40 lb/ft | ID 6.625 in': dict(od=7.625, wt=26.40, id=6.625),
    '7.625 × 29.70 lb/ft | ID 6.625 in': dict(od=7.625, wt=29.70, id=6.625),
    '7.000 × 20.00 lb/ft | ID 6.456 in': dict(od=7.000, wt=20.00, id=6.456),
    '7.000 × 23.00 lb/ft | ID 6.366 in': dict(od=7.000, wt=23.00, id=6.366),
    '7.000 × 26.00 lb/ft | ID 6.276 in': dict(od=7.000, wt=26.00, id=6.276),
    '7.000 × 29.00 lb/ft | ID 6.184 in': dict(od=7.000, wt=29.00, id=6.184),
    '7.000 × 32.00 lb/ft | ID 6.094 in': dict(od=7.000, wt=32.00, id=6.094),
    '5.500 × 15.50 lb/ft | ID 4.950 in': dict(od=5.500, wt=15.50, id=4.950),
    '5.500 × 17.00 lb/ft | ID 4.778 in': dict(od=5.500, wt=17.00, id=4.778),
    '5.500 × 20.00 lb/ft | ID 4.670 in': dict(od=5.500, wt=20.00, id=4.670),
    '4.500 × 9.50 lb/ft | ID 4.090 in': dict(od=4.500, wt=9.50, id=4.090),
    '4.500 × 10.50 lb/ft | ID 4.000 in': dict(od=4.500, wt=10.50, id=4.000),
    '4.500 × 11.60 lb/ft | ID 3.920 in': dict(od=4.500, wt=11.60, id=3.920),
    '4.500 × 13.50 lb/ft | ID 3.826 in': dict(od=4.500, wt=13.50, id=3.826),
}

# Keep the requested common sizes at the top in this exact order. All other sizes follow
# after 7 in, even when their OD is larger than 7 in (e.g. 9-5/8 in).
_CASING_PRIORITY = [30.000, 20.000, 13.375, 10.750, 8.625, 7.000]
CASING_DATABASE = dict(sorted(
    CASING_DATABASE.items(),
    key=lambda item: (
        0, _CASING_PRIORITY.index(item[1]['od']), list(CASING_DATABASE).index(item[0])
    ) if item[1]['od'] in _CASING_PRIORITY else (1, 0, list(CASING_DATABASE).index(item[0]))
))

DP_DATABASE = {
    '5.000 × 4.276 × 19.50 lb/ft': dict(od=5.000, id=4.276, wt=19.50, tj=6.625, joint=9.144, conn=0.433),
    '5.000 × 4.276 × 21.92 lb/ft': dict(od=5.000, id=4.276, wt=21.92, tj=6.625, joint=9.144, conn=0.433),
    '5.000 × 4.000 × 19.50 lb/ft': dict(od=5.000, id=4.000, wt=19.50, tj=6.625, joint=9.144, conn=0.433),
    '4.500 × 3.826 × 16.60 lb/ft': dict(od=4.500, id=3.826, wt=16.60, tj=6.250, joint=9.144, conn=0.433),
    '4.000 × 3.340 × 14.00 lb/ft': dict(od=4.000, id=3.340, wt=14.00, tj=5.500, joint=9.144, conn=0.433),
    '3.500 × 2.764 × 13.30 lb/ft': dict(od=3.500, id=2.764, wt=13.30, tj=4.500, joint=9.144, conn=0.433),
    '3.500 × 2.750 × 13.00 lb/ft': dict(od=3.500, id=2.750, wt=13.00, tj=4.500, joint=9.144, conn=0.433),
    '3.500 × 2.602 × 15.50 lb/ft': dict(od=3.500, id=2.602, wt=15.50, tj=4.500, joint=9.144, conn=0.433),
    '3.500 × 2.500 × 15.50 lb/ft': dict(od=3.500, id=2.500, wt=15.50, tj=4.500, joint=9.144, conn=0.433),
}

HWDP_DATABASE = {
    '5.000 × 3.000 × 50.00 lb/ft': dict(od=5.000, id=3.000, wt=50.00, tj=6.750, joint=9.144, conn=0.600),
    '5.000 × 3.000 × 49.30 lb/ft': dict(od=5.000, id=3.000, wt=49.30, tj=6.750, joint=9.144, conn=0.600),
    '4.500 × 2.750 × 38.00 lb/ft': dict(od=4.500, id=2.750, wt=38.00, tj=6.250, joint=9.144, conn=0.600),
}

DC_DATABASE = {
    '8.000 × 2.813 × 147.00 lb/ft': dict(od=8.000, id=2.813, wt=147.00),
    '8.000 × 2.813 × 148.00 lb/ft': dict(od=8.000, id=2.813, wt=148.00),
    '8.000 × 2.750 × 147.00 lb/ft': dict(od=8.000, id=2.750, wt=147.00),
    '8.000 × 2.813 × 132.00 lb/ft': dict(od=8.000, id=2.813, wt=132.00),
    '8.000 × 2.813 × 100.00 lb/ft': dict(od=8.000, id=2.813, wt=100.00),
    '6.500 × 2.813 × 91.76 lb/ft': dict(od=6.500, id=2.813, wt=91.76),
    '6.500 × 2.813 × 100.00 lb/ft': dict(od=6.500, id=2.813, wt=100.00),
    '6.500 × 2.750 × 100.00 lb/ft': dict(od=6.500, id=2.750, wt=100.00),
    '6.000 × 2.750 × 83.00 lb/ft': dict(od=6.000, id=2.750, wt=83.00),
}


def _init_state():
    defaults = {
        'td': 1500.0, 'kop': 300.0, 'eob': 700.0, 'final_inc': 30.0,
        'casing_shoe': 400.0,
        'section_count': 4,
        'liner1_overlap': 50.0, 'liner1_shoe': 700.0,
        'liner1_spec': '10.750 × 40.50 lb/ft | ID 9.950 in',
        'liner2_overlap': 50.0, 'liner2_shoe': 1300.0,
        'liner2_spec': '8.625 × 24.00 lb/ft | ID 7.921 in',
        'casing_spec': '13.375 × 72.00 lb/ft | ID 12.191 in',
        'open_hole_id': 7.875,
        'mud_weight': 8.8, 'r600': 50.0, 'r300': 40.0, 'mud_fill_percent': 100.0,
        'ff_casing_min': 0.30, 'ff_casing_max': 0.50,
        'ff_open_min': 0.30, 'ff_open_max': 0.70,
        'dc_length': 113.0, 'hwdp_length': 214.0,
        'analysis_string': 'Drilling',
        'casing_string_spec': '13.375 × 68.00 lb/ft | ID 12.259 in',
        'casing_string_length': 1300.0,
        'dp_std': '5.000 × 4.276 × 19.50 lb/ft',
        'hwdp_std': '5.000 × 3.000 × 49.30 lb/ft',
        'dc_std': '6.500 × 2.813 × 91.76 lb/ft',
        'rpm': 60.0, 'wob': 38000.0, 'bit_torque_lbf_ft': 2000.0,
        'drilling_mode': 'On Bottom', 'max_iter': 200,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


_init_state()

# ============================================================
# HELPERS
# ============================================================

def create_j_type_survey(td, kop, final_inc, eob):
    build_end = min(max(float(eob), float(kop)), float(td))
    points = sorted(set([0.0, float(kop), float(build_end), float(td)] +
                        [float(x) for x in np.arange(0.0, float(td), 30.0)]))
    inc = [
        0.0 if md <= kop else
        final_inc if md >= build_end else
        final_inc * (md-kop)/(build_end-kop) if build_end > kop else final_inc
        for md in points
    ]
    return Survey(np.array(points), np.array(inc), np.zeros(len(points)))


def create_components(analysis_string, dp_std, hwdp_std, dc_std, td_m, hwdp_length, dc_length, casing_string_spec, casing_string_length):
    """Build the final surface-to-bottom string for the selected operation.

    Drilling:       DP + HWDP + DC
    Running Liner:  DP + Liner
    Running Casing: Casing only (full string)

    DP length is always derived from:
        DP = TD - HWDP - DC - Running Casing/Liner
    """
    td_m = float(td_m)
    hwdp_length = float(hwdp_length)
    dc_length = float(dc_length)
    running_length = float(casing_string_length) if analysis_string in ('Running Liner', 'Running Casing') else 0.0

    if analysis_string == 'Drilling':
        dp = DP_DATABASE[dp_std]
        hw = HWDP_DATABASE[hwdp_std]
        dc = DC_DATABASE[dc_std]
        dp_length = td_m - hwdp_length - dc_length
        if dp_length < -1e-9:
            raise ValueError('HWDP + DC length cannot exceed TD.')
        return [
            Component('Drill Pipe', dp['od'], dp['id'], dp['wt'], max(0.0, dp_length), dp['tj'], dp['joint'], dp['conn']),
            Component('HWDP', hw['od'], hw['id'], hw['wt'], hwdp_length, hw['tj'], hw['joint'], hw['conn']),
            Component('Drill Collar', dc['od'], dc['id'], dc['wt'], dc_length, 0.0, 9.144, 0.0),
        ]

    cs = CASING_DATABASE[casing_string_spec]
    if analysis_string == 'Running Casing':
        # Full-string casing: casing length = TD and no DP/HWDP/DC.
        return [Component('Casing String', cs['od'], cs['id'], cs['wt'], td_m, 0.0, 9.144, 0.0)]

    # Running liner: DP + liner, with HWDP/DC forced to zero.
    if running_length <= 0.0 or running_length > td_m + 1e-9:
        raise ValueError('Liner length must be greater than zero and cannot exceed TD.')
    dp = DP_DATABASE[dp_std]
    dp_length = td_m - running_length
    return [
        Component('Drill Pipe', dp['od'], dp['id'], dp['wt'], max(0.0, dp_length), dp['tj'], dp['joint'], dp['conn']),
        Component('Liner String', cs['od'], cs['id'], cs['wt'], running_length, 0.0, 9.144, 0.0),
    ]


def run_case(survey, components, base, ff_casing, ff_open, bit_torque):
    cfg = TDConfig(
        mud_weight_ppg=float(base.mud_weight_ppg),
        mud_fill_fraction=float(base.mud_fill_fraction),
        td_m=float(base.td_m),
        casing_shoe_m=float(base.casing_shoe_m),
        casing_id_in=float(base.casing_id_in),
        open_hole_id_in=float(base.open_hole_id_in),
        section_count=int(base.section_count),
        liner1_top_m=float(base.liner1_top_m), liner1_shoe_m=float(base.liner1_shoe_m), liner1_id_in=float(base.liner1_id_in),
        liner2_top_m=float(base.liner2_top_m), liner2_shoe_m=float(base.liner2_shoe_m), liner2_id_in=float(base.liner2_id_in),
        ff_casing=float(ff_casing),
        ff_open_hole=float(ff_open),
        r600=float(base.r600), r300=float(base.r300), rpm=float(base.rpm),
        wob_lbf=float(base.wob_lbf), drilling_mode=base.drilling_mode,
        max_iter=int(base.max_iter), tolerance=base.tolerance, relaxation=base.relaxation,
    )
    return solve_depth_sweep(survey, components, cfg, bit_torque_lbf_ft=float(bit_torque))


def build_sensitivity(survey, components, base, ff_casing_min, ff_casing_max, ff_open_min, ff_open_max, bit_torque):
    cases = {
        'Low-Low': (ff_casing_min, ff_open_min),
        'Low-High': (ff_casing_min, ff_open_max),
        'High-Low': (ff_casing_max, ff_open_min),
        'High-High': (ff_casing_max, ff_open_max),
    }
    dfs = {}
    for name, (fc, fo) in cases.items():
        dfs[name] = run_case(survey, components, base, fc, fo, bit_torque)
    return dfs


def _annotate_depth_markers(ax, ri):
    markers = [('KOP', ri.get('kop')) , ('EOB', ri.get('eob')), ('Casing Shoe', ri.get('casing_shoe'))]
    if int(ri.get('section_count', 2)) >= 3:
        markers.append(('Liner 1 Shoe', ri.get('liner1_shoe')))
    if int(ri.get('section_count', 2)) >= 4:
        markers.append(('Liner 2 Shoe', ri.get('liner2_shoe')))
    seen = set()
    for label, depth in markers:
        if depth is None:
            continue
        depth = float(depth)
        if depth in seen or depth < 0 or depth > float(ri.get('td', depth)):
            continue
        seen.add(depth)
        ax.axhline(depth, color='#777777', linewidth=0.7, linestyle=(0, (3, 3)), alpha=0.65, zorder=0)
        ax.plot([0.0], [depth], marker='o', markersize=3.2, color='#555555',
                transform=ax.get_yaxis_transform(), clip_on=False, zorder=6)
        ax.annotate(f'{label} @{depth:,.0f} mMD', xy=(0.012, depth), xycoords=('axes fraction', 'data'),
                    xytext=(2, 2), textcoords='offset points', ha='left', va='bottom', fontsize=8,
                    color='#333333', bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85),
                    clip_on=True, zorder=7)

def plot_hookload(result, sensitivity, td, plot_top, ri=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    md = result['MD_m'].to_numpy()
    puw_stack = np.stack([df['PUW_lbf'].to_numpy() for df in sensitivity.values()])
    sow_stack = np.stack([df['SOW_lbf'].to_numpy() for df in sensitivity.values()])
    ax.fill_betweenx(md, puw_stack.min(axis=0), puw_stack.max(axis=0), alpha=0.14, label='PUW friction sensitivity')
    ax.fill_betweenx(md, sow_stack.min(axis=0), sow_stack.max(axis=0), alpha=0.10, label='SOW friction sensitivity')
    ax.plot(result['PUW_lbf'], md, linewidth=2.2, label='PUW — nominal')
    ax.plot(result['ROTW_lbf'], md, linewidth=2.2, label='ROTW — nominal')
    ax.plot(result['SOW_lbf'], md, linewidth=2.2, label='SOW — nominal')
    ax.set_ylim(float(td), float(plot_top))
    if ri is not None:
        _annotate_depth_markers(ax, ri)
    ax.set_xlabel('Hookload (lbf)')
    ax.set_ylabel('Measured Depth (m)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_torque(result, sensitivity, td, plot_top, ri=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    sens_stack = np.stack([df['SurfaceTorque_lbf_ft'].to_numpy() for df in sensitivity.values()])
    sens_min = sens_stack.min(axis=0)
    sens_max = sens_stack.max(axis=0)
    md = result['MD_m'].to_numpy()
    ax.fill_betweenx(md, sens_min, sens_max, alpha=0.16, label='Friction sensitivity envelope')
    ax.plot(result['SurfaceTorque_lbf_ft'], md, linewidth=2.5, label='Surface Torque — nominal')
    ax.set_ylim(float(td), float(plot_top))
    if ri is not None:
        _annotate_depth_markers(ax, ri)
    ax.set_xlabel('Surface Torque (lbf-ft)')
    ax.set_ylabel('Measured Depth (m)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


# ============================================================
# HEADER / TITLE
# ============================================================

if os.path.exists(ICON_PATH):
    h1, h2 = st.columns([0.09, 0.91], vertical_alignment='center')
    h1.image(ICON_PATH, width=64)
    h2.markdown('## Preliminary Torque & Drag Analysis')
else:
    st.markdown('## Preliminary Torque & Drag Analysis')

st.markdown(
    '**J-Type 2D Soft-String Force-Balance | Hookload + RPM-Dependent Surface Torque | '
    'Developed by Didin Irwansyah for Rigsis Drilling Team**'
)
st.caption(
    'This prototype is intended for preliminary assessment only. It is not a replacement for detailed '
    'Torque and Drag Analysis from Directional Drilling Company.'
)
st.divider()

# ============================================================
# INPUTS — MAIN PAGE, NO SIDEBAR
# ============================================================

st.header('Input Data')

def _on_section_count_change():
    # Apply the requested default open-hole size when switching between the
    # 3-section and 4-section configurations. Two-section mode preserves the
    # user's current selection.
    if int(st.session_state.get('section_count', 4)) == 3:
        st.session_state['open_hole_id'] = 9.875
    elif int(st.session_state.get('section_count', 4)) == 4:
        st.session_state['open_hole_id'] = 7.875


OPEN_HOLE_DIAMETERS = [26.0, 17.5, 12.25, 9.875, 7.875, 8.5, 6.5]

with st.expander('1. Well Geometry & Schematic', expanded=True):
    geometry_col, schematic_col = st.columns([1.08, 1], gap='large')

    # Geometry inputs are arranged in the requested sequence, with each
    # section's inputs on a single row. The schematic remains alongside them.
    with geometry_col:
        st.markdown('**Trajectory**')
        t1, t2, t3, t4 = st.columns(4)
        td = t1.number_input('TD (m)', min_value=100.0, step=50.0, key='td')
        kop = t2.number_input('KOP (m)', min_value=0.0, max_value=float(td), step=25.0, key='kop')
        eob = t3.number_input('EOB (m)', min_value=float(kop), max_value=float(td), step=25.0, key='eob')
        final_inc = t4.number_input('Final Inclination (°)', min_value=0.0, max_value=90.0, step=0.5, key='final_inc')

        st.markdown('**Number of Well Sections**')
        section_count = st.selectbox(
            'Configuration', [2, 3, 4],
            format_func=lambda x: {
                2: '2 Sections — Cased + Open Hole',
                3: '3 Sections — Cased + 1 Liner + Open Hole',
                4: '4 Sections — Cased + 2 Liners + Open Hole',
            }[x], key='section_count', on_change=_on_section_count_change,
        )

        st.markdown('**Casing Section Data**')
        c1, c2 = st.columns([1.65, 1])
        casing_spec = c1.selectbox('Casing — API OD × Weight | ID', list(CASING_DATABASE), key='casing_spec')
        casing_shoe = c2.number_input('Casing Shoe Depth (mMD)', min_value=0.0, max_value=float(td), step=25.0, key='casing_shoe')
        casing_id = CASING_DATABASE[casing_spec]['id']
        st.caption(f"OD {CASING_DATABASE[casing_spec]['od']:.3f} in | {CASING_DATABASE[casing_spec]['wt']:.2f} lb/ft | ID {casing_id:.3f} in")

        liner1_overlap = float(st.session_state.liner1_overlap)
        liner1_top = max(0.0, float(casing_shoe) - liner1_overlap)
        liner1_shoe = float(st.session_state.liner1_shoe)
        liner1_spec = st.session_state.liner1_spec
        liner1_id = CASING_DATABASE[liner1_spec]['id']
        if section_count >= 3:
            st.markdown('**Liner 1 Data**')
            l1a, l1b, l1c = st.columns([1.7, 0.85, 1])
            liner1_spec = l1a.selectbox('Liner 1 — API OD × Weight | ID', list(CASING_DATABASE), key='liner1_spec')
            liner1_overlap = l1b.number_input('Overlap (m)', min_value=0.0, step=10.0, key='liner1_overlap')
            liner1_shoe = l1c.number_input('Liner 1 Shoe (mMD)', min_value=0.0, max_value=float(td), step=25.0, key='liner1_shoe')
            liner1_top = max(0.0, float(casing_shoe) - float(liner1_overlap))
            liner1_id = CASING_DATABASE[liner1_spec]['id']
            st.caption(f"OD {CASING_DATABASE[liner1_spec]['od']:.3f} in | {CASING_DATABASE[liner1_spec]['wt']:.2f} lb/ft | ID {liner1_id:.3f} in | Calculated TOL: {liner1_top:,.0f} mMD")

        liner2_overlap = float(st.session_state.liner2_overlap)
        liner2_top = max(0.0, float(liner1_shoe) - liner2_overlap)
        liner2_shoe = float(st.session_state.liner2_shoe)
        liner2_spec = st.session_state.liner2_spec
        liner2_id = CASING_DATABASE[liner2_spec]['id']
        if section_count >= 4:
            st.markdown('**Liner 2 Data**')
            l2a, l2b, l2c = st.columns([1.7, 0.85, 1])
            liner2_spec = l2a.selectbox('Liner 2 — API OD × Weight | ID', list(CASING_DATABASE), key='liner2_spec')
            liner2_overlap = l2b.number_input('Overlap (m)', min_value=0.0, step=10.0, key='liner2_overlap')
            liner2_shoe = l2c.number_input('Liner 2 Shoe (mMD)', min_value=0.0, max_value=float(td), step=25.0, key='liner2_shoe')
            liner2_top = max(0.0, float(liner1_shoe) - float(liner2_overlap))
            liner2_id = CASING_DATABASE[liner2_spec]['id']
            st.caption(f"OD {CASING_DATABASE[liner2_spec]['od']:.3f} in | {CASING_DATABASE[liner2_spec]['wt']:.2f} lb/ft | ID {liner2_id:.3f} in | Calculated TOL: {liner2_top:,.0f} mMD")

        st.markdown('**Open Hole Data**')
        open_hole_id = st.selectbox(
            'Open Hole Diameter (in)', OPEN_HOLE_DIAMETERS,
            format_func=lambda x: f'{x:g} in', key='open_hole_id',
        )

    with schematic_col:
        st.markdown('**Interactive Well Schematic**')
        schematic_svg = build_well_schematic_svg(
            td=td, section_count=section_count, casing_shoe=casing_shoe,
            liner1_top=liner1_top, liner1_shoe=liner1_shoe,
            liner2_top=liner2_top, liner2_shoe=liner2_shoe,
            kop=kop, eob=eob, final_inc=final_inc,
            operation=st.session_state.get('analysis_string', 'Drilling'),
            dp_length=max(0.0, float(td)-float(st.session_state.get('hwdp_length',214.0))-float(st.session_state.get('dc_length',113.0))),
            hwdp_length=float(st.session_state.get('hwdp_length',214.0)), dc_length=float(st.session_state.get('dc_length',113.0)),
            running_tubular_length=float(st.session_state.get('casing_string_length',800.0)),
            casing_label=f"{CASING_DATABASE[casing_spec]['od']:g} in × {CASING_DATABASE[casing_spec]['wt']:g} ppf | ID {casing_id:.3f} in",
            liner1_label=f"{CASING_DATABASE[liner1_spec]['od']:g} in × {CASING_DATABASE[liner1_spec]['wt']:g} ppf | ID {liner1_id:.3f} in",
            liner2_label=f"{CASING_DATABASE[liner2_spec]['od']:g} in × {CASING_DATABASE[liner2_spec]['wt']:g} ppf | ID {liner2_id:.3f} in",
            open_hole_label=f"Open hole | {open_hole_id:.3f} in",
        )
        # Fixed-size container so the schematic scale is locked and always fully visible
        # regardless of screen width or browser zoom.
        html_wrapper = f"""
        <div style="display:flex; justify-content:center; overflow:auto;">
            {schematic_svg}
        </div>
        """
        st_components.html(html_wrapper, height=920, scrolling=False)
        st.caption('Depths and string composition update from the inputs. Hover over schematic labels for section information. Schematic scale is locked for consistent display.')

with st.expander('2. Drillstring / Running String', expanded=True):
    analysis_string = st.selectbox(
        'Operation', ['Drilling', 'Running Liner', 'Running Casing'], key='analysis_string'
    )

    # DP
    r1, r2 = st.columns([2.6, 1.0])
    dp_std = r1.selectbox('Drill Pipe — OD × ID × Weight', list(DP_DATABASE), key='dp_std')

    # In all modes DP is derived, never manually entered.
    hwdp_length_state = float(st.session_state.get('hwdp_length', 214.0))
    dc_length_state = float(st.session_state.get('dc_length', 113.0))
    casing_length_state = float(st.session_state.get('casing_string_length', 1300.0))
    if analysis_string == 'Drilling':
        dp_length = float(td) - hwdp_length_state - dc_length_state
    elif analysis_string == 'Running Liner':
        dp_length = float(td) - casing_length_state
    else:
        dp_length = 0.0
    r2.number_input('DP Length (m) — Auto', min_value=0.0, step=10.0, value=max(0.0, dp_length), disabled=True, key='dp_length_display')

    # HWDP
    r3, r4 = st.columns([2.6, 1.0])
    hwdp_std = r3.selectbox('HWDP — OD × ID × Weight', list(HWDP_DATABASE), key='hwdp_std', disabled=analysis_string != 'Drilling')
    if analysis_string == 'Drilling':
        hwdp_length = r4.number_input('HWDP Length (m)', min_value=0.0, step=10.0, key='hwdp_length')
    else:
        # HWDP is not part of the active running string for casing/liner operations.
        # Keep the drilling value in session state, but show 0.00 m automatically.
        r4.number_input('HWDP Length (m) — Auto 0', min_value=0.0, max_value=0.0, value=0.0, step=10.0, disabled=True, key='hwdp_length_auto_zero')
        hwdp_length = 0.0

    # DC: show only body ODs that fit inside the selected open hole while drilling.
    # This prevents the default 8-in DC from being selected with a 7-7/8-in open hole.
    r5, r6 = st.columns([2.6, 1.0])
    if analysis_string == 'Drilling':
        compatible_dc_options = [name for name, spec in DC_DATABASE.items() if spec['od'] < float(open_hole_id)]
        if not compatible_dc_options:
            st.error(f'No Drill Collar option has an OD smaller than the selected open-hole diameter ({float(open_hole_id):g} in). Choose a larger open-hole diameter or add a smaller DC to the catalogue.')
            st.stop()
        if st.session_state.get('dc_std') not in compatible_dc_options:
            # Pick the largest compatible DC OD as a sensible default.
            compatible_dc_options.sort(key=lambda name: DC_DATABASE[name]['od'], reverse=True)
            st.session_state['dc_std'] = compatible_dc_options[0]
        dc_options = compatible_dc_options
    else:
        dc_options = list(DC_DATABASE)
    dc_std = r5.selectbox('Drill Collar — OD × ID × Weight', dc_options, key='dc_std', disabled=analysis_string != 'Drilling')
    if analysis_string == 'Drilling':
        dc_length = r6.number_input('DC Length (m)', min_value=0.0, step=10.0, key='dc_length')
    else:
        # DC is not part of the active running string for casing/liner operations.
        # Keep the drilling value in session state, but show 0.00 m automatically.
        r6.number_input('DC Length (m) — Auto 0', min_value=0.0, max_value=0.0, value=0.0, step=10.0, disabled=True, key='dc_length_auto_zero')
        dc_length = 0.0

    # Running casing / liner length
    if analysis_string in ('Running Liner', 'Running Casing'):
        r7, r8 = st.columns([2.6, 1.0])
        casing_string_spec = r7.selectbox(
            'Casing / Liner String — API OD × Weight | ID',
            list(CASING_DATABASE), key='casing_string_spec'
        )
        if analysis_string == 'Running Casing':
            # Full string: length is forced to TD.
            r8.number_input('String Length (m) — Auto = TD', min_value=0.0, value=float(td), disabled=True, key='casing_string_length_auto')
            casing_string_length = float(td)
        else:
            casing_string_length = r8.number_input('Liner Length (m)', min_value=0.0, max_value=float(td), step=25.0, key='casing_string_length')
    else:
        casing_string_spec = st.session_state.get('casing_string_spec', '13.375 × 68.00 lb/ft | ID 12.259 in')
        casing_string_length = 0.0

# ============================================================
# MUD & FRICTION
# ============================================================

with st.expander('3. Mud Properties & Friction Factors', expanded=True):
    m1, m2, m3, m4 = st.columns(4)
    mud_weight = m1.number_input('Mud Weight (ppg)', min_value=6.0, max_value=20.0, step=0.1, key='mud_weight')
    r600 = m2.number_input('R600', min_value=0.0, step=1.0, key='r600')
    r300 = m3.number_input('R300', min_value=0.0, step=1.0, key='r300')
    mud_fill_percent = m4.number_input('Mud Fill (%)', min_value=0.0, max_value=100.0, step=5.0, key='mud_fill_percent')

    f1, f2, f3, f4 = st.columns(4)
    ff_casing_min = f1.number_input('FF Casing Min', min_value=0.0, max_value=1.0, step=0.05, key='ff_casing_min')
    ff_casing_max = f2.number_input('FF Casing Max', min_value=0.0, max_value=1.0, step=0.05, key='ff_casing_max')
    ff_open_min = f3.number_input('FF Open Hole Min', min_value=0.0, max_value=1.0, step=0.05, key='ff_open_min')
    ff_open_max = f4.number_input('FF Open Hole Max', min_value=0.0, max_value=1.0, step=0.05, key='ff_open_max')

# ============================================================
# OPERATING PARAMETERS
# ============================================================

with st.expander('4. Operating Parameters', expanded=True):
    o1, o2, o3, o4 = st.columns(4)
    rpm = o1.number_input('RPM', min_value=0.0, step=10.0, key='rpm')
    wob = o2.number_input('WOB (lbf)', min_value=0.0, step=1000.0, key='wob')
    bit_torque = o3.number_input('Bit Torque (lbf-ft)', min_value=0.0, step=100.0, key='bit_torque_lbf_ft')
    drilling_mode = o4.selectbox('Drilling Mode', ['On Bottom', 'Off Bottom'], key='drilling_mode')

# ============================================================
# RUN ANALYSIS
# ============================================================

st.divider()
if st.button('Run Torque & Drag Analysis', type='primary', use_container_width=True):
    try:
        survey = create_j_type_survey(td, kop, final_inc, eob)
        components = create_components(
            analysis_string, dp_std, hwdp_std, dc_std, td,
            hwdp_length, dc_length, casing_string_spec, casing_string_length
        )

        # Base config object (simplified for the call)
        class BaseConfig:
            def __init__(self):
                self.mud_weight_ppg = mud_weight
                self.mud_fill_fraction = mud_fill_percent / 100.0
                self.td_m = td
                self.casing_shoe_m = casing_shoe
                self.casing_id_in = casing_id
                self.open_hole_id_in = open_hole_id
                self.section_count = section_count
                self.liner1_top_m = liner1_top if section_count >= 3 else 0.0
                self.liner1_shoe_m = liner1_shoe if section_count >= 3 else 0.0
                self.liner1_id_in = liner1_id if section_count >= 3 else 0.0
                self.liner2_top_m = liner2_top if section_count >= 4 else 0.0
                self.liner2_shoe_m = liner2_shoe if section_count >= 4 else 0.0
                self.liner2_id_in = liner2_id if section_count >= 4 else 0.0
                self.r600 = r600
                self.r300 = r300
                self.rpm = rpm
                self.wob_lbf = wob
                self.drilling_mode = drilling_mode
                self.max_iter = 200
                self.tolerance = 1e-4
                self.relaxation = 0.5

        base = BaseConfig()
        base_fc = (ff_casing_min + ff_casing_max) / 2
        base_fo = (ff_open_min + ff_open_max) / 2

        result = run_case(survey, components, base, base_fc, base_fo, bit_torque)
        sensitivity = build_sensitivity(
            survey, components, base,
            ff_casing_min, ff_casing_max, ff_open_min, ff_open_max, bit_torque
        )

        st.success('Analysis completed.')

        # Plots
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_hookload(result, sensitivity, td, 0.0, {'kop': kop, 'eob': eob, 'casing_shoe': casing_shoe, 'td': td, 'section_count': section_count, 'liner1_shoe': liner1_shoe if section_count>=3 else None, 'liner2_shoe': liner2_shoe if section_count>=4 else None}))
        with col2:
            st.pyplot(plot_torque(result, sensitivity, td, 0.0, {'kop': kop, 'eob': eob, 'casing_shoe': casing_shoe, 'td': td, 'section_count': section_count, 'liner1_shoe': liner1_shoe if section_count>=3 else None, 'liner2_shoe': liner2_shoe if section_count>=4 else None}))

        # Report download
        if st.button('Generate PDF Report'):
            pdf_bytes = generate_tnd_report(
                result, sensitivity, td, kop, eob, final_inc,
                casing_shoe, casing_spec, open_hole_id,
                mud_weight, rpm, wob, bit_torque,
                schematic_svg=schematic_svg
            )
            st.download_button(
                label='Download PDF Report',
                data=pdf_bytes,
                file_name='torque_drag_report.pdf',
                mime='application/pdf'
            )

    except Exception as e:
        st.error(f'Error during analysis: {e}')
