"""
Torque & Drag - J-Type Soft-String Force-Balance Solver (V6)

IMPORTANT
---------
This is a RUNNABLE engineering prototype intended for development/validation
before any future extension toward a full Mitchell SPE-163477 discrete beam-column solver.

V1 scope:
- Hookload only: PUW, ROTW, SOW
- No torque calculation
- Minimum Curvature well trajectory
- Discrete string nodes
- Casing/open-hole friction-factor sensitivity
- Iterative axial-force / contact-force calculation
- CSV + PNG output

V6 uses a J-Type-specific 2D soft-string force-balance formulation:
    kappa = dI/ds
    N = |T*kappa + w_b*sin(I)| * ds
    F_f = mu*N
The model is deliberately limited to J-Type geometry with constant azimuth.
It is benchmark-validated against the published Mitchell SPE-163477 cases,
but it is NOT the complete Mitchell discrete beam-column formulation.

Units:
- Internal calculation: SI (m, N, Pa, kg/m3)
- User inputs below are common drilling units.
"""

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# UNIT CONVERSIONS
# ============================================================

LBMFT_TO_KGM = 1.48816394357       # lbm/ft -> kg/m
PPG_TO_KGM3 = 119.8264273          # ppg -> kg/m3
LBF_TO_N = 4.4482216152605
N_TO_LBF = 1.0 / LBF_TO_N
LBF_FT_TO_N_M = LBF_TO_N * 0.3048
N_M_TO_LBF_FT = 1.0 / LBF_FT_TO_N_M
M_TO_FT = 3.280839895013123
DEG_TO_RAD = math.pi / 180.0


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Survey:
    md_m: np.ndarray
    inc_deg: np.ndarray
    azi_deg: np.ndarray


@dataclass
class Component:
    name: str
    od_in: float
    id_in: float
    weight_lbft: float
    length_m: float
    tool_joint_od_in: float = 0.0
    joint_length_m: float = 9.144
    connection_length_m: float = 0.433


@dataclass
class TDConfig:
    mud_weight_ppg: float = 9.2
    mud_fill_fraction: float = 1.0
    td_m: float = 2500.0

    casing_shoe_m: float = 1000.0
    casing_id_in: float = 12.25
    open_hole_id_in: float = 12.25
    section_count: int = 2
    liner1_top_m: float = 500.0
    liner1_shoe_m: float = 1300.0
    liner1_id_in: float = 12.347
    liner2_top_m: float = 1250.0
    liner2_shoe_m: float = 2100.0
    liner2_id_in: float = 8.835

    ff_casing: float = 0.20
    ff_open_hole: float = 0.30

    # Power-Law mud rheology / drilling parameters for V7 torque.
    r600: float = 50.0
    r300: float = 40.0
    rpm: float = 0.0
    wob_lbf: float = 0.0
    drilling_mode: str = "Off Bottom"

    # Solver controls
    max_iter: int = 200
    tolerance: float = 1e-7
    relaxation: float = 0.60

    # Kept for backward compatibility with earlier saved configs.
    include_buoyant_weight_normal: bool = True

    output_dir: str = "td_output"


# ============================================================
# SAMPLE INPUTS
# ============================================================

def build_sample_survey() -> Survey:
    """
    Example J-type trajectory.

    Surface: 0 deg
    KOP:      400 m
    EOB:      700 m
    Final:     30 deg
    TD:      2500 m
    """
    md = np.array([
        0, 400, 500, 600, 700, 1000, 1300, 1600, 1900, 2200, 2500
    ], dtype=float)

    inc = np.array([
        0, 0, 10, 20, 30, 30, 30, 30, 30, 30, 30
    ], dtype=float)

    azi = np.zeros_like(md)

    return Survey(md, inc, azi)


def build_sample_string() -> list[Component]:
    """
    Example drillstring.

    Values can be replaced with the user's actual DP/HWDP/DC data.
    """
    return [
        Component(
            name="Drill Collar",
            od_in=8.000,
            id_in=2.750,
            weight_lbft=100.0,
            length_m=372 * 0.3048,
            tool_joint_od_in=8.000,
        ),
        Component(
            name="HWDP",
            od_in=5.000,
            id_in=3.000,
            weight_lbft=50.0,
            length_m=840 * 0.3048,
            tool_joint_od_in=6.625,
        ),
        Component(
            name="Drill Pipe",
            od_in=5.000,
            id_in=4.276,
            weight_lbft=19.5,
            length_m=10988 * 0.3048,
            tool_joint_od_in=6.625,
        ),
    ]


# ============================================================
# VECTOR / TRAJECTORY FUNCTIONS
# ============================================================

def tangent_vector(inc_deg, azi_deg):
    """
    Minimum-curvature tangent vector:
        t = [cos(Az) sin(I), sin(Az) sin(I), cos(I)]
    """
    i = np.deg2rad(inc_deg)
    a = np.deg2rad(azi_deg)

    return np.column_stack([
        np.cos(a) * np.sin(i),
        np.sin(a) * np.sin(i),
        np.cos(i),
    ])


def minimum_curvature(survey: Survey, step_m: float = 5.0) -> pd.DataFrame:
    """
    Generate a minimum-curvature trajectory.

    Returns:
        MD, X, Y, TVD, Inclination, Azimuth, Dogleg
    """
    md = survey.md_m
    inc = np.deg2rad(survey.inc_deg)
    azi = np.deg2rad(survey.azi_deg)

    x = [0.0]
    y = [0.0]
    z = [0.0]

    dogleg = [0.0]

    for j in range(len(md) - 1):
        ds = md[j + 1] - md[j]

        t1 = np.array([
            np.cos(azi[j]) * np.sin(inc[j]),
            np.sin(azi[j]) * np.sin(inc[j]),
            np.cos(inc[j]),
        ])

        t2 = np.array([
            np.cos(azi[j + 1]) * np.sin(inc[j + 1]),
            np.sin(azi[j + 1]) * np.sin(inc[j + 1]),
            np.cos(inc[j + 1]),
        ])

        dot = np.clip(np.dot(t1, t2), -1.0, 1.0)
        dl = math.acos(dot)
        dogleg.append(np.rad2deg(dl))

        if dl < 1e-12:
            rf = 1.0
        else:
            rf = 2.0 / dl * math.tan(dl / 2.0)

        dN = ds / 2.0 * (
            np.sin(inc[j]) * np.cos(azi[j]) +
            np.sin(inc[j + 1]) * np.cos(azi[j + 1])
        ) * rf

        dE = ds / 2.0 * (
            np.sin(inc[j]) * np.sin(azi[j]) +
            np.sin(inc[j + 1]) * np.sin(azi[j + 1])
        ) * rf

        dTVD = ds / 2.0 * (
            np.cos(inc[j]) +
            np.cos(inc[j + 1])
        ) * rf

        x.append(x[-1] + dN)
        y.append(y[-1] + dE)
        z.append(z[-1] + dTVD)

    return pd.DataFrame({
        "MD_m": md,
        "X_m": x,
        "Y_m": y,
        "TVD_m": z,
        "Inclination_deg": survey.inc_deg,
        "Azimuth_deg": survey.azi_deg,
        "Dogleg_deg": dogleg,
    })


def interpolate_trajectory(survey_df: pd.DataFrame, md_nodes: np.ndarray):
    """
    Interpolate inclination and azimuth to string-node MDs.
    """
    inc = np.interp(
        md_nodes,
        survey_df["MD_m"],
        survey_df["Inclination_deg"]
    )

    azi = np.interp(
        md_nodes,
        survey_df["MD_m"],
        survey_df["Azimuth_deg"]
    )

    return inc, azi


def average_cos_inclination(inc1_deg, inc2_deg):
    """Average cos(I) over a segment whose inclination varies linearly."""
    a = inc1_deg * DEG_TO_RAD
    b = inc2_deg * DEG_TO_RAD
    if abs(b - a) < 1e-12:
        return math.cos(a)
    return (math.sin(b) - math.sin(a)) / (b - a)


def average_sin_inclination(inc1_deg, inc2_deg):
    """Average sin(I) over a segment whose inclination varies linearly."""
    a = inc1_deg * DEG_TO_RAD
    b = inc2_deg * DEG_TO_RAD
    if abs(b - a) < 1e-12:
        return math.sin(a)
    return (math.cos(a) - math.cos(b)) / (b - a)


# ============================================================
# DRILLSTRING NODE GENERATION
# ============================================================

def build_string_nodes(components: list[Component], td_m: float) -> pd.DataFrame:
    """
    Create discrete nodes along the string.

    For this prototype, nodes are placed at:
    - component boundaries
    - approximately 30 m intervals inside the component

    Later this can be changed to exact tool-joint positions.
    """
    rows = []
    md = 0.0

    for comp in components:
        start = md
        end = min(md + comp.length_m, td_m)

        if end <= start:
            break

        # ~30 m discrete interval
        n = max(1, int(math.ceil((end - start) / 30.0)))

        positions = np.linspace(start, end, n + 1)

        for k, p in enumerate(positions):
            row = {
                "MD_m": p,
                "Component": comp.name,
                "OD_in": comp.od_in,
                "ID_in": comp.id_in,
                "Weight_lbft": comp.weight_lbft,
                "ToolJointOD_in": comp.tool_joint_od_in,
                "JointLength_m": comp.joint_length_m,
                "ConnectionLength_m": comp.connection_length_m,
            }
            if rows and abs(p - rows[-1]["MD_m"]) < 1e-9:
                # A component boundary belongs to the component BELOW it
                # for the forward interval. Keep one node, but update its
                # segment properties to the new component. The preceding
                # interval still uses the preceding node's properties.
                rows[-1] = row
                continue
            rows.append(row)

        md = end

    return pd.DataFrame(rows)


# ============================================================
# BUOYANCY / WEIGHT
# ============================================================

def pipe_area_m2(od_in, id_in):
    od = od_in * 0.0254
    id_ = id_in * 0.0254
    return math.pi / 4.0 * (od**2 - id_**2)


def displaced_area_m2(od_in):
    od = od_in * 0.0254
    return math.pi / 4.0 * od**2


def wellbore_id_at_md(md: float, config: TDConfig) -> float:
    """Return the effective internal wellbore diameter at measured depth.

    Liner intervals take precedence over the parent casing where they overlap.
    Below the deepest active shoe, the open-hole diameter applies.
    """
    md = float(md)
    td = max(0.0, float(config.td_m))
    count = int(getattr(config, "section_count", 2))
    if count >= 4 and config.liner2_top_m <= md <= config.liner2_shoe_m:
        return max(0.001, float(config.liner2_id_in))
    if count >= 3 and config.liner1_top_m <= md <= config.liner1_shoe_m:
        return max(0.001, float(config.liner1_id_in))
    open_top = float(config.casing_shoe_m)
    if count >= 3:
        open_top = max(open_top, float(config.liner1_shoe_m))
    if count >= 4:
        open_top = max(open_top, float(config.liner2_shoe_m))
    if md <= min(td, max(0.0, float(config.casing_shoe_m))):
        return max(0.001, float(config.casing_id_in))
    return max(0.001, float(config.open_hole_id_in))


def _wellbore_geometry_intervals(config: TDConfig):
    """Build non-overlapping MD intervals and effective IDs for fluid volume."""
    td = max(0.0, float(config.td_m))
    breaks = {0.0, td, min(td, max(0.0, float(config.casing_shoe_m)))}
    if int(getattr(config, "section_count", 2)) >= 3:
        breaks.update([min(td, max(0.0, float(config.liner1_top_m))), min(td, max(0.0, float(config.liner1_shoe_m)))])
    if int(getattr(config, "section_count", 2)) >= 4:
        breaks.update([min(td, max(0.0, float(config.liner2_top_m))), min(td, max(0.0, float(config.liner2_shoe_m)))])
    ordered = sorted(breaks)
    out = []
    for a, b in zip(ordered[:-1], ordered[1:]):
        if b > a:
            out.append((a, b, wellbore_id_at_md((a+b)/2.0, config)))
    return out


def mud_top_md_for_fill_fraction(config: TDConfig) -> float:
    """Return mud top MD for a wellbore volume fraction filled from bottom up."""
    td = max(0.0, float(config.td_m))
    fill = min(1.0, max(0.0, float(config.mud_fill_fraction)))
    if td <= 0.0 or fill >= 1.0:
        return 0.0
    if fill <= 0.0:
        return td

    intervals = _wellbore_geometry_intervals(config)
    volumes = []
    total_volume = 0.0
    for a, b, diameter_in in intervals:
        area = math.pi / 4.0 * (diameter_in * 0.0254) ** 2
        vol = area * (b-a)
        volumes.append((a, b, area, vol))
        total_volume += vol
    dry_volume = total_volume * (1.0 - fill)
    remaining = dry_volume
    # Dry portion occupies the well from surface downward.
    for a, b, area, vol in volumes:
        if remaining <= vol + 1e-12:
            return min(td, a + remaining / area if area > 0 else b)
        remaining -= vol
    return td


def segment_submerged_fraction(md0: float, md1: float, config: TDConfig) -> float:
    """Fraction of a string segment submerged in mud (0 to 1)."""
    if md1 <= md0:
        return 0.0
    top = mud_top_md_for_fill_fraction(config)
    submerged = max(0.0, md1 - max(md0, top))
    return min(1.0, submerged / (md1 - md0))


def buoyed_weight_N_per_m(row, mud_density, submerged_fraction=1.0):
    """Effective axial weight per unit length for the submerged fraction.

    Above the mud level the string carries air weight.  Below the mud level,
    the standard engineering buoyancy factor is applied.  A partially
    submerged calculation interval is weighted by its submerged fraction.
    """
    air_weight = row["Weight_lbft"] * LBF_TO_N / 0.3048
    steel_density_kgm3 = 65.4 * PPG_TO_KGM3
    sf = min(1.0, max(0.0, float(submerged_fraction)))
    buoyancy_reduction = sf * (mud_density / steel_density_kgm3)
    return air_weight * (1.0 - buoyancy_reduction)


# ============================================================
# CONTACT FORCE MODEL
# ============================================================

def local_dogleg_rad(inc1, azi1, inc2, azi2):
    t1 = tangent_vector(np.array([inc1]), np.array([azi1]))[0]
    t2 = tangent_vector(np.array([inc2]), np.array([azi2]))[0]

    return math.acos(np.clip(np.dot(t1, t2), -1.0, 1.0))


def calculate_jtype_contact_force(
    nodes: pd.DataFrame,
    survey_df: pd.DataFrame,
    tension_N: np.ndarray,
    mud_density: float,
    config: TDConfig | None = None,
):
    """
    Calculate the J-Type 2D soft-string normal contact force per segment.

    For the restricted J-Type geometry (constant azimuth), the local normal
    balance is represented by:

        kappa = dI/ds
        n_per_length = |T*kappa + w_b*sin(I)|
        N_segment = n_per_length * ds

    The first term is the curvature/tension contribution and the second is
    the normal component of buoyed gravity.  This replaces the previous
    opaque ``T * dogleg + normal_weight`` approximation with quantities that
    have explicit physical meaning and units.
    """
    md = nodes["MD_m"].to_numpy(dtype=float)
    inc_deg, azi_deg = interpolate_trajectory(survey_df, md)
    n = len(nodes)
    if n < 2:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

    ds = np.diff(md)
    inc_rad = inc_deg * DEG_TO_RAD

    t_x = np.cos(np.deg2rad(azi_deg)) * np.sin(inc_rad)
    t_y = np.sin(np.deg2rad(azi_deg)) * np.sin(inc_rad)
    t_z = np.cos(inc_rad)
    dot = np.clip(
        t_x[:-1] * t_x[1:] + t_y[:-1] * t_y[1:] + t_z[:-1] * t_z[1:],
        -1.0, 1.0,
    )
    dogleg = np.arccos(dot)
    kappa = np.divide(dogleg, ds, out=np.zeros_like(dogleg), where=ds > 0.0)

    if config is None:
        wb = np.array([buoyed_weight_N_per_m(row, mud_density) for _, row in nodes.iloc[:-1].iterrows()], dtype=float)
    else:
        wb = np.array([
            buoyed_weight_N_per_m(nodes.iloc[i], mud_density, segment_submerged_fraction(md[i], md[i+1], config))
            for i in range(n - 1)
        ], dtype=float)
    avg_sin = np.array(
        [average_sin_inclination(inc_deg[i], inc_deg[i + 1]) for i in range(n - 1)],
        dtype=float,
    )

    t_avg = 0.5 * (tension_N[:-1] + tension_N[1:])
    curvature_normal_per_m = t_avg * kappa
    gravity_normal_per_m = wb * avg_sin
    contact_per_m = np.abs(curvature_normal_per_m + gravity_normal_per_m)
    contact_N = contact_per_m * ds

    contact_full = np.zeros(n)
    contact_full[:-1] = contact_N
    contact_full[-1] = contact_N[-1]

    return contact_full, np.rad2deg(dogleg), curvature_normal_per_m, gravity_normal_per_m


def calculate_contact_forces(
    nodes: pd.DataFrame,
    survey_df: pd.DataFrame,
    tension_N: np.ndarray,
    mud_density: float,
    config: TDConfig,
):
    """Backward-compatible wrapper for the V6 J-Type contact-force engine."""
    contact, dogleg, _, _ = calculate_jtype_contact_force(
        nodes, survey_df, tension_N, mud_density, config
    )
    return contact, dogleg


# ============================================================
# AXIAL FORCE SOLVER
# ============================================================

def is_cased_or_lined_at_md(md, config: TDConfig):
    md = float(md)
    if md <= float(config.casing_shoe_m):
        return True
    if int(getattr(config, "section_count", 2)) >= 3 and float(config.liner1_top_m) <= md <= float(config.liner1_shoe_m):
        return True
    if int(getattr(config, "section_count", 2)) >= 4 and float(config.liner2_top_m) <= md <= float(config.liner2_shoe_m):
        return True
    return False


def friction_factor_for_md(md, config: TDConfig):
    return config.ff_casing if is_cased_or_lined_at_md(md, config) else config.ff_open_hole


def calculate_rotating_weight(nodes: pd.DataFrame, mud_density, survey_df: pd.DataFrame, config: TDConfig | None = None):
    """ROTW = axial component of string weight with partial mud buoyancy."""
    inc, _ = interpolate_trajectory(survey_df, nodes["MD_m"].to_numpy())
    total = 0.0
    for i in range(len(nodes) - 1):
        ds = nodes.iloc[i + 1]["MD_m"] - nodes.iloc[i]["MD_m"]
        sf = segment_submerged_fraction(nodes.iloc[i]['MD_m'], nodes.iloc[i+1]['MD_m'], config) if config is not None else 1.0
        w = buoyed_weight_N_per_m(nodes.iloc[i], mud_density, sf)
        tz = average_cos_inclination(inc[i], inc[i + 1])
        total += w * ds * tz
    return total


def solve_hookload(
    nodes: pd.DataFrame,
    survey_df: pd.DataFrame,
    config: TDConfig,
    direction: str,
):
    """
    Iterative discrete hookload solver.

    The implementation keeps the prototype contact-force approximation, but
    uses vectorized segment calculations so benchmark depth sweeps remain
    practical. Gravity is projected onto the local drillstring tangent:

        dF_axial = w_b cos(I) ds

    This is required for the Mitchell zero-friction benchmark.
    """
    if direction not in ("PUW", "SOW"):
        raise ValueError("direction must be PUW or SOW")

    mud_density = config.mud_weight_ppg * PPG_TO_KGM3

    md = nodes["MD_m"].to_numpy(dtype=float)
    ds = np.diff(md)
    n = len(nodes)

    # Trajectory values at string nodes.
    inc_deg, azi_deg = interpolate_trajectory(survey_df, md)
    inc_rad = inc_deg * DEG_TO_RAD
    azi_rad = azi_deg * DEG_TO_RAD

    # Local tangent vectors and discrete dogleg angle.
    tx = np.cos(azi_rad) * np.sin(inc_rad)
    ty = np.sin(azi_rad) * np.sin(inc_rad)
    tz = np.cos(inc_rad)
    dot = tx[:-1] * tx[1:] + ty[:-1] * ty[1:] + tz[:-1] * tz[1:]
    dogleg = np.arccos(np.clip(dot, -1.0, 1.0))

    # Buoyed weight at each segment's left node.
    weight_N_m = np.empty(n - 1, dtype=float)
    for i in range(n - 1):
        sf = segment_submerged_fraction(md[i], md[i + 1], config)
        weight_N_m[i] = buoyed_weight_N_per_m(nodes.iloc[i], mud_density, sf)

    # Axial gravity component and optional normal gravity component.
    avg_cos = np.empty(n - 1)
    avg_sin = np.empty(n - 1)
    for i in range(n - 1):
        avg_cos[i] = average_cos_inclination(inc_deg[i], inc_deg[i + 1])
        avg_sin[i] = (
            math.cos(inc_rad[i]) - math.cos(inc_rad[i + 1])
        ) / (inc_rad[i + 1] - inc_rad[i]) if abs(inc_rad[i + 1] - inc_rad[i]) > 1e-12 else math.sin(inc_rad[i])

    axial_weight = weight_N_m * ds * avg_cos
    if config.include_buoyant_weight_normal:
        normal_weight = np.abs(weight_N_m * avg_sin * ds)
    else:
        normal_weight = np.zeros(n - 1)

    # Friction factor at each segment.
    mu = np.array([friction_factor_for_md(v, config) for v in md[:-1]], dtype=float)

    # Initial effective tension profile from gravity only.
    tension = np.zeros(n)
    tension[:-1] = np.cumsum(axial_weight[::-1])[::-1]

    # V6 J-Type contact mechanics.  ``normal_weight`` is retained above only
    # for compatibility/diagnostic continuity; contact is now calculated by
    # the explicit curvature + normal-gravity balance below.
    for iteration in range(config.max_iter):
        contact_full, _, _, _ = calculate_jtype_contact_force(
            nodes, survey_df, tension, mud_density, config
        )
        contact = contact_full[:-1]
        friction_force = mu * contact

        if direction == "PUW":
            increments = axial_weight + friction_force
        else:
            increments = axial_weight - friction_force

        new_tension = np.zeros(n)
        new_tension[:-1] = np.cumsum(increments[::-1])[::-1]

        relaxed = (
            config.relaxation * new_tension
            + (1.0 - config.relaxation) * tension
        )

        error = np.max(
            np.abs(relaxed - tension)
            / np.maximum(1.0, np.abs(tension))
        )
        hookload = relaxed[0]
        tension = relaxed

        if error < config.tolerance:
            break

    # Final contact values for diagnostics.
    contact_full, dogleg_from_engine, curvature_normal_per_m, gravity_normal_per_m = calculate_jtype_contact_force(
        nodes, survey_df, tension, mud_density, config
    )
    contact = contact_full[:-1]

    dogleg_full = np.zeros(n)
    dogleg_full[:-1] = np.rad2deg(dogleg)
    if n > 1:
        dogleg_full[-1] = dogleg_full[-2]

    result = nodes.copy()
    result["Inclination_deg"] = inc_deg
    result["Dogleg_deg_interval"] = dogleg_full
    result["ContactForce_N"] = contact_full
    result["ContactForce_lbf"] = contact_full * N_TO_LBF
    result["CurvatureContact_N_per_m"] = np.r_[curvature_normal_per_m, curvature_normal_per_m[-1]]
    result["GravityNormal_N_per_m"] = np.r_[gravity_normal_per_m, gravity_normal_per_m[-1]]
    result["Tension_N"] = tension
    result["Hookload_lbf"] = tension * N_TO_LBF
    result["FrictionFactor"] = np.array([friction_factor_for_md(v, config) for v in md], dtype=float)

    return result, iteration + 1


# ============================================================
# TORSIONAL / SURFACE TORQUE CALCULATION
# ============================================================

def calculate_rotating_tension_profile(
    nodes: pd.DataFrame,
    survey_df: pd.DataFrame,
    mud_density: float,
    config: TDConfig | None = None,
) -> np.ndarray:
    """Return the zero-axial-friction rotating tension profile in N.

    For a rotating string, the V6 axial model uses rotating weight as the
    zero-drag reference.  The tension at each node is therefore the axial
    component of buoyed string weight below that node:

        T(s) = integral_s^TD w_b cos(I) ds

    This profile is used as the first-order contact-force basis for the
    surface-torque calculation.
    """
    md = nodes["MD_m"].to_numpy(dtype=float)
    inc_deg, _ = interpolate_trajectory(survey_df, md)
    n = len(nodes)
    tension = np.zeros(n, dtype=float)
    if n < 2:
        return tension

    ds = np.diff(md)
    wb = np.array([
        buoyed_weight_N_per_m(
            nodes.iloc[i], mud_density,
            segment_submerged_fraction(md[i], md[i+1], config) if config is not None else 1.0
        ) for i in range(n - 1)
    ], dtype=float)
    axial = np.empty(n - 1, dtype=float)
    for i in range(n - 1):
        axial[i] = wb[i] * ds[i] * average_cos_inclination(inc_deg[i], inc_deg[i + 1])
    tension[:-1] = np.cumsum(axial[::-1])[::-1]
    return tension


def power_law_rheology(r600: float, r300: float):
    """Convert standard Fann 600/300 dial readings to Power-Law K and n.

    Internal units are SI:
        shear rate: 1/s
        shear stress: Pa
        K: Pa.s^n

    For standard Fann Model 35 geometry:
        gamma_600 ~= 1022 1/s
        gamma_300 ~= 511 1/s
        tau ~= 0.511 * dial reading Pa
    """
    r600 = float(r600)
    r300 = float(r300)
    if r600 <= 0 or r300 <= 0:
        raise ValueError("R600 and R300 must be greater than zero.")
    n = math.log(r600 / r300) / math.log(2.0)
    gamma_300 = 511.0
    tau_300 = 0.511 * r300
    K = tau_300 / (gamma_300 ** n)
    return float(K), float(n)


def _outer_radius_for_md(md: float, config: TDConfig):
    """Return the effective wellbore inner-wall radius at measured depth."""
    return wellbore_id_at_md(md, config) * 0.0254 / 2.0


def _power_law_couette_torque(K, n, omega, ri, ro, length_m):
    """Viscous torque for a concentric annulus with a rotating inner wall.

    T = 2*pi*L*K*[2*omega/(n*(ri^(-2/n)-ro^(-2/n)))]^n

    Returns N.m.  The calculation is intentionally isolated so its units and
    assumptions can be audited independently from the T&D contact model.
    """
    if length_m <= 0 or omega <= 0 or K <= 0:
        return 0.0
    if ro <= ri:
        raise ValueError("Outer hole/casing radius must be larger than the rotating component radius.")
    denom = n * (ri ** (-2.0 / n) - ro ** (-2.0 / n))
    if denom <= 0:
        raise ValueError("Invalid Power-Law Couette geometry: check n and annular radii.")
    shear_factor = (2.0 * omega / denom) ** n
    return 2.0 * math.pi * length_m * K * shear_factor


def _segment_geometry_for_component(comp: Component):
    """Split a component into body and connection/TJ lengths.

    DP/HWDP use connection OD; DC intentionally has no tool-joint split.
    The number of connections is estimated from standard joint length. The
    connection length is capped so total split length never exceeds input length.
    """
    L = max(0.0, float(comp.length_m))
    if L <= 0:
        return []
    if comp.name.lower().startswith("drill collar") or comp.tool_joint_od_in <= 0:
        return [("Body", L, comp.od_in)]

    joint_len = max(float(comp.joint_length_m), 1e-6)
    conn_len = max(0.0, float(comp.connection_length_m))
    n_joints = max(1, int(math.ceil(L / joint_len)))
    total_conn = min(L, n_joints * conn_len)
    body_len = max(0.0, L - total_conn)
    parts = []
    if body_len > 0:
        parts.append(("Body", body_len, comp.od_in))
    if total_conn > 0:
        parts.append(("Tool Joint/Connection", total_conn, comp.tool_joint_od_in))
    return parts


def calculate_surface_torque(
    nodes: pd.DataFrame,
    survey_df: pd.DataFrame,
    config: TDConfig,
    bit_torque_lbf_ft: float = 0.0,
    torque_ff_casing: float | None = None,
    torque_ff_open_hole: float | None = None,
):
    """V7 preliminary surface torque: Bit + contact + RPM-dependent viscous.

    Contact torque uses the existing V6 J-Type contact-force field and the
    actual contact diameter: DP TJ OD, HWDP connection OD, or DC OD.

    Viscous torque uses a Power-Law concentric-annulus Couette approximation.
    DP and HWDP are split into body and connection/TJ lengths; DC is not split.
    """
    if bit_torque_lbf_ft < 0:
        raise ValueError("Bit torque must be non-negative.")

    mud_density = config.mud_weight_ppg * PPG_TO_KGM3
    md = nodes["MD_m"].to_numpy(dtype=float)
    n_nodes = len(nodes)
    if n_nodes < 2:
        return pd.DataFrame(nodes), {"BitTorque_lbf_ft": float(bit_torque_lbf_ft)}

    K, n_power = power_law_rheology(config.r600, config.r300)
    omega = 2.0 * math.pi * max(0.0, config.rpm) / 60.0
    wob_N = max(0.0, float(config.wob_lbf)) * LBF_TO_N
    if str(config.drilling_mode).lower().startswith("off"):
        wob_N = 0.0

    # Rotating tension includes the bit axial boundary condition (WOB when on bottom).
    tension = calculate_rotating_tension_profile(nodes, survey_df, mud_density, config) + wob_N
    contact_full, dogleg, curvature_normal_per_m, gravity_normal_per_m = calculate_jtype_contact_force(
        nodes, survey_df, tension, mud_density, config
    )
    contact = contact_full[:-1]

    if torque_ff_casing is None:
        torque_ff_casing = config.ff_casing
    if torque_ff_open_hole is None:
        torque_ff_open_hole = config.ff_open_hole
    mu = np.array([float(torque_ff_casing) if is_cased_or_lined_at_md(v, config) else float(torque_ff_open_hole) for v in md[:-1]], dtype=float)

    # Contact lever arm: DP/HWDP connection OD; DC OD.  At component boundaries,
    # the segment uses the properties stored on its left node.
    comp_name = nodes["Component"].astype(str).to_numpy()[:-1]
    tj_od = nodes["ToolJointOD_in"].to_numpy(dtype=float)[:-1]
    pipe_od = nodes["OD_in"].to_numpy(dtype=float)[:-1]
    is_dc = np.array(["drill collar" in name.lower() for name in comp_name], dtype=bool)
    contact_od = np.where(is_dc, pipe_od, np.where(tj_od > 0.0, tj_od, pipe_od))
    effective_radius_m = contact_od * 0.0254 / 2.0
    contact_torque_Nm = mu * contact * effective_radius_m

    # Viscous torque is evaluated by component/geometry intervals, using the
    # same total component lengths as the string. This avoids assigning TJ OD
    # to the entire DP/HWDP body.
    viscous_rows = []
    viscous_total_Nm = 0.0
    viscous_by_component = {}
    component_lengths = {}
    # The node table may contain one node per component boundary. Reconstruct
    # component lengths from MD boundaries, including the final segment.
    boundaries = []
    for idx, row in nodes.iterrows():
        if idx == 0 or row["Component"] != nodes.iloc[idx - 1]["Component"]:
            boundaries.append((float(row["MD_m"]), row))
    boundaries.append((float(nodes.iloc[-1]["MD_m"]), None))

    for j in range(len(boundaries) - 1):
        start_md, row = boundaries[j]
        end_md = boundaries[j + 1][0]
        L = max(0.0, end_md - start_md)
        if L <= 0:
            continue
        comp = Component(
            name=str(row["Component"]), od_in=float(row["OD_in"]), id_in=float(row["ID_in"]),
            weight_lbft=float(row["Weight_lbft"]), length_m=L,
            tool_joint_od_in=float(row["ToolJointOD_in"]),
            joint_length_m=float(row.get("JointLength_m", 9.144)),
            connection_length_m=float(row.get("ConnectionLength_m", 0.433)),
        )
        outer_r = _outer_radius_for_md(start_md + 1e-6, config)
        component_viscous = 0.0
        for part, part_len, inner_od in _segment_geometry_for_component(comp):
            ri = inner_od * 0.0254 / 2.0
            if ri >= outer_r:
                raise ValueError(
                    f"{comp.name} {part} OD {inner_od:.3f} in is not smaller than the active hole/casing ID."
                )
            t_part = _power_law_couette_torque(K, n_power, omega, ri, outer_r, part_len)
            component_viscous += t_part
            viscous_rows.append({"Component": comp.name, "Part": part, "Length_m": part_len,
                                 "OD_in": inner_od, "OuterID_in": outer_r * 2.0 / 0.0254,
                                 "ViscousTorque_Nm": t_part})
        viscous_by_component[str(comp.name)] = component_viscous
        component_lengths[str(comp.name)] = L
        viscous_total_Nm += component_viscous

    # Distribute each component's total viscous resistance over its MD intervals
    # so the transmitted-torque curve starts at bit torque and accumulates both
    # contact and viscous resistance toward surface.
    segment_viscous_Nm = np.zeros(n_nodes - 1, dtype=float)
    for i in range(n_nodes - 1):
        name = str(nodes.iloc[i]["Component"])
        Lc = component_lengths.get(name, 0.0)
        if Lc > 0:
            segment_viscous_Nm[i] = viscous_by_component.get(name, 0.0) * (md[i+1] - md[i]) / Lc

    torque_at_node = np.zeros(n_nodes, dtype=float)
    torque_at_node[-1] = bit_torque_lbf_ft * LBF_FT_TO_N_M
    torque_at_node[:-1] = (
        torque_at_node[-1]
        + np.cumsum((contact_torque_Nm + segment_viscous_Nm)[::-1])[::-1]
    )

    result = nodes.copy()
    inc_deg, _ = interpolate_trajectory(survey_df, md)
    result["Inclination_deg"] = inc_deg
    result["Tension_ROT_N"] = tension
    result["Tension_ROT_lbf"] = tension * N_TO_LBF
    result["ContactForce_N"] = contact_full
    result["ContactForce_lbf"] = contact_full * N_TO_LBF
    result["TorqueFrictionFactor"] = np.r_[mu, mu[-1]]
    result["ContactTorqueRadius_in"] = np.r_[effective_radius_m / 0.0254, effective_radius_m[-1] / 0.0254]
    result["ContactTorqueLoss_lbf_ft_segment"] = np.r_[contact_torque_Nm * N_M_TO_LBF_FT, contact_torque_Nm[-1] * N_M_TO_LBF_FT]
    result["ViscousTorqueLoss_lbf_ft_segment"] = np.r_[segment_viscous_Nm * N_M_TO_LBF_FT, segment_viscous_Nm[-1] * N_M_TO_LBF_FT]
    result["TorqueAtNode_lbf_ft"] = torque_at_node * N_M_TO_LBF_FT

    viscous_df = pd.DataFrame(viscous_rows)
    if not viscous_df.empty:
        viscous_df["ViscousTorque_lbf_ft"] = viscous_df["ViscousTorque_Nm"] * N_M_TO_LBF_FT

    summary = {
        "BitTorque_lbf_ft": float(bit_torque_lbf_ft),
        "ContactTorque_lbf_ft": float(np.sum(contact_torque_Nm) * N_M_TO_LBF_FT),
        "ViscousTorque_lbf_ft": float(viscous_total_Nm * N_M_TO_LBF_FT),
        "SurfaceTorque_lbf_ft": float(torque_at_node[0] * N_M_TO_LBF_FT),
        "R600": float(config.r600),
        "R300": float(config.r300),
        "PowerLaw_K_Pa_s_n": float(K),
        "PowerLaw_n": float(n_power),
        "RPM": float(config.rpm),
        "WOB_lbf": float(config.wob_lbf if str(config.drilling_mode).lower().startswith("on") else 0.0),
        "DrillingMode": str(config.drilling_mode),
        "ViscousBreakdown": viscous_df,
    }
    return result, summary


# ============================================================
# DEPTH SWEEP
# ============================================================

def _active_string_at_depth(components: list[Component], depth_m: float) -> list[Component]:
    """Build the *actual* surface-to-bit string present at Run MD.

    ``components`` are defined in final-string order, surface to bit:

        DP -> HWDP -> DC

    During RIH/while drilling at an intermediate depth, however, the bottom
    of the final string enters the well first.  Therefore the active string
    at Run MD ``D`` is the **bottom D metres of the final drillstring**,
    expressed again in surface-to-bit order for the mechanical solve.

    Example for TD=1,300 m, DP=973 m, HWDP=214 m and DC=113 m:

        D=900 m  -> DP 573 + HWDP 214 + DC 113
        D=1,100 m -> DP 773 + HWDP 214 + DC 113
        D=1,300 m -> DP 973 + HWDP 214 + DC 113

    This is the key distinction from the previous implementation, which
    incorrectly filled the active string from the surface downward and thus
    produced DP-only calculations until the full DP length was reached.

    Every Run MD is consequently solved as one continuous active string,
    including all BHA components that are physically in the hole at that
    depth.
    """
    depth = max(0.0, float(depth_m))
    total_length = sum(max(0.0, float(c.length_m)) for c in components)

    if depth > total_length + 1e-8:
        raise ValueError(
            f"Run MD {depth:.2f} m exceeds available drillstring length "
            f"{total_length:.2f} m."
        )
    if depth <= 1e-12:
        return []

    # The portion in the hole is the suffix of the final string adjacent to
    # the bit.  ``start_pos`` is its top position measured from the surface
    # along the final string.
    start_pos = total_length - depth
    local_components: list[Component] = []
    cursor = 0.0

    for comp in components:
        comp_len = max(0.0, float(comp.length_m))
        comp_start = cursor
        comp_end = cursor + comp_len
        cursor = comp_end

        overlap_start = max(comp_start, start_pos)
        overlap_end = min(comp_end, total_length)
        take = overlap_end - overlap_start

        if take > 1e-10:
            local_components.append(
                Component(
                    name=comp.name,
                    od_in=comp.od_in,
                    id_in=comp.id_in,
                    weight_lbft=comp.weight_lbft,
                    length_m=take,
                    tool_joint_od_in=comp.tool_joint_od_in,
                    joint_length_m=comp.joint_length_m,
                    connection_length_m=comp.connection_length_m,
                )
            )

    active_total = sum(c.length_m for c in local_components)
    if abs(active_total - depth) > 1e-7:
        raise RuntimeError(
            f"Active-string construction error at Run MD {depth:.3f} m: "
            f"constructed {active_total:.6f} m instead of {depth:.6f} m."
        )

    return local_components



def insert_well_section_boundaries(nodes: pd.DataFrame, config: TDConfig) -> pd.DataFrame:
    """Insert exact casing/liner TOL and shoe nodes into the active string grid."""
    if nodes.empty:
        return nodes
    md = nodes["MD_m"].to_numpy(dtype=float)
    bounds = [float(config.casing_shoe_m)]
    if int(getattr(config, "section_count", 2)) >= 3:
        bounds += [float(config.liner1_top_m), float(config.liner1_shoe_m)]
    if int(getattr(config, "section_count", 2)) >= 4:
        bounds += [float(config.liner2_top_m), float(config.liner2_shoe_m)]
    add_rows = []
    for depth in bounds:
        if depth <= md[0] or depth >= md[-1] or np.min(np.abs(md-depth)) < 1e-7:
            continue
        left_idx = int(np.searchsorted(md, depth, side="right") - 1)
        row = nodes.iloc[left_idx].to_dict()
        row["MD_m"] = depth
        add_rows.append(row)
    if not add_rows:
        return nodes
    out = pd.concat([nodes, pd.DataFrame(add_rows)], ignore_index=True)
    return out.sort_values("MD_m").drop_duplicates(subset=["MD_m"], keep="last").reset_index(drop=True)

def solve_depth_sweep(
    survey: Survey,
    components: list[Component],
    config: TDConfig,
    bit_torque_lbf_ft: float = 0.0,
):
    """Calculate depth-dependent PUW, ROTW, SOW and surface torque.

    At each Run MD the complete portion of the drillstring that has entered
    the well is reconstructed from surface to bit.  Hookload and surface
    torque are then solved from that same active string.  This makes the
    result answer the operational question: *when drilling at depth X, what
    are the expected hookloads and surface torque at that moment?*
    """
    trajectory = minimum_curvature(survey)
    well_td = float(survey.md_m[-1])
    total_string = sum(float(c.length_m) for c in components)
    if total_string <= 1e-9:
        raise ValueError("The analyzed string has zero total length.")

    # For a drillstring, total string length normally equals well TD.
    # For casing/liner running, the string can be shorter than well TD; in
    # that case the depth sweep ends at the available string length.
    td = min(well_td, total_string)

    # Keep the 25 m calculation interval used by the current prototype and
    # always append exact analysis TD.
    depth_grid = np.unique(
        np.concatenate([
            np.arange(100.0, td, 25.0),
            np.array([td]),
        ])
    )

    output = []

    for depth in depth_grid:
        local_components = _active_string_at_depth(components, depth)
        nodes = build_string_nodes(local_components, depth)
        nodes = insert_well_section_boundaries(nodes, config)

        puw, puw_iter = solve_hookload(nodes, trajectory, config, "PUW")
        sow, sow_iter = solve_hookload(nodes, trajectory, config, "SOW")

        mud_density = config.mud_weight_ppg * PPG_TO_KGM3
        rotw = calculate_rotating_weight(nodes, mud_density, trajectory, config)

        # Surface torque is evaluated at this same Run MD using exactly the
        # same active string.  Therefore the torque represents the surface
        # torque while drilling at this depth, not only the TD torque.
        _, torque_summary = calculate_surface_torque(
            nodes, trajectory, config, bit_torque_lbf_ft=float(bit_torque_lbf_ft)
        )

        output.append({
            "MD_m": depth,
            "PUW_lbf": float(puw.iloc[0]["Hookload_lbf"]),
            "ROTW_lbf": float(rotw * N_TO_LBF),
            "SOW_lbf": float(sow.iloc[0]["Hookload_lbf"]),
            "SurfaceTorque_lbf_ft": float(torque_summary["SurfaceTorque_lbf_ft"]),
            "BitTorque_lbf_ft": float(torque_summary["BitTorque_lbf_ft"]),
            "ContactTorque_lbf_ft": float(torque_summary["ContactTorque_lbf_ft"]),
            "ViscousTorque_lbf_ft": float(torque_summary["ViscousTorque_lbf_ft"]),
            "PUW_iterations": puw_iter,
            "SOW_iterations": sow_iter,
            "ActiveString_m": float(depth),
            "DP_in_hole_m": sum(c.length_m for c in local_components if c.name == "Drill Pipe"),
            "HWDP_in_hole_m": sum(c.length_m for c in local_components if c.name == "HWDP"),
            "DC_in_hole_m": sum(c.length_m for c in local_components if c.name == "Drill Collar"),
            "Casing_in_hole_m": sum(c.length_m for c in local_components if "casing" in c.name.lower()),
            "Liner_in_hole_m": sum(c.length_m for c in local_components if "liner" in c.name.lower()),
        })

    return pd.DataFrame(output)


# ============================================================
# SENSITIVITY
# ============================================================

def run_sensitivity(
    survey: Survey,
    components: list[Component],
    base_config: TDConfig,
    casing_values,
    open_hole_values,
):
    """
    Full rerun for every FF pair.
    """
    cases = []

    for ff_casing in casing_values:
        for ff_open in open_hole_values:
            cfg = TDConfig(
                casing_shoe_m=base_config.casing_shoe_m,
                casing_id_in=base_config.casing_id_in,
                open_hole_id_in=base_config.open_hole_id_in,
                section_count=base_config.section_count,
                liner1_top_m=base_config.liner1_top_m,
                liner1_shoe_m=base_config.liner1_shoe_m,
                liner1_id_in=base_config.liner1_id_in,
                liner2_top_m=base_config.liner2_top_m,
                liner2_shoe_m=base_config.liner2_shoe_m,
                liner2_id_in=base_config.liner2_id_in,
                mud_weight_ppg=base_config.mud_weight_ppg,
                mud_fill_fraction=base_config.mud_fill_fraction,
                td_m=base_config.td_m,
                ff_casing=ff_casing,
                ff_open_hole=ff_open,
                max_iter=base_config.max_iter,
                tolerance=base_config.tolerance,
                relaxation=base_config.relaxation,
                include_buoyant_weight_normal=(
                    base_config.include_buoyant_weight_normal
                ),
            )

            df = solve_depth_sweep(
                survey,
                components,
                cfg,
            )

            df["FF_Casing"] = ff_casing
            df["FF_OpenHole"] = ff_open

            cases.append(df)

    return pd.concat(cases, ignore_index=True)


# ============================================================
# PLOTS
# ============================================================

def plot_base_result(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))

    plt.plot(
        df["PUW_lbf"],
        df["MD_m"],
        label="PUW",
        linewidth=2.0,
    )

    plt.plot(
        df["ROTW_lbf"],
        df["MD_m"],
        label="ROTW",
        linewidth=2.0,
    )

    plt.plot(
        df["SOW_lbf"],
        df["MD_m"],
        label="SOW",
        linewidth=2.0,
    )

    plt.gca().invert_yaxis()
    plt.xlabel("Hookload (lbf)")
    plt.ylabel("Measured Depth (m)")
    plt.title("Torque & Drag Prototype - Hookload")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "hookload_base.png",
        dpi=200,
    )

    plt.show()


def plot_sensitivity(
    sensitivity_df,
    output_dir,
    variable="FF_OpenHole",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))

    for value in sorted(sensitivity_df[variable].unique()):
        sub = sensitivity_df[
            sensitivity_df[variable] == value
        ]

        # Use the first casing FF as the fixed value if plotting OH FF
        if variable == "FF_OpenHole":
            label = (
                f"OH FF={value:.2f}, "
                f"Casing FF={sub['FF_Casing'].iloc[0]:.2f}"
            )
        else:
            label = (
                f"Casing FF={value:.2f}, "
                f"OH FF={sub['FF_OpenHole'].iloc[0]:.2f}"
            )

        plt.plot(
            sub["PUW_lbf"],
            sub["MD_m"],
            label=label,
        )

    plt.gca().invert_yaxis()
    plt.xlabel("Pickup Hookload (lbf)")
    plt.ylabel("Measured Depth (m)")
    plt.title(f"PUW Sensitivity - {variable}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / f"puw_sensitivity_{variable}.png",
        dpi=200,
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("TORQUE & DRAG - HYBRID-INSPIRED PROTOTYPE")
    print("=" * 72)

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    survey = build_sample_survey()
    components = build_sample_string()

    config = TDConfig(
        mud_weight_ppg=9.8,
        casing_shoe_m=1000.0,
        ff_casing=0.20,
        ff_open_hole=0.30,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # TRAJECTORY
    # --------------------------------------------------------

    trajectory = minimum_curvature(survey)

    trajectory.to_csv(
        output_dir / "trajectory_minimum_curvature.csv",
        index=False,
    )

    # --------------------------------------------------------
    # BASE CASE
    # --------------------------------------------------------

    print("\nRunning base case...")

    base = solve_depth_sweep(
        survey,
        components,
        config,
    )

    base.to_csv(
        output_dir / "hookload_base.csv",
        index=False,
    )

    print("\nBASE CASE")
    print("-" * 72)
    print(
        base[
            [
                "MD_m",
                "PUW_lbf",
                "ROTW_lbf",
                "SOW_lbf",
            ]
        ].tail(10).to_string(index=False)
    )

    # --------------------------------------------------------
    # SENSITIVITY
    # --------------------------------------------------------

    print("\nRunning friction-factor sensitivity...")

    sensitivity = run_sensitivity(
        survey,
        components,
        config,
        casing_values=[0.10, 0.20, 0.30],
        open_hole_values=[0.20, 0.30, 0.40],
    )

    sensitivity.to_csv(
        output_dir / "hookload_sensitivity.csv",
        index=False,
    )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    plot_base_result(base, output_dir)

    # Keep one casing FF fixed while showing OH sensitivity
    oh_sens = sensitivity[
        np.isclose(sensitivity["FF_Casing"], config.ff_casing)
    ]

    plot_sensitivity(
        oh_sens,
        output_dir,
        variable="FF_OpenHole",
    )

    print("\nFiles written to:")
    print(output_dir.resolve())

    print("\nDone.")


if __name__ == "__main__":
    main()
