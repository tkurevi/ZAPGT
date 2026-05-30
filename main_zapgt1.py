"""
main_zapgt1.py
==============

Nodal-analysis driver for the Zaprešić GT-1 (ZapGT-1) well.

ZapGT-1 is a vertical exploratory geothermal well in the Sava
depression, Croatia.  The well intersects a highly-fractured dolomite
at 1537-1698 m TVD (top dolomite, casing shoe to lithology change).
The 96-hour pressure build-up interpreted by the operator's reservoir
engineering team (KAPPA software) returned an exceptionally high
permeability k = 9128 mD and a skin of +7.4, with three boundaries
detected (faults / property changes).

The well has been tested in two phases:
  1.  Nitrogen-lifted eruptive flow with a 22.23 mm fixed choke, ~30
      m^3/h (8.3 L/s) with drawdown ~ 0.06 bar.
  2.  ESP-driven test (Crosco, Feb 11 2026):  ESP at 517 m TVD inside
      the 9-5/8" casing, coiled tubing 88.6 / 76 mm OD/ID providing
      the production conduit above the pump.  24-hour test stepping
      30 -> 35 -> ... -> 62.5 Hz with rates 7 to 38.5 L/s.

Configs provided
----------------
* DEFAULT_CONFIG     -- linear-PI infinite-acting IPR with PBU
                         parameters, no pump.  Reproduces the
                         "WellPerform overpredicts" anomaly.
* CALIBRATED_CONFIG  -- adds Forchheimer non-Darcy D_nonDarcy on the
                         dolomite, calibrated to the 60 Hz steady-state
                         field point (q = 38 L/s, P_intake = 53.4 bar,
                         WHP = 25.8 bar).  Uses the three-conduit
                         wellbore (CT above ESP, casing in middle,
                         slotted liner at bottom).
"""
from __future__ import annotations
import os, sys, json, copy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pvt import ppm_to_molality, bar_to_Pa, Pa_to_bar, m3h_to_ls, m3h_to_m3d
from vlp import WellGeometry, ThermalContext, FluidStream, march_VLP
from ipr_multilayer import Layer, CommingledReservoir
from nodal_multi import (solve_operating_point_multi,
                          print_operating_point_multi,
                          export_summary_csv_multi,
                          export_layer_breakdown_csv,
                          ipr_curve_commingled,
                          vlp_curve_at_depth)


# =====================================================================
# DEFAULT_CONFIG -- linear-PI infinite-acting IPR, no pump, no Forchheimer
# =====================================================================
DEFAULT_CONFIG = dict(
    case_name='ZapGT-1 eruptive (5" DP + 7" liner, Forchheimer D=1000)',

    # ---- Reservoir (single layer: top dolomite) -------------------
    layers=[
        dict(
            name='dolomite',
            top_depth_m=1537.0,
            bottom_depth_m=1698.0,
            h_net_m=161.0,         # 1537-1698 m (user-specified)
            k_md=9128.92,          # KAPPA-fitted permeability (PBU)
            P_res_bar=159.92,      # @ z_mid = 1617.5 m  (user-specified)
            T_res_C=71.0,
            r_w=0.108,             # 8-1/2" bit -> radius 108 mm
            r_e=700.0,             # half of 1.4 km inter-well spacing
            NaCl_ppm=1320.0,       # very low salinity (near-fresh water)
            mode='darcy_radial',
            regime='pss',
            skin_total=7.396,      # from PBU
            porosity=0.033,        # 3.3 %
            t_prod_days=4.0,       # 96-hour PBU duration
            D_nonDarcy=1000.0,     # Forchheimer, calibrated from ESP test
        ),
    ],

    # ---- Commingled config (single layer is OK, just collapses) ---
    commingled=dict(
        reference_depth_m=1617.5,  # mid of producing interval
        wellbore_density_kg_m3=983.0,  # ~71 C, 1320 ppm NaCl
    ),

    # ---- Wellbore geometry (eruptive case by default: 3-segment) -----
    # Based on the HDI day-by-day equipment record for ZapGT-1:
    #
    #   0 -> 1475 m: 5" NC50 X-95 19.5# drill pipe (FID ~ 108.6 mm)
    #                with RTTS packer 9-5/8" set at 1475 m
    #   1475 -> 1485 m: 9-5/8" 47# FB-2 production casing alone
    #                   (FID = 220.5 mm)
    #   1485 -> 1617.5 m: 7" 29# BTC slotted liner (FID = 154.8 mm)
    #                     hung from 1485 m, slotted to 2489 m TD
    #
    # During Day 2-3 eruptive testing the drill string was IN the well
    # with the packer active.  Production logging on Days 5-6 (when q
    # = 8.3 L/s was measured) was performed in dynamic eruptive
    # conditions WITH the drill string in place (slick line / wireline
    # in the well; documented in the day-by-day record).
    well=dict(
        tubing_ID=0.1086,         # 5" NC50 19.5#  --  default upper conduit
        tubing_OD=0.127,
        roughness=46e-6,
        # 3-conduit segments (overrides 2-conduit when present)
        segments=[
            dict(z_top=0.0,    z_bot=1475.0, ID=0.1086, OD=0.127,
                 roughness=46e-6),     # 5" DP (NC50 X-95 19.5#)
            dict(z_top=1475.0, z_bot=1485.0, ID=0.2205, OD=0.2445,
                 roughness=46e-6),     # 9-5/8" 47# casing (only 10 m)
            dict(z_top=1485.0, z_bot=1617.5, ID=0.1548, OD=0.1778,
                 roughness=46e-6),     # 7" 29# BTC slotted liner
        ],
    ),

    # ---- Thermal --------------------------------------------------
    thermal=dict(
        T_surface_C=12.0,          # north-Croatia surface T
        geo_gradient_K_m=0.0365,   # gives ~71 C at z=1617.5 m
        U_overall=20.0,
        t_prod_days=1.0,           # 24 h test
    ),

    # ---- Fluid (water, no gas) ------------------------------------
    fluid=dict(
        GWR_std=0.0,               # ESP test: NO gas observed
        NaCl_ppm=1320.0,
        w_CO2=0.0,
    ),

    # ---- Operating: eruptive test, WHP near atmospheric -----------
    # WHP_gauge = 1.22 bar (measured during PLT period of Days 5-6),
    # so WHP_absolute = 1.0 + 1.22 = 2.22 bar.  The downstream surface
    # system was a 50 m run of 76.2 mm pipe with a 50.8 mm restriction
    # at the lagoon inlet; this restriction sets the wellhead back-
    # pressure but is not modelled explicitly in the wellbore VLP --
    # the measured WHP = 2.22 bar is used as the upstream boundary.
    operating=dict(
        WHP_bar=2.22,
        n_segments=50,
        pump=None,                 # eruptive, no pump
    ),
)


# =====================================================================
# CALIBRATED_CONFIG -- adds Forchheimer D + 3-conduit ESP wellbore
# =====================================================================
CALIBRATED_CONFIG = copy.deepcopy(DEFAULT_CONFIG)
CALIBRATED_CONFIG['case_name'] = (
    'ZapGT-1 (calibrated: Forchheimer non-Darcy + ESP at 517 m)')

# Forchheimer D fitted to the 60 Hz field-stable ESP point.
# At q = 38 L/s, P_intake = 53.4 bar at z = 517 m TVD, the
# linear-PI infinite-acting J from the PBU (k=9128, h=161, s=7.4,
# re=700) is ~ 140 L/s/bar.  PROSPER fitted an APPARENT J of
# 40.5 L/s/bar from the ESP-test points, which is the Forchheimer-
# reduced effective J at q ~ 38 L/s.  Solving
#     1 + D * q / denom0  =  J0 / J_eff  =  140 / 40.5  =  3.46
# with q = 0.038 m^3/s and denom0 = ln(re/rw) - 0.75 + s = 15.42
# gives  D ~ 1000  (m^3/s)^-1.
for L in CALIBRATED_CONFIG['layers']:
    if L['name'] == 'dolomite':
        L['D_nonDarcy'] = 1000.0   # (m^3/s)^-1

# Three-conduit ESP wellbore (CT above pump, casing in middle, liner
# at bottom).  Default keeps the eruptive 2-segment well; the ESP
# version is added by build_well_to_ref when operating['pump'] is set.
# The 'segments' field is constructed dynamically in build_well_to_ref.

# Operating conditions for the 60 Hz steady ESP point (target match)
# WHP = 25.8 bar (field-measured at 60 Hz, choke wide open).
# dP_pump tuned so the nodal operating point lands at q = 38 L/s.
# In the three-conduit model the CT friction at 38 L/s in 76 mm ID
# is the dominant loss above the pump (~ 35 bar), so the model
# requires a slightly higher dP_pump than a field back-calc using
# nominal Moody friction would give.  The match is q = 38.95 L/s,
# Pwf @ z_ref = 158.97 bar -- within +2.5% on rate and within
# 1 bar on Pwf relative to field 38 L/s / ~159 bar.
CALIBRATED_CONFIG['operating']['WHP_bar'] = 25.8
CALIBRATED_CONFIG['operating']['pump'] = dict(
    z_intake_m=517.0,
    dP_bar=70.0,
)
# Mark this config as ESP-test mode (used by build_well_to_ref)
CALIBRATED_CONFIG['well']['_mode'] = 'esp_test'

# Three-segment well for ESP test
CALIBRATED_CONFIG['well']['esp_segments'] = [
    dict(z_top=0.0,    z_bot=517.0,  ID=0.076,  OD=0.0886,
         roughness=46e-6),         # Coiled tubing 88.6/76 mm
    dict(z_top=517.0,  z_bot=1485.0, ID=0.2204, OD=0.2445,
         roughness=46e-6),         # 9-5/8" 47# casing
    dict(z_top=1485.0, z_bot=1617.5, ID=0.110,  OD=0.127,
         roughness=46e-6),         # 5" slotted liner
]


# =====================================================================
# Builders
# =====================================================================
def build_commingled(config):
    layers = [Layer(**L) for L in config['layers']]
    return CommingledReservoir(
        layers,
        reference_depth_m=config['commingled'].get('reference_depth_m'),
        wellbore_density_kg_m3=config['commingled'].get(
            'wellbore_density_kg_m3'))


def build_well_to_ref(config, z_ref):
    """Build a WellGeometry.  Selects 3-segment ESP geometry vs 2-segment
    eruptive geometry based on whether the operating dict has 'pump'."""
    w = config['well']
    if config['operating'].get('pump') is not None and 'esp_segments' in w:
        segments = w['esp_segments']
    else:
        segments = w.get('segments')
    return WellGeometry(
        depth_TVD=float(z_ref),
        tubing_ID=w['tubing_ID'],
        tubing_OD=w.get('tubing_OD'),
        roughness=w.get('roughness', 46e-6),
        segments=segments,
    )


def build_thermal(config, z_ref):
    th = config['thermal']
    return ThermalContext(
        T_surface=273.15 + th['T_surface_C'],
        geo_gradient=th['geo_gradient_K_m'],
        T_BH=273.15 + th['T_surface_C'] + th['geo_gradient_K_m'] * z_ref,
        U_overall=th.get('U_overall', 20.0),
        k_formation=th.get('k_earth', 2.5),
        alpha_formation=th.get('alpha_earth', 1.0e-6),
        time_seconds=th.get('t_prod_days', 30.0) * 86400.0,
    )


# =====================================================================
# Full analysis run
# =====================================================================
def run_full_analysis(config, out_dir, verbose=True):
    os.makedirs(out_dir, exist_ok=True)

    comm = build_commingled(config)
    well = build_well_to_ref(config, comm.z_ref)
    thermal = build_thermal(config, comm.z_ref)
    GWR    = config['fluid']['GWR_std']
    m_NaCl = ppm_to_molality(config['fluid']['NaCl_ppm'])
    WHP    = config['operating']['WHP_bar']
    n_seg  = config['operating'].get('n_segments', 50)

    pump_cfg = config['operating'].get('pump')
    pump = None
    if pump_cfg is not None:
        pump = dict(
            z_intake_m=float(pump_cfg['z_intake_m']),
            dP_Pa=bar_to_Pa(float(pump_cfg['dP_bar'])),
        )

    if verbose:
        print("=" * 70)
        print(f"CASE: {config['case_name']}")
        print("=" * 70)
        print(comm.describe())
        print(f"\nWell: {well}")
        print(f"GWR_std = {GWR}, NaCl_avg = {config['fluid']['NaCl_ppm']:.0f} "
              f"ppm, WHP = {WHP:.2f} bar")
        if pump is not None:
            print(f"ESP at {pump['z_intake_m']:.0f} m, dP_pump = "
                  f"{Pa_to_bar(pump['dP_Pa']):.2f} bar")
        else:
            print("No pump (eruptive)")

    # Solve operating point
    op = solve_operating_point_multi(
        WHP, well, GWR, m_NaCl, thermal, comm,
        n_segments=n_seg, pump=pump,
        q_min_m3h=0.5, q_max_m3h=2.0 * comm.AOF() * 3600.0,
    )
    if verbose:
        print_operating_point_multi(op)

    # Save summary
    export_summary_csv_multi(op, os.path.join(out_dir, 'summary.csv'))
    export_layer_breakdown_csv(
        comm, os.path.join(out_dir, 'layer_breakdown.csv'))

    # Save config used
    cfg_clean = copy.deepcopy(config)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(cfg_clean, f, indent=2, default=str)

    return op, comm, well, thermal


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) > 1 and sys.argv[1] == 'calibrated':
        cfg = CALIBRATED_CONFIG
        out_dir = os.path.join(SCRIPT_DIR, 'zapgt1_run_calibrated')
    else:
        cfg = DEFAULT_CONFIG
        out_dir = os.path.join(SCRIPT_DIR, 'zapgt1_run_eruptive')
    print(f"Output directory: {out_dir}")
    run_full_analysis(cfg, out_dir)
