"""
plotting.py
===========

Matplotlib plots for the geothermal nodal-analysis tool.

Conventions:
  * Depth is plotted on the y-axis with the surface at the TOP and
    bottomhole at the BOTTOM (the Petroleum standard).  Y-axis is
    inverted, so depth values increase downward.
  * Pressure on x-axis is in bar; temperature in degC.
  * IPR and VLP curves use the Cartesian convention: q on x, Pwf on y.
  * Sensitivity sweeps plot the swept value on x and a chosen
    dependent variable on y.

All functions accept a `save_path=None` argument; if provided, the
figure is saved and closed (no GUI required).  Otherwise the figure
is returned and shown via the caller's plt.show().
"""

from __future__ import annotations
import numpy as np
try:
    import plot_style
except Exception:
    pass

import matplotlib
matplotlib.use('Agg')           # no GUI required
import matplotlib.pyplot as plt

from pvt import bar_to_Pa, Pa_to_bar, fmt_q, m3h_to_ls, m3h_to_m3d
from vlp import march_VLP, FluidStream
from nodal import ipr_curve, vlp_curve, solve_operating_point


# Common style helpers
def _style():
    plt.rcParams.update({
        'figure.dpi':   110,
        'savefig.dpi':  140,
        'font.size':     10,
        'axes.grid':     True,
        'grid.alpha':    0.3,
        'lines.linewidth': 1.6,
    })

_style()


# =====================================================================
# 1. Pressure profile vs depth
# =====================================================================
def plot_pressure_profile(profile, title=None, save_path=None,
                          mark_bubble_point=True):
    """Plot P(z) with depth on y-axis (inverted)."""
    fig, ax = plt.subplots(figsize=(5.5, 7.0))
    z   = profile['z']
    P   = profile['P'] * 1e-5
    ax.plot(P, z, 'C0-', label='Wellbore P(z)')
    ax.invert_yaxis()
    ax.set_xlabel('Pressure  (bar)')
    ax.set_ylabel('TVD  (m)')
    ax.set_title(title or 'Wellbore pressure profile')
    if mark_bubble_point and profile['bubble_point_depth'] is not None:
        zb = profile['bubble_point_depth']
        ax.axhline(zb, color='C3', linestyle='--', alpha=0.7,
                   label=f'Bubble point ~ {zb:.0f} m')
    ax.scatter([P[0]], [z[0]], color='C0',  s=42, zorder=5,
               label=f'WHP = {P[0]:.1f} bar')
    ax.scatter([P[-1]], [z[-1]], color='C2', s=42, zorder=5,
               label=f'BHFP = {P[-1]:.1f} bar')
    ax.legend(loc='best')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path); plt.close(fig); return None
    return fig


# =====================================================================
# 2. Temperature profile vs depth (fluid + formation)
# =====================================================================
def plot_temperature_profile(profile, title=None, save_path=None):
    """Plot T_fluid(z) and T_formation(z)."""
    fig, ax = plt.subplots(figsize=(5.5, 7.0))
    z  = profile['z']
    Tf = profile['T']    - 273.15
    Te = profile['T_e']  - 273.15
    ax.plot(Tf, z, 'C0-', label='Fluid T (Ramey)')
    ax.plot(Te, z, 'C1--', label='Formation T (geo gradient)')
    ax.invert_yaxis()
    ax.set_xlabel('Temperature  (degC)')
    ax.set_ylabel('TVD  (m)')
    ax.set_title(title or 'Wellbore temperature profile')
    ax.legend(loc='best')
    # Annotate WH cooling
    dT_top = Te[0] - Tf[0]
    ax.text(0.05, 0.05, f'T cooling at WH:\n{abs(dT_top):.1f} degC '
                        'below formation',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path); plt.close(fig); return None
    return fig


# =====================================================================
# 3. Holdup, density, velocity profiles  (3-panel)
# =====================================================================
def plot_flow_profiles(profile, title=None, save_path=None):
    """Three-panel: H_l, rho_m, v_m vs depth (depth on y, inverted)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 7), sharey=True)
    z = profile['z']
    axes[0].plot(profile['H_l'], z, 'C0-')
    axes[0].set_xlabel('Liquid holdup  H_L  (-)')
    axes[0].set_ylabel('TVD  (m)')
    axes[0].set_xlim(0.0, 1.05)

    axes[1].plot(profile['rho_m'], z, 'C2-')
    axes[1].set_xlabel('Mixture (slip) density  (kg/m^3)')

    axes[2].plot(profile['v_m'], z, 'C3-')
    axes[2].set_xlabel('Mixture velocity  v_m  (m/s)')

    for ax in axes:
        ax.invert_yaxis()
    if title:
        fig.suptitle(title)
    if profile['bubble_point_depth'] is not None:
        for ax in axes:
            ax.axhline(profile['bubble_point_depth'], color='gray',
                       linestyle=':', alpha=0.6)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path); plt.close(fig); return None
    return fig


# =====================================================================
# 4. Pressure-gradient breakdown (dP_grav vs dP_fric)
# =====================================================================
def plot_dP_breakdown(profile, title=None, save_path=None):
    """Plot dPg, dPf and total gradient vs depth (in mbar/m)."""
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    z   = profile['z']
    dpg = profile['dpdL_grav'] * 1e-2     # mbar/m
    dpf = profile['dpdL_fric'] * 1e-2
    dpt = dpg + dpf
    ax.plot(dpg, z, 'C0-', label='Hydrostatic dP/dL')
    ax.plot(dpf, z, 'C3-', label='Friction dP/dL')
    ax.plot(dpt, z, 'k--', label='Total dP/dL', alpha=0.6)
    ax.invert_yaxis()
    ax.set_xlabel('dP / dL  (mbar / m)')
    ax.set_ylabel('TVD  (m)')
    ax.set_title(title or 'Pressure-gradient breakdown')
    ax.legend(loc='best')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path); plt.close(fig); return None
    return fig


# =====================================================================
# 5. CLASSIC NODAL PLOT  (IPR + VLP + operating point)
# =====================================================================
def plot_ipr_only(reservoir, n_points=30, q_max_m3h=None,
                  title=None, save_path=None):
    """Standalone IPR plot - reservoir inflow performance only.

    Useful when no natural flow is achieved at the chosen WHP (you may
    plan to add ESP/lift) but you still want to see the reservoir's
    pressure-rate response.
    """
    AOF_m3h = reservoir.AOF() * 3600.0
    if q_max_m3h is None:
        q_max_m3h = max(AOF_m3h * 1.05, 10.0)

    q_ipr, Pwf_ipr = ipr_curve(reservoir, n_points=n_points,
                               q_max_m3h=q_max_m3h)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.plot(m3h_to_ls(q_ipr), Pwf_ipr, 'C2-',
            label='IPR (reservoir)', linewidth=2)
    # Mark P_res and AOF
    P_res_bar = reservoir.P_res * 1.0e-5
    ax.scatter([0.0], [P_res_bar], color='C2', s=70, zorder=5,
               marker='s', label=f"P_res = {P_res_bar:.1f} bar")
    ax.scatter([m3h_to_ls(AOF_m3h)], [0.0], color='C3', s=70, zorder=5,
               marker='v',
               label=(f"AOF = {m3h_to_ls(AOF_m3h):.2f} l/s "
                      f"({m3h_to_m3d(AOF_m3h):.0f} m^3/d)"))
    ax.set_xlabel('q_brine, standard  (l/s)')
    ax.set_ylabel('Pwf  (bar)')
    ax.set_title(title or 'IPR (reservoir inflow performance)')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path); plt.close(fig); return None
    return fig


def plot_esp_requirement(WHP_bar, well, GWR_std, m_NaCl, thermal,
                         reservoir, n_segments=40, n_points=25,
                         q_max_m3h=None, title=None, save_path=None):
    """ESP (or any artificial-lift) requirement plot.

    For each rate q, compute:
      * IPR's Pwf (what the reservoir delivers to the bottomhole)
      * VLP's Pwf (what the wellbore needs at the bottomhole to reach
                   the surface at the chosen WHP)
      * dP_lift = Pwf_VLP - Pwf_IPR
                = the bottomhole pressure boost the pump must provide

    Where dP_lift > 0, you need lift; where it would be <0, the well
    flows naturally and no lift is needed.
    """
    AOF_m3h = reservoir.AOF() * 3600.0
    if q_max_m3h is None:
        q_max_m3h = max(AOF_m3h * 0.95, 10.0)

    q_arr = np.linspace(max(1.0, q_max_m3h * 0.02),
                        q_max_m3h, n_points)
    Pwf_ipr_arr = np.array(
        [reservoir.Pwf_at_q(q / 3600.0) * 1.0e-5 for q in q_arr])
    _, Pwf_vlp_arr = vlp_curve(WHP_bar, well, GWR_std, m_NaCl,
                                thermal, q_arr, n_segments=n_segments)
    dP_lift = Pwf_vlp_arr - Pwf_ipr_arr

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 8.0), sharex=True)

    # Top: IPR + VLP
    axes[0].plot(m3h_to_ls(q_arr), Pwf_ipr_arr, 'C2-',
                 label='IPR (reservoir)', linewidth=2)
    axes[0].plot(m3h_to_ls(q_arr), Pwf_vlp_arr, 'C0-',
                 label=f'VLP @ WHP={WHP_bar:.1f} bar', linewidth=2)
    axes[0].fill_between(m3h_to_ls(q_arr), Pwf_ipr_arr, Pwf_vlp_arr,
                          where=(Pwf_vlp_arr > Pwf_ipr_arr),
                          alpha=0.20, color='C3',
                          label='ESP lift required')
    axes[0].set_ylabel('Pwf  (bar)')
    axes[0].set_title(title or
        'ESP / artificial-lift requirement analysis')
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Bottom: dP_lift required vs rate
    axes[1].plot(m3h_to_ls(q_arr), dP_lift, 'C3-', linewidth=2)
    axes[1].axhline(0, color='k', linewidth=0.8, alpha=0.6)
    axes[1].fill_between(m3h_to_ls(q_arr), 0, dP_lift,
                          where=(dP_lift > 0),
                          alpha=0.20, color='C3')
    axes[1].set_xlabel('q_brine, standard  (l/s)')
    axes[1].set_ylabel('dP_lift required  (bar)')
    axes[1].set_title('Bottom-hole pressure boost from pump '
                      '(positive = ESP needed)')
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path); plt.close(fig); return None
    return fig


# =====================================================================
# 6. CLASSIC NODAL PLOT  (IPR + VLP + operating point)
# =====================================================================
def plot_nodal(WHP_bar, well, GWR_std, m_NaCl, thermal, reservoir,
               n_segments=40, n_points=20, q_max_m3h=None,
               extra_VLP=None, title=None, save_path=None):
    """The Cartesian nodal plot.

    `extra_VLP` (optional): list of dicts each with keys
        WHP_bar, label  (and optionally a different `well` etc.)
        for overlaying multiple VLP curves.
    """
    AOF_m3h = reservoir.AOF() * 3600.0
    if q_max_m3h is None:
        q_max_m3h = max(AOF_m3h * 1.05, 50.0)

    # IPR
    q_ipr, Pwf_ipr = ipr_curve(reservoir, n_points=n_points,
                               q_max_m3h=q_max_m3h)
    # VLP (main)
    q_vlp_in = np.linspace(max(1.0, q_max_m3h * 0.02), q_max_m3h,
                           n_points)
    q_vlp, Pwf_vlp = vlp_curve(WHP_bar, well, GWR_std, m_NaCl,
                                thermal, q_vlp_in,
                                n_segments=n_segments)
    # Operating point
    op = solve_operating_point(WHP_bar, well, GWR_std, m_NaCl, thermal,
                                reservoir, n_segments=n_segments)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.plot(m3h_to_ls(q_ipr), Pwf_ipr, 'C2-',
            label='IPR (reservoir)', linewidth=2)
    ax.plot(m3h_to_ls(q_vlp), Pwf_vlp, 'C0-',
            label=f'VLP @ WHP={WHP_bar:.1f} bar', linewidth=2)

    if extra_VLP:
        for spec in extra_VLP:
            qv = np.linspace(max(1.0, q_max_m3h * 0.02),
                             q_max_m3h, n_points)
            _, Pv = vlp_curve(spec['WHP_bar'],
                              spec.get('well', well),
                              spec.get('GWR_std', GWR_std),
                              spec.get('m_NaCl', m_NaCl),
                              spec.get('thermal', thermal), qv,
                              n_segments=n_segments)
            ax.plot(m3h_to_ls(qv), Pv, '--', alpha=0.7,
                    label=spec.get('label',
                                   f"VLP @ WHP={spec['WHP_bar']:.1f}"))

    if op['converged']:
        q_ls  = m3h_to_ls(op['q_op_m3h'])
        q_m3d = m3h_to_m3d(op['q_op_m3h'])
        ax.scatter([q_ls], [op['Pwf_op_bar']],
                   color='red', s=80, zorder=5, marker='o',
                   label=f"Op pt: q={q_ls:.2f} l/s ({q_m3d:.0f} m^3/d), "
                         f"Pwf={op['Pwf_op_bar']:.1f} bar")
    ax.set_xlabel('q_brine, standard  (l/s)')
    ax.set_ylabel('Pwf  (bar)')
    ax.set_title(title or 'Nodal analysis  (IPR / VLP at bottomhole)')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path); plt.close(fig); return op
    return fig, op


# =====================================================================
# 6. Sensitivity plot
# =====================================================================
def plot_sensitivity(rows, x_label, y_keys=('q_op_ls',),
                     y_labels=None, title=None, save_path=None,
                     log_x=False):
    """Plot sensitivity-sweep results.

    `rows` is the list of records returned by a sweep_*() function.
    `y_keys` is a tuple of keys to plot vs the swept value.
    """
    if y_labels is None:
        y_labels = list(y_keys)
    xs = np.array([r['value'] for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for k, label in zip(y_keys, y_labels):
        ys = np.array([r.get(k, np.nan) for r in rows], dtype=float)
        ax.plot(xs, ys, 'o-', label=label)
    ax.set_xlabel(x_label)
    if log_x:
        ax.set_xscale('log')
    ax.set_title(title or 'Sensitivity')
    ax.legend(loc='best')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path); plt.close(fig); return None
    return fig


# =====================================================================
# 7. Multi-panel "report" of one operating point
# =====================================================================
def plot_operating_point_report(op, save_path=None, title=None):
    """One figure with four panels: P(z), T(z), H_l(z), dP breakdown."""
    p = op['profile']
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # P(z)
    z = p['z']
    axes[0, 0].plot(p['P'] * 1e-5, z, 'C0-')
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel('P (bar)')
    axes[0, 0].set_ylabel('TVD (m)')
    axes[0, 0].set_title('Pressure')
    if p['bubble_point_depth']:
        axes[0, 0].axhline(p['bubble_point_depth'], color='C3', ls='--',
                           alpha=0.6,
                           label=f"BP ~ {p['bubble_point_depth']:.0f} m")
        axes[0, 0].legend()

    # T(z)
    axes[0, 1].plot(p['T'] - 273.15, z, 'C0-', label='Fluid')
    axes[0, 1].plot(p['T_e'] - 273.15, z, 'C1--', label='Formation')
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel('T (degC)')
    axes[0, 1].set_ylabel('TVD (m)')
    axes[0, 1].set_title('Temperature')
    axes[0, 1].legend()

    # H_l(z)
    axes[1, 0].plot(p['H_l'], z, 'C2-')
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel('Liquid holdup H_L')
    axes[1, 0].set_ylabel('TVD (m)')
    axes[1, 0].set_title('Holdup')
    axes[1, 0].set_xlim(0, 1.05)

    # dP breakdown
    axes[1, 1].plot(p['dpdL_grav'] * 1e-2, z, 'C0-', label='Gravity')
    axes[1, 1].plot(p['dpdL_fric'] * 1e-2, z, 'C3-', label='Friction')
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel('dP/dL (mbar/m)')
    axes[1, 1].set_ylabel('TVD (m)')
    axes[1, 1].set_title('Pressure-gradient breakdown')
    axes[1, 1].legend()

    res = op['reservoir']
    suptitle = (title or
        f"Operating point  q = {fmt_q(op['q_op_m3h'])}  "
        f"Pwf={op['Pwf_op_bar']:.1f} bar  "
        f"WHP={op['WHP_bar']:.1f} bar  "
        f"(s_total={res.s_total:.2f}, P_res={res.P_res*1e-5:.0f} bar)")
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path); plt.close(fig); return None
    return fig


# =====================================================================
# 8. SELF-TEST
# =====================================================================
def _validate():
    print("=" * 78)
    print("PLOTTING MODULE - generating reference figures")
    print("=" * 78)

    from vlp import WellGeometry, ThermalContext
    from ipr import ReservoirModel
    from pvt import ppm_to_molality
    import os

    well = WellGeometry(depth_TVD=2500.0, tubing_ID=0.114, tubing_OD=0.127)
    therm = ThermalContext(T_surface=288.15, geo_gradient=0.080,
                           U_overall=20.0, time_seconds=60 * 86400.0)
    res = ReservoirModel(
        k_md=80.0, h_m=20.0, P_res_bar=320.0, T_res_C=215.0,
        r_w=0.108, r_e=300.0, m_NaCl=0.713,
        completion='perforated',
        perf_length=0.30, perf_diameter=0.012,
        perf_SPF_per_m=13.12, perf_phasing=90,
        damage_radius=0.30, damage_k_ratio=0.5)
    GWR = 10.0
    m_NaCl = ppm_to_molality(40_000)

    # Solve operating point
    op = solve_operating_point(8.0, well, GWR, m_NaCl, therm, res)

    out = '/tmp'
    plot_pressure_profile(op['profile'],
                          save_path=f'{out}/p_profile.png')
    plot_temperature_profile(op['profile'],
                             save_path=f'{out}/t_profile.png')
    plot_flow_profiles(op['profile'],
                       save_path=f'{out}/flow_profiles.png')
    plot_dP_breakdown(op['profile'],
                      save_path=f'{out}/dP_breakdown.png')
    plot_nodal(8.0, well, GWR, m_NaCl, therm, res,
               extra_VLP=[
                   dict(WHP_bar=3.0,  label='VLP @ 3 bar'),
                   dict(WHP_bar=15.0, label='VLP @ 15 bar'),
                   dict(WHP_bar=25.0, label='VLP @ 25 bar'),
               ],
               save_path=f'{out}/nodal_plot.png')
    plot_operating_point_report(op,
                                save_path=f'{out}/op_report.png')

    # Sensitivity example
    from sensitivities import sweep_skin_total
    rows = sweep_skin_total([-2, 0, 2, 5, 8, 12, 18, 25],
                            8.0, well, GWR, m_NaCl, therm, res)
    plot_sensitivity(rows, x_label='Total skin (-)',
                     y_keys=('q_op_ls',), y_labels=('q (l/s)',),
                     title='Skin sensitivity (WHP = 8 bar)',
                     save_path=f'{out}/sens_skin.png')

    print(f"  Saved figures to {out}/:")
    for f in ('p_profile', 't_profile', 'flow_profiles', 'dP_breakdown',
              'nodal_plot', 'op_report', 'sens_skin'):
        full = f'{out}/{f}.png'
        if os.path.exists(full):
            print(f"    {full}  ({os.path.getsize(full)//1024} KB)")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    _validate()
