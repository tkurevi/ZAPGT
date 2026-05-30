"""ZapGT-1 VERSION 2: new deeper PRODUCER, existing ZapGT-1 = injector.
Built by depth-shifting ZapGT-1's calibrated config by -430 m (deeper):
all properties identical (k, h, porosity, skin, D); only depth +430 m,
temperature +430*gradient, reservoir pressure +430*hydrostatic.
"""
import copy
import main_zapgt1 as _base
from main_zapgt1 import build_commingled            # reuse the identical builder

SHIFT_M   = 430.0
GRAD_K_m  = 0.0365
RHOG_bar_m = 983.0 * 9.81 / 1e5                       # ~0.0964 bar/m (cold brine column)

def _shift(cfg):
    cfg = copy.deepcopy(cfg)
    for L in cfg['layers']:
        for k in ('top_depth_m', 'bottom_depth_m', 'z_mid_m', 'mid_depth_m'):
            if k in L and L[k] is not None: L[k] += SHIFT_M
        if L.get('T_res_C')  is not None: L['T_res_C']  += SHIFT_M * GRAD_K_m
        if L.get('P_res_bar') is not None: L['P_res_bar'] += SHIFT_M * RHOG_bar_m
    if cfg.get('commingled', {}).get('reference_depth_m') is not None:
        cfg['commingled']['reference_depth_m'] += SHIFT_M
    # deepen well segments so the wellbore reaches the deeper datum
    for seg in cfg.get('well', {}).get('segments', []) or []:
        if isinstance(seg, dict):
            for k in list(seg):
                if 'depth' in k or k in ('top','bottom','z_top','z_bot'): seg[k]+=SHIFT_M
    if cfg.get('operating',{}).get('pump') and isinstance(cfg['operating']['pump'],dict):
        if 'z_intake_m' in cfg['operating']['pump']: cfg['operating']['pump']['z_intake_m']+=SHIFT_M
    return cfg

CALIBRATED_CONFIG = _shift(_base.CALIBRATED_CONFIG)
CALIBRATED_CONFIG['case_name'] = 'ZapGT-1 V2 (deeper producer, +430 m, ~87 C)'
