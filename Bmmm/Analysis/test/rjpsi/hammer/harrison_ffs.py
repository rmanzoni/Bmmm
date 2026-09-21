#!/usr/bin/env python3
"""
Harrison-side FF evaluator for the Bc->J/psi BGLVar reweighting (route b).

Reuses Harrison's ancillary machinery (arXiv:2503.15090) to produce the PHYSICAL
helicity form factors g, f, F1, F2 (as correlated gvars) on any q2 grid. These
are the fit target for the Hammer CLL-BGL stage (next module).

Run next to the ancillary files:
    continuum_fit_posteriors.pydat
    CC_extrapolation_utilities.py, load_chi_u12.py, dispersive_functions.py,
    poly.py, CC_fit_parameters.py

Physical V, A0, A1, A12 come from Harrison's dispersive fit via
CC_extrapolation_utilities.ChiContFF -- the fitprintA path, replicated here
WITHOUT importing CC_extrapolation.py (which plots/prints on import). A2 from A12
via Harrison Eq. 8. Helicity basis (Harrison PRD 97 054502 / 2007.06957 App. B):
    g  = 2 V / (M_Bc + M_Jpsi)
    f  = (M_Bc + M_Jpsi) A1
    F1 = (M_Bc + M_Jpsi)/M_Jpsi [ -2 M_Bc^2 |p'|^2/(M_Bc+M_Jpsi)^2 * A2
                                  - 1/2 (q2 - M_Bc^2 + M_Jpsi^2) * A1 ]
    F2 = 2 A0
"""
import numpy as np
import gvar as gv

import load_chi_u12 as CHI
import CC_extrapolation_utilities as cce
from CC_fit_parameters import *   # MBCPHYS, MJPPHYS, mbphys, uphys, chidict, lambdaqcdphys

POSTERIORS_FILE = "continuum_fit_posteriors.pydat"
_POST = None
_SUSC_CURRS = ["A0", "A1", "A12", "V", "T1", "T2", "T23"]


def posteriors():
    global _POST
    if _POST is None:
        _POST = gv.load(POSTERIORS_FILE)
    return _POST


def fitprintA(curr, s, p=None, MHc=MBCPHYS, MJp=MJPPHYS, mhbar=mbphys, u=uphys):
    """Physical FF value at q2 = s (gvar). Mirrors CC_extrapolation.fitprintA."""
    if p is None:
        p = posteriors()
    chisusc = {C: CHI.fitprint(p, u, chidict[C]) for C in _SUSC_CURRS}
    if curr != "A2":
        return cce.ChiContFF(p, curr, 0, 0, 0, 0, MHc, MJp, lambdaqcdphys, MHc, chisusc, mhbar, s)
    A1q  = fitprintA("A1",  s, p, MHc, MJp, mhbar, u)
    A12q = fitprintA("A12", s, p, MHc, MJp, mhbar, u)
    lam  = cce.lambda_kin(MHc, MJp, s)
    return ((MHc + MJp)**2 * (MHc**2 - MJp**2 - s) * A1q
            - 16.0 * MHc * MJp**2 * (MHc + MJp) * A12q) / lam


def helicity(q2, p=None):
    """Physical helicity FFs g, f, F1, F2 (gvar) at a single q2."""
    if p is None:
        p = posteriors()
    V  = fitprintA("V",  q2, p)
    A0 = fitprintA("A0", q2, p)
    A1 = fitprintA("A1", q2, p)
    A2 = fitprintA("A2", q2, p)
    s   = MBCPHYS + MJPPHYS
    pp2 = cce.lambda_kin(MBCPHYS, MJPPHYS, q2) / (4.0 * MBCPHYS**2)   # |p'|^2
    return {
        "g":  2.0 * V / s,
        "f":  s * A1,
        "F2": 2.0 * A0,
        "F1": (s / MJPPHYS) * (-2.0 * MBCPHYS**2 * pp2 / s**2 * A2
                               - 0.5 * (q2 - MBCPHYS**2 + MJPPHYS**2) * A1),
    }


def helicity_curves(q2grid, p=None):
    """Correlated gvar arrays {g,f,F1,F2} over q2grid (shared posteriors keep
    the cross-correlations that feed the BGL covariance)."""
    if p is None:
        p = posteriors()
    cols = {k: [] for k in ("g", "f", "F1", "F2")}
    for q2 in q2grid:
        h = helicity(q2, p)
        for k in cols:
            cols[k].append(h[k])
    return {k: np.array(v, dtype=object) for k, v in cols.items()}


def main():
    q2max = (MBCPHYS - MJPPHYS)**2
    grid = np.array([0.01, 3.35, 6.70, q2max - 1e-6])
    hc = helicity_curves(grid)
    print("  q2        g               f               F1              F2")
    for i, q2 in enumerate(grid):
        print("%6.3f  %s  %s  %s  %s"
              % (q2, hc["g"][i], hc["f"][i], hc["F1"][i], hc["F2"][i]))
    # strong self-check: on the PHYSICAL curve, F1/f -> (M_Bc - M_Jpsi) at q2max
    ratio = hc["F1"][-1] / hc["f"][-1]
    print("\nendpoint F1/f = %s   (must equal M_Bc - M_Jpsi = %.5f)"
          % (ratio, MBCPHYS - MJPPHYS))


if __name__ == "__main__":
    main()
