#!/usr/bin/env python3
"""
Fit Harrison-2024 (arXiv:2503.15090) into Hammer's EXACT BctoJpsiBGL convention.

This supersedes bgl_fit.py. The old fit produced coefficients in Harrison's
dispersive convention, which Hammer (a different z/phi/Rb convention, ported in
hammer_bgl_forward.py) mis-evaluated -> the ~20x bug. Here we fit through Hammer's
own forward model, so the emitted avec/bvec/cvec/dvec load correctly via
set_options and reproduce Harrison's FFs.

Targets (physical helicity FFs from harrison_ffs, with covariance):
  g -> avec ; f -> bvec ; F1 -> cvec (a0 fixed by endpoint constraint) ;
  F2 -> dvec via P1 = sqrt(r)/(1+r) * F2   (hammer_bgl_forward.p1_from_f2).
  NOT P1 = F2/2: that was the first version's assumption, read from evalAtPSPoint
  alone. Hammer's amplitude reads P1 through the a- form factor and the implied
  A0 is (1+r)/(2 sqrt r) * P1, a 1.063 factor that cancelled in every port-side
  test and was caught only against Hammer's C++ (see P1_OVER_F2).

Each FF is linear in its coefficient vector, so every fit is a linear LSQ; the
basis is extracted by probing hammer_bgl_forward with unit vectors.
"""
import argparse
import datetime
import json

import numpy as np
import gvar as gv
import harrison_ffs as HF
import hammer_bgl_forward as HB

q2max = (HB.MBC - HB.MJPSI)**2
GRID = np.linspace(0.10, q2max - 1e-3, 40)


def _probe(name, n, q2, dims):
    kw = {"avec": [0.]*4, "bvec": [0.]*4, "cvec": [0.]*3, "dvec": [0.]*4}
    kw[name] = [1.0 if i == n else 0.0 for i in range(dims)]
    return HB.hammer_bgl_ff(q2, **kw)

# design matrices (kinematics only, coefficient-independent)
_Ag  = np.array([[_probe("avec", n, q2, 4)[0] for n in range(4)] for q2 in GRID])
_Af  = np.array([[_probe("bvec", n, q2, 4)[1] for n in range(4)] for q2 in GRID])
_Cbv = np.array([[_probe("bvec", n, q2, 4)[2] for n in range(4)] for q2 in GRID])  # F1 <- bvec (constraint)
_Bcv = np.array([[_probe("cvec", n, q2, 3)[2] for n in range(3)] for q2 in GRID])  # F1 <- cvec
_Ap  = np.array([[_probe("dvec", n, q2, 4)[3] for n in range(4)] for q2 in GRID])


# The z-basis is nearly degenerate over the physical grid, so plain LSQ amplifies
# the small (2020->2024) FF change into large, unitarity-violating coefficients.
# Regularise toward Hammer's default (Harrison-2020) coefficients: this fixes only
# the curve-null direction, leaving the reproduced FF unchanged (pull ~1e-4) but
# giving physical, unitary coefficients. LAM=1e-6 validated on synthetic tests.
LAM = 1e-6
_D_a = np.array(HB.DEFAULT["avec"]); _D_b = np.array(HB.DEFAULT["bvec"])
_D_c = np.array(HB.DEFAULT["cvec"]); _D_d = np.array(HB.DEFAULT["dvec"])


def _tikh(A, b, x0):
    n = A.shape[1]
    return np.linalg.solve(A.T @ A + LAM * np.eye(n), A.T @ b + LAM * x0)


def fit_coeffs(g, f, F1, P1):
    avec = _tikh(_Ag, g, _D_a)
    bvec = _tikh(_Af, f, _D_b)
    cvec = _tikh(_Bcv, F1 - _Cbv @ bvec, _D_c)
    dvec = _tikh(_Ap, P1, _D_d)
    return avec, bvec, cvec, dvec


def _tikh_matrix(A, x0):
    """The Tikhonov solve is affine in the target b: x = M b + c. Returns (M, c)."""
    n = A.shape[1]
    inv = np.linalg.inv(A.T @ A + LAM * np.eye(n))
    return inv @ A.T, LAM * (inv @ x0)


def coefficient_jacobian():
    """J (15 x 4*len(GRID)): d(coefficients)/d(stacked curves [g, f, F1, F2]).

    fit_coeffs is LINEAR in the curves, so the coefficient covariance is exactly
        C_coeff = J . C_Harrison . J^T
    -- no toys, no seed, no NSAMP. Mirrors fit_coeffs including the bvec-mediated
    dependence of cvec on f (endpoint constraint), and the P1 <- F2 convention.
    """
    nq = len(GRID)
    Ma, _ = _tikh_matrix(_Ag, _D_a)
    Mb, _ = _tikh_matrix(_Af, _D_b)
    Mc, _ = _tikh_matrix(_Bcv, _D_c)
    Md, _ = _tikh_matrix(_Ap, _D_d)
    J = np.zeros((15, 4 * nq))
    sg, sf, sF1, sF2 = (slice(k * nq, (k + 1) * nq) for k in range(4))
    J[0:4, sg] = Ma
    J[4:8, sf] = Mb
    J[8:11, sF1] = Mc
    J[8:11, sf] = -Mc @ _Cbv @ Mb
    J[11:15, sF2] = Md * HB.P1_OVER_F2
    return J


def coefficient_covariance(hc):
    """Exact 15x15 covariance of the fitted coefficients."""
    stacked = np.concatenate([hc[k] for k in ("g", "f", "F1", "F2")])
    J = coefficient_jacobian()
    return J @ gv.evalcov(stacked) @ J.T


def flat(a, b, c, d):
    return np.concatenate([a, b, c, d])          # a's, b's, c's, d's (Hammer row order)


def write_card(path, avec, bvec, cvec, dvec, evecs, sqrt_evals, order):
    """Emit the JSON card that Bmmm/Analysis/python/HammerFF.py loads.

    covariance_status is written as 'unvalidated': flip it to 'validated' by
    hand once check_ff_band.py shows the implied FF band matches Harrison's own
    uncertainty. HammerFF refuses to write eigenvariations from anything else,
    which is the point -- the coefficients and the covariance have gone out of
    sync once already.
    """
    card = {
        "name": "Harrison-2024 in Hammer BctoJpsiBGLVar",
        "source": "arXiv:2503.15090, fit through Hammer's forward model by "
                  "harrison_to_hammer_bgl.py",
        "fit_date": datetime.date.today().isoformat(),
        "process": "BcJpsi", "xtoy": "BctoJpsi",
        "ff_input": "Kiselev", "ff_target": "BGLVar",
        "regularisation_lambda": LAM,
        "covariance_method": "exact linear propagation J C J^T",
        "param_order": order,
        "avec": [float(x) for x in avec], "bvec": [float(x) for x in bvec],
        "cvec": [float(x) for x in cvec], "dvec": [float(x) for x in dvec],
        "covariance_status": "unvalidated",
        # HammerFF refuses any card without exactly this string
        "p1_convention": "P1 = sqrt(r)/(1+r) * F2",
        "sqrt_evals": [float(x) for x in sqrt_evals],
        "evecs": [[float(x) for x in row] for row in evecs],
    }
    with open(path, "w") as fout:
        json.dump(card, fout, indent=1)
        fout.write("\n")
    print("\n# wrote FF card -> %s" % path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--toys", type=int, default=0,
                    help="also run N toys, only to cross-check the exact "
                         "linear covariance (the card always gets the exact one)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--json", default=None,
                    help="also write the coefficients + covariance eigenbasis as "
                         "the JSON card HammerFF.py reads "
                         "(Bmmm/Analysis/data/harrison_bglvar.json)")
    args = ap.parse_args()

    hc = HF.helicity_curves(GRID)                 # {g,f,F1,F2} correlated gvars
    g, f, F1 = gv.mean(hc["g"]), gv.mean(hc["f"]), gv.mean(hc["F1"])
    P1 = HB.p1_from_f2(gv.mean(hc["F2"]))          # NOT F2/2: see P1_OVER_F2
    avec, bvec, cvec, dvec = fit_coeffs(g, f, F1, P1)

    # fit quality: reproduce Harrison's curves through Hammer's model
    print("fit closure (max pull over the grid):")
    mg = np.array([HB.hammer_bgl_ff(q2, avec, bvec, cvec, dvec)[0] for q2 in GRID])
    mf = np.array([HB.hammer_bgl_ff(q2, avec, bvec, cvec, dvec)[1] for q2 in GRID])
    mF = np.array([HB.hammer_bgl_ff(q2, avec, bvec, cvec, dvec)[2] for q2 in GRID])
    mP = np.array([HB.hammer_bgl_ff(q2, avec, bvec, cvec, dvec)[3] for q2 in GRID])
    for nm, mod, tg in [("g", mg, hc["g"]), ("f", mf, hc["f"]),
                        ("F1", mF, hc["F1"]), ("P1", mP, HB.p1_from_f2(hc["F2"]))]:
        pull = np.abs(mod - gv.mean(tg)) / np.maximum(gv.sdev(tg), 1e-12)
        print("  %-9s max pull = %.2f" % (nm, pull.max()))

    print("unitarity (sum a_n^2 < 1):  avec=%.3f bvec=%.3f cvec=%.3f dvec=%.3f  %s"
          % ((avec**2).sum(), (bvec**2).sum(), (cvec**2).sum(), (dvec**2).sum(),
             "ALL OK" if max((avec**2).sum(), (bvec**2).sum(),
                             (cvec**2).sum(), (dvec**2).sum()) < 1 else "CHECK"))

    print("\n# ---- Harrison-2024 in Hammer convention: set on BctoJpsiBGL(Var) ----")
    for nm, v in [("avec", avec), ("bvec", bvec), ("cvec", cvec), ("dvec", dvec)]:
        print('ham.set_options("BctoJpsiBGL: { %s: %s }")'
              % (nm, np.array2string(v, precision=8, separator=", ")))

    # covariance -> eigenbasis (for the delta_e/abcdmatrix variations, later)
    order = (["avec%d" % n for n in range(4)] + ["bvec%d" % n for n in range(4)]
             + ["cvec%d" % n for n in range(3)] + ["dvec%d" % n for n in range(4)])
    # EXACT linear propagation. The toy loop is kept only as a cross-check of
    # the algebra (--toys N): its eigenvalues scatter as sqrt(2/N), which is
    # what made two consecutive runs disagree by ~7% at N = 400.
    cov = coefficient_covariance(hc)
    if args.toys:
        gv.ranseed(args.seed)
        samples = []
        for d in gv.raniter(hc, n=args.toys):
            a, b, c, dd = fit_coeffs(d["g"], d["f"], d["F1"], HB.p1_from_f2(d["F2"]))
            samples.append(flat(a, b, c, dd))
        cov_toy = np.cov(np.array(samples).T)
        print("# toys vs exact covariance: max|dC|/max|C| = %.2e  (expect ~%.1e)"
              % (np.abs(cov_toy - cov).max() / np.abs(cov).max(),
                 np.sqrt(2. / args.toys)))
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    print("\n# 15-parameter order:", order)
    print("# sqrt(eigenvalues):", np.array2string(np.sqrt(np.abs(evals)), precision=5))
    np.set_printoptions(precision=6, suppress=True)
    print("HARRISON_HAMMER_EVECS =", repr(evecs))
    print("HARRISON_HAMMER_SQRT_EVALS =", repr(np.sqrt(np.abs(evals))))

    if args.json:
        write_card(args.json, avec, bvec, cvec, dvec, evecs,
                   np.sqrt(np.abs(evals)), order)
    print("\n# CLOSURE: load avec..dvec into Hammer and check R(J/psi)=0.2597 and <w>_mu ~ 0.5")


if __name__ == "__main__":
    main()
