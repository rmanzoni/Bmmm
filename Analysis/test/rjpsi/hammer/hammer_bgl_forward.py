#!/usr/bin/env python3
"""
Faithful Python port of Hammer's FFBctoJpsiBGL::evalAtPSPoint (v2.0.0).
Computes the helicity FFs (g, f, F1, P1) from the BGL coefficients avec/bvec/
cvec/dvec, in HAMMER'S EXACT convention. NOTE: P1 is NOT A0 -- see P1_OVER_F2
below for the relation, and use f2_from_p1() before building any amplitude:
  * two z-variables: z0 (g, Rb0 branch), z1 (f/F1/P1, Rb* branch)
  * branch points from Rb*, Rb0, Rd0; OptZ toggles t0 (q2max vs optimized)
  * Hammer's outer functions phig/phif/phiF1/phiP1 (nc=1), Blaschke on BcStates*
  * F1 endpoint constraint F1(q2max)=(Mb-Mc) f(q2max) built in
  * NO Vcb division (unlike B->D*)
Defaults below are Hammer's active BctoJpsiBGL settings (Harrison 2020, 2007.06957).
"""
import numpy as np

# Hammer default settings (FFBctoJpsiBGL::defineSettings, active block)
MBC, MJPSI = 6.2745, 3.0969
CHIM  = 0.00737058 / (4.78 * 4.78)
CHIP  = 0.0126835  / (4.78 * 4.78)
CHIML = 0.0249307
BCSTATESF  = [6.730, 6.736, 7.135, 7.142]
BCSTATESG  = [6.337, 6.899, 7.012]
BCSTATESP1 = [6.2749, 6.842]
RBS = 5.324 / 6.275     # Rb*
RBZ = 5.280 / 6.275     # Rb0
RDZ = 1.865 / 3.096     # Rd0
OPTZ = False
NC = 1.0

# ---------------------------------------------------------------------------
# P1 <-> A0 <-> F2: THE convention, in one place
# ---------------------------------------------------------------------------
# evalAtPSPoint returns P1, but Hammer's AMPLITUDE never uses P1 as A0. It maps
# (f, g, F1, P1) onto the Manohar-Wise basis (f, g, a+, a-), with P1 entering
# only a- through
#     FmP1 = sqrt(r) (1 + r) / (Mb (1 + r^2 - 2 r w)) .
# Contracting the axial current with q fixes A0 via
#     2 m A0 = f + (M^2 - m^2) a+ + q^2 a- ,
# the f and F1 terms cancel identically, and what is left is
#     A0 = (1 + r) / (2 sqrt r) * P1 ,        r = M_Jpsi / M_Bc ,
# constant in q2. With Harrison's F2 = 2 A0:
#     P1 = sqrt(r) / (1 + r) * F2_Harrison .
#
# The first version of this port assumed A0 = P1 (i.e. P1 = F2/2), read from
# evalAtPSPoint alone. That factor, 1.063 in amplitude, entered BOTH the fit
# target and the rate formula, so it cancelled in every port-side test (Fig. 6,
# R(J/psi), A_lambda_tau all closed) and showed up only when Hammer's own C++
# evaluated the card: an excess in |H_t|^2 of 0.120 +/- 0.012 measured against
# 0.130 predicted. See hammer_q2_from_scratch.py --scale-coeff dvec:2.
RATIO_R = MJPSI / MBC
P1_OVER_F2 = np.sqrt(RATIO_R) / (1. + RATIO_R)       # = 0.4704


def p1_from_f2(F2):
    """Hammer's P1 from Harrison's helicity F2 (= 2 A0)."""
    return F2 * P1_OVER_F2


def f2_from_p1(P1):
    """Harrison's helicity F2 (= 2 A0) from Hammer's P1."""
    return P1 / P1_OVER_F2


DEFAULT = dict(
    avec=[0.0257, -0.129, -0.21, -0.09],
    bvec=[0.01781, -0.104, 0.30, -0.23],
    cvec=[-0.0149, -0.028, 0.21],
    dvec=[0.0374, -0.192, -0.34, -0.16],
)


def hammer_bgl_ff(q2, avec, bvec, cvec, dvec,
                  Mb=MBC, Mc=MJPSI, chim=CHIM, chip=CHIP, chimL=CHIML,
                  BcStatesf=BCSTATESF, BcStatesg=BCSTATESG, BcStatesP1=BCSTATESP1,
                  Rbs=RBS, Rbz=RBZ, Rdz=RDZ, OptZ=OPTZ, nc=NC):
    Mb2, Mb3, Mc2 = Mb*Mb, Mb**3, Mc*Mc
    rC = Mc / Mb; rC2 = rC*rC; sqrC = np.sqrt(rC)
    w = (Mb2 + Mc2 - q2) / (2. * Mb * Mc)
    if abs(w - 1.0) < 1e-12:
        w += 1e-6
    w2 = w*w

    sqtbd1  = Rbs + rC * Rdz
    sqtst1  = np.sqrt(sqtbd1**2 - rC2 + 2.*rC*w - 1.)
    sqtstm1 = np.sqrt(sqtbd1**2 - (rC - 1.)**2)
    sqtst01 = np.sqrt(sqtbd1 * sqtstm1) if OptZ else sqtstm1

    sqtbd0  = Rbz + rC * Rdz
    sqtst0  = np.sqrt(sqtbd0**2 - rC2 + 2.*rC*w - 1.)
    sqtstm0 = np.sqrt(sqtbd0**2 - (rC - 1.)**2)
    sqtst00 = np.sqrt(sqtbd0 * sqtstm0) if OptZ else sqtstm0

    z1    = (sqtst1 - sqtst01) / (sqtst1 + sqtst01)
    z1min = (sqtstm1 - sqtst01) / (sqtstm1 + sqtst01)
    z0    = (sqtst0 - sqtst00) / (sqtst0 + sqtst00)
    z1pow    = [z1**n for n in range(4)]
    z1minpow = [z1min**n for n in range(4)]
    z0pow    = [z0**n for n in range(4)]

    def blaschke(states, sqtbd, sqtst0_, z):
        P = 1.0
        for mP in states:
            wM = (Mb2 + Mc2 - mP*mP) / (2. * Mb * Mc)
            sqtstM = np.sqrt(sqtbd**2 - rC2 + 2.*rC*wM - 1.)
            zM = (sqtstM - sqtst0_) / (sqtstM + sqtst0_)
            P *= (z - zM) / (1. - z*zM)
        return P
    Pf  = blaschke(BcStatesf,  sqtbd1, sqtst01, z1); PF1 = Pf
    Pg  = blaschke(BcStatesg,  sqtbd0, sqtst00, z0)
    PP1 = blaschke(BcStatesP1, sqtbd1, sqtst01, z1)

    pi = np.pi
    phig  = np.sqrt(nc/(96*pi*chip)) * (sqtst0/sqtst00)**0.5 * (sqtst0+sqtst00) * sqtst0**1.5 * (sqtst0+sqtstm0)**1.5 / (sqtst0+sqtbd0)**4
    phif  = (1./Mb2)*np.sqrt(nc/(24*pi*chim)) * (sqtst1/sqtst01)**0.5 * (sqtst1+sqtst01) * sqtst1**0.5 * (sqtst1+sqtstm1)**0.5 / (sqtst1+sqtbd1)**4
    phiF1 = phif / (Mb*np.sqrt(2.)*(sqtst1+sqtbd1))
    phiP1 = np.sqrt(nc/(64*pi*chimL)) * (sqtst1/sqtst01)**0.5 * (sqtst1+sqtst01) * sqtst1**1.5 * (sqtst1+sqtstm1)**1.5 / (sqtst1+sqtbd1)**4
    phif_0  = (1./Mb2)*np.sqrt(nc/(24*pi*chim)) * (sqtstm1/sqtst01)**0.5 * (sqtstm1+sqtst01) * sqtstm1**0.5 * (sqtstm1+sqtstm1)**0.5 / (sqtstm1+sqtbd1)**4
    phiF1_0 = phif_0 / (Mb*np.sqrt(2.)*(sqtstm1+sqtbd1))

    g = sum(avec[n]*z0pow[n] for n in range(len(avec))) / (Pg*phig)
    f   = sum(bvec[n]*z1pow[n]    for n in range(len(bvec)))
    f_0 = sum(bvec[n]*z1minpow[n] for n in range(len(bvec)))
    f /= (Pf*phif)
    F1 = (Mb - Mc)*f_0*phiF1_0/phif_0
    for n in range(len(cvec)):
        F1 += cvec[n]*(z1pow[n+1] - z1minpow[n+1])
    F1 /= (PF1*phiF1)
    P1 = sum(dvec[n]*z1pow[n] for n in range(len(dvec)))
    P1 *= sqrC / ((1+rC)*PP1*phiP1)
    return g, f, F1, P1


if __name__ == "__main__":
    q2max = (MBC - MJPSI)**2
    print("self-test with Hammer's DEFAULT (Harrison 2020) coefficients:")
    print("  q2      g          f          F1         P1(=F2)")
    for q2 in (0.05, 3.0, 6.0, q2max - 1e-4):
        g, f, F1, P1 = hammer_bgl_ff(q2, **DEFAULT)
        print("  %5.2f  %9.5f  %9.5f  %9.4f  %9.5f" % (q2, g, f, F1, P1))
    # endpoint constraint must hold by construction
    g, f, F1, P1 = hammer_bgl_ff(q2max - 1e-6, **DEFAULT)
    print("\nendpoint: F1/f = %.5f   must equal (M_Bc-M_Jpsi) = %.5f"
          % (F1/f, MBC - MJPSI))
