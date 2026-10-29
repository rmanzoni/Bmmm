#!/usr/bin/env python3
"""
Bc -> J/psi l nu differential rate from the Hammer-convention BGL coefficients.

Split out of ff_fit_validation.py so that anything needing the rate -- the
lattice-side validation, and the Hammer-side one in hammer_q2_validation.py --
shares ONE implementation. This module deliberately imports only numpy and
hammer_bgl_forward: no harrison_ffs, no gvar, no lattice ancillary, so it runs
anywhere the FF card does.

Validated against Fig. 6 of Harrison arXiv:2503.15090: the mu and tau spectra
agree in shape and in relative normalisation, and R(J/psi) = 0.2602 +/- 0.0034
against the published 0.2597(27).
"""
import numpy as np

import hammer_bgl_forward as HB

SLICES = {'avec': slice(0, 4), 'bvec': slice(4, 8),
          'cvec': slice(8, 11), 'dvec': slice(11, 15)}
FFS = ('g', 'f', 'F1', 'F2')
NCOEF = 15


def unflatten(vec):
    return dict((k, vec[s]) for k, s in SLICES.items())


def curves_from_coeffs(vec, grid):
    out = np.array([HB.hammer_bgl_ff(q2, **unflatten(vec)) for q2 in grid])
    return {'g': out[:, 0], 'f': out[:, 1], 'F1': out[:, 2],
            'F2': HB.f2_from_p1(out[:, 3])}


def coeffs_from_card(card):
    """The 15-vector, in Hammer row order, from a loaded FF card dict."""
    return np.concatenate([np.asarray(card[k], dtype=float)
                           for k in ('avec', 'bvec', 'cvec', 'dvec')])


def sigma_directions_from_card(card):
    """Columns = the 1 sigma principal directions, i.e. Hammer's abcdmatrix."""
    return np.asarray(card['evecs'], dtype=float) * np.asarray(
        card['sqrt_evals'], dtype=float)[None, :]


# ---------------------------------------------------------------------------
# differential rate: where the FF uncertainty actually lands
# ---------------------------------------------------------------------------
M_MU, M_TAU = 0.1056583755, 1.77686
LEPTONS = {'mu': M_MU, 'tau': M_TAU}


def _momentum(q2):
    """|p| of the J/psi in the Bc rest frame."""
    mb2, mc2 = HB.MBC ** 2, HB.MJPSI ** 2
    lam = (mb2 + mc2 - q2) ** 2 - 4. * mb2 * mc2
    return np.sqrt(np.maximum(lam, 0.)) / (2. * HB.MBC)


def helicity_amplitudes(q2, vec):
    r"""H_+, H_-, H_0, H_t from the Hammer BGL coefficients.

        H_\pm = f -+ M_Bc |p| g          H_0 = F1 / sqrt(q2)
        H_t   = M_Bc |p| F2 / sqrt(q2)   with Harrison F2 = 2 A_0

    hammer_bgl_ff returns Hammer's P1 as its fourth value. P1 is NOT A_0:
    F2 = 2 A_0 = f2_from_p1(P1) = (1+r)/sqrt(r) * P1. The first version
    assumed A_0 = P1 and got H_t low by 1.063 -- see P1_OVER_F2.
    """
    kw = dict((k, vec[s]) for k, s in SLICES.items())
    g, f, F1, P1 = HB.hammer_bgl_ff(q2, **kw)
    p = _momentum(q2)
    Hp = f - HB.MBC * p * g
    Hm = f + HB.MBC * p * g
    H0 = F1 / np.sqrt(q2)
    Ht = HB.MBC * p * HB.f2_from_p1(P1) / np.sqrt(q2)
    return Hp, Hm, H0, Ht


def dgamma_dq2(q2, vec, mlep):
    """dGamma/dq^2 up to the common constant G_F^2 |V_cb|^2 eta_EW^2 / (96 pi^3 M^2),
    which cancels in R(J/psi) and in every shape."""
    q2 = np.atleast_1d(np.asarray(q2, dtype=float))
    out = np.zeros_like(q2)
    ml2 = mlep ** 2
    for i, x in enumerate(q2):
        if x <= ml2:
            continue
        Hp, Hm, H0, Ht = helicity_amplitudes(x, vec)
        kin = _momentum(x) * x * (1. - ml2 / x) ** 2
        out[i] = kin * ((1. + ml2 / (2. * x)) * (Hp ** 2 + Hm ** 2 + H0 ** 2)
                        + (3. * ml2 / (2. * x)) * Ht ** 2)
    return out


def rate_grid(mlep, npoints=400):
    return np.linspace(mlep ** 2 * 1.0001, (HB.MBC - HB.MJPSI) ** 2 - 1e-6, npoints)


def _sigma_directions(cov_coeff):
    """Columns = 1 sigma principal directions of the coefficient covariance."""
    ev, evec = np.linalg.eigh(cov_coeff)
    idx = np.argsort(ev)[::-1]
    return evec[:, idx] * np.sqrt(np.maximum(ev[idx], 0.))[None, :]


def rate_with_band(vec, cov_coeff, mlep, npoints=400, shape=True):
    """Central dGamma/dq^2 and its 1 sigma band, linearised over the 15
    coefficient directions -- the same construction Hammer's delta_e
    eigenvariations give, so the band here is the band the templates will see.

    shape=True normalises every variation to unit area first, which is the band
    that matters for the fit: bc_norm floats, so a coherent up/down shift of the
    whole rate costs nothing and would otherwise inflate the plotted band. The
    normalisation uncertainty is reported separately.
    """
    grid = rate_grid(mlep, npoints)
    central = dgamma_dq2(grid, vec, mlep)
    norm = np.trapezoid(central, grid)
    ref = central / norm if shape else central
    mat = _sigma_directions(cov_coeff)
    var = np.zeros_like(central)
    for j in range(mat.shape[1]):
        var_curve = dgamma_dq2(grid, vec + mat[:, j], mlep)
        if shape:
            var_curve = var_curve / np.trapezoid(var_curve, grid)
        var += (var_curve - ref) ** 2
    return grid, ref, np.sqrt(var)


def r_jpsi(vec, npoints=2000):
    num = np.trapezoid(dgamma_dq2(rate_grid(M_TAU, npoints), vec, M_TAU),
                       rate_grid(M_TAU, npoints))
    den = np.trapezoid(dgamma_dq2(rate_grid(M_MU, npoints), vec, M_MU),
                       rate_grid(M_MU, npoints))
    return num / den


def r_jpsi_with_error(vec, cov_coeff):
    central = r_jpsi(vec)
    mat = _sigma_directions(cov_coeff)
    var = sum((r_jpsi(vec + mat[:, j]) - central) ** 2
              for j in range(mat.shape[1]))
    return central, np.sqrt(var)


# ---------------------------------------------------------------------------
# angular observables -- what dGamma/dq2 cannot see
# ---------------------------------------------------------------------------
# The q2 spectrum depends on H+^2 + H-^2, which is EVEN in g (since
# H_pm = f -+ M|p| g). A sign flip of g, or any g <-> f mix-up that preserves
# that sum, is invisible in every q2-only test, in both channels, and in
# R(J/psi). The observables below do see it, through the H+^2 - H-^2 and the
# H0 H_t interference, and Harrison publishes all three:
#
#     A_lambda_tau = 0.5093(42)   F_L^{J/psi} = 0.4421(55)   A_FB = -0.0567(61)
#
# Definitions (Harrison eq. 34), written in terms of the helicity amplitudes:
#   dGamma^{lambda_l = -1/2} ~ (H+^2 + H-^2 + H0^2)
#   dGamma^{lambda_l = +1/2} ~ (m^2/2q^2)(H+^2 + H-^2 + H0^2 + 3 H_t^2)
#   longitudinal (lambda_V = 0) = the H0 and H_t pieces
#   A_FB numerator = (3/4)(H-^2 - H+^2) - (3/2)(m^2/q^2) H0 H_t
# The sum of the two lepton-helicity pieces reproduces dgamma_dq2 exactly, which
# is the internal consistency check run by check_angular_observables.py.


def _amp_pieces(q2, vec, mlep):
    Hp, Hm, H0, Ht = helicity_amplitudes(q2, vec)
    ml2 = mlep ** 2
    kin = _momentum(q2) * q2 * (1. - ml2 / q2) ** 2
    sumH = Hp ** 2 + Hm ** 2 + H0 ** 2
    return kin, sumH, Hp, Hm, H0, Ht, ml2


def observable_integrands(grid, vec, mlep):
    """Returns dict of q2-differential numerators and the total, all up to the
    same common constant."""
    out = dict((k, np.zeros(len(grid))) for k in
               ('total', 'minus_half', 'plus_half', 'longitudinal', 'afb'))
    for i, q2 in enumerate(grid):
        if q2 <= mlep ** 2:
            continue
        kin, sumH, Hp, Hm, H0, Ht, ml2 = _amp_pieces(q2, vec, mlep)
        out['minus_half'][i] = kin * sumH
        out['plus_half'][i] = kin * (ml2 / (2. * q2)) * (sumH + 3. * Ht ** 2)
        out['total'][i] = out['minus_half'][i] + out['plus_half'][i]
        out['longitudinal'][i] = kin * ((1. + ml2 / (2. * q2)) * H0 ** 2
                                        + 3. * ml2 / (2. * q2) * Ht ** 2)
        out['afb'][i] = kin * (0.75 * (Hm ** 2 - Hp ** 2)
                               - 1.5 * (ml2 / q2) * H0 * Ht)
    return out


def angular_observables(vec, mlep, npoints=2000):
    """F_L, A_lambda and A_FB, integrated over q2."""
    grid = rate_grid(mlep, npoints)
    it = observable_integrands(grid, vec, mlep)
    tot = np.trapezoid(it['total'], grid)
    closure = abs(np.trapezoid(dgamma_dq2(grid, vec, mlep), grid) / tot - 1.)
    return {
        'F_L': np.trapezoid(it['longitudinal'], grid) / tot,
        'A_lambda': (np.trapezoid(it['minus_half'], grid)
                     - np.trapezoid(it['plus_half'], grid)) / tot,
        'A_FB': np.trapezoid(it['afb'], grid) / tot,
        'rate_closure': closure,
    }


# ---------------------------------------------------------------------------
# R(J/psi) straight from the lattice curves, with NO fit in the loop
# ---------------------------------------------------------------------------
def _interp_amplitudes(q2, grid, curves):
    g = np.interp(q2, grid, curves['g'])
    f = np.interp(q2, grid, curves['f'])
    F1 = np.interp(q2, grid, curves['F1'])
    F2 = np.interp(q2, grid, curves['F2'])
    p = _momentum(q2)
    return (f - HB.MBC * p * g, f + HB.MBC * p * g, F1 / np.sqrt(q2),
            HB.MBC * p * F2 / np.sqrt(q2))


def rate_from_curves(grid, curves, mlep, npoints=1500):
    """dGamma/dq2 integrated, taking the helicity FFs directly as tabulated
    curves rather than through BGL coefficients."""
    lo = max(mlep ** 2 * 1.0001, grid[0])
    qs = np.linspace(lo, grid[-1], npoints)
    Hp, Hm, H0, Ht = _interp_amplitudes(qs, grid, curves)
    ml2 = mlep ** 2
    kin = np.array([_momentum(q) for q in qs]) * qs * (1. - ml2 / qs) ** 2
    val = kin * ((1. + ml2 / (2. * qs)) * (Hp ** 2 + Hm ** 2 + H0 ** 2)
                 + (3. * ml2 / (2. * qs)) * Ht ** 2)
    return np.trapezoid(val, qs)


def r_jpsi_from_curves(grid, curves):
    return (rate_from_curves(grid, curves, M_TAU)
            / rate_from_curves(grid, curves, M_MU))


# ---------------------------------------------------------------------------
# the W helicity angle, from four-vectors -- used identically on every side
# ---------------------------------------------------------------------------
# theta_W: angle between the charged lepton, in the W* rest frame, and the W*
# flight direction in the Bc rest frame (Harrison fig. 5). Computed from plain
# four-vectors [E, px, py, pz] so that the SAME function measures the angle in
# Hammer's events, in EvtGen's events, and nowhere is a convention assumed.


def _boost_to_rest(p, frame):
    """Boost four-vector p into the rest frame of four-vector frame."""
    p = np.asarray(p, dtype=float)
    frame = np.asarray(frame, dtype=float)
    b = frame[1:] / frame[0]
    b2 = float(np.dot(b, b))
    if b2 < 1e-24:
        return p.copy()
    g = 1. / np.sqrt(1. - b2)
    bp = float(np.dot(b, p[1:]))
    e = g * (p[0] - bp)
    vec = p[1:] + ((g - 1.) * bp / b2 - g * p[0]) * b
    return np.concatenate([[e], vec])


def cos_theta_w(bc, jpsi, lep):
    """cos(theta_W) from the Bc, J/psi and charged-lepton four-vectors."""
    rest = np.array([np.sqrt(max(np.dot(bc, bc * np.array([1, -1, -1, -1])), 0.)),
                     0., 0., 0.])
    j = _boost_to_rest(jpsi, bc)
    l_ = _boost_to_rest(lep, bc)
    w = rest - j
    wdir = w[1:] / np.linalg.norm(w[1:])
    l_w = _boost_to_rest(l_, w)
    return float(np.dot(l_w[1:], wdir) / np.linalg.norm(l_w[1:]))


def dgamma_dcos(cos_grid, vec, mlep, npoints=600, flip=False):
    """dGamma/dcos(theta_W) integrated over q2, up to a constant.

    Angular structure (Becirevic et al., as cited in Harrison sec. IV A):
      (1+c)^2 H-^2 + (1-c)^2 H+^2 + 2(1-c^2) H0^2
        + (m^2/q^2) [ (1-c^2)(H+^2 + H-^2) + 2 (c H0 - H_t)^2 ]
    integrates to exactly dgamma_dq2 (checked in check_angular_observables.py).
    Which sign of c this convention corresponds to in cos_theta_w() is NOT
    assumed: the comparison reports both, and a mirror-image match is a
    convention statement, not a physics result.
    """
    qs = rate_grid(mlep, npoints)
    out = np.zeros(len(cos_grid))
    ml2 = mlep ** 2
    for q2 in qs:
        Hp, Hm, H0, Ht = helicity_amplitudes(q2, vec)
        kin = _momentum(q2) * q2 * (1. - ml2 / q2) ** 2
        c = -np.asarray(cos_grid) if flip else np.asarray(cos_grid)
        br = ((1 + c) ** 2 * Hm ** 2 + (1 - c) ** 2 * Hp ** 2
              + 2 * (1 - c ** 2) * H0 ** 2
              + (ml2 / q2) * ((1 - c ** 2) * (Hp ** 2 + Hm ** 2)
                              + 2 * (c * H0 - Ht) ** 2))
        out += kin * br
    return out


def ht_share(vec, mlep, npoints=2000):
    """Fraction of Gamma carried by the H_t (scalar/timelike) term.

    The one piece of the rate driven by dvec alone, and the lepton-mass
    suppressed one: ~8% of Gamma_tau, <0.4% of Gamma_mu. A discrepancy between
    Hammer and the analytic R that scales with this is an H_t discrepancy.
    """
    qs = rate_grid(mlep, npoints)
    tot = ht = 0.
    ml2 = mlep ** 2
    for q2 in qs:
        Hp, Hm, H0, Ht = helicity_amplitudes(q2, vec)
        kin = _momentum(q2) * q2 * (1. - ml2 / q2) ** 2
        tot += kin * ((1 + ml2 / (2 * q2)) * (Hp ** 2 + Hm ** 2 + H0 ** 2)
                      + 3 * ml2 / (2 * q2) * Ht ** 2)
        ht += kin * 3 * ml2 / (2 * q2) * Ht ** 2
    return ht / tot
