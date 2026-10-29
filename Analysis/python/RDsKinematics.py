'''
Pure-numpy kinematics of Bs -> Ds(-> phi(-> K K) pi) mu nu, shared by the
R(Ds) reconstruction (RDsCandidate) and the gen-level truth (RDsGenHistory).

No ROOT, no CMSSW: everything here works on (E, px, py, pz) numpy arrays and
is unit-tested in test/rds/test_rds_kinematics.py. Anything that exposes
energy()/px()/py()/pz() (a ROOT LorentzVector, a gen particle) is accepted
wherever a four-vector is expected.

Contents
  masses                    PDG masses [GeV] used by the R(Ds) code
  p4 helpers                np4, mass, pt, eta, phi, boost, ...
  Bs momentum hypotheses    collinear, equal velocity along a direction
  observables               q2, m_miss2, pt_miss, E*_mu, E#_mu, helicity angles
  soft photon               collinear expectation of the Ds* -> Ds gamma energy
  combinatorics             vectorised KK / KKpi preselection on raw momenta
'''

import numpy as np

##########################################################################################
#####      MASSES  [GeV]
##########################################################################################
M_BS   = 5.36692        # PDG; also the value the signal MC was generated with ('531:m0')
M_DS   = 1.96835
M_DSST = 2.1122
M_DP   = 1.86966        # D+, for the reflection masses
M_LC   = 2.28646        # Lambda_c+
M_PHI  = 1.019461
M_K    = 0.493677
M_PI   = 0.13957039
M_MU   = 0.1056583755
M_P    = 0.93827208816

##########################################################################################
#####      FOUR-VECTORS
##########################################################################################


def np4(p):
    '''(E, px, py, pz) float array from an array-like or anything exposing
    energy()/px()/py()/pz().'''
    if hasattr(p, 'energy'):
        return np.array([p.energy(), p.px(), p.py(), p.pz()], dtype=float)
    return np.asarray(p, dtype=float)


def p4_from_p3(px, py, pz, m):
    return np.array([np.sqrt(px * px + py * py + pz * pz + m * m), px, py, pz], dtype=float)


def p4_from_ptetaphi(pt, eta, phi, m):
    return p4_from_p3(pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), m)


def mass2(p):
    return p[0] * p[0] - p[1] * p[1] - p[2] * p[2] - p[3] * p[3]


def mass(p):
    m2 = mass2(p)
    return np.sqrt(m2) if m2 >= 0 else -np.sqrt(-m2)


def p3mag(p):
    return np.sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3])


def pt(p):
    return np.hypot(p[1], p[2])


def eta(p):
    t = pt(p)
    return np.arcsinh(p[3] / t) if t > 0 else np.sign(p[3]) * np.inf


def phi(p):
    return np.arctan2(p[2], p[1])


def unit3(v):
    n = np.sqrt(float(np.dot(v, v)))
    return v / n if n > 0 else v


def boost(four, beta):
    '''Express `four` in a frame moving with velocity `beta` (3-array). Same
    sign convention as JpsiChargedCandidate._boost and RJPsiGenHistory.'''
    four = np.asarray(four, dtype=float)
    b2 = float(np.dot(beta, beta))
    if b2 <= 0.:
        return four.copy()
    g  = 1. / np.sqrt(1. - b2)
    p  = four[1:]
    bp = float(np.dot(beta, p))
    e  = g * (four[0] - bp)
    pp = p + ((g - 1.) * bp / b2 - g * four[0]) * beta
    return np.array([e, pp[0], pp[1], pp[2]])


def beta_of(p):
    '''Velocity 3-vector of a time-like four-vector (None if not time-like).'''
    if p[0] <= 0. or mass2(p) <= 0.:
        return None
    return p[1:] / p[0]


##########################################################################################
#####      Bs MOMENTUM HYPOTHESES
##########################################################################################


def collinear_p4(visible, m_parent=M_BS):
    '''Collinear approximation (AN Sec. 5.4.1): the visible four-vector scaled
    by m_parent / m_vis.'''
    return np.asarray(visible, dtype=float) * (m_parent / mass(visible))


def equal_velocity_p4(direction, visible, m_parent=M_BS):
    '''"xyz" approximation (AN Sec. 5.4.2): |p| = |p_vis| m_parent / m_vis,
    along `direction` (a 3-array, e.g. PV -> Bs vertex).'''
    d   = unit3(np.asarray(direction, dtype=float))
    mag = p3mag(visible) * m_parent / mass(visible)
    p3  = d * mag
    return np.array([np.sqrt(m_parent ** 2 + mag * mag), p3[0], p3[1], p3[2]])


##########################################################################################
#####      OBSERVABLES
##########################################################################################


def semileptonic_observables(p_b, p_ds, p_mu):
    '''AN Sec. 5.6, for one Bs momentum hypothesis p_b. Returns a dict:
        pt, eta, phi, beta          of the Bs hypothesis
        q2          (p_b - p_ds)^2
        m_miss2     (p_b - p_ds - p_mu)^2
        pt_miss     |(p_b - p_ds - p_mu)_T|  (vector)
        pt_miss_sca pT(b) - pT(ds) - pT(mu)  (scalar)
        e_star_mu   muon energy in the Bs rest frame
        e_hash_mu   muon energy in the W* rest frame (NaN if q2 <= 0)
    '''
    p_b, p_ds, p_mu = np4(p_b), np4(p_ds), np4(p_mu)
    w    = p_b - p_ds
    miss = w - p_mu
    m_b  = mass(p_b)
    q2   = mass2(w)
    out = {
        'pt'         : pt(p_b),
        'eta'        : eta(p_b),
        'phi'        : phi(p_b),
        'beta'       : p3mag(p_b) / p_b[0] if p_b[0] > 0 else np.nan,
        'q2'         : q2,
        'm_miss2'    : mass2(miss),
        'pt_miss'    : pt(miss),
        'pt_miss_sca': pt(p_b) - pt(p_ds) - pt(p_mu),
        'e_star_mu'  : minkowski(p_b, p_mu) / m_b if m_b > 0 else np.nan,
        'e_hash_mu'  : minkowski(w, p_mu) / np.sqrt(q2) if q2 > 0 else np.nan,
    }
    return out


def minkowski(a, b):
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


def cos_angle_in_frame(p_daughter, p_frame, p_ref):
    '''cos of the angle between p_daughter, seen in the rest frame of p_frame,
    and the flight direction of p_frame seen from p_ref, i.e. minus p_ref in
    the p_frame rest frame. The usual helicity angle (as Bmmm cos_theta_v).'''
    b = beta_of(p_frame)
    if b is None or float(np.dot(b, b)) >= 1.:
        return np.nan
    d = boost(p_daughter, b)[1:]
    r = boost(p_ref, b)[1:]
    if not np.any(d) or not np.any(r):
        return np.nan
    return float(np.dot(unit3(d), -unit3(r)))


def helicity_angles(p_b, p_ds, p_phi, p_pi, p_k, p_mu):
    '''AN Fig. 9, for one Bs momentum hypothesis:
        cos_theta_k_pi    K vs pi in the phi rest frame (no Bs dependence;
                          the kaon passed is the one with the pion's charge)
        cos_theta_phi_ds  phi in the Ds rest frame vs the Ds flight direction
                          in the Bs rest frame
        cos_theta_mu_w    mu in the W* rest frame vs the W* flight direction
                          in the Bs rest frame (W* = Bs - Ds)
        chi               signed angle between the Ds (phi, pi) and the
                          W* (mu, nu) decay planes, in the Bs rest frame
    NaN where not computable (e.g. q2 <= 0 for cos_theta_mu_w).'''
    p_b, p_ds, p_phi, p_pi, p_k, p_mu = (np4(x) for x in (p_b, p_ds, p_phi, p_pi, p_k, p_mu))
    out = {}

    # K vs pi in the phi rest frame
    b_phi = beta_of(p_phi)
    if b_phi is not None and float(np.dot(b_phi, b_phi)) < 1.:
        k_ = boost(p_k, b_phi)[1:]
        pi_ = boost(p_pi, b_phi)[1:]
        out['cos_theta_k_pi'] = float(np.dot(unit3(k_), unit3(pi_))) if np.any(k_) and np.any(pi_) else np.nan
    else:
        out['cos_theta_k_pi'] = np.nan

    out['cos_theta_phi_ds'] = cos_angle_in_frame(p_phi, p_ds, p_b)
    w = p_b - p_ds
    out['cos_theta_mu_w'] = cos_angle_in_frame(p_mu, w, p_b) if mass2(w) > 0 else np.nan

    chi = np.nan
    b_b = beta_of(p_b)
    if b_b is not None and float(np.dot(b_b, b_b)) < 1.:
        z  = unit3(boost(p_ds, b_b)[1:])
        hb = boost(p_phi, b_b)[1:]
        lb = boost(p_mu, b_b)[1:]
        nv = unit3(np.cross(hb, z))
        nl = unit3(np.cross(lb, z))
        if np.any(nv) and np.any(nl):
            cc  = float(np.clip(np.dot(nv, nl), -1., 1.))
            ss  = float(np.dot(np.cross(nl, nv), z))
            chi = float(np.arctan2(ss, cc))
    out['chi'] = chi
    return out


##########################################################################################
#####      SOFT PHOTON
##########################################################################################


def e_gamma_collinear(p_ds):
    '''AN Sec. 5.6: expected photon energy of Ds* -> Ds gamma with the
    collinear approximation p(Ds*) = m(Ds*)/m(Ds) p(Ds):
        E = sqrt(m(Ds*)^2 + p(Ds*)^2) - sqrt(m(Ds)^2 + p(Ds)^2)'''
    p = p3mag(np4(p_ds))
    p_st = p * M_DSST / M_DS
    return np.sqrt(M_DSST ** 2 + p_st ** 2) - np.sqrt(M_DS ** 2 + p ** 2)


##########################################################################################
#####      REFLECTIONS
##########################################################################################


def mass_of(p3s, ms):
    '''Invariant mass of particles with 3-momenta p3s (list of 3-arrays) under
    the mass hypotheses ms.'''
    tot = np.zeros(4)
    for p3, m in zip(p3s, ms):
        tot += p4_from_p3(p3[0], p3[1], p3[2], m)
    return mass(tot)


def reflection_masses(p3_k1, p3_k2, p3_pi):
    '''m(KKpi) under swapped mass hypotheses, raw momenta:
        k1_as_pi, k2_as_pi   that kaon taken as a pion   (D+ -> K pi pi)
        k1_as_p,  k2_as_p    that kaon taken as a proton (Lc+ -> p K pi)
        k1_pi, k2_pi         m(K pi) of that kaon with the pion  (D0 -> K pi)
    '''
    return {
        'k1_as_pi': mass_of([p3_k1, p3_k2, p3_pi], [M_PI, M_K,  M_PI]),
        'k2_as_pi': mass_of([p3_k1, p3_k2, p3_pi], [M_K,  M_PI, M_PI]),
        'k1_as_p' : mass_of([p3_k1, p3_k2, p3_pi], [M_P,  M_K,  M_PI]),
        'k2_as_p' : mass_of([p3_k1, p3_k2, p3_pi], [M_K,  M_P,  M_PI]),
        'k1_pi'   : mass_of([p3_k1, p3_pi], [M_K, M_PI]),
        'k2_pi'   : mass_of([p3_k2, p3_pi], [M_K, M_PI]),
    }


##########################################################################################
#####      COMBINATORICS
##########################################################################################


def _pair_mass(px, py, pz, i, j, m1, m2):
    e1 = np.sqrt(px[i] ** 2 + py[i] ** 2 + pz[i] ** 2 + m1 * m1)
    e2 = np.sqrt(px[j] ** 2 + py[j] ** 2 + pz[j] ** 2 + m2 * m2)
    sx, sy, sz = px[i] + px[j], py[i] + py[j], pz[i] + pz[j]
    m2_ = (e1 + e2) ** 2 - sx * sx - sy * sy - sz * sz
    return np.sqrt(np.clip(m2_, 0., None)), e1 + e2, sx, sy, sz


def kkpi_triplets(px, py, pz, mu_p4, phi_window, kkpi_min, kkpi_max, max_mass,
                  phi_mass=M_PHI, mk=M_K, mpi=M_PI):
    '''All (i, j, k) with i, j the kaons and k the pion, from the tracks with
    momenta px, py, pz (numpy arrays, the tracks already preselected around one
    muon), such that

        |m(K_i K_j) - phi_mass| < phi_window
        kkpi_min < m(K_i K_j pi_k) < kkpi_max
        m(K_i K_j pi_k mu) < max_mass

    with raw momenta. No charge requirement (wrong-sign combinations are kept).
    i < j in input order: sort the inputs by pT (descending) to get k1 = leading.
    Returns an int array of shape (n, 3).'''
    n = len(px)
    if n < 3:
        return np.zeros((0, 3), dtype=int)
    ii, jj = np.triu_indices(n, k=1)
    mkk, ekk, sx, sy, sz = _pair_mass(px, py, pz, ii, jj, mk, mk)
    keep = np.abs(mkk - phi_mass) < phi_window
    ii, jj, ekk, sx, sy, sz = ii[keep], jj[keep], ekk[keep], sx[keep], sy[keep], sz[keep]
    if len(ii) == 0:
        return np.zeros((0, 3), dtype=int)

    # every pair x every pion candidate
    epi = np.sqrt(px ** 2 + py ** 2 + pz ** 2 + mpi * mpi)
    E  = ekk[:, None] + epi[None, :]
    X  = sx[:, None] + px[None, :]
    Y  = sy[:, None] + py[None, :]
    Z  = sz[:, None] + pz[None, :]
    m3 = np.sqrt(np.clip(E ** 2 - X ** 2 - Y ** 2 - Z ** 2, 0., None))
    kidx = np.arange(n)[None, :]
    ok = (kidx != ii[:, None]) & (kidx != jj[:, None]) & (m3 > kkpi_min) & (m3 < kkpi_max)

    mu = np4(mu_p4)
    E4 = E + mu[0]
    m4 = np.sqrt(np.clip(E4 ** 2 - (X + mu[1]) ** 2 - (Y + mu[2]) ** 2 - (Z + mu[3]) ** 2, 0., None))
    ok &= m4 < max_mass

    pair_idx, pi_idx = np.nonzero(ok)
    return np.stack([ii[pair_idx], jj[pair_idx], pi_idx], axis=1).astype(int)


def charge_class(q_k1, q_k2, q_pi, q_mu):
    '''AN Sec. 5.9:
        0  right sign      KK neutral and mu pi neutral
        1  mu pi wrong     KK neutral, mu pi charged
        2  KK wrong        KK charged (inclusive in mu pi)'''
    if q_k1 + q_k2 != 0:
        return 2
    if q_mu + q_pi != 0:
        return 1
    return 0
