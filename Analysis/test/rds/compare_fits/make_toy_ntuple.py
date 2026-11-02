#!/usr/bin/env python3
'''
Toy R(Ds) signal ntuple with the branch names of the real one, to test the
comparison suite without the production file.

Generated: Bs -> Ds(*) (mu | tau(-> mu nu nu)) nu, Ds* -> Ds gamma,
Ds -> phi pi, phi -> K K, with simple phase-space decays (no form factors),
the Bs, Ds and tau flights, smeared tracks and vertices. The fit schemes are
emulated, not fitted:
    free  Ds = K + K + pi
    phic  KK 3-momentum with the phi mass, + pi
    cons  the phic Ds 3-momentum with the Ds mass
    dsc   the free Ds 3-momentum with the Ds mass
and the Bs vertex resolution is scaled per scheme (cons 0.85, phic 0.95,
dsc 0.92, free 1.0), so the suite must rank them in that order on the vertex
metrics. Everything downstream (Bs hypotheses, q2, m_miss2, helicity angles)
is computed with the same RDsKinematics / neutrino-solver code as the
ntuplizer.

    python3 make_toy_ntuple.py -o toy.root -n 5000 [--schemes cons,phic,free,dsc]

NOT a physics model: the shapes are phase space and the resolutions guesses.
'''

import os
import sys
import argparse
import types

import numpy as np
from scipy.special import chdtrc

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import Bmmm.Analysis.RDsKinematics  # noqa: F401
except Exception:
    pkg = types.ModuleType('Bmmm'); pkg.__path__ = []
    ana = types.ModuleType('Bmmm.Analysis'); ana.__path__ = [os.path.join(HERE, '..', '..', '..', 'python')]
    sys.modules['Bmmm'] = pkg
    sys.modules['Bmmm.Analysis'] = ana
from Bmmm.Analysis import RDsKinematics as K

M_TAU = 1.77686
CT_BS, CT_DS, CT_TAU = 0.04527, 0.01510, 0.008711     # cm
SCHEME_VTX_SCALE = {'cons': 0.85, 'phic': 0.95, 'dsc': 0.92, 'free': 1.0}
HYPS = ('coll', 'xyz_bs', 'math1_bs', 'math2_bs')
HYP_OBS = ('pt', 'eta', 'phi', 'beta', 'q2', 'm_miss2', 'pt_miss', 'pt_miss_sca',
           'e_star_mu', 'e_hash_mu', 'cos_theta_phi_ds', 'cos_theta_mu_w', 'chi')
PATHS = ('HLT_Mu7_IP4', 'HLT_Mu9_IP6', 'HLT_Mu12_IP6')


##########################################################################################
#####      vectorised kinematics
##########################################################################################

def p2body(M, m1, m2):
    return np.sqrt(np.clip((M * M - (m1 + m2) ** 2) * (M * M - (m1 - m2) ** 2), 0, None)) / (2 * M)


def iso(rng, n):
    c = rng.uniform(-1, 1, n)
    f = rng.uniform(-np.pi, np.pi, n)
    s = np.sqrt(1 - c * c)
    return np.stack([s * np.cos(f), s * np.sin(f), c], axis=1)


def four(m, p3):
    return np.concatenate([np.sqrt(m * m + np.sum(p3 * p3, axis=1))[:, None], p3], axis=1)


def boost_to_lab(p_rest, p_parent):
    '''p_rest in the parent rest frame -> the frame where the parent has p_parent.'''
    beta = -p_parent[:, 1:] / p_parent[:, :1]
    b2 = np.sum(beta * beta, axis=1)
    g = 1. / np.sqrt(1. - b2)
    bp = np.sum(beta * p_rest[:, 1:], axis=1)
    e = g * (p_rest[:, 0] - bp)
    with np.errstate(invalid='ignore', divide='ignore'):
        coef = np.where(b2 > 0, (g - 1.) * bp / b2, 0.) - g * p_rest[:, 0]
    pp = p_rest[:, 1:] + coef[:, None] * beta
    return np.concatenate([e[:, None], pp], axis=1)


def decay2(rng, parent, m1, m2):
    M = np.sqrt(np.clip(K_m2(parent), 0, None))
    p = p2body(M, m1, m2)
    d = iso(rng, len(parent)) * p[:, None]
    a, b = four(m1, d), four(m2, -d)
    return boost_to_lab(a, parent), boost_to_lab(b, parent)


def K_m2(p):
    return p[:, 0] ** 2 - np.sum(p[:, 1:] ** 2, axis=1)


def sample_q2(rng, n, mx, ml):
    lo, hi = ml * ml + 1e-6, (K.M_BS - mx) ** 2
    out = np.empty(0)
    while len(out) < n:
        q2 = rng.uniform(lo, hi, 4 * n)
        w = p2body(K.M_BS, mx, np.sqrt(q2)) * (1 - ml * ml / q2) ** 2 * (1 + ml * ml / (2 * q2))
        keep = rng.uniform(0, w.max() * 1.05, len(w)) < w
        out = np.concatenate([out, q2[keep]])
    return out[:n]


def nu_solutions(vis, d):
    '''Vectorised neutrino-pz solutions along d (unit), Bs mass; RJPsiNuReco.'''
    u = d / np.linalg.norm(d, axis=1, keepdims=True)
    E = vis[:, 0]
    ppar = np.sum(vis[:, 1:] * u, axis=1)
    pT_vec = vis[:, 1:] - ppar[:, None] * u
    pT2 = np.sum(pT_vec * pT_vec, axis=1)
    Kc = K.M_BS ** 2 + ppar ** 2 - pT2 - E ** 2
    A = 4 * (ppar ** 2 - E ** 2)
    B = 4 * ppar * Kc
    C = Kc ** 2 - 4 * E ** 2 * pT2
    disc = B * B - 4 * A * C
    real = disc >= 0
    sq = np.sqrt(np.clip(disc, 0, None))
    out = []
    for sgn in (+1, -1):
        x = (-B + sgn * sq) / (2 * A)
        nu3 = x[:, None] * u - pT_vec
        nu = np.concatenate([np.linalg.norm(nu3, axis=1)[:, None], nu3], axis=1)
        out.append(vis + nu)
    return out[0], out[1], real.astype(float)


def line_distance(p1, d1, p2, d2):
    '''Distance between two lines and the position of the closest point of
    line 2 along d2 (from p2).'''
    n = np.cross(d1, d2)
    nn = np.linalg.norm(n, axis=1)
    w = p2 - p1
    dist = np.abs(np.sum(w * n, axis=1)) / nn
    d2u = d2 / np.linalg.norm(d2, axis=1, keepdims=True)
    d1u = d1 / np.linalg.norm(d1, axis=1, keepdims=True)
    b = np.sum(d1u * d2u, axis=1)
    dd = np.sum(d1u * w, axis=1)
    e = np.sum(d2u * w, axis=1)
    t2 = (b * dd - e) / (1 - b * b)
    return dist, t2


def point_line_distance(p, a, d):
    du = d / np.linalg.norm(d, axis=1, keepdims=True)
    w = p - a
    return np.linalg.norm(w - np.sum(w * du, axis=1)[:, None] * du, axis=1)


##########################################################################################
#####      generation
##########################################################################################

def generate(rng, n, code, schemes):
    star, tau = code in (1, 3), code in (2, 3)
    mx = K.M_DSST if star else K.M_DS
    ml = M_TAU if tau else K.M_MU
    out = {}

    # ---- Bs production and decay ------------------------------------------
    pt_b = 10 + rng.exponential(8., n)
    eta_b = rng.uniform(-1.2, 1.2, n)
    phi_b = rng.uniform(-np.pi, np.pi, n)
    bs = four(K.M_BS, np.stack([pt_b * np.cos(phi_b), pt_b * np.sin(phi_b), pt_b * np.sinh(eta_b)], axis=1))
    pv = np.stack([rng.normal(0, 0.001, n), rng.normal(0, 0.001, n), rng.normal(0, 4., n)], axis=1)
    bs_dir = bs[:, 1:] / np.linalg.norm(bs[:, 1:], axis=1, keepdims=True)
    bg_b = np.linalg.norm(bs[:, 1:], axis=1) / K.M_BS
    sv = pv + bs_dir * (rng.exponential(CT_BS, n) * bg_b)[:, None]

    q2 = sample_q2(rng, n, mx, ml)
    # Bs -> X W* in the Bs frame
    p = p2body(K.M_BS, mx, np.sqrt(q2))
    dirx = iso(rng, n)
    x_rest = four(mx, dirx * p[:, None])
    w_rest = four(np.sqrt(q2), -dirx * p[:, None])
    X = boost_to_lab(x_rest, bs)
    W = boost_to_lab(w_rest, bs)
    lep, nu = decay2(rng, W, ml, 0.)
    if star:
        ds, gam = decay2(rng, X, K.M_DS, 0.)
    else:
        ds, gam = X, None
    if tau:
        mnn = rng.uniform(0., M_TAU - K.M_MU, n) * rng.uniform(0.6, 1., n)
        mu, nunu = _tau_decay(rng, lep, mnn)
        bg_t = np.linalg.norm(lep[:, 1:], axis=1) / M_TAU
        tdir = lep[:, 1:] / np.linalg.norm(lep[:, 1:], axis=1, keepdims=True)
        mu_origin = sv + tdir * (rng.exponential(CT_TAU, n) * bg_t)[:, None]
    else:
        mu, mu_origin = lep, sv
    phi_, pi = decay2(rng, ds, K.M_PHI, K.M_PI)
    k1, k2 = decay2(rng, phi_, K.M_K, K.M_K)
    bg_d = np.linalg.norm(ds[:, 1:], axis=1) / K.M_DS
    ddir = ds[:, 1:] / np.linalg.norm(ds[:, 1:], axis=1, keepdims=True)
    dsv = sv + ddir * (rng.exponential(CT_DS, n) * bg_d)[:, None]

    # ---- gen branches -------------------------------------------------------
    def put4(pre, p4):
        out[pre + '_pt'] = np.hypot(p4[:, 1], p4[:, 2])
        out[pre + '_eta'] = np.arcsinh(p4[:, 3] / out[pre + '_pt'])
        out[pre + '_phi'] = np.arctan2(p4[:, 2], p4[:, 1])
        out[pre + '_mass'] = np.sqrt(np.clip(K_m2(p4), 0, None))
    put4('gen_bs', bs)
    out['gen_bs_beta'] = np.linalg.norm(bs[:, 1:], axis=1) / bs[:, 0]
    for i, a in enumerate('xyz'):
        out['gen_bs_sv_' + a] = sv[:, i]
        out['gen_pv_' + a] = pv[:, i]
    out['gen_bs_lxyz'] = np.linalg.norm(sv - pv, axis=1)
    out['gen_bs_ct'] = out['gen_bs_lxyz'] / bg_b
    put4('gen_ham_dsx', X)
    if star:
        put4('gen_ham_ds', ds)
    else:
        for c in ('pt', 'eta', 'phi', 'mass'):
            out['gen_ham_ds_' + c] = np.full(n, np.nan)
    put4('gen_vis_mu', mu)
    out['gen_q2'] = K_m2(bs - X)
    g_obs = dict((k, np.empty(n)) for k in ('m_miss2', 'pt_miss', 'e_star_mu', 'e_hash_mu',
                                             'cos_theta_k_pi', 'cos_theta_phi_ds', 'cos_theta_mu_w', 'chi'))
    for i in range(n):
        o = K.semileptonic_observables(bs[i], ds[i], mu[i])
        h = K.helicity_angles(bs[i], ds[i], phi_[i], pi[i], k1[i], mu[i])
        o.update(h)
        for k in g_obs:
            g_obs[k][i] = o[k]
    for k, v in g_obs.items():
        out['gen_' + k] = v
    out['gen_cand'] = np.full(n, code, dtype=float)
    out['gen_sig'] = np.full(n, code, dtype=float)
    for o_, pdg in (('mu', 13), ('k1', 321), ('k2', -321), ('pi', 211)):
        out[o_ + '_gen_pdgid'] = np.full(n, pdg, dtype=float)

    # ---- reco tracks ----------------------------------------------------------
    def smear(p4, m, rel):
        ptv = np.hypot(p4[:, 1], p4[:, 2])
        f = 1 + rng.normal(0, rel + 0.0005 * ptv, len(p4))
        dphi = rng.normal(0, 5e-4, len(p4))
        c, s = np.cos(dphi), np.sin(dphi)
        px, py = p4[:, 1] * c - p4[:, 2] * s, p4[:, 1] * s + p4[:, 2] * c
        pz = p4[:, 3] * (1 + rng.normal(0, 5e-4, len(p4)))
        return four(m, np.stack([px, py, pz], axis=1) * f[:, None])
    rk1, rk2, rpi = smear(k1, K.M_K, 0.008), smear(k2, K.M_K, 0.008), smear(pi, K.M_PI, 0.008)
    rmu = smear(mu, K.M_MU, 0.010)
    # k1 = the leading kaon
    swap = np.hypot(rk2[:, 1], rk2[:, 2]) > np.hypot(rk1[:, 1], rk1[:, 2])
    rk1[swap], rk2[swap] = rk2[swap].copy(), rk1[swap].copy()
    for o_, p4 in (('mu', rmu), ('k1', rk1), ('k2', rk2), ('pi', rpi)):
        out[o_ + '_pt'] = np.hypot(p4[:, 1], p4[:, 2])
        out[o_ + '_eta'] = np.arcsinh(p4[:, 3] / out[o_ + '_pt'])
    kk = rk1 + rk2
    kkpi = kk + rpi
    out['kk_mass'] = np.sqrt(K_m2(kk))
    out['kkpi_mass'] = np.sqrt(K_m2(kkpi))
    out['mass'] = np.sqrt(K_m2(kkpi + rmu))
    out['dr_kk'] = _dr(rk1, rk2)
    out['dr_ds_mu'] = _dr(kkpi, rmu)
    out['dr_phi_pi'] = _dr(kk, rpi)
    pv_r = pv + rng.normal(0, 1, (n, 3)) * np.array([0.0010, 0.0010, 0.0030])
    for i, a in enumerate('xyz'):
        out['pv_' + a] = pv_r[:, i]
    ipmu = point_line_distance(pv, mu_origin, mu[:, 1:])
    out['mu_bs_dxy_sig'] = ipmu / 0.0025 + rng.normal(0, 1, n)
    out['mu_dxy_sig'] = out['mu_bs_dxy_sig'] * 0.9
    out['mu_reliso_clean_03'] = rng.exponential(0.08, n)
    out['b_reliso_clean_03'] = rng.exponential(0.12, n)

    # ---- photon ---------------------------------------------------------------
    if star:
        found = rng.uniform(0, 1, n) < 0.6
        g = gam.copy()
        g[:, 1:] *= (1 + rng.normal(0, 0.1, n))[:, None]
        g[:, 0] = np.linalg.norm(g[:, 1:], axis=1)
    else:
        found = np.zeros(n, dtype=bool)
        g = np.zeros((n, 4))
    rnd = ~found & (rng.uniform(0, 1, n) < 0.5)
    rg_dir = ddir + rng.normal(0, 0.08, (n, 3))
    rg_e = 0.2 + rng.exponential(0.4, n)
    g[rnd] = four(0., rg_dir[rnd] / np.linalg.norm(rg_dir[rnd], axis=1, keepdims=True) * rg_e[rnd, None])
    has_g = found | rnd

    # ---- schemes --------------------------------------------------------------
    for s in schemes:
        sc = SCHEME_VTX_SCALE.get(s, 1.0)
        if s == 'free':
            dsr, phir = kkpi, kk
        elif s == 'phic':
            phir = four(K.M_PHI, kk[:, 1:]); dsr = phir + rpi
        elif s == 'cons':
            phir = four(K.M_PHI, kk[:, 1:]); dsr = four(K.M_DS, (phir + rpi)[:, 1:])
        else:   # dsc
            phir = kk; dsr = four(K.M_DS, kkpi[:, 1:])
        vis = dsr + rmu
        # vertices
        sig_ds = np.array([0.0020, 0.0020, 0.0040]) * sc
        dsv_r = dsv + rng.normal(0, 1, (n, 3)) * sig_ds
        pull_to = sv + (0.6 * (mu_origin - sv) if tau else 0.)
        sig_bs = np.array([0.0025, 0.0025, 0.0050]) * sc
        bsv_r = pull_to + rng.normal(0, 1, (n, 3)) * sig_bs
        for v, pos, sig, p4 in (('phi', dsv_r, sig_ds, phir), ('ds', dsv_r, sig_ds, dsr), ('bs', bsv_r, sig_bs, vis)):
            pre = '%s_%s_' % (s, v)
            out[pre + 'good'] = np.ones(n)
            out[pre + 'status'] = np.zeros(n)
            for i, a in enumerate('xyz'):
                out[pre + a] = pos[:, i]
            for lab, (i, j) in zip(('xx', 'xy', 'xz', 'yy', 'yz', 'zz'),
                                   ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
                out[pre + 'cov_' + lab] = np.full(n, sig[i] ** 2 if i == j else 0.)
            ndof = 2. if (v == 'phi' and s in ('cons', 'phic')) or (v == 'ds' and s in ('cons', 'dsc')) else 1.
            chi2 = rng.chisquare(ndof, n)
            if v == 'bs' and tau:
                doca_true, _ = line_distance(dsv, ds[:, 1:], mu_origin, mu[:, 1:])
                chi2 = chi2 + (doca_true / 0.0015) ** 2
            out[pre + 'chi2'], out[pre + 'ndof'] = chi2, np.full(n, ndof)
            out[pre + 'prob'] = chdtrc(ndof, chi2)
            out[pre + 'mass'] = np.sqrt(np.clip(K_m2(p4), 0, None))
            out[pre + 'pt'] = np.hypot(p4[:, 1], p4[:, 2])
            fl = pos - pv_r
            out[pre + 'lxyz'] = np.linalg.norm(fl, axis=1)
            out[pre + 'lxyz_sig'] = out[pre + 'lxyz'] / np.sqrt(np.sum(sig ** 2) / 3 + 0.0015 ** 2)
            out[pre + 'cos3d'] = np.sum(fl * p4[:, 1:], axis=1) / (out[pre + 'lxyz'] * np.linalg.norm(p4[:, 1:], axis=1))
        if s == 'free':
            fail = rng.uniform(0, 1, n) < 0.01
            for k in list(out):
                if k.startswith('free_bs_') and k not in ('free_bs_good', 'free_bs_status'):
                    out[k][fail] = np.nan
            out['free_bs_good'][fail] = 0.
            out['free_bs_status'][fail] = 1.
        # topology
        doca, t2 = line_distance(dsv_r, dsr[:, 1:], mu_origin + rng.normal(0, 0.0012, (n, 3)), rmu[:, 1:])
        out[s + '_mu_doca_ds'] = doca
        out[s + '_mu_doca_ds_sig'] = doca / (0.0015 * sc)
        _, tds = line_distance(mu_origin, rmu[:, 1:], dsv_r, dsr[:, 1:])
        out[s + '_mu_poca_along_ds'] = tds
        bdir = bsv_r - pv_r
        out[s + '_mu_dist_to_bs_dir'] = line_distance(pv_r, bdir, mu_origin, rmu[:, 1:])[0]
        out[s + '_mu_ip3d_bs_pv_sig'] = point_line_distance(pv_r, mu_origin, rmu[:, 1:]) / 0.0025
        out[s + '_ds_lxyz_bs'] = np.linalg.norm(dsv_r - bsv_r, axis=1)
        out[s + '_ds_lxyz_bs_sig'] = out[s + '_ds_lxyz_bs'] / (0.0060 * sc)
        u = bdir / np.linalg.norm(bdir, axis=1, keepdims=True)
        for lab, p4 in (('', rk1 + rk2 + rpi + rmu), ('ds_', dsr)):
            par = np.sum(p4[:, 1:] * u, axis=1)
            perp = np.sqrt(np.clip(np.sum(p4[:, 1:] ** 2, axis=1) - par ** 2, 0, None))
            out[s + '_%sp_perp' % lab] = perp
            if lab == '':
                out[s + '_mcorr'] = np.sqrt(np.clip(K_m2(p4), 0, None) + perp ** 2) + perp
        # hypotheses
        hyps = {'coll': vis * (K.M_BS / np.sqrt(K_m2(vis)))[:, None]}
        mag = np.linalg.norm(vis[:, 1:], axis=1) * K.M_BS / np.sqrt(K_m2(vis))
        hyps['xyz_bs'] = four(K.M_BS, u * mag[:, None])
        m1, m2, real = nu_solutions(vis, u)
        hyps['math1_bs'], hyps['math2_bs'] = m1, m2
        out[s + '_math_real_bs'] = real
        for h, pb in hyps.items():
            vals = dict((q, np.empty(n)) for q in HYP_OBS)
            for i in range(n):
                o = K.semileptonic_observables(pb[i], dsr[i], rmu[i])
                o.update(K.helicity_angles(pb[i], dsr[i], phir[i], rpi[i], rk1[i], rmu[i]))
                for q in HYP_OBS:
                    vals[q][i] = o[q]
            for q in HYP_OBS:
                out['%s_%s_%s' % (s, h, q)] = vals[q]
        out[s + '_cos_theta_k_pi'] = np.array([K.helicity_angles(hyps['coll'][i], dsr[i], phir[i], rpi[i], rk1[i], rmu[i])['cos_theta_k_pi'] for i in range(n)])
        # photon
        pdg_ = dsr + g
        mdg = np.sqrt(np.clip(K_m2(pdg_), 0, None))
        ok = has_g & (np.abs(mdg - K.M_DSST) < 0.1)
        out[s + '_gamma_n'] = ok.astype(float)
        out[s + '_gamma_pt'] = np.where(ok, np.hypot(g[:, 1], g[:, 2]), np.nan)
        out[s + '_gamma_dm'] = np.where(ok, mdg - np.sqrt(K_m2(dsr)), np.nan)
        ecoll = np.array([K.e_gamma_collinear(dsr[i]) for i in range(n)])
        out[s + '_gamma_e_coll_over_pt'] = np.where(ok, ecoll / np.maximum(out[s + '_gamma_pt'], 1e-9), np.nan)
        visg = pdg_ + rmu
        cg = visg * (K.M_BS / np.sqrt(K_m2(visg)))[:, None]
        out[s + '_gamma_coll_q2'] = np.where(ok, K_m2(cg - pdg_), np.nan)
        out[s + '_gamma_coll_m_miss2'] = np.where(ok, K_m2(cg - pdg_ - rmu), np.nan)
    return out


def _tau_decay(rng, tau, mnn):
    '''tau -> mu + (nu nu) with m(nu nu) = mnn, isotropic.'''
    n = len(tau)
    p = p2body(M_TAU, K.M_MU, mnn)
    d = iso(rng, n) * p[:, None]
    mu = boost_to_lab(four(K.M_MU, d), tau)
    nn = np.concatenate([np.sqrt(mnn ** 2 + p ** 2)[:, None], -d], axis=1)
    return mu, boost_to_lab(nn, tau)


def _dr(a, b):
    def ep(p):
        t = np.hypot(p[:, 1], p[:, 2])
        return np.arcsinh(p[:, 3] / t), np.arctan2(p[:, 2], p[:, 1])
    e1, f1 = ep(a)
    e2, f2 = ep(b)
    df = (f1 - f2 + np.pi) % (2 * np.pi) - np.pi
    return np.hypot(e1 - e2, df)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('-n', type=int, default=5000, help='events per signal channel')
    ap.add_argument('--schemes', default='cons,phic,dsc,free')
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    import uproot
    rng = np.random.default_rng(args.seed)
    schemes = args.schemes.split(',')
    parts = [generate(rng, args.n, c, schemes) for c in (0, 1, 2, 3)]
    keys = parts[0].keys()
    data = dict((k, np.concatenate([p[k] for p in parts]).astype(np.float32)) for k in keys)
    n = len(data['gen_cand'])
    data['run'] = np.full(n, 1, dtype=np.int64)
    data['lumi'] = np.full(n, 1, dtype=np.int64)
    data['event'] = np.arange(n, dtype=np.int64)
    data['cand_idx'] = np.zeros(n, dtype=np.float32)
    data['charge_class'] = np.zeros(n, dtype=np.float32)
    data['mu_trig_match'] = (rng.uniform(0, 1, n) < 0.95).astype(np.float32)
    for p in PATHS:
        data[p] = (rng.uniform(0, 1, n) < 0.7).astype(np.float32)
    data['HLT_Mu9_IP6'] = np.ones(n, dtype=np.float32)
    # a few extra rows: same event, swapped K/pi species (removed by the
    # species check) and a second candidate (removed by one-per-event)
    extra = rng.choice(n, n // 30, replace=False)
    dup = dict((k, v[extra].copy()) for k, v in data.items())
    half = len(extra) // 2
    dup['k1_gen_pdgid'][:half] = 211
    dup['cand_idx'][:] = 1
    data = dict((k, np.concatenate([data[k], dup[k]])) for k in data)
    order = np.argsort(data['event'], kind='mergesort')
    data = dict((k, v[order]) for k, v in data.items())
    with uproot.recreate(args.output) as f:
        f['tree'] = data
    print('wrote %s: %d rows, %d branches' % (args.output, len(data['event']), len(data)))


if __name__ == '__main__':
    main()
