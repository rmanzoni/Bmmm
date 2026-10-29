#!/usr/bin/env python3
'''
Unit tests of the ROOT-free parts of the R(Ds) ntuplizer:

  RDsKinematics      combinatorics, Bs hypotheses, observables, helicity
                     angles (checked on decays generated with known angles),
                     reflection masses
  RJPsiNuReco        the neutrino solver with the Bs mass
  RDsGenHistory      signal classification and candidate truth, on stand-in
                     gen particles (B mixing, Ds*, tau, double charm, D+)
  HammerDsFF         channel / leaf bookkeeping (no Hammer needed)
  TrackMomentumCalib loading the momcal applier and the kind check
  L1Seeds2018        merged L1 tables vs the per-path pickles

Run with pytest or as a script:
    python3 -m pytest -q test_rds_kinematics.py
    python3 test_rds_kinematics.py
Inside CMSSW the Bmmm.Analysis package is used; outside, it is mapped onto
../../python. Optional inputs: MOMCAL_DIR (a momcal checkout), BMMM_DATADIR.
'''

import os
import sys
import json
import types
import itertools

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import Bmmm.Analysis.RDsKinematics  # noqa: F401
except Exception:
    pkg = types.ModuleType('Bmmm'); pkg.__path__ = []
    ana = types.ModuleType('Bmmm.Analysis'); ana.__path__ = [os.path.join(HERE, '..', '..', 'python')]
    sys.modules['Bmmm'] = pkg
    sys.modules['Bmmm.Analysis'] = ana

from Bmmm.Analysis import RDsKinematics as K
from Bmmm.Analysis import RDsGenHistory as G
from Bmmm.Analysis import HammerDsFF as HF
from Bmmm.Analysis.RJPsiNuReco import reconstruct

rng = np.random.default_rng(20181)


##########################################################################################
#####      helpers: two-body decays with chosen angles
##########################################################################################
def p_two_body(M, m1, m2):
    return np.sqrt((M * M - (m1 + m2) ** 2) * (M * M - (m1 - m2) ** 2)) / (2 * M)


def four(m, p, cos_t, phi):
    s = np.sqrt(1 - cos_t * cos_t)
    v = p * np.array([s * np.cos(phi), s * np.sin(phi), cos_t])
    return np.array([np.sqrt(m * m + p * p), v[0], v[1], v[2]])


def to_lab(p_rest, p_parent):
    '''p_rest is in the rest frame of p_parent: express it in the frame where
    the parent has four-momentum p_parent.'''
    return K.boost(p_rest, -p_parent[1:] / p_parent[0])


def make_decay(q2, cos_phi, cos_l, chi_in, cos_kpi_dir, bs_lab):
    '''Bs -> Ds W*, Ds -> phi pi, phi -> K K, W* -> mu nu with the given
    helicity angles (Ds along +z in the Bs frame, phi in the x-z plane).
    Returns lab four-vectors.'''
    mw = np.sqrt(q2)
    p = p_two_body(K.M_BS, K.M_DS, mw)
    ds = four(K.M_DS, p, 1., 0.)
    w = np.array([np.sqrt(mw * mw + p * p), 0., 0., -p])
    # Ds -> phi pi in the Ds frame; the reference is +z (Ds flight in the Bs frame)
    pp = p_two_body(K.M_DS, K.M_PHI, K.M_PI)
    phi_ds = four(K.M_PHI, pp, cos_phi, 0.)
    pi_ds = np.array([np.sqrt(K.M_PI ** 2 + pp ** 2), *(-phi_ds[1:])])
    # phi -> K K in the phi frame
    pk = p_two_body(K.M_PHI, K.M_K, K.M_K)
    k1_phi = four(K.M_K, pk, cos_kpi_dir, 1.0)
    k2_phi = np.array([k1_phi[0], *(-k1_phi[1:])])
    # W -> mu nu in the W frame; reference is -z (W flight in the Bs frame)
    pm = p_two_body(mw, K.M_MU, 0.)
    mu_w = four(K.M_MU, pm, -cos_l, chi_in)       # angle with -z
    nu_w = np.array([pm, *(-mu_w[1:])])
    # to the Bs frame
    phi_b = to_lab(phi_ds, ds); pi_b = to_lab(pi_ds, ds)
    k1_b = to_lab(to_lab(k1_phi, phi_ds), ds); k2_b = to_lab(to_lab(k2_phi, phi_ds), ds)
    mu_b = to_lab(mu_w, w); nu_b = to_lab(nu_w, w)
    bs_b = np.array([K.M_BS, 0., 0., 0.])
    rest = dict(bs=bs_b, ds=ds, phi=phi_b, pi=pi_b, k1=k1_b, k2=k2_b, mu=mu_b, nu=nu_b, w=w)
    lab = dict((k, to_lab(v, bs_lab)) for k, v in rest.items())
    return rest, lab


##########################################################################################
#####      RDsKinematics
##########################################################################################
def test_combinatorics_against_brute_force():
    for trial in range(20):
        n = rng.integers(3, 25)
        px, py, pz = rng.normal(0, 2, (3, n))
        mu = K.p4_from_p3(*rng.normal(0, 5, 3), K.M_MU)
        got = K.kkpi_triplets(px, py, pz, mu, 0.3, 1.2, 3.5, 9.0)
        got = set(map(tuple, got.tolist()))
        exp = set()
        p3 = np.stack([px, py, pz], axis=1)
        for i, j in itertools.combinations(range(n), 2):
            mkk = K.mass_of([p3[i], p3[j]], [K.M_K, K.M_K])
            if abs(mkk - K.M_PHI) >= 0.3:
                continue
            for k in range(n):
                if k in (i, j):
                    continue
                m3 = K.mass_of([p3[i], p3[j], p3[k]], [K.M_K, K.M_K, K.M_PI])
                tot = sum(K.p4_from_p3(*p3[x], m) for x, m in ((i, K.M_K), (j, K.M_K), (k, K.M_PI))) + mu
                if 1.2 < m3 < 3.5 and K.mass(tot) < 9.0:
                    exp.add((i, j, k))
        assert got == exp, (trial, got ^ exp)


def test_bs_hypotheses_have_bs_mass():
    vis = K.p4_from_p3(3., -1., 20., 2.6)
    assert abs(K.mass(K.collinear_p4(vis)) - K.M_BS) < 1e-9
    xyz = K.equal_velocity_p4(np.array([0.1, -0.05, 1.]), vis)
    assert abs(K.mass(xyz) - K.M_BS) < 1e-9
    assert abs(K.p3mag(xyz) - K.p3mag(vis) * K.M_BS / K.mass(vis)) < 1e-9


def test_observables_and_helicity_on_generated_decays():
    for trial in range(200):
        q2 = rng.uniform(0.05, (K.M_BS - K.M_DS) ** 2 - 0.05)
        cos_phi, cos_l = rng.uniform(-0.99, 0.99, 2)
        chi_in = rng.uniform(-np.pi + 0.01, np.pi - 0.01)
        bs_lab = K.p4_from_p3(*rng.normal(0, 6, 3), K.M_BS)
        rest, lab = make_decay(q2, cos_phi, cos_l, chi_in, rng.uniform(-0.99, 0.99), bs_lab)

        obs = K.semileptonic_observables(lab['bs'], lab['ds'], lab['mu'])
        assert abs(obs['q2'] - q2) < 1e-6
        assert abs(obs['m_miss2']) < 1e-6                                  # massless nu
        assert abs(obs['e_star_mu'] - rest['mu'][0]) < 1e-6
        e_mu_w = (q2 + K.M_MU ** 2) / (2 * np.sqrt(q2))
        assert abs(obs['e_hash_mu'] - e_mu_w) < 1e-6

        # the kaon with the pion's charge: either kaon works for the angle test,
        # the angle between K and pi in the phi frame is frame independent
        hl = K.helicity_angles(lab['bs'], lab['ds'], lab['phi'], lab['pi'], lab['k1'], lab['mu'])
        hr = K.helicity_angles(rest['bs'], rest['ds'], rest['phi'], rest['pi'], rest['k1'], rest['mu'])
        for k in hl:
            assert abs(hl[k] - hr[k]) < 1e-6, (k, hl[k], hr[k])
        assert abs(hl['cos_theta_phi_ds'] - cos_phi) < 1e-6
        assert abs(hl['cos_theta_mu_w'] - cos_l) < 1e-6
        # sign convention of chi: as JpsiChargedCandidate.helicity_angles
        d = (hl['chi'] + chi_in + np.pi) % (2 * np.pi) - np.pi
        assert abs(d) < 1e-6, (hl['chi'], chi_in)


def test_neutrino_solver_with_bs_mass():
    for trial in range(100):
        q2 = rng.uniform(0.5, 10.)
        bs_lab = K.p4_from_p3(*rng.normal(0, 8, 3), K.M_BS)
        _, lab = make_decay(q2, *rng.uniform(-0.9, 0.9, 2), rng.uniform(-3, 3), 0.3, bs_lab)
        vis = lab['ds'] + lab['mu']
        sols = reconstruct(vis, bs_lab[1:], m_parent=K.M_BS, clamp_negative_disc=True)
        best = min(sols, key=lambda s: np.max(np.abs(K.np4(s.p4_parent) - bs_lab)))
        assert np.max(np.abs(K.np4(best.p4_parent) - bs_lab)) < 1e-5


def test_reflections_and_charge_class():
    pk1, pk2, ppi = np.array([1., 2., 3.]), np.array([-0.5, 1., 2.]), np.array([0.2, 0.3, 4.])
    r = K.reflection_masses(pk1, pk2, ppi)
    assert abs(r['k1_as_pi'] - K.mass_of([pk1, pk2, ppi], [K.M_PI, K.M_K, K.M_PI])) < 1e-12
    assert r['k1_as_p'] > K.mass_of([pk1, pk2, ppi], [K.M_K, K.M_K, K.M_PI]) > r['k1_as_pi']
    assert K.charge_class(+1, -1, -1, +1) == 0
    assert K.charge_class(+1, -1, +1, +1) == 1
    assert K.charge_class(+1, +1, -1, +1) == 2
    assert K.charge_class(-1, -1, +1, +1) == 2


def test_e_gamma_collinear():
    # at rest the expected photon energy is the Ds*-Ds mass difference
    assert abs(K.e_gamma_collinear(np.array([K.M_DS, 0., 0., 0.])) - (K.M_DSST - K.M_DS)) < 1e-12


##########################################################################################
#####      RDsGenHistory on stand-ins
##########################################################################################
class GP(object):
    '''Minimal reco::GenParticle stand-in.'''
    def __init__(self, pdg, p4, status=2, vtx=(0., 0., 0.)):
        self._pdg, self._p4, self._st, self._v = pdg, np.asarray(p4, float), status, vtx
        self._mothers, self._daus = [], []

    def add(self, *daus):
        for d in daus:
            d._mothers.append(self)
            self._daus.append(d)
        return self

    def pdgId(self): return self._pdg
    def status(self): return self._st
    def charge(self):
        a = abs(self._pdg)
        q = {13: -1, 11: -1, 15: -1, 211: 1, 321: 1, 431: 1, 433: 1, 411: 1, 521: 1, 2212: 1}.get(a, 0)
        return q * (1 if self._pdg > 0 else -1)
    def energy(self): return self._p4[0]
    def px(self): return self._p4[1]
    def py(self): return self._p4[2]
    def pz(self): return self._p4[3]
    def pt(self): return K.pt(self._p4)
    def eta(self): return K.eta(self._p4)
    def phi(self): return K.phi(self._p4)
    def vx(self): return self._v[0]
    def vy(self): return self._v[1]
    def vz(self): return self._v[2]
    def numberOfMothers(self): return len(self._mothers)
    def mother(self, i): return self._mothers[i]
    def numberOfDaughters(self): return len(self._daus)
    def daughter(self, i): return self._daus[i]


def build_signal(code, mixing=False):
    '''Bs(-> mixed copy) -> Ds(*) l nu with Ds -> phi pi; lab four-vectors
    from make_decay. Returns (genparticles, dict of key particles).'''
    bs_lab = K.p4_from_p3(2., 1., 15., K.M_BS)
    _, lab = make_decay(4.0, 0.3, -0.2, 1.0, 0.5, bs_lab)
    sv = (0.01, 0.02, 0.05)
    bs = GP(531, lab['bs'], vtx=(0., 0., 0.))
    dec = bs
    if mixing:
        bsbar = GP(-531, lab['bs'], vtx=(0., 0., 0.))
        bs.add(bsbar)
        dec = bsbar
    tau = code in (2, 3)
    ds = GP(-431 if dec.pdgId() > 0 else 431, lab['ds'], vtx=sv)
    sgn = 1 if ds.pdgId() > 0 else -1
    if code in (1, 3):
        dsst = GP(433 * sgn, lab['ds'] + np.array([0.15, 0, 0, 0.1]), vtx=sv)
        gam = GP(22, [0.15, 0, 0, 0.1], status=1, vtx=sv)
        dsst.add(ds, gam)
        had = dsst
    else:
        had = ds
    phi = GP(333, lab['phi'], vtx=sv)
    pi = GP(211 * sgn, lab['pi'], status=1, vtx=sv)
    kp = GP(321, lab['k1'], status=1, vtx=sv)
    km = GP(-321, lab['k2'], status=1, vtx=sv)
    phi.add(kp, km)
    ds.add(phi, pi)
    mu = GP(13 * sgn, lab['mu'], status=1, vtx=sv)
    nu = GP(-14 * sgn, lab['nu'], status=1, vtx=sv)
    if tau:
        t = GP(15 * sgn, lab['mu'] + lab['nu'], vtx=sv)
        tnu = GP(16 * sgn, [0.1, 0, 0.1, 0], status=1, vtx=sv)
        tnumu = GP(-14 * sgn, [0.1, 0.1, 0, 0], status=1, vtx=sv)
        mu_t = GP(13 * sgn, lab['mu'], status=1, vtx=sv)
        t.add(mu_t, tnu, tnumu)
        nut = GP(-16 * sgn, lab['nu'], status=1, vtx=sv)
        dec.add(had, t, nut)
        mu = mu_t
    else:
        dec.add(had, mu, nu)
    parts = [bs] + ([dec] if mixing else []) + [had, ds, phi, pi, kp, km, mu]
    return parts, dict(bs=bs, dec=dec, ds=ds, pi=pi, kp=kp, km=km, mu=mu)


def test_signal_classification():
    for code in range(4):
        for mixing in (False, True):
            parts, p = build_signal(code, mixing)
            dec = G.BsSignalDecay.from_genparticles(parts)
            assert dec.code == code, (code, mixing, dec.code)
            assert dec.visible_complete
            assert G.same(dec.mu, p['mu'])
            info = G.bs_gen_info(dec, gen_pv=(0., 0., 0.))
            assert info['gen_bs_pdgid'] == 531
            assert abs(info['gen_bs_mass'] - K.M_BS) < 1e-9
            assert info['gen_bs_ct'] > 0
            obs = G.gen_observables(dec)
            if code == 0:
                assert abs(obs['gen_q2'] - 4.0) < 1e-6
                assert abs(obs['gen_m_miss2']) < 1e-6
            ham = G.gen_hammer_leaves(dec)
            assert np.isfinite(ham['gen_ham_lep_pt'])
            assert np.isfinite(ham['gen_ham_taumu_pt']) == (code in (2, 3))
            assert np.isfinite(ham['gen_ham_dsg_pt']) == (code in (1, 3))
            # candidate truth: the right tracks give the signal code back
            c = G.classify_candidate(p['kp'], p['km'], p['pi'], p['mu'])
            assert c['gen_cand'] == code, (code, mixing, c)
            assert c['gen_ds_true'] == 1. and c['gen_same_b'] == 1.
            # a muon from elsewhere
            other_b = GP(521, [6, 0, 0, 3])
            mu2 = GP(13, [3, 0.5, 0.5, 2.9], status=1)
            other_b.add(mu2)
            c = G.classify_candidate(p['kp'], p['km'], p['pi'], mu2)
            assert c['gen_cand'] == 20 and c['gen_same_b'] == 0.
            c = G.classify_candidate(p['kp'], p['km'], p['pi'], GP(13, [3, 0.5, 0.5, 2.9], status=1))
            assert c['gen_cand'] == 21
            c = G.classify_candidate(p['kp'], p['km'], None, p['mu'])
            assert c['gen_cand'] == -1


def test_hb_categories():
    # B0 -> Ds+ D-, D- -> mu nu X  (double charm, B0)
    b0 = GP(511, [6., 0., 0., 2.])
    ds = GP(431, [2.5, 0, 0, 1])
    dm = GP(-411, [2.5, 0, 0, -1])
    phi = GP(333, [1.2, 0, 0, 0.5])
    kp, km = GP(321, [0.6, 0.1, 0, 0.2], 1), GP(-321, [0.6, -0.1, 0, 0.3], 1)
    pi = GP(211, [0.6, 0, 0.1, 0.2], 1)
    mu = GP(-13, [1., 0, 0.2, 0.5], 1)
    phi.add(kp, km); ds.add(phi, pi); dm.add(mu); b0.add(ds, dm)
    c = G.classify_candidate(kp, km, pi, mu)
    assert c['gen_cand'] == 12 and c['gen_ds_b_pdgid'] == 511
    # Bs -> Ds1(2460) mu nu, Ds1 -> Ds pi0  (feed-down)
    bs = GP(531, [6., 0., 0., 2.])
    ds1 = GP(-20433, [2.6, 0, 0, 1])
    ds = GP(-431, [2.4, 0, 0, 1])
    pi0 = GP(111, [0.2, 0, 0, 0], 1)
    phi = GP(333, [1.2, 0, 0, 0.5]); kp, km = GP(321, [0.6, 0.1, 0, 0.2], 1), GP(-321, [0.6, -0.1, 0, 0.3], 1)
    pi = GP(-211, [0.6, 0, 0.1, 0.2], 1)
    mu = GP(13, [1., 0, 0.2, 0.5], 1)
    nu = GP(-14, [1., 0, 0.2, -0.5], 1)
    phi.add(kp, km); ds.add(phi, pi); ds1.add(ds, pi0); bs.add(ds1, mu, nu)
    c = G.classify_candidate(kp, km, pi, mu)
    assert c['gen_cand'] == 10 and c['gen_ds_mother_pdgid'] == -20433
    # D+ -> phi pi (not a Ds): charm ancestor is the D+
    dp = GP(411, [2.4, 0, 0, 1])
    phi = GP(333, [1.2, 0, 0, 0.5]); kp, km = GP(321, [0.6, 0.1, 0, 0.2], 1), GP(-321, [0.6, -0.1, 0, 0.3], 1)
    pi = GP(211, [0.6, 0, 0.1, 0.2], 1)
    phi.add(kp, km); dp.add(phi, pi)
    c = G.classify_candidate(kp, km, pi, mu)
    assert c['gen_cand'] == 30 and c['gen_charm_pdgid'] == 411 and c['gen_ds_true'] == 0.


def test_b_hadron_digits():
    assert all(G.is_b_hadron(x) for x in (511, -521, 531, 541, 5122, -5232, 513, 10531))
    assert not any(G.is_b_hadron(x) for x in (553, 100553, 431, 443, 4122, 321, 22))


##########################################################################################
#####      HammerDsFF bookkeeping, TrackMomentumCalib, L1 tables
##########################################################################################
def test_hammer_channel_bookkeeping():
    for code in range(4):
        parts, p = build_signal(code)
        dec = G.BsSignalDecay.from_genparticles(parts)
        leaves = dict(G.gen_hammer_leaves(dec))
        leaves.update(G.bs_gen_info(dec))
        leaves['gen_sig'] = float(dec.code)
        ch, st = HF.channel_of(leaves)
        assert ch == code and st == HF.STATUS_OK
        leaves['gen_ham_lep_pt'] = float('nan')
        assert HF.channel_of(leaves) == (None, HF.STATUS_NO_LEAVES)
    assert HF.channel_of({'gen_sig': -1.})[1] == HF.STATUS_NOT_SIGNAL
    assert len(HF.BRANCH_NAMES) == 2 + 2 * HF.N_MAX_COORDS
    assert set(HF.INPUT_BRANCHES) >= set(G.HAM_BRANCHES)


def test_track_momentum_calib():
    momcal = os.environ.get('MOMCAL_DIR', '')
    jpath = os.path.join(momcal, 'applier', 'data', 'rds_trkcalib_2018BParking_v3.json')
    if not os.path.isfile(jpath):
        print('    (skipped: set MOMCAL_DIR to a momcal checkout)')
        return
    import shutil, tempfile
    from Bmmm.Analysis.TrackMomentumCalib import TrackMomentumCalib, make_trk_calib, KEY_OFFSET
    tmp = tempfile.mkdtemp()
    shutil.copy(jpath, tmp)
    shutil.copy(os.path.join(momcal, 'applier', 'python', 'rds_trkcalib.py'), tmp)
    cal = TrackMomentumCalib(os.path.join(tmp, os.path.basename(jpath)), verbose=False)
    z, f = cal.z_and_f(320000, 10, 123456789, TrackMomentumCalib.key('lost', 7), 2.0, 0.5, -1)
    # same numbers as the applier called directly
    zz = cal.cal.gauss(320000, 10, 123456789, KEY_OFFSET['lost'] + 7)
    ff = cal.cal.factor(2.0 * np.cosh(0.5), 2.0, 0.5, -1, zz)
    assert z == zz and f == ff and 0.9 < f < 1.1
    # a file with muon terms and an applier without `kind` must be refused
    cfg = json.load(open(jpath))
    cfg['per_bin_muon'] = cfg['per_bin']
    fake = os.path.join(tmp, 'fake_v4.json')
    json.dump(cfg, open(fake, 'w'))
    try:
        TrackMomentumCalib(fake, verbose=False)
        raise AssertionError('a muon-term file with a kind-less applier was accepted')
    except RuntimeError:
        pass
    # data must be refused
    try:
        make_trk_calib(jpath, mc=False)
        raise AssertionError('--trk-calib accepted on data')
    except RuntimeError:
        pass


def test_l1_tables():
    datadir = os.environ.get('BMMM_DATADIR', os.path.join(HERE, '..', '..', 'data'))
    if not os.path.isfile(os.path.join(datadir, 'HLT_Mu7_IP4.pickle')):
        print('    (skipped: no L1 tables)')
        return
    import pickle, tempfile
    from Bmmm.Analysis.L1Seeds2018 import load_tables, write_merged, MERGED_NAME
    paths = ['HLT_Mu7_IP4', 'HLT_Mu8_IP3', 'HLT_Mu8_IP5', 'HLT_Mu8_IP6', 'HLT_Mu8p5_IP3p5',
             'HLT_Mu9_IP4', 'HLT_Mu9_IP5', 'HLT_Mu9_IP6', 'HLT_Mu10p5_IP3p5', 'HLT_Mu12_IP6']
    merged, _ = load_tables(paths, datadir)
    # every value of every per-path table is in the merged one, unchanged
    for p in paths:
        t = pickle.load(open(os.path.join(datadir, p + '.pickle'), 'rb'))
        for seed, runs in t.items():
            for run, lss in runs.items():
                for ls, v in lss.items():
                    assert merged[seed][run][ls] == v
    tmp = tempfile.mkdtemp()
    write_merged(paths, datadir, os.path.join(tmp, MERGED_NAME))
    again, src = load_tables(paths, tmp)
    assert again == merged and src.endswith(MERGED_NAME)


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_') and callable(f)]
    for name, func in tests:
        print('%-50s' % name, end='', flush=True)
        func()
        print(' ok')
    print('%d tests passed' % len(tests))
