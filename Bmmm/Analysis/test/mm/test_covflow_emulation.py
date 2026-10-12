"""Functional test of the MC emulation (CovflowEmulation.MCTrackEmulator,
inspector --emulate) without CMSSW.

ROOT is replaced by small python stand-ins (track, muon, Lorentz vector) that
implement exactly the reco::TrackBase parameter definitions; the covflow runs
are small random flows written with covflow's own write_manifest; the kill
maps are synthetic (one dead L1 region, data less efficient than MC there,
more efficient elsewhere). Checked against INDEPENDENT recomputations:

  1. track_with_params round trip: parameters and reference point unchanged
  2. clip_increment: the sampled covariance equals the clipped increment;
     clipped directions keep the base
  3. per muon: hit content after the kill, C' = morph_mc(C; c, c'),
     B = C + clip(C' - C), C_fit = B + clip(C* - B) with C* = covflow(B; c_f),
     fitted parameters = raw + delta_1 + delta_2, PDG mass, weights
  4. kill rate in the dead region = luminosity-averaged P(kill)
  5. delta_1 and delta_2 statistics over many events: covariance = clipped targets
  6. same (run, lumi, event) -> same result

Usage (any python with torch, zuko, numpy, scipy, particle):
    python3 test_covflow_emulation.py /path/to/covflow-repo
"""
import argparse, json, os, sys, tempfile
import numpy as np
from unittest import mock

COVFLOW = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                          os.environ.get('COVFLOW_REPO', '../../../../../covflow'))
if not os.path.isfile(os.path.join(COVFLOW, 'hitemu.py')):
    raise SystemExit('covflow repository not found at %s: pass it as the argument' % COVFLOW)
sys.path.insert(0, os.path.dirname(COVFLOW))
sys.path.insert(0, COVFLOW)


# ----------------------------------------------------------------- ROOT stand-ins
class P3:
    def __init__(self, x, y, z): self._v = (float(x), float(y), float(z))
    def x(self): return self._v[0]
    def y(self): return self._v[1]
    def z(self): return self._v[2]


class FakeTrack:
    '''reco::Track from (chi2, ndof, point, momentum, q, cov, algo, quality), with
    the reco::TrackBase parameter definitions.'''
    def __init__(self, chi2, ndof, point, mom, q, cov, algo, quality):
        self._ref, self._mom, self._q, self._cov = point, mom, int(q), np.array(cov)
        self._chi2, self._ndof, self._algo, self._qual = chi2, ndof, algo, quality
    def referencePoint(self): return self._ref
    def momentum(self): return self._mom
    def charge(self): return self._q
    def covariance(self): return self._cov
    def chi2(self): return self._chi2
    def ndof(self): return self._ndof
    def algo(self): return self._algo
    def qualityMask(self): return self._qual
    def vx(self): return self._ref.x()
    def vy(self): return self._ref.y()
    def vz(self): return self._ref.z()
    def pt(self): return float(np.hypot(self._mom.x(), self._mom.y()))
    def p(self): return float(np.sqrt(self.pt() ** 2 + self._mom.z() ** 2))
    def theta(self): return float(np.arctan2(self.pt(), self._mom.z()))
    def eta(self): return float(-np.log(np.tan(self.theta() / 2)))
    def phi(self): return float(np.arctan2(self._mom.y(), self._mom.x()))
    def qoverp(self): return self._q / self.p()
    def dxy(self):
        return (-self.vx() * self._mom.y() + self.vy() * self._mom.x()) / self.pt()
    def dsz(self):
        return (self.vz() * self.pt() / self.p()
                - (self.vx() * self._mom.x() + self.vy() * self._mom.y()) / self.pt()
                * self._mom.z() / self.p())


class FakeLV:
    def __init__(self, px, py, pz, E): self.v = np.array([px, py, pz, E], float)
    def pt(self): return float(np.hypot(*self.v[:2]))
    def mass(self): return float(np.sqrt(max(self.v[3] ** 2 - (self.v[:3] ** 2).sum(), 0)))
    def eta(self): return float(np.arcsinh(self.v[2] / self.pt()))
    def phi(self): return float(np.arctan2(self.v[1], self.v[0]))


class FakeMuon:
    '''pat::Muon: bestTrack(), p4; the copy constructor copies the C++ part only.'''
    def __init__(self, other=None, track=None, p4=None):
        if other is not None:
            track, p4 = other._trk, other._p4
        self._trk, self._p4 = track, p4
    def bestTrack(self): return self._trk
    def p4(self): return self._p4
    def pt(self): return self._p4.pt()
    def setP4(self, lv): self._p4 = lv


class FakeEvent:
    def __init__(self, run, lumi, evt):
        self.aux = mock.Mock(**{'run.return_value': run, 'luminosityBlock.return_value': lumi,
                                'event.return_value': evt})
    def eventAuxiliary(self): return self.aux


ROOT = mock.MagicMock()
ROOT.reco.Track = FakeTrack
ROOT.reco.TrackBase.Point = P3
ROOT.reco.TrackBase.Vector = P3
ROOT.reco.TrackBase.TrackQuality = lambda m: m
ROOT.reco.Candidate.LorentzVector = FakeLV
ROOT.pat.Muon = FakeMuon
sys.modules['ROOT'] = ROOT
for _m in ('PhysicsTools', 'PhysicsTools.HeppyCore', 'PhysicsTools.HeppyCore.utils',
           'PhysicsTools.HeppyCore.utils.deltar'):
    sys.modules.setdefault(_m, mock.MagicMock())
try:
    import Bmmm.Analysis.utils                   # inside CMSSW, after scram b
except ImportError:
    pkg = tempfile.mkdtemp()
    os.makedirs(os.path.join(pkg, 'Bmmm'))
    open(os.path.join(pkg, 'Bmmm', '__init__.py'), 'w').close()
    os.symlink(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            '..', '..', 'python')),
               os.path.join(pkg, 'Bmmm', 'Analysis'))
    sys.path.insert(0, pkg)

import torch
from covflow import flows as FL, data as D, features as F
import Bmmm.Analysis.utils as U
U.smatrix55_from_cov = lambda cov: np.array(cov)
import Bmmm.Analysis.CovflowEmulation as CE
CE.convert_cov = lambda m: np.array(m)
import train_covflow as T
import hitemu as H

rng = np.random.default_rng(7)
fails = []


def check(cond, what):
    print('  %-4s %s' % ('ok' if cond else 'FAIL', what))
    if not cond:
        fails.append(what)


# ----------------------------------------------------------------- covflow runs
def random_cov(n, r):
    s = np.array([0.02, 1e-3, 1.5e-3, 3e-3, 4e-3])          # relative q/p, then absolute
    A = r.normal(size=(n, 5, 5)) * 0.4 + np.eye(5)[None] * 1.0
    C = A @ np.swapaxes(A, 1, 2)
    return C * np.outer(s, s)[None]


def make_run(d, role, seed):
    ctx_br = ['%s_best_trk_pt' % role, '%s_best_trk_eta' % role, '%s_n_pix_hit' % role,
              '%s_pix_first_b_layer' % role, '%s_pix_first_e_disk' % role]
    cfg = FL.FlowConfig(n_features=15, n_context=5, transforms=2, hidden=(16, 16), bins=4, seed=seed)
    torch.manual_seed(seed);       fmc = FL.build_flow(cfg)
    torch.manual_seed(seed + 100); fdt = FL.build_flow(cfg)
    with torch.no_grad():
        for p in fdt.parameters():
            p.add_(0.05 * torch.randn_like(p))
    os.makedirs(d, exist_ok=True)
    FL.save_flow(fmc, os.path.join(d, 'flow_mc.pt')); FL.save_flow(fdt, os.path.join(d, 'flow_data.pt'))
    r = np.random.default_rng(seed)
    C = random_cov(4000, r); C[:, 0, :] *= 0.15; C[:, :, 0] *= 0.15
    X = F.packed_to_features(F.matrix_to_packed(C), 'logsigma_corr')
    Cx = np.column_stack([np.log(r.uniform(3, 20, 4000)), r.uniform(-2.4, 2.4, 4000),
                          r.integers(2, 6, 4000), r.integers(0, 3, 4000), r.integers(0, 3, 4000)])
    D.Standardiser.fit(X, Cx).save(os.path.join(d, 'scalers.json'))
    a = argparse.Namespace(param='logsigma_corr', features='all', cov_prefix='%s_cov_' % role,
                           context=ctx_br, log_pt='%s_best_trk_pt' % role, transforms=2,
                           hidden=[16, 16], bins=4, seed=seed, dtype='float32')
    T.write_manifest(d, a, ['log_pt', 'eta', 'n_pix_hit', 'b', 'e'], list(range(15)))


tmp = tempfile.mkdtemp()
for role, seed in (('mu1', 1), ('mu2', 2)):
    make_run(os.path.join(tmp, role), role, seed)
corrs = {r: U.make_cov_corrector(os.path.join(tmp, r), attr_suffix='_' + r, role=r)
         for r in ('mu1', 'mu2')}

# ----------------------------------------------------------------- kill maps
EPS_MC, EPS_DEAD, EPS_GOOD = 0.98, (0.30, 0.90), 0.995
DEAD_PHI = (0.0, 0.8)
surfaces = ['L1', 'D1+', 'D1-']
edges = {s: H.surface_edges(s, 48, 20) for s in surfaces}
dn, dd, mn, md, mn2, md2 = {}, {}, {}, {}, {}, {}
for s in surfaces:
    xe, ye = edges[s]
    nx, ny = len(xe) - 1, len(ye) - 1
    yc = 0.5 * (ye[1:] + ye[:-1])
    eps = np.full((2, nx, ny), EPS_MC if s != 'L1' else EPS_GOOD)
    if s == 'L1':
        dead = (yc > DEAD_PHI[0]) & (yc < DEAD_PHI[1])
        for r in (0, 1):
            eps[r][:, dead] = EPS_DEAD[r]
    dd[s] = np.full((2, nx, ny), 1e4); dn[s] = eps * dd[s]
    md[s] = np.full((nx, ny), 1e5); mn[s] = EPS_MC * md[s]
    md2[s] = np.full((nx, ny), 1e5); mn2[s] = EPS_MC * md2[s]
km = H.KillMaps(surfaces, edges, [(1, 10), (11, 20)], np.array([0.5, 0.5]), dn, dd, mn, md,
                mn2, md2, dict(epoch='toy', helix_origin_mc='dimuon_vertex',
                               helix_origin_data='dimuon_vertex')).finalise(30, 3.0, 0.0, 0.0, 'mcshape')
stem = os.path.join(tmp, 'killmaps_toy')
km.save(stem)

emu = CE.MCTrackEmulator(corrs, killmaps=stem, seed=99)


# ----------------------------------------------------------------- muons
def make_muon(r, pt=None, phi=None, eta=None, q=None):
    pt = r.uniform(4, 10) if pt is None else pt
    eta = r.uniform(-1., 1.) if eta is None else eta
    phi = r.uniform(-np.pi, np.pi) if phi is None else phi
    q = r.choice([-1, 1]) if q is None else q
    th = 2 * np.arctan(np.exp(-eta))
    p = pt / np.sin(th)
    mom = P3(pt * np.cos(phi), pt * np.sin(phi), p * np.cos(th))
    # reference point: the PCA to the beam line near (0.01, -0.02)
    bx, by = 0.01, -0.02
    s = bx * np.cos(phi) + by * np.sin(phi)
    ref = P3(s * np.cos(phi) + 0.002 * -np.sin(phi), s * np.sin(phi) + 0.002 * np.cos(phi), r.normal(0, 3))
    C = random_cov(1, r)[0]; C[0, :] *= 0.02 / p * 50; C[:, 0] *= 0.02 / p * 50
    trk = FakeTrack(10., 20, ref, mom, q, C, 4, 4)
    E = np.sqrt(p ** 2 + CE.MUON_MASS ** 2)
    mu = FakeMuon(track=trk, p4=FakeLV(mom.x(), mom.y(), mom.z(), E))
    count = sum(1 << (2 * b) for b in range(4))
    mu._hit_content = dict(n_pix_hit=4, n_pix_b_hit=4, n_pix_e_hit=0, n_pix_layer=4, n_trk_layer=13,
                           pix_first_b_layer=1, pix_first_e_disk=0, pix_first_layer=1,
                           n_pix_miss_inner=0, n_pix_inact_inner=0, pix_valid_mask=0b1111,
                           pix_miss_mask=0, pix_inact_mask=0, pix_hit_count=count)
    return mu


# ----------------------------------------------------------------- 1. round trip
print('1. track_with_params round trip')
mu = make_muon(rng)
t0 = mu.bestTrack()
t1 = U.track_with_params(t0, U.track_params(t0), t0.covariance())
check(np.allclose(U.track_params(t1), U.track_params(t0), rtol=0, atol=1e-12)
      and np.allclose(t1.referencePoint()._v, t0.referencePoint()._v, atol=1e-12)
      and np.allclose(t1.momentum()._v, t0.momentum()._v, atol=1e-12),
      'parameters, reference point and momentum unchanged')
par = U.track_params(t0) + np.array([1e-4, 2e-3, -3e-3, 5e-3, -4e-3])
t2 = U.track_with_params(t0, par, t0.covariance())
check(np.allclose(U.track_params(t2), par, atol=1e-12), 'shifted parameters reproduced exactly')
check(U.track_with_params(t0, -U.track_params(t0), t0.covariance()) is None, 'charge flip refused')

# ----------------------------------------------------------------- 2. clipping
print('2. clip_increment')
Bm = random_cov(1, rng)[0]
L = np.linalg.cholesky(Bm)
lam_true = np.array([0.5, 0.2, 0.0, -0.1, -0.3])
Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
Dm = L @ Q @ np.diag(lam_true) @ Q.T @ L.T
Dp, A, lam = CE.clip_increment(Dm[None], Bm[None])
check(np.allclose(np.sort(lam[0]), np.sort(lam_true), atol=1e-9), 'relative eigenvalues recovered')
v_neg = L @ Q[:, 4]                         # a clipped direction, in B-metric
Cfit = Bm + Dp[0]
w_dir = np.linalg.solve(Bm, v_neg)          # a functional that sees only that direction
check(abs(w_dir @ Cfit @ w_dir - w_dir @ Bm @ w_dir) < 1e-12 * abs(w_dir @ Bm @ w_dir) * 1e3,
      'clipped direction keeps the base (MC) variance')
xs = A[0] @ rng.standard_normal((5, 400000))
S = np.cov(xs)
Wm = np.linalg.inv(L)
check(np.abs(Wm @ (S - Dp[0]) @ Wm.T).max() < 0.01, 'sampled covariance = clipped increment')

# ----------------------------------------------------------------- 3. per-muon checks
print('3. per muon: kill bookkeeping, C\', C_fit, parameters')
n_check, bad = 0, []
killed_seen = False
for iev in range(400):
    mus = sorted([make_muon(rng), make_muon(rng)], key=lambda m: m.pt(), reverse=True)
    ev = FakeEvent(5, 1, iev)
    out = emu.process(mus, ev)
    orig = {id(m): m for m in mus}
    for cl in out:
        # find its original by the raw p4
        m0 = next(m for m in mus if m.p4() is cl._p4_raw)
        role = 'mu1' if m0 is mus[0] else 'mu2'
        corr = corrs[role]
        C = m0.bestTrack().covariance()
        e = cl.emu
        if e['killed']:
            killed_seen = True
            hc = cl._hit_content
            if not (hc['n_pix_hit'] == 3 and hc['pix_first_b_layer'] == 2
                    and hc['pix_valid_mask'] == 0b1110 and hc['n_trk_layer'] == 12
                    and cl._hit_content_raw['n_pix_hit'] == 4):
                bad.append('hit content after kill %s' % hc)
            c_raw = corr.context(m0, m0.bestTrack(), ev)
            c_fin = corr.context(cl, m0.bestTrack(), ev)
            Cp = corr.CFEAT.packed_to_matrix(corr.morph_mc_batch(
                np.array([U.cov_upper_triangle(C)]), np.array([c_raw]), np.array([c_fin])))[0]
            B = C + CE.clip_increment((Cp - C)[None], C[None])[0][0]
        else:
            if cl._hit_content != cl._hit_content_raw:
                bad.append('unkilled muon changed hit content')
            B = C
        c_fin = corr.context(cl, m0.bestTrack(), ev)
        Cs = corr.CFEAT.packed_to_matrix(corr._morph_batch(np.array([U.cov_upper_triangle(B)]),
                                                           np.array([c_fin]))[0])[0]
        Cfit = B + CE.clip_increment((Cs - B)[None], B[None])[0][0]
        Wb = np.linalg.inv(np.linalg.cholesky(Cfit))
        if np.abs(Wb @ (cl.cov_corr - Cfit) @ Wb.T).max() > 1e-6:
            bad.append('C_fit mismatch')
        par_fit = U.track_params(cl._emu_track)
        if not np.allclose(par_fit, U.track_params(m0.bestTrack()) + e['d1'] + e['d2'], rtol=0, atol=1e-12):
            bad.append('parameters != raw + delta_1 + delta_2')
        if (not e['killed']) and np.any(e['d1'] != 0):
            bad.append('delta_1 on an unkilled muon')
        if abs(cl.p4().mass() - CE.MUON_MASS) > 1e-9:
            bad.append('mass %g' % cl.p4().mass())
        n_check += 1
check(killed_seen and not bad, '%d muons recomputed independently%s'
      % (n_check, '' if not bad else ': ' + '; '.join(sorted(set(bad))[:4])))
check(out[0].pt() >= out[1].pt(), 'clones sorted by the smeared pT')

# ----------------------------------------------------------------- 4. kill rate
print('4. kill rate and weights')
n_dead = n_kill = 0
w_good = []
for iev in range(3000):
    mu = make_muon(rng, phi=rng.uniform(0.1, 0.7), eta=rng.uniform(-0.5, 0.5))
    out = emu.process([mu], FakeEvent(6, 1, iev))
    n_dead += 1; n_kill += out[0].emu['killed_L1']
    mu = make_muon(rng, phi=rng.uniform(-3.0, -1.0), eta=rng.uniform(-0.5, 0.5))
    w_good.append(emu.process([mu], FakeEvent(7, 1, iev))[0].emu['weight'])
expect = 0.5 * (1 - EPS_DEAD[0] / EPS_MC) + 0.5 * (1 - EPS_DEAD[1] / EPS_MC)
f = n_kill / n_dead
check(abs(f - expect) < 4 * np.sqrt(expect * (1 - expect) / n_dead),
      'kill rate in the dead region %.3f, expected %.3f (luminosity-averaged P(kill))' % (f, expect))
check(np.allclose(w_good, EPS_GOOD / EPS_MC), 'weight where data is more efficient = %.4f'
      % (EPS_GOOD / EPS_MC))

# ----------------------------------------------------------------- 5. smearing statistics
print('5. delta_1, delta_2 statistics (one muon, many events)')
base = make_muon(np.random.default_rng(11), pt=5., phi=0.4, eta=0.2, q=1)
d1s, d2s, d2u = [], [], []
for iev in range(6000):
    cl = emu.process([base], FakeEvent(8, 1, iev))[0]
    (d1s if cl.emu['killed'] else d2u).append(cl.emu['d2'])
    if cl.emu['killed']:
        d1s[-1] = (cl.emu['d1'], cl.emu['d2'])
ev = FakeEvent(8, 1, 0)
C = base.bestTrack().covariance()
corr = corrs['mu1']
c_raw = corr.context(base, base.bestTrack(), ev)
probe = FakeMuon(base); probe._hit_content = dict(base._hit_content, n_pix_hit=3, pix_first_b_layer=2,
                                                  pix_first_layer=2)
c_kill = corr.context(probe, base.bestTrack(), ev)
Cp = corr.CFEAT.packed_to_matrix(corr.morph_mc_batch(np.array([U.cov_upper_triangle(C)]),
                                                     np.array([c_raw]), np.array([c_kill])))[0]
D1p = CE.clip_increment((Cp - C)[None], C[None])[0][0]
Bk = C + D1p
Cs_u = corr.CFEAT.packed_to_matrix(corr._morph_batch(np.array([U.cov_upper_triangle(C)]), np.array([c_raw]))[0])[0]
D2u = CE.clip_increment((Cs_u - C)[None], C[None])[0][0]
Cs_k = corr.CFEAT.packed_to_matrix(corr._morph_batch(np.array([U.cov_upper_triangle(Bk)]), np.array([c_kill]))[0])[0]
D2k = CE.clip_increment((Cs_k - Bk)[None], Bk[None])[0][0]


def wdev(samples, target, ref):
    W = np.linalg.inv(np.linalg.cholesky(ref))
    return np.abs(W @ (np.cov(np.array(samples).T) - target) @ W.T).max()


n = len(d2u)
check(wdev(d2u, D2u, C) < 6 / np.sqrt(n), 'delta_2, unkilled (%d events): covariance = clip(C* - C), '
      'max whitened deviation %.3f' % (n, wdev(d2u, D2u, C)))
nk = len(d1s)
check(wdev([a for a, b in d1s], D1p, C) < 6 / np.sqrt(nk),
      'delta_1, killed (%d events): covariance = clip(C\' - C), max whitened deviation %.3f'
      % (nk, wdev([a for a, b in d1s], D1p, C)))
check(wdev([b for a, b in d1s], D2k, Bk) < 6 / np.sqrt(nk),
      'delta_2, killed: covariance = clip(C\'* - C\'), max whitened deviation %.3f'
      % wdev([b for a, b in d1s], D2k, Bk))
c12 = np.mean([np.outer(a, b) for a, b in d1s], 0)
W = np.linalg.inv(np.linalg.cholesky(C))
check(np.abs(W @ c12 @ W.T).max() < 6 / np.sqrt(nk), 'delta_1 and delta_2 independent')

# ----------------------------------------------------------------- 6. determinism
print('6. reproducibility')
a1 = emu.process([base], FakeEvent(9, 3, 12345))[0].emu
a2 = emu.process([base], FakeEvent(9, 3, 12345))[0].emu
a3 = emu.process([base], FakeEvent(9, 3, 12346))[0].emu
check(np.array_equal(a1['d2'], a2['d2']) and a1['killed'] == a2['killed'], 'same event -> same result')
check(not np.array_equal(a1['d2'], a3['d2']), 'next event -> another draw')

print('\n' + emu.summary())
if fails:
    raise SystemExit('\nFAILED: %d check(s): %s' % (len(fails), fails))
print('ALL OK')
