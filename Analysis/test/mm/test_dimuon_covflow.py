"""Functional test of the dimuon covflow path (MuMuCandidate --covflow-mu1/-mu2).

No CMSSW needed: ROOT and the HeppyCore helpers are mocked, so only the covflow
plumbing runs. Two covflow runs (mu1, mu2) with small random flows are written
with covflow's OWN write_manifest (schema covflow/1, as the Run 3 productions),
then:
  * the manifest is translated into the Bmmm context getters;
  * a mu2 flow installed for mu1 is refused;
  * the per-role correction equals covflow.correct.morph to the last bit;
  * a muon that is mu1 in one pair and mu2 in another carries both, apart;
  * the hand-written covflow.json of the J/psi + track channel still loads;
  * MuMuCandidate.fit_track / expose_cov / set_cov_correctors behave.

Usage (covflow env, or any python with torch, zuko, numpy, scipy, particle):
    python3 test_dimuon_covflow.py /path/to/covflow-repo
"""
import os, sys, types, json, argparse, tempfile
import numpy as np
from unittest import mock

COVFLOW = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                          os.environ.get('COVFLOW_REPO', '../../../../../covflow'))
if not os.path.isfile(os.path.join(COVFLOW, 'train_covflow.py')):
    raise SystemExit('covflow repository not found at %s: pass it as the argument' % COVFLOW)
sys.path.insert(0, os.path.dirname(COVFLOW))     # `from covflow import ...` (namespace package)
sys.path.insert(0, COVFLOW)                      # train_covflow.write_manifest
sys.modules['ROOT'] = mock.MagicMock()
for _m in ('PhysicsTools', 'PhysicsTools.HeppyCore', 'PhysicsTools.HeppyCore.utils',
           'PhysicsTools.HeppyCore.utils.deltar'):
    sys.modules.setdefault(_m, mock.MagicMock())
try:
    import Bmmm.Analysis.utils                   # inside CMSSW, after scram b
except ImportError:                              # a bare checkout: python/ as Bmmm.Analysis
    pkg = tempfile.mkdtemp()
    os.makedirs(os.path.join(pkg, 'Bmmm'))
    open(os.path.join(pkg, 'Bmmm', '__init__.py'), 'w').close()
    os.symlink(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            '..', '..', 'python')),
               os.path.join(pkg, 'Bmmm', 'Analysis'))
    sys.path.insert(0, pkg)

import torch
from covflow import flows as FL, data as D, features as F, correct as CF
import Bmmm.Analysis.utils as U
import train_covflow as T

rng = np.random.default_rng(1)

def random_cov(n):
    A = rng.normal(size=(n, 5, 5)) * np.array([1e-3, 1e-3, 1e-3, 1e-2, 1e-2])[None, :, None]
    return A @ np.swapaxes(A, 1, 2) + np.eye(5)[None] * 1e-6

def make_run(d, role, seed):
    ctx_br = ['%s_best_trk_pt' % role, '%s_best_trk_eta' % role, '%s_n_pix_hit' % role,
              '%s_pix_first_b_layer' % role, '%s_pix_first_e_disk' % role]
    cfg = FL.FlowConfig(n_features=15, n_context=5, transforms=2, hidden=(16, 16), bins=4, seed=seed)
    torch.manual_seed(seed);       fmc = FL.build_flow(cfg)
    torch.manual_seed(seed + 100); fdt = FL.build_flow(cfg)
    # perturb the data flow so the morph is not the identity
    with torch.no_grad():
        for p in fdt.parameters():
            p.add_(0.05 * torch.randn_like(p))
    os.makedirs(d, exist_ok=True)
    FL.save_flow(fmc, os.path.join(d, 'flow_mc.pt')); FL.save_flow(fdt, os.path.join(d, 'flow_data.pt'))
    X = F.packed_to_features(F.matrix_to_packed(random_cov(2000)), 'logsigma_corr')
    C = np.column_stack([np.log(rng.uniform(3, 20, 2000)), rng.uniform(-2.4, 2.4, 2000),
                         rng.integers(2, 6, 2000), rng.integers(0, 3, 2000), rng.integers(0, 3, 2000)])
    D.Standardiser.fit(X, C).save(os.path.join(d, 'scalers.json'))
    a = argparse.Namespace(param='logsigma_corr', features='all', cov_prefix='%s_cov_' % role,
                           context=ctx_br, log_pt='%s_best_trk_pt' % role, transforms=2,
                           hidden=[16, 16], bins=4, seed=seed, dtype='float32')
    T.write_manifest(d, a, ['log_pt', 'eta', 'n_pix_hit', 'b', 'e'], list(range(15)))
    return fmc, fdt, D.Standardiser.load(os.path.join(d, 'scalers.json'))

tmp = tempfile.mkdtemp()
runs = {r: make_run(os.path.join(tmp, '2024', r, 'task_0'), r, s) for r, s in (('mu1', 1), ('mu2', 2))}

class Trk:
    def __init__(self, pt, eta): self._pt, self._eta = pt, eta
    def pt(self): return self._pt
    def eta(self): return self._eta
class Mu:
    def __init__(self, pt, eta, nhit, fb, fe):
        self._t = Trk(pt, eta)
        self.cov = random_cov(1)[0]; self.is_cov_pos_def = True
        self._hit_content = dict(n_pix_hit=nhit, pix_first_b_layer=fb, pix_first_e_disk=fe)
    def bestTrack(self): return self._t

corr = {r: U.make_cov_corrector(os.path.join(tmp, '2024', r, 'task_0'), attr_suffix='_' + r, role=r)
        for r in ('mu1', 'mu2')}
for r, c in corr.items():
    assert c.schema == 'covflow/1' and c.role == r, (c.schema, c.role)
    assert c.context_names == ['log_pt', 'eta', 'n_pix_hit', 'pix_first_b_layer', 'pix_first_e_disk'], c.context_names
print('manifest translation  ok:', corr['mu1'].context_names)

# wrong role refused
try:
    U.make_cov_corrector(os.path.join(tmp, '2024', 'mu2', 'task_0'), attr_suffix='_mu1', role='mu1')
    raise SystemExit('FAILED: a mu2 flow was accepted for mu1')
except ValueError as e:
    print('wrong role refused    ok:', str(e)[:70], '...')

muons = sorted([Mu(rng.uniform(3, 15), rng.uniform(-2, 2), 4, 1, 0) for _ in range(4)],
               key=lambda m: m.bestTrack().pt(), reverse=True)
corr['mu1'].prime(muons[:-1]); corr['mu2'].prime(muons[1:])

# reference: covflow.correct.morph per role
for r, sub in (('mu1', muons[:-1]), ('mu2', muons[1:])):
    fmc, fdt, sc = runs[r]
    packed = np.array([U.cov_upper_triangle(m.cov) for m in sub])
    ctx = np.array([[np.log(m.bestTrack().pt()), m.bestTrack().eta(), 4, 1, 0] for m in sub])
    ref, _, _ = CF.morph(packed, ctx, fmc, fdt, sc, param='logsigma_corr', feature_indices='all')
    got = np.array([U.cov_upper_triangle(getattr(m, 'cov_corr_' + r)) for m in sub])
    rel = np.max(np.abs(got - ref) / np.abs(ref))
    moved = np.max(np.abs(got - packed) / np.abs(packed))
    assert rel < 1e-5, rel
    assert all(getattr(m, 'covflow_ok_' + r) for m in sub)
    print('%s vs covflow.morph   ok: max rel diff %.1e (correction itself moves up to %.0f%%)'
          % (r, rel, 100 * moved))

# a muon in both roles carries two different corrections
mid = muons[1]
d12 = np.max(np.abs(mid.cov_corr_mu1 - mid.cov_corr_mu2))
assert d12 > 0 and not hasattr(muons[0], 'cov_corr_mu2') and not hasattr(muons[-1], 'cov_corr_mu1')
print('roles kept apart      ok: middle muon differs by %.2e between the mu1 and mu2 flows' % d12)
print(corr['mu1'].summary()); print(corr['mu2'].summary())

# --- the hand-written manifest of the J/psi + track channel still loads, unsuffixed
import shutil
old = os.path.join(tmp, 'handwritten'); shutil.copytree(os.path.join(tmp, '2024', 'mu1', 'task_0'), old)
json.dump({"context": ["log_pt", "eta", "n_pix_hit", "pix_first_b_layer", "pix_first_e_disk"],
           "features": "all", "param": "logsigma_corr", "transforms": 2, "hidden": [16, 16],
           "bins": 4, "seed": 1}, open(os.path.join(old, 'covflow.json'), 'w'))
c0 = U.make_cov_corrector(old)
m0 = Mu(7.0, 0.3, 4, 1, 0); c0.prime([m0])
assert c0.schema == 'handwritten' and hasattr(m0, 'cov_corr') and m0.covflow_ok
print('hand-written manifest ok: J/psi-channel layout loads, memoized as cov_corr')

# --- MuMuCandidate hooks (ROOT mocked: only the covflow plumbing is exercised)
from Bmmm.Analysis import MuMuCandidate as MMC
built = []
MMC.track_with_cov = lambda raw, cov: ('rebuilt', cov)
corrs = MMC.Candidate.set_cov_correctors({'mu1': os.path.join(tmp, '2024', 'mu1', 'task_0'),
                                          'mu2': os.path.join(tmp, '2024', 'mu2', 'task_0')})
mus = sorted([Mu(rng.uniform(3, 15), rng.uniform(-2, 2), 4, 1, 0) for _ in range(3)],
             key=lambda m: m.bestTrack().pt(), reverse=True)
MMC.Candidate.prime_cov_correctors(mus)
cand = MMC.Candidate.__new__(MMC.Candidate)
t1 = cand.fit_track(mus[1], 'mu1'); t2 = cand.fit_track(mus[1], 'mu2')
assert t1[0] == 'rebuilt' and np.array_equal(t1[1], mus[1].cov_corr_mu1)
assert np.array_equal(t2[1], mus[1].cov_corr_mu2) and cand.fit_track(mus[1], 'mu1') is t1
cand.expose_cov(mus[1], 'mu2')
assert mus[1]._cov_fit_track is t2 and np.array_equal(mus[1].cov_corr, mus[1].cov_corr_mu2)
try:
    MMC.Candidate.set_cov_correctors({'mu1': os.path.join(tmp, '2024', 'mu1', 'task_0'), 'mu2': ''})
    raise SystemExit('FAILED: one role only was accepted')
except RuntimeError:
    pass
MMC.Candidate.set_cov_correctors({})
assert cand.fit_track(mus[0], 'mu1') is mus[0].bestTrack()
print('MuMuCandidate hooks   ok: per-role fit tracks, expose_cov, both roles required, off = bestTrack()')

# --- raw refit: the *_raw quantities come from the fit with the RAW covariances,
#     through the same displacement / flight code as the nominal ones
class P3:
    def __init__(self, x, y, z): self._v = np.array([x, y, z], float)
    def x(self): return self._v[0]
    def y(self): return self._v[1]
    def z(self): return self._v[2]
class Vec(P3):
    def Dot(self, o): return float(self._v @ o._v)
    def R(self): return float(np.linalg.norm(self._v))
    def Mag2(self): return float(self._v @ self._v)
    def eta(self): return float(np.arcsinh(self._v[2] / np.hypot(*self._v[:2])))
    def phi(self): return float(np.arctan2(self._v[1], self._v[0]))
    def __truediv__(self, a): return Vec(*(self._v / a))
class Meas:
    def __init__(self, v, e): self.v, self.e = v, e
    def value(self): return self.v
    def error(self): return self.e
    def significance(self): return self.v / self.e
class Vtx:
    def __init__(self, x, y, z, chi2):
        self.p = P3(x, y, z); self.c = chi2
    def isValid(self): return True
    def position(self): return self.p
    def vertexState(self): return self.p
    def normalisedChiSquared(self): return self.c
class Dist:
    def __init__(self, nd): self.nd = nd
    def distance(self, ref, st):
        d = st._v - ref.position()._v
        return Meas(float(np.linalg.norm(d[:self.nd])), 0.002)
class P4:
    def Vect(self): return Vec(1.0, 2.0, 0.5)
    def mass(self): return 3.0969
MMC.Vector3D = Vec
MMC.ROOT.Math.XYZVector = Vec
MMC.ROOT.VertexDistanceXY = lambda: Dist(2)
MMC.ROOT.VertexDistance3D = lambda: Dist(3)
class FitCounter:
    n = 0
    def Fit(self, tracks):
        FitCounter.n += 1
        return Vtx(0.031, 0.052, 0.4, 0.8)         # the raw fit lands somewhere else
MMC.vtxfit = FitCounter()
MMC.tofit = mock.MagicMock()

def make_cand():
    c = MMC.Candidate.__new__(MMC.Candidate)
    c.muons = mus[:2]
    c.vtx = Vtx(0.030, 0.050, 0.4, 1.1); c.vtx.chi2, c.vtx.prob = 1.1, 0.3
    c.pv_bs = Vtx(0.0, 0.0, 0.39, 0); c.bs = Vtx(0.001, 0.0, 0.39, 0)
    c.p4 = lambda: P4()
    c.px, c.py, c.pz = (lambda: 1.0), (lambda: 2.0), (lambda: 0.5)
    c.compute_displacement(); c.compute_flight(c.vtx)
    return c

MMC.Candidate.set_cov_correctors({})                 # covflow off
c = make_cand(); c.compute_raw_vertex()
assert FitCounter.n == 0 and c.vtx_raw is c.vtx and c.cos2d_raw == c.cos2d and c.mcorr_raw == c.mcorr
MMC.Candidate.set_cov_correctors({'mu1': os.path.join(tmp, '2024', 'mu1', 'task_0'),
                                  'mu2': os.path.join(tmp, '2024', 'mu2', 'task_0')})
c = make_cand(); c.compute_raw_vertex()
assert FitCounter.n == 1 and c.vtx_raw.position().x() == 0.031
for k in ('cos2d', 'cos3d', 'cos2d_bs', 'mcorr', 'Bdir_phi'):
    assert getattr(c, k + '_raw') != getattr(c, k), k
assert c.lxy_raw.value() != c.lxy.value() and c.lxyz_raw.value() != c.lxyz.value()
print('raw refit             ok: off -> *_raw = nominal, no 2nd fit; on -> lxy %.4f vs %.4f, '
      'cos2d %.5f vs %.5f' % (c.lxy.value(), c.lxy_raw.value(), c.cos2d, c.cos2d_raw))
print('ALL OK')
