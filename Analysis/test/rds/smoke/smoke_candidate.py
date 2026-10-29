'''Run outside CMSSW (fake ROOT, see fakeroot.py): python3 smoke_candidate.py

Run RDsCandidate on a generated signal decay with fake fit trees and check
that every quantity it computes has a branch, and vice versa.'''
import sys, types, math
import numpy as np
import fakeroot
from fakeroot import Meas, LV

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import test_rds_kinematics as T
from Bmmm.Analysis import RDsCandidate as C
from Bmmm.Analysis import RDsKinematics as K
from Bmmm.Analysis import RDsBranches as B
from Bmmm.Analysis import RDsGenHistory as G
from Bmmm.Analysis.RDsCuts import cuts as _c
cuts = _c['rds']

rng = np.random.default_rng(1)
bs_lab = K.p4_from_p3(25., 8., 5., K.M_BS)
rest, lab = T.make_decay(4.0, 0.3, -0.2, 1.0, 0.5, bs_lab)
sv = bs_lab[1:] / K.p3mag(bs_lab) * 0.3
dsv = sv + 0.02 * lab['ds'][1:] / K.p3mag(lab['ds'])


class Cov(object):
    kRows = kCols = 5
    def __init__(self): self.m = np.diag([1e-4, 1e-6, 1e-6, 1e-5, 1e-5])
    def __call__(self, i, j): return self.m[i][j]


class Trk(object):
    def __init__(self, p4): self.p4_ = p4
    def covariance(self): return Cov()
    def dz(self, *a): return 0.01
    def dxy(self, *a): return 0.01
    def pt(self): return K.pt(self.p4_)
    def eta(self): return K.eta(self.p4_)


class Pat(object):
    def __init__(self, p4, q, pdg):
        self.p4_, self.q, self.pdg = p4, q, pdg
    def px(self): return self.p4_[1]
    def py(self): return self.p4_[2]
    def pz(self): return self.p4_[3]
    def pt(self): return K.pt(self.p4_)
    def eta(self): return K.eta(self.p4_)
    def phi(self): return K.phi(self.p4_)
    def energy(self): return self.p4_[0]
    def charge(self): return self.q
    def pdgId(self): return self.pdg
    def vz(self): return 0.
    def bestTrack(self): return Trk(self.p4_)
    def pfIsolationR03(self): return types.SimpleNamespace(sumChargedHadronPt=1., sumNeutralHadronEt=0.5, sumPhotonEt=0.2, sumPUPt=0.1, sumChargedParticlePt=1.1)
    pfIsolationR04 = pfIsolationR03
    def fromPV(self, *a): return 3
    def pvAssociationQuality(self): return 7
    def trackHighPurity(self): return True
    def lostInnerHits(self): return 0


class Vtx(object):
    def __init__(self, pos): self.pos = np.asarray(pos, float)
    def position(self): return types.SimpleNamespace(x=lambda: self.pos[0], y=lambda: self.pos[1], z=lambda: self.pos[2])
    def chiSquared(self): return 1.2
    def degreesOfFreedom(self): return 1.
    def error(self): return types.SimpleNamespace(matrix=lambda: (lambda i, j: 1e-6 * (i == j)))
    def vertexState(self): return self
    def tracksSize(self): return 20
    def chi2(self): return 10.
    def ndof(self): return 15.
    def z(self): return self.pos[2]


class RecoVtx(Vtx):
    def error(self): return (lambda i, j: 1e-6 * (i == j))


class KinPart(object):
    def __init__(self, p4): self.p4_ = p4
    def currentState(self):
        p = self.p4_
        m = math.sqrt(max(K.mass2(p), 0))
        vec = types.SimpleNamespace(At=lambda i: [0, 0, 0, p[1], p[2], p[3], m][i])
        return types.SimpleNamespace(kinematicParameters=lambda: types.SimpleNamespace(vector=lambda: vec))


class Tree(object):
    def __init__(self, top, children, pos):
        self.top, self.ch, self.v, self.i = top, children, Vtx(pos), -1
    def get(self): return True
    def isEmpty(self): return False
    def isValid(self): return True
    def movePointerToTheTop(self): self.i = -1
    def currentParticle(self): return KinPart(self.top if self.i < 0 else self.ch[self.i])
    def currentDecayVertex(self): return types.SimpleNamespace(get=lambda: self.v)
    def movePointerToTheFirstChild(self):
        self.i = 0; return len(self.ch) > 0
    def movePointerToTheNextChild(self):
        self.i += 1
        if self.i >= len(self.ch):
            self.i = len(self.ch) - 1; return False
        return True


class FakeFitter(object):
    def FitVertexCut(self, *a):
        phi = lab['k1'] + lab['k2']; dsphi = phi + lab['pi']
        return [Tree(phi, [lab['k1'], lab['k2']], dsv), Tree(dsphi, [lab['pi'], phi], dsv),
                Tree(phi, [lab['k1'], lab['k2']], dsv), Tree(dsphi, [lab['pi'], phi], dsv)]
    def DsMomentaBeforeBs(self):
        phi = lab['k1'] + lab['k2']; dsphi = phi + lab['pi']
        return [*dsphi[1:], *lab['ds'][1:], *dsphi[1:]]
    def FitRest(self, *a):
        phi = lab['k1'] + lab['k2']; ds = lab['ds']; dsphi = phi + lab['pi']
        return [Tree(dsphi + lab['mu'], [lab['mu'], dsphi], sv),
                Tree(ds, [lab['pi'], phi], dsv), Tree(ds + lab['mu'], [lab['mu'], ds], sv),
                Tree(dsphi + lab['mu'], [lab['mu'], dsphi], sv)]


class FakeIP(object):
    def signedIP3D(self, *a): return Meas(0.01, 0.002)
    def signedDecayLength3D(self, *a): return Meas(0.02, 0.003)
    def jetTrackDistance(self, *a): return types.SimpleNamespace(first=0.1, second=Meas(-0.003))


C.kinfit, C.iptools = FakeFitter(), FakeIP()

mu = C.RDsTrack(C.TrackRecord(Pat(lab['mu'], -1, 13), 'muon', 0), K.M_MU)
k1 = C.RDsTrack(C.TrackRecord(Pat(lab['k1'], +1, 211), 'pf', 1), K.M_K)
k2 = C.RDsTrack(C.TrackRecord(Pat(lab['k2'], -1, 211), 'pf', 2), K.M_K)
pi = C.RDsTrack(C.TrackRecord(Pat(lab['pi'], +1, 211), 'lost', 3), K.M_PI)
if k1.pt() < k2.pt():
    k1, k2 = k2, k1

cand = C.RDsCandidate(mu, k1, k2, pi)
cand.fit()
assert cand.passes_vertex_cuts(cuts)


def fake_choose_pv(self, vertices, beamspot, pf, lost):
    self.pv_idx, self.pv_from_scheme = 0, 0
    self.pv = self.pv_bs = self.bs = RecoVtx([0., 0., 0.])
    self.pv_refit_valid, self.pv_refit_ntrk = True, 19
    self.pv_pos = np.zeros(3)


C.RDsCandidate.choose_pv = fake_choose_pv
photons = [Pat(np.array([0.2, 0.05, 0.02, 0.19]) + 0 * lab['ds'], 0, 22)]
# a photon pointing along the Ds with the Ds*-Ds energy
# a photon collinear with the Ds, with the energy that gives m(Ds gamma) = m(Ds*)
e_g = (K.M_DSST ** 2 - K.M_DS ** 2) / (2 * (lab['ds'][0] - K.p3mag(lab['ds'])))
pg = lab['ds'][1:] / K.p3mag(lab['ds']) * e_g
photons.append(Pat(np.array([e_g, *pg]), 0, 22))
pf = [Pat(lab['k1'] * 0.3 + np.array([1, .1, 0, 0]), 1, 211), Pat(np.array([1., 0.2, 0.3, 0.9]), 0, 130)]
cand.compute([Vtx([0, 0, 0])], None, pf, [], photons, cuts)
cand.out.update(G.classify_candidate(None, None, None, None))
cand.out['cand_idx'] = 0

produced = set(cand.out)
schema = set(B.cand_names)
print('produced %d, schema %d' % (len(produced), len(schema)))
print('PRODUCED BUT NOT IN SCHEMA (silently dropped):', sorted(produced - schema))
print('IN SCHEMA BUT NOT PRODUCED here:', sorted(schema - produced))

# physics sanity on the cons scheme with the true momenta
o = cand.out
print('cons_coll_q2 %.3f  cons_xyz_bs_q2 %.3f  true q2 4.0' % (o['cons_coll_q2'], o['cons_xyz_bs_q2']))
print('math1/2 q2 %.3f %.3f real %s' % (o['cons_math1_bs_q2'], o['cons_math2_bs_q2'], o['cons_math_real_bs']))
print('cos_theta_phi_ds math %.3f %.3f (true 0.3)' % (o['cons_math1_bs_cos_theta_phi_ds'], o['cons_math2_bs_cos_theta_phi_ds']))
print('kkpi_mass %.4f kk_mass %.4f gamma n %d dm %.4f' % (o['kkpi_mass'], o['kk_mass'], o['cons_gamma_n'], o.get('cons_gamma_dm', float('nan'))))
print('charge class', o['charge_class'], 'mupi', o['mupi_charge'])

# ---------------------------------------------------------------------------
# inspector hooks with a fake event
# ---------------------------------------------------------------------------
from Bmmm.Analysis import RDsInspector as I
from Bmmm.Analysis.RDsBranches import branches, gen_tree_branches, EVENT_GEN_KEYS, EVENT_EXTRA_KEYS
import fakeroot as F

parts, pp = T.build_signal(1)


class Names(object):
    names = ['HLT_Mu9_IP6_part0_v3', 'HLT_Mu7_IP4_part0_v2', 'HLT_Other_v1']
    def triggerNames(self): return self.names
    def triggerIndex(self, n): return self.names.index(n)


class Ev(object):
    def eventAuxiliary(self):
        return types.SimpleNamespace(run=lambda: 320500, luminosityBlock=lambda: 12, event=lambda: 987654321)
    def object(self):
        return types.SimpleNamespace(triggerNames=lambda r: Names())


ev = Ev()
ev.trg_res = types.SimpleNamespace(size=lambda: 3, accept=lambda i: i == 0)
ev.genpr = parts
ev.genpk = []
ev.jets = []
ev.vtx = [RecoVtx([0, 0, 0])]
ev.bs = None
ev.pf = pf
ev.ltrk = []
ev.glb_alg = None
ev.l1min = types.SimpleNamespace(getPrescaleForIndex=lambda i: 1)
ev.l1max = types.SimpleNamespace(getPrescaleForIndex=lambda i: 3)
ev.pu_at_bx0 = types.SimpleNamespace(getPU_NumInteractions=lambda: 30, getTrueNumInteractions=lambda: 31.5)

insp = I.RDsInspector()
opts = types.SimpleNamespace(mc=True, verbose=True, inputFiles='/store/mc/x.root', trk_calib='', no_l1=True)
insp.setup(opts)
insp.GEN_PV = types.SimpleNamespace(__call__=None)
insp.GEN_PV = lambda e: (0., 0., 0.)
insp.extra_rows = {'tree_gen': []}
insp.HAMMER = None
insp.process_event_gen(ev, opts)
row = insp.extra_rows['tree_gen'][0]
print('gen row: missing', sorted(set(gen_tree_branches) - set(row)), 'extra', sorted(set(row) - set(gen_tree_branches)))
print('gen_sig', row['gen_sig'], 'gen_q2 %.3f' % row['gen_q2'], 'ham dsg pt', row['gen_ham_dsg_pt'])

event_tofill = {}
insp.fill_event_extra(ev, opts, event_tofill)
gen_state = insp.setup_event_gen(ev, opts, event_tofill)
print('event keys missing', sorted((set(EVENT_EXTRA_KEYS) | set(EVENT_GEN_KEYS)) - set(event_tofill)))
print('l1min/max', event_tofill['HLT_Mu9_IP6_l1min_ps'], event_tofill['HLT_Mu9_IP6_l1max_ps'], event_tofill['HLT_Mu8_IP3_l1min_ps'])

insp._photons = photons
cand2 = C.RDsCandidate(mu, k1, k2, pi)
cand2.fit(); cand2.cand_idx = 0
tofill = insp.fill_candidate(cand2, ev, opts, cuts, gen_state, {})
row = {}
row.update(event_tofill); row.update(tofill)
row.update(dict((p, 1.) for p in insp.PATHS)); row.update(dict((p + '_ps', 1.) for p in insp.PATHS))
missing = [b for b in branches if b not in row]
nan = [b for b in branches if b in row and isinstance(row[b], float) and row[b] != row[b]]
print('row has %d / %d branches; missing %s' % (len(set(branches) & set(row)), len(branches), missing[:10]))
print('NaN branches (%d):' % len(nan))
import collections
groups = collections.Counter(b.split('_')[0] for b in nan)
print(dict(groups))
print([b for b in nan if not b.startswith(('cons_gamma', 'phic_gamma', 'free_gamma', 'hammer'))][:80])
