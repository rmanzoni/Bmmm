'''
Standalone check of Bmmm/Analysis/python/GenVertices.py with stand-in gen
particles (no CMSSW needed):

    python3 test_gen_vertices.py

Cases: B -> J/psi (-> mu mu) with PHOTOS copies of a muon and a J/psi copy,
prompt J/psi, one reco muon matched elsewhere, a muon matched to a pion,
nothing matched, psi(2S) -> J/psi, no gen PV, and the fail-soft gen-PV reader.
'''
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'python'))
import GenVertices as G     # noqa: E402

FAILED = []


def check(name, ok, detail=''):
    print('%-58s %s %s' % (name, 'ok  ' if ok else 'FAIL', detail))
    if not ok:
        FAILED.append(name)


class P(object):
    '''reco::GenParticle stand-in. A copy shares pdgId and vertex with its
    parent but is a different object.'''
    def __init__(self, pdg, p3, v, mother=None, m=0.):
        self._pdg, self._p3, self._v, self._mom, self._m = pdg, p3, v, mother, m
    def pdgId(self): return self._pdg
    def px(self): return self._p3[0]
    def py(self): return self._p3[1]
    def pz(self): return self._p3[2]
    def vx(self): return self._v[0]
    def vy(self): return self._v[1]
    def vz(self): return self._v[2]
    def pt(self): return math.hypot(self._p3[0], self._p3[1])
    def eta(self): return math.asinh(self._p3[2] / self.pt())
    def phi(self): return math.atan2(self._p3[1], self._p3[0])
    def mass(self): return self._m
    def numberOfMothers(self): return 0 if self._mom is None else 1
    def mother(self, i):
        # FWLite returns a new proxy every time: emulate it with a shallow copy
        m = self._mom
        return P(m._pdg, m._p3, m._v, m._mom, m._m)


class Mu(object):
    def __init__(self, g=None):
        if g is not None:
            self.gen_match = g


PV = (0.01, -0.02, 1.5)
SV = (0.11, 0.08, 1.9)

# B+ -> J/psi K+, J/psi (with a shower copy) -> mu+ mu-, mu- radiates twice
B = P(521, (5., 3., 2.), PV, m=5.279)
J0 = P(443, (4., 2., 1.5), SV, mother=B, m=3.0969)
J = P(443, (4., 2., 1.5), SV, mother=J0, m=3.0969)        # copy
mup = P(-13, (2., 1., 0.8), SV, mother=J)
mum0 = P(13, (2., 1., 0.7), SV, mother=J)
mum1 = P(13, (1.9, 1., 0.7), SV, mother=mum0)                # PHOTOS copy
mum2 = P(13, (1.8, 0.9, 0.7), SV, mother=mum1)               # PHOTOS copy

o = G.gen_info(Mu(mup), Mu(mum2), PV)
check('B -> J/psi: both muons from the J/psi', o['jpsi_nmu'] == 2, o['jpsi_nmu'])
check('B -> J/psi: J/psi vertex = muon vertex', (o['jpsi_vx'], o['jpsi_vy'], o['jpsi_vz']) == SV)
check('B -> J/psi: mother = B+ (through the J/psi copy)', o['jpsi_mom_pdgid'] == 521, o['jpsi_mom_pdgid'])
lxy = math.hypot(SV[0] - PV[0], SV[1] - PV[1])
lxyz = math.sqrt(lxy ** 2 + (SV[2] - PV[2]) ** 2)
check('B -> J/psi: lxy, lxyz', abs(o['lxy'] - lxy) < 1e-12 and abs(o['lxyz'] - lxyz) < 1e-12,
      '%.4f %.4f' % (o['lxy'], o['lxyz']))
check('B -> J/psi: J/psi kinematics', abs(o['jpsi_pt'] - math.hypot(4, 2)) < 1e-12
      and o['jpsi_mass'] == 3.0969)

o = G.gen_info(Mu(mum2), Mu(mup), PV)
check('roles swapped: same answer', o['jpsi_nmu'] == 2 and o['jpsi_mom_pdgid'] == 521)

# prompt J/psi, no mother in the pruned record
Jp = P(443, (3., 0., 0.), PV, m=3.0969)
a, b = P(13, (1.5, 0.1, 0.), PV, mother=Jp), P(-13, (1.5, -0.1, 0.), PV, mother=Jp)
o = G.gen_info(Mu(a), Mu(b), PV)
check('prompt J/psi: nmu 2, mother 0, lxy 0', o['jpsi_nmu'] == 2 and o['jpsi_mom_pdgid'] == 0
      and o['lxy'] == 0.)

# psi(2S) -> J/psi pi pi in a B decay: the mother is the psi(2S)
Bs = P(531, (5., 0., 0.), PV)
psi2 = P(100443, (4., 0., 0.), SV, mother=Bs)
J2 = P(443, (3.5, 0., 0.), SV, mother=psi2)
o = G.gen_info(Mu(P(13, (2., 0., 0.), SV, mother=J2)), Mu(P(-13, (1.5, 0., 0.), SV, mother=J2)), PV)
check('psi(2S) -> J/psi: mother 100443', o['jpsi_mom_pdgid'] == 100443)

# one muon from the J/psi, the other from another J/psi of the same event
Jx = P(443, (1., 1., 1.), (0.5, 0.5, 2.), mother=B)
mux = P(13, (0.5, 0.5, 0.5), (0.5, 0.5, 2.), mother=Jx)
o = G.gen_info(Mu(mup), Mu(mux), PV)
check('two different J/psi: nmu 1, the J/psi of mu1', o['jpsi_nmu'] == 1
      and (o['jpsi_vx'], o['jpsi_vy'], o['jpsi_vz']) == SV)

# mu1 matched to a pion, mu2 from the J/psi: the J/psi of mu2
pi = P(211, (1., 0., 0.), SV, mother=B)
o = G.gen_info(Mu(pi), Mu(mum2), PV)
check('mu1 matched to a pion: J/psi from mu2, nmu 1', o['jpsi_nmu'] == 1
      and (o['jpsi_vx'], o['jpsi_vy'], o['jpsi_vz']) == SV)

# a muon from a B semileptonic decay (no J/psi), the other unmatched
o = G.gen_info(Mu(P(13, (1., 0., 0.), SV, mother=B)), Mu(), PV)
check('no J/psi: nmu 0, everything NaN', o['jpsi_nmu'] == 0
      and all(math.isnan(v) for k, v in o.items() if k != 'jpsi_nmu'))

# no gen PV: the J/psi is there, the distances are not
o = G.gen_info(Mu(mup), Mu(mum2), None)
check('no gen PV: J/psi filled, lxy / lxyz NaN', o['jpsi_nmu'] == 2 and math.isnan(o['lxy'])
      and math.isnan(o['lxyz']))


# --- the reader, with stand-in FWLite pieces ---------------------------------
class XYZ(object):
    def __init__(self, v): self.v = v
    def x(self): return self.v[0]
    def y(self): return self.v[1]
    def z(self): return self.v[2]


class Handle(object):
    def __init__(self, t):
        if t == 'math::XYZPointF':
            raise RuntimeError('typedef unknown here')     # exercises the fallback
        self.t, self._p = t, None
    def isValid(self): return self._p is not None
    def product(self):
        if self._p is None:
            raise RuntimeError('no product')
        return self._p


class Event(object):
    def __init__(self, v): self.v = v
    def getByLabel(self, label, h):
        assert label == ('genParticles', 'xyz0'), label
        h._p = XYZ(self.v) if self.v is not None else None


r = G.GenPVReader(Handle)
check('reader: falls back to the full type name', r.handle is not None and 'PositionVector3D' in r.handle.t)
check('reader: reads xyz0', r(Event(PV)) == PV)
check('reader: missing product -> None, no exception', r(Event(None)) is None)
check('reader: counts', (r.n_ok, r.n_missing) == (1, 1), r.summary())

print('\nALL OK' if not FAILED else '\nFAILED: %s' % ', '.join(FAILED))
sys.exit(1 if FAILED else 0)
