'''Fake ROOT / FWLite / heppy so the R(Ds) python modules can be imported and
exercised outside CMSSW. Values from the fake are meaningless; the point is to
run the code paths and check the branch names.'''
import sys
import types
import math
import numpy as np


class Any(object):
    def __init__(self, *a, **k): pass
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return Any()
    def __call__(self, *a, **k): return Any()
    def __mro_entries__(self, bases): return (object,)
    def __iter__(self): return iter([])
    def __bool__(self): return True
    def __float__(self): return 0.
    def __int__(self): return 0
    def __getitem__(self, i): return Any()


class LV(object):
    '''PxPyPzE4D LorentzVector stand-in.'''
    def __init__(self, px=0., py=0., pz=0., e=0.):
        self.v = np.array([e, px, py, pz], float)
    def px(self): return self.v[1]
    def py(self): return self.v[2]
    def pz(self): return self.v[3]
    def energy(self): return self.v[0]
    def E(self): return self.v[0]
    def mass(self): return math.sqrt(max(self.v[0]**2 - self.v[1]**2 - self.v[2]**2 - self.v[3]**2, 0))
    def pt(self): return math.hypot(self.v[1], self.v[2])
    def eta(self): return math.asinh(self.v[3] / self.pt())
    def phi(self): return math.atan2(self.v[2], self.v[1])
    def __add__(self, o):
        r = LV(); r.v = self.v + o.v; return r


class Meas(object):
    def __init__(self, v=0.1, e=0.01): self.v, self.e = v, e
    def value(self): return self.v
    def error(self): return self.e
    def significance(self): return self.v / self.e


class Math(Any):
    def LorentzVector(self, *a):
        return LV
    def XYZVector(self, x, y, z):
        return np.array([x, y, z])


root = types.ModuleType('ROOT')
root.gSystem = Any()
root.Math = Math()
root.reco = Any()
root.std = Any()
root.TLorentzVector = Any
for name in ('KVFitter', 'RDsKinVtxFitter', 'RJpsiKinVtxFitter', 'B4MuKinVtxFitter'):
    setattr(root, name, Any)
root.VertexDistanceXY = lambda: types.SimpleNamespace(distance=lambda a, b: Meas())
root.VertexDistance3D = lambda: types.SimpleNamespace(distance=lambda a, b: Meas())
sys.modules['ROOT'] = root

fw = types.ModuleType('DataFormats.FWLite')
fw.Events = Any
fw.Handle = Any
sys.modules['DataFormats'] = types.ModuleType('DataFormats')
sys.modules['DataFormats.FWLite'] = fw


def deltaR(*args):
    if len(args) == 2:
        a, b = args
        e1, p1, e2, p2 = a.eta(), a.phi(), b.eta(), b.phi()
    else:
        e1, p1, e2, p2 = args
    dp = (p1 - p2 + math.pi) % (2 * math.pi) - math.pi
    return math.sqrt((e1 - e2) ** 2 + dp ** 2)


def bestMatch(obj, coll):
    best, d2 = None, 1e9
    for c in coll:
        d = deltaR(obj, c) ** 2
        if d < d2:
            best, d2 = c, d
    return best, d2


dr = types.ModuleType('PhysicsTools.HeppyCore.utils.deltar')
dr.deltaR, dr.bestMatch, dr.deltaPhi = deltaR, bestMatch, lambda a, b: (a - b + math.pi) % (2 * math.pi) - math.pi
for m in ('PhysicsTools', 'PhysicsTools.HeppyCore', 'PhysicsTools.HeppyCore.utils'):
    sys.modules[m] = types.ModuleType(m)
sys.modules['PhysicsTools.HeppyCore.utils.deltar'] = dr

pkg = types.ModuleType('Bmmm'); pkg.__path__ = []
ana = types.ModuleType('Bmmm.Analysis'); import os
ana.__path__ = [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'python')]
sys.modules['Bmmm'] = pkg
sys.modules['Bmmm.Analysis'] = ana
