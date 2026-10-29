'''Run outside CMSSW (fake ROOT, see fakeroot.py): python3 smoke_main.py
Needs numpy, scipy, uproot, particle.

End-to-end run of RDsInspector.main() on fake events (fake FWLite, fake
fits): argument parsing, the base looper with the new hooks, the uproot
output with the gen tree, the cutflow.'''
import sys, types, os
import numpy as np
import fakeroot
from fakeroot import Any
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'smoke_candidate.py')).read().split('# ---------------------------------------------------------------------------\n# inspector hooks')[0])

from Bmmm.Analysis import JpsiChargedInspector as JCI
from Bmmm.Analysis import RDsInspector as I
import uproot


parts, pp = T.build_signal(1)


class Names(object):
    names = ['HLT_Mu9_IP6_part0_v3', 'HLT_Mu7_IP4_part0_v2', 'HLT_Other_v1']
    def triggerNames(self): return self.names
    def triggerIndex(self, n): return self.names.index(n)
    def __len__(self): return len(self.names)


class H(object):
    def __init__(self, typ): self._p = None
    def product(self): return self._p
    def isValid(self): return self._p is not None


class Mu(Pat):
    def isPFMuon(self): return True
    def bestTrack(self):
        t = Trk(self.p4_); t.dxy = lambda *a: 0.01; return t


class TObj(object):
    def __init__(self, p4): self.p4_ = p4
    def pt(self): return K.pt(self.p4_)
    def eta(self): return K.eta(self.p4_)
    def phi(self): return K.phi(self.p4_)
    def unpackNamesAndLabels(self, *a): pass
    def hasFilterLabel(self, l): return l == 'hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered9Q'


class Tk(Pat):
    def hasTrackDetails(self): return True


class FakeEvent(object):
    def __init__(self, n):
        self.n = n
        mu_p4 = lab['mu']
        if K.pt(mu_p4) < 7:
            mu_p4 = mu_p4 * (7.5 / K.pt(mu_p4))   # fake: harder muon to pass pT > 7
            mu_p4[0] = np.sqrt(np.sum(mu_p4[1:] ** 2) + K.M_MU ** 2)
        self.prod = {
            'slimmedMuons': [Mu(mu_p4, -1, 13)],
            'packedPFCandidates': [Tk(lab['k1'], 1, 211), Tk(lab['k2'], -1, 211), Tk(lab['pi'], 1, 211),
                                   Tk(np.array([0.3, 0.1, 0.2, 0.1]), 0, 22)],
            'lostTracks': [],
            'offlineSlimmedPrimaryVertices': [RecoVtx([0, 0, 0])],
            'offlineBeamSpot': types.SimpleNamespace(x0=lambda: 0., y0=lambda: 0., z0=lambda: 0.,
                                                     x0Error=lambda: 1e-3, y0Error=lambda: 1e-3, z0Error=lambda: 1.),
            'slimmedPatTrigger': [TObj(mu_p4)],
            'slimmedJets': [],
            'prunedGenParticles': parts, 'packedGenParticles': [],
            'slimmedAddPileupInfo': [types.SimpleNamespace(getBunchCrossing=lambda: 0, getPU_NumInteractions=lambda: 30,
                                                           getTrueNumInteractions=lambda: 31.2)],
        }
        self.prod[('TriggerResults', '', 'HLT')] = types.SimpleNamespace(size=lambda: 3, accept=lambda i: i == 0)
        self.prod[('patTrigger', '')] = types.SimpleNamespace(getPrescaleForIndex=lambda i: 1)
        self.prod[('patTrigger', 'l1min')] = types.SimpleNamespace(getPrescaleForIndex=lambda i: 1)
        self.prod[('patTrigger', 'l1max')] = types.SimpleNamespace(getPrescaleForIndex=lambda i: 2)

    def getByLabel(self, label, handle):
        handle._p = self.prod.get(label)
    def eventAuxiliary(self):
        return types.SimpleNamespace(run=lambda: 320500, luminosityBlock=lambda: 12, event=lambda: 10**10 + self.n)
    def object(self):
        return types.SimpleNamespace(triggerNames=lambda r: Names())


class FakeEvents(object):
    def __init__(self, files): self.files = files
    def size(self): return 3
    def __iter__(self): return iter([FakeEvent(i) for i in range(3)])


JCI.Events = FakeEvents
C.track_with_cov = lambda trk, cov: trk   # fake tracks cannot be rebuilt
import Bmmm.Analysis.Handles as Hd
for d in (Hd.handles, Hd.handles_mc):
    for k, (lab_, h) in list(d.items()):
        d[k] = (lab_, H(None))
for k, (lab_, h) in list(I.extra_handles.items()):
    I.extra_handles[k] = (lab_, H(None))

import tempfile
out = tempfile.mkdtemp()
sys.argv = ['inspector_rds.py', '--inputFiles=root://x//store/mc/a.root', '--filename=smoke',
            '--destination=' + out, '--mc', '--savenontrig', '--no-l1', '--logger=' + out + '/logger',
            '--cov-scale=dxy=1.1']
I.RDsInspector.choose_pv = fake_choose_pv
I.RDsInspector().main()
f = uproot.open(out + '/smoke.root')
print(f.keys())
t, g = f['tree'], f['tree_gen']
print('tree entries', t.num_entries, 'branches', len(t.keys()), '; tree_gen entries', g.num_entries, 'branches', len(g.keys()))
a = t.arrays(['event', 'mu_pt', 'cons_ds_mass', 'kkpi_mass', 'cons_math1_bs_q2', 'mu_trig_match',
              'mu_HLT_Mu9_IP6_tag', 'HLT_Mu9_IP6', 'charge_class', 'k1_cov_scale_dxy', 'cand_idx', 'ncands'], library='np')
for k, v in a.items():
    print('%-22s %s' % (k, v))
print(open(out + '/logger.txt').read())
