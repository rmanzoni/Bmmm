'''
Generator-level vertices of the dimuon ntuple (MC only).

  gen PV       genParticles:xyz0, the position of the first HepMC vertex after
               the vertex smearing: the true primary interaction point. It is
               what NanoAOD stores as GenVtx_x/y/z, and MiniAOD keeps it
               ('keep *_genParticles_xyz0_*'). Event level.

  gen J/psi    the J/psi (pdgId 443) the two reco muons come from, found by
               walking up the mothers of each muon's gen match through the
               copies of the same particle (PHOTOS radiation, shower
               bookkeeping). Its DECAY vertex is the production vertex of its
               daughter muons, i.e. the true dimuon vertex. (The J/psi's own
               vertex() is its production point; for a B -> J/psi X decay that
               is the B decay vertex, the same point to within the J/psi
               lifetime, c*tau ~ 2e-12 cm.)

The J/psi is taken from mu1 if mu1 comes from one, else from mu2.
gen_jpsi_nmu says how many of the two reco muons are its daughters: 2 is the
clean case, 1 means the other reco muon is matched elsewhere (or not at all),
0 means no J/psi was found (branches NaN).

Pure python on top of the reco::GenParticle interface (pdgId, vx/vy/vz,
numberOfMothers, mother, p4 accessors), so it can be tested with stand-ins.
'''

import numpy as np

JPSI_PDGID = 443
MUON_PDGID = 13

NAN_GEN = dict(jpsi_nmu=0, jpsi_vx=np.nan, jpsi_vy=np.nan, jpsi_vz=np.nan,
               jpsi_pt=np.nan, jpsi_eta=np.nan, jpsi_phi=np.nan, jpsi_mass=np.nan,
               jpsi_mom_pdgid=np.nan, lxy=np.nan, lxyz=np.nan)


def _same(a, b):
    '''Two pointers to the same gen particle: FWLite hands out a new python
    proxy per call, so compare the content, not the proxies.'''
    return (a.pdgId() == b.pdgId() and a.px() == b.px() and a.py() == b.py()
            and a.pz() == b.pz() and a.vx() == b.vx() and a.vy() == b.vy()
            and a.vz() == b.vz())


def _first_copy(p):
    '''Walk up through mothers with the same pdgId; returns the first copy.'''
    while p.numberOfMothers() > 0 and p.mother(0).pdgId() == p.pdgId():
        p = p.mother(0)
    return p


def jpsi_of(gen_match):
    '''(J/psi first copy, muon first copy) the gen match descends from, or
    (None, None) if the match is not a muon or not from a J/psi.'''
    if gen_match is None or abs(gen_match.pdgId()) != MUON_PDGID:
        return None, None
    mu = _first_copy(gen_match)
    if mu.numberOfMothers() == 0:
        return None, None
    mom = mu.mother(0)
    if mom.pdgId() != JPSI_PDGID:
        return None, None
    return _first_copy(mom), mu


def gen_info(mu1, mu2, gen_pv):
    '''Per-candidate gen bookkeeping. mu1 / mu2 may carry gen_match;
    gen_pv is (x, y, z) or None.'''
    out = dict(NAN_GEN)
    m1, m2 = getattr(mu1, 'gen_match', None), getattr(mu2, 'gen_match', None)
    j1, d1 = jpsi_of(m1)
    j2, d2 = jpsi_of(m2)
    jpsi, dau = (j1, d1) if j1 is not None else (j2, d2)
    if jpsi is None:
        return out
    out['jpsi_nmu'] = int(j1 is not None and _same(j1, jpsi)) + int(j2 is not None and _same(j2, jpsi))
    out['jpsi_vx'], out['jpsi_vy'], out['jpsi_vz'] = dau.vx(), dau.vy(), dau.vz()
    out['jpsi_pt'], out['jpsi_eta'], out['jpsi_phi'] = jpsi.pt(), jpsi.eta(), jpsi.phi()
    out['jpsi_mass'] = jpsi.mass()
    out['jpsi_mom_pdgid'] = jpsi.mother(0).pdgId() if jpsi.numberOfMothers() > 0 else 0
    if gen_pv is not None:
        dx, dy, dz = (out['jpsi_vx'] - gen_pv[0], out['jpsi_vy'] - gen_pv[1],
                      out['jpsi_vz'] - gen_pv[2])
        out['lxy'] = float(np.hypot(dx, dy))
        out['lxyz'] = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    return out


class GenPVReader(object):
    '''Reads genParticles:xyz0 per event, fail-soft: a sample without it gets
    NaN gen-PV branches and one loud warning, not a dead job.'''

    LABEL = ('genParticles', 'xyz0')
    TYPES = ('math::XYZPointF',
             'ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<float>,'
             'ROOT::Math::DefaultCoordinateSystemTag>')

    def __init__(self, Handle):
        self.handle = None
        for t in self.TYPES:
            try:
                self.handle = Handle(t)
                break
            except Exception:
                continue
        self.n_ok = 0
        self.n_missing = 0
        self.warned = False

    def __call__(self, event):
        p = None
        if self.handle is not None:
            try:
                event.getByLabel(self.LABEL, self.handle)
                if self.handle.isValid():
                    q = self.handle.product()
                    p = (float(q.x()), float(q.y()), float(q.z()))
            except Exception:
                p = None
        if p is None:
            self.n_missing += 1
            if not self.warned:
                print('WARNING: genParticles:xyz0 not readable in this sample: gen_pv_* '
                      'and gen_lxy / gen_lxyz will be NaN')
                self.warned = True
        else:
            self.n_ok += 1
        return p

    def summary(self):
        return ('#### gen PV (genParticles:xyz0): read in %d event(s), missing in %d'
                % (self.n_ok, self.n_missing))
