'''
GEN-level tau -> 3mu candidate for the displaced-scalar model:

    D_s -> tau nu ; tau -> mu + a ; a (-> mu mu) long-lived

The candidate is built from the long-lived scalar `a` (pdgId 9900015).
Its two muon daughters form the displaced opposite-sign pair; the tau
ancestor provides the prompt ("other") muon.

All GEN vertices are in cm, hence all decay lengths below are in cm.
'''

from PhysicsTools.HeppyCore.utils.deltar import deltaR


##########################################################################################
#####      GEN NAVIGATION HELPERS
##########################################################################################
def get_daughters(p):
    return [p.daughter(i) for i in range(p.numberOfDaughters())]


def get_mothers(p):
    return [p.mother(i) for i in range(p.numberOfMothers())]


def is_last_copy(p):
    '''
    True if no daughter carries the same pdgId (i.e. p is the last copy in
    its FSR / Pythia copy chain). Avoids relying on the statusFlags being
    correctly propagated to the GEN-SIM reco::GenParticle collection.
    '''
    return not any(d.pdgId() == p.pdgId() for d in get_daughters(p))


def last_copy(p):
    '''
    Descend through same-pdgId daughters, returning the last copy.
    '''
    pdg = p.pdgId()
    while True:
        same = [d for d in get_daughters(p) if d.pdgId() == pdg]
        if not same:
            return p
        p = same[0]


def ancestor_with_pdgid(p, pdgid):
    '''
    Walk up the mother chain and return the first ancestor with |pdgId|==pdgid,
    or None if there is none.
    '''
    for mom in get_mothers(p):
        if abs(mom.pdgId()) == pdgid:
            return mom
        res = ancestor_with_pdgid(mom, pdgid)
        if res is not None:
            return res
    return None


##########################################################################################
#####      CANDIDATE
##########################################################################################
class Tau3MuGenCandidate():
    '''
    GEN-level tau -> 3mu candidate built from the displaced scalar `a`.
    '''
    SCALAR_PDGID = 9900015
    TAU_PDGID    = 15

    def __init__(self, scalar):
        # last copy of the scalar, so that its daughters are the real decay products
        self.scalar = last_copy(scalar)

        # the two muons from the displaced scalar decay, pt-sorted
        disp_muons = sorted([d for d in get_daughters(self.scalar) if abs(d.pdgId()) == 13],
                            key=lambda x: x.pt(), reverse=True)
        if len(disp_muons) != 2:
            raise ValueError('scalar does not decay to exactly two muons')

        # keep the first-copy daughters for the decay vertex (FSR-safe),
        # and the last copies for the momenta (post-radiation, status-1-like)
        self._disp_dau = disp_muons[0]
        self.mu_disp1  = last_copy(disp_muons[0])
        self.mu_disp2  = last_copy(disp_muons[1])

        # the tau ancestor and the prompt muon coming straight from the tau
        self.tau = ancestor_with_pdgid(self.scalar, self.TAU_PDGID)
        if self.tau is None:
            raise ValueError('no tau ancestor found')

        tau_muons = [d for d in get_daughters(last_copy(self.tau)) if abs(d.pdgId()) == 13]
        if len(tau_muons) < 1:
            raise ValueError('tau has no prompt muon daughter')
        self.mu_tau = last_copy(tau_muons[0])

        # convenience pt-sorted list of the three muons
        self.muons = sorted([self.mu_tau, self.mu_disp1, self.mu_disp2],
                           key=lambda x: x.pt(), reverse=True)

    ##########################################################################
    #####      MOMENTA
    ##########################################################################
    def p4(self):
        '''Four-momentum of the whole three-muon system.'''
        return self.mu_tau.p4() + self.mu_disp1.p4() + self.mu_disp2.p4()

    def pair_p4(self):
        '''Four-momentum of the displaced opposite-sign muon pair.'''
        return self.mu_disp1.p4() + self.mu_disp2.p4()

    def pt(self):     return self.p4().pt()
    def eta(self):    return self.p4().eta()
    def phi(self):    return self.p4().phi()
    def mass(self):   return self.p4().mass()
    def energy(self): return self.p4().energy()

    def charge(self):
        return self.mu_tau.charge() + self.mu_disp1.charge() + self.mu_disp2.charge()

    def pair_charge(self):
        return self.mu_disp1.charge() + self.mu_disp2.charge()

    ##########################################################################
    #####      DISPLACEMENT  (everything in cm)
    ##########################################################################
    def prod_vertex(self):
        '''Production vertex of the scalar = decay vertex of the tau.'''
        return self.scalar.vertex()

    def decay_vertex(self):
        '''Decay vertex of the scalar = production vertex of its muon daughters.'''
        return self._disp_dau.vertex()

    def _disp(self):
        pv = self.prod_vertex()
        sv = self.decay_vertex()
        return (sv.x() - pv.x(), sv.y() - pv.y(), sv.z() - pv.z())

    def decay_length(self):
        '''3D decay length of the displaced pair (cm).'''
        dx, dy, dz = self._disp()
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def lxy(self):
        '''Transverse decay length of the displaced pair (cm).'''
        dx, dy, _ = self._disp()
        return (dx * dx + dy * dy) ** 0.5

    def lz(self):
        return self._disp()[2]

    def ctau(self):
        '''
        Proper decay length L * m / |p| (cm). Useful cross-check: should
        peak at the generated ctau (0.1 cm for ctau = 1 mm).
        '''
        return self.decay_length() * self.scalar.mass() / self.scalar.p()

    ##########################################################################
    #####      ANGLES
    ##########################################################################
    def dr_scalar_mu(self):
        '''Angular separation between the scalar and the prompt (tau) muon.'''
        return deltaR(self.scalar, self.mu_tau)

    def __str__(self):
        return '\n'.join([
            'gen tau3mu cand  mass %.3f pt %.2f eta %.2f phi %.2f' % (
                self.mass(), self.pt(), self.eta(), self.phi()),
            '  pair mass %.3f  L3D %.4f cm  Lxy %.4f cm  ctau %.4f cm  dR(a,mu) %.3f' % (
                self.pair_p4().mass(), self.decay_length(), self.lxy(),
                self.ctau(), self.dr_scalar_mu()),
            '  mu_tau   pt %.2f eta %.2f phi %.2f q %+d' % (
                self.mu_tau.pt(), self.mu_tau.eta(), self.mu_tau.phi(), self.mu_tau.charge()),
            '  mu_disp1 pt %.2f eta %.2f phi %.2f q %+d' % (
                self.mu_disp1.pt(), self.mu_disp1.eta(), self.mu_disp1.phi(), self.mu_disp1.charge()),
            '  mu_disp2 pt %.2f eta %.2f phi %.2f q %+d' % (
                self.mu_disp2.pt(), self.mu_disp2.eta(), self.mu_disp2.phi(), self.mu_disp2.charge()),
        ])
