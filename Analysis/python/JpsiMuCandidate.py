from Bmmm.Analysis.utils import masses
from Bmmm.Analysis.JpsiChargedCandidate import JpsiChargedCandidate, kinfit


class JpsiMuCandidate(JpsiChargedCandidate):
    '''
    Bc -> J/psi(-> mu mu) mu  candidate  (the RJpsi golden path).

    A JpsiChargedCandidate whose bachelor is the third muon. Nothing is
    overridden: the bachelor mass hypothesis is the muon mass (the base default,
    _bachelor_mass = masses['mu'], so the full 2mu+mu vertex assigns the muon
    mass to every track, exactly as the original RJpsiCandidate.fit_vertex did),
    and the composite mother pdgId is the Bc (541, the base default). The muon-
    side ntuple is therefore numerically identical to the original RJpsi ntuple.

    self.mu is the bachelor muon; self.jpsi_muons the two J/psi muons; self.muons
    the pt-sorted three. compute_vtx_quantities / compute_helicity_angles and the
    whole machinery are inherited unchanged.
    '''

    _bachelor_mass = masses['mu']

    def __init__(self, jpsi_muons, mu3):
        # mother_pdgid defaults to 541 (Bc), matching the original RJpsiCandidate
        super().__init__(jpsi_muons, mu3, mother_pdgid=541)
