from Bmmm.Analysis.utils import masses, p4_with_mass
from Bmmm.Analysis.JpsiChargedCandidate import (
    JpsiChargedCandidate, BachelorTrack, kinfit,
)


class JpsiTkCandidate(JpsiChargedCandidate):
    '''
    J/psi(-> mu mu) + charged track  candidate, reconstructed under TWO bachelor
    mass hypotheses at once:

        KAON  (primary, unprefixed) :  B+  -> J/psi K+   (mother pdgId 521)
        PION  (alternative, 'pi_')  :  Bc+ -> J/psi pi+  (mother pdgId 541)

    The bachelor track (a pat::PackedCandidate pseudo-track, from
    packedPFCandidates or lostTracks) is wrapped twice with BachelorTrack, once
    per mass hypothesis. self.mu carries the KAON hypothesis, so the base class
    treats the kaon as the primary bachelor and the whole inherited machinery --
    the two vertex fits (mass-constrained J/psi dimuon + full 2mu+K), the PV
    choice, the beamspot-constrained signal-removed PV refit, the displacement /
    pointing quantities, the bachelor impact parameters, the jet-track distances
    and the custom PF isolation -- runs for the KAON hypothesis with UNPREFIXED
    attribute names, byte-for-byte the same code path as JpsiMuCandidate.

    The PION hypothesis is added by compute_alt_bachelor (called after
    compute_vtx_quantities): it DUPLICATES the full 2mu+bachelor vertex fit with
    the pion mass on the same tracks and recomputes every hypothesis-dependent
    quantity under the 'pi_' prefix, reusing the shared PV / J/psi fit /
    isolation. self.pi_p4 is the pion bachelor 4-momentum, handed to
    compute_alt_bachelor by the inspector.

    Only the bachelor mass hypotheses and the mother pdgId differ from the base;
    everything else is inherited.
    '''

    # the KAON mass is the primary hypothesis: the base fits the full 2mu+bachelor
    # vertex with this mass and stores the result unprefixed
    _bachelor_mass = masses['k']

    def __init__(self, jpsi_muons, track):
        # primary bachelor = the track under the CHARGED-KAON hypothesis
        kaon = BachelorTrack(track, masses['k'])
        super().__init__(jpsi_muons, kaon, mother_pdgid=521)  # B+
        # keep the raw track and the PION-hypothesis 4-momentum around for the
        # alternative reconstruction (compute_alt_bachelor)
        self.track = track
        self.pi_bachelor_p4 = p4_with_mass(track, masses['pi'], root_type=1)

    def compute_alt_hypothesis(self):
        '''Convenience wrapper: run the pion (Bc+ -> J/psi pi+) hypothesis, storing
        every quantity under the 'pi_' prefix. Must be called after
        compute_vtx_quantities. Kept as a named method so the inspector reads
        cleanly (icand.compute_alt_hypothesis()).'''
        self.compute_alt_bachelor(self.pi_bachelor_p4, masses['pi'], prefix='pi_')
