'''
J/psi + charged-track inspector: B+ -> J/psi K+  and  Bc+ -> J/psi pi+, built as
ONE candidate per (dimuon, track) pair carrying BOTH mass hypotheses (kaon
unprefixed, pion pi_-prefixed; see JpsiTkCandidate / compute_alt_bachelor).

The muon side is selected exactly as in the J/psi mu analysis (shared muon
selection in BaseInspector + the shared 'rjpsi'/'jpsi_tk' cuts). The bachelor is a
charged pat::PackedCandidate pseudo-track (from packedPFCandidates + lostTracks).

MC gen matching here is PROVISIONAL and sample-specific (assumes the B+ ->
J/psi K+ topology: J/psi muons from a 443, bachelor from a 521). It is a simple
Delta-R match, guarded by --mc; on data it is skipped and all gen_* branches are
NaN. Adjust to the actual signal sample before using the gen branches.
'''

import ROOT
import numpy as np
from itertools import product
from PhysicsTools.HeppyCore.utils.deltar import deltaR, bestMatch

from Bmmm.Analysis.JpsiChargedInspector import BaseInspector
from Bmmm.Analysis.JpsiTkBranches import (
    branches, paths, muon_branches, cand_branches, event_branches, k_branches, safe_get,
)
from Bmmm.Analysis.JpsiTkCandidate import JpsiTkCandidate as Candidate
from Bmmm.Analysis.JpsiTkCuts import cuts
from Bmmm.Analysis.utils import cutflow, masses, p4_with_mass
from Bmmm.Analysis.RJPsiNuReco import reconstruct, M_BC

# charged-track pdgId set the bachelor may take (exclude neutrals / leptons that
# would never be a kaon/pion hadron track)
_TRACK_VETO_PDG = (11, 13, 22, 130, 310, 111, 2112)


class JpsiTkInspector(BaseInspector):

    DESCRIPTION    = 'J/psi + track (B+ -> J/psi K+ / Bc+ -> J/psi pi+) ntuplizer'
    CHANNEL        = 'jpsi_tk'
    CUTS           = cuts
    CANDIDATE      = Candidate
    BRANCHES       = branches
    PATHS          = paths
    EVENT_BRANCHES = event_branches
    CAND_BRANCHES  = cand_branches
    MUON_BRANCHES  = muon_branches
    SAFE_GET       = safe_get
    EVENT_GEN_KEYS = ()          # no event-level gen block for this channel
    MIN_MUONS      = 2

    # ---- bachelor-track quality -------------------------------------------
    @staticmethod
    def _good_track(trk, cuts):
        if trk.charge() == 0:
            return False
        if not trk.hasTrackDetails():
            return False
        if trk.pt() <= cuts['k_pt']:
            return False
        if abs(trk.eta()) >= cuts['k_eta']:
            return False
        if abs(trk.pdgId()) in _TRACK_VETO_PDG:
            return False
        if abs(trk.bestTrack().dxy()) >= cuts['k_dxy']:
            return False
        if abs(trk.bestTrack().dz()) >= cuts['k_dz']:
            return False
        return True

    @staticmethod
    def _overlaps_jpsi_muon(trk, jpsi_muons, cuts):
        '''Overlap removal: drop a track that is one of the J/psi muons (same
        charge and within k_dr_mu).'''
        for imu in jpsi_muons:
            if trk.charge() == imu.charge() and \
               deltaR(imu.eta(), imu.phi(), trk.eta(), trk.phi()) < cuts['k_dr_mu']:
                return True
        return False

    # ---- build (J/psi + track) candidates ---------------------------------
    def build_candidates(self, muons, event, options, cuts, good_tobjs):
        # collect the good bachelor tracks once per event
        tracks = [trk for coll in (event.pf, event.ltrk) for trk in coll
                  if self._good_track(trk, cuts)]

        cands  = []
        nmuons = len(muons)
        for ii in range(nmuons):
            for jj in range(ii + 1, nmuons):
                mu1, mu2 = muons[ii], muons[jj]

                if mu1.charge() + mu2.charge() != 0:
                    continue
                if mu1.pt() < cuts['tight_mu_pt'] or mu2.pt() < cuts['tight_mu_pt']:
                    continue
                if np.abs((mu1.p4() + mu2.p4()).mass() - masses['jpsi']) > cuts['jpsi_mass_window']:
                    continue
                cutflow['\tpass jpsi mass cut (pair)'] += 1
                jpsi_p4 = mu1.p4() + mu2.p4()

                for trk in tracks:
                    if self._overlaps_jpsi_muon(trk, [mu1, mu2], cuts):
                        continue

                    # keep if EITHER mass hypothesis lands in the window (computed
                    # from raw p4s, before building the candidate)
                    k_mass  = (jpsi_p4 + p4_with_mass(trk, masses['k'],  root_type=1)).mass()
                    pi_mass = (jpsi_p4 + p4_with_mass(trk, masses['pi'], root_type=1)).mass()
                    if not ((cuts['min_mass'] < k_mass  < cuts['max_mass']) or
                            (cuts['min_mass'] < pi_mass < cuts['max_mass'])):
                        continue
                    cutflow['\tpass mass window (K or pi)'] += 1

                    cands.append(self.CANDIDATE([mu1, mu2], trk))
                    cutflow['\tcandidates after HLT and 2mu+trk'] += 1
        return cands

    def sort_candidates(self, cands):
        # rank OS candidates first, then by pt, then by proximity of the KAON mass
        # to the B+ mass
        cands.sort(key=lambda x: (abs(x.charge()) == 0, x.pt(),
                                  -np.abs(x.mass() - masses['bs'])), reverse=True)
        return cands

    # ---- provisional MC gen matching --------------------------------------
    def setup_event_gen(self, event, options, event_tofill):
        if not options.mc:
            return None
        # status-1 muons from a J/psi (443) and status-1 charged hadrons from a
        # B (|pdgId| 521). PROVISIONAL: adapt to the real signal sample.
        gen_mus, gen_had = [], []
        for gp in event.genpr:
            if gp.status() != 1:
                continue
            if abs(gp.pdgId()) == 13:
                gen_mus.append(gp)
            elif abs(gp.pdgId()) in (321, 211):
                gen_had.append(gp)
        return (gen_mus, gen_had)

    @staticmethod
    def _gen_match(obj, gen_list, dr2_max):
        if not gen_list:
            return
        best, dr2 = bestMatch(obj, gen_list)
        if dr2 < dr2_max:
            obj.gen_match = best
            obj.gen_dr    = np.sqrt(dr2)

    # ---- per-candidate row --------------------------------------------------
    def fill_candidate(self, icand, event, options, cuts, gen_state, good_tobjs):

        icand.trig_match = self.trig_match(icand, cuts, good_tobjs, icand.jpsi_muons)

        # kaon (primary) reconstruction, then the pion (alternative) hypothesis
        icand.compute_vtx_quantities(event.vtx, event.bs, event.pf, event.ltrk)
        icand.compute_alt_hypothesis()

        cand_tofill = self._CAND_TEMPLATE.copy()

        # ---- per-muon (mu1 / mu2 = the two J/psi muons) ---------------------
        muons_dict = {1: lambda x: x.jpsi_muons[0], 2: lambda x: x.jpsi_muons[1]}
        for idx in (1, 2):
            imu = muons_dict[idx](icand)
            self.fill_muon(imu, icand, event, options)
            if options.mc and gen_state is not None:
                self._gen_match(imu, gen_state[0], self._GEN_DR2)
            for branch, getter in muon_branches.items():
                cand_tofill['mu%d_%s' % (idx, branch)] = safe_get(getter, imu, verbose=options.verbose, name=branch)

        # ---- bachelor track (k_*) -------------------------------------------
        ik    = icand.mu                 # the kaon-hypothesis BachelorTrack
        ik.pv = icand.pv
        ik.bs = icand.bs
        jet, dr2 = bestMatch(ik, event.jets)
        if dr2 < self._JET_MATCH_DR2:
            ik.jet = jet
        if options.mc and gen_state is not None:
            self._gen_match(ik, gen_state[1], self._GEN_DR2)
        for branch, getter in k_branches.items():
            cand_tofill['k_%s' % branch] = safe_get(getter, ik, verbose=options.verbose, name=branch)

        # ---- neutrino reconstruction + reco helicity, BOTH hypotheses -------
        if icand.jpsi_rfp4 is not None:
            # kaon (unprefixed)
            self._reco_neutrinos(icand, prefix='',    bachelor_p4=icand.mu.p4())
            # pion (pi_)
            self._reco_neutrinos(icand, prefix='pi_', bachelor_p4=icand.pi_bachelor_p4)

            icand.compute_helicity_angles()  # kaon
            icand.compute_helicity_angles(prefix='pi_',
                                          bachelor_p4=icand.pi_bachelor_p4,
                                          visible_p4=icand.jpsi_rfp4 + icand.pi_bachelor_p4)

        for branch, getter in cand_branches.items():
            cand_tofill[branch] = safe_get(getter, icand, verbose=options.verbose, name=branch)

        return cand_tofill

    @staticmethod
    def _reco_neutrinos(icand, prefix, bachelor_p4):
        '''Exact 2-fold neutrino-pz reconstruction for one mass hypothesis, one
        set of solutions per flight direction, storing {prefix}sols_/{prefix}math*
        _b_p4_ . Mirrors the J/psi mu inspector, prefix-aware.'''
        visible_p4 = icand.jpsi_rfp4 + bachelor_p4
        for label in ('jpsi', 'sv'):
            bdir = getattr(icand, 'Bdirection_%s%s' % (prefix, label), None)
            if bdir is None:
                continue
            sols = reconstruct(visible_p4, bdir, m_parent=M_BC, clamp_negative_disc=True)
            if len(sols) < 2:
                continue
            setattr(icand, '%ssols_%s'       % (prefix, label), sols)
            setattr(icand, '%smath1_b_p4_%s' % (prefix, label), visible_p4 + sols[0].p4_nu)
            setattr(icand, '%smath2_b_p4_%s' % (prefix, label), visible_p4 + sols[1].p4_nu)
