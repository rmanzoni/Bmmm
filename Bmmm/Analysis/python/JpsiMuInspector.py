'''
J/psi mu (RJpsi) inspector: Bc -> J/psi(-> mu mu) mu.

A thin BaseInspector subclass whose hooks reproduce the original inspector_rjpsi
byte-for-byte (3-muon combinatorics, Bc/Hb gen-truth classifier, in-loop neutrino
reconstruction, reco helicity angles), so the produced ntuple is identical.
'''

import ROOT
import numpy as np
from itertools import product
from PhysicsTools.HeppyCore.utils.deltar import deltaR, bestMatch

from Bmmm.Analysis.JpsiChargedInspector import BaseInspector
from Bmmm.Analysis.JpsiMuBranches import (
    branches, paths, muon_branches, cand_branches, event_branches, bc_branches, safe_get,
)
from Bmmm.Analysis.JpsiMuCandidate import JpsiMuCandidate as Candidate
from Bmmm.Analysis.JpsiMuCuts import cuts
from Bmmm.Analysis.utils import cutflow, masses
from Bmmm.Analysis.RJPsiGenHistory import BcGenDecay, gen_kinematics, gen_helicity_angles, gen_hammer_p4
from Bmmm.Analysis.RJPsiMuonMatcher import match_candidate_muons, signal_gen_muons, ROLE
from Bmmm.Analysis.RJPsiHbMatcher import match_hb_candidate, hb_status1_muons
from Bmmm.Analysis.RJPsiNuReco import reconstruct, M_BC


class JpsiMuInspector(BaseInspector):

    DESCRIPTION    = 'RJpsi (J/psi mu) ntuplizer'
    CHANNEL        = 'jpsi_mu'
    CUTS           = cuts
    CANDIDATE      = Candidate
    BRANCHES       = branches
    PATHS          = paths
    EVENT_BRANCHES = event_branches
    CAND_BRANCHES  = cand_branches
    MUON_BRANCHES  = muon_branches
    SAFE_GET       = safe_get
    EVENT_GEN_KEYS = tuple(bc_branches.keys())
    MIN_MUONS      = 3

    # ---- build 3-muon (J/psi + bachelor mu) candidates ---------------------
    def build_candidates(self, muons, event, options, cuts, good_tobjs):
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

                for kk in range(nmuons):
                    if kk == ii or kk == jj:
                        continue
                    mu3 = muons[kk]

                    cand = self.CANDIDATE([mu1, mu2], mu3)
                    cutflow['\tcandidates after HLT and 3mu'] += 1

                    if cand.mass() > cuts['max_mass']:
                        continue
                    cutflow['\tpass 3mu mass cut'] += 1

                    cands.append(cand)
        return cands

    def sort_candidates(self, cands):
        cands.sort(key=lambda x: (abs(x.charge()) == 0, x.pt(),
                                  -np.abs(x.jpsi.mass() - masses['jpsi'])), reverse=True)
        return cands

    # ---- once-per-event Bc / Hb gen-truth classifier -----------------------
    def setup_event_gen(self, event, options, event_tofill):
        gen_info   = None   # signal_gen_muons() result, reused by every candidate's matcher
        hb_gen_mus = None   # status-1 gen muons for the Hb matcher (no-Bc events)
        bc         = None
        if options.mc:
            event.bc_gen = BcGenDecay.from_genparticles(event.genpr)

            if event.bc_gen is not None:
                bc = event.bc_gen.bc

            if bc is not None:
                bc.bc_code = event.bc_gen.code

                if event.bc_gen.name in ['Jpsi_mu_nu', 'Jpsi_tau_nu']:
                    gk = gen_kinematics(event.bc_gen)
                    for key in ('q2', 'm_miss2', 'm_miss2_vis', 'e_mu_bc', 'e_mu_jpsi'):
                        setattr(bc, key, gk[key])
                    ha = gen_helicity_angles(event.bc_gen)
                    for key in ('cos_theta_v', 'cos_theta_l', 'chi'):
                        setattr(bc, key, ha[key])

                    # pre-FSR gen four-momenta for Hammer FF reweighting
                    for key, val in gen_hammer_p4(event.bc_gen).items():
                        setattr(bc, key, val)

                for branch, getter in bc_branches.items():
                    event_tofill[branch] = safe_get(getter, bc, verbose=options.verbose, name=branch)
            else:
                for branch in bc_branches:
                    event_tofill[branch] = np.nan

            gen_info = signal_gen_muons(event.genpr)
            if gen_info is None:
                hb_gen_mus = hb_status1_muons(event.genpr)
        else:
            for branch in bc_branches:
                event_tofill[branch] = np.nan

        return (gen_info, hb_gen_mus, bc)

    # ---- per-candidate row --------------------------------------------------
    def fill_candidate(self, icand, event, options, cuts, gen_state, good_tobjs):
        gen_info, hb_gen_mus, bc = gen_state

        if options.mc:
            if gen_info is not None:
                match_candidate_muons(icand, event.genpr, dr_max=0.04, info=gen_info)
            else:
                match_hb_candidate(icand, event.genpr, dr_max=0.04, gen_muons=hb_gen_mus)

        icand.trig_match = self.trig_match(icand, cuts, good_tobjs, icand.muons)

        icand.compute_vtx_quantities(event.vtx, event.bs, event.pf, event.ltrk)

        cand_tofill = self._CAND_TEMPLATE.copy()

        # mu1 -> leading jpsi muon, mu2 -> trailing jpsi muon, mu3 -> bachelor
        muons_dict = {1: lambda x: x.jpsi_muons[0],
                      2: lambda x: x.jpsi_muons[1],
                      3: lambda x: x.mu}

        for idx in range(1, 4):
            imu = muons_dict[idx](icand)
            self.fill_muon(imu, icand, event, options)
            for branch, getter in muon_branches.items():
                cand_tofill['mu%d_%s' % (idx, branch)] = safe_get(getter, imu, verbose=options.verbose, name=branch)

        # ---- neutrino reconstruction + reco helicity angles -----------------
        have_dir = (getattr(icand, 'Bdirection_jpsi', None) is not None or
                    getattr(icand, 'Bdirection_sv',   None) is not None)

        if have_dir and icand.jpsi_rfp4 is not None:

            visible_p4   = icand.jpsi_rfp4 + icand.mu.p4()
            visible_mass = visible_p4.mass()

            for label in ('jpsi', 'sv'):
                bdir = getattr(icand, 'Bdirection_%s' % label, None)
                if bdir is None:
                    continue
                sols = reconstruct(visible_p4, bdir, m_parent=M_BC, clamp_negative_disc=True)
                if len(sols) < 2:
                    continue
                setattr(icand, 'sols_%s'       % label, sols)
                setattr(icand, 'math1_b_p4_%s' % label, visible_p4 + sols[0].p4_nu)
                setattr(icand, 'math2_b_p4_%s' % label, visible_p4 + sols[1].p4_nu)

            if options.mc and bc is not None:
                gen_dir   = ROOT.Math.XYZVector(bc.daughter(0).vx() - bc.vx(),
                                                bc.daughter(0).vy() - bc.vy(),
                                                bc.daughter(0).vz() - bc.vz())
                bc_gen_p  = gen_dir.unit() * M_BC / visible_mass * visible_p4.P()
                bc_gen_p4 = ROOT.Math.LorentzVector('ROOT::Math::PxPyPzE4D<double>')(
                    bc_gen_p.x(), bc_gen_p.y(), bc_gen_p.z(), np.sqrt(M_BC**2 + bc_gen_p.Mag2()))

                icand.q2_gen      = (bc_gen_p4 - icand.jpsi_rfp4).mass2()
                icand.m_miss2_gen = (bc_gen_p4 - icand.jpsi_rfp4 - icand.mu.p4()).mass2()

            icand.compute_helicity_angles()

        for branch, getter in cand_branches.items():
            cand_tofill[branch] = safe_get(getter, icand, verbose=options.verbose, name=branch)

        return cand_tofill
