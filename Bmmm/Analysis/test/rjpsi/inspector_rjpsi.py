'''
Example:

ipython -i -- inspector_rjpsi.py --inputFiles="root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/HbToPsiX_JMM_TuneCP5_13TeV-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2540000/D388E03F-0214-E842-9905-26008B393E50.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/HbToPsiX_JMM_TuneCP5_13TeV-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2540000/261075CA-133E-7A4E-B619-8846615BBD44.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/HbToPsiX_JMM_TuneCP5_13TeV-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/50000/C82BE32E-7DC6-394D-A213-04F8E087875A.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/HbToPsiX_JMM_TuneCP5_13TeV-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/50000/11B8EB1F-C40C-1441-AD14-B9ABBF52B97F.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/HbToPsiX_JMM_TuneCP5_13TeV-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2520000/1EC3014E-488E-AB44-9F1A-20821D6C0189.root" --filename=rjpsi_hb --mc --maxevents=-1

ipython -i -- inspector_rjpsi.py --inputFiles="root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/BcToJPsiMuMu_inclusive_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/230000/0443354B-2D3F-CF41-A1F0-0FC4F92E718E.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/BcToJPsiMuMu_inclusive_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/230000/05C316B4-BD3D-CC4E-8BDB-C603259F1016.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/BcToJPsiMuMu_inclusive_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/230000/0D7329CE-CC25-1A4C-848D-CDF63DA314B5.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/BcToJPsiMuMu_inclusive_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/230000/127F4AE1-CE44-6E4D-95A5-C25986F28A1B.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/BcToJPsiMuMu_inclusive_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/230000/17FDD58D-FEAD-5B49-83DD-3A8C1C1A3960.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/BcToJPsiMuMu_inclusive_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/230000/189B54A2-6B0D-ED49-A7FA-A94DF21D41A0.root" --filename=rjpsi_bc_signal --mc --maxevents=-1 --savenontrig

ipython -i -- inspector_rjpsi.py --inputFiles=0443354B-2D3F-CF41-A1F0-0FC4F92E718E.root --filename=rjpsi_bc_signal_small --mc --maxevents=-1 --savenontrig




    

ipython -i -- inspector_rjpsi.py --inputFiles=root://cms-xrd-global.cern.ch///store/data/Run2022D/ParkingDoubleMuonLowMass0/MINIAOD/PromptReco-v1/000/357/539/00000/8abcc7e1-c6f0-4fcd-9be9-e07fb6878777.root,root://cms-xrd-global.cern.ch///store/data/Run2022D/ParkingDoubleMuonLowMass0/MINIAOD/PromptReco-v1/000/357/542/00000/c5941f20-7f8f-40ae-976c-cb354a3f1a06.root,root://cms-xrd-global.cern.ch///store/data/Run2022D/ParkingDoubleMuonLowMass0/MINIAOD/PromptReco-v1/000/357/542/00000/fc709d38-043c-4529-85fb-567ca09214a2.root,root://cms-xrd-global.cern.ch///store/data/Run2022D/ParkingDoubleMuonLowMass0/MINIAOD/PromptReco-v1/000/357/542/00000/6af139f3-2242-42ce-b2a5-09387387c1db.root,root://cms-xrd-global.cern.ch///store/data/Run2022D/ParkingDoubleMuonLowMass0/MINIAOD/PromptReco-v1/000/357/550/00000/9f95cf6b-dfb7-466a-9ab4-6213ffd5b080.root,root://cms-xrd-global.cern.ch///store/data/Run2022D/ParkingDoubleMuonLowMass0/MINIAOD/PromptReco-v1/000/357/542/00000/87bf23a8-e77d-44a6-8623-961db9319eda.root,root://cms-xrd-global.cern.ch///store/data/Run2022D/ParkingDoubleMuonLowMass0/MINIAOD/PromptReco-v1/000/357/542/00000/38d1ee7b-83c0-4e86-8510-204b7f737e7f.root,root://cms-xrd-global.cern.ch///store/data/Run2022D/ParkingDoubleMuonLowMass0/MINIAOD/PromptReco-v1/000/357/542/00000/ecdd5c5f-a1cd-418e-a4a1-e77c53db4d3a.root,root://cms-xrd-global.cern.ch///store/data/Run2022D/ParkingDoubleMuonLowMass0/MINIAOD/PromptReco-v1/000/357/542/00000/66783f3e-8d0d-43ec-97fe-c10dfa92e095.root --filename=data_2022d_partial --maxevents=-1 


'''

import gc
import sys
import ROOT
import argparse
import numpy as np
import pandas as pd
import uproot
from time import time
from datetime import datetime, timedelta
from glob import glob
from collections import namedtuple
from DataFormats.FWLite import Events, Handle
from PhysicsTools.HeppyCore.utils.deltar import deltaR, bestMatch
from itertools import product, combinations
from Bmmm.Analysis.RJpsiBranches import branches, paths, muon_branches, cand_branches, event_branches, bc_branches, safe_get
from Bmmm.Analysis.RJpsiCandidate import RJpsiCandidate as Candidate
from Bmmm.Analysis.utils import drop_hlt_version, cutflow, p4_with_mass, masses, compute_mass, is_b_hadron
from Bmmm.Analysis.RJpsiCuts import cuts
# from Bmmm.Analysis.Handles import handles, handles_mc
from Bmmm.Analysis.Handles import handles_mc
from Bmmm.Analysis.Handles import handles_skim as handles # includes BS constrained vertices
from Bmmm.Analysis.RJPsiGenHistory import BcGenDecay, gen_kinematics, gen_helicity_angles
from Bmmm.Analysis.RJPsiMuonMatcher import match_candidate_muons, signal_gen_muons, ROLE
from Bmmm.Analysis.RJPsiHbMatcher import match_hb_candidate, hb_status1_muons
from Bmmm.Analysis.RJPsiNuReco import gen_nu_reco, reconstruct, solve_nu_pz, pick_closest, M_BC

# pre-compute constants used in the hot loop
_JET_MATCH_DR2 = cuts['rjpsi']['jet_dr'] ** 2

# template for candidate-level NaN initialisation: covers muon, cand, bs, jpsi, phi branches
_TRIGGER_KEYS = set(k for p in paths for k in (p, p + '_ps'))
_EVENT_KEYS   = set(event_branches.keys())
_GEN_KEYS     = set(bc_branches.keys())
_CAND_KEYS    = [b for b in branches if b not in _TRIGGER_KEYS and b not in _EVENT_KEYS and b not in _GEN_KEYS]
_CAND_TEMPLATE = dict.fromkeys(_CAND_KEYS, np.nan)

######################################################################################
#####      INCREMENTAL (BATCHED) OUTPUT
######################################################################################
# Memory footprint of the job is bounded by WRITE_EVERY rows instead of growing
# with the whole file: rows are flushed to the TTree and the buffer is cleared.
# Each row is ~400 scalar branches (~15-20 kB), so 50k rows ~ <1 GB transient.
WRITE_EVERY  = 50000                 # rows buffered before a flush; tune to memory
INT_BRANCHES = ('run', 'lumi', 'event')

def build_branch_types(branches):
    '''Fixed schema for the TTree: int64 for the event identifiers, float32 for
    everything else. Used by mktree so every extend() call matches it exactly.'''
    return {c: (np.int64 if c in INT_BRANCHES else np.float32) for c in branches}

def rows_to_columns(rows, branches):
    '''list-of-dicts -> {branch: numpy array} with a FIXED dtype per branch, so
    every uproot extend() presents an identical schema. pd.to_numeric turns
    bools into 0/1 and leaves NaN where a branch was never filled.'''
    df = pd.DataFrame(rows, columns=branches)
    out = {}
    for col in branches:
        s = pd.to_numeric(df[col], errors='coerce')
        if col in INT_BRANCHES:
            out[col] = s.fillna(0).astype(np.int64).to_numpy()
        else:
            out[col] = s.astype(np.float32).to_numpy()
    return out

def flush(fout, row_list, branches):
    '''Append the buffered rows to the TTree and clear the buffer in place
    (row_list is the same list object the looper holds).'''
    if not row_list:
        return
    fout['tree'].extend(rows_to_columns(row_list, branches))
    row_list.clear()
    # gc.collect()   # uncomment if you still see a slow baseline creep

######################################################################################
#####      LOOPER
######################################################################################
def looper(events, options, handles, handles_mc, row_list, start, fout, branches):

    i = 0
    for i, event in enumerate(events, 1):

        if i > options.maxevents:
            flush(fout, row_list, branches)
            return i - 1, cutflow

        if i % options.logfreq == 0:
            percentage = float(i) / options.maxevents * 100.
            speed      = float(i) / (time() - start)
            eta        = datetime.now() + timedelta(seconds=(options.maxevents - i) / max(0.1, speed))
            print('\t===> processing %d / %d event \t completed %.1f%s \t %.1f ev/s \t ETA %s'
                  % (i, options.maxevents, percentage, '%', speed, eta.strftime('%Y-%m-%d %H:%M:%S')))

        # ---- load handles -------------------------------------------------------
        for k, v in handles.items():
            event.getByLabel(v[0], v[1])
            setattr(event, k, v[1].product())

        event.mc = False

        if options.mc:
            event.mc = True
            for k, v in handles_mc.items():
                event.getByLabel(v[0], v[1])
                setattr(event, k, v[1].product())
            event.pu_at_bx0 = [ipu for ipu in event.pu if ipu.getBunchCrossing() == 0][0]

        cutflow['all processed events'] += 1
    
        irun = event.eventAuxiliary().run()
        lumi = event.eventAuxiliary().luminosityBlock()
        iev  = event.eventAuxiliary().event()
        
        ######################################################################################
        #####      TRIGGERS — filled ONCE per event
        ######################################################################################
        trg_names = event.object().triggerNames(event.trg_res)
        _trg_len  = len(trg_names)

        # one entry per path and its prescale, initialised to NaN
        trigger_tofill = {}
        for ipath in paths:
            trigger_tofill[ipath]          = np.nan
            trigger_tofill[ipath + '_ps']  = np.nan

        # map trigger index → stripped path name (strips _vN suffix, skips unknowns)
        idx_to_path = {}
        for iname in trg_names.triggerNames():
            iname    = str(iname)
            stripped = drop_hlt_version(iname)
            if stripped in paths:
                idx_to_path[trg_names.triggerIndex(iname)] = stripped

        # OR across all versions of the same path that fired
        for idx, ipath in idx_to_path.items():
            accept = int(idx < _trg_len and event.trg_res.accept(idx))
            ps     = event.trg_ps.getPrescaleForIndex(idx)
            trigger_tofill[ipath]         = np.nanmax([trigger_tofill[ipath],         accept])
            trigger_tofill[ipath + '_ps'] = np.nanmax([trigger_tofill[ipath + '_ps'], ps    ])

        # nan > 0 is False in numpy, so this is safe even before any path fires
        hlt_passed = any(trigger_tofill[p] > 0 for p in paths)

        if not (options.savenontrig or hlt_passed):
            continue

        cutflow['pass HLT'] += 1
        
        ######################################################################################
        #####      TRIGGER OBJECTS — built ONCE per event
        ######################################################################################
        good_tobjs      = {key: []    for key in paths}
        good_tobjs_seen = {key: set() for key in paths}

        # only worth unpacking filter labels if some path actually fired
        if hlt_passed:
            for to in event.tobjs:
                if to.pt() < cuts['rjpsi']['to_pt'] or abs(to.eta()) >= cuts['rjpsi']['to_eta']:
                    continue
                to.unpackNamesAndLabels(event.object(), event.trg_res)
                for k, v in paths.items():
                    if trigger_tofill[k] != 1:
                        continue
                    for ilabel in v:
                        if to.hasFilterLabel(ilabel) and id(to) not in good_tobjs_seen[k]:
                            good_tobjs[k].append(to)
                            good_tobjs_seen[k].add(id(to))

        ######################################################################################
        #####      MUON SELECTION
        ######################################################################################
        muons = [mu for mu in event.muons
                 if mu.pt()                    >  cuts['rjpsi']['mu_pt']
                 and abs(mu.eta())             <  cuts['rjpsi']['mu_eta']
                 and cuts['rjpsi']['mu_id'](mu)
                 and abs(mu.bestTrack().dxy()) <  cuts['rjpsi']['mu_dxy']]
        muons.sort(key=lambda x: x.pt(), reverse=True)

        if len(muons) < 3:
            continue

        cutflow['at least 3 muons'] += 1
                
        ######################################################################################
        #####      BUILD AND SELECT 3MU CANDIDATES
        ######################################################################################
        cands = []

        nmuons = len(muons)
        for ii in range(nmuons):
            for jj in range(ii + 1, nmuons):
                mu1, mu2 = muons[ii], muons[jj]
        
                if mu1.charge() + mu2.charge() != 0:
                    continue

                if mu1.pt()<cuts['rjpsi']['tight_mu_pt'] or mu2.pt()<cuts['rjpsi']['tight_mu_pt']:
                    continue

                # J/psi mass window on the dimuon, hoisted out of the bachelor loop:
                # if this OS pair is not a J/psi there is no point looping over the
                # third muon at all. Uses the same J/psi mass as the vertex constraint.
                if np.abs((mu1.p4() + mu2.p4()).mass() - masses['jpsi']) > cuts['rjpsi']['jpsi_mass_window']:
                    continue
                cutflow['\tpass jpsi mass cut (pair)'] += 1

                for kk in range(nmuons):
                    if kk == ii or kk == jj:
                        continue
                    
                    mu3 = muons[kk]
                                        
                    cand = Candidate([mu1, mu2], mu3)
                    cutflow['\tcandidates after HLT and 3mu'] += 1

                    if cand.mass()>cuts['rjpsi']['max_mass']:
                        continue
                    cutflow['\tpass 3mu mass cut'] += 1

                    cands.append(cand)
                    
        if len(cands) == 0:
            continue

#         import ipdb ; ipdb.set_trace()

        event.ncands = len(cands)
        cutflow['at least one cand pass presel'] += 1

        cands.sort(key=lambda x: (abs(x.charge()) == 0, x.pt(), -np.abs(x.jpsi.mass() - masses['jpsi'])), reverse=True)
        
        ######################################################################################
        #####      EVENT-LEVEL TOFILL — filled ONCE, shared across all candidates
        ######################################################################################
        event_tofill = {}
        for branch, getter in event_branches.items():
            event_tofill[branch] = getter(event)

        ######################################################################################
        #####      BC MC TRUTH CLASSIFIER
        ######################################################################################
        gen_info = None   # signal_gen_muons() result, reused by every candidate's matcher
        hb_gen_mus = None # status-1 gen muons for the Hb matcher (no-Bc events)
        bc       = None
        if options.mc:
            event.bc_gen = BcGenDecay.from_genparticles(event.genpr)

            # from_genparticles returns None for samples with no decayed Bc
            # (e.g. the entire HbToPsiX background) -> leave bc None and NaN-fill
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

                for branch, getter in bc_branches.items():
                    event_tofill[branch] = safe_get(getter, bc, verbose=options.verbose, name=branch)
            else:
                for branch in bc_branches:
                    event_tofill[branch] = np.nan

            # gen muon collection for reco<->gen matching: identical for every
            # candidate, so build it once per event
            gen_info = signal_gen_muons(event.genpr)

            # no decayed Bc (e.g. the whole HbToPsiX background): fall back to the
            # Hb matcher, whose gen side (status-1 gen muons) is also event-level
            if gen_info is None:
                hb_gen_mus = hb_status1_muons(event.genpr)
        else:
            for branch in bc_branches:
                event_tofill[branch] = np.nan

        ######################################################################################
        #####      FILL ONE ROW PER CANDIDATE
        ######################################################################################
        for icand in cands:

            if options.mc:
                # reco<->gen matching; tags each muon's gen_role/gen_match/gen_dr.
                # gen_info is computed once per event and reused here.
                if gen_info is not None:
                    match_candidate_muons(icand, event.genpr, dr_max=0.04, info=gen_info)
                else:
                    # Hb background: tag mu*_gen_* and the candidate-level
                    # gen_hb_* truth (same-mother flag + b-ancestor pdgIds).
                    match_hb_candidate(icand, event.genpr, dr_max=0.04, gen_muons=hb_gen_mus)

            # trigger matching (informational, NOT a selection cut): does the
            # candidate have >=2 muons within hlt_dr of the fired HLT objects?
            hlt_objs = good_tobjs.get(cuts['rjpsi']['hlt'], [])
            icand.trig_match = sum(
                deltaR(imu, to) < cuts['rjpsi']['hlt_dr']
                for imu, to in product(icand.muons, hlt_objs)
            ) >= 2

            # event.pf / event.ltrk = packedPFCandidates / lostTracks (kept by
            # the skim): enable the custom PF isolation and the on-the-fly PV
            # refit. The refit rebuilds the chosen PV's track set from these
            # candidates' pseudo-tracks (closest-z association) + the beamspot,
            # so the persisted unpackedTracksAndVertices / primaryVertexRefit
            # collections are no longer needed.
            icand.compute_vtx_quantities(event.vtx, event.bs, event.pf, event.ltrk)

            # start from NaN for all candidate-level branches
            cand_tofill = _CAND_TEMPLATE.copy()
                        
            # organize muons in the ntuples such that
            # mu1 --> leading pt muon from jpsi
            # mu2 --> trailing pt muon from jpsi
            # mu3 --> bachelor muon
            muons_dict = {1 : lambda x : x.jpsi_muons[0], 
                          2 : lambda x : x.jpsi_muons[1],
                          3 : lambda x : x.mu}

            # ---- per-muon quantities -------------------------------------------------
            for idx in range(1,4):
                imu = muons_dict[idx](icand)
                imu.pv    = icand.pv
                imu.bs    = icand.bs
                imu.iso03 = imu.pfIsolationR03()
                imu.iso04 = imu.pfIsolationR04()

                # jet matching
                jet, dr2 = bestMatch(imu, event.jets)
                if dr2 < _JET_MATCH_DR2:
                    imu.jet = jet

                for branch, getter in muon_branches.items():
                    mygetter = safe_get(getter, imu, verbose=options.verbose, name=branch)                    
                    cand_tofill['mu%d_%s' % (idx, branch)] = mygetter


            # ---- neutrino reconstruction + reco helicity angles ---------------------
            # everything here needs the refitted, mass-constrained J/psi. With
            # per-label vertex gating Bdirection_sv can exist while the J/psi fit
            # failed (jpsi_rfp4 is None), so guard on it explicitly.
            have_dir = (getattr(icand, 'Bdirection_jpsi', None) is not None or
                        getattr(icand, 'Bdirection_sv',   None) is not None)

            if have_dir and icand.jpsi_rfp4 is not None:

                visible_p4   = icand.jpsi_rfp4 + icand.mu.p4()
                visible_mass = visible_p4.mass()

                # exact 2-fold neutrino-pz reconstruction, one set of solutions per
                # flight direction. Solve with the SAME refitted visible p4 that the
                # solutions are added back to, otherwise the Bc mass constraint that
                # defines the quadratic is violated.
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

                # gen-level equal-betagamma reference: TRUE Bc flight direction,
                # independent of the reco vertex choice. MC only, and only when a
                # decayed Bc was actually found (bc is None for the Hb background).
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
                      
            # ---- candidate-level quantities -----------------------------------------
            for branch, getter in cand_branches.items():
                cand_tofill[branch] = safe_get(getter, icand, verbose=options.verbose, name=branch)

            # ---- merge the three scopes into one flat row ---------------------------
            # trigger_tofill  : filled once per event, same for every candidate
            # event_tofill    : filled once per event, same for every candidate
            # cand_tofill     : filled fresh for each candidate
            row = {**trigger_tofill, **event_tofill, **cand_tofill}
            row_list.append(row)

        # ---- stream to disk in batches: keep resident memory bounded ------------
        # flushed once per event (never splitting an event across two baskets) as
        # soon as the buffer reaches WRITE_EVERY rows.
        if len(row_list) >= WRITE_EVERY:
            flush(fout, row_list, branches)

    # final partial batch of the file
    flush(fout, row_list, branches)
    return i, cutflow

######################################################################################
#####      MAIN
######################################################################################
def main():

    parser = argparse.ArgumentParser(description='RJpsi ntuplizer')
    parser.add_argument('--inputFiles',  dest='inputFiles',  required=True,          type=str)
    parser.add_argument('--verbose',     dest='verbose',     action='store_true')
    parser.add_argument('--destination', dest='destination', default='./',           type=str)
    parser.add_argument('--filename',    dest='filename',    required=True,          type=str)
    parser.add_argument('--maxevents',   dest='maxevents',   default=-1,             type=int)
    parser.add_argument('--mc',          dest='mc',          action='store_true')
    parser.add_argument('--logfreq',     dest='logfreq',     default=100,            type=int)
    parser.add_argument('--logger',      dest='logger',      default='',             type=str)
    parser.add_argument('--savenontrig', dest='savenontrig', action='store_true')
    parser.add_argument('--maxfiles',    dest='maxfiles',    default=-1,             type=int)
    parser.add_argument('--redirector',  dest='redirector',  default='root://cms-xrd-global.cern.ch//', type=str)

    args    = parser.parse_args()
    options = namedtuple('options', args.__dict__.keys())(*args.__dict__.values())

    if 'txt' in options.inputFiles:
        with open(options.inputFiles) as f:
            files = [options.redirector + line for line in f.read().splitlines()]
    elif (',' in options.inputFiles
          or 'cms-xrd-global'    in options.inputFiles
          or 'cms03.lcg.cscs.ch' in options.inputFiles
          or 't3dcachedb'        in options.inputFiles):
        files = options.inputFiles.split(',')
    else:
        files = glob(options.inputFiles)

    if options.maxfiles > 0:
        files = files[:options.maxfiles]

    print('files:', files)
        
    events  = Events(files)
    options = options._replace(
        maxevents=options.maxevents if options.maxevents >= 0 else events.size()
    )

    fout      = uproot.recreate(options.destination + '/' + options.filename + '.root', compression=uproot.ZSTD(5))

    # create the (empty) TTree up front with a fixed schema, so the looper can
    # append batches with extend() while it processes. An empty tree is written
    # if nothing is selected, which is harmless for a downstream hadd.
    fout.mktree('tree', build_branch_types(branches))

    row_list  = []
    start     = time()
    mytimestamp = datetime.now().strftime('%Y-%m-%d__%Hh%Mm%Ss')
    print('#### STARTING NOW', mytimestamp)

    ##########################################################################################
    #####      PROCESS EVENTS  (rows are flushed to disk inside the looper)
    ##########################################################################################
    n_proc_events, cutflow_result = looper(events, options, handles, handles_mc, row_list, start, fout, branches)

    ##########################################################################################
    #####      ANY REMAINING ROWS  (defensive: the looper already flushes its tail)
    ##########################################################################################
    flush(fout, row_list, branches)

    n_written = fout['tree'].num_entries
    print('\nnumber of selected events', n_written)
    print('\nntuple saved, processed all desired events?',
          (n_proc_events == options.maxevents),
          'processed', n_proc_events, 'maxevents', options.maxevents)

    ##########################################################################################
    #####      SAVE LOGGER
    ##########################################################################################
    logger_name = options.logger if len(options.logger) > 0 else 'logger_' + mytimestamp
    with open('%s.txt' % logger_name, 'w') as logger_file:
        for k, v in cutflow_result.items():
            print(k, v, file=logger_file)

    finish = time()
    print('done in %.1f hours' % ((finish - start) / 3600.))

######################################################################################
#####      ENTRY POINT
######################################################################################
if __name__ == '__main__':
    main()
