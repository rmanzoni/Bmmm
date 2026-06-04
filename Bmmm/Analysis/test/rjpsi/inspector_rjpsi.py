'''
Example:

ipython -i -- inspector_rjpsi.py \
--inputFiles="/pnfs/psi.ch/cms/trivcat/store/user/manzoni/Bs4Mu_MINIAOD_05sep23_v1/Bs4Mu_MINIAOD_41.root" \
--filename=rjpsi \
--mc \
--maxevents=5000

ipython -i -- inspector_b4m_analysis.py \
--inputFiles="7d3aac57-14ea-40d8-b898-0f14ee0d45c3.root" \
--filename=lifetime_check \
--mc \
--maxevents=1000


ipython -i -- inspector_b4m_analysis.py \
--inputFiles="files_ParkingDoubleMuonLowMass0-PromptReco-v2.txt" \
--filename=data \
--maxevents=100000







ipython -i -- inspector_rjpsi.py --inputFiles="root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/HbToPsiX_JMM_TuneCP5_13TeV-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2540000/D388E03F-0214-E842-9905-26008B393E50.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/HbToPsiX_JMM_TuneCP5_13TeV-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2540000/261075CA-133E-7A4E-B619-8846615BBD44.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/HbToPsiX_JMM_TuneCP5_13TeV-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/50000/C82BE32E-7DC6-394D-A213-04F8E087875A.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/HbToPsiX_JMM_TuneCP5_13TeV-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/50000/11B8EB1F-C40C-1441-AD14-B9ABBF52B97F.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/HbToPsiX_JMM_TuneCP5_13TeV-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2520000/1EC3014E-488E-AB44-9F1A-20821D6C0189.root" --filename=rjpsi_hb --mc --maxevents=-1

ipython -i -- inspector_rjpsi.py --inputFiles="root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/BcToJPsiMuMu_inclusive_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/230000/0443354B-2D3F-CF41-A1F0-0FC4F92E718E.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/BcToJPsiMuMu_inclusive_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/230000/05C316B4-BD3D-CC4E-8BDB-C603259F1016.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/BcToJPsiMuMu_inclusive_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/230000/0D7329CE-CC25-1A4C-848D-CDF63DA314B5.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/BcToJPsiMuMu_inclusive_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/230000/127F4AE1-CE44-6E4D-95A5-C25986F28A1B.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/BcToJPsiMuMu_inclusive_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/230000/17FDD58D-FEAD-5B49-83DD-3A8C1C1A3960.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/BcToJPsiMuMu_inclusive_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/230000/189B54A2-6B0D-ED49-A7FA-A94DF21D41A0.root" --filename=rjpsi_bc_signal --mc --maxevents=-1 --savenontrig

ipython -i -- inspector_rjpsi.py --inputFiles=0443354B-2D3F-CF41-A1F0-0FC4F92E718E.root --filename=rjpsi_bc_signal_small --mc --maxevents=-1 --savenontrig

'''

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
from Bmmm.Analysis.RJpsiBranches import branches, paths, muon_branches, cand_branches, event_branches, bc_branches, jpsi_branches, phi_branches, safe_get
from Bmmm.Analysis.RJpsiCandidate import RJpsiCandidate as Candidate
from Bmmm.Analysis.utils import drop_hlt_version, cutflow, p4_with_mass, masses, compute_mass, is_b_hadron
from Bmmm.Analysis.RJpsiCuts import cuts
from Bmmm.Analysis.Handles import handles, handles_mc
from Bmmm.Analysis.RJPsiGenHistory import * # FIXME! BAD PRACTICE
from Bmmm.Analysis.RJPsiMuonMatcher import match_candidate_muons, save_bc_gen, ROLE

## tighten muon ID
cuts['rjpsi']['mu_id'] = lambda mu : mu.isMediumMuon() and mu.isPFMuon() and mu.isGlobalMuon()

_JPSI_MASS = 3.0969 # GeV

# pre-compute constants used in the hot loop
_JET_MATCH_DR2 = 0.2 ** 2
_GEN_MATCH_DR2 = 0.02 ** 2

# template for candidate-level NaN initialisation: covers muon, cand, bs, jpsi, phi branches
_TRIGGER_KEYS = set(k for p in paths for k in (p, p + '_ps'))
_EVENT_KEYS   = set(event_branches.keys())
_GEN_KEYS     = set(bc_branches.keys())
_CAND_KEYS    = [b for b in branches if b not in _TRIGGER_KEYS and b not in _EVENT_KEYS and b not in _GEN_KEYS]
_CAND_TEMPLATE = dict.fromkeys(_CAND_KEYS, np.nan)

######################################################################################
#####      LOOPER
######################################################################################
def looper(events, options, handles, handles_mc, row_list, start):

    i = 0
    for i, event in enumerate(events, 1):

        if i > options.maxevents:
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

        for to in event.tobjs:
            if to.pt() < 3. or abs(to.eta()) >= 2.6:
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
        for i in range(nmuons):
            for j in range(i + 1, nmuons):
                mu1, mu2 = muons[i], muons[j]
        
                if mu1.charge() + mu2.charge() != 0:
                    continue
                
                for k in range(nmuons):
                    if k == i or k == j:
                        continue
                    
                    mu3 = muons[k]
                    
                    # print(f'numer of muons {nmuons}, jpsi idx {i} {j}, bachelor {k}')
                    
                    cand = Candidate([mu1, mu2], mu3)
                    cutflow['\tcandidates after HLT and 3mu'] += 1

                    if cand.mass()>cuts['rjpsi']['max_mass']:
                        continue
                    cutflow['\tpass 3mu mass cut'] += 1

                    if np.abs(cand.jpsi.mass()-_JPSI_MASS)>cuts['rjpsi']['jpsi_mass_window']:
                        continue
                        
                    cutflow['\tpass jpsi mass cut'] += 1
                    cands.append(cand)
                    
        if len(cands) == 0:
            continue

        event.ncands = len(cands)
        cutflow['at least one cand pass presel'] += 1

        cands.sort(key=lambda x: (abs(x.charge()) == 0, x.pt(), -np.abs(cand.jpsi.mass()-_JPSI_MASS)), reverse=True)
        
        ######################################################################################
        #####      EVENT-LEVEL TOFILL — filled ONCE, shared across all candidates
        ######################################################################################
        event_tofill = {}
        for branch, getter in event_branches.items():
            event_tofill[branch] = getter(event)

        ######################################################################################
        #####      BC MC TRUTH CLASSIFIER
        ######################################################################################
        if options.mc:
            event.bc_code, event.bc_info = classify_bc_event(event.genpr)
            bc = event.bc_info['bc']
            if bc is not None:
                bc.bc_code = event.bc_code
                for branch, getter in bc_branches.items():
                    event_tofill[branch] = safe_get(getter, bc, verbose=options.verbose, name=branch)
        else:
            for branch in bc_branches:
                event_tofill[branch] = np.nan

        ######################################################################################
        #####      FILL ONE ROW PER CANDIDATE
        ######################################################################################
        for icand in cands:

            if options.mc:
                # run the reco<->gen matching once; it tags final_cand.mu1/mu2/mu3
                match_candidate_muons(icand, event.genpr, dr_max=0.04)
            
            icand.compute_vtx_quantities(event.vtx, event.bs)

            # start from NaN for all candidate-level branches
            cand_tofill = _CAND_TEMPLATE.copy()
                        
            # ---- candidate-level quantities -----------------------------------------
            for branch, getter in cand_branches.items():
                cand_tofill[branch] = safe_get(getter, icand, verbose=options.verbose, name=branch)

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

            # ---- merge the three scopes into one flat row ---------------------------
            # trigger_tofill  : filled once per event, same for every candidate
            # event_tofill    : filled once per event, same for every candidate
            # cand_tofill     : filled fresh for each candidate
            row = {**trigger_tofill, **event_tofill, **cand_tofill}
            row_list.append(row)

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

    args    = parser.parse_args()
    options = namedtuple('options', args.__dict__.keys())(*args.__dict__.values())

    if 'txt' in options.inputFiles:
        with open(options.inputFiles) as f:
            files = ['root://cms-xrd-global.cern.ch//' + line for line in f.read().splitlines()]
    elif (',' in options.inputFiles
          or 'cms-xrd-global' in options.inputFiles
          or 't3dcachedb'     in options.inputFiles):
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

    fout      = uproot.recreate(options.destination + '/' + options.filename + '.root')
    row_list  = []
    start     = time()
    mytimestamp = datetime.now().strftime('%Y-%m-%d__%Hh%Mm%Ss')
    print('#### STARTING NOW', mytimestamp)

    ##########################################################################################
    #####      PROCESS EVENTS
    ##########################################################################################
    n_proc_events, cutflow_result = looper(events, options, handles, handles_mc, row_list, start)

    ##########################################################################################
    #####      WRITE TO DISK
    ##########################################################################################
    ntuple = pd.DataFrame(row_list, columns=branches)
    print('\nnumber of selected events', len(ntuple))
    if len(ntuple) > 0:
        fout['tree'] = ntuple
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
