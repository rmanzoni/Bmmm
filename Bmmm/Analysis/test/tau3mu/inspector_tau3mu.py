'''
Displaced tau -> 3mu ntuplizer  (Ds -> tau nu , tau -> mu a , a -> mu mu ; a long-lived, pdgId 9900015)

Follows the rjpsi inspector philosophy:
  - one flat row per candidate, scalar branches streamed to a TTree in batches
  - mu1, mu2 = the displaced opposite-sign pair (a -> mu mu, fires HLT_DoubleMu4_3_LowMass)
  - mu3      = the bachelor muon (tau -> mu a)
In ADDITION, every candidate carries a variable-length list of ALL the PF
candidates in a R=0.4 cone around the 3mu (tau) direction, written as jagged
awkward branches (pf_*): their p4, impact parameters and pdgId.

The 3mu vertex is fitted sequentially: first the displaced OS pair (the a),
then the a-mother together with the bachelor muon in a DIFFERENT, upstream
vertex (the tau vertex), itself separated from the PV (Tau3MuKinVtxFitter).

Example:
ipython -i -- inspector_tau3mu.py --inputFiles=tau3mu_displaced_1GeV_ctau1mm-RunIII2024Summer24MiniAODv6-00065_9.root --filename=tau3mu_signal --mc --maxevents=-1
ipython -i -- inspector_tau3mu.py --inputFiles="root://cms-xrd-global.cern.ch///store/data/.../file.root" --filename=data_2024 --maxevents=-1
'''

import os
import sys
import ROOT
import argparse
import numpy as np
import pandas as pd
import uproot
import awkward as ak
from time import time
from datetime import datetime, timedelta
from glob import glob
from collections import namedtuple
from DataFormats.FWLite import Events, Handle
from PhysicsTools.HeppyCore.utils.deltar import deltaR

from Bmmm.Analysis.Tau3MuBranches import (branches, paths, muon_branches, cand_branches,
                                          event_branches, gen_branches, pf_fields, PF_BRANCH,
                                          pf_branch_type, safe_get)
from Bmmm.Analysis.Tau3MuCandidate import Tau3MuCandidate as Candidate
from Bmmm.Analysis.Tau3MuCuts import cuts
from Bmmm.Analysis.Tau3MuHandles import handles, handles_mc
from Bmmm.Analysis.Tau3MuGenHistory import Tau3MuGenDecay, match_candidate_muons
from Bmmm.Analysis.utils import drop_hlt_version, cutflow

# template for candidate-level NaN initialisation (muon + cand branches only)
_TRIGGER_KEYS  = set(k for p in paths for k in (p, p + '_ps'))
_EVENT_KEYS    = set(event_branches.keys())
_GEN_KEYS      = set(gen_branches.keys())
_CAND_KEYS     = [b for b in branches if b not in _TRIGGER_KEYS and b not in _EVENT_KEYS and b not in _GEN_KEYS]
_CAND_TEMPLATE = dict.fromkeys(_CAND_KEYS, np.nan)

######################################################################################
#####      INCREMENTAL (BATCHED) OUTPUT
######################################################################################
WRITE_EVERY  = 20000
INT_BRANCHES = ('run', 'lumi', 'event')

def build_branch_types(branches):
    '''Fixed schema: int64 for the event identifiers, float32 for the other
    scalars, plus the single jagged PF record branch (one shared 'npf' counter).'''
    types = {c: (np.int64 if c in INT_BRANCHES else np.float32) for c in branches}
    types[PF_BRANCH] = pf_branch_type()
    return types

def rows_to_columns(rows, branches):
    '''list-of-dicts -> {branch: numpy array} with a FIXED dtype per branch.'''
    df  = pd.DataFrame(rows, columns=branches)
    out = {}
    for col in branches:
        s = pd.to_numeric(df[col], errors='coerce')
        if col in INT_BRANCHES:
            out[col] = s.fillna(0).astype(np.int64).to_numpy()
        else:
            out[col] = s.astype(np.float32).to_numpy()
    return out

def pf_to_columns(pf_rows):
    '''list-of-dicts (one per candidate, each {pf_attr: [..per pf cand..]}) -> a
    single jagged record column {'pf': ak.zip(...)} so ROOT writes one shared
    counter (npf) + the pf_<field> leaves.'''
    fields = {}
    for fname, (attr, dt) in pf_fields.items():
        sublists      = [pr[attr] for pr in pf_rows]
        fields[fname] = ak.values_astype(ak.Array(sublists), dt)
    return {PF_BRANCH: ak.zip(fields)}

def flush(fout, row_list, pf_list, branches):
    '''Append the buffered scalar rows AND the jagged pf rows to the TTree and
    clear both buffers in place (kept in lockstep: one pf row per scalar row).'''
    if not row_list:
        return
    cols = rows_to_columns(row_list, branches)
    cols.update(pf_to_columns(pf_list))
    fout['tree'].extend(cols)
    row_list.clear()
    pf_list.clear()

######################################################################################
#####      LOOPER
######################################################################################
def looper(events, options, handles, handles_mc, row_list, pf_list, start, fout, branches):

    i = 0
    for i, event in enumerate(events, 1):

        if i > options.maxevents:
            flush(fout, row_list, pf_list, branches)
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

        ######################################################################################
        #####      TRIGGERS — filled ONCE per event
        ######################################################################################
        trg_names = event.object().triggerNames(event.trg_res)
        _trg_len  = len(trg_names)

        trigger_tofill = {}
        for ipath in paths:
            trigger_tofill[ipath]         = np.nan
            trigger_tofill[ipath + '_ps'] = np.nan

        idx_to_path = {}
        for iname in trg_names.triggerNames():
            iname    = str(iname)
            stripped = drop_hlt_version(iname)
            if stripped in paths:
                idx_to_path[trg_names.triggerIndex(iname)] = stripped

        for idx, ipath in idx_to_path.items():
            accept = int(idx < _trg_len and event.trg_res.accept(idx))
            ps     = event.trg_ps.getPrescaleForIndex(idx)
            trigger_tofill[ipath]         = np.nanmax([trigger_tofill[ipath],         accept])
            trigger_tofill[ipath + '_ps'] = np.nanmax([trigger_tofill[ipath + '_ps'], ps    ])

        hlt_passed = any(trigger_tofill[p] > 0 for p in paths)
        if not (options.savenontrig or hlt_passed):
            continue
        cutflow['pass HLT'] += 1

        ######################################################################################
        #####      TRIGGER OBJECTS — built ONCE per event
        ######################################################################################
        good_tobjs      = {key: []    for key in paths}
        good_tobjs_seen = {key: set() for key in paths}
        if hlt_passed:
            for to in event.tobjs:
                if to.pt() < cuts['tau3mu']['to_pt'] or abs(to.eta()) >= cuts['tau3mu']['to_eta']:
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
                 if mu.pt()                    >  cuts['tau3mu']['mu_pt']
                 and abs(mu.eta())             <  cuts['tau3mu']['mu_eta']
                 and cuts['tau3mu']['mu_id'](mu)
                 and abs(mu.bestTrack().dxy()) <  cuts['tau3mu']['mu_dxy']]
        muons.sort(key=lambda x: x.pt(), reverse=True)

        if len(muons) < 3:
            continue
        cutflow['at least 3 muons'] += 1

        ######################################################################################
        #####      BUILD AND SELECT 3MU CANDIDATES  (OS displaced pair + bachelor)
        ######################################################################################
        cands  = []
        nmuons = len(muons)
        for ii in range(nmuons):
            for jj in range(ii + 1, nmuons):
                mu1, mu2 = muons[ii], muons[jj]

                # the a -> mu mu pair is opposite-sign
                if mu1.charge() + mu2.charge() != 0:
                    continue
                # both fire the displaced low-mass dimuon trigger
                if mu1.pt() < cuts['tau3mu']['tight_mu_pt'] or mu2.pt() < cuts['tau3mu']['tight_mu_pt']:
                    continue
                # loose upper bound on m(mu mu) (m(a) is the search variable)
                if (mu1.p4() + mu2.p4()).mass() > cuts['tau3mu']['pair_max_mass']:
                    continue
                cutflow['\tpass OS displaced pair presel'] += 1

                for kk in range(nmuons):
                    if kk == ii or kk == jj:
                        continue
                    mu3  = muons[kk]
                    cand = Candidate([mu1, mu2], mu3)

                    m3 = cand.mass()
                    if m3 < cuts['tau3mu']['min_3mu_mass'] or m3 > cuts['tau3mu']['max_3mu_mass']:
                        continue
                    cutflow['\tpass 3mu (tau) mass window'] += 1
                    cands.append(cand)

        if len(cands) == 0:
            continue

        event.ncands = len(cands)
        cutflow['at least one cand pass presel'] += 1

        # most displaced-friendly first: trigger-friendly hard pair, tau-mass closeness
        cands.sort(key=lambda c: (c.a.pt(), -abs(c.mass() - 1.77686)), reverse=True)

        ######################################################################################
        #####      EVENT-LEVEL TOFILL (shared across all candidates) + GEN TRUTH
        ######################################################################################
        event_tofill = {}
        for branch, getter in event_branches.items():
            event_tofill[branch] = getter(event)

        gen_info = None
        if options.mc:
            gen_info = Tau3MuGenDecay.from_genparticles(event.genpr)
            if gen_info is not None:
                for branch, getter in gen_branches.items():
                    event_tofill[branch] = safe_get(getter, gen_info, verbose=options.verbose, name=branch)
            else:
                for branch in gen_branches:
                    event_tofill[branch] = np.nan
        else:
            for branch in gen_branches:
                event_tofill[branch] = np.nan

        ######################################################################################
        #####      FILL ONE ROW PER CANDIDATE
        ######################################################################################
        for icand in cands:

            if options.mc and gen_info is not None:
                match_candidate_muons(icand, event.genpr, dr_max=cuts['tau3mu']['gen_dr'], info=gen_info)

            # trigger matching (informational): the two a-muons matched to fired
            # HLT objects of the displaced low-mass dimuon path
            hlt_objs = good_tobjs.get(cuts['tau3mu']['hlt'], [])
            icand.trig_match = sum(
                any(deltaR(imu.eta(), imu.phi(), to.eta(), to.phi()) < cuts['tau3mu']['hlt_dr']
                    for to in hlt_objs)
                for imu in icand.a_muons
            ) >= 2

            # sequential vertex fit + PV refit (signal muons removed)
            icand.compute_vtx_quantities(event.vtx, event.bs, event.pf, event.ltrk)
            # the R=0.4 PF-candidate cone around the 3mu direction
            icand.compute_pf_cone(event.pf,
                                  cone_dr=cuts['tau3mu']['pf_cone_dr'],
                                  min_pt =cuts['tau3mu']['pf_min_pt'])

            cand_tofill = _CAND_TEMPLATE.copy()

            # mu1, mu2 = a (OS) pair ; mu3 = bachelor
            muons_dict = {1: lambda c: c.a_muons[0],
                          2: lambda c: c.a_muons[1],
                          3: lambda c: c.mu}
            for idx in range(1, 4):
                imu       = muons_dict[idx](icand)
                imu.pv    = icand.pv_bs
                imu.bs    = icand.bs
                imu.iso03 = imu.pfIsolationR03()
                imu.iso04 = imu.pfIsolationR04()
                for branch, getter in muon_branches.items():
                    cand_tofill['mu%d_%s' % (idx, branch)] = safe_get(getter, imu, verbose=options.verbose, name=branch)

            # compute W kinematics
            icand.compute_w_kinematics(event.met[0])

            for branch, getter in cand_branches.items():
                cand_tofill[branch] = safe_get(getter, icand, verbose=options.verbose, name=branch)

            # merge the three scalar scopes into one flat row
            row = {**trigger_tofill, **event_tofill, **cand_tofill}
            row_list.append(row)
            
            # the jagged PF cone for THIS candidate (kept in lockstep with row_list)
            pf_list.append({attr: list(getattr(icand, attr)) for _f, (attr, _dt) in pf_fields.items()})
                        
        if len(row_list) >= WRITE_EVERY:
            flush(fout, row_list, pf_list, branches)

    flush(fout, row_list, pf_list, branches)
    return i, cutflow

######################################################################################
#####      MAIN
######################################################################################
def main():

    parser = argparse.ArgumentParser(description='Tau3Mu (displaced) ntuplizer')
    parser.add_argument('--inputFiles',  dest='inputFiles',  required=True,           type=str)
    parser.add_argument('--verbose',     dest='verbose',     action='store_true')
    parser.add_argument('--destination', dest='destination', default='./',            type=str)
    parser.add_argument('--filename',    dest='filename',    required=True,           type=str)
    parser.add_argument('--maxevents',   dest='maxevents',   default=-1,              type=int)
    parser.add_argument('--mc',          dest='mc',          action='store_true')
    parser.add_argument('--logfreq',     dest='logfreq',     default=100,             type=int)
    parser.add_argument('--logger',      dest='logger',      default='',              type=str)
    parser.add_argument('--savenontrig', dest='savenontrig', action='store_true')
    parser.add_argument('--maxfiles',    dest='maxfiles',    default=-1,              type=int)
    parser.add_argument('--redirector',  dest='redirector',  default='root://cms-xrd-global.cern.ch//', type=str)

    args    = parser.parse_args()
    options = namedtuple('options', args.__dict__.keys())(*args.__dict__.values())

    if 'txt' in options.inputFiles:
        with open(options.inputFiles) as f:
            files = [options.redirector + line for line in f.read().splitlines()]
    elif (',' in options.inputFiles
          or 'cms-xrd-global' in options.inputFiles
          or 't3dcachedb'     in options.inputFiles):
        files = options.inputFiles.split(',')
    else:
        files = glob(options.inputFiles)

        good = []
        for f in files:
            tf = None
        
            if os.path.getsize(f) == 0:
                print('WARNING: skipping unreadable/empty file', f)
                continue
        
            tf = ROOT.TFile.Open(f)
        
            if tf and not tf.IsZombie() and tf.Get('Events'):
                good.append(f)
            else:
                print('WARNING: skipping unreadable/invalid ROOT file', f)
        
            if tf:
                tf.Close()
                
        files = good



    if options.maxfiles > 0:
        files = files[:options.maxfiles]

    print('files:', files)

    events  = Events(files)
    options = options._replace(
        maxevents=options.maxevents if options.maxevents >= 0 else events.size()
    )

    fout = uproot.recreate(options.destination + '/' + options.filename + '.root', compression=uproot.ZSTD(5))
    fout.mktree('tree', build_branch_types(branches))

    row_list, pf_list = [], []
    start             = time()
    mytimestamp       = datetime.now().strftime('%Y-%m-%d__%Hh%Mm%Ss')
    print('#### STARTING NOW', mytimestamp)

    n_proc_events, cutflow_result = looper(events, options, handles, handles_mc, row_list, pf_list, start, fout, branches)
    flush(fout, row_list, pf_list, branches)

    n_written = fout['tree'].num_entries
    print('\nnumber of selected candidates', n_written)
    print('\nntuple saved, processed all desired events?',
          (n_proc_events == options.maxevents),
          'processed', n_proc_events, 'maxevents', options.maxevents)

    logger_name = options.logger if len(options.logger) > 0 else 'logger_' + mytimestamp
    with open('%s.txt' % logger_name, 'w') as logger_file:
        for k, v in cutflow_result.items():
            print(k, v, file=logger_file)

    finish = time()
    print('done in %.1f hours' % ((finish - start) / 3600.))

######################################################################################
if __name__ == '__main__':
    main()
