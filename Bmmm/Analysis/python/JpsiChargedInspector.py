'''
Shared machinery for the J/psi + charged-object ntuplizers.

BaseInspector holds everything that does NOT depend on the nature of the bachelor:
argument parsing, file handling, the incremental (batched) uproot output, and the
per-event skeleton (trigger decision + trigger objects + muon selection + the
event loop + the flush cadence). Each channel is a thin subclass that provides its
branch/cut/candidate modules and overrides the handful of hooks that genuinely
differ:

    MIN_MUONS               how many selected muons are required
    build_candidates(...)   how (dimuon [+ bachelor]) candidates are built
    sort_candidates(...)    the ranking of the candidates within an event
    setup_event_gen(...)    the once-per-event MC gen-truth setup (event block)
    fill_candidate(...)     the per-candidate row (matching, vertexing, filling)

JpsiMuInspector reproduces the original inspector_rjpsi exactly; JpsiTkInspector
adds the track (kaon + pion) reconstruction.
'''

import ROOT
import argparse
import numpy as np
import pandas as pd
import uproot
from time import time
from datetime import datetime, timedelta
from glob import glob
from collections import namedtuple
from itertools import product, combinations
from DataFormats.FWLite import Events, Handle
from PhysicsTools.HeppyCore.utils.deltar import deltaR, bestMatch

from Bmmm.Analysis.utils import drop_hlt_version, cutflow
from Bmmm.Analysis.Handles import handles_mc
from Bmmm.Analysis.Handles import handles
from Bmmm.Analysis.Handles import handles_skim # includes BS constrained vertices

######################################################################################
#####      INCREMENTAL (BATCHED) OUTPUT   (channel-agnostic)
######################################################################################
# Memory footprint of the job is bounded by WRITE_EVERY rows instead of growing
# with the whole file: rows are flushed to the TTree and the buffer is cleared.
WRITE_EVERY  = 50000
INT_BRANCHES = ('run', 'lumi', 'event')

def build_branch_types(branches):
    '''Fixed schema for the TTree: int64 for the event identifiers, float32 for
    everything else. Used by mktree so every extend() call matches it exactly.'''
    return {c: (np.int64 if c in INT_BRANCHES else np.float32) for c in branches}

def rows_to_columns(rows, branches):
    '''list-of-dicts -> {branch: numpy array} with a FIXED dtype per branch, so
    every uproot extend() presents an identical schema.'''
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
    '''Append the buffered rows to the TTree and clear the buffer in place.'''
    if not row_list:
        return
    fout['tree'].extend(rows_to_columns(row_list, branches))
    row_list.clear()


class BaseInspector(object):
    '''Base class of the J/psi + charged-object inspectors. Subclasses set the
    class attributes below and override the hooks.'''

    # ----- channel configuration (set by subclasses) -----
    DESCRIPTION     = 'J/psi + charged-object ntuplizer'
    CHANNEL         = None     # cuts key, e.g. 'jpsi_mu' / 'jpsi_tk'
    CUTS            = None     # the cuts dict
    CANDIDATE       = None     # the Candidate class
    BRANCHES        = None     # flat branch list
    PATHS           = None     # HLT paths dict
    EVENT_BRANCHES  = None     # event-level getters
    CAND_BRANCHES   = None     # candidate-level getters
    MUON_BRANCHES   = None     # per-muon getters
    SAFE_GET        = None     # safe_get helper
    EVENT_GEN_KEYS  = ()       # event-level branches filled by setup_event_gen (e.g. bc_branches)
    MIN_MUONS       = 2        # minimum selected muons

    def __init__(self):
        cuts = self.CUTS[self.CHANNEL]
        self._JET_MATCH_DR2 = cuts['jet_dr'] ** 2
        self._GEN_DR2       = cuts['gen_dr'] ** 2

        self._TRIGGER_KEYS = set(k for p in self.PATHS for k in (p, p + '_ps'))
        self._EVENT_KEYS   = set(self.EVENT_BRANCHES.keys()) | set(self.EVENT_GEN_KEYS)
        self._CAND_KEYS    = [b for b in self.BRANCHES
                              if b not in self._TRIGGER_KEYS and b not in self._EVENT_KEYS]
        self._CAND_TEMPLATE = dict.fromkeys(self._CAND_KEYS, np.nan)

    # ==================================================================
    #  HOOKS  (overridden by the channel subclasses)
    # ==================================================================
    def build_candidates(self, muons, event, options, cuts, good_tobjs):
        raise NotImplementedError

    def sort_candidates(self, cands):
        return cands

    def setup_event_gen(self, event, options, event_tofill):
        '''Fill any event-level gen branches into event_tofill and return a
        per-event gen state (reused by every candidate). Default: nothing.'''
        return None

    def fill_candidate(self, icand, event, options, cuts, gen_state, good_tobjs):
        raise NotImplementedError

    # ==================================================================
    #  SHARED per-candidate helpers
    # ==================================================================
    def fill_muon(self, imu, icand, event, options):
        '''Per-muon housekeeping shared by both channels: attach the PV / BS /
        muon-POG isolation, do the jet match, and return nothing (attributes are
        set on the muon object, read later by MUON_BRANCHES).'''
        imu.pv    = icand.pv
        imu.bs    = icand.bs
        imu.iso03 = imu.pfIsolationR03()
        imu.iso04 = imu.pfIsolationR04()
        jet, dr2 = bestMatch(imu, event.jets)
        if dr2 < self._JET_MATCH_DR2:
            imu.jet = jet

    def trig_match(self, icand, cuts, good_tobjs, objects):
        '''Informational (NOT a cut): >= 2 of `objects` within hlt_dr of a fired
        HLT object.'''
        hlt_objs = good_tobjs.get(cuts['hlt'], [])
        return sum(deltaR(io, to) < cuts['hlt_dr']
                   for io, to in product(objects, hlt_objs)) >= 2

    # ==================================================================
    #  LOOPER  (invariant skeleton)
    # ==================================================================
    def looper(self, events, options, handles, handles_mc, row_list, start, fout, branches):

        cuts  = self.CUTS[self.CHANNEL]
        paths = self.PATHS

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

            # ---- load handles ---------------------------------------------------
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

            ##################################################################
            #####      TRIGGERS -- filled ONCE per event
            ##################################################################
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

            ##################################################################
            #####      TRIGGER OBJECTS -- built ONCE per event
            ##################################################################
            good_tobjs      = {key: []    for key in paths}
            good_tobjs_seen = {key: set() for key in paths}

            if hlt_passed:
                for to in event.tobjs:
                    if to.pt() < cuts['to_pt'] or abs(to.eta()) >= cuts['to_eta']:
                        continue
                    to.unpackNamesAndLabels(event.object(), event.trg_res)
                    for k, v in paths.items():
                        if trigger_tofill[k] != 1:
                            continue
                        for ilabel in v:
                            if to.hasFilterLabel(ilabel) and id(to) not in good_tobjs_seen[k]:
                                good_tobjs[k].append(to)
                                good_tobjs_seen[k].add(id(to))

            ##################################################################
            #####      MUON SELECTION  (shared)
            ##################################################################
            muons = [mu for mu in event.muons
                     if mu.pt()                    >  cuts['mu_pt']
                     and abs(mu.eta())             <  cuts['mu_eta']
                     and cuts['mu_id'](mu)
                     and abs(mu.bestTrack().dxy()) <  cuts['mu_dxy']]
            muons.sort(key=lambda x: x.pt(), reverse=True)

            if len(muons) < self.MIN_MUONS:
                continue

            cutflow['at least %d muons' % self.MIN_MUONS] += 1

            ##################################################################
            #####      BUILD AND SELECT CANDIDATES  (hook)
            ##################################################################
            cands = self.build_candidates(muons, event, options, cuts, good_tobjs)

            if len(cands) == 0:
                continue

            event.ncands = len(cands)
            cutflow['at least one cand pass presel'] += 1

            cands = self.sort_candidates(cands)

            ##################################################################
            #####      EVENT-LEVEL TOFILL -- filled ONCE
            ##################################################################
            event_tofill = {}
            for branch, getter in self.EVENT_BRANCHES.items():
                event_tofill[branch] = getter(event)

            gen_state = self.setup_event_gen(event, options, event_tofill)

            ##################################################################
            #####      FILL ONE ROW PER CANDIDATE  (hook)
            ##################################################################
            for icand in cands:
                cand_tofill = self.fill_candidate(icand, event, options, cuts, gen_state, good_tobjs)
                row = {**trigger_tofill, **event_tofill, **cand_tofill}
                row_list.append(row)

            if len(row_list) >= WRITE_EVERY:
                flush(fout, row_list, branches)

        flush(fout, row_list, branches)
        return i, cutflow

    # ==================================================================
    #  MAIN
    # ==================================================================
    def parse_args(self):
        parser = argparse.ArgumentParser(description=self.DESCRIPTION)
        parser.add_argument('--inputFiles',  dest='inputFiles',  required=True,          type=str)
        parser.add_argument('--verbose',     dest='verbose',     action='store_true')
        parser.add_argument('--destination', dest='destination', default='./',           type=str)
        parser.add_argument('--filename',    dest='filename',    required=True,          type=str)
        parser.add_argument('--maxevents',   dest='maxevents',   default=-1,             type=int)
        parser.add_argument('--mc',          dest='mc',          action='store_true')
        parser.add_argument('--logfreq',     dest='logfreq',     default=100,            type=int)
        parser.add_argument('--logger',      dest='logger',      default='',             type=str)
        parser.add_argument('--savenontrig', dest='savenontrig', action='store_true')
        parser.add_argument('--useskim',     dest='useskim',     action='store_true')
        parser.add_argument('--maxfiles',    dest='maxfiles',    default=-1,             type=int)
        parser.add_argument('--redirector',  dest='redirector',  default='root://cms-xrd-global.cern.ch//', type=str)
        args = parser.parse_args()
        return namedtuple('options', args.__dict__.keys())(*args.__dict__.values())

    def main(self):
        options = self.parse_args()

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

        branches = self.BRANCHES
        fout = uproot.recreate(options.destination + '/' + options.filename + '.root',
                               compression=uproot.ZSTD(5))
        fout.mktree('tree', build_branch_types(branches))

        row_list    = []
        start       = time()
        mytimestamp = datetime.now().strftime('%Y-%m-%d__%Hh%Mm%Ss')
        print('#### STARTING NOW', mytimestamp)

        myhandles = handles_skim if options.useskim else handles

        n_proc_events, cutflow_result = self.looper(
            events, options, myhandles, handles_mc, row_list, start, fout, branches)

        flush(fout, row_list, branches)

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
