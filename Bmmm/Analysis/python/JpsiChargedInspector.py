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
import uproot
from time import time
from datetime import datetime, timedelta
from glob import glob
from collections import namedtuple
from itertools import product, combinations
from DataFormats.FWLite import Events, Handle
from PhysicsTools.HeppyCore.utils.deltar import deltaR, bestMatch

from Bmmm.Analysis.utils import drop_hlt_version, cutflow, make_cov_scaler, make_cov_corrector, resolve_input_files
from Bmmm.Analysis.HammerFF import make_hammer_session
from Bmmm.Analysis.Handles import handles_mc
from Bmmm.Analysis.Handles import handles      as handles_std   # full MINIAOD collections
from Bmmm.Analysis.Handles import handles_skim                  # SKIM collections (BS-constrained vertices)

# Batched output lives in NtupleWriter, shared with the other channels.
# Re-exported here so existing "from ...JpsiChargedInspector import flush" style
# imports keep working.
from Bmmm.Analysis.NtupleWriter import (
    WRITE_EVERY, INT_BRANCHES, build_branch_types, rows_to_columns, flush,
)


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
    HAMMER          = None     # HammerFF.HammerSession, built by main() when --hammer is given
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
        self._event_keys_checked = False

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

            # covflow, batched over the event's muons in one torch call, BEFORE
            # any candidate exists: a muon shared by several candidates must
            # carry the same corrected covariance in all of them. No-op without
            # --covflow.
            self.CANDIDATE.prime_cov_corrector(muons, event)

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

            # fail loud, once: anything setup_event_gen fills that is also a
            # candidate key would be silently overwritten by the NaN template
            # in the row merge below (this is how the hammer_* block was lost).
            if not self._event_keys_checked:
                clobbered = set(event_tofill) & set(self._CAND_KEYS)
                if clobbered:
                    raise RuntimeError(
                        'setup_event_gen filled %s, which are candidate keys and '
                        'would be overwritten by the NaN candidate template; add '
                        'them to %s.EVENT_GEN_KEYS'
                        % (sorted(clobbered), type(self).__name__))
                self._event_keys_checked = True

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
        parser.add_argument('--skim',        dest='skim',        action='store_true',
                            help='read the SKIM collections (handles_skim) instead of the full MINIAOD handles')
        parser.add_argument('--maxfiles',    dest='maxfiles',    default=-1,             type=int)
        parser.add_argument('--redirector',  dest='redirector',  default='root://cms-xrd-global.cern.ch//', type=str)
        parser.add_argument('--cov-scale',   dest='cov_scale',   default='',             type=str,
                            help='rescale the track covariance before every vertex fit / IP '
                                 'computation, preserving all correlations. Empty (default) = no '
                                 'scaling: the ntuple then carries the RAW cov_* branches, which '
                                 'is what you want while measuring the correction. '
                                 "Accepts 'dxy=1.05,dsz=1.02' (flat), a binned-table JSON "
                                 "(pt_edges/abs_eta_edges/scales), or a correctionlib file, "
                                 "optionally as 'file.json:dxy=<correction name>'. "
                                 'See utils.make_cov_scaler.')
        parser.add_argument('--covflow',     dest='covflow',     default='',             type=str,
                            help='CORRECT the full track covariance with a trained covflow '
                                 'model before every vertex fit / IP computation -- all 5 '
                                 'scales and all 10 correlations, not just the diagonal. '
                                 'Takes the directory a covflow training run wrote '
                                 '(flow_mc.pt, flow_data.pt, scalers.json) plus a covflow.json '
                                 'describing the context and the flow hyperparameters, '
                                 "optionally as 'dir:key=value,...' to override them. "
                                 'MC only, and mutually exclusive with --cov-scale. The raw '
                                 'cov_* branches are untouched; the corrected matrix is written '
                                 'alongside as cov_corr_*. See utils.make_cov_corrector.')
        parser.add_argument('--hammer',      dest='hammer',      default='',             type=str,
                            help='REWEIGHT the Bc -> J/psi l nu form factors from the '
                                 'generated Kiselev model to Harrison-2024 lattice QCD with '
                                 'Hammer, at production time. Takes the FF coefficient card: '
                                 "'default' for Bmmm/Analysis/data/harrison_bglvar.json, or a "
                                 'path to another one. Appending \':nominal\' writes only the '
                                 "central weight, ':allow-stale' writes the eigenvariations "
                                 'from a card whose covariance is not yet validated. MC only, '
                                 'and it needs Hammer importable (see '
                                 'test/rjpsi/hammer/hammer_env.sh). The hammer_* branches are '
                                 'in the schema either way -- without the flag they are NaN, '
                                 'and the same weights can be added afterwards with '
                                 'test/rjpsi/hammer/add_hammer_weights.py.')
        args = parser.parse_args()
        return namedtuple('options', args.__dict__.keys())(*args.__dict__.values())

    def main(self):
        options = self.parse_args()

        # track-covariance rescaling, applied to every track before the vertex
        # fits and the IP computations (JpsiChargedCandidate.fit_track). Off by
        # default, in which case nothing at all changes and the cov_* branches
        # hold the raw covariance -- the input to the measurement.
        cov_scaler = make_cov_scaler(getattr(options, 'cov_scale', ''))
        self.CANDIDATE.set_cov_scaler(cov_scaler)
        if cov_scaler is not None:
            print('#### rescaling the track covariance with %s (%r)'
                  % (type(cov_scaler).__name__, options.cov_scale))

        # covflow: the full-matrix version of the same correction, at the same
        # insertion point. MC only -- the morph is f_data^-1 . f_mc, defined
        # MC -> data, so running it on data is not a closure test, it is wrong.
        # The data file for the comparison is the plain no-flag run, which is
        # bit-for-bit the uncorrected reconstruction.
        covflow_spec = getattr(options, 'covflow', '')
        if covflow_spec and cov_scaler is not None:
            raise RuntimeError('--cov-scale and --covflow are two different '
                               'corrections of the same matrix at the same '
                               'point in the chain. Pass one, not both.')
        if covflow_spec and not options.mc:
            raise RuntimeError('--covflow without --mc: the covflow morph maps '
                               'MC onto data and must not be applied to data.')
        cov_corrector = make_cov_corrector(covflow_spec)
        self.CANDIDATE.set_cov_corrector(cov_corrector)
        if cov_corrector is not None:
            print('#### correcting the track covariance with %s (%r)'
                  % (type(cov_corrector).__name__, covflow_spec))
            print('####   context    %s' % cov_corrector.context_names)
            print('####   features   %s' % cov_corrector.idx)
            print('####   flow cfg   %s' % (cov_corrector.flow_config,))

        # Hammer FF reweighting, off by default. Built here rather than lazily on
        # the first signal event so that a bad card, a missing Hammer install or
        # a non-unitary fit kills the job now instead of after two hours.
        hammer_spec = getattr(options, 'hammer', '')
        if hammer_spec and not options.mc:
            raise RuntimeError('--hammer without --mc: the form-factor weights '
                               'are a generator-truth quantity and exist only for MC.')
        self.HAMMER = make_hammer_session(hammer_spec)

        files = resolve_input_files(options.inputFiles, options.redirector)

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

        handles = handles_skim if options.skim else handles_std
        print('#### reading %s collections' % ('SKIM' if options.skim else 'full-MINIAOD'))

        n_proc_events, cutflow_result = self.looper(
            events, options, handles, handles_mc, row_list, start, fout, branches)

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

        if cov_corrector is not None:
            print(cov_corrector.summary())

        if self.HAMMER is not None:
            print(self.HAMMER.summary())

        finish = time()
        print('done in %.1f hours' % ((finish - start) / 3600.))
