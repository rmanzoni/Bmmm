'''
J/psi + track ntuplizer for the 2018 B-parking data (ParkingBPH1-6, UL2018
MiniAODv2): B+ -> J/psi K+ and Bc+ -> J/psi pi+, exactly as JpsiTkInspector,
with the trigger swapped from the Run 3 dimuon path to the 2018 single-muon
parking paths.

Everything except the trigger is inherited UNCHANGED from JpsiTkInspector (muon
selection, J/psi window, bachelor-track selection, vertexing, both mass
hypotheses, all candidate branches). The Run 3 classes are not touched, so the
Run 3 ntuple schema is unchanged.

What is different, and why
--------------------------
1. PATHS = the ten 2018 parking paths HLT_Mu7_IP4 ... HLT_Mu12_IP6, with their
   last L3 filter labels, taken from MuMuBranches.paths (the single place they
   are defined; the dimuon ntuplizer has used them on the same datasets).

   HLT_DoubleMu4_3_LowMass (the Run 3 path of JpsiTkInspector) is dropped: it
   does not exist in the 2018 menu, and keeping it would make the inspector
   reject every ParkingBPH event.

2. Event selection: the base looper keeps an event if ANY path in PATHS fired
   (without --savenontrig). With PATHS = the parking paths this is the
   requested OR of Mu*_IP*.

3. Trigger matching. The base trig_match asks for >= 2 objects matched to the
   fired dimuon path. The parking paths are SINGLE-muon triggers, so here

     trig_match              = 1 if at least ONE of the two J/psi muons is within
                               hlt_dr of an HLT object of ANY fired parking path
                               (informational, NOT a cut -- as in the base class)
     mu{1,2}_<path>_tag      = 1/0: this J/psi muon is within hlt_dr of an object
                               carrying <path>'s filter label. NaN when <path>
                               did not fire (or is not in that run's menu), the
                               same convention as the dimuon ntuple.

   A candidate with trig_match == 0 comes from an event triggered by a muon
   that is NOT one of the J/psi muons (tag side). Those are kept on purpose:
   the requirement asked for is the OR of the paths, not signal-side triggering.
   Select trig_match == 1 offline if you want the signal-side sample.

   hlt_dr is the J/psi-channel value (cuts['jpsi_tk']['hlt_dr'] = 0.1), not the
   0.15 the dimuon ntuple used for its _tag branches.

4. Candidate mass window: max_mass = 7.0 GeV (jpsi_tk: 7.5), through its own
   cuts key 'jpsi_tk_bparking2018'; every other cut is the jpsi_tk one.
'''

import numpy as np
from copy import copy
from collections import OrderedDict
from PhysicsTools.HeppyCore.utils.deltar import deltaR, bestMatch

from Bmmm.Analysis.utils import drop_hlt_version
from Bmmm.Analysis.JpsiTkInspector import JpsiTkInspector
from Bmmm.Analysis.JpsiTkCuts import cuts as _jpsi_tk_cuts
from Bmmm.Analysis.JpsiTkBranches import branches as _run3_branches
from Bmmm.Analysis.JpsiTkBranches import paths as _run3_paths
from Bmmm.Analysis.MuMuBranches import paths as _mm_paths

##########################################################################################
#####      2018 PARKING PATHS
##########################################################################################
BPARKING2018_PATH_NAMES = (
    'HLT_Mu7_IP4'     ,
    'HLT_Mu8_IP3'     ,
    'HLT_Mu8_IP5'     ,
    'HLT_Mu8_IP6'     ,
    'HLT_Mu8p5_IP3p5' ,
    'HLT_Mu9_IP4'     ,
    'HLT_Mu9_IP5'     ,
    'HLT_Mu9_IP6'     ,
    'HLT_Mu10p5_IP3p5',
    'HLT_Mu12_IP6'    ,
)

# fail loud: MuMuBranches can be restricted with BMMM_MM_HLT_PATHS, and a job
# that silently lost a path would select a different trigger without saying so
_missing = [p for p in BPARKING2018_PATH_NAMES if p not in _mm_paths]
if _missing:
    raise RuntimeError(
        'JpsiTkBParking2018Inspector: MuMuBranches.paths lacks %s. Is '
        'BMMM_MM_HLT_PATHS set in this shell? Unset it for the 2018 parking ntuples.'
        % _missing)

paths = OrderedDict((p, list(_mm_paths[p])) for p in BPARKING2018_PATH_NAMES)

##########################################################################################
#####      BRANCH LIST: Run 3 J/psi + track schema, trigger block swapped
##########################################################################################
_run3_trigger_keys = set(_run3_paths) | set(p + '_ps' for p in _run3_paths)

# per-muon trigger tags, filled in fill_candidate below (candidate keys)
tag_branches = ['mu%d_%s_tag' % (idx, p) for p in paths for idx in (1, 2)]

branches  = [b for b in _run3_branches if b not in _run3_trigger_keys]
branches += tag_branches
branches += list(paths)
branches += [p + '_ps' for p in paths]

if len(branches) != len(set(branches)):
    raise RuntimeError('JpsiTkBParking2018Inspector: duplicated branch names')

##########################################################################################
#####      CUTS: the jpsi_tk selection, with a tighter upper mass edge
##########################################################################################
# A copy under its own key, so the shared cuts dict -- and with it the Run 3
# jpsi_tk selection (max_mass = 7.5, JpsiTkCuts) -- is left untouched.
# max_mass is used only by the J/psi + track mass window in
# JpsiTkInspector.build_candidates: a candidate is kept if EITHER the kaon- or
# the pion-hypothesis mass lies in (min_mass, max_mass) = (4.5, 7.0) GeV.
CHANNEL = 'jpsi_tk_bparking2018'
cuts = dict(_jpsi_tk_cuts)
cuts[CHANNEL] = copy(_jpsi_tk_cuts['jpsi_tk'])
cuts[CHANNEL]['max_mass'] = 7.0   # [GeV]   (jpsi_tk: 7.5, baseline: 10)


class JpsiTkBParking2018Inspector(JpsiTkInspector):

    DESCRIPTION = ('J/psi + track (B+ -> J/psi K+ / Bc+ -> J/psi pi+) ntuplizer, '
                   '2018 B-parking (HLT_Mu*_IP*)')
    CHANNEL     = CHANNEL
    CUTS        = cuts
    PATHS       = paths
    BRANCHES    = branches

    def trig_match(self, icand, cuts, good_tobjs, objects):
        '''>= 1 of `objects` (the two J/psi muons) within hlt_dr of an HLT object
        of any FIRED parking path. good_tobjs only holds objects of fired paths,
        already filtered on the path's filter label by the base looper.'''
        for path in self.PATHS:
            for to in good_tobjs.get(path, []):
                for io in objects:
                    if deltaR(io, to) < cuts['hlt_dr']:
                        return True
        return False

    def fill_candidate(self, icand, event, options, cuts, gen_state, good_tobjs):
        cand_tofill = super(JpsiTkBParking2018Inspector, self).fill_candidate(
            icand, event, options, cuts, gen_state, good_tobjs)

        # per-muon, per-path tag: 1/0 for a FIRED path, NaN (the template) for a
        # path that did not fire. The decision is read from TriggerResults, not
        # inferred from good_tobjs: a path can fire with none of its objects
        # passing the to_pt / to_eta preselection, and that is a 0, not a NaN.
        hlt_dr2 = cuts['hlt_dr'] ** 2
        for path in self._fired_paths(event):
            tobjs = good_tobjs.get(path, [])
            label = self.PATHS[path][0]
            for idx, imu in enumerate(icand.jpsi_muons, 1):
                matched = False
                if tobjs:
                    to, dr2 = bestMatch(imu, tobjs)
                    matched = (to is not None and dr2 < hlt_dr2 and to.hasFilterLabel(label))
                cand_tofill['mu%d_%s_tag' % (idx, path)] = float(matched)
        return cand_tofill

    def _fired_paths(self, event):
        '''Parking paths that fired in this event, computed once per event (the
        FWLite event object is reused across the loop, hence the explicit key).'''
        aux = event.eventAuxiliary()
        key = (aux.run(), aux.luminosityBlock(), aux.event())
        if getattr(self, '_fired_key', None) != key:
            trg_names = event.object().triggerNames(event.trg_res)
            n_results = event.trg_res.size()
            fired = set()
            for iname in trg_names.triggerNames():
                iname = str(iname)
                path  = drop_hlt_version(iname)
                if path not in self.PATHS:
                    continue
                idx = trg_names.triggerIndex(iname)
                if idx < n_results and event.trg_res.accept(idx):
                    fired.add(path)
            self._fired_key   = key
            self._fired_cache = fired
        return self._fired_cache
