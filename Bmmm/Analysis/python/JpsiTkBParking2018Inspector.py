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
'''

import numpy as np

from Bmmm.Analysis.JpsiTkInspector import JpsiTkInspector
from Bmmm.Analysis.JpsiTkBranches import branches as _run3_branches
from Bmmm.Analysis.JpsiTkBranches import paths as _run3_paths
# the parking path list, the matching and the fired-path bookkeeping live in
# BParking2018Trigger, shared with the R(Ds) ntuplizer; BPARKING2018_PATH_NAMES
# and paths are re-exported here for existing imports
from Bmmm.Analysis.BParking2018Trigger import (
    BPARKING2018_PATH_NAMES, paths, BParking2018TriggerMixin,
)

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


class JpsiTkBParking2018Inspector(BParking2018TriggerMixin, JpsiTkInspector):

    DESCRIPTION = ('J/psi + track (B+ -> J/psi K+ / Bc+ -> J/psi pi+) ntuplizer, '
                   '2018 B-parking (HLT_Mu*_IP*)')
    PATHS       = paths
    BRANCHES    = branches

    # trig_match: BParking2018TriggerMixin (>= 1 J/psi muon matched to an
    # object of any fired parking path; informational, not a cut)

    def fill_candidate(self, icand, event, options, cuts, gen_state, good_tobjs):
        cand_tofill = super(JpsiTkBParking2018Inspector, self).fill_candidate(
            icand, event, options, cuts, gen_state, good_tobjs)

        # per-muon, per-path tag: 1/0 for a FIRED path, NaN (the template) for a
        # path that did not fire
        tags = self.path_tags(icand.jpsi_muons, event, good_tobjs, cuts['hlt_dr'])
        for (idx, path), val in tags.items():
            cand_tofill['mu%d_%s_tag' % (idx, path)] = val
        return cand_tofill
