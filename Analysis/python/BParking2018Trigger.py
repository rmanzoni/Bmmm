'''
2018 B-parking trigger: the ten single-muon parking paths HLT_Mu7_IP4 ...
HLT_Mu12_IP6, shared by every ntuplizer that runs on ParkingBPH1-6.

Moved here from JpsiTkBParking2018Inspector so the J/psi + track (Vcb) and the
R(Ds) ntuplizers use the same path list, the same per-muon matching and the
same fired-path bookkeeping. The J/psi + track ntuple is unchanged by the move.

  paths                       {path: [last L3 filter label]}, taken from
                              MuMuBranches.paths (the single place they are
                              defined; the dimuon ntuplizer uses them too)
  BParking2018TriggerMixin    mixed into a BaseInspector subclass:
      trig_match(...)         >= 1 of the given objects within hlt_dr of an
                              HLT object of ANY fired parking path
      fired_paths(event)      the parking paths that fired, once per event
      path_tags(...)          per object and per path: 1/0 for a fired path,
                              NaN (left to the caller's template) otherwise
      trigger_dr(...)         dR of an object to the closest object of any
                              fired parking path (NaN if none)
'''

import numpy as np
from collections import OrderedDict
from PhysicsTools.HeppyCore.utils.deltar import deltaR, bestMatch

from Bmmm.Analysis.utils import drop_hlt_version
from Bmmm.Analysis.MuMuBranches import paths as _mm_paths

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
        'BParking2018Trigger: MuMuBranches.paths lacks %s. Is '
        'BMMM_MM_HLT_PATHS set in this shell? Unset it for the 2018 parking ntuples.'
        % _missing)

paths = OrderedDict((p, list(_mm_paths[p])) for p in BPARKING2018_PATH_NAMES)


class BParking2018TriggerMixin(object):
    '''Trigger logic of the 2018 parking ntuplizers. Expects self.PATHS to be
    the `paths` above (or a subset with the same structure).'''

    def trig_match(self, icand, cuts, good_tobjs, objects):
        '''>= 1 of `objects` within hlt_dr of an HLT object of any FIRED parking
        path. good_tobjs only holds objects of fired paths, already filtered on
        the path's filter label by the base looper.'''
        for path in self.PATHS:
            for to in good_tobjs.get(path, []):
                for io in objects:
                    if deltaR(io, to) < cuts['hlt_dr']:
                        return True
        return False

    def path_tags(self, objects, event, good_tobjs, hlt_dr):
        '''{(idx, path): 1./0.} for every object (idx counts from 1) and every
        FIRED path; paths that did not fire are absent (the caller's template
        leaves them NaN). The decision is read from TriggerResults, not
        inferred from good_tobjs: a path can fire with none of its objects
        passing the to_pt / to_eta preselection, and that is a 0, not a NaN.'''
        out = {}
        hlt_dr2 = hlt_dr ** 2
        for path in self.fired_paths(event):
            tobjs = good_tobjs.get(path, [])
            label = self.PATHS[path][0]
            for idx, io in enumerate(objects, 1):
                matched = False
                if tobjs:
                    to, dr2 = bestMatch(io, tobjs)
                    matched = (to is not None and dr2 < hlt_dr2 and to.hasFilterLabel(label))
                out[(idx, path)] = float(matched)
        return out

    def trigger_dr(self, obj, good_tobjs):
        '''dR of obj to the closest HLT object of any fired parking path.'''
        tobjs = [to for path in self.PATHS for to in good_tobjs.get(path, [])]
        if not tobjs:
            return np.nan
        _, dr2 = bestMatch(obj, tobjs)
        return np.sqrt(dr2)

    def fired_paths(self, event):
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

    # name kept for the J/psi + track inspector, which called it this way
    _fired_paths = fired_paths
