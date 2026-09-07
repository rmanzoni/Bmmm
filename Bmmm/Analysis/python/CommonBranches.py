'''
Branch machinery shared by every channel.

What lives here is the DEFINITION of a quantity, not the layout of any one
ntuple: `npv` must mean len(event.vtx) in the dimuon ntuple exactly as it does
in the RJpsi one, or the two cannot be compared. Each channel still assembles
its own branch list in its own order, because those orders are historical and
changing them would reshuffle existing files for no gain.

Only quantities that are already spelled and computed identically in every
channel belong in event_branches. Anything a channel spells differently
(the dimuon ntuple's n_pu / n_true_int against the RJpsi npu / nti) stays with
that channel until the names are deliberately harmonised.
'''

import ROOT
import numpy as np

from Bmmm.Analysis.utils import COV_ELEMENT_NAMES, COV_INDEX_PAIRS

##########################################################################################
#####      EVENT-LEVEL QUANTITIES, COMMON TO ALL CHANNELS
##########################################################################################
# Getters take the FWLite event, decorated by the inspector with the products it
# read (ev.vtx, ev.bs, ...) and with ev.ncands, the number of candidates the
# event yielded before the best one was picked.
event_branches = {
    'run'    : lambda ev : ev.eventAuxiliary().run()             ,
    'lumi'   : lambda ev : ev.eventAuxiliary().luminosityBlock() ,
    'event'  : lambda ev : ev.eventAuxiliary().event()           ,
    'ncands' : lambda ev : ev.ncands                             ,
    'npv'    : lambda ev : len(ev.vtx)                           ,
    'bs_x0'  : lambda ev : ev.bs.x0()                            ,
    'bs_y0'  : lambda ev : ev.bs.y0()                            ,
    'bs_z0'  : lambda ev : ev.bs.z0()                            ,
}


##########################################################################################
#####      PER-MUON QUANTITIES, COMMON TO ALL CHANNELS
##########################################################################################
# Getters take a pat::Muon decorated by the inspector with the per-candidate
# context they need: imu.pv, imu.bs, imu.iso03, imu.iso04, and optionally
# imu.jet / imu.gen_match. A channel that does not set one of those still gets a
# NaN rather than a crash, through safe_get.
#
# Every channel's per-muon block is built from this one, each in its own order
# and with its own additions -- the dimuon ntuple happens to use exactly this
# list, the RJpsi one interleaves the refitted-J/psi rf_* quantities and appends
# the gen bookkeeping.
muon_branches = {
    'pt'             :  lambda imu : imu.pt()                            ,
    'eta'            :  lambda imu : imu.eta()                           , 
    'phi'            :  lambda imu : imu.phi()                           ,
    'e'              :  lambda imu : imu.energy()                        ,
    'mass'           :  lambda imu : imu.mass()                          ,
    'charge'         :  lambda imu : imu.charge()                        ,
    'id_loose'       :  lambda imu : imu.isLooseMuon()                   ,
    'id_soft'        :  lambda imu : imu.isSoftMuon(imu.pv)              ,
    'id_medium'      :  lambda imu : imu.isMediumMuon()                  ,
    'id_tight'       :  lambda imu : imu.isTightMuon(imu.pv)             ,
    'id_soft_mva_raw':  lambda imu : imu.softMvaValue()                  ,
    'id_soft_mva'    :  lambda imu : imu.passed(ROOT.reco.Muon.SoftMvaId),
    'id_pf'          :  lambda imu : imu.isPFMuon()                      ,
    'id_global'      :  lambda imu : imu.isGlobalMuon()                  ,
    'id_tracker'     :  lambda imu : imu.isTrackerMuon()                 ,
    'id_standalone'  :  lambda imu : imu.isStandAloneMuon()              ,
    'pfiso03'        :  lambda imu : (imu.iso03.sumChargedHadronPt + max(imu.iso03.sumNeutralHadronEt + imu.iso03.sumPhotonEt - 0.5 * imu.iso03.sumPUPt, 0.0))           ,
    'pfiso04'        :  lambda imu : (imu.iso04.sumChargedHadronPt + max(imu.iso04.sumNeutralHadronEt + imu.iso04.sumPhotonEt - 0.5 * imu.iso04.sumPUPt, 0.0))           ,
    'pfreliso03'     :  lambda imu : (imu.iso03.sumChargedHadronPt + max(imu.iso03.sumNeutralHadronEt + imu.iso03.sumPhotonEt - 0.5 * imu.iso03.sumPUPt, 0.0)) / imu.pt(),
    'pfreliso04'     :  lambda imu : (imu.iso04.sumChargedHadronPt + max(imu.iso04.sumNeutralHadronEt + imu.iso04.sumPhotonEt - 0.5 * imu.iso04.sumPUPt, 0.0)) / imu.pt(),
#     'rf_pfreliso03'  :  lambda imu : (imu.iso03.sumChargedHadronPt + max(imu.iso03.sumNeutralHadronEt + imu.iso03.sumPhotonEt - 0.5 * imu.iso03.sumPUPt, 0.0)) / imu.rfp4.pt(),
#     'rf_pfreliso04'  :  lambda imu : (imu.iso04.sumChargedHadronPt + max(imu.iso04.sumNeutralHadronEt + imu.iso04.sumPhotonEt - 0.5 * imu.iso04.sumPUPt, 0.0)) / imu.rfp4.pt(),
    'pfiso03_ch'     :  lambda imu : imu.iso03.sumChargedHadronPt  ,
    'pfiso03_cp'     :  lambda imu : imu.iso03.sumChargedParticlePt,
    'pfiso03_nh'     :  lambda imu : imu.iso03.sumNeutralHadronEt  ,
    'pfiso03_ph'     :  lambda imu : imu.iso03.sumPhotonEt         ,
    'pfiso03_pu'     :  lambda imu : imu.iso03.sumPUPt             ,
    'pfiso04_ch'     :  lambda imu : imu.iso04.sumChargedHadronPt  ,
    'pfiso04_cp'     :  lambda imu : imu.iso04.sumChargedParticlePt,
    'pfiso04_nh'     :  lambda imu : imu.iso04.sumNeutralHadronEt  ,
    'pfiso04_ph'     :  lambda imu : imu.iso04.sumPhotonEt         ,
    'pfiso04_pu'     :  lambda imu : imu.iso04.sumPUPt             ,
    'dxy'            :  lambda imu : imu.bestTrack().dxy(imu.pv.position()),
    'dxy_e'          :  lambda imu : imu.bestTrack().dxyError(imu.pv.position(), imu.pv.error()),
    'dxy_sig'        :  lambda imu : imu.bestTrack().dxy(imu.pv.position()) / imu.bestTrack().dxyError(imu.pv.position(), imu.pv.error()),
    'dz'             :  lambda imu : imu.bestTrack().dz(imu.pv.position()),
    'dz_e'           :  lambda imu : imu.bestTrack().dzError(),
    'dz_sig'         :  lambda imu : imu.bestTrack().dz(imu.pv.position()) / imu.bestTrack().dzError(),
    'bs_dxy'         :  lambda imu : imu.bestTrack().dxy(imu.bs.position()),
    'bs_dxy_e'       :  lambda imu : imu.bestTrack().dxyError(imu.bs.position(), imu.bs.error()),
    'bs_dxy_sig'     :  lambda imu : imu.bestTrack().dxy(imu.bs.position()) / imu.bestTrack().dxyError(imu.bs.position(), imu.bs.error()),
#    'rf_dxy'         :  lambda imu : imu.rf_track.dxy(imu.pv.position()),
#    'rf_dxy_e'       :  lambda imu : imu.rf_track.dxyError(imu.pv.position(), imu.pv.error()),
#    'rf_dxy_sig'     :  lambda imu : imu.rf_track.dxy(imu.pv.position()) / imu.rf_track.dxyError(imu.pv.position(), imu.pv.error()),
#    'rf_dz'          :  lambda imu : imu.rf_track.dz(imu.pv.position()),
#    'rf_dz_e'        :  lambda imu : imu.rf_track.dzError(),
#    'rf_dz_sig'      :  lambda imu : imu.rf_track.dz(imu.pv.position()) / imu.rf_track.dzError(),
#    'rf_bs_dxy'      :  lambda imu : imu.rf_track.dxy(imu.bs.position()),
#    'rf_bs_dxy_e'    :  lambda imu : imu.rf_track.dxyError(imu.bs.position(), imu.bs.error()),
#    'rf_bs_dxy_sig'  :  lambda imu : imu.rf_track.dxy(imu.bs.position()) / imu.rf_track.dxyError(imu.bs.position(), imu.bs.error()),
    'cov_pos_def'    :  lambda imu : imu.is_cov_pos_def,
    'jet_pt'         :  lambda imu : imu.jet.pt()      if hasattr(imu, 'jet') else np.nan,
    'jet_eta'        :  lambda imu : imu.jet.eta()     if hasattr(imu, 'jet') else np.nan,
    'jet_phi'        :  lambda imu : imu.jet.phi()     if hasattr(imu, 'jet') else np.nan,
    'jet_e'          :  lambda imu : imu.jet.energy()  if hasattr(imu, 'jet') else np.nan,
    'gen_pt'         :  lambda imu : imu.gen_match.pt()     if hasattr(imu, 'gen_match') else np.nan,
    'gen_eta'        :  lambda imu : imu.gen_match.eta()    if hasattr(imu, 'gen_match') else np.nan,
    'gen_phi'        :  lambda imu : imu.gen_match.phi()    if hasattr(imu, 'gen_match') else np.nan,
    'gen_e'          :  lambda imu : imu.gen_match.energy() if hasattr(imu, 'gen_match') else np.nan,
    'gen_pdgid'      :  lambda imu : imu.gen_match.pdgId()  if hasattr(imu, 'gen_match') else np.nan,


}

# the 15 independent elements of the track covariance, as <obj>_cov_<par_i>_<par_j>
track_cov_branches = {}
for _name, (_i, _j) in zip(COV_ELEMENT_NAMES, COV_INDEX_PAIRS):
    track_cov_branches['cov_%s' % _name] = (lambda iobj, i=_i, j=_j : iobj.cov[i][j])

##########################################################################################
#####      FILLING
##########################################################################################

def safe_get(getter, cand, default=np.nan, verbose=False, name=None):
    '''Apply getter(cand); on any failure return default instead of crashing.

    A getter is allowed to fail: a vertex fit that did not converge, a gen match
    that does not exist, a collection absent in data. The branch is then the
    default (NaN) rather than the job dying thousands of events in. Pass
    verbose=True while developing to see what is failing and why -- silence is
    the price of the safety, so do not run a new schema blind.
    '''
    try:
        return getter(cand)
    except Exception as exc:
        if verbose:
            label = name if name is not None else getattr(getter, '__name__', repr(getter))
            print('[safe_get] %r failed on %s: %s: %s' % (
                  label, type(cand).__name__, type(exc).__name__, exc))
        return default
