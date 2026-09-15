'''
Branch machinery shared by every channel.

What lives here is the DEFINITION of a quantity, not the layout of any one
ntuple: `npv` must mean len(event.vtx) in the dimuon ntuple exactly as it does
in the RJpsi one, or the two cannot be compared. Each channel still assembles
its own branch list in its own order, because those orders are historical and
changing them would reshuffle existing files for no gain.

Only quantities that are already spelled and computed identically in every
channel belong in event_branches. A channel that spells one differently either
gets harmonised onto the common name or keeps its own definition locally --
what must not happen is two names for one quantity living in two files.
'''

import ROOT
import numpy as np

from Bmmm.Analysis.utils import COV_ELEMENT_NAMES, COV_INDEX_PAIRS, COV_NAN_5X5

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
    'npu'    : lambda ev : ev.pu_at_bx0.getPU_NumInteractions()  if ev.mc else np.nan,
    'nti'    : lambda ev : ev.pu_at_bx0.getTrueNumInteractions() if ev.mc else np.nan,
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
def cov_track(imu):
    '''The track whose covariance the FITS actually used.

    With --covflow (or --cov-scale) installed, Candidate.fit_track memoizes the
    rebuilt track on the muon as _cov_fit_track and hands that to the vertex
    fitters and the IP machinery. bestTrack() still returns the raw one. Any
    branch describing an UNCERTAINTY should come from here, so the ntuple is
    self-consistent: an error bar computed from a covariance no fit ever saw is
    a number with no owner.

    A branch describing a POSITION or a MOMENTUM should stay on bestTrack():
    the two tracks carry identical parameters by construction (track_with_cov
    copies momentum, charge and reference point), so using bestTrack() there
    keeps <obj>_dxy an independent check that nothing moved the trajectory.

    Falls back to bestTrack() when no corrector is installed, which is what
    makes this a no-op for every channel and every job that does not use one.
    '''
    trk = getattr(imu, '_cov_fit_track', None)
    return imu.bestTrack() if trk is None else trk


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
    'dxy_e'          :  lambda imu : cov_track(imu).dxyError(imu.pv.position(), imu.pv.error()),
    'dxy_sig'        :  lambda imu : imu.bestTrack().dxy(imu.pv.position()) / cov_track(imu).dxyError(imu.pv.position(), imu.pv.error()),
    'dz'             :  lambda imu : imu.bestTrack().dz(imu.pv.position()),
    'dz_e'           :  lambda imu : cov_track(imu).dzError(),
    'dz_sig'         :  lambda imu : imu.bestTrack().dz(imu.pv.position()) / cov_track(imu).dzError(),
    'bs_dxy'         :  lambda imu : imu.bestTrack().dxy(imu.bs.position()),
    'bs_dxy_e'       :  lambda imu : cov_track(imu).dxyError(imu.bs.position(), imu.bs.error()),
    'bs_dxy_sig'     :  lambda imu : imu.bestTrack().dxy(imu.bs.position()) / cov_track(imu).dxyError(imu.bs.position(), imu.bs.error()),
    # The same three errors from the RAW covariance. Without a corrector these
    # equal the branches above exactly; with one, the pair is what lets a single
    # file say how much the correction moved the impact-parameter resolution --
    # the cov_* / cov_corr_* pattern, one level up.
    'dxy_e_raw'      :  lambda imu : imu.bestTrack().dxyError(imu.pv.position(), imu.pv.error()),
    'dz_e_raw'       :  lambda imu : imu.bestTrack().dzError(),
    'bs_dxy_e_raw'   :  lambda imu : imu.bestTrack().dxyError(imu.bs.position(), imu.bs.error()),
    'n_pix_hit'      :  lambda imu : imu.bestTrack().hitPattern().numberOfValidPixelHits(),
    'n_pix_b_hit'    :  lambda imu : imu.bestTrack().hitPattern().numberOfValidPixelBarrelHits(),
    'n_pix_e_hit'    :  lambda imu : imu.bestTrack().hitPattern().numberOfValidPixelEndcapHits(),
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

# The covflow-corrected covariance, as <obj>_cov_corr_<par_i>_<par_j>, plus two
# per-track diagnostics. All NaN / False when running without --covflow.
#
# This block exists because the cov_* branches above are, and must remain, RAW:
# they are the input to the measurement, so overwriting them with the corrected
# values would make the ntuple unable to say what was done to it. With both sets
# present a single file answers "what did the correction do to this track",
# which is the whole point of running --covflow in the first place.
#
#   <obj>_covflow_ok    the correction was actually applied to this track.
#                       False means the raw matrix was used for the fits too --
#                       a non-PD input covariance, or a morph failure.
#   <obj>_covflow_zmax  max_i |z_i| of the MC latent. Large values are tracks
#                       sitting where the data flow saw little or nothing, i.e.
#                       where the morph extrapolates. Cut on it offline; the
#                       honest per-component bound from the training run can be
#                       passed in through covflow.json (latent_bounds), in which
#                       case the job also counts how many tracks fall outside.
track_cov_corr_branches = {}
for _name, (_i, _j) in zip(COV_ELEMENT_NAMES, COV_INDEX_PAIRS):
    track_cov_corr_branches['cov_corr_%s' % _name] = (
        lambda iobj, i=_i, j=_j : getattr(iobj, 'cov_corr', COV_NAN_5X5)[i][j])
track_cov_corr_branches['covflow_ok']   = (
    lambda iobj : int(getattr(iobj, 'covflow_ok', False)))
track_cov_corr_branches['covflow_zmax'] = (
    lambda iobj : getattr(iobj, 'covflow_zmax', np.nan))

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
