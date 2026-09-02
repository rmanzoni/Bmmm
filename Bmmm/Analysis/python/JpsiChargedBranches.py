import ROOT
import numpy as np

# Shared branch definitions for the J/psi + charged-object ntuples. These are
# channel-agnostic and imported verbatim by BOTH JpsiMuBranches (J/psi mu) and
# JpsiTkBranches (J/psi + track), so the event block, the per-muon block, the Bc
# gen-truth block, the J/psi gen block, the HLT paths and safe_get are identical
# across the two channels.

event_branches = {
    'run'     : lambda ev : ev.eventAuxiliary().run()               ,
    'lumi'    : lambda ev : ev.eventAuxiliary().luminosityBlock()   ,
    'event'   : lambda ev : ev.eventAuxiliary().event()             ,

    'ncands'  : lambda ev : ev.ncands                               ,

#     'qscale'  : lambda ev : ev.genInfo.qScale()                     ,
    'npv'     : lambda ev : len(ev.vtx)                             ,
    'npu'     : lambda ev : ev.pu_at_bx0.getPU_NumInteractions()  if ev.mc else np.nan,
    'nti'     : lambda ev : ev.pu_at_bx0.getTrueNumInteractions() if ev.mc else np.nan,

    'bs_x0'   : lambda ev : ev.bs.x0()                              ,
    'bs_x0e'  : lambda ev : ev.bs.x0Error()                         ,
    'bs_y0'   : lambda ev : ev.bs.y0()                              ,
    'bs_y0e'  : lambda ev : ev.bs.y0Error()                         ,
    'bs_z0'   : lambda ev : ev.bs.z0()                              ,
    'bs_z0e'  : lambda ev : ev.bs.z0Error()                         ,
}

muon_branches = {
    'pt'             :  lambda imu : imu.pt()                            ,
    'eta'            :  lambda imu : imu.eta()                           , 
    'phi'            :  lambda imu : imu.phi()                           ,
    'e'              :  lambda imu : imu.energy()                        ,
    'rf_pt'          :  lambda imu : imu.jpsi_rfp4.pt()                  ,
    'rf_eta'         :  lambda imu : imu.jpsi_rfp4.eta()                 , 
    'rf_phi'         :  lambda imu : imu.jpsi_rfp4.phi()                 ,
    'rf_e'           :  lambda imu : imu.jpsi_rfp4.energy()             ,
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
    'gen_charge'     :  lambda imu : imu.gen_match.charge() if hasattr(imu, 'gen_match') else np.nan,

    'gen_role'       :  lambda imu : imu.gen_role,
    'gen_dr'         :  lambda imu : imu.gen_dr  ,

}

bc_branches = {
    'gen_bc_decay'      :  lambda ib : ib.bc_code    ,
    'gen_bc_q2'         :  lambda ib : ib.q2         ,
    'gen_bc_m_miss2'    :  lambda ib : ib.m_miss2    ,
    'gen_bc_m_miss2_vis':  lambda ib : ib.m_miss2_vis,
    'gen_bc_e_mu_bc'    :  lambda ib : ib.e_mu_bc    ,
    'gen_bc_e_mu_jpsi'  :  lambda ib : ib.e_mu_jpsi  , # helicity angles
    'gen_bc_cos_theta_v':  lambda ib : ib.cos_theta_v, # helicity angles
    'gen_bc_cos_theta_l':  lambda ib : ib.cos_theta_l, # helicity angles
    'gen_bc_chi'        :  lambda ib : ib.chi        , # helicity angles

    'gen_b_e'      :  lambda ib : ib.energy(),
    'gen_b_pt'     :  lambda ib : ib.pt()    ,
    'gen_b_eta'    :  lambda ib : ib.eta()   , 
    'gen_b_phi'    :  lambda ib : ib.phi()   ,
    'gen_b_mass'   :  lambda ib : ib.mass()  ,
    'gen_b_pdgid'  :  lambda ib : ib.pdgId() ,
    'gen_b_charge' :  lambda ib : ib.charge(),

    'gen_b_beta'   :  lambda ib : ib.p4().Beta(),
    'gen_b_gamma'  :  lambda ib : ib.p4().Gamma(),
    'gen_b_ct'     :  lambda ib : np.sqrt( (ib.daughter(0).vx() - ib.vx())**2 + (ib.daughter(0).vy() - ib.vy())**2 + (ib.daughter(0).vz() - ib.vz())**2 )/ib.p4().Beta()/ib.p4().Gamma(),

    'gen_pv_x'     :  lambda ib : ib.vx()    ,
    'gen_pv_y'     :  lambda ib : ib.vy()    ,
    'gen_pv_z'     :  lambda ib : ib.vz()    ,

    'gen_sv_x'     :  lambda ib : ib.daughter(0).vx(),
    'gen_sv_y'     :  lambda ib : ib.daughter(0).vy(),
    'gen_sv_z'     :  lambda ib : ib.daughter(0).vz(),

    'gen_lxy'      :  lambda ib : np.sqrt( (ib.daughter(0).vx() - ib.vx())**2 + (ib.daughter(0).vy() - ib.vy())**2 ) ,
    'gen_lxyz'     :  lambda ib : np.sqrt( (ib.daughter(0).vx() - ib.vx())**2 + (ib.daughter(0).vy() - ib.vy())**2 + (ib.daughter(0).vz() - ib.vz())**2 ) ,
}

jpsi_branches = {
    'gen_jpsi_pt'     :  lambda ib : ib.pt()    ,
    'gen_jpsi_eta'    :  lambda ib : ib.eta()   , 
    'gen_jpsi_phi'    :  lambda ib : ib.phi()   ,
    'gen_jpsi_e'      :  lambda ib : ib.energy(),
    'gen_jpsi_mass'   :  lambda ib : ib.mass()  ,
    'gen_jpsi_pdgid'  :  lambda ib : ib.pdgId() ,
    'gen_jpsi_charge' :  lambda ib : ib.charge(),
}

paths = {}
paths['HLT_DoubleMu4_3_LowMass'] = ['hltDisplacedmumuFilterDoubleMu43LowMass', 'hltDisplacedmumuFilterDoubleMu43LowMass']

##########################################################################################
def safe_get(getter, cand, default=np.nan, verbose=False, name=None):
    '''Apply getter(cand); on any failure return default instead of crashing.'''
    try:
        return getter(cand)
    except Exception as exc:
        if verbose:
            label = name if name is not None else getattr(getter, '__name__', repr(getter))
            print('[safe_get] %r failed on %s: %s: %s' % (
                  label, type(cand).__name__, type(exc).__name__, exc))
        return default
