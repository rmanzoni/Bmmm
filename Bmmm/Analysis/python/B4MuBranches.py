import ROOT
import numpy as np

branches = [
    'run'               ,
    'lumi'              ,
    'event'             ,
    
    'ncands'            ,
    'npv'               ,
    'npu'               ,
    'nti'               ,

    'bs_x0'             ,
    'bs_y0'             ,
    'bs_z0'             ,
]

cand_branches = {
    'mass'              : lambda cand : cand.mass()            ,
    'mcorr'             : lambda cand : cand.mass_corrected()  ,
    'pt'                : lambda cand : cand.pt()              ,
    'eta'               : lambda cand : cand.eta()             ,
    'phi'               : lambda cand : cand.phi()             ,
    'charge'            : lambda cand : cand.charge()          ,
 
    'dr'                : lambda cand : cand.r()               ,
    'dr_max'            : lambda cand : cand.max_dr()          ,
    'dr_12'             : lambda cand : cand.dr12()            ,
    'dr_13'             : lambda cand : cand.dr13()            ,
    'dr_14'             : lambda cand : cand.dr14()            ,
    'dr_23'             : lambda cand : cand.dr23()            ,
    'dr_24'             : lambda cand : cand.dr24()            ,
    'dr_34'             : lambda cand : cand.dr34()            ,
 
    'charge_12'         : lambda cand : cand.charge12()        ,
    'charge_13'         : lambda cand : cand.charge13()        ,
    'charge_14'         : lambda cand : cand.charge14()        ,
    'charge_23'         : lambda cand : cand.charge23()        ,
    'charge_24'         : lambda cand : cand.charge24()        ,
    'charge_34'         : lambda cand : cand.charge34()        ,
 
    'mass_12'           : lambda cand : cand.mass12()          ,
    'mass_13'           : lambda cand : cand.mass13()          ,
    'mass_14'           : lambda cand : cand.mass14()          ,
    'mass_23'           : lambda cand : cand.mass23()          ,
    'mass_24'           : lambda cand : cand.mass24()          ,
    'mass_34'           : lambda cand : cand.mass34()          ,

    'min_mass'          : lambda cand : min([cand.mass12(), 
                                             cand.mass13(), 
                                             cand.mass14(), 
                                             cand.mass23(), 
                                             cand.mass24(), 
                                             cand.mass34()])   ,
 
    'rf_mass'           : lambda cand : cand.rf_mass()         ,
    'rf_pt'             : lambda cand : cand.rf_pt()           ,
    'rf_eta'            : lambda cand : cand.rf_eta()          ,
    'rf_phi'            : lambda cand : cand.rf_phi()          ,
 
    'rf_dr'             : lambda cand : cand.rf_r()            ,
    'rf_dr_max'         : lambda cand : cand.rf_max_dr()       ,
    'rf_dr_12'          : lambda cand : cand.rf_dr12()         ,
    'rf_dr_13'          : lambda cand : cand.rf_dr13()         ,
    'rf_dr_14'          : lambda cand : cand.rf_dr14()         ,
    'rf_dr_23'          : lambda cand : cand.rf_dr23()         ,
    'rf_dr_24'          : lambda cand : cand.rf_dr24()         ,
    'rf_dr_34'          : lambda cand : cand.rf_dr34()         ,
  
    'rf_mass_12'        : lambda cand : cand.rf_mass12()       ,
    'rf_mass_13'        : lambda cand : cand.rf_mass13()       ,
    'rf_mass_14'        : lambda cand : cand.rf_mass14()       ,
    'rf_mass_23'        : lambda cand : cand.rf_mass23()       ,
    'rf_mass_24'        : lambda cand : cand.rf_mass24()       ,
    'rf_mass_34'        : lambda cand : cand.rf_mass34()       ,

    'rf_min_mass'       : lambda cand : min([cand.rf_mass12(), 
                                             cand.rf_mass13(), 
                                             cand.rf_mass14(), 
                                             cand.rf_mass23(), 
                                             cand.rf_mass24(), 
                                             cand.rf_mass34()]),

    'pv_x'              : lambda cand : cand.pv.position().x() ,
    'pv_y'              : lambda cand : cand.pv.position().y() ,
    'pv_z'              : lambda cand : cand.pv.position().z() ,
 
    'bs_x'              : lambda cand : cand.bs.position().x() ,
    'bs_y'              : lambda cand : cand.bs.position().y() ,

    'vx'                : lambda cand : cand.vtx.position().x(),
    'vy'                : lambda cand : cand.vtx.position().y(),
    'vz'                : lambda cand : cand.vtx.position().z(),
    'vtx_chi2'          : lambda cand : cand.vtx.chi2          ,
    'vtx_prob'          : lambda cand : cand.vtx.prob          ,

    'cos2d'             : lambda cand : cand.vtx.cos           ,
    'rf_cos2d'          : lambda cand : cand.vtx.rf_cos        ,
    'lxy'               : lambda cand : cand.lxy.value()       ,
    'lxy_err'           : lambda cand : cand.lxy.error()       ,
    'lxy_sig'           : lambda cand : cand.lxy.significance(),

    'trig_match'        : lambda cand : cand.trig_match        ,
}


muon_branches = {
    'pt'             :  lambda imu : imu.pt()                            ,
    'eta'            :  lambda imu : imu.eta()                           , 
    'phi'            :  lambda imu : imu.phi()                           ,
    'e'              :  lambda imu : imu.energy()                        ,
    'rf_pt'          :  lambda imu : imu.rfp4.pt()                       ,
    'rf_eta'         :  lambda imu : imu.rfp4.eta()                      , 
    'rf_phi'         :  lambda imu : imu.rfp4.phi()                      ,
    'rf_e'           :  lambda imu : imu.energy()                        ,
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
    'rf_pfreliso03'  :  lambda imu : (imu.iso03.sumChargedHadronPt + max(imu.iso03.sumNeutralHadronEt + imu.iso03.sumPhotonEt - 0.5 * imu.iso03.sumPUPt, 0.0)) / imu.rfp4.pt(),
    'rf_pfreliso04'  :  lambda imu : (imu.iso04.sumChargedHadronPt + max(imu.iso04.sumNeutralHadronEt + imu.iso04.sumPhotonEt - 0.5 * imu.iso04.sumPUPt, 0.0)) / imu.rfp4.pt(),
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
    'rf_dxy'         :  lambda imu : imu.rf_track.dxy(imu.pv.position()),
    'rf_dxy_e'       :  lambda imu : imu.rf_track.dxyError(imu.pv.position(), imu.pv.error()),
    'rf_dxy_sig'     :  lambda imu : imu.rf_track.dxy(imu.pv.position()) / imu.rf_track.dxyError(imu.pv.position(), imu.pv.error()),
    'rf_dz'          :  lambda imu : imu.rf_track.dz(imu.pv.position()),
    'rf_dz_e'        :  lambda imu : imu.rf_track.dzError(),
    'rf_dz_sig'      :  lambda imu : imu.rf_track.dz(imu.pv.position()) / imu.rf_track.dzError(),
    'rf_bs_dxy'      :  lambda imu : imu.rf_track.dxy(imu.bs.position()),
    'rf_bs_dxy_e'    :  lambda imu : imu.rf_track.dxyError(imu.bs.position(), imu.bs.error()),
    'rf_bs_dxy_sig'  :  lambda imu : imu.rf_track.dxy(imu.bs.position()) / imu.rf_track.dxyError(imu.bs.position(), imu.bs.error()),
    'cov_pos_def'    :  lambda imu : imu.is_cov_pos_def,
    'jet_pt'         :  lambda imu : imu.jet.pt()      if hasattr(imu, 'jet') else np.nan,
    'jet_eta'        :  lambda imu : imu.jet.eta()     if hasattr(imu, 'jet') else np.nan,
    'jet_phi'        :  lambda imu : imu.jet.phi()     if hasattr(imu, 'jet') else np.nan,
    'jet_e'          :  lambda imu : imu.jet.energy()  if hasattr(imu, 'jet') else np.nan,
    'gen_pt'         :  lambda imu : imu.genp.pt()     if hasattr(imu, 'genp') else np.nan,
    'gen_eta'        :  lambda imu : imu.genp.eta()    if hasattr(imu, 'genp') else np.nan,
    'gen_phi'        :  lambda imu : imu.genp.phi()    if hasattr(imu, 'genp') else np.nan,
    'gen_e'          :  lambda imu : imu.genp.energy() if hasattr(imu, 'genp') else np.nan,
    'gen_pdgid'      :  lambda imu : imu.genp.pdgId()  if hasattr(imu, 'genp') else np.nan,
}

for idx in [1,2,3,4]:
    for ibr in muon_branches.keys():
        branches.append('mu%d_%s' %(idx, ibr))

for ibranch in cand_branches.keys():
    branches.append(ibranch)

paths = {}
paths['HLT_DoubleMu4_3_LowMass'] = ['hltDisplacedmumuFilterDoubleMu43LowMass', 'hltDisplacedmumuFilterDoubleMu43LowMass']

#paths = [
#    'HLT_Mu7_IP4'     ,
#    'HLT_Mu8_IP3'     ,
#    'HLT_Mu8_IP5'     ,
#    'HLT_Mu8_IP6'     ,
#    'HLT_Mu8p5_IP3p5' ,
#    'HLT_Mu9_IP4'     ,
#    'HLT_Mu9_IP5'     ,
#    'HLT_Mu9_IP6'     ,
#    'HLT_Mu10p5_IP3p5',
#    'HLT_Mu12_IP6'    ,
#]

branches += paths
branches += [path+'_ps' for path in paths]
