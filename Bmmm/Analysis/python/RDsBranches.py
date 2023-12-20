import ROOT
import numpy as np
from collections import OrderedDict # still using old python unfortunately

##########################################################################################
event_branches = OrderedDict()
event_branches['run'   ] = lambda ev : ev.eventAuxiliary().run()            
event_branches['lumi'  ] = lambda ev : ev.eventAuxiliary().luminosityBlock()
event_branches['event' ] = lambda ev : ev.eventAuxiliary().event()          
event_branches['qscale'] = lambda ev : getattr(ev, 'qscale', np.nan)        
event_branches['ncands'] = lambda ev : len(ev.reco_candidates)              
event_branches['npv'   ] = lambda ev : len(ev.vtx)                          
event_branches['npu'   ] = lambda ev : getattr(ev, 'npu', np.nan)           
event_branches['nti'   ] = lambda ev : getattr(ev, 'nti', np.nan)           
event_branches['bs_x0' ] = lambda ev : ev.bs.x0()                           
event_branches['bs_y0' ] = lambda ev : ev.bs.y0()                           
event_branches['bs_z0' ] = lambda ev : ev.bs.z0()                           
event_branches['sig'   ] = lambda ev : getattr(ev, 'which_signal', np.nan)

##########################################################################################
cand_branches = OrderedDict()
cand_branches['mass'            ] = lambda cand : cand.mass()                
cand_branches['mcorr'           ] = lambda cand : cand.mass_corrected()      
cand_branches['pt'              ] = lambda cand : cand.pt()                  
cand_branches['eta'             ] = lambda cand : cand.eta()                 
cand_branches['phi'             ] = lambda cand : cand.phi()                 
cand_branches['charge'          ] = lambda cand : cand.charge()              
cand_branches['mass_coll'       ] = lambda cand : cand.mass_collinear()      
cand_branches['pt_coll'         ] = lambda cand : cand.pt_collinear()        
cand_branches['eta_coll'        ] = lambda cand : cand.eta_collinear()       
cand_branches['phi_coll'        ] = lambda cand : cand.phi_collinear()       
cand_branches['ds_mass'         ] = lambda cand : cand.ds.mass()             
cand_branches['ds_pt'           ] = lambda cand : cand.ds.pt()               
cand_branches['ds_eta'          ] = lambda cand : cand.ds.eta()              
cand_branches['ds_phi'          ] = lambda cand : cand.ds.phi()              
cand_branches['ds_charge'       ] = lambda cand : cand.ds.charge()           
cand_branches['phi_mass'        ] = lambda cand : cand.phi1020.mass()        
cand_branches['phi_pt'          ] = lambda cand : cand.phi1020.pt()          
cand_branches['phi_eta'         ] = lambda cand : cand.phi1020.eta()         
cand_branches['phi_phi'         ] = lambda cand : cand.phi1020.phi()         
cand_branches['phi_charge'      ] = lambda cand : cand.phi1020.charge()      
cand_branches['beta'            ] = lambda cand : cand.beta                  
cand_branches['gamma'           ] = lambda cand : cand.gamma                 
cand_branches['ct'              ] = lambda cand : cand.vtx.lxy.value() / (cand.beta * cand.gamma)
cand_branches['m2_miss'         ] = lambda cand : cand.m2_miss()             
cand_branches['q2'              ] = lambda cand : cand.q2()                  
cand_branches['e_star_mu'       ] = lambda cand : cand.e_star_mu()           
cand_branches['e_hash_mu'       ] = lambda cand : cand.e_hash_mu()           
cand_branches['pt_miss_sca'     ] = lambda cand : cand.pt_miss_sca()                           
cand_branches['pt_miss_vec'     ] = lambda cand : cand.pt_miss_vec()                           
cand_branches['ptvar'           ] = lambda cand : cand.ptvar()                           
cand_branches['cos_theta_pi_ds' ] = lambda cand : np.cos(cand.theta_pi_ds() )
cand_branches['cos_theta_phi_ds'] = lambda cand : np.cos(cand.theta_phi_ds())
cand_branches['cos_theta_mu_w'  ] = lambda cand : np.cos(cand.theta_mu_w()  )
cand_branches['cos_theta_k_pi'  ] = lambda cand : np.cos(cand.theta_k_pi()  )
cand_branches['cos_phi_w_ds'    ] = lambda cand : np.cos(cand.phi_w_ds()    )
cand_branches['cos_phi_phi_ds'  ] = lambda cand : np.cos(cand.phi_phi_ds()  )
cand_branches['pv_x'            ] = lambda cand : cand.pv.position().x()     
cand_branches['pv_y'            ] = lambda cand : cand.pv.position().y()     
cand_branches['pv_z'            ] = lambda cand : cand.pv.position().z()     
cand_branches['bs_x'            ] = lambda cand : cand.bs.position().x()     
cand_branches['bs_y'            ] = lambda cand : cand.bs.position().y()     
cand_branches['vx'              ] = lambda cand : cand.vtx.position().x()    
cand_branches['vy'              ] = lambda cand : cand.vtx.position().y()    
cand_branches['vz'              ] = lambda cand : cand.vtx.position().z()    
cand_branches['vtx_chi2'        ] = lambda cand : cand.vtx.chi2              
cand_branches['vtx_prob'        ] = lambda cand : cand.vtx.prob              
cand_branches['cos2d'           ] = lambda cand : cand.vtx.cos2d               
cand_branches['lxy'             ] = lambda cand : cand.vtx.lxy.value()       
cand_branches['lxy_err'         ] = lambda cand : cand.vtx.lxy.error()       
cand_branches['lxy_sig'         ] = lambda cand : cand.vtx.lxy.significance()
cand_branches['cos3d'           ] = lambda cand : cand.vtx.cos3d               
cand_branches['lxyz'            ] = lambda cand : cand.vtx.lxyz.value()       
cand_branches['lxyz_err'        ] = lambda cand : cand.vtx.lxyz.error()       
cand_branches['lxyz_sig'        ] = lambda cand : cand.vtx.lxyz.significance()
  
cand_branches['phi_vx'          ] = lambda cand : cand.phi1020.vtx[0].position().x()    
cand_branches['phi_vy'          ] = lambda cand : cand.phi1020.vtx[0].position().y()    
cand_branches['phi_vz'          ] = lambda cand : cand.phi1020.vtx[0].position().z()    
cand_branches['phi_vtx_chi2'    ] = lambda cand : cand.phi1020.vtx[0].chi2              
cand_branches['phi_vtx_prob'    ] = lambda cand : cand.phi1020.vtx[0].prob              
cand_branches['phi_cos2d'       ] = lambda cand : cand.phi1020.vtx[0].cos2d               
cand_branches['phi_lxy'         ] = lambda cand : cand.phi1020.vtx[0].lxy.value()       
cand_branches['phi_lxy_err'     ] = lambda cand : cand.phi1020.vtx[0].lxy.error()       
cand_branches['phi_lxy_sig'     ] = lambda cand : cand.phi1020.vtx[0].lxy.significance()
cand_branches['phi_cos3d'       ] = lambda cand : cand.phi1020.vtx[0].cos3d               
cand_branches['phi_lxyz'        ] = lambda cand : cand.phi1020.vtx[0].lxyz.value()       
cand_branches['phi_lxyz_err'    ] = lambda cand : cand.phi1020.vtx[0].lxyz.error()       
cand_branches['phi_lxyz_sig'    ] = lambda cand : cand.phi1020.vtx[0].lxyz.significance()
  
cand_branches['ds_vx'           ] = lambda cand : cand.ds.vtx[0].position().x()    
cand_branches['ds_vy'           ] = lambda cand : cand.ds.vtx[0].position().y()    
cand_branches['ds_vz'           ] = lambda cand : cand.ds.vtx[0].position().z()    
cand_branches['ds_vtx_chi2'     ] = lambda cand : cand.ds.vtx[0].chi2              
cand_branches['ds_vtx_prob'     ] = lambda cand : cand.ds.vtx[0].prob              
cand_branches['ds_cos2d'        ] = lambda cand : cand.ds.vtx[0].cos2d               
cand_branches['ds_lxy'          ] = lambda cand : cand.ds.vtx[0].lxy.value()       
cand_branches['ds_lxy_err'      ] = lambda cand : cand.ds.vtx[0].lxy.error()       
cand_branches['ds_lxy_sig'      ] = lambda cand : cand.ds.vtx[0].lxy.significance()
cand_branches['ds_cos3d'        ] = lambda cand : cand.ds.vtx[0].cos3d               
cand_branches['ds_lxyz'         ] = lambda cand : cand.ds.vtx[0].lxyz.value()       
cand_branches['ds_lxyz_err'     ] = lambda cand : cand.ds.vtx[0].lxyz.error()       
cand_branches['ds_lxyz_sig'     ] = lambda cand : cand.ds.vtx[0].lxyz.significance()
  
cand_branches['dr_bs_pi'        ] = lambda cand : cand.dr_bs_pi()
cand_branches['dr_bs_k1'        ] = lambda cand : cand.dr_bs_k1()
cand_branches['dr_bs_k2'        ] = lambda cand : cand.dr_bs_k2()
cand_branches['dr_bs_mu'        ] = lambda cand : cand.dr_bs_mu()
cand_branches['dr_bs_ds'        ] = lambda cand : cand.dr_bs_ds()
cand_branches['dr_mu_ds'        ] = lambda cand : cand.dr_mu_ds()
  
cand_branches['dr_bs_pi'      ] = lambda cand : cand.dr_bs_pi()
cand_branches['dr_bs_k1'      ] = lambda cand : cand.dr_bs_k1()
cand_branches['dr_bs_k2'      ] = lambda cand : cand.dr_bs_k2()
cand_branches['dr_bs_mu'      ] = lambda cand : cand.dr_bs_mu()
cand_branches['dr_bs_ds'      ] = lambda cand : cand.dr_bs_ds()
cand_branches['dr_mu_ds'      ] = lambda cand : cand.dr_mu_ds()
cand_branches['dr_mu_k1'      ] = lambda cand : cand.dr_mu_k1()
cand_branches['dr_mu_k2'      ] = lambda cand : cand.dr_mu_k2()
cand_branches['dr_mu_pi'      ] = lambda cand : cand.dr_mu_pi()
cand_branches['max_dr_mu_kkpi'] = lambda cand : cand.max_dr_mu_kkpi()

cand_branches['dr_phi_pi'     ] = lambda cand : cand.ds.dr_phi_pi()
cand_branches['dr_ds_k1'      ] = lambda cand : cand.ds.dr_ds_k1()
cand_branches['dr_ds_k2'      ] = lambda cand : cand.ds.dr_ds_k2()
cand_branches['dr_ds_pi'      ] = lambda cand : cand.ds.dr_ds_pi()
cand_branches['dr_ds_k1'      ] = lambda cand : cand.ds.dr_ds_k1()
cand_branches['dr_ds_k2'      ] = lambda cand : cand.ds.dr_ds_k2()
  
cand_branches['dr_k1_k2'      ] = lambda cand : cand.phi1020.dr_kk()
cand_branches['dr_phi_k1'     ] = lambda cand : cand.phi1020.dr_phi_k1()
cand_branches['dr_phi_k2'     ] = lambda cand : cand.phi1020.dr_phi_k2()
cand_branches['max_dr_phi_k'  ] = lambda cand : cand.phi1020.max_dr_phi_k()

##########################################################################################
track_branches = OrderedDict()
track_branches['pt'        ] = lambda itk : itk.pt()    
track_branches['eta'       ] = lambda itk : itk.eta()   
track_branches['phi'       ] = lambda itk : itk.phi()   
track_branches['e'         ] = lambda itk : itk.energy()
track_branches['mass'      ] = lambda itk : itk.mass()  
track_branches['charge'    ] = lambda itk : itk.charge()
track_branches['dxy'       ] = lambda itk : itk.bestTrack().dxy(itk.pv.position())
track_branches['dxy_e'     ] = lambda itk : itk.bestTrack().dxyError(itk.pv.position(), itk.pv.error())
track_branches['dxy_sig'   ] = lambda itk : itk.bestTrack().dxy(itk.pv.position()) / itk.bestTrack().dxyError(itk.pv.position(), itk.pv.error())
track_branches['dz'        ] = lambda itk : itk.bestTrack().dz(itk.pv.position())
track_branches['dz_e'      ] = lambda itk : itk.bestTrack().dzError()
track_branches['dz_sig'    ] = lambda itk : itk.bestTrack().dz(itk.pv.position()) / itk.bestTrack().dzError()
track_branches['bs_dxy'    ] = lambda itk : itk.bestTrack().dxy(itk.bs.position())
track_branches['bs_dxy_e'  ] = lambda itk : itk.bestTrack().dxyError(itk.bs.position(), itk.bs.error())
track_branches['bs_dxy_sig'] = lambda itk : itk.bestTrack().dxy(itk.bs.position()) / itk.bestTrack().dxyError(itk.bs.position(), itk.bs.error())

##########################################################################################
muon_branches = OrderedDict()
muon_branches['pt'             ] = lambda imu : imu.pt()       
muon_branches['eta'            ] = lambda imu : imu.eta()      
muon_branches['phi'            ] = lambda imu : imu.phi()                           
muon_branches['e'              ] = lambda imu : imu.energy()                        
muon_branches['mass'           ] = lambda imu : imu.mass()                          
muon_branches['charge'         ] = lambda imu : imu.charge()                        
muon_branches['id_loose'       ] = lambda imu : imu.isLooseMuon()                   
muon_branches['id_soft'        ] = lambda imu : imu.isSoftMuon(imu.pv)                  
muon_branches['id_medium'      ] = lambda imu : imu.isMediumMuon()              
muon_branches['id_tight'       ] = lambda imu : imu.isTightMuon(imu.pv)             
muon_branches['id_soft_mva_raw'] = lambda imu : imu.softMvaValue()                  
muon_branches['id_soft_mva'    ] = lambda imu : imu.passed(ROOT.reco.Muon.SoftMvaId)
muon_branches['id_pf'          ] = lambda imu : imu.isPFMuon()                      
muon_branches['id_global'      ] = lambda imu : imu.isGlobalMuon()                  
muon_branches['id_tracker'     ] = lambda imu : imu.isTrackerMuon()                 
muon_branches['id_standalone'  ] = lambda imu : imu.isStandAloneMuon()              
muon_branches['pfiso03'        ] = lambda imu : (imu.iso03.sumChargedHadronPt + max(imu.iso03.sumNeutralHadronEt + imu.iso03.sumPhotonEt - 0.5 * imu.iso03.sumPUPt, 0.0))           
muon_branches['pfiso04'        ] = lambda imu : (imu.iso04.sumChargedHadronPt + max(imu.iso04.sumNeutralHadronEt + imu.iso04.sumPhotonEt - 0.5 * imu.iso04.sumPUPt, 0.0))           
muon_branches['pfreliso03'     ] = lambda imu : (imu.iso03.sumChargedHadronPt + max(imu.iso03.sumNeutralHadronEt + imu.iso03.sumPhotonEt - 0.5 * imu.iso03.sumPUPt, 0.0)) / imu.pt()
muon_branches['pfreliso04'     ] = lambda imu : (imu.iso04.sumChargedHadronPt + max(imu.iso04.sumNeutralHadronEt + imu.iso04.sumPhotonEt - 0.5 * imu.iso04.sumPUPt, 0.0)) / imu.pt()
muon_branches['pfiso03_ch'     ] = lambda imu : imu.iso03.sumChargedHadronPt  
muon_branches['pfiso03_cp'     ] = lambda imu : imu.iso03.sumChargedParticlePt
muon_branches['pfiso03_nh'     ] = lambda imu : imu.iso03.sumNeutralHadronEt  
muon_branches['pfiso03_ph'     ] = lambda imu : imu.iso03.sumPhotonEt         
muon_branches['pfiso03_pu'     ] = lambda imu : imu.iso03.sumPUPt             
muon_branches['pfiso04_ch'     ] = lambda imu : imu.iso04.sumChargedHadronPt  
muon_branches['pfiso04_cp'     ] = lambda imu : imu.iso04.sumChargedParticlePt
muon_branches['pfiso04_nh'     ] = lambda imu : imu.iso04.sumNeutralHadronEt  
muon_branches['pfiso04_ph'     ] = lambda imu : imu.iso04.sumPhotonEt         
muon_branches['pfiso04_pu'     ] = lambda imu : imu.iso04.sumPUPt             
muon_branches['dxy'            ] = lambda imu : imu.bestTrack().dxy(imu.pv.position())
muon_branches['dxy_e'          ] = lambda imu : imu.bestTrack().dxyError(imu.pv.position(), imu.pv.error())
muon_branches['dxy_sig'        ] = lambda imu : imu.bestTrack().dxy(imu.pv.position()) / imu.bestTrack().dxyError(imu.pv.position(), imu.pv.error())
muon_branches['dz'             ] = lambda imu : imu.bestTrack().dz(imu.pv.position())
muon_branches['dz_e'           ] = lambda imu : imu.bestTrack().dzError()
muon_branches['dz_sig'         ] = lambda imu : imu.bestTrack().dz(imu.pv.position()) / imu.bestTrack().dzError()
muon_branches['bs_dxy'         ] = lambda imu : imu.bestTrack().dxy(imu.bs.position())
muon_branches['bs_dxy_e'       ] = lambda imu : imu.bestTrack().dxyError(imu.bs.position(), imu.bs.error())
muon_branches['bs_dxy_sig'     ] = lambda imu : imu.bestTrack().dxy(imu.bs.position()) / imu.bestTrack().dxyError(imu.bs.position(), imu.bs.error())
#muon_branches['cov_pos_def'    ] = lambda imu : imu.is_cov_pos_def
muon_branches['jet_pt'         ] = lambda imu : imu.jet.pt()      if hasattr(imu, 'jet') else np.nan
muon_branches['jet_eta'        ] = lambda imu : imu.jet.eta()     if hasattr(imu, 'jet') else np.nan
muon_branches['jet_phi'        ] = lambda imu : imu.jet.phi()     if hasattr(imu, 'jet') else np.nan
muon_branches['jet_e'          ] = lambda imu : imu.jet.energy()  if hasattr(imu, 'jet') else np.nan
muon_branches['gen_pt'         ] = lambda imu : imu.genp.pt()     if hasattr(imu, 'genp') else np.nan
muon_branches['gen_eta'        ] = lambda imu : imu.genp.eta()    if hasattr(imu, 'genp') else np.nan
muon_branches['gen_phi'        ] = lambda imu : imu.genp.phi()    if hasattr(imu, 'genp') else np.nan
muon_branches['gen_e'          ] = lambda imu : imu.genp.energy() if hasattr(imu, 'genp') else np.nan
muon_branches['gen_pdgid'      ] = lambda imu : imu.genp.pdgId()  if hasattr(imu, 'genp') else np.nan

##########################################################################################
##########################################################################################

branches = []

for ibr in event_branches.keys():
    branches.append(ibr)

for ibranch in cand_branches.keys():
    branches.append(ibranch)

for ibr in muon_branches.keys():
    branches.append('mu_%s' %(ibr))

for itk in ['k1', 'k2', 'pi']:
    for ibr in track_branches.keys():
        branches.append('%s_%s' %(itk,ibr))
    
# https://hlt-config-editor-confdbv3.app.cern.ch/open?cfg=%2Fcdaq%2Fphysics%2FRun2018%2F2e34%2Fv3.6.1%2FHLT%2FV2&db=online
paths = {}
paths['HLT_Mu7_IP4'     ] = ['hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered7IP4Q']
#paths['HLT_Mu8_IP3'     ] = ['hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered8Q'] # why different?!
#paths['HLT_Mu8_IP5'     ] = ['hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered8IP5Q']
#paths['HLT_Mu8_IP6'     ] = ['hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered8IP6Q']
##paths['HLT_Mu8p5_IP3p5' ] = []
#paths['HLT_Mu9_IP4'     ] = ['hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered9IP4Q']
#paths['HLT_Mu9_IP5'     ] = ['hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered9IP5Q']
#paths['HLT_Mu9_IP6'     ] = ['hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered9Q'] # why different?!
##paths['HLT_Mu10p5_IP3p5'] = []
#paths['HLT_Mu12_IP6'    ] = ['hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered12Q'] # why different?!

branches += paths
branches += [path+'_ps' for path in paths]

















##########################################################################################
##########################################################################################
##########################################################################################

# 
# branches = [
#     'run'          
#     'lumi'         
#     'event'        
# 
#     'n_cands'      
# 
#     'ds_m_mass'    
#     'ds_m_pt'      
#     'ds_m_eta'     
#     'ds_m_phi'     
#     'max_dr'       
#     'm2_miss'      
#     'q2'           
#     'e_star_mu'    
# 
#     'mu_pt'        
#     'mu_eta'       
#     'mu_phi'       
#     'mu_e'         
#     'mu_mass'      
#     'mu_charge'    
# 
#     'tau_pt'       
#     'tau_eta'      
#     'tau_phi'      
#     'tau_e'        
#     'tau_mass'     
#     'tau_charge'   
# 
#     'hb_pt'        
#     'hb_eta'       
#     'hb_phi'       
#     'hb_e'         
#     'hb_mass'      
#     'hb_charge'    
#     'hb_pdgid'     
# 
#     'ds_pt'        
#     'ds_eta'       
#     'ds_phi'       
#     'ds_e'         
#     'ds_mass'      
#     'ds_charge'    
# 
#     'ds_st_pt'     
#     'ds_st_eta'    
#     'ds_st_phi'    
#     'ds_st_e'      
#     'ds_st_mass'   
#     'ds_st_charge' 
# 
#     'phi_pt'       
#     'phi_eta'      
#     'phi_phi'      
#     'phi_e'        
#     'phi_mass'     
#     'phi_charge'   
# 
#     'kp_pt'        
#     'kp_eta'       
#     'kp_phi'       
#     'kp_e'         
#     'kp_mass'      
#     'kp_charge'    
# 
#     'km_pt'        
#     'km_eta'       
#     'km_phi'       
#     'km_e'         
#     'km_mass'      
#     'km_charge'    
# 
#     'pi_pt'        
#     'pi_eta'       
#     'pi_phi'       
#     'pi_e'         
#     'pi_mass'      
#     'pi_charge'    
# 
#     'pv_x'         
#     'pv_y'         
#     'pv_z'         
# 
#     'ds_st_vx'     
#     'ds_st_vy'     
#     'ds_st_vz'     
# 
#     'ds_vx'        
#     'ds_vy'        
#     'ds_vz'        
# 
#     'cos'          
#     
#     'same_b'       
#     'sig'          
# ]


# 
# 
# branches_reco = [
#     
#     'ds_m_mass'    
#     'ds_m_pt'      
#     'ds_m_eta'     
#     'ds_m_phi'     
#     'b_pt'         
#     'b_eta'        
#     'b_phi'        
#     'b_beta'       
#     'b_gamma'      
#     'b_ct'         
#     'm2_miss'      
#     'q2'           
#     'e_star_mu'    
#     'pt_miss_sca'  
#     'pt_miss_vec'  
#     'ptvar'        
# 
#     'mu_pt'        
#     'mu_eta'       
#     'mu_phi'       
#     'mu_e'         
#     'mu_mass'      
#     'mu_charge'    
#     'mu_id_loose'  
#     'mu_id_soft'   
#     'mu_id_medium' 
#     'mu_id_tight'  
#     'mu_ch_iso'    
#     'mu_db_n_iso'  
#     'mu_abs_iso'   
#     'mu_rel_iso'   
# #     'mu_iso'       
#     'mu_dxy'       
#     'mu_dxy_e'     
#     'mu_dxy_sig'   
#     'mu_dz'        
#     'mu_dz_e'      
#     'mu_dz_sig'    
#     'mu_bs_dxy'    
#     'mu_bs_dxy_e'  
#     'mu_bs_dxy_sig'
# 
#     'ds_pt'        
#     'ds_eta'       
#     'ds_phi'       
#     'ds_e'         
#     'ds_mass'      
# 
#     'phi_pt'       
#     'phi_eta'      
#     'phi_phi'      
#     'phi_e'        
#     'phi_mass'     
# 
#     'kp_pt'        
#     'kp_eta'       
#     'kp_phi'       
#     'kp_e'         
#     'kp_mass'      
#     'kp_charge'    
#     'kp_dxy'       
#     'kp_dxy_e'     
#     'kp_dxy_sig'   
#     'kp_dz'        
#     'kp_dz_e'      
#     'kp_dz_sig'    
#     'kp_bs_dxy'    
#     'kp_bs_dxy_e'  
#     'kp_bs_dxy_sig'
# 
#     'km_pt'        
#     'km_eta'       
#     'km_phi'       
#     'km_e'         
#     'km_mass'      
#     'km_charge'    
#     'km_dxy'       
#     'km_dxy_e'     
#     'km_dxy_sig'   
#     'km_dz'        
#     'km_dz_e'      
#     'km_dz_sig'    
#     'km_bs_dxy'    
#     'km_bs_dxy_e'  
#     'km_bs_dxy_sig'
# 
#     'pi_pt'        
#     'pi_eta'       
#     'pi_phi'       
#     'pi_e'         
#     'pi_mass'      
#     'pi_charge'    
#     'pi_dxy'       
#     'pi_dz'        
# 
#     'dr_m_kp'      
#     'dr_m_km'      
#     'dr_m_pi'      
#     'dr_m_ds'      
# 
#     'pv_x'         
#     'pv_y'         
#     'pv_z'         
# 
#     'phi_vx'       
#     'phi_vy'       
#     'phi_vz'       
#     'phi_vtx_chi2' 
#     'phi_vtx_prob' 
# 
#     'ds_vx'        
#     'ds_vy'        
#     'ds_vz'        
#     'ds_vtx_chi2'  
#     'ds_vtx_prob'  
# 
#     'ds_m_vx'      
#     'ds_m_vy'      
#     'ds_m_vz'      
#     'ds_m_vtx_chi2'
#     'ds_m_vtx_prob'
# 
#     'bs_x0'        
#     'bs_y0'        
#     'bs_z0'        
#     
#     'cos3D_ds'     
#     'lxyz_ds'      
#     'lxyz_ds_err'  
#     'lxyz_ds_sig'  
# 
#     'cos2D_ds'     
#     'lxy_ds'       
#     'lxy_ds_err'   
#     'lxy_ds_sig'   
# 
#     'cos3D_ds_m'   ,    
#     'lxyz_ds_m'    
#     'lxyz_ds_m_err'
#     'lxyz_ds_m_sig'
# 
#     'cos2D_ds_m'   
#     'lxy_ds_m'     
#     'lxy_ds_m_err' 
#     'lxy_ds_m_sig' 
# 
#     'sig'          
# ]
# 
# 
# 
