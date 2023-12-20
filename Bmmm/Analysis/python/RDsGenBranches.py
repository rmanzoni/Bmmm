import ROOT
import numpy as np
from collections import OrderedDict # still using old python unfortunately
from RDsBranches  import event_branches

##########################################################################################
cand_branches = OrderedDict()
cand_branches['mass'          ] = lambda cand : cand.mass()                
cand_branches['pt'            ] = lambda cand : cand.pt()                  
cand_branches['eta'           ] = lambda cand : cand.eta()                 
cand_branches['phi'           ] = lambda cand : cand.phi()                 
cand_branches['charge'        ] = lambda cand : cand.charge()              

cand_branches['vis_mass'      ] = lambda cand : cand.mass()                
cand_branches['vis_pt'        ] = lambda cand : cand.pt()                  
cand_branches['vis_eta'       ] = lambda cand : cand.eta()                 
cand_branches['vis_phi'       ] = lambda cand : cand.phi()                 

cand_branches['mass_coll'     ] = lambda cand : cand.mass_collinear()      
cand_branches['pt_coll'       ] = lambda cand : cand.pt_collinear()        
cand_branches['eta_coll'      ] = lambda cand : cand.eta_collinear()       
cand_branches['phi_coll'      ] = lambda cand : cand.phi_collinear()       

cand_branches['ds_mass'       ] = lambda cand : cand.ds.mass()             
cand_branches['ds_pt'         ] = lambda cand : cand.ds.pt()               
cand_branches['ds_eta'        ] = lambda cand : cand.ds.eta()              
cand_branches['ds_phi'        ] = lambda cand : cand.ds.phi()              
cand_branches['ds_charge'     ] = lambda cand : cand.ds.charge()           
cand_branches['phi_mass'      ] = lambda cand : cand.phi1020.mass()        
cand_branches['phi_pt'        ] = lambda cand : cand.phi1020.pt()          
cand_branches['phi_eta'       ] = lambda cand : cand.phi1020.eta()         
cand_branches['phi_phi'       ] = lambda cand : cand.phi1020.phi()         
cand_branches['phi_charge'    ] = lambda cand : cand.phi1020.charge()      
cand_branches['beta'          ] = lambda cand : cand.beta                  
cand_branches['gamma'         ] = lambda cand : cand.gamma                 
#cand_branches[#'ct'           ] =  lambda cand : cand.vtx.lxy.value() / (cand.beta * cand.gamma)
cand_branches['m2_miss'       ] = lambda cand : cand.m2_miss()             
cand_branches['q2'            ] = lambda cand : cand.q2()                  
cand_branches['e_star_mu'     ] = lambda cand : cand.e_star_mu()           
#cand_branches[#'pt_miss_sca'  ] =  lambda cand :                            
#cand_branches[#'pt_miss_vec'  ] =  lambda cand :                            
#cand_branches[#'ptvar'        ] =  lambda cand :                            
cand_branches['pv_x'          ] = lambda cand : cand.pv.position().x()     
cand_branches['pv_y'          ] = lambda cand : cand.pv.position().y()     
cand_branches['pv_z'          ] = lambda cand : cand.pv.position().z()     
cand_branches['bs_x'          ] = lambda cand : cand.bs.position().x()     
cand_branches['bs_y'          ] = lambda cand : cand.bs.position().y()     
cand_branches['vx'            ] = lambda cand : cand.vtx.position().x()    
cand_branches['vy'            ] = lambda cand : cand.vtx.position().y()    
cand_branches['vz'            ] = lambda cand : cand.vtx.position().z()    
cand_branches['vtx_chi2'      ] = lambda cand : cand.vtx.chi2              
cand_branches['vtx_prob'      ] = lambda cand : cand.vtx.prob              
cand_branches['cos2d'         ] = lambda cand : cand.vtx.cos               
cand_branches['lxy'           ] = lambda cand : cand.vtx.lxy.value()       
cand_branches['lxy_err'       ] = lambda cand : cand.vtx.lxy.error()       
cand_branches['lxy_sig'       ] = lambda cand : cand.vtx.lxy.significance()

cand_branches['phi_vx'        ] = lambda cand : cand.phi1020.vtx[0].position().x()    
cand_branches['phi_vy'        ] = lambda cand : cand.phi1020.vtx[0].position().y()    
cand_branches['phi_vz'        ] = lambda cand : cand.phi1020.vtx[0].position().z()    
cand_branches['phi_vtx_chi2'  ] = lambda cand : cand.phi1020.vtx[0].chi2              
cand_branches['phi_vtx_prob'  ] = lambda cand : cand.phi1020.vtx[0].prob              
cand_branches['phi_cos2d'     ] = lambda cand : cand.phi1020.vtx[0].cos               
cand_branches['phi_lxy'       ] = lambda cand : cand.phi1020.vtx[0].lxy.value()       
cand_branches['phi_lxy_err'   ] = lambda cand : cand.phi1020.vtx[0].lxy.error()       
cand_branches['phi_lxy_sig'   ] = lambda cand : cand.phi1020.vtx[0].lxy.significance()

cand_branches['ds_vx'         ] = lambda cand : cand.ds.vtx[0].position().x()    
cand_branches['ds_vy'         ] = lambda cand : cand.ds.vtx[0].position().y()    
cand_branches['ds_vz'         ] = lambda cand : cand.ds.vtx[0].position().z()    
cand_branches['ds_vtx_chi2'   ] = lambda cand : cand.ds.vtx[0].chi2              
cand_branches['ds_vtx_prob'   ] = lambda cand : cand.ds.vtx[0].prob              
cand_branches['ds_cos2d'      ] = lambda cand : cand.ds.vtx[0].cos               
cand_branches['ds_lxy'        ] = lambda cand : cand.ds.vtx[0].lxy.value()       
cand_branches['ds_lxy_err'    ] = lambda cand : cand.ds.vtx[0].lxy.error()       
cand_branches['ds_lxy_sig'    ] = lambda cand : cand.ds.vtx[0].lxy.significance()

#cand_branches[#'trig_match'   ] =  lambda cand : cand.trig_match        

cand_branches['dr_bs_pi'      ] = lambda cand : cand.dr_bs_pi()
cand_branches['dr_bs_k1'      ] = lambda cand : cand.dr_bs_k1()
cand_branches['dr_bs_k2'      ] = lambda cand : cand.dr_bs_k2()
cand_branches['dr_bs_mu'      ] = lambda cand : cand.dr_bs_mu()
cand_branches['dr_bs_ds'      ] = lambda cand : cand.dr_bs_ds()
cand_branches['dr_mu_ds'      ] = lambda cand : cand.dr_mu_ds()
  
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
muon_branches['id_soft'        ] = lambda imu : imu.isMediumMuon()                  
muon_branches['id_medium'      ] = lambda imu : imu.isSoftMuon(imu.pv)              
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
#muon_branches['rf_pfreliso03'  ] = lambda imu : (imu.iso03.sumChargedHadronPt + max(imu.iso03.sumNeutralHadronEt + imu.iso03.sumPhotonEt - 0.5 * imu.iso03.sumPUPt, 0.0)) / imu.rfp4.pt()
#muon_branches['rf_pfreliso04'  ] = lambda imu : (imu.iso04.sumChargedHadronPt + max(imu.iso04.sumNeutralHadronEt + imu.iso04.sumPhotonEt - 0.5 * imu.iso04.sumPUPt, 0.0)) / imu.rfp4.pt()
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
    
















##########################################################################################
##########################################################################################
##########################################################################################

#         tofill['run'          ] = event.eventAuxiliary().run()
#         tofill['lumi'         ] = event.eventAuxiliary().luminosityBlock()
#         tofill['event'        ] = event.eventAuxiliary().event()
#         
#         tofill['n_cands'      ] = len(candidates)
# 
#         tofill['ds_m_mass'    ] = b_lab_p4.mass()
#         tofill['ds_m_pt'      ] = b_lab_p4.pt()
#         tofill['ds_m_eta'     ] = b_lab_p4.eta()
#         tofill['ds_m_phi'     ] = b_lab_p4.phi()
#         tofill['max_dr'       ] = max([deltaR(the_mu, pp) for pp in [the_kp, the_km, the_pi]])
#         tofill['m2_miss'      ] = (b_scaled_p4 - the_mu.p4() - the_ds.p4()).mass2()
#         tofill['q2'           ] = (b_scaled_p4 - the_ds.p4()).mass2()
#         tofill['e_star_mu'    ] = the_mu_p4_in_b_rf.E()
# 
#         tofill['mu_pt'        ] = the_mu.pt()
#         tofill['mu_eta'       ] = the_mu.eta()
#         tofill['mu_phi'       ] = the_mu.phi()
#         tofill['mu_e'         ] = the_mu.energy()
#         tofill['mu_mass'      ] = the_mu.mass()
#         tofill['mu_charge'    ] = the_mu.charge()
# 
#         if the_tau:
#             tofill['tau_pt'       ] = the_tau.pt()
#             tofill['tau_eta'      ] = the_tau.eta()
#             tofill['tau_phi'      ] = the_tau.phi()
#             tofill['tau_e'        ] = the_tau.energy()
#             tofill['tau_mass'     ] = the_tau.mass()
#             tofill['tau_charge'   ] = the_tau.charge()
# 
#         if the_bs:
#             tofill['hb_pt'        ] = the_bs.pt()
#             tofill['hb_eta'       ] = the_bs.eta()
#             tofill['hb_phi'       ] = the_bs.phi()
#             tofill['hb_e'         ] = the_bs.energy()
#             tofill['hb_mass'      ] = the_bs.mass()
#             tofill['hb_charge'    ] = the_bs.charge()
#             tofill['hb_pdgid'     ] = the_bs.pdgId()
#             tofill['pv_x'         ] = the_bs.vertex().x()
#             tofill['pv_y'         ] = the_bs.vertex().y()
#             tofill['pv_z'         ] = the_bs.vertex().z()
#         
#     #         import pdb ; pdb.set_trace()
#             L = ROOT.Math.DisplacementVector3D('ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')( 
#                                                 the_ds.vertex().x() - the_bs.vertex().x(),
#                                                 the_ds.vertex().y() - the_bs.vertex().y(),
#                                                 the_ds.vertex().z() - the_bs.vertex().z() )
#             if L.R() > 0.:
#                 tofill['cos'       ] = b_lab_p4.Vect().Dot(L) / (b_lab_p4.Vect().R() * L.R())
# 
#         tofill['ds_pt'        ] = the_ds.pt()
#         tofill['ds_eta'       ] = the_ds.eta()
#         tofill['ds_phi'       ] = the_ds.phi()
#         tofill['ds_e'         ] = the_ds.energy()
#         tofill['ds_mass'      ] = the_ds.mass()
#         tofill['ds_charge'    ] = the_ds.charge()
# 
#         tofill['ds_st_vx'     ] = the_ds.vertex().x()
#         tofill['ds_st_vy'     ] = the_ds.vertex().y()
#         tofill['ds_st_vz'     ] = the_ds.vertex().z()
# 
#         if the_ds_st:
#             tofill['ds_st_pt'     ] = the_ds_st.pt()
#             tofill['ds_st_eta'    ] = the_ds_st.eta()
#             tofill['ds_st_phi'    ] = the_ds_st.phi()
#             tofill['ds_st_e'      ] = the_ds_st.energy()
#             tofill['ds_st_mass'   ] = the_ds_st.mass()
#             tofill['ds_st_charge' ] = the_ds_st.charge()
# 
#             tofill['ds_st_vx'     ] = the_ds_st.vertex().x()
#             tofill['ds_st_vy'     ] = the_ds_st.vertex().y()
#             tofill['ds_st_vz'     ] = the_ds_st.vertex().z()
# 
#         tofill['phi_pt'       ] = the_phi.pt()
#         tofill['phi_eta'      ] = the_phi.eta()
#         tofill['phi_phi'      ] = the_phi.phi()
#         tofill['phi_e'        ] = the_phi.energy()
#         tofill['phi_mass'     ] = the_phi.mass()
#         tofill['phi_charge'   ] = the_phi.charge()
# 
#         tofill['kp_pt'        ] = the_kp.pt()
#         tofill['kp_eta'       ] = the_kp.eta()
#         tofill['kp_phi'       ] = the_kp.phi()
#         tofill['kp_e'         ] = the_kp.energy()
#         tofill['kp_mass'      ] = the_kp.mass()
#         tofill['kp_charge'    ] = the_kp.charge()
# 
#         tofill['km_pt'        ] = the_km.pt()
#         tofill['km_eta'       ] = the_km.eta()
#         tofill['km_phi'       ] = the_km.phi()
#         tofill['km_e'         ] = the_km.energy()
#         tofill['km_mass'      ] = the_km.mass()
#         tofill['km_charge'    ] = the_km.charge()
# 
#         tofill['pi_pt'        ] = the_pi.pt()
#         tofill['pi_eta'       ] = the_pi.eta()
#         tofill['pi_phi'       ] = the_pi.phi()
#         tofill['pi_e'         ] = the_pi.energy()
#         tofill['pi_mass'      ] = the_pi.mass()
#         tofill['pi_charge'    ] = the_pi.charge()
#     
#         tofill['same_b'       ] = (the_ds.ancestors[-1] == the_mu.ancestors[-1]) if (len(the_ds.ancestors)>0 and len(the_mu.ancestors)>0) else 0.
#         tofill['sig'          ] = which_signal
#     
#         ntuple.Fill(array('f', tofill.values()))
# 