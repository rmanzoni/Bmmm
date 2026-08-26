import numpy as np

# J/psi mu (RJpsi) branch schema. The event / per-muon / Bc-gen / J/psi-gen
# blocks, the HLT paths and safe_get are imported UNCHANGED from the shared
# JpsiChargedBranches, so the muon side and the event side are identical to the
# J/psi + track ntuple. Only the candidate-level block (cand_branches) and the
# flat 'branches' assembly live here; they are a verbatim copy of the original
# RJpsiBranches, so 'branches' reproduces the RJpsi schema branch-for-branch.
from Bmmm.Analysis.JpsiChargedBranches import (
    event_branches, muon_branches, bc_branches, jpsi_branches, paths, safe_get,
)

cand_branches = {
    # ----- 3-muon (Bc) kinematics -----
    'mass'              : lambda cand : cand.mass()            ,
    'pt'                : lambda cand : cand.pt()              ,
    'eta'               : lambda cand : cand.eta()             ,
    'phi'               : lambda cand : cand.phi()             ,
    'charge'            : lambda cand : cand.charge()          ,

    'q2_coll'           : lambda cand : (cand.p4_collinear - cand.jpsi_rfp4).mass2(),
    'm_miss2_coll'      : lambda cand : (cand.p4_collinear - cand.jpsi_rfp4 - cand.mu.p4()).mass2() ,
    'mu_b_e_coll'       : lambda cand : cand.p4_collinear.Dot(cand.mu.p4()) / cand.p4_collinear.mass(),

    'nu1_q2_jpsi'       : lambda cand : (cand.math1_b_p4_jpsi - cand.jpsi_rfp4).mass2()   ,
    'nu2_q2_jpsi'       : lambda cand : (cand.math2_b_p4_jpsi - cand.jpsi_rfp4).mass2()   ,
    'nu1_q2_sv'         : lambda cand : (cand.math1_b_p4_sv - cand.jpsi_rfp4).mass2()   ,
    'nu2_q2_sv'         : lambda cand : (cand.math2_b_p4_sv - cand.jpsi_rfp4).mass2()   ,
    'nu1_mu_b_e_jpsi'   : lambda cand : cand.math1_b_p4_jpsi.Dot(cand.mu.p4()) / cand.math1_b_p4_jpsi.mass(),
    'nu2_mu_b_e_jpsi'   : lambda cand : cand.math2_b_p4_jpsi.Dot(cand.mu.p4()) / cand.math2_b_p4_jpsi.mass(),
    'nu1_mu_b_e_sv'     : lambda cand : cand.math1_b_p4_sv  .Dot(cand.mu.p4()) / cand.math1_b_p4_sv  .mass(),
    'nu2_mu_b_e_sv'     : lambda cand : cand.math2_b_p4_sv  .Dot(cand.mu.p4()) / cand.math2_b_p4_sv  .mass(),

    'q2_jpsi'           : lambda cand : (cand.bc_full_p4_jpsi - cand.jpsi_rfp4).mass2(),
    'q2_sv'             : lambda cand : (cand.bc_full_p4_sv   - cand.jpsi_rfp4).mass2(),
    'm_miss2_jpsi'      : lambda cand : (cand.bc_full_p4_jpsi - cand.jpsi_rfp4 - cand.mu.p4()).mass2() ,
    'm_miss2_sv'        : lambda cand : (cand.bc_full_p4_sv   - cand.jpsi_rfp4 - cand.mu.p4()).mass2() ,
    'mu_b_e_jpsi'       : lambda cand : cand.bc_full_p4_jpsi.Dot(cand.mu.p4()) / cand.bc_full_p4_jpsi.mass(),
    'mu_b_e_sv'         : lambda cand : cand.bc_full_p4_sv  .Dot(cand.mu.p4()) / cand.bc_full_p4_sv  .mass(),
    'mu_jpsi_e'         : lambda cand : cand.jpsi_rfp4.Dot(cand.mu.p4()) / cand.jpsi_rfp4.mass()  ,

    'p4_par_jpsi'       : lambda cand: cand.p4_par_jpsi  ,
    'p4_perp_jpsi'      : lambda cand: cand.p4_perp_jpsi ,
    'mcorr_jpsi'        : lambda cand: cand.mcorr_jpsi   ,

    'p4_par_sv'         : lambda cand: cand.p4_par_sv  ,
    'p4_perp_sv'        : lambda cand: cand.p4_perp_sv ,
    'mcorr_sv'          : lambda cand: cand.mcorr_sv   ,

    'q2_gen'            : lambda cand : cand.q2_gen           ,
    'm_miss2_gen'       : lambda cand : cand.m_miss2_gen      ,

    # ----- Hb background gen truth (set by RJPsiHbMatcher; NaN otherwise) -----
    # 1.0 : bachelor mu and J/psi share the same b-hadron ancestor
    # 0.0 : different b hadron, or bachelor not from a b (combinatorial)
    # NaN : J/psi b mother not defined (e.g. reco dimuon not a single-b J/psi) / not Hb MC
    'gen_hb_same_mother'        : lambda cand : cand.gen_hb_same_mother                                            ,
    'gen_hb_jpsi_b_pdgid'       : lambda cand : cand.gen_jpsi_b.pdgId()          if cand.gen_jpsi_b          is not None else np.nan,
    'gen_hb_bachelor_b_pdgid'   : lambda cand : cand.gen_bachelor_b.pdgId()      if cand.gen_bachelor_b      is not None else np.nan,
    'gen_hb_bachelor_mom_pdgid' : lambda cand : cand.gen_bachelor_mother.pdgId() if cand.gen_bachelor_mother is not None else np.nan,

    # ----- 3-muon (Bc) kinematics, jpsi constraint -----
    'rf_mass'           : lambda cand : cand.rfp4.mass()       ,
    'rf_pt'             : lambda cand : cand.rfp4.pt()         ,
    'rf_eta'            : lambda cand : cand.rfp4.eta()        ,
    'rf_phi'            : lambda cand : cand.rfp4.phi()        ,

    # ----- mathematical neutrino solutions -----
    'nu1_jpsi_bc_e'          : lambda cand : cand.sols_jpsi[0].p4_parent.energy(),
    'nu1_jpsi_bc_pt'         : lambda cand : cand.sols_jpsi[0].p4_parent.pt()    ,
    'nu1_jpsi_bc_eta'        : lambda cand : cand.sols_jpsi[0].p4_parent.eta()   ,
    'nu1_jpsi_bc_phi'        : lambda cand : cand.sols_jpsi[0].p4_parent.phi()   ,
    'nu1_jpsi_pz'            : lambda cand : cand.sols_jpsi[0].pz                ,
    'nu1_jpsi_e'             : lambda cand : cand.sols_jpsi[0].p4_nu.energy()    ,
    'nu1_jpsi_pt'            : lambda cand : cand.sols_jpsi[0].p4_nu.pt()        ,
    'nu1_jpsi_eta'           : lambda cand : cand.sols_jpsi[0].p4_nu.eta()       ,
    'nu1_jpsi_phi'           : lambda cand : cand.sols_jpsi[0].p4_nu.phi()       ,

    'nu2_jpsi_bc_e'          : lambda cand : cand.sols_jpsi[1].p4_parent.energy(),
    'nu2_jpsi_bc_pt'         : lambda cand : cand.sols_jpsi[1].p4_parent.pt()    ,
    'nu2_jpsi_bc_eta'        : lambda cand : cand.sols_jpsi[1].p4_parent.eta()   ,
    'nu2_jpsi_bc_phi'        : lambda cand : cand.sols_jpsi[1].p4_parent.phi()   ,
    'nu2_jpsi_pz'            : lambda cand : cand.sols_jpsi[1].pz                ,
    'nu2_jpsi_e'             : lambda cand : cand.sols_jpsi[1].p4_nu.energy()    ,
    'nu2_jpsi_pt'            : lambda cand : cand.sols_jpsi[1].p4_nu.pt()        ,
    'nu2_jpsi_eta'           : lambda cand : cand.sols_jpsi[1].p4_nu.eta()       ,
    'nu2_jpsi_phi'           : lambda cand : cand.sols_jpsi[1].p4_nu.phi()       ,

    'nu1_sv_bc_e'          : lambda cand : cand.sols_sv[0].p4_parent.energy(),
    'nu1_sv_bc_pt'         : lambda cand : cand.sols_sv[0].p4_parent.pt()    ,
    'nu1_sv_bc_eta'        : lambda cand : cand.sols_sv[0].p4_parent.eta()   ,
    'nu1_sv_bc_phi'        : lambda cand : cand.sols_sv[0].p4_parent.phi()   ,
    'nu1_sv_pz'            : lambda cand : cand.sols_sv[0].pz                ,
    'nu1_sv_e'             : lambda cand : cand.sols_sv[0].p4_nu.energy()    ,
    'nu1_sv_pt'            : lambda cand : cand.sols_sv[0].p4_nu.pt()        ,
    'nu1_sv_eta'           : lambda cand : cand.sols_sv[0].p4_nu.eta()       ,
    'nu1_sv_phi'           : lambda cand : cand.sols_sv[0].p4_nu.phi()       ,

    'nu2_sv_bc_e'          : lambda cand : cand.sols_sv[1].p4_parent.energy(),
    'nu2_sv_bc_pt'         : lambda cand : cand.sols_sv[1].p4_parent.pt()    ,
    'nu2_sv_bc_eta'        : lambda cand : cand.sols_sv[1].p4_parent.eta()   ,
    'nu2_sv_bc_phi'        : lambda cand : cand.sols_sv[1].p4_parent.phi()   ,
    'nu2_sv_pz'            : lambda cand : cand.sols_sv[1].pz                ,
    'nu2_sv_e'             : lambda cand : cand.sols_sv[1].p4_nu.energy()    ,
    'nu2_sv_pt'            : lambda cand : cand.sols_sv[1].p4_nu.pt()        ,
    'nu2_sv_eta'           : lambda cand : cand.sols_sv[1].p4_nu.eta()       ,
    'nu2_sv_phi'           : lambda cand : cand.sols_sv[1].p4_nu.phi()       ,

    # ----- J/psi (dimuon) kinematics -----
    'jpsi_mass'         : lambda cand : cand.jpsi.mass()       ,
    'jpsi_pt'           : lambda cand : cand.jpsi.pt()         ,
    'jpsi_eta'          : lambda cand : cand.jpsi.eta()        ,
    'jpsi_phi'          : lambda cand : cand.jpsi.phi()        ,
    'jpsi_charge'       : lambda cand : cand.jpsi.charge()     ,

    # ----- J/psi (dimuon) kinematics -----
    'jpsi_rf_mass'      : lambda cand : cand.jpsi_rfp4.mass()  ,
    'jpsi_rf_pt'        : lambda cand : cand.jpsi_rfp4.pt()    ,
    'jpsi_rf_eta'       : lambda cand : cand.jpsi_rfp4.eta()   ,
    'jpsi_rf_phi'       : lambda cand : cand.jpsi_rfp4.phi()   ,

    # ----- pairwise muon kinematics (pt-sorted muons) -----
    'dr'                : lambda cand : cand.r()               ,
    'dr_max'            : lambda cand : cand.max_dr()          ,
#     'dr_12'             : lambda cand : cand.dr12()            ,
#     'dr_13'             : lambda cand : cand.dr13()            ,
#     'dr_23'             : lambda cand : cand.dr23()            ,
# 
#     'charge_12'         : lambda cand : cand.charge12()        ,
#     'charge_13'         : lambda cand : cand.charge13()        ,
#     'charge_23'         : lambda cand : cand.charge23()        ,
# 
#     'mass_12'           : lambda cand : cand.mass12()          ,
#     'mass_13'           : lambda cand : cand.mass13()          ,
#     'mass_23'           : lambda cand : cand.mass23()          ,
# 
#     'min_mass'          : lambda cand : min([cand.mass12(),
#                                              cand.mass13(),
#                                              cand.mass23()])   ,
 
#     'rf_mass'           : lambda cand : cand.rf_mass()         ,
#     'rf_mass_err'       : lambda cand : np.sqrt(cand.b4refUnc.At(6,6)),
#     'rf_mass12_err'     : lambda cand : cand.dimuon12_mass_unc ,
#     'rf_mass13_err'     : lambda cand : cand.dimuon13_mass_unc ,
#     'rf_mass23_err'     : lambda cand : cand.dimuon23_mass_unc ,
#     'rf_pt'             : lambda cand : cand.rf_pt()           ,
#     'rf_eta'            : lambda cand : cand.rf_eta()          ,
#     'rf_phi'            : lambda cand : cand.rf_phi()          ,
#  
#     'rf_dr'             : lambda cand : cand.rf_r()            ,
#     'rf_dr_max'         : lambda cand : cand.rf_max_dr()       ,
#     'rf_dr_12'          : lambda cand : cand.rf_dr12()         ,
#     'rf_dr_13'          : lambda cand : cand.rf_dr13()         ,
#     'rf_dr_23'          : lambda cand : cand.rf_dr23()         ,
#   
#     'rf_mass_12'        : lambda cand : cand.rf_mass12()       ,
#     'rf_mass_13'        : lambda cand : cand.rf_mass13()       ,
#     'rf_mass_23'        : lambda cand : cand.rf_mass23()       ,
# 
#     'rf_min_mass'       : lambda cand : min([cand.rf_mass12()  , 
#                                              cand.rf_mass13()  , 
#                                              cand.rf_mass23()]),

    # ----- primary vertex / beamspot -----
    'pv_x'              : lambda cand : cand.pv.position().x()  ,
    'pv_y'              : lambda cand : cand.pv.position().y()  ,
    'pv_z'              : lambda cand : cand.pv.position().z()  ,
 
    'bs_x'              : lambda cand : cand.bs.position().x()  ,
    'bs_y'              : lambda cand : cand.bs.position().y()  ,
 
    # ----- 3-muon (Bc) secondary vertex ----- 
    'sv_good'           : lambda cand : cand.good_vtx           ,
    'sv_x'              : lambda cand : cand.vtx.position().x() ,
    'sv_y'              : lambda cand : cand.vtx.position().y() ,
    'sv_z'              : lambda cand : cand.vtx.position().z() ,
    
    'sv_chi2'           : lambda cand : cand.vtx_chi2           ,
    'sv_ndof'           : lambda cand : cand.vtx_ndof          ,
    'sv_prob'           : lambda cand : cand.vtx_prob           ,
 
    'cos2d'             : lambda cand : cand.cos2d              ,
    'cos3d'             : lambda cand : cand.cos3d              ,

    'cos3dbs'           : lambda cand : cand.cos3dbs            ,
 
    'lxy'               : lambda cand : cand.lxy.value()        ,
    'lxy_err'           : lambda cand : cand.lxy.error()        ,
    'lxy_sig'           : lambda cand : cand.lxy.significance() ,

    'lxyz'              : lambda cand : cand.lxyz.value()       ,
    'lxyz_err'          : lambda cand : cand.lxyz.error()       ,
    'lxyz_sig'          : lambda cand : cand.lxyz.significance(),

    # ----- J/psi (dimuon) secondary vertex -----
    'jpsi_good_vtx'     : lambda cand : cand.jpsi_good_vtx           ,
    'jpsi_x'            : lambda cand : cand.jpsi_vtx.position().x() ,
    'jpsi_y'            : lambda cand : cand.jpsi_vtx.position().y() ,
    'jpsi_z'            : lambda cand : cand.jpsi_vtx.position().z() ,
    'jpsi_vtx_chi2'     : lambda cand : cand.jpsi_vtx_chi2           ,
    'jpsi_vtx_ndof'     : lambda cand : cand.jpsi_vtx_ndof           ,
    'jpsi_vtx_prob'     : lambda cand : cand.jpsi_vtx_prob           ,
 
    'jpsi_cos2d'        : lambda cand : cand.jpsi_cos2d              ,
    'jpsi_cos3d'        : lambda cand : cand.jpsi_cos3d              ,
 
    'jpsi_lxy'          : lambda cand : cand.jpsi_lxy.value()        ,
    'jpsi_lxy_err'      : lambda cand : cand.jpsi_lxy.error()        ,
    'jpsi_lxy_sig'      : lambda cand : cand.jpsi_lxy.significance() ,
      
    'jpsi_lxyz'         : lambda cand : cand.jpsi_lxyz.value()       ,
    'jpsi_lxyz_err'     : lambda cand : cand.jpsi_lxyz.error()       ,
    'jpsi_lxyz_sig'     : lambda cand : cand.jpsi_lxyz.significance(),

    # ----- bachelor-muon signed 3D impact parameter (IPTools) -----

    # mu_ip3d_jpsi_pv     # jpsi-vertex direction, wrt PV
    # mu_ip3d_jpsi_sv     # jpsi-vertex direction, wrt the J/ψ vertex
    # mu_ip3d_sv_pv       # 3μ-vertex direction, wrt PV
    # mu_ip3d_sv_sv       # 3μ-vertex direction, wrt the 3μ vertex 

    'mu_ip3d_jpsi_pv'    : lambda cand : cand.mu_ip3d_jpsi_pv      ,
    'mu_ip3d_jpsi_pv_err': lambda cand : cand.mu_ip3d_jpsi_pv_err  ,
    'mu_ip3d_jpsi_pv_sig': lambda cand : cand.mu_ip3d_jpsi_pv_sig  ,

    'mu_ip3d_jpsi_sv'    : lambda cand : cand.mu_ip3d_jpsi_sv      ,
    'mu_ip3d_jpsi_sv_err': lambda cand : cand.mu_ip3d_jpsi_sv_err  ,
    'mu_ip3d_jpsi_sv_sig': lambda cand : cand.mu_ip3d_jpsi_sv_sig  ,

    'mu_ip3d_sv_pv'      : lambda cand : cand.mu_ip3d_sv_pv       ,
    'mu_ip3d_sv_pv_err'  : lambda cand : cand.mu_ip3d_sv_pv_err   ,
    'mu_ip3d_sv_pv_sig'  : lambda cand : cand.mu_ip3d_sv_pv_sig   ,

    'mu_ip3d_sv_sv'      : lambda cand : cand.mu_ip3d_sv_sv       ,
    'mu_ip3d_sv_sv_err'  : lambda cand : cand.mu_ip3d_sv_sv_err   ,
    'mu_ip3d_sv_sv_sig'  : lambda cand : cand.mu_ip3d_sv_sv_sig   ,

    'mu_dist_to_b_dir_jpsi'     : lambda cand : abs(cand.mu_dist_to_b_dir_jpsi)    ,
#     'mu_dist_to_b_dir_jpsi_err' : lambda cand : cand.mu_dist_to_b_dir_jpsi_err,
#     'mu_dist_to_b_dir_jpsi_sig' : lambda cand : cand.mu_dist_to_b_dir_jpsi_sig,
 
    'mu_dist_to_b_dir_sv'       : lambda cand : abs(cand.mu_dist_to_b_dir_sv)    ,
#     'mu_dist_to_b_dir_sv_err'   : lambda cand : cand.mu_dist_to_b_dir_sv_err,
#     'mu_dist_to_b_dir_sv_sig'   : lambda cand : cand.mu_dist_to_b_dir_sv_sig,

    'mu_dist_along_b_dir_jpsi_pv' : lambda cand : cand.mu_dist_along_b_dir_jpsi_pv,
    'mu_dist_along_b_dir_jpsi_sv' : lambda cand : cand.mu_dist_along_b_dir_jpsi_sv,
    'mu_dist_along_b_dir_sv_pv'   : lambda cand : cand.mu_dist_along_b_dir_sv_pv  ,
    'mu_dist_along_b_dir_sv_sv'   : lambda cand : cand.mu_dist_along_b_dir_sv_sv  ,

    # ----- per-candidate PV (AVF beamspot-constrained refit, signal muons out) -----
    # cand.pv_bs is the active PV reference: the refit when valid, else the Run2
    # hybrid PV; pv_refit_valid tells the two apart.
    'pv_refit_valid'    : lambda cand : int(cand.pv_refit_valid)  ,
    'pv_x'              : lambda cand : cand.pv_bs.position().x() ,
    'pv_y'              : lambda cand : cand.pv_bs.position().y() ,
    'pv_z'              : lambda cand : cand.pv_bs.position().z() ,
    'pv_ntrk'           : lambda cand : cand.pv_bs.tracksSize()   ,
    'pv_chi2'           : lambda cand : cand.pv_bs.chi2()         ,
    'pv_ndof'           : lambda cand : cand.pv_bs.ndof()         ,

    'trig_match'         : lambda cand : cand.trig_match       ,
}

helicity_branches = {
    'cos_theta_v_%s' % k : (lambda c, k=k: getattr(c, 'cos_theta_v_%s' % k, np.nan)) for k in ('jpsi','sv','coll','nu1','nu2')
}
helicity_branches.update({'cos_theta_l_%s' % k : (lambda c, k=k: getattr(c, 'cos_theta_l_%s' % k, np.nan)) for k in ('jpsi','sv','coll','nu1','nu2')})
helicity_branches.update({'chi_%s' % k : (lambda c, k=k: getattr(c, 'chi_%s' % k, np.nan)) for k in ('jpsi','sv','coll','nu1','nu2')})

cand_branches.update(helicity_branches)

# ----- custom PF isolation (bachelor mu + J/psi, recomputed vs the refit PV) -----
# generated to match the attributes set by RJpsiCandidate.compute_isolation:
#   <obj>_<quantity>_<RR>,  obj in {mu, jpsi}, RR in {03, 04}
# missing (e.g. pf not available) -> NaN via getattr default.
iso_branches = {}
for _obj in ('mu', 'jpsi'):
    for _rr in ('03', '04'):
        for _q in ('iso_ch', 'iso_ch_clean', 'iso_pu', 'iso_pu_clean',
                   'iso_nh', 'iso_ph', 'iso', 'iso_clean', 'reliso', 'reliso_clean'):
            _name = '%s_%s_%s' % (_obj, _q, _rr)
            iso_branches[_name] = (lambda c, n=_name: getattr(c, n, np.nan))
cand_branches.update(iso_branches)

branches =[]

for ibranch in event_branches.keys():
    branches.append(ibranch)

for idx in [1,2,3]:
    for ibr in muon_branches.keys():
        branches.append('mu%d_%s' %(idx, ibr))

for ibranch in cand_branches.keys():
    branches.append(ibranch)

for ibranch in bc_branches.keys():
    branches.append(ibranch)

# for ibranch in jpsi_branches.keys():
#     branches.append(ibranch)

paths = {}
paths['HLT_DoubleMu4_3_LowMass'] = ['hltDisplacedmumuFilterDoubleMu43LowMass', 'hltDisplacedmumuFilterDoubleMu43LowMass']

branches += paths
branches += [path+'_ps' for path in paths]
