import ROOT
import numpy as np
from Bmmm.Analysis.B4MuBranches import event_branches, cand_branches, muon_branches, bs_branches, jpsi_branches, phi_branches, paths

event_branches['nmuons'] = lambda ev : ev.nmuons
event_branches['ntracks'] = lambda ev : ev.ntracks

track_branches = {
    'pt'             :  lambda itk : itk.pt()                            ,
    'eta'            :  lambda itk : itk.eta()                           , 
    'phi'            :  lambda itk : itk.phi()                           ,
    'e'              :  lambda itk : itk.energy()                        ,
    'rf_pt'          :  lambda itk : itk.rfp4.pt()                       ,
    'rf_eta'         :  lambda itk : itk.rfp4.eta()                      , 
    'rf_phi'         :  lambda itk : itk.rfp4.phi()                      ,
    'rf_e'           :  lambda itk : itk.energy()                        ,
    'mass'           :  lambda itk : itk.mass()                          ,
    'charge'         :  lambda itk : itk.charge()                        ,
    'dxy'            :  lambda itk : itk.bestTrack().dxy(itk.pv.position()),
    'dxy_e'          :  lambda itk : itk.bestTrack().dxyError(itk.pv.position(), itk.pv.error()),
    'dxy_sig'        :  lambda itk : itk.bestTrack().dxy(itk.pv.position()) / itk.bestTrack().dxyError(itk.pv.position(), itk.pv.error()),
    'dz'             :  lambda itk : itk.bestTrack().dz(itk.pv.position()),
    'dz_e'           :  lambda itk : itk.bestTrack().dzError(),
    'dz_sig'         :  lambda itk : itk.bestTrack().dz(itk.pv.position()) / itk.bestTrack().dzError(),
    'bs_dxy'         :  lambda itk : itk.bestTrack().dxy(itk.bs.position()),
    'bs_dxy_e'       :  lambda itk : itk.bestTrack().dxyError(itk.bs.position(), itk.bs.error()),
    'bs_dxy_sig'     :  lambda itk : itk.bestTrack().dxy(itk.bs.position()) / itk.bestTrack().dxyError(itk.bs.position(), itk.bs.error()),
    'cov_pos_def'    :  lambda itk : itk.is_cov_pos_def,
    'jet_pt'         :  lambda itk : itk.jet.pt()      if hasattr(itk, 'jet') else np.nan,
    'jet_eta'        :  lambda itk : itk.jet.eta()     if hasattr(itk, 'jet') else np.nan,
    'jet_phi'        :  lambda itk : itk.jet.phi()     if hasattr(itk, 'jet') else np.nan,
    'jet_e'          :  lambda itk : itk.jet.energy()  if hasattr(itk, 'jet') else np.nan,
    'gen_pt'         :  lambda itk : itk.genp.pt()     if hasattr(itk, 'genp') else np.nan,
    'gen_eta'        :  lambda itk : itk.genp.eta()    if hasattr(itk, 'genp') else np.nan,
    'gen_phi'        :  lambda itk : itk.genp.phi()    if hasattr(itk, 'genp') else np.nan,
    'gen_e'          :  lambda itk : itk.genp.energy() if hasattr(itk, 'genp') else np.nan,
    'gen_pdgid'      :  lambda itk : itk.genp.pdgId()  if hasattr(itk, 'genp') else np.nan,
    'gen_charge'     :  lambda itk : itk.genp.charge() if hasattr(itk, 'genp') else np.nan,
}

branches =[]

for ibranch in event_branches.keys():
    branches.append(ibranch)

for idx in [1,2]:
    for ibr in muon_branches.keys():
        branches.append('mu%d_%s' %(idx, ibr))
    for ibr in track_branches.keys():
        branches.append('tk%d_%s' %(idx, ibr))

for ibranch in cand_branches.keys():
    branches.append(ibranch)

for ibranch in bs_branches.keys():
    branches.append(ibranch)

for ibranch in jpsi_branches.keys():
    branches.append(ibranch)

for ibranch in phi_branches.keys():
    branches.append(ibranch)


branches += paths
branches += [path+'_ps' for path in paths]
