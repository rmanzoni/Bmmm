import ROOT
import numpy as np

from Bmmm.Analysis.utils import (
    COV_ELEMENT_NAMES, COV_INDEX_PAIRS, COV_PARAM_NAMES, COV_NO_SCALE,
)
# the quantities every channel defines the same way, plus the filling helper.
# safe_get is re-exported so the existing
#   from Bmmm.Analysis.JpsiChargedBranches import ..., safe_get
# in JpsiMuBranches / JpsiTkBranches keeps working.
from Bmmm.Analysis.CommonBranches import event_branches as _common_event
from Bmmm.Analysis.CommonBranches import muon_branches as _common_muon
from Bmmm.Analysis.CommonBranches import track_cov_branches as _track_cov
from Bmmm.Analysis.CommonBranches import safe_get

# Shared branch definitions for the J/psi + charged-object ntuples. These are
# channel-agnostic and imported verbatim by BOTH JpsiMuBranches (J/psi mu) and
# JpsiTkBranches (J/psi + track), so the event block, the per-muon block, the Bc
# gen-truth block, the J/psi gen block, the HLT paths and safe_get are identical
# across the two channels.

# Shared getters pulled in by name, channel-specific ones defined here. Written
# out entry by entry rather than as common + extras because the ORDER is the
# ntuple's existing layout and is worth preserving exactly.
event_branches = {
    'run'     : _common_event['run']    ,
    'lumi'    : _common_event['lumi']   ,
    'event'   : _common_event['event']  ,

    'ncands'  : _common_event['ncands'] ,

#     'qscale'  : lambda ev : ev.genInfo.qScale()                     ,
    'npv'     : _common_event['npv']    ,
    'npu'     : _common_event['npu']    ,
    'nti'     : _common_event['nti']    ,

    'bs_x0'   : _common_event['bs_x0']  ,
    'bs_x0e'  : lambda ev : ev.bs.x0Error()                         ,
    'bs_y0'   : _common_event['bs_y0']  ,
    'bs_y0e'  : lambda ev : ev.bs.y0Error()                         ,
    'bs_z0'   : _common_event['bs_z0']  ,
    'bs_z0e'  : lambda ev : ev.bs.z0Error()                         ,
}

# Built from the shared per-muon block, with the RJpsi-only quantities inserted
# where they have always been: the refitted-J/psi rf_* right after the raw
# kinematics, the gen bookkeeping after gen_pdgid. Assembled key by key rather
# than as common + extras so the existing branch order is preserved exactly.
muon_branches = {}
for _k in ('pt', 'eta', 'phi', 'e'):
    muon_branches[_k] = _common_muon[_k]
muon_branches.update({
    'rf_pt'          :  lambda imu : imu.jpsi_rfp4.pt()                  ,
    'rf_eta'         :  lambda imu : imu.jpsi_rfp4.eta()                 , 
    'rf_phi'         :  lambda imu : imu.jpsi_rfp4.phi()                 ,
    'rf_e'           :  lambda imu : imu.jpsi_rfp4.energy()             ,
})
for _k, _v in _common_muon.items():
    if _k in ('pt', 'eta', 'phi', 'e'):
        continue
    muon_branches[_k] = _v
muon_branches.update({
    'gen_charge'     :  lambda imu : imu.gen_match.charge() if hasattr(imu, 'gen_match') else np.nan,
    'gen_role'       :  lambda imu : imu.gen_role,
    'gen_dr'         :  lambda imu : imu.gen_dr  ,
})

##########################################################################################
#####      TRACK COVARIANCE MATRIX BLOCK  (shared by muons and bachelor tracks)
##########################################################################################
# The 15 independent elements of the 5x5 curvilinear covariance of the object's
# best track, RAW -- exactly as reconstruction stored them, before any scaling.
# These are the inputs to the data/MC covariance-mismodelling measurement, hence
# raw: the correction is derived FROM them, so persisting a corrected version
# here would be circular.
#
#   <obj>_cov_<par_i>_<par_j>   with par in (qoverp, lambda, phi, dxy, dsz)
#
# sigma_i = sqrt(cov_<par_i>_<par_i>), rho_ij = cov_ij / (sigma_i sigma_j): both
# trivially derived offline, so they are not duplicated in the ntuple. Note these
# are the BARE track uncertainties -- unlike the dxy_e / dz_e branches above,
# which fold in the primary-vertex error.
#
# <obj>_cov_scale_<par> records the scale factor actually applied to that
# parameter before the vertex fit (1 when running without --cov-scale, i.e.
# always while the measurement is still being made). It is written so an ntuple
# says for itself whether, and by how much, it was corrected.
cov_branches = dict(_track_cov)          # the 15 elements, shared with every channel
for _i, _par in enumerate(COV_PARAM_NAMES):
    cov_branches['cov_scale_%s' % _par] = (
        lambda iobj, i=_i : getattr(iobj, 'cov_scale', COV_NO_SCALE)[i])

muon_branches.update(cov_branches)

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

