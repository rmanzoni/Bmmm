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
from Bmmm.Analysis.CommonBranches import track_cov_corr_branches as _track_cov_corr
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

# the covflow-corrected matrix (--covflow) and its two per-track diagnostics.
# NaN / 0 when running without it, exactly as cov_scale_* is 1 without
# --cov-scale: the ntuple says for itself what was done to it.
cov_branches.update(_track_cov_corr)

muon_branches.update(cov_branches)

##########################################################################################
#####      Bc LIFETIME REWEIGHTING   (AN-20-223 Sec. 7.3.1)
##########################################################################################
# The Bc sample is generated with a lifetime that differs from the PDG one:
#
#     tau_MC  = 0.507            e-12 s     (generator setting)
#     tau_PDG = (0.510 +/- 0.010)e-12 s     (AN-20-223 Sec. 7.3.1)
#
# Every Bc-derived channel (both signals AND the whole cocktail: feeddown,
# J/psi+Hc, psi(2S), ...) is therefore reweighted from the generated proper-time
# exponential to the PDG one, and the +/-1 sigma targets provide the `ctau`
# shape nuisance of the fit (Table 5, syst. #4).
#
# The natural variable is the proper decay length ell = L3D/(beta*gamma) [cm],
# i.e. c*t_proper, which is exactly the existing gen_b_ct branch. The weight is
# the ratio of the two exponential pdfs in ell,
#
#     w(ell) = (ctau_MC / ctau_tgt) * exp( ell * (1/ctau_MC - 1/ctau_tgt) )
#
# whose mean is 1 by construction over the *generated* ell spectrum (shape
# only). After selection <w> deviates from 1 at the permille level: that is the
# genuine efficiency dependence on the lifetime, not a normalisation bug.
_C_CM_PER_S         = 2.99792458e10        # speed of light [cm/s]
BC_TAU_MC_S         = 0.507e-12            # generated Bc lifetime [s]  <-- verify vs Run 3 DEC table
BC_TAU_PDG_S        = 0.510e-12            # PDG central value [s]
BC_TAU_PDG_UNC_S    = 0.010e-12            # PDG uncertainty [s]

CTAU_BC_MC_CM       = _C_CM_PER_S *  BC_TAU_MC_S                        # 0.0151995 cm (generated)
CTAU_BC_PDG_CM      = _C_CM_PER_S *  BC_TAU_PDG_S                       # 0.0152894 cm (nominal target)
CTAU_BC_PDG_UP_CM   = _C_CM_PER_S * (BC_TAU_PDG_S + BC_TAU_PDG_UNC_S)   # 0.0155892 cm
CTAU_BC_PDG_DOWN_CM = _C_CM_PER_S * (BC_TAU_PDG_S - BC_TAU_PDG_UNC_S)   # 0.0149896 cm


def bc_proper_decay_length(ib):
    '''Bc proper decay length L3D/(beta*gamma) in cm (== c*t_proper).

    Single source of truth for gen_b_ct and for the lifetime weights, so the
    weight can never be evaluated at a different ell than the one written out.
    daughter(0) of the last-copy Bc carries the decay vertex.
    '''
    dx   = ib.daughter(0).vx() - ib.vx()
    dy   = ib.daughter(0).vy() - ib.vy()
    dz   = ib.daughter(0).vz() - ib.vz()
    l3d  = np.sqrt(dx*dx + dy*dy + dz*dz)
    return l3d / ib.p4().Beta() / ib.p4().Gamma()


def bc_ctau_weight(ib, ctau_target_cm):
    '''Per-event Bc lifetime weight, target vs generated exponential in ell.

    Returns NaN (not 1.0) when ell is unusable: an event that HAS a Bc
    (gen_bc_decay is not NaN) but a NaN weight flags a broken gen record
    instead of silently entering the histograms unweighted.
    '''
    ell = bc_proper_decay_length(ib)
    if not np.isfinite(ell) or ell < 0.:
        return np.nan
    return (CTAU_BC_MC_CM / ctau_target_cm) * \
           np.exp(ell * (1. / CTAU_BC_MC_CM - 1. / ctau_target_cm))


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
    'gen_b_ct'     :  lambda ib : bc_proper_decay_length(ib),

    # --- Bc lifetime reweighting, AN-20-223 Sec. 7.3.1. Filled for EVERY event
    #     with a gen Bc, i.e. the whole cocktail, not just the two signals.
    'gen_bc_ctau_weight'      :  lambda ib : bc_ctau_weight(ib, CTAU_BC_PDG_CM)     , # 0.507 -> 0.510 ps (nominal)
    'gen_bc_ctau_weight_up'   :  lambda ib : bc_ctau_weight(ib, CTAU_BC_PDG_UP_CM)  , # 0.507 -> 0.520 ps (ctau up)
    'gen_bc_ctau_weight_down' :  lambda ib : bc_ctau_weight(ib, CTAU_BC_PDG_DOWN_CM), # 0.507 -> 0.500 ps (ctau down)

    'gen_pv_x'     :  lambda ib : ib.vx()    ,
    'gen_pv_y'     :  lambda ib : ib.vy()    ,
    'gen_pv_z'     :  lambda ib : ib.vz()    ,

    'gen_sv_x'     :  lambda ib : ib.daughter(0).vx(),
    'gen_sv_y'     :  lambda ib : ib.daughter(0).vy(),
    'gen_sv_z'     :  lambda ib : ib.daughter(0).vz(),

    'gen_lxy'      :  lambda ib : np.sqrt( (ib.daughter(0).vx() - ib.vx())**2 + (ib.daughter(0).vy() - ib.vy())**2 ) ,
    'gen_lxyz'     :  lambda ib : np.sqrt( (ib.daughter(0).vx() - ib.vx())**2 + (ib.daughter(0).vy() - ib.vy())**2 + (ib.daughter(0).vz() - ib.vz())**2 ) ,
}

# ----- Hammer FF-reweighting inputs (pre-FSR gen leaves; NaN off signal) -----
# Attributes ham_<particle>_<component> are attached to the bc gen particle by
# RJPsiGenHistory.gen_hammer_p4 (called in the inspector's setup_event_gen); the
# getters read them with an NaN default so non-signal / non-MC rows stay clean.
# The Bc four-momentum is NOT duplicated -- use the gen_b_* branches above.
for _ham_p in ('mup', 'mum', 'lep', 'nu', 'taumu', 'taunut', 'taunum'):
    for _ham_c in ('pt', 'eta', 'phi', 'mass', 'pdgid'):
        _ham_attr = 'ham_%s_%s' % (_ham_p, _ham_c)
        bc_branches['gen_%s' % _ham_attr] = (lambda ib, n=_ham_attr: getattr(ib, n, np.nan))

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

