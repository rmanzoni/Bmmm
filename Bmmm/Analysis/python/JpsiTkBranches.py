import numpy as np

# J/psi + charged-track branch schema (B+ -> J/psi K+ / Bc+ -> J/psi pi+).
#
# The event block, the per-muon block (mu1, mu2 = the two J/psi muons), the HLT
# paths and safe_get are imported UNCHANGED from the shared JpsiChargedBranches,
# so the muon side and the event side are identical to the J/psi mu ntuple.
#
# The candidate block is the J/psi mu cand block reused verbatim for the KAON
# (primary, unprefixed) hypothesis -- so every well-defined quantity is directly
# comparable with the RJpsi ntuple -- with three changes:
#   * the bachelor IP / jet-track-distance / isolation branches are renamed
#     mu_* -> k_* (the getters are unchanged: they still read the cand.mu_*
#     attributes the base class fills for the bachelor);
#   * the RJpsi signal-only gen-truth cand branches (q2_gen, m_miss2_gen,
#     gen_hb_*) are dropped -- they are Bc->J/psi mu nu specific;
#   * a full PION-hypothesis mirror is appended under the pi_ prefix (the
#     hypothesis-dependent block only; the J/psi-vertex block, the PV and the
#     isolation are shared and NOT duplicated).
from Bmmm.Analysis.JpsiChargedBranches import (
    event_branches, muon_branches, paths, safe_get,
)
from Bmmm.Analysis.JpsiMuBranches import cand_branches as _mu_cand_branches

##########################################################################################
#####      KAON (bachelor track) quantities  ->  k_<name>
##########################################################################################
# Same track quantities the other track-based ntuples in this package persist:
# kinematics with the CHARGED-KAON mass hypothesis, dxy/dz (wrt PV and BS) with
# errors and significances, the covariance-pos-def flag, jet matching and (MC)
# gen matching. The muon-only fields (muon IDs, muon-POG PFIso) are absent (there
# is no muon to compute them on); the bachelor's custom PF isolation lives in the
# candidate-level k_iso_* branches instead.
k_branches = {
    'pt'          : lambda itk : itk.pt()     ,
    'eta'         : lambda itk : itk.eta()    ,
    'phi'         : lambda itk : itk.phi()    ,
    'e'           : lambda itk : itk.energy() ,
    'mass'        : lambda itk : itk.mass()   ,
    'charge'      : lambda itk : itk.charge() ,

    'dxy'         : lambda itk : itk.bestTrack().dxy(itk.pv.position()),
    'dxy_e'       : lambda itk : itk.bestTrack().dxyError(itk.pv.position(), itk.pv.error()),
    'dxy_sig'     : lambda itk : itk.bestTrack().dxy(itk.pv.position()) / itk.bestTrack().dxyError(itk.pv.position(), itk.pv.error()),
    'dz'          : lambda itk : itk.bestTrack().dz(itk.pv.position()),
    'dz_e'        : lambda itk : itk.bestTrack().dzError(),
    'dz_sig'      : lambda itk : itk.bestTrack().dz(itk.pv.position()) / itk.bestTrack().dzError(),
    'bs_dxy'      : lambda itk : itk.bestTrack().dxy(itk.bs.position()),
    'bs_dxy_e'    : lambda itk : itk.bestTrack().dxyError(itk.bs.position(), itk.bs.error()),
    'bs_dxy_sig'  : lambda itk : itk.bestTrack().dxy(itk.bs.position()) / itk.bestTrack().dxyError(itk.bs.position(), itk.bs.error()),

    'cov_pos_def' : lambda itk : itk.is_cov_pos_def,

    'jet_pt'      : lambda itk : itk.jet.pt()      if hasattr(itk, 'jet') else np.nan,
    'jet_eta'     : lambda itk : itk.jet.eta()     if hasattr(itk, 'jet') else np.nan,
    'jet_phi'     : lambda itk : itk.jet.phi()     if hasattr(itk, 'jet') else np.nan,
    'jet_e'       : lambda itk : itk.jet.energy()  if hasattr(itk, 'jet') else np.nan,

    'gen_pt'      : lambda itk : itk.gen_match.pt()     if hasattr(itk, 'gen_match') else np.nan,
    'gen_eta'     : lambda itk : itk.gen_match.eta()    if hasattr(itk, 'gen_match') else np.nan,
    'gen_phi'     : lambda itk : itk.gen_match.phi()    if hasattr(itk, 'gen_match') else np.nan,
    'gen_e'       : lambda itk : itk.gen_match.energy() if hasattr(itk, 'gen_match') else np.nan,
    'gen_pdgid'   : lambda itk : itk.gen_match.pdgId()  if hasattr(itk, 'gen_match') else np.nan,
    'gen_charge'  : lambda itk : itk.gen_match.charge() if hasattr(itk, 'gen_match') else np.nan,
    'gen_dr'      : lambda itk : itk.gen_dr             if hasattr(itk, 'gen_dr')   else np.nan,
}

##########################################################################################
#####      CANDIDATE block: KAON hypothesis (unprefixed), reused from J/psi mu
##########################################################################################
# Drop the RJpsi signal-only gen-truth cand branches; rename the bachelor
# IP / jet-track-distance / isolation branches mu_* -> k_* (getters unchanged).
_DROP_CAND = {
    'q2_gen', 'm_miss2_gen',
    'gen_hb_same_mother', 'gen_hb_jpsi_b_pdgid',
    'gen_hb_bachelor_b_pdgid', 'gen_hb_bachelor_mom_pdgid',
}

def _bachelor_rename(key):
    '''mu_* -> k_* for the bachelor IP / jet-track-distance / isolation branches;
    everything else (including the Bc-frame energies mu_b_e_*, mu_jpsi_e, which
    are kept RJpsi-identical for comparability) is left untouched.'''
    for pfx in ('mu_ip3d_', 'mu_dist_', 'mu_iso', 'mu_reliso'):
        if key.startswith(pfx):
            return 'k_' + key[3:]
    return key

cand_branches = {}
for _k, _v in _mu_cand_branches.items():
    if _k in _DROP_CAND:
        continue
    cand_branches[_bachelor_rename(_k)] = _v

##########################################################################################
#####      CANDIDATE block: PION hypothesis mirror  ->  pi_<name>
##########################################################################################
# Only the bachelor-mass-DEPENDENT quantities are duplicated (the J/psi-vertex
# block, the PV and the isolation are shared). Composite/visible p4 come from
# cand.pi_p4 / cand.pi_rfp4, the bachelor p4 from cand.pi_bachelor_p4; the
# 3-body-pi vertex, flight directions, IP grid, jet-track distances, mcorr and
# equal-velocity block from the pi_-prefixed attributes set by
# JpsiChargedCandidate.compute_alt_bachelor; the neutrino solutions and helicity
# angles from the pi_-prefixed attributes set by the inspector.
pi_branches = {
    # composite (Bc+ -> J/psi pi+) kinematics
    'pi_mass'    : lambda c : c.pi_p4.mass() ,
    'pi_pt'      : lambda c : c.pi_p4.pt()   ,
    'pi_eta'     : lambda c : c.pi_p4.eta()  ,
    'pi_phi'     : lambda c : c.pi_p4.phi()  ,

    'pi_rf_mass' : lambda c : c.pi_rfp4.mass(),
    'pi_rf_pt'   : lambda c : c.pi_rfp4.pt()  ,
    'pi_rf_eta'  : lambda c : c.pi_rfp4.eta() ,
    'pi_rf_phi'  : lambda c : c.pi_rfp4.phi() ,

    # collinear / equal-velocity q2, m_miss2, bachelor energy in the Bc frame
    'pi_q2_coll'      : lambda c : (c.pi_p4_collinear - c.jpsi_rfp4).mass2(),
    'pi_m_miss2_coll' : lambda c : (c.pi_p4_collinear - c.jpsi_rfp4 - c.pi_bachelor_p4).mass2(),
    'pi_mu_b_e_coll'  : lambda c : c.pi_p4_collinear.Dot(c.pi_bachelor_p4) / c.pi_p4_collinear.mass(),

    'pi_q2_jpsi'      : lambda c : (c.pi_bc_full_p4_jpsi - c.jpsi_rfp4).mass2(),
    'pi_q2_sv'        : lambda c : (c.pi_bc_full_p4_sv   - c.jpsi_rfp4).mass2(),
    'pi_m_miss2_jpsi' : lambda c : (c.pi_bc_full_p4_jpsi - c.jpsi_rfp4 - c.pi_bachelor_p4).mass2(),
    'pi_m_miss2_sv'   : lambda c : (c.pi_bc_full_p4_sv   - c.jpsi_rfp4 - c.pi_bachelor_p4).mass2(),
    'pi_mu_b_e_jpsi'  : lambda c : c.pi_bc_full_p4_jpsi.Dot(c.pi_bachelor_p4) / c.pi_bc_full_p4_jpsi.mass(),
    'pi_mu_b_e_sv'    : lambda c : c.pi_bc_full_p4_sv  .Dot(c.pi_bachelor_p4) / c.pi_bc_full_p4_sv  .mass(),
    'pi_mu_jpsi_e'    : lambda c : c.jpsi_rfp4.Dot(c.pi_bachelor_p4) / c.jpsi_rfp4.mass(),

    'pi_nu1_q2_jpsi'  : lambda c : (c.pi_math1_b_p4_jpsi - c.jpsi_rfp4).mass2(),
    'pi_nu2_q2_jpsi'  : lambda c : (c.pi_math2_b_p4_jpsi - c.jpsi_rfp4).mass2(),
    'pi_nu1_q2_sv'    : lambda c : (c.pi_math1_b_p4_sv - c.jpsi_rfp4).mass2(),
    'pi_nu2_q2_sv'    : lambda c : (c.pi_math2_b_p4_sv - c.jpsi_rfp4).mass2(),
    'pi_nu1_mu_b_e_jpsi' : lambda c : c.pi_math1_b_p4_jpsi.Dot(c.pi_bachelor_p4) / c.pi_math1_b_p4_jpsi.mass(),
    'pi_nu2_mu_b_e_jpsi' : lambda c : c.pi_math2_b_p4_jpsi.Dot(c.pi_bachelor_p4) / c.pi_math2_b_p4_jpsi.mass(),
    'pi_nu1_mu_b_e_sv'   : lambda c : c.pi_math1_b_p4_sv  .Dot(c.pi_bachelor_p4) / c.pi_math1_b_p4_sv  .mass(),
    'pi_nu2_mu_b_e_sv'   : lambda c : c.pi_math2_b_p4_sv  .Dot(c.pi_bachelor_p4) / c.pi_math2_b_p4_sv  .mass(),

    # corrected mass / momentum projections along each flight direction
    'pi_p4_par_jpsi'  : lambda c : c.pi_p4_par_jpsi  ,
    'pi_p4_perp_jpsi' : lambda c : c.pi_p4_perp_jpsi ,
    'pi_mcorr_jpsi'   : lambda c : c.pi_mcorr_jpsi   ,
    'pi_p4_par_sv'    : lambda c : c.pi_p4_par_sv    ,
    'pi_p4_perp_sv'   : lambda c : c.pi_p4_perp_sv   ,
    'pi_mcorr_sv'     : lambda c : c.pi_mcorr_sv     ,

    # pion (Bc) secondary vertex (2mu + pi), pi_-prefixed
    'pi_sv_good'  : lambda c : c.pi_good_vtx           ,
    'pi_sv_x'     : lambda c : c.pi_vtx.position().x() ,
    'pi_sv_y'     : lambda c : c.pi_vtx.position().y() ,
    'pi_sv_z'     : lambda c : c.pi_vtx.position().z() ,
    'pi_sv_chi2'  : lambda c : c.pi_vtx_chi2           ,
    'pi_sv_ndof'  : lambda c : c.pi_vtx_ndof           ,
    'pi_sv_prob'  : lambda c : c.pi_vtx_prob           ,
    'pi_cos2d'    : lambda c : c.pi_cos2d              ,
    'pi_cos3d'    : lambda c : c.pi_cos3d              ,
    'pi_cos3dbs'  : lambda c : c.pi_cos3dbs            ,
    'pi_lxy'      : lambda c : c.pi_lxy.value()        ,
    'pi_lxy_err'  : lambda c : c.pi_lxy.error()        ,
    'pi_lxy_sig'  : lambda c : c.pi_lxy.significance() ,
    'pi_lxyz'     : lambda c : c.pi_lxyz.value()       ,
    'pi_lxyz_err' : lambda c : c.pi_lxyz.error()       ,
    'pi_lxyz_sig' : lambda c : c.pi_lxyz.significance(),
}

# ----- mathematical neutrino solutions (pion hypothesis) -----
for _lbl in ('jpsi', 'sv'):
    for _si, _idx in ((1, 0), (2, 1)):
        _base = 'pi_nu%d_%s' % (_si, _lbl)
        _sols = 'pi_sols_%s' % _lbl
        pi_branches['%s_bc_e'  % _base] = (lambda c, s=_sols, i=_idx : getattr(c, s)[i].p4_parent.energy())
        pi_branches['%s_bc_pt' % _base] = (lambda c, s=_sols, i=_idx : getattr(c, s)[i].p4_parent.pt())
        pi_branches['%s_bc_eta'% _base] = (lambda c, s=_sols, i=_idx : getattr(c, s)[i].p4_parent.eta())
        pi_branches['%s_bc_phi'% _base] = (lambda c, s=_sols, i=_idx : getattr(c, s)[i].p4_parent.phi())
        pi_branches['%s_pz'    % _base] = (lambda c, s=_sols, i=_idx : getattr(c, s)[i].pz)
        pi_branches['%s_e'     % _base] = (lambda c, s=_sols, i=_idx : getattr(c, s)[i].p4_nu.energy())
        pi_branches['%s_pt'    % _base] = (lambda c, s=_sols, i=_idx : getattr(c, s)[i].p4_nu.pt())
        pi_branches['%s_eta'   % _base] = (lambda c, s=_sols, i=_idx : getattr(c, s)[i].p4_nu.eta())
        pi_branches['%s_phi'   % _base] = (lambda c, s=_sols, i=_idx : getattr(c, s)[i].p4_nu.phi())

# ----- bachelor signed-IP 3D grid (pion hypothesis)  ->  pi_k_ip3d_* -----
for _d in ('jpsi', 'sv'):
    for _r in ('pv', 'sv'):
        for _suf in ('', '_err', '_sig'):
            _br  = 'pi_k_ip3d_%s_%s%s'  % (_d, _r, _suf)
            _att = 'pi_mu_ip3d_%s_%s%s' % (_d, _r, _suf)
            pi_branches[_br] = (lambda c, a=_att : getattr(c, a, np.nan))

# ----- bachelor jet-track distances (pion hypothesis)  ->  pi_k_dist_* -----
pi_branches['pi_k_dist_to_b_dir_jpsi'] = lambda c : abs(c.pi_mu_dist_to_b_dir_jpsi)
pi_branches['pi_k_dist_to_b_dir_sv']   = lambda c : abs(c.pi_mu_dist_to_b_dir_sv)
for _d in ('jpsi', 'sv'):
    for _r in ('pv', 'sv'):
        _br  = 'pi_k_dist_along_b_dir_%s_%s'  % (_d, _r)
        _att = 'pi_mu_dist_along_b_dir_%s_%s' % (_d, _r)
        pi_branches[_br] = (lambda c, a=_att : getattr(c, a, np.nan))

# ----- helicity angles (pion hypothesis) -----
for _q in ('cos_theta_v', 'cos_theta_l', 'chi'):
    for _k in ('jpsi', 'sv', 'coll', 'nu1', 'nu2'):
        _br = 'pi_%s_%s' % (_q, _k)
        pi_branches[_br] = (lambda c, a=_br : getattr(c, a, np.nan))

cand_branches.update(pi_branches)

##########################################################################################
#####      FLAT BRANCH LIST (schema order)
##########################################################################################
branches = []

for ibranch in event_branches.keys():
    branches.append(ibranch)

# mu1 / mu2 = the two J/psi muons (identical to the J/psi mu ntuple)
for idx in [1, 2]:
    for ibr in muon_branches.keys():
        branches.append('mu%d_%s' % (idx, ibr))

# k = the bachelor track (track quantities, no muon IDs)
for ibr in k_branches.keys():
    branches.append('k_%s' % ibr)

for ibranch in cand_branches.keys():
    branches.append(ibranch)

branches += list(paths)
branches += [path + '_ps' for path in paths]
