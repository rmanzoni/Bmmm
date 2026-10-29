'''
R(Ds) ntuple schema: Bs -> Ds(-> phi(-> K K) pi) mu (nu), one row per candidate.

Blocks, in schema order:

  event          run, lumi, event, ncands, npv, npu, nti, beamspot  (as the
                 J/psi ntuples), era (1-4 = 2018 A-D), bph_part (1-6), the L1
                 seeds of the parking paths (<seed>, <seed>_ps), the L1
                 prescale range of each HLT path (<path>_l1min_ps, _l1max_ps),
                 the MC gen PV
  mu_*           the shared per-muon block (CommonBranches) + covariance +
                 trigger matching (mu_trig_dr, mu_trig_match, mu_<path>_tag) +
                 momentum calibration (mu_calib_f, mu_calib_z)
  k1_*, k2_*, pi_*   per track: calibrated and raw kinematics, packed-candidate
                 information, IPs, covariance, hit content, jet and gen match
  candidate      scheme-independent quantities (masses, charges, reflections,
                 PV, isolation, cand_idx), then everything that depends on the
                 kinematic fits, once per scheme with the prefix cons_ / phic_ /
                 free_
                 (see RDsCandidate)
  gen            MC event block: signal code, Bs kinematics and flight, gen
                 analysis quantities, Hammer inputs and weights
  HLT            the ten parking paths and their prescales

All getters run through safe_get: a failed fit or a missing gen match is a NaN.
'''

import numpy as np

from Bmmm.Analysis.CommonBranches import muon_branches as _common_muon
from Bmmm.Analysis.CommonBranches import track_cov_branches, safe_get, cov_track
from Bmmm.Analysis.JpsiChargedBranches import event_branches as _jpsi_event
from Bmmm.Analysis.TrackHitContent import hit_content_branches, best_track_kinematics_branches
from Bmmm.Analysis.BParking2018Trigger import paths
from Bmmm.Analysis.utils import COV_PARAM_NAMES
from Bmmm.Analysis.RDsCandidate import SCHEMES, SCHEME_VERTICES, HYPOTHESES, HYP_OBS
from Bmmm.Analysis.RDsGenHistory import CAND_GEN_NAMES, HAM_BRANCHES, GEN_OBS_NAMES
from Bmmm.Analysis.HammerDsFF import BRANCH_NAMES as HAMMER_BRANCH_NAMES

# the L1 seeds of the ten parking paths (all ten share them; checked against the
# brilcalc tables by L1Seeds2018 at the start of every job)
L1_SEEDS = (
    'L1_SingleMu6er1p5' , 'L1_SingleMu7er1p5' , 'L1_SingleMu8er1p5' ,
    'L1_SingleMu9er1p5' , 'L1_SingleMu10er1p5', 'L1_SingleMu12er1p5',
    'L1_SingleMu14er1p5', 'L1_SingleMu18er1p5', 'L1_SingleMu22'     ,
    'L1_SingleMu25'     ,
)

##########################################################################################
#####      EVENT
##########################################################################################
event_branches = dict(_jpsi_event)

l1_branches = []
for _s in L1_SEEDS:
    l1_branches += [_s, _s + '_ps']
l1_path_branches = []
for _p in paths:
    l1_path_branches += [_p + '_l1min_ps', _p + '_l1max_ps']

event_extra_branches = ['era', 'bph_part'] + l1_branches + l1_path_branches + \
                       ['gen_pv_x', 'gen_pv_y', 'gen_pv_z']

##########################################################################################
#####      PER TRACK (k1, k2, pi) and per-muon additions
##########################################################################################
# IPs: the same getters as the muon block (they read obj.pv, obj.bs and the
# track the fits used, CommonBranches.cov_track)
_IP_KEYS = ('dxy', 'dxy_e', 'dxy_sig', 'dz', 'dz_e', 'dz_sig',
            'bs_dxy', 'bs_dxy_e', 'bs_dxy_sig', 'dxy_e_raw', 'dz_e_raw', 'bs_dxy_e_raw')

track_branches = {
    'pt'              : lambda t : t.pt()     ,
    'eta'             : lambda t : t.eta()    ,
    'phi'             : lambda t : t.phi()    ,
    'e'               : lambda t : t.energy() ,
    'mass'            : lambda t : t.mass()   ,
    'charge'          : lambda t : t.charge() ,
    'px'              : lambda t : t.px()     ,
    'py'              : lambda t : t.py()     ,
    'pz'              : lambda t : t.pz()     ,
    'pt_raw'          : lambda t : t.raw_pt() ,
    'pdgid'           : lambda t : t.pdgId()  ,
    'from_pv'         : lambda t : t.fromPV() ,
    'pv_assoc_q'      : lambda t : int(t.pvAssociationQuality()),
    'high_purity'     : lambda t : t.trackHighPurity(),
    'lost_inner_hits' : lambda t : t.lostInnerHits(),
    'is_lost'         : lambda t : t.record.collection == 'lost',
}
for _k in _IP_KEYS:
    track_branches[_k] = _common_muon[_k]

# shared by tracks and muon
common_track_extras = {
    'calib_f'      : lambda t : t.record.f,
    'calib_z'      : lambda t : t.record.z,
    'cov_pos_def'  : lambda t : t.record.is_cov_pos_def,
    'cov_fixed'    : lambda t : t.record.cov_fixed,
}
for _i, _par in enumerate(COV_PARAM_NAMES):
    common_track_extras['cov_scale_%s' % _par] = (lambda t, i=_i : t.record.cov_scale[i])

track_branches.update(hit_content_branches)
track_branches.update(best_track_kinematics_branches)
track_branches.update(common_track_extras)
track_branches.update(track_cov_branches)
track_branches.update({
    'jet_pt'          : lambda t : t.jet.pt()     if hasattr(t, 'jet') else np.nan,
    'jet_eta'         : lambda t : t.jet.eta()    if hasattr(t, 'jet') else np.nan,
    'jet_phi'         : lambda t : t.jet.phi()    if hasattr(t, 'jet') else np.nan,
    'jet_e'           : lambda t : t.jet.energy() if hasattr(t, 'jet') else np.nan,
})

gen_track_branches = {
    'gen_pt'          : lambda t : t.gen_match.pt()    ,
    'gen_eta'         : lambda t : t.gen_match.eta()   ,
    'gen_phi'         : lambda t : t.gen_match.phi()   ,
    'gen_pdgid'       : lambda t : t.gen_match.pdgId() ,
    'gen_dr'          : lambda t : t.gen_dr            ,
    'gen_mother_pdgid': lambda t : t.gen_mother_pdgid  ,
}
track_branches.update(gen_track_branches)

muon_branches = dict(_common_muon)
muon_branches.update(common_track_extras)
muon_branches.update(track_cov_branches)
muon_branches['gen_mother_pdgid'] = gen_track_branches['gen_mother_pdgid']
muon_branches['gen_dr']           = gen_track_branches['gen_dr']
muon_branches['pt_raw']           = lambda t : t.raw_pt()
# trigger matching, filled by the inspector
muon_trigger_names = ['trig_dr', 'trig_match'] + ['%s_tag' % p for p in paths]

##########################################################################################
#####      CANDIDATE
##########################################################################################
# Most candidate quantities are computed into cand.out by RDsCandidate.compute,
# so their names are listed here and read with cand.out.get(name, NaN).
cand_names = ['cand_idx', 'mass', 'pt', 'eta', 'phi', 'charge',
              'kk_charge', 'mupi_charge', 'charge_class',
              'kk_mass', 'kkpi_mass',
              'refl_k1_as_pi', 'refl_k2_as_pi', 'refl_k1_as_p', 'refl_k2_as_p',
              'refl_k1_pi', 'refl_k2_pi',
              'dr_kk', 'dr_phi_pi', 'dr_ds_mu', 'dr_mu_k1', 'dr_mu_k2', 'dr_mu_pi', 'max_dr_mu_tk',
              'pv_idx', 'pv_from_scheme', 'pv_refit_valid', 'pv_refit_ntrk',
              'pv_x', 'pv_y', 'pv_z', 'pv_ntrk', 'pv_chi2', 'pv_ndof',
              'pv_cov_xx', 'pv_cov_xy', 'pv_cov_xz', 'pv_cov_yy', 'pv_cov_yz', 'pv_cov_zz',
              'bs_x', 'bs_y']

for _obj in ('mu', 'ds', 'b'):
    for _rr in ('03', '04'):
        for _q in ('iso_ch', 'iso_ch_clean', 'iso_pu', 'iso_nh', 'iso_ph',
                   'iso', 'iso_clean', 'reliso', 'reliso_clean'):
            cand_names.append('%s_%s_%s' % (_obj, _q, _rr))

_VTX_Q = ('good', 'status', 'x', 'y', 'z', 'chi2', 'ndof', 'prob', 'mass', 'pt',
          'lxy', 'lxy_err', 'lxy_sig', 'lxyz', 'lxyz_err', 'lxyz_sig', 'cos2d', 'cos3d',
          'cov_xx', 'cov_xy', 'cov_xz', 'cov_yy', 'cov_yz', 'cov_zz')

for _s in SCHEMES:
    for _v in SCHEME_VERTICES:
        for _q in _VTX_Q:
            cand_names.append('%s_%s_%s' % (_s, _v, _q))
    cand_names += ['%s_ds_%s' % (_s, q) for q in
                   ('lxy_bs', 'lxy_bs_err', 'lxy_bs_sig', 'lxyz_bs', 'lxyz_bs_sig',
                    'cos3d_bs', 'cos2d_bs')]
    # refitted momenta of the four tracks
    for _o in ('k1', 'k2', 'pi', 'mu'):
        cand_names += ['%s_%s_rf_%s' % (_s, _o, q) for q in ('pt', 'eta', 'phi')]
    cand_names += ['%s_math_real_bs' % _s]
    for _h in HYPOTHESES:
        cand_names += ['%s_%s_%s' % (_s, _h, q) for q in HYP_OBS]
    cand_names += ['%s_cos_theta_k_pi' % _s, '%s_p_perp' % _s, '%s_ds_p_perp' % _s, '%s_mcorr' % _s]
    # the muon vs the Bs flight axis and vs the Ds trajectory (RDsCandidate)
    for _r in ('pv', 'sv'):
        cand_names += ['%s_mu_ip3d_bs_%s%s' % (_s, _r, x) for x in ('', '_err', '_sig')]
        cand_names += ['%s_mu_dist_along_bs_dir_%s' % (_s, _r)]
    cand_names += ['%s_mu_dist_to_bs_dir' % _s]
    cand_names += ['%s_mu_doca_ds' % _s, '%s_mu_doca_ds_sig' % _s, '%s_mu_poca_along_ds' % _s]
    cand_names += ['%s_gamma_%s' % (_s, q) for q in
                   ('n', 'e_coll', 'pt', 'eta', 'phi', 'dr', 'm_ds_gamma', 'dm', 'e_coll_over_pt',
                    'coll_q2', 'coll_m_miss2', 'xyz_bs_q2', 'xyz_bs_m_miss2')]

cand_names += list(CAND_GEN_NAMES)

##########################################################################################
#####      GEN (MC, event level)
##########################################################################################
BS_GEN_NAMES = ['gen_bs_pt', 'gen_bs_eta', 'gen_bs_phi', 'gen_bs_y', 'gen_bs_mass',
                'gen_bs_pdgid', 'gen_bs_beta', 'gen_bs_gamma', 'gen_bs_lxyz', 'gen_bs_lxy',
                'gen_bs_ct', 'gen_bs_sv_x', 'gen_bs_sv_y', 'gen_bs_sv_z']
VIS_GEN_NAMES = ['gen_vis_%s_%s' % (o, q) for o in ('mu', 'kp', 'km', 'pi')
                 for q in ('pt', 'eta', 'phi')]
gen_event_names = (['gen_sig', 'gen_n_bs'] + BS_GEN_NAMES
                   + ['gen_%s' % k for k in GEN_OBS_NAMES] + VIS_GEN_NAMES
                   + list(HAM_BRANCHES))

##########################################################################################
#####      FLAT LISTS
##########################################################################################
branches = list(event_branches.keys()) + list(event_extra_branches)
for _ibr in muon_branches:
    branches.append('mu_%s' % _ibr)
branches += ['mu_%s' % n for n in muon_trigger_names]
for _obj in ('k1', 'k2', 'pi'):
    branches += ['%s_%s' % (_obj, b) for b in track_branches]
branches += cand_names
branches += gen_event_names
branches += list(HAMMER_BRANCH_NAMES)
branches += list(paths)
branches += [p + '_ps' for p in paths]

_dup = sorted(set(b for b in branches if branches.count(b) > 1))
if _dup:
    raise RuntimeError('RDsBranches: duplicated branch names %s' % _dup)

# the per-event gen tree (MC): one row per processed event
gen_tree_branches = ['run', 'lumi', 'event', 'npu', 'nti', 'gen_pv_x', 'gen_pv_y', 'gen_pv_z'] + \
                    gen_event_names + list(HAMMER_BRANCH_NAMES)

# event-level keys filled outside EVENT_BRANCHES (BaseInspector bookkeeping)
EVENT_EXTRA_KEYS = tuple(event_extra_branches)
EVENT_GEN_KEYS   = tuple(gen_event_names) + tuple(HAMMER_BRANCH_NAMES)
