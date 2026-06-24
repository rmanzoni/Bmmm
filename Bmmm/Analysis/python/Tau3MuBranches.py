import numpy as np
from collections import OrderedDict

##########################################################################################
#####      EVENT-LEVEL BRANCHES (filled once per event)
##########################################################################################
event_branches = {
    'run'    : lambda ev : ev.eventAuxiliary().run()            ,
    'lumi'   : lambda ev : ev.eventAuxiliary().luminosityBlock(),
    'event'  : lambda ev : ev.eventAuxiliary().event()          ,

    'ncands' : lambda ev : ev.ncands                            ,
    'npv'    : lambda ev : len(ev.vtx)                          ,
    'npu'    : lambda ev : ev.pu_at_bx0.getPU_NumInteractions()  if ev.mc else np.nan,
    'nti'    : lambda ev : ev.pu_at_bx0.getTrueNumInteractions() if ev.mc else np.nan,

    'bs_x0'  : lambda ev : ev.bs.x0()     ,
    'bs_x0e' : lambda ev : ev.bs.x0Error(),
    'bs_y0'  : lambda ev : ev.bs.y0()     ,
    'bs_y0e' : lambda ev : ev.bs.y0Error(),
    'bs_z0'  : lambda ev : ev.bs.z0()     ,
    'bs_z0e' : lambda ev : ev.bs.z0Error(),
}

##########################################################################################
#####      CANDIDATE-LEVEL BRANCHES (filled once per candidate)
##########################################################################################
cand_branches = {
    # ----- tau (full 3mu) kinematics -----
    'mass'        : lambda c : c.mass()   ,
    'pt'          : lambda c : c.pt()     ,
    'eta'         : lambda c : c.eta()    ,
    'phi'         : lambda c : c.phi()    ,
    'charge'      : lambda c : c.charge() ,

    'rf_mass'     : lambda c : c.rfp4.mass(),
    'rf_pt'       : lambda c : c.rfp4.pt()  ,
    'rf_eta'      : lambda c : c.rfp4.eta() ,
    'rf_phi'      : lambda c : c.rfp4.phi() ,

    # ----- a (displaced OS pair, a -> mu mu) kinematics -----
    'a_mass'      : lambda c : c.a.mass()   ,
    'a_pt'        : lambda c : c.a.pt()     ,
    'a_eta'       : lambda c : c.a.eta()    ,
    'a_phi'       : lambda c : c.a.phi()    ,
    'a_charge'    : lambda c : c.a.charge() ,

    'a_rf_mass'   : lambda c : c.a_rfp4.mass(),
    'a_rf_pt'     : lambda c : c.a_rfp4.pt()  ,
    'a_rf_eta'    : lambda c : c.a_rfp4.eta() ,
    'a_rf_phi'    : lambda c : c.a_rfp4.phi() ,

    # ----- opening angles -----
    'dr'          : lambda c : c.r()      ,
    'dr_max'      : lambda c : c.max_dr() ,
    'dr_a'        : lambda c : c.dr12()   ,   # dR of the two a muons
    'dr_a_mu'     : lambda c : c.dr_a_mu(),   # dR(a, bachelor)

    # ----- primary vertex / beamspot (refit PV reference, self.pv_bs) -----
    'pv_refit_valid' : lambda c : int(c.pv_refit_valid)  ,
    'pv_x'           : lambda c : c.pv_bs.position().x() ,
    'pv_y'           : lambda c : c.pv_bs.position().y() ,
    'pv_z'           : lambda c : c.pv_bs.position().z() ,
    'pv_ntrk'        : lambda c : c.pv_bs.tracksSize()   ,
    'pv_chi2'        : lambda c : c.pv_bs.chi2()         ,
    'pv_ndof'        : lambda c : c.pv_bs.ndof()         ,
    'bs_x'           : lambda c : c.bs.position().x()    ,
    'bs_y'           : lambda c : c.bs.position().y()    ,

    # ----- tau vertex (sequential a + bachelor fit), wrt PV -----
    'sv_good'     : lambda c : int(c.good_vtx)        ,
    'sv_x'        : lambda c : c.sv_vtx.position().x(),
    'sv_y'        : lambda c : c.sv_vtx.position().y(),
    'sv_z'        : lambda c : c.sv_vtx.position().z(),
    'sv_chi2'     : lambda c : c.sv_vtx_chi2          ,
    'sv_ndof'     : lambda c : c.sv_vtx_ndof          ,
    'sv_prob'     : lambda c : c.sv_vtx_prob          ,
    'sv_cos2d'    : lambda c : c.sv_cos2d             ,
    'sv_cos3d'    : lambda c : c.sv_cos3d             ,
    'sv_lxy'      : lambda c : c.sv_lxy.value()       ,
    'sv_lxy_err'  : lambda c : c.sv_lxy.error()       ,
    'sv_lxy_sig'  : lambda c : c.sv_lxy.significance(),
    'sv_lxyz'     : lambda c : c.sv_lxyz.value()       ,
    'sv_lxyz_err' : lambda c : c.sv_lxyz.error()       ,
    'sv_lxyz_sig' : lambda c : c.sv_lxyz.significance(),

    # ----- a vertex (displaced OS pair), wrt PV -----
    'a_good'      : lambda c : int(c.a_good_vtx)     ,
    'a_x'         : lambda c : c.a_vtx.position().x(),
    'a_y'         : lambda c : c.a_vtx.position().y(),
    'a_z'         : lambda c : c.a_vtx.position().z(),
    'a_vtx_chi2'  : lambda c : c.a_vtx_chi2          ,
    'a_vtx_ndof'  : lambda c : c.a_vtx_ndof          ,
    'a_vtx_prob'  : lambda c : c.a_vtx_prob          ,
    'a_cos2d'     : lambda c : c.a_cos2d             ,
    'a_cos3d'     : lambda c : c.a_cos3d             ,
    'a_lxy'       : lambda c : c.a_lxy.value()       ,
    'a_lxy_err'   : lambda c : c.a_lxy.error()       ,
    'a_lxy_sig'   : lambda c : c.a_lxy.significance(),
    'a_lxyz'      : lambda c : c.a_lxyz.value()       ,
    'a_lxyz_err'  : lambda c : c.a_lxyz.error()       ,
    'a_lxyz_sig'  : lambda c : c.a_lxyz.significance(),

    # ----- a vertex displaced FROM the tau vertex (the a flight) -----
    'a_wrt_tau_cos2d'    : lambda c : c.a_wrt_tau_cos2d             ,
    'a_wrt_tau_cos3d'    : lambda c : c.a_wrt_tau_cos3d             ,
    'a_wrt_tau_lxy'      : lambda c : c.a_wrt_tau_lxy.value()       ,
    'a_wrt_tau_lxy_err'  : lambda c : c.a_wrt_tau_lxy.error()       ,
    'a_wrt_tau_lxy_sig'  : lambda c : c.a_wrt_tau_lxy.significance(),
    'a_wrt_tau_lxyz'     : lambda c : c.a_wrt_tau_lxyz.value()       ,
    'a_wrt_tau_lxyz_err' : lambda c : c.a_wrt_tau_lxyz.error()       ,
    'a_wrt_tau_lxyz_sig' : lambda c : c.a_wrt_tau_lxyz.significance(),

    # ----- bachelor mu3 signed 3D IP wrt the a vertex (mu3 not in the a fit) -----
    'mu3_ip3d_a'     : lambda c : c.mu3_ip3d_a    ,
    'mu3_ip3d_a_err' : lambda c : c.mu3_ip3d_a_err,
    'mu3_ip3d_a_sig' : lambda c : c.mu3_ip3d_a_sig,

    'trig_match'  : lambda c : int(c.trig_match),
}

##########################################################################################
#####      PER-MUON BRANCHES (mu1, mu2 = a pair; mu3 = bachelor)
##########################################################################################
muon_branches = {
    'pt'         : lambda m : m.pt()       ,
    'eta'        : lambda m : m.eta()      ,
    'phi'        : lambda m : m.phi()      ,
    'e'          : lambda m : m.energy()   ,
    'mass'       : lambda m : m.mass()     ,
    'charge'     : lambda m : m.charge()   ,

    'rf_pt'      : lambda m : m.rfp4.pt()  ,
    'rf_eta'     : lambda m : m.rfp4.eta() ,
    'rf_phi'     : lambda m : m.rfp4.phi() ,
    'rf_e'       : lambda m : m.rfp4.energy(),

    'id_loose'   : lambda m : m.isLooseMuon()    ,
    'id_soft'    : lambda m : m.isSoftMuon(m.pv) ,
    'id_medium'  : lambda m : m.isMediumMuon()   ,
    'id_tight'   : lambda m : m.isTightMuon(m.pv),
    'id_pf'      : lambda m : m.isPFMuon()       ,
    'id_global'  : lambda m : m.isGlobalMuon()   ,
    'id_tracker' : lambda m : m.isTrackerMuon()  ,

    'pfiso03'    : lambda m : (m.iso03.sumChargedHadronPt + max(m.iso03.sumNeutralHadronEt + m.iso03.sumPhotonEt - 0.5 * m.iso03.sumPUPt, 0.0)),
    'pfiso04'    : lambda m : (m.iso04.sumChargedHadronPt + max(m.iso04.sumNeutralHadronEt + m.iso04.sumPhotonEt - 0.5 * m.iso04.sumPUPt, 0.0)),
    'pfreliso03' : lambda m : (m.iso03.sumChargedHadronPt + max(m.iso03.sumNeutralHadronEt + m.iso03.sumPhotonEt - 0.5 * m.iso03.sumPUPt, 0.0)) / m.pt(),
    'pfreliso04' : lambda m : (m.iso04.sumChargedHadronPt + max(m.iso04.sumNeutralHadronEt + m.iso04.sumPhotonEt - 0.5 * m.iso04.sumPUPt, 0.0)) / m.pt(),

    # IPs wrt the refit PV
    'dxy'        : lambda m : m.bestTrack().dxy(m.pv.position()),
    'dxy_e'      : lambda m : m.bestTrack().dxyError(m.pv.position(), m.pv.error()),
    'dxy_sig'    : lambda m : m.bestTrack().dxy(m.pv.position()) / m.bestTrack().dxyError(m.pv.position(), m.pv.error()),
    'dz'         : lambda m : m.bestTrack().dz(m.pv.position()),
    'dz_e'       : lambda m : m.bestTrack().dzError(),
    'dz_sig'     : lambda m : m.bestTrack().dz(m.pv.position()) / m.bestTrack().dzError(),
    'bs_dxy'     : lambda m : m.bestTrack().dxy(m.bs.position()),
    'bs_dxy_e'   : lambda m : m.bestTrack().dxyError(m.bs.position(), m.bs.error()),
    'bs_dxy_sig' : lambda m : m.bestTrack().dxy(m.bs.position()) / m.bestTrack().dxyError(m.bs.position(), m.bs.error()),

    # signed 3D IP wrt the refit PV (lifetime-signed along the tau flight)
    'ip3d'       : lambda m : m.ip3d    ,
    'ip3d_err'   : lambda m : m.ip3d_err,
    'ip3d_sig'   : lambda m : m.ip3d_sig,

    'cov_pos_def': lambda m : m.is_cov_pos_def,

    # gen matching (signal MC; NaN otherwise)
    'gen_pt'     : lambda m : m.gen_match.pt()     if hasattr(m, 'gen_match') else np.nan,
    'gen_eta'    : lambda m : m.gen_match.eta()    if hasattr(m, 'gen_match') else np.nan,
    'gen_phi'    : lambda m : m.gen_match.phi()    if hasattr(m, 'gen_match') else np.nan,
    'gen_e'      : lambda m : m.gen_match.energy() if hasattr(m, 'gen_match') else np.nan,
    'gen_pdgid'  : lambda m : m.gen_match.pdgId()  if hasattr(m, 'gen_match') else np.nan,
    'gen_charge' : lambda m : m.gen_match.charge() if hasattr(m, 'gen_match') else np.nan,
    'gen_role'   : lambda m : m.gen_role,
    'gen_dr'     : lambda m : m.gen_dr  ,
}

##########################################################################################
#####      GEN BRANCHES (signal MC; operate on a Tau3MuGenDecay 'info')
##########################################################################################
def _gen_lxyz(prod, decay):
    return np.sqrt((decay.vx() - prod.vx())**2 + (decay.vy() - prod.vy())**2 + (decay.vz() - prod.vz())**2)
def _gen_lxy(prod, decay):
    return np.sqrt((decay.vx() - prod.vx())**2 + (decay.vy() - prod.vy())**2)

gen_branches = {
    'gen_ds_pt'    : lambda i : i.ds.pt()    if i.ds is not None else np.nan,
    'gen_ds_eta'   : lambda i : i.ds.eta()   if i.ds is not None else np.nan,
    'gen_ds_phi'   : lambda i : i.ds.phi()   if i.ds is not None else np.nan,
    'gen_ds_mass'  : lambda i : i.ds.mass()  if i.ds is not None else np.nan,
    'gen_ds_pdgid' : lambda i : i.ds.pdgId() if i.ds is not None else np.nan,

    'gen_tau_pt'   : lambda i : i.tau.pt()  ,
    'gen_tau_eta'  : lambda i : i.tau.eta() ,
    'gen_tau_phi'  : lambda i : i.tau.phi() ,
    'gen_tau_mass' : lambda i : i.tau.mass(),
    'gen_tau_pdgid': lambda i : i.tau.pdgId(),

    'gen_a_pt'     : lambda i : i.a.pt()  ,
    'gen_a_eta'    : lambda i : i.a.eta() ,
    'gen_a_phi'    : lambda i : i.a.phi() ,
    'gen_a_mass'   : lambda i : i.a.mass(),

    # tau flight (production -> tau decay vertex)
    'gen_tau_lxy'  : lambda i : _gen_lxy (i.tau, i.tau.daughter(0)),
    'gen_tau_lxyz' : lambda i : _gen_lxyz(i.tau, i.tau.daughter(0)),
    'gen_tau_ct'   : lambda i : _gen_lxyz(i.tau, i.tau.daughter(0)) / i.tau.p4().Beta() / i.tau.p4().Gamma(),

    # a flight (a production == tau decay vertex -> a decay vertex)
    'gen_a_lxy'    : lambda i : _gen_lxy (i.a, i.a.daughter(0)),
    'gen_a_lxyz'   : lambda i : _gen_lxyz(i.a, i.a.daughter(0)),
    'gen_a_ct'     : lambda i : _gen_lxyz(i.a, i.a.daughter(0)) / i.a.p4().Beta() / i.a.p4().Gamma(),

    # vertices
    'gen_pv_x'     : lambda i : i.tau.vx(),
    'gen_pv_y'     : lambda i : i.tau.vy(),
    'gen_pv_z'     : lambda i : i.tau.vz(),
    'gen_tau_sv_x' : lambda i : i.tau.daughter(0).vx(),
    'gen_tau_sv_y' : lambda i : i.tau.daughter(0).vy(),
    'gen_tau_sv_z' : lambda i : i.tau.daughter(0).vz(),
    'gen_a_sv_x'   : lambda i : i.a.daughter(0).vx(),
    'gen_a_sv_y'   : lambda i : i.a.daughter(0).vy(),
    'gen_a_sv_z'   : lambda i : i.a.daughter(0).vz(),
}

##########################################################################################
#####      JAGGED PF-CONE BRANCH (variable length, one sublist per candidate)
#####
#####  All the per-PF-candidate quantities are written as ONE jagged record branch
#####  'pf' (a  var * {pt, eta, ...}  structure). ROOT/uproot then writes a SINGLE
#####  shared counter 'npf' plus the leaves pf_pt, pf_eta, ... -- the NanoAOD-style
#####  layout  npf + pf_<field>[npf]  -- instead of one (identical) counter per
#####  field. Each entry maps  record-field-name -> (candidate attribute, dtype).
##########################################################################################
PF_BRANCH = 'pf'

pf_fields = OrderedDict([
    ('pt'       , ('pf_pt'       , np.float32)),
    ('eta'      , ('pf_eta'      , np.float32)),
    ('phi'      , ('pf_phi'      , np.float32)),
    ('mass'     , ('pf_mass'     , np.float32)),
    ('energy'   , ('pf_energy'   , np.float32)),
    ('puppiweight', ('pf_puppiweight', np.float32)),
    ('pdgid'    , ('pf_pdgid'    , np.int32  )),
    ('charge'   , ('pf_charge'   , np.int32  )),
    ('dr'       , ('pf_dr'       , np.float32)),
    ('dxy'      , ('pf_dxy'      , np.float32)),
    ('dxy_err'  , ('pf_dxy_err'  , np.float32)),
    ('dz'       , ('pf_dz'       , np.float32)),
    ('dz_err'   , ('pf_dz_err'   , np.float32)),
    ('ip3d'     , ('pf_ip3d'     , np.float32)),
    ('ip3d_sig' , ('pf_ip3d_sig' , np.float32)),
    ('is_signal', ('pf_is_signal', np.int32  )),
])

_PF_ROOT_TYPE = {np.float32: 'float32', np.int32: 'int32'}
def pf_branch_type():
    '''uproot type string for the single jagged record branch.'''
    inner = ', '.join('%s: %s' % (fname, _PF_ROOT_TYPE[dt]) for fname, (_a, dt) in pf_fields.items())
    return 'var * {%s}' % inner

##########################################################################################
#####      ASSEMBLE THE FLAT (SCALAR) BRANCH LIST + the trigger paths
##########################################################################################
branches = []
for ibranch in event_branches.keys():
    branches.append(ibranch)

for idx in [1, 2, 3]:
    for ibr in muon_branches.keys():
        branches.append('mu%d_%s' % (idx, ibr))

for ibranch in cand_branches.keys():
    branches.append(ibranch)

for ibranch in gen_branches.keys():
    branches.append(ibranch)

paths = {}
paths['HLT_DoubleMu4_3_LowMass'] = ['hltDisplacedmumuFilterDoubleMu43LowMass']

branches += list(paths)
branches += [path + '_ps' for path in paths]

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
