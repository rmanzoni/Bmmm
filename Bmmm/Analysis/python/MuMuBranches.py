import os

import numpy as np

from Bmmm.Analysis.utils import VTX_COV_ELEMENT_NAMES, VTX_COV_INDEX_PAIRS, vertex_cov_element
from Bmmm.Analysis.CommonBranches import event_branches as _common_event
from Bmmm.Analysis.CommonBranches import muon_branches as _common_muon
from Bmmm.Analysis.CommonBranches import track_cov_branches as _track_cov

##########################################################################################
#####      EVENT-LEVEL
##########################################################################################
# Every event-level quantity of this ntuple is now the shared one, under the
# shared name: the pileup pair used to be spelled n_pu / n_true_int here and
# npu / nti in the RJpsi ntuples, for the same two numbers.
event_branches = {
    'run'    : _common_event['run']    ,
    'lumi'   : _common_event['lumi']   ,
    'event'  : _common_event['event']  ,
    'ncands' : _common_event['ncands'] ,
    'npv'    : _common_event['npv']    ,
    'npu'    : _common_event['npu']    ,
    'nti'    : _common_event['nti']    ,
}

##########################################################################################
#####      CANDIDATE-LEVEL
##########################################################################################
# Order is this ntuple's historical layout, kept as it was. bs_x0/y0/z0 are
# event quantities that happen to sit in the middle of it; the candidate carries
# the beamspot it was built from (cand.beamspot), so they are read from there
# rather than breaking the block in two.
cand_branches = {
    'mass'     : lambda cand : cand.mass()             ,
    'mcorr'    : lambda cand : cand.mass_corrected()   ,
    'pt'       : lambda cand : cand.pt()               ,
    'eta'      : lambda cand : cand.eta()              ,
    'phi'      : lambda cand : cand.phi()              ,
    'charge'   : lambda cand : cand.charge()           ,

    'dr'       : lambda cand : cand.r()                ,
    'dr_max'   : lambda cand : cand.max_dr()           ,
    'dr_12'    : lambda cand : cand.dr12()             ,

    'pv_x'     : lambda cand : cand.pv.position().x()  ,
    'pv_y'     : lambda cand : cand.pv.position().y()  ,
    'pv_z'     : lambda cand : cand.pv.position().z()  ,

    'bs_x0'    : lambda cand : cand.beamspot.x0()      ,
    'bs_y0'    : lambda cand : cand.beamspot.y0()      ,
    'bs_z0'    : lambda cand : cand.beamspot.z0()      ,

    'bs_x'     : lambda cand : cand.bs.position().x()  ,
    'bs_y'     : lambda cand : cand.bs.position().y()  ,

    'vx'       : lambda cand : cand.vtx.position().x() ,
    'vy'       : lambda cand : cand.vtx.position().y() ,
    'vz'       : lambda cand : cand.vtx.position().z() ,
    'vtx_chi2' : lambda cand : cand.vtx.chi2           ,
    'vtx_prob' : lambda cand : cand.vtx.prob           ,

    # Displacement wrt the refit PV (cand.pv_bs): beamspot constrained, signal
    # muons removed. Same reference and same names as the RJpsi channel.
    'cos2d'     : lambda cand : cand.cos2d              ,
    'cos3d'     : lambda cand : cand.cos3d              ,
    'lxy'       : lambda cand : cand.lxy.value()        ,
    'lxy_err'   : lambda cand : cand.lxy.error()        ,
    'lxy_sig'   : lambda cand : cand.lxy.significance() ,
    'lxyz'      : lambda cand : cand.lxyz.value()       ,
    'lxyz_err'  : lambda cand : cand.lxyz.error()       ,
    'lxyz_sig'  : lambda cand : cand.lxyz.significance(),

    # the pre-refit convention (bare beamspot), kept as a fixed comparator:
    # lxy - lxy_bs is the size of the change
    'cos2d_bs'  : lambda cand : cand.cos2d_bs              ,
    'lxy_bs'    : lambda cand : cand.lxy_bs.value()        ,
    'lxy_bs_err': lambda cand : cand.lxy_bs.error()        ,
    'lxy_bs_sig': lambda cand : cand.lxy_bs.significance() ,
}

branches = list(event_branches) + list(cand_branches)

# The per-muon block of this ntuple is exactly the channel-agnostic one in
# CommonBranches -- same quantities, same order (the RJpsi block is this list
# with the refitted-J/psi rf_* and the gen bookkeeping added). So take it whole
# rather than restating it, and the two channels can no longer drift: the
# swapped soft/medium IDs fixed a few commits back were precisely that kind of
# drift.
muon_branches = dict(_common_muon)

##########################################################################################
#####      COVARIANCE MATRICES
##########################################################################################
# Same blocks the J/psi + charged-object ntuples carry, so the covariance
# mismodelling can be measured HERE -- on the 2018 dimuon tag-and-probe sample,
# where the statistics are -- and the correction applied there.
#
# Per muon: the 15 independent elements of the 5x5 curvilinear track covariance,
# mu<i>_cov_<par_j>_<par_k>. Note these are the BARE track uncertainties: unlike
# mu<i>_dxy_e, which is dxyError(pv.position(), pv.error()) and therefore folds
# in the primary-vertex error, sqrt(mu<i>_cov_dxy_dxy) is the track's own
# sigma_dxy. For measuring a track-resolution mismodelling that is the cleaner
# quantity -- and in this channel it matters more than in the RJpsi one, because
# the PV here is the plain chosen vertex, NOT refitted with the signal muons
# removed, so the muon under study is itself in the PV fit and dxy_e is
# correlated with it.
muon_branches.update(_track_cov)

# The per-candidate refit PV: position, provenance and the track count that fed
# it. pv_refit_valid==0 means the refit was not available and every pv_bs_*
# quantity below is the hybrid fallback (beamspot x,y + PV z, PV covariance),
# so cut on it before using the refit as such.
cand_branches['pv_bs_x'        ] = lambda c : c.pv_bs.position().x()
cand_branches['pv_bs_y'        ] = lambda c : c.pv_bs.position().y()
cand_branches['pv_bs_z'        ] = lambda c : c.pv_bs.position().z()
cand_branches['pv_refit_valid' ] = lambda c : int(c.pv_refit_valid)
cand_branches['pv_refit_ntrk'  ] = lambda c : c.pv_refit_ntrk
branches += ['pv_bs_x', 'pv_bs_y', 'pv_bs_z', 'pv_refit_valid', 'pv_refit_ntrk']

# Per vertex: the 6 independent elements of the 3x3 position covariance.
#   pv_cov_*    the CHOSEN primary vertex as it comes out of the collection
#               (cand.pv), kept as the pre-refit comparator
#   pv_bs_cov_* the reference actually used for every wrt-PV quantity: the
#               beamspot-constrained refit with the signal muons removed, or
#               the hybrid fallback when pv_refit_valid==0
#   vtx_cov_*   the dimuon vertex from the Kalman fit (cand.vtx)
# vtx_ rather than sv_ to match the vx/vy/vz/vtx_chi2/vtx_prob already in this
# ntuple; the RJpsi channel calls the same thing sv_cov_* because that is what
# its own vertex block is called.
for _iname, (_ii, _jj) in zip(VTX_COV_ELEMENT_NAMES, VTX_COV_INDEX_PAIRS):
    cand_branches['pv_cov_%s'    % _iname] = (lambda c, i=_ii, j=_jj : vertex_cov_element(c.pv,    i, j))
    cand_branches['pv_bs_cov_%s' % _iname] = (lambda c, i=_ii, j=_jj : vertex_cov_element(c.pv_bs, i, j))
    cand_branches['vtx_cov_%s'   % _iname] = (lambda c, i=_ii, j=_jj : vertex_cov_element(c.vtx,   i, j))
branches += ['pv_cov_%s'    % iname for iname in VTX_COV_ELEMENT_NAMES]
branches += ['pv_bs_cov_%s' % iname for iname in VTX_COV_ELEMENT_NAMES]
branches += ['vtx_cov_%s'   % iname for iname in VTX_COV_ELEMENT_NAMES]

for idx in [1,2]:
    for ibr in muon_branches:
        branches.append('mu%d_%s' %(idx, ibr))

# paths and filters
# check online confDB https://hlt-config-editor-confdbv3.app.cern.ch/

paths = dict()



##########################################################################################                                                                  
##########################################################################################                                                                  
# these filters don't seem to be in our samples... I've taken them from /cdaq/physics/Run2018/2e34/v1.2.3/HLT/V2
# paths['HLT_Mu17'                                  ] = ['hltL3fL1sMu10lqL1f0L2f10L3Filtered17']
# paths['HLT_Mu19'                                  ] = ['hltL3fL1sMu10lqL1f0L2f10L3Filtered19']
# in MC this menu was used
# hltInfo C1ACDC94-EBC6-1745-A410-359FFEAB28BC.root
# /frozen/2018/2e34/v3.2/HLT/V1
# https://hlt-config-editor-confdbv3.app.cern.ch/open?cfg=%2Ffrozen%2F2018%2F2e34%2Fv3.2%2FHLT%2FV1&db=offline-run2 
# for data (same filters, yay!)
# hltInfo 5EBF575A-A990-CB41-8EC8-28A3F2035C1B.root
# /cdaq/physics/Run2018/2e34/v3.6.1/HLT/V2
# https://hlt-config-editor-confdbv3.app.cern.ch/open?cfg=%2Fcdaq%2Fphysics%2FRun2018%2F2e34%2Fv3.6.1%2FHLT%2FV2&db=online

# HLT_Dimuon0_Jpsi3p5_Muon2
# L1_TripleMu_5SQ_3SQ_0OQ_DoubleMu_5_3_SQ_OS_Mass_Max9 OR L1_TripleMu_5SQ_3SQ_0_DoubleMu_5_3_SQ_OS_Mass_Max9
  
##########################################################################################                                                                  
##########################################################################################                                                                  


##########################################################################################                                                                  
##########################################################################################                                                                  
##   _____ _                                      _                                     ##
##  / ____| |                                    (_)                                    ##
## | |    | |__   __ _ _ __ _ __ ___   ___  _ __  _ _   _ _ __ ___                      ##
## | |    | '_ \ / _` | '__| '_ ` _ \ / _ \| '_ \| | | | | '_ ` _ \                     ##
## | |____| | | | (_| | |  | | | | | | (_) | | | | | |_| | | | | | |                    ##
##  \_____|_| |_|\__,_|_|  |_| |_| |_|\___/|_| |_|_|\__,_|_| |_| |_|                    ##
##########################################################################################                                                                  
##########################################################################################                                                                  
       
# rates from here https://cmsoms.cern.ch/cms/triggers/hlt_trigger_rates?cms_run=319579 
paths['HLT_Mu7p5_Track2_Jpsi'                     ] = ['hltL3fLMu7p5TrackL3Filtered7p5'               , 'hltMu7p5Track2JpsiTrackMassFiltered'          ] # run 319579 rate 0.56 Hz # L1_SingleMu5 OR L1_SingleMu7
paths['HLT_Mu7p5_Track3p5_Jpsi'                   ] = ['hltL3fLMu7p5TrackL3Filtered7p5'               , 'hltMu7p5Track3p5JpsiTrackMassFiltered'        ] # run 319579 rate 0.47 Hz # L1_SingleMu5 OR L1_SingleMu7
paths['HLT_Mu7p5_Track7_Jpsi'                     ] = ['hltL3fLMu7p5TrackL3Filtered7p5'               , 'hltMu7p5Track7JpsiTrackMassFiltered'          ] # run 319579 rate 0.15 Hz # L1_SingleMu5 OR L1_SingleMu7
paths['HLT_Mu7p5_L2Mu2_Jpsi'                      ] = ['hltSQMu7p5L2Mu2JpsiTrackMassFiltered'         , 'hltSQMu7p5L2Mu2JpsiTrackMassFiltered'         ] # run 319579 rate 0.11 Hz # L1_DoubleMu0_SQ
paths['HLT_Dimuon0_Jpsi'                          ] = ['hltDisplacedmumuFilterDimuon0Jpsi'            , 'hltDisplacedmumuFilterDimuon0Jpsi'            ] # run 319579 rate 0.05 Hz # L1_DoubleMu0_SQ_OS OR L1_DoubleMu0_SQ
paths['HLT_Dimuon0_Jpsi_L1_NoOS'                  ] = ['hltDisplacedmumuFilterDimuon0JpsiL1sNoOS'     , 'hltDisplacedmumuFilterDimuon0JpsiL1sNoOS'     ] # run 319579 rate 0.05 Hz # L1_DoubleMu0_SQ
paths['HLT_Dimuon0_Jpsi_L1_4R_0er1p5R'            ] = ['hltDisplacedmumuFilterDimuon0JpsiL1s4R0er1p5R', 'hltDisplacedmumuFilterDimuon0JpsiL1s4R0er1p5R'] # run 319579 rate 0.69 Hz # L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4 OR L1_DoubleMu0er1p4_SQ_OS_dR_Max1p4 OR L1_DoubleMu4p5_SQ_OS_dR_Max1p2 OR L1_DoubleMu4_SQ_OS_dR_Max1p2
paths['HLT_Dimuon0_Jpsi_NoVertexing'              ] = ['hltDimuon0JpsiL3Filtered'                     , 'hltDimuon0JpsiL3Filtered'                     ] # run 319579 rate 0.06 Hz # L1_DoubleMu0_SQ_OS OR L1_DoubleMu0_SQ
paths['HLT_Dimuon0_Jpsi_NoVertexing_L1_NoOS'      ] = ['hltDimuon0JpsiNoVtxNoOSL3Filtered'            , 'hltDimuon0JpsiNoVtxNoOSL3Filtered'            ] # run 319579 rate 0.05 Hz # L1_DoubleMu0_SQ
paths['HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R'] = ['hltDimuon0JpsiL1s4R0er1p5RL3Filtered'         , 'hltDimuon0JpsiL1s4R0er1p5RL3Filtered'         ] # run 319579 rate 0.75 Hz # L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4 OR L1_DoubleMu0er1p4_SQ_OS_dR_Max1p4 OR L1_DoubleMu4p5_SQ_OS_dR_Max1p2 OR L1_DoubleMu4_SQ_OS_dR_Max1p2
paths['HLT_DoubleMu4_3_Jpsi'                      ] = ['hltmumuFilterDoubleMu43Jpsi'                  , 'hltmumuFilterDoubleMu43Jpsi'                  ] # run 319579 rate 6.30 Hz # L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4 OR L1_DoubleMu0er1p4_SQ_OS_dR_Max1p4
paths['HLT_DoubleMu4_Jpsi_NoVertexing'            ] = ['hltDoubleMu4JpsiDisplacedL3Filtered'          , 'hltDoubleMu4JpsiDisplacedL3Filtered'          ] # run 319579 rate 0.65 Hz # L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4 OR L1_DoubleMu0er1p4_SQ_OS_dR_Max1p4 OR L1_DoubleMu4p5_SQ_OS_dR_Max1p2 OR L1_DoubleMu4_SQ_OS_dR_Max1p2
paths['HLT_DoubleMu4_Jpsi_Displaced'              ] = ['hltDisplacedmumuFilterDoubleMu4Jpsi'          , 'hltDisplacedmumuFilterDoubleMu4Jpsi'          ] # run 319579 rate 0.77 Hz # L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4 OR L1_DoubleMu0er1p4_SQ_OS_dR_Max1p4 OR L1_DoubleMu4p5_SQ_OS_dR_Max1p2 OR L1_DoubleMu4_SQ_OS_dR_Max1p2

##########################################################################################                                                                  
##########################################################################################                                                                  
##   _____              _     _      __  __                                             ##
##  |  __ \            | |   | |    |  \/  |                                            ##
##  | |  | | ___  _   _| |__ | | ___| \  / |_   _  ___  _ __                            ##
##  | |  | |/ _ \| | | | '_ \| |/ _ \ |\/| | | | |/ _ \| '_ \                           ##
##  | |__| | (_) | |_| | |_) | |  __/ |  | | |_| | (_) | | | |                          ##
##  |_____/ \___/ \__,_|_.__/|_|\___|_|  |_|\__,_|\___/|_| |_|                          ##
##########################################################################################                                                               
##########################################################################################                                                                  
paths['HLT_Mu8' ] = ['hltL3fL1sMu5L1f0L2f5L3Filtered8'       ] # run 319579 rate Hz 1.51 # L1_SingleMu3 OR L1_SingleMu5 OR L1_SingleMu7
paths['HLT_Mu17'] = ['hltL3fL1sMu15DQlqL1f0L2f10L3Filtered17'] # run 319579 rate Hz 1.11 # L1_SingleMu15_DQ
paths['HLT_Mu19'] = ['hltL3fL1sMu15DQlqL1f0L2f10L3Filtered19'] # run 319579 rate Hz 0.75 # L1_SingleMu15_DQ

########################################################################################################                                                                  
########################################################################################################                                                    
##  _____              _     _      __  __                   _                   __  __               ##
## |  __ \            | |   | |    |  \/  |                 | |                 |  \/  |              ##
## | |  | | ___  _   _| |__ | | ___| \  / |_   _  ___  _ __ | |     _____      _| \  / | __ _ ___ ___ ##
## | |  | |/ _ \| | | | '_ \| |/ _ \ |\/| | | | |/ _ \| '_ \| |    / _ \ \ /\ / / |\/| |/ _` / __/ __|##
## | |__| | (_) | |_| | |_) | |  __/ |  | | |_| | (_) | | | | |___| (_) \ V  V /| |  | | (_| \__ \__ \##
## |_____/ \___/ \__,_|_.__/|_|\___|_|  |_|\__,_|\___/|_| |_|______\___/ \_/\_/ |_|  |_|\__,_|___/___/##
########################################################################################################                                                 
########################################################################################################                                                    




##########################################################################################                                                                  
##########################################################################################                                                                  
##   _____ _             _      __  __                                                  ##
##  / ____(_)           | |    |  \/  |                                                 ##
## | (___  _ _ __   __ _| | ___| \  / |_   _  ___  _ __                                 ##
##  \___ \| | '_ \ / _` | |/ _ \ |\/| | | | |/ _ \| '_ \                                ##
##  ____) | | | | | (_| | |  __/ |  | | |_| | (_) | | | |                               ##
## |_____/|_|_| |_|\__, |_|\___|_|  |_|\__,_|\___/|_| |_|                               ##
##                  __/ |                                                               ##
##                 |___/                                                                ##
##########################################################################################                                                               
##########################################################################################                                                                  
paths['HLT_IsoMu24'] = ['hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p07']
paths['HLT_Mu12'   ] = ['hltL3fL1sMu15DQlqL1f0L2f10L3Filtered12'  ]   # L1_SingleMu15_DQ
paths['HLT_Mu15'   ] = ['hltL3fL1sMu15DQlqL1f0L2f10L3Filtered15'  ]   # L1_SingleMu15_DQ
paths['HLT_Mu20'   ] = ['hltL3fL1sMu18L1f0L2f10QL3Filtered20Q'    ]   # L1_SingleMu18
paths['HLT_Mu27'   ] = ['hltL3fL1sMu22Or25L1f0L2f10QL3Filtered27Q']   # L1_SingleMu22 OR L1_SingleMu25

# analysis triggers, not straightforward to define T&P filters...
# paths['HLT_Dimuon0_Jpsi3p5_Muon2'                 ] = ['hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p07']
# paths['HLT_DoubleMu4_JpsiTrk_Displaced'           ] = ['hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p07']



##########################################################################################                                                                  
##########################################################################################                                                                  
##  ____  _____           _    _                                                        ##
## |  _ \|  __ \         | |  (_)                                                       ##
## | |_) | |__) |_ _ _ __| | ___ _ __   __ _                                            ##
## |  _ <|  ___/ _` | '__| |/ / | '_ \ / _` |                                           ##
## | |_) | |  | (_| | |  |   <| | | | | (_| |                                           ##
## |____/|_|   \__,_|_|  |_|\_\_|_| |_|\__, |                                           ##
##                                      __/ |                                           ##
##                                     |___/                                            ##
##########################################################################################                                                               
##########################################################################################                                                                  

paths['HLT_Mu7_IP4'     ] = ["hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered7IP4Q"]
paths['HLT_Mu8_IP3'     ] = ["hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered8Q"   ]
paths['HLT_Mu8_IP5'     ] = ["hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered8IP5Q"]
paths['HLT_Mu8_IP6'     ] = ["hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered8IP6Q"]
paths['HLT_Mu8p5_IP3p5' ] = ["hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered8p5Q" ]
paths['HLT_Mu9_IP4'     ] = ["hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered9IP4Q"]
paths['HLT_Mu9_IP5'     ] = ["hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered9IP5Q"]
paths['HLT_Mu9_IP6'     ] = ["hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered9Q"   ]
paths['HLT_Mu10p5_IP3p5'] = ["hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered10p5Q"]
paths['HLT_Mu12_IP6'    ] = ["hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered12Q"  ]


##########################################################################################
#####      RUN 3
##########################################################################################
# The list above is the 2018 menu. This is the Run 3 low-mass dimuon path, the one
# the RJpsi Run 3 skims trigger on (same filter labels as
# JpsiChargedBranches.paths), so the dimuon and RJpsi Run 3 samples are selected
# by the same trigger.
paths['HLT_DoubleMu4_3_LowMass'] = ['hltDisplacedmumuFilterDoubleMu43LowMass',
                                    'hltDisplacedmumuFilterDoubleMu43LowMass']

# Opt-in restriction of the path set, applied BEFORE the branch list is built
# because the schema is derived from it. Unset (the default) keeps every path,
# so 2018 running is untouched; a Run 3 job exports
#
#     BMMM_MM_HLT_PATHS=HLT_DoubleMu4_3_LowMass
#
# and gets an ntuple carrying only that path's decision, prescale and
# tag-and-probe branches instead of ~200 columns of 2018 paths that no Run 3
# menu contains. An env var rather than a command-line flag because the branch
# list is assembled at import, before any argument parsing happens.
_ONLY = os.environ.get('BMMM_MM_HLT_PATHS', '').strip()
if _ONLY:
    _keep = set(n.strip() for n in _ONLY.split(',') if n.strip())
    _unknown = _keep - set(paths)
    if _unknown:
        raise ValueError(
            'BMMM_MM_HLT_PATHS names paths that are not defined: %s\n'
            'It is a comma-separated list of HLT path names to KEEP, not a flag '
            '-- e.g. BMMM_MM_HLT_PATHS=HLT_DoubleMu4_3_LowMass. Leave it unset '
            'to keep all %d. Defined here:\n    %s'
            % (sorted(_unknown), len(paths), '\n    '.join(sorted(paths))))
    for _drop in [k for k in paths if k not in _keep]:
        del paths[_drop]
    print('[MuMuBranches] BMMM_MM_HLT_PATHS -> keeping %d path(s): %s'
          % (len(paths), ', '.join(paths)))

# add branches for T&P
for k, v in paths.items():
    for idx in [1,2]:
        branches.append('mu%d_%s_tag' %(idx, k))
        branches.append('mu%d_%s_probe' %(idx, k))
    
branches += paths
branches += [path+'_ps' for path in paths]
