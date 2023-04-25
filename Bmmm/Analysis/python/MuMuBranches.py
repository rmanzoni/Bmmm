branches = [
    'run'               ,
    'lumi'              ,
    'event'             ,
    'ncands'            ,
    'npv'               ,
    'n_pu'              ,
    'n_true_int'        ,

    'mass'              ,
    'mcorr'             ,
    'pt'                ,
    'eta'               ,
    'phi'               ,
    'charge'            ,

    'dr'                ,
    'dr_max'            ,
    'dr_12'             ,

    'pv_x'              ,
    'pv_y'              ,
    'pv_z'              ,

    'bs_x0'             ,
    'bs_y0'             ,
    'bs_z0'             ,

    'bs_x'              ,
    'bs_y'              ,

    'vx'                ,
    'vy'                ,
    'vz'                ,
    'vtx_chi2'          ,
    'vtx_prob'          ,

    'cos2d'             ,
    'lxy'               ,
    'lxy_err'           ,
    'lxy_sig'           ,
]

muon_branches = [
    'pt'             ,
    'eta'            ,
    'phi'            ,
    'e'              ,
    'mass'           ,
    'charge'         ,
    'id_loose'       ,
    'id_soft'        ,
    'id_medium'      ,
    'id_tight'       ,
    'id_soft_mva_raw',
    'id_soft_mva'    ,
    'id_pf'          ,
    'id_global'      ,
    'id_tracker'     ,
    'id_standalone'  ,
    'pfiso03'        ,
    'pfiso04'        ,
    'pfreliso03'     ,
    'pfreliso04'     ,
    'pfiso03_ch'     ,
    'pfiso03_cp'     ,
    'pfiso03_nh'     ,
    'pfiso03_ph'     ,
    'pfiso03_pu'     ,
    'pfiso04_ch'     ,
    'pfiso04_cp'     ,
    'pfiso04_nh'     ,
    'pfiso04_ph'     ,
    'pfiso04_pu'     ,
    'dxy'            ,
    'dxy_e'          ,
    'dxy_sig'        ,
    'dz'             ,
    'dz_e'           ,
    'dz_sig'         ,
    'bs_dxy'         ,
    'bs_dxy_e'       ,
    'bs_dxy_sig'     ,
    'cov_pos_def'    ,
    'jet_pt'         ,
    'jet_eta'        ,
    'jet_phi'        ,
    'jet_e'          ,
    'gen_pt'         ,
    'gen_eta'        ,
    'gen_phi'        ,
    'gen_e'          ,
    'gen_pdgid'      ,
]

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
# paths['HLT_Mu12'   ] = ['hltL3fL1sMu15DQlqL1f0L2f10L3Filtered12']




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

#    'HLT_Mu7_IP4'     ,
#    'HLT_Mu8_IP3'     ,
#    'HLT_Mu8_IP5'     ,
#    'HLT_Mu8_IP6'     ,
#    'HLT_Mu8p5_IP3p5' ,
#    'HLT_Mu9_IP4'     ,
#    'HLT_Mu9_IP5'     ,
#    'HLT_Mu9_IP6'     ,
#    'HLT_Mu10p5_IP3p5',
#    'HLT_Mu12_IP6'    ,

# add branches for T&P
for k, v in paths.items():
    for idx in [1,2]:
        branches.append('mu%d_%s_tag' %(idx, k))
        branches.append('mu%d_%s_probe' %(idx, k))
    
branches += paths
branches += [path+'_ps' for path in paths]
