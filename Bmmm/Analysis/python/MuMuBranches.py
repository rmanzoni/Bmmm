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

paths['HLT_Mu7p5_Track2_Jpsi'                     ] = ['hltL3fLMu7p5TrackL3Filtered7p5', 'hltMu7p5Track2JpsiTrackMassFiltered'  ]
paths['HLT_Mu7p5_Track3p5_Jpsi'                   ] = ['hltL3fLMu7p5TrackL3Filtered7p5', 'hltMu7p5Track3p5JpsiTrackMassFiltered']
paths['HLT_Mu7p5_Track7_Jpsi'                     ] = ['hltL3fLMu7p5TrackL3Filtered7p5', 'hltMu7p5Track7JpsiTrackMassFiltered'  ]
paths['HLT_Mu8'                                   ] = ['hltL3fL1sMu5L1f0L2f5L3Filtered8']
paths['HLT_Mu17'                                  ] = ['hltL3fL1sMu10lqL1f0L2f10L3Filtered17']
paths['HLT_Mu19'                                  ] = ['hltL3fL1sMu10lqL1f0L2f10L3Filtered19']
paths['HLT_DoubleMu4_3_Jpsi'                      ] = ['hltmumuFilterDoubleMu43Jpsi', 'hltmumuFilterDoubleMu43Jpsi']
paths['HLT_Dimuon0_Jpsi_NoVertexing'              ] = ['hltDimuon0JpsiL3Filtered', 'hltDimuon0JpsiL3Filtered']
paths['HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R'] = ['hltDimuon0JpsiL1s4R0er1p5RL3Filtered', 'hltDimuon0JpsiL1s4R0er1p5RL3Filtered']
paths['HLT_IsoMu24'                               ] = ['hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p07']

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
