branches = [
    'run'               ,
    'lumi'              ,
    'event'             ,
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

    'mu1_pt'            ,
    'mu1_eta'           ,
    'mu1_phi'           ,
    'mu1_e'             ,
    'mu1_mass'          ,
    'mu1_charge'        ,
    'mu1_id_loose'      ,
    'mu1_id_soft'       ,
    'mu1_id_medium'     ,
    'mu1_id_tight'      ,
    'mu1_id_soft_mva_raw',
    'mu1_id_soft_mva'   ,
    'mu1_id_pf'         ,
    'mu1_id_global'     ,
    'mu1_id_tracker'    ,
    'mu1_id_standalone' ,
    'mu1_pfiso03'       ,
    'mu1_pfiso04'       ,
    'mu1_pfreliso03'    ,
    'mu1_pfreliso04'    ,
    'mu1_dxy'           ,
    'mu1_dxy_e'         ,
    'mu1_dxy_sig'       ,
    'mu1_dz'            ,
    'mu1_dz_e'          ,
    'mu1_dz_sig'        ,
    'mu1_bs_dxy'        ,
    'mu1_bs_dxy_e'      ,
    'mu1_bs_dxy_sig'    ,
    'mu1_cov_pos_def'   ,

    'mu2_pt'            ,
    'mu2_eta'           ,
    'mu2_phi'           ,
    'mu2_e'             ,
    'mu2_mass'          ,
    'mu2_charge'        ,
    'mu2_id_loose'      ,
    'mu2_id_soft'       ,
    'mu2_id_medium'     ,
    'mu2_id_tight'      ,
    'mu2_id_soft_mva_raw',
    'mu2_id_soft_mva'   ,
    'mu2_id_pf'         ,
    'mu2_id_global'     ,
    'mu2_id_tracker'    ,
    'mu2_id_standalone' ,
    'mu2_pfiso03'       ,
    'mu2_pfiso04'       ,
    'mu2_pfreliso03'    ,
    'mu2_pfreliso04'    ,
    'mu2_dxy'           ,
    'mu2_dxy_e'         ,
    'mu2_dxy_sig'       ,
    'mu2_dz'            ,
    'mu2_dz_e'          ,
    'mu2_dz_sig'        ,
    'mu2_bs_dxy'        ,
    'mu2_bs_dxy_e'      ,
    'mu2_bs_dxy_sig'    ,
    'mu2_cov_pos_def'   ,
]


# paths and filters
# check online confDB https://hlt-config-editor-confdbv3.app.cern.ch/

paths = dict()

paths['HLT_Mu7p5_Track2_Jpsi'  ] = ['hltL3fLMu7p5TrackL3Filtered7p5', 'hltMu7p5Track2JpsiTrackMassFiltered'  ]
paths['HLT_Mu7p5_Track3p5_Jpsi'] = ['hltL3fLMu7p5TrackL3Filtered7p5', 'hltMu7p5Track3p5JpsiTrackMassFiltered']
paths['HLT_Mu7p5_Track7_Jpsi'  ] = ['hltL3fLMu7p5TrackL3Filtered7p5', 'hltMu7p5Track7JpsiTrackMassFiltered'  ]
paths['HLT_Mu8'                ] = ['hltL3fL1sMu5L1f0L2f5L3Filtered8']

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
