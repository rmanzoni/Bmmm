branches = [
    'run'               ,
    'lumi'              ,
    'event'             ,
    'npv'               ,
    'neles'             ,
    'neecands'          ,
    
    'mass'              ,
    'mcorr'             ,
    'pt'                ,
    'eta'               ,
    'phi'               ,
    'charge'            ,
    'energy'            ,

    'dr'                ,
    'dr_max'            ,

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

    'e1_pt'             ,
    'e1_eta'            ,
    'e1_phi'            ,
    'e1_e'              ,
    'e1_mass'           ,
    'e1_charge'         ,
    'e1_energy'         ,
    'e1_id_loose'       ,
    'e1_id_wp90'        ,
    'e1_id_wp80'        ,
    'e1_dxy'            ,
    'e1_dxy_e'          ,
    'e1_dxy_sig'        ,
    'e1_dz'             ,
    'e1_dz_e'           ,
    'e1_dz_sig'         ,
    'e1_bs_dxy'         ,
    'e1_bs_dxy_e'       ,
    'e1_bs_dxy_sig'     ,
    'e1_cov_pos_def'    ,
    'e1_det_cov'        ,

    'e2_pt'             ,
    'e2_eta'            ,
    'e2_phi'            ,
    'e2_e'              ,
    'e2_mass'           ,
    'e2_charge'         ,
    'e2_energy'         ,
    'e2_id_loose'       ,
    'e2_id_wp90'        ,
    'e2_id_wp80'        ,
    'e2_dxy'            ,
    'e2_dxy_e'          ,
    'e2_dxy_sig'        ,
    'e2_dz'             ,
    'e2_dz_e'           ,
    'e2_dz_sig'         ,
    'e2_bs_dxy'         ,
    'e2_bs_dxy_e'       ,
    'e2_bs_dxy_sig'     ,
    'e2_cov_pos_def'    ,
    'e2_det_cov'        ,

]

paths = [
    'HLT_DoubleEle4_eta1p22_mMax6_v1'  ,
    'HLT_DoubleEle4p5_eta1p22_mMax6_v1',
    'HLT_DoubleEle5_eta1p22_mMax6_v1'  ,
    'HLT_DoubleEle5p5_eta1p22_mMax6_v1',
    'HLT_DoubleEle6_eta1p22_mMax6_v1'  ,
    'HLT_DoubleEle6p5_eta1p22_mMax6_v1',
    'HLT_DoubleEle7_eta1p22_mMax6_v1'  ,
    'HLT_DoubleEle7p5_eta1p22_mMax6_v1',
    'HLT_DoubleEle8_eta1p22_mMax6_v1'  ,
    'HLT_DoubleEle8p5_eta1p22_mMax6_v1',
    'HLT_DoubleEle9_eta1p22_mMax6_v1'  ,
    'HLT_DoubleEle9p5_eta1p22_mMax6_v1',
    'HLT_DoubleEle10_eta1p22_mMax6_v1' ,
]

branches += paths
branches += [path+'_ps' for path in paths]



