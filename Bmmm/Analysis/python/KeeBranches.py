branches = [
    'run'                     ,
    'lumi'                    ,
    'event'                   ,
    'npv'                     ,
    'neles'                   ,
    'nbcands'                 ,
    'neecands'                ,
      
    'b_mass'                  ,
    'b_mcorr'                 ,
    'b_pt'                    ,
    'b_eta'                   ,
    'b_phi'                   ,
    'b_tk_mass'               ,
    'b_tk_mcorr'              ,
    'b_tk_pt'                 ,
    'b_tk_eta'                ,
    'b_tk_phi'                ,
    'b_sc_mass'               ,
    'b_sc_mcorr'              ,
    'b_sc_pt'                 ,
    'b_sc_eta'                ,
    'b_sc_phi'                ,
    'b_charge'                ,
      
    'b_dr'                    ,
    'b_dr_max'                ,
    'b_dr_eek'                ,
      
    'b_abs_tk_iso'            ,
    'b_rel_tk_iso'            ,
          
    'ee_tk_mass'              ,
    'ee_tk_mcorr'             ,
    'ee_tk_pt'                ,
    'ee_tk_eta'               ,
    'ee_tk_phi'               ,
    'ee_sc_mass'              ,
    'ee_sc_mcorr'             ,
    'ee_sc_pt'                ,
    'ee_sc_eta'               ,
    'ee_sc_phi'               ,
    'ee_mass'                 ,
    'ee_mcorr'                ,
    'ee_pt'                   ,
    'ee_eta'                  ,
    'ee_phi'                  ,
    'ee_charge'               ,
      
    'ee_dr'                   ,
    'ee_dr_max'               ,
      
    'e1k_charge'              ,
    'e2k_charge'              ,
    'e1k_mass'                ,
    'e2k_mass'                ,
    'p1k_mass'                ,
    'p2k_mass'                ,
    'e1k_dr'                  ,
    'e2k_dr'                  ,
      
    'pv_x'                    ,
    'pv_y'                    ,
    'pv_z'                    ,
      
    'bs_x0'                   ,
    'bs_y0'                   ,
    'bs_z0'                   ,
      
    'bs_x'                    ,
    'bs_y'                    ,
      
    'ee_vx'                   ,
    'ee_vy'                   ,
    'ee_vz'                   ,
    'ee_vtx_chi2'             ,
    'ee_vtx_prob'             ,
      
    'ee_cos2d'                ,
    'ee_lxy'                  ,
    'ee_lxy_err'              ,
    'ee_lxy_sig'              ,
      
    'b_vx'                    ,
    'b_vy'                    ,
    'b_vz'                    ,
    'b_vtx_chi2'              ,
    'b_vtx_prob'              ,
      
    'b_cos2d'                 ,
    'b_lxy'                   ,
    'b_lxy_err'               ,
    'b_lxy_sig'               ,
      
    'ele1_pt'                 ,
    'ele1_eta'                ,
    'ele1_phi'                ,
    'ele1_e'                  ,
    'ele1_tk_pt'              ,
    'ele1_tk_eta'             ,
    'ele1_tk_phi'             ,
    'ele1_tk_e'               ,
    'ele1_sc_pt'              ,
    'ele1_sc_eta'             ,
    'ele1_sc_phi'             ,
    'ele1_sc_e'               ,
    'ele1_mass'               ,
    'ele1_charge'             ,
    'ele1_id_loose'           ,
    'ele1_id_wp90'            ,
    'ele1_id_wp80'            ,
    'ele1_dxy'                ,
    'ele1_dxy_e'              ,
    'ele1_dxy_sig'            ,
    'ele1_dz'                 ,
    'ele1_dz_e'               ,
    'ele1_dz_sig'             ,
    'ele1_bs_dxy'             ,
    'ele1_bs_dxy_e'           ,
    'ele1_bs_dxy_sig'         ,
    'ele1_cov_pos_def'        ,
    'ele1_det_cov'            ,
    'ele1_fbrem'              ,  # fbrem	 
    'ele1_deltaetain'         ,  # abs(deltaEtaSuperClusterTrackAtVtx)	 
    'ele1_deltaphiin'         ,  # abs(deltaPhiSuperClusterTrackAtVtx)	 
    'ele1_oldsigmaietaieta'   ,  # full5x5_sigmaIetaIeta	 
    'ele1_oldhe'              ,  # full5x5_hcalOverEcal
    'ele1_ep'                 ,  # eSuperClusterOverP	 
    'ele1_olde15'             ,  # full5x5_e1x5	 
    'ele1_eelepout'           ,  # eEleClusterOverPout	 
    'ele1_kfchi2'             ,  # closestCtfTrackNormChi2	 
    'ele1_kfhits'             ,  # closestCtfTrackNLayers	 
    'ele1_expected_inner_hits',  # gsfTrack.hitPattern.numberOfLostHits('MISSING_INNER_HITS')	 
    'ele1_convDist'           ,  # convDist	 
    'ele1_convDcot'           ,  # convDcot	  
    'ele1_r9'                 ,  # r9
    'ele1_r9_5x5'             ,  # full5x5_r9
    'ele1_scl_eta'            ,  # superCluster.eta 
    'ele1_dr03TkSumPt'        ,  # dr03TkSumPt 
    'ele1_dr03EcalRecHitSumEt',  # dr03EcalRecHitSumEt 
    'ele1_dr03HcalTowerSumEt' ,  # dr03HcalTowerSumEt

    'ele2_pt'                 ,
    'ele2_eta'                ,
    'ele2_phi'                ,
    'ele2_e'                  ,
    'ele2_tk_pt'              ,
    'ele2_tk_eta'             ,
    'ele2_tk_phi'             ,
    'ele2_tk_e'               ,
    'ele2_sc_pt'              ,
    'ele2_sc_eta'             ,
    'ele2_sc_phi'             ,
    'ele2_sc_e'               ,
    'ele2_mass'               ,
    'ele2_charge'             ,
    'ele2_id_loose'           ,
    'ele2_id_wp90'            ,
    'ele2_id_wp80'            ,
    'ele2_dxy'                ,
    'ele2_dxy_e'              ,
    'ele2_dxy_sig'            ,
    'ele2_dz'                 ,
    'ele2_dz_e'               ,
    'ele2_dz_sig'             ,
    'ele2_bs_dxy'             ,
    'ele2_bs_dxy_e'           ,
    'ele2_bs_dxy_sig'         ,
    'ele2_cov_pos_def'        ,
    'ele2_det_cov'            ,
    'ele2_fbrem'              ,  # fbrem	 
    'ele2_deltaetain'         ,  # abs(deltaEtaSuperClusterTrackAtVtx)	 
    'ele2_deltaphiin'         ,  # abs(deltaPhiSuperClusterTrackAtVtx)	 
    'ele2_oldsigmaietaieta'   ,  # full5x5_sigmaIetaIeta	 
    'ele2_oldhe'              ,  # full5x5_hcalOverEcal
    'ele2_ep'                 ,  # eSuperClusterOverP	 
    'ele2_olde15'             ,  # full5x5_e1x5	 
    'ele2_eelepout'           ,  # eEleClusterOverPout	 
    'ele2_kfchi2'             ,  # closestCtfTrackNormChi2	 
    'ele2_kfhits'             ,  # closestCtfTrackNLayers	 
    'ele2_expected_inner_hits',  # gsfTrack.hitPattern.numberOfLostHits('MISSING_INNER_HITS')	 
    'ele2_convDist'           ,  # convDist	 
    'ele2_convDcot'           ,  # convDcot	  
    'ele2_r9'                 ,  # r9
    'ele2_r9_5x5'             ,  # full5x5_r9
    'ele2_scl_eta'            ,  # superCluster.eta 
    'ele2_dr03TkSumPt'        ,  # dr03TkSumPt 
    'ele2_dr03EcalRecHitSumEt',  # dr03EcalRecHitSumEt 
    'ele2_dr03HcalTowerSumEt' ,  # dr03HcalTowerSumEt

    'k_pt'                    ,
    'k_eta'                   ,
    'k_phi'                   ,
    'k_e'                     ,
    'k_mass'                  ,
    'k_charge'                ,
    'k_dxy'                   ,
    'k_dxy_e'                 ,
    'k_dxy_sig'               ,
    'k_dz'                    ,
    'k_dz_e'                  ,
    'k_dz_sig'                ,
    'k_bs_dxy'                ,
    'k_bs_dxy_e'              ,
    'k_bs_dxy_sig'            ,
    'k_cov_pos_def'           ,
    'k_det_cov'               ,
]


branches_mc = [
    'ele1_gen_pt'   ,
    'ele1_gen_eta'  ,
    'ele1_gen_phi'  ,
    'ele1_gen_e'    ,
    'ele1_gen_match',
    
    'ele2_gen_pt'   ,
    'ele2_gen_eta'  ,
    'ele2_gen_phi'  ,
    'ele2_gen_e'    ,
    'ele2_gen_match',

    'k_gen_pt'      ,
    'k_gen_eta'     ,
    'k_gen_phi'     ,
    'k_gen_e'       ,
    'k_gen_match'   ,

    'ee_gen_pt'     ,
    'ee_gen_eta'    ,
    'ee_gen_phi'    ,
    'ee_gen_mass'   ,

    'b_gen_pt'      ,
    'b_gen_eta'     ,
    'b_gen_phi'     ,
    'b_gen_mass'    ,
    'b_gen_q2bin'   ,
    'b_gen_match'   ,
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



