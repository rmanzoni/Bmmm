from copy import copy

cuts = {} 

cuts['baseline'] = {}

cuts['baseline']['to_pt'   ] = 3.
cuts['baseline']['to_eta'  ] = 2.6
cuts['baseline']['to_dr'   ] = 0.1
cuts['baseline']['mu_pt'   ] = 1.
cuts['baseline']['mu_eta'  ] = 2.5
cuts['baseline']['mu_id'   ] = lambda mu : mu.isPFMuon() and mu.isGlobalMuon()
cuts['baseline']['min_mass'] = 4
cuts['baseline']['max_mass'] = 7
cuts['baseline']['max_dz'  ] = 1.2 
cuts['baseline']['hlt'     ] = 'HLT_DoubleMu4_3_LowMass'
cuts['baseline']['hlt_dr'  ] = 0.1 
cuts['baseline']['jet_dr'  ] = 0.2 
cuts['baseline']['gen_dr'  ] = 0.02

######################################

cuts['4mu'  ] = copy(cuts['baseline'])
cuts['2mu2k'] = copy(cuts['baseline'])

######################################
cuts['4mu']['min_mass'] = 4.5
cuts['4mu']['max_mass'] = 6.5
cuts['4mu']['mu_dxy'] = 1.2

######################################

cuts['2mu2k']['min_mass'] = 4.8
cuts['2mu2k']['max_mass'] = 6.
cuts['2mu2k']['mu_pt' ] = 3.
cuts['2mu2k']['tk_pt' ] = 1.5
cuts['2mu2k']['tk_eta'] = 2.5
cuts['2mu2k']['tk_dxy'] = 1.2
cuts['2mu2k']['tk_dz' ] = 24.
cuts['2mu2k']['max_dz_presel' ] = 4.
cuts['2mu2k']['tk_id' ] = lambda tk : tk.trackHighPurity()
cuts['2mu2k']['max_dr_k_mm'] = 1.2
cuts['2mu2k']['min_dimuon_mass'] = 3.1 - 0.25
cuts['2mu2k']['max_dimuon_mass'] = 3.1 + 0.25
cuts['2mu2k']['min_dikaon_mass'] = 1.02 - 0.03
cuts['2mu2k']['max_dikaon_mass'] = 1.02 + 0.03
cuts['2mu2k']['dr_cleaning'] = 0.01 # reduce!
cuts['2mu2k']['dpt_cleaning'] = 0.01 # 0.01 corresponds to 1%
