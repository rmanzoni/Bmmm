from copy import copy

cuts = {} 

cuts['baseline'] = {}

cuts['baseline']['to_pt'   ] = 3.
cuts['baseline']['to_eta'  ] = 2.6
cuts['baseline']['to_dr'   ] = 0.1
cuts['baseline']['mu_pt'   ] = 3.
cuts['baseline']['mu_eta'  ] = 2.5
cuts['baseline']['mu_id'   ] = lambda mu : mu.isPFMuon() and mu.isGlobalMuon()
cuts['baseline']['min_mass'] = 0
cuts['baseline']['max_mass'] = 10
cuts['baseline']['max_dz'  ] = 1.2 
cuts['baseline']['hlt'     ] = 'HLT_DoubleMu4_3_LowMass'
cuts['baseline']['hlt_dr'  ] = 0.1 
cuts['baseline']['jet_dr'  ] = 0.2 
cuts['baseline']['gen_dr'  ] = 0.02

######################################

cuts['rjpsi'] = copy(cuts['baseline'])

######################################
cuts['rjpsi']['mu_dxy'] = 1.2
cuts['rjpsi']['jpsi_mass_window'] = 0.4
