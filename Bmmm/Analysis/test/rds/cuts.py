cuts_loose = {}

cuts_loose['HLT'            ] = 'HLT_Mu7_IP4'
cuts_loose['mu_trig_match'  ] = 0.1
cuts_loose['mu_pt'          ] = 7. # GeV
cuts_loose['mu_eta'         ] = 1.5 
cuts_loose['mu_basic_id'    ] = lambda mu : (mu.isPFMuon() and mu.isGlobalMuon())
cuts_loose['tk_pt'          ] = 0.7 # GeV
cuts_loose['tk_eta'         ] = 2.1 
cuts_loose['tk_dxy'         ] = 0.6 # cm 
cuts_loose['min_dr_m_tk'    ] = 0.005 
cuts_loose['max_dr_m_tk'    ] = 1.2 
cuts_loose['max_dz_m_tk'    ] = 0.6 # cm
cuts_loose['max_dr_k1_k1'   ] = 0.6 
cuts_loose['phi_mass_window'] = 0.03 # GeV 
cuts_loose['ds_mass_window' ] = 0.15 # GeV 
cuts_loose['max_bs_mass'    ] = 8. # GeV 
cuts_loose['phi_vtx_prob'   ] = 1e-3 # GeV 
cuts_loose['ds_vtx_prob'    ] = 1e-3 # GeV 

cuts_tight = cuts_loose.copy()

cuts_tight['tk_pt'          ] = 1.5 # GeV
cuts_tight['tk_eta'         ] = 2.0 
cuts_tight['tk_dxy'         ] = 0.5 # cm 
cuts_tight['phi_mass_window'] = 0.025 # GeV 
cuts_tight['ds_mass_window' ] = 0.12 # GeV 
cuts_tight['phi_vtx_prob'   ] = 1e-2 # GeV 
cuts_tight['ds_vtx_prob'    ] = 1e-2 # GeV 


cuts_gen = {}

cuts_gen['mu_pt'          ] = 7. # GeV
cuts_gen['mu_eta'         ] = 1.5 
cuts_gen['k_pt'           ] = 0.7 # GeV
cuts_gen['k_eta'          ] = 2.1 
cuts_gen['k_dxy'          ] = 0.6 # cm 
cuts_gen['pi_pt'          ] = 0.7 # GeV
cuts_gen['pi_eta'         ] = 2.1 
cuts_gen['pi_dxy'         ] = 0.6 # cm 
cuts_gen['max_bs_mass'    ] = 8. # GeV 
