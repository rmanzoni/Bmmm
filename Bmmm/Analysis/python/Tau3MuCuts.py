from copy import copy

# ----------------------------------------------------------------------------
# Selection cuts for the displaced tau -> 3mu analysis
#     Ds -> tau nu , tau -> mu a , a -> mu mu        (a long-lived, pdgId 9900015)
#
# The whole tau decay is visible, so m(3mu) ~ m(tau) = 1.777 GeV is a narrow
# peak, while m(mu mu) of the displaced OS pair = m(a) is the SEARCH variable and
# is therefore NOT windowed here. Because the muons come from a long-lived
# particle the transverse-IP (dxy) cut is deliberately loose -- a tight dxy would
# throw the signal away.
# ----------------------------------------------------------------------------

cuts = {}

cuts['baseline'] = {}
cuts['baseline']['to_pt'        ] = 2.
cuts['baseline']['to_eta'       ] = 2.6
cuts['baseline']['to_dr'        ] = 0.1
cuts['baseline']['mu_pt'        ] = 2.
cuts['baseline']['mu_eta'       ] = 2.5
cuts['baseline']['mu_id'        ] = lambda mu : mu.isPFMuon() and (mu.isGlobalMuon() or mu.isTrackerMuon())
cuts['baseline']['mu_dxy'       ] = 20.    # cm; LOOSE on purpose (displaced muons)
cuts['baseline']['tight_mu_pt'  ] = 3.     # leading two (trigger) muons
cuts['baseline']['min_mass'     ] = 0.
cuts['baseline']['max_mass'     ] = 10.
cuts['baseline']['max_dz'       ] = 1.2
cuts['baseline']['hlt'          ] = 'HLT_DoubleMu4_3_LowMass'
cuts['baseline']['hlt_dr'       ] = 0.1
cuts['baseline']['jet_dr'       ] = 0.2
cuts['baseline']['gen_dr'       ] = 0.03
cuts['baseline']['pf_cone_dr'   ] = 0.6    # R of the PF-candidate cone (3mu axis)
cuts['baseline']['pf_min_pt'    ] = 0.     # min pt of a PF candidate to be stored

######################################
cuts['tau3mu'] = copy(cuts['baseline'])
# the displaced OS pair (a -> mu mu)
cuts['tau3mu']['pair_max_mass'] = 4.       # loose upper bound on m(mu mu)
# the full tau candidate, m(3mu) ~ m(tau)
cuts['tau3mu']['min_3mu_mass' ] = 1.2
cuts['tau3mu']['max_3mu_mass' ] = 2.4
