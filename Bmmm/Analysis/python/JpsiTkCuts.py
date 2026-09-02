from copy import copy
from Bmmm.Analysis.JpsiChargedCuts import cuts

# J/psi + track (B+ -> J/psi K+ / Bc+ -> J/psi pi+) selection.
# Inherit the ENTIRE RJpsi selection (trigger-object, muon pt/eta/id, tight-muon
# pt, J/psi mass window, muon dxy, HLT path + matching dR, jet-match dR, gen dR)
# so the muon side is selected identically to the J/psi mu analysis.
cuts['jpsi_tk'] = copy(cuts['rjpsi'])

# ---- bachelor track (pseudo-track) selection ----------------------------------
cuts['jpsi_tk']['k_pt']    = 2.0    # [GeV]
cuts['jpsi_tk']['k_eta']   = 2.5
cuts['jpsi_tk']['k_dxy']   = 2.0    # [cm], wrt the track's own reference point
cuts['jpsi_tk']['k_dz']    = 20.0   # [cm]
cuts['jpsi_tk']['k_dr_mu'] = 0.01   # overlap removal vs a J/psi muon (same charge, dR < this)

# ---- candidate mass window (kept if EITHER hypothesis mass falls in-window) ----
# B+ ~ 5.279 GeV (kaon), Bc+ ~ 6.274 GeV (pion): one loose window covers both.
cuts['jpsi_tk']['min_mass'] = 4.5   # [GeV]
cuts['jpsi_tk']['max_mass'] = 7.5   # [GeV]
