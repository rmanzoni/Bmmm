'''
R(Ds) preselection, AN-24-173 v4 Sec. 5.1-5.3, with the changes agreed for the
Bmmm ntuples (29 Oct 2026):

  * the m(KKpi) window is widened to include D+ -> phi pi (1.87 GeV), for the
    mass-scale calibration and the reflection studies;
  * no candidate arbitration: every candidate passing these cuts is a row,
    wrong-sign ones included;
  * no trigger-matching requirement: the muon's match to the parking objects
    is stored (mu_trig_dr, mu_trig_match, mu_<path>_tag) and applied offline;
  * the vertex-probability cuts are kept as in the AN (P(V_phi) with the phi
    constraint, P(V_Ds) of phi + pi without the Ds constraint), but a
    candidate passes if EITHER the phi-constrained or the free fit passes
    them (RDsCandidate.CUT_SCHEMES), so all schemes are compared on the same
    candidates.

The keys used by the shared BaseInspector muon selection are mu_pt, mu_eta,
mu_id, mu_dxy; to_pt / to_eta preselect the trigger objects; hlt_dr is the
trigger-matching radius.
'''

from Bmmm.Analysis.RDsKinematics import M_PHI

cuts = {}
cuts['rds'] = {
    # ---- trigger objects (2018 parking muons: pT > 7, |eta| < 1.5) ----
    'to_pt'          : 5.0,     # [GeV]
    'to_eta'         : 2.0,
    'hlt_dr'         : 0.05,    # AN Sec. 5.1

    # ---- muon (AN Sec. 5.1) ----
    'mu_pt'          : 7.0,     # [GeV]
    'mu_eta'         : 1.5,
    'mu_id'          : lambda mu : mu.isPFMuon(),
    'mu_dxy'         : 1.0e9,   # [cm] no cut (the BaseInspector selection needs the key)

    # ---- tracks (AN Sec. 5.2) ----
    'tk_pt'          : 0.7,     # [GeV]
    'tk_eta'         : 2.4,
    'tk_veto_pdgid'  : (11, 13),
    'min_dr_mu_tk'   : 0.005,
    'max_dr_mu_tk'   : 1.2,
    'max_dz_mu_tk'   : 1.0,     # [cm] |dz(mu) - dz(tk)|, wrt the PV closest in z to the muon
    'max_dxy_tk'     : 0.6,     # [cm] |dxy(tk)|, wrt the same PV

    # ---- resonances (AN Sec. 5.3, Ds window widened) ----
    'phi_window'     : 0.020,   # [GeV] |m(KK) - m(phi)|, raw momenta
    'phi_mass'       : M_PHI,
    'kkpi_min'       : 1.75,    # [GeV] m(KKpi), raw momenta; includes D+ (1.870) and Ds (1.968)
    'kkpi_max'       : 2.15,    # [GeV]
    'max_mass'       : 8.0,     # [GeV] m(KKpi mu), raw momenta

    # ---- vertex probabilities (AN Sec. 5.3), OR of phic and free ----------
    'phi_vtx_prob'   : 0.01,
    'ds_vtx_prob'    : 0.01,

    # ---- matching radii ----
    'jet_dr'         : 0.2,     # as the other channels
    'gen_dr'         : 0.03,

    # ---- soft photon (AN Sec. 5.5) ----
    'gamma_window'   : 0.100,   # [GeV] |m(Ds gamma) - m(Ds*)|
}
