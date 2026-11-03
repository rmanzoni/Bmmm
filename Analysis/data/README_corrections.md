# Corrections used by the R(Ds) ntuples (2018 B-parking, UL MC)

Every correction is MC only; data are never corrected. Each one is a
directory with the payload the ntuplizer reads, the code that derived or
applies it, and its provenance. Plots, fit caches and ROOT files stay out of
git (`.gitignore`); they are on disk at PSI only.

| correction | ntuplizer option | payload | applied to |
|---|---|---|---|
| track momentum scale + smearing (momcal v4b) | `--trk-calib` | `trkcalib/RDs_2018BParking_trkcalib_v4b_base_laptop/rds_trkcalib_RDs_2018BParking_trkcalib_v4b_base_laptop.json` (applier: `applier/python/rds_trkcalib.py` of the same directory) | kaons, pion (`kind=hadron`), muon (`kind=muon`), before combinatorics and fits |
| muon sigma_dxy quantile morph | `--cov-scale ... --cov-scale-muon-only` | `impact_parameter_quantile_morph/dxy_morphing/sigma_dxy_morph_splot_fixed_muonid_fixed_decay_length.json` (correctionlib, `mu_bs_dxy_e_morph`) | muon track covariance (dxy), before the fits |
| muon trigger scale factors | `--trig-sf` | `bparking_trigger_sf/scale_factors.json`, map `pt_vs_dxy_sig` (derived by `efficiency_sf_splot.py`) | per-candidate weight branches `mu_trig_sf{,_up,_dn,_inrange}` |
| pileup | (offline) | not yet: `nti` is stored in both trees | |

Also offline: B lifetime reweighting from `gen_bs_ct` (event, signal decay)
and `gen_ds_b_ct`, `gen_mu_b_ct` (candidate, the b hadron the Ds / the muon
comes from; with `gen_ds_b_pdgid`, `gen_mu_b_pdgid`).

Production recipe: the `rds-production` Claude skill and
`test/rds/submitter_rds_slurm.py`.
