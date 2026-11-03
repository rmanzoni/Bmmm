plots/ numerical outputs — trigger efficiency & scale factors
=============================================================

JSON  (human-readable, NaN → null)
───────────────────────────────────
  efficiency_data.json      full 1D+2D binned efficiency for DATA
  efficiency_mc.json        full 1D+2D binned efficiency for MC
  scale_factors.json        1D+2D scale factors (SF = ε_data / ε_MC)

  Structure:
    {
      "1d": { "<var>": { bin_edges, bin_centers, n_total, n_pass,
                         efficiency, err_lo, err_hi }, ... },
      "2d": { "<v0>_vs_<v1>": { bin_edges_x/y, bin_centers_x/y,
                                n_total, n_pass,
                                efficiency, err_lo, err_hi }, ... }
    }

  Uncertainties are Clopper-Pearson (1σ ≈ 68.27 %), asymmetric.
  err_lo / err_hi are ERRORS (not bounds):
      lower bound = efficiency − err_lo
      upper bound = efficiency + err_hi

CSV – 1D (one row per bin)
──────────────────────────
  eff_1d_{var}.csv          var ∈ {pt, eta, dxy_sig}
  Columns: bin_lo, bin_hi, bin_center,
           n_total_data, n_pass_data, eff_data, err_lo_data, err_hi_data,
           n_total_mc,   n_pass_mc,   eff_mc,   err_lo_mc,   err_hi_mc,
           sf, sf_err_lo, sf_err_hi

CSV – 2D, tidy format (one row per cell)
─────────────────────────────────────────
  eff_2d_{v0}_vs_{v1}_tidy.csv
  Columns: bin_lo_x, bin_hi_x, bin_center_x,
           bin_lo_y, bin_hi_y, bin_center_y,
           n_total_data, n_pass_data, eff_data, err_lo_data, err_hi_data,
           n_total_mc,   n_pass_mc,   eff_mc,   err_lo_mc,   err_hi_mc,
           sf, sf_err_lo, sf_err_hi

CSV – 2D, matrix format (rows = var0 bins, cols = var1 bins)
─────────────────────────────────────────────────────────────
  eff_2d_{v0}_vs_{v1}_data.csv   DATA efficiency matrix
  eff_2d_{v0}_vs_{v1}_mc.csv     MC efficiency matrix
  eff_2d_{v0}_vs_{v1}_sf.csv     Scale factor matrix
  Each cell contains three sub-columns: value | err_lo | err_hi

NPZ  (NumPy binary, for programmatic access)
─────────────────────────────────────────────
  efficiency_arrays.npz
  Key naming convention:
    1D:  "1d__{var}__{quantity}"
    2D:  "{v0}_vs_{v1}__{quantity}"
  Quantities: bin_edges, bin_centers, bin_edges_x/y, bin_centers_x/y,
              n_total_data, n_pass_data, eff_data, err_lo_data, err_hi_data,
              n_total_mc,   n_pass_mc,   eff_mc,   err_lo_mc,   err_hi_mc,
              sf, sf_err_lo, sf_err_hi

  Usage example:
      import numpy as np
      d = np.load("plots/efficiency_arrays.npz")
      eff_pt_data = d["1d__pt__eff_data"]
      sf_2d       = d["pt_vs_dxy_sig__sf"]

