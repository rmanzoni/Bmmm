# Per-candidate m(KKπ) uncertainty from the track covariances

`sigma_m_demo.py` computes σ(m_KKπ) from the (q/p, λ, φ) block of the K1, K2, π helix
covariances (analytic Jacobian, tracks independent), with an optional momentum-calibration
term added to the q/p variance and evaluated without z. Method: the note
`sigma_m_kkpi.pdf` (σ(m_KKπ) from calibrated tracks).

```
source ../compare_fits/cfenv.sh
python3 sigma_m_demo.py /pnfs/psi.ch/cms/trivcat/store/user/manzoni/rds_bs_dslnu_ul18_v3_500f_dsc_02nov26 --files 50 -o demo_50f
```

## Results: 50 of 500 chunks, 778k truth-matched signal candidates (2026-11-03)

| check | what it tests | result |
|---|---|---|
| m recomputed vs `kkpi_mass` | inputs, mass hypotheses | median \|Δm\| 5e-8 GeV |
| A: Jacobian vs finite differences | the derivation | 99th percentile of relative difference 1.6e-6 |
| B: toy round trip (gen → smear with each track's C → σ_m) | propagation, linearity | pull core width 0.998 |
| C: same + calibration-like smear f = 1 + s z | the σ_cal term | pull 1.326 without it, 1.001 with it |
| D: reconstruction, (m_reco − m_gen)/σ_m | the reco covariance in MC | core 1.13, s68 1.15, median +0.07 |
| E: residual width per σ_m decile | σ_m as a category variable | linear, slope 1.15, intercept −0.15 MeV |

- σ_m: median 9.0 MeV, 5–95% range 6.1–18.7 MeV. The distribution is bimodal: a barrel peak at about 7 MeV and a second peak at about 11 MeV.
- Variance budget: π q/p 68%, K1 q/p 15%, K2 q/p 5%, angles 13%, correlations about 0.
- 52% of candidates have at least one track whose 5×5 packed covariance is not positive definite (`cov_pos_def`).
  The 3×3 momentum block is positive definite for all but about 1e-4 of the candidates. The pull is the same with and without a non-positive-definite 5×5 (1.12 vs 1.14), so these candidates are kept.
- The MC reco covariance underestimates the core resolution by 13% on average. The underestimate grows from 9% to 18% across the σ_m deciles, from 1.11 in the barrel to 1.16 in the endcap, and up to 1.20 for a minimum track pT around 4 GeV.
- Tails: ⟨pull²⟩ = 2.46 (1.21 within ±3). 3.4% of candidates are beyond 3σ and 1.0% beyond 5σ, against 0.27% and 6e-7 for a Gaussian.
  Against the PDG mass ⟨pull²⟩ = 4.27, which is the ≈4 per degree of freedom of the D_s mass constraint in the dsc scheme (cmp_v3). That excess therefore comes from the tails, including FSR, and not from a global underestimate of the errors.

## Not done yet
- The real σ_cal needs `rds_trkcalib.py` and its JSON, which are not in `Analysis/data/trkcalib/` (README only). C uses a toy σ_cal (0.6–1.2% vs η, + 0.1%/GeV).
- No calibrated MC and no data yet: validation checks 3–5 of the note remain.
