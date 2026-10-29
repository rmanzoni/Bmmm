# R(Ds) ntuples from the 2018 B-parking data

Bs → Ds(→ φ(K K) π) μ ν candidates, one row per candidate, from the 22 UL2018
MiniAODv2 datasets `ParkingBPH1–6` and the UL2018 MC. Third channel of the
J/ψ + charged-object ntuplizers: argument parsing, lumi mask, trigger
bookkeeping, batched uproot output and cutflow are the shared ones
(`JpsiChargedInspector.BaseInspector`); the 2018 parking trigger is the code of
the Vcb parking ntuples (`BParking2018Trigger`).

**Status (29 Oct 2026): first version, never run in CMSSW.** What has been run is
listed under *Checks* below. The first job on a real file will find what
the fake-event tests could not.

## Files

| file | content |
|---|---|
| `test/rds/inspector_rds.py` | entry point |
| `python/RDsInspector.py` | per-event logic: track selection, combinatorics, gen tree, filling |
| `python/RDsCandidate.py` | per-candidate reconstruction: fits, PV, displacements, Bs hypotheses, photon, isolation |
| `python/RDsKinematics.py` | ROOT-free kinematics: masses, Bs hypotheses, q², helicity angles, reflections, combinatorics |
| `python/RDsGenHistory.py` | generator truth: signal decay, Bs flight, candidate classification (AN Sec. 6.1) |
| `python/RDsBranches.py`, `python/RDsCuts.py` | schema and preselection |
| `interface/RDsKinVtxFitter.h` | the two sequential fit schemes |
| `python/TrackMomentumCalib.py` | momcal correction at production (MC) |
| `python/L1Seeds2018.py` | L1 seed decisions and prescales |
| `python/HammerDsFF.py`, `test/rds/hammer/` | Bs → Ds(*) form-factor weights, production time or afterwards |
| `test/rds/crab/`, `test/rds/submitter_rds_slurm.py` | CRAB and PSI SLURM submission |
| `test/rds/test_rds_kinematics.py` | unit tests of the ROOT-free parts |

## What is selected

| step | selection |
|---|---|
| trigger | OR of the ten `HLT_Mu*_IP*` paths (MC with `--savenontrig`: every event) |
| muon | pT > 7 GeV, \|η\| < 1.5, PF. **No trigger match required**: stored as `mu_trig_dr`, `mu_trig_match` (ΔR < 0.05), `mu_<path>_tag` |
| tracks | packedPFCandidates + lostTracks, pT > 0.7 GeV, \|η\| < 2.4, charged, with track details, not e/μ; 0.005 < ΔR(μ) < 1.2; \|dz(μ) − dz(t)\| < 1 cm and \|dxy(t)\| < 0.6 cm wrt the PV closest in z to the muon |
| masses (raw momenta) | \|m(KK) − m(φ)\| < 20 MeV; **1.75 < m(KKπ) < 2.15 GeV** (D⁺ and Ds); m(KKπμ) < 8 GeV |
| charges | none: right-sign, μπ-wrong and KK-wrong candidates are all kept (`charge_class` 0/1/2) |
| vertices | P(V_φ) > 0.01 and P(V_Ds) > 0.01 in **either** fit scheme |
| arbitration | none: every candidate is a row; `cand_idx` numbers them by decreasing pT(KKπμ), `ncands` counts them |

## The two fit schemes

Every fit-dependent quantity is written twice, with the prefix `cons_` or `free_`.

| | `cons_` | `free_` |
|---|---|---|
| φ vertex (`*_phi_*`) | KK with the φ mass constraint | KK, no constraint |
| `cons_dsphi_*` | φ + π without the Ds constraint: `cons_dsphi_mass` is m(KKπ) with the φ constraint (the AN v4 definition) | — |
| Ds vertex (`*_ds_*`) | φ + π with the Ds mass constraint; its momentum enters the Bs kinematics | KK + π, no constraint |
| Bs vertex (`*_bs_*`) | Ds + μ | KKπ + μ |

Scheme-independent masses from the (calibrated) track momenta: `kk_mass`,
`kkpi_mass`, `mass` (KKπμ), and the reflection masses `refl_k{1,2}_as_pi`
(D⁺ → Kππ), `refl_k{1,2}_as_p` (Λc → pKπ), `refl_k{1,2}_pi` (D⁰ → Kπ).

The primary vertex (`pv_*`) is shared by the two schemes: the PV closest to the
line through the Bs vertex (cons, else free) along the KKπμ momentum, refitted
with the beamspot constraint and the four tracks removed (the same code as the
RJpsi and dimuon ntuples). `pv_refit_valid` = 0 means the hybrid PV (beamspot
x, y, PV z) was used.

Per scheme, Bs momentum hypotheses `coll`, `xyz_bs`, `xyz_ds` (equal velocity
along PV → V_Bs, PV → V_Ds), `math1/2_bs`, `math1/2_ds` (the two neutrino
solutions along each direction; `*_math_real_*` = 0 when the discriminant was
negative and clamped). For each: `pt eta phi beta q2 m_miss2 pt_miss pt_miss_sca
e_star_mu e_hash_mu cos_theta_phi_ds cos_theta_mu_w chi`, e.g.
`cons_math1_bs_q2`. Also `*_cos_theta_k_pi`, `*_mcorr`, `*_p_perp`, the muon
impact parameters wrt the Ds flight line (`*_mu_ip3d_ds_{pv,sv}`,
`*_mu_dl3d_ds` = the AN "IP3D", `*_mu_dist_to_ds_dir`), the Ds vertex wrt the
Bs vertex (`*_ds_lxy_bs`, `*_ds_cos3d_bs`), and the soft photon (`*_gamma_*`,
AN Sec. 5.5, with q² and m²_miss recomputed with Ds + γ).

## Production-time corrections (MC)

All go through one place (`TrackRecord.fit_track`), so the fits, the IPs and
their significances see the same track. Order:

1. a non-positive-definite packed covariance is repaired (`*_cov_fixed`);
2. `--trk-calib <rds_trkcalib_2018BParking_v4.json>[:variation]`: momentum
   calibration on the kaons, the pion (hadron terms) and the muon (muon terms).
   The momenta used everywhere, the combinatorics included, are the corrected
   ones; q/p and its covariance row and column are scaled before the fits.
   Stored per track: `*_calib_f` (p'/p), `*_calib_z` (the Gaussian number),
   `*_pt_raw`. Copy `rds_trkcalib.py` and the JSON of the same momcal version
   into `data/trkcalib/` (see the README there);
3. `--cov-scale <table>`: covariance scaling, on the tracks and the muon.
   Besides the existing formats, the σ_dxy quantile morph of
   `test/mm/derive_sigma_correction_v3.py` is accepted as it is
   (`--cov-scale sigma_dxy_morph.json`): the beamspot part of its input error is
   kept and the track part rescaled (`utils.MorphCovScaler`). Stored:
   `*_cov_scale_<par>`; the raw covariance stays in `*_cov_<par>_<par>`.

`--covflow` is refused (the flows are Run 3). Pileup: `nti` and `npu` are
stored; the Run 3 `pu_weight_*` block is not in this schema.

## Generator truth (MC)

* `tree_gen`: one row per processed event, whatever the trigger or the
  reconstruction: `gen_sig` (0 Ds μν, 1 Ds* μν, 2 Ds τν, 3 Ds* τν, −1 none), Bs
  kinematics and flight (`gen_bs_pt`, `gen_bs_y`, `gen_bs_ct`, ... for the
  lifetime and FONLL weights), gen q², m²_miss, E*_μ, helicity angles, the
  visible particles, the Hammer inputs (`gen_ham_*`) and weights.
* in `tree`: the same event block, plus per candidate `gen_cand` (AN Sec. 6.1:
  0–3 signal, 10 Bs feed-down, 11 Bs double charm, 12 B⁰, 13 B⁺, 14 Λb,
  15 other b, 20 μ from another b, 21 μ not from a b, 30 not a Ds, −1 a track
  not matched), `gen_charm_pdgid` (the common charm ancestor of KKπ: Ds, D⁺,
  Λc, ...), `gen_ds_mother_pdgid`, `gen_ds_b_pdgid`, `gen_mu_b_pdgid`,
  `gen_same_b`, and per track `*_gen_pdgid`, `*_gen_mother_pdgid`.

Hammer: `--hammer <card>` at production, or
`test/rds/hammer/add_hammer_weights_rds.py` afterwards (same code, `--closure`
compares the two). The card format is in `HammerDsFF.py`;
`test/rds/hammer/card_template.json` lists what must be filled in from the
setup that produced the AN v4 weights. Not tested (no Hammer here).

## Running

Local test (tcsh, PSI, CMSSW_16_0_8):

```tcsh
cd $CMSSW_BASE/src && cmsenv && git checkout rds_ntuplizer && scram b -j 8
cd Bmmm/Analysis/test/rds
set f = `dasgoclient -query "file dataset=/ParkingBPH1/Run2018D-UL2018_MiniAODv2-v1/MINIAOD" -limit 1`
python3 inspector_rds.py --inputFiles=$f --filename=test_rds_data --lumi-mask=golden \
    --maxevents=500 --logfreq=50 --verbose
```

`--verbose` prints every getter that fails; expect the gen ones on data, and
nothing else. MC: add `--mc --savenontrig`, and the corrections above.

SLURM (PSI): `python3 submitter_rds_slurm.py --mode data|mc --files <list> --out-dir <name> [...]`
(`--dry-run` writes the job scripts only).

CRAB: `test/rds/crab/crab_rds_data.py` and `crab_rds_mc.py`, after
`./make_pylibs.sh`. Both start as costing runs (one dataset, `max_files = 5`).
The MC dataset names are placeholders and the submitter refuses to run until
they are replaced.

## Checks

Done here, without CMSSW:

* `test_rds_kinematics.py` (12 tests): combinatorics vs brute force; Bs
  hypotheses; q², m²_miss, E*_μ, E#_μ and the four helicity angles recovered on
  decays generated with known angles, in any frame; the neutrino solver with the
  Bs mass recovers the true Bs; gen classification of the four signals (with
  and without B mixing), feed-down, B⁰ double charm, D⁺ → φπ, foreign muons;
  the momcal applier loaded as in production; merged L1 tables equal to the
  per-path ones.
* With fake ROOT/FWLite objects (`test/rds/smoke/`, runs without CMSSW):
  `RDsInspector.main()` end to end on fake
  events (uproot output with `tree` and `tree_gen`, int64 event numbers,
  trigger tags, covariance scaling, cutflow); every quantity the candidate
  computes has a branch and every candidate branch is produced.
* The J/ψ μ, J/ψ + track and Vcb parking branch lists are identical before and
  after the shared-code changes.
* `RDsKinVtxFitter.h` compiles against stub headers (syntax and types only).

To do on the first real files: run with `--verbose`, compare `cons_` and
`free_` on signal MC, check the speed (ev/s in the log) before choosing the
CRAB splitting, and sync a few events against the previous ntuples.
