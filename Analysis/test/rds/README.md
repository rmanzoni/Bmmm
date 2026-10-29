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
| `interface/RDsKinVtxFitter.h` | the single kinematic-fit steps the three schemes are chained from |
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
| vertices | P(V_φ) > 0.01 and P(V_Ds) > 0.01 in **either** the `phic` or the `free` fit scheme |
| arbitration | none: every candidate is a row; `cand_idx` numbers them by decreasing pT(KKπμ), `ncands` counts them |

## The three fit schemes

Every fit-dependent quantity is written three times, with the prefix `cons_`
(φ and Ds mass constraints), `phic_` (φ mass constraint only) or `free_` (no
mass constraint).

| | `cons_` | `phic_` | `free_` |
|---|---|---|---|
| φ vertex (`*_phi_*`) | KK with the φ mass constraint | the same fit as `cons_` | KK, no constraint |
| Ds vertex (`*_ds_*`) | φ + π with the Ds mass constraint | φ + π, no constraint: `phic_ds_mass` is m(KKπ) with the φ constraint (the AN v4 definition) | KK + π, no constraint |
| Bs vertex (`*_bs_*`) | Ds + μ | Ds + μ | KKπ + μ |

The Ds momentum of each scheme enters its Bs kinematics.

**Every vertex is read from its own fit.** A fit updates, in place, the top
particle of the tree it is built from: after the Bs fit the Ds of the Ds tree is
the Ds refitted at the Bs vertex, carrying the muon information (it moved by a
median of 4 MeV, by up to 12 GeV). Each tree is therefore read before it enters
the next fit, so `*_ds_*`, the Ds-tree children and the Ds trajectory are the Ds
of the Ds fit, and `*_phi_*` the φ of the φ fit. A composite enters one further
fit only (`cons` runs its own φ fit, identical to the `phic` one).

**The fitters.** Every step is the sequential recipe of the CMSSW KinematicFit
package (Prokofiev and Speer): a Kalman vertex fit of the inputs
(`KinematicParticleVertexFitter`), then, where a mass is constrained,
`KinematicParticleFitter` with `MassKinematicConstraint` (σ = 10 keV, i.e.
exact) on the resulting particle, which is what enters the next step. The
constraint updates the top particle and its decay vertex: `*_phi_{chi2,ndof,prob}`
of `cons`/`phic` and `cons_ds_*` are vertex + mass (2 ndof for two inputs), as
the vertex cut requires. The daughters keep their vertex-fit momenta, so
`*_{k1,k2}_rf_*` and the π of `cons_ds` are not mass-constrained.

This replaced `KinematicConstrainedVertexFitter` (one Lagrange-multiplier fit
of vertex and constraint), which broke numerically on the unconstrained fits.
It starts the vertex from a 100 cm² variance ("huge error method") and obtains
the ~1e-6 cm² final covariance as the difference of two terms of that size,
after inverting a matrix that mixes them with ~1e-8 track terms. On nearly
collinear inputs (boosted Ds, small ΔR(Ds, μ)) the rounding exceeded the true
transverse variance, and χ² came from the same algebra; its convergence test
divides by that covariance, so it could even stop on a broken one. On the
signal MC ~20% of the Bs vertices and ~15% of the free KK and KKπ ones came back
"valid" with χ² < 0 or a covariance that was not positive definite, and a
negative χ² gives P = 1, so broken vertices passed the vertex cuts: ~37% of the
non-signal candidates were there only because of that.

**Failed fits.** `*_<vtx>_good` = 0 and every other quantity of that vertex and
of the vertices after it in the chain are NaN when the fit failed or was not
run, and also, as a safety net, when a fit returned as valid has χ² < 0 or a
vertex covariance that is not positive definite (none with the current
fitters). `*_<vtx>_status` says which: 0 good, 1 no fit (failed or not run),
2 χ² < 0, 3 covariance not positive definite.

Scheme-independent masses from the (calibrated) track momenta: `kk_mass`,
`kkpi_mass`, `mass` (KKπμ), and the reflection masses `refl_k{1,2}_as_pi`
(D⁺ → Kππ), `refl_k{1,2}_as_p` (Λc → pKπ), `refl_k{1,2}_pi` (D⁰ → Kπ).

The primary vertex (`pv_*`) is shared by the three schemes: the PV closest to the
line through the Bs vertex (cons, else phic, else free; `pv_from_scheme` = 0, 1, 2)
along the KKπμ momentum, refitted
with the beamspot constraint and the four tracks removed (the same code as the
RJpsi and dimuon ntuples). `pv_refit_valid` = 0 means the hybrid PV (beamspot
x, y, PV z) was used.

Per scheme, Bs momentum hypotheses `coll`, `xyz_bs` (equal velocity along the
Bs flight direction PV → V_Bs), `math1/2_bs` (the two neutrino solutions along
it; `*_math_real_bs` = 0 when the discriminant was negative and clamped). PV →
V_Ds is not the Bs direction, so no hypothesis is built along it. For each: `pt eta phi beta q2 m_miss2 pt_miss pt_miss_sca
e_star_mu e_hash_mu cos_theta_phi_ds cos_theta_mu_w chi`, e.g.
`cons_math1_bs_q2`. Also `*_cos_theta_k_pi`, `*_mcorr`, `*_p_perp`, the Ds vertex wrt the
Bs vertex (`*_ds_lxy_bs`, `*_ds_cos3d_bs`), and the soft photon (`*_gamma_*`,
AN Sec. 5.5, with q² and m²_miss recomputed with Ds + γ).

### The Bs vertex in the τ channels, and the muon observables

V_Bs (`*_bs_*`) is the Ds + μ vertex, for every candidate. In Bs → Ds(*) τ ν,
τ → μ ν ν the muon comes from the τ decay point, so V_Bs is not the Bs decay
vertex there: it is pulled downstream along the flight (by ~30–150 µm on the
signal MC, with a ~2× worse longitudinal resolution) and its χ² is worse. The
reconstruction cannot depend on the channel it is measuring, so the definition
is kept and the τ flight is turned into observables; the MC models it in the
templates. Do not cut tightly on the Bs vertex probability.

Per scheme, the muon relative to

* the Bs flight axis, PV → V_Bs (as the RJpsi `mu_ip3d_*` / `mu_dist_*`):
  `*_mu_ip3d_bs_{pv,sv}{,_err,_sig}` (lifetime-signed IP3D wrt the PV and wrt
  V_Bs; the latter is small by construction, the muon being in that fit),
  `*_mu_dist_along_bs_dir_{pv,sv}` (IPTools `jetTrackDistance`, along the axis
  from the reference vertex) and `*_mu_dist_to_bs_dir` (line to line);
* the Ds trajectory, V_Ds along the Ds momentum of the Ds fit (taken before the
  Bs fit, so muon-free). It passes through the true Bs decay vertex: the muon
  meets it there in the μ channels and misses it by the τ impact parameter in
  the τ ones.
  `*_mu_doca_ds` — distance of closest approach muon–Ds trajectory;
  `*_mu_poca_along_ds` — where that is along the Ds momentum, from V_Ds
  (negative upstream; minus the Ds flight length in the μ channels);
  `*_mu_doca_ds_sig` = √χ²(Ds + μ vertex): with one degree of freedom that is
  the DOCA significance, with the full covariances (IPTools gives no error on
  the line-to-line distance).

## Branch reference

`tree` has one row per candidate; `tree_gen` (MC) one row per processed event,
with `run lumi event npu nti gen_pv_{x,y,z}` and the *Generator truth* blocks.
Everything is float32 except `run`, `lumi`, `event` (int64); a value that could
not be computed is NaN. Placeholders:

| placeholder | values |
|---|---|
| `<s>` | fit scheme: `cons`, `phic`, `free` |
| `<v>` | vertex of a scheme: `phi`, `ds`, `bs` |
| `<o>` | object: `mu`, `k1`, `k2`, `pi` (`k1` the leading kaon) |
| `<t>` | track: `k1`, `k2`, `pi` |
| `<h>` | Bs momentum hypothesis: `coll`, `xyz_bs`, `math1_bs`, `math2_bs` |
| `<q>` | hypothesis observable: `pt eta phi beta q2 m_miss2 pt_miss pt_miss_sca e_star_mu e_hash_mu cos_theta_phi_ds cos_theta_mu_w chi` |
| `<i>` | isolated object: `mu`, `ds` (KKπ), `b` (KKπμ) |
| `<rr>` | isolation cone: `03`, `04` |
| `<path>` | the ten parking paths `HLT_Mu7_IP4` … `HLT_Mu12_IP6` |
| `<seed>` | the ten L1 seeds `L1_SingleMu6er1p5` … `L1_SingleMu25` |
| `<par>` | track parameter: `qoverp`, `lambda`, `phi`, `dxy`, `dsz` |
| `<ij>` | the 15 pairs `<par>_<par>` of the upper triangle, e.g. `qoverp_lambda` |
| `<g>` | gen observable: `q2 m_miss2 pt_miss e_star_mu e_hash_mu cos_theta_k_pi cos_theta_phi_ds cos_theta_mu_w chi` |
| `<p>` | Hammer particle: `dsx ds dsg lep nu taumu taunut taunum` |
| `<NN>` | form-factor eigenvector: `00` … `09` |

### Event (repeated on every candidate of the event)

| branches | content |
|---|---|
| `run`, `lumi`, `event` | event identifiers (int64) |
| `ncands` | candidates of the event (rows with the same run, lumi, event) |
| `npv` | reconstructed primary vertices |
| `npu`, `nti` | MC: in-time pile-up interactions, true mean number of interactions; NaN in data |
| `bs_{x0,y0,z0}`, `bs_{x0,y0,z0}e` | beamspot position and its errors |
| `era`, `bph_part` | data: 2018 era (1–4 = A–D), ParkingBPH part (1–6); NaN in MC |
| `gen_pv_{x,y,z}` | MC: generated primary vertex |
| `<seed>`, `<seed>_ps` | L1 seed decision and prescale |
| `<path>`, `<path>_ps` | HLT path decision and prescale |
| `<path>_{l1min,l1max}_ps` | smallest and largest L1 prescale among the seeds of that path |

### Muon `mu_*` and tracks `k1_*`, `k2_*`, `pi_*`

| branches | content |
|---|---|
| `<o>_{pt,eta,phi,e,mass,charge}` | kinematics; pT and E with the momentum calibration (MC), the mass the hypothesis assigns |
| `<t>_{px,py,pz}` | calibrated momentum components |
| `<o>_pt_raw` | pT before the momentum calibration |
| `<o>_calib_{f,z}` | calibration factor p'/p and its Gaussian number (1, NaN without `--trk-calib`) |
| `mu_id_{loose,soft,medium,tight,soft_mva,pf,global,tracker,standalone}` | muon identification flags |
| `mu_id_soft_mva_raw` | soft-MVA output |
| `mu_pfiso{03,04}`, `mu_pfreliso{03,04}` | standard PAT PF isolation of the muon (default PV), absolute and relative |
| `mu_pfiso{03,04}_{ch,cp,nh,ph,pu}` | its components: charged hadrons, charged particles, neutral hadrons, photons, pile-up |
| `<o>_{dxy,dz}`, `<o>_{dxy,dz}_{e,sig}` | transverse and longitudinal IP wrt the candidate PV (`pv_*`), error, significance, from the track the fits used |
| `<o>_bs_dxy`, `<o>_bs_dxy_{e,sig}` | transverse IP wrt the beamspot |
| `<o>_{dxy,dz,bs_dxy}_e_raw` | the same errors from the raw track (before `--cov-scale`) |
| `<t>_pdgid`, `<t>_from_pv`, `<t>_pv_assoc_q`, `<t>_high_purity`, `<t>_lost_inner_hits` | packed-candidate information: PF id, PV association and its quality, high-purity flag, lost inner hits |
| `<t>_is_lost` | 1 if from `lostTracks`, 0 if from `packedPFCandidates` |
| `<o>_n_pix_{hit,b_hit,e_hit}`, `<o>_n_pix_layer`, `<o>_n_trk_layer` | valid pixel hits (all, barrel, endcap), pixel layers, tracker layers with a measurement |
| `<o>_pix_first_{b_layer,e_disk,layer}` | innermost barrel layer, endcap disk, and both in one code (`TrackHitContent.py`) |
| `<o>_n_pix_{miss,inact}_inner` | pixel layers missing before the first hit, on active / inactive modules |
| `<o>_pix_{valid,miss,inact}_mask`, `<o>_pix_hit_count` | per-layer bit masks and packed hit counts (`TrackHitContent.py`) |
| `<o>_best_trk_{pt,eta}`, `mu_best_trk_type` | the best track's pT, η; for the muon its type |
| `<o>_cov_<ij>` | 5×5 curvilinear covariance of the track, before corrections |
| `<o>_cov_pos_def`, `<o>_cov_fixed` | covariance positive definite as packed; repaired before the fits |
| `<o>_cov_scale_<par>` | `--cov-scale` factor applied to σ of each parameter (1 without it) |
| `<o>_jet_{pt,eta,phi,e}` | matched jet (ΔR < 0.2) |
| `mu_gen_{pt,eta,phi,e,pdgid}`, `<t>_gen_{pt,eta,phi,pdgid}` | MC: matched generator particle |
| `<o>_gen_dr`, `<o>_gen_mother_pdgid` | MC: ΔR of the match, pdgId of the matched particle's mother |
| `mu_trig_dr`, `mu_trig_match` | ΔR to the closest parking trigger object; 1 if < 0.05 |
| `mu_<path>_tag` | path fired and the muon matched to its object (1), fired and not matched (0), not fired (NaN) |

### Candidate, scheme-independent

| branches | content |
|---|---|
| `cand_idx` | rank of the candidate in the event by decreasing pT(KKπμ) |
| `mass`, `pt`, `eta`, `phi` | KKπμ, calibrated track momenta |
| `charge`, `kk_charge`, `mupi_charge` | total, KK and μπ charges |
| `charge_class` | 0 right sign, 1 μπ wrong sign, 2 KK wrong sign |
| `kk_mass`, `kkpi_mass` | m(KK), m(KKπ), calibrated track momenta |
| `refl_k{1,2}_as_{pi,p}`, `refl_k{1,2}_pi` | m(KKπ) with that kaon as a pion (D⁺ → Kππ) or a proton (Λc → pKπ); m(Kπ) of that kaon and the pion (D⁰ → Kπ) |
| `dr_{kk,phi_pi,ds_mu}`, `dr_mu_{k1,k2,pi}`, `max_dr_mu_tk` | ΔR between the pieces |
| `pv_idx`, `pv_from_scheme` | the chosen PV in the vertex collection; scheme whose Bs vertex chose it (0 cons, 1 phic, 2 free, −1 none: closest in z to the muon) |
| `pv_refit_{valid,ntrk}` | PV refitted (beamspot-constrained, signal tracks removed); its tracks. 0: hybrid PV (beamspot x, y, PV z) |
| `pv_{x,y,z}`, `pv_{ntrk,chi2,ndof}`, `pv_cov_{xx,xy,xz,yy,yz,zz}` | the PV used everywhere |
| `bs_{x,y}` | beamspot at the z of the PV |
| `<i>_iso_{ch,ch_clean,pu,nh,ph}_<rr>` | PF isolation wrt the refitted PV: charged hadrons from the PV (`clean`: signal tracks removed), charged from other vertices, neutral hadrons, photons |
| `<i>_iso{,_clean}_<rr>`, `<i>_reliso{,_clean}_<rr>` | ch + max(0, nh + ph − ½ pu), absolute and relative. `ds` and `b` include their own tracks; use `clean` |
| `gen_cand` | MC truth of the four tracks: 0–3 signal (Ds μ, Ds* μ, Ds τ, Ds* τ), 10 Bs feed-down, 11 Bs double charm, 12–15 Ds from B⁰, B⁺, Λb, other b, 20 Ds and μ from different b, 21 μ not from a b, 30 KKπ not from one Ds, −1 fewer than four tracks matched |
| `gen_ds_true`, `gen_charm_pdgid`, `gen_ds_{mother,b}_pdgid` | MC: KKπ from one Ds; their common charm ancestor; the Ds mother and b ancestor |
| `gen_mu_{mother,b}_pdgid`, `gen_mu_from_tau`, `gen_same_b`, `gen_n_matched` | MC: muon mother and b ancestor; muon from a τ; Ds and μ from the same b; tracks matched (0–4) |

### Per fit scheme `<s>_*`

| branches | content |
|---|---|
| `<s>_<v>_good`, `<s>_<v>_status` | fit usable (1/0); 0 good, 1 no fit, 2 χ² < 0, 3 covariance not positive definite |
| `<s>_<v>_{x,y,z}`, `<s>_<v>_cov_{xx,xy,xz,yy,yz,zz}` | vertex position and covariance |
| `<s>_<v>_{chi2,ndof,prob}` | vertex fit quality |
| `<s>_<v>_{mass,pt}` | the particle of that fit (φ, Ds, Ds + μ), from its own fit |
| `<s>_<v>_{lxy,lxyz}`, `<s>_<v>_{lxy,lxyz}_{err,sig}` | 2D / 3D distance from the PV |
| `<s>_<v>_cos{2d,3d}` | cosine between PV → vertex and the momentum of that fit |
| `<s>_ds_{lxy,lxyz}_bs`, `<s>_ds_lxy_bs_{err,sig}`, `<s>_ds_lxyz_bs_sig`, `<s>_ds_cos{2d,3d}_bs` | Ds vertex relative to the Bs vertex |
| `<s>_{k1,k2,pi,mu}_rf_{pt,eta,phi}` | track momenta refitted in the φ (k1, k2), Ds (π) and Bs (μ) fits |
| `<s>_<h>_<q>` | Bs momentum hypothesis and the observables with it: Bs pT, η, φ, β; q² = (B − Ds)², m²_miss = (B − Ds − μ)², pT_miss (vector, scalar), E\*_μ (Bs frame), E#_μ (W* frame), helicity angles cos θ_φ(Ds), cos θ_μ(W), χ |
| `<s>_math_real_bs` | 1 if the neutrino solutions are real, 0 if the discriminant was negative (then `math1` = `math2` are meaningless) |
| `<s>_cos_theta_k_pi` | helicity angle of the φ decay (no Bs dependence) |
| `<s>_p_perp`, `<s>_ds_p_perp`, `<s>_mcorr` | momentum of KKπμ / Ds transverse to PV → V_Bs; corrected mass √(m² + p⊥²) + p⊥ |
| `<s>_mu_ip3d_bs_{pv,sv}`, `<s>_mu_ip3d_bs_{pv,sv}_{err,sig}` | muon IP3D wrt the PV / V_Bs, signed along PV → V_Bs |
| `<s>_mu_dist_along_bs_dir_{pv,sv}`, `<s>_mu_dist_to_bs_dir` | muon closest approach to the Bs axis: position along it from the PV / V_Bs; line-to-line distance |
| `<s>_mu_doca_ds`, `<s>_mu_doca_ds_sig`, `<s>_mu_poca_along_ds` | muon vs the Ds trajectory (see above): distance of closest approach, its significance (√χ² of the Ds + μ vertex), position along the Ds from V_Ds |
| `<s>_gamma_{n,e_coll}` | photons with \|m(Dsγ) − m(Ds*)\| < 100 MeV; expected collinear photon energy |
| `<s>_gamma_{pt,eta,phi,dr,m_ds_gamma,dm,e_coll_over_pt}` | the one closest to the Ds in ΔR: kinematics, ΔR, m(Dsγ), m(Dsγ) − m(Ds), E_coll / pT |
| `<s>_gamma_{coll,xyz_bs}_{q2,m_miss2}` | q² and m²_miss with Ds + γ in place of the Ds |

### Generator truth (MC, also in `tree_gen`)

| branches | content |
|---|---|
| `gen_sig` | signal decay in the event: 0 Ds μ ν, 1 Ds* μ ν, 2 Ds τ ν, 3 Ds* τ ν, −1 none |
| `gen_n_bs` | Bs in the event |
| `gen_bs_{pt,eta,phi,y,mass,pdgid,beta,gamma}` | the signal Bs |
| `gen_bs_{lxyz,lxy,ct}`, `gen_bs_sv_{x,y,z}` | its flight and decay vertex |
| `gen_<g>` | the analysis quantities with the true Bs momentum (Ds, not Ds*; visible muon) |
| `gen_vis_{mu,kp,km,pi}_{pt,eta,phi}` | the visible final-state particles |
| `gen_ham_<p>_{pt,eta,phi,mass,pdgid}` | Hammer inputs, before FSR: Ds(*) from the Bs, the Ds and γ of Ds* → Ds γ, lepton, neutrino, and for τ the μ and two neutrinos |
| `hammer_weight`, `hammer_ff_ev<NN>_{up,dn}` | form-factor weight to the target scheme and its eigenvector variations |
| `hammer_status` | 0 ok, 1 not signal, 2 inputs missing, 3 declined by Hammer, 4 non-finite |

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

To do on the first real files: run with `--verbose`, compare `cons_`, `phic_`
and `free_` on signal MC, check the speed (ev/s in the log) before choosing the
CRAB splitting, and sync a few events against the previous ntuples.
