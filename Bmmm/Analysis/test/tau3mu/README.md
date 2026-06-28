# Displaced τ → 3μ ntuplizer (`Bmmm`, branch `tau3mu_disp`)

This package produces flat ROOT ntuples for the search for a **long-lived scalar
`a`** in the decay chain

```
D_s±  →  τ± ν
          └─ τ±  →  μ±  a
                      └─ a  →  μ+ μ−      (a is long-lived, PDG id 9900015)
```

The whole τ decay is visible (three muons), so the three-muon invariant mass is a
narrow peak at `m(τ) ≈ 1.777 GeV`, while the invariant mass of the **displaced
opposite-sign muon pair** `m(μμ) = m(a)` is the **search variable**, scanned over a
mass × lifetime grid. Because `a` is long-lived, its two muons are produced at a
secondary vertex displaced from the τ vertex, which is itself displaced from the
primary vertex.

The code reads (private) **MiniAOD** with FWLite and writes the ntuple with
`uproot`. It is built as an additive sub-package on top of the existing `RJPsi`
machinery: nothing in the `rjpsi` code is modified.

> **Audience.** This README assumes you are comfortable with CMSSW, FWLite,
> kinematic vertex fitting and the PSI Tier-3, but new to *this* analysis. It is
> meant to let you rerun the full ntuple production and understand every branch
> without further input from me.

Everything below refers to files actually in the branch (`tau3mu_disp`,
HEAD `7d42863`).

---

## 1. What is in the repository

The package modules live under `Bmmm/Analysis/python`, the C++ fitter under
`interface`/`src`, and the runnable scripts under `Bmmm/Analysis/test/tau3mu`.

**Analysis modules** (`Bmmm/Analysis/python`):

| File | Role |
|------|------|
| `Tau3MuHandles.py` | FWLite `Handle`s: which MiniAOD collections are read |
| `Tau3MuCandidate.py` | reco candidate — vertexing, displacement, IP, W-kinematics, PF cone |
| `Tau3MuBranches.py` | the ntuple schema: every branch and its getter |
| `Tau3MuCuts.py` | selection thresholds (`baseline`, `tau3mu`) |
| `Tau3MuGenHistory.py` | gen-truth navigation + reco↔gen muon matching (signal MC) |
| `Tau3MuGenBranches.py` | branch schema for the **gen-only** inspector |
| `Tau3MuGenCandidate.py` | gen-level candidate (built from the scalar `a`) |

**C++ vertex fitter:** `interface/Tau3MuKinVtxFitter.h`, `src/Tau3MuKinVtxFitter.cc`.

**Runnable scripts** (`Bmmm/Analysis/test/tau3mu`):

| File | Role |
|------|------|
| `inspector_tau3mu.py` | the **reco** ntuplizer (data + signal MC) |
| `inspector_tau3mu_gen.py` | the **gen-level** inspector (runs on GEN-SIM `genParticles`) |
| `submitter_tau3mu_mc.py` | SLURM submitter for the 21-point signal MC grid |
| `submitter_tau3mu_data.py` | SLURM submitter for 2024 `ParkingDoubleMuonLowMass` |
| `crab_submitter_tau3mu_data.py` | CRAB3 submitter for the same data (preferred at scale) |
| `PSet.py` | dummy CMSSW parameter-set required by CRAB's `scriptExe` mode |

The data input file lists (one LFN per line, per dataset) are committed under
`Bmmm/Analysis/test/files/*.txt`.

Shared infrastructure the above depend on: `python/utils.py` (`masses`,
`convert_cov`, `is_pos_def`, `compute_IP3D`, `drop_hlt_version`, `cutflow`),
`python/RJPsiNuReco.py` (`solve_nu_pz`), and the inherited
`interface/RJpsiKinVtxFitter.h`.

> **Not committed** (you must create/install them before the CRAB path works, see
> §5.5): `crab_script.sh`, `FrameworkJobReport.xml`, and the `pylibs/` tree.

---

## 2. Installation

The ntuplizer runs natively under recent (el9) CMSSW; no apptainer/el7 wrapper is
needed, because the release OS matches the Tier-3 worker nodes.

```bash
# 1) analysis release (Run 3, el9)
cmsrel CMSSW_15_1_1
cd CMSSW_15_1_1/src
cmsenv

# 2) get the package
git clone -b tau3mu_disp git@github.com:rmanzoni/Bmmm.git Bmmm

# 3) build libBmmmAnalysis (contains Tau3MuKinVtxFitter)
scram b -j 8
```

`scram b` compiles `src/Tau3MuKinVtxFitter.cc` and registers the class in the ROOT
dictionary (`src/classes.h`, `src/classes_def.xml`). The Python candidate loads it
at import time (`ROOT.gSystem.Load('libBmmmAnalysis')` then
`from ROOT import Tau3MuKinVtxFitter`). If that import fails, the library did not
build — re-run `scram b` and read the log.

### 2.1 Python dependencies

The reco inspector imports `numpy`, `scipy`, `uproot`, `awkward`, `pandas` and
(via `utils.py`) `particle` (Scikit-HEP). Most are in the CMSSW python stack; the
one reliably **missing** is `particle`. Two cases:

- **Interactively / login node:** `pip install --user particle` (found via
  `~/.local`).
- **Batch worker nodes:** `~/.local` is not visible, so ship the package with the
  job. Install it into a local tree and add it to the job `PYTHONPATH`:

  ```bash
  cd $CMSSW_BASE/src/Bmmm/Analysis/test/tau3mu
  PYTHONNOUSERSITE=1 pip install --no-cache-dir --target=pylibs particle
  ```

  The CRAB submitter ships this `pylibs/` tree and prepends it to `PYTHONPATH` (the
  SLURM jobs run from `$CMSSW_BASE/src`, so they pick up your `~/.local` instead).
  To reproduce the worker-node environment locally (hide `~/.local`):

  ```bash
  PYTHONNOUSERSITE=1 PYTHONPATH=$PWD/pylibs:$PYTHONPATH python3 inspector_tau3mu.py ...
  ```

---

## 3. Inputs

| | Data | Signal MC |
|---|---|---|
| Sample | 2024 `ParkingDoubleMuonLowMass` **MINIAOD** | private `D_s→τν, τ→μa, a→μμ` |
| Eras / grid | eras C, D, E, F, G, H, I (+ I `_v2`); parts 0–7 each | `m(a)` ∈ {0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5} GeV × `cτ` ∈ {10, 30, 100} mm |
| Location | global xrootd (`cms-xrd-global.cern.ch`) | phys03 DBS, mirrored on `/pnfs` at PSI |
| Trigger | `HLT_DoubleMu4_3_LowMass` | same |

The era/version map used by the submitters is C-v1, D-v1, E-v1, F-v3, G-v3, H-v3,
I-v3, I `_v2-v2`. The signal grid and MC chain are summarised in §8.

---

## 4. Quick sanity run (interactive)

Run a few events locally first, to confirm the build and an input file. The
inspector's own header gives the canonical form:

```bash
cd $CMSSW_BASE/src/Bmmm/Analysis/test/tau3mu

# signal MC
ipython -i -- inspector_tau3mu.py \
    --inputFiles=tau3mu_displaced_1GeV_ctau1mm-...-MiniAODv6-00065_9.root \
    --filename=tau3mu_signal --mc --maxevents=2000

# data (xrootd URL; --mc dropped)
ipython -i -- inspector_tau3mu.py \
    --inputFiles="root://cms-xrd-global.cern.ch///store/data/.../file.root" \
    --filename=data_2024 --maxevents=2000
```

This writes `./<filename>.root` (TTree `tree`) plus a `logger_*.txt` cutflow. Open
it and check `mass` peaks at ~1.78 GeV, `a_mass` is broad, and (MC) `gen_*` are
populated. The `-i` keeps the IPython session alive afterwards for poking at the
objects — handy for debugging.

To study the signal straight off **GEN-SIM** (no reco), use the gen inspector,
which reads the `genParticles` collection:

```bash
ipython -i -- inspector_tau3mu_gen.py \
    --inputFiles="file:/path/to/tau3mu_displaced_1GeV_ctau1mm.root" \
    --filename=tau3mu_gen --maxevents=-1
```

---

## 5. Running the ntuple production

### 5.1 The reco inspector command-line interface

`inspector_tau3mu.py` (argparse → a `namedtuple` of options):

| Flag | Meaning |
|------|---------|
| `--inputFiles` | input(s); **required**. Accepts a `.txt` list (one LFN/line, `--redirector` prepended), a comma-separated list, or a glob (globbed files are opened and the empty/zombie/`Events`-less ones dropped) |
| `--filename` | output basename → `<filename>.root`; **required** |
| `--destination` | output directory (default `./`) |
| `--maxevents` | events to process (`-1` = all) |
| `--mc` | turn on the gen + pileup handles and the gen branches |
| `--savenontrig` | also keep candidates in events that did **not** fire the HLT path |
| `--redirector` | prefix prepended to bare LFNs from a `.txt` list (default `root://cms-xrd-global.cern.ch//`) |
| `--maxfiles` | cap the number of input files |
| `--logfreq` | progress-print frequency (events) |
| `--logger` | cutflow logger filename (default timestamped) |
| `--verbose` | print the offending getter when `safe_get` swallows an exception |

**Output granularity: one row per *candidate*, not per event.** The TTree `tree`
is created up front with a fixed schema (`uproot.recreate` + `mktree`, ZSTD-5):
event-id branches are `int64`, all other scalars `float32`, and the PF cone is the
single jagged record branch `pf` (one shared `npf` counter). Rows are buffered and
flushed every `WRITE_EVERY = 20000`. An empty tree is written if nothing is
selected (harmless for a downstream `hadd`). A per-run `logger_*.txt` holds the
cutflow.

### 5.2 What the inspector does, per event and per candidate

Per **event**: load the handles; evaluate `HLT_DoubleMu4_3_LowMass` (accept +
prescale) and skip the event unless it fired (or `--savenontrig`); collect the
fired HLT filter objects (`hltDisplacedmumuFilterDoubleMu43LowMass`, with
`to_pt > 2`, `|to_eta| < 2.6`); preselect muons
(`pt > 2`, `|η| < 2.5`, `isPFMuon and (global or tracker)`, `|dxy| < 20 cm` —
**loose on purpose** for displaced muons); require ≥ 3 muons.

Candidate building: for every **opposite-sign** muon pair with **both** `pt > 3`
(the trigger pair) and `m(μμ) < 4 GeV` (loose; `m(a)` is the search variable),
take each remaining muon as the bachelor `mu3` and keep the candidate if the
three-muon mass is in `1.2 < m(3μ) < 2.4 GeV`. Candidates are then sorted by
`(a.pt(), −|m(3μ) − 1.77686|)` descending — hardest trigger pair and closest to
the τ mass first.

Per **candidate**, in this exact order:

1. (MC) `match_candidate_muons(cand, genpr, dr_max=0.03, info=gen_info)` — attach
   the gen muons for the per-muon `gen_*` branches;
2. `cand.trig_match` — set to 1 if **both** `a`-muons match a fired HLT object
   within ΔR < 0.1;
3. `cand.compute_vtx_quantities(vtx, bs, pf, ltrk)` — the sequential vertex fit, PV
   choice, beamspot-constrained PV refit (signal muons removed), displacement, IPs;
4. `cand.compute_pf_cone(pf, cone_dr=0.6, min_pt=0)` — the PF-candidate cone (the
   radius is `cuts['tau3mu']['pf_cone_dr'] = 0.6`; see §6.11 note);
5. per-muon branches — each muon gets `imu.pv = pv_bs`, `imu.bs`, and its PF
   isolation (`pfIsolationR03/04`), then the `muon_branches` getters run;
6. `cand.compute_w_kinematics(met[0])` — transverse mass + the longitudinal-ν
   solutions (the flight-direction roots need the τ vertex from step 3);
7. the `cand_branches` getters run, the three scalar scopes are merged into one
   flat row, and the candidate's PF cone is appended in lockstep.

Every getter is wrapped in `safe_get`, so a failing getter writes `NaN` (or its
default) rather than killing the job.

### 5.3 SLURM — signal MC (`submitter_tau3mu_mc.py`)

Run from a shell where you have `cmsenv`-d the release you want the jobs to use
(`$CMSSW_BASE`/`$SCRAM_ARCH` are read and baked into each job script). The
submitter:

- iterates the `masses × ctaus` grid and **globs** the local `/pnfs` mount,
  `…/manzoni/a_mass_<m>gev_ctau_<ct>mm/<mc_prod_tag>/*/*/*.root` (the CRAB timestamp
  and counter dirs are wildcarded; mass naming uses the `p` convention, e.g.
  `0.3 → 0p3`). `mc_prod_tag = tau3mu_displaced_run3_MiniAOD_25jun26`;
- chunks the files (`files_per_job = 100`), and for each chunk writes a
  `submitter_chunk<N>.sh` that sets up the CMSSW runtime natively, stages a private
  grid-proxy copy into scratch, runs `inspector_tau3mu.py` over the
  **comma-separated** chunk with `--mc` into `/scratch/manzoni/<out_dir>`, then
  `xrdcp`s the single output to `/pnfs/.../manzoni/<out_dir>/` via
  `root://t3dcachedb03.psi.ch:1094//` and cleans scratch on success;
- `sbatch`es each to the `standard` queue, pinned to `t3wn[80-91]`.

There is **no `hadd`** step here — `Events()` reads the whole chunk at once, so one
chunk → one output file. Knobs at the top: `masses`, `ctaus`, `files_per_job`,
`queue`/`time`, `out_dir`, `out_file_name`, `mc_prod_tag`, `redirector`, plus
`resubmit`/`toresubmit` for targeted resubmission.

### 5.4 SLURM — data (`submitter_tau3mu_data.py`)

Same job-script machinery, but it reads pre-generated DAS file lists. For each of
the 8 era/version entries × 8 parts it expects
`Bmmm/Analysis/test/files/<dataset-with-slashes-as-dashes>.txt` (generate them with
`dasgoclient`), prefixes each LFN with the **global xrootd** redirector
(`root://cms-xrd-global.cern.ch//`; the dCache redirector is commented out),
chunks at `files_per_job = 35`, and submits one job per chunk (again one output
file per chunk, no `hadd`). Missing/empty list files are skipped with a warning.

### 5.5 CRAB3 — data (`crab_submitter_tau3mu_data.py`, preferred at scale)

The SLURM data jobs read remote files over the WAN, which is slow and flaky once a
few thousand jobs hammer the global redirector. CRAB instead schedules each job at
a site hosting the data (LAN reads), auto-resubmits failures, and splits the
datasets for you. Because the inspector is **not** a `cmsRun` job, the task runs in
`scriptExe` mode:

- `JobType.psetName = 'PSet.py'` — a dummy parameter-set; CRAB injects each job's
  input files into `process.source.fileNames`, and `crab_script.sh` reads them back
  via `import PSet`;
- `JobType.scriptExe = 'crab_script.sh'` — the wrapper that prepends `./pylibs` to
  `PYTHONPATH` and launches `inspector_tau3mu.py`;
- `JobType.inputFiles` ships **all** `*.py`/`*.h` from `test/tau3mu/` **plus** the
  `pylibs/` tree **plus** `FrameworkJobReport.xml`. Shipping only the inspector
  makes every job die at import time — the sibling helper modules must travel in
  the sandbox (that directory is not an importable package on the WN);
- `JobType.disableAutomaticOutputCollection = True` and
  `JobType.outputFiles = ['tau3mu.root']` — there is no EDM output to harvest;
- `Data.splitting = 'FileBased'`, `unitsPerJob = 10`, `inputDBS = 'global'`,
  `outLFNDirBase = /store/user/manzoni/<out_dir>/Run2024<era>`,
  `Site.storageSite = 'T3_CH_PSI'`, `publication = False`;
- each dataset is submitted **in its own `multiprocessing.Process`** to dodge the
  FWCore "pset already cached" error you get from submitting multiple configs in one
  interpreter. The `already_submitted` list skips request names you've already done.

Setup and submit (order matters):

```bash
cmsenv                                        # in the src you built Bmmm in
source /cvmfs/cms.cern.ch/crab3/crab.sh
voms-proxy-init -rfc -voms cms -valid 192:00
cd $CMSSW_BASE/src/Bmmm/Analysis/test/tau3mu
python3 crab_submitter_tau3mu_data.py
```

> **Before this works** you must provide three files that are *not* in the repo,
> in `test/tau3mu/`: `crab_script.sh` (the `scriptExe` wrapper), a static
> `FrameworkJobReport.xml`, and the `pylibs/` tree (§2.1). The submitter raises a
> `RuntimeError` at config-build time if `pylibs/` (or the inspector) is missing.
>
> **Quirk to know:** the submit loop currently starts with `if ii==0: continue`,
> i.e. it **skips the first dataset** (`ParkingDoubleMuonLowMass0`, `Run2024C`,
> `v1`). Remove that line, or submit that one dataset separately, if you want full
> coverage.
>
> If some input blocks sit only at `T2_CH_CSCS` and trip the global blacklist (as
> happened for the private MC), uncomment `cfg.Site.ignoreGlobalBlacklist = True`.
> To restrict to certified lumis, set `cfg.Data.lumiMask` to the 2024 golden JSON.

Monitor with `crab status/getlog/resubmit/report -d <work_area>/crab_<requestName>`.

### 5.6 Output layout

Each era/version (or grid point) lands in its own top-level directory under
`/store/user/manzoni/<out_dir>/`. CRAB additionally forces a dataset-name
sub-directory level that cannot be collapsed. `hadd` per era/point as needed.

---

## 6. Branch reference (the physics of every branch)

The schema is assembled in `Tau3MuBranches.py`. Units: momenta/energies/masses in
**GeV**, lengths in **cm**, angles in **rad**; charges integer. Quantities default
to `NaN` when not computable (e.g. a failed vertex fit).

Muon naming (`rjpsi` convention):

- **`mu1`, `mu2`** — the displaced opposite-sign pair, i.e. the `a → μμ` muons (the
  pair that fires `HLT_DoubleMu4_3_LowMass`), pt-sorted;
- **`mu3`** — the bachelor muon from `τ → μ a`.

The candidate object is the full 3μ = τ system; `a` is the sub-composite
`mu1+mu2`.

### 6.1 Event-level (constant across the candidates of an event)

| Branch | Meaning |
|--------|---------|
| `run`, `lumi`, `event` | event id (`int64`) |
| `ncands` | number of selected τ candidates in the event |
| `npv` | number of reconstructed primary vertices |
| `npu` | in-time pileup `getPU_NumInteractions()` (MC only, else `NaN`) |
| `nti` | true number of interactions `getTrueNumInteractions()` (MC only) |
| `bs_x0/y0/z0` (+`…e`) | beamspot position and error |
| `met_pt`, `met_phi`, `met_px`, `met_py`, `met_sumet` | **PUPPI** MET (`slimmedMETsPuppi`, a one-entry vector → `[0]`) |

### 6.2 Candidate — τ (full 3μ) and `a` kinematics

| Branch | Meaning |
|--------|---------|
| `mass`, `pt`, `eta`, `phi`, `charge` | the raw 3μ = τ four-momentum sum; `mass` peaks at `m(τ)` |
| `rf_mass/pt/eta/phi` | the **refitted** τ from the sequential vertex fit (stage 2) |
| `a_mass`, `a_pt`, `a_eta`, `a_phi`, `a_charge` | the displaced OS pair `a = μ1+μ2`; **`a_mass` is the search variable** |
| `a_rf_mass/pt/eta/phi` | the **refitted** `a` from the OS-pair vertex fit (stage 1) |
| `dr` | cone radius: max ΔR between the 3μ direction and any of its muons |
| `dr_max` | max pairwise ΔR among the three muons |
| `dr_a` | ΔR between the two `a` muons |
| `dr_a_mu` | ΔR between the `a` direction and the bachelor muon |

### 6.3 Candidate — primary vertex / beamspot

All displacement/IP quantities use the **per-candidate refit PV** `pv_bs` (chosen
PV, refit with the three signal muons removed, beamspot-constrained; hybrid
beamspot-xy/PV-z fallback). See §7.2.

| Branch | Meaning |
|--------|---------|
| `pv_refit_valid` | 1 if the beamspot-constrained PV refit succeeded, 0 if the hybrid fallback was used |
| `pv_x/y/z` | position of the reference PV (`pv_bs`) |
| `pv_ntrk`, `pv_chi2`, `pv_ndof` | its track multiplicity / fit quality |
| `bs_x`, `bs_y` | beamspot position at the PV z |

### 6.4 Candidate — τ vertex (sequential `a` + bachelor fit), wrt PV

| Branch | Meaning |
|--------|---------|
| `sv_good` | 1 if the τ-vertex fit succeeded |
| `sv_x/y/z` | τ decay-vertex position |
| `sv_chi2`, `sv_ndof`, `sv_prob` | fit χ², ndof, χ² survival probability |
| `sv_cos2d`, `sv_cos3d` | pointing angle between the τ momentum and the PV→SV flight (2D/3D) |
| `sv_lxy(_err/_sig)` | transverse flight length PV→SV, error, significance |
| `sv_lxyz(_err/_sig)` | 3D flight length PV→SV, error, significance |

### 6.5 Candidate — `a` vertex (displaced OS pair), wrt PV

| Branch | Meaning |
|--------|---------|
| `a_good` | 1 if the `a`-vertex fit succeeded |
| `a_x/y/z` | `a` decay-vertex position |
| `a_vtx_chi2/ndof/prob` | fit quality |
| `a_cos2d`, `a_cos3d` | pointing angle of the `a` momentum wrt PV→(a vtx) |
| `a_lxy(_err/_sig)`, `a_lxyz(_err/_sig)` | `a` flight length from the PV, transverse and 3D |

### 6.6 Candidate — the `a` flight **from the τ vertex** (the smoking gun)

Displacement of the `a` vertex measured from the **τ vertex**, not the PV — the
genuine `a` flight, the direct handle on the scalar lifetime.

| Branch | Meaning |
|--------|---------|
| `a_wrt_tau_cos2d/cos3d` | pointing angle of the `a` momentum wrt (τ vtx)→(a vtx) |
| `a_wrt_tau_lxy(_err/_sig)` | transverse `a` flight from the τ vertex |
| `a_wrt_tau_lxyz(_err/_sig)` | 3D `a` flight from the τ vertex |

### 6.7 Candidate — bachelor IP, W-kinematics, trigger

| Branch | Meaning |
|--------|---------|
| `mu3_ip3d_a`, `…_err`, `…_sig` | signed 3D IP of the bachelor muon wrt the **`a` vertex** (mu3 is *not* in the `a` fit; non-zero tags the displaced topology) |
| `mt` | transverse mass between the 3μ system and PUPPI MET (uses the raw 3μ p4, so defined even if the τ fit failed) |
| `nu_pz_1`, `nu_pz_2` | the two longitudinal-ν solutions, `\|pz\|`-sorted (smaller first), **flight-direction** method + `m_W` constraint (§7.3); need the τ vertex, else `NaN` |
| `nu_has_real` | 1 if the discriminant ≥ 0 (two distinct real roots) |
| `nu_disc` | the quadratic discriminant (diagnostic) |
| `nu_pz_met_1`, `nu_pz_met_2` | same `m_W` constraint but ν transverse momentum from **PUPPI MET** (standard `W→ℓν`, vertex-free; filled whenever MET exists) |
| `nu_met_has_real` | 1 if the MET-based discriminant ≥ 0 |
| `nu_met_disc` | the MET-based discriminant |
| `trig_match` | 1 if **both** `a`-muons matched a fired HLT filter object within ΔR < 0.1 |

### 6.8 Per-muon — `mu1_*`, `mu2_*`, `mu3_*`

The same block for each muon (`mu1`,`mu2` = the `a` pair; `mu3` = the bachelor).

| Branch (per `muN_`) | Meaning |
|--------|---------|
| `pt`, `eta`, `phi`, `e`, `mass`, `charge` | the muon four-momentum |
| `rf_pt/eta/phi/e` | the **refitted** muon momentum from the relevant vertex fit |
| `id_loose/soft/medium/tight/pf/global/tracker` | standard muon IDs (soft/tight evaluated against the candidate PV) |
| `pfiso03`, `pfiso04` | PF isolation sums, Δβ-PU-corrected, R = 0.3 / 0.4 |
| `pfreliso03`, `pfreliso04` | the same divided by muon pt |
| `dxy`, `dxy_e`, `dxy_sig` | transverse IP wrt the refit PV (+ error, significance) |
| `dz`, `dz_e`, `dz_sig` | longitudinal IP wrt the refit PV |
| `bs_dxy`, `bs_dxy_e`, `bs_dxy_sig` | transverse IP wrt the beamspot |
| `ip3d`, `ip3d_err`, `ip3d_sig` | signed 3D IP wrt the refit PV, **lifetime-signed along the τ flight** |
| `cov_pos_def` | 1 if the muon track covariance is positive-definite |
| `gen_pt/eta/phi/e/pdgid/charge` | matched **gen** muon (signal MC; `NaN` if unmatched/data) |
| `gen_role` | role of the gen match: **1 = `mu_a`** (from `a→μμ`), **2 = `mu_tau`** (bachelor) |
| `gen_dr` | ΔR of the gen match |

### 6.9 Gen branches (signal MC only)

Built from `prunedGenParticles` by `Tau3MuGenDecay.from_genparticles`. Vertices in
cm; `_ct` are proper decay lengths (cross-checks).

| Branch | Meaning |
|--------|---------|
| `gen_ds_pt/eta/phi/mass/pdgid` | the `D_s` (if found; `NaN` otherwise) |
| `gen_tau_pt/eta/phi/mass/pdgid` | the τ |
| `gen_a_pt/eta/phi/mass` | the long-lived scalar `a` |
| `gen_tau_lxy`, `gen_tau_lxyz` | τ flight (production → τ decay vertex), transverse/3D |
| `gen_tau_ct` | τ proper decay length `L·m/(βγ\|p\|)` |
| `gen_a_lxy`, `gen_a_lxyz` | `a` flight (τ decay vertex → `a` decay vertex) |
| `gen_a_ct` | `a` proper decay length — should peak at the generated `cτ` (0.1 cm for 1 mm) |
| `gen_pv_x/y/z` | gen "PV" = τ production vertex |
| `gen_tau_sv_x/y/z` | τ decay vertex |
| `gen_a_sv_x/y/z` | `a` decay vertex |

### 6.10 Trigger branches

| Branch | Meaning |
|--------|---------|
| `HLT_DoubleMu4_3_LowMass` | 1/0 (event max) if the path fired |
| `HLT_DoubleMu4_3_LowMass_ps` | the path prescale |

The matching filter object is `hltDisplacedmumuFilterDoubleMu43LowMass`.

### 6.11 Jagged PF-candidate cone (`pf`, one sublist per candidate)

All PF candidates within ΔR < `pf_cone_dr` of the 3μ (τ) axis, written as a single
NanoAOD-style jagged record branch `pf` (`var * {pt, eta, …}`): one shared counter
`npf` + the `pf_<field>` leaves. IPs are wrt the refit PV `pv_bs`.

| Field (`pf_…`) | Meaning |
|--------|---------|
| `pt`, `eta`, `phi`, `mass`, `energy` | PF-candidate four-momentum |
| `puppiweight` | PUPPI weight |
| `pdgid`, `charge` | identity (note `pdgId==130` = PF neutral hadron — it borrows the K⁰_L number, mass set to 0 by `PFAlgo`; not a real K⁰_L) |
| `dr` | ΔR to the τ axis |
| `dxy`, `dxy_err`, `dz`, `dz_err` | packed-candidate IPs wrt the PV |
| `ip3d`, `ip3d_sig` | signed 3D IP wrt the PV (full helical extrapolation; `NaN` without track details) |
| `is_signal` | 1 if the PF candidate is one of the three signal muons (cross-collection proximity match), so it can be removed downstream |

> **Cone radius caveat.** The inspector passes `cuts['tau3mu']['pf_cone_dr']`, which
> is **0.6**, so the stored cone is R = 0.6. The `compute_pf_cone` default (0.4) and
> a couple of stale "R=0.4" comments in the docstrings are *not* what runs — the cut
> value wins.

### 6.12 Gen-only inspector branches (`Tau3MuGenBranches.py`)

`inspector_tau3mu_gen.py` runs on GEN-SIM `genParticles`, builds a
`Tau3MuGenCandidate` per last-copy scalar, and writes one row per gen candidate:
`run/lumi/event/ncands`; per-muon `mu_tau_*`, `mu_disp1_*`, `mu_disp2_*`
(`pt,eta,phi,mass,e,px,py,pz,charge,pdgid,vx,vy,vz`); the whole-3μ system
(`tau3mu_*`), the displaced pair (`pair_*`), the pair displacement
(`decay_length`, `lxy`, `lz`, `ctau`, `pv_*`, `sv_*`), the gen scalar (`scalar_*`),
and `dr_scalar_mu`. Its CLI is the small one shown in §4 (no `--mc`, no trigger
flags). Use it to validate lifetime reweighting and acceptance before touching
reco.

---

## 7. How it works (so you know what you're running)

### 7.1 Sequential (hierarchical) vertex fit

`Tau3MuCandidate.compute_vtx_quantities` + the C++ `Tau3MuKinVtxFitter`. **Not** a
flat three-track fit:

1. **`a` vertex.** Fit the displaced OS pair `(μ1,μ2)` to a common vertex (muon
   mass each, **no** mass constraint, since `m(μμ)=m(a)` is the search variable).
   This yields the refitted `a` particle with its full covariance.
2. **τ vertex.** Feed that mother particle and the bachelor muon to
   `FitMotherPlusTrack(aTree, mu3.bestTrack(), m_μ)`, fitting them to a *different*,
   upstream common vertex. The `a` enters as a single `KinematicParticle`, so the τ
   and `a` vertices are properly correlated.

The returned τ tree is structurally identical to a plain `Fit` tree, so all the
downstream handling is shared with `rjpsi`.

### 7.2 Primary-vertex choice and refit

- **Choice:** the PV with the smallest 3D IP wrt the τ flight line (through the τ SV
  along the 3μ momentum); if the τ fit failed, the PV closest in `dz` to the
  leading muon.
- **Refit:** a beamspot-constrained Adaptive Vertex Fit of that PV with the three
  signal muons removed (`refitPVRemovingTracks`). The PV track set is rebuilt in the
  loop from `packedPFCandidates` + `lostTracks` — primarily via the offline
  fit-track assignment (`fromPV == PVUsedInFit`), with a closest-z + offline-quality
  track-filter fallback. Success → `pv_refit` (`pv_refit_valid=1`); failure → a
  hybrid PV (beamspot x,y at the PV z). The reference used everywhere is `pv_bs`.

The displaced muons make a tight `dxy` cut counter-productive, so the PV definition
leans on the beamspot.

### 7.3 Longitudinal-neutrino reconstruction

For `W → τ ν, τ → 3μ` the ν `pz` is reconstructed two ways, both imposing
`m(3μ + ν) = m_W` (`W_MASS = 80.3692 GeV`), each a quadratic with two
`|pz|`-sorted roots:

- **Flight-direction (`nu_pz_1/2`):** the W is assumed to fly along the PV→SV (τ)
  flight direction, so it has no momentum transverse to that axis and the ν
  transverse momentum is fixed by balance — deliberately **MET-independent**.
  Solved by `RJPsiNuReco.solve_nu_pz` with `m_parent = m_W`; needs the τ vertex.
- **MET-based (`nu_pz_met_1/2`):** ν transverse momentum from PUPPI MET; standard
  `W→ℓν` `pz` reconstruction, vertex-free.

A negative discriminant (resolution) is clamped: both roots collapse to the real
part and `…_has_real = 0`.

### 7.4 Selection (`Tau3MuCuts.py`)

The `tau3mu` working point extends `baseline`. Highlights: trigger-object `pt > 2`,
`|η| < 2.6`; per-muon `pt > 2`, `|η| < 2.5`, `isPFMuon and (global or tracker)`,
**`dxy < 20 cm` — loose on purpose**; the trigger pair `pt > 3`;
`HLT_DoubleMu4_3_LowMass` with ΔR match 0.1; the OS pair `m(μμ) < 4 GeV` (loose
upper bound — *not* an `a` window, since `m(a)` is the search variable); the τ
candidate `1.2 < m(3μ) < 2.4 GeV`; PF cone R = 0.6; gen-matching ΔR = 0.03.

---

## 8. Signal MC production (upstream of the ntuples)

Recap so you can regenerate or extend the grid:

- **Fragment.** `Pythia8GeneratorFilter` + `EvtGen130`, `comEnergy = 13600 GeV`.
  EvtGen forces `myDs± → myTau± ν`, `myTau± → μ± myX`, `myX → μ+ μ−`, with `myX`
  aliased to `hnl` (pdgId 9900015). Per grid point the mass and `cτ` live in a
  dedicated `.pdl`,
  `cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt_tau3mu_displaced_mass_<m>gev_ctau_<ct>mm.pdl')`.
  Generator filters: `MCMultiParticleFilter` on the `D_s` (431);
  `MCParticlePairFilter` for an OS muon pair (`|η|<2.55`, `pt>3`, `M(μμ)<2 GeV`);
  `PythiaFilterMultiMother` for a muon from the τ.
- **EvtGen gotchas.** `ChargeConj` on **aliases** (`myDs+`/`myDs-`), not real
  particles; `CDecay` a standalone statement *after* `Enddecay`, never inside a
  decay block; the `Status`/`ParticleID` filter vectors must have matching sizes.
  The `.pdl` must sit at `GeneratorInterface/EvtGenInterface/data/` and be built
  into the release/sandbox *before* CRAB submission so `FileInPath` resolves; a
  partial local checkout of `GeneratorInterface/EvtGenInterface` that shadows the
  release copy breaks the import.
- **Chain.** GEN-SIM (`CMSSW_14_0_19`) → DR step 1 (DIGI/L1/DIGI2RAW/HLT) → DR step
  2 (RECO/AODSIM) → MiniAODv6, each a standalone CRAB3 task. Tag chain
  `DR_step1_25jun26 → AODSIM_25jun26 → MiniAOD_25jun26`; GEN-SIM published to phys03
  DBS under `tau3mu_displaced_run3_25jun26_v2`. The MiniAOD step must point at the
  **AODSIM** output, not the RAW dataset.

---

## 9. Gotchas / FAQ

- **`from ROOT import Tau3MuKinVtxFitter` fails** → `libBmmmAnalysis` did not build;
  re-run `scram b`.
- **`ModuleNotFoundError: particle` on the grid** → ship `pylibs/` and prepend it to
  `PYTHONPATH` (§2.1); CRAB jobs that ship only the inspector die at import (exit 5,
  0% CPU) — ship **all** the `test/tau3mu/*.py`/`*.h` siblings too.
- **CRAB submit raises `RuntimeError`** → `pylibs/`, `crab_script.sh` or
  `FrameworkJobReport.xml` missing from `test/tau3mu/` (none are committed; §5.5).
- **First data dataset never submits** → the CRAB loop has `if ii==0: continue`,
  which skips `ParkingDoubleMuonLowMass0 / Run2024C / v1`; remove it for full
  coverage.
- **CRAB output to CSCS rejected** → uncomment `cfg.Site.ignoreGlobalBlacklist =
  True`.
- **Empty `nu_pz_*`** → the flight-direction roots are `NaN` whenever the τ vertex
  fit failed (`sv_good==0`); the MET-based `nu_pz_met_*` and `mt` only need MET, so
  if *those* are empty check the `slimmedMETsPuppi` handle is loaded.
- **The PF cone looks bigger than 0.4** → it is 0.6 by the cut; the 0.4 in the code
  comments/`compute_pf_cone` default is overridden (§6.11).
- **`pdgId==130` in the PF cone** → PF *neutral hadron* category label, mass 0 by
  construction; don't read a K⁰_L into it.
