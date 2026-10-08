# J/ψ + track ntuples from the 2018 B-parking data

Ntuples of **B⁺ → J/ψ K⁺** and **B_c⁺ → J/ψ π⁺** candidates (one candidate per
dimuon + track pair, carrying both mass hypotheses: kaon unprefixed, pion
`pi_`-prefixed) from the 22 UL2018 MiniAODv2 datasets `ParkingBPH1–6`.

It is the Run 3 J/ψ + track ntuplizer (`test/rjpsi/inspector_jpsi_tk.py`) with
only the trigger changed:

```
                         Run 3 (JpsiTkInspector)          2018 parking (this directory)
                         -----------------------          -----------------------------
HLT paths                HLT_DoubleMu4_3_LowMass          HLT_Mu7_IP4 ... HLT_Mu12_IP6 (10 paths)
event kept if            that path fired                  ANY of the 10 fired (OR)
trig_match (no cut)      >= 2 objects matched             >= 1 J/psi muon matched, any fired path
per-muon trigger tags    -                                mu{1,2}_<path>_tag  (1/0, NaN if path not fired)
K/pi mass window        4.5 - 7.5 GeV                    4.5 - 7.0 GeV  (either hypothesis)
everything else          identical                        identical
```

`trig_match == 0` means the event was triggered by a muon that is **not** one of
the two J/ψ muons (tag-side trigger). Those candidates are kept; select
`trig_match == 1` offline for the signal-side sample.

## Files

| file | what it is |
|---|---|
| `inspector_jpsi_tk_bparking2018.py` | entry point; sets `max_mass = 7.0 GeV` for this ntuple. The rest of the logic is in `Bmmm/Analysis/python/JpsiTkBParking2018Inspector.py` |
| `crab/crab_jpsi_tk_bparking2018.py` | CRAB submitter for the 22 datasets |
| `crab/crab_jpsi_tk_bparking2018_mc.py` | CRAB submitter for the UL2018 MC (B⁺ → J/ψ K⁺) |
| `crab/crab_script.sh` | job wrapper (runs the inspector instead of cmsRun) |
| `crab/PSet.py`, `crab/FrameworkJobReport.xml` | dummy PSet / job report CRAB needs for a scriptExe job |
| `crab/make_pylibs.sh` | builds `crab/pylibs/` (the `particle` package, missing from CMSSW) |

## Before the first submission

1. The 2018 certification JSON must be in `Bmmm/Analysis/data/golden_jsons`. Its
   official name works as it is (`LumiMask` reads `Collisions18` as 2018):

   ```
   cp Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt \
      $CMSSW_BASE/src/Bmmm/Analysis/data/golden_jsons/
   ```

   Without it the submitter stops and says so.

2. Do a costing run: set `max_files = 5` in the submitter, keep one dataset in
   `productions`, submit, check runtime / memory / output size, then set
   `files_per_job` (default 1) and submit the rest.

## Local test (tcsh, at PSI)

```tcsh
cd /work/manzoni/rjpsi_run3/CMSSW_16_0_8/src
cmsenv
git fetch origin ; git checkout bparking2018_jpsi_tk
scram b -j 8
voms-proxy-init -rfc -voms cms -valid 24:00
cd $CMSSW_BASE/src/Bmmm/Analysis/test/bparking2018
set f = `dasgoclient -query "file dataset=/ParkingBPH1/Run2018D-UL2018_MiniAODv2-v1/MINIAOD" -limit 1`
python3 inspector_jpsi_tk_bparking2018.py --inputFiles=$f --filename=test_jpsitk_bph1_2018D \
    --maxevents=5000 --logfreq=500 --lumi-mask=none
```

`--lumi-mask=none` is for this functional test only (the job prints a warning).
Once the 2018 JSON is in `data/golden_jsons`, use `--lumi-mask=golden`; a random
file may then contain no certified lumi and give an empty tree, which is
correct. The cutflow is written to `logger_<timestamp>.txt`.

Same test on the B⁺ → J/ψ K⁺ MC (no mask; `--savenontrig` as in the grid jobs):

```tcsh
set f = `dasgoclient -query "file dataset=/BuToJpsiK_BMuonFilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/RunIISummer20UL18MiniAODv2-BPH_106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM" -limit 1`
python3 inspector_jpsi_tk_bparking2018.py --inputFiles=$f --filename=test_jpsitk_mc_bu_ul18 \
    --maxevents=2000 --logfreq=200 --mc --savenontrig
```

## MC

`crab/crab_jpsi_tk_bparking2018_mc.py` runs the same inspector with `--mc
--savenontrig`:

- every event is kept, whether or not a parking path fired. The path decisions,
  prescales and `mu{1,2}_<path>_tag` are filled, so the data selection is
  reproduced offline by requiring the OR of the paths.
- no certification mask.
- no pileup weights: the card has only Run 3 campaigns, so `pu_weight_*` are
  NaN. `nti` (true number of interactions) is stored for an offline reweighting.
- the gen matching is the provisional ΔR match of `JpsiTkInspector`, written for
  the B⁺ → J/ψ K⁺ topology, which is this sample.

## Submitting (tcsh)

```tcsh
cd $CMSSW_BASE/src/Bmmm/Analysis/test/bparking2018/crab
source /cvmfs/cms.cern.ch/crab3/crab.csh
voms-proxy-init -rfc -voms cms -valid 192:00
./make_pylibs.sh            # once
python3 crab_jpsi_tk_bparking2018.py       # data
python3 crab_jpsi_tk_bparking2018_mc.py    # MC
```

Output at `T3_CH_PSI`, one `jpsi_tk_ntuple_<job>.root` per job:

- data: `/store/user/manzoni/jpsi_tk_ntuples_bparking2018_08oct2026/Run2018<era>/ParkingBPH<N>/...`
- MC: `/store/user/manzoni/jpsi_tk_ntuples_mc_ul2018_08oct2026/BuToJpsiK_BMuonFilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/...`
