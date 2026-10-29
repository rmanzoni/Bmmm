# Pileup reweighting (Run 3 MC)

The weight maps the MC "true" pileup (`nti` = `getTrueNumInteractions()` at BX 0) onto the official data pileup distribution:

    w(nti) = P_data(k) / P_MC(k),    k = floor(nti)

- **P_data** is the pileupCalc `--calcMode true` histogram of one data period, normalised to 1.
- **P_MC** is the profile the campaign was *generated* with, i.e. the `probValue` of its `SimGeneral/MixingModule` cfi. It is **not** the `nti` distribution of our ntuples, because the skim and the selection depend on pileup.
- `PileUp.cc` draws `nti` uniformly in `[k, k+1)`, so bin k of the profile is exactly `floor(nti) = k`.

## Files

| file | role |
|---|---|
| `python/PileupWeights.py` | the one implementation: card reader, weight table, `PileupSession`, branch schema |
| `python/JpsiChargedInspector.py` | `--pu <campaign>` at production time (both channels) |
| `test/rjpsi/pileup/pu_config_run3.py` | data periods and MC campaigns (inputs of the card) |
| `test/rjpsi/pileup/build_pu_card.py` | builds `data/pu_weights_run3.json` (lxplus, after cmsenv) |
| `test/rjpsi/pileup/add_pu_weights.py` | same weights on existing ntuples, as a friend tree; `--closure` |
| `test/rjpsi/pileup/mc_pu_profile_from_miniaod.py` | option (b): measure `nti` on unskimmed MINIAODSIM and rank the profiles |

## Branches

There are 15 of them, always in the schema and NaN unless `--pu` is given. The pattern is `pu_weight_<year>`, `pu_weight_<year>_up` and `pu_weight_<year>_down`, for years 2022 to 2026. Nominal uses 69200 ub, up 72400 ub, down 66000 ub, as on the TWiki.

| campaign | fills | data period behind it |
|---|---|---|
| Summer22 | `pu_weight_2022*` | 2022 BCD |
| Summer22EE | `pu_weight_2022*` | 2022 EFG |
| Summer23 | `pu_weight_2023*` | 2023 BC |
| Summer23BPix | `pu_weight_2023*` | 2023 D |
| Summer24 | `pu_weight_2024*`, `_2025*`, `_2026*` | 2024, 2025, 2026 |

The era split matches the TWiki correctionlib list (2022 BCD↔Summer22, EFG↔Summer22EE, 2023 BC↔Summer23, D↔Summer23BPix).

In the plotter:
- Pick `pu_weight_<year>` for the year a sample stands for.
- Use `_up`/`_down` as the pileup systematic.
- **Never renormalise after the selection.** ⟨w⟩ ≠ 1 on selected events is the pileup dependence of the efficiency.

## Workflow

1. **Confirm the MC profiles (cheap).** For each campaign, measure about 20k events of the unskimmed MINIAODSIM:
   ```
   python3 mc_pu_profile_from_miniaod.py measure --files <list> --label Summer24 -o nti_Summer24.json
   python3 mc_pu_profile_from_miniaod.py compare nti_Summer24.json --campaign Summer24 --plot nti_Summer24.png
   ```
   The configured profile should come out on top with χ²/ndf ≈ 1. If it does, set `confirmed=True` in `pu_config_run3.py`. Otherwise, replace `mc_cfi` with the winner.
2. **Fill the `# CONFIRM` paths** in `pu_config_run3.py` (see open items below).
3. **Build the card** on lxplus, after `cmsenv`, and commit it:
   ```
   python3 build_pu_card.py -o $CMSSW_BASE/src/Bmmm/Analysis/data/pu_weights_run3.json --plots pu_card_plots
   ```
   For each campaign × year × variation it prints:
   - the fraction of data covered by the MC range,
   - the effective MC fraction after weighting,
   - the maximum weight.

   It stops if more than 0.1% of the data lies where the MC has no events, unless you pass `--allow-uncovered`.
4. **Produce the ntuples** with `--mc --pu Summer22EE` (etc.). The run summary prints the same numbers plus a count of unusable `nti` values.
5. **Or add the weights afterwards:**
   ```
   python3 add_pu_weights.py file.root --campaign Summer24 -o pu_friend.root
   ```
   Once a file has been produced with `--pu`, prove the two paths agree:
   ```
   python3 add_pu_weights.py file.root --campaign Summer24 -o /dev/null --closure
   ```

Spec forms, identical for `--pu` and `--campaign`:
- `Summer24`
- `Summer24:/path/card.json`
- `Summer24:allow-unconfirmed`, which accepts a campaign not yet confirmed in step 1. The inspector refuses such a campaign otherwise.

## Open items — to confirm

1. **Which profile each campaign was generated with.** This is set in `pu_config_run3.py` but not verified: it is fixed by the premix library request in McM, not by the dataset.
   - Current guesses:
     - Summer22 and 22EE → `Run3_2022_LHC_Simulation_10h_2h`
     - Summer23 and 23BPix → `Run3_2023_LHC_Simulation_12p5h_9h_hybrid2p23`
     - Summer24 → `mix_2024_25ns_RunIII2024Summer24_PoissonOOTPU`
   - The 2023 choice sits between three hybrid profiles with means of about 57, 62 and 67. Step 1 settles it.
2. **2025 and 2026 pileup JSON paths.** The TWiki (r53) lists none; the `Collisions25` / `Collisions26` folders on `/eos/user/c/cmsdqm/...` presumably have them.
3. **Golden JSONs for 2024, 2025 and 2026** (the pileupCalc input), to be filled in.
   - For 2024 the widest pileup JSON (BCDEFGHI) is used: pileupCalc only needs it to *contain* the golden lumisections.
4. **Central 2022/2023 directories.**
   - The file for each cross section is found by `<xsec>ub` in its name.
   - If a directory holds several candidates (e.g. JSON versions, or 99 vs 100 bins), the builder stops and lists them. Set `files={...}` for that period.
   - Any binning with unit-width bins on integer edges is accepted.
5. **Holes in the Summer24 profile.** It is zero for pileup 75–83 and 85–99, with an isolated bin at 84 of probability about 1e-8.
   - (a) Data at those values is not covered. For 2025/2026, with higher pileup, the builder will likely stop on this. Decide whether `--allow-uncovered` is acceptable, or how to extend.
   - (b) Any MC event that lands in bin 84 gets an enormous weight. With the fake data of the smoke test, one such event carried 7% of the sample.
   - No weight cap is applied on purpose: it would be an analysis choice, not a technical one.
6. **TWiki stress test.** The TWiki suggests checking the most pileup-sensitive variables after reweighting (number of vertices, ρ) and enlarging the 4.6% uncertainty if it does not cover data/MC. `npv` is in the ntuple, so this can be done on the J/ψ control sample.

## Verification done (smoke test, no CMSSW)

The test used the real MixingModule cfis, shims for the CMSSW config machinery, fake data histograms, and a stand-in `pileupCalc.py`.

- **Card builder:**
  - regrids 99-bin histograms onto the 100-bin grid;
  - runs pileupCalc with a heartbeat and counts its warnings;
  - reports coverage and writes plots;
  - the fail-loud paths trigger: too many warnings, uncovered data.
- **Physics:** after reweighting, 200k MC events reproduce the data profile. Pulls have mean −0.01 and RMS 0.85 over the 58 populated bins.
- **Inline vs post-hoc:** the inline path (scalar `weights()`, as in the inspector) and the post-hoc friend agree bit for bit on 200k × 15 values, NaNs included.
  - A single altered value makes `--closure` exit 1.
- **Fail-loud checks** that raise as they should: unconfirmed campaign, unknown campaign, all-NaN `nti` (a data file), missing card.
- **Branch lists and inspectors:**
  - both channels' branch lists hold the 15 branches exactly once;
  - both inspectors instantiate;
  - the PU block is treated as event-level, so the NaN candidate template cannot overwrite it.
- **Option (b):** on 5k events drawn from the 2022 profile, `compare` ranks it first (χ²/ndf 0.95 against ≥118 for the others) and flags a wrong configured profile.

Not tested here, because it needs CMSSW: the real FWLite event loop, the `pileupCalc.py` command line in the release, and the cfi import through `FWCore`.
