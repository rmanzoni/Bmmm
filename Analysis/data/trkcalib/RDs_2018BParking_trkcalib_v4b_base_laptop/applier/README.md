# rds_trkcalib — track momentum calibration for R(Ds) MC

Corrects the momentum of every MC charged track (kaons, pion, muon) so that the MC mass peaks match 2018 BParking data. v4 is fitted to three peaks at once: Ds → φ(KK)π, J/ψ → μμ and B⁺ → J/ψK⁺; v1–v3 to the Ds peak only.

```
k' = k (1 + A - e*k + q*M/k) (1 + sqrt(a^2 + c^2/k^2) z),   k = 1/pT
pT' = pT / [ (1 + A - e/pT + q*M*pT) (1 + sqrt(max(a2 + c2*pT^2, 0)) z) ]
```

This is the convention of the CMS W-mass measurement ([arXiv:2412.13872](https://arxiv.org/abs/2412.13872), Eqs. 2–3). The terms are A (field, scale), e = ε (energy loss, GeV), M (sagitta, GeV⁻¹, sign follows the track charge q), a (multiple scattering) and c (hit resolution, GeV⁻¹). Here (A, ε, M) describe data relative to MC, and (a, c) the resolution MC is missing. The file stores a2 = a² and c2 = c² [GeV⁻²], so a resolution variation can cross zero. Files of schema `rds_trkcalib/2` use this convention.

The v1 file (schema `rds_trkcalib/1`) uses the first convention, p' = p (1 + s0 + s1/p + k q pT)(1 + sqrt(v0·1e-6 + v1·1e-8 pT²) z). Both appliers read both schemas and pick the formula from the file.

Only the magnitude of the momentum changes; the direction stays. **Apply to MC only.** Parameters per |η| bin (0–0.7, 0.7–1.2, 1.2–2.5; beyond 2.5 uses the last bin). How the correction was derived is explained in the document *Track momentum calibration for 2018 BParking (v4): how it works and how to use it*.

**Track type.** In v4 all tracks share A, ε and M, but muons have their own a and c, stored in `per_bin_muon` (and its variations in `variations.shift_per_bin_muon`). Pass the track type to every call: `kind="muon"` (Python) or `rdscalib::Kind::Muon` (C++) for muons; the default, hadron, for kaons and pions. A file may also hold `per_bin_kaon` (`kind="kaon"`, `Kind::Kaon`); v4 does not. For v1–v3 the track type makes no difference.

## Contents

| Path | What |
| --- | --- |
| `data/rds_trkcalib_2018BParking_v4.json` | the current correction (three channels; muons with their own a, c): per-bin values, `per_bin_muon`, fit values + covariance, 15 statistical eigen-variations (MC-template statistics included), random-number convention, derivation record |
| `data/rds_trkcalib_2018BParking_v3.json` | the Ds-only correction (CMS convention, κ_eff categories), with 10 MC-template variations |
| `data/rds_trkcalib_2018BParking_v2.json` | the previous correction (CMS convention, softest-pT categories) |
| `data/rds_trkcalib_2018BParking_v1.json` | the first correction, old convention (kept for comparison) |
| `python/rds_trkcalib.py` | Python applier (numpy only), vectorised |
| `python/example_uproot.py` | corrects a flat ntuple and writes a friend tree (tested) |
| `cpp/RDsTrackCalib.h` | C++14 header-only applier, no dependencies (own JSON reader) |
| `cpp/example_standalone.cpp` | one candidate + muon, compiled and run (tested with g++ and clang++) |
| `cpp/example_rdf.C` | ROOT RDataFrame macro (written for ROOT ≥ 6.24; not run here, no ROOT in the test area) |
| `test/crosscheck.py` | builds the C++ test, runs C++ and Python on 200 000 tracks, compares everything |

## Use

Python:
```python
from rds_trkcalib import TrackCalib
cal = TrackCalib("data/rds_trkcalib_2018BParking_v4.json")
z = cal.gauss(run, lumi, event, track_id)                  # fixed per track
f = cal.factor(p, pt, eta, q, z)                           # p'/p ; arrays or scalars ; kaon or pion
f_mu = cal.factor(p, pt, eta, q, z, kind="muon")           # a muon
f_up = cal.factor(p, pt, eta, q, z, variation="eig3_up")
```

C++:
```cpp
#include "RDsTrackCalib.h"
rdscalib::TrackCalib cal("rds_trkcalib_2018BParking_v4.json");   // build once, e.g. in the constructor of the producer
double z = cal.gauss(run, lumi, event, trackIndex);
double f = cal.factor(p, pt, eta, q, z);                   // kaon or pion: multiply px, py, pz by f, recompute E
double fMu = cal.factor(p, pt, eta, q, z, "nominal", rdscalib::Kind::Muon);   // a muon
```

In a CMSSW ntuplizer the natural key is the index of the track in its input collection, for example:
```cpp
const auto& id = iEvent.id();
double z = cal.gauss(id.run(), id.luminosityBlock(), id.event(), iPFCand);          // packedPFCandidates index
double zLost = cal.gauss(id.run(), id.luminosityBlock(), id.event(), 1000000 + iLost); // offset for lostTracks
```
Any key works as long as the same physical track gets the same key in every candidate of the event. Flat ntuples without a track index can use `track_id_from_kinematics(pt, eta, phi)` / `rdscalib::trackIdFromKinematics` (pT rounded to 1 MeV, η and φ to 1e-4).

## Variations

| Name | Meaning |
| --- | --- |
| `nominal` | the fit result |
| `eig0_up/down` … | ±1σ along each eigenvector of the fit covariance (statistical; in v4 including the MC-template statistics); uncorrelated, add the effects in quadrature. Each moves the hadron and the muon terms together, so use the same name for all tracks of an event |
| `mctemplate_<term>_up/down` | v3 only: ±1σ of the MC-template uncertainty of one term, from refits that change the template random numbers and resample the MC; add in quadrature with the statistical ones |
| `scale_only` | scale terms only (A, ε, M), no extra smearing (for checks) |
| `smear_only` | extra smearing only (a, c), no scale change (for checks) |

Not included yet: kernel, mass window, background shape, category edges, D⁺-veto dependence of A.

## The muon

Use the muon's own pT, charge and |η|, with the muon track type (v4). Since v4 the muon terms are measured directly on J/ψ → μμ (muons of 3–30 GeV), so the muon correction is no longer an extrapolation from hadrons. The R(Ds) slim ntuples of 26 Oct 2026 store the muon pT, η and φ.

## Checks done

- C++ vs Python, all correction files (v1–v4), every variation and every track type (hadron, kaon, muon), 200 000 random tracks including bin edges, |η| > 2.5 and negative event numbers: the Gaussian z agrees to 4e-16 and the factor to 2e-16; the kinematic track id is identical.
- Applier vs fitter for v4-type files (muon a, c and kaon ε of their own), J/ψK toy tracks: factors identical to 1e-14 for muons and kaons (`tests/test_multi.py`).
- Applier vs fitter, 3.4 M signal-MC tracks with the fitter's z: factors identical to 2e-16.
- v2 vs v1 on 3.36 M MC tracks: scale factors differ by 2×10⁻⁵ on average; corrected m(KKπ) mean 1967.178 vs 1967.181 MeV, rms 9.655 vs 9.656 MeV (with the D⁺ veto).
- Hash-based z vs the fitter's numpy z on the same MC (v1): corrected m(KKπ) mean 1967.853 vs 1967.854 MeV, rms 9.830 vs 9.830 MeV (1945–1991 MeV window; uncorrected 1968.727 / 9.695).

Run `python test/crosscheck.py` after any change (needs numpy and a C++14 compiler; set `CXX` to choose it).
