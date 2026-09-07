# Bmmm

## installation
```
cmsrel CMSSW_10_6_28
cd CMSSW_10_6_28/src
cmsenv
git init
git remote add origin git@github.com:rmanzoni/Bmmm.git
git fetch origin
git checkout main
scram b
```

## run

```
cd $CMSSW_BASE/src/Bmmm/Analysis/test
ipython -i -- inspector_bmmm_analysis.py --inputFiles="../../../../../rds/CMSSW_10_6_28/src/Bmmm/MINI/Bmmm_signal_MINI.root" --filename=signal --mc
```

## covariance matrices

The J/psi + charged-object ntuples (`inspector_jpsi_mu.py`, `inspector_jpsi_tk.py`)
persist the covariance matrices needed to measure, and then correct, the data/MC
track-resolution mismodelling.

**Tracks** -- the 15 independent elements of the 5x5 curvilinear covariance of
each object's best track, plus the scale factor actually applied to each
parameter:

```
<obj>_cov_<par_i>_<par_j>     par in (qoverp, lambda, phi, dxy, dsz)
<obj>_cov_scale_<par>         1 when running without --cov-scale
```

for `<obj>` in `mu1`, `mu2` (`mu3` in the J/psi mu channel) and `k`. These are
the RAW uncertainties -- unlike `<obj>_dxy_e` / `<obj>_dz_e`, which fold in the
primary-vertex error. Derive `sigma_i = sqrt(cov_<par_i>_<par_i>)` and
`rho_ij = cov_ij / (sigma_i sigma_j)` offline.

**Vertices** -- the 6 independent elements of the 3x3 position covariance of
every vertex whose position is stored:

```
pv_cov_<xx|xy|xz|yy|yz|zz>      the PV actually used as reference (cand.pv_bs)
sv_cov_*                        2mu + bachelor vertex
jpsi_cov_*                      mass-constrained dimuon vertex
pi_sv_cov_*                     pion-hypothesis vertex (J/psi + track channel)
```

`sv_cov_*` and `jpsi_cov_*` are derived from the covariances of the tracks that
were fitted, so correcting the tracks and refitting propagates into them
automatically -- they are stored to *check* that, not to be corrected. `pv_cov_*`
does not follow: the PV is fitted from the whole event's track set, of which
`--cov-scale` only touches the candidate's own tracks, and the refit is
beamspot-constrained, so its transverse covariance is largely the beamspot's.
It follows `pv_bs`, i.e. the beamspot-constrained AVF refit with the signal muons
removed when `pv_refit_valid`, and the Run2 hybrid PV otherwise -- in that
fallback the position comes from the beamspot but the covariance from the
unrefitted PV, so cut on `pv_refit_valid` before using it quantitatively.

**Applying a correction.** Once measured, feed it back with `--cov-scale`. Every
track is rebuilt around `D cov D` (`D = diag(scales)`) before the vertex fits, the
IP3D grid and the jet-track distances, so each `sigma_i` is scaled and every
correlation is left exactly where it was:

```
--cov-scale 'dxy=1.05,dsz=1.02'            # flat, for closure tests / systematics
--cov-scale table.json                     # binned in (pt, |eta|)
--cov-scale corr.json:dxy=sigma_dxy_scale  # correctionlib
```

Without `--cov-scale` nothing is scaled and the reconstruction is bit-for-bit the
one before this existed -- the mode to run in while the correction is still being
measured. See `Bmmm/Analysis/utils.py` (`scale_cov`, `CovScaler`,
`vertex_cov_element`), `JpsiChargedCandidate.fit_track`, and
`test/rjpsi/test_cov_scaling.py`.
