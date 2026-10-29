'''
Self-test for the covflow track-covariance CORRECTION (utils.CovFlowCorrector,
JpsiChargedCandidate.fit_track). Companion to test_cov_scaling.py, which covers
the diagonal --cov-scale path.

Needs a CMSSW environment (it builds a real reco::Track) AND torch + zuko +
covflow on the PYTHONPATH. No input file, no event loop, no trained model: the
flows are built here and the run directory is synthesised in a tmpdir.

    cmsenv
    python3 test_covflow_correction.py

What it actually checks -- in order of how badly a silent failure would hurt:

  1. covflow.features.PACK_NAMES and Bmmm's COV_ELEMENT_NAMES are the same 15
     elements in the same order. A permutation here is the catastrophic silent
     failure: every corrected matrix would be valid and wrong.
  2. CovFlowCorrector._morph_batch reproduces covflow.correct.morph exactly. The
     corrector inlines the morph to keep the latent z for the diagnostic branch;
     that inlining must not drift from the function it copies.
  3. IDENTITY CLOSURE: with flow_data and flow_mc the same network,
     f_data^-1 . f_mc is the identity, so the corrected covariance must come
     back equal to the raw one. This is the end-to-end test of the whole feature
     round trip (pack -> log-sigma/CPC -> standardise -> flow -> back), and the
     only one here that exercises it against an independent expectation.
     It is checked on sigma_i and on the CORRELATIONS separately, never as a
     relative error on cov_ij: cov_ij = sigma_i sigma_j rho_ij, so wherever
     rho_ij happens to sit near zero -- which for a random PD matrix is most
     places -- a relative error on cov_ij measures the smallness of rho, not the
     accuracy of anything. The two diagnostics printed alongside (the feature
     map alone in float64, the flow alone in standardised units) localise a
     failure to one of the two halves instead of leaving it to guesswork.
  4. a non-positive-definite input is left UNCORRECTED and flagged, not patched.
  5. prime() memoizes: the same object gets one correction, reused everywhere.
  6. the rebuilt reco::Track carries the corrected matrix and nothing else moved.
  7. covflow.json validation: a context list that disagrees with scalers.json is
     a hard error, not a silently misaligned conditioning vector.
'''

import json
import os
import shutil
import tempfile

import numpy as np
import ROOT

from Bmmm.Analysis.utils import (
    COV_ELEMENT_NAMES, COV_INDEX_PAIRS,
    convert_cov, cov_upper_triangle, is_pos_def, track_with_cov,
    CovFlowCorrector, make_cov_corrector,
)

FAILURES = []

# reco::TrackBase stores float covariance_[15]: single precision. Anything that
# crosses a real track is only good to float32 epsilon. See test_cov_scaling.py.
FLOAT_RTOL = 1e-6

# The identity-closure floor. The flows are float32, so a round trip through one
# of them lands ~1e-6 in standardised feature space, and the standardiser
# (std ~0.4) plus the log-sigma / CPC maps carry that into sigma and rho at the
# same order. Real corrections move sigma at the percent level and rho at ~1e-2,
# so this floor sits four orders below anything worth quoting -- but it IS the
# level below which a covariance correction means nothing, which is a fair
# question to be asked about the method.
IDENTITY_TOL = 1e-5


def check(name, condition, extra=''):
    print(('  OK   ' if condition else '  FAIL ') + name + (('   ' + extra) if extra else ''))
    if not condition:
        FAILURES.append(name)


def random_cov(seed=0, scale=1e-4):
    '''A random, symmetric, positive-definite 5x5 with track-like magnitudes.'''
    rng = np.random.RandomState(seed)
    a = rng.normal(size=(5, 5))
    return scale ** 2 * (a.dot(a.T) + 5. * np.eye(5))


def fake_track(cov, pt=7.0, eta=0.3):
    '''A real reco::Track with a given covariance, momentum and reference point.'''
    from Bmmm.Analysis.utils import smatrix55_from_cov
    p3 = ROOT.math.XYZVector(pt, 0.0, pt * np.sinh(eta))
    ref = ROOT.math.XYZPoint(0.01, -0.02, 0.5)
    return ROOT.reco.Track(12.0, 10.0, ref, p3, 1, smatrix55_from_cov(cov),
                           ROOT.reco.TrackBase.undefAlgorithm,
                           ROOT.reco.TrackBase.undefQuality)


class FakeMuon(object):
    '''Just enough of a pat::Muon for the corrector: bestTrack() and the
    attributes the candidate memoizes on it.'''
    def __init__(self, track):
        self._track = track

    def bestTrack(self):
        return self._track


def make_run_directory(path, context, n_features=15, seed=0):
    '''Write a covflow run directory whose two flows are the SAME network, so
    the morph is the identity by construction.'''
    from covflow import data as CD
    from covflow import flows as CFL

    cfg = CFL.FlowConfig(n_features=n_features, n_context=len(context),
                         transforms=2, hidden=(16, 16), bins=6, seed=seed)
    flow = CFL.build_flow(cfg)
    CFL.save_flow(flow, os.path.join(path, 'flow_mc.pt'))
    CFL.save_flow(flow, os.path.join(path, 'flow_data.pt'))

    # a standardiser fitted on a spread of plausible features/contexts
    rng = np.random.RandomState(1)
    X = rng.normal(-7.0, 0.4, size=(2000, 15))
    C = rng.normal(2.0, 0.4, size=(2000, len(context)))
    CD.Standardiser.fit(X, C).save(os.path.join(path, 'scalers.json'))

    with open(os.path.join(path, 'covflow.json'), 'w') as fout:
        json.dump({'context': list(context), 'features': 'all',
                   'param': 'logsigma_corr', 'transforms': 2,
                   'hidden': [16, 16], 'bins': 6, 'seed': seed}, fout, indent=2)


def main():
    from covflow import correct as CF
    from covflow import features as CFEAT
    from covflow import flows as CFL

    context = ['log_pt', 'eta', 'n_pix_hit']
    tmp = tempfile.mkdtemp(prefix='covflow_test_')
    try:
        print('\n== 1. the packed 15-element order is the SAME on both sides ==')
        check('covflow PACK_NAMES == Bmmm COV_ELEMENT_NAMES',
              list(CFEAT.PACK_NAMES) == list(COV_ELEMENT_NAMES),
              '%s' % list(CFEAT.PACK_NAMES)[:3])
        check('covflow PACK_PAIRS == Bmmm COV_INDEX_PAIRS',
              [tuple(p) for p in CFEAT.PACK_PAIRS] == list(COV_INDEX_PAIRS))

        make_run_directory(tmp, context)
        corr = make_cov_corrector(tmp)
        check('make_cov_corrector builds a CovFlowCorrector',
              isinstance(corr, CovFlowCorrector))
        check('context read from covflow.json', corr.context_names == context)

        print('\n== 2. _morph_batch reproduces covflow.correct.morph ==')
        packed = np.array([cov_upper_triangle(random_cov(seed=i)) for i in range(64)])
        ctx = np.column_stack([np.log(np.linspace(3., 30., 64)),
                               np.linspace(-2.2, 2.2, 64),
                               np.random.RandomState(0).randint(1, 6, 64)])
        mine, zmax = corr._morph_batch(packed, ctx)
        theirs, _, _ = CF.morph(packed, ctx, corr.flow_mc, corr.flow_data,
                                corr.scaler, param=corr.param,
                                active_features=corr.active_features,
                                feature_indices=corr.idx, device=corr.device)
        check('inlined morph == covflow.correct.morph',
              np.allclose(mine, theirs, rtol=1e-9, atol=0),
              'max rel diff %.2e' % np.max(np.abs(mine / theirs - 1.)))
        check('zmax is finite and positive',
              np.all(np.isfinite(zmax)) and np.all(zmax > 0))

        print('\n== 3. identity closure: same flow twice -> covariance unchanged ==')

        # Localise any failure BEFORE judging it: the two halves of the morph
        # fail for completely different reasons, and only one of them would be a
        # bug in this package.
        M_raw = CFEAT.packed_to_matrix(packed)
        back  = CFEAT.matrix_to_packed(corr.to_mat(corr.to_feat(M_raw)))
        print('   feature map round trip      max rel %.2e   (float64, expect ~1e-15)'
              % np.max(np.abs(back / packed - 1.)))

        ys  = corr.scaler.x(corr.to_feat(M_raw))
        cs  = corr.scaler.c(ctx)
        z   = CFL.data_to_latent(corr.flow_mc, ys[:, corr.idx], cs)
        ys2 = CFL.latent_to_data(corr.flow_mc, z, cs)
        print('   flow round trip (std units) max abs %.2e   (float32, expect ~1e-6)'
              % np.max(np.abs(ys2 - ys[:, corr.idx])))

        # sigma and rho, separately. NOT cov_ij / cov_ij - 1: see the header.
        M_corr = CFEAT.packed_to_matrix(mine)
        s_raw  = np.sqrt(np.diagonal(M_raw,  axis1=1, axis2=2))
        s_corr = np.sqrt(np.diagonal(M_corr, axis1=1, axis2=2))
        iu     = np.triu_indices(5, 1)
        d_sig  = np.abs(s_corr / s_raw - 1.)
        d_rho  = (np.abs(M_corr - M_raw)
                  / (s_raw[:, :, None] * s_raw[:, None, :]))[:, iu[0], iu[1]]

        check('sigma_i unchanged when f_data is f_mc',
              np.max(d_sig) < IDENTITY_TOL, 'max rel %.2e' % np.max(d_sig))
        check('correlations unchanged when f_data is f_mc',
              np.max(d_rho) < IDENTITY_TOL, 'max abs %.2e' % np.max(d_rho))
        check('every corrected matrix is positive definite',
              bool(np.all(CFEAT.is_positive_definite(M_corr))))

        print('\n== 4. a non-PD input is left alone, flagged, and counted ==')
        bad = random_cov(seed=3)
        bad[3][3] = -1e-6
        check('the test matrix really is not positive definite', not is_pos_def(bad))
        mu_bad = FakeMuon(fake_track(np.abs(bad)))
        mu_bad.cov = bad
        mu_bad.is_cov_pos_def = False
        before = corr.n_not_pos_def
        corr.prime([mu_bad])
        check('covflow_ok is False', mu_bad.covflow_ok is False)
        check('cov_corr is all NaN', bool(np.all(np.isnan(mu_bad.cov_corr))))
        check('correct() returns None -> fit_track uses the raw track',
              corr.correct(mu_bad, mu_bad.bestTrack(), bad) is None)
        check('counted as not positive definite', corr.n_not_pos_def == before + 1)

        print('\n== 5. prime() memoizes: one correction per object per event ==')
        cov = random_cov(seed=7)
        mu = FakeMuon(fake_track(cov))
        corr.prime([mu])
        first = np.array(mu.cov_corr, copy=True)
        n_after_first = corr.n_corrected
        corr.prime([mu])
        check('a second prime() does not recompute',
              corr.n_corrected == n_after_first)
        check('the memoized matrix is unchanged',
              np.array_equal(first, mu.cov_corr))
        check('correct() returns the same matrix',
              np.array_equal(corr.correct(mu, mu.bestTrack(), cov), first))

        print('\n== 6. the rebuilt track carries the corrected matrix ==')
        raw_trk = mu.bestTrack()
        new_trk = track_with_cov(raw_trk, mu.cov_corr)
        check('covariance is the corrected one',
              np.allclose(convert_cov(new_trk.covariance()), mu.cov_corr,
                          rtol=FLOAT_RTOL, atol=0))
        check('momentum untouched',
              new_trk.momentum().x() == raw_trk.momentum().x() and
              new_trk.momentum().z() == raw_trk.momentum().z())
        check('charge untouched', new_trk.charge() == raw_trk.charge())
        check('reference point untouched',
              new_trk.referencePoint().z() == raw_trk.referencePoint().z())
        check('chi2/ndof untouched',
              new_trk.chi2() == raw_trk.chi2() and new_trk.ndof() == raw_trk.ndof())

        print('\n== 7. covflow.json must agree with scalers.json ==')
        with open(os.path.join(tmp, 'covflow.json')) as fin:
            payload = json.load(fin)
        payload['context'] = ['log_pt', 'eta']          # one short
        with open(os.path.join(tmp, 'covflow.json'), 'w') as fout:
            json.dump(payload, fout)
        try:
            make_cov_corrector(tmp)
        except ValueError as exc:
            check('a context/scaler length mismatch raises', 'context' in str(exc))
        else:
            check('a context/scaler length mismatch raises', False)

        payload['context'] = ['log_pt', 'eta', 'not_a_variable']
        with open(os.path.join(tmp, 'covflow.json'), 'w') as fout:
            json.dump(payload, fout)
        try:
            make_cov_corrector(tmp)
        except KeyError as exc:
            check('an unknown context variable raises', 'not_a_variable' in str(exc))
        else:
            check('an unknown context variable raises', False)

        check('an empty --covflow is None (the default)',
              make_cov_corrector('') is None)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print('\n' + ('ALL CHECKS PASSED' if not FAILURES
                  else 'FAILURES: %s' % FAILURES))
    return 1 if FAILURES else 0


if __name__ == '__main__':
    raise SystemExit(main())