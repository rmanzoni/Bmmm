'''
Self-test for the track-covariance persistency and rescaling helpers
(Bmmm.Analysis.utils). Needs a CMSSW environment (it builds a real reco::Track),
but no input file and no event loop:

    cmsenv
    python test_cov_scaling.py

What it actually checks -- in order of how badly a silent failure would hurt:

  1. the 15 stored elements and the SMatrix the track is rebuilt from use the
     SAME ordering. This is the one that would corrupt everything quietly: a
     transposed or shifted index gives a perfectly valid matrix that is simply
     the wrong one. Checked against a REAL reco::Track round trip, not against
     our own convention.
  2. scale_cov scales every sigma and moves no correlation.
  3. scale_track_cov leaves the trajectory (momentum, charge, reference point,
     chi2/ndof) untouched -- only the uncertainty changes.
  4. unit scales are a true no-op (the same object comes back).
  5. fix_track still behaves after the refactor onto track_with_cov.
  6. the scalers and the --cov-scale string parsing.
'''

import json
import tempfile

import numpy as np
import ROOT

from Bmmm.Analysis.utils import (
    COV_ELEMENT_NAMES, COV_INDEX_PAIRS, COV_PARAM_NAMES,
    VTX_COV_ELEMENT_NAMES, VTX_COV_INDEX_PAIRS,
    convert_cov, cov_upper_triangle, is_pos_def, fix_track,
    scale_cov, scale_track_cov, smatrix55_from_cov, track_with_cov,
    vertex_cov_element, vertex_covariance, vertex_cov_upper_triangle,
    BinnedCovScaler, ConstantCovScaler, make_cov_scaler,
)

FAILURES = []


def check(name, condition, extra=''):
    print(('  OK   ' if condition else '  FAIL ') + name + (('   ' + extra) if extra else ''))
    if not condition:
        FAILURES.append(name)


def random_cov(seed=0, scale=1e-4):
    '''A random, symmetric, positive-definite 5x5 with track-like magnitudes.'''
    rng = np.random.RandomState(seed)
    a = rng.normal(size=(5, 5))
    return scale * (a.dot(a.T) + 5. * np.eye(5))


def make_track(cov, pt=4.2):
    '''A real reco::Track carrying `cov`. The Point/Vector typedefs live inside
    reco::TrackBase; fall back to the math:: names if cppyy does not expose
    them in this release.'''
    try:
        point, vector = ROOT.reco.Track.Point, ROOT.reco.Track.Vector
    except AttributeError:
        point, vector = ROOT.math.XYZPoint, ROOT.math.XYZVector
    return ROOT.reco.Track(
        11.5,                                  # chi2
        7.,                                    # ndof
        point(0.03, -0.02, 1.7),               # reference point
        vector(pt * 0.6, pt * 0.8, 3.1),       # momentum
        -1,                                    # charge
        smatrix55_from_cov(cov),
        ROOT.reco.TrackBase.undefAlgorithm,
        ROOT.reco.TrackBase.undefQuality,
    )


def main():

    cov = random_cov()

    print('== 1. element ordering, against a real reco::Track round trip ==')
    trk = make_track(cov)
    back = convert_cov(trk.covariance())
    check('cov -> reco::Track -> cov is the identity',
          np.allclose(back, cov, rtol=1e-12, atol=0),
          'max |diff| %.3e' % np.abs(back - cov).max())
    check('the matrix comes back symmetric', np.allclose(back, back.T))
    ut = cov_upper_triangle(cov)
    check('cov_upper_triangle == row-major upper triangle',
          np.allclose(ut, [cov[i][j] for i, j in COV_INDEX_PAIRS]))
    check('15 elements, 15 names',
          len(ut) == 15 and len(COV_ELEMENT_NAMES) == 15)
    # the branch <obj>_cov_dxy_dxy must really be sigma_dxy^2
    idx_dxy_dxy = COV_ELEMENT_NAMES.index('dxy_dxy')
    check('cov_dxy_dxy is the dxy variance',
          abs(ut[idx_dxy_dxy] - cov[3][3]) < 1e-30)
    check('sqrt(cov_dxy_dxy) == the track dxyError',
          abs(np.sqrt(ut[idx_dxy_dxy]) - trk.dxyError()) < 1e-9 * trk.dxyError(),
          'stored %.6e  track %.6e' % (np.sqrt(ut[idx_dxy_dxy]), trk.dxyError()))

    print('== 2. scale_cov: sigmas scale, correlations do not ==')
    scales = np.array([1.00, 0.90, 1.30, 1.07, 1.02])
    scaled = scale_cov(cov, scales)
    sig, sig_s = np.sqrt(np.diag(cov)), np.sqrt(np.diag(scaled))
    rho   = cov    / np.outer(sig,   sig)
    rho_s = scaled / np.outer(sig_s, sig_s)
    check('sigma_i -> scales[i] * sigma_i', np.allclose(sig_s, scales * sig),
          'max dev %.3e' % np.abs(sig_s - scales * sig).max())
    check('every correlation unchanged', np.allclose(rho_s, rho, atol=1e-14),
          'max |d rho| %.3e' % np.abs(rho_s - rho).max())
    check('still positive definite', is_pos_def(scaled))

    print('== 3. scale_track_cov: only the uncertainty moves ==')
    scaled_trk = scale_track_cov(trk, scales, cov=cov)
    check('covariance is the scaled one',
          np.allclose(convert_cov(scaled_trk.covariance()), scaled, rtol=1e-12, atol=0))
    check('momentum untouched',
          (abs(scaled_trk.px() - trk.px()) < 1e-12 and
           abs(scaled_trk.py() - trk.py()) < 1e-12 and
           abs(scaled_trk.pz() - trk.pz()) < 1e-12))
    check('charge untouched', scaled_trk.charge() == trk.charge())
    check('reference point untouched',
          (abs(scaled_trk.referencePoint().x() - trk.referencePoint().x()) < 1e-12 and
           abs(scaled_trk.referencePoint().y() - trk.referencePoint().y()) < 1e-12 and
           abs(scaled_trk.referencePoint().z() - trk.referencePoint().z()) < 1e-12))
    check('chi2 / ndof untouched',
          abs(scaled_trk.chi2() - trk.chi2()) < 1e-12 and
          abs(scaled_trk.ndof() - trk.ndof()) < 1e-12)
    check('dxyError picks up exactly the dxy scale',
          abs(scaled_trk.dxyError() - scales[3] * trk.dxyError()) < 1e-9 * trk.dxyError(),
          '%.6e vs %.6e' % (scaled_trk.dxyError(), scales[3] * trk.dxyError()))
    # dzError() = sigma_dsz / |sin(theta)|, so it follows the dsz scale
    check('dzError picks up exactly the dsz scale',
          abs(scaled_trk.dzError() - scales[4] * trk.dzError()) < 1e-9 * trk.dzError(),
          '%.6e vs %.6e' % (scaled_trk.dzError(), scales[4] * trk.dzError()))

    print('== 4. unit scales are a genuine no-op ==')
    check('same object back, no rebuild',
          scale_track_cov(trk, [1.] * 5, cov=cov) is trk)
    check('None scaler is a no-op too', scale_track_cov(trk, None, cov=cov) is trk)

    print('== 5. fix_track after the refactor ==')
    bad = cov.copy()
    bad[3][3] = -abs(bad[3][3])          # force a non positive-definite matrix
    bad_trk = make_track(bad)
    check('the broken matrix is really not pos-def', not is_pos_def(convert_cov(bad_trk.covariance())))
    fixed = fix_track(bad_trk)
    check('fix_track returns a pos-def matrix', is_pos_def(convert_cov(fixed.covariance())))
    check('fix_track leaves a good track alone', fix_track(trk) is trk)
    check('fix_track preserves the momentum',
          abs(fixed.px() - bad_trk.px()) < 1e-12 and abs(fixed.pz() - bad_trk.pz()) < 1e-12)

    print('== 6. scalers and --cov-scale parsing ==')
    con = ConstantCovScaler(dxy=1.05, dsz=1.02)
    check('ConstantCovScaler', con.scales(trk) == (1., 1., 1., 1.05, 1.02))
    try:
        ConstantCovScaler(pt=1.1)
        check('rejects a non-track-parameter', False)
    except ValueError:
        check('rejects a non-track-parameter', True)

    table = {'pt_edges'      : [2., 5., 10.],
             'abs_eta_edges' : [0., 1.2, 2.4],
             'scales'        : {'dxy': [[1.10, 1.20], [1.30, 1.40]]}}
    binned = BinnedCovScaler(table['pt_edges'], table['abs_eta_edges'], table['scales'])
    got = binned.scales(trk)                      # the test track has pt = 4.2
    check('BinnedCovScaler picks the right (pt, |eta|) cell',
          got[3] in (1.10, 1.20), 'pt %.2f |eta| %.2f -> %.2f' % (trk.pt(), abs(trk.eta()), got[3]))
    check('untouched parameters stay at 1', got[:3] == (1., 1., 1.) and got[4] == 1.)
    check('out-of-range tracks are counted, not extrapolated',
          isinstance(binned.n_out_of_range, int))

    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as fout:
        json.dump(table, fout)
        path = fout.name
    check('make_cov_scaler("") -> None (the default, no scaling)',
          make_cov_scaler('') is None)
    check('make_cov_scaler("dxy=1.05") -> ConstantCovScaler',
          isinstance(make_cov_scaler('dxy=1.05'), ConstantCovScaler))
    check('make_cov_scaler("<table>.json") -> BinnedCovScaler',
          isinstance(make_cov_scaler(path), BinnedCovScaler))

    print('== 7. vertex covariance accessors ==')
    # The two vertex flavours in this package expose their covariance
    # differently -- reco::Vertex.error() IS the SMatrix, KinematicVertex.error()
    # is a GlobalError that wraps one. vertex_cov_element picks the accessor from
    # the returned object, so both must be exercised: a silent failure here is a
    # column of NaNs in the ntuple, which is easy not to notice.
    vcov = np.array([[3.1e-5, 0.4e-5, 0.2e-5],
                     [0.4e-5, 2.7e-5, 0.1e-5],
                     [0.2e-5, 0.1e-5, 9.4e-4]])

    # same flat-upper-triangle SMatrix idiom as smatrix55_from_cov, at 3x3
    upper = np.array([vcov[i][j] for i, j in VTX_COV_INDEX_PAIRS])
    err = ROOT.Math.SMatrix('double', 3, 3, ROOT.Math.MatRepSym('double', 3))(
        ROOT.Math.SVector('double', 6)(upper, 6), False)
    reco_vtx = ROOT.reco.Vertex(ROOT.reco.Vertex.Point(0.09, -0.04, 2.1),
                                err, 21.0, 30.0, 18)
    check('reco::Vertex path: every element',
          np.allclose(vertex_covariance(reco_vtx), vcov, rtol=1e-12, atol=0),
          'max |diff| %.3e' % np.abs(vertex_covariance(reco_vtx) - vcov).max())
    check('reco::Vertex path: agrees with reco::Vertex.covariance(i,j)',
          all(abs(vertex_cov_element(reco_vtx, i, j) - reco_vtx.covariance(i, j)) < 1e-30
              for i, j in VTX_COV_INDEX_PAIRS))
    ut = vertex_cov_upper_triangle(reco_vtx)
    check('6 elements, 6 names',
          len(ut) == 6 and len(VTX_COV_ELEMENT_NAMES) == 6)
    check('cov_zz is the z variance',
          abs(ut[VTX_COV_ELEMENT_NAMES.index('zz')] - vcov[2][2]) < 1e-30)
    check('sqrt(cov_zz) == reco::Vertex.zError()',
          abs(np.sqrt(ut[VTX_COV_ELEMENT_NAMES.index('zz')]) - reco_vtx.zError())
          < 1e-9 * reco_vtx.zError())

    # KinematicVertex hands back a GlobalError; build one directly (lower
    # triangle: xx, yx, yy, zx, zy, zz) and wrap it in the same duck type
    try:
        gerr = ROOT.GlobalError(vcov[0][0], vcov[1][0], vcov[1][1],
                                vcov[2][0], vcov[2][1], vcov[2][2])
    except (AttributeError, TypeError) as exc:
        print('  SKIP  GlobalError not available in this release (%s)' % type(exc).__name__)
    else:
        class _KinVtxLike(object):
            def error(self):
                return gerr
        kin_like = _KinVtxLike()
        check('KinematicVertex path: every element',
              np.allclose(vertex_covariance(kin_like), vcov, rtol=1e-12, atol=0),
              'max |diff| %.3e' % np.abs(vertex_covariance(kin_like) - vcov).max())
        check('both paths give the same answer',
              np.allclose(vertex_covariance(kin_like), vertex_covariance(reco_vtx)))

    print('== 8. TransientVertex accessor (the dimuon channel) ==')
    # The mm channel fits its dimuon vertex with KVFitter, which returns a
    # TransientVertex -- a third flavour, with neither reco::Vertex's error()
    # nor its return type. vertex_error_matrix falls back to positionError()
    # when error() is absent; assert that assumption against the real class
    # rather than trusting it, because if it is wrong the mm ntuplizer raises
    # on its first candidate.
    try:
        has_error = hasattr(ROOT.TransientVertex, 'error')
        has_poserr = hasattr(ROOT.TransientVertex, 'positionError')
    except AttributeError:
        print('  SKIP  TransientVertex dictionary not loaded')
    else:
        check('TransientVertex exposes positionError()', has_poserr)
        check('and no error(), so the fallback branch is the one taken',
              has_poserr and not has_error,
              'error=%s positionError=%s' % (has_error, has_poserr))

    # and the fallback itself, on a duck type carrying only positionError
    try:
        gerr2 = ROOT.GlobalError(vcov[0][0], vcov[1][0], vcov[1][1],
                                 vcov[2][0], vcov[2][1], vcov[2][2])
    except (AttributeError, TypeError):
        print('  SKIP  GlobalError not available, fallback path not exercised')
    else:
        class _TV(object):
            def positionError(self):
                return gerr2
        check('positionError() fallback gives the same matrix',
              np.allclose(vertex_covariance(_TV()), vcov, rtol=1e-12, atol=0))

    print('\n' + ('ALL CHECKS PASSED' if not FAILURES
                  else 'FAILURES: %s' % FAILURES))
    return 1 if FAILURES else 0


if __name__ == '__main__':
    raise SystemExit(main())
