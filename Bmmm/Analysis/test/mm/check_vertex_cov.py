'''
Is a stored vertex covariance actually a covariance matrix?

Written to answer one question: vtx_cov_yy coming out comparable to vtx_cov_zz
-- is the six-element storage scrambled, or is the dimuon vertex genuinely that
shape? The two have different signatures and this separates them without
needing to re-run the ntuplizer.

  1. Positive-definiteness. The stored order is (xx, xy, xz, yy, yz, zz), the
     upper triangle read row-major. If names and index pairs were zipped in
     different orders, an off-diagonal would land on a diagonal slot, and the
     rebuilt 3x3 would stop being positive definite for a large fraction of
     candidates. A real covariance is positive definite essentially always.
     THIS IS THE DECISIVE TEST: it passes only if the mapping is right.

  2. |rho| <= 1 for the three correlations. Same idea, cheaper to read.

  3. The resolutions themselves, in microns, so the numbers can be compared
     with what is expected: a primary vertex is ~10-20 um transverse and a bit
     worse longitudinally; a two-track dimuon vertex is much looser and NOT
     isotropic.

  4. The anisotropy, resolved in the candidate's own frame. A two-track vertex
     is poorly determined ALONG the pair's direction of flight and well
     determined across it, so sigma_xx and sigma_yy are not each other's twin
     -- which of the two is the large one depends on the candidate's phi. This
     rotates the transverse block into (along, across) and reports both. If the
     spread in x/y collapses once rotated, the anisotropy is geometry, not a
     bug.

    python check_vertex_cov.py data_2024d_test.root
    python check_vertex_cov.py data_2024d_test.root --prefix pv_cov
'''

from __future__ import print_function

import argparse

import numpy as np
import uproot


ELEMENTS = ['xx', 'xy', 'xz', 'yy', 'yz', 'zz']
# how the six stored elements map onto the 3x3, i.e. what utils.py wrote
INDEX_PAIRS = [(i, j) for i in range(3) for j in range(i, 3)]

CM_TO_UM = 1e4


def rebuild(arrays, prefix):
    '''(N, 3, 3) covariance stack from the six stored branches.'''
    n   = len(arrays['%s_xx' % prefix])
    cov = np.zeros((n, 3, 3))
    for name, (i, j) in zip(ELEMENTS, INDEX_PAIRS):
        v = np.asarray(arrays['%s_%s' % (prefix, name)], dtype=float)
        cov[:, i, j] = v
        cov[:, j, i] = v
    return cov


def summarise(label, values, unit=''):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        print('  %-26s (no finite entries)' % label)
        return
    print('  %-26s median %10.3f   [16%%, 84%%] = [%9.3f, %9.3f] %s'
          % (label, np.median(v), np.percentile(v, 16), np.percentile(v, 84), unit))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('infile')
    parser.add_argument('--tree'  , default='tree')
    parser.add_argument('--prefix', default='vtx_cov',
                        help='vtx_cov (dimuon SV), pv_cov (chosen PV), pv_bs_cov (refit PV)')
    args = parser.parse_args()

    wanted = ['%s_%s' % (args.prefix, e) for e in ELEMENTS]
    # phi of the pair, for the rotation in step 4; optional
    extra  = ['mu1_pt', 'mu1_phi', 'mu2_pt', 'mu2_phi']

    with uproot.open('%s:%s' % (args.infile, args.tree)) as tree:
        have = set(tree.keys())
        missing = [b for b in wanted if b not in have]
        if missing:
            raise SystemExit('missing branches: %s\navailable %s_* : %s'
                             % (missing, args.prefix,
                                sorted(b for b in have if b.startswith(args.prefix))))
        arrays = tree.arrays(wanted + [b for b in extra if b in have], library='np')

    cov = rebuild(arrays, args.prefix)
    n   = len(cov)
    finite = np.isfinite(cov).all(axis=(1, 2))
    cov, nfin = cov[finite], int(finite.sum())
    print('\n%s: %d candidates, %d with a finite %s_* matrix\n'
          % (args.infile, n, nfin, args.prefix))
    if not nfin:
        raise SystemExit('nothing to check')

    # ---- 1. positive definiteness -------------------------------------------
    eig    = np.linalg.eigvalsh(cov)
    posdef = (eig > 0).all(axis=1)
    frac   = posdef.mean()
    print('1. positive definite : %6.2f%% of candidates' % (100 * frac))
    print('   min eigenvalue    : %s' % np.array2string(
        np.percentile(eig[:, 0], [0, 1, 50]), precision=3))
    if frac > 0.99:
        print('   -> the six-element mapping is SOUND. A scrambled name/index zip '
              'would\n      put an off-diagonal on a diagonal and break this.')
    else:
        print('   -> NOT a covariance matrix for %.1f%% of candidates. Suspect the '
              'storage\n      mapping, or entries written for an INVALID vertex.'
              % (100 * (1 - frac)))

    # ---- 2. correlations -----------------------------------------------------
    sig  = np.sqrt(np.clip(np.diagonal(cov, axis1=1, axis2=2), 0, None))
    with np.errstate(divide='ignore', invalid='ignore'):
        rho_xy = cov[:, 0, 1] / (sig[:, 0] * sig[:, 1])
        rho_xz = cov[:, 0, 2] / (sig[:, 0] * sig[:, 2])
        rho_yz = cov[:, 1, 2] / (sig[:, 1] * sig[:, 2])
    print('\n2. correlations')
    for name, r in (('rho_xy', rho_xy), ('rho_xz', rho_xz), ('rho_yz', rho_yz)):
        fin = r[np.isfinite(r)]
        bad = float(np.mean(np.abs(fin) > 1.)) if len(fin) else 0.
        summarise(name, r)
        if bad > 1e-3:
            print('     ^ |rho|>1 for %.2f%% -- not a covariance' % (100 * bad))

    # ---- 3. the resolutions themselves ---------------------------------------
    print('\n3. resolutions [um]')
    for k, ax in enumerate('xyz'):
        summarise('sigma_%s' % ax, sig[:, k] * CM_TO_UM, 'um')
    with np.errstate(divide='ignore', invalid='ignore'):
        summarise('sigma_y / sigma_x', sig[:, 1] / sig[:, 0])
        summarise('sigma_z / sigma_x', sig[:, 2] / sig[:, 0])

    # ---- 3b. x/y symmetry: the real test -------------------------------------
    # Candidates are distributed uniformly in phi, so however anisotropic an
    # INDIVIDUAL vertex is, <cov_xx> and <cov_yy> must come out equal over the
    # sample. That makes this a clean null test with no modelling in it: a
    # significant xx/yy difference is a bug, and the per-candidate geometry
    # cannot explain it away. (It only widens the yy/xx distribution; it cannot
    # move its centre.)
    #
    # Compared on medians rather than means -- the per-candidate distribution is
    # long-tailed -- with a bootstrap uncertainty, so "consistent" has a number.
    print('\n3b. x/y symmetry over the sample (phi is uniform, so this must hold)')
    rng   = np.random.default_rng(0)
    nboot = 400
    idx   = rng.integers(0, len(cov), size=(nboot, len(cov)))
    r_obs = np.median(cov[:, 1, 1]) / np.median(cov[:, 0, 0])
    r_bs  = np.median(cov[idx, 1, 1], axis=1) / np.median(cov[idx, 0, 0], axis=1)
    err   = float(np.std(r_bs))
    pull  = (r_obs - 1.) / err if err > 0 else np.inf
    print('   median(cov_yy) / median(cov_xx) = %.4f +/- %.4f  (%.1f sigma from 1)'
          % (r_obs, err, pull))
    if abs(pull) < 3.:
        print('   -> consistent with 1: x and y are treated identically. Whatever')
        print('      makes zz look comparable to yy is NOT an x/y mix-up.')
    else:
        print('   -> NOT consistent with 1. phi symmetry says these must agree, so')
        print('      this is a real asymmetry between the x and y elements: a bug.')

    # and the comparison that prompted all this
    r_zx = np.median(cov[:, 2, 2]) / np.median(cov[:, 0, 0])
    r_zy = np.median(cov[:, 2, 2]) / np.median(cov[:, 1, 1])
    print('   median(cov_zz) / median(cov_xx) = %.2f' % r_zx)
    print('   median(cov_zz) / median(cov_yy) = %.2f' % r_zy)
    print('   (a two-track vertex is not dramatically worse in z than in the')
    print('    transverse plane -- a ratio of a few is ordinary, not a symptom.')
    print('    Compare against pv_cov, where many tracks make the PV tighter in')
    print('    both, to see whether the RATIO differs or only the scale.)')

    # ---- 4. anisotropy in the candidate frame --------------------------------
    if all(b in arrays for b in ('mu1_pt', 'mu1_phi', 'mu2_pt', 'mu2_phi')):
        px  = arrays['mu1_pt'] * np.cos(arrays['mu1_phi']) + \
              arrays['mu2_pt'] * np.cos(arrays['mu2_phi'])
        py  = arrays['mu1_pt'] * np.sin(arrays['mu1_phi']) + \
              arrays['mu2_pt'] * np.sin(arrays['mu2_phi'])
        phi = np.arctan2(py, px)[finite]
        c, s = np.cos(phi), np.sin(phi)
        # transverse block rotated into (along pair pT, across it)
        along  = c*c*cov[:, 0, 0] + 2*c*s*cov[:, 0, 1] + s*s*cov[:, 1, 1]
        across = s*s*cov[:, 0, 0] - 2*c*s*cov[:, 0, 1] + c*c*cov[:, 1, 1]
        print('\n4. transverse plane rotated into the candidate frame [um]')
        summarise('sigma_along  (pair pT)', np.sqrt(np.clip(along,  0, None)) * CM_TO_UM, 'um')
        summarise('sigma_across (pair pT)', np.sqrt(np.clip(across, 0, None)) * CM_TO_UM, 'um')
        spread_lab = np.std(np.log(sig[:, :2].clip(1e-12)), axis=0).mean()
        rot        = np.sqrt(np.clip(np.stack([along, across], axis=1), 1e-24, None))
        spread_rot = np.std(np.log(rot), axis=0).mean()
        print('   log-spread  lab (x,y) = %.3f   rotated (along,across) = %.3f'
              % (spread_lab, spread_rot))
        if spread_rot < spread_lab:
            print('   -> the frame absorbs part of the x/y scatter: the anisotropy '
                  'follows the\n      candidate direction, i.e. it is geometry, not '
                  'a storage bug.')
    else:
        print('\n4. skipped (needs mu1_pt/mu1_phi/mu2_pt/mu2_phi in the tree)')

    print()


if __name__ == '__main__':
    main()
