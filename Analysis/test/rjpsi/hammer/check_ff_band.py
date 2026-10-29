#!/usr/bin/env python3
'''
Acceptance test for the covariance in an FF card: how big is the form-factor
band it actually implies?

The fitted BGL coefficients are strongly correlated -- the z-basis is nearly
degenerate over the narrow physical range -- so a coefficient sigma of O(0.3) on
a coefficient of O(0.03) is NOT automatically absurd: almost all of it lies along
directions that barely move the curve. The only meaningful check is to push each
eigendirection through Hammer's forward model and look at dF/F.

    python3 check_ff_band.py ../../../data/harrison_bglvar.json

What to expect: the quadrature band should be comparable to Harrison-2024's own
quoted precision (order 1-2%). Several times that means the covariance does not
belong to these central values -- which is exactly how the September card went
out of sync -- and the eigenvariations must not be used.

Needs only numpy and hammer_bgl_forward.py; no Hammer install, no lattice files.
'''
import argparse
import json

import numpy as np

import hammer_bgl_forward as HB

FF_NAMES = ('g', 'f', 'F1', 'F2')
SLICES = {'avec': slice(0, 4), 'bvec': slice(4, 8),
          'cvec': slice(8, 11), 'dvec': slice(11, 15)}


def unflatten(vec):
    return dict((k, vec[s]) for k, s in SLICES.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('card')
    ap.add_argument('--npoints', type=int, default=40)
    ap.add_argument('--per-direction', action='store_true',
                    help='also print the band of each eigendirection separately')
    args = ap.parse_args()

    with open(args.card) as fin:
        card = json.load(fin)
    central = np.concatenate([np.asarray(card[k], dtype=float)
                              for k in ('avec', 'bvec', 'cvec', 'dvec')])
    evecs = np.asarray(card['evecs'], dtype=float)
    sqev = np.asarray(card['sqrt_evals'], dtype=float)
    mat = evecs * sqev[None, :]                       # column j = 1 sigma direction j

    print('card %r   fit %s   covariance %s'
          % (card.get('name'), card.get('fit_date'), card.get('covariance_status')))
    for key in ('avec', 'bvec', 'cvec', 'dvec'):
        norm = float(np.sum(np.asarray(card[key]) ** 2))
        print('  unitarity %-5s sum a_n^2 = %.4f  %s'
              % (key, norm, 'OK' if norm < 1 else 'VIOLATED'))
    ndof = int(np.sum(sqev > 1e-4 * sqev.max()))
    print('  %d of %d eigendirections are non-degenerate '
          '(the rest are pinned by the regularisation and give w = nominal)'
          % (ndof, len(sqev)))

    q2max = (HB.MBC - HB.MJPSI) ** 2
    grid = np.linspace(0.10, q2max - 1e-3, args.npoints)
    base = np.array([HB.hammer_bgl_ff(q2, **unflatten(central)) for q2 in grid])

    total = np.zeros_like(base)
    per_dir = []
    for j in range(len(sqev)):
        up = np.array([HB.hammer_bgl_ff(q2, **unflatten(central + mat[:, j]))
                       for q2 in grid])
        delta = up - base
        total += delta ** 2
        per_dir.append(np.max(np.abs(delta) / np.abs(base), axis=0))

    if args.per_direction:
        print('\n  dir     sigma    max |dF/F| over the grid')
        print('                   ' + '  '.join('%7s' % n for n in FF_NAMES))
        for j, rel in enumerate(per_dir):
            print('  e%02d  %9.5f   ' % (j, sqev[j])
                  + '  '.join('%6.2f%%' % (100 * x) for x in rel))

    band = np.sqrt(total) / np.abs(base)
    print('\nquadrature band over all %d directions:' % len(sqev))
    print('   q2  ' + '  '.join('%8s' % n for n in FF_NAMES))
    for q2, row in zip(grid, band):
        if q2 == grid[0] or q2 == grid[-1] or abs(q2 % 2.0) < (grid[1] - grid[0]):
            print(' %5.2f  ' % q2 + '  '.join('%7.2f%%' % (100 * x) for x in row))
    print('\nlargest band anywhere: %.2f%% (on %s)'
          % (100 * band.max(), FF_NAMES[int(np.argmax(band.max(axis=0)))]))


if __name__ == '__main__':
    main()
