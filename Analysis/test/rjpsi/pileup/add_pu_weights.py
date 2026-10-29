#!/usr/bin/env python3
'''
Add the pileup weights to MC ntuples that were produced WITHOUT --pu.

Reads the `nti` branch the ntuplizer always writes and calls the same
Bmmm.Analysis.PileupWeights session the inspector calls, so the result is
identical to a production-time run -- see --closure, which proves it on a file
that has both.

    python3 add_pu_weights.py rjpsi_bc_summer24.root --campaign Summer24 -o pu_friend.root
    python3 add_pu_weights.py rjpsi_bc_summer24.root --campaign Summer24 -o /dev/null --closure

The output is a friend tree (default name 'pu'), row-aligned 1:1 with the input
tree(s), read in the order given.
'''
import argparse
import sys
import time

import numpy as np
import uproot

from Bmmm.Analysis.PileupWeights import BRANCH_NAMES, INPUT_BRANCH, make_pu_session


def progress(done, total, t0, width=40):
    frac = done / float(total) if total else 1.
    rate = done / max(time.time() - t0, 1e-9)
    eta  = (total - done) / rate if rate > 0 else float('inf')
    bar  = '#' * int(width * frac) + '-' * (width - int(width * frac))
    sys.stderr.write('\r  [%s] %5.1f%%  %d/%d rows  %.0f rows/s  ETA %.0f s   '
                     % (bar, 100 * frac, done, total, rate, eta))
    sys.stderr.flush()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--campaign', required=True,
                        help="MC campaign, e.g. Summer22EE; optionally 'campaign:card.json' "
                             "and/or ':allow-unconfirmed', as for the inspector's --pu")
    parser.add_argument('-t', '--tree', default='tree')
    parser.add_argument('--out-tree', default='pu')
    parser.add_argument('--chunk', type=int, default=500000)
    parser.add_argument('--closure', action='store_true',
                        help='the input already carries pu_weight_* branches from a --pu '
                             'run: recompute and compare instead of writing a friend. '
                             'Any difference is a bug, not a tolerance.')
    args = parser.parse_args()

    total = 0
    for f in args.inputs:
        with uproot.open('%s:%s' % (f, args.tree)) as t:
            have = set(t.keys())
            if INPUT_BRANCH not in have:
                sys.exit('[FATAL] %s has no %r branch' % (f, INPUT_BRANCH))
            if args.closure:
                missing = [b for b in BRANCH_NAMES if b not in have]
                if missing:
                    sys.exit('[FATAL] --closure needs the reference branches; %s lacks %s'
                             % (f, missing))
            total += t.num_entries

    pu = make_pu_session(args.campaign)
    read = [INPUT_BRANCH] + (BRANCH_NAMES if args.closure else [])

    fout = None
    if not args.closure:
        fout = uproot.recreate(args.output, compression=uproot.ZSTD(5))
        fout.mktree(args.out_tree, {b: np.float32 for b in BRANCH_NAMES})

    n_rows = n_nan_nti = 0
    sums   = dict.fromkeys(BRANCH_NAMES, 0.)
    counts = dict.fromkeys(BRANCH_NAMES, 0)
    diff   = dict.fromkeys(BRANCH_NAMES, 0)
    t0 = time.time()
    for f in args.inputs:
        for arrays in uproot.iterate('%s:%s' % (f, args.tree), read,
                                     step_size=args.chunk, library='np'):
            nti = arrays[INPUT_BRANCH]
            if n_rows == 0 and nti.size and np.all(np.isnan(nti)):
                sys.exit('[FATAL] %s: nti is NaN everywhere -- this is a data file, '
                         'pileup weights exist only for MC' % f)
            w = pu.weights_array(nti)
            for b in BRANCH_NAMES:
                ok = np.isfinite(w[b])
                sums[b]   += float(w[b][ok].sum())
                counts[b] += int(ok.sum())
                if args.closure:
                    ref = arrays[b].astype(np.float32)
                    same = (ref == w[b]) | (np.isnan(ref) & np.isnan(w[b]))
                    diff[b] += int(np.count_nonzero(~same))
            if fout is not None:
                fout[args.out_tree].extend(w)
            n_nan_nti += int(np.count_nonzero(~np.isfinite(nti)))
            n_rows    += nti.size
            progress(n_rows, total, t0)
    sys.stderr.write('\n')
    if fout is not None:
        fout.close()

    if n_rows != total:
        sys.exit('[FATAL] read %d rows, expected %d' % (n_rows, total))

    print(pu.summary())
    print('\nrows: %d   rows with NaN nti: %d' % (n_rows, n_nan_nti))
    print('average weight over the rows of THIS ntuple (after skim and selection;')
    print('it is not expected to be 1, and must not be forced to 1):')
    for b in BRANCH_NAMES:
        if counts[b]:
            print('    %-22s %.4f   (%d rows)' % (b, sums[b] / counts[b], counts[b]))

    if args.closure:
        bad = {b: n for b, n in diff.items() if n}
        if bad:
            print('\n[CLOSURE FAILED] rows differing between the ntuple and the recomputation:')
            for b, n in bad.items():
                print('    %-22s %d' % (b, n))
            sys.exit(1)
        print('\n[CLOSURE OK] all %d x %d values identical, NaNs included'
              % (n_rows, len(BRANCH_NAMES)))
    else:
        print('\nfriend tree %r written to %s' % (args.out_tree, args.output))


if __name__ == '__main__':
    main()
