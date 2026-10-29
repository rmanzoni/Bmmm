#!/usr/bin/env python3
'''
Add the Bs -> Ds(*) form-factor weights to R(Ds) ntuples produced WITHOUT
--hammer.

Reads the gen_sig / gen_bs_* / gen_ham_* leaves the ntuplizer writes (in both
'tree' and 'tree_gen') and calls the same Bmmm.Analysis.HammerDsFF session as
the inspector, so the weights are identical to a production-time run; check it
with --closure on a file that has both.

    python3 add_hammer_weights_rds.py rds_signal.root --card card.json -o hammer_friend.root
    python3 add_hammer_weights_rds.py rds_signal.root --card card.json -t tree_gen -o gen_friend.root
    python3 add_hammer_weights_rds.py rds_signal.root --card card.json -o /dev/null --closure

The output is a friend tree, row-aligned 1:1 with the input tree. With the
weights of tree_gen (all generated events of an unfiltered sample) the
per-channel mean weight gives the normalisation correction of AN Sec. 8.2.
'''
import argparse
import math
import sys

import numpy as np
import uproot

from Bmmm.Analysis.HammerDsFF import (
    BRANCH_NAMES, BRANCH_NOMINAL, BRANCH_STATUS, INPUT_BRANCHES, STATUS_OK, CHANNELS,
    leaves_from_row, make_hammer_session,
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-t', '--tree', default='tree')
    parser.add_argument('--out-tree', default='hammer')
    parser.add_argument('--card', required=True,
                        help="card JSON; append ':nominal' for the central weight only, "
                             "':allow-stale' for a not-yet-validated covariance")
    parser.add_argument('--chunk', type=int, default=50000)
    parser.add_argument('-n', '--max-events', type=int, default=None)
    parser.add_argument('--closure', action='store_true',
                        help='the input already has hammer_* branches: recompute and '
                             'compare instead of writing a friend tree')
    args = parser.parse_args()

    with uproot.open('%s:%s' % (args.inputs[0], args.tree)) as tin:
        have = set(tin.keys())
    missing = [b for b in INPUT_BRANCHES if b not in have]
    if missing:
        sys.exit('[FATAL] input has no %s' % missing)
    read = list(INPUT_BRANCHES) + (list(BRANCH_NAMES) if args.closure else [])

    ham = make_hammer_session(args.card)
    cols = dict((b, []) for b in BRANCH_NAMES)
    worst, sumw, nfin = {}, {}, {}
    nrow = 0
    src = ['%s:%s' % (p, args.tree) for p in args.inputs]
    for row in uproot.iterate(src, expressions=read, step_size=args.chunk, library='np'):
        for i in range(len(row['gen_sig'])):
            if args.max_events is not None and nrow >= args.max_events:
                break
            nrow += 1
            out = ham.weights(leaves_from_row(row, i))
            for b in BRANCH_NAMES:
                cols[b].append(out[b])
            if out[BRANCH_STATUS] == STATUS_OK and math.isfinite(out[BRANCH_NOMINAL]):
                code = int(round(float(row['gen_sig'][i])))
                sumw[code] = sumw.get(code, 0.) + out[BRANCH_NOMINAL]
                nfin[code] = nfin.get(code, 0) + 1
            if args.closure:
                for b in BRANCH_NAMES:
                    new, ref = out[b], float(row[b][i])
                    if not (math.isnan(new) and math.isnan(ref)):
                        worst[b] = max(worst.get(b, 0.), abs(new - ref))

    print('[ok] rows=%d' % nrow)
    for code in sorted(nfin):
        print('[info] channel %d (%s, %s): <w> = %.4f over %d events'
              % (code, CHANNELS[code][0], CHANNELS[code][1], sumw[code] / nfin[code], nfin[code]))
    print(ham.summary())

    if args.closure:
        bad = dict((k, v) for k, v in worst.items() if v != 0.)
        if bad:
            sys.exit('[closure][FAIL] inline and post-hoc disagree: %s' % bad)
        print('[closure] inline == post-hoc on %d rows' % nrow)
        return

    data = dict((b, np.asarray(v, dtype=np.float64)) for b, v in cols.items())
    with uproot.recreate(args.output) as fout:
        fout.mktree(args.out_tree, dict((b, 'float64') for b in BRANCH_NAMES))
        fout[args.out_tree].extend(data)
    print('[ok] wrote %d rows -> %s:%s' % (nrow, args.output, args.out_tree))


if __name__ == '__main__':
    main()
