#!/usr/bin/env python3
'''
Add the Bc form-factor weights to ntuples that were produced WITHOUT --hammer.

Reads the gen_b_* / gen_ham_* / gen_bc_decay leaves the ntuplizer already writes
and calls the same Bmmm.Analysis.HammerFF session the inspector calls, so the
result is identical to a production-time run -- see --closure, which proves it
on a file that has both.

    python3 add_hammer_weights.py bc_signal.root -o hammer_friend.root
    python3 add_hammer_weights.py bc_signal.root -o /dev/null --closure

The output is a friend tree, row-aligned 1:1 with the input tree.
'''
import argparse
import math
import sys

import numpy as np
import uproot

from Bmmm.Analysis.HammerFF import (
    BRANCH_NAMES, BRANCH_NOMINAL, BRANCH_STATUS, INPUT_BRANCHES, STATUS_OK,
    leaves_from_row, make_hammer_session,
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-t', '--tree', default='tree')
    parser.add_argument('--out-tree', default='hammer')
    parser.add_argument('--card', default='default',
                        help="FF card: 'default', or a path; append ':nominal' "
                             "for the central weight only, ':allow-stale' to use "
                             'a not-yet-validated covariance.')
    parser.add_argument('--chunk', type=int, default=50000)
    parser.add_argument('-n', '--max-events', type=int, default=None)
    parser.add_argument('--max-signal', type=int, default=None,
                        help='stop after N reweighted signal events (smoke test)')
    parser.add_argument('--closure', action='store_true',
                        help='the input already carries hammer_* branches: '
                             'recompute and compare instead of writing a friend. '
                             'Any non-zero difference is a bug, not a tolerance.')
    args = parser.parse_args()

    with uproot.open('%s:%s' % (args.inputs[0], args.tree)) as tin:
        have = set(tin.keys())
    missing = [b for b in INPUT_BRANCHES if b not in have]
    if missing:
        sys.exit('[FATAL] input has no %s -- the ntuple predates the Hammer gen '
                 'leaves and cannot be reweighted without reprocessing.' % missing)
    read = list(INPUT_BRANCHES)
    if args.closure:
        ref_missing = [b for b in BRANCH_NAMES if b not in have]
        if ref_missing:
            sys.exit('[FATAL] --closure needs the reference branches: missing %s'
                     % ref_missing)
        read += list(BRANCH_NAMES)

    ham = make_hammer_session(args.card)
    if ham is None:
        sys.exit('[FATAL] --card resolved to nothing.')

    cols = dict((b, []) for b in BRANCH_NAMES)
    worst = {}
    nrow = nsig = 0
    sumw = {'mu': 0., 'tau': 0.}
    nfin = {'mu': 0, 'tau': 0}

    src = ['%s:%s' % (p, args.tree) for p in args.inputs]
    step = args.chunk if args.max_events is None else min(args.chunk, args.max_events)

    stop = False
    for row in uproot.iterate(src, expressions=read, step_size=step, library='np'):
        for i in range(len(row['gen_bc_decay'])):
            if ((args.max_events is not None and nrow >= args.max_events) or
                    (args.max_signal is not None and nsig >= args.max_signal)):
                stop = True
                break
            nrow += 1
            out = ham.weights(leaves_from_row(row, i))
            for branch in BRANCH_NAMES:
                cols[branch].append(out[branch])

            if out[BRANCH_STATUS] == STATUS_OK:
                nsig += 1
                code = int(round(float(row['gen_bc_decay'][i])))
                key = 'mu' if code == 1 else 'tau'
                if math.isfinite(out[BRANCH_NOMINAL]):
                    sumw[key] += out[BRANCH_NOMINAL]
                    nfin[key] += 1

            if args.closure:
                for branch in BRANCH_NAMES:
                    new, ref = out[branch], float(row[branch][i])
                    if math.isnan(new) and math.isnan(ref):
                        continue
                    worst[branch] = max(worst.get(branch, 0.), abs(new - ref))
        if stop:
            break

    print('[ok] rows=%d  reweighted=%d' % (nrow, nsig))
    for key in ('mu', 'tau'):
        if nfin[key]:
            print('[closure] <w>_%-3s = %.4f  (finite %d)'
                  % (key, sumw[key] / nfin[key], nfin[key]))
    if nfin['mu'] and nfin['tau']:
        print('[closure] R(J/psi)_Harrison / R(J/psi)_Kiselev = %.4f  (target R = 0.2597)'
              % ((sumw['tau'] / nfin['tau']) / (sumw['mu'] / nfin['mu'])))
    print(ham.summary())

    if args.closure:
        bad = {k: v for k, v in worst.items() if v != 0.}
        if bad:
            print('[closure][FAIL] inline and post-hoc disagree: %s' % bad, file=sys.stderr)
            sys.exit(1)
        print('[closure] inline == post-hoc, bit for bit, on %d rows x %d branches'
              % (nrow, len(BRANCH_NAMES)))
        return

    data = dict((b, np.asarray(v, dtype=np.float64)) for b, v in cols.items())
    with uproot.recreate(args.output) as fout:
        fout.mktree(args.out_tree, dict((b, 'float64') for b in BRANCH_NAMES))
        fout[args.out_tree].extend(data)
    print('[ok] wrote %d rows x %d branches -> %s:%s'
          % (nrow, len(BRANCH_NAMES), args.output, args.out_tree))


if __name__ == '__main__':
    main()
