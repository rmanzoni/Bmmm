#!/usr/bin/env python3
'''
Read the output of a --test production and decide whether to launch the rest.

    python3 covariance/check_production_weights.py <out_dir>/rjpsi_chunk*.root
    python3 covariance/check_production_weights.py file.root --tree tree

Checks, per channel (gen_bc_decay 1 = J/psi mu nu, 7 = J/psi tau nu):

  1. hammer_status   : share of signal rows with a valid weight; declined (3) or
                       non-finite (4) above a few per mille is a problem
  2. hammer_weight   : finite, positive, mean printed. For reference, the MC-level
                       mean Kiselev -> Harrison weight in the mu channel was ~0.5 in
                       the standalone study (this is per selected CANDIDATE, so
                       selection shifts it: a sanity scale, not a closure)
  3. variations      : each of the 14 non-degenerate directions must move the weight;
                       direction 14 (null for this card) must not. All variations
                       equal to nominal = the eigenvector plumbing did not take.
  4. non-signal rows : hammer_* must be NaN there
  5. lifetime        : gen_bc_ctau_weight* present, finite, mean close to 1, and
                       up/down on opposite sides of nominal

Exit code 1 if any HARD check fails, so it can gate a submission script.
'''
import argparse
import sys

import numpy as np
import uproot

N_DIR = 15
NULL_DIRS = {14}          # pinned by the regulariser for the current card
SIGNAL = {1: 'mu', 7: 'tau'}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--tree', default='tree')
    args = ap.parse_args()

    var = ['hammer_ff_ev%02d_%s' % (j, t) for j in range(N_DIR) for t in ('up', 'dn')]
    need = (['gen_bc_decay', 'hammer_weight', 'hammer_status',
             'gen_bc_ctau_weight', 'gen_bc_ctau_weight_up', 'gen_bc_ctau_weight_down']
            + var)

    with uproot.open('%s:%s' % (args.inputs[0], args.tree)) as t:
        missing = [b for b in need if b not in t.keys()]
    if missing:
        sys.exit('[FAIL] branches missing from the ntuple: %s\n'
                 '       (was the production run from a checkout with the Hammer '
                 'and lifetime patches?)' % missing[:6])

    arr = {}
    for chunk in uproot.iterate(['%s:%s' % (f, args.tree) for f in args.inputs],
                                expressions=need, library='np', step_size='200 MB'):
        for k, v in chunk.items():
            arr.setdefault(k, []).append(np.asarray(v, dtype=float))
    arr = dict((k, np.concatenate(v)) for k, v in arr.items())
    code = np.round(np.nan_to_num(arr['gen_bc_decay'], nan=-1)).astype(int)
    print('rows: %d from %d file(s)' % (code.size, len(args.inputs)))

    hard_fail = []

    # ---- 1-3: signal channels -------------------------------------------------
    for c, name in SIGNAL.items():
        sel = code == c
        n = int(sel.sum())
        print('\n[%s] %d signal rows' % (name, n))
        if not n:
            print('    none in this sample')
            continue
        st = arr['hammer_status'][sel]
        for s, lab in ((0, 'ok'), (2, 'gen leaves missing'), (3, 'declined by Hammer'),
                       (4, 'non-finite')):
            k = int(np.sum(st == s))
            if k:
                print('    status %d (%s): %d  (%.3f%%)' % (s, lab, k, 100. * k / n))
        if np.all(np.isnan(st)):
            hard_fail.append('%s: hammer_status all NaN -- Hammer did not run' % name)
            continue
        bad = np.sum((st == 3) | (st == 4)) / float(n)
        if bad > 5e-3:
            hard_fail.append('%s: %.2f%% of signal rows declined or non-finite'
                             % (name, 100 * bad))

        w = arr['hammer_weight'][sel & (arr['hammer_status'] == 0)]
        w = w[np.isfinite(w)]
        if w.size == 0:
            hard_fail.append('%s: no finite nominal weight' % name)
            continue
        print('    hammer_weight: mean %.4f  median %.4f  min %.3g  max %.3g'
              % (w.mean(), np.median(w), w.min(), w.max()))
        if np.any(w <= 0):
            hard_fail.append('%s: non-positive nominal weights' % name)

        ok = sel & (arr['hammer_status'] == 0)
        nom = arr['hammer_weight'][ok]
        moved, frozen = [], []
        for j in range(N_DIR):
            up = arr['hammer_ff_ev%02d_up' % j][ok]
            dn = arr['hammer_ff_ev%02d_dn' % j][ok]
            shift = np.nanmean(np.abs(up - nom) / np.maximum(np.abs(nom), 1e-12))
            asym = np.nanmean((up + dn) / 2. - nom) / max(np.nanmean(nom), 1e-12)
            (moved if shift > 1e-9 else frozen).append(j)
            if j < 3 or j in NULL_DIRS:
                print('    ev%02d: <|up-nom|/nom> %.2e   <(up+dn)/2-nom>/<nom> %+.1e'
                      % (j, shift, asym))
        print('    directions that move the weight: %d of %d  (expected %d)'
              % (len(moved), N_DIR, N_DIR - len(NULL_DIRS)))
        should_move = [j for j in range(N_DIR) if j not in NULL_DIRS]
        if not moved:
            hard_fail.append('%s: NO variation moves the weight -- the eigenvector '
                             'plumbing did not take' % name)
        elif set(should_move) - set(moved):
            hard_fail.append('%s: directions %s should move the weight and do not'
                             % (name, sorted(set(should_move) - set(moved))))

    # ---- 4: non-signal rows ----------------------------------------------------
    other = ~np.isin(code, list(SIGNAL))
    if other.any():
        leak = np.isfinite(arr['hammer_weight'][other]).sum()
        print('\nnon-signal rows: %d, with a finite hammer_weight: %d' % (other.sum(), leak))
        if leak:
            hard_fail.append('%d non-signal rows carry a Hammer weight' % leak)

    # ---- 5: lifetime -------------------------------------------------------------
    has_bc = np.isfinite(arr['gen_bc_ctau_weight'])
    print('\nlifetime weights on %d rows with a gen Bc' % has_bc.sum())
    if has_bc.any():
        n0 = arr['gen_bc_ctau_weight'][has_bc]
        nu = arr['gen_bc_ctau_weight_up'][has_bc]
        nd = arr['gen_bc_ctau_weight_down'][has_bc]
        print('    mean nominal %.4f  up %.4f  down %.4f' % (n0.mean(), nu.mean(), nd.mean()))
        if not (0.95 < n0.mean() < 1.05):
            hard_fail.append('ctau nominal weight mean %.3f far from 1' % n0.mean())
        if not ((nu.mean() - n0.mean()) * (nd.mean() - n0.mean()) < 0):
            print('    [WARN] up and down are not on opposite sides of nominal on '
                  'average (can happen after selection; check the lifetime shape)')
    else:
        hard_fail.append('no row carries a lifetime weight')

    print()
    if hard_fail:
        for msg in hard_fail:
            print('[FAIL] ' + msg)
        sys.exit(1)
    print('[OK] all hard checks passed -- safe to launch the full production')


if __name__ == '__main__':
    main()
