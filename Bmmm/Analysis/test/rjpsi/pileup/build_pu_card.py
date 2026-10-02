#!/usr/bin/env python3
'''
Build the pileup card read by Bmmm.Analysis.PileupWeights.

Run on lxplus (or anywhere /eos/user/c/cmsdqm is mounted) after `cmsenv`, since
it needs the central files on /eos, pileupCalc.py, and the MixingModule cfis:

    python3 build_pu_card.py -o $CMSSW_BASE/src/Bmmm/Analysis/data/pu_weights_run3.json \
                             --plots pu_card_plots

    # only some campaigns, e.g. while the 2025/2026 inputs are not there yet
    python3 build_pu_card.py -o card.json --campaigns Summer22 Summer22EE Summer23 Summer23BPix

The card stores the INPUTS (data histograms, MC generation profiles) and where
they came from (path, sha256, cross section, pileupCalc command); the weights
are formed in PileupWeights.weight_table and only reported here.
'''
import argparse
import datetime
import glob
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import threading
import time

import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pu_config_run3 as cfg                                          # noqa: E402

from Bmmm.Analysis.PileupWeights import (                             # noqa: E402
    SCHEMA_VERSION, VARIATIONS, branch_name, load_card, weight_table,
)


def fatal(msg):
    sys.exit('[FATAL] ' + msg)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class Heartbeat(object):
    '''Prints the elapsed time every `every` seconds while a slow step runs, so
    a silent pileupCalc is never mistaken for a hung one.'''

    def __init__(self, label, every=30.):
        self.label, self.every = label, every
        self._stop = threading.Event()

    def _run(self):
        t0 = time.time()
        while not self._stop.wait(self.every):
            print('        ... %s still running, %.0f s' % (self.label, time.time() - t0),
                  flush=True)

    def __enter__(self):
        self._t0 = time.time()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._th.join()
        print('        done in %.0f s' % (time.time() - self._t0), flush=True)


# ---------------------------------------------------------------------------
# MC generation profile, straight from the MixingModule cfi
# ---------------------------------------------------------------------------
def mc_profile(cfi):
    try:
        mod = importlib.import_module('SimGeneral.MixingModule.' + cfi)
    except ImportError as e:
        fatal('cannot import SimGeneral.MixingModule.%s (%s) -- run after cmsenv' % (cfi, e))
    nb   = mod.mix.input.nbPileupEvents
    var  = list(nb.probFunctionVariable)
    prob = np.asarray(list(nb.probValue), dtype=np.float64)
    if var != list(range(len(var))) or len(prob) != len(var):
        fatal('%s: probFunctionVariable is not 0..N-1 or does not match probValue' % cfi)
    if np.any(prob < 0) or abs(prob.sum() - 1.) > 1e-4:
        fatal('%s: probValue is not a normalised probability (sum %.6f)' % (cfi, prob.sum()))
    # a shorter profile (e.g. the 99-bin UL ones) has zero probability in the
    # missing bins: pad. A longer one is only acceptable if it is empty beyond.
    if len(prob) > cfg.NBINS:
        if prob[cfg.NBINS:].sum() > 0:
            fatal('%s has probability beyond pileup %d, the card grid' % (cfi, cfg.NBINS))
        prob = prob[:cfg.NBINS]
    prob = np.pad(prob, (0, cfg.NBINS - len(prob)))
    centres = np.arange(cfg.NBINS) + 0.5
    return {
        'cfi'   : 'SimGeneral.MixingModule.' + cfi,
        'cmssw' : os.environ.get('CMSSW_VERSION', '?'),
        'prob'  : (prob / prob.sum()).tolist(),
        'mean'  : float(np.sum(centres * prob) / prob.sum()),
    }


# ---------------------------------------------------------------------------
# data histograms
# ---------------------------------------------------------------------------
def to_unit_grid(path, hname='pileup'):
    '''Read a pileupCalc-style histogram and put it on the card grid, [0, NBINS)
    in unit bins. Anything outside the grid (including ROOT under/overflow) is
    returned separately so it still counts in the normalisation.'''
    with uproot.open(path) as f:
        if hname not in f:
            fatal('%s has no histogram %r (keys: %s)' % (path, hname, f.keys()))
        h = f[hname]
        vals  = h.values(flow=True)
        edges = h.axis().edges()
    under, content, over = float(vals[0]), vals[1:-1], float(vals[-1])
    if not (np.allclose(edges, np.round(edges)) and np.allclose(np.diff(edges), 1.)):
        fatal('%s: binning is not unit-width on integer edges (%g..%g, %d bins); '
              'rerun pileupCalc with --maxPileupBin %d --numPileupBins %d'
              % (path, edges[0], edges[-1], len(edges) - 1, cfg.NBINS, cfg.NBINS))
    grid, outside = np.zeros(cfg.NBINS), under + over
    lo = int(round(edges[0]))
    for i, c in enumerate(content):
        k = lo + i
        if 0 <= k < cfg.NBINS:
            grid[k] += c
        else:
            outside += c
    if np.any(grid < 0) or grid.sum() <= 0:
        fatal('%s: empty or negative histogram' % path)
    return grid, outside, (float(edges[0]), float(edges[-1]), len(edges) - 1)


def find_central(directory, xsec):
    files = sorted(glob.glob(os.path.join(directory, '*.root')))
    if not files:
        fatal('no ROOT files in %s' % directory)
    match = [f for f in files if re.search(r'(?<!\d)%dub' % xsec, os.path.basename(f))]
    if len(match) != 1:
        fatal('%s: %d files match %d ub: %s\n        set files={...} for this period '
              'in pu_config_run3.py. All files there:\n          %s'
              % (directory, len(match), xsec, match, '\n          '.join(files)))
    return match[0]


# TWiki: "if this error only pops up once or twice, you can safely ignore it"
MAX_WARN_ABS = 5


def count_lumisections(golden):
    with open(golden) as f:
        js = json.load(f)
    return sum(hi - lo + 1 for ranges in js.values() for lo, hi in ranges)


def run_pileupcalc(golden, pileup_json, xsec, out, logfile):
    cmd = ['pileupCalc.py', '-i', golden, '--inputLumiJSON', pileup_json,
           '--calcMode', 'true', '--minBiasXsec', str(xsec),
           '--maxPileupBin', str(cfg.NBINS), '--numPileupBins', str(cfg.NBINS), out]
    with Heartbeat('pileupCalc %d ub' % xsec):
        with open(logfile, 'w') as log:
            ret = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    text = open(logfile).read()
    if ret.returncode != 0 or not os.path.isfile(out):
        fatal('pileupCalc failed (exit %d), see %s\n        %s'
              % (ret.returncode, logfile, ' '.join(cmd)))
    return cmd, {
        'ls_not_found'     : len(re.findall(r'not found in Lumi/Pileup input file', text)),
        'outside_histogram': len(re.findall(r'Significant probability density outside', text)),
    }


def build_period(name, per, workdir, max_warn_frac):
    print('\n==> data period %s (year %s, %s)' % (name, per['year'], per['kind']))
    out = {'year': per['year'], 'hist': {}, 'outside': {}, 'inputs': {},
           'source': {'kind': per['kind']}}

    if per['kind'] == 'central':
        files = per.get('files') or {v: find_central(per['dir'], cfg.XSEC_UB[v]) for v in VARIATIONS}
        out['source']['dir'] = per.get('dir')
        for var in VARIATIONS:
            path = files[var]
            grid, outside, binning = to_unit_grid(path)
            out['hist'][var], out['outside'][var] = grid.tolist(), outside
            out['inputs'][var] = {'path': path, 'sha256': sha256(path),
                                  'xsec_ub': cfg.XSEC_UB[var], 'binning': binning}
            print('    %-8s %s' % (var, path))

    elif per['kind'] == 'pileupCalc':
        for key in ('golden_json', 'pileup_json'):
            if not per.get(key):
                fatal('period %s: %s is not set in pu_config_run3.py' % (name, key))
            if not os.path.isfile(per[key]):
                fatal('period %s: %s %s does not exist' % (name, key, per[key]))
        n_ls = count_lumisections(per['golden_json'])
        out['source'].update(golden_json=per['golden_json'], pileup_json=per['pileup_json'],
                             golden_sha256=sha256(per['golden_json']),
                             pileup_sha256=sha256(per['pileup_json']),
                             n_lumisections=n_ls)
        print('    golden JSON %s (%d lumisections)' % (per['golden_json'], n_ls))
        print('    pileup JSON %s' % per['pileup_json'])
        for i, var in enumerate(VARIATIONS, 1):
            xsec = cfg.XSEC_UB[var]
            root = os.path.join(workdir, 'pileup_%s_%dub.root' % (name, xsec))
            log  = root.replace('.root', '.log')
            print('    [%d/%d] pileupCalc at %d ub -> %s' % (i, len(VARIATIONS), xsec, root),
                  flush=True)
            cmd, warn = run_pileupcalc(per['golden_json'], per['pileup_json'], xsec, root, log)
            # TWiki: a few of these are harmless, many mean a wrong JSON pairing
            # or a histogram range that is too short.
            for kind, n in warn.items():
                if n:
                    print('        %d "%s" warnings (%.3f%% of lumisections)'
                          % (n, kind, 100. * n / n_ls))
                if n > max(MAX_WARN_ABS, max_warn_frac * n_ls):
                    fatal('period %s: %d "%s" warnings out of %d lumisections, see %s'
                          % (name, n, kind, n_ls, log))
            grid, outside, binning = to_unit_grid(root)
            out['hist'][var], out['outside'][var] = grid.tolist(), outside
            out['inputs'][var] = {'path': root, 'sha256': sha256(root), 'xsec_ub': xsec,
                                  'binning': binning, 'command': ' '.join(cmd),
                                  'warnings': warn}
    else:
        fatal('period %s: unknown kind %r' % (name, per['kind']))
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def report(card, max_uncovered, allow_uncovered):
    print('\n' + '=' * 100)
    print('What the numbers below mean, for each MC campaign and data year:')
    print('  data covered : fraction of the DATA pileup distribution that falls in pileup values')
    print('                 the MC was generated with. 1 means every data event has MC counterparts;')
    print('                 it is also the average weight over the full generated MC sample.')
    print('  effective MC : statistical power left after weighting, as a fraction of the raw')
    print('                 MC size ((sum of weights)^2 / sum of squared weights, per event).')
    print('  max weight   : the largest weight any MC event can get.')
    print('=' * 100)
    problems = []
    for cname, camp in card['campaigns'].items():
        prof = card['mc_profiles'][camp['mc_profile']]
        print('\n%s  (MC profile %s, mean %.1f%s)'
              % (cname, camp['mc_profile'], prof['mean'],
                 '' if camp['confirmed'] else ', NOT YET CONFIRMED'))
        for year, period in sorted(camp['periods'].items()):
            per = card['data'][period]
            for var in VARIATIONS:
                w, info = weight_table(per['hist'][var], per['outside'][var], prof['prob'])
                d = np.asarray(per['hist'][var]); d = d / d.sum()
                dmean = float(np.sum((np.arange(card['nbins']) + 0.5) * d))
                print('    %-22s data %-8s mean %.1f   data covered %.5f   '
                      'effective MC %.3f   max weight %7.2f'
                      % (branch_name(year, var), period, dmean, info['data_covered'],
                         info['neff_frac'], info['w_max']))
                if info['data_covered'] < 1. - max_uncovered:
                    problems.append('%s %s: %.3f%% of the data pileup distribution is in '
                                    'bins the MC never populates'
                                    % (cname, branch_name(year, var),
                                       100. * (1. - info['data_covered'])))
    if problems:
        print('\n' + '\n'.join('[WARNING] ' + p for p in problems))
        if not allow_uncovered:
            fatal('data outside the MC pileup range above %.2f%% (see above). Pass '
                  '--allow-uncovered to write the card anyway; those events are then '
                  'simply missing from the reweighted MC.' % (100. * max_uncovered))


def make_plots(card, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    x = np.arange(card['nbins'] + 1)
    colors = {'nominal': '#3f90da', 'up': '#bd1f01', 'down': '#94a4a2'}
    for cname, camp in card['campaigns'].items():
        prof = card['mc_profiles'][camp['mc_profile']]
        for year, period in sorted(camp['periods'].items()):
            per = card['data'][period]
            fig, (ax, rx) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                         gridspec_kw={'height_ratios': [2, 1]})
            ax.stairs(prof['prob'], x, color='black', lw=2,
                      label='MC generation profile\n%s' % camp['mc_profile'])
            for var in VARIATIONS:
                h = np.asarray(per['hist'][var])
                ax.stairs(h / (h.sum() + per['outside'][var]), x, color=colors[var],
                          label='data %s, %s (%d ub)'
                          % (period, var, per['inputs'][var]['xsec_ub']))
                w, _ = weight_table(h, per['outside'][var], prof['prob'])
                rx.stairs(w, x, color=colors[var], label=var)      # NaN (no MC) left as a gap
            ax.set_ylabel('fraction per unit of true pileup')
            ax.set_title('%s  ->  pu_weight_%s' % (cname, year))
            ax.legend(fontsize=8)
            rx.axhline(1., color='black', lw=0.5)
            rx.set_ylabel('weight = data / MC')
            rx.set_xlabel('true pileup (nti)')
            rx.set_ylim(0, min(5., 1.2 * max(np.nanmax(weight_table(
                per['hist'][v], per['outside'][v], prof['prob'])[0]) for v in VARIATIONS)))
            fig.tight_layout()
            for ext in ('png', 'pdf'):
                fig.savefig(os.path.join(outdir, 'pu_%s_%s.%s' % (cname, year, ext)))
            plt.close(fig)
    print('plots in %s' % outdir)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-o', '--output', required=True)
    p.add_argument('--campaigns', nargs='+', default=sorted(cfg.CAMPAIGNS))
    p.add_argument('--workdir', default='pu_card_work',
                   help='where pileupCalc outputs and logs go (kept for provenance)')
    p.add_argument('--plots', default='', help='directory for data/MC/weight plots')
    p.add_argument('--max-uncovered', type=float, default=1e-3,
                   help='largest tolerated fraction of data outside the MC range')
    p.add_argument('--allow-uncovered', action='store_true')
    p.add_argument('--max-warn-frac', type=float, default=1e-3,
                   help='largest tolerated fraction of lumisections with pileupCalc '
                        'warnings (at least %d are always tolerated)' % MAX_WARN_ABS)
    args = p.parse_args()

    unknown = set(args.campaigns) - set(cfg.CAMPAIGNS)
    if unknown:
        fatal('unknown campaign(s) %s; known: %s' % (sorted(unknown), sorted(cfg.CAMPAIGNS)))
    campaigns = {c: cfg.CAMPAIGNS[c] for c in args.campaigns}
    periods   = sorted(set(p for c in campaigns.values() for p in c['periods'].values()))
    os.makedirs(args.workdir, exist_ok=True)

    card = {
        'schema'  : SCHEMA_VERSION,
        'created' : datetime.datetime.now().isoformat(timespec='seconds'),
        'cmssw'   : os.environ.get('CMSSW_VERSION', '?'),
        'nbins'   : cfg.NBINS,
        'binning' : 'unit bins [k, k+1), k = 0..nbins-1; bin of an event = floor(nti)',
        'xsec_ub' : cfg.XSEC_UB,
        'mc_profiles': {}, 'data': {}, 'campaigns': {},
    }

    print('==> MC generation profiles')
    for c in campaigns.values():
        if c['mc_cfi'] not in card['mc_profiles']:
            card['mc_profiles'][c['mc_cfi']] = mc_profile(c['mc_cfi'])
            print('    %-55s mean %.2f' % (c['mc_cfi'], card['mc_profiles'][c['mc_cfi']]['mean']))

    for name in periods:
        card['data'][name] = build_period(name, cfg.PERIODS[name], args.workdir,
                                          args.max_warn_frac)

    for cname, c in campaigns.items():
        card['campaigns'][cname] = {'mc_profile': c['mc_cfi'], 'periods': dict(c['periods']),
                                    'confirmed': bool(c['confirmed']), 'gt': c.get('gt')}

    report(card, args.max_uncovered, args.allow_uncovered)

    tmp = args.output + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(card, f, indent=1)
    load_card(tmp)                         # the reader must accept what we wrote
    os.replace(tmp, args.output)
    print('\ncard written to %s' % args.output)

    if args.plots:
        make_plots(card, args.plots)


if __name__ == '__main__':
    main()
