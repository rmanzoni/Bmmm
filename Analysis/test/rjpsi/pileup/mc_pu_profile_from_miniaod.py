#!/usr/bin/env python3
'''
Cross-check (option b) of the MC pileup profiles used in the card: histogram the
true pileup of the UNSKIMMED MINIAODSIM and compare it with the MixingModule
profiles. This is also how a campaign gets `confirmed=True` in pu_config_run3.py.

Why unskimmed MINIAODSIM: the pileup overlay happens after generation, so the
GEN-level filter of the sample does not bias nti; the skim and the ntuple
selection (vertexing, isolation) do.

A few thousand events per campaign are enough to tell the candidate profiles
apart (their means range from ~37 to ~67). After cmsenv:

    # 1) measure, one campaign at a time (file list = one LFN or PFN per line)
    python3 mc_pu_profile_from_miniaod.py measure --files files_bc_Summer24_miniaod.txt \
            --label Summer24 --max-events 20000 -o nti_Summer24.json

    # 2) rank every Run 3 profile of this CMSSW release against the measurement,
    #    and show where the one configured for the campaign ends up
    python3 mc_pu_profile_from_miniaod.py compare nti_Summer24.json --campaign Summer24 \
            --plot nti_Summer24.png
'''
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pu_config_run3 as cfg                                          # noqa: E402

REDIRECTOR = 'root://cms-xrd-global.cern.ch//'


def measure(args):
    from DataFormats.FWLite import Events, Handle

    with open(args.files) as f:
        files = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    files = [(args.redirector + x) if x.startswith('/store') else x for x in files]
    if args.max_files > 0:
        files = files[:args.max_files]

    handle = Handle('std::vector<PileupSummaryInfo>')
    counts = np.zeros(cfg.NBINS)
    outside = n = 0
    t0 = time.time()
    events = Events(files)
    total = events.size() if args.max_events < 0 else min(args.max_events, events.size())
    for i, ev in enumerate(events):
        if i >= total:
            break
        ev.getByLabel('slimmedAddPileupInfo', handle)
        nti = [p.getTrueNumInteractions() for p in handle.product() if p.getBunchCrossing() == 0][0]
        k = int(np.floor(np.float32(nti)))
        if 0 <= k < cfg.NBINS:
            counts[k] += 1
        else:
            outside += 1
        n += 1
        if n % 1000 == 0 or n == total:
            rate = n / max(time.time() - t0, 1e-9)
            sys.stderr.write('\r  %d/%d events  %.0f ev/s  ETA %.0f s   '
                             % (n, total, rate, (total - n) / max(rate, 1e-9)))
    sys.stderr.write('\n')
    out = {'label': args.label, 'files': files, 'n_events': n,
           'counts': counts.tolist(), 'outside': outside,
           'binning': 'unit bins [k, k+1), k = 0..%d; bin = floor(nti)' % (cfg.NBINS - 1)}
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=1)
    print('%d events, mean nti bin centre %.2f -> %s'
          % (n, np.sum((np.arange(cfg.NBINS) + 0.5) * counts) / max(counts.sum(), 1), args.output))


def candidate_cfis():
    """Every Run 3 profile cfi importable in this environment.

    Looks where Python itself would import SimGeneral.MixingModule from (the
    package __path__), plus the src/ areas of the local and release checkouts:
    in a release, python/SimGeneral/MixingModule/ holds only the package
    __init__, the cfis themselves sit in src/SimGeneral/MixingModule/python/.
    Names are merged; which copy is actually read is decided by the import in
    mc_profile(), i.e. the usual CMSSW order (local checkout first).
    """
    dirs = []
    try:
        import SimGeneral.MixingModule as pkg
        dirs += list(pkg.__path__)
    except ImportError:
        pass
    for base in ('CMSSW_BASE', 'CMSSW_RELEASE_BASE'):
        root = os.environ.get(base, '')
        if root:
            dirs.append(os.path.join(root, 'src', 'SimGeneral', 'MixingModule', 'python'))
            dirs.append(os.path.join(root, 'python', 'SimGeneral', 'MixingModule'))
    names = set()
    for d in dirs:
        for pat in ('Run3_*_cfi.py', 'mix_202[2-9]*_cfi.py', 'mix_Run3*_cfi.py'):
            names.update(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(d, pat)))
    return sorted(names), dirs


def _bracketed_numbers(text, key):
    """Every list of numbers assigned to `key` in `text`, whatever the syntax:
    cms.vdouble(...) in a cfi or cmsDriver cfg, {...} in an edmProvDump
    parameter-set dump, [...] in JSON."""
    out = []
    for m in re.finditer(r'\b%s\b[^({\[]*?[({\[]([^)}\]]*)[)}\]]' % key, text):
        nums = re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', m.group(1))
        if nums:
            out.append([float(x) for x in nums])
    return out


def profile_from_file(path):
    """A pileup profile from a file instead of an importable cfi.

    Accepts any text holding exactly one probFunctionVariable / probValue pair
    (a cfi, a cmsDriver cfg with --customise_commands, the edmProvDump of a
    PREMIX library file), or a JSON with a "prob" list (bin k = [k, k+1)).
    Shorter profiles are padded with zeros to the NBINS grid; a longer one with
    probability beyond it is refused.
    """
    text = open(path).read()
    if path.endswith('.json'):
        prob = np.asarray(json.loads(text)['prob'], dtype=np.float64)
    else:
        variables = {tuple(v) for v in _bracketed_numbers(text, 'probFunctionVariable')}
        values    = {tuple(v) for v in _bracketed_numbers(text, 'probValue')}
        if len(variables) != 1 or len(values) != 1:
            raise SystemExit('%s: found %d distinct probFunctionVariable and %d distinct probValue '
                             'lists, need exactly one of each' % (path, len(variables), len(values)))
        var, prob = np.asarray(variables.pop()), np.asarray(values.pop())
        if len(var) != len(prob) or np.any(var != np.arange(len(var))):
            raise SystemExit('%s: probFunctionVariable is not 0..N-1 with one probValue each' % path)
    if np.any(prob < 0) or prob.sum() <= 0:
        raise SystemExit('%s: not a probability list' % path)
    if len(prob) > cfg.NBINS:
        if prob[cfg.NBINS:].sum() > 0:
            raise SystemExit('%s: profile has probability beyond %d' % (path, cfg.NBINS))
        prob = prob[:cfg.NBINS]
    prob = np.pad(prob, (0, cfg.NBINS - len(prob)))
    centres = np.arange(cfg.NBINS) + 0.5
    return {'prob': (prob / prob.sum()).tolist(),
            'mean': float(np.sum(centres * prob) / prob.sum())}


def chi2(counts, prob):
    '''Pearson chi2 of the measured counts against N * prob, with every bin
    expecting fewer than 5 events merged into one, so sparse tails do not
    dominate. Returns (chi2, ndf).'''
    counts, prob = np.asarray(counts, float), np.asarray(prob, float)
    exp = counts.sum() * prob / prob.sum()
    big = exp >= 5
    o = np.append(counts[big], counts[~big].sum())
    e = np.append(exp[big], exp[~big].sum())
    keep = e > 0
    return float(np.sum((o[keep] - e[keep]) ** 2 / e[keep])), int(keep.sum() - 1)


def compare(args):
    from build_pu_card import mc_profile    # same cfi reader as the card builder

    with open(args.measurement) as f:
        meas = json.load(f)
    counts = np.asarray(meas['counts'])
    centres = np.arange(cfg.NBINS) + 0.5
    print('measurement %s: %d events, mean %.2f, %d outside [0, %d)'
          % (meas['label'], meas['n_events'], np.sum(centres * counts) / counts.sum(),
             meas['outside'], cfg.NBINS))

    configured = cfg.CAMPAIGNS[args.campaign]['mc_cfi'] if args.campaign else None
    if args.candidates:
        names = list(args.candidates)
    else:
        names, searched = candidate_cfis()
        # a ranking of one is not a check: with no alternatives the configured
        # profile would always come out "best"
        if len(names) < 2:
            sys.exit('[FATAL] found %d MixingModule profile(s) to compare against in:\n  %s\n'
                     'pass them explicitly with --candidates' % (len(names), '\n  '.join(searched)))
        print('%d candidate profiles found' % len(names))
    if configured and configured not in names:
        names.append(configured)
    rows = []
    for name in names:
        # cfi module names never contain '.' or '/': anything that does is a file
        if ('.' in name or '/' in name) and not os.path.isfile(name):
            print('    skip %s: no such file (looked in %s)' % (name, os.getcwd()))
            continue
        try:
            prof = profile_from_file(name) if os.path.isfile(name) else mc_profile(name)
        except SystemExit as e:             # not a usable 100-bin probFunction profile
            print('    skip %s: %s' % (name, e))
            continue
        c2, ndf = chi2(counts, prof['prob'])
        rows.append((c2 / max(ndf, 1), c2, ndf, prof['mean'], name, prof['prob']))
    rows.sort()
    print('\nprofiles ranked by chi2/ndf against the measured nti distribution '
          '(about 1 = compatible):')
    for r, c2, ndf, mean, name, _ in rows:
        print('    %9.2f   chi2 %10.1f / %3d   mean %.2f   %s%s'
              % (r, c2, ndf, mean, name, '   <== configured for %s' % args.campaign
                 if name == configured else ''))
    if configured and rows and rows[0][4] != configured:
        print('\n[WARNING] the configured profile %s is NOT the best match (%s)'
              % (configured, rows[0][4]))

    if args.plot and rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        x = np.arange(cfg.NBINS + 1)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.stairs(counts / counts.sum(), x, color='black', lw=2,
                  label='measured, %s (%d ev)' % (meas['label'], meas['n_events']))
        for r, _, _, _, name, prob in rows[:3]:
            ax.stairs(prob, x, label='%s  (chi2/ndf %.1f)'
                      % (os.path.basename(name).replace('_cfi', ''), r))
        ax.set_xlabel('true pileup (nti)')
        ax.set_ylabel('fraction per unit')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.plot)
        print('plot: %s' % args.plot)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('measure')
    m.add_argument('--files', required=True, help='text file, one MINIAODSIM file per line')
    m.add_argument('--label', required=True)
    m.add_argument('-o', '--output', required=True)
    m.add_argument('--max-events', type=int, default=20000)
    m.add_argument('--max-files', type=int, default=-1)
    m.add_argument('--redirector', default=REDIRECTOR)
    c = sub.add_parser('compare')
    c.add_argument('measurement')
    c.add_argument('--campaign', choices=sorted(cfg.CAMPAIGNS))
    c.add_argument('--candidates', nargs='*',
                   help='cfi module names, or files: a cfi / cmsDriver cfg / edmProvDump output '
                        'holding one probFunctionVariable + probValue pair, or a JSON with '
                        '"prob". Default: all Run 3 cfis found')
    c.add_argument('--plot', default='')
    args = p.parse_args()
    measure(args) if args.cmd == 'measure' else compare(args)


if __name__ == '__main__':
    main()
