#!/usr/bin/env python3
'''
covflow_pulls.py -- are the MC track pulls Gaussian with width 1, before and
after covflow?

For every gen-matched MC muon of a dimuon ntuple (inspector_mm_analysis.py
--mc, with or without --covflow) and for the three track parameters the
generator truth can check,

    pull = (reco - gen) / sigma        for  q/p,  lambda,  phi

with sigma from the RAW covariance (mu<i>_cov_*) and, if the ntuple was made
with covflow, from the CORRECTED one (mu<i>_cov_corr_*). This decides between
the two readings of what covflow measures:

  raw width ~ 1, corrected width < 1   MC reports its own scatter honestly and
                                       data tracks are truly less precise: the
                                       parameters need the extra smearing
                                       delta_2 ~ N(0, clip(C* - C))
  raw width != 1, corrected ~ 1        the MC errors were off, covflow alone fixes
                                       them, no delta_2

WHAT IS COMPARED, EXACTLY
  reco   bestTrack() parameters at its reference point (point of closest
         approach to the beam line): q/p and lambda from mu<i>_best_trk_pt /
         _best_trk_eta and mu<i>_charge; phi from mu<i>_phi, the pat::Muon
         direction (no best_trk_phi branch yet). The two agree for muons whose
         pat p4 IS the best track: by default only those are used (pT equal to
         --same-pt-tol, relative).
  gen    the matched generated muon (mu<i>_gen_*; |pdgId| = 13, dR < 0.1), at its
         production point. q/p and lambda are constant along the helix. phi is
         not: it is transported from the production point to the reco
         reference point, with the production point taken as the reco dimuon
         vertex (vx, vy; the gen vertex is not stored) and the reference point
         as the beam spot (bs_x0, bs_y0):
             phi_ref = phi_gen + q * s / R,   s = (v - b) . u_T,  R = pT / (0.3 B)
         The vertex resolution (tens of um) makes this exact to ~1e-5 rad. Use
         --no-phi-transport to see the size of the effect.
  dxy, dsz   NOT tested: they need the generated vertex (placeholder branches
         mu<i>_gen_vx/vy/vz in hitemu, not in the ntuples yet).

WIDTH ESTIMATORS (all = 1 for a unit Gaussian)
  sigma68   half the distance between the 15.87% and 84.13% quantiles: the
            Gaussian core, insensitive to tails -- the headline number
  rms5      RMS of the pulls within |pull| < 5
  >3sigma   fraction of |pull| > 3 (0.27% for a unit Gaussian)
  Errors: asymptotic, for a Gaussian (sigma68: 0.96 sigma / sqrt(n)).

Usage (any python with numpy, uproot, matplotlib; e.g. the covflow conda env):
  python3 covflow_pulls.py pilot/*.root
  python3 covflow_pulls.py pilot/*.root --out pulls_pilot --max-entries 200000
Outputs <out>.pdf and <out>.json. The table is printed per muon role (mu1, mu2)
and for both roles merged, each split into mu+ and mu-. The pdf merges the roles:
  page 1  all muons: pull distributions, core width vs pT and |eta|, mean vs pT
  page 2  mu+ and mu- apart: distributions, core width and mean vs pT per charge
          (a mean of opposite sign for mu+ and mu-, growing as 1/pT, points at
          the phi transport; the same sign for both, at a reco/gen offset)
'''

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

PARAMS = ('qoverp', 'lambda', 'phi')
LABEL = {'qoverp': 'q/p', 'lambda': 'λ', 'phi': 'φ'}
B_FIELD = 3.8                      # T
PT_EDGES = np.array([0., 4., 5., 6., 8., 10., 15., 1e4])
ETA_EDGES = np.array([0., 0.5, 0.9, 1.2, 1.6, 2.0, 2.5])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('files', nargs='+', help='dimuon MC ntuples (globs allowed)')
    p.add_argument('--tree', default='tree')
    p.add_argument('--muons', nargs='+', default=['mu1', 'mu2'])
    p.add_argument('--out', default='covflow_pulls', help='output stem (default: %(default)s)')
    p.add_argument('--max-entries', type=int, default=None)
    p.add_argument('--same-pt-tol', type=float, default=1e-4,
                   help='keep muons with |pT(pat) / pT(bestTrack) - 1| below this, so that '
                        'mu<i>_phi is the best-track phi; negative = keep all (default: %(default)s)')
    p.add_argument('--no-phi-transport', action='store_true',
                   help='compare phi at the production point and at the reference point as is')
    p.add_argument('--selection', default='',
                   help="extra numpy expression on the branches, e.g. 'abs(mass-3.0969)<0.1'")
    return p.parse_args()


class Progress:
    def __init__(self, total, label):
        self.total, self.label, self.t0 = max(total, 1), label, time.time()

    def update(self, done):
        f = min(done / self.total, 1.0)
        bar = '#' * int(30 * f) + '.' * (30 - int(30 * f))
        rate = done / max(time.time() - self.t0, 1e-9)
        sys.stdout.write('\r[%s] %5.1f%%  %s  %d/%d  %.0f/s' % (bar, 100 * f, self.label,
                                                                done, self.total, rate))
        sys.stdout.flush()
        if done >= self.total:
            sys.stdout.write('\n')


def branches_for(mu, have):
    want = ['best_trk_pt', 'best_trk_eta', 'pt', 'phi', 'charge', 'gen_pt', 'gen_eta',
            'gen_phi', 'gen_pdgid', 'covflow_ok']
    want += ['cov_%s_%s' % (p, p) for p in PARAMS]
    want += ['cov_corr_%s_%s' % (p, p) for p in PARAMS]
    out = ['%s_%s' % (mu, w) for w in want]
    return [b for b in out if b in have]


def read(files, args):
    import uproot
    with uproot.open(files[0]) as fh:
        have = set(fh[args.tree].keys())
    need = {'vx', 'vy', 'bs_x0', 'bs_y0'} & have
    for mu in args.muons:
        need |= set(branches_for(mu, have))
    if args.selection:
        import re
        need |= {n for n in re.findall(r'\b[A-Za-z_]\w*\b', args.selection) if n in have}
    missing = [b for mu in args.muons for b in ('%s_gen_pt' % mu, '%s_cov_qoverp_qoverp' % mu)
               if b not in have]
    if missing:
        sys.exit('ERROR: %s lacks %s -- not a dimuon MC ntuple?' % (files[0], missing))
    total = 0
    for f in files:
        with uproot.open(f) as fh:
            total += fh[args.tree].num_entries
    if args.max_entries:
        total = min(total, args.max_entries)
    prog = Progress(total, 'reading')
    cols, done = {}, 0
    for chunk in uproot.iterate([{f: args.tree} for f in files], expressions=sorted(need),
                                library='np', step_size='100 MB'):
        n = len(next(iter(chunk.values())))
        if args.max_entries and done + n > args.max_entries:
            n = args.max_entries - done
        for k, v in chunk.items():
            cols.setdefault(k, []).append(np.asarray(v[:n], np.float64))
        done += n
        prog.update(done)
        if args.max_entries and done >= args.max_entries:
            break
    cols = {k: np.concatenate(v) for k, v in cols.items()}
    has_corr = all('%s_cov_corr_qoverp_qoverp' % mu in cols for mu in args.muons)
    return cols, has_corr


def muon_pulls(cols, mu, args, sel):
    g = lambda k: cols['%s_%s' % (mu, k)]
    pt, eta, q = g('best_trk_pt'), g('best_trk_eta'), np.sign(g('charge'))
    gpt, geta, gphi = g('gen_pt'), g('gen_eta'), g('gen_phi')
    gq = -np.sign(g('gen_pdgid'))                     # pdgId 13 is the mu-
    ok = sel & np.isfinite(pt) & (pt > 0) & np.isfinite(gpt) & (gpt > 0) \
        & (np.abs(g('gen_pdgid')) == 13) & (q == gq)
    if args.same_pt_tol >= 0:
        ok &= np.abs(g('pt') / pt - 1.) < args.same_pt_tol
    lam = lambda e: 0.5 * np.pi - 2. * np.arctan(np.exp(-e))
    p, gp = pt * np.cosh(eta), gpt * np.cosh(geta)
    reco = {'qoverp': q / p, 'lambda': lam(eta), 'phi': g('phi')}
    gphi_ref = gphi
    if not args.no_phi_transport and all(k in cols for k in ('vx', 'vy', 'bs_x0', 'bs_y0')):
        # transverse distance travelled from the reference point (PCA to the beam
        # spot) to the production point, along the track direction; B along +z:
        # dphi/ds = -q / R, so phi_ref = phi_prod + q s / R
        s = (cols['vx'] - cols['bs_x0']) * np.cos(gphi) + (cols['vy'] - cols['bs_y0']) * np.sin(gphi)
        R = gpt / (0.299792458 * B_FIELD) * 100.          # cm
        gphi_ref = gphi + gq * s / R
    gen = {'qoverp': gq / gp, 'lambda': lam(geta), 'phi': gphi_ref}
    res = {}
    for prm in PARAMS:
        d = reco[prm] - gen[prm]
        if prm == 'phi':
            d = (d + np.pi) % (2 * np.pi) - np.pi
        for kind, pre in (('raw', 'cov_'), ('corr', 'cov_corr_')):
            key = '%s_%s%s_%s' % (mu, pre, prm, prm)
            if key not in cols:
                continue
            var = cols[key]
            use = ok & np.isfinite(var) & (var > 0)
            if kind == 'corr' and '%s_covflow_ok' % mu in cols:
                use &= cols['%s_covflow_ok' % mu] > 0.5
            res[(prm, kind)] = dict(pull=(d / np.sqrt(np.where(var > 0, var, np.nan)))[use],
                                    pt=pt[use], aeta=np.abs(eta)[use], q=q[use])
    return res, int(ok.sum())


def widths(x):
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return dict(n=n)
    q16, q50, q84 = np.quantile(x, [0.158655, 0.5, 0.841345])
    s68 = 0.5 * (q84 - q16)
    core = x[np.abs(x) < 5]
    return dict(n=n, mean=float(np.mean(core)), mean_err=float(np.std(core) / np.sqrt(len(core))),
                median=float(q50), sigma68=float(s68),
                # two correlated quantiles of a Gaussian: var(sigma68) = 0.925 sigma^2 / n
                sigma68_err=float(0.9617 * s68 / np.sqrt(n)),
                rms5=float(np.std(core)), rms5_err=float(np.std(core) / np.sqrt(2 * len(core))),
                frac_gt3=float(np.mean(np.abs(x) > 3)))


def merge(results):
    '''Concatenate the per-role results: (param, kind) -> one set of arrays.'''
    out = {}
    for res in results:
        for key, r in res.items():
            t = out.setdefault(key, {k: [] for k in r})
            for k, v in r.items():
                t[k].append(v)
    return {key: {k: np.concatenate(v) for k, v in t.items()} for key, t in out.items()}


def print_table(name, res, nmu, a, store):
    print('\n%s: %d gen-matched muons (same charge%s)'
          % (name, nmu, ', pat pT = best-track pT' if a.same_pt_tol >= 0 else ''))
    print('   %-7s %-5s %8s %9s %9s %9s %9s %8s'
          % ('param', 'cov', 'n', 'mean', 'sigma68', '+-', 'rms(<5)', '>3sig'))
    for (prm, kind), r in sorted(res.items(), key=lambda kv: (PARAMS.index(kv[0][0]), kv[0][1] != 'raw')):
        w = widths(r['pull'])
        store['%s_%s' % (prm, kind)] = w
        if w['n'] < 20:
            print('   %-7s %-5s %8d   (too few)' % (prm, kind, w['n']))
            continue
        print('   %-7s %-5s %8d %9.3f %9.3f %9.3f %9.3f %7.2f%%'
              % (prm, kind, w['n'], w['mean'], w['sigma68'], w['sigma68_err'], w['rms5'],
                 100 * w['frac_gt3']))
        for sgn, cname in ((1, 'mu+'), (-1, 'mu-')):
            ws = widths(r['pull'][r['q'] == sgn])
            store['%s_%s_%s' % (prm, kind, cname)] = ws
            if ws['n'] < 20:
                print('   %7s %-5s %8d   (too few)' % (cname, '', ws['n']))
                continue
            print('   %7s %-5s %8d %9.3f %9.3f %9.3f %9.3f %7.2f%%'
                  % (cname, '', ws['n'], ws['mean'], ws['sigma68'], ws['sigma68_err'], ws['rms5'],
                     100 * ws['frac_gt3']))


def binned(r, var, edges, what, sub=None):
    '''(x, y, err) of the core width (what='sigma68') or the mean (what='mean')
    in bins of var, for the pulls selected by sub.'''
    m0 = np.ones(len(r['pull']), bool) if sub is None else sub
    xs, ys, es = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = m0 & (r[var] >= lo) & (r[var] < hi)
        w = widths(r['pull'][m])
        if w['n'] >= 30:
            xs.append(0.5 * (lo + min(hi, 30.)))
            ys.append(w[what]); es.append(w[what + '_err'])
    return xs, ys, es


def main():
    a = parse_args()
    files = sorted(f for pat in a.files for f in glob.glob(pat))
    if not files:
        sys.exit('ERROR: no input files match %s' % a.files)
    print('covflow pulls: %d file(s), tree %s' % (len(files), a.tree))
    cols, has_corr = read(files, a)
    n_ev = len(next(iter(cols.values())))
    sel = np.ones(n_ev, bool)
    if a.selection:
        env = dict(cols, abs=np.abs, sqrt=np.sqrt, np=np)
        sel = np.asarray(eval(a.selection, {'__builtins__': {}}, env), bool)
    print('%d candidates, %d selected; covariance: raw%s'
          % (n_ev, sel.sum(), ' and covflow-corrected' if has_corr else
             ' only (no mu<i>_cov_corr_* branches: ntuple made without --covflow)'))

    allres, summary = {}, dict(files=files, n_candidates=n_ev, selection=a.selection,
                               phi_transport=not a.no_phi_transport,
                               same_pt_tol=a.same_pt_tol, muons={})
    nused = {}
    for mu in a.muons:
        res, nused[mu] = muon_pulls(cols, mu, a, sel)
        allres[mu] = res
        summary['muons'][mu] = dict(n_used=nused[mu])
        print_table(mu, res, nused[mu], a, summary['muons'][mu])
    both = merge([allres[mu] for mu in a.muons])
    name = ' + '.join(a.muons)
    summary['muons'][name] = dict(n_used=sum(nused.values()))
    print_table(name, both, sum(nused.values()), a, summary['muons'][name])
    print('\nunit Gaussian: mean 0, sigma68 1, rms(<5) 1, >3sig 0.27%')

    # ------------------------------------------------------------------ plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams['mathtext.default'] = 'regular'
    kinds = ['raw', 'corr'] if has_corr else ['raw']
    col = {'raw': '#2a5a9a', 'corr': '#b5651d'}
    lab = {'raw': 'raw covariance', 'corr': 'covflow-corrected'}
    chg = (('mu+', 1, 'o', '-'), ('mu-', -1, 's', '--'))
    hb = np.linspace(-6, 6, 81)
    hc = 0.5 * (hb[1:] + hb[:-1])
    gauss = lambda n: n * (hb[1] - hb[0]) * np.exp(-0.5 * hc ** 2) / np.sqrt(2 * np.pi)
    who = ' + '.join(a.muons)

    def dist(A, prm, sub=None, title=''):
        n = 0
        for k in kinds:
            r = both.get((prm, k))
            if r is None:
                continue
            x = r['pull'] if sub is None else r['pull'][sub(r)]
            if len(x) < 20:
                continue
            w = widths(x)
            A.hist(np.clip(x, -6, 6), hb, histtype='step', lw=1.6, color=col[k],
                   label='%s: mean %+.3f, σ68 %.3f, rms %.3f'
                         % (lab[k], w['mean'], w['sigma68'], w['rms5']))
            n = len(x)
        if n:
            A.plot(hc, gauss(n), 'k--', lw=1, label='N(0, 1)')
        A.set_yscale('log'); A.set_ylim(bottom=0.5)
        A.set_xlabel('pull of %s' % LABEL[prm]); A.legend(fontsize=7.5, frameon=False, loc='lower center')
        A.set_title(title)

    def finish(A, xl, yl, title, ref):
        A.axhline(ref, color='0.5', lw=1, ls=':')
        A.set_xlabel(xl); A.set_ylabel(yl); A.set_title(title)
        A.legend(fontsize=7.5, frameon=False)

    with PdfPages(a.out + '.pdf') as pdf:
        # ---- page 1: all muons, both roles merged
        fig, ax = plt.subplots(3, 4, figsize=(20, 12.5))
        fig.suptitle('%s merged, all charges' % who, fontsize=14)
        for i, prm in enumerate(PARAMS):
            dist(ax[i, 0], prm, title='%s: pull distribution' % LABEL[prm])
            for j, (var, edges, xl, what, yl, ref) in enumerate((
                    ('pt', PT_EDGES, 'p$_T$ [GeV]', 'sigma68', 'σ68 of the pull', 1.),
                    ('aeta', ETA_EDGES, '|η|', 'sigma68', 'σ68 of the pull', 1.),
                    ('pt', PT_EDGES, 'p$_T$ [GeV]', 'mean', 'mean pull', 0.))):
                A = ax[i, 1 + j]
                for k in kinds:
                    r = both.get((prm, k))
                    if r is not None:
                        A.errorbar(*binned(r, var, edges, what)[:2], yerr=binned(r, var, edges, what)[2],
                                   fmt='o-', color=col[k], label=lab[k], ms=4)
                finish(A, xl, yl, '%s: %s' % (LABEL[prm], 'core width' if what == 'sigma68' else 'mean'), ref)
        fig.tight_layout(rect=(0, 0, 1, 0.97)); pdf.savefig(fig); plt.close(fig)

        # ---- page 2: mu+ and mu- apart
        fig, ax = plt.subplots(3, 4, figsize=(20, 12.5))
        fig.suptitle('%s merged, μ⁺ and μ⁻ apart' % who, fontsize=14)
        for i, prm in enumerate(PARAMS):
            for j, (cname, sgn, _, _) in enumerate(chg):
                dist(ax[i, j], prm, sub=lambda r, s_=sgn: r['q'] == s_,
                     title='%s, %s' % (LABEL[prm], 'μ⁺' if sgn > 0 else 'μ⁻'))
            for j, (what, yl, ref) in enumerate((('sigma68', 'σ68 of the pull', 1.),
                                                 ('mean', 'mean pull', 0.))):
                A = ax[i, 2 + j]
                for k in kinds:
                    r = both.get((prm, k))
                    if r is None:
                        continue
                    for cname, sgn, mk, ls in chg:
                        x, y, e = binned(r, 'pt', PT_EDGES, what, sub=r['q'] == sgn)
                        A.errorbar(x, y, yerr=e, fmt=mk, ls=ls, color=col[k], ms=4,
                                   mfc=col[k] if sgn > 0 else 'none',
                                   label='%s, %s' % (lab[k], 'μ⁺' if sgn > 0 else 'μ⁻'))
                finish(A, 'p$_T$ [GeV]', yl, '%s: %s vs p$_T$, by charge'
                       % (LABEL[prm], 'core width' if what == 'sigma68' else 'mean'), ref)
        fig.tight_layout(rect=(0, 0, 1, 0.97)); pdf.savefig(fig); plt.close(fig)
    with open(a.out + '.json', 'w') as f:
        json.dump(summary, f, indent=1)
    print('[out] %s.pdf, %s.json' % (a.out, a.out))


if __name__ == '__main__':
    main()
