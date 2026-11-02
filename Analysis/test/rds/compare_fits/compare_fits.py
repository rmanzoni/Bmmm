#!/usr/bin/env python3
'''
Comparison of the kinematic-fit options of the R(Ds) ntuples on the signal MC:
which mass constraints (cons: phi + Ds, phic: phi, dsc: Ds, free: none) and
which Bs momentum reconstruction (collinear, xyz, the two neutrino solutions
and choices between them) to use for q2 and the other observables.

    python3 compare_fits.py /pnfs/psi.ch/cms/trivcat/store/user/manzoni/rds_prod.root -o cmp_v1

Writes, in the output directory:
    index.html          the report: tables and every plot
    summary.md          the summary tables, markdown
    results.json        every number; results_long.csv one per row
    plots/              the figures (PNG; --pdf for PDF too)
    log.txt, status.json  progress (status.json is rewritten at every stage)

Metrics (details in README.md):
    tau / mu separation   AUC per observable (tau vs mu, within Ds and within
                          Ds*); expected precision on R(Ds), R(Ds*) from a
                          binned four-template fit (Fisher information), with
                          single observables, (q2, m_miss2), (q2, E*_mu),
                          (q2, m_miss2, E*_mu) and a multiclass BDT
    resolution            reco - gen: bias, sigma68, tails, rank correlation
    topology              fit success, vertex-cut efficiency, P(V) flatness,
                          Bs vertex residuals and pulls, Bs flight direction,
                          Ds mass peak, Ds pT

Needs numpy, scipy, matplotlib, uproot; scikit-learn for the BDT (skipped
without it). No ROOT.
'''

import os
import sys
import json
import time
import argparse
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cfio
import cfcore
import cfanalysis as A
import cfreport
from cfreport import KEY_SORT
from cfplots import Plotter, HYP_LABEL
from cfio import SCHEME_LABEL, GROUPS, CLASSES, OBS_LABEL


class Log(object):
    '''Timestamped lines to stdout and log.txt; the current stage to
    status.json, so a long run can be followed from another shell.'''

    def __init__(self, outdir):
        self.f = open(os.path.join(outdir, 'log.txt'), 'a')
        self.status = os.path.join(outdir, 'status.json')
        self.t0 = time.time()
        self.stage = None

    def __call__(self, msg):
        line = '[%s +%6.0fs] %s' % (datetime.now().strftime('%H:%M:%S'), time.time() - self.t0, msg)
        print(line, flush=True)
        self.f.write(line + '\n')
        self.f.flush()

    def begin(self, stage, i, n):
        self.stage = stage
        self('==== [%d/%d] %s' % (i, n, stage))
        with open(self.status, 'w') as f:
            json.dump(dict(stage=stage, step=i, of=n, elapsed_s=round(time.time() - self.t0),
                           at=datetime.now().isoformat(timespec='seconds')), f)

    def done(self):
        with open(self.status, 'w') as f:
            json.dump(dict(stage='done', elapsed_s=round(time.time() - self.t0),
                           at=datetime.now().isoformat(timespec='seconds')), f)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='R(Ds) ntuple (path or root:// URL)')
    ap.add_argument('-o', '--outdir', required=True)
    ap.add_argument('--tree', default='tree')
    ap.add_argument('--max-entries', type=int, default=None, help='read at most this many rows')
    ap.add_argument('--max-candidates', type=int, default=1000000,
                    help='stop reading once this many truth-matched signal candidates are kept; the file is read '
                         'in blocks in random order, so they are an unbiased sample (0: whole file)')
    ap.add_argument('-j', '--jobs', type=int, default=4, help='parallel reader processes')
    ap.add_argument('--block', type=int, default=100000, help='entries per block')
    ap.add_argument('--no-cache', action='store_true', help='do not cache the selected arrays')
    ap.add_argument('--no-trigger', action='store_true', help='do not require the parking trigger and the muon match')
    ap.add_argument('--no-species', action='store_true', help='do not require K, K, pi, mu matched to K, K, pi, mu')
    ap.add_argument('--all-candidates', action='store_true', help='keep every truth-matched candidate of an event')
    ap.add_argument('--no-common', action='store_true',
                    help='compare each scheme on its own good Bs vertices instead of on the events where every scheme has one')
    ap.add_argument('--r-target', default='0.299,0.244', help='R(Ds),R(Ds*) the tau channels are weighted to')
    ap.add_argument('--r-mc', default='1,1', help='R(Ds),R(Ds*) the MC was generated with (AN: 1, 1)')
    ap.add_argument('--n-ref', type=float, default=1e5, help='precision quoted for this many Ds mu nu candidates')
    ap.add_argument('--nbins', type=int, default=20, help='bins of the 1D precision estimate')
    ap.add_argument('--no-bdt', action='store_true')
    ap.add_argument('--bdt-max', type=int, default=50000, help='BDT candidates per channel')
    ap.add_argument('--bdt-sets', default='coll,xyz,math,all_hyps')
    ap.add_argument('--pdf', action='store_true', help='PDF of every plot too')
    return ap.parse_args()


def main():
    args = parse()
    os.makedirs(args.outdir, exist_ok=True)
    log = Log(args.outdir)
    log('compare_fits %s' % ' '.join(sys.argv[1:]))
    N = 9
    res = {}

    def checkpoint():
        with open(os.path.join(args.outdir, 'results.json'), 'w') as f:
            json.dump(res, f, indent=1, default=lambda x: None)

    # ---- 1. read ----------------------------------------------------------
    log.begin('read and select', 1, N)
    layout, d, read_info = cfio.read(args.input, tree=args.tree, max_entries=args.max_entries,
                          max_candidates=args.max_candidates, block=args.block, jobs=args.jobs,
                          cache_dir=None if args.no_cache else os.path.join(args.outdir, 'cache'), log=log)
    cfio.add_derived(layout, d)
    m, cutflow = cfio.select(layout, d, trigger=not args.no_trigger, species=not args.no_species,
                             one_per_event=not args.all_candidates, log=log)
    cm = cfio.common_mask(layout, d)
    mc = m if args.no_common else (m & cm)
    cls = d['gen_cand'].astype(int)
    cutflow.append(('every scheme has a Bs vertex', [int(np.sum(m & cm & (cls == c))) for c in CLASSES]))
    log('  %-34s %s' % (cutflow[-1][0], '  '.join('%d' % k for k in cutflow[-1][1])))
    r_t = tuple(float(x) for x in args.r_target.split(','))
    r_m = tuple(float(x) for x in args.r_mc.split(','))
    w = cfio.class_weights(cls, r_t, r_m)
    sel_txt = 'truth-matched signal' + ('' if args.no_species else ', correct species') + \
              ('' if args.no_trigger else ', trigger and muon match') + \
              ('' if args.all_candidates else ', one candidate per event') + \
              ('' if args.no_common else ', Bs vertex in every scheme')
    if read_info.get('fraction_read', 1.) < 0.999:
        sel_txt += '; %.1f%% of the file read (random blocks of %d entries)' % (
            100 * read_info['fraction_read'], read_info['block'])
    meta = dict(input=args.input, when=datetime.now().strftime('%Y-%m-%d %H:%M'), selection=sel_txt,
                r_target=r_t, r_mc=r_m, n_ref=int(args.n_ref), schemes=layout.schemes,
                hyps=layout.hyps, read=read_info, args=vars(args))
    res['meta'] = meta
    res['cutflow'] = cutflow
    if np.sum(mc) < 100:
        log('FATAL: only %d candidates selected' % np.sum(mc))
        sys.exit(1)

    # ---- 2. efficiencies, fit quality, masses, vertex ---------------------
    log.begin('efficiencies, fit quality, masses, vertices', 2, N)
    res['efficiency'] = A.efficiencies(layout, d, m)
    res['fit_quality'] = A.fit_quality(layout, d, mc)
    res['masses'] = A.masses(layout, d, mc)
    res['vertex'] = A.vertex(layout, d, mc)
    for s in layout.schemes:
        e = res['efficiency'][s]
        log('  %-5s Bs fit ok %s   vertex cuts %s' % (
            s, ' '.join('%.3f' % e[str(c)].get('good_bs', np.nan) for c in CLASSES),
            ' '.join('%.3f' % e[str(c)].get('vtx_cut', np.nan) for c in CLASSES)))
    checkpoint()

    # ---- 3. resolution -------------------------------------------------------
    log.begin('resolution', 3, N)
    res['resolution'] = A.resolution(layout, d, mc, log=log)
    checkpoint()

    # ---- 4. single-observable separation -------------------------------------
    log.begin('tau/mu separation, single observables', 4, N)
    res['separation'] = A.separation_1d(layout, d, mc, w, nbins=args.nbins, n_ref=args.n_ref, log=log)
    log('  %d observables' % len(res['separation']))
    checkpoint()

    # ---- 5. multidimensional ---------------------------------------------------
    log.begin('expected precision, 2D and 3D', 5, N)
    res['multidim'] = A.multidim(layout, d, mc, w, n_ref=args.n_ref)
    res['multidim_generator'] = A.multidim_generator(d, mc, w, n_ref=args.n_ref)
    res['degeneracy'] = A.degeneracy(layout, d, mc)
    for name, f in res['multidim_generator'].items():
        log('  generator level %-16s sigma tau/mu %.4f  R(Ds) %.4f  R(Ds*) %.4f' % (
            name, f['r_taumu'] or np.nan, f['r_ds'] or np.nan, f['r_dsst'] or np.nan))
    for s in layout.schemes:
        deg = [h for h, r in res['degeneracy'].get(s, {}).items() if (r.get('max_abs') or 0) > 0.99]
        if deg:
            log('  %-5s q2, m_miss2, E*_mu nearly one variable (|rho| > 0.99) for: %s' % (s, ', '.join(deg)))
    checkpoint()

    # ---- 6. BDT ---------------------------------------------------------------
    scores = {}
    if args.no_bdt:
        res['bdt'] = {}
    else:
        log.begin('BDT', 6, N)
        res['bdt'], scores = A.bdt(layout, d, mc, w, sets=tuple(args.bdt_sets.split(',')),
                                   max_per_class=args.bdt_max, n_ref=args.n_ref, log=log)
    checkpoint()

    # ---- 7. plots ----------------------------------------------------------------
    log.begin('plots', 7, N)
    figs = make_plots(args, layout, d, m, mc, res, scores, cutflow, log)

    # ---- 8. report -------------------------------------------------------------------
    log.begin('report', 8, N)
    tabs = cfreport.write(args.outdir, res, layout, figs, meta)
    h, rows = tabs['key']
    log('  best by sigma N(tau)/N(mu) from (q2, m_miss2, E*_mu):')
    for r in rows[:6]:
        log('    %-5s %-12s tau/mu %s   R(Ds) %s   R(Ds*) %s' % (r[0], r[1], cfreport._fmt(r[KEY_SORT]),
                                                              cfreport._fmt(r[KEY_SORT + 1]), cfreport._fmt(r[KEY_SORT + 2])))
    log.begin('done: %s' % os.path.join(args.outdir, 'index.html'), 9, N)
    log.done()


def make_plots(args, L, d, m, mc, res, scores, cutflow, log):
    P = Plotter(args.outdir, L, d, mc, pdf=args.pdf)
    figs = {}

    def do(key, fn, *a, **k):
        try:
            out = fn(*a, **k)
            if out:
                figs[key] = out
        except Exception as exc:          # one broken plot must not kill the report
            log('  plot %s failed: %s: %s' % (key, type(exc).__name__, exc))

    do('cutflow', P.cutflow, cutflow)
    do('efficiency', P.efficiency, res['efficiency'])
    do('probabilities', P.probabilities)
    do('masses', P.masses)
    do('vertex', P.vertex)
    do('math_choice', P.math_choice)
    do('q2_vs_gen', P.q2_vs_gen)
    log('  overview plots done')

    obs = [q for q in cfio.HYP_OBS]
    for q in obs:
        do('res_%s' % q, P.residuals, q)
        do('shape_%s' % q, P.shapes, q, res['separation'])
    log('  residual and shape plots done')
    do('topology_shapes', P.topology_shapes, res['separation'],
       ['mu_doca_ds_sig', 'mu_poca_along_ds', 'mu_ip3d_bs_pv_sig', 'mu_dist_to_bs_dir', 'bs_prob',
        'ds_lxyz_bs_sig', 'mcorr', 'gamma_e_coll_over_pt', 'gamma_dm'])

    # ---- heat maps --------------------------------------------------------------
    cols = [(s, h) for s in L.schemes for h in L.hyps[s]]
    col_lab = ['%s:%s' % (s, h) for s, h in cols]
    oracle = [j for j, (s, h) in enumerate(cols) if h in cfio.ORACLE_HYPS]
    rows = [q for q in obs if any('%s_%s_%s' % (s, h, q) in res['separation'] for s, h in cols)]
    cls = d['gen_cand'].astype(int)
    for pair, key, title in (('tau_vs_mu', 'auc_tau_mu', 'AUC, $\\tau$ vs $\\mu$ channels'),
                             ('ds_tau_vs_mu', 'auc_ds_tau_mu', 'AUC, D$_s\\tau\\nu$ vs D$_s\\mu\\nu$'),
                             ('dsst_tau_vs_mu', 'auc_dsst_tau_mu', 'AUC, D$_s^*\\tau\\nu$ vs D$_s^*\\mu\\nu$'),
                             ('dsst_vs_ds_mu', 'auc_dsst_ds_mu', 'AUC, D$_s^*\\mu\\nu$ vs D$_s\\mu\\nu$')):
        V = [[(res['separation'].get('%s_%s_%s' % (s, h, q), {}).get(pair, {}) or {}).get('auc') for s, h in cols]
             for q in rows]
        V = [[np.nan if v is None else v for v in r] for r in V]
        # first column: the same observable at generator level, the separation
        # a perfect reconstruction would give
        a, b = A.PAIRS[pair]
        gen = []
        for q in rows:
            gb = cfio.TARGETS.get(q, (None,))[0]
            if gb in d:
                x = d[gb]
                gen.append(cfcore.auc_oriented(x[mc & np.isin(cls, a)], x[mc & np.isin(cls, b)])[0])
            else:
                gen.append(np.nan)
        V = [[g] + r for g, r in zip(gen, V)]
        do(key, P.heatmap, key, [OBS_LABEL.get(q, q) for q in rows], ['generator'] + col_lab, V, title,
           fmt='%.3f', mode='auc', ref_cols=[0] + [j + 1 for j in oracle])
    for k, key, title in (('r_taumu', 'fisher_1d_taumu', 'expected $\\sigma_{rel}$ N($\\tau$)/N($\\mu$), fit to one observable'),
                          ('r_ds', 'fisher_1d_ds', 'expected $\\sigma_{rel}$ R(D$_s$), fit to one observable'),
                          ('r_dsst', 'fisher_1d_dsst', 'expected $\\sigma_{rel}$ R(D$_s^*$), fit to one observable')):
        V = [[((res['separation'].get('%s_%s_%s' % (s, h, q), {}) or {}).get('fisher') or {}).get(k) for s, h in cols]
             for q in rows]
        V = [[np.nan if v is None else v for v in r] for r in V]
        do(key, P.heatmap, key, [OBS_LABEL.get(q, q) for q in rows], col_lab, V, title, fmt='%.3f', mode='lower',
           ref_cols=oracle)
    for g in ('Ds mu', 'Ds* mu', 'Ds tau', 'Ds* tau'):
        V = [[(((res['resolution'].get(s, {}).get(h, {}) or {}).get(q, {}) or {}).get(g) or {}).get('s68')
              for s, h in cols] for q in rows]
        V = [[np.nan if v is None else v for v in r] for r in V]
        key = 'res_summary_%s' % g.replace(' ', '_').replace('*', 'st')
        do(key, P.heatmap, key, [OBS_LABEL.get(q, q) for q in rows], col_lab, V,
           '$\\sigma_{68}$(reco $-$ gen), %s' % g, fmt='%.3g', mode='lower', ref_cols=oracle)
    V = [[(((res['resolution'].get(s, {}).get(h, {}) or {}).get(q, {}) or {}).get('Ds mu') or {}).get('median')
          for s, h in cols] for q in rows]
    V = [[np.nan if v is None else abs(v) for v in r] for r in V]
    do('bias_summary_Ds_mu', P.heatmap, 'bias_summary_Ds_mu', [OBS_LABEL.get(q, q) for q in rows], col_lab, V,
       '|median(reco $-$ gen)|, Ds mu', fmt='%.3g', mode='lower', ref_cols=oracle)

    # ---- precision bars ------------------------------------------------------------
    entries = []
    for s in L.schemes:
        for h in L.hyps[s]:
            for name, lab in (('q2_mmiss2', 'q$^2$, m$^2_{miss}$'), ('q2_mmiss2_estar', 'q$^2$, m$^2_{miss}$, E$^*_\\mu$')):
                f = (res['multidim'].get(s, {}).get(h, {}) or {}).get(name)
                if f and h not in cfio.ORACLE_HYPS:
                    entries.append(('%s:%s  [%s]' % (s, h, lab), P.color(s), f))
    entries.sort(key=lambda e: e[2]['r_taumu'] if e[2].get('r_taumu') is not None else 9)
    entries = entries[:38]
    for name, lab in (('q2_mmiss2', 'q$^2$, m$^2_{miss}$'), ('q2_mmiss2_estar', 'q$^2$, m$^2_{miss}$, E$^*_\\mu$')):
        f = res.get('multidim_generator', {}).get(name)
        if f:
            entries.insert(0, ('GENERATOR  [%s]' % lab, '#999999', f))
    do('fisher_multidim', P.fisher_bars, entries, 'fisher_multidim',
       'Expected precision, fit to 2D / 3D distributions: generator level (grey), then the best 38 by N(tau)/N(mu)')
    if res.get('bdt'):
        ent = []
        for name, r in res['bdt'].items():
            s = name.split(':')[0]
            col = P.color(s) if s in L.schemes else '#555555'
            ent.append((name, col, r['fisher']))
        ent.sort(key=lambda e: e[2]['r_taumu'] if e[2].get('r_taumu') is not None else 9)
        do('fisher_bdt', P.fisher_bars, ent, 'fisher_bdt', 'Expected precision, fit to the BDT outputs')
        do('bdt_scores', P.bdt_scores, scores, [n for n in scores if n.endswith(':all_hyps')])
        cur = []
        for name, (idx, p_tau, _) in scores.items():
            if not name.endswith(':all_hyps'):
                continue
            y = d['gen_cand'][idx].astype(int)
            s = name.split(':')[0]
            cur.append((name, P.color(s) if s in L.schemes else '#555555', '-',
                        p_tau[np.isin(y, (2, 3))], p_tau[np.isin(y, (0, 1))]))
        do('roc_bdt', P.rocs, cur, 'roc_bdt', 'BDT, $\\tau$ vs $\\mu$')

    # ---- ROC curves of single observables ---------------------------------------
    tau, mu = P.sel((2, 3)), P.sel((0, 1))
    for q, key in (('q2', 'roc_q2'), ('m_miss2', 'roc_mmiss2'), ('e_star_mu', 'roc_estar')):
        cur = []
        for s in L.schemes:
            for h, ls in (('coll', '-'), ('xyz_bs', '--'), ('math_avg', ':'), ('math1_bs', '-.')):
                b = '%s_%s_%s' % (s, h, q)
                if b in d:
                    cur.append(('%s:%s' % (s, h), P.color(s), ls, d[b][tau], d[b][mu]))
        do(key, P.rocs, cur, key, '%s, $\\tau$ vs $\\mu$' % OBS_LABEL.get(q, q))
    log('  %d figures' % len(figs))
    return figs


if __name__ == '__main__':
    main()
