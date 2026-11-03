#!/usr/bin/env python3
'''Do the coll and math Bs hypotheses add information to xyz_bs?

Every Bs hypothesis is a deterministic function of the same measurements:
the visible Ds and mu four-vectors and the PV -> V_Bs direction (coll uses
only the visible part). This study tests the consequence, and the procedure
behind the cmp_v2 BDT ablation, on the free scheme:

  1. closure: the stored coll and xyz_bs q2 recomputed from the refitted
     tracks, PV and V_Bs (are the hypotheses built from what we think?)
  2. BDTs on xyz_obs, coll_obs, math_obs, their combinations, and on the raw
     measurements (tracks + flight direction), 3 seeds each for the spread
  3. positive controls, which must show a gain if the method can see one:
     the truth-picked neutrino solution (math_oracle) and a smeared generator
     q2 added to xyz_obs
  4. robustness: a larger BDT, 20x20 bins of the output plane, an unbinned
     Fisher information from the calibrated outputs, the out-of-fold log-loss
  5. binned fits: every 2D (12 bins/axis) and 3D (6 bins/axis) combination of
     q2, m_miss2, E*_mu over the hypotheses

    source cfenv.sh; export OMP_NUM_THREADS=32
    python3 study_hypinfo.py /pnfs/.../rds_prod.root -o hypinfo
'''
import os
import sys
import json
import time
import argparse
import itertools
import numpy as np

import cfio
import cfcore
import cfanalysis

S = 'free'
M_K, M_PI, M_MU, M_BS = 0.493677, 0.13957039, 0.1056583755, 5.36688
TRACKS = ('k1', 'k2', 'pi', 'mu')
RAW = ['%s_%s_rf_%s' % (S, t, q) for t in TRACKS for q in ('pt', 'eta', 'phi')]
OBS = cfanalysis.BDT_HYP_OBS


def log(*a):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), ' '.join(str(x) for x in a)), flush=True)


def p4(pt, eta, phi, m):
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    return np.stack([np.sqrt(px ** 2 + py ** 2 + pz ** 2 + m ** 2), px, py, pz], axis=1)


def m2(p):
    return p[:, 0] ** 2 - np.sum(p[:, 1:] ** 2, axis=1)


def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def raw_features(d):
    '''(raw, inv, closure): the measurements in two parametrisations, and the
    coll / xyz_bs q2 recomputed from them.'''
    t = dict((k, p4(d['%s_%s_rf_pt' % (S, k)].astype(float), d['%s_%s_rf_eta' % (S, k)].astype(float),
                    d['%s_%s_rf_phi' % (S, k)].astype(float), mk))
             for k, mk in (('k1', M_K), ('k2', M_K), ('pi', M_PI), ('mu', M_MU)))
    ds = t['k1'] + t['k2'] + t['pi']
    mu = t['mu']
    vis = ds + mu
    dv = np.stack([d['%s_bs_%s' % (S, c)] - d['pv_%s' % c] for c in 'xyz'], axis=1).astype(float)
    u = unit(dv)
    eta_d = np.arcsinh(u[:, 2] / np.hypot(u[:, 0], u[:, 1]))
    phi_d = np.arctan2(u[:, 1], u[:, 0])
    raw = {}
    for k in TRACKS:          # azimuth relative to the flight direction (the rest is symmetric)
        raw['%s_pt' % k] = d['%s_%s_rf_pt' % (S, k)]
        raw['%s_eta' % k] = d['%s_%s_rf_eta' % (S, k)]
        raw['%s_dphi' % k] = np.angle(np.exp(1j * (d['%s_%s_rf_phi' % (S, k)] - phi_d)))
    raw['d_eta'] = eta_d
    # rotation invariants of (Ds, mu, d) + the lab orientation
    pds, pmu = ds[:, 1:], mu[:, 1:]
    nds, nmu = np.linalg.norm(pds, axis=1), np.linalg.norm(pmu, axis=1)
    inv = dict(p_ds=nds, p_mu=nmu, m_kkpi=np.sqrt(np.maximum(m2(ds), 0)),
               m_kk=np.sqrt(np.maximum(m2(t['k1'] + t['k2']), 0)),
               c_ds_mu=np.sum(pds * pmu, 1) / (nds * nmu), c_ds_d=np.sum(pds * u, 1) / nds,
               c_mu_d=np.sum(pmu * u, 1) / nmu,
               triple=np.sum(np.cross(pds, pmu) * u, 1) / (nds * nmu), d_eta=eta_d,
               cos_k_pi=d['%s_cos_theta_k_pi' % S])
    # closure: coll and xyz_bs as in RDsKinematics
    mvis = np.sqrt(np.maximum(m2(vis), 0))
    coll = vis * (M_BS / mvis)[:, None]
    mag = np.linalg.norm(vis[:, 1:], axis=1) * M_BS / mvis
    xyz = np.concatenate([np.sqrt(M_BS ** 2 + mag ** 2)[:, None], u * mag[:, None]], axis=1)
    clo = dict(coll_q2=m2(coll - ds), xyz_q2=m2(xyz - ds), coll_mm2=m2(coll - ds - mu), xyz_mm2=m2(xyz - ds - mu))
    if 'mu_phi' in d:       # the same with the unrefitted muon
        mu0 = p4(d['mu_pt'].astype(float), d['mu_eta'].astype(float), d['mu_phi'].astype(float), M_MU)
        vis0 = ds + mu0
        mv0 = np.sqrt(np.maximum(m2(vis0), 0))
        mag0 = np.linalg.norm(vis0[:, 1:], axis=1) * M_BS / mv0
        xyz0 = np.concatenate([np.sqrt(M_BS ** 2 + mag0 ** 2)[:, None], u * mag0[:, None]], axis=1)
        clo['xyz_q2_mu_unrefit'] = m2(xyz0 - ds)
        clo['coll_q2_mu_unrefit'] = m2(vis0 * (M_BS / mv0)[:, None] - ds)
    return ({k: v.astype(np.float32) for k, v in raw.items()},
            {k: v.astype(np.float32) for k, v in inv.items()}, clo)


def hyp(h, qs=OBS):
    return ['%s_%s_%s' % (S, h, q) for q in qs]


PARAMS = dict(
    # trained to convergence: stops only after 100 iterations without a gain on the validation loss
    long=dict(max_iter=3000, learning_rate=0.05, max_leaf_nodes=31, early_stopping=True,
              validation_fraction=0.1, n_iter_no_change=100, tol=1e-7),
    # the cmp_v2 setting (stopped after ~45 iterations at 1M candidates)
    short=dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, early_stopping=True,
               validation_fraction=0.15))


def oof(X, y, sw, folds, seed, params):
    from sklearn.ensemble import HistGradientBoostingClassifier
    fold = np.random.default_rng(seed).integers(0, folds, len(y))
    P = np.full((len(y), 4), np.nan)
    iters = []
    for k in range(folds):
        tr, te = fold != k, fold == k
        clf = HistGradientBoostingClassifier(random_state=seed * 100 + k, **params)
        clf.fit(X[tr], y[tr], sample_weight=sw[tr])
        P[te] = clf.predict_proba(X[te])
        iters.append(int(clf.n_iter_))
    return P, iters


def unbinned_fisher(P, y, wi, n_ref):
    '''Fisher information of the four yields from the per-event class
    densities, f_a/sum_k N_k f_k = q_a/N_a, with q the posterior at the
    sample proportions (the BDT was trained on balanced classes).'''
    N = np.array([wi[y == c].sum() for c in range(4)])
    sc = n_ref / N[0]
    q = P * N[None, :]
    q /= q.sum(1, keepdims=True)
    Ns = N * sc
    F = (q * (wi * sc)[:, None] / Ns[None, :]).T @ (q / Ns[None, :])
    cov = np.linalg.inv(F)
    out = dict(r_ds=cfcore.ratio_rel_error(cov, Ns, 2, 0), r_dsst=cfcore.ratio_rel_error(cov, Ns, 3, 1))
    ql = np.stack([q[:, 2] + q[:, 3], q[:, 0] + q[:, 1]], 1)
    Nl = np.array([Ns[2] + Ns[3], Ns[0] + Ns[1]])
    F2 = (ql * (wi * sc)[:, None] / Nl[None, :]).T @ (ql / Nl[None, :])
    out['r_taumu'] = cfcore.ratio_rel_error(np.linalg.inv(F2), Nl, 0, 1)
    return out


def metrics(P, y, sw, wi, n_ref):
    p_tau, p_st = P[:, 2] + P[:, 3], P[:, 1] + P[:, 3]
    r = {}
    for nb in (10, 20):
        e = [cfcore.equal_pop_edges(p_tau, nb, wi), cfcore.equal_pop_edges(p_st, nb, wi)]
        f = cfcore.fisher_ratios([[p_tau[y == c], p_st[y == c]] for c in range(4)],
                                 [wi[y == c] for c in range(4)], e, n_ref=n_ref)
        r['b%d' % nb] = dict(r_taumu=f['r_taumu'], r_ds=f['r_ds'], r_dsst=f['r_dsst'], min_mc=f['min_mc'],
                             frac_low=f['frac_low'])
    r['unb'] = unbinned_fisher(P, y, wi, n_ref)
    pc = np.clip(P[np.arange(len(y)), y], 1e-12, 1)
    r['logloss'] = float(-np.sum(sw * np.log(pc)) / np.sum(sw))
    r['auc_taumu'] = cfcore.auc_oriented(p_tau[y >= 2], p_tau[y < 2])[0]
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input')
    ap.add_argument('-o', '--outdir', default='hypinfo')
    ap.add_argument('--max-candidates', type=int, default=0, help='0: read everything')
    ap.add_argument('--max-per-class', type=int, default=0, help='BDT events per class (0: all)')
    ap.add_argument('--cache-dir', default='/scratch/manzoni/compare_fits_cache',
                    help='where the cache and predictions are written; OUTDIR/cache gets symlinks')
    ap.add_argument('--folds', type=int, default=4)
    ap.add_argument('--seeds', default='1')
    ap.add_argument('-j', '--jobs', type=int, default=8)
    ap.add_argument('--n-ref', type=float, default=1e5)
    ap.add_argument('--boot', type=int, default=50, help='bootstrap replicas of the evaluation events')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(',')]
    res = dict(args=vars(args))

    def save():
        with open(os.path.join(args.outdir, 'hypinfo.json'), 'w') as f:
            json.dump(res, f, indent=1, default=lambda x: None)

    # ---- read: the suite's reader, plus the refitted tracks ------------------
    COMMON0 = tuple(cfio.COMMON_OBS)
    cfio.COMMON_OBS = COMMON0 + tuple(RAW) + ('pv_x', 'pv_y', 'pv_z', 'mu_phi')
    linkd = os.path.join(args.outdir, 'cache')
    layout, d, info = cfio.read(args.input, max_candidates=args.max_candidates, jobs=args.jobs,
                                cache_dir=args.cache_dir, link_dir=linkd, schemes=[S], log=log)
    res['read'] = info
    cfio.add_derived(layout, d)
    m, _ = cfio.select(layout, d, log=log)
    m &= cfio.common_mask(layout, d)
    cls = d['gen_cand'].astype(int)
    w = cfio.class_weights(cls, (0.299, 0.244), (1.0, 1.0))
    raw, inv, clo = raw_features(d)
    for k, v in list(raw.items()) + list(inv.items()):
        d['raw_' + k if k in raw else 'inv_' + k] = v
    RAWF = ['raw_' + k for k in raw]
    INVF = ['inv_' + k for k in inv]
    res['selected'] = [int(np.sum(m & (cls == c))) for c in range(4)]
    log('selected per class:', [int(np.sum(m & (cls == c))) for c in range(4)])

    # ---- 1. closure -----------------------------------------------------------
    res['closure'] = {}
    for h, q, k in (('coll', 'q2', 'coll_q2'), ('xyz_bs', 'q2', 'xyz_q2'), ('coll', 'm_miss2', 'coll_mm2'),
                    ('xyz_bs', 'm_miss2', 'xyz_mm2'), ('xyz_bs', 'q2', 'xyz_q2_mu_unrefit'),
                    ('coll', 'q2', 'coll_q2_mu_unrefit')):
        if k not in clo:
            continue
        a, b = d['%s_%s_%s' % (S, h, q)][m].astype(float), clo[k][m]
        ok = np.isfinite(a) & np.isfinite(b)
        dif = np.abs(a[ok] - b[ok])
        res['closure'][k] = dict(median_abs=float(np.median(dif)), q99_abs=float(np.quantile(dif, .99)),
                                                frac_gt_0p01=float(np.mean(dif > 0.01)), n=int(ok.sum()))
        log('closure %-16s median |diff| %.2e  99%% %.2e  frac>0.01 %.4f'
            % (k, np.median(dif), np.quantile(dif, .99), np.mean(dif > 0.01)))
    # how often the two neutrino solutions are a clamped vertex (identical)
    res['math_real_frac'] = float(np.nanmean(d['%s_math_real_bs' % S][m]))
    log('math: real solutions for %.3f of the candidates' % res['math_real_frac'])
    # correlations: is math a near-copy of xyz?
    res['corr'] = {}
    for q in ('q2', 'm_miss2', 'e_star_mu'):
        for h in ('coll', 'math1_bs', 'math2_bs'):
            a, b = d['%s_xyz_bs_%s' % (S, q)][m], d['%s_%s_%s' % (S, h, q)][m]
            ok = np.isfinite(a) & np.isfinite(b)
            ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
            res['corr']['%s_%s' % (h, q)] = float(np.corrcoef(ra, rb)[0, 1])
    log('rank correlation with xyz_bs:', json.dumps({k: round(v, 3) for k, v in res['corr'].items()}))

    # ---- 2.-4. BDTs -----------------------------------------------------------
    rng = np.random.default_rng(1)
    use = np.zeros(len(cls), bool)
    restore = np.ones(len(cls))
    for c in range(4):
        idx = np.nonzero(m & (cls == c))[0]
        n_all = len(idx)
        if args.max_per_class and n_all > args.max_per_class:
            idx = rng.choice(idx, args.max_per_class, replace=False)
        use[idx] = True
        restore[idx] = n_all / float(len(idx))
    idx = np.nonzero(use)[0]
    y = cls[idx]
    cw = dict((c, len(y) / (4. * np.sum(y == c))) for c in range(4))
    sw = np.array([cw[c] for c in y])
    wi = (w * restore)[idx]
    # positive control 2: an independent q2 measurement with 1 GeV2 resolution
    d['gen_q2_smeared'] = (d['gen_q2'] + np.random.default_rng(7).normal(0, 1.0, len(cls))).astype(np.float32)

    X3 = hyp('xyz_bs', ('q2', 'm_miss2', 'e_star_mu'))
    XYZ, COLL, MATH = hyp('xyz_bs'), hyp('coll'), hyp('math1_bs') + hyp('math2_bs')
    ORC = hyp('math_oracle')
    # the cmp_v2 context: topology, photon quality and common observables
    CTX = ['%s_%s' % (S, q) for q in cfanalysis.TOPO_FOR_BDT] + [q for q in COMMON0 if q in d]
    GAM = ['%s_gamma_xyz_bs_q2' % S, '%s_gamma_xyz_bs_m_miss2' % S, '%s_gamma_coll_q2' % S,
           '%s_gamma_coll_m_miss2' % S]
    SETS = [
        ('q3', X3), ('xyz_obs', XYZ), ('coll_obs', COLL), ('math_obs', MATH),
        ('xyz+coll', XYZ + COLL), ('xyz+math', XYZ + MATH), ('xyz+coll+math', XYZ + COLL + MATH),
        ('raw', RAWF), ('inv', INVF), ('raw+inv', RAWF + INVF), ('raw+inv+xyz', RAWF + INVF + XYZ),
        ('raw+inv+xyz+coll+math', RAWF + INVF + XYZ + COLL + MATH),
        ('CONTROL xyz+oracle', XYZ + ORC), ('CONTROL xyz+gen_q2_smeared', XYZ + ['gen_q2_smeared']),
        ('ctx xyz', XYZ + CTX), ('ctx xyz+coll+math', XYZ + COLL + MATH + CTX),
        ('ctx all', XYZ + COLL + MATH + GAM + CTX),
    ]
    LARGE = ('xyz_obs', 'xyz+coll+math')
    res['bdt'] = {}
    PRED = {}
    for name, feats in SETS:
        feats = [f for f in feats if f in d]
        X = np.stack([d[f][idx] for f in feats], 1).astype(np.float32)
        for pname in ('long', 'short'):
            if pname != 'long' and name not in LARGE:
                continue
            runs = []
            for sd in (seeds if pname == 'long' else seeds[:1]):
                t0 = time.time()
                P, iters = oof(X, y, sw, args.folds, sd, PARAMS[pname])
                r = metrics(P, y, sw, wi, args.n_ref)
                r['iters'], r['seconds'] = iters, round(time.time() - t0, 1)
                runs.append(r)
                if pname == 'long' and sd == seeds[0]:
                    PRED[name] = P.astype(np.float32)
                log('%-26s %-5s seed %d  %2d feats  b10 %.4f %.3f %.3f  b20 %.4f %.3f %.3f  unb %.4f %.3f %.3f  '
                    'logloss %.4f  iters %s  %.0fs'
                    % (name, pname, sd, len(feats), r['b10']['r_taumu'], r['b10']['r_ds'], r['b10']['r_dsst'],
                       r['b20']['r_taumu'], r['b20']['r_ds'], r['b20']['r_dsst'], r['unb']['r_taumu'],
                       r['unb']['r_ds'], r['unb']['r_dsst'], r['logloss'], iters, r['seconds']))
            res['bdt']['%s|%s' % (name, pname)] = dict(features=feats, runs=runs)
            save()

    pf = os.path.join(args.cache_dir, 'hypinfo_predictions_%s.npz' % os.path.basename(os.path.abspath(args.outdir)))
    np.savez(pf, y=y, wi=wi, sw=sw, idx=idx, **dict((k.replace(' ', '_'), v) for k, v in PRED.items()))
    cfio._link(pf, linkd, log)

    # ---- paired bootstrap over the events: the statistical uncertainty of the
    # figure of merit, and of the difference between two input sets (same
    # events, so the differences are much less noisy than the values) --------
    FOMS = ('b10', 'b20', 'unb')

    def foms(P, sel):
        pt, ps = P[sel, 2] + P[sel, 3], P[sel, 1] + P[sel, 3]
        yy, ww = y[sel], wi[sel]
        out = []
        for nb in (10, 20):
            e = [cfcore.equal_pop_edges(pt, nb, ww), cfcore.equal_pop_edges(ps, nb, ww)]
            f = cfcore.fisher_ratios([[pt[yy == c], ps[yy == c]] for c in range(4)],
                                     [ww[yy == c] for c in range(4)], e, n_ref=args.n_ref)
            out.append([f['r_taumu'], f['r_ds'], f['r_dsst']])
        f = unbinned_fisher(P[sel], yy, ww, args.n_ref)
        out.append([f['r_taumu'], f['r_ds'], f['r_dsst']])
        return out
    brng = np.random.default_rng(11)
    names = list(PRED)
    boot = dict((k, []) for k in names)
    for ib in range(args.boot):
        sel = brng.integers(0, len(y), len(y))
        for k in names:
            boot[k].append(foms(PRED[k], sel))
        if ib % 10 == 9:
            log('bootstrap %d/%d' % (ib + 1, args.boot))
    boot = dict((k, np.array(v)) for k, v in boot.items())       # [replica, fom, quantity]
    res['bootstrap'] = {}
    for k in names:
        res['bootstrap'][k] = {}
        for j, fm in enumerate(FOMS):
            v = boot[k][:, j, :]
            rr = dict(mean=v.mean(0).tolist(), std=v.std(0).tolist())
            for ref in ('xyz_obs', 'ctx xyz'):
                rel = v / boot[ref][:, j, :] - 1.
                rr['rel_to_' + ref] = dict(mean=rel.mean(0).tolist(), std=rel.std(0).tolist())
            res['bootstrap'][k][fm] = rr
            rel = v / boot['xyz_obs'][:, j, :] - 1.
            log('bootstrap %-26s %s  tau/mu %.4f+-%.4f  R(Ds) %.3f+-%.3f  R(Ds*) %.3f+-%.3f   vs xyz_obs: '
                '%+.1f+-%.1f%%  %+.1f+-%.1f%%  %+.1f+-%.1f%%'
                % ((k, fm) + tuple(x for i in range(3) for x in (v[:, i].mean(), v[:, i].std()))
                   + tuple(100 * x for i in range(3) for x in (rel[:, i].mean(), rel[:, i].std()))))
    save()

    # ---- 5. binned fits over hypothesis combinations --------------------------
    hyps = ('coll', 'xyz_bs', 'math1_bs', 'math2_bs', 'math_avg')
    var = ['%s_%s_%s' % (S, h, q) for h in hyps for q in ('q2', 'm_miss2', 'e_star_mu')]
    var = [v for v in var if v in d]
    res['binned'] = {}
    for k, nb in ((1, 20), (2, 12), (3, 6)):
        out = []
        for combo in itertools.combinations(var, k):
            f = cfanalysis._fisher_axes(d, m, w, cls, list(combo), nb, args.n_ref)
            out.append(dict(vars=list(combo), **{q: f.get(q) for q in ('r_taumu', 'r_ds', 'r_dsst', 'min_mc', 'frac_low')}))
        res['binned']['%dd' % k] = out
        log('binned %dD: %d combinations' % (k, len(out)))
        save()
    log('done')


if __name__ == '__main__':
    main()
