'''
The metrics of the fit-scheme comparison, computed on the selected signal
candidates. Each function returns plain python (dicts, lists, floats) that goes
into results.json and into the report tables; plots are made separately
(cfplots) from the same arrays.

  efficiencies      fit success, vertex-cut efficiency, real neutrino solutions
  fit_quality       vertex-probability distributions (flatness, P < 1%)
  masses            Ds and phi mass peaks, Ds pT resolution
  vertex            Bs vertex residuals and pulls along / across the true Bs
                    flight, Bs flight direction, PV
  resolution        reco - gen for every scheme x hypothesis x observable
  separation        tau / mu (and Ds / Ds*) separation of every observable:
                    AUC, <S^2>, expected precision on R(Ds), R(Ds*)
  multidim          the same precision with 2D and 3D binnings
  bdt               a multiclass BDT per scheme (and all schemes together)
'''

import time

import numpy as np

from cfcore import (robust_stats, rank_correlation, auc_oriented, separation,
                    equal_pop_edges, fisher_ratios, NAN)
from cfio import (CLASSES, GROUPS, VERTICES, TARGETS, GAMMA_TARGETS, DERIVED_HYPS,
                  ORACLE_HYPS, residual)

PDG_DS, PDG_PHI = 1.96835, 1.019461


def _cls(d, m):
    return d['gen_cand'][m].astype(int)


def _f(x):
    x = float(x)
    return x if np.isfinite(x) else None


##########################################################################################
#####      EFFICIENCIES
##########################################################################################

def efficiencies(layout, d, m, cuts=(0.01, 0.01)):
    '''Per scheme and class, on the selected candidates (before requiring the
    Bs vertex in every scheme):
        good_<v>     fit usable
        vtx_cut      P(V_phi) > 0.01 and P(V_Ds) > 0.01 in this scheme (the AN
                     cuts; the ntuplizer keeps a candidate if phic or free pass)
        math_real    real neutrino solutions along PV -> V_Bs
        bs_status_<k>  fraction with Bs-vertex status k (0 good, 1 no fit,
                     2 chi2 < 0, 3 covariance not positive definite)'''
    cls = d['gen_cand'].astype(int)
    out = {}
    for s in layout.schemes:
        r = {}
        for c in CLASSES:
            mc = m & (cls == c)
            nc = max(int(mc.sum()), 1)
            e = {}
            for v in VERTICES:
                b = '%s_%s_good' % (s, v)
                if b in d:
                    e['good_' + v] = float(np.sum(d[b][mc] > 0.5)) / nc
            pp, pd = d.get('%s_phi_prob' % s), d.get('%s_ds_prob' % s)
            if pp is not None and pd is not None:
                e['vtx_cut'] = float(np.sum((pp[mc] > cuts[0]) & (pd[mc] > cuts[1]))) / nc
            mr = d.get('%s_math_real_bs' % s)
            if mr is not None:
                e['math_real'] = float(np.sum(mr[mc] > 0.5)) / nc
            st = d.get('%s_bs_status' % s)
            if st is not None:
                for k in (0, 1, 2, 3):
                    e['bs_status_%d' % k] = float(np.sum(st[mc] == k)) / nc
            e['n'] = int(mc.sum())
            r[str(c)] = e
        out[s] = r
    return out


##########################################################################################
#####      FIT QUALITY
##########################################################################################

def fit_quality(layout, d, m):
    '''Per scheme, vertex and channel group (mu: Ds(*) mu nu, tau: Ds(*) tau
    nu): fraction with P < 0.01 and P < 0.05, Kolmogorov distance of the P
    distribution from uniform (0 = flat: the vertex errors describe the
    scatter), mean chi2/ndof, ndof.'''
    cls = d['gen_cand'].astype(int)
    out = {}
    for s in layout.schemes:
        rs = {}
        for v in VERTICES:
            pb, cb, nb = ('%s_%s_%s' % (s, v, q) for q in ('prob', 'chi2', 'ndof'))
            if pb not in d:
                continue
            rv = {}
            for g in ('mu', 'tau'):
                mg = m & np.isin(cls, GROUPS[g])
                p = d[pb][mg]
                p = p[np.isfinite(p)]
                if len(p) < 10:
                    continue
                ps = np.sort(p)
                ecdf = np.arange(1, len(ps) + 1) / len(ps)
                ks = float(np.max(np.abs(ecdf - ps)))
                chi2, ndof = d[cb][mg], d[nb][mg]
                ok = np.isfinite(chi2) & (ndof > 0)
                rv[g] = dict(n=int(len(p)), p_lt_001=float(np.mean(p < 0.01)),
                             p_lt_005=float(np.mean(p < 0.05)), ks_uniform=ks,
                             chi2_ndof=_f(np.mean(chi2[ok] / ndof[ok])) if ok.any() else None,
                             ndof=_f(np.median(ndof[ok])) if ok.any() else None)
            rs[v] = rv
        out[s] = rs
    return out


##########################################################################################
#####      MASSES
##########################################################################################

def masses(layout, d, m):
    '''Ds mass of each scheme where it is not constrained (phic, free), phi
    mass where not constrained (free, dsc), the scheme-independent m(KK pi)
    and m(KK), all on the selected candidates (every scheme's own good fit):
    median - PDG, s68, tail3. Ds pT: reco / gen - 1 per scheme.'''
    out = {}
    rows = []
    for s in layout.schemes:
        for v, pdg in (('ds', PDG_DS), ('phi', PDG_PHI)):
            b = '%s_%s_mass' % (s, v)
            if b not in d:
                continue
            x = d[b][m]
            st = robust_stats(x - pdg)
            if st['s68'] is not None and np.isfinite(st['s68']) and st['s68'] < 1e-5:
                continue       # constrained in this scheme
            rows.append(('%s_%s' % (s, v), b, pdg, st))
    for b, pdg in (('kkpi_mass', PDG_DS), ('kk_mass', PDG_PHI)):
        if b in d:
            rows.append((b, b, pdg, robust_stats(d[b][m] - pdg)))
    out['mass'] = dict((name, dict(branch=b, pdg=pdg, **st)) for name, b, pdg, st in rows)
    dspt = {}
    for s in layout.schemes:
        b = '%s_ds_pt' % s
        if b in d and 'gen_ds_pt' in d:
            dspt[s] = robust_stats(d[b][m] / d['gen_ds_pt'][m] - 1.)
    out['ds_pt'] = dspt
    return out


##########################################################################################
#####      VERTEX
##########################################################################################

def bs_vertex_residuals(d, s, m):
    '''Residual of V_Bs w.r.t. the true Bs decay vertex, decomposed along the
    true Bs flight (long) and across it (t1 in the transverse plane, t2 the
    third axis); pulls with the vertex covariance. Returns dict of arrays.'''
    u = np.stack([np.cos(d['gen_bs_phi'][m]), np.sin(d['gen_bs_phi'][m]),
                  np.sinh(d['gen_bs_eta'][m])])
    u /= np.linalg.norm(u, axis=0)
    zhat = np.array([0., 0., 1.])[:, None]
    e1 = np.cross(u.T, zhat.T).T
    e1 /= np.linalg.norm(e1, axis=0)
    e2 = np.cross(u.T, e1.T).T
    r = np.stack([d['%s_bs_%s' % (s, a)][m] - d['gen_bs_sv_%s' % a][m] for a in 'xyz'])
    C = np.empty((3, 3, r.shape[1]))
    names = {(0, 0): 'xx', (0, 1): 'xy', (0, 2): 'xz', (1, 1): 'yy', (1, 2): 'yz', (2, 2): 'zz'}
    for (i, j), nm in names.items():
        C[i, j] = C[j, i] = d['%s_bs_cov_%s' % (s, nm)][m]
    out = {}
    for lab, ax in (('long', u), ('t1', e1), ('t2', e2)):
        res = np.sum(r * ax, axis=0)
        var = np.einsum('in,ijn,jn->n', ax, C, ax)
        with np.errstate(invalid='ignore', divide='ignore'):
            out[lab] = res
            out[lab + '_pull'] = res / np.sqrt(var)
    # flight direction: PV (reco) -> V_Bs vs the true Bs momentum
    fd = np.stack([d['%s_bs_%s' % (s, a)][m] - d['pv_%s' % a][m] for a in 'xyz'])
    with np.errstate(invalid='ignore'):
        cosang = np.sum(fd * u, axis=0) / np.linalg.norm(fd, axis=0)
    out['angle'] = np.arccos(np.clip(cosang, -1., 1.))
    if 'gen_bs_lxyz' in d and '%s_bs_lxyz' % s in d:
        out['lxyz_rel'] = d['%s_bs_lxyz' % s][m] / d['gen_bs_lxyz'][m] - 1.
    return out


def vertex(layout, d, m):
    '''Per scheme and channel group, robust statistics of the Bs-vertex
    residuals (long, t1, t2, in micron), pulls, flight-direction angle (mrad),
    relative flight length; and the PV residual (scheme independent).'''
    cls = d['gen_cand'].astype(int)
    out = {}
    need = ['gen_bs_sv_x', 'gen_bs_eta', 'gen_bs_phi', 'pv_x']
    if not all(b in d for b in need):
        return out
    for s in layout.schemes:
        if '%s_bs_cov_xx' % s not in d:
            continue
        rs = {}
        for g in ('Ds mu', 'Ds* mu', 'Ds tau', 'Ds* tau', 'mu', 'tau'):
            mg = m & np.isin(cls, GROUPS[g])
            if mg.sum() < 10:
                continue
            res = bs_vertex_residuals(d, s, mg)
            rg = {}
            for k, v in res.items():
                scale = 1e4 if k in ('long', 't1', 't2') else (1e3 if k == 'angle' else 1.)
                rg[k] = robust_stats(v * scale)
            rs[g] = rg
        out[s] = rs
    if 'gen_pv_x' in d:
        pv = {}
        for a in 'xyz':
            pv[a] = robust_stats((d['pv_%s' % a][m] - d['gen_pv_%s' % a][m]) * 1e4)
        out['pv'] = pv
    return out


##########################################################################################
#####      RESOLUTION
##########################################################################################

def hyp_branch(s, h, q):
    return '%s_%s_%s' % (s, h, q)


def resolution(layout, d, m, log=print):
    '''reco - gen of every observable of every (scheme, hypothesis), per
    channel group: robust statistics (median = bias, s68, tail3) and the rank
    correlation reco vs gen.'''
    cls = d['gen_cand'].astype(int)
    out = {}
    for s in layout.schemes:
        for h in layout.hyps[s] + layout.gamma_hyps[s]:
            targets = GAMMA_TARGETS if h.startswith('gamma_') else TARGETS
            for q in layout.hyp_obs(s, h):
                if q not in targets:
                    continue
                gb, kind = targets[q]
                b = hyp_branch(s, h, q)
                if b not in d or gb not in d:
                    continue
                r = residual(d[b], d[gb], kind)
                for g, cs in GROUPS.items():
                    mg = m & np.isin(cls, cs)
                    if mg.sum() < 10:
                        continue
                    st = robust_stats(r[mg])
                    st['corr'] = rank_correlation(d[b][mg], d[gb][mg])
                    out.setdefault(s, {}).setdefault(h, {}).setdefault(q, {})[g] = st
    return out


##########################################################################################
#####      SEPARATION
##########################################################################################

PAIRS = {
    'tau_vs_mu'        : ((2, 3), (0, 1)),
    'ds_tau_vs_mu'     : ((2,), (0,)),
    'dsst_tau_vs_mu'   : ((3,), (1,)),
    'dsst_vs_ds_mu'    : ((1,), (0,)),
}


def _values(d, b, m, cls, cs):
    return d[b][m & np.isin(cls, cs)]


def observable_list(layout):
    '''Every observable entering the separation study: (key, branch, scheme,
    hypothesis or None, observable name).'''
    obs = []
    for s in layout.schemes:
        for h in layout.hyps[s] + layout.gamma_hyps[s]:
            for q in layout.hyp_obs(s, h):
                obs.append((hyp_branch(s, h, q), hyp_branch(s, h, q), s, h, q))
        for q in layout.scheme_obs(s):
            b = '%s_%s' % (s, q)
            obs.append((b, b, s, None, q))
    for q in layout.common_obs():
        obs.append((q, q, None, None, q))
    return obs


def separation_1d(layout, d, m, w, nbins=20, n_ref=1e5, log=print):
    '''For every observable: AUC (oriented) and <S^2> for each pair in PAIRS,
    and the expected precision on R(Ds), R(Ds*) from a fit of the four signal
    yields to its distribution (nbins equal-population bins + one bin for the
    candidates where it is not defined).'''
    cls = d['gen_cand'].astype(int)
    out = {}
    obs = observable_list(layout)
    t0 = time.time()
    for i, (key, b, s, h, q) in enumerate(obs):
        if b not in d:
            continue
        x = d[b]
        if not np.any(np.isfinite(x[m])):
            continue
        r = dict(scheme=s, hyp=h, obs=q)
        for pname, (a, c) in PAIRS.items():
            ma, mb = m & np.isin(cls, a), m & np.isin(cls, c)
            auc_v, sign = auc_oriented(x[ma], x[mb])
            edges = equal_pop_edges(x[ma | mb], nbins)
            r[pname] = dict(auc=_f(auc_v), sign=sign, sep=_f(separation(x[ma], x[mb], edges)))
        edges = equal_pop_edges(x[m], nbins, w[m])
        cols = [[x[m & (cls == c)]] for c in CLASSES]
        ws = [w[m & (cls == c)] for c in CLASSES]
        fr = fisher_ratios(cols, ws, [edges], n_ref=n_ref)
        r['fisher'] = dict((k, _f(v)) for k, v in fr.items())
        out[key] = r
        if (i + 1) % 100 == 0:
            log('    %d / %d observables (%.0f s)' % (i + 1, len(obs), time.time() - t0))
    return out


def multidim_generator(d, m, w, n_ref=1e5):
    '''The multidimensional precisions with the generator-level observables:
    what a perfect Bs reconstruction would give (reference, not ranked).'''
    cls = d['gen_cand'].astype(int)
    out = {}
    for name, qs in MULTIDIM_SETS.items():
        bs = [TARGETS[q][0] for q in qs]
        if not all(b in d for b in bs):
            continue
        nb = MULTIDIM_BINS[len(qs)]
        edges = [equal_pop_edges(d[b][m], nb, w[m]) for b in bs]
        cols = [[d[b][m & (cls == c)] for b in bs] for c in CLASSES]
        ws = [w[m & (cls == c)] for c in CLASSES]
        out[name] = dict((k, _f(v)) for k, v in fisher_ratios(cols, ws, edges, n_ref=n_ref).items())
    return out


def degeneracy(layout, d, m, qs=('q2', 'm_miss2', 'e_star_mu')):
    '''Rank correlation between the observables of the same (scheme,
    hypothesis) that the multidimensional fits combine. |rho| ~ 1 means one
    is a monotonic function of the other and the 2D/3D fit adds nothing: the
    case of the collinear approximation, where every Lorentz invariant built
    from p(Bs) = p(Ds mu) M(Bs) / m(Ds mu) depends on m(Ds mu) only.'''
    out = {}
    for s in layout.schemes:
        for h in layout.hyps[s]:
            avail = layout.hyp_obs(s, h)
            r = {}
            for i, a in enumerate(qs):
                for b in qs[i + 1:]:
                    if a in avail and b in avail:
                        r['%s:%s' % (a, b)] = _f(rank_correlation(d[hyp_branch(s, h, a)][m], d[hyp_branch(s, h, b)][m]))
            if r:
                vals = [abs(v) for v in r.values() if v is not None]
                r['max_abs'] = max(vals) if vals else None
                out.setdefault(s, {})[h] = r
    return out


MULTIDIM_SETS = {
    'q2_mmiss2'        : ('q2', 'm_miss2'),
    'q2_estar'         : ('q2', 'e_star_mu'),
    'q2_mmiss2_estar'  : ('q2', 'm_miss2', 'e_star_mu'),
}
MULTIDIM_BINS = {2: 12, 3: 6}


def multidim(layout, d, m, w, n_ref=1e5):
    '''Expected precision on R(Ds), R(Ds*) from a fit to the joint
    distribution of a few observables of the same (scheme, hypothesis),
    equal-population bins in each (12 per axis in 2D, 6 in 3D).'''
    cls = d['gen_cand'].astype(int)
    out = {}
    for s in layout.schemes:
        for h in layout.hyps[s]:
            avail = layout.hyp_obs(s, h)
            for name, qs in MULTIDIM_SETS.items():
                if not all(q in avail for q in qs):
                    continue
                bs = [hyp_branch(s, h, q) for q in qs]
                nb = MULTIDIM_BINS[len(qs)]
                edges = [equal_pop_edges(d[b][m], nb, w[m]) for b in bs]
                cols = [[d[b][m & (cls == c)] for b in bs] for c in CLASSES]
                ws = [w[m & (cls == c)] for c in CLASSES]
                fr = fisher_ratios(cols, ws, edges, n_ref=n_ref)
                out.setdefault(s, {}).setdefault(h, {})[name] = dict((k, _f(v)) for k, v in fr.items())
    return out


##########################################################################################
#####      BDT
##########################################################################################

TOPO_FOR_BDT = ('mu_doca_ds_sig', 'mu_doca_ds', 'mu_poca_along_ds', 'mu_ip3d_bs_pv_sig',
                'mu_dist_to_bs_dir', 'bs_prob', 'bs_lxyz_sig', 'ds_lxyz_bs_sig', 'mcorr',
                'p_perp', 'cos_theta_k_pi', 'gamma_pt', 'gamma_dm', 'gamma_e_coll_over_pt')
BDT_HYP_SETS = {
    'coll'    : ('coll',),
    'xyz'     : ('xyz_bs',),
    'math'    : ('math1_bs', 'math2_bs'),
    'all_hyps': ('coll', 'xyz_bs', 'math1_bs', 'math2_bs'),
}
BDT_HYP_OBS = ('q2', 'm_miss2', 'pt_miss', 'e_star_mu', 'e_hash_mu',
               'cos_theta_phi_ds', 'cos_theta_mu_w', 'chi', 'pt')


def bdt_features(layout, s, hyp_set):
    '''Branches of the BDT of scheme s with the Bs hypotheses of hyp_set: their
    observables, the scheme's topology observables, the common ones. s None:
    every scheme together (hyp_set 'all_hyps').'''
    schemes = layout.schemes if s is None else [s]
    feats = list(layout.common_obs())
    for ss in schemes:
        for h in BDT_HYP_SETS[hyp_set]:
            if h not in layout.hyps[ss]:
                continue
            feats += [hyp_branch(ss, h, q) for q in BDT_HYP_OBS if q in layout.hyp_obs(ss, h)]
        feats += ['%s_%s' % (ss, q) for q in TOPO_FOR_BDT if q in layout.scheme_obs(ss)]
    return [f for f in feats if f in layout.keys]


def bdt(layout, d, m, w, sets=('coll', 'xyz', 'math', 'all_hyps'), combined=True,
        max_per_class=100000, folds=2, n_ref=1e5, seed=1, log=print):
    '''A multiclass gradient-boosted classifier of the four signal channels
    per scheme and set of Bs hypotheses (sklearn HistGradientBoosting, fixed
    hyperparameters, out-of-fold predictions). Reported:
        auc_tau_vs_mu        p(Ds tau) + p(Ds* tau), tau vs mu channels
        auc_ds_tau_vs_mu     the same within the Ds channels, and Ds*
        auc_dsst_vs_ds_mu    p(Ds*) between the two mu channels
        fisher               precision on R(Ds), R(Ds*) from a fit to the
                             (p_tau, p_Ds*) plane, 10 x 10 equal-population bins
    Returns (results, scores) where scores[name] = (mask of the events with a
    score, p_tau, p_dsst) for the plots.'''
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError:
        log('    sklearn not available: BDT comparison skipped')
        return {}, {}
    cls = d['gen_cand'].astype(int)
    rng = np.random.default_rng(seed)
    # the events used: at most max_per_class per class, the same for every BDT
    use = np.zeros(len(cls), dtype=bool)
    # the subsampling changes the class proportions: undo it in the Fisher
    # weights, so the yields stay those of the selected sample
    restore = np.ones(len(cls))
    for c in CLASSES:
        idx = np.nonzero(m & (cls == c))[0]
        n_all = len(idx)
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        use[idx] = True
        if len(idx):
            restore[idx] = n_all / float(len(idx))
    idx = np.nonzero(use)[0]
    fold = rng.integers(0, folds, len(idx))
    y = cls[idx]
    # balanced classes in the training
    cw = dict((c, len(y) / (4. * max(np.sum(y == c), 1))) for c in CLASSES)
    sw = np.array([cw[c] for c in y])

    jobs = [(s, hs) for s in layout.schemes for hs in sets]
    if combined and len(layout.schemes) > 1:
        jobs.append((None, 'all_hyps'))
    out, scores = {}, {}
    for s, hs in jobs:
        feats = bdt_features(layout, s, hs)
        name = '%s:%s' % (s if s is not None else 'all_schemes', hs)
        if not feats:
            continue
        X = np.stack([d[f][idx] for f in feats], axis=1).astype(np.float32)
        P = np.full((len(idx), 4), np.nan)
        t0 = time.time()
        for k in range(folds):
            tr, te = fold != k, fold == k
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                                 early_stopping=True, validation_fraction=0.15,
                                                 random_state=seed + k)
            clf.fit(X[tr], y[tr], sample_weight=sw[tr])
            P[te] = clf.predict_proba(X[te])
        p_tau = P[:, 2] + P[:, 3]
        p_st = P[:, 1] + P[:, 3]
        r = dict(n_features=len(feats), features=feats, seconds=round(time.time() - t0, 1))
        r['auc_tau_vs_mu'] = _f(auc_oriented(p_tau[np.isin(y, (2, 3))], p_tau[np.isin(y, (0, 1))])[0])
        r['auc_ds_tau_vs_mu'] = _f(auc_oriented(p_tau[y == 2], p_tau[y == 0])[0])
        r['auc_dsst_tau_vs_mu'] = _f(auc_oriented(p_tau[y == 3], p_tau[y == 1])[0])
        r['auc_dsst_vs_ds_mu'] = _f(auc_oriented(p_st[y == 1], p_st[y == 0])[0])
        wi = w[idx] * restore[idx]
        edges = [equal_pop_edges(p_tau, 10, wi), equal_pop_edges(p_st, 10, wi)]
        cols = [[p_tau[y == c], p_st[y == c]] for c in CLASSES]
        ws = [wi[y == c] for c in CLASSES]
        r['fisher'] = dict((k, _f(v)) for k, v in fisher_ratios(cols, ws, edges, n_ref=n_ref).items())
        out[name] = r
        scores[name] = (idx, p_tau, p_st)
        log('    BDT %-22s %3d features  AUC tau/mu %.4f  sigma tau/mu %.4f  R(Ds) %.4f  R(Ds*) %.4f  (%.0f s)'
            % (name, len(feats), r['auc_tau_vs_mu'] or NAN, r['fisher']['r_taumu'] or NAN,
               r['fisher']['r_ds'] or NAN, r['fisher']['r_dsst'] or NAN, r['seconds']))
    return out, scores


def is_ranked(h):
    return h not in ORACLE_HYPS
