'''
Pure-numpy metrics of the fit-scheme comparison. No ROOT, no uproot, no
matplotlib: everything here is unit-tested in test_compare_fits.py.

  robust_stats       bias and width of a residual distribution, robust to tails
  rank_correlation   Spearman correlation reco vs gen
  auc, roc_curve     separation of two classes by one score (weighted)
  separation         <S^2> = 1/2 sum (p - q)^2 / (p + q) on a binning
  equal_pop_edges    bin edges with equal (weighted) population
  fisher_ratios      expected precision of R(Ds), R(Ds*) from a binned
                     template fit of the four signal yields (Fisher matrix)
'''

import numpy as np

NAN = float('nan')


##########################################################################################
#####      RESIDUALS
##########################################################################################

def wrap_angle(x):
    '''x in [-pi, pi).'''
    return (np.asarray(x, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def robust_stats(r, w=None):
    '''Statistics of a residual distribution r (NaN entries ignored):
        n        finite entries
        frac_ok  fraction of entries that are finite
        median   bias
        s68      half the width of the central 68% interval (sigma for a
                 Gaussian, robust to tails)
        s95      half the width of the central 95% interval / 1.96
        mad      1.4826 * median absolute deviation
        rms5     RMS within median +- 5 s68
        tail3    fraction with |r - median| > 3 s68
    Weights w (optional) enter the quantiles.'''
    r = np.asarray(r, dtype=float)
    ok = np.isfinite(r)
    out = dict(n=int(ok.sum()), frac_ok=float(ok.mean()) if len(r) else NAN)
    if ok.sum() < 5:
        out.update(median=NAN, s68=NAN, s95=NAN, mad=NAN, rms5=NAN, tail3=NAN)
        return out
    x = r[ok]
    ww = None if w is None else np.asarray(w, dtype=float)[ok]
    q = weighted_quantile(x, [0.025, 0.16, 0.5, 0.84, 0.975], ww)
    med = q[2]
    s68 = 0.5 * (q[3] - q[1])
    s95 = 0.5 * (q[4] - q[0]) / 1.959964
    mad = 1.4826 * weighted_quantile(np.abs(x - med), [0.5], ww)[0]
    core = np.abs(x - med) < 5 * s68 if s68 > 0 else np.ones_like(x, dtype=bool)
    if ww is None:
        rms5 = float(np.sqrt(np.mean((x[core] - med) ** 2))) if core.any() else NAN
        tail3 = float(np.mean(np.abs(x - med) > 3 * s68)) if s68 > 0 else NAN
    else:
        rms5 = float(np.sqrt(np.average((x[core] - med) ** 2, weights=ww[core]))) if core.any() else NAN
        tail3 = float(np.average(np.abs(x - med) > 3 * s68, weights=ww)) if s68 > 0 else NAN
    out.update(median=float(med), s68=float(s68), s95=float(s95), mad=float(mad),
               rms5=rms5, tail3=tail3)
    return out


def weighted_quantile(x, qs, w=None):
    '''Quantiles of x (finite) with weights w; w None: np.quantile (linear).'''
    x = np.asarray(x, dtype=float)
    if w is None:
        return np.quantile(x, qs)
    w = np.asarray(w, dtype=float)
    order = np.argsort(x, kind='mergesort')
    x, w = x[order], w[order]
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return np.interp(qs, cw, x)


def rankdata(x):
    '''Average ranks (ties share their mean rank), 0-based.'''
    x = np.asarray(x)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    xs = x[order]
    # average over tied groups
    _, first, counts = np.unique(xs, return_index=True, return_counts=True)
    tied = counts > 1
    for f, c in zip(first[tied], counts[tied]):
        ranks[order[f:f + c]] = f + 0.5 * (c - 1)
    return ranks


def rank_correlation(a, b):
    '''Spearman correlation of the finite pairs.'''
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return NAN
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else NAN


##########################################################################################
#####      SEPARATION OF TWO CLASSES
##########################################################################################

def auc(sig, bkg, w_sig=None, w_bkg=None):
    '''Area under the ROC curve = P(score_sig > score_bkg) (+ 1/2 P(tie)),
    finite entries only. 0.5 = no separation; values below 0.5 mean the
    observable separates with the opposite sign (use auc_oriented).'''
    s = np.asarray(sig, dtype=float)
    b = np.asarray(bkg, dtype=float)
    ws = np.ones(len(s)) if w_sig is None else np.asarray(w_sig, dtype=float)
    wb = np.ones(len(b)) if w_bkg is None else np.asarray(w_bkg, dtype=float)
    ms, mb = np.isfinite(s), np.isfinite(b)
    s, ws, b, wb = s[ms], ws[ms], b[mb], wb[mb]
    if len(s) == 0 or len(b) == 0:
        return NAN
    x = np.concatenate([s, b])
    is_s = np.concatenate([np.ones(len(s), bool), np.zeros(len(b), bool)])
    w = np.concatenate([ws, wb])
    order = np.argsort(x, kind='mergesort')
    x, is_s, w = x[order], is_s[order], w[order]
    # group ties
    uniq, start = np.unique(x, return_index=True)
    ends = np.append(start[1:], len(x))
    wb_grp = np.add.reduceat(np.where(is_s, 0., w), start)
    ws_grp = np.add.reduceat(np.where(is_s, w, 0.), start)
    wb_below = np.cumsum(wb_grp) - wb_grp
    num = np.sum(ws_grp * (wb_below + 0.5 * wb_grp))
    return float(num / (ws_grp.sum() * wb_grp.sum()))


def auc_oriented(sig, bkg, w_sig=None, w_bkg=None):
    '''(max(AUC, 1 - AUC), +1 if sig is above bkg, -1 otherwise).'''
    a = auc(sig, bkg, w_sig, w_bkg)
    if not np.isfinite(a):
        return NAN, 0
    return (a, +1) if a >= 0.5 else (1. - a, -1)


def roc_curve(sig, bkg, n_points=200, direction=+1):
    '''(eff_bkg, eff_sig) for thresholds over the pooled finite scores,
    cutting score > t (direction +1) or score < t (-1).'''
    s = np.asarray(sig, dtype=float)
    b = np.asarray(bkg, dtype=float)
    s, b = s[np.isfinite(s)], b[np.isfinite(b)]
    if len(s) == 0 or len(b) == 0:
        return np.array([0., 1.]), np.array([0., 1.])
    if direction < 0:
        s, b = -s, -b
    thr = np.quantile(np.concatenate([s, b]), np.linspace(0, 1, n_points))[::-1]
    ss, bs = np.sort(s), np.sort(b)
    eff_s = 1. - np.searchsorted(ss, thr, side='right') / len(ss)
    eff_b = 1. - np.searchsorted(bs, thr, side='right') / len(bs)
    return np.concatenate([[0.], eff_b, [1.]]), np.concatenate([[0.], eff_s, [1.]])


def equal_pop_edges(x, nbins, w=None):
    '''Edges with equal (weighted) population of the finite x; duplicated
    edges (discrete values) merged.'''
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return np.array([-np.inf, np.inf])
    ww = None if w is None else np.asarray(w, dtype=float)[ok]
    e = weighted_quantile(x[ok], np.linspace(0, 1, nbins + 1), ww)
    e = np.unique(e)
    if len(e) < 2:
        e = np.array([e[0] - 0.5, e[0] + 0.5])
    e[0], e[-1] = -np.inf, np.inf
    return e


def separation(sig, bkg, edges, w_sig=None, w_bkg=None):
    '''<S^2> = 1/2 sum_i (p_i - q_i)^2 / (p_i + q_i), in [0, 1], on `edges`
    (plus one bin for non-finite values).'''
    p = _hist_with_nan(sig, edges, w_sig)
    q = _hist_with_nan(bkg, edges, w_bkg)
    if p.sum() <= 0 or q.sum() <= 0:
        return NAN
    p, q = p / p.sum(), q / q.sum()
    s = p + q
    m = s > 0
    return float(0.5 * np.sum((p[m] - q[m]) ** 2 / s[m]))


def _hist_with_nan(x, edges, w=None):
    x = np.asarray(x, dtype=float)
    w = np.ones(len(x)) if w is None else np.asarray(w, dtype=float)
    ok = np.isfinite(x)
    h, _ = np.histogram(x[ok], bins=edges, weights=w[ok])
    return np.append(h, w[~ok].sum())


##########################################################################################
#####      EXPECTED PRECISION ON R(Ds), R(Ds*)
##########################################################################################

def bin_index(columns, edges_list):
    '''Global bin index of every event in an N-dimensional binning; events
    with any non-finite coordinate go to one extra bin (the last). Returns
    (index array, number of bins).'''
    n = len(columns[0])
    idx = np.zeros(n, dtype=np.int64)
    bad = np.zeros(n, dtype=bool)
    stride = 1
    for col, edges in zip(columns, edges_list):
        col = np.asarray(col, dtype=float)
        bad |= ~np.isfinite(col)
        k = np.clip(np.searchsorted(edges, col, side='right') - 1, 0, len(edges) - 2)
        idx += k * stride
        stride *= len(edges) - 1
    idx[bad] = stride
    return idx, stride + 1


def fisher_yields(templates, yields):
    '''Covariance of the yields of a binned extended Poisson fit
        nu_i = sum_c N_c t_ci,   t_c normalised templates (rows),
    at the true values: F_ab = sum_i t_ai t_bi / nu_i, cov = F^-1.
    Bins with nu = 0 are dropped. Returns (cov, F).'''
    t = np.asarray(templates, dtype=float)
    N = np.asarray(yields, dtype=float)
    nu = N @ t
    m = nu > 0
    tm = t[:, m]
    F = (tm / nu[m]) @ tm.T
    try:
        cov = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        cov = np.full_like(F, np.nan)
    return cov, F


def ratio_rel_error(cov, yields, num, den):
    '''sigma(N_num / N_den) / (N_num / N_den) from the yield covariance.'''
    Nn, Nd = yields[num], yields[den]
    v = cov[num, num] / Nn ** 2 + cov[den, den] / Nd ** 2 - 2 * cov[num, den] / (Nn * Nd)
    return float(np.sqrt(v)) if v >= 0 else NAN


# class order of the four signal channels (gen_cand codes)
CLASS_DSMU, CLASS_DSSTMU, CLASS_DSTAU, CLASS_DSSTTAU = 0, 1, 2, 3


def fisher_ratios(columns_by_class, weights_by_class, edges_list, n_ref=None):
    '''Expected relative precision on R(Ds) = N(Ds tau)/N(Ds mu) and R(Ds*) =
    N(Ds* tau)/N(Ds* mu) from a binned fit of the four signal yields to the
    distributions of the given columns (perfect templates, signal only, no
    background, no systematics: a FIGURE OF MERIT to compare observables, not
    the sensitivity of the analysis).

      columns_by_class   [class][dim] -> array, class order 0..3 (gen_cand)
      weights_by_class   [class] -> per-event weights (yields = their sums)
      edges_list         [dim] -> bin edges (non-finite values: extra bin)
      n_ref              if given, the result is scaled to N(Ds mu) = n_ref
                         (sigma_rel goes as 1/sqrt(N))

    Returns dict(r_ds, r_dsst: sigma_rel; r_ds_count, r_dsst_count: the
    counting limit sqrt(1/N_tau + 1/N_mu), i.e. perfect separation; dil_ds,
    dil_dsst: their ratio >= 1; r_taumu, r_taumu_count, dil_taumu: the same
    for N(tau)/N(mu) from a two-yield fit (Ds and Ds* merged); nbins; min_mc: the smallest number of MC
    events (all classes, unweighted) in a populated bin, and frac_low: the
    fraction of populated bins with fewer than 10. The templates come from the
    same finite MC, so sparse bins make the precision look better than it is:
    compare observables at similar min_mc).'''
    H, Hraw, yields = [], [], []
    nb = None
    for cols, w in zip(columns_by_class, weights_by_class):
        idx, nb = bin_index(cols, edges_list)
        H.append(np.bincount(idx, weights=w, minlength=nb))
        Hraw.append(np.bincount(idx, minlength=nb))
        yields.append(np.sum(w))
    H = np.array(H)
    raw = np.array(Hraw).sum(axis=0)
    yields = np.array(yields, dtype=float)
    scale = 1.
    if n_ref is not None and yields[CLASS_DSMU] > 0:
        scale = n_ref / yields[CLASS_DSMU]
    t = H / yields[:, None]
    Ns = yields * scale
    cov, _ = fisher_yields(t, Ns)
    out = dict(
        r_ds=ratio_rel_error(cov, Ns, CLASS_DSTAU, CLASS_DSMU),
        r_dsst=ratio_rel_error(cov, Ns, CLASS_DSSTTAU, CLASS_DSSTMU),
        r_ds_count=float(np.sqrt(1. / Ns[CLASS_DSTAU] + 1. / Ns[CLASS_DSMU])),
        r_dsst_count=float(np.sqrt(1. / Ns[CLASS_DSSTTAU] + 1. / Ns[CLASS_DSSTMU])),
        nbins=int(nb),
    )
    out['dil_ds'] = out['r_ds'] / out['r_ds_count']
    out['dil_dsst'] = out['r_dsst'] / out['r_dsst_count']
    # tau/mu only: two yields, N(tau) = N(Ds tau) + N(Ds* tau) and N(mu), each
    # template the sum of its Ds and Ds* channels at their sample proportions.
    # Insensitive to how well Ds and Ds* are told apart, which in the
    # four-yield fit can dominate r_ds and r_dsst.
    Hl = np.array([H[CLASS_DSTAU] + H[CLASS_DSSTTAU], H[CLASS_DSMU] + H[CLASS_DSSTMU]])
    yl = np.array([yields[CLASS_DSTAU] + yields[CLASS_DSSTTAU], yields[CLASS_DSMU] + yields[CLASS_DSSTMU]])
    Nl = yl * scale
    cov2, _ = fisher_yields(Hl / yl[:, None], Nl)
    out['r_taumu'] = ratio_rel_error(cov2, Nl, 0, 1)
    out['r_taumu_count'] = float(np.sqrt(1. / Nl[0] + 1. / Nl[1]))
    out['dil_taumu'] = out['r_taumu'] / out['r_taumu_count']
    pop = raw[raw > 0]
    out['min_mc'] = int(pop.min()) if pop.size else 0
    out['frac_low'] = float(np.mean(pop < 10)) if pop.size else NAN
    return out
