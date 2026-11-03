#!/usr/bin/env python3
'''Per-candidate m(KKpi) uncertainty from the track helix covariances, with an
optional momentum-calibration resolution term, and its validation.

    sigma^2(m) = sum_{K1,K2,pi} J_i C_i J_i^T          C_i: (q/p, lambda, phi) block
    C_i[q/p,q/p] += (sigma_cal_i * q/p)^2               calibration term (no z)

Checks (all on truth-matched signal, any of the four channels):
  A  Jacobian: analytic vs finite differences
  B  toy round trip: generator momenta -> smeared with each track's own C ->
     mass -> pull against sigma_m; must be N(0,1) if the propagation is right
  C  same toy with an extra calibration-like smear f = 1 + s_cal * z, and
     sigma_m with the s_cal term (no z): the pull must return to N(0,1)
  D  the real reconstruction: (m_reco - m_gen) / sigma_m, m_gen from the
     generator-matched momenta of the three tracks; width vs sigma_m, eta, p
  E  the 'Ds peak per category' test done on MC: residual width per sigma_m
     quantile vs <sigma_m>

    source ../compare_fits/cfenv.sh
    python3 sigma_m_demo.py /pnfs/.../rds_bs_dslnu_ul18_v3_500f_dsc_02nov26 --files 20 -o demo
'''
import os
import sys
import glob
import json
import time
import argparse
import numpy as np

M_K, M_PI = 0.493677, 0.13957039
TRACKS = (('k1', M_K, 321), ('k2', M_K, 321), ('pi', M_PI, 211))
PARS = ('qoverp', 'lambda', 'phi')


def log(*a):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), ' '.join(str(x) for x in a)), flush=True)


# ---------------------------------------------------------------------------
# helix parameters <-> momenta
# ---------------------------------------------------------------------------
def helix(pt, eta, phi, q):
    '''(q/p, lambda, phi) from (pt, eta, phi, charge).'''
    return np.stack([q / (pt * np.cosh(eta)), np.arctan(np.sinh(eta)), phi], -1)


def p3_from_helix(h, q):
    p = q / h[..., 0]
    lam, phi = h[..., 1], h[..., 2]
    return np.stack([p * np.cos(lam) * np.cos(phi), p * np.cos(lam) * np.sin(phi), p * np.sin(lam)], -1)


def mass(p3s, masses):
    E = sum(np.sqrt(np.sum(p ** 2, -1) + m ** 2) for p, m in zip(p3s, masses))
    P = sum(p3s)
    return np.sqrt(np.maximum(E ** 2 - np.sum(P ** 2, -1), 0))


# ---------------------------------------------------------------------------
# the propagation
# ---------------------------------------------------------------------------
def jacobians(h, q, masses):
    '''dm/d(q/p, lambda, phi) per track, analytic. h: [track](n,3)'''
    p3 = [p3_from_helix(hi, qi) for hi, qi in zip(h, q)]
    Ei = [np.sqrt(np.sum(p ** 2, -1) + m ** 2) for p, m in zip(p3, masses)]
    E, P = sum(Ei), sum(p3)
    m = np.sqrt(E ** 2 - np.sum(P ** 2, -1))
    J = []
    for hi, qi, pi, ei in zip(h, q, p3, Ei):
        g = (E[:, None] * pi / ei[:, None] - P) / m[:, None]              # dm/dp_vec = gamma (beta_i - beta)
        p = qi / hi[:, 0]
        lam, phi = hi[:, 1], hi[:, 2]
        e_lam = np.stack([-np.sin(lam) * np.cos(phi), -np.sin(lam) * np.sin(phi), np.cos(lam)], 1)
        e_phi = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], 1)
        J.append(np.stack([-qi * p * np.sum(g * pi, 1),                    # d/d(q/p)
                           p * np.sum(g * e_lam, 1),                       # d/d(lambda)
                           p * np.cos(lam) * np.sum(g * e_phi, 1)], 1))    # d/d(phi), p cos(lambda) = pt
    return m, J


def sigma_m(h, q, C, masses, s_cal=None):
    '''m and sigma(m). C: [track](n,3,3); s_cal: [track](n,) relative width or None.'''
    m, J = jacobians(h, q, masses)
    var = np.zeros(len(m))
    for i, (Ji, Ci) in enumerate(zip(J, C)):
        if s_cal is not None:
            Ci = Ci.copy()
            Ci[:, 0, 0] += (s_cal[i] * h[i][:, 0]) ** 2
        var += np.einsum('ni,nij,nj->n', Ji, Ci, Ji)
    return m, np.sqrt(var), J


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------
def branches():
    b = ['gen_cand', 'kkpi_mass', 'event']
    for t, _, _ in TRACKS:
        b += ['%s_%s' % (t, q) for q in ('pt', 'eta', 'phi', 'charge', 'pt_raw', 'calib_f', 'cov_pos_def',
                                         'gen_pt', 'gen_eta', 'gen_phi', 'gen_pdgid')]
        b += ['%s_cov_%s_%s' % (t, PARS[a], PARS[c]) for a in range(3) for c in range(a, 3)]
    return b


def read(path, nfiles, seed):
    import uproot
    files = sorted(glob.glob(os.path.join(path, 'rds_chunk*.root'))) if os.path.isdir(path) else [path]
    rng = np.random.default_rng(seed)
    if nfiles and nfiles < len(files):
        files = sorted(rng.choice(files, nfiles, replace=False))
    log('%d files' % len(files))
    out = {}
    br = branches()
    for i, f in enumerate(files):
        t = uproot.open(f)['tree']
        a = t.arrays(br, library='np')
        sel = (a['gen_cand'] >= 0) & (a['gen_cand'] <= 3)
        for k in br:
            out.setdefault(k, []).append(a[k][sel])
        if i % 5 == 4:
            log('  read %d/%d' % (i + 1, len(files)))
    return files, dict((k, np.concatenate(v)) for k, v in out.items())


def prepare(d):
    n0 = len(d['gen_cand'])
    ok = np.ones(n0, bool)
    for t, _, pdg in TRACKS:
        ok &= np.abs(d['%s_gen_pdgid' % t]) == pdg                 # species
        ok &= np.isfinite(d['%s_gen_pt' % t]) & (d['%s_gen_pt' % t] > 0)
        for a in range(3):
            for c in range(a, 3):
                ok &= np.isfinite(d['%s_cov_%s_%s' % (t, PARS[a], PARS[c])])
    # the mass needs only the (q/p, lambda, phi) block: require THAT to be
    # positive definite (the 5x5 flag cov_pos_def fails for ~25% of the packed
    # tracks, mostly in the impact-parameter sector)
    for t, _, _ in TRACKS:
        B = np.empty((n0, 3, 3))
        for a in range(3):
            for c in range(a, 3):
                B[:, a, c] = B[:, c, a] = np.nan_to_num(d['%s_cov_%s_%s' % (t, PARS[a], PARS[c])])
        ok &= np.all(np.linalg.eigvalsh(B) > 0, axis=1)
    n_species = int(ok.sum())
    # one candidate per event (the first), so the events are independent
    _, first = np.unique(d['event'][ok], return_index=True)
    idx = np.nonzero(ok)[0][first]
    log('candidates: %d truth-matched signal, %d with species + positive-definite 3x3 block, %d one per event'
        % (n0, n_species, len(idx)))
    T = {}
    T['q'] = [d['%s_charge' % t][idx].astype(float) for t, _, _ in TRACKS]
    T['h'] = [helix(d['%s_pt' % t][idx].astype(float), d['%s_eta' % t][idx].astype(float),
                    d['%s_phi' % t][idx].astype(float), q) for (t, _, _), q in zip(TRACKS, T['q'])]
    T['hgen'] = [helix(d['%s_gen_pt' % t][idx].astype(float), d['%s_gen_eta' % t][idx].astype(float),
                       d['%s_gen_phi' % t][idx].astype(float), q) for (t, _, _), q in zip(TRACKS, T['q'])]
    C = []
    for t, _, _ in TRACKS:
        Ci = np.empty((len(idx), 3, 3))
        for a in range(3):
            for c in range(a, 3):
                Ci[:, a, c] = Ci[:, c, a] = d['%s_cov_%s_%s' % (t, PARS[a], PARS[c])][idx]
        f = np.nan_to_num(d['%s_calib_f' % t][idx].astype(float), nan=1.0)
        Ci[:, 0, :] /= f[:, None]
        Ci[:, :, 0] /= f[:, None]
        C.append(Ci)
    T['C'] = C
    T['kkpi_mass'] = d['kkpi_mass'][idx].astype(float)
    T['eta'] = [d['%s_eta' % t][idx].astype(float) for t, _, _ in TRACKS]
    T['pt'] = [d['%s_pt' % t][idx].astype(float) for t, _, _ in TRACKS]
    T['cls'] = d['gen_cand'][idx].astype(int)
    T['pd5'] = np.all(np.stack([d['%s_cov_pos_def' % t][idx] > 0 for t, _, _ in TRACKS]), 0)
    log('  of which with all three 5x5 covariances positive definite: %.3f' % T['pd5'].mean())
    return T


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def widths(x):
    '''robust core width (IQR/1.349), sigma_68 (half width of the central
    68%), RMS within +-5, mean.'''
    x = x[np.isfinite(x)]
    q = np.percentile(x, [15.865, 25, 50, 75, 84.135])
    c = x[np.abs(x - q[2]) < 5]
    return dict(iqr=float((q[3] - q[1]) / 1.349), s68=float((q[4] - q[0]) / 2), rms5=float(c.std()),
                median=float(q[2]), n=int(len(x)))


def binned(x, by, nb=10):
    e = np.percentile(by, np.linspace(0, 100, nb + 1))
    out = []
    for lo, hi in zip(e[:-1], e[1:]):
        s = (by >= lo) & (by <= hi)
        w = widths(x[s])
        w.update(lo=float(lo), hi=float(hi), mean_by=float(by[s].mean()), median_by=float(np.median(by[s])))
        out.append(w)
    return out


def toy_scal(h):
    '''A stand-in for the calibration's extra relative resolution: 0.6% in the
    barrel rising to 1.2% at |eta| = 2.5, + 0.1% per GeV of pt. Only used to
    test that the term is propagated correctly; the real one comes from
    rds_trkcalib.'''
    eta = np.arcsinh(np.tan(h[:, 1]))
    pt = 1.0 / np.abs(h[:, 0]) * np.cos(h[:, 1])
    return 0.006 * (1 + (np.abs(eta) / 2.5) ** 2) + 0.001 * pt


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('-o', '--outdir', default='demo')
    ap.add_argument('--files', type=int, default=20, help='random chunks to read (0: all)')
    ap.add_argument('--seed', type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    res = dict(args=vars(args))
    rng = np.random.default_rng(args.seed)
    masses = [m for _, m, _ in TRACKS]

    files, d = read(args.input, args.files, args.seed)
    res['files'] = files
    T = prepare(d)
    h, q, C = T['h'], T['q'], T['C']
    n = len(q[0])
    res['n'] = n

    # ---- 0. consistency with the ntuple --------------------------------------
    m, sm, J = sigma_m(h, q, C, masses)
    dm = m - T['kkpi_mass']
    res['m_vs_ntuple'] = dict(median_abs_diff=float(np.median(np.abs(dm))), max_abs_diff=float(np.max(np.abs(dm))))
    log('m(KKpi) recomputed vs kkpi_mass branch: median |diff| %.2e GeV, max %.2e' % (
        res['m_vs_ntuple']['median_abs_diff'], res['m_vs_ntuple']['max_abs_diff']))
    sq = np.percentile(sm, [5, 16, 50, 84, 95]) * 1e3
    res['sigma_m_MeV_quantiles_5_16_50_84_95'] = sq.tolist()
    log('sigma_m [MeV] quantiles 5/16/50/84/95%%: %s' % ' '.join('%.2f' % x for x in sq))
    # contribution of each track and parameter (diagonal terms, fraction of the variance)
    contrib = {}
    for (t, _, _), Ji, Ci in zip(TRACKS, J, C):
        for a, pa in enumerate(PARS):
            contrib['%s_%s' % (t, pa)] = float(np.mean(Ji[:, a] ** 2 * Ci[:, a, a] / sm ** 2))
    contrib['correlations'] = 1 - sum(contrib.values())
    res['variance_fractions'] = contrib
    log('variance fractions: ' + ' '.join('%s %.3f' % kv for kv in contrib.items()))

    # ---- A. Jacobian vs finite differences -------------------------------------
    k = min(2000, n)
    worst = 0.
    eps = (1e-6, 1e-6, 1e-6)
    for i in range(3):
        for a in range(3):
            hp = [x[:k].copy() for x in h]
            hm = [x[:k].copy() for x in h]
            step = eps[a] * (np.abs(h[i][:k, 0]) if a == 0 else 1.0)
            hp[i][:, a] += step
            hm[i][:, a] -= step
            num = (mass([p3_from_helix(x, y[:k]) for x, y in zip(hp, q)], masses)
                   - mass([p3_from_helix(x, y[:k]) for x, y in zip(hm, q)], masses)) / (2 * step)
            scale = np.maximum(np.abs(J[i][:k, a]), 1e-3 * np.abs(J[i][:k, a]).max())
            worst = max(worst, float(np.percentile(np.abs(num - J[i][:k, a]) / scale, 99)))
    res['A_jacobian_rel_diff_p99'] = worst
    log('A  Jacobian vs finite differences: 99th percentile of the relative difference %.1e' % worst)

    # ---- B. toy round trip ------------------------------------------------------
    def smear(hc, Cs, extra=None):
        out = []
        for hi, Ci, ex in zip(hc, Cs, extra if extra is not None else [None] * 3):
            L = np.linalg.cholesky(Ci)
            hs = hi + np.einsum('nij,nj->ni', L, rng.standard_normal((len(hi), 3)))
            if ex is not None:            # calibration-like: p -> f p, f = 1 + s z, direction unchanged
                hs[:, 0] /= 1 + ex * rng.standard_normal(len(hi))
            out.append(hs)
        return out
    mt = mass([p3_from_helix(x, y) for x, y in zip(T['hgen'], q)], masses)
    hs = smear(T['hgen'], C)
    ms, sms, _ = sigma_m(hs, q, C, masses)
    pullB = (ms - mt) / sms
    res['B_toy'] = dict(all=widths(pullB), vs_sigma_m=binned(pullB, sms))
    log('B  toy round trip, pull (m_smeared - m_gen)/sigma_m: core %.3f  s68 %.3f  median %+.3f'
        % (res['B_toy']['all']['iqr'], res['B_toy']['all']['s68'], res['B_toy']['all']['median']))

    # ---- C. toy with a calibration-like smear ------------------------------------
    s_cal = [toy_scal(x) for x in T['hgen']]
    hc = smear(T['hgen'], C, extra=s_cal)
    s_cal_reco = [toy_scal(x) for x in hc]        # evaluated on the smeared (= 'measured') momenta
    mc, smc0, _ = sigma_m(hc, q, C, masses)
    _, smc1, _ = sigma_m(hc, q, C, masses, s_cal=s_cal_reco)
    pC0, pC1 = (mc - mt) / smc0, (mc - mt) / smc1
    res['C_toy_cal'] = dict(without_scal=widths(pC0), with_scal=widths(pC1),
                            with_scal_vs_sigma_m=binned(pC1, smc1),
                            sigma_m_ratio_median=float(np.median(smc1 / smc0)))
    log('C  toy + calibration smear: pull core without s_cal %.3f, with s_cal %.3f (sigma_m x%.3f)'
        % (res['C_toy_cal']['without_scal']['iqr'], res['C_toy_cal']['with_scal']['iqr'],
           res['C_toy_cal']['sigma_m_ratio_median']))

    # ---- D. real reconstruction ---------------------------------------------------
    pullD = (m - mt) / sm
    eta_max = np.max(np.abs(np.stack(T['eta'])), 0)
    pt_min = np.min(np.stack(T['pt']), 0)
    res['D_reco'] = dict(all=widths(pullD), residual_MeV=dict((k2, v * (1e3 if k2 not in ('n',) else 1))
                                                              for k2, v in widths(m - mt).items()),
                         vs_sigma_m=binned(pullD, sm), vs_eta_max=binned(pullD, eta_max, 6),
                         vs_pt_min=binned(pullD, pt_min, 6),
                         pd5_true=widths(pullD[T['pd5']]), pd5_false=widths(pullD[~T['pd5']]))
    log('D  reco pull (m_reco - m_gen)/sigma_m: core %.3f  s68 %.3f  median %+.3f;  residual core %.2f MeV'
        % (res['D_reco']['all']['iqr'], res['D_reco']['all']['s68'], res['D_reco']['all']['median'],
           res['D_reco']['residual_MeV']['iqr']))
    log('     5x5 covariances all positive definite: pull core %.3f;  not: %.3f'
        % (res['D_reco']['pd5_true']['iqr'], res['D_reco']['pd5_false']['iqr']))
    M_DS = 1.96835
    pdg = (m - M_DS) / sm
    res['D_reco']['tails'] = dict(mean_pull2=float(np.mean(pullD ** 2)),
                                  mean_pull2_within3=float(np.mean(pullD[np.abs(pullD) < 3] ** 2)),
                                  frac_gt3=float(np.mean(np.abs(pullD) > 3)), frac_gt5=float(np.mean(np.abs(pullD) > 5)),
                                  mean_pull2_vs_pdg=float(np.mean(pdg ** 2)), core_vs_pdg=widths(pdg)['iqr'],
                                  m_gen_minus_pdg_MeV_p1_p50_p99=((np.percentile(mt, [1, 50, 99]) - M_DS) * 1e3).tolist())
    t = res['D_reco']['tails']
    log('     tails: <pull^2> %.2f (|pull|<3: %.2f), |pull|>3 %.4f, >5 %.4f (Gaussian 0.0027, 6e-7); '
        'vs PDG mass <pull^2> %.2f' % (t['mean_pull2'], t['mean_pull2_within3'], t['frac_gt3'], t['frac_gt5'],
                                         t['mean_pull2_vs_pdg']))
    for b in res['D_reco']['vs_sigma_m']:
        log('     sigma_m %5.2f-%5.2f MeV  pull core %.3f' % (b['lo'] * 1e3, b['hi'] * 1e3, b['iqr']))

    # ---- E. residual width per sigma_m category ------------------------------------
    E = binned(m - mt, sm, 10)
    x = np.array([b['median_by'] for b in E])     # median: the top decile has a long tail
    y = np.array([b['iqr'] for b in E])
    slope = float(np.sum(x * y) / np.sum(x * x))
    lin = np.polyfit(x, y, 1)
    res['E_categories'] = dict(bins=E, slope_through_origin=slope, linear_fit=lin.tolist())
    log('E  residual core width vs <sigma_m> in deciles: slope through 0 = %.3f; linear fit slope %.3f, '
        'intercept %.2f MeV' % (slope, lin[0], lin[1] * 1e3))

    json.dump(res, open(os.path.join(args.outdir, 'sigma_m_demo.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(args.outdir, 'sigma_m_demo.npz'), m=m, mt=mt, sm=sm, ms=ms, sms=sms, mc=mc,
                        smc0=smc0, smc1=smc1, eta_max=eta_max, pt_min=pt_min, cls=T['cls'])
    plots(args.outdir, res, dict(m=m, mt=mt, sm=sm, pullB=pullB, pC0=pC0, pC1=pC1, pullD=pullD))
    log('done')


def plots(outdir, res, a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.linspace(-5, 5, 201)
    gauss = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    ax = ax.ravel()
    ax[0].hist(a['sm'] * 1e3, 100, range=(0, 30), histtype='step', color='k')
    ax[0].set_xlabel(r'$\sigma_m$ [MeV]')
    ax[0].set_title('per-candidate $\\sigma(m_{KK\\pi})$')
    for i, (key, lab, col) in enumerate((('pullB', 'B: toy round trip', 'C0'),
                                         ('pC1', 'C: toy + cal. smear, with $\\sigma_{cal}$', 'C2'),
                                         ('pC0', 'C: toy + cal. smear, without', 'C1'),
                                         ('pullD', 'D: reconstruction vs generator', 'C3'))):
        axx = ax[1] if i < 3 else ax[2]
        axx.hist(np.clip(a[key], -5, 5), 100, range=(-5, 5), density=True, histtype='step', color=col,
                 label='%s (core %.2f)' % (lab, np.subtract(*np.percentile(a[key], [75, 25])) / 1.349))
    for axx in ax[1:3]:
        axx.plot(x, gauss, 'k:', label='N(0,1)')
        axx.set_xlabel('pull')
        axx.legend(fontsize=8)
    ax[1].set_title('round trips (toys)')
    ax[2].set_title('real reconstruction')
    ax[2].set_yscale('log')
    for key, lab, col in (('B_toy', 'B toy', 'C0'), ('D_reco', 'D reco', 'C3')):
        b = res[key]['vs_sigma_m']
        ax[3].plot([u['mean_by'] * 1e3 for u in b], [u['iqr'] for u in b], 'o-', color=col, label=lab)
    b = res['C_toy_cal']['with_scal_vs_sigma_m']
    ax[3].plot([u['mean_by'] * 1e3 for u in b], [u['iqr'] for u in b], 's-', color='C2', label='C toy + cal.')
    ax[3].axhline(1, color='k', ls=':')
    ax[3].set_xlabel(r'$\langle\sigma_m\rangle$ in the decile [MeV]')
    ax[3].set_ylabel('pull core width')
    ax[3].legend(fontsize=8)
    ax[3].set_title('pull width per $\\sigma_m$ decile')
    E = res['E_categories']
    xx = np.array([u['median_by'] for u in E['bins']]) * 1e3
    yy = np.array([u['iqr'] for u in E['bins']]) * 1e3
    ax[4].plot(xx, yy, 'o', color='C3', label='MC: residual core width')
    xl = np.linspace(0, xx.max() * 1.1, 10)
    ax[4].plot(xl, xl, 'k:', label='slope 1')
    ax[4].plot(xl, E['slope_through_origin'] * xl, 'C3--', label='fit through 0: %.2f' % E['slope_through_origin'])
    ax[4].set_xlabel(r'median $\sigma_m$ in the decile [MeV]')
    ax[4].set_ylabel(r'core width of $m_{reco}-m_{gen}$ [MeV]')
    ax[4].legend(fontsize=8)
    ax[4].set_title('E: mass resolution per $\\sigma_m$ category')
    for key, lab in (('vs_eta_max', r'max $|\eta|$ of the 3 tracks'), ('vs_pt_min', r'min $p_T$ [GeV]')):
        b = res['D_reco'][key]
        ax[5].plot([u['mean_by'] for u in b], [u['iqr'] for u in b], 'o-', label='D reco vs ' + lab)
    ax[5].axhline(1, color='k', ls=':')
    ax[5].set_ylabel('pull core width')
    ax[5].legend(fontsize=8)
    ax[5].set_title('D: pull width vs kinematics')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'sigma_m_demo.png'), dpi=110)
    fig.savefig(os.path.join(outdir, 'sigma_m_demo.pdf'))


if __name__ == '__main__':
    main()
