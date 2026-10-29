#!/usr/bin/env python3
'''
fit_track_scales.py -- from the prompt J/psi pull widths to one error scale
factor per kind of track error, for the MC emulation (CovflowEmulation,
--track-scales, "delta_3").

WHY NOT ONE NUMBER (pull-study note, section 12)
  Each vertex measurement sees a different mix of the muons' track errors:
    across the J/psi flight    the transverse positions d_xy only
    along the J/psi flight     d_xy, plus d_sz when the muons differ in eta
    two-muon chi2              mostly d_sz
  and none of them sees the directions (phi, lambda) or q/p. The mix is
  computed per candidate by vertex_pull_study.py (error_shares, pv_shares)
  and stored per bin as profiles. In addition the refit PV makes up a
  sizeable part of the ACROSS variance (~10 um PV against ~10-20 um SV) and
  dilutes any understatement of the SV there; along the flight it is ~1%.

THE MODEL (per bin; X = s_xy^2, Z = s_sz^2, P = s_pv^2, fixed, default 1)
    r_across^2 = (1 - p_ac) X + p_ac P
    r_chi2^2   = f_ch X + (1 - f_ch) Z
    r_along^2  = (1 - p_al) (f_al X + (1 - f_al) Z) + p_al P
  r = width(data) / width(MC corrected): corrected MC is honest by
  construction (note section 11), so r is the data's own understatement.
  f_al, f_ch: d_xy shares of the along error and of the chi2; p_ac, p_al: PV
  shares of the across and along variance. X and Z come from a weighted
  least-squares fit to across and chi2 in all bins; along is NOT used in the
  fit: it is the test (note 12.5). If X, Z describe along too, two factors are
  enough; if along comes out wider, something links the two muons.

  Widths: --width core (core Gaussian; for the chi2 the median-based k) or
  s68 (sigma68; the chi2 has only k). Along is always the LEFT-HALF fit
  (vtx_pull_par_left), immune to non-prompt J/psi.

OUTPUT
  <out>.json   schema trackscales/1, read by CovflowEmulation.TrackScales:
               s_dxy, s_dsz (inclusive fit, clamped at 1: a random step
               cannot remove scatter), s_qoverp = 1 and pair_angles = true
               (not measured here), plus the full fit record under "fit"
  <out>.pdf    ratios per bin with the fitted model, per data view; the
               per-J/psi-pT fit if pt_x_share is present
  stdout       the numbers and the along test

USAGE
  python3 fit_track_scales.py pulls/vps_2026.json --out pulls/trackscales_2026
  python3 fit_track_scales.py pulls/vps_2026.json --width s68 --out pulls/trackscales_2026_s68
'''
import argparse
import json
import sys

import numpy as np

OBS_AC, OBS_AL, OBS_CH = 'vtx_pull_perp', 'vtx_pull_par_left', 'dimu_sqrtchi2'


def die(msg):
    sys.exit('[fit_track_scales] ERROR: ' + msg)


def widths(R, obs, b, view, width):
    rr = R.get('%s|%s|%s' % (obs, b, view))
    if rr is None:
        return None, None
    key = 'sigma' if width == 'core' or obs == OBS_CH else 'sigma68'
    w = np.array([r.get(key, np.nan) if r.get('ok') else np.nan for r in rr], float)
    # sigma68 has no fitted error: the core error x 1.3 (sigma68 of a Gaussian
    # sample has ~1.3x the variance of the fitted sigma), a stated approximation
    e = np.array([r.get('sigma_err', np.nan) for r in rr], float) * (1.3 if key == 'sigma68' else 1.0)
    return w, e


def ratio(R, obs, b, dview, width):
    wd, ed = widths(R, obs, b, dview, width)
    wm, em = widths(R, obs, b, 'mc_corr', width)
    if wd is None or wm is None:
        return None, None
    with np.errstate(all='ignore'):
        r = wd / wm
        er = r * np.hypot(ed / wd, em / wm)
    return r, er


def prof(P, q, b, view):
    x = P.get('%s|%s|%s' % (q, b, view))
    return None if x is None else np.array(x, float)


def fit_bins(r_ac, e_ac, r_ch, e_ch, p_ac, f_ch, s_pv, sel=None):
    '''weighted LSQ for X = s_xy^2, Z = s_sz^2 on r^2 (linear in X, Z)'''
    rows, ys, ws = [], [], []
    n = len(r_ac)
    sel = np.ones(n, bool) if sel is None else sel
    for i in range(n):
        if not sel[i]:
            continue
        if np.isfinite(r_ac[i]) and np.isfinite(e_ac[i]) and e_ac[i] > 0 and np.isfinite(p_ac[i]):
            rows.append([1 - p_ac[i], 0.]); ys.append(r_ac[i] ** 2 - p_ac[i] * s_pv ** 2)
            ws.append(1. / (2 * r_ac[i] * e_ac[i]) ** 2)
        if np.isfinite(r_ch[i]) and np.isfinite(e_ch[i]) and e_ch[i] > 0 and np.isfinite(f_ch[i]):
            rows.append([f_ch[i], 1 - f_ch[i]]); ys.append(r_ch[i] ** 2)
            ws.append(1. / (2 * r_ch[i] * e_ch[i]) ** 2)
    if len(rows) < 2:
        return None
    A, y, W = np.array(rows), np.array(ys), np.array(ws)
    ATA = A.T @ (W[:, None] * A)
    if abs(np.linalg.det(ATA)) < 1e-300:
        return None
    cov = np.linalg.inv(ATA)
    x = cov @ (A.T @ (W * y))
    chi2 = float(np.sum(W * (A @ x - y) ** 2))
    X, Z = x
    sx, sz = np.sqrt(max(X, 0)), np.sqrt(max(Z, 0))
    # errors on s from those on s^2
    esx = np.sqrt(cov[0, 0]) / (2 * max(sx, 1e-9))
    esz = np.sqrt(cov[1, 1]) / (2 * max(sz, 1e-9))
    return dict(s_dxy=sx, s_dxy_err=esx, s_dsz=sz, s_dsz_err=esz,
                rho=float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])), chi2=chi2,
                ndf=len(y) - 2, X=X, Z=Z, cov=cov.tolist())


def along_test(fit, r_al, e_al, p_al, f_al, s_pv):
    '''predicted along ratio from the fitted X, Z; the extra variance
    fraction needed on top: r_al^2 - pred^2'''
    X, Z = fit['X'], fit['Z']
    pred = np.sqrt((1 - p_al) * (f_al * X + (1 - f_al) * Z) + p_al * s_pv ** 2)
    with np.errstate(all='ignore'):
        pull = (r_al - pred) / e_al
        extra = r_al ** 2 - pred ** 2
        eextra = 2 * r_al * e_al
    ok = np.isfinite(pull)
    if not ok.any():
        return dict(pred=pred.tolist(), pull=pull.tolist())
    wmean = float(np.sum(extra[ok] / eextra[ok] ** 2) / np.sum(1 / eextra[ok] ** 2))
    werr = float(1 / np.sqrt(np.sum(1 / eextra[ok] ** 2)))
    return dict(pred=pred.tolist(), pull=pull.tolist(), chi2=float(np.sum(pull[ok] ** 2)),
                n=int(ok.sum()), extra_var=wmean, extra_var_err=werr)


def analyse_view(R, P, b, dview, width, s_pv):
    r_ac, e_ac = ratio(R, OBS_AC, b, dview, width)
    r_ch, e_ch = ratio(R, OBS_CH, b, dview, width)
    r_al, e_al = ratio(R, OBS_AL, b, dview, width)
    if r_ac is None or r_ch is None or r_al is None:
        return None
    # shares from the data candidates of that view (the stated errors of data
    # and corrected MC are matched by covflow); PV shares likewise
    pv = 'data' if dview.startswith('data') else dview
    f_al = prof(P, 'share_along', b, pv)
    f_ch = prof(P, 'share_chi2', b, pv)
    p_ac = prof(P, 'pvshare_across', b, pv)
    p_al = prof(P, 'pvshare_along', b, pv)
    if any(x is None for x in (f_al, f_ch, p_ac, p_al)):
        die('profiles missing for binning %s: vps json made by an older vertex_pull_study.py?' % b)
    fit = fit_bins(r_ac, e_ac, r_ch, e_ch, p_ac, f_ch, s_pv)
    if fit is None:
        return None
    test = along_test(fit, r_al, e_al, p_al, f_al, s_pv)
    return dict(binning=b, view=dview, width=width, fit=fit, along=test,
                r_across=r_ac.tolist(), e_across=e_ac.tolist(), r_chi2=r_ch.tolist(),
                e_chi2=e_ch.tolist(), r_along=r_al.tolist(), e_along=e_al.tolist(),
                share_along=f_al.tolist(), share_chi2=f_ch.tolist(),
                pvshare_across=p_ac.tolist(), pvshare_along=p_al.tolist())


def per_pt(R, P, js, dview, width, s_pv):
    b = 'pt_x_share'
    if '%s|%s|%s' % (OBS_AC, b, dview) not in R:
        return None
    pte, she = js['pt2_edges'], js['sh2_edges']
    npt, nsh = len(pte) - 1, len(she) - 1
    A = analyse_view(R, P, b, dview, width, s_pv)
    if A is None:
        return None
    out = []
    for ip in range(npt):
        sel = np.zeros(npt * nsh, bool)
        sel[ip * nsh:(ip + 1) * nsh] = True
        g = lambda k: np.asarray(A[k], float)
        fit = fit_bins(g('r_across'), g('e_across'), g('r_chi2'), g('e_chi2'),
                       g('pvshare_across'), g('share_chi2'), s_pv, sel)
        if fit is None:
            out.append(dict(pt=[pte[ip], pte[ip + 1]], fit=None))
            continue
        t = along_test(fit, g('r_along')[sel], g('e_along')[sel], g('pvshare_along')[sel],
                       g('share_along')[sel], s_pv)
        out.append(dict(pt=[pte[ip], pte[ip + 1]], fit=fit, along=t))
    return out


def report(A, label):
    f = A['fit']
    L = ['%s  [%s, %s widths]' % (label, A['binning'], A['width']),
         '  s_dxy = %.4f +- %.4f    s_dsz = %.4f +- %.4f    (correlation %.2f; chi2 %.1f / %d)'
         % (f['s_dxy'], f['s_dxy_err'], f['s_dsz'], f['s_dsz_err'], f['rho'], f['chi2'], f['ndf'])]
    t = A['along']
    if 'extra_var' in t:
        L.append('  along test: chi2 %.1f / %d; extra variance needed along %+.4f +- %.4f '
                 '(%s)' % (t['chi2'], t['n'], t['extra_var'], t['extra_var_err'],
                           'two factors are enough' if abs(t['extra_var']) < 2 * t['extra_var_err']
                           else 'along is NOT described by per-track factors (note 12.5)'))
    L.append('  %-4s %8s %8s %8s %8s | %8s %8s | %8s %8s %8s' % ('bin', 'f_al', 'f_ch', 'p_ac',
                                                               'p_al', 'r_ac', 'r_ch', 'r_al',
                                                               'pred', 'pull'))
    for i in range(len(A['r_across'])):
        L.append('  %-4d %8.3f %8.3f %8.3f %8.3f | %8.4f %8.4f | %8.4f %8.4f %8.2f'
                 % (i, A['share_along'][i], A['share_chi2'][i], A['pvshare_across'][i],
                    A['pvshare_along'][i], A['r_across'][i], A['r_chi2'][i], A['r_along'][i],
                    t['pred'][i], t['pull'][i]))
    return L


def plot(results, ptres, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out + '.pdf') as pdf:
        for A in results:
            fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
            x = np.arange(len(A['r_across']))
            X, Z = A['fit']['X'], A['fit']['Z']
            for k, (key, lab, pred) in enumerate((
                    ('across', 'across: sees d$_{xy}$ (and the PV)',
                     np.sqrt((1 - np.array(A['pvshare_across'])) * X + np.array(A['pvshare_across']))),
                    ('chi2', 'two-muon $\\chi^2$: mostly d$_{sz}$',
                     np.sqrt(np.array(A['share_chi2']) * X + (1 - np.array(A['share_chi2'])) * Z)),
                    ('along', 'along (left half): the TEST, not fitted',
                     np.array(A['along']['pred'])))):
                ax[k].errorbar(x, A['r_' + key], yerr=A['e_' + key], fmt='o', color='black',
                               label='data / MC corrected')
                ax[k].plot(x, pred, 's--', color='#d1495b', mfc='none',
                           label='model: s$_{xy}$ = %.3f, s$_{sz}$ = %.3f'
                           % (A['fit']['s_dxy'], A['fit']['s_dsz']))
                ax[k].axhline(1, color='grey', lw=0.8, ls=':')
                ax[k].set_title(lab, fontsize=10, loc='left')
                ax[k].set_xlabel('bin of %s' % A['binning'])
                ax[k].set_ylabel('width ratio')
                ax[k].legend(fontsize=8, frameon=False)
            fig.suptitle('%s, %s widths' % (A['view'], A['width']), fontsize=11)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        if ptres:
            fig, ax = plt.subplots(1, 1, figsize=(7, 4.4))
            ok = [p for p in ptres if p['fit']]
            xc = np.arange(len(ok))
            for key, col, lab in (('s_dxy', '#2f6eba', 's$_{xy}$'), ('s_dsz', '#e09f3e', 's$_{sz}$')):
                ax.errorbar(xc, [p['fit'][key] for p in ok], yerr=[p['fit'][key + '_err'] for p in ok],
                            fmt='o-', color=col, label=lab)
            ax.set_xticks(xc)
            ax.set_xticklabels(['%g-%g' % (p['pt'][0], min(p['pt'][1], 999)) for p in ok])
            ax.set_xlabel('J/$\\psi$ $p_T$ [GeV]')
            ax.set_ylabel('fitted factor')
            ax.axhline(1, color='grey', lw=0.8, ls=':')
            ax.legend(frameon=False)
            ax.set_title('factors per J/$\\psi$ $p_T$ (data, inclusive in era)', fontsize=10, loc='left')
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('json', help='the .json written by vertex_pull_study.py plot')
    ap.add_argument('--out', required=True)
    ap.add_argument('--binning', default='share_along',
                    help='binning used for the inclusive fit (default share_along)')
    ap.add_argument('--width', choices=['core', 's68'], default='core')
    ap.add_argument('--s-pv', type=float, default=1.0,
                    help='assumed scale factor of the refit-PV errors (default 1: honest)')
    a = ap.parse_args()
    js = json.load(open(a.json))
    R, P = js['results'], js.get('profiles')
    if not P:
        die('%s has no profiles: made by an older vertex_pull_study.py' % a.json)
    views = sorted({k.split('|')[2] for k in R if k.startswith(OBS_AC + '|incl|data')
                    and k.split('|')[2] != 'data_SR'})
    results, lines = [], []
    for dv in views:
        A = analyse_view(R, P, a.binning, dv, a.width, a.s_pv)
        if A is None:
            print('  %s: not enough bins for a fit' % dv)
            continue
        results.append(A)
        lines += report(A, dv) + ['']
    if not results or results[0]['view'] != 'data':
        die('no fit for the combined data view')
    ptres = per_pt(R, P, js, 'data', a.width, a.s_pv)
    if ptres:
        lines.append('per J/psi pT (data):')
        for p in ptres:
            if p['fit']:
                f = p['fit']
                t = p['along']
                lines.append('  pT %5g-%-5g  s_dxy %.4f +- %.4f  s_dsz %.4f +- %.4f   along extra '
                             'variance %+.4f +- %.4f' % (p['pt'][0], min(p['pt'][1], 999), f['s_dxy'],
                                                         f['s_dxy_err'], f['s_dsz'], f['s_dsz_err'],
                                                         t.get('extra_var', np.nan),
                                                         t.get('extra_var_err', np.nan)))
    print('\n'.join(lines))
    f = results[0]['fit']
    clamp = {k: max(f[k], 1.0) for k in ('s_dxy', 's_dsz')}
    notes = ['%s fitted %.4f < 1, set to 1: a random step cannot remove scatter' % (k, f[k])
             for k in ('s_dxy', 's_dsz') if f[k] < 1]
    for n_ in notes:
        print('WARNING: ' + n_)
    out = dict(schema='trackscales/1', s_dxy=clamp['s_dxy'], s_dsz=clamp['s_dsz'], s_qoverp=1.0,
               pair_angles=True,
               fit=dict(source=a.json, binning=a.binning, width=a.width, s_pv=a.s_pv, notes=notes,
                        not_measured='phi, lambda (paired with dxy, dsz by assumption) and q/p '
                                     '(=1): invisible to the vertex pulls, to be measured with the '
                                     'J/psi mass resolution',
                        views=results, per_pt=ptres))
    with open(a.out + '.json', 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    plot(results, ptres, a.out)
    print('wrote %s.json, %s.pdf' % (a.out, a.out))


if __name__ == '__main__':
    main()
