#!/usr/bin/env python3
'''
make_t_table.py -- the extra scatter MC needs to get data's RESOLUTION, per
muon pT x |eta| (pull-study note, section 10b and 13.3):

    t = s x E / C

  s   the data's own understatement of its track errors (pulls; honesty),
      from fit_track_scales.py, per J/psi pT x muon |eta| (fit/per_pt_eta)
  E   median stated error of the data muons in the cell (vertex_errors.py)
  C   median stated error of the corrected-MC muons in the cell

  Corrected MC is honest, so its real scatter is ~C; data's is s E. MC
  widened by t then scatters like data. Where t < 1 corrected MC is already
  wider than data: a random step cannot narrow it, so the table holds 1 and
  the cell is listed under "floored" (the emulator counts those muons).

  d_xy uses s_dxy and sigma(d_xy); d_sz uses s_dsz and sigma(d_sz); phi and
  lambda follow d_xy and d_sz (pair_angles), q/p stays 1.

GRID
  muon pT [GeV]: 0-5, 5-8, 8-15, >15 (vertex_errors.py bins merged; the
  first holds 3-5 GeV muons) <-> J/psi pT 0-10, 10-16, 16-30, >30 in the
  pull study (muon pT ~ J/psi pT / 2). muon |eta|: 0-0.9, 0.9-1.5, 1.5-2.5.

INPUT FALLBACK
  If the pull-study JSON had no pt_eta_x_share binning (filled before Oct
  16), s is taken from fit/per_pt (J/psi pT only, the same in every |eta|)
  and a warning is printed.

OUTPUT
  <out>.json  schema trackscales/1 with pt_edges, eta_edges, s_dxy_pt_eta,
              s_dsz_pt_eta (floored at 1), floored, and the full record
              (s, E, C, raw t, counts) under "t_table"
  <out>.pdf   t per cell, with s alone for comparison
  stdout      the table

USAGE
  python3 make_t_table.py --scales pulls/trackscales_2026.json \
      --errs errs/data2026B.npz errs/data2026D.npz errs/mc.npz --out pulls/ttable_2026
'''
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vertex_errors as VE       # noqa: E402
import vertex_pull_study as V    # noqa: E402

PT_GROUPS = [[0, 1], [2, 3], [4, 5], [6]]        # of vertex_errors.MU_PT bins
PT_EDGES = [0, 5, 8, 15, 1e9]                     # muon pT, the first bin from 0
ETA_EDGES = list(VE.MU_ETA)


def die(msg):
    sys.exit('[make_t_table] ERROR: ' + msg)


def merged_medians(H, q, view):
    '''medians of the muon stated error on the t grid'''
    key = (q, 'mu_pt_x_eta', view)
    if key not in H:
        die('no %s histograms for %s in the vertex_errors inputs' % (q, view))
    w = H[key]                                   # (n_pt * n_eta, nhist)
    ne = len(VE.MU_ETA) - 1
    e = VE.edges_of(q)
    med = np.full((len(PT_GROUPS), ne), np.nan)
    cnt = np.zeros((len(PT_GROUPS), ne))
    for ig, grp in enumerate(PT_GROUPS):
        for ie in range(ne):
            h = sum(w[ip * ne + ie] for ip in grp)
            med[ig, ie] = 10 ** V.quantile(h, e, 0.5)
            cnt[ig, ie] = np.clip(h, 0, None).sum()
    return med, cnt


def s_tables(ts):
    '''s_dxy, s_dsz (and errors) on the grid, from per_pt_eta or the fallback'''
    fit = ts.get('fit', {})
    npt, ne = len(PT_EDGES) - 1, len(ETA_EDGES) - 1
    S = {k: np.full((npt, ne), np.nan) for k in ('s_dxy', 's_dsz', 's_dxy_err', 's_dsz_err')}
    pe = fit.get('per_pt_eta')
    source = 'per_pt_eta'
    if pe:
        if len(pe) != npt * ne:
            die('per_pt_eta has %d cells, expected %d' % (len(pe), npt * ne))
        for k_, cell in enumerate(pe):
            ip, ie = divmod(k_, ne)
            if cell['fit'] is None:
                continue
            for k in S:
                S[k][ip, ie] = cell['fit'][k]
    pp = fit.get('per_pt')
    missing = ~np.isfinite(S['s_dxy'])
    if missing.any():
        if not pp:
            die('no per_pt_eta and no per_pt fits in %s' % ts.get('_path'))
        if not pe:
            source = 'per_pt (J/psi pT only, the same in every |eta|)'
            print('WARNING: the scales have no per_pt_eta fit (pull study filled before the '
                  'pt_eta_x_share binning): s from per_pt, independent of |eta|')
        else:
            print('WARNING: %d cell(s) without a per_pt_eta fit: s from per_pt there'
                  % int(missing.sum()))
        for ip in range(min(npt, len(pp))):
            if pp[ip]['fit'] is None:
                continue
            for ie in range(ne):
                if missing[ip, ie]:
                    for k in S:
                        S[k][ip, ie] = pp[ip]['fit'][k]
    if not np.all(np.isfinite(S['s_dxy'])):
        die('cells without any s: %s' % np.argwhere(~np.isfinite(S['s_dxy'])).tolist())
    return S, source


def plot(T, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labs = ['%g-%s' % (PT_EDGES[i], ('%g' % PT_EDGES[i + 1]) if PT_EDGES[i + 1] < 1e3 else 'inf')
            for i in range(len(PT_EDGES) - 1)]
    x = np.arange(len(labs))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for k, p in enumerate(('dxy', 'dsz')):
        a = ax[k]
        a.axhspan(0.85, 1.0, color='#d1495b', alpha=0.08, lw=0)
        t, s = np.array(T['t_raw_' + p]), np.array(T['s_' + p])
        for j, col in enumerate(('#2f6eba', '#7a3db8', '#e09f3e')):
            lab = '|$\\eta$| %g-%g' % (ETA_EDGES[j], ETA_EDGES[j + 1])
            a.plot(x + 0.05 * (j - 1), t[:, j], 'o-', color=col, label='t, ' + lab)
            a.plot(x + 0.05 * (j - 1), s[:, j], ':', color=col, alpha=0.7)
        a.axhline(1, color='grey', lw=0.7, ls=':')
        a.set_xticks(x)
        a.set_xticklabels(labs)
        a.set_xlabel('muon $p_T$ [GeV]')
        a.set_title('$d_{%s}$: t (solid), s alone (dotted); red: t < 1, floored at 1'
                    % p[1:], fontsize=9.5, loc='left')
        a.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out + '.pdf')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scales', required=True, help='JSON from fit_track_scales.py')
    ap.add_argument('--errs', nargs='+', required=True, help='.npz files from vertex_errors.py fill')
    ap.add_argument('--out', required=True)
    ap.add_argument('--sr', type=float, default=0.05)
    ap.add_argument('--sb', type=float, nargs=2, default=(0.15, 0.25))
    a = ap.parse_args()

    ts = json.load(open(a.scales))
    ts['_path'] = a.scales
    S, source = s_tables(ts)
    H, eras = VE.combine([VE.load_store(p) for p in a.errs], a)
    if not eras:
        die('no data among --errs')
    T = dict(source_scales=a.scales, source_errs=a.errs, s_source=source, eras=eras,
             pt_edges=PT_EDGES, eta_edges=ETA_EDGES)
    out = dict(schema='trackscales/1', s_qoverp=1.0, pair_angles=True,
               pt_edges=PT_EDGES, eta_edges=ETA_EDGES, floored=[])
    floored = set()
    lines = ['t = s x E / C per muon pT x |eta|   (s from %s; data: %s)' % (source, ', '.join(eras)), '']
    for p, q in (('dxy', 'trk_dxy'), ('dsz', 'trk_dsz')):
        E, nE = merged_medians(H, q, 'data')
        C, nC = merged_medians(H, q, 'mc_corr')
        s, se = S['s_' + p], S['s_%s_err' % p]
        t = s * E / C
        T.update({'s_' + p: s.tolist(), 's_%s_err' % p: se.tolist(), 'E_' + p: E.tolist(),
                  'C_' + p: C.tolist(), 'C_over_E_' + p: (C / E).tolist(), 't_raw_' + p: t.tolist(),
                  't_err_' + p: (se * E / C).tolist(), 'n_data_' + p: nE.tolist(),
                  'n_mc_' + p: nC.tolist()})
        out['s_%s_pt_eta' % p] = np.maximum(t, 1.0).tolist()
        for ip, ie in np.argwhere(t < 1):
            floored.add((int(ip), int(ie)))
        lines.append('%s: %-10s %-9s %8s %8s %8s %8s %8s %9s %9s' % (p, 'muon pT', '|eta|', 's', 'E [um]',
                                                                  'C [um]', 'C/E', 't', 'n data', 'n MC'))
        for ip in range(len(PT_EDGES) - 1):
            for ie in range(len(ETA_EDGES) - 1):
                lines.append('     %-10s %-9s %8.4f %8.2f %8.2f %8.3f %8.4f %9.0f %9.0f%s' % (
                    '%g-%s' % (PT_EDGES[ip], ('%g' % PT_EDGES[ip + 1]) if PT_EDGES[ip + 1] < 1e3 else 'inf'),
                    '%g-%g' % (ETA_EDGES[ie], ETA_EDGES[ie + 1]), s[ip, ie], E[ip, ie], C[ip, ie],
                    C[ip, ie] / E[ip, ie], t[ip, ie], nE[ip, ie], nC[ip, ie],
                    '   <1: floored' if t[ip, ie] < 1 else ''))
        lines.append('')
    out['floored'] = sorted([list(c) for c in floored])
    # inclusive values for describe() and older readers: the event-weighted mean of the table
    w = np.array(T['n_data_dxy'])
    out['s_dxy'] = float(np.sum(w * np.array(out['s_dxy_pt_eta'])) / w.sum())
    out['s_dsz'] = float(np.sum(w * np.array(out['s_dsz_pt_eta'])) / w.sum())
    out['t_table'] = T
    lines.append('%d cell(s) floored at 1 (MC already wider than data, to be fixed in covflow): %s'
                 % (len(floored), out['floored']))
    print('\n'.join(lines))
    with open(a.out + '.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    plot(T, a.out)
    print('wrote %s.json, %s.pdf' % (a.out, a.out))


if __name__ == '__main__':
    main()
