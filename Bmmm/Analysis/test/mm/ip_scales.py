#!/usr/bin/env python3
'''
ip_scales.py -- the honesty factors straight from the muon impact parameters
of prompt J/psi, per muon pT x |eta| (the simple cross-check of the vertex fit).

IDEA
  A prompt muon comes from the PV, so its d_xy and d_z wrt the refit PV (signal
  muons removed) are pure measurement error. Pull = IP / sqrt(sigma_trk^2 + PV
  term). Non-prompt J/psi push the IP to the side the J/psi flew: with the
  LIFETIME SIGN (+ when the IP points that way) they sit on the + side only,
  so the width is taken from the LEFT half, as for the along pull. In MC the
  generated flight is subtracted, so MC pulls are prompt-like by construction.

  r = data width / corrected-MC width                   (per cell)
  s = sqrt((r^2 - p) / (1 - p))                          PV part taken as honest
      p = the refit PV's share of the pull variance in data (profiled)
  d_z and d_sz have the same pull (d_sz = d_z cos(lambda)), so s_dz = s_dsz.

INPUT
  the JSON of vertex_pull_study.py plot, filled with the Oct 19 version or
  later (trk_spull_* observables, mu_pt_x_eta binning, pvshare_* profiles)

OUTPUT
  stdout      per cell: widths, PV share, r, s, and the vertex-fit s if --scales
  <out>.json  schema trackscales/1 with fit.per_pt_eta on the t-table grid, so
              make_t_table.py --scales <out>.json builds a t table from it

USAGE (tcsh, covflow env)
  python3 ip_scales.py pulls/vps_2026.json --out pulls/ipscales_2026 \\
      [--scales pulls/trackscales_2026.json]
'''
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vertex_pull_study as V    # noqa: E402

GRID = 'mu_pt_x_eta'


def die(msg):
    sys.exit('[ip_scales] ERROR: ' + msg)


def width(R, o, b, v, i):
    r = R.get('%s|%s|%s' % (o, b, v))
    if not r or i >= len(r) or not r[i] or not r[i].get('ok'):
        return np.nan, np.nan
    return r[i]['sigma'], r[i]['sigma_err']


def s_from(R, P, q, b, i, data='data'):
    o = 'trk_spull_%s_left' % q
    wd, ed = width(R, o, b, data, i)
    wm, em = width(R, o, b, 'mc_corr', i)
    pr = P.get('pvshare_%s|%s|%s' % (q, b, 'data'))
    p = float(pr[i]) if pr is not None and i < len(pr) and np.isfinite(pr[i]) else 0.0
    r = wd / wm
    re = r * np.hypot(ed / wd, em / wm)
    s2 = (r * r - p) / (1 - p)
    s = np.sqrt(s2) if s2 > 0 else np.nan
    se = r * re / (s * (1 - p)) if np.isfinite(s) else np.nan
    return dict(w_data=wd, w_mc=wm, p=p, r=r, r_err=re, s=s, s_err=se)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('vps', help='JSON from vertex_pull_study.py plot')
    ap.add_argument('--out', required=True)
    ap.add_argument('--scales', default=None, help='fit_track_scales.py JSON, to compare with')
    a = ap.parse_args()
    js = json.load(open(a.vps))
    R, P = js['results'], js.get('profiles', {})
    if 'trk_spull_dxy_left|%s|data' % GRID not in R:
        die('%s has no lifetime-signed IP pulls on the %s grid: refill with vertex_pull_study.py '
            'of Oct 19 or later' % (a.vps, GRID))
    pte, ete = V.MU_PT_EDGES, V.MU_ETA_EDGES
    npt, ne = len(pte) - 1, len(ete) - 1
    eras = sorted({k.split('|')[2] for k in R if k.split('|')[2].startswith('data@')})

    vfit = None
    if a.scales:
        pe = json.load(open(a.scales)).get('fit', {}).get('per_pt_eta')
        if pe and len(pe) == npt * ne:
            vfit = pe
        else:
            print('note: %s has no per_pt_eta fit on this grid; no comparison' % a.scales)

    lab = lambda e, i: '%g-%s' % (e[i], ('%g' % e[i + 1]) if e[i + 1] < 1e3 else 'inf')
    lines, cells = [], []
    lines.append('muon IP pulls, lifetime-signed, left-half widths; s = sqrt((r^2 - p)/(1 - p)), '
                 'r = data/MC corr, p = PV share (data)')
    for q in ('dxy', 'dz'):
        lines.append('')
        hdr = '%-3s %-9s %-8s %7s %7s %6s %7s %15s' % (q, 'muon pT', '|eta|', 'data', 'MC corr', 'p',
                                                     'r', 's')
        hdr += ''.join(' %7s' % ('s ' + e[5:][-1:]) for e in eras)
        hdr += '   vertex-fit s' if vfit else ''
        lines.append(hdr)
        for ip in range(npt):
            for ie in range(ne):
                k = ip * ne + ie
                x = s_from(R, P, q, GRID, k)
                row = '    %-9s %-8s %7.4f %7.4f %6.3f %7.4f %7.4f +- %.4f' % (
                    lab(pte, ip), lab(ete, ie), x['w_data'], x['w_mc'], x['p'], x['r'], x['s'], x['s_err'])
                for e in eras:
                    row += ' %7.4f' % s_from(R, P, q, GRID, k, e)['s']
                if vfit and vfit[k]['fit']:
                    kk = 's_dxy' if q == 'dxy' else 's_dsz'
                    row += '   %.4f +- %.4f' % (vfit[k]['fit'][kk], vfit[k]['fit'][kk + '_err'])
                lines.append(row)
                if q == 'dxy':
                    cells.append({'pt': [pte[ip], pte[ip + 1]], 'eta': [ete[ie], ete[ie + 1]], 'dxy': x})
                else:
                    cells[k]['dz'] = x
        x = s_from(R, P, q, 'incl', 0)
        lines.append('    %-18s %7.4f %7.4f %6.3f %7.4f %7.4f +- %.4f' % (
            'inclusive', x['w_data'], x['w_mc'], x['p'], x['r'], x['s'], x['s_err'])
            + ''.join(' %7.4f' % s_from(R, P, q, 'incl', 0, e)['s'] for e in eras))
        xs = s_from(R, P, q, 'incl', 0)
        o = 'trk_pull_%s' % q
        ws, _ = width(R, o, 'incl', 'data', 0)
        wm, _ = width(R, o, 'incl', 'mc_corr', 0)
        lines.append('    (unsigned, full core fit: r = %.4f; the left-half r = %.4f. The gap is the '
                     'non-prompt J/psi)' % (ws / wm, xs['r']))
    print('\n'.join(lines))

    per = []
    for c in cells:
        ok = np.isfinite(c['dxy']['s']) and np.isfinite(c['dz']['s'])
        per.append(dict(pt=c['pt'], eta=c['eta'],
                        fit=dict(s_dxy=c['dxy']['s'], s_dxy_err=c['dxy']['s_err'],
                                 s_dsz=c['dz']['s'], s_dsz_err=c['dz']['s_err']) if ok else None,
                        detail=c))
    inc = {q: s_from(R, P, q, 'incl', 0) for q in ('dxy', 'dz')}
    out = dict(schema='trackscales/1', source=a.vps, method='muon IP pulls, lifetime-signed, left half',
               s_dxy=inc['dxy']['s'], s_dsz=inc['dz']['s'], s_qoverp=1.0, pair_angles=True,
               mu_pt_edges=pte, mu_eta_edges=ete, fit=dict(per_pt_eta=per, inclusive=inc))
    with open(a.out + '.json', 'w') as fh:
        json.dump(out, fh, indent=1, default=lambda v: None if isinstance(v, float) and not np.isfinite(v) else v)
    print('\nwrote %s.json (make_t_table.py --scales %s.json reads it)' % (a.out, a.out))


if __name__ == '__main__':
    main()
