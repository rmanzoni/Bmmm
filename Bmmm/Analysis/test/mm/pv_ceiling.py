#!/usr/bin/env python3
'''
pv_ceiling.py -- why the refit-PV transverse error has a hard wall in MC and
not in data (pull-study note, section 10b).

THE PICTURE BEING TESTED
  The refit PV is an AVF fit of the offline-PV tracks, signal muons removed,
  with the beamspot as a transverse constraint. Along any transverse direction u

      1 / sigma_refit^2  =  1 / C^2  +  1 / T^2

  C  beamspot width along u (the ceiling: sigma_refit can never exceed it)
  T  what the PV tracks alone would give along u

  MC has ONE beamspot width, so every event shares the same ceiling and the
  weak-track PVs (large T) all pile up just below it: bump + spike + wall.
  Data's beamspot width changes lumisection by lumisection (7.5-11 um in x
  over a fill), so the walls are smeared into a broad peak.

WHAT IS MEASURED (existing ntuple branches, no new production needed)
  sigma_refit(u)  from pv_bs_cov_*           (the BS-constrained refit)
  sigma_off(u)    from pv_cov_*              (offline PV: no BS, muons in)
  T(u)            offline PV information minus the two muons' share:
                    1/T^2 = 1/sigma_off^2 - sum_mu (u.n_mu)^2 / sigma(d_xy,mu)^2
                  n_mu = transverse direction a muon's d_xy measures
                  (approximate: ignores AVF weights and track correlations)
  C(u)            the implied ceiling  1/C^2 = 1/sigma_refit^2 - 1/T^2
  luminous region pv_x - bs_x, pv_y - bs_y for well-measured PVs
                  (pv_refit_ntrk >= --ntrk-lum); MC also gen_pv - bs
  If the ntuple has bs_width_x / bs_width_y (Bmmm after Oct 16), also the
  beamspot width projected on u, to compare with the implied C directly.

  Expectation if the picture is right:
    MC    C is a spike at one value (~8.4 um), T is smooth with no spike
    data  C spreads like the beamspot width plot (2026B: ~7.5-11 um)
    T     MC / data about the same as for the PV z error (0.69)

USAGE (covflow conda env: numpy, uproot, matplotlib; tcsh at PSI)
  set D = /pnfs/psi.ch/cms/trivcat/store/user/manzoni/rjpsi_run3/covflow_dimuon_ntuples_12oct26
  python3 pv_ceiling.py fill --files $D/data2026B_PromptReco_v1.root --label data2026B --out pvc/data2026B.npz
  python3 pv_ceiling.py fill --files $D/data2026D_PromptReco_v1.root --label data2026D --out pvc/data2026D.npz
  python3 pv_ceiling.py fill --files $D/hb_Summer24_2026.root        --label mc        --out pvc/mc.npz
  python3 pv_ceiling.py plot pvc/data2026B.npz pvc/data2026D.npz pvc/mc.npz --out pvc/pv_ceiling
  Quick test: --max-entries 2000000 on fill.

OUTPUT
  <out>.pdf   one page per quantity, data eras and MC overlaid
  <out>.json  percentiles per quantity and sample
  stdout      the same, as a table
'''
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vertex_pull_study as V    # noqa: E402

UM = 1e4
LOG_EDGES = np.linspace(0, 3.5, 701)          # log10(sigma / um)
LIN_EDGES = np.linspace(-80, 80, 641)         # um
NTRK_EDGES = np.arange(0, 151) - 0.5
ERR_Q = ('ref', 'off', 'T', 'C', 'Cbs')       # per direction
DIRS = ('across', 'along')
LUM_Q = ('lum_x', 'lum_y', 'gen_x', 'gen_y')
LABELS = dict(ref='refit PV (BS constraint, muons removed)', off='offline PV (no BS, muons in)',
              T='tracks only (offline PV minus the muons)', C='implied beamspot ceiling C',
              Cbs='beamspot width from the ntuple (bs_width_x/y)')
BS_W = ('bs_width_x', 'bs_width_y')


def die(msg):
    sys.exit('[pv_ceiling] ERROR: ' + msg)


def needed(is_mc, a, have):
    b = V.needed_branches(is_mc, a) | {'pv_cov_' + e for e in V.VTX_COV} | {'bs_x', 'bs_y'}
    opt = {n for n in BS_W if n in have}
    return b, opt


def quantities(ch, is_mc):
    '''per-candidate errors (um) along/across the J/psi direction, luminous offsets'''
    f = lambda n: np.asarray(ch[n], np.float64)
    v = V.View(ch, 'corr' if is_mc else 'data')
    D = V.displacement(v, is_mc)
    Cref, Coff = V.cov3(v, 'pv_bs_cov_'), V.cov3(v, 'pv_cov_')
    out = {}
    with np.errstate(all='ignore'):
        for d, u in (('across', D['uperp']), ('along', D['upar'])):
            sref = np.sqrt(np.einsum('ni,nij,nj->n', u, Cref, u)) * UM
            soff = np.sqrt(np.einsum('ni,nij,nj->n', u, Coff, u)) * UM
            imu = 0.
            for mu in ('mu1', 'mu2'):
                phi = f(mu + '_phi')
                proj = -u[:, 0] * np.sin(phi) + u[:, 1] * np.cos(phi)      # u . n_mu
                imu = imu + proj ** 2 / (f(mu + '_cov_dxy_dxy') * UM ** 2)  # raw track: what the offline PV used
            inotrk = 1. / soff ** 2 - imu
            T = np.where(inotrk > 0, 1. / np.sqrt(np.abs(inotrk)), np.inf)
            ic = 1. / sref ** 2 - np.where(inotrk > 0, inotrk, 0.)
            C = np.where(ic > 0, 1. / np.sqrt(np.abs(ic)), np.nan)
            out['ref_' + d], out['off_' + d], out['T_' + d], out['C_' + d] = sref, soff, T, C
            if all(n in ch for n in BS_W):
                wx, wy = f('bs_width_x') * UM, f('bs_width_y') * UM
                out['Cbs_' + d] = np.sqrt(u[:, 0] ** 2 * wx ** 2 + u[:, 1] ** 2 * wy ** 2)
        out['lum_x'] = (f('pv_x') - f('bs_x')) * UM
        out['lum_y'] = (f('pv_y') - f('bs_y')) * UM
        if is_mc:
            out['gen_x'] = (f('gen_pv_x') - f('bs_x')) * UM
            out['gen_y'] = (f('gen_pv_y') - f('bs_y')) * UM
    out['ntrk'] = f('pv_refit_ntrk')
    return out


class Store(dict):
    def fill(self, key, edges, x, w):
        h, _ = np.histogram(x, edges, weights=w)
        self[key] = self[key] + h if key in self else h.astype(np.float64)


def fill_q(q, w, store, view, ntrk_lum):
    for d in DIRS:
        for k in ERR_Q:
            x = q.get('%s_%s' % (k, d))
            if x is None:
                continue
            ok = np.isfinite(x) & (x > 0)
            store.fill('%s_%s|%s' % (k, d, view), LOG_EDGES, np.log10(x[ok]), w[ok])
        store['nan_C_%s|%s' % (d, view)] = store.get('nan_C_%s|%s' % (d, view), 0.) + \
            float(w[~np.isfinite(q['C_' + d])].sum())
    good = q['ntrk'] >= ntrk_lum
    for k in LUM_Q:
        if k in q:
            store.fill('%s|%s' % (k, view), LIN_EDGES, q[k][good], w[good])
    store.fill('ntrk|' + view, NTRK_EDGES, q['ntrk'], w)


def fill(a):
    import uproot
    with uproot.open(a.files[0]) as fh:
        have = set(fh['tree'].keys())
    is_mc = 'gen_pv_x' in have
    need, opt = needed(is_mc, a, have)
    missing = sorted(n for n in need if n not in have)
    if missing:
        die('%s: branches missing: %s' % (a.files[0], ', '.join(missing)))
    if not opt:
        print('  (no bs_width_x/y in this ntuple: the direct beamspot comparison is skipped)')
    total = 0
    for fn in a.files:
        with uproot.open(fn) as fh:
            total += fh['tree'].num_entries
    if a.max_entries:
        total = min(total, a.max_entries)
    print('=== fill %s (%s): %d file(s), %s entries' % (a.label, 'MC' if is_mc else 'data',
                                                     len(a.files), f'{total:,}'))
    store, counts, done = Store(), {}, 0
    f_sb = (2 * a.sr) / (2 * (a.sb[1] - a.sb[0]))
    prog = V.Progress(total, a.label)
    for ch in uproot.iterate([{fn: 'tree'} for fn in a.files], expressions=sorted(need | opt),
                             library='np', step_size=a.step):
        n = len(ch['mass'])
        if a.max_entries and done + n > a.max_entries:
            n = a.max_entries - done
            ch = {k: x[:n] for k, x in ch.items()}
        ch = {k: np.asarray(x) for k, x in ch.items()}
        v = V.View(ch, 'corr' if is_mc else 'data')
        sel, dm = V.base_selection(v, is_mc, a, 'mass')
        if is_mc:
            m = sel & (dm < a.sr)
            w = np.asarray(ch[a.pu_weight], np.float64)
        else:
            sr, sb = sel & (dm < a.sr), sel & (dm > a.sb[0]) & (dm < a.sb[1])
            m = sr | sb
            w = np.where(sr, 1.0, -f_sb)
            counts['SB'] = counts.get('SB', 0) + int(sb.sum())
        counts['sel'] = counts.get('sel', 0) + int((sel & (dm < a.sr)).sum())
        if m.any():
            q = quantities({k: x[m] for k, x in ch.items()}, is_mc)
            fill_q(q, w[m], store, 'mc' if is_mc else 'data', a.ntrk_lum)
        done += n
        prog.update(done, 'selected %s' % f'{counts["sel"]:,}')
        if a.max_entries and done >= a.max_entries:
            break
    prog.update(total)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    meta = dict(label=a.label, is_mc=is_mc, files=a.files, entries=done, counts=counts,
                ntrk_lum=a.ntrk_lum, has_bs_width=bool(opt))
    np.savez_compressed(a.out, _meta=json.dumps(meta), **{k: np.asarray(x) for k, x in store.items()})
    print('\n  selected %s (data: + %s sideband, subtracted)   -> %s'
          % (f'{counts["sel"]:,}', f'{counts.get("SB", 0):,}', a.out))


# ---------------------------------------------------------------------------
def load(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z['_meta']))
    return meta, {k: z[k] for k in z.files if k != '_meta'}


def pct(h, centres, ps=(0.10, 0.25, 0.50, 0.75, 0.90, 0.99)):
    h = np.clip(h, 0, None)
    if h.sum() <= 0:
        return [np.nan] * len(ps)
    cs = np.cumsum(h) / h.sum()
    return [float(centres[min(np.searchsorted(cs, p), len(centres) - 1)]) for p in ps]


def rms_core(h, centres, r=2.5):
    '''Gaussian-core width of a centred distribution: iterated +-r sigma window'''
    h = np.clip(h, 0, None)
    if h.sum() <= 0:
        return np.nan, np.nan
    m = np.sum(h * centres) / h.sum()
    s = np.sqrt(np.sum(h * (centres - m) ** 2) / h.sum())
    for _ in range(20):
        k = np.abs(centres - m) < r * s
        if h[k].sum() <= 0:
            break
        m = np.sum(h[k] * centres[k]) / h[k].sum()
        s_new = np.sqrt(np.sum(h[k] * (centres[k] - m) ** 2) / h[k].sum()) / 0.9558  # truncation at 2.5 sigma
        if abs(s_new - s) < 1e-3:
            s = s_new
            break
        s = s_new
    return m, s


def plot(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    samples = [load(p) for p in a.inputs]
    lc = 10 ** (0.5 * (LOG_EDGES[1:] + LOG_EDGES[:-1]))
    cc = 0.5 * (LIN_EDGES[1:] + LIN_EDGES[:-1])
    cols = ['#222222', '#2f6eba', '#e09f3e', '#7a3db8', '#d1495b']
    R, lines = {}, []
    hdr = '%-26s %-10s' + ' %7s' * 6
    with PdfPages(a.out + '.pdf') as pdf:
        for d in DIRS:
            lines.append('== %s the J/psi flight, stated sigma [um]: percentiles 10/25/50/75/90/99' % d)
            lines.append(hdr % ('quantity', 'sample', '10%', '25%', '50%', '75%', '90%', '99%'))
            fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
            for iq, k in enumerate(('ref', 'T', 'C', 'off')):
                axx = ax.flat[iq]
                for j, (meta, H) in enumerate(samples):
                    key = '%s_%s|%s' % (k, d, 'mc' if meta['is_mc'] else 'data')
                    if key not in H:
                        continue
                    h = np.clip(H[key], 0, None)
                    p = pct(h, lc)
                    R['%s|%s|%s' % (k, d, meta['label'])] = p
                    lines.append(hdr.replace('%7s', '%7.2f').replace('%-10s', '%-10s', 1) % (
                        (LABELS[k][:26], meta['label']) + tuple(p)))
                    if k == 'C':
                        nn = float(H.get('nan_C_%s|%s' % (d, 'mc' if meta['is_mc'] else 'data'), 0.))
                        lines.append('%-26s %-10s fraction with 1/sigma_refit^2 < 1/T^2 (no C): %.3f'
                                     % ('', meta['label'], nn / max(nn + h.sum(), 1e-9)))
                    axx.step(lc, h / max(h.sum(), 1e-9), where='mid', color=cols[j % len(cols)],
                             label='%s (median %.2f)' % (meta['label'], p[2]))
                    if k == 'C':
                        kb = 'Cbs_%s|%s' % (d, 'mc' if meta['is_mc'] else 'data')
                        if kb in H:
                            hb = np.clip(H[kb], 0, None)
                            R['Cbs|%s|%s' % (d, meta['label'])] = pct(hb, lc)
                            lines.append(hdr.replace('%7s', '%7.2f') % (
                                ('beamspot width (ntuple)', meta['label']) + tuple(R['Cbs|%s|%s' % (d, meta['label'])])))
                            axx.step(lc, hb / max(hb.sum(), 1e-9), where='mid', ls=':',
                                     color=cols[j % len(cols)], label='%s: bs_width' % meta['label'])
                axx.set_xscale('log')
                if k == 'C':
                    axx.set_yscale('log')
                    axx.set_ylim(1e-4, 2)
                axx.set_xlim(*((4, 40) if k != 'T' else (5, 150)))
                axx.set_xlabel('$\\sigma$ [$\\mu$m]')
                axx.set_title('%s, %s' % (LABELS[k], d), fontsize=9.5, loc='left')
                axx.legend(fontsize=7.5, frameon=False)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            lines.append('')
        # luminous region
        lines.append('== luminous region: PV - beamspot (PVs with >= %d tracks); Gaussian core [um]'
                     % samples[0][0]['ntrk_lum'])
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        for ia, xy in enumerate('xy'):
            axx = ax[ia]
            for j, (meta, H) in enumerate(samples):
                view = 'mc' if meta['is_mc'] else 'data'
                for k, ls in (('lum_' + xy, '-'), ('gen_' + xy, ':')):
                    key = '%s|%s' % (k, view)
                    if key not in H:
                        continue
                    h = np.clip(H[key], 0, None)
                    m, s = rms_core(h, cc)
                    R['%s|%s' % (k, meta['label'])] = [m, s]
                    tag = 'reco PV' if k.startswith('lum') else 'GEN PV'
                    lines.append('  %-10s %s - bs_%s: mean %6.2f  width %6.2f' % (meta['label'], tag, xy, m, s))
                    axx.step(cc, h / max(h.sum(), 1e-9), where='mid', ls=ls, color=cols[j % len(cols)],
                             label='%s, %s (width %.1f)' % (meta['label'], tag, s))
            axx.set_xlim(-50, 50)
            axx.set_xlabel('PV %s - beamspot %s [$\\mu$m]' % (xy, xy))
            axx.set_title('luminous region in %s (reco PV includes its own error)' % xy, fontsize=9.5, loc='left')
            axx.legend(fontsize=7.5, frameon=False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        # track multiplicity
        fig, axx = plt.subplots(figsize=(7, 4.2))
        for j, (meta, H) in enumerate(samples):
            h = np.clip(H['ntrk|' + ('mc' if meta['is_mc'] else 'data')], 0, None)
            axx.step(np.arange(150), h / max(h.sum(), 1e-9), where='mid', color=cols[j % len(cols)],
                     label=meta['label'])
        axx.set_xlabel('tracks fed to the PV refit (pv_refit_ntrk)')
        axx.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print('\n'.join(lines))
    with open(a.out + '.json', 'w') as fh:
        json.dump(dict(inputs=a.inputs, percentiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.99], values=R), fh, indent=1)
    print('wrote %s.pdf, %s.json' % (a.out, a.out))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fill')
    f.add_argument('--files', nargs='+', required=True)
    f.add_argument('--label', required=True)
    f.add_argument('--out', required=True)
    f.add_argument('--sr', type=float, default=0.05)
    f.add_argument('--sb', type=float, nargs=2, default=(0.15, 0.25))
    f.add_argument('--ct-max', type=float, default=0.01)
    f.add_argument('--pu-weight', default='pu_weight_2026')
    f.add_argument('--ntrk-lum', type=int, default=30, help='PVs used for the luminous region')
    f.add_argument('--max-entries', type=int, default=None)
    f.add_argument('--step', default='300 MB')
    g = sub.add_parser('plot')
    g.add_argument('inputs', nargs='+')
    g.add_argument('--out', required=True)
    return p.parse_args()


def main():
    a = parse_args()
    fill(a) if a.cmd == 'fill' else plot(a)


if __name__ == '__main__':
    main()
