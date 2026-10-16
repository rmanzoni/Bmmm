#!/usr/bin/env python3
'''
vertex_errors.py -- are the STATED errors of corrected MC data-like?
(pull-study note, section 12c)

The pulls of vertex_pull_study.py divide every measurement by its own stated
error, so they cannot see whether corrected MC states the SAME errors as data.
The PV shares hint that it may not: the PV makes up 11% of the across variance
in data, 15% in raw MC and 7% in corrected MC. With the same PV in all three,
corrected MC would state dimuon-vertex errors ~30% larger than data (in sigma);
with a narrower beamspot in MC, the PV would explain it instead. This script
looks at the stated errors themselves.

WHAT IS FILLED (same selection as vertex_pull_study.py: medium ID, opposite
charge, valid refit PV, prompt cut ct < CT_MAX, J/psi signal window; data SR
and SB, the SB subtracted at plot time)
  per candidate, stated sigma in um, histogrammed as log10(sigma / um):
    sv_along, sv_across, sv_z   dimuon vertex, along / across the J/psi
                                transverse direction, and z
    pv_along, pv_across, pv_z   refit PV (pv_bs_*), same directions
    in bins of J/psi pT
  per muon:
    trk_dxy, trk_dsz            sigma(d_xy), sigma(d_sz) = dz_e cos(lambda)
    in bins of muon pT, |eta|, and pT x |eta|
  MC only, the SAME event before and after the emulation:
    r_sv_along, r_sv_across     stated SV variance, corrected / raw
    r_trk_dxy, r_trk_dsz        stated track variance, corrected / raw
    histogrammed as log10(ratio); section 10 of the note needs 1.3-1.5 across

HOW TO READ THE SUMMARY
  medians per bin, and corrected / data, raw / data. If pv_* agree between
  data and MC while sv_* corrected / data ~ 1.3: the emulation overshoots at
  the vertex. If pv_* is smaller in MC: the beamspot, nothing wrong.
  Kinematics differ (prompt J/psi in data, J/psi from B in MC): compare per
  bin, not inclusive.

USAGE (covflow conda env: numpy, uproot, matplotlib)
  python3 vertex_errors.py all --out errs/verr_2026
  python3 vertex_errors.py all --out errs/verr_2026 --max-entries 3000000   # quick look
  or one job per file, then plot:
  python3 vertex_errors.py fill --files <data2026B.root> --label data2026B --out errs/data2026B.npz
  python3 vertex_errors.py fill --files <data2026D.root> --label data2026D --out errs/data2026D.npz
  python3 vertex_errors.py fill --files <hb_Summer24_2026.root> --label mc --out errs/mc.npz
  python3 vertex_errors.py plot errs/data2026B.npz errs/data2026D.npz errs/mc.npz --out errs/verr_2026

OUTPUTS
  <out>.pdf   summary page; per quantity the inclusive distributions and the
              medians vs bin with the MC / data ratios
  <out>.json  every median
'''

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vertex_pull_study as V   # noqa: E402  (selection, views, geometry, progress bar)

UM = 1e4                                        # cm -> um
LOG_EDGES = np.linspace(0.0, 3.5, 701)          # log10(sigma / um): 1 um .. 3 mm
RAT_EDGES = np.linspace(-1.0, 1.5, 501)         # log10(variance ratio): 0.1 .. 30

VTX_Q = ('sv_along', 'sv_across', 'sv_z', 'pv_along', 'pv_across', 'pv_z')
TRK_Q = ('trk_dxy', 'trk_dsz')
RAT_Q = ('r_sv_along', 'r_sv_across', 'r_trk_dxy', 'r_trk_dsz')

MU_PT = [3, 4, 5, 6, 8, 10, 15, 1e9]
MU_ETA = [0, 0.9, 1.5, 2.5]
BINS = {
    'incl':        [-1e9, 1e9],
    'jpsi_pt':     V.BINNINGS['jpsi_pt'][1],
    'mu_pt':       MU_PT,
    'mu_abseta':   MU_ETA,
    'mu_pt_x_eta': list(np.arange((len(MU_PT) - 1) * (len(MU_ETA) - 1) + 1).astype(float)),
}
VTX_BINS = ('incl', 'jpsi_pt')
TRK_BINS = ('incl', 'mu_pt', 'mu_abseta', 'mu_pt_x_eta')
LABEL = {
    'sv_along': 'dimuon vertex, along the flight', 'sv_across': 'dimuon vertex, across the flight',
    'sv_z': 'dimuon vertex, z', 'pv_along': 'refit PV, along the flight',
    'pv_across': 'refit PV, across the flight', 'pv_z': 'refit PV, z',
    'trk_dxy': r'muon $\sigma(d_{xy})$', 'trk_dsz': r'muon $\sigma(d_{sz})$',
    'r_sv_along': 'SV variance along, corrected / raw (same event)',
    'r_sv_across': 'SV variance across, corrected / raw (same event)',
    'r_trk_dxy': r'$\sigma^2(d_{xy})$, corrected / raw (same muon)',
    'r_trk_dsz': r'$\sigma^2(d_{sz})$, corrected / raw (same muon)',
}


def edges_of(q):
    return RAT_EDGES if q.startswith('r_') else LOG_EDGES


# ---------------------------------------------------------------------------
# quantities
# ---------------------------------------------------------------------------
def vertex_errors(v, is_mc):
    D = V.displacement(v, is_mc)
    Csv, Cpv = V.cov3(v, 'vtx_cov_'), V.cov3(v, 'pv_bs_cov_')
    out = {}
    with np.errstate(all='ignore'):
        for nm, u in (('along', D['upar']), ('across', D['uperp'])):
            out['sv_' + nm] = np.sqrt(np.einsum('ni,nij,nj->n', u, Csv, u)) * UM
            out['pv_' + nm] = np.sqrt(np.einsum('ni,nij,nj->n', u, Cpv, u)) * UM
        out['sv_z'] = np.sqrt(Csv[:, 2, 2]) * UM
        out['pv_z'] = np.sqrt(Cpv[:, 2, 2]) * UM
    return out


def jpsi_pt(v):
    f = lambda n: np.asarray(v(n), np.float64)
    px = f('mu1_pt') * np.cos(f('mu1_phi')) + f('mu2_pt') * np.cos(f('mu2_phi'))
    py = f('mu1_pt') * np.sin(f('mu1_phi')) + f('mu2_pt') * np.sin(f('mu2_phi'))
    return np.hypot(px, py)


def track_errors(v, mu):
    f = lambda n: np.asarray(v(n), np.float64)
    eta = f(mu + '_eta')
    with np.errstate(all='ignore'):
        dxy = np.sqrt(f(mu + '_cov_dxy_dxy')) * UM
        dsz = f(mu + '_dz_e') * np.cos(np.arctan(np.sinh(eta))) * UM
    return dict(trk_dxy=dxy, trk_dsz=dsz)


def track_bins(v, mu):
    f = lambda n: np.asarray(v(n), np.float64)
    pt, ae = f(mu + '_pt'), np.abs(f(mu + '_eta'))
    ne = len(MU_ETA) - 1
    ip = np.searchsorted(MU_PT, pt, side='right') - 1
    ie = np.searchsorted(MU_ETA, ae, side='right') - 1
    ok = (ip >= 0) & (ip < len(MU_PT) - 1) & (ie >= 0) & (ie < ne)
    return dict(incl=np.zeros(len(pt)), mu_pt=pt, mu_abseta=ae,
                mu_pt_x_eta=np.where(ok, ip * ne + ie + 0.5, np.nan))


# ---------------------------------------------------------------------------
# store: (quantity, binning, view) -> [sum w, sum w^2] (nbins x nhist)
# ---------------------------------------------------------------------------
class Store(object):
    def __init__(self):
        self.h = {}

    def fill(self, q, binvars, view, x, w, log=True):
        e = edges_of(q)
        nh = len(e) - 1
        with np.errstate(all='ignore'):
            y = np.log10(x) if log else x
        ok = np.isfinite(y) & np.isfinite(w)
        ih = np.clip(((y - e[0]) / (e[1] - e[0])).astype(np.int64), 0, nh - 1)
        for bname, bv in binvars.items():
            be = np.asarray(BINS[bname], float)
            nb = len(be) - 1
            okb = ok & np.isfinite(bv) & (bv >= be[0]) & (bv < be[-1])
            ib = np.clip(np.searchsorted(be, bv, side='right') - 1, 0, nb - 1)
            flat = (ib * nh + ih)[okb]
            key = (q, bname, view)
            if key not in self.h:
                self.h[key] = [np.zeros(nb * nh), np.zeros(nb * nh)]
            self.h[key][0] += np.bincount(flat, weights=w[okb], minlength=nb * nh)
            self.h[key][1] += np.bincount(flat, weights=w[okb] ** 2, minlength=nb * nh)

    def save(self, path, meta):
        arr = {}
        for (q, b, v), (w, w2) in self.h.items():
            arr['w|%s|%s|%s' % (q, b, v)] = w
            arr['w2|%s|%s|%s' % (q, b, v)] = w2
        arr['meta'] = np.array(json.dumps(meta))
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        np.savez_compressed(path, **arr)


def load_store(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z['meta']))
    h = {}
    for k in z.files:
        if k.startswith('w|'):
            _, q, b, v = k.split('|')
            nh, nb = len(edges_of(q)) - 1, len(BINS[b]) - 1
            h[(q, b, v)] = (z[k].reshape(nb, nh), z['w2|%s|%s|%s' % (q, b, v)].reshape(nb, nh))
    return meta, h


# ---------------------------------------------------------------------------
# fill
# ---------------------------------------------------------------------------
def fill_view(v, is_mc, view, w, store):
    vb = dict(incl=np.zeros(len(w)), jpsi_pt=jpsi_pt(v))
    for q, x in vertex_errors(v, is_mc).items():
        store.fill(q, vb, view, x, w)
    for mu in ('mu1', 'mu2'):
        tb = track_bins(v, mu)
        for q, x in track_errors(v, mu).items():
            store.fill(q, tb, view, x, w)


def fill_ratios(ch, m, w, store):
    '''same events, corrected / raw stated variances (MC only)'''
    sub = {k: x[m] for k, x in ch.items()}
    vc, vr = V.View(sub, 'corr'), V.View(sub, 'raw')
    D = V.displacement(vc, True)
    Cc, Cr = V.cov3(vc, 'vtx_cov_'), V.cov3(vr, 'vtx_cov_')
    vb = dict(incl=np.zeros(int(m.sum())), jpsi_pt=jpsi_pt(vc))
    with np.errstate(all='ignore'):
        for nm, u in (('along', D['upar']), ('across', D['uperp'])):
            r = np.einsum('ni,nij,nj->n', u, Cc, u) / np.einsum('ni,nij,nj->n', u, Cr, u)
            store.fill('r_sv_' + nm, vb, 'mc_ratio', r, w)
        for mu in ('mu1', 'mu2'):
            tb = track_bins(vc, mu)
            ec, er = track_errors(vc, mu), track_errors(vr, mu)
            for q in TRK_Q:
                store.fill('r_' + q, tb, 'mc_ratio', (ec[q] / er[q]) ** 2, w)


def fill_chunk(ch, is_mc, store, counts, a):
    ch = {k: np.asarray(x) for k, x in ch.items()}
    if not is_mc:
        v = V.View(ch, 'data')
        sel, dm = V.base_selection(v, False, a, 'mass')
        for reg, m in dict(SR=sel & (dm < a.sr), SB=sel & (dm > a.sb[0]) & (dm < a.sb[1])).items():
            counts['data_' + reg] = counts.get('data_' + reg, 0) + int(m.sum())
            if m.any():
                fill_view(V.View({k: x[m] for k, x in ch.items()}, 'data'), False, 'data_' + reg,
                          np.ones(int(m.sum())), store)
        counts['sel'] = counts.get('data_SR', 0)
        return
    masks = {}
    for mode, mass_name in (('raw', 'mass_raw'), ('corr', 'mass')):
        v = V.View(ch, mode)
        sel, dm = V.base_selection(v, True, a, mass_name)
        m = sel & (dm < a.sr)
        masks[mode] = m
        counts['mc_' + mode] = counts.get('mc_' + mode, 0) + int(m.sum())
        if not m.any():
            continue
        w = np.asarray(ch[a.pu_weight][m], np.float64)
        if mode == 'corr':
            w = w * np.asarray(ch['emu_weight'][m], np.float64)
        fill_view(V.View({k: x[m] for k, x in ch.items()}, mode), True, 'mc_' + mode, w, store)
    both = masks['raw'] & masks['corr']
    if both.any():
        w = np.asarray(ch[a.pu_weight][both], np.float64) * np.asarray(ch['emu_weight'][both], np.float64)
        fill_ratios(ch, both, w, store)
    counts['sel'] = counts.get('mc_corr', 0)


def fill(files, label, out, a):
    import uproot
    with uproot.open(files[0]) as fh:
        have = set(fh['tree'].keys())
    is_mc = 'gen_pv_x' in have if a.kind == 'auto' else a.kind == 'mc'
    need = V.needed_branches(is_mc, a)
    missing = sorted(n for n in need if n not in have)
    if missing:
        V.die('%s: branches missing: %s' % (files[0], ', '.join(missing)))
    total = 0
    for fn in files:
        with uproot.open(fn) as fh:
            total += fh['tree'].num_entries
    if a.max_entries:
        total = min(total, a.max_entries)
    print('=== fill %s (%s): %d file(s), %s entries' % (label, 'MC' if is_mc else 'data',
                                                     len(files), f'{total:,}'))
    store, counts = Store(), {}
    prog = V.Progress(total, label)
    done = 0
    for chunk in uproot.iterate([{fn: 'tree'} for fn in files], expressions=sorted(need),
                                library='np', step_size=a.step):
        n = len(chunk['mass'])
        if a.max_entries and done + n > a.max_entries:
            n = a.max_entries - done
            chunk = {k: x[:n] for k, x in chunk.items()}
        fill_chunk(chunk, is_mc, store, counts, a)
        done += n
        prog.update(done, 'selected %s' % f'{counts.get("sel", 0):,}')
        if a.max_entries and done >= a.max_entries:
            break
    prog.update(total)
    store.save(out, dict(label=label, is_mc=is_mc, files=files, entries=done, counts=counts,
                         args=dict(sr=a.sr, sb=list(a.sb), ct_max=a.ct_max, pu_weight=a.pu_weight)))
    print('  selected: %s   -> %s' % (', '.join('%s %s' % (k, f'{int(x):,}')
                                                for k, x in counts.items()), out))


# ---------------------------------------------------------------------------
# combine, medians, plot
# ---------------------------------------------------------------------------
def combine(stores, a):
    f_sb = (2 * a.sr) / (2 * (a.sb[1] - a.sb[0]))
    H, eras = {}, []

    def add(key, w):
        H[key] = H[key] + w if key in H else w.copy()

    for meta, h in stores:
        if not meta['is_mc']:
            eras.append(meta['label'])
            for (q, b, v), (w, _) in h.items():
                if v == 'data_SR':
                    sb = h.get((q, b, 'data_SB'), (np.zeros_like(w),))[0]
                    add((q, b, 'data'), w - f_sb * sb)
                    add((q, b, 'data@' + meta['label']), w - f_sb * sb)
        else:
            for k, (w, _) in h.items():
                add(k, w)
    return H, eras


def median_err(h, e, log=True):
    '''median of a (log10) histogram and a rough error: the median of a
    Gaussian has sigma ~ 1.25 sigma / sqrt(N), sigma from the 16-84% quantiles'''
    m = V.quantile(h, e, 0.5)
    q16, q84 = V.quantile(h, e, 0.16), V.quantile(h, e, 0.84)
    n = max(np.clip(h, 0, None).sum(), 1.0)
    s = 1.25 * 0.5 * (q84 - q16) / np.sqrt(n)
    return (10 ** m, 10 ** m * np.log(10) * s) if log else (m, s)


def medians(H):
    M = {}
    for (q, b, v), w in H.items():
        e = edges_of(q)
        M[(q, b, v)] = [median_err(w[i], e) for i in range(w.shape[0])]
    return M


def bin_labels(b):
    if b == 'mu_pt_x_eta':
        ne = len(MU_ETA) - 1
        return ['%g-%s|%g-%g' % (MU_PT[i // ne], ('%g' % MU_PT[i // ne + 1]) if MU_PT[i // ne + 1] < 1e3 else 'inf',
                                 MU_ETA[i % ne], MU_ETA[i % ne + 1]) for i in range((len(MU_PT) - 1) * ne)]
    be = BINS[b]
    return ['%s-%s' % (V._fe(be[i]), V._fe(be[i + 1])) for i in range(len(be) - 1)]


def summary_lines(M, eras):
    L = ['vertex_errors: stated errors (medians, um), data (SR - SB) vs MC raw vs MC corrected',
         'data: %s' % ', '.join(eras), '']
    L.append('%-12s %-10s %9s %9s %9s %9s %9s' % ('quantity', 'bin', 'data', 'MC raw', 'MC corr',
                                                   'raw/data', 'corr/data'))
    for q in VTX_Q + TRK_Q:
        bins = ('incl', 'jpsi_pt') if q in VTX_Q else ('incl', 'mu_pt')
        for b in bins:
            d, r, c = (M.get((q, b, v)) for v in ('data', 'mc_raw', 'mc_corr'))
            if d is None or r is None or c is None:
                continue
            for i, lab in enumerate(bin_labels(b)):
                L.append('%-12s %-10s %9.2f %9.2f %9.2f %9.3f %9.3f'
                         % (q, lab, d[i][0], r[i][0], c[i][0], r[i][0] / d[i][0], c[i][0] / d[i][0]))
        L.append('')
    L.append('same MC event, corrected / raw stated VARIANCE (median):')
    for q in RAT_Q:
        x = M.get((q, 'incl', 'mc_ratio'))
        if x is not None:
            L.append('  %-12s %.3f   (sigma ratio %.3f)' % (q, x[0][0], np.sqrt(x[0][0])))
    L += ['', 'reading: pv_* corr/data ~ 1 and sv_* corr/data ~ 1.3 -> the emulation overshoots at the',
          'vertex; pv_* corr/data < 1 -> the MC PV (beamspot) is narrower, not a covflow problem.',
          'Compare per bin: data are prompt J/psi, MC J/psi from B.']
    return L


def plot(paths, out, a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    stores = [load_store(p) for p in paths]
    H, eras = combine(stores, a)
    M = medians(H)
    lines = summary_lines(M, eras)
    print('\n'.join(lines))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
    json.dump({'|'.join(k): v for k, v in M.items()}, open(out + '.json', 'w'), indent=0)
    style = {'data': dict(color='black', marker='o', label='data (SR - SB)'),
             'mc_raw': dict(color='#2f6eba', marker='s', label='MC raw'),
             'mc_corr': dict(color='#d1495b', marker='D', label='MC corrected')}
    with PdfPages(out + '.pdf') as pdf:
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(0.02, 0.98, '\n'.join(lines), family='monospace', fontsize=5.6, va='top')
        pdf.savefig(fig)
        plt.close(fig)
        for q in VTX_Q + TRK_Q:
            b = 'jpsi_pt' if q in VTX_Q else 'mu_pt_x_eta'
            e = edges_of(q)
            c = 0.5 * (e[1:] + e[:-1])
            fig = plt.figure(figsize=(13, 7.5))
            gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.3])
            a00 = fig.add_subplot(gs[:, 0])
            a01 = fig.add_subplot(gs[0, 1])
            a11 = fig.add_subplot(gs[1, 1], sharex=a01)
            ax = np.array([[a00, a01], [None, a11]], dtype=object)
            a01.tick_params(labelbottom=False)
            lo, hi_ = [], []
            for v, st in style.items():
                hi = H.get((q, 'incl', v))
                if hi is not None:
                    y = np.clip(hi[0], 0, None)
                    ax[0, 0].plot(10 ** c, y / max(y.sum(), 1e-300), drawstyle='steps-mid',
                                  color=st['color'], label=st['label'])
                    lo.append(V.quantile(y, e, 0.001)); hi_.append(V.quantile(y, e, 0.999))
            ax[0, 0].set_xscale('log')
            if lo:
                ax[0, 0].set_xlim(10 ** (min(lo) - 0.05), 10 ** (max(hi_) + 0.05))
            ax[0, 0].set_ylabel('fraction per bin')
            ax[0, 0].set_title(LABEL[q] + ', inclusive', fontsize=10, loc='left')
            ax[0, 0].legend(fontsize=8, frameon=False)
            labs = bin_labels(b)
            x = np.arange(len(labs))
            md = M.get((q, b, 'data'))
            for v, st in style.items():
                mv = M.get((q, b, v))
                if mv is None:
                    continue
                y, ye = np.array(mv).T
                ax[0, 1].errorbar(x, y, yerr=ye, fmt=st['marker'] + '-', color=st['color'], ms=4,
                                  label=st['label'])
                if md is not None and v != 'data':
                    yd = np.array(md)[:, 0]
                    ax[1, 1].plot(x, y / yd, st['marker'] + '-', color=st['color'], ms=4)
            ax[0, 1].set_ylabel(r'median stated $\sigma$ [$\mu$m]')
            ax[0, 1].set_title('medians vs ' + b, fontsize=10, loc='left')
            ax[1, 1].axhline(1, color='grey', lw=0.7, ls=':')
            ax[1, 1].set_ylabel('MC / data')
            ax[1, 1].set_xticks(x)
            ax[1, 1].set_xticklabels(labs, rotation=60, fontsize=6.5)
            ax[0, 0].set_xlabel(r'stated $\sigma$ [$\mu$m]')
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        e = RAT_EDGES
        c = 0.5 * (e[1:] + e[:-1])
        for q, col in zip(RAT_Q, ('#7a3db8', '#e09f3e', '#2f6eba', '#d1495b')):
            hi = H.get((q, 'incl', 'mc_ratio'))
            if hi is not None:
                y = np.clip(hi[0], 0, None)
                ax[0].plot(10 ** c, y / max(y.sum(), 1e-300), drawstyle='steps-mid', color=col, label=LABEL[q])
            mv = M.get((q, 'jpsi_pt' if q.startswith('r_sv') else 'mu_pt', 'mc_ratio'))
            if mv is not None:
                y = np.array(mv)[:, 0]
                ax[1].plot(np.arange(len(y)), y, 'o-', color=col, label=q)
        ax[0].set_xscale('log')
        ax[0].axvspan(1.3, 1.5, color='grey', alpha=0.2, lw=0, label='needed by note sec. 10 (across)')
        ax[0].set_xlabel('stated variance, MC corrected / MC raw, same event')
        ax[0].legend(fontsize=7, frameon=False)
        ax[1].set_ylabel('median ratio')
        ax[1].set_xlabel('bin (SV: J/psi pT; tracks: muon pT)')
        ax[1].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print('-> %s.pdf, %s.json' % (out, out))


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(q):
        q.add_argument('--sr', type=float, default=0.05, help='signal region |m - mJ/psi| < SR [GeV]')
        q.add_argument('--sb', type=float, nargs=2, default=(0.15, 0.25),
                       help='sidebands SB0 < |m - mJ/psi| < SB1 [GeV]')

    def filling(q):
        q.add_argument('--ct-max', type=float, default=0.01,
                       help='prompt cut ct < CT_MAX [cm], same in data and MC (default 0.01)')
        q.add_argument('--pu-weight', default='pu_weight_2026')
        q.add_argument('--kind', choices=['auto', 'data', 'mc'], default='auto')
        q.add_argument('--max-entries', type=int, default=None)
        q.add_argument('--step', default='300 MB', help='uproot chunk size')

    f = sub.add_parser('fill', help='read ntuples, write summable histograms (.npz)')
    f.add_argument('--files', nargs='+', required=True)
    f.add_argument('--label', required=True)
    f.add_argument('--out', required=True)
    common(f)
    filling(f)
    g = sub.add_parser('plot', help='combine .npz files, medians, plot')
    g.add_argument('inputs', nargs='+')
    g.add_argument('--out', required=True)
    common(g)
    al = sub.add_parser('all', help='fill the default files one after the other, then plot')
    al.add_argument('--out', required=True)
    al.add_argument('--hist-dir', default=None, help='where the .npz go (default: next to --out)')
    common(al)
    filling(al)
    return p.parse_args()


def main():
    a = parse_args()
    if a.cmd == 'fill':
        fill(a.files, a.label, a.out, a)
    elif a.cmd == 'plot':
        plot(a.inputs, a.out, a)
    else:
        hd = a.hist_dir or (os.path.dirname(os.path.abspath(a.out)) or '.')
        paths = []
        for lab, fn in V.DEFAULT_FILES.items():
            pth = os.path.join(hd, 'verr_%s.npz' % lab)
            fill([fn], lab, pth, a)
            paths.append(pth)
        plot(paths, a.out, a)


if __name__ == '__main__':
    main()
