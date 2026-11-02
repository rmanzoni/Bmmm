'''
Plots of the fit-scheme comparison. Every function writes one figure (PNG,
and PDF with pdf=True) and returns its file name, relative to the output
directory, for the report.
'''

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cfcore import robust_stats, roc_curve, auc_oriented
from cfio import (CLASSES, CLASS_LABEL, CLASS_COLOR, GROUPS, SCHEME_LABEL, TARGETS,
                  GAMMA_TARGETS, OBS_LABEL, UNITS, VERTICES, scheme_color, residual)
from cfanalysis import bs_vertex_residuals, hyp_branch

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9, 'legend.fontsize': 7,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'figure.dpi': 100,
    'axes.grid': True, 'grid.alpha': 0.25, 'axes.spines.top': False, 'axes.spines.right': False,
})

HYP_LABEL = {'coll': 'collinear', 'xyz_bs': 'xyz (PV$\\to$V$_{Bs}$)', 'math1_bs': 'math 1',
             'math2_bs': 'math 2', 'math_avg': 'math (average)', 'math_oracle': 'math (truth-picked)',
             'gamma_coll': 'collinear, D$_s\\gamma$', 'gamma_xyz_bs': 'xyz, D$_s\\gamma$'}


class Plotter(object):

    def __init__(self, outdir, layout, d, m, pdf=False):
        self.outdir = outdir
        self.layout = layout
        self.d = d
        self.m = m
        self.cls = d['gen_cand'].astype(int)
        self.pdf = pdf
        os.makedirs(os.path.join(outdir, 'plots'), exist_ok=True)

    # ---- helpers ------------------------------------------------------------
    def save(self, fig, name):
        rel = os.path.join('plots', name + '.png')
        fig.savefig(os.path.join(self.outdir, rel), bbox_inches='tight', dpi=110)
        if self.pdf:
            fig.savefig(os.path.join(self.outdir, 'plots', name + '.pdf'), bbox_inches='tight')
        plt.close(fig)
        return rel

    def color(self, s):
        return scheme_color(s, self.layout.schemes)

    def sel(self, classes, extra=None):
        mm = self.m & np.isin(self.cls, classes)
        if extra is not None:
            mm &= extra
        return mm

    @staticmethod
    def _hist(ax, x, bins, rng, **kw):
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return
        ax.hist(np.clip(x, rng[0], rng[1]), bins=bins, range=rng, histtype='step',
                density=True, linewidth=1.3, **kw)

    def all_hyps(self):
        out = []
        for s in self.layout.schemes:
            for h in self.layout.hyps[s]:
                if h not in out:
                    out.append(h)
        return out

    # ---- overview ---------------------------------------------------------
    def cutflow(self, rows):
        fig, ax = plt.subplots(figsize=(7, 3))
        names = [r[0] for r in rows]
        x = np.arange(len(names))
        wd = 0.2
        for i, c in enumerate(CLASSES):
            ax.bar(x + (i - 1.5) * wd, [r[1][i] for r in rows], wd, color=CLASS_COLOR[c], label=CLASS_LABEL[c])
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.set_ylabel('candidates')
        ax.legend(ncol=4)
        return self.save(fig, 'cutflow')

    def efficiency(self, eff):
        keys = [k for k in ('good_phi', 'good_ds', 'good_bs', 'vtx_cut', 'math_real')
                if any(k in eff[s]['0'] for s in eff)]
        fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 2.8), squeeze=False)
        schemes = list(eff)
        wd = 0.8 / len(schemes)
        for ax, k in zip(axes[0], keys):
            for i, s in enumerate(schemes):
                v = [eff[s][str(c)].get(k, np.nan) for c in CLASSES]
                ax.bar(np.arange(4) + (i - (len(schemes) - 1) / 2.) * wd, v, wd,
                       color=self.color(s), label=SCHEME_LABEL.get(s, s))
            ax.set_xticks(np.arange(4))
            ax.set_xticklabels([CLASS_LABEL[c] for c in CLASSES])
            lo = np.nanmin([eff[s][str(c)].get(k, np.nan) for s in schemes for c in CLASSES])
            ax.set_ylim(max(0., lo - 0.05), 1.005)
            ax.set_title(k.replace('_', ' '))
        axes[0][0].legend(loc='lower left')
        return self.save(fig, 'efficiency')

    # ---- fit quality -----------------------------------------------------------
    def probabilities(self):
        L = self.layout
        fig, axes = plt.subplots(len(VERTICES), len(L.schemes),
                                 figsize=(2.9 * len(L.schemes), 2.3 * len(VERTICES)), squeeze=False)
        for i, v in enumerate(VERTICES):
            for j, s in enumerate(L.schemes):
                ax = axes[i][j]
                b = '%s_%s_prob' % (s, v)
                if b not in self.d:
                    ax.axis('off')
                    continue
                for g, col in (('mu', '#1b6ca8'), ('tau', '#c0392b')):
                    x = self.d[b][self.sel(GROUPS[g])]
                    self._hist(ax, x, 50, (0, 1), color=col, label=g + ' channels')
                ax.set_yscale('log')
                ax.set_title('%s: P(V$_{%s}$)' % (s, v))
                if i == 0 and j == 0:
                    ax.legend()
        return self.save(fig, 'probabilities')

    def masses(self):
        L, d = self.layout, self.d
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
        ax = axes[0]
        m = self.sel((0, 1, 2, 3))
        series = [('kkpi_mass', 'm(KK$\\pi$), tracks', 'k')]
        for s in L.schemes:
            b = '%s_ds_mass' % s
            if b in d and np.nanstd(d[b][m]) > 1e-5:
                series.append((b, '%s Ds fit' % s, self.color(s)))
        for b, lab, col in series:
            if b in d:
                st = robust_stats(d[b][m])
                self._hist(ax, d[b][m], 80, (1.90, 2.04), color=col, label='%s, $\\sigma_{68}$ %.1f MeV' % (lab, 1e3 * st['s68']))
        ax.set_xlabel('m(KK$\\pi$) [GeV]')
        ax.legend()
        ax = axes[1]
        series = [('kk_mass', 'm(KK), tracks', 'k')]
        for s in L.schemes:
            b = '%s_phi_mass' % s
            if b in d and np.nanstd(d[b][m]) > 1e-5:
                series.append((b, '%s $\\phi$ fit' % s, self.color(s)))
        for b, lab, col in series:
            if b in d:
                st = robust_stats(d[b][m])
                self._hist(ax, d[b][m], 80, (1.0, 1.04), color=col, label='%s, $\\sigma_{68}$ %.2f MeV' % (lab, 1e3 * st['s68']))
        ax.set_xlabel('m(KK) [GeV]')
        ax.legend()
        ax = axes[2]
        if 'gen_ds_pt' in d:
            for s in L.schemes:
                b = '%s_ds_pt' % s
                if b in d:
                    r = d[b][m] / d['gen_ds_pt'][m] - 1.
                    st = robust_stats(r)
                    self._hist(ax, r, 80, (-0.05, 0.05), color=self.color(s),
                               label='%s, $\\sigma_{68}$ %.2f%%' % (s, 100 * st['s68']))
            ax.set_xlabel('p$_T$(D$_s$) reco / gen $-$ 1')
            ax.legend()
        return self.save(fig, 'masses')

    def vertex(self):
        L, d = self.layout, self.d
        if not all(b in d for b in ('gen_bs_sv_x', 'gen_bs_eta', 'pv_x')):
            return None
        rows = [('long', 'V$_{Bs}$ along the B$_s$ flight [$\\mu$m]', 1e4, None),
                ('t1', 'V$_{Bs}$ across, transverse plane [$\\mu$m]', 1e4, None),
                ('long_pull', 'pull, along', 1., (-6, 6)),
                ('t1_pull', 'pull, across', 1., (-6, 6)),
                ('angle', 'angle(PV$\\to$V$_{Bs}$, true B$_s$) [mrad]', 1e3, None)]
        groups = ('mu', 'tau')
        fig, axes = plt.subplots(len(rows), 2, figsize=(9.5, 2.7 * len(rows)), squeeze=False)
        res = {}
        for s in L.schemes:
            if '%s_bs_cov_xx' % s not in d:
                continue
            for g in groups:
                res[(s, g)] = bs_vertex_residuals(d, s, self.sel(GROUPS[g]))
        for i, (k, lab, sc, rng) in enumerate(rows):
            for j, g in enumerate(groups):
                ax = axes[i][j]
                vals = [res[(s, g)][k] * sc for s in L.schemes if (s, g) in res]
                if not vals:
                    continue
                r = rng          # per panel: the tau channels need a wider range
                if r is None:
                    allv = np.concatenate([v[np.isfinite(v)] for v in vals])
                    if k == 'angle':
                        r = (0., np.quantile(allv, 0.95))
                    else:
                        q = np.quantile(allv, [0.03, 0.97])
                        r = (q[0], q[1])
                for s in L.schemes:
                    if (s, g) not in res:
                        continue
                    v = res[(s, g)][k] * sc
                    st = robust_stats(v)
                    self._hist(ax, v, 60, r, color=self.color(s),
                               label='%s: med %.3g, $\\sigma_{68}$ %.3g' % (s, st['median'], st['s68']))
                ax.set_xlabel(lab, fontsize=8)
                if i == 0:
                    ax.set_title('%s channels (reco $-$ true; first and last bins: overflow)' % g, fontsize=9)
                ax.legend(fontsize=6)
        return self.save(fig, 'vertex')

    # ---- resolution ---------------------------------------------------------
    def residuals(self, q):
        L, d = self.layout, self.d
        hyps = [h for h in self.all_hyps() if any(q in L.hyp_obs(s, h) for s in L.schemes)]
        gam = sorted(set(h for s in L.schemes for h in L.gamma_hyps[s] if q in L.hyp_obs(s, h)))
        hyps = hyps + gam
        if q not in TARGETS or not hyps:
            return None
        rows = CLASSES
        fig, axes = plt.subplots(len(rows), len(hyps), figsize=(2.6 * len(hyps), 2.1 * len(rows)), squeeze=False)
        for i, c in enumerate(rows):
            mc = self.sel((c,))
            for j, h in enumerate(hyps):
                ax = axes[i][j]
                tg = GAMMA_TARGETS if h.startswith('gamma_') else TARGETS
                gb, kind = tg[q]
                if gb not in d:
                    ax.axis('off')
                    continue
                rs = []
                for s in L.schemes:
                    b = hyp_branch(s, h, q)
                    if b in d:
                        rs.append((s, residual(d[b][mc], d[gb][mc], kind)))
                if not rs:
                    ax.axis('off')
                    continue
                st_all = [robust_stats(r) for _, r in rs]
                med = np.nanmedian([st['median'] for st in st_all])
                wid = np.nanmax([st['s68'] for st in st_all])
                if not np.isfinite(wid) or wid <= 0:
                    ax.axis('off')
                    continue
                rng = (med - 4 * wid, med + 4 * wid)
                for (s, r), st in zip(rs, st_all):
                    self._hist(ax, r, 60, rng, color=self.color(s),
                               label='%s %.3g / %.3g' % (s, st['median'], st['s68']))
                ax.axvline(0, color='grey', lw=0.6)
                if i == 0:
                    ax.set_title(HYP_LABEL.get(h, h))
                if j == 0:
                    ax.set_ylabel(CLASS_LABEL[c])
                if i == len(rows) - 1:
                    ax.set_xlabel('%s reco $-$ gen %s' % (OBS_LABEL.get(q, q), UNITS.get(q, '')))
                ax.legend(title='median / $\\sigma_{68}$', title_fontsize=6)
        return self.save(fig, 'res_%s' % q)

    def q2_vs_gen(self, hyps=('coll', 'xyz_bs', 'math_avg')):
        L, d = self.layout, self.d
        if 'gen_q2_ds' not in d:
            return None
        hyps = [h for h in hyps if any(h in L.hyps[s] for s in L.schemes)]
        fig, axes = plt.subplots(2 * len(hyps), len(L.schemes),
                                 figsize=(2.6 * len(L.schemes), 2.5 * 2 * len(hyps)), squeeze=False)
        for k, h in enumerate(hyps):
            for gi, g in enumerate(('mu', 'tau')):
                mg = self.sel(GROUPS[g])
                for j, s in enumerate(L.schemes):
                    ax = axes[2 * k + gi][j]
                    b = hyp_branch(s, h, 'q2')
                    if b not in d:
                        ax.axis('off')
                        continue
                    x, y = d['gen_q2_ds'][mg], d[b][mg]
                    ok = np.isfinite(x) & np.isfinite(y)
                    ax.hist2d(x[ok], np.clip(y[ok], -4, 14), bins=60, range=((-1, 12), (-4, 14)), cmin=1, cmap='viridis')
                    ax.plot([-1, 12], [-1, 12], color='w', lw=0.7)
                    ax.set_title('%s, %s, %s' % (s, HYP_LABEL.get(h, h), g), fontsize=7)
                    if gi == 1 and k == len(hyps) - 1:
                        ax.set_xlabel('gen q$^2$ = (B$_s$ $-$ D$_s$)$^2$ [GeV$^2$]')
                    if j == 0:
                        ax.set_ylabel('reco q$^2$')
        return self.save(fig, 'q2_vs_gen')

    def heatmap(self, name, rows, cols, values, title, fmt='%.3g', mode='lower', cmap='RdYlGn_r', ref_cols=()):
        '''rows x cols matrix of values, coloured by how far each is from the
        best of its row: mode 'lower' (smaller is better): value / row minimum;
        mode 'auc' (AUC, larger is better): (best - 0.5) / (value - 0.5). Both
        are 1 for the best entry, and the colour scale saturates at 1.5.
        ref_cols: reference columns (generator truth, truth-picked solution),
        printed but grey and left out of the row best.'''
        V = np.array(values, dtype=float)
        C = np.full_like(V, np.nan)
        rank = np.ones(V.shape[1], dtype=bool)
        rank[list(ref_cols)] = False
        for i in range(V.shape[0]):
            row = V[i][rank & np.isfinite(V[i])]
            if not len(row):
                continue
            if mode == 'auc':
                best = np.max(row)
                with np.errstate(divide='ignore', invalid='ignore'):
                    C[i] = (best - 0.5) / (V[i] - 0.5)
            else:
                best = np.min(row)
                with np.errstate(divide='ignore', invalid='ignore'):
                    C[i] = V[i] / best if best > 0 else np.nan
        fig, ax = plt.subplots(figsize=(0.62 * len(cols) + 2.8, 0.30 * len(rows) + 1.8))
        C[:, ~rank] = np.nan
        ax.imshow(np.where(rank[None, :], np.nan, 1.) * np.ones_like(V), aspect='auto', cmap='Greys', vmin=0, vmax=6)
        im = ax.imshow(np.clip(C, 1., 1.5), aspect='auto', cmap=cmap, vmin=1., vmax=1.5)
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                if np.isfinite(V[i, j]):
                    ax.text(j, i, fmt % V[i, j], ha='center', va='center', fontsize=5.5)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=60, ha='right', fontsize=7)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=7)
        ax.grid(False)
        ax.set_title(title)
        cb = fig.colorbar(im, ax=ax, fraction=0.03)
        cb.set_label('value / best of the row' if mode == 'lower' else '(best $-$ 0.5) / (AUC $-$ 0.5)')
        return self.save(fig, name)

    # ---- separation -------------------------------------------------------
    def shapes(self, q, sep):
        L, d = self.layout, self.d
        hyps = [h for h in self.all_hyps() if any(q in L.hyp_obs(s, h) for s in L.schemes)]
        if not hyps:
            return None
        fig, axes = plt.subplots(len(L.schemes), len(hyps),
                                 figsize=(2.5 * len(hyps), 2.0 * len(L.schemes)), squeeze=False)
        allx = np.concatenate([d[hyp_branch(s, h, q)][self.m] for s in L.schemes for h in hyps
                               if hyp_branch(s, h, q) in d])
        allx = allx[np.isfinite(allx)]
        if not len(allx):
            plt.close(fig)
            return None
        # range: that of the generator-level quantity, widened by 25% on each
        # side (the badly reconstructed solutions would otherwise set it)
        gb = TARGETS.get(q, (None,))[0]
        if gb in d and np.any(np.isfinite(d[gb][self.m])):
            g = d[gb][self.m]
            lo, hi = np.quantile(g[np.isfinite(g)], [0.005, 0.995])
            rng = (lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo))
        else:
            rng = tuple(np.quantile(allx, [0.02, 0.98]))
        for i, s in enumerate(L.schemes):
            for j, h in enumerate(hyps):
                ax = axes[i][j]
                b = hyp_branch(s, h, q)
                if b not in d:
                    ax.axis('off')
                    continue
                for c in CLASSES:
                    self._hist(ax, d[b][self.sel((c,))], 50, rng, color=CLASS_COLOR[c], label=CLASS_LABEL[c])
                r = sep.get(b, {})
                a = r.get('tau_vs_mu', {}).get('auc')
                f = r.get('fisher', {})
                txt = 'AUC $\\tau/\\mu$ %.3f' % a if a else ''
                if f.get('r_taumu'):
                    txt += '\n$\\sigma$ $\\tau/\\mu$ %.4f' % f['r_taumu']
                ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha='right', va='top', fontsize=6)
                if i == 0:
                    ax.set_title(HYP_LABEL.get(h, h))
                if j == 0:
                    ax.set_ylabel(s)
                if i == len(L.schemes) - 1:
                    ax.set_xlabel(OBS_LABEL.get(q, q) + ' ' + UNITS.get(q, ''))
        axes[0][0].legend(fontsize=6)
        return self.save(fig, 'shape_%s' % q)

    def topology_shapes(self, sep, obs):
        L, d = self.layout, self.d
        obs = [q for q in obs if any('%s_%s' % (s, q) in d for s in L.schemes)]
        if not obs:
            return None
        fig, axes = plt.subplots(len(obs), len(L.schemes),
                                 figsize=(2.6 * len(L.schemes), 1.9 * len(obs)), squeeze=False)
        for i, q in enumerate(obs):
            allx = np.concatenate([d['%s_%s' % (s, q)][self.m] for s in L.schemes if '%s_%s' % (s, q) in d])
            allx = allx[np.isfinite(allx)]
            if not len(allx):
                continue
            rng = tuple(np.quantile(allx, [0.005, 0.99]))
            if rng[0] == rng[1]:
                rng = (rng[0] - 0.5, rng[1] + 0.5)
            for j, s in enumerate(L.schemes):
                ax = axes[i][j]
                b = '%s_%s' % (s, q)
                if b not in d:
                    ax.axis('off')
                    continue
                for c in CLASSES:
                    self._hist(ax, d[b][self.sel((c,))], 50, rng, color=CLASS_COLOR[c], label=CLASS_LABEL[c])
                a = sep.get(b, {}).get('tau_vs_mu', {}).get('auc')
                if a:
                    ax.text(0.98, 0.95, 'AUC %.3f' % a, transform=ax.transAxes, ha='right', va='top', fontsize=6)
                ax.set_title('%s_%s' % (s, q), fontsize=7)
        axes[0][0].legend(fontsize=6)
        return self.save(fig, 'topology_shapes')

    def rocs(self, curves, name, title):
        '''curves: list of (label, color, linestyle, sig values, bkg values).'''
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        for lab, col, ls, sg, bg in curves:
            a, sign = auc_oriented(sg, bg)
            if not np.isfinite(a):
                continue
            eb, es = roc_curve(sg, bg, direction=sign)
            ax.plot(eb, es, color=col, ls=ls, lw=1.3, label='%s (AUC %.3f)' % (lab, a))
        ax.plot([0, 1], [0, 1], color='grey', lw=0.6)
        ax.set_xlabel('$\\mu$-channel efficiency')
        ax.set_ylabel('$\\tau$-channel efficiency')
        ax.set_title(title)
        ax.legend(fontsize=6, loc='lower right')
        return self.save(fig, name)

    def fisher_bars(self, entries, name, title):
        '''entries: list of (label, color, fisher dict from cfcore.fisher_ratios).
        Three panels: R(Ds), R(Ds*) (four-yield fit) and N(tau)/N(mu)
        (two-yield fit); the number in brackets is the factor above the
        counting limit.'''
        if not entries:
            return None
        panels = (('r_ds', 'dil_ds', 'R(D$_s$)'), ('r_dsst', 'dil_dsst', 'R(D$_s^*$)'),
                  ('r_taumu', 'dil_taumu', 'N($\\tau$)/N($\\mu$), D$_s$+D$_s^*$'))
        fig, axes = plt.subplots(1, 3, figsize=(14, 0.24 * len(entries) + 1.4), sharey=True)
        y = np.arange(len(entries))[::-1]
        for ax, (k, kd, lab) in zip(axes, panels):
            v = np.array([np.nan if e[2].get(k) is None else e[2][k] for e in entries], dtype=float)
            dil = [e[2].get(kd) for e in entries]
            ax.barh(y, np.nan_to_num(v), color=[e[1] for e in entries])
            for yy, vv, dd in zip(y, v, dil):
                if np.isfinite(vv):
                    ax.text(vv, yy, ' %.4f (x%.1f)' % (vv, dd if dd is not None else np.nan), va='center', fontsize=6)
            ax.set_xlabel('expected $\\sigma_{rel}$, %s' % lab, fontsize=8)
            if np.any(np.isfinite(v)):
                ax.set_xlim(0, 1.35 * np.nanmax(v))
        axes[0].set_yticks(y)
        axes[0].set_yticklabels([e[0] for e in entries], fontsize=7)
        fig.suptitle(title, fontsize=9)
        return self.save(fig, name)

    def bdt_scores(self, scores, names):
        d = self.d
        names = [n for n in names if n in scores]
        if not names:
            return None
        fig, axes = plt.subplots(1, len(names), figsize=(2.8 * len(names), 2.5), squeeze=False)
        for ax, n in zip(axes[0], names):
            idx, p_tau, _ = scores[n]
            y = d['gen_cand'][idx].astype(int)
            for c in CLASSES:
                self._hist(ax, p_tau[y == c], 40, (0, 1), color=CLASS_COLOR[c], label=CLASS_LABEL[c])
            ax.set_title(n, fontsize=7)
            ax.set_xlabel('p($\\tau$) of the BDT')
        axes[0][0].legend(fontsize=6)
        return self.save(fig, 'bdt_scores')

    def math_choice(self):
        L, d = self.layout, self.d
        keys = [s for s in L.schemes if '%s_math_oracle_is_1' % s in d]
        if not keys:
            return None
        fig, axes = plt.subplots(1, 3, figsize=(10, 2.7))
        wd = 0.8 / len(keys)
        for k, (ax, (b, title)) in enumerate(zip(axes, (('%s_math_real_bs', 'real solutions'),
                                                       ('%s_math_oracle_is_1', 'math 1 closer to the true |p(B$_s$)|'),
                                                       ('%s_math_choice_low_is_1', 'math 1 has the lower |p(B$_s$)|')))):
            for i, s in enumerate(keys):
                bb = b % s
                if bb not in d:
                    continue
                v = [np.nanmean(d[bb][self.sel((c,))] > 0.5) for c in CLASSES]
                ax.bar(np.arange(4) + (i - (len(keys) - 1) / 2.) * wd, v, wd, color=self.color(s), label=s)
            ax.set_xticks(np.arange(4))
            ax.set_xticklabels([CLASS_LABEL[c] for c in CLASSES])
            ax.set_ylim(0, 1.02)
            ax.set_title(title)
        axes[0].legend()
        return self.save(fig, 'math_choice')
