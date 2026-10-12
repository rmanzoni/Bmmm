#!/usr/bin/env python3
'''
covflow_closure_plots.py -- does the emulated MC describe the data?

Three distributions per quantity, per epoch and for all epochs together:

  data             the dimuon data ntuples, background subtracted with sPlot
                   (J/psi mass fit, covflow.splot, the model covflow uses)
  MC raw           the emulated MC ntuples read through their *_raw branches:
                   the MC as it was before the emulation
  MC corrected     the same MC events, nominal branches: hit kill, route A +
                   delta_1, covflow, delta_2 (inspector --emulate)

plus the dimuon mass RESOLUTION: the sWeighted data mass is not shown (it is
the fitted signal model by construction), the fitted data signal shape is
compared with the MC raw / corrected mass distributions instead, with sigma68
(half the 15.87-84.13% interval) for each.

Both MC curves come from the SAME events, so their difference is the effect of
the emulation alone; the ratio panel and a chi2/ndf against data say which of
the two describes the data.

WEIGHTS
  data           signal sWeights from an extended ML fit of the dimuon mass in
                 +-250 MeV (2 DSCB + exp by default, binned, 1 MeV). The weights
                 subtract the background only summed over the WHOLE fit window,
                 so the mass terms of the selection are dropped and the window
                 is applied instead, to data AND MC (as covflow's training does).
                 Negative weights are kept.
  MC raw         pu_weight_<year> (the year of the epoch, as in the training)
  MC corrected   pu_weight_<year> x emu_weight (hit-efficiency weight of the
                 emulation, product of the two muons': cells where data is more
                 efficient than MC are not killed but weighted)
  --kin-reweight optional: on top, a per-event weight in the J/psi (pT, |eta|)
                 from sWeighted data / MC corrected, applied to both MC views.
                 Off by default: the kinematics are an input, not what is tested.
  Normalisation: each epoch's MC to that epoch's data signal yield (sum of
                 sWeights), so "all epochs" is luminosity weighted by construction.

RAW VIEW OF THE MC
  Every branch X is read as X_raw when the ntuple has it (mass_raw, lxy_sig_raw,
  mu1_pt_raw, mu1_dxy_e_raw, ...). Derived raw quantities: mu<i>_dxy_sig_raw =
  dxy_raw / dxy_e_raw, mu<i>_dz_sig_raw = dz_raw / dz_e_raw. Branches the
  emulation does not touch (trigger, muon ID, charge, gen, cov_* which are RAW by
  construction) fall back to the nominal branch; the list of fallbacks is
  printed. A quantity the emulation changes but whose raw value is not stored
  (candidate eta / phi, mu<i>_bs_dxy) is refused rather than silently taken
  from the corrected MC.
  The emulated muons are ordered by the SMEARED pT; in the raw view mu1 and mu2
  are swapped back where the raw pT order differs, so "leading muon" means the
  same thing as in a raw ntuple.

The selection is covflow's Run 3 training selection (configs/run3_epochs.py:
trigger, both muons matched to the dimuon filter, medium ID, displaced J/psi),
mass term replaced by the sPlot window, plus gen-matched muons in MC.

INPUT FILES
  Every *.root file under --data-dir and under --mc-dir is read, searched
  recursively (a CRAB output tree <tag>/<timestamp>/0000/*.root works as is).
  The two may contain {epoch} (and --mc-dir {sample}), filled per epoch:
      --data-dir /pnfs/.../data/{epoch}  --mc-dir /pnfs/.../mc/{sample}_{epoch}
  Without {epoch}, a directory holds ONE epoch: give a single --epochs value.
  MC: one directory per (sample, epoch) -- Summer24 is three productions (2024,
  2025, 2026 flows).

Usage (covflow conda env: numpy, uproot, matplotlib, iminuit; mplhep optional)
  python3 covflow_closure_plots.py --data-dir <DATA_DIR>/{epoch} --mc-dir <MC_DIR>/{sample}_{epoch}
  python3 covflow_closure_plots.py --epochs 2026 --data-dir <DATA_DIR> --mc-dir <MC_DIR> --out cl_2026
  python3 covflow_closure_plots.py ... --sample bc          # the validation sample
  python3 covflow_closure_plots.py ... --max-entries 2000000 --prescale-data 0.2

Outputs
  <out>_<epoch>.pdf     per epoch: the sPlot fit, then one page per quantity
  <out>_all.pdf         all requested epochs summed
  <out>.json            yields, fit summaries, and per quantity and epoch the
                        chi2/ndf, means and RMS of the three distributions
'''

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

JPSI = 3.0969

# ---------------------------------------------------------------------------
# epochs: the pileup-weight year of each (MC is Summer24 for 2024-2026)
# ---------------------------------------------------------------------------
EPOCHS = {
    '2022_preEE'   : dict(pu_year='2022'),
    '2022_postEE'  : dict(pu_year='2022'),
    '2023_preBPix' : dict(pu_year='2023'),
    '2023_postBPix': dict(pu_year='2023'),
    '2024'         : dict(pu_year='2024'),
    '2025'         : dict(pu_year='2025'),
    '2026'         : dict(pu_year='2026'),
}

# covflow's Run 3 training selection (configs/run3_epochs.py), mass term
# included: it is dropped and replaced by the sPlot window below
SELECTION = ('(HLT_DoubleMu4_3_LowMass > 0.5) & '
             '(mu1_HLT_DoubleMu4_3_LowMass_tag > 0.5) & '
             '(mu2_HLT_DoubleMu4_3_LowMass_tag > 0.5) & '
             '(mu1_pt > 4.5) & (abs(mu1_eta) < 2.4) & (mu1_id_medium>0.5) & '
             '(mu2_id_medium>0.5) & (abs(mass-3.0969)<0.1) & '
             '((lxy * cos2d /pt * 3.0969)>0.008)')
MC_SELECTION = '(abs(mu1_gen_pdgid)==13) & (abs(mu2_gen_pdgid)==13)'

# ---------------------------------------------------------------------------
# what is plotted
#   kind 'cand' : one entry per candidate; 'mu' : mu1 and mu2 merged ({mu})
#   expr        nominal expression (data, MC corrected)
#   corr        MC-corrected override (the covariance the fit used)
#   raw         MC-raw override; default: the nominal one read through *_raw
#   bins        (lo, hi, n); values outside go to the edge bins (shown)
# ---------------------------------------------------------------------------
P = 'qoverp lambda phi dxy dsz'.split()
VARS = [
    # candidate
    dict(name='pt',          kind='cand', expr='pt', bins=(0, 40, 80), label=r'$p_T(\mu\mu)$ [GeV]'),
    dict(name='vtx_prob',    kind='cand', expr='vtx_prob', bins=(0, 1, 50), logy=True,
         label='vertex probability'),
    dict(name='vtx_chi2',    kind='cand', expr='vtx_chi2', bins=(0, 15, 60), logy=True,
         label=r'vertex $\chi^2$'),
    dict(name='lxy',         kind='cand', expr='lxy', bins=(0, 0.6, 60), logy=True,
         label=r'$L_{xy}$ [cm]'),
    dict(name='lxy_err',     kind='cand', expr='lxy_err', bins=(0, 0.012, 60),
         label=r'$\sigma(L_{xy})$ [cm]'),
    dict(name='lxy_sig',     kind='cand', expr='lxy_sig', bins=(0, 150, 75), logy=True,
         label=r'$L_{xy}/\sigma$'),
    dict(name='lxyz_sig',    kind='cand', expr='lxyz_sig', bins=(0, 150, 75), logy=True,
         label=r'$L_{xyz}/\sigma$'),
    dict(name='lxy_bs_sig',  kind='cand', expr='lxy_bs_sig', bins=(0, 150, 75), logy=True,
         label=r'$L_{xy}/\sigma$ (beam spot)'),
    dict(name='ct',          kind='cand', expr='lxy * cos2d / pt * 3.0969', bins=(0, 0.3, 60),
         logy=True, label=r'$ct = L_{xy} \cos\alpha_{2D}\, m/p_T$ [cm]'),
    dict(name='log_1mcos2d', kind='cand', expr='log10(clip(1 - cos2d, 1e-9, 1))',
         bins=(-9, 0, 54), logy=True, label=r'$\log_{10}(1-\cos\alpha_{2D})$'),
    dict(name='log_1mcos3d', kind='cand', expr='log10(clip(1 - cos3d, 1e-9, 1))',
         bins=(-9, 0, 54), logy=True, label=r'$\log_{10}(1-\cos\alpha_{3D})$'),
    # muons, mu1 and mu2 merged
    dict(name='mu_pt',       kind='mu', expr='{mu}_pt', bins=(0, 30, 60), label=r'$\mu$ $p_T$ [GeV]'),
    dict(name='mu_eta',      kind='mu', expr='{mu}_eta', bins=(-2.5, 2.5, 50), label=r'$\mu$ $\eta$'),
    dict(name='mu_dxy',      kind='mu', expr='{mu}_dxy', bins=(-0.3, 0.3, 60), logy=True,
         label=r'$\mu$ $d_{xy}$ (PV) [cm]'),
    dict(name='mu_dxy_e',    kind='mu', expr='{mu}_dxy_e', bins=(0, 0.01, 50),
         label=r'$\mu$ $\sigma(d_{xy})$ [cm]'),
    dict(name='mu_dxy_sig',  kind='mu', expr='{mu}_dxy_sig', bins=(-60, 60, 60), logy=True,
         label=r'$\mu$ $d_{xy}/\sigma$'),
    dict(name='mu_dz',       kind='mu', expr='{mu}_dz', bins=(-0.5, 0.5, 50), logy=True,
         label=r'$\mu$ $d_z$ (PV) [cm]'),
    dict(name='mu_dz_e',     kind='mu', expr='{mu}_dz_e', bins=(0, 0.02, 50),
         label=r'$\mu$ $\sigma(d_z)$ [cm]'),
    dict(name='mu_dz_sig',   kind='mu', expr='{mu}_dz_sig', bins=(-60, 60, 60), logy=True,
         label=r'$\mu$ $d_z/\sigma$'),
    # the track errors themselves: data and MC raw from cov_*, MC corrected
    # from cov_corr_* (what the fit used)
    dict(name='mu_relerr_p', kind='mu',
         expr='sqrt({mu}_cov_qoverp_qoverp) * {mu}_pt * cosh({mu}_eta)',
         corr='sqrt({mu}_cov_corr_qoverp_qoverp) * {mu}_pt * cosh({mu}_eta)',
         bins=(0, 0.04, 50), label=r'$\mu$ $\sigma(p)/p$'),
] + [
    dict(name='mu_sigma_%s' % p, kind='mu', expr='sqrt({mu}_cov_%s_%s)' % (p, p),
         corr='sqrt({mu}_cov_corr_%s_%s)' % (p, p),
         bins=dict(qoverp=(0, 0.01, 50), **{'lambda': (0, 0.003, 50)}, phi=(0, 0.003, 50),
                   dxy=(0, 0.01, 50), dsz=(0, 0.02, 50))[p],
         label=r'$\mu$ $\sigma(%s)$' % {'qoverp': 'q/p', 'lambda': r'\lambda', 'phi': r'\phi',
                                        'dxy': 'd_{xy}', 'dsz': 'd_{sz}'}[p])
    for p in P
]

# raw counterparts that are computed, not stored
DERIVED_RAW = {
    'mu1_dxy_sig_raw': 'mu1_dxy_raw / mu1_dxy_e_raw',
    'mu2_dxy_sig_raw': 'mu2_dxy_raw / mu2_dxy_e_raw',
    'mu1_dz_sig_raw' : 'mu1_dz_raw / mu1_dz_e_raw',
    'mu2_dz_sig_raw' : 'mu2_dz_raw / mu2_dz_e_raw',
}
# changed by the emulation but no raw value stored: refused in the raw view
NO_RAW = {'eta', 'phi', 'mu1_bs_dxy', 'mu2_bs_dxy', 'mu1_bs_dxy_sig', 'mu2_bs_dxy_sig',
          'mu1_bs_dxy_e', 'mu2_bs_dxy_e'}

FUNCS = dict(abs=np.abs, sqrt=np.sqrt, log10=np.log10, log=np.log, exp=np.exp,
             cosh=np.cosh, sinh=np.sinh, cos=np.cos, sin=np.sin, clip=np.clip,
             minimum=np.minimum, maximum=np.maximum, where=np.where, np=np)
IDENT = re.compile(r'\b[A-Za-z_][A-Za-z0-9_]*\b')


def names_in(expr):
    return {n for n in IDENT.findall(expr) if n not in FUNCS}


def die(msg):
    sys.exit('[closure] ERROR: ' + msg)


# ---------------------------------------------------------------------------
# progress bar (stdout, throttled)
# ---------------------------------------------------------------------------
class Progress:
    def __init__(self, total, label):
        self.total, self.label, self.t0, self.last = max(int(total), 1), label, time.time(), 0.
        self.closed = False

    def update(self, done, extra=''):
        now = time.time()
        if self.closed or (done < self.total and now - self.last < 0.5):
            return
        self.last = now
        f = min(done / self.total, 1.0)
        bar = '#' * int(30 * f) + '.' * (30 - int(30 * f))
        rate = done / max(now - self.t0, 1e-9)
        eta = (self.total - done) / max(rate, 1e-9)
        sys.stdout.write('\r  [%s] %5.1f%%  %-28s %s/%s  %.0f/s  ETA %3.0fs %s'
                         % (bar, 100 * f, self.label[:28], f'{done:,}', f'{self.total:,}',
                            rate, eta, extra))
        sys.stdout.flush()
        if done >= self.total:
            sys.stdout.write('\n')
            self.closed = True


# ---------------------------------------------------------------------------
# selections
# ---------------------------------------------------------------------------
def split_and(expr):
    '''Top-level & terms of an expression.'''
    out, depth, cur = [], 0, ''
    for ch in expr:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == '&' and depth == 0:
            out.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def drop_mass_terms(expr):
    keep, dropped = [], []
    for t in split_and(expr):
        (dropped if {'mass', 'mass_raw'} & names_in(t) else keep).append(t)
    return ' & '.join('(%s)' % t for t in keep), dropped


# ---------------------------------------------------------------------------
# column views
# ---------------------------------------------------------------------------
class View(object):
    '''Resolves branch names to arrays for one reading of a sample:
       'nominal' : as stored (data, MC corrected)
       'raw'     : X_raw where stored or derivable, the nominal branch for what
                   the emulation leaves alone; mu1 / mu2 swapped back to the
                   raw pT order'''

    def __init__(self, cols, mode):
        self.c, self.mode = cols, mode
        self.fallbacks = set()
        self.cache = {}
        if mode == 'raw':
            for k in ('mu1_pt_raw', 'mu2_pt_raw'):
                if k not in cols:
                    die('the MC has no %s: not an emulated ntuple (inspector --emulate)?' % k)
            self.swap = cols['mu2_pt_raw'] > cols['mu1_pt_raw']

    def _raw(self, name):
        c = self.c
        if name + '_raw' in c:
            return c[name + '_raw']
        if name + '_raw' in DERIVED_RAW:
            return self.eval(DERIVED_RAW[name + '_raw'], raw_names=True)
        if name in NO_RAW:
            die('%s is changed by the emulation and its raw value is not stored: it '
                'cannot be shown for the raw MC' % name)
        if name not in c:
            die('branch %s (nor %s_raw) not in the MC ntuple' % (name, name))
        self.fallbacks.add(name)
        return c[name]

    def get(self, name):
        if name in self.cache:
            return self.cache[name]
        if self.mode == 'nominal':
            if name not in self.c:
                die('branch %s not in the ntuple' % name)
            v = self.c[name]
        else:
            if name.startswith('mu1_') or name.startswith('mu2_'):
                me, other = name, ('mu2_' if name.startswith('mu1_') else 'mu1_') + name[4:]
                a, b = self._raw(me), self._raw(other)
                v = np.where(self.swap, b, a)
            else:
                v = self._raw(name)
        self.cache[name] = v
        return v

    def eval(self, expr, raw_names=False):
        ns = dict(FUNCS)
        for n in names_in(expr):
            ns[n] = self.c[n] if raw_names else self.get(n)
        with np.errstate(all='ignore'):
            out = eval(expr, {'__builtins__': {}}, ns)
        return np.broadcast_to(np.asarray(out, np.float64), (self.n,)).copy() \
            if np.ndim(out) == 0 else np.asarray(out, np.float64)

    @property
    def n(self):
        return len(next(iter(self.c.values())))


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def root_files(directory, what):
    '''Every *.root file under `directory`, recursively, sorted.'''
    if '<' in directory and '>' in directory:
        die('%s directory %s is still a placeholder' % (what, directory))
    if not os.path.isdir(directory):
        die('%s directory %s not found' % (what, directory))
    files = sorted(glob.glob(os.path.join(directory, '**', '*.root'), recursive=True))
    if not files:
        die('no .root file under %s' % directory)
    return files


def read(files, branches, label, pre_sel, max_entries=None, prescale=1.0, seed=1):
    '''Only the branches needed, only the rows passing pre_sel (a numpy
    expression on the stored branches, None = all).'''
    import uproot
    with uproot.open(files[0]) as fh:
        have = set(fh['tree'].keys())
    need = sorted(b for b in branches if b in have)
    total = 0
    for f in files:
        with uproot.open(f) as fh:
            total += fh['tree'].num_entries
    if max_entries:
        total = min(total, max_entries)
    rng = np.random.default_rng(seed)
    prog = Progress(total, 'reading ' + label)
    cols, done, kept = {}, 0, 0
    for chunk in uproot.iterate([{f: 'tree'} for f in files], expressions=need,
                                library='np', step_size='200 MB'):
        n = len(next(iter(chunk.values())))
        if max_entries and done + n > max_entries:
            n = max_entries - done
        # float32 in memory, as stored: the data samples are large
        ch = {k: np.asarray(v[:n], np.float32) for k, v in chunk.items()}
        m = np.ones(n, bool)
        if pre_sel:
            ns = dict(FUNCS)
            ns.update({k: ch[k] for k in names_in(pre_sel)})
            with np.errstate(all='ignore'):
                m &= np.asarray(eval(pre_sel, {'__builtins__': {}}, ns), bool)
        if prescale < 1:
            m &= rng.random(n) < prescale
        for k, v in ch.items():
            cols.setdefault(k, []).append(v[m])
        done += n
        kept += int(m.sum())
        prog.update(done, 'kept %s' % f'{kept:,}')
        if max_entries and done >= max_entries:
            break
    prog.update(total)
    return {k: np.concatenate(v) for k, v in cols.items()}, have


# ---------------------------------------------------------------------------
# histograms and comparison
# ---------------------------------------------------------------------------
def values_and_weights(view, var, which, w_evt, sel):
    '''x and w for one variable in one view; muon variables concatenate both
    muons with the event weight repeated.'''
    expr = var.get(which, var['expr']) if which in ('corr', 'raw') else var['expr']
    if var['kind'] == 'mu':
        xs = [view.eval(expr.format(mu=mu))[sel] for mu in ('mu1', 'mu2')]
        return np.concatenate(xs), np.concatenate([w_evt[sel], w_evt[sel]])
    return view.eval(expr)[sel], w_evt[sel]


def hist(x, w, bins):
    lo, hi, n = bins
    edges = np.linspace(lo, hi, n + 1)
    ok = np.isfinite(x) & np.isfinite(w)
    xc = np.clip(x[ok], lo, hi - 1e-12 * (hi - lo))     # overflow into the edge bins
    h, _ = np.histogram(xc, bins=edges, weights=w[ok])
    e2, _ = np.histogram(xc, bins=edges, weights=w[ok] ** 2)
    return h, e2, edges


def wstats(x, w):
    ok = np.isfinite(x) & np.isfinite(w)
    x, w = x[ok], w[ok]
    s = w.sum()
    if s == 0:
        return np.nan, np.nan
    m = np.sum(w * x) / s
    return float(m), float(np.sqrt(max(np.sum(w * (x - m) ** 2) / s, 0.)))


def chi2(hd, ed2, hm, em2):
    v = ed2 + em2
    ok = v > 0
    c = float(np.sum((hd[ok] - hm[ok]) ** 2 / v[ok]))
    return c, int(ok.sum())


# ---------------------------------------------------------------------------
# sPlot
# ---------------------------------------------------------------------------
def sweights(mass, a, out_pdf):
    from covflow import splot as SP
    lo, hi = JPSI - a.sweight_window, JPSI + a.sweight_window
    lo, hi = min(lo, float(mass.min())), max(hi, float(mass.max()))
    model = SP.MassModel(lo, hi, signal=a.sweight_signal, tails=a.sweight_tails,
                         background=a.sweight_background)
    edges = np.linspace(lo, hi, int(round((hi - lo) / 0.001)) + 1)
    cnt, _ = np.histogram(mass, bins=edges)
    res = SP.fit(model, counts=cnt, edges=edges, label='sPlot')
    gof = SP.goodness_of_fit(res, mass=mass)
    sp = SP.SPlot(res, mass)
    SP.plot_fit(res, out_pdf, mass=mass, gof=gof, rlabel='')
    dev = sp.info['max_abs_sw_s_plus_sw_b_minus_1']
    if dev > 1e-2:
        die('sPlot: signal + background weights deviate from 1 by %.2e, the fit is not '
            'at a maximum (tune it with covflow fit_splot.py on the same selection)' % dev)
    sw, _ = sp.weights(mass)
    p = res.values
    # the fitted data signal shape on the common mass binning, as expected
    # signal counts per bin (fine grid, integrated per bin)
    me = mass_edges(a)
    xf = np.linspace(me[0], me[-1], 20 * (len(me) - 1) + 1)
    f = model.signal_pdf(np.clip(xf, model.lo, model.hi), p)
    f[(xf < model.lo) | (xf > model.hi)] = 0.
    cdf = np.concatenate([[0.], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(xf))])
    curve = p['Ns'] * np.diff(np.interp(me, xf, cdf))
    info = dict(Ns=float(p['Ns']), Ns_err=float(res.errors['Ns']), Nb=float(p['Nb']),
                purity=float(p['Ns'] / (p['Ns'] + p['Nb'])), chi2=gof['chi2'], ndf=gof['ndf'],
                valid=bool(res.valid), negative_fraction=sp.info['negative_signal_weight_fraction'],
                effective_n=sp.info['effective_n_signal'], mu=float(p['mu']),
                curve=curve)
    print('  sPlot: Ns %.0f +- %.0f, purity %.1f%%, chi2/ndf %.0f/%d, %.1f%% negative weights'
          % (info['Ns'], info['Ns_err'], 100 * info['purity'], info['chi2'], info['ndf'],
             100 * info['negative_fraction']))
    return sw, info


def mass_edges(a):
    return np.linspace(JPSI - a.sweight_window, JPSI + a.sweight_window,
                       int(round(2 * a.sweight_window / 0.002)) + 1)


def sigma68(h, edges):
    '''Half the 15.87-84.13% interval of a binned distribution (linear inside
    the bins).'''
    c = np.concatenate([[0.], np.cumsum(np.clip(h, 0, None))])
    if c[-1] <= 0:
        return np.nan
    c = c / c[-1]
    q = np.interp([0.158655, 0.841345], c, edges)
    return float(0.5 * (q[1] - q[0]))


def kin_weights(data_v, sw, sel_d, mc_v, w_mc, sel_m, nbins=(12, 6)):
    '''Per-event MC weight in (pT, |eta|) of the dimuon, sWeighted data over MC
    corrected, normalised to unit mean on the selected MC.'''
    pe = np.r_[0, np.quantile(data_v.get('pt')[sel_d], np.linspace(0, 1, nbins[0] + 1))[1:-1], 1e4]
    ee = np.linspace(0, 2.4, nbins[1] + 1)
    ee[-1] = 10.
    hd, _, _ = np.histogram2d(data_v.get('pt')[sel_d], np.abs(data_v.get('eta')[sel_d]),
                              bins=(pe, ee), weights=sw[sel_d])
    ptm, etm = mc_v.get('pt'), np.abs(mc_v.get('eta'))
    hm, _, _ = np.histogram2d(ptm[sel_m], etm[sel_m], bins=(pe, ee), weights=w_mc[sel_m])
    with np.errstate(all='ignore'):
        r = np.where((hm > 0) & (hd > 0), (hd / hd.sum()) / (hm / hm.sum()), 1.0)
    i = np.clip(np.digitize(ptm, pe) - 1, 0, len(pe) - 2)
    j = np.clip(np.digitize(etm, ee) - 1, 0, len(ee) - 2)
    w = r[i, j]
    w /= np.average(w[sel_m], weights=w_mc[sel_m])
    return w


# ---------------------------------------------------------------------------
# one epoch
# ---------------------------------------------------------------------------
def run_epoch(ep, data_files, mc_files, pu_year, a):
    print('\n=== %s: %d data file(s), %d MC file(s)' % (ep, len(data_files), len(mc_files)))
    sel_nomass, dropped = drop_mass_terms(a.selection)
    win = '(abs(mass - %.4f) <= %.4f)' % (JPSI, a.sweight_window)
    sel_data = '(%s) & %s' % (sel_nomass, win) if sel_nomass else win
    pu_branch = a.pu_weight.format(pu_year=pu_year)

    var_names = set()
    for v in VARS:
        for which in ('expr', 'corr', 'raw'):
            if which in v:
                for mu in ('mu1', 'mu2'):
                    var_names |= names_in(v[which].format(mu=mu))
    var_names |= {'pt', 'eta'}
    sel_names = names_in(sel_data) | names_in(a.mc_selection or '')

    # ---- data: read the window, fit, sWeights
    d_cols, _ = read(data_files, var_names | sel_names | {'mass'}, '%s data' % ep, sel_data,
                     a.max_entries, a.prescale_data)
    if not d_cols or len(d_cols['mass']) < 1000:
        die('%s: only %d data candidates selected' % (ep, len(d_cols.get('mass', []))))
    sw, sp_info = sweights(d_cols['mass'].astype(np.float64), a, '%s_%s_splot.pdf' % (a.out, ep))
    dv = View(d_cols, 'nominal')
    sel_d = np.ones(dv.n, bool)

    # ---- MC: read everything that passes EITHER view's preselection, then
    # select per view (the raw view uses the raw mass, pT, Lxy, ...)
    def raw_names_of(names):
        return {n + '_raw' for n in names} | {m for n in names
                                               for m in names_in(DERIVED_RAW.get(n + '_raw', ''))}
    need_mc = var_names | sel_names | raw_names_of(var_names | sel_names) | \
        {'mass', 'mass_raw', 'emu_weight', pu_branch, 'mu1_pt_raw', 'mu2_pt_raw'}
    m_cols, have = read(mc_files, need_mc, '%s MC' % ep,
                        '(abs(mass - %.4f) <= %.4f) | (abs(mass_raw - %.4f) <= %.4f)'
                        % (JPSI, a.sweight_window, JPSI, a.sweight_window),
                        a.max_entries)
    for b in ('emu_weight', pu_branch, 'mass_raw'):
        if b not in m_cols:
            die('%s MC: branch %s missing (emulated MC ntuple with pileup weights expected)'
                % (ep, b))
    mc_full = '(%s) & %s' % (sel_data, a.mc_selection) if a.mc_selection else sel_data
    cv, rv = View(m_cols, 'nominal'), View(m_cols, 'raw')
    sel_c = cv.eval(mc_full).astype(bool)
    sel_r = rv.eval(mc_full).astype(bool)
    pu = m_cols[pu_branch]
    emu = m_cols['emu_weight']
    if np.any(~np.isfinite(pu[sel_c | sel_r])):
        die('%s MC: %s not finite for selected events (wrong pu year?)' % (ep, pu_branch))
    w_c = pu * (emu if not a.no_emu_weight else 1.0)
    w_r = pu.copy()
    if a.kin_reweight:
        wk = kin_weights(dv, sw, sel_d, cv, w_c, sel_c)
        w_c, w_r = w_c * wk, w_r * wk
        print('  kinematic weights: mean %.3f, RMS %.3f' % (wk[sel_c].mean(), wk[sel_c].std()))
    print('  selected: data %s (sum sw %.0f), MC raw %s, MC corrected %s; mean emu_weight %.4f'
          % (f'{dv.n:,}', sw.sum(), f'{int(sel_r.sum()):,}', f'{int(sel_c.sum()):,}',
             float(np.mean(emu[sel_c]))))
    # normalise each MC view to the data signal yield
    k_c = sw.sum() / w_c[sel_c].sum()
    k_r = sw.sum() / w_r[sel_r].sum()
    w_c, w_r = w_c * k_c, w_r * k_r

    me = mass_edges(a)
    hmr, _ = np.histogram(rv.get('mass')[sel_r], bins=me, weights=w_r[sel_r])
    hmc, _ = np.histogram(cv.get('mass')[sel_c], bins=me, weights=w_c[sel_c])
    curve = sp_info.pop('curve')
    hists = {'__mass__': dict(edges=me, data=curve, raw=hmr, corr=hmc)}
    s68 = {k: sigma68(v, me) for k, v in (('data', curve), ('raw', hmr), ('corr', hmc))}
    sp_info['sigma68_MeV'] = {k: 1e3 * v for k, v in s68.items()}
    print('  mass resolution sigma68: data %.1f MeV (fitted signal), MC raw %.1f, '
          'MC corrected %.1f' % tuple(1e3 * s68[k] for k in ('data', 'raw', 'corr')))
    prog = Progress(len(VARS), '%s histograms' % ep)
    for i, v in enumerate(VARS):
        xd, wd = values_and_weights(dv, v, 'data', sw, sel_d)
        xc, wc = values_and_weights(cv, v, 'corr', w_c, sel_c)
        xr, wr = values_and_weights(rv, v, 'raw', w_r, sel_r)
        hd, ed, edges = hist(xd, wd, v['bins'])
        hc, ec, _ = hist(xc, wc, v['bins'])
        hr, er, _ = hist(xr, wr, v['bins'])
        hists[v['name']] = dict(edges=edges, data=(hd, ed), corr=(hc, ec), raw=(hr, er),
                                stats=dict(data=wstats(xd, wd), corr=wstats(xc, wc),
                                           raw=wstats(xr, wr)))
        prog.update(i + 1)
    if rv.fallbacks:
        print('  raw MC view: unchanged by the emulation, nominal branch used for: %s'
              % ', '.join(sorted(rv.fallbacks)))
    info = dict(files=dict(data=data_files, mc=mc_files), pu_weight=pu_branch,
                selection=dict(data=sel_data, mc_extra=a.mc_selection, dropped=dropped),
                splot=sp_info, n=dict(data=int(dv.n), mc_raw=int(sel_r.sum()),
                                      mc_corr=int(sel_c.sum())),
                emu_weight_mean=float(np.mean(emu[sel_c])), raw_fallbacks=sorted(rv.fallbacks))
    return hists, info


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
COLORS = dict(data='black', raw='#2f6eba', corr='#d1495b')
LABELS = dict(raw='MC raw', corr='MC corrected')


def summarise(h):
    out = {}
    for which in ('raw', 'corr'):
        c, n = chi2(h['data'][0], h['data'][1], h[which][0], h[which][1])
        out[which] = dict(chi2=c, ndf=n)
    out['stats'] = h['stats']
    return out


def plot_pages(hists, title, pdf_path, extra_first=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    table = {}
    with PdfPages(pdf_path) as pdf:
        hm = hists['__mass__']
        e = hm['edges']
        s68 = {k: sigma68(hm[k], e) for k in ('data', 'raw', 'corr')}
        table['__mass__'] = {'sigma68_MeV': {k: 1e3 * v for k, v in s68.items()}}
        fig, ax = plt.subplots(figsize=(7, 5.5))
        norm = lambda h: h / max(h.sum(), 1e-300) / np.diff(e)
        ax.stairs(norm(hm['data']), e, color=COLORS['data'], lw=2,
                  label=r'data, fitted signal shape  $\sigma_{68}$ = %.1f MeV' % (1e3 * s68['data']))
        for which, ls in (('raw', '--'), ('corr', '-')):
            ax.stairs(norm(hm[which]), e, color=COLORS[which], ls=ls, lw=1.6,
                      label=r'%s  $\sigma_{68}$ = %.1f MeV' % (LABELS[which], 1e3 * s68[which]))
        ax.set_yscale('log')
        ax.set_xlabel(r'$m(\mu\mu)$ [GeV]')
        ax.set_ylabel('normalised')
        ax.set_title('%s: dimuon mass resolution (the sWeighted data mass would be the fit '
                     'itself)' % title, fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        for v in VARS:
            h = hists[v['name']]
            s = summarise(h)
            table[v['name']] = s
            e = h['edges']
            c = 0.5 * (e[1:] + e[:-1])
            fig, (ax, rx) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True,
                                         gridspec_kw=dict(height_ratios=(3, 1), hspace=0.05))
            hd, ed = h['data']
            ax.errorbar(c, hd, yerr=np.sqrt(ed), fmt='o', ms=3, color=COLORS['data'],
                        label='data (sWeighted)', zorder=3)
            for which, ls in (('raw', '--'), ('corr', '-')):
                hm, em = h[which]
                ax.stairs(hm, e, color=COLORS[which], ls=ls, lw=1.6,
                          label='%s  $\\chi^2$/ndf %.0f/%d' % (LABELS[which], s[which]['chi2'],
                                                               s[which]['ndf']))
                with np.errstate(all='ignore'):
                    r = np.where(hd != 0, hm / hd, np.nan)
                    rerr = np.where(hd != 0, np.sqrt(em) / np.abs(hd), np.nan)
                rx.errorbar(c, r, yerr=rerr, fmt='o' if which == 'corr' else 's', ms=2.5,
                            color=COLORS[which], alpha=0.9)
            with np.errstate(all='ignore'):
                dr = np.where(hd != 0, np.sqrt(ed) / np.abs(hd), np.nan)
            rx.fill_between(c, 1 - dr, 1 + dr, step='mid', color='grey', alpha=0.25, lw=0)
            rx.axhline(1, color='black', lw=0.8)
            rx.set_ylim(0.5, 1.5)
            rx.set_ylabel('MC / data')
            rx.set_xlabel(v['label'] + ('   (both muons)' if v['kind'] == 'mu' else ''))
            if v.get('logy'):
                ax.set_yscale('log')
            ax.set_ylabel('candidates' if v['kind'] == 'cand' else 'muons')
            st = s['stats']
            ax.set_title('%s   mean: data %.4g, raw %.4g, corr %.4g' % (
                title, st['data'][0], st['raw'][0], st['corr'][0]), fontsize=9)
            ax.legend(fontsize=8, frameon=False)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        # summary page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        rows = [['quantity', 'chi2/ndf raw', 'chi2/ndf corr', 'mean data', 'mean raw',
                 'mean corr', 'RMS data', 'RMS raw', 'RMS corr']]
        for v in VARS:
            s = table[v['name']]
            st = s['stats']
            rows.append([v['name'],
                         '%.1f' % (s['raw']['chi2'] / max(s['raw']['ndf'], 1)),
                         '%.1f' % (s['corr']['chi2'] / max(s['corr']['ndf'], 1)),
                         '%.4g' % st['data'][0], '%.4g' % st['raw'][0], '%.4g' % st['corr'][0],
                         '%.4g' % st['data'][1], '%.4g' % st['raw'][1], '%.4g' % st['corr'][1]])
        t = ax.table(cellText=rows, loc='upper center', cellLoc='center')
        t.auto_set_font_size(False)
        t.set_fontsize(7)
        t.scale(1, 1.25)
        s68 = table['__mass__']['sigma68_MeV']
        ax.set_title('%s: summary (chi2 over the shown bins, MC normalised to the data '
                     'signal yield)\nmass sigma68: data %.1f MeV, MC raw %.1f, MC corrected %.1f'
                     % (title, s68['data'], s68['raw'], s68['corr']), fontsize=9)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    return table


def print_table(title, table):
    print('\n  %s' % title)
    print('  mass sigma68 [MeV]: data %.1f, MC raw %.1f, MC corrected %.1f'
          % tuple(table['__mass__']['sigma68_MeV'][k] for k in ('data', 'raw', 'corr')))
    print('  %-16s %14s %14s %11s %11s %11s' % ('quantity', 'chi2/ndf raw', 'chi2/ndf corr',
                                                 'mean data', 'mean raw', 'mean corr'))
    for v in VARS:
        s = table[v['name']]
        st = s['stats']
        print('  %-16s %14s %14s %11.4g %11.4g %11.4g' % (
            v['name'], '%.0f/%d' % (s['raw']['chi2'], s['raw']['ndf']),
            '%.0f/%d' % (s['corr']['chi2'], s['corr']['ndf']),
            st['data'][0], st['raw'][0], st['corr'][0]))


def combine(per_epoch):
    '''Sum the histograms over epochs; means / RMS combined from the per-epoch
    moments, each epoch weighted by its (data-normalised) yield.'''
    out = {'__mass__': dict(edges=per_epoch[0]['__mass__']['edges'],
                            **{w: sum(h['__mass__'][w] for h in per_epoch)
                               for w in ('data', 'raw', 'corr')})}
    for v in VARS:
        k = v['name']
        hs = [h[k] for h in per_epoch]
        o = dict(edges=hs[0]['edges'])
        for w in ('data', 'raw', 'corr'):
            o[w] = (sum(h[w][0] for h in hs), sum(h[w][1] for h in hs))
        st = {}
        for w in ('data', 'raw', 'corr'):
            W = np.array([h['data'][0].sum() for h in hs])          # data yield per epoch
            m = np.array([h['stats'][w][0] for h in hs])
            s = np.array([h['stats'][w][1] for h in hs])
            ok = np.isfinite(m) & np.isfinite(s) & (W > 0)
            if not ok.any():
                st[w] = (np.nan, np.nan)
                continue
            M = np.sum(W[ok] * m[ok]) / W[ok].sum()
            V = np.sum(W[ok] * (s[ok] ** 2 + (m[ok] - M) ** 2)) / W[ok].sum()
            st[w] = (float(M), float(np.sqrt(V)))
        o['stats'] = st
        out[k] = o
    return out


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--epochs', nargs='+', default=list(EPOCHS), choices=list(EPOCHS))
    p.add_argument('--sample', default='hb', help='MC sample in the file names: hb (default) or bc')
    p.add_argument('--data-dir', default='<DATA_DIR>/{epoch}',
                   help='all *.root under it, recursively; {epoch} filled per epoch '
                        '(default, a PLACEHOLDER: %(default)s)')
    p.add_argument('--mc-dir', default='<MC_DIR>/{sample}_{epoch}',
                   help='all *.root under it, recursively; {epoch} and {sample} filled '
                        '(default, a PLACEHOLDER: %(default)s)')
    p.add_argument('--selection', default=SELECTION,
                   help='common selection; its mass terms are replaced by the sPlot window')
    p.add_argument('--mc-selection', default=MC_SELECTION, help='extra MC selection (gen matching)')
    p.add_argument('--pu-weight', default='pu_weight_{pu_year}',
                   help='MC pileup weight branch, {pu_year} = the epoch\'s (default %(default)s)')
    p.add_argument('--no-emu-weight', action='store_true',
                   help='do not apply emu_weight to the corrected MC (check only)')
    p.add_argument('--kin-reweight', action='store_true',
                   help='also reweight both MC views in the J/psi (pT, |eta|) to sWeighted data')
    p.add_argument('--sweight-window', type=float, default=0.25)
    p.add_argument('--sweight-signal', default='2dscb', choices=['2dscb', 'dscb'])
    p.add_argument('--sweight-tails', default='shared', choices=['shared', 'independent'])
    p.add_argument('--sweight-background', default='exp', choices=['exp', '2exp'])
    p.add_argument('--covflow-repo', default='/work/manzoni/correct_track_covariance/covflow',
                   help='covflow checkout, for covflow.splot (default %(default)s)')
    p.add_argument('--max-entries', type=int, default=None, help='per sample and epoch')
    p.add_argument('--prescale-data', type=float, default=1.0,
                   help='keep this random fraction of the data (fast iterations)')
    p.add_argument('--out', default='covflow_closure')
    a = p.parse_args()
    for opt, d in (('--data-dir', a.data_dir), ('--mc-dir', a.mc_dir)):
        if '{epoch}' not in d and len(a.epochs) != 1:
            p.error('%s has no {epoch}: it holds one epoch, give a single --epochs value' % opt)
    return a


def main():
    a = parse_args()
    repo = os.path.abspath(a.covflow_repo)
    if not os.path.isfile(os.path.join(repo, 'splot.py')):
        die('%s is not a covflow checkout (no splot.py): --covflow-repo' % repo)
    sys.path.insert(0, os.path.dirname(repo))
    if os.path.basename(repo) != 'covflow':
        die('the covflow checkout must be a directory named covflow (imported as a package)')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)

    report = dict(args=vars(a), epochs={})
    per_epoch = []
    for ep in a.epochs:
        cfg = EPOCHS[ep]
        data_files = root_files(a.data_dir.replace('{epoch}', ep), 'data')
        mc_files = root_files(a.mc_dir.replace('{epoch}', ep).replace('{sample}', a.sample), 'MC')
        hists, info = run_epoch(ep, data_files, mc_files, cfg['pu_year'], a)
        table = plot_pages(hists, '%s %s' % (ep, a.sample), '%s_%s.pdf' % (a.out, ep))
        print_table('%s (%s)' % (ep, a.sample), table)
        info['table'] = table
        report['epochs'][ep] = info
        per_epoch.append(hists)
    if len(per_epoch) > 1:
        allh = combine(per_epoch)
        table = plot_pages(allh, 'all epochs %s' % a.sample, '%s_all.pdf' % a.out)
        print_table('all epochs (%s)' % a.sample, table)
        report['all'] = table
    with open(a.out + '.json', 'w') as fh:
        json.dump(report, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print('\nwrote %s_<epoch>.pdf%s, %s_<epoch>_splot.pdf, %s.json'
          % (a.out, ', %s_all.pdf' % a.out if len(per_epoch) > 1 else '', a.out, a.out))


if __name__ == '__main__':
    main()
