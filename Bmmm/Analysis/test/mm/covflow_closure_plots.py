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
(half the 15.87-84.13% interval) for each. That page is indicative only (data
model vs MC histogram, window-truncated, background in the no-sPlot view): the
proper comparison is in _resolution.pdf, see below.

RESOLUTION AND sPlot-FREE COMPARISONS (<out>_<epoch>_resolution.pdf)
  sPlot assumes that the plotted quantity is independent of the mass for the
  signal. That is FALSE for anything that sets the mass resolution: a muon
  with a large sigma(q/p) sits in a wider mass peak, its candidate is more
  background-like in the fit and its sWeight is pulled down. The sWeighted
  tails of sigma(q/p), sigma(phi), sigma(lambda), sigma(p)/p, sigma(m) are
  biased low; the vertex chi2 mildly so. For these, the data signal yield is
  instead taken from a separate mass fit in each bin of the quantity (same
  model, shape parameters fixed to the global fit, width and background
  free), which needs no independence assumption.
    1. summary: the mass resolution with ONE method for data, MC raw and MC
       corrected -- the same signal model fitted to each, sigma68 of the fitted
       signal (no window truncation, no background in the data number); the
       vertex-chi2 scale factor k (k^2 = signal median chi2 / 0.455, the
       median of a chi2 with 1 dof); the MC-truth widths below.
    2. calibration of the predicted mass uncertainty: in bins of sigma(m)
       (propagated per event from the (q/p, lambda, phi) covariance block of
       the two muons: cov_* for data and MC raw, cov_corr_* for MC corrected),
       the fitted sigma68 of the mass peak against the mean predicted sigma(m).
       Honest errors lie on the diagonal; the ratio panel is the in-data
       equivalent of a mass pull width.
    3. per-bin-fit yields of the resolution-correlated quantities, with the
       sWeighted data overlaid (the size of the sPlot bias); after each, the
       mass fits behind it, one panel per bin (linear and log pages): data
       with its fit and fitted background, and both MC views (no background)
       scaled to the data signal yield and drawn on top of the fitted data
       background, histogram and own signal fit. The global fits (whole
       sample) are shown the same way after the summary.
    4. MC truth: m - m_gen (m_gen from the two gen-matched muons, final-state
       radiation therefore removed) and the mass pull (m - m_gen)/sigma(m),
       raw vs corrected, for all events, events with no hit killed, events
       with at least one muon hit killed. Needs mu<i>_gen_pt/eta/phi.

Every quantity is shown twice, in two pdfs, each page linear (left) and log
(right):
  _sw    with sPlot: data sWeighted over the whole fit window, background
         subtracted; MC normalised to the data signal yield
  _nosw  without sPlot: the selection's own mass cut (|m - 3.0969| < 0.1 by
         default), every data candidate with weight 1, background INCLUDED --
         the conditions covflow was trained in; MC normalised to the data count.
         The signal fraction in that cut (from the sWeights) is printed.

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
  The EPOCHS table below is the usual association map (covflow's
  configs/run3_epochs.py): data eras per epoch, and one MC file per (sample,
  epoch), <sample>_<campaign>_<epoch>.root -- Summer24 is three productions
  (2024, 2025, 2026 flows). <DATA_DIR> and <MC_DIR> are --data-dir / --mc-dir,
  both by default the merged ntuples of 12 Oct 2026 (NTUPLE_DIR). Or give
  --data / --mc explicitly for a single --epochs value.

Usage (covflow conda env: numpy, uproot, matplotlib, iminuit; mplhep optional)
  python3 covflow_closure_plots.py --epochs 2026 --out closure/hb_2026
  python3 covflow_closure_plots.py --epochs 2026 --data 'd/*.root' --mc 'm/*.root' --out cl_2026
  python3 covflow_closure_plots.py ... --sample bc          # the validation sample
  python3 covflow_closure_plots.py ... --max-entries 2000000 --prescale-data 0.2

Outputs
  <out>_<epoch>_sw.pdf, <out>_<epoch>_nosw.pdf
                        per epoch: mass page, one page per quantity, summary
  <out>_<epoch>_resolution.pdf   mass resolution, sigma(m) calibration,
                        per-bin-fit yields, vertex-chi2 k, MC truth (above)
  <out>_<epoch>_splot.pdf   the data mass fit behind the sWeights
  <out>_all_sw.pdf, <out>_all_nosw.pdf, <out>_all_resolution.pdf
                        all requested epochs summed (fits redone on the sums)
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
# the merged ntuples; {sample} = hb / bc
# Data eras per epoch as in covflow's configs/run3_epochs.py.
# ---------------------------------------------------------------------------
NTUPLE_DIR = '/pnfs/psi.ch/cms/trivcat/store/user/manzoni/rjpsi_run3/covflow_dimuon_ntuples_12oct26'

EPOCHS = {
    '2022_preEE': dict(pu_year='2022', campaign='Summer22',
                       data=['<DATA_DIR>/data2022C_PromptReco_v1.root',
                             '<DATA_DIR>/data2022D_PromptReco_v1.root',
                             '<DATA_DIR>/data2022D_PromptReco_v2.root'],
                       mc=['<MC_DIR>/{sample}_Summer22_2022_preEE.root']),
    '2022_postEE': dict(pu_year='2022', campaign='Summer22EE',
                        data=['<DATA_DIR>/data2022E_PromptReco_v1.root',
                              '<DATA_DIR>/data2022F_22Sep2023_v1.root',
                              '<DATA_DIR>/data2022G_22Sep2023_v1.root'],
                        mc=['<MC_DIR>/{sample}_Summer22EE_2022_postEE.root']),
    '2023_preBPix': dict(pu_year='2023', campaign='Summer23',
                         data=['<DATA_DIR>/data2023C_22Sep2023_v1_v2.root',
                               '<DATA_DIR>/data2023C_22Sep2023_v2_v1.root',
                               '<DATA_DIR>/data2023C_22Sep2023_v3_v1.root',
                               '<DATA_DIR>/data2023C_22Sep2023_v4_v1.root'],
                         mc=['<MC_DIR>/{sample}_Summer23_2023_preBPix.root']),
    '2023_postBPix': dict(pu_year='2023', campaign='Summer23BPix',
                          data=['<DATA_DIR>/data2023D_22Sep2023_v1_v1.root',
                                '<DATA_DIR>/data2023D_22Sep2023_v2_v1.root'],
                          mc=['<MC_DIR>/{sample}_Summer23BPix_2023_postBPix.root']),
    '2024': dict(pu_year='2024', campaign='Summer24',
                 data=['<DATA_DIR>/data2024%s.root' % e for e in
                       ('C_MINIv6NANOv15_v1', 'D_MINIv6NANOv15_v1', 'E_MINIv6NANOv15_v1',
                        'F_MINIv6NANOv15_v3', 'G_MINIv6NANOv15_v1', 'H_MINIv6NANOv15_v1',
                        'I_MINIv6NANOv15_v2_v2', 'I_MINIv6NANOv15_v3')],
                 mc=['<MC_DIR>/{sample}_Summer24_2024.root']),
    '2025': dict(pu_year='2025', campaign='Summer24',
                 data=['<DATA_DIR>/data2025%s.root' % e for e in
                       ('B_PromptReco_v1', 'C_PromptReco_v1', 'C_PromptReco_v2',
                        'D_PromptReco_v1', 'E_PromptReco_v1', 'F_PromptReco_v1',
                        'F_PromptReco_v2', 'G_PromptReco_v1')],
                 mc=['<MC_DIR>/{sample}_Summer24_2025.root']),
    '2026': dict(pu_year='2026', campaign='Summer24',
                 data=['<DATA_DIR>/data2026B_PromptReco_v1.root',
                       '<DATA_DIR>/data2026D_PromptReco_v1.root'],
                 mc=['<MC_DIR>/{sample}_Summer24_2026.root']),
}

# covflow's Run 3 training selection (configs/run3_epochs.py), mass term
# included: it is dropped and replaced by the sPlot window below
SELECTION = ('(HLT_DoubleMu4_3_LowMass > 0.5) & '
             '(mu1_HLT_DoubleMu4_3_LowMass_tag > 0.5) & '
             '(mu2_HLT_DoubleMu4_3_LowMass_tag > 0.5) & '
             '(mu1_pt > 4.5) & (abs(mu1_eta) < 2.4) & (mu1_id_medium>0.5) & '
             '(mu2_pt > 3.5) & (abs(mu2_eta) < 2.4) & '
             '(mu2_id_medium>0.5) & (abs(mass-3.0969)<0.1) & '
             '((lxy * cos2d /pt * 3.0969)>0.015) & '
             '(lxy > 0.1)')
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
] + [
    # per-event predicted dimuon mass uncertainty, propagated from the (q/p,
    # lambda, phi) block of both muons' covariance (cov_corr_* for MC
    # corrected): computed, not stored (predicted_mass below)
    dict(name='sigma_m', kind='cand', expr='sigma_m', bins=(0, 0.08, 80),
         label=r'predicted $\sigma(m_{\mu\mu})$ [GeV]'),
] + [
    # ---- hit pattern: the covflow context and what the hit kill changes.
    # Integer quantities, one bin per value. The raw view reads <hit>_raw (the
    # hit content before the kill), the corrected view the content after it.
    dict(name='mu_n_pix_hit',       kind='mu', expr='{mu}_n_pix_hit',       bins=(-0.5, 10.5, 11),
         label=r'$\mu$ pixel hits'),
    dict(name='mu_n_pix_b_hit',     kind='mu', expr='{mu}_n_pix_b_hit',     bins=(-0.5, 8.5, 9),
         label=r'$\mu$ BPix hits'),
    dict(name='mu_n_pix_e_hit',     kind='mu', expr='{mu}_n_pix_e_hit',     bins=(-0.5, 8.5, 9),
         label=r'$\mu$ FPix hits'),
    dict(name='mu_n_pix_layer',     kind='mu', expr='{mu}_n_pix_layer',     bins=(-0.5, 7.5, 8),
         label=r'$\mu$ pixel layers with a hit'),
    dict(name='mu_n_trk_layer',     kind='mu', expr='{mu}_n_trk_layer',     bins=(-0.5, 20.5, 21),
         label=r'$\mu$ tracker layers with a hit'),
    dict(name='mu_pix_first_b_layer', kind='mu', expr='{mu}_pix_first_b_layer', bins=(-0.5, 4.5, 5),
         label=r'$\mu$ first BPix layer (0 = none)'),
    dict(name='mu_pix_first_e_disk', kind='mu', expr='{mu}_pix_first_e_disk', bins=(-0.5, 3.5, 4),
         label=r'$\mu$ first FPix disk (0 = none)'),
    dict(name='mu_pix_first_layer', kind='mu', expr='{mu}_pix_first_layer', bins=(-0.5, 7.5, 8),
         ticks=['none', 'L1', 'L2', 'L3', 'L4', 'D1', 'D2', 'D3'],
         label=r'$\mu$ first pixel layer'),
    dict(name='mu_pix_valid_layers', kind='mask', expr='{mu}_pix_valid_mask', bins=(-0.5, 6.5, 7),
         ticks=['L1', 'L2', 'L3', 'L4', 'D1', 'D2', 'D3'],
         label=r'$\mu$ pixel layer / disk with a valid hit (muons per layer)'),
    # the covflow context cell the flows see: pixel hits x first BPix layer
    dict(name='mu_context_cell', kind='mu',
         expr='clip({mu}_n_pix_hit, 0, 6) + 7 * {mu}_pix_first_b_layer', bins=(-0.5, 34.5, 35),
         label=r'$\mu$ context cell = min(pixel hits, 6) + 7 $\times$ first BPix layer'),
    # not changed by the emulation (raw = corrected): the reason a first hit is
    # where it is, as reconstructed
    dict(name='mu_n_pix_miss_inner', kind='mu', expr='{mu}_n_pix_miss_inner', bins=(-0.5, 4.5, 5),
         label=r'$\mu$ missing inner pixel layers (not emulated)'),
    dict(name='mu_n_pix_inact_inner', kind='mu', expr='{mu}_n_pix_inact_inner', bins=(-0.5, 4.5, 5),
         label=r'$\mu$ inactive inner pixel layers (not emulated)'),
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
          'mu1_bs_dxy_e', 'mu2_bs_dxy_e',
          # changed by the hit kill, no raw copy stored
          'mu1_pix_miss_mask', 'mu2_pix_miss_mask'}
# computed per view (View.cache), never read from the ntuple
COMPUTED = {'sigma_m', 'mass_mu'}

# ---------------------------------------------------------------------------
# sPlot-free comparison: quantities correlated with the mass resolution, in
# bins (the last bin is the overflow). In each bin the data signal yield comes
# from its own mass fit. Edges in the variable's units; sigma_m also gets the
# per-bin mass width (calibration of the predicted mass error).
# ---------------------------------------------------------------------------
BINFIT = {
    'sigma_m':         [0, .015, .018, .021, .024, .027, .030, .034, .040, .050, .070, 1e3],
    'vtx_chi2':        [0, .05, .1, .2, .3, .455, .7, 1., 1.5, 2., 3., 5., 8., 15., 1e9],
    'mu_sigma_qoverp': [0, 6e-4, 8e-4, 1e-3, 1.2e-3, 1.4e-3, 1.6e-3, 1.8e-3, 2.2e-3, 3e-3, 5e-3, 1e3],
    'mu_sigma_phi':    [0, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 1e-3, 1.5e-3, 3e-3, 1e3],
    'mu_sigma_lambda': [0, 1.5e-4, 2e-4, 2.5e-4, 3e-4, 3.5e-4, 4e-4, 5e-4, 7e-4, 1.2e-3, 1e3],
    'mu_relerr_p':     [0, .007, .008, .009, .010, .012, .014, .016, .018, .022, .030, 1e3],
}
CALIB_VAR = 'sigma_m'
CHI2_VAR = 'vtx_chi2'
CHI2_1DOF_MEDIAN = 0.454936
WEAK_REL_ERR = 0.10          # per-bin data yield error above which a bin is 'weak'
MU_MASS = 0.1056583755
SIGMA_M_COV = ('qoverp_qoverp', 'qoverp_lambda', 'qoverp_phi',
               'lambda_lambda', 'lambda_phi', 'phi_phi')
KILL_CATS = (('all', 'all events'), ('nokill', 'no muon hit killed'),
             ('killed', '>= 1 muon hit killed'))

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
def expand(patterns):
    files = []
    for p in patterns:
        if '<' in p and '>' in p:
            die('placeholder path %s: give --data-dir / --mc-dir, or --data / --mc' % p)
        g = sorted(glob.glob(p)) if any(ch in p for ch in '*?[') else [p]
        if not g:
            die('no file matches %s' % p)
        files += g
    for f in files:
        if not f.startswith('root://') and not os.path.isfile(f):
            die('%s not found' % f)
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
    if var['kind'] == 'mask':
        # one entry per (muon, layer with the bit set), at the layer index
        xs, ws = [], []
        for mu in ('mu1', 'mu2'):
            m = view.eval(expr.format(mu=mu))[sel]
            ok = np.isfinite(m)
            m, w = m[ok].astype(np.int64), w_evt[sel][ok]
            for b in range(var['bins'][2]):
                on = ((m >> b) & 1).astype(bool)
                xs.append(np.full(int(on.sum()), float(b)))
                ws.append(w[on])
        return np.concatenate(xs), np.concatenate(ws)
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
                curve=curve, _res=res)
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
# predicted mass uncertainty and MC truth
# ---------------------------------------------------------------------------
def sigma_m_inputs(prefix):
    '''Branches predicted_mass reads for a covariance prefix (cov / cov_corr).'''
    out = set()
    for mu in ('mu1', 'mu2'):
        out |= {'%s_%s' % (mu, k) for k in ('pt', 'eta', 'phi', 'charge')}
        out |= {'%s_%s_%s' % (mu, prefix, e) for e in SIGMA_M_COV}
    return out


def gen_inputs():
    return {'%s_gen_%s' % (mu, k) for mu in ('mu1', 'mu2') for k in ('pt', 'eta', 'phi')}


def _p3(pt, eta, phi):
    return np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)], -1)


def _mass(p1, p2):
    e1 = np.sqrt(np.sum(p1 ** 2, -1) + MU_MASS ** 2)
    e2 = np.sqrt(np.sum(p2 ** 2, -1) + MU_MASS ** 2)
    m2 = 2 * MU_MASS ** 2 + 2 * (e1 * e2 - np.sum(p1 * p2, -1))
    return np.sqrt(np.clip(m2, 0, None)), e1, e2


def predicted_mass(view, prefix):
    '''Dimuon mass from the two muons and its uncertainty, propagated linearly
    from the (q/p, lambda, phi) block of each muon's covariance (dxy, dsz do
    not enter the momentum; the two muons are independent).
        p = |q / (q/p)| (cos l cos f, cos l sin f, sin l)
        dm/dtheta = (d m^2 / d p) . (d p / d theta) / 2m,
        d m^2 / d p1 = 2 (E2 p1 / E1 - p2)'''
    idx = {'qoverp': 0, 'lambda': 1, 'phi': 2}
    P, D, C = [], [], []
    for mu in ('mu1', 'mu2'):
        pt, eta, phi = (np.asarray(view.get('%s_%s' % (mu, k)), np.float64)
                        for k in ('pt', 'eta', 'phi'))
        q = np.sign(np.asarray(view.get(mu + '_charge'), np.float64))
        lam = np.arctan(np.sinh(eta))
        pv = _p3(pt, eta, phi)
        pmag = pt * np.cosh(eta)
        with np.errstate(all='ignore'):
            x = q / pmag
            d_x = -pv / x[:, None]
        d_l = np.stack([-pmag * np.sin(lam) * np.cos(phi), -pmag * np.sin(lam) * np.sin(phi),
                        pmag * np.cos(lam)], -1)
        d_f = np.stack([-pt * np.sin(phi), pt * np.cos(phi), np.zeros_like(pt)], -1)
        M = np.empty((len(pt), 3, 3))
        for e in SIGMA_M_COV:
            a_, b_ = e.split('_')
            v = np.asarray(view.get('%s_%s_%s' % (mu, prefix, e)), np.float64)
            M[:, idx[a_], idx[b_]] = v
            M[:, idx[b_], idx[a_]] = v
        P.append(pv)
        D.append((d_x, d_l, d_f))
        C.append(M)
    m, e1, e2 = _mass(P[0], P[1])
    E = (e1, e2)
    var = np.zeros_like(m)
    with np.errstate(all='ignore'):
        for k in (0, 1):
            o = 1 - k
            g = 2 * (E[o][:, None] * P[k] / E[k][:, None] - P[o])
            J = np.stack([np.sum(g * d, -1) for d in D[k]], -1) / (2 * m)[:, None]
            var += np.einsum('ni,nij,nj->n', J, C[k], J)
    return m, np.sqrt(np.where(var >= 0, var, np.nan))


def add_computed(view, prefix, label):
    '''sigma_m and mass_mu into the view; prints how well the two-muon mass
    reproduces the ntuple mass (a check of the inputs, not of the physics).'''
    m, s = predicted_mass(view, prefix)
    view.cache['mass_mu'] = m
    view.cache['sigma_m'] = s
    mm = np.asarray(view.get('mass'), np.float64)
    ok = np.isfinite(m) & np.isfinite(mm) & np.isfinite(s)
    if ok.any():
        print('  %-12s sigma(m): median %.1f MeV (%s); two-muon mass vs ntuple mass: '
              'median |diff| %.2f MeV' % (label, 1e3 * np.median(s[ok]), prefix,
                                           1e3 * np.median(np.abs(m[ok] - mm[ok]))))
    else:
        print('  %-12s sigma(m): no finite value (%s branches missing or NaN?)' % (label, prefix))


# ---------------------------------------------------------------------------
# sPlot-free histograms: per-bin mass histograms (summable over epochs)
# ---------------------------------------------------------------------------
def fit_edges(a):
    return np.linspace(JPSI - a.sweight_window, JPSI + a.sweight_window,
                       int(round(2 * a.sweight_window / 0.001)) + 1)


def var_by_name(name):
    for v in VARS:
        if v['name'] == name:
            return v
    die('BINFIT variable %s is not in VARS' % name)


def entries(view, var, which, sel):
    '''Values of a variable and the event index of each entry (muon variables:
    two entries per event).'''
    expr = var.get(which, var['expr']) if which in ('corr', 'raw') else var['expr']
    ev = np.nonzero(sel)[0]
    if var['kind'] == 'mu':
        xs = [view.eval(expr.format(mu=mu))[sel] for mu in ('mu1', 'mu2')]
        return np.concatenate(xs), np.concatenate([ev, ev])
    return view.eval(expr)[sel], ev


def build_binfit(dv, sw, sel_d, cv, w_c, sel_c, rv, w_r, sel_r, a, ep):
    '''For every BINFIT variable and bin: the data mass histogram (unweighted,
    1 MeV, for the per-bin fit), the sum of sWeights, the MC yields; for the
    calibration variable also the MC mass histograms and the sums of x for
    the bin means. Plus the full-window mass histograms (global fits) and the
    MC truth histograms. Everything is a sum: epochs add.'''
    fe = fit_edges(a)
    out = dict(fit_edges=fe, vars={})
    samples = dict(data=(dv, 'data', np.ones(dv.n), sel_d),
                   raw=(rv, 'raw', w_r, sel_r), corr=(cv, 'corr', w_c, sel_c))
    out['mass_all'] = {k: np.histogram(np.asarray(v.get('mass'))[s], bins=fe, weights=w[s])[0]
                       for k, (v, _, w, s) in samples.items()}
    prog = Progress(len(BINFIT), '%s per-bin mass histograms' % ep)
    for i, (name, edges) in enumerate(BINFIT.items()):
        var = var_by_name(name)
        edges = np.asarray(edges, float)
        nb = len(edges) - 1
        o = dict(edges=edges, kind=var['kind'], label=var['label'])
        for k, (view, which, w, s) in samples.items():
            x, ev = entries(view, var, which, s)
            ok = np.isfinite(x)
            x, ev = x[ok], ev[ok]
            b = np.clip(np.digitize(x, edges) - 1, 0, nb - 1)
            ww = w[ev]
            o['%s_yield' % k] = np.bincount(b, weights=ww, minlength=nb)
            o['%s_sx' % k] = np.bincount(b, weights=ww * x, minlength=nb)
            if True:                    # per-bin mass histograms, every sample
                mass = np.asarray(view.get('mass'), np.float64)[ev]
                h, _, _ = np.histogram2d(b, mass, bins=(np.arange(nb + 1) - 0.5, fe), weights=ww)
                o['%s_mass' % k] = h
            if k == 'data':
                o['data_swsum'] = np.bincount(b, weights=sw[ev], minlength=nb)
        out['vars'][name] = o
        prog.update(i + 1)
    # ---- MC truth
    have_gen = all(n in cv.c for n in gen_inputs())
    if have_gen:
        p1 = _p3(*(np.asarray(cv.get('mu1_gen_%s' % k), np.float64) for k in ('pt', 'eta', 'phi')))
        p2 = _p3(*(np.asarray(cv.get('mu2_gen_%s' % k), np.float64) for k in ('pt', 'eta', 'phi')))
        m_gen = _mass(p1, p2)[0]
        kill = np.zeros(cv.n, bool)
        for mu in ('mu1', 'mu2'):
            if mu + '_emu_killed' in cv.c:
                kill |= np.asarray(cv.c[mu + '_emu_killed']) > 0.5
        dm_e = np.linspace(-0.2, 0.2, 201)
        pu_e = np.linspace(-8, 8, 161)
        tr = dict(dm_edges=dm_e, pull_edges=pu_e,
                  have_kill=any(mu + '_emu_killed' in cv.c for mu in ('mu1', 'mu2')))
        for k, (view, _, w, s) in samples.items():
            if k == 'data':
                continue
            m = np.asarray(view.get('mass'), np.float64)
            sm = np.asarray(view.get('sigma_m'), np.float64)
            with np.errstate(all='ignore'):
                dm = m - m_gen
                pull = dm / sm
            for cat, _ in KILL_CATS:
                c = s & np.isfinite(dm) & ({'all': True, 'nokill': ~kill, 'killed': kill}[cat])
                tr['%s_%s_dm' % (k, cat)] = np.histogram(np.clip(dm[c], -0.2, 0.2 - 1e-9),
                                                         bins=dm_e, weights=w[c])[0]
                c &= np.isfinite(pull)
                tr['%s_%s_pull' % (k, cat)] = np.histogram(np.clip(pull[c], -8, 8 - 1e-9),
                                                           bins=pu_e, weights=w[c])[0]
        out['truth'] = tr
    else:
        print('  MC truth page skipped: %s not in the MC ntuple' % ', '.join(sorted(gen_inputs())))
    return out


KEEP_FIRST = {'edges', 'fit_edges', 'dm_edges', 'pull_edges', 'kind', 'label', 'have_kill'}


def sum_tree(items, key=None):
    '''Element-wise sum over epochs of the nested dicts from build_binfit;
    edges, labels and flags (KEEP_FIRST) are taken from the first epoch. A
    key missing in some epoch (e.g. no MC truth) is dropped.'''
    first = items[0]
    if isinstance(first, dict):
        return {k: sum_tree([it[k] for it in items], k)
                for k in first if all(k in it for it in items)}
    if key in KEEP_FIRST:
        return first
    return sum(np.asarray(it, float) for it in items)


# ---------------------------------------------------------------------------
# fits for the resolution pdf
# ---------------------------------------------------------------------------
def sigma68_model(model, p):
    '''sigma68 of the fitted signal shape (over the fit window).'''
    x = np.linspace(model.lo, model.hi, 20001)
    f = model.signal_pdf(x, p)
    c = np.concatenate([[0.], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(x))])
    if not np.isfinite(c[-1]) or c[-1] <= 0:
        return np.nan
    q = np.interp([0.158655, 0.841345], c / c[-1], x)
    return float(0.5 * (q[1] - q[0]))


def analyse_resolution(R, a, title):
    '''All fits behind the resolution pdf, from the summed histograms R.'''
    from covflow import splot as SP
    fe = np.asarray(R['fit_edges'])
    model = SP.MassModel(fe[0], fe[-1], signal=a.sweight_signal, tails=a.sweight_tails,
                         background=a.sweight_background)
    names = model.names
    shape = [k for k in ('rsig', 'fcore') if k in names] + list(model.tail_names) + \
        [k for k in ('fb',) if k in names]
    bkg0 = {k: v for k, v in dict(Nb=0., lam1=0., lam2=0., fb=0.5).items() if k in names}

    def fit(counts, start=None, fixed=None):
        counts = np.asarray(counts, float)
        if counts.sum() < a.binfit_min:
            return None
        try:
            r = SP.fit(model, counts=counts, edges=fe, start=start, fixed=fixed,
                       verbose=False, heartbeat=False, label=title)
        except Exception as e:     # noqa: BLE001 -- a failed bin is reported, not fatal
            print('\n  fit failed (%s): %s' % (title, e))
            return None
        return r

    res = dict(title=title, model=model.describe(), glob={}, vars={})
    n_fits = 3 + 3 * sum(len(R['vars'][n]['edges']) - 1 for n in R['vars'])
    prog = Progress(n_fits, '%s resolution fits' % title)
    done = [0]

    def tick():
        done[0] += 1
        prog.update(done[0])

    # ---- global fits: one method for the three samples
    G = {}
    G['data'] = fit(R['mass_all']['data'])
    tick()
    if G['data'] is None:
        die('%s: the global data mass fit failed' % title)
    gd = dict(G['data'].values)
    for k in ('raw', 'corr'):
        cnt = np.asarray(R['mass_all'][k], float)
        st = dict(gd, Ns=float(cnt.sum()), **bkg0)
        G[k] = fit(cnt, start=st, fixed=bkg0)
        tick()
    for k, r in G.items():
        if r is None:
            res['glob'][k] = dict(valid=False)
            continue
        res['glob'][k] = dict(valid=bool(r.valid), sigma68=sigma68_model(model, r.values),
                              sigma_eff=float(model.sigma_eff(r.values)),
                              s1=float(r.values['s1']), mu=float(r.values['mu']),
                              Ns=float(r.values['Ns']), par=dict(r.values))
    # ---- per-bin fits
    for name, o in R['vars'].items():
        nb = len(o['edges']) - 1
        tot = float(np.asarray(R['mass_all']['data']).sum())
        rv_ = dict(edges=np.asarray(o['edges']), kind=o['kind'], label=o['label'],
                   data_yield=np.full(nb, np.nan), data_err=np.full(nb, np.nan),
                   data_s68=np.full(nb, np.nan), data_valid=np.zeros(nb, bool),
                   data_swsum=np.asarray(o['data_swsum'], float))
        for k in ('raw', 'corr'):
            rv_['%s_yield' % k] = np.asarray(o['%s_yield' % k], float)
            rv_['%s_s68' % k] = np.full(nb, np.nan)
        for k in ('data', 'raw', 'corr'):
            rv_['%s_par' % k] = [None] * nb       # fitted parameters per bin (plots)
        for k in ('data', 'raw', 'corr'):
            with np.errstate(all='ignore'):
                rv_['%s_mean' % k] = np.asarray(o['%s_sx' % k], float) / \
                    np.asarray(o['%s_yield' % k], float)
        for b in range(nb):
            cnt = np.asarray(o['data_mass'][b], float)
            f_ = cnt.sum() / max(tot, 1.)
            st = dict(gd, Ns=gd['Ns'] * f_, Nb=max(gd['Nb'] * f_, 1.))
            r = fit(cnt, start=st, fixed={k: gd[k] for k in shape})
            tick()
            if r is not None:
                rv_['data_yield'][b] = r.values['Ns']
                rv_['data_err'][b] = r.errors['Ns']
                rv_['data_s68'][b] = sigma68_model(model, r.values)
                rv_['data_valid'][b] = bool(r.valid)
                rv_['data_par'][b] = dict(r.values)
            # MC: signal only (no background), shape as in the MC global fit,
            # width and peak free -- the same treatment as the data bin
            for k in ('raw', 'corr'):
                if G[k] is None:
                    tick()
                    continue
                gm = dict(G[k].values)
                cm = np.asarray(o['%s_mass' % k][b], float)
                fx = dict({s: gm[s] for s in shape}, **bkg0)
                r = fit(cm, start=dict(gm, Ns=float(cm.sum()), **bkg0), fixed=fx)
                tick()
                if r is not None:
                    rv_['%s_s68' % k][b] = sigma68_model(model, r.values)
                    rv_['%s_par' % k][b] = dict(r.values)
        with np.errstate(all='ignore'):
            rv_['data_precise'] = rv_['data_valid'] & \
                (rv_['data_err'] / rv_['data_yield'] < WEAK_REL_ERR)
        res['vars'][name] = rv_
    prog.update(n_fits)
    # ---- vertex chi2 scale factor from the signal median
    if CHI2_VAR in res['vars']:
        o = res['vars'][CHI2_VAR]
        k_ = {}
        for k in ('data', 'raw', 'corr'):
            y = np.nan_to_num(o['%s_yield' % k]) if k != 'data' else np.nan_to_num(o['data_yield'])
            med = binned_median(y, o['edges'])
            k_[k] = dict(median=med, k=float(np.sqrt(med / CHI2_1DOF_MEDIAN)) if med > 0 else np.nan)
        swmed = binned_median(np.clip(o['data_swsum'], 0, None), o['edges'])
        k_['data_sweights'] = dict(median=swmed, k=float(np.sqrt(swmed / CHI2_1DOF_MEDIAN))
                                   if swmed > 0 else np.nan)
        res['chi2_k'] = k_
    # ---- MC truth widths
    if 'truth' in R:
        T = R['truth']
        tw = {}
        for k in ('raw', 'corr'):
            for cat, _ in KILL_CATS:
                h1, h2 = T['%s_%s_dm' % (k, cat)], T['%s_%s_pull' % (k, cat)]
                tw['%s_%s' % (k, cat)] = dict(dm_s68=sigma68(h1, T['dm_edges']),
                                              pull_s68=sigma68(h2, T['pull_edges']),
                                              n=float(np.sum(h1)))
        res['truth'] = tw
    return res


def binned_median(y, edges):
    '''Median of a binned distribution, linear inside the bin (the overflow
    bin's upper edge is only used if the median falls in it).'''
    y = np.clip(np.asarray(y, float), 0, None)
    if y.sum() <= 0:
        return np.nan
    c = np.concatenate([[0.], np.cumsum(y)]) / y.sum()
    return float(np.interp(0.5, c, np.asarray(edges, float)))


# ---------------------------------------------------------------------------
# one epoch
# ---------------------------------------------------------------------------
def build_hists(ep, label, dv, wd_evt, sel_d, cv, w_c, sel_c, rv, w_r, sel_r, me,
                data_mass_curve=None):
    '''Histograms of every VARS entry for data and both MC views, plus the
    dimuon mass (the fitted data signal shape with sPlot, the data mass
    itself without).'''
    hists = {}
    hmr, _ = np.histogram(rv.get('mass')[sel_r], bins=me, weights=w_r[sel_r])
    hmc, _ = np.histogram(cv.get('mass')[sel_c], bins=me, weights=w_c[sel_c])
    if data_mass_curve is None:
        hmd, _ = np.histogram(dv.get('mass')[sel_d], bins=me, weights=wd_evt[sel_d])
    else:
        hmd = data_mass_curve
    hists['__mass__'] = dict(edges=me, data=hmd, raw=hmr, corr=hmc)
    prog = Progress(len(VARS), '%s histograms, %s' % (ep, label))
    for i, v in enumerate(VARS):
        xd, wd = values_and_weights(dv, v, 'data', wd_evt, sel_d)
        xc, wc = values_and_weights(cv, v, 'corr', w_c, sel_c)
        xr, wr = values_and_weights(rv, v, 'raw', w_r, sel_r)
        hd, ed, edges = hist(xd, wd, v['bins'])
        hc, ec, _ = hist(xc, wc, v['bins'])
        hr, er, _ = hist(xr, wr, v['bins'])
        hists[v['name']] = dict(edges=edges, data=(hd, ed), corr=(hc, ec), raw=(hr, er),
                                stats=dict(data=wstats(xd, wd), corr=wstats(xc, wc),
                                           raw=wstats(xr, wr)))
        prog.update(i + 1)
    return hists


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
    var_names |= {'pt', 'eta'} | sigma_m_inputs('cov')
    var_names -= COMPUTED
    sel_names = names_in(sel_data) | names_in(a.mc_selection or '')

    # ---- data: read the window, fit, sWeights
    d_cols, _ = read(data_files, var_names | sel_names | {'mass'}, '%s data' % ep, sel_data,
                     a.max_entries, a.prescale_data)
    if not d_cols or len(d_cols['mass']) < 1000:
        die('%s: only %d data candidates selected' % (ep, len(d_cols.get('mass', []))))
    sw, sp_info = sweights(d_cols['mass'].astype(np.float64), a, '%s_%s_splot.pdf' % (a.out, ep))
    sp_info.pop('_res')
    dv = View(d_cols, 'nominal')
    sel_d = np.ones(dv.n, bool)
    add_computed(dv, 'cov', 'data')

    # ---- MC: read everything that passes EITHER view's preselection, then
    # select per view (the raw view uses the raw mass, pT, Lxy, ...)
    def raw_names_of(names):
        return {n + '_raw' for n in names} | {m for n in names
                                               for m in names_in(DERIVED_RAW.get(n + '_raw', ''))}
    need_mc = var_names | sel_names | raw_names_of(var_names | sel_names) | \
        sigma_m_inputs('cov_corr') | gen_inputs() | \
        {'mass', 'mass_raw', 'emu_weight', pu_branch, 'mu1_pt_raw', 'mu2_pt_raw',
         'mu1_emu_killed', 'mu2_emu_killed'}
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
    add_computed(rv, 'cov', 'MC raw')
    add_computed(cv, 'cov_corr', 'MC corrected')
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
    # (both now sum to the data signal yield on their sPlot selection)

    me = mass_edges(a)
    curve = sp_info.pop('curve')
    hists = {}
    # --- with sPlot: data sWeighted over the whole fit window, MC normalised
    # to the data signal yield
    hists['sw'] = build_hists(ep, 'sPlot', dv, sw, sel_d, cv, w_c, sel_c, rv, w_r, sel_r,
                              me, data_mass_curve=curve)
    s68 = {k: sigma68(v, me) for k, v in hists['sw']['__mass__'].items() if k != 'edges'}
    sp_info['sigma68_MeV'] = {k: 1e3 * v for k, v in s68.items()}
    print('  mass resolution sigma68: data %.1f MeV (fitted signal), MC raw %.1f, '
          'MC corrected %.1f' % tuple(1e3 * s68[k] for k in ('data', 'raw', 'corr')))
    # --- without sPlot: the selection's own mass cut back in (the covflow
    # training conditions: no sWeights, background included), every data
    # candidate weight 1, MC normalised to the data count
    if dropped:
        mcut = ' & '.join('(%s)' % t for t in dropped)
        sel_d_ns = sel_d & dv.eval(mcut).astype(bool)
        sel_c_ns = sel_c & cv.eval(mcut).astype(bool)
        sel_r_ns = sel_r & rv.eval(mcut).astype(bool)
    else:
        mcut = 'the fit window'
        sel_d_ns, sel_c_ns, sel_r_ns = sel_d, sel_c, sel_r
    ones = np.ones(dv.n)
    n_ns = float(sel_d_ns.sum())
    # signal inside the cut from the FITTED signal shape (curve, expected signal
    # counts per mass bin); NOT from the sum of sWeights inside the cut, which
    # is biased (the weights are valid only summed over the whole window)
    pur = np.nan
    try:
        cen = 0.5 * (me[1:] + me[:-1])
        ns = dict(FUNCS)
        ns['mass'] = cen
        inside = np.asarray(eval(mcut, {'__builtins__': {}}, ns), bool) if dropped else \
            np.ones(cen.size, bool)
        pur = float(curve[inside].sum() / max(n_ns, 1.))
    except Exception:
        pass
    print('  without sPlot: %s data candidates in %s, signal fraction there %.1f%% (fitted '
          'signal shape)' % (f'{int(n_ns):,}', mcut, 100 * pur))
    hists['nosw'] = build_hists(ep, 'no sPlot', dv, ones, sel_d_ns,
                                cv, w_c * n_ns / w_c[sel_c_ns].sum(), sel_c_ns,
                                rv, w_r * n_ns / w_r[sel_r_ns].sum(), sel_r_ns, me)
    sp_info['no_splot'] = dict(mass_cut=mcut, n_data=int(n_ns), signal_fraction=pur,
                               n_mc_raw=int(sel_r_ns.sum()), n_mc_corr=int(sel_c_ns.sum()))
    # --- sPlot-free: per-bin mass histograms for the resolution pdf
    if not a.no_resolution:
        hists['res'] = build_binfit(dv, sw, sel_d, cv, w_c, sel_c, rv, w_r, sel_r, a, ep)
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


MODES = dict(sw=dict(tag='sw', data='data (sWeighted)', norm='MC normalised to the data '
                     'signal yield', mass='data: fitted signal shape'),
             nosw=dict(tag='nosw', data='data (no background subtraction)',
                       norm='no sPlot: data candidates in the selection mass cut, background '
                            'included; MC normalised to the data count',
                       mass='data (background included)'))


def _panel(ax, rx, h, s, v, mode, logy):
    e = h['edges']
    c = 0.5 * (e[1:] + e[:-1])
    hd, ed = h['data']
    ax.errorbar(c, hd, yerr=np.sqrt(ed), fmt='o', ms=2.5, color=COLORS['data'],
                label=MODES[mode]['data'], zorder=3)
    for which, ls in (('raw', '--'), ('corr', '-')):
        hm, em = h[which]
        ax.stairs(hm, e, color=COLORS[which], ls=ls, lw=1.5,
                  label='%s  $\\chi^2$/ndf %.0f/%d' % (LABELS[which], s[which]['chi2'],
                                                       s[which]['ndf']))
        with np.errstate(all='ignore'):
            r = np.where(hd != 0, hm / hd, np.nan)
            rerr = np.where(hd != 0, np.sqrt(em) / np.abs(hd), np.nan)
        rx.errorbar(c, r, yerr=rerr, fmt='o' if which == 'corr' else 's', ms=2,
                    color=COLORS[which], alpha=0.9)
    with np.errstate(all='ignore'):
        dr = np.where(hd != 0, np.sqrt(ed) / np.abs(hd), np.nan)
    rx.fill_between(c, 1 - dr, 1 + dr, step='mid', color='grey', alpha=0.25, lw=0)
    rx.axhline(1, color='black', lw=0.8)
    rx.set_ylim(0.5, 1.5)
    rx.set_ylabel('MC / data', fontsize=11)
    rx.set_xlabel(v['label'] + ('   (both muons)' if v['kind'] in ('mu', 'mask') else ''),
                  fontsize=13)
    if logy:
        pos = np.concatenate([x[x > 0] for x in (hd, h['raw'][0], h['corr'][0])])
        if pos.size:
            ax.set_yscale('log')
            ax.set_ylim(0.3 * pos.min(), 5 * pos.max())
    else:
        ax.set_ylim(0, 1.25 * max(np.nanmax(hd + np.sqrt(ed)), h['raw'][0].max(),
                                   h['corr'][0].max()))
    ax.set_ylabel('candidates' if v['kind'] == 'cand' else 'muons', fontsize=13)
    if v.get('ticks'):
        rx.set_xticks(range(len(v['ticks'])))
        rx.set_xticklabels(v['ticks'])
    ax.legend(fontsize=7, frameon=False)
    for a_ in (ax, rx):
        a_.tick_params(labelsize=10)


def plot_pages(hists, title, pdf_path, mode):
    """One page per quantity, linear (left) and log (right) scale; the mass
    page first, a summary table last."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    M = MODES[mode]
    table = {}
    with PdfPages(pdf_path) as pdf:
        hm = hists['__mass__']
        e = hm['edges']
        s68 = {k: sigma68(hm[k], e) for k in ('data', 'raw', 'corr')}
        table['__mass__'] = {'sigma68_MeV': {k: 1e3 * v for k, v in s68.items()}}
        fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
        norm = lambda h: h / max(h.sum(), 1e-300) / np.diff(e)
        for ax, logy in zip(axs, (False, True)):
            ax.stairs(norm(hm['data']), e, color=COLORS['data'], lw=2,
                      label=r'%s  $\sigma_{68}$ = %.1f MeV' % (M['mass'], 1e3 * s68['data']))
            for which, ls in (('raw', '--'), ('corr', '-')):
                ax.stairs(norm(hm[which]), e, color=COLORS[which], ls=ls, lw=1.6,
                          label=r'%s  $\sigma_{68}$ = %.1f MeV' % (LABELS[which], 1e3 * s68[which]))
            if logy:
                ax.set_yscale('log')
            ax.set_xlabel(r'$m(\mu\mu)$ [GeV]', fontsize=13)
            ax.set_ylabel('normalised', fontsize=13)
            ax.legend(fontsize=8, frameon=False)
        fig.suptitle('%s, %s: dimuon mass%s' % (title, mode, ' (the sWeighted data mass would be '
                     'the fit itself)' if mode == 'sw' else ''), fontsize=10)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        for v in VARS:
            h = hists[v['name']]
            s = summarise(h)
            table[v['name']] = s
            fig = plt.figure(figsize=(14, 6.5))
            gs = fig.add_gridspec(2, 2, height_ratios=(3, 1), hspace=0.06, wspace=0.22)
            for col, logy in enumerate((False, True)):
                ax = fig.add_subplot(gs[0, col])
                rx = fig.add_subplot(gs[1, col], sharex=ax)
                ax.tick_params(labelbottom=False)
                _panel(ax, rx, h, s, v, mode, logy)
                ax.set_title('log scale' if logy else 'linear scale', fontsize=10)
            st = s['stats']
            fig.suptitle('%s, %s   mean: data %.4g, MC raw %.4g, MC corrected %.4g' % (
                title, M['data'], st['data'][0], st['raw'][0], st['corr'][0]), fontsize=10)
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
        ax.set_title('%s: summary, %s (chi2 over the shown bins; %s)\nmass sigma68: data %.1f MeV, '
                     'MC raw %.1f, MC corrected %.1f   (indicative: window-truncated%s; the '
                     'like-for-like numbers are in _resolution.pdf)'
                     % (title, M['data'], M['norm'], s68['data'], s68['raw'], s68['corr'],
                        ', data with background' if mode == 'nosw' else ''),
                     fontsize=8)
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
    '''Per mode (sw, nosw): see _combine.'''
    return {m: _combine([h[m] for h in per_epoch]) for m in ('sw', 'nosw')}


def _combine(per_epoch):
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
# the resolution pdf
# ---------------------------------------------------------------------------
def _fmt_edge(x):
    if x >= 1e2:
        return 'inf'
    return ('%.3g' % x) if x >= 1e-2 or x == 0 else ('%.2g' % x)


def resolution_lines(res):
    g = res['glob']
    L = ['MASS RESOLUTION, one method for all: the same signal model (%s) fitted to each '
         'sample, sigma68 of the fitted signal' % res['model']['signal'],
         '  (data: background fitted, not included; no window truncation)']
    for k, lab in (('data', 'data'), ('raw', 'MC raw'), ('corr', 'MC corrected')):
        x = g.get(k, {})
        if x.get('sigma68') is None:
            L.append('  %-13s fit failed' % lab)
            continue
        L.append('  %-13s sigma68 %5.1f MeV   sigma_eff %5.1f MeV   peak %.4f GeV%s'
                 % (lab, 1e3 * x['sigma68'], 1e3 * x['sigma_eff'], x['mu'],
                    '' if x['valid'] else '   (fit not valid)'))
    if 'chi2_k' in res:
        c = res['chi2_k']
        L += ['', 'VERTEX CHI2 SCALE FACTOR k = sqrt(signal median chi2 / %.3f)  '
              '(1 = honest Gaussian errors)' % CHI2_1DOF_MEDIAN]
        for k, lab in (('data', 'data, per-bin fits'), ('data_sweights', 'data, sWeights'),
                       ('raw', 'MC raw'), ('corr', 'MC corrected')):
            L.append('  %-20s median chi2 %.3f   k %.3f' % (lab, c[k]['median'], c[k]['k']))
    if CALIB_VAR in res['vars']:
        o = res['vars'][CALIB_VAR]
        L += ['', 'MASS PULL WIDTH IN DATA = fitted sigma68 / mean predicted sigma(m), per '
              'sigma(m) bin (1 = honest errors)']
        hdr = '  %-15s' % 'sigma(m) bin' + ''.join('%9s' % s for s in ('data', 'MC raw', 'MC corr'))
        L.append(hdr)
        e = o['edges']
        for b in range(len(e) - 1):
            r = []
            for k in ('data', 'raw', 'corr'):
                s68 = o['data_s68' if k == 'data' else '%s_s68' % k][b]
                mn = o['%s_mean' % k][b]
                r.append(s68 / mn if np.isfinite(s68) and mn > 0 else np.nan)
            weak = not o['data_precise'][b]
            L.append('  %-15s' % ('%s-%s' % (_fmt_edge(e[b]), _fmt_edge(e[b + 1])))
                     + ''.join('%9.3f' % x for x in r) + ('   * weak' if weak and
                                                          np.isfinite(r[0]) else ''))
        L.append('  * weak: data signal yield known to worse than %.0f%% in the bin '
                 '(background-dominated), hollow in the plot' % (100 * WEAK_REL_ERR))
    if 'truth' in res:
        t = res['truth']
        L += ['', 'MC TRUTH: sigma68 of m - m_gen [MeV] and of the pull (m - m_gen)/sigma(m)']
        for cat, lab in KILL_CATS:
            r_, c_ = t['raw_%s' % cat], t['corr_%s' % cat]
            L.append('  %-22s raw %5.1f MeV, pull %.3f   corrected %5.1f MeV, pull %.3f   '
                     '(%.1f%% of MC)' % (lab, 1e3 * r_['dm_s68'], r_['pull_s68'],
                                        1e3 * c_['dm_s68'], c_['pull_s68'],
                                        100 * c_['n'] / max(t['corr_all']['n'], 1e-300)))
    return L


def print_resolution(res):
    print('\n  %s: resolution' % res['title'])
    for line in resolution_lines(res):
        print('  ' + line)


def _rebin(h, k):
    h = np.asarray(h, float)
    return h.reshape(-1, k).sum(1)


def bin_fit_pages(pdf, plt, res, R, name, model):
    """The mass fit in every bin of one variable, data and both MC views in
    the same panel. Data: points, the fitted total (signal + background) and
    the fitted background. MC has no background: each MC view is scaled to the
    data signal yield of the bin and drawn ON TOP of the fitted data
    background, histogram (thin step) and its own signal fit (line), so it
    reads directly against the data points. One page linear, one page log."""
    o, h = res['vars'][name], R['vars'][name]
    fe = np.asarray(R['fit_edges'], float)
    nfe = len(fe) - 1
    rb = max(d for d in (5, 4, 2, 1) if nfe % d == 0)
    de = fe[::rb]
    dc = 0.5 * (de[1:] + de[:-1])
    e = o['edges']
    nb = len(e) - 1
    ncol = 4 if nb > 1 else 1
    nrow = int(np.ceil(nb / ncol))
    size = (4.2 * ncol, 3.4 * nrow) if nb > 1 else (10, 6.5)
    for logy in (False, True):
        fig, axs = plt.subplots(nrow, ncol, figsize=size, squeeze=False)
        for b in range(nrow * ncol):
            ax = axs[b // ncol, b % ncol]
            if b >= nb:
                ax.axis('off')
                continue
            cnt = _rebin(h['data_mass'][b], rb)
            ax.errorbar(1e3 * dc, cnt, yerr=np.sqrt(np.clip(cnt, 0, None)), fmt='o', ms=2,
                        color=COLORS['data'], lw=0.8, zorder=4, label='data')
            pd = o['data_par'][b]
            if pd is not None:
                tot = _rebin(model.bin_expect(fe, pd), rb)
                bkg = _rebin(model.bin_expect(fe, dict(pd, Ns=0.)), rb)
                ns_d = pd['Ns']
                ax.plot(1e3 * dc, tot, color=COLORS['data'], lw=1.2, zorder=5,
                        label=r'data fit: $\sigma_{68}$ %.1f MeV' % (1e3 * o['data_s68'][b]))
                ax.plot(1e3 * dc, bkg, color='grey', ls='--', lw=1, zorder=3,
                        label='data fit: background')
            else:
                bkg = np.zeros_like(cnt)
                ns_d = cnt.sum()
            for k, ls in (('raw', '--'), ('corr', '-')):
                hm = _rebin(h['%s_mass' % k][b], rb)
                if hm.sum() <= 0:
                    continue
                sc = ns_d / hm.sum()
                ax.stairs(bkg + sc * hm, 1e3 * de, color=COLORS[k], lw=0.7, alpha=0.55, ls=ls)
                pm = o['%s_par' % k][b]
                if pm is not None:
                    curve = _rebin(model.bin_expect(fe, dict(pm, Ns=ns_d, Nb=0.)), rb) + bkg
                    ax.plot(1e3 * dc, curve, color=COLORS[k], lw=1.3, ls=ls, zorder=4,
                            label=r'%s fit: $\sigma_{68}$ %.1f MeV'
                            % (LABELS[k], 1e3 * o['%s_s68' % k][b]))
            weak = '' if o['data_precise'][b] else '  (weak)'
            ny = ('no fit' if not np.isfinite(o['data_yield'][b]) else
                  r'%.0f $\pm$ %.0f' % (o['data_yield'][b], o['data_err'][b])
                  if np.isfinite(o['data_err'][b]) else '%.0f' % o['data_yield'][b])
            where = 'whole sample' if name == 'all' else '%s: %s - %s%s' % (
                name, _fmt_edge(e[b]), _fmt_edge(e[b + 1]), weak)
            ax.set_title('%s   N$_S$ %s' % (where, ny), fontsize=8 if nb > 1 else 11)
            if logy:
                pos = cnt[cnt > 0]
                if pos.size:
                    ax.set_yscale('log')
                    ax.set_ylim(0.5 * max(pos.min(), 0.5), 3 * pos.max())
            else:
                ax.set_ylim(0, 1.25 * max(cnt.max(), 1))
            ax.tick_params(labelsize=7)
            ax.set_xlabel(r'$m(\mu\mu)$ [MeV]', fontsize=8)
            ax.legend(fontsize=5.5 if nb > 1 else 9, frameon=False, loc='upper left')
        fig.suptitle('%s: mass fit %s (%s).\nMC (no background) scaled to the '
                     'data signal yield and drawn on top of the fitted data background;\nthin '
                     'step = MC histogram, line = its signal fit'
                     % (res['title'], 'of the whole sample' if name == 'all' else
                        'in each bin of %s' % name, 'log' if logy else 'linear'), fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.94 if nb > 1 else 0.88))
        pdf.savefig(fig)
        plt.close(fig)


def plot_resolution(res, R, pdf_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from covflow import splot as SP
    md = res['model']
    model = SP.MassModel(md['window'][0], md['window'][1], signal=md['signal'],
                         tails=md['tails'], background=md['background'])
    with PdfPages(pdf_path) as pdf:
        # ---- 1. summary
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.03, 0.97, '%s: resolution and sPlot-free checks' % res['title'],
                 fontsize=12, weight='bold', va='top')
        fig.text(0.03, 0.92, '\n'.join(resolution_lines(res)), family='monospace',
                 fontsize=7.5, va='top')
        pdf.savefig(fig)
        plt.close(fig)
        # ---- 1b. the global fits, same layout as the per-bin ones
        g = res['glob']
        if g.get('data', {}).get('par') is not None:
            G1 = dict(title=res['title'], vars={'all': dict(
                edges=np.array([0., 1e3]), data_par=[g['data']['par']],
                data_s68=np.array([g['data']['sigma68']]),
                data_yield=np.array([g['data']['Ns']]), data_err=np.array([np.nan]),
                data_precise=np.array([True]),
                **{'%s_%s' % (k, f): ([g[k]['par']] if f == 'par' else np.array([g[k]['sigma68']]))
                   if g.get(k, {}).get('par') is not None else
                   ([None] if f == 'par' else np.array([np.nan]))
                   for k in ('raw', 'corr') for f in ('par', 's68')})})
            R1 = dict(fit_edges=R['fit_edges'], vars={'all': {
                '%s_mass' % k: [R['mass_all'][k]] for k in ('data', 'raw', 'corr')}})
            bin_fit_pages(pdf, plt, G1, R1, 'all', model)
        # ---- 2. calibration of sigma(m)
        if CALIB_VAR in res['vars']:
            o = res['vars'][CALIB_VAR]
            fig = plt.figure(figsize=(8, 7.5))
            gs = fig.add_gridspec(2, 1, height_ratios=(3, 1.3), hspace=0.06)
            ax, rx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
            hi = 0
            for k, mk, lab in (('data', 'o', 'data (fit per bin)'), ('raw', 's', LABELS['raw']),
                               ('corr', 'D', LABELS['corr'])):
                x = 1e3 * o['%s_mean' % k]
                y = 1e3 * o['data_s68' if k == 'data' else '%s_s68' % k]
                ok = np.isfinite(x) & np.isfinite(y)
                if k == 'data':
                    good = ok & o['data_precise']
                    weak = ok & ~o['data_precise']
                    for sel_, mfc, lb in ((good, COLORS[k], lab),
                                          (weak, 'none', 'data, weak bin (yield error > %.0f%%)'
                                           % (100 * WEAK_REL_ERR))):
                        if sel_.any():
                            ax.plot(x[sel_], y[sel_], 'o', color=COLORS[k], mfc=mfc, ms=5,
                                    label=lb)
                            rx.plot(x[sel_], y[sel_] / x[sel_], 'o', color=COLORS[k], mfc=mfc,
                                    ms=4)
                else:
                    ax.plot(x[ok], y[ok], mk + '-', color=COLORS[k], ms=5, lw=1, label=lab)
                    rx.plot(x[ok], y[ok] / x[ok], mk + '-', color=COLORS[k], ms=4, lw=1)
                if ok.any():
                    hi = max(hi, np.nanmax(x[ok]), np.nanmax(y[ok]))
            ax.plot([0, 1.1 * hi], [0, 1.1 * hi], color='grey', lw=0.8, ls=':',
                    label='honest errors')
            ax.set_xlim(0, 1.1 * hi)
            ax.set_ylim(0, 1.1 * hi)
            ax.set_ylabel('fitted mass width $\\sigma_{68}$ [MeV]', fontsize=12)
            ax.tick_params(labelbottom=False)
            ax.legend(fontsize=9, frameon=False)
            ax.set_title('%s: does the predicted $\\sigma(m)$ match the mass width?\n'
                         'bins of predicted $\\sigma(m)$ (data/MC raw: cov_*, MC corrected: '
                         'cov_corr_*)' % res['title'], fontsize=10)
            rx.axhline(1, color='black', lw=0.8)
            rx.set_xlim(0, 1.1 * hi)
            rx.set_ylim(0.6, 1.6)
            rx.set_xlabel('mean predicted $\\sigma(m)$ in the bin [MeV]', fontsize=12)
            rx.set_ylabel('width / predicted', fontsize=10)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        # ---- 3. per-bin-fit yields
        for name, o in res['vars'].items():
            e = o['edges']
            nb = len(e) - 1
            c = np.arange(nb)
            yd, ed = o['data_yield'], o['data_err']
            tot = np.nansum(yd)
            if not tot > 0:
                continue
            fd, fe_ = yd / tot, ed / tot
            fsw = o['data_swsum'] / max(o['data_swsum'].sum(), 1e-300)
            fr = o['raw_yield'] / max(o['raw_yield'].sum(), 1e-300)
            fc = o['corr_yield'] / max(o['corr_yield'].sum(), 1e-300)
            fig = plt.figure(figsize=(14, 6.5))
            gs = fig.add_gridspec(2, 2, height_ratios=(3, 1), hspace=0.06, wspace=0.22)
            for col, logy in enumerate((False, True)):
                ax = fig.add_subplot(gs[0, col])
                rx = fig.add_subplot(gs[1, col], sharex=ax)
                ax.tick_params(labelbottom=False)
                ax.errorbar(c, fd, yerr=fe_, fmt='o', ms=4, color=COLORS['data'],
                            label='data: signal yield from a mass fit per bin')
                ax.plot(c, fsw, 'o', ms=5, mfc='none', color='grey',
                        label='data: sum of sWeights (biased if correlated with the mass)')
                ax.stairs(fr, np.arange(nb + 1) - 0.5, color=COLORS['raw'], ls='--', lw=1.5,
                          label=LABELS['raw'])
                ax.stairs(fc, np.arange(nb + 1) - 0.5, color=COLORS['corr'], lw=1.5,
                          label=LABELS['corr'])
                with np.errstate(all='ignore'):
                    for f_, mk, col_ in ((fr, 's', COLORS['raw']), (fc, 'o', COLORS['corr']),
                                         (fsw, 'o', 'grey')):
                        rx.plot(c, f_ / fd, mk, ms=3, color=col_,
                                mfc='none' if col_ == 'grey' else col_)
                    rx.fill_between(c, 1 - fe_ / fd, 1 + fe_ / fd, step='mid', color='grey',
                                    alpha=0.25, lw=0)
                rx.axhline(1, color='black', lw=0.8)
                rx.set_ylim(0.5, 1.5)
                rx.set_ylabel('/ data (fit)', fontsize=10)
                rx.set_xticks(np.arange(nb + 1) - 0.5)
                rx.set_xticklabels([_fmt_edge(x) for x in e], rotation=60, fontsize=8)
                rx.set_xlabel(o['label'] + ('   (both muons)' if o['kind'] == 'mu' else '')
                              + '   [bin edges; last bin = overflow]', fontsize=11)
                ax.set_ylabel('fraction per bin', fontsize=12)
                if logy:
                    pos = np.concatenate([x[np.isfinite(x) & (x > 0)] for x in (fd, fr, fc)])
                    if pos.size:
                        ax.set_yscale('log')
                        ax.set_ylim(0.3 * pos.min(), 5 * pos.max())
                ax.set_title('log scale' if logy else 'linear scale', fontsize=10)
                ax.legend(fontsize=7, frameon=False)
            extra = ''
            if name == CHI2_VAR and 'chi2_k' in res:
                k_ = res['chi2_k']
                extra = '   k: data %.3f (sWeights %.3f), MC raw %.3f, MC corrected %.3f' % (
                    k_['data']['k'], k_['data_sweights']['k'], k_['raw']['k'], k_['corr']['k'])
            fig.suptitle('%s: %s, sPlot-free (mass fit in each bin)%s'
                         % (res['title'], name, extra), fontsize=10)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            bin_fit_pages(pdf, plt, res, R, name, model)
        # ---- 4. MC truth
        if 'truth' in R and 'truth' in res:
            T, t = R['truth'], res['truth']
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            cat_ls = dict(all='-', nokill=':', killed='-.')
            for row, cats in enumerate((('all',), ('nokill', 'killed'))):
                for col, (q, ek, unit) in enumerate((('dm', 'dm_edges', 1e3),
                                                     ('pull', 'pull_edges', 1.))):
                    ax = axs[row, col]
                    e = np.asarray(T[ek]) * unit
                    for cat in cats:
                        for k in ('raw', 'corr'):
                            h = np.asarray(T['%s_%s_%s' % (k, cat, q)], float)
                            if h.sum() <= 0:
                                continue
                            s = t['%s_%s' % (k, cat)]['%s_s68' % q] * unit
                            ax.stairs(h / h.sum(), e, color=COLORS[k],
                                      ls=cat_ls[cat] if row else ('--' if k == 'raw' else '-'),
                                      lw=1.5, label='%s, %s: $\\sigma_{68}$ %.3g%s'
                                      % (LABELS[k], dict(KILL_CATS)[cat], s,
                                         ' MeV' if q == 'dm' else ''))
                    ax.set_yscale('log')
                    ax.set_xlabel('$m - m_{gen}$ [MeV]' if q == 'dm'
                                  else 'mass pull $(m - m_{gen})/\\sigma(m)$', fontsize=12)
                    ax.set_ylabel('fraction', fontsize=12)
                    ax.legend(fontsize=7, frameon=False)
            fig.suptitle('%s: MC truth (m_gen from the gen-matched muons: radiation removed); '
                         'top all events, bottom by hit-kill status' % res['title'], fontsize=10)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--epochs', nargs='+', default=list(EPOCHS), choices=list(EPOCHS))
    p.add_argument('--sample', default='hb', help='MC sample in the file names: hb (default) or bc')
    p.add_argument('--data-dir', default=NTUPLE_DIR,
                   help='replaces <DATA_DIR> in the EPOCHS table (default %(default)s)')
    p.add_argument('--mc-dir', default=NTUPLE_DIR,
                   help='replaces <MC_DIR> in the EPOCHS table (default %(default)s)')
    p.add_argument('--data', nargs='+', default=None,
                   help='data files / globs (one --epochs value only)')
    p.add_argument('--mc', nargs='+', default=None, help='MC files / globs (one --epochs value only)')
    p.add_argument('--selection', default=SELECTION,
                   help='common selection; its mass terms are replaced by the sPlot window')
    p.add_argument('--mc-selection', default=MC_SELECTION, help='extra MC selection (gen matching)')
    p.add_argument('--pu-weight', default='pu_weight_{pu_year}',
                   help='MC pileup weight branch, {pu_year} = the epoch\'s (default %(default)s)')
    p.add_argument('--no-emu-weight', action='store_true',
                   help='do not apply emu_weight to the corrected MC (check only)')
    p.add_argument('--kin-reweight', action='store_true',
                   help='also reweight both MC views in the J/psi (pT, |eta|) to sWeighted data')
    p.add_argument('--no-resolution', action='store_true',
                   help='skip the resolution pdf (per-bin mass fits, sigma(m) calibration, '
                        'MC truth)')
    p.add_argument('--binfit-min', type=float, default=500,
                   help='minimum candidates for a per-bin mass fit (default %(default)s)')
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
    if (a.data or a.mc) and len(a.epochs) != 1:
        p.error('--data / --mc need exactly one --epochs value')
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
        fill = lambda s: s.replace('<DATA_DIR>', a.data_dir or '<DATA_DIR>') \
                          .replace('<MC_DIR>', a.mc_dir or '<MC_DIR>').format(sample=a.sample)
        data_files = expand(a.data or [fill(f) for f in cfg['data']])
        mc_files = expand(a.mc or [fill(f) for f in cfg['mc']])
        hists, info = run_epoch(ep, data_files, mc_files, cfg['pu_year'], a)
        info['table'] = {}
        for mode in ('sw', 'nosw'):
            table = plot_pages(hists[mode], '%s %s' % (ep, a.sample),
                               '%s_%s_%s.pdf' % (a.out, ep, mode), mode)
            print_table('%s (%s), %s' % (ep, a.sample, MODES[mode]['data']), table)
            info['table'][mode] = table
        if 'res' in hists:
            res = analyse_resolution(hists['res'], a, '%s %s' % (ep, a.sample))
            plot_resolution(res, hists['res'], '%s_%s_resolution.pdf' % (a.out, ep))
            print_resolution(res)
            info['resolution'] = res
        report['epochs'][ep] = info
        per_epoch.append(hists)
    if len(per_epoch) > 1:
        allh = combine(per_epoch)
        report['all'] = {}
        for mode in ('sw', 'nosw'):
            table = plot_pages(allh[mode], 'all epochs %s' % a.sample,
                               '%s_all_%s.pdf' % (a.out, mode), mode)
            print_table('all epochs (%s), %s' % (a.sample, MODES[mode]['data']), table)
            report['all'][mode] = table
        if all('res' in h for h in per_epoch):
            R = sum_tree([h['res'] for h in per_epoch])
            res = analyse_resolution(R, a, 'all epochs %s' % a.sample)
            plot_resolution(res, R, '%s_all_resolution.pdf' % a.out)
            print_resolution(res)
            report['all']['resolution'] = res
    with open(a.out + '.json', 'w') as fh:
        json.dump(report, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print('\nwrote %s_<epoch>_{sw,nosw%s}.pdf%s, %s_<epoch>_splot.pdf (mass fit), %s.json'
          % (a.out, '' if a.no_resolution else ',resolution',
             ', %s_all_{sw,nosw%s}.pdf' % (a.out, '' if a.no_resolution else ',resolution')
             if len(per_epoch) > 1 else '', a.out, a.out))


if __name__ == '__main__':
    main()
