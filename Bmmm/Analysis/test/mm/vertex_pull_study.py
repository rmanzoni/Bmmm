#!/usr/bin/env python3
'''
vertex_pull_study.py -- are the vertex and track uncertainties honest, in data
and in MC, and by how much do they differ? (the "k" factor)

A pull is (measured - true) / stated uncertainty. With honest Gaussian errors
its distribution is N(0, 1). Its CORE width (Gaussian fit restricted to the
core, iterated) is the scale factor k by which the stated errors would have to
be multiplied. Tails (wrong hits, non-prompt and combinatorial leftovers) are
not part of k and are kept out of every fit.

WHAT IS MEASURED
  Data has no truth: the J/psi is required to be PROMPT (ct < 0.01 cm, ct
  measured wrt the refit PV) so that its true decay point is the primary
  vertex. The PV is the per-candidate refit pv_bs_* (beamspot constrained,
  the two muons REMOVED, pv_refit_valid == 1), hence independent of the muons
  and of the dimuon vertex: the stated covariance of a difference is the sum.
  The plain offline PV (pv_*) contains the muons and must not be used: the
  difference with the dimuon vertex then has covariance C_SV - C_PV, and the
  pull comes out ~0.87 even with honest errors.

  MC (Hb, non-prompt) uses the generator truth in the same estimator: the true
  displacement (gen J/psi decay vertex - gen PV) is subtracted, so the
  numerator is pure measurement error with the SAME denominator as in data
  ("data-like"). The ct used for the prompt cut and the ct bins is computed
  from that same displacement, in data and MC alike:
      ct = (SV - PV)_T . u_T(J/psi) * m / pT     (MC: truth-subtracted)
  so in MC it is the ct a prompt J/psi with this measurement error would
  have, and the cut shapes MC exactly as it shapes data. In addition, MC has
  the truth-only pulls, which leave the PV out entirely ("truth only").

  Vertex observables (one entry per candidate)
    vtx_pull_x/y/z    (SV - PV)_i / sqrt(C_SV + C_PV)_ii
    vtx_pull_perp     transverse projection perpendicular to the J/psi pT:
                      least sensitive to residual non-prompt J/psi (but not
                      immune: B-J/psi angle x Lxy)
    vtx_pull_par      transverse projection along the J/psi pT: the classic
                      lifetime resolution function; non-prompt leakage goes
                      into its positive tail
    vtx_sqrtchi2_3d   sqrt of the 3-dof chi2 of the difference; k from its
                      median: median / 1.5382 (= sqrt of the chi2(3) median)
    dimu_sqrtchi2     sqrt of the dimuon vertex chi2 (1 dof): the two muons
                      tested against each other, no PV involved;
                      k = median / 0.67449
  Track observables (one entry per muon)
    trk_pull_dxy      dxy wrt the refit PV, / sqrt(sigma_trk^2 + PV term)
    trk_pull_dz       dz  wrt the refit PV, same
      The ntuple stores dxy, dz wrt the offline PV (pv_*); they are moved to
      the refit PV (pv_bs_*) exactly, for straight tracks:
        dxy(Q) = dxy(P) + (Qx-Px) sin(phi) - (Qy-Py) cos(phi)
        dz(Q)  = dz(P)  - (Qz-Pz) + ((Qx-Px) cos(phi) + (Qy-Py) sin(phi)) sinh(eta)
      sigma_trk: sqrt(cov_dxy_dxy) and dz_e (track only; cov_corr_dxy_dxy for
      MC corrected), PV term g^T C_PV g with g the gradient above. In MC the
      reference is the gen muon production vertex G and the PV term is the
      refit-PV measurement error (pv_bs - gen PV), as in data.

  Views
    data         signal region |m - m_J/psi| < 0.05 minus the sidebands
                 0.15 < |m - m_J/psi| < 0.25 scaled by the width ratio (linear
                 background); "data SR only" kept as a check of the subtraction
    MC raw       the emulated MC read through its *_raw branches (vx_raw,
                 vtx_cov_*_raw, dxy_raw, ...), weight pu_weight_<year>
    MC corr      nominal branches (emulated), weight pu x emu_weight
    + truth-only variants of both (MC)

  Differential (every observable in bins of)
    vertex: J/psi pT, J/psi |eta|, dR(mu, mu), refit-PV track count, number of
            muons whose first pixel hit is on BPix L1
    track:  muon pT, muon |eta|, first pixel layer, number of pixel hits
    both:   ct (the non-prompt leakage check: ct < 0 is nearly leakage-free)
  and per data file (era).

  Which numbers to trust (2026 data, pull-study note v2, 15 Oct 2026):
    - the prompt selection keeps >= 8% non-prompt J/psi in data (up to ~30%
      at high pT). They push the SV only FORWARD: the along pull gets a
      one-sided excess, x / y / z are inflated, the across pull and the
      two-muon chi2 are immune.
    - vtx_pull_par_left: the along pull fitted on its LEFT half only (window
      [mu - r sigma, mu + 0.3 sigma], mu free; sigma68 from the left side),
      which non-prompt J/psi cannot reach. Reads ~+0.01 high at 8%
      contamination, +0.02-0.04 at 25% (toys). Its forward excess
      (N+ - N-)/N is reported per bin: a lower limit on the non-prompt share.
    - the clean set: vtx_pull_par_left, vtx_pull_perp, dimu_sqrtchi2.
      x / y / z are kept as cross-checks only.
  Which track errors each sees (straight-track model, note Fig. 13): across
  = d_xy only; along = d_xy + d_sz, the d_sz share growing with |deta|;
  two-muon chi2 = mostly d_sz; the directions (phi, lambda) and q/p are
  invisible to all of them. Per candidate, the d_xy share of the along error
  and of the chi2 is computed from the muon kinematics and errors
  (error_shares) and used as a binning (share_along, pt_x_share) and as a
  profile (mean share per bin), the input of fit_track_scales.py.

CORE FIT
  Gaussian, least squares on the histogram (errors from sum w^2, so it is
  right for weighted MC and for sideband-subtracted data), restricted to
  [mu - r sigma, mu + r sigma], iterated from the median and sigma68 until
  sigma moves by < 1e-4 (r = --core, default 1.0: an unbiased width for a
  Gaussian, and blind to tails beyond ~1.5 sigma). Also reported: sigma68 from
  the quantiles, and the fraction beyond 3 sigma_core.

USAGE (covflow conda env: numpy, uproot, matplotlib, iminuit)
  Everything, sequentially, default files:
    python3 vertex_pull_study.py all --out pulls/vps_2026
  Or one job per file, then plot (fill writes summable histograms):
    python3 vertex_pull_study.py fill --files <data2026B.root> --label data2026B --out pulls/data2026B.npz
    python3 vertex_pull_study.py fill --files <data2026D.root> --label data2026D --out pulls/data2026D.npz
    python3 vertex_pull_study.py fill --files <hb_Summer24_2026.root> --label mc --out pulls/mc.npz
    python3 vertex_pull_study.py plot pulls/data2026B.npz pulls/data2026D.npz pulls/mc.npz --out pulls/vps_2026
  Quick test: add --max-entries 2000000 to fill / all.

OUTPUTS
  <out>.pdf        summary page; per observable the inclusive distributions
                   with the core fits; per observable and binning sigma_core
                   vs bin (data, MC raw, MC corr, truth-only) and data/MC
  <out>_fits.pdf   every per-bin core fit, one panel per bin
  <out>.json       every number
'''

import argparse
import json
import os
import sys
import time

import numpy as np

JPSI = 3.0969
NTUPLE_DIR = '/pnfs/psi.ch/cms/trivcat/store/user/manzoni/rjpsi_run3/covflow_dimuon_ntuples_12oct26'
DEFAULT_FILES = dict(data2026B=NTUPLE_DIR + '/data2026B_PromptReco_v1.root',
                     data2026D=NTUPLE_DIR + '/data2026D_PromptReco_v1.root',
                     mc=NTUPLE_DIR + '/hb_Summer24_2026.root')

# pull histograms: [-8, 8] in steps of 0.025; |pull| (sqrt chi2): [0, 8]
PULL_EDGES = np.linspace(-8, 8, 641)
ABS_EDGES = np.linspace(0, 8, 641)

CHI2_1_MEDIAN_SQRT = 0.674490      # median of |N(0,1)|
CHI2_3_MEDIAN_SQRT = 1.538172      # sqrt(median of chi2(3)) = sqrt(2.36597)

OBS = {
    # name: (level, kind, label)
    'vtx_pull_x':      ('vtx', 'pull', r'$(SV-PV)_x/\sigma$'),
    'vtx_pull_y':      ('vtx', 'pull', r'$(SV-PV)_y/\sigma$'),
    'vtx_pull_z':      ('vtx', 'pull', r'$(SV-PV)_z/\sigma$'),
    'vtx_pull_perp':   ('vtx', 'pull', r'$(SV-PV)_{\perp p_T}/\sigma$'),
    'vtx_pull_par':    ('vtx', 'pull', r'$(SV-PV)_{\parallel p_T}/\sigma$'),
    # derived in combine() from vtx_pull_par, fitted on its left half only
    'vtx_pull_par_left': ('vtx', 'pull', r'$(SV-PV)_{\parallel p_T}/\sigma$, left half'),
    'vtx_sqrtchi2_3d': ('vtx', 'abs3', r'$\sqrt{\chi^2_{3D}}$ of SV - PV'),
    'dimu_sqrtchi2':   ('vtx', 'abs1', r'$\sqrt{\chi^2}$ of the dimuon vertex (1 dof)'),
    'trk_pull_dxy':    ('trk', 'pull', r'muon $d_{xy}$(PV)$/\sigma$'),
    'trk_pull_dz':     ('trk', 'pull', r'muon $d_z$(PV)$/\sigma$'),
}

BINNINGS = {
    # name: (level, edges, label)
    'incl':        ('any', [-1e9, 1e9], 'inclusive'),
    'jpsi_pt':     ('vtx', [0, 8, 10, 12, 15, 20, 25, 35, 1e9], r'J/$\psi$ $p_T$ [GeV]'),
    'jpsi_abseta': ('vtx', [0, 0.6, 1.2, 1.6, 2.0, 2.5], r'J/$\psi$ $|\eta|$'),
    'dr_mumu':     ('vtx', [0, 0.1, 0.15, 0.2, 0.3, 0.45, 0.7, 10], r'$\Delta R(\mu,\mu)$'),
    'pv_ntrk':     ('vtx', [0, 5, 10, 15, 25, 40, 1e9], 'refit-PV tracks'),
    'n_mu_l1':     ('vtx', [-0.5, 0.5, 1.5, 2.5], 'muons with first pixel hit on L1'),
    # ct: the non-prompt leakage test. Prompt J/psi are symmetric around 0,
    # non-prompt only on the positive side: a width that grows with ct > 0 is
    # leakage, the ct < 0 bins are (nearly) free of it. Not meaningful for
    # vtx_pull_par, whose numerator IS ct.
    'ct':          ('both', [-1e9, -0.005, -0.0025, -0.001, 0, 0.001, 0.0025, 0.005, 1e9],
                    r'$ct$ [cm] (J/$\psi$, wrt the refit PV)'),
    'mu_pt':       ('trk', [0, 4, 5, 6, 8, 10, 14, 20, 1e9], r'muon $p_T$ [GeV]'),
    'mu_abseta':   ('trk', [0, 0.5, 0.9, 1.2, 1.6, 2.0, 2.5], r'muon $|\eta|$'),
    'first_layer': ('trk', [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
                    'first pixel layer (0 none, 1-4 L1-L4, 5-7 D1-D3)'),
    'n_pix_hit':   ('trk', [0, 2.5, 3.5, 4.5, 5.5, 99], 'pixel hits'),
    # which track errors the vertex sees (fit_track_scales.py)
    'abs_deta':    ('vtx', [0, 0.1, 0.2, 0.3, 0.45, 0.7, 10], r'$|\Delta\eta(\mu,\mu)|$'),
    'share_along': ('vtx', [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.001],
                    r'share of the along error from $d_{xy}$'),
    # an effect tied to the muon charge shifts the along pull in opposite
    # directions in the two bins (note section 12.5)
    'mu_plus_side': ('vtx', [-1.5, 0, 1.5], r'$\mu^+$ right ($-1$) / left ($+1$) of the J/$\psi$'),
}
# 2D binning J/psi pT x share_along, flattened (index = ipt * NS + ishare);
# level 'vtx2d': no per-bin plots, read by fit_track_scales.py
PT2_EDGES = [0, 10, 15, 25, 1e9]
SH2_EDGES = [0, 0.7, 0.85, 0.95, 1.001]
BINNINGS['pt_x_share'] = ('vtx2d', list(np.arange((len(PT2_EDGES) - 1) * (len(SH2_EDGES) - 1) + 1)
                                        .astype(float)), r'J/$\psi$ $p_T$ $\times$ $d_{xy}$ share')
PROFILES = ('share_along', 'share_chi2', 'pvshare_along', 'pvshare_across')

VTX_COV = ('xx', 'xy', 'xz', 'yy', 'yz', 'zz')


def die(msg):
    sys.exit('[vertex_pull_study] ERROR: ' + msg)


class Progress:
    def __init__(self, total, label):
        self.total, self.label, self.t0, self.last = max(int(total), 1), label, time.time(), 0.

    def update(self, done, extra=''):
        now = time.time()
        if done < self.total and now - self.last < 1.0:
            return
        self.last = now
        f = min(done / self.total, 1.0)
        rate = done / max(now - self.t0, 1e-9)
        eta = (self.total - done) / max(rate, 1e-9)
        sys.stdout.write('\r  [%-30s] %5.1f%%  %-22s %s/%s  %.0f/s  ETA %4.0fs %s   '
                         % ('#' * int(30 * f), 100 * f, self.label[:22], f'{done:,}',
                            f'{self.total:,}', rate, eta, extra))
        sys.stdout.flush()
        if done >= self.total:
            sys.stdout.write('\n')


# ---------------------------------------------------------------------------
# branches
# ---------------------------------------------------------------------------
def needed_branches(is_mc, a):
    b = {'mass', 'charge', 'pv_refit_valid', 'pv_refit_ntrk', 'pt',
         'vx', 'vy', 'vz', 'vtx_chi2', 'pv_x', 'pv_y', 'pv_z',
         'pv_bs_x', 'pv_bs_y', 'pv_bs_z'}
    b |= {'pv_bs_cov_' + e for e in VTX_COV} | {'vtx_cov_' + e for e in VTX_COV}
    for mu in ('mu1', 'mu2'):
        b |= {'%s_%s' % (mu, k) for k in ('pt', 'eta', 'phi', 'id_medium', 'dxy', 'dz', 'dz_e',
                                           'cov_dxy_dxy', 'pix_first_layer', 'n_pix_hit',
                                           'charge')}
    if is_mc:
        b |= {'mass_raw', 'pt_raw', 'vx_raw', 'vy_raw', 'vz_raw',
              'vtx_chi2_raw', 'emu_weight', a.pu_weight, 'gen_pv_x', 'gen_pv_y', 'gen_pv_z',
              'gen_jpsi_vx', 'gen_jpsi_vy', 'gen_jpsi_vz', 'gen_jpsi_nmu'}
        b |= {'vtx_cov_%s_raw' % e for e in VTX_COV}
        for mu in ('mu1', 'mu2'):
            b |= {'%s_%s' % (mu, k) for k in ('pt_raw', 'eta_raw', 'phi_raw', 'dxy_raw', 'dz_raw',
                                               'dz_e_raw', 'cov_corr_dxy_dxy', 'gen_pdgid',
                                               'gen_vx', 'gen_vy', 'gen_vz',
                                               'pix_first_layer_raw', 'n_pix_hit_raw')}
    return b


class View(object):
    '''Branch access for one reading of a chunk: data, MC raw (X_raw where it
    exists, nominal otherwise: PV, gen, ID, cov_* which is raw by
    construction) or MC corrected (nominal).'''
    RAW_MAP = {'cov_dxy_dxy': 'cov_dxy_dxy'}        # raw by construction

    def __init__(self, cols, mode):
        self.c, self.mode = cols, mode

    def __call__(self, name):
        c = self.c
        if self.mode == 'raw':
            base = name[4:] if name[:4] in ('mu1_', 'mu2_') else name
            if base in self.RAW_MAP:
                return c[name]
            if base.startswith('vtx_cov_') and name + '_raw' in c:
                return c[name + '_raw']
            if name + '_raw' in c:
                return c[name + '_raw']
            return c[name]
        if self.mode == 'corr':
            base = name[4:] if name[:4] in ('mu1_', 'mu2_') else name
            if base == 'cov_dxy_dxy':
                return c[name[:4] + 'cov_corr_dxy_dxy']
        return c[name]


# ---------------------------------------------------------------------------
# observables
# ---------------------------------------------------------------------------
def cov3(get, prefix, suffix=''):
    C = np.empty((len(get(prefix + 'xx' + suffix)), 3, 3))
    for k, (i, j) in zip(VTX_COV, ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
        v = np.asarray(get(prefix + k + suffix), np.float64)
        C[:, i, j] = v
        C[:, j, i] = v
    return C


def displacement(v, is_mc):
    '''SV - refit PV (MC: minus the true displacement, i.e. pure measurement
    error, as for a prompt J/psi in data), the J/psi transverse direction from
    the two muons, and ct = (displacement . u_T) m / pT -- in MC therefore the
    ct a PROMPT J/psi with this measurement error would have, so that the
    prompt cut and the ct bins shape MC exactly as they shape data.'''
    f = lambda n: np.asarray(v(n), np.float64)
    sv = np.stack([f('vx'), f('vy'), f('vz')], -1)
    pv = np.stack([f('pv_bs_x'), f('pv_bs_y'), f('pv_bs_z')], -1)
    d = sv - pv
    gsv = None
    if is_mc:
        gsv = np.stack([f('gen_jpsi_vx'), f('gen_jpsi_vy'), f('gen_jpsi_vz')], -1)
        gpv = np.stack([f('gen_pv_x'), f('gen_pv_y'), f('gen_pv_z')], -1)
        d = d - (gsv - gpv)
    px = f('mu1_pt') * np.cos(f('mu1_phi')) + f('mu2_pt') * np.cos(f('mu2_phi'))
    py = f('mu1_pt') * np.sin(f('mu1_phi')) + f('mu2_pt') * np.sin(f('mu2_phi'))
    ptj = np.hypot(px, py)
    with np.errstate(all='ignore'):
        upar = np.stack([px / ptj, py / ptj, np.zeros_like(px)], -1)
        ct = np.einsum('ni,ni->n', d, upar) * JPSI / ptj
    uperp = np.stack([-upar[:, 1], upar[:, 0], np.zeros_like(px)], -1)
    return dict(sv=sv, pv=pv, d=d, gsv=gsv, upar=upar, uperp=uperp, ct=ct)


def vertex_observables(v, is_mc):
    '''Per-candidate vertex pulls for one view. Returns {name: array} for the
    data-like estimator and, in MC, {name + ':truth': array} for truth-only.'''
    f = lambda n: np.asarray(v(n), np.float64)
    D = displacement(v, is_mc)
    sv, d, gsv, upar, uperp = D['sv'], D['d'], D['gsv'], D['upar'], D['uperp']
    Csv = cov3(v, 'vtx_cov_')
    Cpv = cov3(v, 'pv_bs_cov_')

    def pulls(delta, C, tag):
        out = {}
        with np.errstate(all='ignore'):
            for i, ax in enumerate('xyz'):
                out['vtx_pull_' + ax + tag] = delta[:, i] / np.sqrt(C[:, i, i])
            for nm, u in (('par', upar), ('perp', uperp)):
                out['vtx_pull_' + nm + tag] = np.einsum('ni,ni->n', delta, u) / \
                    np.sqrt(np.einsum('ni,nij,nj->n', u, C, u))
            ok = np.all(np.isfinite(C.reshape(len(C), -1)), 1) & np.all(np.isfinite(delta), 1)
            chi2 = np.full(len(C), np.nan)
            if ok.any():
                chi2[ok] = np.einsum('ni,ni->n', delta[ok],
                                     np.linalg.solve(C[ok], delta[ok][..., None])[..., 0])
            out['vtx_sqrtchi2_3d' + tag] = np.sqrt(np.where(chi2 >= 0, chi2, np.nan))
        return out

    res = pulls(d, Csv + Cpv, '')
    res['dimu_sqrtchi2'] = np.sqrt(np.clip(f('vtx_chi2'), 0, None))
    if is_mc:
        res.update(pulls(sv - gsv, Csv, ':truth'))
    return res


def track_observables(v, mu, is_mc):
    '''Per-muon dxy, dz pulls wrt the refit PV (data-like) and, in MC, wrt the
    gen production vertex with the track error only (truth).'''
    f = lambda n: np.asarray(v(n), np.float64)
    phi, eta = f(mu + '_phi'), f(mu + '_eta')
    s, c, sh = np.sin(phi), np.cos(phi), np.sinh(eta)
    P = [f('pv_x'), f('pv_y'), f('pv_z')]           # dxy, dz are stored wrt this
    Q = [f('pv_bs_x'), f('pv_bs_y'), f('pv_bs_z')]  # the refit PV
    dxyP, dzP = f(mu + '_dxy'), f(mu + '_dz')
    with np.errstate(all='ignore'):
        s_dxy = np.sqrt(f(mu + '_cov_dxy_dxy'))
    s_dz = f(mu + '_dz_e')
    Cq = cov3(v, 'pv_bs_cov_')
    g_dxy = np.stack([s, -c, np.zeros_like(s)], -1)
    g_dz = np.stack([c * sh, s * sh, -np.ones_like(s)], -1)
    v_dxy = np.einsum('ni,nij,nj->n', g_dxy, Cq, g_dxy)
    v_dz = np.einsum('ni,nij,nj->n', g_dz, Cq, g_dz)

    def move(dxy0, dz0, frm, to):
        dx, dy, dzz = to[0] - frm[0], to[1] - frm[1], to[2] - frm[2]
        return dxy0 + dx * s - dy * c, dz0 - dzz + (dx * c + dy * s) * sh

    out = {}
    with np.errstate(all='ignore'):
        if not is_mc:
            dxyQ, dzQ = move(dxyP, dzP, P, Q)
            out['trk_pull_dxy'] = dxyQ / np.sqrt(s_dxy ** 2 + v_dxy)
            out['trk_pull_dz'] = dzQ / np.sqrt(s_dz ** 2 + v_dz)
        else:
            G = [f(mu + '_gen_vx'), f(mu + '_gen_vy'), f(mu + '_gen_vz')]
            T = [f('gen_pv_x'), f('gen_pv_y'), f('gen_pv_z')]
            dxyG, dzG = move(dxyP, dzP, P, G)          # wrt the true origin: pure error
            # data-like: + the refit-PV measurement error, projected
            dxyD, dzD = move(dxyG, dzG, T, Q)
            out['trk_pull_dxy'] = dxyD / np.sqrt(s_dxy ** 2 + v_dxy)
            out['trk_pull_dz'] = dzD / np.sqrt(s_dz ** 2 + v_dz)
            out['trk_pull_dxy:truth'] = dxyG / s_dxy
            out['trk_pull_dz:truth'] = dzG / s_dz
    return out


def vertex_binvars(v):
    f = lambda n: np.asarray(v(n), np.float64)
    px = f('mu1_pt') * np.cos(f('mu1_phi')) + f('mu2_pt') * np.cos(f('mu2_phi'))
    py = f('mu1_pt') * np.sin(f('mu1_phi')) + f('mu2_pt') * np.sin(f('mu2_phi'))
    pz = f('mu1_pt') * np.sinh(f('mu1_eta')) + f('mu2_pt') * np.sinh(f('mu2_eta'))
    pt = np.hypot(px, py)
    with np.errstate(all='ignore'):
        eta = np.arcsinh(pz / pt)
    dphi = np.angle(np.exp(1j * (f('mu1_phi') - f('mu2_phi'))))
    dr = np.hypot(f('mu1_eta') - f('mu2_eta'), dphi)
    nl1 = (f('mu1_pix_first_layer') == 1).astype(float) + (f('mu2_pix_first_layer') == 1)
    sh = error_shares(v)
    # side of the mu+: sign of (u_J/psi x p_mu+)_z
    q1 = f('mu1_charge')
    pxp = np.where(q1 > 0, f('mu1_pt') * np.cos(f('mu1_phi')), f('mu2_pt') * np.cos(f('mu2_phi')))
    pyp = np.where(q1 > 0, f('mu1_pt') * np.sin(f('mu1_phi')), f('mu2_pt') * np.sin(f('mu2_phi')))
    side = np.sign(px * pyp - py * pxp)
    ns = len(SH2_EDGES) - 1
    ipt = np.clip(np.searchsorted(PT2_EDGES, pt, side='right') - 1, 0, len(PT2_EDGES) - 2)
    ish = np.clip(np.searchsorted(SH2_EDGES, sh['share_along'], side='right') - 1, 0, ns - 1)
    ptsh = np.where(np.isfinite(sh['share_along']) & np.isfinite(pt), ipt * ns + ish + 0.5, np.nan)
    return dict(incl=np.zeros(len(px)), jpsi_pt=pt, jpsi_abseta=np.abs(eta), dr_mumu=dr,
                pv_ntrk=f('pv_refit_ntrk'), n_mu_l1=nl1, ct=cand_ct(v),
                abs_deta=np.abs(f('mu1_eta') - f('mu2_eta')), share_along=sh['share_along'],
                mu_plus_side=side, pt_x_share=ptsh), sh


def pv_shares(v, is_mc):
    '''Share of the stated SV - PV variance that comes from the PV, along and
    across the J/psi flight direction. Across the flight the dimuon vertex is
    known to ~10-20 um, comparable with the beamspot-constrained PV (~10 um):
    there the PV is a sizeable part of the pull's denominator, and dilutes any
    understatement of the SV errors. Along the flight (~80 um) it is ~1%.'''
    D = displacement(v, is_mc)
    Csv = cov3(v, 'vtx_cov_')
    Cpv = cov3(v, 'pv_bs_cov_')
    out = {}
    for nm, u in (('along', D['upar']), ('across', D['uperp'])):
        with np.errstate(all='ignore'):
            a = np.einsum('ni,nij,nj->n', u, Cpv, u)
            b = np.einsum('ni,nij,nj->n', u, Csv, u)
            out['pvshare_' + nm] = a / (a + b)
    return out


def error_shares(v):
    '''Per candidate: the share of the vertex error ALONG the J/psi flight
    direction, and of the two-muon distance (the chi2), that comes from the
    muons' transverse position errors d_xy rather than from d_sz.

    Model (pull-study note, section 12.3 and Fig. 13): two straight tracks
    through a common point near the beam line; the vertex is the weighted
    meeting point, each track contributing its offset in the plane
    perpendicular to it with weights 1/sigma(d_xy)^2 (transverse direction
    e1) and 1/sigma(d_sz)^2 (e2 = u x e1). Then, exactly for this model,
        d v / d dxy_i = A^-1 e1_i / sigma_dxy_i^2,
        d v / d dsz_i = A^-1 e2_i / sigma_dsz_i^2,   A = sum_i E_i^T W_i E_i
    and the two-muon distance g = (P2 - P1) . n, n = u1 x u2 / |u1 x u2|,
        d g / d dxy_i = -+ n . e1_i,   d g / d dsz_i = -+ n_z / cos(lambda_i).
    The direction errors (phi, lambda) do not enter for a vertex at the
    track reference point. sigma(d_sz) = dz_e cos(lambda).'''
    f = lambda n: np.asarray(v(n), np.float64)
    E1, E2, Wd, Wz, U = [], [], [], [], []
    for mu in ('mu1', 'mu2'):
        phi, eta = f(mu + '_phi'), f(mu + '_eta')
        lam = np.arctan(np.sinh(eta))
        cl = np.cos(lam)
        u = np.stack([cl * np.cos(phi), cl * np.sin(phi), np.sin(lam)], -1)
        e1 = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], -1)
        e2 = np.cross(u, e1)
        with np.errstate(all='ignore'):
            sd2 = f(mu + '_cov_dxy_dxy')
            sz2 = (f(mu + '_dz_e') * cl) ** 2
        E1.append(e1); E2.append(e2); Wd.append(1. / sd2); Wz.append(1. / sz2); U.append(u)
    A = sum(Wd[i][:, None, None] * E1[i][:, :, None] * E1[i][:, None, :] +
            Wz[i][:, None, None] * E2[i][:, :, None] * E2[i][:, None, :] for i in range(2))
    ok = np.all(np.isfinite(A.reshape(len(A), -1)), 1)
    Ainv = np.full_like(A, np.nan)
    if ok.any():
        Ainv[ok] = np.linalg.inv(A[ok])
    pxj = U[0][:, :2] + U[1][:, :2]
    uJ = np.concatenate([pxj / np.linalg.norm(pxj, axis=1, keepdims=True),
                         np.zeros((len(A), 1))], 1)
    vxy = vsz = 0.
    for i in range(2):
        gx = np.einsum('ni,nij,nj->n', uJ, Ainv, E1[i]) * Wd[i]
        gz = np.einsum('ni,nij,nj->n', uJ, Ainv, E2[i]) * Wz[i]
        vxy = vxy + gx ** 2 / Wd[i]
        vsz = vsz + gz ** 2 / Wz[i]
    n = np.cross(U[0], U[1])
    with np.errstate(all='ignore'):
        n = n / np.linalg.norm(n, axis=1, keepdims=True)
        cxy = sum((np.einsum('ni,ni->n', n, E1[i])) ** 2 / Wd[i] for i in range(2))
        csz = sum((n[:, 2] / np.cos(np.arctan(U[i][:, 2] / np.hypot(U[i][:, 0], U[i][:, 1]))))
                  ** 2 / Wz[i] for i in range(2))
        return dict(share_along=vxy / (vxy + vsz), share_chi2=cxy / (cxy + csz))


def cand_ct(v):
    return displacement(v, 'gen_pv_x' in v.c)['ct']


def track_binvars(v, mu):
    f = lambda n: np.asarray(v(n), np.float64)
    return dict(incl=np.zeros(len(f(mu + '_pt'))), mu_pt=f(mu + '_pt'),
                mu_abseta=np.abs(f(mu + '_eta')), first_layer=f(mu + '_pix_first_layer'),
                n_pix_hit=f(mu + '_n_pix_hit'), ct=cand_ct(v))


# ---------------------------------------------------------------------------
# histogram store: key (obs, binning, view) -> [sum w, sum w^2], (nbins, npull)
# ---------------------------------------------------------------------------
class Store(object):
    def __init__(self):
        self.h = {}
        self.p = {}         # (quantity, binning, view) -> [sum w, sum w x] per bin

    def fill_prof(self, binvars, view, quants, w):
        for bname, bv in binvars.items():
            be = np.asarray(BINNINGS[bname][1], float)
            nb = len(be) - 1
            ib = np.clip(np.searchsorted(be, bv, side='right') - 1, 0, nb - 1)
            for q, x in quants.items():
                ok = np.isfinite(bv) & (bv >= be[0]) & (bv < be[-1]) & np.isfinite(x) & np.isfinite(w)
                key = (q, bname, view)
                if key not in self.p:
                    self.p[key] = [np.zeros(nb), np.zeros(nb)]
                self.p[key][0] += np.bincount(ib[ok], weights=w[ok], minlength=nb)
                self.p[key][1] += np.bincount(ib[ok], weights=(w * x)[ok], minlength=nb)

    def fill(self, obs, binvars, view, x, w):
        kind = OBS[obs.split(':')[0]][1]
        edges = PULL_EDGES if kind == 'pull' else ABS_EDGES
        npb = len(edges) - 1
        ok = np.isfinite(x) & np.isfinite(w)
        ip = np.clip(((x - edges[0]) / (edges[1] - edges[0])).astype(np.int64), 0, npb - 1)
        for bname, bv in binvars.items():
            be = np.asarray(BINNINGS[bname][1], float)
            nb = len(be) - 1
            okb = ok & np.isfinite(bv) & (bv >= be[0]) & (bv < be[-1])
            ib = np.clip(np.searchsorted(be, bv, side='right') - 1, 0, nb - 1)
            flat = (ib * npb + ip)[okb]
            ww = w[okb]
            key = (obs, bname, view)
            if key not in self.h:
                self.h[key] = [np.zeros(nb * npb), np.zeros(nb * npb)]
            self.h[key][0] += np.bincount(flat, weights=ww, minlength=nb * npb)
            self.h[key][1] += np.bincount(flat, weights=ww * ww, minlength=nb * npb)

    def save(self, path, meta):
        arr = {}
        for (o, b, v), (w, w2) in self.h.items():
            arr['w|%s|%s|%s' % (o, b, v)] = w
            arr['w2|%s|%s|%s' % (o, b, v)] = w2
        for (q, b, v), (sw, swx) in self.p.items():
            arr['p|%s|%s|%s' % (q, b, v)] = np.stack([sw, swx])
        arr['meta'] = np.array(json.dumps(meta))
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        np.savez_compressed(path, **arr)


def load_store(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z['meta']))
    h = {}
    for k in z.files:
        if k.startswith('w|'):
            _, o, b, v = k.split('|')
            kind = OBS[o.split(':')[0]][1]
            npb = len((PULL_EDGES if kind == 'pull' else ABS_EDGES)) - 1
            nb = len(BINNINGS[b][1]) - 1
            h[(o, b, v)] = (z[k].reshape(nb, npb), z['w2|%s|%s|%s' % (o, b, v)].reshape(nb, npb))
        elif k.startswith('p|'):
            _, q, b, v = k.split('|')
            h[('prof:' + q, b, v)] = (z[k][0], z[k][1])
    return meta, h


# ---------------------------------------------------------------------------
# fill
# ---------------------------------------------------------------------------
def fill(files, label, out, a):
    import uproot
    with uproot.open(files[0]) as fh:
        have = set(fh['tree'].keys())
    is_mc = 'gen_pv_x' in have if a.kind == 'auto' else a.kind == 'mc'
    need = needed_branches(is_mc, a)
    missing = sorted(n for n in need if n not in have)
    hard = [n for n in missing if not n.startswith(('gen_jpsi', 'mu1_gen_v', 'mu2_gen_v'))]
    if hard:
        die('%s: branches missing: %s' % (files[0], ', '.join(hard)))
    if missing:
        die('%s: MC truth branches missing (%s): ntuples older than Bmmm 0d2d5e2?'
            % (files[0], ', '.join(missing)))
    total = 0
    for fn in files:
        with uproot.open(fn) as fh:
            total += fh['tree'].num_entries
    if a.max_entries:
        total = min(total, a.max_entries)
    print('=== fill %s (%s): %d file(s), %s entries' % (label, 'MC' if is_mc else 'data',
                                                     len(files), f'{total:,}'))
    store = Store()
    counts = {}
    prog = Progress(total, label)
    done = 0
    for chunk in uproot.iterate([{fn: 'tree'} for fn in files], expressions=sorted(need),
                                library='np', step_size=a.step):
        n = len(chunk['mass'])
        if a.max_entries and done + n > a.max_entries:
            n = a.max_entries - done
            chunk = {k: v[:n] for k, v in chunk.items()}
        fill_chunk(chunk, is_mc, store, counts, a)
        done += n
        prog.update(done, 'selected %s' % f'{counts.get("sel", 0):,}')
        if a.max_entries and done >= a.max_entries:
            break
    prog.update(total)
    meta = dict(label=label, is_mc=is_mc, files=files, entries=done, counts=counts,
                args=dict(sr=a.sr, sb=a.sb, ct_max=a.ct_max, pu_weight=a.pu_weight))
    store.save(out, meta)
    print('  selected: %s   -> %s' % (', '.join('%s %s' % (k, f'{int(v):,}')
                                                for k, v in counts.items()), out))


def base_selection(v, is_mc, a, mass_name):
    f = lambda n: np.asarray(v(n), np.float64)
    sel = (f('mu1_id_medium') > 0.5) & (f('mu2_id_medium') > 0.5) & (f('charge') == 0) & \
        (f('pv_refit_valid') > 0.5)
    if a.ct_max is not None:
        with np.errstate(all='ignore'):
            sel &= displacement(v, is_mc)['ct'] < a.ct_max
    dm = np.abs(f(mass_name) - JPSI)
    if is_mc:
        sel &= (np.abs(f('mu1_gen_pdgid')) == 13) & (np.abs(f('mu2_gen_pdgid')) == 13) & \
            (f('gen_jpsi_nmu') > 1.5)
    return sel, dm


def fill_chunk(ch, is_mc, store, counts, a):
    ch = {k: np.asarray(v) for k, v in ch.items()}
    if not is_mc:
        v = View(ch, 'data')
        sel, dm = base_selection(v, False, a, 'mass')
        regions = dict(SR=sel & (dm < a.sr), SB=sel & (dm > a.sb[0]) & (dm < a.sb[1]))
        for reg, m in regions.items():
            counts['data_' + reg] = counts.get('data_' + reg, 0) + int(m.sum())
            if not m.any():
                continue
            sub = View({k: x[m] for k, x in ch.items()}, 'data')
            w = np.ones(int(m.sum()))
            fill_view(sub, False, 'data_' + reg, w, store)
        counts['sel'] = counts.get('data_SR', 0)
        return
    for mode, mass_name in (('raw', 'mass_raw'), ('corr', 'mass')):
        v = View(ch, mode)
        sel, dm = base_selection(v, True, a, mass_name)
        m = sel & (dm < a.sr)
        counts['mc_' + mode] = counts.get('mc_' + mode, 0) + int(m.sum())
        if not m.any():
            continue
        sub = View({k: x[m] for k, x in ch.items()}, mode)
        w = np.asarray(ch[a.pu_weight][m], np.float64)
        if mode == 'corr':
            w = w * np.asarray(ch['emu_weight'][m], np.float64)
        fill_view(sub, True, 'mc_' + mode, w, store)
    counts['sel'] = counts.get('mc_corr', 0)


def fill_view(v, is_mc, view, w, store):
    vo = vertex_observables(v, is_mc)
    vb, sh = vertex_binvars(v)
    for name, x in vo.items():
        base, _, tag = name.partition(':')
        store.fill(base, vb, view + (':truth' if tag else ''), x, w)
    sh = dict(sh, **pv_shares(v, is_mc))
    store.fill_prof(vb, view, {q: sh[q] for q in PROFILES}, w)
    for mu in ('mu1', 'mu2'):
        to = track_observables(v, mu, is_mc)
        tb = track_binvars(v, mu)
        for name, x in to.items():
            base, _, tag = name.partition(':')
            store.fill(base, tb, view + (':truth' if tag else ''), x, w)


# ---------------------------------------------------------------------------
# core fit
# ---------------------------------------------------------------------------
def quantile(h, edges, q):
    c = np.concatenate([[0.], np.cumsum(np.clip(h, 0, None))])
    if c[-1] <= 0:
        return np.nan
    return float(np.interp(q * c[-1], c, edges))


def core_fit(h, w2, edges, kind, r=1.0, min_core=100.):
    '''Iterated Gaussian fit to the core of a (possibly weighted / subtracted)
    histogram. For |pull| histograms (kind abs1 / abs3) the width is the robust
    median estimate instead (no Gaussian there).'''
    out = dict(n=float(np.sum(h)), sigma=np.nan, sigma_err=np.nan, mu=np.nan, mu_err=np.nan,
               sigma68=np.nan, tail3=np.nan, ok=False)
    if out['n'] <= 0:
        return out
    if kind in ('abs1', 'abs3'):
        med = quantile(h, edges, 0.5)
        ref = CHI2_1_MEDIAN_SQRT if kind == 'abs1' else CHI2_3_MEDIAN_SQRT
        # error of the median: 1 / (2 f(med) sqrt(N_eff)), f from the histogram
        neff = np.sum(h) ** 2 / max(np.sum(w2), 1e-300)
        i = np.clip(np.searchsorted(edges, med) - 1, 0, len(h) - 1)
        lo, hi = max(i - 4, 0), min(i + 5, len(h))
        dens = np.sum(h[lo:hi]) / max(np.sum(h), 1e-300) / (edges[hi] - edges[lo])
        err = 1. / (2 * dens * np.sqrt(neff)) if dens > 0 else np.nan
        out.update(sigma=med / ref, sigma_err=err / ref, mu=med, sigma68=med / ref,
                   ok=bool(neff > min_core))
        return out
    c = 0.5 * (edges[1:] + edges[:-1])
    bw = edges[1] - edges[0]
    q16, q50, q84 = (quantile(h, edges, q) for q in (0.158655, 0.5, 0.841345))
    out['sigma68'] = 0.5 * (q84 - q16)
    mu, sg = q50, out['sigma68']
    if not (np.isfinite(sg) and sg > 0):
        return out
    err = np.sqrt(np.where(w2 > 0, w2, 1.0))
    from iminuit import Minuit
    from iminuit.cost import LeastSquares

    def model(x, A, m, s):
        return A * bw / (np.sqrt(2 * np.pi) * s) * np.exp(-0.5 * ((x - m) / s) ** 2)

    res = None
    for _ in range(30):
        win = (c > mu - r * sg) & (c < mu + r * sg)
        if win.sum() < 5 or h[win].sum() < min_core:
            return out
        lsq = LeastSquares(c[win], h[win], err[win], model)
        A0 = max(float(np.sum(h[win])) / 0.6827 * (1.0 if r == 1.0 else 1.0), 1.0)
        mi = Minuit(lsq, A=A0, m=mu, s=sg)
        mi.limits['s'] = (1e-3, 20)
        mi.migrad()
        if not mi.valid:
            mi.migrad()
        new_mu, new_sg = float(mi.values['m']), float(mi.values['s'])
        conv = abs(new_sg - sg) / sg < 1e-4 and abs(new_mu - mu) < 1e-4
        mu, sg, res = new_mu, new_sg, mi
        if conv:
            break
    if res is None:
        return out
    beyond = (c < mu - 3 * sg) | (c > mu + 3 * sg)
    out.update(mu=mu, mu_err=float(res.errors['m']), sigma=sg, sigma_err=float(res.errors['s']),
               tail3=float(np.sum(h[beyond]) / max(np.sum(h), 1e-300)), ok=bool(res.valid),
               A=float(res.values['A']), chi2=float(res.fval),
               ndf=int(((c > mu - r * sg) & (c < mu + r * sg)).sum() - 3))
    return out


def left_half_fit(h, w2, edges, r=1.0, up=0.3, min_core=200.):
    '''The along pull without the non-prompt J/psi, which only reach its right
    (positive) side: a Gaussian fitted in [mu - r sigma, mu + up sigma], mean
    free, iterated from the full core fit; sigma68 from the left side alone,
    F(mu - s68) = 0.3173 F(mu); tail3 = left-side fraction beyond 3 sigma,
    doubled. Also excess = (N(>0) - N(<0)) / N: a lower limit on the
    non-prompt share. On toys with 8% (25%) contamination the width reads
    ~+0.01 (+0.02-0.04) high.'''
    out = core_fit(h, w2, edges, 'pull', r, min_core)
    c = 0.5 * (edges[1:] + edges[:-1])
    n = float(np.sum(h))
    out['excess'] = float((h[c > 0].sum() - h[c < 0].sum()) / n) if n > 0 else np.nan
    if not out['ok']:
        return out
    bw = edges[1] - edges[0]
    mu, sg = out['mu'], out['sigma']
    err = np.sqrt(np.where(w2 > 0, w2, 1.0))
    from iminuit import Minuit
    from iminuit.cost import LeastSquares

    def model(x, A, m, s):
        return A * bw / (np.sqrt(2 * np.pi) * s) * np.exp(-0.5 * ((x - m) / s) ** 2)

    res = None
    for _ in range(40):
        win = (c > mu - r * sg) & (c < mu + up * sg)
        if win.sum() < 6 or h[win].sum() < min_core:
            out['ok'] = False
            return out
        mi = Minuit(LeastSquares(c[win], h[win], err[win], model), A=n, m=mu, s=sg)
        mi.limits['s'] = (0.05, 20)
        mi.migrad()
        nm, ns_ = float(mi.values['m']), float(mi.values['s'])
        conv = abs(ns_ - sg) < 1e-5 and abs(nm - mu) < 1e-5
        mu, sg, res = nm, ns_, mi
        if conv:
            break
    cdf = np.concatenate([[0.], np.cumsum(np.clip(h, 0, None))])
    F = lambda x: np.interp(x, edges, cdf)
    xs = np.linspace(mu - 6, mu, 6001)
    s68 = mu - float(np.interp(0.317311 * F(mu), F(xs), xs))
    out.update(mu=mu, mu_err=float(res.errors['m']), sigma=sg, sigma_err=float(res.errors['s']),
               sigma68=s68, tail3=float(2 * F(mu - 3 * sg) / max(n, 1e-300)),
               ok=bool(res.valid), A=float(res.values['A']), chi2=float(res.fval),
               ndf=int(((c > mu - r * sg) & (c < mu + up * sg)).sum() - 3))
    return out


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------
VIEW_STYLE = {
    'data':           dict(color='black', marker='o', ls='none', label='data (SR - SB)'),
    'data_SR':        dict(color='grey', marker='o', ls='none', mfc='none', label='data, SR only'),
    'mc_raw':         dict(color='#2f6eba', marker='s', ls='-', label='MC raw'),
    'mc_corr':        dict(color='#d1495b', marker='D', ls='-', label='MC corrected'),
    'mc_raw:truth':   dict(color='#2f6eba', marker='s', ls=':', mfc='none', label='MC raw, truth only'),
    'mc_corr:truth':  dict(color='#d1495b', marker='D', ls=':', mfc='none', label='MC corr, truth only'),
}


def combine(stores, a):
    '''Sum the fill outputs: data files into "data" (SR - f SB) and "data_SR",
    and per era "data@<label>"; MC as is.'''
    f_sb = (2 * a.sr) / (2 * (a.sb[1] - a.sb[0]))
    H = {}
    eras = []

    def add(key, w, w2):
        if key in H:
            H[key] = (H[key][0] + w, H[key][1] + w2)
        else:
            H[key] = (w.copy(), w2.copy())

    for meta, h in stores:
        lab = meta['label']
        if not meta['is_mc']:
            eras.append(lab)
            for (o, b, v), (w, w2) in h.items():
                if v == 'data_SR':
                    add((o, b, 'data_SR'), w, w2)
                    sb = h.get((o, b, 'data_SB'), (np.zeros_like(w), np.zeros_like(w2)))
                    if o.startswith('prof:'):       # (sum w, sum w x): both subtracted
                        sub = (w - f_sb * sb[0], w2 - f_sb * sb[1])
                    else:
                        sub = (w - f_sb * sb[0], w2 + f_sb ** 2 * sb[1])
                    add((o, b, 'data'), *sub)
                    add((o, b, 'data@' + lab), *sub)
        else:
            for k, (w, w2) in h.items():
                add(k, w, w2)
    # the along pull, to be fitted on its left half (analyse)
    for (o, b, v), wv in list(H.items()):
        if o == 'vtx_pull_par':
            H[('vtx_pull_par_left', b, v)] = wv
    # the ct < 0 half: prompt J/psi only (non-prompt leakage lives at ct > 0)
    be = np.asarray(BINNINGS['ct'][1])
    neg = np.nonzero(be[1:] <= 0)[0]
    for (o, b, v), (w, w2) in list(H.items()):
        if b == 'ct':
            H[(o, 'ct<0', v)] = (w[neg].sum(0, keepdims=True), w2[neg].sum(0, keepdims=True))
    return H, eras, f_sb


def analyse(H, a):
    R = {}
    for (o, b, v), (w, w2) in H.items():
        if o.startswith('prof:'):
            continue
        kind = OBS[o][1]
        edges = PULL_EDGES if kind == 'pull' else ABS_EDGES
        if o == 'vtx_pull_par_left':
            R[(o, b, v)] = [left_half_fit(w[i], w2[i], edges, a.core) for i in range(len(w))]
        else:
            R[(o, b, v)] = [core_fit(w[i], w2[i], edges, kind, a.core) for i in range(len(w))]
    return R


def profiles(H):
    '''mean d_xy shares per bin: {(quantity, binning, view): array}'''
    P = {}
    for (o, b, v), (sw, swx) in H.items():
        if o.startswith('prof:'):
            with np.errstate(all='ignore'):
                P[(o[5:], b, v)] = swx / sw
    return P


def views_present(R, o, b):
    order = ['data', 'data_SR', 'mc_raw', 'mc_corr', 'mc_raw:truth', 'mc_corr:truth']
    return [v for v in order if (o, b, v) in R]


def summary_lines(R, eras, meta_all, f_sb, a):
    L = ['vertex_pull_study: core width of the pulls (Gaussian, iterated, |x - mu| < %.2f sigma); '
         'for sqrt(chi2): k from the median' % a.core,
         'selection: medium ID, opposite charge, pv_refit_valid, ct < %s cm; data SR |dm| < %.3f, '
         'SB %.3f-%.3f GeV (scale %.3f)' % ('/'.join(sorted({str(m['args']['ct_max'])
                                                            for m in meta_all})),
                                            a.sr, a.sb[0], a.sb[1], f_sb), '']
    for m in meta_all:
        L.append('  %-12s %s  entries %s  selected %s' % (m['label'], 'MC  ' if m['is_mc'] else 'data',
                                                         f'{m["entries"]:,}',
                                                         ', '.join('%s %s' % (k, f'{int(x):,}')
                                                                   for k, x in m['counts'].items()
                                                                   if k != 'sel')))
    for sel_name, sel_lab in (('incl', 'all ct < ct_max'),
                              ('ct<0', 'ct < 0 only: free of non-prompt leakage, half the '
                                       'statistics -- compare with the line above')):
        L += ['', sel_lab, '%-16s %15s %15s %15s %15s %15s %9s %9s' % (
              'observable', 'data', 'MC raw', 'MC corr', 'raw truth', 'corr truth', 'data/raw',
              'data/corr')]
        for o in OBS:
            if sel_name == 'ct<0' and o == 'vtx_pull_par_left':
                continue                    # ct < 0 IS the left half: nothing left to fit
            def g(v, sel_name=sel_name):
                r = R.get((o, sel_name, v))
                return r[0] if r else None
            cells = []
            for v in ('data', 'mc_raw', 'mc_corr', 'mc_raw:truth', 'mc_corr:truth'):
                x = g(v)
                cells.append('%.4f+-%.4f' % (x['sigma'], x['sigma_err']) if x and np.isfinite(x['sigma'])
                             else '-')
            d, r_, c_ = g('data'), g('mc_raw'), g('mc_corr')
            rat = lambda p, q: ('%.4f' % (p['sigma'] / q['sigma'])) if p and q and \
                np.isfinite(p['sigma']) and np.isfinite(q['sigma']) else '-'
            L.append('%-16s %15s %15s %15s %15s %15s %9s %9s' % (o, *cells, rat(d, r_), rat(d, c_))
                     + ('   (numerator is ct: biased here)' if sel_name == 'ct<0' and
                        o == 'vtx_pull_par' else ''))
    L += ['', 'per era (data, inclusive core width):']
    for o in OBS:
        L.append('  %-16s ' % o + '  '.join(
            '%s %.4f' % (e, R[(o, 'incl', 'data@' + e)][0]['sigma'])
            for e in eras if (o, 'incl', 'data@' + e) in R))
    # the clean set and the two tests of the note's section 12.5
    L += ['', 'CLEAN SET (immune to non-prompt J/psi): core / sigma68, data / MC corr',
          '%-22s %17s %17s %17s %15s' % ('', 'data', 'MC raw', 'MC corr', 'data/corr')]
    for o, lab in (('vtx_pull_par_left', 'along (left half)'), ('vtx_pull_perp', 'across'),
                   ('dimu_sqrtchi2', 'two-muon chi2, k')):
        rr = [R.get((o, 'incl', v), [None])[0] for v in ('data', 'mc_raw', 'mc_corr')]
        cell = lambda r: '%.4f / %.4f' % (r['sigma'], r['sigma68']) if r and np.isfinite(r['sigma']) \
            else '-'
        rat = ('%.4f / %.4f' % (rr[0]['sigma'] / rr[2]['sigma'], rr[0]['sigma68'] / rr[2]['sigma68'])
               if rr[0] and rr[2] and np.isfinite(rr[0]['sigma']) and np.isfinite(rr[2]['sigma'])
               else '-')
        L.append('%-22s %17s %17s %17s %15s' % (lab, *[cell(r) for r in rr], rat))
    ex = R.get(('vtx_pull_par_left', 'incl', 'data'))
    if ex:
        L.append('forward excess (N+ - N-)/N of the along pull, data: %.4f  (lower limit on the '
                 'non-prompt share)' % ex[0]['excess'])
    cs = R.get(('vtx_pull_par_left', 'mu_plus_side', 'data'))
    if cs and all(r['ok'] for r in cs):
        d = cs[1]['mu'] - cs[0]['mu']
        e = np.hypot(cs[0]['mu_err'], cs[1]['mu_err'])
        L.append('charge test: along-pull centre, mu+ left minus mu+ right: %+.4f +- %.4f '
                 '(%.1f sigma; 0 if nothing is tied to the muon charge)' % (d, e, d / e))
    L += ['', 'reading: honest errors give 1 (pulls) / k = 1 (sqrt chi2). The factors to apply to',
          'the track errors come from fit_track_scales.py, not from these numbers directly.']
    return L


def plot(paths, out, a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    stores = [load_store(p) for p in paths]
    for m, _ in stores:
        if m['args']['sr'] != a.sr or m['args']['sb'] != list(a.sb):
            print('WARNING: %s was filled with SR %s SB %s, plotting with SR %s SB %s'
                  % (m['label'], m['args']['sr'], m['args']['sb'], a.sr, list(a.sb)))
    H, eras, f_sb = combine(stores, a)
    print('fitting %d histogram sets ...' % len(H))
    R = analyse(H, a)
    P = profiles(H)
    lines = summary_lines(R, eras, [m for m, _ in stores], f_sb, a)
    print('\n'.join(lines))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
    with PdfPages(out + '.pdf') as pdf:
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(0.02, 0.98, '\n'.join(lines), family='monospace', fontsize=6.8, va='top')
        pdf.savefig(fig)
        plt.close(fig)
        # inclusive distributions with the core fits
        for o, (lvl, kind, lab) in OBS.items():
            edges = PULL_EDGES if kind == 'pull' else ABS_EDGES
            c = 0.5 * (edges[1:] + edges[:-1])
            fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
            for ax, logy in zip(axs, (False, True)):
                for v in views_present(R, o, 'incl'):
                    w = H[(o, 'incl', v)][0][0]
                    tot = w.sum()
                    if tot <= 0:
                        continue
                    st = VIEW_STYLE[v]
                    r = R[(o, 'incl', v)][0]
                    y = w / tot
                    if v.startswith('data'):
                        ax.plot(c, y, st['marker'], ms=2, color=st['color'], mfc=st.get('mfc'),
                                label='%s: %.4f' % (st['label'], r['sigma']))
                    else:
                        ax.stairs(y, edges, color=st['color'], ls=st['ls'], lw=1.2,
                                  label='%s: %.4f' % (st['label'], r['sigma']))
                    if kind == 'pull' and np.isfinite(r['sigma']) and 'A' in r:
                        xs = np.linspace(r['mu'] - 2.5 * r['sigma'], r['mu'] + 2.5 * r['sigma'], 200)
                        bw = edges[1] - edges[0]
                        g = r['A'] * bw / (np.sqrt(2 * np.pi) * r['sigma']) * \
                            np.exp(-0.5 * ((xs - r['mu']) / r['sigma']) ** 2) / tot
                        core = np.abs(xs - r['mu']) < a.core * r['sigma']
                        ax.plot(xs, g, color=st['color'], lw=0.8, ls='--')
                        ax.plot(xs[core], g[core], color=st['color'], lw=2.0)
                if logy:
                    ax.set_yscale('log')
                    ax.set_ylim(1e-6, None)
                ax.set_xlabel(lab, fontsize=12)
                ax.set_ylabel('fraction per bin', fontsize=12)
                ax.set_xlim(-6 if kind == 'pull' else 0, 6)
                ax.legend(fontsize=7, frameon=False,
                          title='core width (thick: fitted range)' if kind == 'pull'
                          else 'k = median / median(honest)', title_fontsize=7)
            fig.suptitle('%s, inclusive' % o, fontsize=11)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        # widths vs bin, per binning
        for o, (lvl, kind, lab) in OBS.items():
            for b, (blvl, be, blab) in BINNINGS.items():
                if b == 'incl' or blvl not in (lvl, 'both'):
                    continue
                vs = views_present(R, o, b)
                if not vs:
                    continue
                nb = len(be) - 1
                x = np.arange(nb)
                fig = plt.figure(figsize=(9, 7))
                gs = fig.add_gridspec(2, 1, height_ratios=(3, 1.4), hspace=0.06)
                ax, rx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharex=ax)
                for v in vs:
                    st = VIEW_STYLE[v]
                    s = np.array([r['sigma'] for r in R[(o, b, v)]])
                    e = np.array([r['sigma_err'] for r in R[(o, b, v)]])
                    ax.errorbar(x + 0.04 * vs.index(v), s, yerr=e, marker=st['marker'], ms=4,
                                color=st['color'], mfc=st.get('mfc', st['color']), ls=st['ls'],
                                lw=1, label=st['label'])
                for e_ in eras:
                    k = (o, b, 'data@' + e_)
                    if k in R:
                        ax.plot(x, [r['sigma'] for r in R[k]], marker='x', ls='none', ms=5,
                                label='data %s' % e_)
                ax.axhline(1, color='black', lw=0.8)
                ax.set_ylabel('core width' if kind == 'pull' else 'k (median)', fontsize=12)
                ax.legend(fontsize=7, frameon=False, ncol=2)
                ax.tick_params(labelbottom=False)
                d = R.get((o, b, 'data'))
                for v in ('mc_raw', 'mc_corr'):
                    if d is None or (o, b, v) not in R:
                        continue
                    sd = np.array([r['sigma'] for r in d])
                    ed = np.array([r['sigma_err'] for r in d])
                    sm = np.array([r['sigma'] for r in R[(o, b, v)]])
                    em = np.array([r['sigma_err'] for r in R[(o, b, v)]])
                    with np.errstate(all='ignore'):
                        rr = sd / sm
                        er = rr * np.hypot(ed / sd, em / sm)
                    rx.errorbar(x, rr, yerr=er, marker=VIEW_STYLE[v]['marker'], ls='none', ms=4,
                                color=VIEW_STYLE[v]['color'], label='data / %s' % VIEW_STYLE[v]['label'])
                rx.axhline(1, color='black', lw=0.8)
                rx.set_ylabel('data / MC', fontsize=10)
                rx.set_xticks(x)
                rx.set_xticklabels(['%s-%s' % (_fe(be[i]), _fe(be[i + 1])) for i in range(nb)],
                                   rotation=45, fontsize=8)
                rx.set_xlabel(blab, fontsize=11)
                rx.legend(fontsize=7, frameon=False)
                fig.suptitle('%s vs %s' % (o, b), fontsize=11)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
    with PdfPages(out + '_fits.pdf') as pdf:
        for o, (lvl, kind, lab) in OBS.items():
            edges = PULL_EDGES if kind == 'pull' else ABS_EDGES
            c = 0.5 * (edges[1:] + edges[:-1])
            bw = edges[1] - edges[0]
            for b, (blvl, be, blab) in BINNINGS.items():
                if blvl not in (lvl, 'any', 'both'):
                    continue
                vs = [v for v in views_present(R, o, b) if v != 'data_SR']
                if not vs:
                    continue
                nb = len(be) - 1
                ncol = min(4, nb)
                nrow = int(np.ceil(nb / ncol))
                fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.3 * nrow), squeeze=False)
                for i in range(nrow * ncol):
                    ax = axs[i // ncol, i % ncol]
                    if i >= nb:
                        ax.axis('off')
                        continue
                    for v in vs:
                        w = H[(o, b, v)][0][i]
                        tot = w.sum()
                        if tot <= 0:
                            continue
                        st = VIEW_STYLE[v]
                        r = R[(o, b, v)][i]
                        if v == 'data':
                            ax.plot(c, w / tot, '.', ms=1.5, color='black',
                                    label='data %.3f' % r['sigma'])
                        else:
                            ax.stairs(w / tot, edges, color=st['color'], ls=st['ls'], lw=0.8,
                                      label='%s %.3f' % (st['label'], r['sigma']))
                        if kind == 'pull' and 'A' in r and np.isfinite(r['sigma']):
                            xs = np.linspace(r['mu'] - a.core * r['sigma'],
                                             r['mu'] + a.core * r['sigma'], 60)
                            ax.plot(xs, r['A'] * bw / (np.sqrt(2 * np.pi) * r['sigma']) *
                                    np.exp(-0.5 * ((xs - r['mu']) / r['sigma']) ** 2) / tot,
                                    color=st['color'], lw=1.8)
                    ax.set_yscale('log')
                    ax.set_xlim(-5 if kind == 'pull' else 0, 5)
                    ax.set_title('%s: %s - %s' % (b, _fe(be[i]), _fe(be[i + 1])), fontsize=8)
                    ax.tick_params(labelsize=7)
                    if ax.get_legend_handles_labels()[0]:
                        ax.legend(fontsize=5.5, frameon=False)
                fig.suptitle('%s in bins of %s (thick line: core fit, over its fitted range)'
                             % (o, b), fontsize=10)
                fig.tight_layout(rect=(0, 0, 1, 0.96))
                pdf.savefig(fig)
                plt.close(fig)
    js = {'%s|%s|%s' % k: v for k, v in R.items()}
    jp = {'%s|%s|%s' % k: v for k, v in P.items()}
    with open(out + '.json', 'w') as fh:
        json.dump(dict(summary=lines, results=js, profiles=jp,
                       binnings={k: v[1] for k, v in BINNINGS.items()},
                       pt2_edges=PT2_EDGES, sh2_edges=SH2_EDGES),
                  fh, indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    print('wrote %s.pdf, %s_fits.pdf, %s.json' % (out, out, out))


def _fe(x):
    if abs(x) >= 1e3:
        return 'inf' if x > 0 else '-inf'
    return '%g' % x


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(q):
        q.add_argument('--sr', type=float, default=0.05, help='signal region |m - mJ/psi| < SR [GeV]')
        q.add_argument('--sb', type=float, nargs=2, default=(0.15, 0.25),
                       help='sidebands SB0 < |m - mJ/psi| < SB1 [GeV]')
        q.add_argument('--core', type=float, default=1.0,
                       help='core fit range in units of the fitted sigma (default 1.0)')

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
    g = sub.add_parser('plot', help='combine .npz files, fit, plot')
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
        for lab, fn in DEFAULT_FILES.items():
            p = os.path.join(hd, 'vps_%s.npz' % lab)
            fill([fn], lab, p, a)
            paths.append(p)
        plot(paths, a.out, a)


if __name__ == '__main__':
    main()
