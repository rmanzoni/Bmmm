'''
Compare a covflow-corrected ntuple against its nominal twin.

    python3 compare_covflow.py --nominal rjpsi_nominal.root \\
                               --covflow rjpsi_covflow.root \\
                               [--data data_2022.root] \\
                               [--out covflow_comparison]

Produces <out>.pdf (multipage) and prints the numbers that matter to stdout.
Needs uproot, numpy, matplotlib -- the same set covflow already uses.

Binning is set by hand, in the RANGES and DELTA_RANGES tables near the top.
Edit those; there is no flag. The right range for q2 is a statement about the
decay, not about a particular pair of files, and the adaptive schemes tried
before this one either handed the axis to an unphysical tail or -- with
equal-occupancy bins on a density axis -- put the shape into the bin widths and
collapsed every skewed variable into a spike at one edge. The fraction of
entries outside the range is printed in each panel title, so clipping a tail
never hides it.

WHY IT IS BUILT THIS WAY

The two MC files come from the same input, the same selection and the same
trajectories: the correction changes uncertainties, never momenta. So the
candidates can be matched ONE TO ONE on (run, lumi, event) and every comparison
done PAIRED -- delta per candidate, not two histograms side by side. A paired
comparison sees a systematic 2% shift that overlaid distributions would bury
under the spread, and it is the only way to tell "the correction moved this
candidate" from "these are two samples of the same thing".

Sections, in the order you should read them:

  1. INTEGRITY. Do the files pair up, and did anything change that must not?
     If mu1_pt moved by so much as a float epsilon, the correction touched the
     trajectory and nothing below this line is worth reading.
  2. CORRECTOR HEALTH. covflow_ok and covflow_zmax, per muon. This is where you
     find out whether the correction was applied at all, and whether mu3 -- a
     softer muon than the mu1 the flow was trained on -- is being extrapolated
     to rather than interpolated.
  3. WHAT MOVED. cov_corr vs cov WITHIN the corrected file, per covariance
     element. Split into the 5 sigmas and the 10 correlations, and the
     correlations split again into within-block and cross-block, because the
     cross-block ones are ~0 in both data and MC and should barely move. If
     they do, that is noise-fitting and the argument for --features block.
  4. PHYSICS, PAIRED. Per-candidate deltas in the fit quantities.
  4b. IMPACT PARAMETERS. The per-muon dxy/dz errors and significances, which
     come from cov_track() and so do move, plus the _raw counterparts that
     give the same shift from within one file.
  4c. KINEMATICS. q2, m_miss2, mcorr, the helicity angles. These move too --
     not through the muon momenta, which are untouched, but through the
     refitted J/psi and through the flight direction, since the fitted vertex
     positions depend on the covariances that weight the fit.
  5. PHYSICS, DISTRIBUTIONS. Overlaid, with data if given.

WHAT THIS SCRIPT CANNOT TELL YOU

Whether the correction is RIGHT. A vertex chi2 that looks more data-like is
compatible with two different stories: the covariance was wrong and is now
right, or the resolution is wrong and a widened sigma is hiding it. Only the
pull probes distinguish those. Read section 7 of the covflow README before
concluding anything from section 4 here.
'''

import argparse
import itertools
import os

import numpy as np
import uproot

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PARAMS = ['qoverp', 'lambda', 'phi', 'dxy', 'dsz']
PAIRS = [(i, j) for i in range(5) for j in range(i, 5)]
NAMES = ['%s_%s' % (PARAMS[i], PARAMS[j]) for i, j in PAIRS]

# r-phi and r-z: the two blocks the track fit actually correlates within.
# Everything else is a cross-block correlation, ~0 in data AND MC.
MUONS = ['mu1', 'mu2', 'mu3']

RPHI, RZ = set([0, 2, 3]), set([1, 4])
WITHIN = [(i, j) for i, j in PAIRS
          if i != j and ((i in RPHI and j in RPHI) or (i in RZ and j in RZ))]
CROSS = [(i, j) for i, j in PAIRS
         if i != j and (i, j) not in WITHIN]

# candidate-level quantities the correction is supposed to move
FIT_VARS = [
    'sv_prob', 'sv_chi2', 'lxy_sig', 'lxyz_sig',
    'jpsi_vtx_prob', 'jpsi_vtx_chi2', 'jpsi_lxy_sig',
    'mu_ip3d_jpsi_pv_sig', 'mu_ip3d_sv_pv_sig',
]
# quantities that MUST be bit-identical: the correction changes uncertainties,
# not trajectories. Any difference here is a bug, not a result. The dxy/dz
# VALUES belong here for the same reason -- they are distances, computed from a
# trajectory the correction does not touch.
INVARIANTS = (['mu1_pt', 'mu2_pt', 'mu3_pt', 'mu1_eta', 'mu2_eta', 'mu3_eta',
               'mass', 'pt', 'jpsi_mass', 'npv'] +
              ['%s_bs_dxy' % m for m in MUONS])

# Measured against the PRIMARY VERTEX, and therefore NOT invariant.
#
# This was found by looking at the plots, not by reasoning ahead: everything
# referred to the beamspot came out exactly flat, everything referred to the PV
# moved at the percent level, with identical event sets. The PV assignment /
# refit depends on the candidate vertex, the vertex moves with the covariances,
# so dxy and dz move with it. dz_e stays flat because reco::Track::dzError()
# takes no PV argument, while dxyError(pv.position(), pv.error()) does.
#
# Worth keeping separate from the fit quantities: this is the correction
# reaching a variable through the vertex REFIT rather than through the fit
# uncertainties, which is a different mechanism and a different size.
PV_VARS = ['%s_%s' % (m, v) for m in MUONS for v in ['dxy', 'dz']]

# Per-muon impact-parameter ERRORS and significances. With CommonBranches
# routing these through cov_track(), they are computed from the SAME matrix the
# fits used, so they move. Each also has a _raw counterpart from the
# uncorrected covariance, which is what makes the shift measurable inside one
# file rather than only across the pair.
IP_VARS = ['%s_%s' % (m, v) for m in MUONS
           for v in ['dxy_e', 'dxy_sig', 'dz_e', 'dz_sig',
                     'bs_dxy_e', 'bs_dxy_sig']]
IP_RAW = [('%s_%s' % (m, v), '%s_%s_raw' % (m, v)) for m in MUONS
          for v in ['dxy_e', 'dz_e', 'bs_dxy_e']]

# Kinematic observables. NONE of these is built from the raw 3-muon p4, so all
# of them move -- by two routes that are worth keeping apart when reading the
# numbers:
#
#   via the REFITTED J/psi (jpsi_rfp4, mass-constrained): the collinear family,
#   which never uses a flight direction at all.
#
#   via the FLIGHT DIRECTION (Bdirection_sv / _jpsi = SV - PV): everything that
#   picks a Bc momentum from a direction. The vertex positions move because the
#   covariances are the weights in the fit, so the direction moves with them.
#
# Both should be small -- a reweighted least-squares solution does not travel
# far -- but small is not zero, which is the point of measuring instead of
# assuming.
KIN_COLL = ['q2_coll', 'm_miss2_coll', 'mu_b_e_coll',
            'mu1_rf_pt', 'mu2_rf_pt', 'jpsi_rf_pt']
KIN_DIR = ['q2_jpsi', 'q2_sv', 'nu1_q2_jpsi', 'nu2_q2_jpsi',
           'nu1_q2_sv', 'nu2_q2_sv',
           'mcorr_jpsi', 'mcorr_sv', 'p4_perp_jpsi', 'p4_perp_sv',
           'cos_theta_v_sv', 'cos_theta_l_sv', 'cos2d', 'jpsi_cos2d']


def load(path, branches, tree='tree'):
    '''Read the branches that exist, quietly skipping the ones that do not --
    channels differ and this script should work on all of them.'''
    f = uproot.open(path)
    t = f[tree]
    have = set(t.keys())
    want = [b for b in branches if b in have]
    missing = [b for b in branches if b not in have]
    return t.arrays(want, library='np'), have, missing


def quantiles(x, qs=(0.16, 0.50, 0.84)):
    x = x[np.isfinite(x)]
    return np.percentile(x, [100 * q for q in qs]) if len(x) else [np.nan] * 3



# ---------------------------------------------------------------------------
# BINNING, set by hand
# ---------------------------------------------------------------------------
# Adaptive binning does not work here and the first version of this script was
# wrong to try. Equal-occupancy bins put the shape into the bin WIDTHS, so on a
# density axis a skewed variable collapses to a spike at one edge; percentile
# ranges, meanwhile, hand the axis to whatever unphysical tail a variable
# happens to have -- q2_sv reconstructed from a flight direction runs to -2500
# GeV^2 when the direction is badly measured, and no percentile cut fixes that
# without also cutting the physics.
#
# So: explicit ranges, chosen per variable from what the quantity MEANS. A
# probability lives on [0,1]. q2 for Bc -> J/psi mu nu lives on [0, ~11] GeV^2
# and anything outside is a failed reconstruction, worth counting (the panel
# title reports the fraction) but not worth plotting.
#
# Keyed by branch name; a mu1_/mu2_/mu3_ prefix is stripped before lookup, so
# one entry covers all three muons. Anything not listed falls back to a robust
# symmetric range.        var : (lo, hi, nbins)
RANGES = {
    # --- fit quality
    'sv_prob'            : (0.,    1.,    50),
    'jpsi_vtx_prob'      : (0.,    1.,    50),
    'sv_chi2'            : (0.,   30.,    50),
    'jpsi_vtx_chi2'      : (0.,   20.,    50),
    'lxy_sig'            : (0.,   60.,    50),
    'lxyz_sig'           : (0.,   60.,    50),
    'jpsi_lxy_sig'       : (0.,   60.,    50),
    'mu_ip3d_jpsi_pv_sig': (-5.,  20.,    50),
    'mu_ip3d_sv_pv_sig'  : (-5.,  20.,    50),
    # --- impact parameters (cm)
    'dxy'                : (-0.02, 0.02,  50),
    'dz'                 : (-0.10, 0.10,  50),
    'bs_dxy'             : (-0.02, 0.02,  50),
    'dxy_e'              : (0.,    0.008, 50),
    'bs_dxy_e'           : (0.,    0.008, 50),
    'dz_e'               : (0.,    0.03,  50),
    'dxy_e_raw'          : (0.,    0.008, 50),
    'bs_dxy_e_raw'       : (0.,    0.008, 50),
    'dz_e_raw'           : (0.,    0.03,  50),
    'dxy_sig'            : (-10.,  10.,   50),
    'dz_sig'             : (-10.,  10.,   50),
    'bs_dxy_sig'         : (-10.,  10.,   50),
    # --- kinematics via the refitted J/psi
    'q2_coll'            : (-1.,   12.,   50),
    'm_miss2_coll'       : (-2.,   10.,   50),
    'mu_b_e_coll'        : (0.,     4.,   50),
    'rf_pt'              : (0.,    30.,   50),
    'jpsi_rf_pt'         : (0.,    40.,   50),
    # --- kinematics via the flight direction. The wide unphysical tails are
    #     the point of clipping: they are failed direction reconstructions, and
    #     the fraction outside is printed rather than hidden.
    'q2_jpsi'            : (-1.,   12.,   50),
    'q2_sv'              : (-1.,   12.,   50),
    'nu1_q2_jpsi'        : (-1.,   12.,   50),
    'nu2_q2_jpsi'        : (-1.,   12.,   50),
    'nu1_q2_sv'          : (-1.,   12.,   50),
    'nu2_q2_sv'          : (-1.,   12.,   50),
    'mcorr_jpsi'         : (0.,    15.,   50),
    'mcorr_sv'           : (0.,    15.,   50),
    'p4_perp_jpsi'       : (0.,    10.,   50),
    'p4_perp_sv'         : (0.,    10.,   50),
    'cos_theta_v_sv'     : (-1.,    1.,   40),
    'cos_theta_l_sv'     : (-1.,    1.,   40),
    # cos2d piles up at 1: a [-1,1] axis shows one bin. This is the window
    # where the candidates actually are.
    'cos2d'              : (0.99,   1.,   50),
    'jpsi_cos2d'         : (0.99,   1.,   50),
    # --- invariants
    'pt'                 : (0.,    50.,   50),
    'eta'                : (-2.5,   2.5,  50),
    'mass'               : (3.5,    8.,   50),
    'jpsi_mass'          : (2.95,   3.25, 50),
    'npv'                : (0.,    80.,   80),
}

# Paired shifts. Symmetric by construction -- the question is always "did this
# move and by how much", so a range that is not centred on zero hides the answer.
DELTA_RANGES = {
    'sv_prob'            : 0.10,
    'jpsi_vtx_prob'      : 0.10,
    'sv_chi2'            : 3.0,
    'jpsi_vtx_chi2'      : 3.0,
    'lxy_sig'            : 3.0,
    'lxyz_sig'           : 3.0,
    'jpsi_lxy_sig'       : 3.0,
    'mu_ip3d_jpsi_pv_sig': 2.0,
    'mu_ip3d_sv_pv_sig'  : 2.0,
    'dxy'                : 2e-3,
    'dz'                 : 1e-2,
    'bs_dxy'             : 2e-3,
    'dxy_e'              : 5e-4,
    'bs_dxy_e'           : 5e-4,
    'dz_e'               : 2e-3,
    'dxy_sig'            : 1.0,
    'dz_sig'             : 1.0,
    'bs_dxy_sig'         : 1.0,
    'q2_coll'            : 0.05,
    'm_miss2_coll'       : 0.05,
    'mu_b_e_coll'        : 0.02,
    'rf_pt'              : 0.05,
    'jpsi_rf_pt'         : 0.05,
    'q2_jpsi'            : 0.5,
    'q2_sv'              : 0.5,
    'nu1_q2_jpsi'        : 0.5,
    'nu2_q2_jpsi'        : 0.5,
    'nu1_q2_sv'          : 0.5,
    'nu2_q2_sv'          : 0.5,
    'mcorr_jpsi'         : 0.3,
    'mcorr_sv'           : 0.3,
    'p4_perp_jpsi'       : 0.2,
    'p4_perp_sv'         : 0.2,
    'cos_theta_v_sv'     : 0.05,
    'cos_theta_l_sv'     : 0.05,
    'cos2d'              : 2e-3,
    'jpsi_cos2d'         : 2e-3,
}


def _strip_mu(name):
    for m in MUONS:
        if name.startswith(m + '_'):
            return name[len(m) + 1:]
    return name


def make_bins(name, values, delta=False, nbins_default=50):
    """Bin edges for one variable, from the hand-set table above.

    Returns (edges, fraction of finite entries outside them). The fraction is
    reported in the panel title rather than silently dropped: a variable with
    20% of its entries off the axis is telling you something about the
    reconstruction, and it should not take a second script to notice.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.linspace(-1., 1., 3), 0.

    key = _strip_mu(name)
    if delta:
        half = DELTA_RANGES.get(key)
        if half is None:
            # not tabulated: symmetric, wide enough to show the core
            half = max(np.percentile(np.abs(x), 99), 1e-12)
        lo, hi, nb = -half, half, nbins_default
    else:
        entry = RANGES.get(key)
        if entry is None:
            lo, hi = np.percentile(x, [0.5, 99.5])
            if lo == hi:
                lo, hi = lo - 1e-9, hi + 1e-9
            nb = nbins_default
        else:
            lo, hi, nb = entry

    outside = float(np.mean((x < lo) | (x > hi)))
    # integer-valued and narrow: one bin per value
    if hi - lo <= nb and np.allclose(np.unique(x[(x >= lo) & (x <= hi)]),
                                     np.round(np.unique(x[(x >= lo) & (x <= hi)]))):
        uniq = np.unique(x[(x >= lo) & (x <= hi)])
        if 0 < len(uniq) <= nb:
            return np.arange(uniq.min() - 0.5, uniq.max() + 1.5), outside
    return np.linspace(lo, hi, nb + 1), outside


def overlay_page(pdf, names, title, nom, cfl, dat=None, per_page=6):
    """Overlaid nominal / covflow distributions with a ratio panel underneath.

    The ratio is the point: the shifts this correction produces are small
    compared with the width of every variable it touches, so two histograms
    drawn on top of each other look identical even when the paired comparison
    says something systematic happened. The ratio panel is where a 2% move
    becomes visible.
    """
    names = [v for v in names if v in nom and v in cfl]
    for page in range(0, len(names), per_page):
        chunk = names[page:page + per_page]
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.28)
        for k, v in enumerate(chunk):
            sub = gs[k // 3, k % 3].subgridspec(2, 1, height_ratios=[3, 1],
                                                hspace=0.05)
            ax, rx = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
            a, b = np.asarray(nom[v]), np.asarray(cfl[v])
            pool = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
            if not len(pool):
                ax.axis('off'); rx.axis('off'); continue
            bins, outside = make_bins(v, pool)
            hn, _ = np.histogram(a[np.isfinite(a)], bins=bins, density=True)
            hc, _ = np.histogram(b[np.isfinite(b)], bins=bins, density=True)
            ctr = 0.5 * (bins[1:] + bins[:-1])
            ax.step(ctr, hn, where='mid', color='tab:blue', label='MC nominal')
            ax.step(ctr, hc, where='mid', color='tab:orange', label='MC covflow')
            if dat is not None and v in dat:
                d = np.asarray(dat[v])
                hd, _ = np.histogram(d[np.isfinite(d)], bins=bins, density=True)
                ax.step(ctr, hd, where='mid', color='k', label='data')
            ax.set_title('%s%s' % (v, '' if outside < 5e-4 else
                                   '   (%.1f%% outside)' % (100 * outside)),
                         fontsize=9)
            ax.tick_params(labelbottom=False, labelsize=7)
            if k == 0:
                ax.legend(fontsize=6)
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.where(hn > 0, hc / hn, np.nan)
            rx.step(ctr, r, where='mid', color='tab:orange')
            rx.axhline(1., color='tab:blue', lw=0.8)
            finite = r[np.isfinite(r)]
            if len(finite):
                span = max(0.02, 1.2 * np.max(np.abs(finite - 1.)))
                rx.set_ylim(1 - span, 1 + span)
            rx.set_ylabel('cfl/nom', fontsize=6)
            rx.tick_params(labelsize=6)
        fig.suptitle('%s  (%d/%d)' % (title, page // per_page + 1,
                                      (len(names) - 1) // per_page + 1))
        pdf.savefig(fig); plt.close(fig)


def section(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nominal', required=True)
    ap.add_argument('--covflow', required=True)
    ap.add_argument('--data', default=None)
    ap.add_argument('--tree', default='tree')
    ap.add_argument('--out', default='covflow_comparison')
    # Binning is the RANGES / DELTA_RANGES tables at the top of this file.
    # Edit those rather than passing a flag: the right range for q2 is a
    # statement about the decay, not about this particular pair of files.
    args = ap.parse_args()

    cov_branches = (['%s_cov_%s' % (m, n) for m in MUONS for n in NAMES] +
                    ['%s_cov_corr_%s' % (m, n) for m in MUONS for n in NAMES] +
                    ['%s_covflow_ok' % m for m in MUONS] +
                    ['%s_covflow_zmax' % m for m in MUONS])
    keys = ['run', 'lumi', 'event']
    ip_raw_names = [r for _, r in IP_RAW]
    wanted = (keys + INVARIANTS + PV_VARS + IP_VARS + ip_raw_names +
              KIN_COLL + KIN_DIR + FIT_VARS + cov_branches)

    nom, have_nom, _ = load(args.nominal, wanted, args.tree)
    cfl, have_cfl, missing = load(args.covflow, wanted, args.tree)

    if any(b.endswith('_covflow_ok') for b in missing):
        raise SystemExit('%s has no covflow_* branches -- it was not produced '
                         'with --covflow, or with a build predating the patch.'
                         % args.covflow)

    have_raw = all(r in cfl for _, r in IP_RAW)
    if not have_raw:
        print('\nNOTE: no <obj>_dxy_e_raw branches. These files predate the '
              'CommonBranches change, so dxy_e / dz_e / bs_dxy_e in them come '
              'from bestTrack(), i.e. the RAW covariance. Section 4b will show '
              'only what moved through the vertex refit, not what the '
              'correction does to the IP errors themselves. Re-run both ntuples '
              'to get that.')

    pdf = PdfPages(args.out + '.pdf')

    # -- 1. integrity --------------------------------------------------------
    section('1. INTEGRITY')

    def key_of(a):
        return np.stack([a['run'], a['lumi'], a['event']], axis=1)

    kn, kc = key_of(nom), key_of(cfl)
    vn = np.array([hash(tuple(r)) for r in kn])
    vc = np.array([hash(tuple(r)) for r in kc])
    print('  candidates   nominal %8d   covflow %8d' % (len(vn), len(vc)))
    if len(vn) != len(set(vn)) or len(vc) != len(set(vc)):
        print('  NOTE: (run,lumi,event) is not unique -- more than one candidate '
              'per event is kept. Pairing uses the first occurrence; if that is '
              'not what you want, add the candidate index to the key.')

    common, in_n, in_c = np.intersect1d(vn, vc, return_indices=True)
    order_n, order_c = in_n[np.argsort(common)], in_c[np.argsort(common)]
    print('  matched      %8d' % len(common))
    print('  nominal only %8d   covflow only %8d'
          % (len(vn) - len(common), len(vc) - len(common)))
    if len(vn) != len(vc):
        print('  -> the two files do NOT contain the same events. The correction '
              'moved a candidate across a selection cut, so part of any '
              'difference you see below is an EFFICIENCY change, not a shape '
              'change. Worth knowing which cut before going further.')

    bad = []
    for b in INVARIANTS:
        if b not in nom or b not in cfl:
            continue
        d = np.abs(np.asarray(nom[b])[order_n] - np.asarray(cfl[b])[order_c])
        d = d[np.isfinite(d)]
        worst = d.max() if len(d) else 0.
        print('  %-12s max |delta| %.3e %s' % (b, worst, '' if worst == 0 else '  <-- NOT ZERO'))
        if worst != 0:
            bad.append(b)
    if bad:
        print('\n  STOP. %s changed. The correction is supposed to touch only the '
              'covariance, so a moved trajectory means the corrected track is '
              'being used somewhere it should not be. Nothing below is '
              'meaningful until this is understood.' % ', '.join(bad))

    # -- 2. corrector health -------------------------------------------------
    section('2. CORRECTOR HEALTH')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for m in MUONS:
        ok = np.asarray(cfl['%s_covflow_ok' % m]).astype(bool)
        z = np.asarray(cfl['%s_covflow_zmax' % m])
        print('  %s  corrected %6.2f%%   zmax  p16 %5.2f  median %5.2f  p84 %5.2f  max %6.2f'
              % ((m, 100. * ok.mean()) + tuple(quantiles(z[ok])) + (np.nanmax(z[ok]) if ok.any() else np.nan,)))
        # deliberately NOT adaptive: the tail is the quantity of interest here,
        # and a per-muon range would stop the three curves being comparable
        axes[0].hist(z[ok], bins=np.linspace(0, 8, 80), histtype='step',
                     label=m, density=True)
    axes[0].set_xlabel('covflow_zmax'); axes[0].set_ylabel('a.u.')
    axes[0].set_yscale('log'); axes[0].legend(); axes[0].set_title('latent distance, per muon')

    frac = [1. - np.asarray(cfl['%s_covflow_ok' % m]).mean() for m in MUONS]
    axes[1].bar(MUONS, frac)
    axes[1].set_ylabel('fraction left UNcorrected')
    axes[1].set_title('covflow_ok == 0')
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print('  A zmax tail on mu3 that mu1 does not have is the flow extrapolating:')
    print('  it was trained on the leading muon, and the bachelor is softer.')

    # -- 3. what moved -------------------------------------------------------
    section('3. WHAT MOVED (within the corrected file)')
    print('  sigma ratios, corrected / raw:')
    sig_ratio = {}
    for m in MUONS:
        for k, p in enumerate(PARAMS):
            n = '%s_%s' % (p, p)
            raw = np.asarray(cfl['%s_cov_%s' % (m, n)])
            cor = np.asarray(cfl['%s_cov_corr_%s' % (m, n)])
            good = np.isfinite(cor) & (raw > 0)
            r = np.sqrt(cor[good] / raw[good])
            sig_ratio[(m, p)] = r
            lo, med, hi = quantiles(r)
            print('    %s sigma_%-7s  median %7.4f   [%7.4f, %7.4f]'
                  % (m, p, med, lo, hi))

    print('\n  correlation shifts, corrected - raw:')
    rho_shift = {}
    for m in MUONS:
        for (i, j) in PAIRS:
            if i == j:
                continue
            n = '%s_%s' % (PARAMS[i], PARAMS[j])
            ii, jj = '%s_%s' % (PARAMS[i], PARAMS[i]), '%s_%s' % (PARAMS[j], PARAMS[j])
            raw = np.asarray(cfl['%s_cov_%s' % (m, n)])
            cor = np.asarray(cfl['%s_cov_corr_%s' % (m, n)])
            s_raw = np.sqrt(np.asarray(cfl['%s_cov_%s' % (m, ii)]) *
                            np.asarray(cfl['%s_cov_%s' % (m, jj)]))
            s_cor = np.sqrt(np.asarray(cfl['%s_cov_corr_%s' % (m, ii)]) *
                            np.asarray(cfl['%s_cov_corr_%s' % (m, jj)]))
            good = np.isfinite(cor) & (s_raw > 0) & (s_cor > 0)
            d = cor[good] / s_cor[good] - raw[good] / s_raw[good]
            rho_shift[(m, (i, j))] = d
            tag = 'within' if (i, j) in WITHIN else 'CROSS '
            if m == 'mu1':
                lo, med, hi = quantiles(d)
                print('    %s %s d_rho_%-16s median %+7.4f   [%+7.4f, %+7.4f]'
                      % (m, tag, n, med, lo, hi))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(PARAMS))
    for off, m in zip([-0.25, 0, 0.25], MUONS):
        med = [np.median(sig_ratio[(m, p)]) for p in PARAMS]
        axes[0].bar(x + off, med, width=0.25, label=m)
    axes[0].set_xticks(x); axes[0].set_xticklabels(PARAMS, rotation=30)
    axes[0].axhline(1., color='k', lw=0.8)
    axes[0].set_ylabel('median sigma_corr / sigma_raw'); axes[0].legend()

    off_pairs = [p for p in PAIRS if p[0] != p[1]]
    x = np.arange(len(off_pairs))
    med = [np.median(rho_shift[('mu1', p)]) for p in off_pairs]
    colors = ['tab:blue' if p in WITHIN else 'tab:red' for p in off_pairs]
    axes[1].bar(x, med, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['%s_%s' % (PARAMS[i], PARAMS[j]) for i, j in off_pairs],
                            rotation=75, fontsize=7)
    axes[1].axhline(0., color='k', lw=0.8)
    axes[1].set_ylabel('median d_rho (mu1)')
    axes[1].set_title('blue = within-block, red = cross-block')
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    cross_med = np.max([abs(np.median(rho_shift[('mu1', p)])) for p in CROSS])
    within_med = np.max([abs(np.median(rho_shift[('mu1', p)])) for p in WITHIN])
    print('\n  largest |median d_rho|:  within-block %.4f   cross-block %.4f'
          % (within_med, cross_med))
    if cross_med > 0.2 * within_med:
        print('  -> the cross-block correlations are moving comparably to the '
              'within-block ones. They are ~0 in both data and MC, so this is '
              'the flow fitting noise: compare against a --features block run.')

    # -- 4. physics, paired --------------------------------------------------
    section('4. PHYSICS, PAIRED PER CANDIDATE')
    present = [v for v in FIT_VARS if v in nom and v in cfl]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, v in zip(axes.flat, present):
        a = np.asarray(nom[v])[order_n]
        b = np.asarray(cfl[v])[order_c]
        good = np.isfinite(a) & np.isfinite(b)
        d = b[good] - a[good]
        lo, med, hi = quantiles(d)
        frac_up = (d > 0).mean() if len(d) else np.nan
        print('  %-22s  median delta %+10.4g   [%+.4g, %+.4g]   %.1f%% move up'
              % (v, med, lo, hi, 100 * frac_up))
        dbins, dout = make_bins(v, d, delta=True)
        ax.hist(d, bins=dbins, histtype='step', density=True)
        ax.axvline(0., color='k', lw=0.8)
        ax.set_title('%s%s' % (v, '' if dout < 5e-4 else
                               '  (%.1f%% outside)' % (100 * dout)), fontsize=9)
        ax.set_xlabel('covflow - nominal', fontsize=8)
    for ax in axes.flat[len(present):]:
        ax.axis('off')
    fig.suptitle('paired per-candidate shifts')
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print('  "% move up" is the sharpest single number here: 50% means the '
          'correction is symmetric noise, anything far from it is a systematic '
          'shift in that variable.')

    # -- 4b. impact parameters -----------------------------------------------
    section('4b. IMPACT-PARAMETER ERRORS AND SIGNIFICANCES')
    print('  These now come from cov_track(), i.e. the same matrix the fits used,')
    print('  so they move. The _raw counterparts in the SAME file give the shift')
    print('  without needing the nominal file at all.')
    ip_present = [v for v in IP_VARS if v in nom and v in cfl]
    for v in ip_present:
        a = np.asarray(nom[v])[order_n]
        b = np.asarray(cfl[v])[order_c]
        good = np.isfinite(a) & np.isfinite(b)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = b[good] / a[good]
        lo, med, hi = quantiles(r[np.isfinite(r)])
        print('  %-20s  covflow/nominal  median %7.4f   [%7.4f, %7.4f]'
              % (v, med, lo, hi))

    print('\n  within the corrected file, corrected / raw:')
    for corrected, raw in IP_RAW:
        if corrected not in cfl or raw not in cfl:
            continue
        a, b = np.asarray(cfl[raw]), np.asarray(cfl[corrected])
        good = np.isfinite(a) & np.isfinite(b) & (a != 0)
        lo, med, hi = quantiles(b[good] / a[good])
        print('  %-20s  median %7.4f   [%7.4f, %7.4f]' % (corrected, med, lo, hi))
        if med == 1.0 and lo == 1.0 and hi == 1.0:
            print('      -> identical to raw. CommonBranches is still filling this '
                  'from bestTrack(); the cov_track() patch did not take effect.')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4)) if have_raw else (None, [])
    for ax, v in zip(axes, ['dxy_e', 'dz_e', 'bs_dxy_e']):
        for m in MUONS:
            raw, cor = '%s_%s_raw' % (m, v), '%s_%s' % (m, v)
            if cor not in cfl or raw not in cfl:
                continue
            a, b = np.asarray(cfl[raw]), np.asarray(cfl[cor])
            good = np.isfinite(a) & np.isfinite(b) & (a != 0)
            ax.hist(b[good] / a[good], bins=np.linspace(0.8, 1.2, 80),
                    histtype='step', density=True, label=m)
        ax.axvline(1., color='k', lw=0.8)
        ax.set_xlabel('%s corrected / raw' % v); ax.legend(fontsize=7)
    if fig is not None:
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    print('\n  measured against the PV, so moved by the vertex refit:')
    for v in [x for x in PV_VARS if x in nom and x in cfl]:
        a = np.asarray(nom[v])[order_n]
        b = np.asarray(cfl[v])[order_c]
        good = np.isfinite(a) & np.isfinite(b)
        d = b[good] - a[good]
        scale = np.percentile(np.abs(a[good]), 68) if good.any() else np.nan
        lo, med, hi = quantiles(d)
        print('  %-16s median delta %+10.3g   [%+.3g, %+.3g]   (delta/scale %.2e)'
              % (v, med, lo, hi, abs(med) / scale if scale else np.nan))
    print('  The bs_ counterparts are referred to the beamspot and do NOT move:')
    print('  that contrast is what identifies the vertex refit as the mechanism.')

    # -- 4c. kinematic observables -------------------------------------------
    section('4c. KINEMATIC OBSERVABLES')
    print('  These move even though no muon momentum did: via the refitted J/psi')
    print('  (collinear family) and via the flight direction SV-PV (the rest).')
    for label, group in [('refitted J/psi', KIN_COLL), ('flight direction', KIN_DIR)]:
        present_k = [v for v in group if v in nom and v in cfl]
        if not present_k:
            continue
        print('\n  -- moves via the %s --' % label)
        for v in present_k:
            a = np.asarray(nom[v])[order_n]
            b = np.asarray(cfl[v])[order_c]
            good = np.isfinite(a) & np.isfinite(b)
            d = b[good] - a[good]
            scale = np.percentile(np.abs(a[good]), 68) if good.any() else np.nan
            lo, med, hi = quantiles(d)
            print('    %-18s median delta %+11.4g   [%+.3g, %+.3g]   '
                  '%5.1f%% up   (delta/scale %.2e)'
                  % (v, med, lo, hi, 100 * (d > 0).mean(),
                     abs(med) / scale if scale else np.nan))
        fig, axes = plt.subplots(2, 4, figsize=(15, 7))
        for ax, v in zip(axes.flat, present_k):
            a = np.asarray(nom[v])[order_n]
            b = np.asarray(cfl[v])[order_c]
            good = np.isfinite(a) & np.isfinite(b)
            d = b[good] - a[good]
            dbins, dout = make_bins(v, d, delta=True)
            ax.hist(d, bins=dbins, histtype='step', density=True)
            ax.axvline(0., color='k', lw=0.8)
            ax.set_title('%s%s' % (v, '' if dout < 5e-4 else
                                   '  (%.1f%% out)' % (100 * dout)), fontsize=8)
        for ax in axes.flat[len(present_k):]:
            ax.axis('off')
        fig.suptitle('kinematics, paired shift -- via the %s' % label)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print('\n  "delta/scale" is the median shift over the 68th percentile of the')
    print('  variable itself: the honest measure of whether this matters. A shift')
    print('  well below the resolution is real but not something to correct for.')

    # -- 5. distributions ----------------------------------------------------
    section('5. DISTRIBUTIONS')
    dat = None
    if args.data:
        dat, _, _ = load(args.data, keys + INVARIANTS + IP_VARS +
                         KIN_COLL + KIN_DIR + FIT_VARS, args.tree)

    for title, group in [
            ('fit quality', present),
            ('impact parameters', ip_present),
            ('kinematics -- via the refitted J/psi', KIN_COLL),
            ('kinematics -- via the flight direction', KIN_DIR),
            ('impact parameters measured against the PV -- these move',
             PV_VARS),
            ('invariants (these two curves must lie on top of each other)',
             INVARIANTS)]:
        overlay_page(pdf, group, title, nom, cfl, dat)
        print('  %-52s %3d variables' % (title, len([v for v in group
                                                     if v in nom and v in cfl])))
    print('\n  The invariants page is a control: blue and orange there must be')
    print('  indistinguishable and the ratio panel flat at 1. If it is not, the')
    print('  two files are not the same events and section 4 is comparing')
    print('  different candidates.')

    pdf.close()
    print('\nwrote %s.pdf' % args.out)
    print('\nA vertex chi2 that now looks more like data does NOT by itself mean '
          'the covariance was fixed: if the MC RESOLUTION is also off, a widened '
          'sigma narrows the pull instead of correcting it, and chi2 improves for '
          'the wrong reason. The pull probes are what separates the two.')


if __name__ == '__main__':
    raise SystemExit(main())