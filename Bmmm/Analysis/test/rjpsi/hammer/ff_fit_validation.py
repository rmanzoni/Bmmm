#!/usr/bin/env python3
'''
Validation of the Harrison-2024 -> Hammer-BGL coefficient fit.

Two things live here.

1. EXACT covariance propagation. fit_coeffs is LINEAR in the input curves
   (four Tikhonov solves, plus one linear back-substitution of bvec into the F1
   target), so the coefficient covariance is

       C_coeff = J . C_Harrison . J^T

   with J the 15 x 160 Jacobian built once from the design matrices. No toys, no
   seed, no run-to-run scatter, no NSAMP to tune. The toy route is kept as a
   cross-check of that algebra, not as the method: at NSAMP the eigenvalue
   scatter is sqrt(2/NSAMP), which is exactly the ~7% wobble seen at 400.

2. Plots. The curves, the toy cloud, the fitted curves, and -- the one that
   matters for the band question -- the input uncertainty and the propagated
   uncertainty on the same axes.

    python3 ff_fit_validation.py --plots-dir plots
    python3 ff_fit_validation.py --plots-dir plots --toys 20000 --seed 1

Imports the fitter itself, so the design matrices, LAM and fit_coeffs are the
ones actually used to produce the card -- never a second copy that can drift.
'''
import argparse
import os
import sys

import numpy as np
import gvar as gv

# harrison_ffs.py is Harrison's ancillary wrapper and is not (yet) vendored
# here; harrison_to_hammer_bgl imports it at module scope, so the path has to be
# in place before that import happens.
_HF_DIR = os.environ.get('HARRISON_FFS_DIR', '')
if _HF_DIR and _HF_DIR not in sys.path:
    sys.path.insert(0, _HF_DIR)
try:
    import harrison_to_hammer_bgl as FIT
except ImportError as _exc:
    raise SystemExit(
        '[FATAL] %s\n'
        '        harrison_ffs.py is not importable. Point HARRISON_FFS_DIR at the\n'
        '        directory holding it (and its data files):\n'
        '            export HARRISON_FFS_DIR=/path/to/harrison/ancillary\n'
        '        It should be vendored into this directory -- the FF systematic\n'
        '        of the measurement should not depend on one laptop.' % _exc)
import hammer_bgl_forward as HB

GRID = FIT.GRID
NQ = len(GRID)
FFS = ('g', 'f', 'F1', 'F2')
NCOEF = 15
SLICES = {'avec': slice(0, 4), 'bvec': slice(4, 8),
          'cvec': slice(8, 11), 'dvec': slice(11, 15)}


# ---------------------------------------------------------------------------
# the linear algebra of the fit
# ---------------------------------------------------------------------------
def coefficient_jacobian():
    """Imported from the fitter: one copy of the algebra that builds the card."""
    return FIT.coefficient_jacobian()


def curve_jacobian():
    '''D (4*NQ x 15): d(curves)/d(coefficients), in Harrison's normalisation.

    Every Hammer FF is linear in the coefficient vector, so the band is exact:
    var(curve) = diag(D C_coeff D^T). No finite differences.
    '''
    D = np.zeros((4 * NQ, NCOEF))
    D[0 * NQ:1 * NQ, SLICES['avec']] = FIT._Ag
    D[1 * NQ:2 * NQ, SLICES['bvec']] = FIT._Af
    D[2 * NQ:3 * NQ, SLICES['bvec']] = FIT._Cbv
    D[2 * NQ:3 * NQ, SLICES['cvec']] = FIT._Bcv
    D[3 * NQ:4 * NQ, SLICES['dvec']] = FIT._Ap / HB.P1_OVER_F2   # back to Harrison's F2
    return D


def stack(hc):
    return np.concatenate([hc[k] for k in FFS])


def coeff_cov_linear(hc):
    J = coefficient_jacobian()
    C = gv.evalcov(stack(hc))
    return J @ C @ J.T


def coeff_cov_toys(hc, nsamp, seed):
    gv.ranseed(seed)
    samples = np.empty((nsamp, NCOEF))
    for i, d in enumerate(gv.raniter(hc, n=nsamp)):
        a, b, c, dd = FIT.fit_coeffs(d['g'], d['f'], d['F1'],
                                     HB.p1_from_f2(d['F2']))
        samples[i] = FIT.flat(a, b, c, dd)
    return np.cov(samples.T), samples


def central_coeffs(hc):
    a, b, c, d = FIT.fit_coeffs(gv.mean(hc['g']), gv.mean(hc['f']),
                                gv.mean(hc['F1']), HB.p1_from_f2(gv.mean(hc['F2'])))
    return FIT.flat(a, b, c, d)


def bands(cov_coeff, vec):
    '''Relative 1-sigma band of each fitted curve, from the coefficient cov.'''
    D = curve_jacobian()
    var = np.diag(D @ cov_coeff @ D.T)
    central = curves_from_coeffs(vec, GRID)
    out = {}
    for i, name in enumerate(FFS):
        sd = np.sqrt(np.maximum(var[i * NQ:(i + 1) * NQ], 0.))
        out[name] = sd / np.abs(central[name])
    return out


# ---------------------------------------------------------------------------
# differential rate: imported, so the lattice-side and Hammer-side validations
# never drift apart
# ---------------------------------------------------------------------------
import ff_rate as FR                                            # noqa: E402
from ff_rate import (                                          # noqa: E402
    LEPTONS, M_MU, M_TAU, _momentum, _sigma_directions, curves_from_coeffs,
    dgamma_dq2, helicity_amplitudes, r_jpsi, r_jpsi_with_error, rate_grid,
    rate_with_band,
)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def make_plots(hc, vec, cov_lin, toys, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fitted = curves_from_coeffs(vec, GRID)
    inp_rel = dict((k, gv.sdev(hc[k]) / np.abs(gv.mean(hc[k]))) for k in FFS)
    out_rel = bands(cov_lin, vec)

    # --- 1. curves: Harrison band, toy cloud, Hammer-model fit ---------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, name in zip(axes.ravel(), FFS):
        mu, sd = gv.mean(hc[name]), gv.sdev(hc[name])
        if toys is not None:
            for row in toys[:200]:
                ax.plot(GRID, curves_from_coeffs(row, GRID)[name], color='0.7',
                        lw=0.4, alpha=0.35, zorder=1)
            ax.plot([], [], color='0.7', lw=0.8, label='fitted toys (200 shown)')
        ax.fill_between(GRID, mu - sd, mu + sd, alpha=0.30, color='C0', zorder=2,
                        label=r'Harrison-2024 $\pm1\sigma$')
        ax.plot(GRID, mu, color='C0', lw=1.6, zorder=3, label='Harrison-2024 central')
        ax.plot(GRID, fitted[name], color='C3', lw=1.4, ls='--', zorder=4,
                label='fit through Hammer BGL')
        ax.set_ylabel(name)
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel(r'$q^2$ [GeV$^2$]')
    axes[1, 1].set_xlabel(r'$q^2$ [GeV$^2$]')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Form factors: lattice input vs Hammer-convention fit')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'ff_curves.png'), dpi=140)
    plt.close(fig)

    # --- 2. the band question ----------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, name in zip(axes.ravel(), FFS):
        ax.plot(GRID, 100 * inp_rel[name], color='C0', lw=1.8,
                label='Harrison-2024 input')
        ax.plot(GRID, 100 * out_rel[name], color='C3', lw=1.8, ls='--',
                label='propagated through the fit')
        ax.set_ylabel(r'$\sigma/|%s|$  [%%]' % name)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel(r'$q^2$ [GeV$^2$]')
    axes[1, 1].set_xlabel(r'$q^2$ [GeV$^2$]')
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('Relative uncertainty: in vs out. The fit is a projection -- '
                 'dashed above solid means the fit is ADDING uncertainty')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'ff_band_in_out.png'), dpi=140)
    plt.close(fig)

    # --- 3. fit residual in units of the lattice error ----------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in FFS:
        pull = (fitted[name] - gv.mean(hc[name])) / np.maximum(gv.sdev(hc[name]), 1e-30)
        ax.plot(GRID, pull, lw=1.5, label=name)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel(r'$q^2$ [GeV$^2$]')
    ax.set_ylabel(r'(fit $-$ lattice) / $\sigma_{\rm lattice}$')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'ff_fit_pull.png'), dpi=140)
    plt.close(fig)

    # --- 4. q2 spectra, mu and tau, with the FF band ------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5),
                             gridspec_kw={'height_ratios': [2.2, 1]})
    rates = {}
    for col, (lep, mlep) in enumerate(LEPTONS.items()):
        q2, cen, sd = rate_with_band(vec, cov_lin, mlep, shape=True)
        rates[lep] = (q2, cen, sd)
        ax = axes[0, col]
        ax.fill_between(q2, cen - sd, cen + sd, alpha=0.35, color='C0',
                        label=r'$\pm1\sigma$ form factors (shape only)')
        ax.plot(q2, cen, color='C0', lw=1.8, label='Harrison-2024')
        old20 = np.concatenate([HB.DEFAULT[k] for k in ('avec', 'bvec', 'cvec', 'dvec')])
        c20 = dgamma_dq2(q2, old20, mlep)
        ax.plot(q2, c20 / np.trapezoid(c20, q2), color='C3', lw=1.3, ls='--',
                label='Harrison-2020 (Hammer default)')
        ax.set_ylabel(r'$(1/\Gamma)\,d\Gamma/dq^2$  [GeV$^{-2}$]')
        ax.set_title(r'$B_c \to J/\psi\,%s\,\nu$' % ('\\mu' if lep == 'mu' else '\\tau'))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

        ax = axes[1, col]
        ax.plot(q2, 100 * sd / np.maximum(cen, 1e-300), color='C0', lw=1.6)
        ax.set_xlabel(r'$q^2$ [GeV$^2$]')
        ax.set_ylabel(r'shape band / central [%]')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    fig.suptitle('Differential rate and its form-factor uncertainty')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'ff_q2_spectra.png'), dpi=140)
    plt.close(fig)

    # --- 5. both channels on one plot, and R(J/psi) -------------------------
    rc, rs = r_jpsi_with_error(vec, cov_lin)
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8.5, 8), sharex=True,
                                  gridspec_kw={'height_ratios': [2, 1]})
    for lep, colour, label in (('mu', 'C0', r'$B_c\to J/\psi\,\mu\,\nu$'),
                               ('tau', 'C3', r'$B_c\to J/\psi\,\tau\,\nu$')):
        # absolute: SAME arbitrary units for both, so the ratio of the two areas
        # IS R(J/psi) -- that is the point of putting them on one axis.
        q2, cen, sd = rate_with_band(vec, cov_lin, LEPTONS[lep], shape=False)
        ax.fill_between(q2, cen - sd, cen + sd, alpha=0.30, color=colour)
        ax.plot(q2, cen, color=colour, lw=1.8, label=label)
        q2n, cn, sdn = rate_with_band(vec, cov_lin, LEPTONS[lep], shape=True)
        axr.fill_between(q2n, cn - sdn, cn + sdn, alpha=0.30, color=colour)
        axr.plot(q2n, cn, color=colour, lw=1.8, label=label)

    ax.axvline(M_TAU ** 2, color='0.4', lw=1.0, ls=':')
    ax.annotate(r'$q^2 = m_\tau^2$', xy=(M_TAU ** 2, ax.get_ylim()[1]),
                xytext=(3, -12), textcoords='offset points', fontsize=8, color='0.4')
    ax.set_ylabel(r'$d\Gamma/dq^2$  [common arbitrary units]')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.25)
    ax.text(0.97, 0.05,
            r'$R(J/\psi) = \int\Gamma_\tau / \int\Gamma_\mu = %.4f \pm %.4f$'
            '\n' r'(form-factor uncertainty only)' % (rc, rs),
            transform=ax.transAxes, ha='right', va='bottom', fontsize=11,
            bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.9))

    axr.set_ylabel(r'$(1/\Gamma)\,d\Gamma/dq^2$  [GeV$^{-2}$]')
    axr.set_xlabel(r'$q^2$ [GeV$^2$]')
    axr.grid(alpha=0.25)
    axr.set_title('same, each normalised to unit area (shape comparison)',
                  fontsize=9)
    fig.suptitle(r'$\mu$ and $\tau$ channels, common normalisation')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'ff_q2_both_channels.png'), dpi=140)
    plt.close(fig)

    # --- 6. coefficient covariance: correlations and spectrum ---------------
    order = (['a%d' % n for n in range(4)] + ['b%d' % n for n in range(4)]
             + ['c%d' % n for n in range(3)] + ['d%d' % n for n in range(4)])
    sd = np.sqrt(np.maximum(np.diag(cov_lin), 0.))
    safe = np.where(sd > 0, sd, 1.)
    corr = cov_lin / np.outer(safe, safe)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im = ax1.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_xticks(range(NCOEF)); ax1.set_xticklabels(order, rotation=90, fontsize=7)
    ax1.set_yticks(range(NCOEF)); ax1.set_yticklabels(order, fontsize=7)
    ax1.set_title('coefficient correlation')
    fig.colorbar(im, ax=ax1, fraction=0.046)
    ev = np.sort(np.linalg.eigvalsh(cov_lin))[::-1]
    ax2.semilogy(np.sqrt(np.abs(ev)), 'o-', ms=4)
    ax2.set_xlabel('eigendirection'); ax2.set_ylabel(r'$\sqrt{\lambda}$')
    ax2.set_title('eigenvalue spectrum (near-degenerate pairs at the top)')
    ax2.grid(alpha=0.25, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'ff_coeff_cov.png'), dpi=140)
    plt.close(fig)

    return inp_rel, out_rel, rates


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plots-dir', default='ff_validation')
    ap.add_argument('--toys', type=int, default=0,
                    help='also run N toys as a cross-check of the linear algebra '
                         '(0 = skip; the linear result is the one to use)')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--r-direct', type=int, default=0, metavar='NSAMP',
                    help='TEST 5: propagate the lattice covariance straight to '
                         'R(J/psi) with NSAMP gvar samples, taking the helicity '
                         'curves as tabulated and never going through the BGL '
                         'fit. If this reproduces 0.2597(27), the rate formula '
                         'AND the covariance handling are validated with no fit '
                         'in the loop, and any excess in the fitted R '
                         'uncertainty is attributable to the regularisation.')
    args = ap.parse_args()

    import harrison_ffs as HF
    hc = HF.helicity_curves(GRID)

    vec = central_coeffs(hc)
    cov_lin = coeff_cov_linear(hc)

    if args.r_direct:
        # R straight from the curves: no fit, no coefficients, no Hammer
        curves_mean = dict((k, gv.mean(hc[k])) for k in FFS)
        r0 = FR.r_jpsi_from_curves(GRID, curves_mean)
        gv.ranseed(args.seed)
        samples = []
        for d in gv.raniter(hc, n=args.r_direct):
            samples.append(FR.r_jpsi_from_curves(GRID, dict((k, d[k]) for k in FFS)))
        samples = np.array(samples)
        print('\nTEST 5 -- R(J/psi) with no fit in the loop')
        print('  from the lattice curves directly : %.4f +/- %.4f  (%d samples)'
              % (r0, samples.std(ddof=1), args.r_direct))
        rc_fit, rs_fit = r_jpsi_with_error(vec, cov_lin)
        print('  through the BGL fit              : %.4f +/- %.4f'
              % (rc_fit, rs_fit))
        print('  published (arXiv:2503.15090)     : 0.2597 +/- 0.0027')
        print('  -> central values agreeing means the fit is faithful; an '
              'inflated\n     uncertainty through the fit points at the '
              'Tikhonov prior, not at a bug.')

    toys = None
    if args.toys:
        cov_toy, toys = coeff_cov_toys(hc, args.toys, args.seed)
        num = np.abs(cov_toy - cov_lin).max()
        den = np.abs(cov_lin).max()
        print('toys vs linear: max |dC| / max|C| = %.3e  (expect ~sqrt(2/N) = %.3e)'
              % (num / den, np.sqrt(2. / args.toys)))

    os.makedirs(args.plots_dir, exist_ok=True)
    inp_rel, out_rel, rates = make_plots(hc, vec, cov_lin, toys, args.plots_dir)

    print('\nrelative uncertainty, input (lattice) -> output (after the fit):')
    print('   q2  ' + '  '.join('%18s' % n for n in FFS))
    for i in (0, NQ // 4, NQ // 2, 3 * NQ // 4, NQ - 1):
        row = '  '.join('%7.2f%% ->%7.2f%%' % (100 * inp_rel[n][i], 100 * out_rel[n][i])
                        for n in FFS)
        print(' %5.2f  %s' % (GRID[i], row))
    print('\nrate-level form-factor uncertainty:')
    for lep in LEPTONS:
        q2, cen, sd = rates[lep]
        rel = sd / np.maximum(cen, 1e-300)
        norm0 = np.trapezoid(dgamma_dq2(q2, vec, LEPTONS[lep]), q2)
        tot = np.sqrt(sum((np.trapezoid(dgamma_dq2(q2, vec + d, LEPTONS[lep]), q2)
                           - norm0) ** 2
                          for d in _sigma_directions(cov_lin).T))
        print('  %-4s shape band %4.2f%% .. %4.2f%%   normalisation %4.2f%% '
              '(absorbed by bc_norm)'
              % (lep, 100 * rel.min(), 100 * rel.max(), 100 * tot / norm0))
    rc, rs = r_jpsi_with_error(vec, cov_lin)
    print('  R(J/psi) from these form factors = %.4f +/- %.4f  '
          '(lattice quotes 0.2597 +/- 0.0027)' % (rc, rs))

    worst = max((out_rel[n] / np.maximum(inp_rel[n], 1e-30)).max() for n in FFS)
    print('\nlargest output/input ratio anywhere: %.2f' % worst)
    print('A least-squares fit projects onto the model space, so this should be '
          '<= 1 wherever\nthe model can represent the curve. Above 1 means the '
          'fit is manufacturing uncertainty.')
    print('\nplots -> %s/' % args.plots_dir)


if __name__ == '__main__':
    main()
