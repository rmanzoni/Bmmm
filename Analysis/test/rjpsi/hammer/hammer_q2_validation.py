#!/usr/bin/env python3
'''
The q2 validation plots, but built on REAL Hammer weights instead of the Python
port of its evaluator.

ff_fit_validation.py answers "do the refitted coefficients reproduce Harrison?"
entirely inside our own port -- it matches Fig. 6 of arXiv:2503.15090, but
Hammer's C++ never ran. This script answers the remaining question: "does Hammer
itself, fed those coefficients, produce the same spectra?" It takes signal MC
with the gen_ham_* leaves, asks Hammer for the weights, and histograms the
generated q2:

  * unweighted  -> the spectrum as generated, i.e. Kiselev
  * weighted    -> the reweighted spectrum, i.e. Harrison-2024 according to Hammer
  * overlaid    -> the analytic Harrison curve from the same FF card, via ff_rate

If the weighted histogram sits on the analytic curve, the last unvalidated link
is closed: option strings parsed, BctoJpsiBGLVar took the coefficients in the
convention we assumed, and Hammer's evaluator agrees with the port.

    python3 hammer_q2_validation.py signal.root -o plots
    python3 hammer_q2_validation.py signal.root -o plots --weight-branch hammer_weight

IMPORTANT -- what the panels can and cannot tell you:
  The top panels compare a RECONSTRUCTED, SELECTED sample against a gen-level
  analytic curve. Acceptance distorts the spectrum, so they agree only for a
  sample without gen filtering or reco selection. On a selected sample, read the
  bottom panels instead: <w> as a function of q2 is a per-event ratio and is
  acceptance-insensitive to first order, so it is the meaningful comparison and
  should follow the analytic Harrison/Kiselev ratio in shape.
'''
import argparse
import math
import os

import numpy as np
import uproot

from Bmmm.Analysis.HammerFF import (
    BRANCH_NAMES, BRANCH_NOMINAL, BRANCH_STATUS, INPUT_BRANCHES, MU_CODE,
    STATUS_OK, TAU_CODE, VAR_LABELS, leaves_from_row, make_hammer_session,
)

import ff_rate as FR
import hammer_bgl_forward as HB

CHANNELS = (('mu', MU_CODE, FR.M_MU, r'$B_c\to J/\psi\,\mu\,\nu$'),
            ('tau', TAU_CODE, FR.M_TAU, r'$B_c\to J/\psi\,\tau\,\nu$'))


def gen_q2(row, i):
    '''q2 = (p_Bc - p_Jpsi)^2 from the stored gen leaves, J/psi = mu+ mu-.'''
    def p4(prefix):
        pt = float(row['%s_pt' % prefix][i])
        eta = float(row['%s_eta' % prefix][i])
        phi = float(row['%s_phi' % prefix][i])
        mass = float(row['%s_mass' % prefix][i])
        px, py = pt * math.cos(phi), pt * math.sin(phi)
        pz = pt * math.sinh(eta)
        return np.array([math.sqrt(px * px + py * py + pz * pz + mass * mass),
                         px, py, pz])

    q = p4('gen_b') - (p4('gen_ham_mup') + p4('gen_ham_mum'))
    return q[0] ** 2 - q[1] ** 2 - q[2] ** 2 - q[3] ** 2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('-o', '--plots-dir', default='hammer_q2_validation')
    ap.add_argument('-t', '--tree', default='tree')
    ap.add_argument('--card', default='default:allow-stale')
    ap.add_argument('--weight-branch', default=None,
                    help='read the weights off the tree (written by --hammer or '
                         'add_hammer_weights.py) instead of recomputing them; the '
                         'variation branches are picked up alongside')
    ap.add_argument('--bins', type=int, default=30)
    ap.add_argument('-n', '--max-events', type=int, default=None)
    ap.add_argument('--chunk', type=int, default=20000)
    args = ap.parse_args()

    precomputed = args.weight_branch is not None
    read = list(INPUT_BRANCHES) + (list(BRANCH_NAMES) if precomputed else [])
    ses = None if precomputed else make_hammer_session(args.card)

    # the card is needed either way, for the analytic overlay
    from Bmmm.Analysis.HammerFF import load_card, DEFAULT_CARD
    card_path = args.card.split(':')[0]
    card = load_card(DEFAULT_CARD if card_path in ('', 'default') else card_path)
    vec = FR.coeffs_from_card(card)
    dirs = FR.sigma_directions_from_card(card)

    acc = dict((name, {'q2': [], 'w': [], 'var': []}) for name, _, _, _ in CHANNELS)
    nrow = 0
    src = ['%s:%s' % (p, args.tree) for p in args.inputs]
    for row in uproot.iterate(src, expressions=read, step_size=args.chunk,
                              library='np'):
        for i in range(len(row['gen_bc_decay'])):
            if args.max_events is not None and nrow >= args.max_events:
                break
            nrow += 1
            code = row['gen_bc_decay'][i]
            if not np.isfinite(code):
                continue
            code = int(round(float(code)))
            name = {MU_CODE: 'mu', TAU_CODE: 'tau'}.get(code)
            if name is None:
                continue

            if precomputed:
                if row[BRANCH_STATUS][i] != STATUS_OK:
                    continue
                w = float(row[args.weight_branch][i])
                var = [float(row['hammer_ff_%s' % lab][i]) for lab in VAR_LABELS]
            else:
                out = ses.weights(leaves_from_row(row, i))
                if out[BRANCH_STATUS] != STATUS_OK:
                    continue
                w = out[BRANCH_NOMINAL]
                var = [out['hammer_ff_%s' % lab] for lab in VAR_LABELS]
            if not math.isfinite(w):
                continue

            acc[name]['q2'].append(gen_q2(row, i))
            acc[name]['w'].append(w)
            acc[name]['var'].append(var)
        if args.max_events is not None and nrow >= args.max_events:
            break

    if ses is not None:
        print(ses.summary())

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(args.plots_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                             gridspec_kw={'height_ratios': [2, 1]})

    for col, (name, code, mlep, label) in enumerate(CHANNELS):
        q2 = np.asarray(acc[name]['q2'], dtype=float)
        w = np.asarray(acc[name]['w'], dtype=float)
        var = np.asarray(acc[name]['var'], dtype=float)
        ax, axr = axes[0, col], axes[1, col]
        if q2.size == 0:
            ax.text(0.5, 0.5, 'no %s events' % name, ha='center',
                    transform=ax.transAxes)
            continue

        lo = mlep ** 2
        hi = (HB.MBC - HB.MJPSI) ** 2
        edges = np.linspace(lo, hi, args.bins + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]

        kis, _ = np.histogram(q2, bins=edges)
        har, _ = np.histogram(q2, bins=edges, weights=w)
        har_sq, _ = np.histogram(q2, bins=edges, weights=w ** 2)
        kis_n = kis / max(kis.sum() * width, 1e-300)
        norm = max(har.sum() * width, 1e-300)
        har_n = har / norm
        # MC statistical error of a weighted histogram. Without this the ratio
        # panel looks like disagreement when it is just finite statistics.
        stat = np.sqrt(har_sq) / norm
        filled = kis > 0

        # FF band from the variation weights, shape-normalised like the templates
        band = np.zeros_like(har_n)
        for j in range(var.shape[1] // 2):
            up, _ = np.histogram(q2, bins=edges, weights=var[:, 2 * j])
            dn, _ = np.histogram(q2, bins=edges, weights=var[:, 2 * j + 1])
            up = up / max(up.sum() * width, 1e-300)
            dn = dn / max(dn.sum() * width, 1e-300)
            band += (0.5 * np.abs(up - dn)) ** 2
        band = np.sqrt(band)

        ax.step(centres, kis_n, where='mid', color='0.45', lw=1.4,
                label='MC as generated (Kiselev)')
        ax.fill_between(centres, har_n - band, har_n + band, step='mid',
                        alpha=0.30, color='C0', label=r'FF $\pm1\sigma$ (shape)')
        ax.errorbar(centres[filled], har_n[filled], yerr=stat[filled], fmt='o',
                    ms=3.5, color='C0',
                    label='Hammer-reweighted (Harrison-2024), MC stat')
        grid = FR.rate_grid(mlep, 400)
        ana = FR.dgamma_dq2(grid, vec, mlep)
        ax.plot(grid, ana / np.trapezoid(ana, grid), color='C3', lw=1.6, ls='--',
                label='analytic, same FF card')
        ax.set_title(label)
        ax.set_ylabel(r'$(1/\Gamma)\,d\Gamma/dq^2$  [GeV$^{-2}$]')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

        ana_binned = np.array([np.trapezoid(FR.dgamma_dq2(np.linspace(a, b, 21),
                                                          vec, mlep),
                                            np.linspace(a, b, 21)) / width
                               for a, b in zip(edges[:-1], edges[1:])])
        ana_binned /= max(np.sum(ana_binned) * width, 1e-300)
        good = (ana_binned > 0) & filled
        ratio = np.where(ana_binned > 0, har_n / np.maximum(ana_binned, 1e-300), 0.)
        axr.axhline(1., color='k', lw=0.8)
        axr.fill_between(centres[good], (1. - band / ana_binned)[good],
                         (1. + band / ana_binned)[good], step='mid', alpha=0.30,
                         color='C0')
        axr.errorbar(centres[good], ratio[good], yerr=(stat / ana_binned)[good],
                     fmt='o', ms=3.5, color='C0')
        axr.set_ylabel('Hammer / analytic')
        axr.set_xlabel(r'$q^2$ [GeV$^2$]')
        axr.grid(alpha=0.25)

        # pull against MC statistics: this is the number that says whether
        # Hammer and the analytic curve agree, not the raw deviation
        pull = ((ratio - 1.) / np.maximum(stat / np.maximum(ana_binned, 1e-300),
                                          1e-300))[good]
        chi2 = float(np.sum(pull ** 2))
        print('[%-3s] %6d events, <w> = %.4f, chi2/ndf vs analytic = %.2f/%d = %.2f'
              % (name, q2.size, w.mean(), chi2, pull.size,
                 chi2 / max(pull.size, 1)))

    fig.suptitle('q2 spectra from Hammer weights vs the analytic form factors')
    fig.tight_layout()
    out = os.path.join(args.plots_dir, 'hammer_q2_spectra.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)

    # --- <w> vs q2: acceptance-insensitive, the meaningful check on a selected sample
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, code, mlep, label in CHANNELS:
        q2 = np.asarray(acc[name]['q2'], dtype=float)
        w = np.asarray(acc[name]['w'], dtype=float)
        if q2.size == 0:
            continue
        edges = np.linspace(mlep ** 2, (HB.MBC - HB.MJPSI) ** 2, args.bins + 1)
        n, _ = np.histogram(q2, bins=edges)
        sw, _ = np.histogram(q2, bins=edges, weights=w)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ok = n > 0
        ax.plot(centres[ok], (sw[ok] / n[ok]), 'o-', ms=3.5, label=label)
    ax.set_xlabel(r'$q^2$ [GeV$^2$]')
    ax.set_ylabel(r'$\langle w \rangle$  (Kiselev $\to$ Harrison)')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.plots_dir, 'hammer_mean_weight.png'), dpi=140)
    plt.close(fig)

    print('plots -> %s/' % args.plots_dir)


if __name__ == '__main__':
    main()
