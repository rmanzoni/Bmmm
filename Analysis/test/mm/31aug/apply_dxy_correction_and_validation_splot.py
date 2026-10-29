#!/usr/bin/env python3
"""
Apply the sigma_dxy quantile-morphing correction (derived on the PROBE muon,
mu2) to the TAG muon (mu1), and compare data vs. MC-uncorrected vs.
MC-corrected.

This is a transfer / closure test: the map is keyed only on (pt, |eta|, sigma),
so evaluating it with mu1's kinematics and mu1's sigma_dxy tells us whether a
probe-derived correction also fixes an independent, harder muon leg. If the
corrected MC lands on data for mu1, the correction behaves as a universal
detector-resolution object (good for other analyses); if not, there is
leg/selection dependence to understand.

Only MC is transformed; data is left untouched. The correction is applied
per-event:

    sigma_corr = corr.evaluate(mu1_pt, |mu1_eta|, mu1_sigma_dxy)

Same tag-and-probe selection as before.



python3 apply_dxy_correction_and_validation_splot.py \
    --data charmonium_2018_24may2026.root --mc hb_2018_24may2026.root \
    --json pseudo_proper_decay_length_100microns/sigma_dxy_morph_splot.json --name mu_bs_dxy_e_morph \
    --mu1-sigma mu1_bs_dxy_e
    
    
"""

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import uproot
import correctionlib

try:
    import mplhep as hep

    plt.style.use(hep.style.CMS)
    HAVE_HEP = True
except Exception:
    HAVE_HEP = False


TREE = "tree"
JPSI_MASS = 3.0969

# Must match the grid the correction was derived on.
# ETA_BINS = [0.0, 0.4, 0.8, 1.2, 1.5]
# PT_BINS = [3.5, 5.0, 7.0, 10.0, 20.0]
ETA_BINS = [0.0, 0.85, 1.2, 1.5]
PT_BINS = [3.5, 5.0, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 12.0, 13.0,
           14.0, 15.0, 17.0, 20.0, 30.0, 500.0]

# Sigma-axis binning for the comparison plots: (lo, hi, nbins) or
# (None, None, nbins) for a [0.5, 99.5]-percentile auto-range.
SIGMA_SPEC = (None, None, 50)

# Branches used by the selection (mu2 = probe) plus the mu1 quantities we plot.
SEL_BRANCHES = [
    "mass", "charge", "vtx_prob", "mu1_bs_dxy_sig", "lxy",
    "mu1_HLT_Mu7p5_Track3p5_Jpsi_tag", "mu1_id_medium",
    "mu2_id_medium", "mu2_HLT_Mu7p5_Track3p5_Jpsi_probe",
    "mu2_eta", "cos2d", "pt",
]

DATA_KW = dict(color="black", marker="o", ms=4, ls="none", label="Data")
RAW_KW = dict(color="#9ec9e2", label="MC (uncorr.)")
COR_KW = dict(color="#c1272d", label="MC (corrected)")


# ----------------------------------------------------------------------------
def load(path, branches, tree=TREE):
    with uproot.open(f"{path}:{tree}") as t:
        present = [b for b in branches if b in t.keys()]
        missing = [b for b in branches if b not in t.keys()]
        if missing:
            print(f"  [warn] {os.path.basename(path)} missing: {missing}")
        return t.arrays(present, library="np")


def selection_mask(a):
    m = np.abs(a["mu2_eta"]) < 1.5
    m &= np.abs(a["mass"] - JPSI_MASS) < 0.1
    m &= a["charge"] == 0
    m &= a["cos2d"] > 0.9
    m &= a["vtx_prob"] > 0.01
    m &= (a["lxy"] * a["cos2d"] * JPSI_MASS / a["pt"]) > 0.01   # was: a["lxy"] > 0.03
    m &= a["mu1_HLT_Mu7p5_Track3p5_Jpsi_tag"].astype(bool)
    m &= a["mu1_id_medium"].astype(bool)
    m &= a["mu2_id_medium"].astype(bool)
    m &= a["mu2_HLT_Mu7p5_Track3p5_Jpsi_probe"] > 0.5
    return m


def resolve_edges(spec, *arrays):
    if isinstance(spec, np.ndarray):
        return spec
    lo, hi, nb = spec
    if lo is None or hi is None:
        pool = np.concatenate([a for a in arrays if len(a)])
        pool = pool[np.isfinite(pool)]
        lo = np.percentile(pool, 0.5) if lo is None else lo
        hi = np.percentile(pool, 99.5) if hi is None else hi
    return np.linspace(lo, hi, int(nb) + 1)


def hist_err(vals, edges):
    c, _ = np.histogram(vals, bins=edges)
    return c.astype(float), np.sqrt(c)


def _norm_to(ref, h):
    s = h.sum()
    return (h * ref.sum() / s) if s > 0 else h


# ----------------------------------------------------------------------------
def draw_inclusive(xlabel, d_sig, m_sig, mc_corr, spec, outdir):
    edges = resolve_edges(spec, d_sig, m_sig, mc_corr)
    ctr = 0.5 * (edges[:-1] + edges[1:])

    d, d_err = hist_err(d_sig, edges)
    m, _ = hist_err(m_sig, edges)
    mc, _ = hist_err(mc_corr, edges)
    m = _norm_to(d, m)
    mc = _norm_to(d, mc)

    fig = plt.figure(figsize=(7, 7.2))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax, axr = fig.add_subplot(gs[0]), None
    axr = fig.add_subplot(gs[1], sharex=ax)

    ax.stairs(m, edges, fill=True, alpha=0.5, color=RAW_KW["color"],
              label=RAW_KW["label"])
    ax.stairs(mc, edges, lw=1.8, color=COR_KW["color"], label=COR_KW["label"])
    ax.errorbar(ctr, d, yerr=d_err, **DATA_KW)
    ax.set_ylabel("Events")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    plt.setp(ax.get_xticklabels(), visible=False)
    if HAVE_HEP:
        hep.cms.label("Preliminary", ax=ax, data=True,
                      rlabel="2018 (13 TeV)", fontsize=13)

    def ratio(num, den):
        return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

    axr.errorbar(ctr, ratio(d, m), yerr=ratio(d_err, m), color=RAW_KW["color"],
                 marker="o", ms=3, ls="none", label="/ uncorr.")
    axr.errorbar(ctr, ratio(d, mc), yerr=ratio(d_err, mc), color=COR_KW["color"],
                 marker="o", ms=3, ls="none", label="/ corrected")
    axr.axhline(1.0, color="black", lw=0.8)
    axr.set_ylim(0.0, 2.0)
    axr.set_ylabel("Data / MC")
    axr.set_xlabel(xlabel)
    axr.set_xlim(edges[0], edges[-1])
    axr.legend(loc="best", fontsize=8, ncol=2)

    out = os.path.join(outdir, "mu1_sigma_dxy_morph_inclusive.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def draw_grid(xlabel, pt, eta, d_sig, m_pt, m_eta, m_sig, mc_corr, spec, outdir):
    edges = resolve_edges(spec, d_sig, m_sig, mc_corr)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    nrow, ncol = len(ETA_BINS) - 1, len(PT_BINS) - 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow),
                             sharex=True, squeeze=False)
    for i in range(nrow):
        e0, e1 = ETA_BINS[i], ETA_BINS[i + 1]
        for j in range(ncol):
            p0, p1 = PT_BINS[j], PT_BINS[j + 1]
            ax = axes[i][j]
            dsel = (np.abs(eta) >= e0) & (np.abs(eta) < e1) & (pt >= p0) & (pt < p1)
            msel = (np.abs(m_eta) >= e0) & (np.abs(m_eta) < e1) \
                & (m_pt >= p0) & (m_pt < p1)

            d, d_err = hist_err(d_sig[dsel], edges)
            m, _ = hist_err(m_sig[msel], edges)
            mc, _ = hist_err(mc_corr[msel], edges)

            def unit(h):
                s = h.sum()
                return h / s if s > 0 else h

            d, m, mc = unit(d), unit(m), unit(mc)
            ax.stairs(m, edges, fill=True, alpha=0.5, color=RAW_KW["color"])
            ax.stairs(mc, edges, lw=1.5, color=COR_KW["color"])
            ax.errorbar(ctr, d, yerr=0, color="black", marker="o", ms=2.2,
                        ls="none")
            ax.set_title(rf"$|\eta|\in[{e0:g},{e1:g})$, "
                         rf"$p_T\in[{p0:g},{p1:g})$", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_ylim(bottom=0)
            if i == nrow - 1:
                ax.set_xlabel(xlabel, fontsize=8)

    handles = [
        plt.Line2D([], [], color="black", marker="o", ms=4, ls="none",
                   label="Data"),
        plt.Rectangle((0, 0), 1, 1, fc=RAW_KW["color"], alpha=0.5,
                      label=RAW_KW["label"]),
        plt.Line2D([], [], color=COR_KW["color"], lw=2, label=COR_KW["label"]),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    fig.suptitle("mu1 sigma_dxy: data vs MC (uncorr.) vs MC (corrected)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(outdir, "mu1_sigma_dxy_morph_grid.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="charmonium_2018_24may2026.root")
    ap.add_argument("--mc", default="hb_2018_24may2026.root")
    ap.add_argument("--tree", default=TREE)
    ap.add_argument("--json", default="sigma_dxy_morph_v2.json")
    ap.add_argument("--name", default="mu2_bs_dxy_e_morph")
    ap.add_argument("--mu1-pt", default="mu1_pt")
    ap.add_argument("--mu1-eta", default="mu1_eta")
    ap.add_argument("--mu1-sigma", default="mu1_bs_dxy_e")
    ap.add_argument("--outdir", default="mu1_morph_plots")
    ap.add_argument("--xlabel", default=r"$|\sigma_{d_{xy}}|\ (\mu_1)$ [cm]")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    branches = SEL_BRANCHES + [args.mu1_pt, args.mu1_eta, args.mu1_sigma]

    data = load(args.data, branches, args.tree)
    mc = load(args.mc, branches, args.tree)
    dm, mm = selection_mask(data), selection_mask(mc)
    print(f"Data: {dm.sum():,} pass; MC: {mm.sum():,} pass")

    d_pt = data[args.mu1_pt][dm]
    d_eta = data[args.mu1_eta][dm]
    d_sig = np.abs(data[args.mu1_sigma][dm])

    m_pt = mc[args.mu1_pt][mm]
    m_eta = mc[args.mu1_eta][mm]
    m_sig = np.abs(mc[args.mu1_sigma][mm])

    # Apply the correction to MC's mu1, per-event and vectorised.
    corr = correctionlib.CorrectionSet.from_file(args.json)[args.name]
    mc_corr = corr.evaluate(m_pt.astype(float),
                            np.abs(m_eta).astype(float),
                            m_sig.astype(float))

    print(f"Applied '{args.name}' to {m_sig.size:,} mu1 in MC "
          f"(mean sigma {m_sig.mean():.5f} -> {mc_corr.mean():.5f})")

    draw_inclusive(args.xlabel, d_sig, m_sig, mc_corr, SIGMA_SPEC, args.outdir)
    draw_grid(args.xlabel, d_pt, d_eta, d_sig, m_pt, m_eta, m_sig, mc_corr,
              SIGMA_SPEC, args.outdir)
    print(f"Done -> {args.outdir}/")


if __name__ == "__main__":
    main()