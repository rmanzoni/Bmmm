#!/usr/bin/env python3
"""
Tag-and-probe data/MC validation of the probe-muon (mu2) transverse impact
parameter, in the 2018 charmonium sample vs. Hb MC.

Reads two ROOT files with a TTree "tree", applies a common J/psi tag-and-probe
selection, and produces data-vs-MC comparisons of

    |mu2_bs_dxy_sig| , |mu2_bs_dxy| , |mu2_bs_dxy_e|

both inclusively (with a data/MC ratio panel) and in a grid of
|mu2_eta| x mu2_pt bins. A handful of validation plots (mass, mu2_pt, mu2_eta)
are produced as well.

I/O is via uproot so no compiled ROOT build is needed. If you'd rather stay in
PyROOT, the only thing to swap is load_branches(); everything downstream works
on plain numpy arrays.
"""

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless / batch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import uproot

try:
    import mplhep as hep

    plt.style.use(hep.style.CMS)
    HAVE_HEP = True
except Exception:  # mplhep optional
    HAVE_HEP = False


# ----------------------------------------------------------------------------
# Configuration  (edit here)
# ----------------------------------------------------------------------------
TREE = "tree"

# Branches we actually need to pull off disk (selection + plotting + binning).
NEEDED = [
    "mass", "charge", "vtx_prob", "vtx_chi2", "lxy",
    "mu1_HLT_Mu7p5_Track3p5_Jpsi_tag", "mu1_id_medium",
    "mu2_id_medium", "mu2_HLT_Mu7p5_Track3p5_Jpsi_probe",
    "mu2_eta", "mu2_pt",
    "mu2_bs_dxy_sig", "mu2_bs_dxy", "mu2_bs_dxy_e",
]

# Optional per-event MC weight branch. Set to None if there isn't one.
MC_WEIGHT = None  # e.g. "weight" or "puWeight"

JPSI_MASS = 3.0969

# Grid binning. Edges are inclusive-low, exclusive-high; last bin includes top.
ETA_BINS = [0.0, 0.85, 1.2, 1.5]        # on |mu2_eta|
PT_BINS = [3.5, 5.0, 7.0, 8.0, 10.0, 15.0, 20.0, 50.0]       # on mu2_pt [GeV]

# Variables to show. spec is one of:
#   np.ndarray                -> explicit bin edges
#   (lo, hi, nbins)           -> fixed uniform binning
#   (None, None, nbins)       -> auto range from [0.5, 99.5] percentiles of data+MC
# transform maps the raw array dict to the plotted quantity.

nbins = 400 # many bins to help quantile regression
DCA_VARS = {
    "abs_mu2_bs_dxy_sig": dict(
        transform=lambda a: np.abs(a["mu2_bs_dxy_sig"]),
        xlabel=r"$|d_{xy}/\sigma_{d_{xy}}|\ (\mu_2)$",
        spec=(0.0, 100.0, nbins),
    ),
    "abs_mu2_bs_dxy": dict(
        transform=lambda a: np.abs(a["mu2_bs_dxy"]),
        xlabel=r"$|d_{xy}|\ (\mu_2)$ [cm]",
        spec=(None, None, nbins),
    ),
    "abs_mu2_bs_dxy_e": dict(
        transform=lambda a: np.abs(a["mu2_bs_dxy_e"]),
        xlabel=r"$|\sigma_{d_{xy}}|\ (\mu_2)$ [cm]",
        spec=(None, None, nbins),
    ),

    "vtx_chi2": dict(
        transform=lambda a: a["vtx_chi2"],
        xlabel=r"vtx $\chi^2$",
        spec=(None, None, nbins),
    ),
    
    "vtx_prob": dict(
        transform=lambda a: a["vtx_prob"],
        xlabel=r"vtx p-value",
        spec=(0, 1, nbins),
    ),


}

VALIDATION_VARS = {
    "mass": dict(
        transform=lambda a: a["mass"],
        xlabel=r"$m(\mu\mu)$ [GeV]",
        spec=(JPSI_MASS - 0.1, JPSI_MASS + 0.1, 50),
    ),
    "mu2_pt": dict(
        transform=lambda a: a["mu2_pt"],
        xlabel=r"$p_{T}(\mu_2)$ [GeV]",
        spec=(0.0, 25.0, 50),
    ),
    "mu2_eta": dict(
        transform=lambda a: a["mu2_eta"],
        xlabel=r"$\eta(\mu_2)$",
        spec=(-2.0, 2.0, 40),
    ),
}

# Which variables get the eta-pt grid (the DCA ones by default).
GRID_VARS = list(DCA_VARS.keys())

DATA_KW = dict(color="black", marker="o", ms=4, ls="none", label="Data")
MC_KW = dict(color="#3f7fbf", label="Hb MC")


# ----------------------------------------------------------------------------
# I/O and selection
# ----------------------------------------------------------------------------
def load_branches(path, branches, tree=TREE):
    """Return a dict {branch: np.ndarray}. Swap this out for PyROOT if desired."""
    with uproot.open(f"{path}:{tree}") as t:
        present = [b for b in branches if b in t.keys()]
        missing = [b for b in branches if b not in t.keys()]
        if missing:
            print(f"  [warn] {os.path.basename(path)} missing branches: {missing}")
        return t.arrays(present, library="np")


def selection_mask(a):
    """The common tag-and-probe J/psi selection, as boolean numpy mask.

    abs(mu2_eta) < 1.5
    & mu1_HLT_Mu7p5_Track3p5_Jpsi_tag & mu1_id_medium & mu2_id_medium
    & mu2_HLT_Mu7p5_Track3p5_Jpsi_probe > 0.5
    & abs(mass - 3.0969) < 0.1 & charge == 0 & vtx_prob > 0.01
    """
    m = np.abs(a["mu2_eta"]) < 1.5
    m &= a["mu1_HLT_Mu7p5_Track3p5_Jpsi_tag"].astype(bool)
    m &= a["mu1_id_medium"].astype(bool)
    m &= a["mu2_id_medium"].astype(bool)
    m &= a["mu2_HLT_Mu7p5_Track3p5_Jpsi_probe"] > 0.5
    m &= np.abs(a["mass"] - JPSI_MASS) < 0.1
    m &= a["charge"] == 0
    m &= a["vtx_prob"] > 0.01
    m &= a["lxy"] > 0.03
    return m


def apply_mask(a, mask):
    return {k: v[mask] for k, v in a.items()}


# ----------------------------------------------------------------------------
# Histogramming helpers
# ----------------------------------------------------------------------------
def resolve_edges(spec, *value_arrays):
    """Turn a spec into concrete bin edges."""
    if isinstance(spec, np.ndarray):
        return spec
    lo, hi, nbins = spec
    if lo is None or hi is None:
        pool = np.concatenate([v for v in value_arrays if len(v)])
        pool = pool[np.isfinite(pool)]
        if len(pool) == 0:
            lo, hi = 0.0, 1.0
        else:
            lo = np.percentile(pool, 0.5) if lo is None else lo
            hi = np.percentile(pool, 99.5) if hi is None else hi
        if hi <= lo:
            hi = lo + 1.0
    return np.linspace(lo, hi, int(nbins) + 1)


def hist_err(vals, edges, weights=None):
    """Histogram counts and (sqrt-sumw2) errors."""
    counts, _ = np.histogram(vals, bins=edges, weights=weights)
    if weights is None:
        err = np.sqrt(counts)
    else:
        sumw2, _ = np.histogram(vals, bins=edges, weights=weights ** 2)
        err = np.sqrt(sumw2)
    return counts.astype(float), err.astype(float)


def norm_mc_to_data(mc, mc_err, data):
    """Area-normalise MC to the data yield (shape comparison)."""
    s = mc.sum()
    if s <= 0:
        return mc, mc_err, 1.0
    scale = data.sum() / s
    return mc * scale, mc_err * scale, scale


# ----------------------------------------------------------------------------
# Plotters
# ----------------------------------------------------------------------------
def draw_comparison(name, xlabel, data_vals, mc_vals, spec,
                    mc_weights=None, outdir=".", cms_label="2018 (13 TeV)",
                    store=None):
    """Inclusive data/MC overlay with a data/MC ratio panel."""
    edges = resolve_edges(spec, data_vals, mc_vals)
    centers = 0.5 * (edges[:-1] + edges[1:])

    d, d_err = hist_err(data_vals, edges)
    m_raw, m_raw_err = hist_err(mc_vals, edges, weights=mc_weights)
    if store is not None:
        store.add(name, "inclusive", edges, (d, d_err), (m_raw, m_raw_err))

    # Normalised copy for the plot only.
    m, m_err, scale = norm_mc_to_data(m_raw.copy(), m_raw_err.copy(), d)

    fig = plt.figure(figsize=(7, 7))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    # MC as a filled step (labelled) with a solid edge (unlabelled); data points.
    ax.stairs(m, edges, fill=True, alpha=0.35, **_step_kw(MC_KW))
    ax.stairs(m, edges, **{k: v for k, v in _step_kw(MC_KW).items()
                           if k != "label"})
    ax.errorbar(centers, d, yerr=d_err, **DATA_KW)
    ax.set_ylabel("Events")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    plt.setp(ax.get_xticklabels(), visible=False)
    if HAVE_HEP:
        hep.cms.label("Preliminary", ax=ax, data=True,
                      rlabel=cms_label, fontsize=13)

    # Ratio.
    ratio = np.divide(d, m, out=np.full_like(d, np.nan), where=m > 0)
    ratio_err = np.divide(d_err, m, out=np.full_like(d, np.nan), where=m > 0)
    axr.errorbar(centers, ratio, yerr=ratio_err, **DATA_KW)
    axr.axhline(1.0, color=MC_KW["color"], lw=1)
    axr.set_ylim(0.0, 2.0)
    axr.set_ylabel("Data / MC")
    axr.set_xlabel(xlabel)
    axr.set_xlim(edges[0], edges[-1])

    fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=140, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def draw_grid(name, xlabel, data, mc, spec, mc_weights=None, outdir=".",
              store=None):
    """Grid of |mu2_eta| (rows) x mu2_pt (cols) data/MC overlays, area-normalised
    per cell."""
    d_eta = np.abs(data["mu2_eta"])
    d_pt = data["mu2_pt"]
    m_eta = np.abs(mc["mu2_eta"])
    m_pt = mc["mu2_pt"]
    d_v = DCA_VARS.get(name, VALIDATION_VARS.get(name))["transform"](data)
    m_v = DCA_VARS.get(name, VALIDATION_VARS.get(name))["transform"](mc)

    # Shared binning across cells so shapes are comparable.
    edges = resolve_edges(spec, d_v, m_v)
    centers = 0.5 * (edges[:-1] + edges[1:])

    nrow = len(ETA_BINS) - 1
    ncol = len(PT_BINS) - 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow),
                             sharex=True, squeeze=False)

    for i in range(nrow):
        eta_lo, eta_hi = ETA_BINS[i], ETA_BINS[i + 1]
        d_eta_sel = (d_eta >= eta_lo) & (d_eta < eta_hi)
        m_eta_sel = (m_eta >= eta_lo) & (m_eta < eta_hi)
        for j in range(ncol):
            pt_lo, pt_hi = PT_BINS[j], PT_BINS[j + 1]
            ax = axes[i][j]
            dsel = d_eta_sel & (d_pt >= pt_lo) & (d_pt < pt_hi)
            msel = m_eta_sel & (m_pt >= pt_lo) & (m_pt < pt_hi)

            d, d_err = hist_err(d_v[dsel], edges)
            w = mc_weights[msel] if mc_weights is not None else None
            m_raw, m_raw_err = hist_err(m_v[msel], edges, weights=w)
            if store is not None:
                tag = "eta_%s_%s__pt_%s_%s" % (_num(eta_lo), _num(eta_hi),
                                               _num(pt_lo), _num(pt_hi))
                store.add(name, tag, edges, (d, d_err), (m_raw, m_raw_err))
            m, m_err, _ = norm_mc_to_data(m_raw.copy(), m_raw_err.copy(), d)

            ax.stairs(m, edges, fill=True, alpha=0.35, **_step_kw(MC_KW))
            ax.errorbar(centers, d, yerr=d_err, ms=2.5, color="black",
                        marker="o", ls="none")
            ax.set_ylim(bottom=0)
            ax.set_title(rf"$|\eta|\in[{eta_lo:g},{eta_hi:g})$, "
                         rf"$p_T\in[{pt_lo:g},{pt_hi:g})$", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == nrow - 1:
                ax.set_xlabel(xlabel, fontsize=8)

    handles = [plt.Line2D([], [], **{k: v for k, v in DATA_KW.items()
                                     if k != "label"}, label="Data"),
               plt.Rectangle((0, 0), 1, 1, fc=MC_KW["color"], alpha=0.35,
                             label=MC_KW["label"])]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    fig.suptitle(name, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(outdir, f"{name}_grid.png"), dpi=130, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{name}_grid.pdf"), bbox_inches="tight")
    plt.close(fig)


def _step_kw(kw):
    """stairs() doesn't take marker/ms; strip them."""
    return {k: v for k, v in kw.items() if k not in ("marker", "ms", "ls")}


# ----------------------------------------------------------------------------
# ROOT output
# ----------------------------------------------------------------------------
class HistStore(list):
    """Collects RAW (unnormalised) histograms so they can be written to ROOT.

    MC is stored with its weights applied but WITHOUT the area-normalisation used
    for the plots, so downstream you can renormalise however you like.
    """

    def add(self, var, tag, edges, data, mc):
        # data / mc are (counts, errors) tuples.
        self.append(dict(var=var, tag=tag, edges=np.asarray(edges, dtype=float),
                         data=data, mc=mc))


def _num(x):
    """Filename-safe float: 0.4 -> '0p4', -1.5 -> 'm1p5'."""
    return ("%g" % x).replace("-", "m").replace(".", "p")


def write_root(store, path):
    """Write every stored histogram as a TH1D into `path`.

    Uses the `hist` package when available so per-bin errors (Sumw2) are
    preserved; otherwise falls back to a plain (counts, edges) tuple, in which
    case ROOT will assume sqrt(N) errors on readback.
    """
    try:
        import hist as _hist

        def make(counts, errs, edges):
            h = _hist.Hist.new.Variable(edges).Weight()
            v = h.view()
            v["value"] = counts
            v["variance"] = np.asarray(errs, dtype=float) ** 2
            return h

        backend = "hist (Sumw2 preserved)"
    except ImportError:
        def make(counts, errs, edges):
            return (np.asarray(counts, dtype=float), edges)

        backend = "numpy tuple (sqrt(N) errors on readback)"

    n = 0
    with uproot.recreate(path) as fout:
        for r in store:
            base = r["var"] if r["tag"] == "inclusive" \
                else "%s__%s" % (r["var"], r["tag"])
            fout["%s__data" % base] = make(*r["data"], r["edges"])
            fout["%s__mc" % base] = make(*r["mc"], r["edges"])
            n += 2
    print("Wrote %d histograms to %s [%s]" % (n, path, backend))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="charmonium_2018_24may2026.root")
    ap.add_argument("--mc", default="hb_2018_24may2026.root")
    ap.add_argument("--tree", default=TREE)
    ap.add_argument("--outdir", default="dca_plots")
    ap.add_argument("--root-out", default="impact_parameter_comparison.root",
                    help="ROOT file for the saved histograms; "
                         "pass '' to skip writing it.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    branches = list(NEEDED)
    if MC_WEIGHT:
        branches.append(MC_WEIGHT)

    print(f"Reading data: {args.data}")
    data_raw = load_branches(args.data, branches, args.tree)
    print(f"Reading MC:   {args.mc}")
    mc_raw = load_branches(args.mc, branches, args.tree)

    d_mask = selection_mask(data_raw)
    m_mask = selection_mask(mc_raw)
    print(f"Data: {d_mask.sum():,} / {len(d_mask):,} pass selection")
    print(f"MC:   {m_mask.sum():,} / {len(m_mask):,} pass selection")

    data = apply_mask(data_raw, d_mask)
    mc = apply_mask(mc_raw, m_mask)
    mc_w = mc[MC_WEIGHT] if MC_WEIGHT else None

    store = HistStore()

    # Inclusive comparisons (DCA + validation), each with a ratio panel.
    for name, cfg in {**DCA_VARS, **VALIDATION_VARS}.items():
        draw_comparison(name, cfg["xlabel"],
                        cfg["transform"](data), cfg["transform"](mc),
                        cfg["spec"], mc_weights=mc_w, outdir=args.outdir,
                        store=store)
        print(f"  wrote {name}.png")

    # eta-pt grids for the DCA variables.
    for name in GRID_VARS:
        cfg = DCA_VARS[name]
        draw_grid(name, cfg["xlabel"], data, mc, cfg["spec"],
                  mc_weights=mc_w, outdir=args.outdir, store=store)
        print(f"  wrote {name}_grid.png")

    if args.root_out:
        write_root(store, args.root_out)

    print(f"Done -> {args.outdir}/")


if __name__ == "__main__":
    main()