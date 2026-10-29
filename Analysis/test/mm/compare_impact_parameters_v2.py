#!/usr/bin/env python3
"""
Symmetric tag-and-probe data/MC validation of the muon transverse impact
parameter, in the 2018 charmonium sample vs. Hb MC.

Trigger requirement (symmetric):

    (mu1 tag & mu2 probe)  |  (mu2 tag & mu1 probe)

so the event is trigger-matched in either configuration. For each event we fill
histograms using whichever leg is the PROBE, binned in that leg's own
(|eta|, pt). Double-tag-and-probe events contribute both legs. We then produce,
per (|eta|, pt) cell and inclusively:

    * leg-1 histograms  (mu1 as probe)        -> keys '<var>__leg1__...'
    * leg-2 histograms  (mu2 as probe)        -> keys '<var>__leg2__...'
    * merged = leg1 + leg2                     -> keys '<var>__...'

The MERGED templates span the full muon kinematics (both a soft-probe and a
hard-tag leg), so a correction derived from them covers mu1 with real events
rather than extrapolation. Variables are leg-agnostic (branch is mu{L}_<stem>):

    |bs_dxy_sig| , |bs_dxy| , |bs_dxy_e|

Event-level validation (mass, vtx) is filled once per trigger-matched event.

I/O is via uproot. Histograms go to a ROOT file (TH1D, Sumw2 preserved when the
`hist` package is available); the merged data-vs-MC comparison is also plotted.
"""

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import uproot

try:
    import mplhep as hep

    plt.style.use(hep.style.CMS)
    HAVE_HEP = True
except Exception:
    HAVE_HEP = False


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
TREE = "tree"
JPSI_MASS = 3.0969

HLT_TAG = "mu{L}_HLT_Mu7p5_Track3p5_Jpsi_tag"
HLT_PROBE = "mu{L}_HLT_Mu7p5_Track3p5_Jpsi_probe"

ETA_BINS = [0.0, 0.85, 1.2, 1.5]                          # on |eta| of the probe
PT_BINS = [3.5, 5.0, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 17.0, 20.0, 30.0, 500.0]   # on pt of the probe

nbins = 500  # fine binning helps the downstream quantile morphing

# Leg-agnostic per-muon variables. `stem` is the branch stem (mu{L}_<stem>).
DCA_STEMS = {
    "abs_mu_bs_dxy_sig": dict(stem="bs_dxy_sig", absval=True,
                              xlabel=r"$|d_{xy}/\sigma_{d_{xy}}|$",
                              spec=(0.0, 100.0, nbins)),
    "abs_mu_bs_dxy": dict(stem="bs_dxy", absval=True,
                          xlabel=r"$|d_{xy}|$ [cm]", spec=(None, None, nbins)),
    "abs_mu_bs_dxy_e": dict(stem="bs_dxy_e", absval=True,
                            xlabel=r"$|\sigma_{d_{xy}}|$ [cm]",
                            spec=(None, None, nbins)),
}

# Event-level validation (filled once per trigger-matched event).
VALIDATION_VARS = {
    "mass": dict(branch="mass", xlabel=r"$m(\mu\mu)$ [GeV]",
                 spec=(JPSI_MASS - 0.1, JPSI_MASS + 0.1, 50)),
    "vtx_chi2": dict(branch="vtx_chi2", xlabel=r"vtx $\chi^2$",
                     spec=(None, None, nbins)),
    "vtx_prob": dict(branch="vtx_prob", xlabel="vtx p-value",
                     spec=(0.0, 1.0, nbins)),
}

MC_WEIGHT = None

DATA_KW = dict(color="black", marker="o", ms=4, ls="none", label="Data")
MC_KW = dict(color="#3f7fbf", label="Hb MC")


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def needed_branches():
    ev = ["mass", "charge", "vtx_prob", "vtx_chi2", "lxy",
          "mu1_id_medium", "mu2_id_medium"]
    for L in (1, 2):
        ev += [HLT_TAG.format(L=L), HLT_PROBE.format(L=L),
               f"mu{L}_eta", f"mu{L}_pt"]
        ev += [f"mu{L}_{d['stem']}" for d in DCA_STEMS.values()]
    if MC_WEIGHT:
        ev.append(MC_WEIGHT)
    return sorted(set(ev))


def load_branches(path, branches, tree=TREE):
    with uproot.open(f"{path}:{tree}") as t:
        present = [b for b in branches if b in t.keys()]
        missing = [b for b in branches if b not in t.keys()]
        if missing:
            print(f"  [warn] {os.path.basename(path)} missing: {missing}")
        return t.arrays(present, library="np")


# ----------------------------------------------------------------------------
# Symmetric selection
# ----------------------------------------------------------------------------
def base_event_mask(a):
    """Leg-agnostic event cuts."""
    m = np.abs(a["mass"] - JPSI_MASS) < 0.1
    m &= a["charge"] == 0
    m &= a["vtx_prob"] > 0.01
    m &= a["lxy"] > 0.03
    m &= a["mu1_id_medium"].astype(bool)
    m &= a["mu2_id_medium"].astype(bool)
    return m


def probe_masks(a):
    """Return (mask_probe1, mask_probe2): events where mu1 (resp. mu2) is the
    probe of a trigger-matched pair, with the probe inside |eta| < 1.5."""
    base = base_event_mask(a)
    tag1 = a[HLT_TAG.format(L=1)].astype(bool)
    tag2 = a[HLT_TAG.format(L=2)].astype(bool)
    probe1 = a[HLT_PROBE.format(L=1)] > 0.5
    probe2 = a[HLT_PROBE.format(L=2)] > 0.5
    # mu1 is probe -> the OTHER leg (mu2) must be the tag
    m1 = base & tag2 & probe1 & (np.abs(a["mu1_eta"]) < 1.5)
    # mu2 is probe -> mu1 is the tag
    m2 = base & tag1 & probe2 & (np.abs(a["mu2_eta"]) < 1.5)
    return m1, m2


def probe_sample(a, mask, L):
    """Unified-key view of one probe leg."""
    out = {"eta": a[f"mu{L}_eta"][mask], "pt": a[f"mu{L}_pt"][mask]}
    for name, cfg in DCA_STEMS.items():
        v = a[f"mu{L}_{cfg['stem']}"][mask]
        out[name] = np.abs(v) if cfg["absval"] else v
    return out


# ----------------------------------------------------------------------------
# Histogramming
# ----------------------------------------------------------------------------
def resolve_edges(spec, *value_arrays):
    if isinstance(spec, np.ndarray):
        return spec
    lo, hi, nb = spec
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
    return np.linspace(lo, hi, int(nb) + 1)


def hist_err(vals, edges):
    c, _ = np.histogram(vals, bins=edges)
    return c.astype(float), np.sqrt(c)


def add_hists(h1, h2):
    """(counts, err) + (counts, err) in quadrature."""
    return h1[0] + h2[0], np.sqrt(h1[1] ** 2 + h2[1] ** 2)


def grid_hists(vals, eta, pt, edges):
    """Return {celltag: (counts, err)} for all cells plus 'inclusive'."""
    out = {}
    aeta = np.abs(eta)
    for i in range(len(ETA_BINS) - 1):
        e0, e1 = ETA_BINS[i], ETA_BINS[i + 1]
        esel = (aeta >= e0) & (aeta < e1)
        for j in range(len(PT_BINS) - 1):
            p0, p1 = PT_BINS[j], PT_BINS[j + 1]
            sel = esel & (pt >= p0) & (pt < p1)
            out[cell_tag(e0, e1, p0, p1)] = hist_err(vals[sel], edges)
    out["inclusive"] = hist_err(vals, edges)
    return out


def cell_tag(e0, e1, p0, p1):
    return "eta_%s_%s__pt_%s_%s" % (_num(e0), _num(e1), _num(p0), _num(p1))


def _num(x):
    return ("%g" % x).replace("-", "m").replace(".", "p")


def norm_mc_to_data(mc, data):
    s = mc.sum()
    return (mc * data.sum() / s) if s > 0 else mc


# ----------------------------------------------------------------------------
# ROOT output (unchanged interface)
# ----------------------------------------------------------------------------
class HistStore(list):
    def add(self, var, tag, edges, data, mc):
        self.append(dict(var=var, tag=tag, edges=np.asarray(edges, float),
                         data=data, mc=mc))


def write_root(store, path):
    try:
        import hist as _hist

        def make(counts, errs, edges):
            h = _hist.Hist.new.Variable(edges).Weight()
            v = h.view()
            v["value"] = counts
            v["variance"] = np.asarray(errs, float) ** 2
            return h

        backend = "hist (Sumw2 preserved)"
    except ImportError:
        def make(counts, errs, edges):
            return (np.asarray(counts, float), edges)

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
# Plotting (from precomputed histograms)
# ----------------------------------------------------------------------------
def _step_kw(kw):
    return {k: v for k, v in kw.items() if k not in ("marker", "ms", "ls")}


def plot_inclusive(name, xlabel, edges, dd, mm, outdir):
    d, d_err = dd
    m = norm_mc_to_data(mm[0].copy(), d)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig = plt.figure(figsize=(7, 7))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)
    ax.stairs(m, edges, fill=True, alpha=0.35, **_step_kw(MC_KW))
    ax.stairs(m, edges, **{k: v for k, v in _step_kw(MC_KW).items()
                           if k != "label"})
    ax.errorbar(centers, d, yerr=d_err, **DATA_KW)
    ax.set_ylabel("Events")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    plt.setp(ax.get_xticklabels(), visible=False)
    if HAVE_HEP:
        hep.cms.label("Preliminary", ax=ax, data=True, rlabel="2018 (13 TeV)",
                      fontsize=13)
    ratio = np.divide(d, m, out=np.full_like(d, np.nan), where=m > 0)
    ratio_err = np.divide(d_err, m, out=np.full_like(d, np.nan), where=m > 0)
    axr.errorbar(centers, ratio, yerr=ratio_err, **DATA_KW)
    axr.axhline(1.0, color=MC_KW["color"], lw=1)
    axr.set_ylim(0.0, 2.0)
    axr.set_ylabel("Data / MC")
    axr.set_xlabel(xlabel)
    axr.set_xlim(edges[0], edges[-1])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=140,
                    bbox_inches="tight")
    plt.close(fig)


def plot_grid(name, xlabel, edges, cell_d, cell_m, outdir):
    centers = 0.5 * (edges[:-1] + edges[1:])
    nrow, ncol = len(ETA_BINS) - 1, len(PT_BINS) - 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow),
                             sharex=True, squeeze=False)
    for i in range(nrow):
        e0, e1 = ETA_BINS[i], ETA_BINS[i + 1]
        for j in range(ncol):
            p0, p1 = PT_BINS[j], PT_BINS[j + 1]
            ax = axes[i][j]
            tag = cell_tag(e0, e1, p0, p1)
            d = cell_d[tag][0]
            m = norm_mc_to_data(cell_m[tag][0].copy(), d)
            ax.stairs(m, edges, fill=True, alpha=0.35, **_step_kw(MC_KW))
            ax.errorbar(centers, d, yerr=cell_d[tag][1], ms=2.5, color="black",
                        marker="o", ls="none")
            ax.set_ylim(bottom=0)
            ax.set_title(rf"$|\eta|\in[{e0:g},{e1:g})$, "
                         rf"$p_T\in[{p0:g},{p1:g})$", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == nrow - 1:
                ax.set_xlabel(xlabel, fontsize=8)
    handles = [plt.Line2D([], [], **{k: v for k, v in DATA_KW.items()
                                     if k != "label"}, label="Data"),
               plt.Rectangle((0, 0), 1, 1, fc=MC_KW["color"], alpha=0.35,
                             label=MC_KW["label"])]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    fig.suptitle(name + "  (merged legs)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}_grid.{ext}"), dpi=130,
                    bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="charmonium_2018_24may2026.root")
    ap.add_argument("--mc", default="hb_2018_24may2026.root")
    ap.add_argument("--tree", default=TREE)
    ap.add_argument("--outdir", default="dca_plots")
    ap.add_argument("--root-out", default="impact_parameter_comparison.root")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    branches = needed_branches()
    data_raw = load_branches(args.data, branches, args.tree)
    mc_raw = load_branches(args.mc, branches, args.tree)

    d1m, d2m = probe_masks(data_raw)
    m1m, m2m = probe_masks(mc_raw)
    print(f"Data probes: leg1={d1m.sum():,}  leg2={d2m.sum():,}  "
          f"(both={np.sum(d1m & d2m):,})")
    print(f"MC   probes: leg1={m1m.sum():,}  leg2={m2m.sum():,}  "
          f"(both={np.sum(m1m & m2m):,})")

    # Unified-key probe samples for each leg.
    dS = {1: probe_sample(data_raw, d1m, 1), 2: probe_sample(data_raw, d2m, 2)}
    mS = {1: probe_sample(mc_raw, m1m, 1), 2: probe_sample(mc_raw, m2m, 2)}

    store = HistStore()

    for name, cfg in DCA_STEMS.items():
        # Shared binning from the pooled probe values (both legs, data + MC).
        edges = resolve_edges(cfg["spec"],
                              dS[1][name], dS[2][name], mS[1][name], mS[2][name])

        d1 = grid_hists(dS[1][name], dS[1]["eta"], dS[1]["pt"], edges)
        d2 = grid_hists(dS[2][name], dS[2]["eta"], dS[2]["pt"], edges)
        m1 = grid_hists(mS[1][name], mS[1]["eta"], mS[1]["pt"], edges)
        m2 = grid_hists(mS[2][name], mS[2]["eta"], mS[2]["pt"], edges)

        merged_d, merged_m = {}, {}
        for tag in d1:
            store.add(f"{name}__leg1", tag, edges, d1[tag], m1[tag])
            store.add(f"{name}__leg2", tag, edges, d2[tag], m2[tag])
            md = add_hists(d1[tag], d2[tag])
            mm = add_hists(m1[tag], m2[tag])
            store.add(name, tag, edges, md, mm)          # merged (leg1 + leg2)
            merged_d[tag], merged_m[tag] = md, mm

        plot_inclusive(name, cfg["xlabel"], edges,
                       merged_d["inclusive"], merged_m["inclusive"], args.outdir)
        plot_grid(name, cfg["xlabel"], edges, merged_d, merged_m, args.outdir)
        print(f"  {name}: leg1+leg2 merged, plotted")

    # Event-level validation (once per trigger-matched event).
    d_evt = d1m | d2m
    m_evt = m1m | m2m
    for name, cfg in VALIDATION_VARS.items():
        dv = data_raw[cfg["branch"]][d_evt]
        mv = mc_raw[cfg["branch"]][m_evt]
        edges = resolve_edges(cfg["spec"], dv, mv)
        dd, mm = hist_err(dv, edges), hist_err(mv, edges)
        store.add(name, "inclusive", edges, dd, mm)
        plot_inclusive(name, cfg["xlabel"], edges, dd, mm, args.outdir)
        print(f"  {name}: event-level validation")

    if args.root_out:
        write_root(store, args.root_out)
    print(f"Done -> {args.outdir}/")


if __name__ == "__main__":
    main()