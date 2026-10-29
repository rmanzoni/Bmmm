#!/usr/bin/env python3
"""
Symmetric tag-and-probe data/MC validation of the muon transverse impact
parameter, in the 2018 charmonium sample vs. Hb MC, with sPlot background
subtraction of the data.

Trigger requirement (symmetric):
    (mu1 tag & mu2 probe) | (mu2 tag & mu1 probe)

BACKGROUND SUBTRACTION (sPlot)
------------------------------
The dimuon invariant mass is fit (J/psi signal = double Gaussian, combinatorial
background = exponential) with an extended maximum likelihood. Signal sWeights
are computed from the fit and used as per-event weights when filling the DATA
histograms, so the combinatorial background is statistically subtracted. The MC
is pure b->J/psi signal and is filled unweighted.

Per (|eta|, pt) cell and inclusively we produce leg-1, leg-2 and merged
(=leg1+leg2) histograms of |bs_dxy_sig|, |bs_dxy|, |bs_dxy_e|.

Notes
-----
* sPlot assumes the discriminant (mass) is independent of the plotted variable
  within each species. That holds well for sigma_dxy (a tracker resolution,
  uncorrelated with dimuon mass); it is weaker for dxy / dxy_sig, whose true
  value differs between prompt and nonprompt muons -- see the prompt/nonprompt
  discussion accompanying this script.
* sWeighted histograms can have (slightly) NEGATIVE bins in sparse tails. When
  feeding the DATA templates to the quantile-morphing derivation, clip negative
  bins to zero first (np.maximum(counts, 0)) so the empirical CDF stays
  monotone.
"""

import argparse
import os

import numpy as np
from scipy.special import erf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import uproot
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL

try:
    import mplhep as hep

    plt.style.use(hep.style.CMS)
    HAVE_HEP = True
except Exception:
    HAVE_HEP = False


_TRAPZ = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

TREE = "tree"
JPSI_MASS = 3.0969
MASS_HALFWIN = 0.25
PPDL_MIN = 0.01   # cm; nonprompt cut on pseudo-proper decay length (tune this)

HLT_TAG = "mu{L}_HLT_Mu7p5_Track3p5_Jpsi_tag"
HLT_PROBE = "mu{L}_HLT_Mu7p5_Track3p5_Jpsi_probe"

ETA_BINS = [0.0, 0.85, 1.2, 1.5]
PT_BINS = [3.5, 5.0, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 12.0, 13.0,
           14.0, 15.0, 17.0, 20.0, 30.0, 500.0]
nbins = 500

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

VALIDATION_VARS = {
    "vtx_chi2": dict(branch="vtx_chi2", xlabel=r"vtx $\chi^2$",
                     spec=(None, None, nbins)),
    "vtx_prob": dict(branch="vtx_prob", xlabel="vtx p-value",
                     spec=(0.0, 1.0, nbins)),
}

MC_WEIGHT = None
DATA_KW = dict(color="black", marker="o", ms=4, ls="none", label="Data (sPlot)")
MC_KW = dict(color="#3f7fbf", label="Hb MC")


def needed_branches():
    ev = ["mass", "charge", "vtx_prob", "vtx_chi2", "lxy",
          "mu1_id_medium", "mu2_id_medium", "pt", "cos2d"]
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


def base_event_mask(a):
    m = np.abs(a["mass"] - JPSI_MASS) < MASS_HALFWIN
    m &= a["charge"] == 0
    m &= a["cos2d"] > 0.9
    m &= a["vtx_prob"] > 0.01
    m &= (a["lxy"] * a["cos2d"] * JPSI_MASS / a["pt"]) > PPDL_MIN   # was: a["lxy"] > 0.03
    m &= a["mu1_id_medium"].astype(bool)
    m &= a["mu2_id_medium"].astype(bool)
    return m


def probe_masks(a):
    base = base_event_mask(a)
    tag1 = a[HLT_TAG.format(L=1)].astype(bool)
    tag2 = a[HLT_TAG.format(L=2)].astype(bool)
    probe1 = a[HLT_PROBE.format(L=1)] > 0.5
    probe2 = a[HLT_PROBE.format(L=2)] > 0.5
    m1 = base & tag2 & probe1 & (np.abs(a["mu1_eta"]) < 1.5)
    m2 = base & tag1 & probe2 & (np.abs(a["mu2_eta"]) < 1.5)
    return m1, m2


def probe_sample(a, mask, L):
    out = {"eta": a[f"mu{L}_eta"][mask], "pt": a[f"mu{L}_pt"][mask]}
    for name, cfg in DCA_STEMS.items():
        v = a[f"mu{L}_{cfg['stem']}"][mask]
        out[name] = np.abs(v) if cfg["absval"] else v
    return out


# ---- mass fit + sPlot -------------------------------------------------------
# Signal  : sum of two double-sided Crystal Balls, common mean, shared tail
#           parameters (aL, nL, aR, nR), two core widths s1, s2 (fraction f).
#           Independent tails per CB are a one-line change (see NOTE below).
# Background: sum of two exponentials (rates lam1, lam2, fraction fb).
def _mass_model(lo, hi):
    r2 = np.sqrt(2.0)
    _grid = np.linspace(lo, hi, 2001)          # grid for numeric normalisation

    def _dscb_unnorm(x, mu, s, aL, nL, aR, nR):
        t = (x - mu) / s
        core = np.exp(-0.5 * t * t)
        AL = (nL / aL) ** nL * np.exp(-0.5 * aL * aL); BL = nL / aL - aL
        AR = (nR / aR) ** nR * np.exp(-0.5 * aR * aR); BR = nR / aR - aR
        inL, inR = t < -aL, t > aR
        # keep the power-law base at 1.0 outside its own tail -> no overflow
        left = AL * np.where(inL, BL - t, 1.0) ** (-nL)
        right = AR * np.where(inR, BR + t, 1.0) ** (-nR)
        return np.where(inL, left, np.where(inR, right, core))

    def ndscb(x, mu, s, aL, nL, aR, nR):
        norm = _TRAPZ(_dscb_unnorm(_grid, mu, s, aL, nL, aR, nR), _grid)
        return _dscb_unnorm(x, mu, s, aL, nL, aR, nR) / norm

    def nexpo(x, lam):
        if abs(lam) < 1e-6:
            return np.full_like(x, 1.0 / (hi - lo))
        nrm = (np.exp(-lam * lo) - np.exp(-lam * hi)) / lam
        return np.exp(-lam * x) / nrm

    def sig_pdf(x, mu, s1, s2, f, aL, nL, aR, nR):
        return (f * ndscb(x, mu, s1, aL, nL, aR, nR)
                + (1 - f) * ndscb(x, mu, s2, aL, nL, aR, nR))
        # NOTE: for independent tails, give the 2nd CB its own aL2,nL2,aR2,nR2
        #       and add them as fit parameters below.

    def bkg_pdf(x, lam1, lam2, fb):
        return fb * nexpo(x, lam1) + (1 - fb) * nexpo(x, lam2)

    def density(x, Ns, Nb, mu, s1, s2, f, aL, nL, aR, nR, lam1, lam2, fb):
        sig = sig_pdf(x, mu, s1, s2, f, aL, nL, aR, nR)
        return Ns + Nb, Ns * sig + Nb * bkg_pdf(x, lam1, lam2, fb)

    return density, sig_pdf, bkg_pdf


def fit_mass(mass):
    lo, hi = JPSI_MASS - MASS_HALFWIN, JPSI_MASS + MASS_HALFWIN
    density, sig_pdf, bkg_pdf = _mass_model(lo, hi)
    n = mass.size
    mi = Minuit(ExtendedUnbinnedNLL(mass, density),
                Ns=0.8 * n, Nb=0.2 * n, mu=JPSI_MASS,
                s1=0.03, s2=0.06, f=0.6,
                aL=1.5, nL=3.0, aR=1.5, nR=3.0,
                lam1=1.0, lam2=-1.0, fb=0.5)
    mi.limits["Ns"] = (0, None); mi.limits["Nb"] = (0, None)
    mi.limits["mu"] = (lo, hi)
    mi.limits["s1"] = (2e-3, 0.1); mi.limits["s2"] = (2e-3, 0.2)
    mi.limits["f"] = (0, 1); mi.limits["fb"] = (0, 1)
    mi.limits["aL"] = (0.2, 10.0); mi.limits["aR"] = (0.2, 10.0)
    mi.limits["nL"] = (1.01, 60.0); mi.limits["nR"] = (1.01, 60.0)
    mi.limits["lam1"] = (-100, 100); mi.limits["lam2"] = (-100, 100)
    mi.strategy = 2
    mi.migrad(); mi.migrad(); mi.hesse()     # second pass helps the 13-par fit
    p = mi.values

    def fs(x):
        return sig_pdf(x, p["mu"], p["s1"], p["s2"], p["f"],
                       p["aL"], p["nL"], p["aR"], p["nR"])

    def fb(x):
        return bkg_pdf(x, p["lam1"], p["lam2"], p["fb"])

    return dict(Ns=p["Ns"], Nb=p["Nb"], fs=fs, fb=fb, lo=lo, hi=hi,
                npar=mi.nfit, valid=bool(mi.valid))


def splot_signal_weights(mass, res):
    fs, fb = res["fs"](mass), res["fb"](mass)
    D = res["Ns"] * fs + res["Nb"] * fb
    Vinv = np.array([[np.sum(fs * fs / D ** 2), np.sum(fs * fb / D ** 2)],
                     [np.sum(fb * fs / D ** 2), np.sum(fb * fb / D ** 2)]])
    V = np.linalg.inv(Vinv)
    return (V[0, 0] * fs + V[0, 1] * fb) / D


# ---- histogramming (weight-aware) -------------------------------------------
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


def hist_err(vals, edges, weights=None):
    if weights is None:
        c, _ = np.histogram(vals, bins=edges)
        return c.astype(float), np.sqrt(c)
    c, _ = np.histogram(vals, bins=edges, weights=weights)
    w2, _ = np.histogram(vals, bins=edges, weights=weights ** 2)
    return c.astype(float), np.sqrt(w2)


def add_hists(h1, h2):
    return h1[0] + h2[0], np.sqrt(h1[1] ** 2 + h2[1] ** 2)


def grid_hists(vals, eta, pt, edges, weights=None):
    out = {}
    aeta = np.abs(eta)
    for i in range(len(ETA_BINS) - 1):
        e0, e1 = ETA_BINS[i], ETA_BINS[i + 1]
        esel = (aeta >= e0) & (aeta < e1)
        for j in range(len(PT_BINS) - 1):
            p0, p1 = PT_BINS[j], PT_BINS[j + 1]
            sel = esel & (pt >= p0) & (pt < p1)
            w = weights[sel] if weights is not None else None
            out[cell_tag(e0, e1, p0, p1)] = hist_err(vals[sel], edges, w)
    out["inclusive"] = hist_err(vals, edges, weights)
    return out


def cell_tag(e0, e1, p0, p1):
    return "eta_%s_%s__pt_%s_%s" % (_num(e0), _num(e1), _num(p0), _num(p1))


def _num(x):
    return ("%g" % x).replace("-", "m").replace(".", "p")


def norm_mc_to_data(mc, data):
    s = mc.sum()
    return (mc * data.sum() / s) if s > 0 else mc


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

        backend = "numpy tuple"

    n = 0
    with uproot.recreate(path) as fout:
        for r in store:
            base = r["var"] if r["tag"] == "inclusive" \
                else "%s__%s" % (r["var"], r["tag"])
            fout["%s__data" % base] = make(*r["data"], r["edges"])
            fout["%s__mc" % base] = make(*r["mc"], r["edges"])
            n += 2
    print("Wrote %d histograms to %s [%s]" % (n, path, backend))


def _step_kw(kw):
    return {k: v for k, v in kw.items() if k not in ("marker", "ms", "ls")}


def plot_inclusive(name, xlabel, edges, dd, mm, outdir):
    d, d_err = dd
    m = norm_mc_to_data(mm[0].copy(), d)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig = plt.figure(figsize=(7, 7))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0]); axr = fig.add_subplot(gs[1], sharex=ax)
    ax.stairs(m, edges, fill=True, alpha=0.35, **_step_kw(MC_KW))
    ax.stairs(m, edges, **{k: v for k, v in _step_kw(MC_KW).items()
                           if k != "label"})
    ax.errorbar(centers, d, yerr=d_err, **DATA_KW)
    ax.set_ylabel("Events / bin (sWeighted)")
    ax.set_ylim(bottom=min(0.0, float(np.nanmin(d)) * 1.1))
    ax.legend(loc="best")
    plt.setp(ax.get_xticklabels(), visible=False)
    if HAVE_HEP:
        hep.cms.label("Preliminary", ax=ax, data=True, rlabel="2018 (13 TeV)",
                      fontsize=13)
    ratio = np.divide(d, m, out=np.full_like(d, np.nan), where=m > 0)
    ratio_err = np.divide(d_err, m, out=np.full_like(d, np.nan), where=m > 0)
    axr.errorbar(centers, ratio, yerr=ratio_err, **DATA_KW)
    axr.axhline(1.0, color=MC_KW["color"], lw=1)
    axr.set_ylim(0.0, 2.0); axr.set_ylabel("Data / MC")
    axr.set_xlabel(xlabel); axr.set_xlim(edges[0], edges[-1])
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
            ax = axes[i][j]; tag = cell_tag(e0, e1, p0, p1)
            d = cell_d[tag][0]
            m = norm_mc_to_data(cell_m[tag][0].copy(), d)
            ax.stairs(m, edges, fill=True, alpha=0.35, **_step_kw(MC_KW))
            ax.errorbar(centers, d, yerr=cell_d[tag][1], ms=2.5, color="black",
                        marker="o", ls="none")
            ax.axhline(0.0, color="0.8", lw=0.6)
            ax.set_title(rf"$|\eta|\in[{e0:g},{e1:g})$, "
                         rf"$p_T\in[{p0:g},{p1:g})$", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == nrow - 1:
                ax.set_xlabel(xlabel, fontsize=8)
    handles = [plt.Line2D([], [], **{k: v for k, v in DATA_KW.items()
                                     if k != "label"}, label="Data (sPlot)"),
               plt.Rectangle((0, 0), 1, 1, fc=MC_KW["color"], alpha=0.35,
                             label=MC_KW["label"])]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    fig.suptitle(name + "  (merged legs, background-subtracted)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}_grid.{ext}"), dpi=130,
                    bbox_inches="tight")
    plt.close(fig)


def plot_mass_fit(mass, res, outdir):
    lo, hi = res["lo"], res["hi"]
    edges = np.linspace(lo, hi, 80)
    ctr = 0.5 * (edges[:-1] + edges[1:]); bw = edges[1] - edges[0]
    counts, _ = np.histogram(mass, bins=edges)
    xs = np.linspace(lo, hi, 400)
    sig = res["Ns"] * res["fs"](xs) * bw
    bkg = res["Nb"] * res["fb"](xs) * bw

    # model expectation per bin (for pulls): integrate the density over each bin
    fine = np.linspace(lo, hi, 8 * len(ctr) + 1)
    dens = res["Ns"] * res["fs"](fine) + res["Nb"] * res["fb"](fine)
    exp_bin = np.array([_TRAPZ(dens[(fine >= edges[k]) & (fine <= edges[k + 1])],
                                 fine[(fine >= edges[k]) & (fine <= edges[k + 1])])
                        for k in range(len(ctr))])
    sigma = np.sqrt(np.clip(exp_bin, 1e-9, None))          # Poisson expectation
    pull = (counts - exp_bin) / sigma
    ndf = np.count_nonzero(exp_bin > 0) - res.get("npar", 0)
    chi2 = float(np.sum(pull[exp_bin > 0] ** 2))

    fig = plt.figure(figsize=(7, 6.2))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    axp = fig.add_subplot(gs[1], sharex=ax)

    ax.errorbar(ctr, counts, yerr=np.sqrt(counts), fmt="o", ms=3, color="black",
                label="Data")
    ax.plot(xs, sig + bkg, color="#c1272d", lw=2, label="fit")
    ax.plot(xs, bkg, color="#3f7fbf", lw=1.6, ls="--", label="background")
    ax.fill_between(xs, bkg, sig + bkg, color="#c1272d", alpha=0.12,
                    label="signal")
    ax.set_ylabel(f"Events / {bw*1e3:.0f} MeV")
    ax.set_ylim(bottom=0, top=1.28 * counts.max()); ax.set_xlim(lo, hi)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.legend(loc="upper right", fontsize=10)
    sob = res["Ns"] / max(res["Nb"], 1e-9)
    ax.text(0.035, 0.94,
            rf"$N_S={res['Ns']:.0f}$" "\n" rf"$N_B={res['Nb']:.0f}$" "\n"
            rf"$S/B={sob:.1f}$" "\n" rf"$\chi^2/\mathrm{{ndf}}={chi2:.0f}/{ndf}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=10)
    if HAVE_HEP:
        hep.cms.label("Preliminary", ax=ax, data=True, rlabel="2018 (13 TeV)",
                      fontsize=13)

    axp.axhspan(-2, 2, color="0.85", zorder=0)
    axp.axhspan(-1, 1, color="0.72", zorder=0)
    axp.axhline(0, color="#c1272d", lw=1)
    axp.plot(ctr, pull, "o", ms=3, color="black")
    axp.set_ylim(-5, 5)
    axp.set_ylabel(r"pull")
    axp.set_xlabel(r"$m(\mu\mu)$ [GeV]")
    axp.set_xlim(lo, hi)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"mass_fit.{ext}"), dpi=140,
                    bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="charmonium_2018_24may2026.root")
    ap.add_argument("--mc", default="hb_2018_24may2026.root")
    ap.add_argument("--tree", default=TREE)
    ap.add_argument("--outdir", default="dca_plots")
    ap.add_argument("--root-out", default="impact_parameter_comparison.root")
    ap.add_argument("--no-sweights", action="store_true",
                    help="disable sPlot; fill data with unit weights")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    branches = needed_branches()
    data_raw = load_branches(args.data, branches, args.tree)
    mc_raw = load_branches(args.mc, branches, args.tree)

    d1m, d2m = probe_masks(data_raw)
    m1m, m2m = probe_masks(mc_raw)
    d_evt, m_evt = d1m | d2m, m1m | m2m
    print(f"Data probes: leg1={d1m.sum():,} leg2={d2m.sum():,} "
          f"(both={np.sum(d1m & d2m):,}); events={d_evt.sum():,}")
    print(f"MC   probes: leg1={m1m.sum():,} leg2={m2m.sum():,}")

    sw_full = np.ones(len(data_raw["mass"]), dtype=float)
    if not args.no_sweights:
        res = fit_mass(data_raw["mass"][d_evt])
        print(f"  mass fit valid={res['valid']} Ns={res['Ns']:.0f} "
              f"Nb={res['Nb']:.0f} S/B={res['Ns']/max(res['Nb'],1e-9):.1f}")
        sw = splot_signal_weights(data_raw["mass"][d_evt], res)
        print(f"  sum(sWeights)={sw.sum():.0f} (~Ns)")
        sw_full[d_evt] = sw
        plot_mass_fit(data_raw["mass"][d_evt], res, args.outdir)

    wD = {1: sw_full[d1m], 2: sw_full[d2m]}
    dS = {1: probe_sample(data_raw, d1m, 1), 2: probe_sample(data_raw, d2m, 2)}
    mS = {1: probe_sample(mc_raw, m1m, 1), 2: probe_sample(mc_raw, m2m, 2)}

    store = HistStore()
    for name, cfg in DCA_STEMS.items():
        edges = resolve_edges(cfg["spec"],
                              dS[1][name], dS[2][name], mS[1][name], mS[2][name])
        d1 = grid_hists(dS[1][name], dS[1]["eta"], dS[1]["pt"], edges, wD[1])
        d2 = grid_hists(dS[2][name], dS[2]["eta"], dS[2]["pt"], edges, wD[2])
        m1 = grid_hists(mS[1][name], mS[1]["eta"], mS[1]["pt"], edges, None)
        m2 = grid_hists(mS[2][name], mS[2]["eta"], mS[2]["pt"], edges, None)
        merged_d, merged_m = {}, {}
        for tag in d1:
            store.add(f"{name}__leg1", tag, edges, d1[tag], m1[tag])
            store.add(f"{name}__leg2", tag, edges, d2[tag], m2[tag])
            md, mm = add_hists(d1[tag], d2[tag]), add_hists(m1[tag], m2[tag])
            store.add(name, tag, edges, md, mm)
            merged_d[tag], merged_m[tag] = md, mm
        plot_inclusive(name, cfg["xlabel"], edges,
                       merged_d["inclusive"], merged_m["inclusive"], args.outdir)
        plot_grid(name, cfg["xlabel"], edges, merged_d, merged_m, args.outdir)
        print(f"  {name}: merged, "
              f"{'sWeighted' if not args.no_sweights else 'raw'}")

    for name, cfg in VALIDATION_VARS.items():
        dv, mv = data_raw[cfg["branch"]][d_evt], mc_raw[cfg["branch"]][m_evt]
        edges = resolve_edges(cfg["spec"], dv, mv)
        dd = hist_err(dv, edges, sw_full[d_evt])
        mm = hist_err(mv, edges, None)
        store.add(name, "inclusive", edges, dd, mm)
        plot_inclusive(name, cfg["xlabel"], edges, dd, mm, args.outdir)
        print(f"  {name}: validation")

    if args.root_out:
        write_root(store, args.root_out)
    print(f"Done -> {args.outdir}/")


if __name__ == "__main__":
    main()