"""
efficiency_sf.py
================
Binned trigger efficiency and scale factors for muon tag-and-probe.

Inputs
------
  probe_skim_charmonium_2018_24may2026.root   (data)
  probe_skim_hb_2018_24may2026.root           (MC)

Each tree is expected to have branches:
  pt        – probe pT  [GeV]
  eta       – probe η  (signed; |η| is used)
  dxy_sig   – probe |dxy / σ_xy|
  pass      – 1 if probe fired the HLT, 0 otherwise

Outputs
-------
  efficiency_data.json     – binned eff ± Clopper-Pearson for data
  efficiency_mc.json       – same for MC
  scale_factors.json       – SF = ε_data / ε_MC  with asymmetric errors
  plots/*.png              – one figure per projection
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import json
import os

# ── third-party ─────────────────────────────────────────────────────────────
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import uproot
from scipy.stats import beta as sp_beta
from scipy.special import erf
from matplotlib.gridspec import GridSpec
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL, ExtendedBinnedNLL
try:
    from scipy.integrate import cumulative_trapezoid as _CUMTRAPZ
except ImportError:
    from scipy.integrate import cumtrapz as _CUMTRAPZ

try:
    import mplhep as hep
    plt.style.use(hep.style.CMS)
    _HAVE_HEP = True
except Exception:
    _HAVE_HEP = False

matplotlib.rcParams.update({"figure.dpi": 150})


# ════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

FILES = {
    "data": "probe_skim_charmonium_2018_14sep2026_morph.root",
    "mc":   "probe_skim_hb_2018_14sep2026_morph.root",
}
TREE_NAME   = "tree"     # adjust if your tree has a different name
PASS_BRANCH = "pass"     # branch holding the HLT decision (0 / 1)
CL          = 0.6827     # confidence level for CP intervals (≈ 1σ)

# ── sPlot: background-subtract the DATA with one inclusive J/psi mass fit ─────
USE_SPLOT    = True
MASS_BRANCH  = "mass"     # dimuon invariant-mass branch in the probe skim
JPSI_MASS    = 3.0969
MASS_HALFWIN = 0.25       # fit-window half-width [GeV]
EFF_HALFWIN  = 0.10       # post-fit window for the efficiency sample [GeV]
_TRAPZ = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

os.makedirs("plots", exist_ok=True)

# ── Bin edges ────────────────────────────────────────────────────────────────
# pT  : fine near threshold (6-12 GeV), coarse above
# PT_BINS  = np.array([6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 13.0, 15.0, 20.0])
# PT_BINS  = np.array([6.0, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 11.0, 11.5, 11.75, 12.0, 12.25, 12.5, 13.0, 15.0, 20.0])
# PT_BINS  = np.array([7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0, 11.0, 11.5, 11.75, 12.0, 12.25, 12.5, 13.0, 15.0, 20.0])
PT_BINS  = np.array([7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0, 11.0, 11.5, 11.75, 12.0, 12.25, 12.5, 13.0, 15.0, 20.0, 25.0, 50.0])

# |η| : uniform inside barrel acceptance
ETA_BINS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5])

# |dxy/σ| : fine near signal region, coarse in tails
# DXY_BINS = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 10.0, 15.0])
DXY_BINS = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 40.0, 100.0])

BINS = {
    "pt":      PT_BINS,
    "eta":     ETA_BINS,
    "dxy_sig": DXY_BINS,
}

LABELS = {
    "pt":      r"probe $p_T$ [GeV]",
    "eta":     r"probe $|\eta|$",
    "dxy_sig": r"probe $|d_{xy}/\sigma_{xy}|$",
}

# Variables whose x/y axis should be drawn in log scale in 2D maps.
# Criteria: uneven bin spacing AND all edges strictly > 0.
LOG_SCALE = {
    "pt":      False,
    "eta":     False,
    "dxy_sig": False,   # starts at 0, so log is not possible
}

PAIRS = [
    ("pt",  "eta"),
    ("pt",  "dxy_sig"),
    ("eta", "dxy_sig"),
]


# ════════════════════════════════════════════════════════════════════════════
# 1.  CLOPPER-PEARSON HELPERS
# ════════════════════════════════════════════════════════════════════════════

_alpha = 1.0 - CL   # two-sided → α/2 per tail


def cp_interval(k, n):
    """
    Clopper-Pearson interval.

    Returns (lo, hi) = downward and upward *errors* (not bounds), i.e.
        efficiency - lo_bound  and  hi_bound - efficiency.
    Works element-wise on arrays.
    """
    k   = np.asarray(k, dtype=float)
    n   = np.asarray(n, dtype=float)
    eff = np.where(n > 0, k / n, np.nan)

    lo_bound = np.where(k > 0,
                        sp_beta.ppf(_alpha / 2,       k,     n - k + 1),
                        0.0)
    hi_bound = np.where(k < n,
                        sp_beta.ppf(1 - _alpha / 2,   k + 1, n - k),
                        1.0)
    return eff - lo_bound, hi_bound - eff


def sf_errors(eff_d, lo_d, hi_d, eff_m, lo_m, hi_m):
    """
    Asymmetric SF = eff_d / eff_m uncertainties via error propagation.

    The upward SF error pairs the upward data error with the downward MC
    error (conservative asymmetric bounds).
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        sf = np.where((eff_m > 0) & (eff_d > 0), eff_d / eff_m, np.nan)

        rel_up = np.sqrt(
            np.where(eff_d > 0, (hi_d / eff_d) ** 2, 0) +
            np.where(eff_m > 0, (lo_m / eff_m) ** 2, 0)
        )
        rel_dn = np.sqrt(
            np.where(eff_d > 0, (lo_d / eff_d) ** 2, 0) +
            np.where(eff_m > 0, (hi_m / eff_m) ** 2, 0)
        )
        sf_hi = np.abs(sf) * rel_up
        sf_lo = np.abs(sf) * rel_dn

    return sf, sf_lo, sf_hi


# ════════════════════════════════════════════════════════════════════════════
# 1b.  MASS FIT  +  sPlot   (two double-sided Crystal Balls  +  double expo)
# ════════════════════════════════════════════════════════════════════════════
# Signal    : sum of two double-sided Crystal Balls, common mean, shared tails
#             (aL, nL, aR, nR), two core widths s1, s2 (fraction f).
# Background : sum of two exponentials (rates lam1, lam2, fraction fb).
# The fit is done ONCE on the inclusive data sample; per-event signal sWeights
# are then used to fill every (pt, |eta|, dxy_sig) bin, so the combinatorial
# background is statistically subtracted before slicing into low-stat bins.

def _mass_model(lo, hi):
    grid = np.linspace(lo, hi, 2001)                     # numeric-normalisation grid

    def _dscb_unnorm(x, mu, s, aL, nL, aR, nR):
        t = (x - mu) / s
        core = np.exp(-0.5 * t * t)
        AL = (nL / aL) ** nL * np.exp(-0.5 * aL * aL); BL = nL / aL - aL
        AR = (nR / aR) ** nR * np.exp(-0.5 * aR * aR); BR = nR / aR - aR
        inL, inR = t < -aL, t > aR
        left = AL * np.where(inL, BL - t, 1.0) ** (-nL)
        right = AR * np.where(inR, BR + t, 1.0) ** (-nR)
        return np.where(inL, left, np.where(inR, right, core))

    def ndscb(x, mu, s, aL, nL, aR, nR):
        norm = _TRAPZ(_dscb_unnorm(grid, mu, s, aL, nL, aR, nR), grid)
        return _dscb_unnorm(x, mu, s, aL, nL, aR, nR) / norm

    def nexpo(x, lam):
        if abs(lam) < 1e-6:
            return np.full_like(x, 1.0 / (hi - lo))
        nrm = (np.exp(-lam * lo) - np.exp(-lam * hi)) / lam
        return np.exp(-lam * x) / nrm

    def sig_pdf(x, mu, s1, s2, f, aL, nL, aR, nR):
        return (f * ndscb(x, mu, s1, aL, nL, aR, nR)
                + (1 - f) * ndscb(x, mu, s2, aL, nL, aR, nR))

    def bkg_pdf(x, lam1, lam2, fb):
        return fb * nexpo(x, lam1) + (1 - fb) * nexpo(x, lam2)

    def density(x, Ns, Nb, mu, s1, s2, f, aL, nL, aR, nR, lam1, lam2, fb):
        sig = sig_pdf(x, mu, s1, s2, f, aL, nL, aR, nR)
        return Ns + Nb, Ns * sig + Nb * bkg_pdf(x, lam1, lam2, fb)

    return density, sig_pdf, bkg_pdf


def fit_mass(mass, nbins_fit=200):
    """Binned extended-ML fit (fast for large samples). sWeights are computed
    per-event afterwards from the fitted pdf, so binning the fit is harmless."""
    lo, hi = JPSI_MASS - MASS_HALFWIN, JPSI_MASS + MASS_HALFWIN
    density, sig_pdf, bkg_pdf = _mass_model(lo, hi)
    counts, edges = np.histogram(mass, bins=nbins_fit, range=(lo, hi))
    grid = np.linspace(lo, hi, 4001)

    def model_cdf(xe, Ns, Nb, mu, s1, s2, f, aL, nL, aR, nR, lam1, lam2, fb):
        sp = sig_pdf(grid, mu, s1, s2, f, aL, nL, aR, nR)   # normalised to 1
        bp = bkg_pdf(grid, lam1, lam2, fb)                  # normalised to 1
        csig = np.concatenate([[0.0], _CUMTRAPZ(sp, grid)])
        cbkg = np.concatenate([[0.0], _CUMTRAPZ(bp, grid)])
        return Ns * np.interp(xe, grid, csig) + Nb * np.interp(xe, grid, cbkg)

    n = mass.size
    mi = Minuit(ExtendedBinnedNLL(counts, edges, model_cdf),
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
    mi.migrad(); mi.hesse()
    p = mi.values

    def fs(x):
        return sig_pdf(x, p["mu"], p["s1"], p["s2"], p["f"],
                       p["aL"], p["nL"], p["aR"], p["nR"])

    def fb(x):
        return bkg_pdf(x, p["lam1"], p["lam2"], p["fb"])

    return dict(Ns=p["Ns"], Nb=p["Nb"], fs=fs, fb=fb, lo=lo, hi=hi,
                npar=mi.nfit, valid=bool(mi.valid))


def splot_signal_weights(mass, res):
    """Per-event signal sWeights (sum ~ fitted Ns)."""
    fs, fb = res["fs"](mass), res["fb"](mass)
    D = res["Ns"] * fs + res["Nb"] * fb
    Vinv = np.array([[np.sum(fs * fs / D ** 2), np.sum(fs * fb / D ** 2)],
                     [np.sum(fb * fs / D ** 2), np.sum(fb * fb / D ** 2)]])
    V = np.linalg.inv(Vinv)
    return (V[0, 0] * fs + V[0, 1] * fb) / D


def plot_mass_fit(mass, res, outdir):
    lo, hi = res["lo"], res["hi"]
    edges = np.linspace(lo, hi, 80)
    ctr = 0.5 * (edges[:-1] + edges[1:]); bw = edges[1] - edges[0]
    counts, _ = np.histogram(mass, bins=edges)
    xs = np.linspace(lo, hi, 400)
    sig = res["Ns"] * res["fs"](xs) * bw
    bkg = res["Nb"] * res["fb"](xs) * bw
    fine = np.linspace(lo, hi, 8 * len(ctr) + 1)
    dens = res["Ns"] * res["fs"](fine) + res["Nb"] * res["fb"](fine)
    exp_bin = np.array([_TRAPZ(dens[(fine >= edges[k]) & (fine <= edges[k + 1])],
                               fine[(fine >= edges[k]) & (fine <= edges[k + 1])])
                        for k in range(len(ctr))])
    sigma = np.sqrt(np.clip(exp_bin, 1e-9, None))
    pull = (counts - exp_bin) / sigma
    ndf = np.count_nonzero(exp_bin > 0) - res.get("npar", 0)
    chi2 = float(np.sum(pull[exp_bin > 0] ** 2))

    fig = plt.figure(figsize=(7, 6.2))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax = fig.add_subplot(gs[0]); axp = fig.add_subplot(gs[1], sharex=ax)
    ax.errorbar(ctr, counts, yerr=np.sqrt(counts), fmt="o", ms=3, color="black",
                label="Data")
    ax.plot(xs, sig + bkg, color="#c1272d", lw=2, label="fit")
    ax.plot(xs, bkg, color="#3f7fbf", lw=1.6, ls="--", label="background")
    ax.fill_between(xs, bkg, sig + bkg, color="#c1272d", alpha=0.12, label="signal")
    ax.set_ylabel(f"Events / {bw*1e3:.0f} MeV")
    ax.set_ylim(bottom=0, top=1.28 * counts.max()); ax.set_xlim(lo, hi)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.legend(loc="upper right", fontsize=10)
    sob = res["Ns"] / max(res["Nb"], 1e-9)
    ax.text(0.035, 0.94,
            rf"$N_S={res['Ns']:.0f}$" "\n" rf"$N_B={res['Nb']:.0f}$" "\n"
            rf"$S/B={sob:.1f}$" "\n" rf"$\chi^2/\mathrm{{ndf}}={chi2:.0f}/{ndf}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=10)
    if _HAVE_HEP:
        hep.cms.label("Preliminary", ax=ax, data=True, rlabel="2018 (13 TeV)",
                      fontsize=13)
    axp.axhspan(-2, 2, color="0.85"); axp.axhspan(-1, 1, color="0.72")
    axp.axhline(0, color="#c1272d", lw=1)
    axp.plot(ctr, pull, "o", ms=3, color="black")
    axp.set_ylim(-5, 5); axp.set_ylabel("pull")
    axp.set_xlabel(r"$m(\mu\mu)$ [GeV]"); axp.set_xlim(lo, hi)
    fig.savefig(os.path.join(outdir, "mass_fit.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# 2.  READ DATA
# ════════════════════════════════════════════════════════════════════════════

def read_skim(path, tree_name=TREE_NAME):
    """Read pt, |η|, |dxy_sig|, pass from a ROOT probe-skim file."""
    print(f"  reading {path} …")
    
    if "hb" in path:
        suffix = "_morph"
    else:
        suffix = ""
    
    with uproot.open(path) as f:
        tree   = f[tree_name]
        want   = [PASS_BRANCH, "pt", "eta", f"dxy_sig{suffix}"]
        has_mass = MASS_BRANCH in tree.keys()
        if has_mass:
            want.append(MASS_BRANCH)
        arrays = tree.arrays(want, library="np")

    out = {
        "pt":      arrays["pt"].astype(np.float32),
        "eta":     np.abs(arrays["eta"]).astype(np.float32),
        "dxy_sig": np.abs(arrays[f"dxy_sig{suffix}"]).astype(np.float32),
        "pass":    arrays[PASS_BRANCH].astype(np.float32),
    }
    if has_mass:
        out["mass"] = arrays[MASS_BRANCH].astype(np.float32)
    n = len(out["pass"]); k = int(out["pass"].sum())
    print(f"    {n:,} probes,  {k:,} passing  ({100*k/n:.1f}%)")
    return out


print("Reading files …")
samples = {name: read_skim(path) for name, path in FILES.items()}


# ── sPlot: one inclusive mass fit of the DATA -> per-event signal weights ────
def _mask_sample(s, m):
    return {k: (v[m] if isinstance(v, np.ndarray) and v.shape == m.shape else v)
            for k, v in s.items()}


if USE_SPLOT:
    ds = samples["data"]
    if "mass" not in ds:
        raise SystemExit(
            f"USE_SPLOT=True but branch '{MASS_BRANCH}' is not in the data file. "
            f"Add the dimuon mass to the probe skim, or set USE_SPLOT=False.")
    win = np.abs(ds["mass"] - JPSI_MASS) < MASS_HALFWIN
    ds = _mask_sample(ds, win)                      # restrict to the fit window
    print(f"\nsPlot: fitting {int(win.sum()):,} data probes in "
          f"|m - {JPSI_MASS}| < {MASS_HALFWIN} GeV …")
    _fit = fit_mass(ds["mass"])
    ds["weight"] = splot_signal_weights(ds["mass"], _fit)
    print(f"  fit valid={_fit['valid']}  Ns={_fit['Ns']:.0f}  Nb={_fit['Nb']:.0f}  "
          f"S/B={_fit['Ns']/max(_fit['Nb'],1e-9):.1f}  "
          f"sum(sWeights)={ds['weight'].sum():.0f}")
    plot_mass_fit(ds["mass"], _fit, "plots")
    # keep the sWeights (already computed on the full fit window), then restrict
    # the efficiency sample to |m - m_Jpsi| < EFF_HALFWIN. The surviving events
    # keep their valid weights; the dropped far sidebands carry ~zero signal
    # weight, so background is still subtracted.
    eff_win = np.abs(ds["mass"] - JPSI_MASS) < EFF_HALFWIN
    ds = _mask_sample(ds, eff_win)
    print(f"  efficiency sample: {int(eff_win.sum()):,} events within "
          f"|m - {JPSI_MASS}| < {EFF_HALFWIN} GeV  "
          f"(sum(sWeights)={ds['weight'].sum():.0f})")
    samples["data"] = ds
    samples["mc"]["weight"] = np.ones(len(samples["mc"]["pass"]), dtype=float)
else:
    for _s in samples.values():
        _s["weight"] = np.ones(len(_s["pass"]), dtype=float)


# ════════════════════════════════════════════════════════════════════════════
# 3.  BINNED EFFICIENCY – 1D
# ════════════════════════════════════════════════════════════════════════════

def _eff_ci(sw, sw2, swp):
    """Weighted efficiency and Clopper-Pearson errors via effective counts.
    For unit weights this reduces exactly to the unweighted CP interval.
    Bins with non-positive signal yield (possible with sWeights in
    background-dominated tails) are returned as NaN -- no signal efficiency
    is defined there."""
    sw  = np.asarray(sw, float)
    sw2 = np.asarray(sw2, float)
    swp = np.asarray(swp, float)
    good = (sw > 0) & (sw2 > 0)
    eff  = np.where(good, np.clip(swp / np.where(good, sw, 1.0), 0.0, 1.0), np.nan)
    neff = np.where(good, sw * sw / np.where(good, sw2, 1.0), 0.0)
    keff = np.where(good, eff * neff, 0.0)
    lo, hi = cp_interval(keff, neff)
    lo = np.where(good, np.clip(lo, 0.0, None), np.nan)   # errors are non-negative
    hi = np.where(good, np.clip(hi, 0.0, None), np.nan)
    return eff, lo, hi, neff


def eff_1d(data, var, bins):
    x     = data[var]
    p     = data["pass"]
    w     = data.get("weight")
    if w is None:
        w = np.ones_like(p, dtype=float)
    edges = np.asarray(bins, dtype=float)
    nb    = len(edges) - 1
    sw = np.zeros(nb); sw2 = np.zeros(nb); swp = np.zeros(nb)

    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask   = (x >= lo) & (x < hi)
        wm     = w[mask]
        sw[i]  = wm.sum()
        sw2[i] = np.sum(wm * wm)
        swp[i] = np.sum(wm * p[mask])

    eff, lo_e, hi_e, neff = _eff_ci(sw, sw2, swp)

    return {
        "bin_edges":   edges.tolist(),
        "bin_centers": (0.5 * (edges[:-1] + edges[1:])).tolist(),
        "n_total":     sw.tolist(),      # weighted (signal) yield
        "n_pass":      swp.tolist(),
        "n_eff":       neff.tolist(),    # effective # of probes (for the CP error)
        "efficiency":  eff.tolist(),
        "err_lo":      lo_e.tolist(),
        "err_hi":      hi_e.tolist(),
    }


# ════════════════════════════════════════════════════════════════════════════
# 4.  BINNED EFFICIENCY – 2D
# ════════════════════════════════════════════════════════════════════════════

def eff_2d(data, var0, var1, bins0, bins1):
    x0, x1 = data[var0], data[var1]
    p      = data["pass"]
    w      = data.get("weight")
    if w is None:
        w = np.ones_like(p, dtype=float)
    e0     = np.asarray(bins0, dtype=float)
    e1     = np.asarray(bins1, dtype=float)
    n0, n1 = len(e0) - 1, len(e1) - 1

    sw  = np.zeros((n0, n1)); sw2 = np.zeros((n0, n1)); swp = np.zeros((n0, n1))

    for i, (lo0, hi0) in enumerate(zip(e0[:-1], e0[1:])):
        m0 = (x0 >= lo0) & (x0 < hi0)
        for j, (lo1, hi1) in enumerate(zip(e1[:-1], e1[1:])):
            mask     = m0 & (x1 >= lo1) & (x1 < hi1)
            wm       = w[mask]
            sw[i, j]  = wm.sum()
            sw2[i, j] = np.sum(wm * wm)
            swp[i, j] = np.sum(wm * p[mask])

    eff, lo, hi_e, neff = _eff_ci(sw, sw2, swp)

    return {
        "bin_edges_x":   e0.tolist(),
        "bin_edges_y":   e1.tolist(),
        "bin_centers_x": (0.5 * (e0[:-1] + e0[1:])).tolist(),
        "bin_centers_y": (0.5 * (e1[:-1] + e1[1:])).tolist(),
        "n_total":       sw.tolist(),
        "n_pass":        swp.tolist(),
        "n_eff":         neff.tolist(),
        "efficiency":    eff.tolist(),
        "err_lo":        lo.tolist(),
        "err_hi":        hi_e.tolist(),
    }


# ════════════════════════════════════════════════════════════════════════════
# 5.  COMPUTE ALL EFFICIENCIES
# ════════════════════════════════════════════════════════════════════════════

results = {}
for name, data in samples.items():
    print(f"\nComputing efficiencies for {name} …")
    r = {"1d": {}, "2d": {}}
    for var, bins in BINS.items():
        print(f"  1D  {var}")
        r["1d"][var] = eff_1d(data, var, bins)
    for v0, v1 in PAIRS:
        key = f"{v0}_vs_{v1}"
        print(f"  2D  {key}")
        r["2d"][key] = eff_2d(data, v0, v1, BINS[v0], BINS[v1])
    results[name] = r


# ════════════════════════════════════════════════════════════════════════════
# 6.  SCALE FACTORS
# ════════════════════════════════════════════════════════════════════════════

print("\nComputing scale factors …")
sf_results = {"1d": {}, "2d": {}}

for var in BINS:
    rd = results["data"]["1d"][var]
    rm = results["mc"]["1d"][var]

    eff_d = np.array(rd["efficiency"]); lo_d = np.array(rd["err_lo"]); hi_d = np.array(rd["err_hi"])
    eff_m = np.array(rm["efficiency"]); lo_m = np.array(rm["err_lo"]); hi_m = np.array(rm["err_hi"])
    sf, sf_lo, sf_hi = sf_errors(eff_d, lo_d, hi_d, eff_m, lo_m, hi_m)

    sf_results["1d"][var] = {
        "bin_edges":   rd["bin_edges"],
        "bin_centers": rd["bin_centers"],
        "sf":       sf.tolist(),    "err_lo":  sf_lo.tolist(), "err_hi": sf_hi.tolist(),
        "eff_data": eff_d.tolist(), "eff_mc":  eff_m.tolist(),
    }

for v0, v1 in PAIRS:
    key = f"{v0}_vs_{v1}"
    rd = results["data"]["2d"][key]
    rm = results["mc"]["2d"][key]

    eff_d = np.array(rd["efficiency"]); lo_d = np.array(rd["err_lo"]); hi_d = np.array(rd["err_hi"])
    eff_m = np.array(rm["efficiency"]); lo_m = np.array(rm["err_lo"]); hi_m = np.array(rm["err_hi"])
    sf, sf_lo, sf_hi = sf_errors(eff_d, lo_d, hi_d, eff_m, lo_m, hi_m)

    sf_results["2d"][key] = {
        "bin_edges_x":   rd["bin_edges_x"],   "bin_edges_y":   rd["bin_edges_y"],
        "bin_centers_x": rd["bin_centers_x"], "bin_centers_y": rd["bin_centers_y"],
        "sf":       sf.tolist(),    "err_lo":  sf_lo.tolist(), "err_hi": sf_hi.tolist(),
        "eff_data": eff_d.tolist(), "eff_mc":  eff_m.tolist(),
    }


# ════════════════════════════════════════════════════════════════════════════
# 7.  SAVE JSON
# ════════════════════════════════════════════════════════════════════════════

def nan_to_none(obj):
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, list):
        return [nan_to_none(x) for x in obj]
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    return obj

for name in ["data", "mc"]:
    path = f"efficiency_{name}.json"
    with open(path, "w") as fout:
        json.dump(nan_to_none(results[name]), fout, indent=2)
    print(f"Saved {path}")

with open("scale_factors.json", "w") as fout:
    json.dump(nan_to_none(sf_results), fout, indent=2)
print("Saved scale_factors.json")


# ════════════════════════════════════════════════════════════════════════════
# 8.  PLOTS
# ════════════════════════════════════════════════════════════════════════════

COLORS = {"data": "#1f77b4", "mc": "#d62728"}


# ── 8a. helper: step-style efficiency curve + CP error bars ─────────────────

def _step_eff(ax, r, color, label, log_x=False):
    edges   = np.array(r["bin_edges"])
    centers = np.array(r["bin_centers"])
    eff     = np.array(r["efficiency"], dtype=float)
    lo      = np.array(r["err_lo"],     dtype=float)
    hi      = np.array(r["err_hi"],     dtype=float)

    # For log-x we draw each bin as a horizontal line between its edges
    if log_x:
        for i in range(len(eff)):
            if np.isfinite(eff[i]):
                ax.plot([edges[i], edges[i+1]], [eff[i], eff[i]],
                        color=color, linewidth=1.8,
                        label=label if i == 0 else "_nolegend_", zorder=3)
                # vertical connector to next bin
                if i < len(eff) - 1 and np.isfinite(eff[i+1]):
                    ax.plot([edges[i+1], edges[i+1]], [eff[i], eff[i+1]],
                            color=color, linewidth=1.8, zorder=3)
    else:
        ax.step(edges[:-1], eff, where="post", color=color,
                linewidth=1.8, label=label, zorder=3)
        ax.step([edges[-2], edges[-1]], [eff[-1], eff[-1]],
                where="post", color=color, linewidth=1.8, zorder=3)

    valid = np.isfinite(eff)
    ax.errorbar(centers[valid], eff[valid],
                yerr=[lo[valid], hi[valid]],
                fmt="none", ecolor=color, capsize=3, linewidth=1.2, zorder=4)


# ── 8b. helper: 2D heat-map with value + asymmetric uncertainty annotation ──

def _heatmap_2d(ax, fig,
                z, err_lo, err_hi,
                edges_x, edges_y,
                vmin=0.0, vmax=1.0, cmap="viridis",
                log_x=False, log_y=False,
                xlabel="", ylabel="", title="",
                cbar_label="",
                val_fmt=".3f", err_fmt=".3f",
                fontsize_val=7, fontsize_err=5.5,
                annotate=True):
    """
    Draw a pcolormesh heatmap.
    Each cell shows:
        <value>
        +<hi> / −<lo>
    Axes whose variable has LOG_SCALE=True are drawn in log scale.
    """
    z      = np.array(z,      dtype=float)    # shape (nx, ny)
    err_lo = np.array(err_lo, dtype=float)
    err_hi = np.array(err_hi, dtype=float)

    ex = np.asarray(edges_x)
    ey = np.asarray(edges_y)

    # pcolormesh expects (ny, nx) with standard orientation
    zT  = z.T
    loT = err_lo.T
    hiT = err_hi.T

    im = ax.pcolormesh(ex, ey, zT, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="flat")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xticks(ex)
        ax.tick_params(axis="x", which="minor", length=0)
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_yticks(ey)
        ax.tick_params(axis="y", which="minor", length=0)

    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title,   fontsize=12)

    # ── cell annotations ────────────────────────────────────────────────────
    if annotate:
        cx = 0.5 * (ex[:-1] + ex[1:])   # geometric mean for log, arithmetic for linear
        cy = 0.5 * (ey[:-1] + ey[1:])
        if log_x:
            cx = np.sqrt(ex[:-1] * ex[1:])
        if log_y:
            cy = np.sqrt(ey[:-1] * ey[1:])

        mid = vmin + 0.5 * (vmax - vmin)

        for i, xi in enumerate(cx):
            for j, yj in enumerate(cy):
                val = zT[j, i]
                lo  = loT[j, i]
                hi  = hiT[j, i]
                if not np.isfinite(val):
                    continue

                txt_col = "white" if val < mid else "black"

                # Central value on the first line
                line1 = format(val, val_fmt)
                # Uncertainty on two short lines below
                if np.isfinite(lo) and np.isfinite(hi):
                    line2 = f"+{format(hi, err_fmt)}"
                    line3 = f"−{format(lo, err_fmt)}"
                else:
                    line2 = line3 = ""

                # Draw central value
                ax.text(xi, yj,  line1,
                        ha="center", va="center" if not line2 else "bottom",
                        fontsize=fontsize_val, color=txt_col,
                        fontweight="bold", transform=ax.transData,
                        linespacing=1.0)

                if line2:
                    # Draw uncertainty lines slightly below centre
                    ax.text(xi, yj,
                            f"\n{line2}\n{line3}",
                            ha="center", va="top",
                            fontsize=fontsize_err, color=txt_col,
                            transform=ax.transData,
                            linespacing=1.1)

    return im


# ────────────────────────────────────────────────────────────────────────────
# 8c. 1D efficiency plots  (data & MC) + SF ratio panel
# ────────────────────────────────────────────────────────────────────────────
print("\nPlotting 1D efficiencies …")

for var, xlabel in LABELS.items():
    log_x = LOG_SCALE[var]

    fig, (ax_eff, ax_sf) = plt.subplots(
        2, 1, figsize=(8, 7),
        gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.08},
        sharex=True
    )
    if log_x:
        ax_eff.set_xscale("log")
        ax_sf.set_xscale("log")
        for ax in (ax_eff, ax_sf):
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    # ── top panel: efficiency ──────────────────────────────────────────────
    for name in ["data", "mc"]:
        _step_eff(ax_eff, results[name]["1d"][var],
                  color=COLORS[name], label=name.upper(), log_x=log_x)

    edges_for_ticks = np.array(results["data"]["1d"][var]["bin_edges"])
    ax_eff.set_xticks(edges_for_ticks)
    ax_eff.set_ylabel("Trigger efficiency", fontsize=12)
    ax_eff.set_ylim(-0.05, 1.15)
    ax_eff.legend(fontsize=11, loc="lower right")
    ax_eff.grid(True, alpha=0.25, which="both")
    ax_eff.set_title(
        rf"$\varepsilon$ vs {xlabel.replace('probe ', '')}",
        fontsize=13, fontweight="bold"
    )

    # ── bottom panel: scale factor ─────────────────────────────────────────
    sfr     = sf_results["1d"][var]
    edges   = np.array(sfr["bin_edges"])
    centers = np.array(sfr["bin_centers"])
    sf      = np.array(sfr["sf"],     dtype=float)
    sf_lo   = np.array(sfr["err_lo"], dtype=float)
    sf_hi   = np.array(sfr["err_hi"], dtype=float)
    valid   = np.isfinite(sf)

    ax_sf.axhline(1.0, color="grey", linestyle="--", linewidth=1.0)

    if log_x:
        for i in range(len(sf)):
            if np.isfinite(sf[i]):
                ax_sf.plot([edges[i], edges[i+1]], [sf[i], sf[i]],
                           color="black", linewidth=1.8, zorder=3)
                if i < len(sf) - 1 and np.isfinite(sf[i+1]):
                    ax_sf.plot([edges[i+1], edges[i+1]], [sf[i], sf[i+1]],
                               color="black", linewidth=1.8, zorder=3)
    else:
        ax_sf.step(edges[:-1], sf, where="post", color="black", linewidth=1.8, zorder=3)
        ax_sf.step([edges[-2], edges[-1]], [sf[-1], sf[-1]],
                   where="post", color="black", linewidth=1.8, zorder=3)

    ax_sf.errorbar(centers[valid], sf[valid],
                   yerr=[sf_lo[valid], sf_hi[valid]],
                   fmt="none", ecolor="black", capsize=3, linewidth=1.2, zorder=4)

    ax_sf.set_xticks(edges)
    ax_sf.tick_params(axis="x", rotation=45)
    ax_sf.set_ylabel("SF = Data/MC", fontsize=11)
    ax_sf.set_xlabel(xlabel, fontsize=12)
    ax_sf.grid(True, alpha=0.25, which="both")

    finite_sf = sf[valid]
    if len(finite_sf):
        pad = max(0.15, 3 * np.nanmax(np.concatenate([sf_lo[valid], sf_hi[valid]])))
        ax_sf.set_ylim(
            max(0.0, np.nanmin(finite_sf) - pad),
            min(2.0, np.nanmax(finite_sf) + pad)
        )

    fig.tight_layout()
    fname = f"plots/eff_sf_1d_{var}.png"
    fname = f"plots/eff_sf_1d_{var}.pdf"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ────────────────────────────────────────────────────────────────────────────
# 8d. 2D efficiency + SF heat maps – TWO VERSIONS per pair
#
#   VERSION A  –  compact row (data | MC | SF), NO cell text
#                 → plots/eff_sf_2d_{key}_compact.pdf
#
#   VERSION B  –  one large figure per panel (data, MC, SF), WITH cell text
#                 → plots/eff_sf_2d_{key}_data.pdf
#                 → plots/eff_sf_2d_{key}_mc.pdf
#                 → plots/eff_sf_2d_{key}_sf.pdf
# ────────────────────────────────────────────────────────────────────────────
print("Plotting 2D efficiencies …")

for v0, v1 in PAIRS:
    key    = f"{v0}_vs_{v1}"
    xlabel = LABELS[v0]
    ylabel = LABELS[v1]
    log_x  = LOG_SCALE[v0]
    log_y  = LOG_SCALE[v1]

    # ── pre-compute SF colour scale (shared between both versions) ────────
    sfr     = sf_results["2d"][key]
    sf_arr  = np.array(sfr["sf"], dtype=float)
    abs_dev = np.nanpercentile(np.abs(sf_arr[np.isfinite(sf_arr)] - 1.0), 95)
    abs_dev = np.clip(abs_dev, 0.02, 0.5)
    sf_vmin, sf_vmax = 1.0 - abs_dev, 1.0 + abs_dev

    suptitle = (
        rf"$\varepsilon$ and SF:  {xlabel.replace('probe ', '')} "
        rf"vs {ylabel.replace('probe ', '')}"
    )

    # ══════════════════════════════════════════════════════════════════════
    # VERSION A – compact 1×3, NO annotations
    # ══════════════════════════════════════════════════════════════════════
    fig_c, axes_c = plt.subplots(1, 3, figsize=(22, 6))
    fig_c.suptitle(suptitle, fontsize=14, fontweight="bold")

    for col, name in enumerate(["data", "mc"]):
        r = results[name]["2d"][key]
        _heatmap_2d(
            axes_c[col], fig_c,
            z       = r["efficiency"],
            err_lo  = r["err_lo"],
            err_hi  = r["err_hi"],
            edges_x = r["bin_edges_x"],
            edges_y = r["bin_edges_y"],
            vmin=0.0, vmax=1.0, cmap="viridis",
            log_x=log_x, log_y=log_y,
            xlabel=xlabel, ylabel=ylabel,
            title=f"{name.upper()} efficiency",
            cbar_label="Efficiency",
            annotate=False,          # ← no cell text
        )

    _heatmap_2d(
        axes_c[2], fig_c,
        z       = sfr["sf"],
        err_lo  = sfr["err_lo"],
        err_hi  = sfr["err_hi"],
        edges_x = sfr["bin_edges_x"],
        edges_y = sfr["bin_edges_y"],
        vmin=sf_vmin, vmax=sf_vmax, cmap="RdBu_r",
        log_x=log_x, log_y=log_y,
        xlabel=xlabel, ylabel=ylabel,
        title="Scale factor (Data / MC)",
        cbar_label="SF",
        annotate=False,              # ← no cell text
    )

    fig_c.tight_layout()
    fname_c = f"plots/eff_sf_2d_{key}_compact.pdf"
    fig_c.savefig(fname_c, dpi=150, bbox_inches="tight")
    plt.close(fig_c)
    print(f"  saved {fname_c}")

    # ══════════════════════════════════════════════════════════════════════
    # VERSION B – individual large figures WITH annotations
    # ══════════════════════════════════════════════════════════════════════

    # Figure dimensions: wider bins need more horizontal room;
    # choose a generous base size so text stays readable.
    n_bins_x = len(sfr["bin_edges_x"]) - 1
    n_bins_y = len(sfr["bin_edges_y"]) - 1
    fig_w = max(14, 0.9 * n_bins_x)
    fig_h = max(8,  0.9 * n_bins_y)

    # font sizes that scale with figure area
    fs_val = max(7,  min(11, int(fig_w * 0.45)))
    fs_err = max(5,  min(8,  int(fig_w * 0.32)))

    # -- data efficiency --------------------------------------------------
    fig_d, ax_d = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    fig_d.suptitle(suptitle, fontsize=13, fontweight="bold")
    r_d = results["data"]["2d"][key]
    _heatmap_2d(
        ax_d, fig_d,
        z       = r_d["efficiency"],
        err_lo  = r_d["err_lo"],
        err_hi  = r_d["err_hi"],
        edges_x = r_d["bin_edges_x"],
        edges_y = r_d["bin_edges_y"],
        vmin=0.0, vmax=1.0, cmap="viridis",
        log_x=log_x, log_y=log_y,
        xlabel=xlabel, ylabel=ylabel,
        title="DATA efficiency",
        cbar_label="Efficiency",
        val_fmt=".3f", err_fmt=".3f",
        fontsize_val=fs_val, fontsize_err=fs_err,
        annotate=True,
    )
    fig_d.tight_layout()
    fname_d = f"plots/eff_sf_2d_{key}_data.pdf"
    fig_d.savefig(fname_d, dpi=150, bbox_inches="tight")
    plt.close(fig_d)
    print(f"  saved {fname_d}")

    # -- MC efficiency ----------------------------------------------------
    fig_m, ax_m = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    fig_m.suptitle(suptitle, fontsize=13, fontweight="bold")
    r_m = results["mc"]["2d"][key]
    _heatmap_2d(
        ax_m, fig_m,
        z       = r_m["efficiency"],
        err_lo  = r_m["err_lo"],
        err_hi  = r_m["err_hi"],
        edges_x = r_m["bin_edges_x"],
        edges_y = r_m["bin_edges_y"],
        vmin=0.0, vmax=1.0, cmap="viridis",
        log_x=log_x, log_y=log_y,
        xlabel=xlabel, ylabel=ylabel,
        title="MC efficiency",
        cbar_label="Efficiency",
        val_fmt=".3f", err_fmt=".3f",
        fontsize_val=fs_val, fontsize_err=fs_err,
        annotate=True,
    )
    fig_m.tight_layout()
    fname_m = f"plots/eff_sf_2d_{key}_mc.pdf"
    fig_m.savefig(fname_m, dpi=150, bbox_inches="tight")
    plt.close(fig_m)
    print(f"  saved {fname_m}")

    # -- scale factor -----------------------------------------------------
    fig_s, ax_s = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    fig_s.suptitle(suptitle, fontsize=13, fontweight="bold")
    _heatmap_2d(
        ax_s, fig_s,
        z       = sfr["sf"],
        err_lo  = sfr["err_lo"],
        err_hi  = sfr["err_hi"],
        edges_x = sfr["bin_edges_x"],
        edges_y = sfr["bin_edges_y"],
        vmin=sf_vmin, vmax=sf_vmax, cmap="RdBu_r",
        log_x=log_x, log_y=log_y,
        xlabel=xlabel, ylabel=ylabel,
        title="Scale factor (Data / MC)",
        cbar_label="SF",
        val_fmt=".3f", err_fmt=".3f",
        fontsize_val=fs_val, fontsize_err=fs_err,
        annotate=True,
    )
    fig_s.tight_layout()
    fname_s = f"plots/eff_sf_2d_{key}_sf.pdf"
    fig_s.savefig(fname_s, dpi=150, bbox_inches="tight")
    plt.close(fig_s)
    print(f"  saved {fname_s}")


# ────────────────────────────────────────────────────────────────────────────
# 8e-NEW.  ε(pT) sliced in dxy_sig – independence check
#
# For each dxy_sig bin, draw ε(pT) as a separate coloured line.
# If pT and dxy_sig are independent the curves should be parallel / identical.
# One figure per sample (data / MC); both saved to plots/.
# ────────────────────────────────────────────────────────────────────────────
print("Plotting ε(pT) per dxy_sig slice …")

# The 2D result is stored with axes (pt-bin, dxy_sig-bin) – shape (n_pt, n_dxy)
_key_2d = "pt_vs_dxy_sig"

# Colour palette: one colour per dxy_sig slice (tab10 up to 10 slices, then tab20)
_n_dxy = len(DXY_BINS) - 1
_cmap  = plt.get_cmap("tab10" if _n_dxy <= 10 else "tab20")
_colors_dxy = [_cmap(i / max(_n_dxy - 1, 1)) for i in range(_n_dxy)]

for _sample_name in ["data", "mc"]:
    _r2d   = results[_sample_name]["2d"][_key_2d]
    _eff2d = np.array(_r2d["efficiency"], dtype=float)   # (n_pt, n_dxy)
    _lo2d  = np.array(_r2d["err_lo"],     dtype=float)
    _hi2d  = np.array(_r2d["err_hi"],     dtype=float)
    _pt_edges   = np.array(_r2d["bin_edges_x"])
    _pt_centers = np.array(_r2d["bin_centers_x"])
    _dxy_edges  = np.array(_r2d["bin_edges_y"])

    fig_ind, ax_ind = plt.subplots(figsize=(10, 6))

    for _j in range(_n_dxy):
        _lo_edge  = _dxy_edges[_j]
        _hi_edge  = _dxy_edges[_j + 1]
        _label    = rf"$|d_{{xy}}/\sigma| \in [{_lo_edge:.4g},\,{_hi_edge:.4g})$"
        _col      = _colors_dxy[_j]

        _eff_slice = _eff2d[:, _j]   # shape (n_pt,)
        _lo_slice  = _lo2d[:, _j]
        _hi_slice  = _hi2d[:, _j]

        # Step-style line (same convention as _step_eff)
        _valid = np.isfinite(_eff_slice)
        if not _valid.any():
            continue

        # Draw horizontal steps between bin edges
        for _i in range(len(_eff_slice)):
            if not np.isfinite(_eff_slice[_i]):
                continue
            ax_ind.plot(
                [_pt_edges[_i], _pt_edges[_i + 1]],
                [_eff_slice[_i], _eff_slice[_i]],
                color=_col, linewidth=2.0,
                label=_label if _i == 0 else "_nolegend_",
                zorder=3,
            )
            # Vertical connector to next bin
            if _i < len(_eff_slice) - 1 and np.isfinite(_eff_slice[_i + 1]):
                ax_ind.plot(
                    [_pt_edges[_i + 1], _pt_edges[_i + 1]],
                    [_eff_slice[_i], _eff_slice[_i + 1]],
                    color=_col, linewidth=2.0, zorder=3,
                )

        # Error bars at bin centres
        ax_ind.errorbar(
            _pt_centers[_valid], _eff_slice[_valid],
            yerr=[_lo_slice[_valid], _hi_slice[_valid]],
            fmt="none", ecolor=_col, capsize=3, linewidth=1.2, zorder=4,
        )

    ax_ind.set_xlabel(LABELS["pt"],     fontsize=12)
    ax_ind.set_ylabel("Trigger efficiency", fontsize=12)
    ax_ind.set_ylim(-0.05, 1.15)
    ax_ind.set_xticks(_pt_edges)
    ax_ind.tick_params(axis="x", rotation=45)
    ax_ind.grid(True, alpha=0.25)
    ax_ind.legend(fontsize=9, loc="lower right", framealpha=0.85,
                  title=r"$|d_{xy}/\sigma_{xy}|$ slice")
    ax_ind.set_title(
        rf"{_sample_name.upper()} — $\varepsilon(p_T)$ in each $|d_{{xy}}/\sigma_{{xy}}|$ bin"
        "\n(curves overlap → variables independent)",
        fontsize=12, fontweight="bold",
    )

    fig_ind.tight_layout()
    _fname_ind = f"plots/eff_pt_per_dxy_slice_{_sample_name}.pdf"
    fig_ind.savefig(_fname_ind, dpi=150, bbox_inches="tight")
    plt.close(fig_ind)
    print(f"  saved {_fname_ind}")

# ── combined data + MC on one canvas (2 rows) ────────────────────────────────
fig_both, axes_both = plt.subplots(2, 1, figsize=(10, 11), sharex=True)
fig_both.suptitle(
    r"$\varepsilon(p_T)$ per $|d_{xy}/\sigma_{xy}|$ slice — independence check",
    fontsize=13, fontweight="bold",
)

for _ax, _sample_name in zip(axes_both, ["data", "mc"]):
    _r2d   = results[_sample_name]["2d"][_key_2d]
    _eff2d = np.array(_r2d["efficiency"], dtype=float)
    _lo2d  = np.array(_r2d["err_lo"],     dtype=float)
    _hi2d  = np.array(_r2d["err_hi"],     dtype=float)
    _pt_edges   = np.array(_r2d["bin_edges_x"])
    _pt_centers = np.array(_r2d["bin_centers_x"])
    _dxy_edges  = np.array(_r2d["bin_edges_y"])

    for _j in range(_n_dxy):
        _lo_edge = _dxy_edges[_j]
        _hi_edge = _dxy_edges[_j + 1]
        _label   = rf"$[{_lo_edge:.4g},\,{_hi_edge:.4g})$"
        _col     = _colors_dxy[_j]
        _eff_slice = _eff2d[:, _j]
        _lo_slice  = _lo2d[:, _j]
        _hi_slice  = _hi2d[:, _j]
        _valid = np.isfinite(_eff_slice)
        if not _valid.any():
            continue

        for _i in range(len(_eff_slice)):
            if not np.isfinite(_eff_slice[_i]):
                continue
            _ax.plot(
                [_pt_edges[_i], _pt_edges[_i + 1]],
                [_eff_slice[_i], _eff_slice[_i]],
                color=_col, linewidth=2.0,
                label=_label if _i == 0 else "_nolegend_",
                zorder=3,
            )
            if _i < len(_eff_slice) - 1 and np.isfinite(_eff_slice[_i + 1]):
                _ax.plot(
                    [_pt_edges[_i + 1], _pt_edges[_i + 1]],
                    [_eff_slice[_i], _eff_slice[_i + 1]],
                    color=_col, linewidth=2.0, zorder=3,
                )

        _ax.errorbar(
            _pt_centers[_valid], _eff_slice[_valid],
            yerr=[_lo_slice[_valid], _hi_slice[_valid]],
            fmt="none", ecolor=_col, capsize=3, linewidth=1.2, zorder=4,
        )

    _ax.set_ylabel("Trigger efficiency", fontsize=11)
    _ax.set_ylim(-0.05, 1.15)
    _ax.grid(True, alpha=0.25)
    _ax.set_title(_sample_name.upper(), fontsize=11, fontweight="bold", loc="left")
    _ax.legend(fontsize=8, loc="lower right", framealpha=0.85, ncol=2,
               title=r"$|d_{xy}/\sigma|$ slice")

axes_both[-1].set_xlabel(LABELS["pt"], fontsize=12)
axes_both[-1].set_xticks(_pt_edges)
axes_both[-1].tick_params(axis="x", rotation=45)

fig_both.tight_layout()
_fname_both = "plots/eff_pt_per_dxy_slice_combined.pdf"
fig_both.savefig(_fname_both, dpi=150, bbox_inches="tight")
plt.close(fig_both)
print(f"  saved {_fname_both}")


# ────────────────────────────────────────────────────────────────────────────
# 8e. Summary: all 1D variables in one figure (2 rows × 3 columns)
# ────────────────────────────────────────────────────────────────────────────
print("Plotting summary figure …")

fig_sum, axes_sum = plt.subplots(
    2, 3, figsize=(18, 10),
    gridspec_kw={"height_ratios": [3, 1.4]}
)
fig_sum.suptitle("Trigger efficiency and scale factors – summary", fontsize=14)

for col, (var, xlabel) in enumerate(LABELS.items()):
    log_x   = LOG_SCALE[var]
    ax_top  = axes_sum[0, col]
    ax_bot  = axes_sum[1, col]

    if log_x:
        for ax in (ax_top, ax_bot):
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    # ── efficiency ─────────────────────────────────────────────────────────
    for name in ["data", "mc"]:
        _step_eff(ax_top, results[name]["1d"][var],
                  color=COLORS[name], label=name.upper(), log_x=log_x)

    edges_t = np.array(results["data"]["1d"][var]["bin_edges"])
    ax_top.set_xticks(edges_t)
    ax_top.set_ylim(-0.05, 1.15)
    ax_top.set_ylabel("Efficiency" if col == 0 else "", fontsize=11)
    ax_top.set_title(xlabel, fontsize=12)
    ax_top.grid(True, alpha=0.25, which="both")
    if col == 0:
        ax_top.legend(fontsize=10)

    # ── SF ─────────────────────────────────────────────────────────────────
    sfr     = sf_results["1d"][var]
    edges   = np.array(sfr["bin_edges"])
    centers = np.array(sfr["bin_centers"])
    sf      = np.array(sfr["sf"],     dtype=float)
    sf_lo   = np.array(sfr["err_lo"], dtype=float)
    sf_hi   = np.array(sfr["err_hi"], dtype=float)
    valid   = np.isfinite(sf)

    ax_bot.axhline(1.0, color="grey", linestyle="--", linewidth=1.0)

    if log_x:
        for i in range(len(sf)):
            if np.isfinite(sf[i]):
                ax_bot.plot([edges[i], edges[i+1]], [sf[i], sf[i]],
                            color="black", linewidth=1.8, zorder=3)
                if i < len(sf) - 1 and np.isfinite(sf[i+1]):
                    ax_bot.plot([edges[i+1], edges[i+1]], [sf[i], sf[i+1]],
                                color="black", linewidth=1.8, zorder=3)
    else:
        ax_bot.step(edges[:-1], sf, where="post", color="black", linewidth=1.8)
        ax_bot.step([edges[-2], edges[-1]], [sf[-1], sf[-1]],
                    where="post", color="black", linewidth=1.8)

    ax_bot.errorbar(centers[valid], sf[valid],
                    yerr=[sf_lo[valid], sf_hi[valid]],
                    fmt="none", ecolor="black", capsize=3, linewidth=1.2)
    ax_bot.set_xticks(edges)
    ax_bot.tick_params(axis="x", rotation=45)
    ax_bot.set_ylabel("SF" if col == 0 else "", fontsize=11)
    ax_bot.set_xlabel(xlabel, fontsize=11)
    ax_bot.grid(True, alpha=0.25, which="both")

    finite_sf = sf[valid]
    if len(finite_sf):
        pad = max(0.15, 3 * np.nanmax(np.concatenate([sf_lo[valid], sf_hi[valid]])))
        ax_bot.set_ylim(
            max(0.0, np.nanmin(finite_sf) - pad),
            min(2.0, np.nanmax(finite_sf) + pad)
        )

plt.tight_layout()
fname_sum = "plots/summary_eff_sf.png"
fname_sum = "plots/summary_eff_sf.pdf"
fig_sum.savefig(fname_sum, dpi=150, bbox_inches="tight")
plt.close(fig_sum)
print(f"  saved {fname_sum}")


# ════════════════════════════════════════════════════════════════════════════
# 8f.  3-panel indexed heatmap  –  pT × |dxy/σ|
#
#  Bins chosen to match the reference histogram (variable-width, physics-
#  motivated).  All cells are drawn at EQUAL visual width (integer indices
#  on both axes) so narrow bins remain readable; the actual range is shown
#  as the tick label on each cell boundary.
#
#  Three axes side-by-side:
#      [0] data efficiency
#      [1] MC efficiency
#      [2] scale factor  (Data / MC)
# ════════════════════════════════════════════════════════════════════════════
print("Plotting indexed 3-panel pT × dxy_sig heatmap …")

# ── Reference bin edges (matching the uploaded reference plot) ───────────────
_PT_REF  = np.array([6.0, 7.0, 8.0, 8.5, 9.0, 10.0, 10.5, 11.0, 12.0, 20.0, 100.0])
_DXY_REF = np.array([0.0, 3.0, 3.5,  4.0, 5.0,  6.0,  8.0, 10.0, 20.0, 500.0])

_n_pt_ref  = len(_PT_REF)  - 1   # 10
_n_dxy_ref = len(_DXY_REF) - 1   #  9

# ── Tick labels: "[lo, hi)" for each bin ────────────────────────────────────
def _bin_labels(edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        lo_s = f"{lo:g}";  hi_s = f"{hi:g}"
        out.append(f"[{lo_s}, {hi_s})")
    return out

_pt_labels  = _bin_labels(_PT_REF)
_dxy_labels = _bin_labels(_DXY_REF)

# Integer cell centres and edges for pcolormesh
_ix = np.arange(_n_pt_ref  + 1)   # 0 … n_pt_ref
_iy = np.arange(_n_dxy_ref + 1)   # 0 … n_dxy_ref
_cx = 0.5 * (_ix[:-1] + _ix[1:])  # cell centres x
_cy = 0.5 * (_iy[:-1] + _iy[1:])  # cell centres y

# ── Re-compute 2D efficiency with reference bins ─────────────────────────────
_ref_r = {}
for _sname, _sdata in samples.items():
    _ref_r[_sname] = eff_2d(_sdata, "pt", "dxy_sig", _PT_REF, _DXY_REF)

# ── Scale factor on the reference grid ──────────────────────────────────────
_rd = np.array(_ref_r["data"]["efficiency"], dtype=float)  # (n_pt, n_dxy)
_rm = np.array(_ref_r["mc"]["efficiency"],   dtype=float)
_lo_d = np.array(_ref_r["data"]["err_lo"],   dtype=float)
_hi_d = np.array(_ref_r["data"]["err_hi"],   dtype=float)
_lo_m = np.array(_ref_r["mc"]["err_lo"],     dtype=float)
_hi_m = np.array(_ref_r["mc"]["err_hi"],     dtype=float)

_sf_ref, _sf_lo_ref, _sf_hi_ref = sf_errors(
    _rd, _lo_d, _hi_d,
    _rm, _lo_m, _hi_m,
)

# Symmetric SF colour scale (95th-percentile of |SF−1|, clipped)
_abs_dev_ref = np.nanpercentile(
    np.abs(_sf_ref[np.isfinite(_sf_ref)] - 1.0), 95
)
_abs_dev_ref = np.clip(_abs_dev_ref, 0.02, 0.5)
_sf_vmin_ref = 1.0 - _abs_dev_ref
_sf_vmax_ref = 1.0 + _abs_dev_ref

# ── Helper: annotate one cell ────────────────────────────────────────────────
def _annotate_cell(ax, ix, iy, val, lo, hi,
                   vmin, vmax, fontsize_val=8, fontsize_err=6):
    """Write  val\n+hi\n−lo  in the cell; colour chosen for contrast."""
    if not np.isfinite(val):
        return
    mid = vmin + 0.5 * (vmax - vmin)
    tc  = "white" if val < mid else "black"
    ax.text(ix, iy, f"{val:.3f}",
            ha="center", va="bottom", fontsize=fontsize_val,
            color=tc, fontweight="bold")
    if np.isfinite(lo) and np.isfinite(hi):
        ax.text(ix, iy, f"\n±{0.5*(lo+hi):.3f}",
                ha="center", va="top", fontsize=fontsize_err, color=tc)

# ── Build figure ─────────────────────────────────────────────────────────────
_fig_ref, _axes_ref = plt.subplots(
    1, 3,
    figsize=(22, 8),
    constrained_layout=True,
)
_fig_ref.suptitle(
    r"Muon trigger efficiency & scale factor  –  $p_T$ × $|d_{xy}/\sigma_{xy}|$"
    "\n(equal-width cells, variable-width physics bins)",
    fontsize=13, fontweight="bold",
)

# Panel specs: (data array, lo, hi, cmap, vmin, vmax, title, cbar label)
_panels = [
    (_rd,      _lo_d,      _hi_d,      "viridis",  0.0, 1.0,
     "DATA efficiency",       "Efficiency"),
    (_rm,      _lo_m,      _hi_m,      "viridis",  0.0, 1.0,
     "MC efficiency",         "Efficiency"),
    (_sf_ref,  _sf_lo_ref, _sf_hi_ref, "RdBu_r",   _sf_vmin_ref, _sf_vmax_ref,
     "Scale factor (Data / MC)", "SF"),
]

for _ax, (_z, _elo, _ehi, _cmap, _vmin, _vmax, _title, _cbarlbl) in \
        zip(_axes_ref, _panels):

    # pcolormesh on integer grid → every cell identical visual size
    _zT = _z.T          # pcolormesh wants (n_dxy, n_pt)
    _im = _ax.pcolormesh(
        _ix, _iy, _zT,
#         cmap=_cmap, vmin=_vmin, vmax=_vmax,
        cmap="cividis", vmin=0, vmax=1,  # <-- CHANGED HERE
        shading="flat",
    )
    _fig_ref.colorbar(_im, ax=_ax, label=_cbarlbl, fraction=0.046, pad=0.02)

    # Cell annotations
    _eloT = _elo.T
    _ehiT = _ehi.T
    for _i, _xi in enumerate(_cx):
        for _j, _yj in enumerate(_cy):
            _annotate_cell(
                _ax, _xi, _yj,
                _zT[_j, _i], _eloT[_j, _i], _ehiT[_j, _i],
                0, 1,                    # <-- CHANGED HERE (updates text color contrast logic)
#                 _vmin, _vmax,
                fontsize_val=8, fontsize_err=6,
            )

    # Replace integer ticks with bin-range labels
    _ax.set_xticks(_cx)
    _ax.set_xticklabels(_pt_labels, rotation=45, ha="right", fontsize=8)
    _ax.set_yticks(_cy)
    _ax.set_yticklabels(_dxy_labels, fontsize=8)

    # White grid lines along cell boundaries
    for _gx in _ix:
        _ax.axvline(_gx, color="white", linewidth=0.6)
    for _gy in _iy:
        _ax.axhline(_gy, color="white", linewidth=0.6)

    _ax.set_xlabel(r"probe $p_T$ [GeV]",        fontsize=11)
    _ax.set_ylabel(r"probe $|d_{xy}/\sigma_{xy}|$", fontsize=11)
    _ax.set_title(_title, fontsize=12, fontweight="bold")
    _ax.set_xlim(0, _n_pt_ref)
    _ax.set_ylim(0, _n_dxy_ref)

_fname_ref = "plots/eff_sf_2d_pt_dxy_indexed.pdf"
_fig_ref.savefig(_fname_ref, dpi=150, bbox_inches="tight")
plt.close(_fig_ref)
print(f"  saved {_fname_ref}")


# ════════════════════════════════════════════════════════════════════════════
# 9.  PRINT QUICK SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 72)
print("SCALE FACTOR SUMMARY  (1D, bin-by-bin)")
print("═" * 72)

for var in BINS:
    sfr   = sf_results["1d"][var]
    edges = sfr["bin_edges"]
    sf    = sfr["sf"];    sf_lo = sfr["err_lo"]; sf_hi = sfr["err_hi"]
    eff_d = sfr["eff_data"]; eff_m = sfr["eff_mc"]

    print(f"\n  {var}:")
    print(f"    {'Bin range':>22s}  {'ε_data':>7s}  {'ε_MC':>7s}  "
          f"{'SF':>7s}  {'−δSF':>7s}  {'+δSF':>7s}")
    print("    " + "─" * 67)

    def _f(x):
        try:
            return f"{float(x):.4f}" if x is not None and np.isfinite(float(x)) else "  ---  "
        except Exception:
            return "  ---  "

    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        print(f"    [{lo:>7.2f}, {hi:>7.2f})  "
              f"{_f(eff_d[i]):>7s}  {_f(eff_m[i]):>7s}  "
              f"{_f(sf[i]):>7s}  {_f(sf_lo[i]):>7s}  {_f(sf_hi[i]):>7s}")

print("\n" + "═" * 72)
print("Done.")
print("  JSON  : efficiency_data.json  efficiency_mc.json  scale_factors.json")
print("  Plots : plots/eff_sf_1d_*.pdf")
print("          plots/eff_sf_2d_*_compact.pdf   (3-up row, no cell text)")
print("          plots/eff_sf_2d_*_{data,mc,sf}.pdf  (large individual, annotated)")
print("          plots/summary_eff_sf.pdf")
print("═" * 72)




"""
save_numerical_outputs.py
=========================
Drop-in addition for efficiency_sf.py.

Expects the following variables to already exist in the namespace
(they are produced by sections 5 and 6 of efficiency_sf.py):

    results     – dict  {"data": {"1d": {...}, "2d": {...}},
                          "mc":   {"1d": {...}, "2d": {...}}}
    sf_results  – dict  {"1d": {...}, "2d": {...}}
    BINS        – dict  {var: bin_edges_array, ...}
    PAIRS       – list  [(var0, var1), ...]
    LABELS      – dict  {var: latex_label, ...}

Insert this block *after* section 7 (SAVE JSON) in efficiency_sf.py,
or run it as a standalone script by replacing the three variable
references at the bottom with actual imports / re-computations.

Outputs written to plots/
─────────────────────────
JSON (mirrors of the root-level files, kept together with the plots):
    plots/efficiency_data.json
    plots/efficiency_mc.json
    plots/scale_factors.json

CSV – 1D (one file per variable):
    plots/eff_1d_{var}.csv
    Columns: bin_lo, bin_hi, bin_center,
             n_total_data, n_pass_data, eff_data, err_lo_data, err_hi_data,
             n_total_mc,   n_pass_mc,   eff_mc,   err_lo_mc,   err_hi_mc,
             sf, sf_err_lo, sf_err_hi

CSV – 2D (one file per variable pair × quantity):
    plots/eff_2d_{var0}_vs_{var1}_data.csv
    plots/eff_2d_{var0}_vs_{var1}_mc.csv
    plots/eff_2d_{var0}_vs_{var1}_sf.csv
    Format: rows = var0 bins, cols = var1 bins
            Each cell: "value ±lo +hi"  (plain value for machine parsing below)

    Also flat (tidy) versions:
    plots/eff_2d_{var0}_vs_{var1}_tidy.csv
    Columns: bin_lo_x, bin_hi_x, bin_lo_y, bin_hi_y,
             n_total_data, n_pass_data, eff_data, err_lo_data, err_hi_data,
             n_total_mc,   n_pass_mc,   eff_mc,   err_lo_mc,   err_hi_mc,
             sf, sf_err_lo, sf_err_hi

NPZ – 2D arrays for programmatic use (all pairs in one file):
    plots/efficiency_2d_arrays.npz
"""

import csv
import json
import os

import numpy as np

# ── ensure output directory exists ──────────────────────────────────────────
os.makedirs("plots", exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# helpers
# ════════════════════════════════════════════════════════════════════════════

def _nan_to_none(obj):
    """Recursively replace NaN/Inf with None for JSON serialisation."""
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, list):
        return [_nan_to_none(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    return obj


def _fmt(x, digits=6):
    """Format a float for CSV output; return '' for NaN/None."""
    try:
        v = float(x)
        return "" if (np.isnan(v) or np.isinf(v)) else f"{v:.{digits}g}"
    except (TypeError, ValueError):
        return ""


# ════════════════════════════════════════════════════════════════════════════
# 1.  JSON mirrors in plots/
# ════════════════════════════════════════════════════════════════════════════

print("\n── Saving numerical outputs to plots/ ──────────────────────────────────")

for name in ["data", "mc"]:
    path = f"plots/efficiency_{name}.json"
    with open(path, "w") as fout:
        json.dump(_nan_to_none(results[name]), fout, indent=2)
    print(f"  {path}")

with open("plots/scale_factors.json", "w") as fout:
    json.dump(_nan_to_none(sf_results), fout, indent=2)
print("  plots/scale_factors.json")


# ════════════════════════════════════════════════════════════════════════════
# 2.  1D CSV  (one file per variable, data + MC + SF side-by-side)
# ════════════════════════════════════════════════════════════════════════════

HEADER_1D = [
    "bin_lo", "bin_hi", "bin_center",
    "n_total_data", "n_pass_data",
    "eff_data", "err_lo_data", "err_hi_data",
    "n_total_mc", "n_pass_mc",
    "eff_mc", "err_lo_mc", "err_hi_mc",
    "sf", "sf_err_lo", "sf_err_hi",
]

for var in BINS:
    rd  = results["data"]["1d"][var]
    rm  = results["mc"]["1d"][var]
    sfr = sf_results["1d"][var]

    edges   = rd["bin_edges"]
    centers = rd["bin_centers"]
    n_bins  = len(centers)

    path = f"plots/eff_1d_{var}.csv"
    with open(path, "w", newline="") as fout:
        writer = csv.writer(fout)

        # metadata header
        writer.writerow([f"# 1D trigger efficiency  –  {var}"])
        writer.writerow([f"# confidence level: {CL:.4f}  (Clopper-Pearson)"])
        writer.writerow([f"# {len(edges)-1} bins"])
        writer.writerow([])
        writer.writerow(HEADER_1D)

        for i in range(n_bins):
            row = [
                _fmt(edges[i]),   _fmt(edges[i + 1]),  _fmt(centers[i]),
                _fmt(rd["n_total"][i]),  _fmt(rd["n_pass"][i]),
                _fmt(rd["efficiency"][i]), _fmt(rd["err_lo"][i]), _fmt(rd["err_hi"][i]),
                _fmt(rm["n_total"][i]),  _fmt(rm["n_pass"][i]),
                _fmt(rm["efficiency"][i]), _fmt(rm["err_lo"][i]), _fmt(rm["err_hi"][i]),
                _fmt(sfr["sf"][i]), _fmt(sfr["err_lo"][i]), _fmt(sfr["err_hi"][i]),
            ]
            writer.writerow(row)

    print(f"  {path}  ({n_bins} bins)")


# ════════════════════════════════════════════════════════════════════════════
# 3.  2D CSV  (tidy / flat format + matrix format per quantity)
# ════════════════════════════════════════════════════════════════════════════

HEADER_2D_TIDY = [
    "bin_lo_x", "bin_hi_x", "bin_center_x",
    "bin_lo_y", "bin_hi_y", "bin_center_y",
    "n_total_data", "n_pass_data",
    "eff_data", "err_lo_data", "err_hi_data",
    "n_total_mc", "n_pass_mc",
    "eff_mc", "err_lo_mc", "err_hi_mc",
    "sf", "sf_err_lo", "sf_err_hi",
]

npz_store = {}  # accumulated for the .npz file

for v0, v1 in PAIRS:
    key    = f"{v0}_vs_{v1}"
    rd     = results["data"]["2d"][key]
    rm     = results["mc"]["2d"][key]
    sfr    = sf_results["2d"][key]

    ex  = np.array(rd["bin_edges_x"])   # shape (nx+1,)
    ey  = np.array(rd["bin_edges_y"])   # shape (ny+1,)
    cx  = np.array(rd["bin_centers_x"]) # shape (nx,)
    cy  = np.array(rd["bin_centers_y"]) # shape (ny,)
    nx, ny = len(cx), len(cy)

    # numpy arrays  – shape (nx, ny)
    eff_d  = np.array(rd["efficiency"], dtype=float)
    lo_d   = np.array(rd["err_lo"],     dtype=float)
    hi_d   = np.array(rd["err_hi"],     dtype=float)
    ntot_d = np.array(rd["n_total"],    dtype=float)
    nps_d  = np.array(rd["n_pass"],     dtype=float)

    eff_m  = np.array(rm["efficiency"], dtype=float)
    lo_m   = np.array(rm["err_lo"],     dtype=float)
    hi_m   = np.array(rm["err_hi"],     dtype=float)
    ntot_m = np.array(rm["n_total"],    dtype=float)
    nps_m  = np.array(rm["n_pass"],     dtype=float)

    sf_arr = np.array(sfr["sf"],     dtype=float)
    sf_lo  = np.array(sfr["err_lo"], dtype=float)
    sf_hi  = np.array(sfr["err_hi"], dtype=float)

    # ── accumulate into npz dict ─────────────────────────────────────────
    prefix = key
    npz_store[f"{prefix}__bin_edges_x"]   = ex
    npz_store[f"{prefix}__bin_edges_y"]   = ey
    npz_store[f"{prefix}__bin_centers_x"] = cx
    npz_store[f"{prefix}__bin_centers_y"] = cy
    npz_store[f"{prefix}__eff_data"]      = eff_d
    npz_store[f"{prefix}__err_lo_data"]   = lo_d
    npz_store[f"{prefix}__err_hi_data"]   = hi_d
    npz_store[f"{prefix}__n_total_data"]  = ntot_d
    npz_store[f"{prefix}__n_pass_data"]   = nps_d
    npz_store[f"{prefix}__eff_mc"]        = eff_m
    npz_store[f"{prefix}__err_lo_mc"]     = lo_m
    npz_store[f"{prefix}__err_hi_mc"]     = hi_m
    npz_store[f"{prefix}__n_total_mc"]    = ntot_m
    npz_store[f"{prefix}__n_pass_mc"]     = nps_m
    npz_store[f"{prefix}__sf"]            = sf_arr
    npz_store[f"{prefix}__sf_err_lo"]     = sf_lo
    npz_store[f"{prefix}__sf_err_hi"]     = sf_hi

    # ── 3a. tidy (long) CSV ──────────────────────────────────────────────
    path_tidy = f"plots/eff_2d_{key}_tidy.csv"
    with open(path_tidy, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow([f"# 2D trigger efficiency  –  {v0} vs {v1}"])
        writer.writerow([f"# confidence level: {CL:.4f}  (Clopper-Pearson)"])
        writer.writerow([f"# shape: {nx} x {ny}  ({v0} x {v1})"])
        writer.writerow([])
        writer.writerow(HEADER_2D_TIDY)

        for i in range(nx):
            for j in range(ny):
                writer.writerow([
                    _fmt(ex[i]), _fmt(ex[i+1]), _fmt(cx[i]),
                    _fmt(ey[j]), _fmt(ey[j+1]), _fmt(cy[j]),
                    _fmt(ntot_d[i, j]), _fmt(nps_d[i, j]),
                    _fmt(eff_d[i, j]), _fmt(lo_d[i, j]), _fmt(hi_d[i, j]),
                    _fmt(ntot_m[i, j]), _fmt(nps_m[i, j]),
                    _fmt(eff_m[i, j]), _fmt(lo_m[i, j]), _fmt(hi_m[i, j]),
                    _fmt(sf_arr[i, j]), _fmt(sf_lo[i, j]), _fmt(sf_hi[i, j]),
                ])

    print(f"  {path_tidy}  ({nx}×{ny} cells)")

    # ── 3b. matrix CSVs (one per quantity) – rows=var0, cols=var1 ────────
    # Column header row uses bin centres; first column is the var0 bin centre.
    def _write_matrix(path, z, lo, hi, row_edges, col_edges,
                      row_centers, col_centers, z_label):
        with open(path, "w", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow([f"# 2D {z_label}  –  rows={v0}  cols={v1}"])
            writer.writerow([f"# row header: [{v0}_lo, {v0}_hi, {v0}_center]"])
            writer.writerow([f"# col header: {v1}_center values"])
            writer.writerow([f"# each cell: value | err_lo | err_hi"])
            writer.writerow([])

            # column header
            col_head = [f"{v0}_lo", f"{v0}_hi", f"{v0}_center"]
            for j in range(len(col_centers)):
                col_head.append(
                    f"[{col_edges[j]:.6g},{col_edges[j+1]:.6g})_val"
                )
                col_head.append(
                    f"[{col_edges[j]:.6g},{col_edges[j+1]:.6g})_err_lo"
                )
                col_head.append(
                    f"[{col_edges[j]:.6g},{col_edges[j+1]:.6g})_err_hi"
                )
            writer.writerow(col_head)

            for i in range(len(row_centers)):
                row = [
                    _fmt(row_edges[i]),
                    _fmt(row_edges[i + 1]),
                    _fmt(row_centers[i]),
                ]
                for j in range(len(col_centers)):
                    row += [_fmt(z[i, j]), _fmt(lo[i, j]), _fmt(hi[i, j])]
                writer.writerow(row)

    for z_arr, lo_arr, hi_arr, suffix, label in [
        (eff_d,  lo_d,  hi_d,  "data", "data efficiency"),
        (eff_m,  lo_m,  hi_m,  "mc",   "MC efficiency"),
        (sf_arr, sf_lo, sf_hi, "sf",   "scale factor"),
    ]:
        path_mat = f"plots/eff_2d_{key}_{suffix}.csv"
        _write_matrix(
            path_mat,
            z_arr, lo_arr, hi_arr,
            ex, ey, cx, cy,
            z_label=label,
        )
        print(f"  {path_mat}")


# ════════════════════════════════════════════════════════════════════════════
# 4.  NPZ  – all 2D arrays in one file for programmatic use
# ════════════════════════════════════════════════════════════════════════════

# Also add 1D arrays to the npz for completeness
for var in BINS:
    rd  = results["data"]["1d"][var]
    rm  = results["mc"]["1d"][var]
    sfr = sf_results["1d"][var]
    p   = f"1d__{var}"
    npz_store[f"{p}__bin_edges"]   = np.array(rd["bin_edges"])
    npz_store[f"{p}__bin_centers"] = np.array(rd["bin_centers"])
    npz_store[f"{p}__n_total_data"]  = np.array(rd["n_total"])
    npz_store[f"{p}__n_pass_data"]   = np.array(rd["n_pass"])
    npz_store[f"{p}__eff_data"]      = np.array(rd["efficiency"], dtype=float)
    npz_store[f"{p}__err_lo_data"]   = np.array(rd["err_lo"],     dtype=float)
    npz_store[f"{p}__err_hi_data"]   = np.array(rd["err_hi"],     dtype=float)
    npz_store[f"{p}__n_total_mc"]    = np.array(rm["n_total"])
    npz_store[f"{p}__n_pass_mc"]     = np.array(rm["n_pass"])
    npz_store[f"{p}__eff_mc"]        = np.array(rm["efficiency"], dtype=float)
    npz_store[f"{p}__err_lo_mc"]     = np.array(rm["err_lo"],     dtype=float)
    npz_store[f"{p}__err_hi_mc"]     = np.array(rm["err_hi"],     dtype=float)
    npz_store[f"{p}__sf"]            = np.array(sfr["sf"],     dtype=float)
    npz_store[f"{p}__sf_err_lo"]     = np.array(sfr["err_lo"], dtype=float)
    npz_store[f"{p}__sf_err_hi"]     = np.array(sfr["err_hi"], dtype=float)

npz_path = "plots/efficiency_arrays.npz"
np.savez(npz_path, **npz_store)
print(f"  {npz_path}  ({len(npz_store)} arrays)")

# ════════════════════════════════════════════════════════════════════════════
# 5.  README
# ════════════════════════════════════════════════════════════════════════════

readme = """\
plots/ numerical outputs — trigger efficiency & scale factors
=============================================================

JSON  (human-readable, NaN → null)
───────────────────────────────────
  efficiency_data.json      full 1D+2D binned efficiency for DATA
  efficiency_mc.json        full 1D+2D binned efficiency for MC
  scale_factors.json        1D+2D scale factors (SF = ε_data / ε_MC)

  Structure:
    {
      "1d": { "<var>": { bin_edges, bin_centers, n_total, n_pass,
                         efficiency, err_lo, err_hi }, ... },
      "2d": { "<v0>_vs_<v1>": { bin_edges_x/y, bin_centers_x/y,
                                n_total, n_pass,
                                efficiency, err_lo, err_hi }, ... }
    }

  Uncertainties are Clopper-Pearson (1σ ≈ 68.27 %), asymmetric.
  err_lo / err_hi are ERRORS (not bounds):
      lower bound = efficiency − err_lo
      upper bound = efficiency + err_hi

CSV – 1D (one row per bin)
──────────────────────────
  eff_1d_{var}.csv          var ∈ {pt, eta, dxy_sig}
  Columns: bin_lo, bin_hi, bin_center,
           n_total_data, n_pass_data, eff_data, err_lo_data, err_hi_data,
           n_total_mc,   n_pass_mc,   eff_mc,   err_lo_mc,   err_hi_mc,
           sf, sf_err_lo, sf_err_hi

CSV – 2D, tidy format (one row per cell)
─────────────────────────────────────────
  eff_2d_{v0}_vs_{v1}_tidy.csv
  Columns: bin_lo_x, bin_hi_x, bin_center_x,
           bin_lo_y, bin_hi_y, bin_center_y,
           n_total_data, n_pass_data, eff_data, err_lo_data, err_hi_data,
           n_total_mc,   n_pass_mc,   eff_mc,   err_lo_mc,   err_hi_mc,
           sf, sf_err_lo, sf_err_hi

CSV – 2D, matrix format (rows = var0 bins, cols = var1 bins)
─────────────────────────────────────────────────────────────
  eff_2d_{v0}_vs_{v1}_data.csv   DATA efficiency matrix
  eff_2d_{v0}_vs_{v1}_mc.csv     MC efficiency matrix
  eff_2d_{v0}_vs_{v1}_sf.csv     Scale factor matrix
  Each cell contains three sub-columns: value | err_lo | err_hi

NPZ  (NumPy binary, for programmatic access)
─────────────────────────────────────────────
  efficiency_arrays.npz
  Key naming convention:
    1D:  "1d__{var}__{quantity}"
    2D:  "{v0}_vs_{v1}__{quantity}"
  Quantities: bin_edges, bin_centers, bin_edges_x/y, bin_centers_x/y,
              n_total_data, n_pass_data, eff_data, err_lo_data, err_hi_data,
              n_total_mc,   n_pass_mc,   eff_mc,   err_lo_mc,   err_hi_mc,
              sf, sf_err_lo, sf_err_hi

  Usage example:
      import numpy as np
      d = np.load("plots/efficiency_arrays.npz")
      eff_pt_data = d["1d__pt__eff_data"]
      sf_2d       = d["pt_vs_dxy_sig__sf"]

"""

with open("plots/README_numerical_outputs.txt", "w") as fout:
    fout.write(readme)
print("  plots/README_numerical_outputs.txt")

print("\n── Numerical outputs complete ──────────────────────────────────────────")