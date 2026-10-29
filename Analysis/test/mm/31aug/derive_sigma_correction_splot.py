#!/usr/bin/env python3
"""
Derive a per-(pt, |eta|) 1D quantile-morphing correction for the muon
beam-spot impact-parameter uncertainty (sigma_dxy), from the grid histograms
written by compare_impact_parameters.py.

For each (pt, |eta|) cell we build the monotone map T(x) that carries an MC
sigma_dxy value onto the value with the same cumulative probability in data.
In 1D this is the optimal-transport map: it reproduces the full data shape and
preserves ordering (larger MC sigma stays larger after correction).

SMOOTHING (this version)
------------------------
The naive map T = F_data^{-1} o F_MC built at native histogram resolution is
piecewise-linear with a kink at every bin edge, and on the steep rising edge
of the distribution the local stretch (T' > 1) magnifies those kinks into
step-like spikes -- visible even at high statistics. To regularise:

  * the map is built on K quantile ANCHORS (default 50) of the two templates
    and interpolated with a MONOTONE PCHIP spline -> one smooth C^1 curve
    instead of ~Nbin kinky segments (K = --nquantiles is the smoothness knob);
  * optionally the templates are lightly Gaussian-smoothed first
    (--kde-bw, in bin units) to remove Poisson wiggle on the steep edge;
  * the map is stored densely (--nsample) so the piecewise-constant
    correctionlib representation does not reintroduce a comb.

Two further, opt-in regularisers aimed at SPARSE cells / uncertainties:

  * --min-events N : where a cell has < N data or MC entries, borrow strength
    by merging templates across pt in the same |eta| row, then inclusively;
  * --bootstrap B : Poisson-resample the templates B times, remonotonise, and
    emit '<name>_up' / '<name>_down' corrections at +-1 sigma of the map.

Output is a correctionlib v2 JSON keyed (pt, abs_eta, sigma_dxy) ->
sigma_dxy_corrected.

    import correctionlib
    cset = correctionlib.CorrectionSet.from_file("sigma_dxy_morph.json")
    sig_corr = cset["mu_bs_dxy_e_morph"].evaluate(pt, abs(eta), sig_mc)




python3 derive_sigma_correction_splot.py --var abs_mu_bs_dxy_e \
    --infile pseudo_proper_decay_length_100microns/impact_parameter_comparison.root \
    --name mu_bs_dxy_e_morph --nquantiles 200 --out sigma_dxy_morph_splot.json

"""

import argparse
import json
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

try:
    import mplhep as hep

    plt.style.use(hep.style.CMS)
except Exception:
    pass


# key format written by compare_impact_parameters.py:
#   {var}__eta_{lo}_{hi}__pt_{lo}_{hi}__{data|mc}
# with floats encoded as  0.4 -> '0p4',  -1.5 -> 'm1p5'
KEY_RE = re.compile(
    r"^(?P<var>.+?)__eta_(?P<e0>[^_]+)_(?P<e1>[^_]+)"
    r"__pt_(?P<p0>[^_]+)_(?P<p1>[^_]+)__(?P<kind>data|mc)$"
)


def _decode(tok):
    return float(tok.replace("p", ".").replace("m", "-"))


# ----------------------------------------------------------------------------
# Read the grid cells for one variable
# ----------------------------------------------------------------------------
def read_cells(infile, var):
    """Return {(e0, e1, p0, p1): dict(edges, data, mc)} for the given variable."""
    cells = {}
    with uproot.open(infile) as f:
        keys = {k.split(";")[0] for k in f.keys()}
        for k in keys:
            m = KEY_RE.match(k)
            if not m or m["var"] != var:
                continue
            cell = (_decode(m["e0"]), _decode(m["e1"]),
                    _decode(m["p0"]), _decode(m["p1"]))
            counts, edges = f[k].to_numpy()
            rec = cells.setdefault(cell, {"edges": edges})
            rec[m["kind"]] = counts.astype(float)
            rec["edges"] = edges
    if not cells:
        avail = sorted({KEY_RE.match(k)["var"] for k in keys if KEY_RE.match(k)})
        raise RuntimeError(f"No histograms for var='{var}' in {infile}. "
                           f"Available prefixes: {avail}")
    return cells


# ----------------------------------------------------------------------------
# CDF / quantile helpers
# ----------------------------------------------------------------------------
def edge_cdf(counts):
    """Cumulative distribution evaluated at bin EDGES (length nbin+1, 0..1)."""
    c = np.concatenate([[0.0], np.cumsum(counts)])
    return c / c[-1] if c[-1] > 0 else None


def _strictly_increasing(y):
    return np.maximum.accumulate(y + 1e-12 * np.arange(y.size))


def hist_quantiles(edges, counts, probs):
    """Inverse-CDF (quantile) positions of a histogram at the given probs."""
    cdf = edge_cdf(counts)
    if cdf is None:
        return None
    return np.interp(probs, _strictly_increasing(cdf), edges)


# ----------------------------------------------------------------------------
# Maps
# ----------------------------------------------------------------------------
def build_map_linear(edges, counts_mc, counts_data):
    """Naive T = F_data^{-1} o F_MC at native binning (piecewise linear).
    Kept for the before/after diagnostic."""
    cdf_mc, cdf_da = edge_cdf(counts_mc), edge_cdf(counts_data)
    if cdf_mc is None or cdf_da is None:
        return lambda x: np.asarray(x, float)
    cdf_da = _strictly_increasing(cdf_da)
    lo, hi = edges[0], edges[-1]

    def T(x):
        x = np.clip(np.asarray(x, float), lo, hi)
        u = np.clip(np.interp(x, edges, cdf_mc), cdf_da[0], cdf_da[-1])
        return np.interp(u, cdf_da, edges)

    return T


def build_map_smooth(edges, counts_mc, counts_data, nq=50, kde_bw=0.0):
    """Smooth monotone map from nq quantile anchors + PCHIP."""
    if kde_bw > 0:
        counts_mc = gaussian_filter1d(counts_mc, kde_bw)
        counts_data = gaussian_filter1d(counts_data, kde_bw)
    if counts_mc.sum() <= 0 or counts_data.sum() <= 0:
        return (lambda x: np.asarray(x, float)), False

    probs = np.linspace(0.0, 1.0, nq + 1)
    xq_mc = hist_quantiles(edges, counts_mc, probs)
    xq_da = hist_quantiles(edges, counts_data, probs)

    # PCHIP needs strictly increasing x; drop coincident MC anchors.
    keep = np.concatenate([[True], np.diff(xq_mc) > 1e-12])
    xq_mc, xq_da = xq_mc[keep], xq_da[keep]
    if xq_mc.size < 2:
        return (lambda x: np.asarray(x, float)), False

    pch = PchipInterpolator(xq_mc, xq_da, extrapolate=False)
    x0, x1 = float(xq_mc[0]), float(xq_mc[-1])
    y0, y1 = float(xq_da[0]), float(xq_da[-1])

    def T(x):
        x = np.asarray(x, float)
        y = pch(np.clip(x, x0, x1))            # PCHIP inside the anchored span
        # slope-1 (identity) extrapolation outside -> no boundary pile-up
        y = np.where(x < x0, y0 + (x - x0), y)
        y = np.where(x > x1, y1 + (x - x1), y)
        return y

    return T, True


# ----------------------------------------------------------------------------
# Adaptive merging for sparse cells
# ----------------------------------------------------------------------------
def merged_templates(cells, cell, min_events):
    """Return (counts_mc, counts_data, edges, level), borrowing strength across
    the |eta| row then inclusively if the cell is below min_events."""
    rec = cells.get(cell, {})
    edges = rec.get("edges")
    if edges is None:
        edges = next(iter(cells.values()))["edges"]
    nb = len(edges) - 1
    mc = rec.get("mc", np.zeros(nb))
    da = rec.get("data", np.zeros(nb))
    if min_events <= 0 or (mc.sum() >= min_events and da.sum() >= min_events):
        return mc, da, edges, "cell"

    e0, e1 = cell[0], cell[1]
    row = [c for c in cells if c[0] == e0 and c[1] == e1]
    mc_r = sum((cells[c].get("mc", np.zeros(nb)) for c in row), np.zeros(nb))
    da_r = sum((cells[c].get("data", np.zeros(nb)) for c in row), np.zeros(nb))
    if mc_r.sum() >= min_events and da_r.sum() >= min_events:
        return mc_r, da_r, edges, "eta-row"

    mc_a = sum((cells[c].get("mc", np.zeros(nb)) for c in cells), np.zeros(nb))
    da_a = sum((cells[c].get("data", np.zeros(nb)) for c in cells), np.zeros(nb))
    return mc_a, da_a, edges, "inclusive"


# ----------------------------------------------------------------------------
# correctionlib assembly
# ----------------------------------------------------------------------------
def _sigma_edges(edges, nsample, pad=0.0):
    """Storage edges, optionally padded beyond the template support so that
    out-of-range application inputs (e.g. a harder leg's low tail) land on the
    extrapolated map rather than the flow='clamp' boundary."""
    lo, hi = edges[0], edges[-1]
    w = hi - lo
    return np.linspace(max(0.0, lo - pad * w), hi + pad * w, nsample + 1)


def sigma_binning(edges, Tfunc, nsample, pad=0.0):
    xs = _sigma_edges(edges, nsample, pad)
    centers = 0.5 * (xs[:-1] + xs[1:])
    content = [float(v) for v in np.atleast_1d(Tfunc(centers))]
    return {"nodetype": "binning", "input": "sigma_dxy",
            "edges": [float(e) for e in xs], "flow": "clamp",
            "content": content}


def content_binning(edges, values, nsample, pad=0.0):
    xs = _sigma_edges(edges, nsample, pad)
    return {"nodetype": "binning", "input": "sigma_dxy",
            "edges": [float(e) for e in xs], "flow": "clamp",
            "content": [float(v) for v in values]}


def assemble(cells, inner_by_cell, name, desc):
    pt_edges = sorted({c[2] for c in cells} | {c[3] for c in cells})
    eta_edges = sorted({c[0] for c in cells} | {c[1] for c in cells})
    pt_content = []
    for i in range(len(pt_edges) - 1):
        eta_content = []
        for j in range(len(eta_edges) - 1):
            cell = (eta_edges[j], eta_edges[j + 1], pt_edges[i], pt_edges[i + 1])
            eta_content.append(inner_by_cell[cell])
        pt_content.append({"nodetype": "binning", "input": "abs_eta",
                           "edges": eta_edges, "flow": "clamp",
                           "content": eta_content})
    data = {"nodetype": "binning", "input": "pt",
            "edges": pt_edges, "flow": "clamp", "content": pt_content}
    return {
        "name": name, "version": 1, "description": desc,
        "inputs": [
            {"name": "pt", "type": "real", "description": "muon pT [GeV]"},
            {"name": "abs_eta", "type": "real", "description": "|eta|"},
            {"name": "sigma_dxy", "type": "real",
             "description": "MC sigma_dxy (bs) [cm]"},
        ],
        "output": {"name": "sigma_dxy_corr", "type": "real"},
        "data": data,
    }


def build_correctionset(cells, name, nq, kde_bw, nsample, min_events,
                        nboot, seed, pad=0.5):
    grid_pt = sorted({c[2] for c in cells} | {c[3] for c in cells})
    grid_eta = sorted({c[0] for c in cells} | {c[1] for c in cells})
    all_cells = [(grid_eta[j], grid_eta[j + 1], grid_pt[i], grid_pt[i + 1])
                 for i in range(len(grid_pt) - 1)
                 for j in range(len(grid_eta) - 1)]

    nominal, up, down = {}, {}, {}
    rng = np.random.default_rng(seed)
    edges0 = next(iter(cells.values()))["edges"]
    xs = _sigma_edges(edges0, nsample, pad)
    xs_c = 0.5 * (xs[:-1] + xs[1:])

    for cell in all_cells:
        mc, da, edges, level = merged_templates(cells, cell, min_events)
        if level != "cell":
            print(f"  cell {cell}: sparse -> {level} templates")
        T, ok = build_map_smooth(edges, mc, da, nq=nq, kde_bw=kde_bw)
        nominal[cell] = sigma_binning(edges, T, nsample, pad)

        if nboot > 0 and ok:
            samples = np.empty((nboot, xs_c.size))
            for b in range(nboot):
                Tb, okb = build_map_smooth(edges, rng.poisson(mc).astype(float),
                                           rng.poisson(da).astype(float),
                                           nq=nq, kde_bw=kde_bw)
                samples[b] = Tb(xs_c) if okb else xs_c
            sd = samples.std(axis=0)
            nom = np.asarray(T(xs_c))
            up[cell] = content_binning(edges,
                                       np.maximum.accumulate(nom + sd),
                                       nsample, pad)
            down[cell] = content_binning(edges,
                                         np.maximum.accumulate(nom - sd),
                                         nsample, pad)

    desc = (f"1D quantile morphing of sigma_dxy (MC->data); {nq} quantile "
            f"anchors + PCHIP"
            + (f", KDE bw={kde_bw} bins" if kde_bw else "")
            + (f", min_events={min_events:g}" if min_events else "") + ".")
    corrs = [assemble(cells, nominal, name, desc)]
    if nboot > 0:
        corrs.append(assemble(cells, up, name + "_up",
                              desc + f" +1sigma of {nboot} bootstraps."))
        corrs.append(assemble(cells, down, name + "_down",
                              desc + f" -1sigma of {nboot} bootstraps."))
    return {"schema_version": 2,
            "description": "sigma_dxy morphing correction, J/psi tag-and-probe "
                           "2018.", "corrections": corrs}


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------
def sample_from_hist(edges, counts, ndraw, rng):
    p = counts / counts.sum()
    b = rng.choice(len(counts), size=ndraw, p=p)
    return rng.uniform(edges[b], edges[b + 1])


def most_populated_cell(cells):
    best, n = None, -1
    for cell, rec in cells.items():
        if "mc" in rec and "data" in rec:
            tot = rec["mc"].sum() + rec["data"].sum()
            if tot > n:
                best, n = cell, tot
    return best


def diagnostic_map(cells, cell, nq, kde_bw, outfile, xlabel):
    rec = cells[cell]
    edges = rec["edges"]
    xs = np.linspace(edges[0], edges[-1], 400)
    T_lin = build_map_linear(edges, rec["mc"], rec["data"])
    T_smo, _ = build_map_smooth(edges, rec["mc"], rec["data"], nq, kde_bw)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    a1.plot(xs, T_lin(xs), color="#999999", lw=1.2, label="naive (per-bin)")
    a1.plot(xs, T_smo(xs), color="#c1272d", lw=2.0,
            label=f"smooth ({nq} quantiles + PCHIP)")
    a1.plot(xs, xs, color="black", lw=0.7, ls=":", label="identity")
    a1.set_xlabel(r"$\sigma_{d_{xy}}$ MC")
    a1.set_ylabel(r"$\sigma_{d_{xy}}$ corr")
    a1.set_title(rf"map: $|\eta|\in[{cell[0]:g},{cell[1]:g})$, "
                 rf"$p_T\in[{cell[2]:g},{cell[3]:g})$", fontsize=9)
    a1.legend(fontsize=8)

    rng = np.random.default_rng(3)
    mc_s = sample_from_hist(edges, rec["mc"], 300000, rng)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    lin, _ = np.histogram(T_lin(mc_s), bins=edges)
    smo, _ = np.histogram(T_smo(mc_s), bins=edges)
    nrm = lambda h: h / h.sum()
    a2.stairs(nrm(rec["mc"]), edges, color="#9ec9e2", fill=True, alpha=0.6,
              label="MC raw")
    a2.stairs(nrm(lin.astype(float)), edges, color="#999999", lw=1.2,
              label="morphed (naive)")
    a2.stairs(nrm(smo.astype(float)), edges, color="#c1272d", lw=1.8,
              label="morphed (smooth)")
    a2.errorbar(ctr, nrm(rec["data"]), yerr=0, color="black", marker="o",
                ms=2.5, ls="none", label="Data")
    a2.set_xlabel(xlabel)
    a2.set_title("closure in the same cell", fontsize=9)
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outfile}")


def closure_grid(cells, mapper, xlabel, outfile, ndraw=200000, seed=7):
    rng = np.random.default_rng(seed)
    eta_edges = sorted({c[0] for c in cells} | {c[1] for c in cells})
    pt_edges = sorted({c[2] for c in cells} | {c[3] for c in cells})
    nrow, ncol = len(eta_edges) - 1, len(pt_edges) - 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow),
                             squeeze=False)
    for i in range(nrow):
        for j in range(ncol):
            ax = axes[i][j]
            cell = (eta_edges[i], eta_edges[i + 1], pt_edges[j], pt_edges[j + 1])
            rec = cells.get(cell)
            if (rec is None or "data" not in rec or "mc" not in rec
                    or rec["mc"].sum() == 0 or rec["data"].sum() == 0):
                ax.set_axis_off()
                continue
            edges = rec["edges"]
            ctr = 0.5 * (edges[:-1] + edges[1:])
            T, _ = mapper(edges, rec["mc"], rec["data"])
            mc_s = sample_from_hist(edges, rec["mc"], ndraw, rng)
            mcm, _ = np.histogram(T(mc_s), bins=edges)
            nrm = lambda h: h / h.sum() if h.sum() > 0 else h
            ax.stairs(nrm(rec["mc"]), edges, color="#9ec9e2", fill=True,
                      alpha=0.7, label="MC (raw)")
            ax.stairs(nrm(mcm.astype(float)), edges, color="#c1272d", lw=1.6,
                      label="MC (morphed)")
            ax.errorbar(ctr, nrm(rec["data"]), yerr=0, color="black",
                        marker="o", ms=2.2, ls="none", label="Data")
            ax.set_title(rf"$|\eta|\in[{cell[0]:g},{cell[1]:g})$, "
                         rf"$p_T\in[{cell[2]:g},{cell[3]:g})$", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == nrow - 1:
                ax.set_xlabel(xlabel, fontsize=8)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", fontsize=9)
    fig.suptitle("sigma_dxy morphing closure (smoothed)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(outfile, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outfile}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--infile", default="impact_parameter_comparison.root")
    ap.add_argument("--var", default="abs_mu_dxy_e",
                    help="set to abs_mu_bs_dxy_e for the beam-spot variable")
    ap.add_argument("--name", default="mu_bs_dxy_e_morph")
    ap.add_argument("--out", default="sigma_dxy_morph.json")
    ap.add_argument("--nquantiles", type=int, default=20,
                    help="quantile anchors for the smooth map (smoothness knob; "
                         "keep well below the template bin count)")
    ap.add_argument("--kde-bw", type=float, default=0.0,
                    help="Gaussian template pre-smoothing, in bin units (0=off)")
    ap.add_argument("--nsample", type=int, default=600,
                    help="dense nodes for the correctionlib storage")
    ap.add_argument("--min-events", type=float, default=0.0,
                    help="merge sparse cells below this many entries (0=off)")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="bootstrap replicas -> up/down corrections (0=off)")
    ap.add_argument("--store-pad", type=float, default=0.5,
                    help="pad the stored sigma range by this fraction of its "
                         "width on each side, so out-of-range inputs "
                         "extrapolate instead of clamping (0 = no pad)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--closure", default="sigma_dxy_closure.pdf")
    ap.add_argument("--diagnostic", default="sigma_dxy_map_diagnostic.pdf")
    ap.add_argument("--xlabel", default=r"$|\sigma_{d_{xy}}|\ (\mu_2)$ [cm]")
    args = ap.parse_args()

    cells = read_cells(args.infile, args.var)
    print(f"Read {len(cells)} cells for '{args.var}' from {args.infile}")

    cset = build_correctionset(cells, args.name, args.nquantiles, args.kde_bw,
                               args.nsample, args.min_events, args.bootstrap,
                               args.seed, args.store_pad)
    with open(args.out, "w") as fh:
        json.dump(cset, fh)
    print(f"Wrote {len(cset['corrections'])} correction(s) -> {args.out}")

    mapper = lambda e, mc, da: build_map_smooth(e, mc, da, args.nquantiles,
                                                args.kde_bw)
    closure_grid(cells, mapper, args.xlabel, args.closure)
    diagnostic_map(cells, most_populated_cell(cells), args.nquantiles,
                   args.kde_bw, args.diagnostic, args.xlabel)

    try:
        import correctionlib
        c = correctionlib.CorrectionSet.from_file(args.out)[args.name]
        probes = np.linspace(0.0015, 0.0045, 7)
        vals = [c.evaluate(6.0, 0.6, float(x)) for x in probes]
        mono = all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))
        print(f"correctionlib load OK; monotone in sigma: {mono}")
    except ImportError:
        print("correctionlib not installed; JSON written but not smoke-tested.")


if __name__ == "__main__":
    main()