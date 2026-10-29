#!/usr/bin/env python3
"""
Derive a per-(pt, |eta|) 1D quantile-morphing correction for the muon
beam-spot impact-parameter uncertainty (sigma_dxy), from the grid histograms
written by compare_impact_parameters.py.

For each (pt, |eta|) cell we build the monotone map

    T(x) = F_data^{-1}( F_MC(x) )

that carries an MC sigma_dxy value onto the value with the same cumulative
probability in data. In 1D this is the optimal-transport map: it reproduces the
full data shape, not just mean/variance, and preserves ordering (a larger MC
sigma stays larger after correction), which is what you want for a quantity you
later divide by.

The correction is exported as a correctionlib v2 JSON keyed
(pt, abs_eta, sigma_dxy) -> sigma_dxy_corrected, so any analysis can apply it
from Python, C++ or RDataFrame with a single lookup and no bespoke dependency:

    import correctionlib
    cset = correctionlib.CorrectionSet.from_file("sigma_dxy_morph.json")
    sig_corr = cset["mu2_bs_dxy_e_morph"].evaluate(pt, abs(eta), sig_mc)

Notes
-----
* The map is built from CDFs, so it is invariant to the overall data/MC
  normalisation -- the earlier area-normalisation choice is irrelevant here.
* The map can only correct within the histogram support; values outside are
  clamped. Derive the input grid on a WIDE, FIXED sigma range with overflow so
  the tails are covered (the plotting script's percentile auto-range clips ~1%
  per side -- fine for eyeballing, not for a correction).
* Background (combinatorial) contamination is neglected here by choice; if you
  revisit it, build the templates on sideband-subtracted / sWeighted data.




cset = correctionlib.CorrectionSet.from_file("sigma_dxy_morph.json")
sig_corr = cset["mu2_bs_dxy_e_morph"].evaluate(pt, abs(eta), sig_mc)
# then recompute e.g. dxy/sig_corr

"""

import argparse
import json
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

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
            rec["edges"] = edges  # identical across data/mc for a cell
    if not cells:
        raise RuntimeError(
            f"No histograms for var='{var}' in {infile}. "
            f"Available prefixes: "
            f"{sorted({KEY_RE.match(k)['var'] for k in keys if KEY_RE.match(k)})}"
        )
    return cells


# ----------------------------------------------------------------------------
# The morph itself
# ----------------------------------------------------------------------------
def edge_cdf(counts):
    """Cumulative distribution evaluated at bin EDGES (length nbin+1, 0..1)."""
    c = np.concatenate([[0.0], np.cumsum(counts)])
    if c[-1] <= 0:
        return None
    return c / c[-1]


def _strictly_increasing(y):
    """Nudge a non-decreasing array to be strictly increasing for inversion."""
    eps = 1e-12
    return np.maximum.accumulate(y + eps * np.arange(y.size))


def build_map(edges, counts_mc, counts_data):
    """Return a vectorised callable T(x) = F_data^{-1}(F_MC(x)), clamped to
    the histogram support. Falls back to identity if a template is empty."""
    cdf_mc = edge_cdf(counts_mc)
    cdf_da = edge_cdf(counts_data)
    if cdf_mc is None or cdf_da is None:
        return (lambda x: np.asarray(x, float)), False

    cdf_da_inc = _strictly_increasing(cdf_da)
    lo, hi = edges[0], edges[-1]

    def T(x):
        x = np.clip(np.asarray(x, float), lo, hi)
        u = np.interp(x, edges, cdf_mc)                 # F_MC(x)
        u = np.clip(u, cdf_da_inc[0], cdf_da_inc[-1])
        return np.interp(u, cdf_da_inc, edges)          # F_data^{-1}(u)

    return T, True


# ----------------------------------------------------------------------------
# correctionlib assembly
# ----------------------------------------------------------------------------
def sigma_binning(edges, Tfunc, nsample):
    """Innermost node: piecewise-constant sample of the map over sigma."""
    xs = np.linspace(edges[0], edges[-1], nsample + 1)
    centers = 0.5 * (xs[:-1] + xs[1:])
    content = [float(v) for v in Tfunc(centers)]
    return {"nodetype": "binning", "input": "sigma_dxy",
            "edges": [float(e) for e in xs], "flow": "clamp",
            "content": content}


def build_correction(cells, name, nsample):
    pt_edges = sorted({c[2] for c in cells} | {c[3] for c in cells})
    eta_edges = sorted({c[0] for c in cells} | {c[1] for c in cells})

    inner = {}
    for cell, rec in cells.items():
        T, ok = build_map(rec["edges"], rec.get("mc"), rec.get("data"))
        if not ok:
            print(f"  [warn] empty template in cell {cell}; identity map used")
        inner[cell] = sigma_binning(rec["edges"], T, nsample)

    pt_content = []
    for i in range(len(pt_edges) - 1):
        eta_content = []
        for j in range(len(eta_edges) - 1):
            cell = (eta_edges[j], eta_edges[j + 1], pt_edges[i], pt_edges[i + 1])
            if cell not in inner:  # identity for a missing cell
                any_edges = next(iter(cells.values()))["edges"]
                eta_content.append(sigma_binning(
                    any_edges, lambda x: np.asarray(x, float), nsample))
            else:
                eta_content.append(inner[cell])
        pt_content.append({"nodetype": "binning", "input": "abs_eta",
                           "edges": eta_edges, "flow": "clamp",
                           "content": eta_content})
    data = {"nodetype": "binning", "input": "pt",
            "edges": pt_edges, "flow": "clamp", "content": pt_content}

    return {
        "schema_version": 2,
        "description": "1D quantile-morphing correction for muon sigma_dxy "
                       "(beam-spot IP uncertainty). MC -> data. "
                       "Derived from J/psi tag-and-probe, 2018.",
        "corrections": [{
            "name": name,
            "version": 1,
            "description": "sigma_dxy_MC -> sigma_dxy_corr via F_data^{-1} o F_MC "
                           "per (pt, |eta|) cell.",
            "inputs": [
                {"name": "pt", "type": "real", "description": "muon pT [GeV]"},
                {"name": "abs_eta", "type": "real", "description": "|eta| of muon"},
                {"name": "sigma_dxy", "type": "real",
                 "description": "MC sigma_dxy (bs) [cm]"},
            ],
            "output": {"name": "sigma_dxy_corr", "type": "real",
                       "description": "corrected sigma_dxy [cm]"},
            "data": data,
        }],
    }


# ----------------------------------------------------------------------------
# Closure: push the MC template through the map and compare to data
# ----------------------------------------------------------------------------
def sample_from_hist(edges, counts, ndraw, rng):
    """Draw continuous pseudo-values from a histogram (uniform within each bin).
    This mirrors how the map is applied per-event and avoids comb artifacts."""
    p = counts / counts.sum()
    b = rng.choice(len(counts), size=ndraw, p=p)
    return rng.uniform(edges[b], edges[b + 1])


def closure_grid(cells, edges_label, outfile, ndraw=200000, seed=7):
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
            if rec is None or "data" not in rec or "mc" not in rec \
                    or rec["mc"].sum() == 0 or rec["data"].sum() == 0:
                ax.set_axis_off()
                continue
            edges = rec["edges"]
            centers = 0.5 * (edges[:-1] + edges[1:])
            T, _ = build_map(edges, rec["mc"], rec["data"])

            # Sample MC continuously, morph each value, re-histogram.
            mc_samp = sample_from_hist(edges, rec["mc"], ndraw, rng)
            mcm, _ = np.histogram(T(mc_samp), bins=edges)

            def norm(h):
                s = h.sum()
                return h / s if s > 0 else h

            d, mc, mcm = norm(rec["data"]), norm(rec["mc"]), norm(mcm.astype(float))
            ax.stairs(mc, edges, color="#9ec9e2", fill=True, alpha=0.7,
                      label="MC (raw)")
            ax.stairs(mcm, edges, color="#c1272d", lw=1.6, label="MC (morphed)")
            ax.errorbar(centers, d, yerr=0, color="black", marker="o",
                        ms=2.2, ls="none", label="Data")
            ax.set_title(rf"$|\eta|\in[{cell[0]:g},{cell[1]:g})$, "
                         rf"$p_T\in[{cell[2]:g},{cell[3]:g})$", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == nrow - 1:
                ax.set_xlabel(edges_label, fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle("sigma_dxy morphing closure (shape, unit-normalised)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(outfile, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outfile}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--infile", default="impact_parameter_comparison.root")
    ap.add_argument("--var", default="abs_mu2_dxy_e",
                    help="histogram-name prefix of the variable to morph "
                         "(set to abs_mu2_bs_dxy_e for the beam-spot one)")
    ap.add_argument("--name", default="mu2_bs_dxy_e_morph",
                    help="correctionlib correction name")
    ap.add_argument("--out", default="sigma_dxy_morph.json")
    ap.add_argument("--nsample", type=int, default=400,
                    help="nodes used to store the map per cell")
    ap.add_argument("--closure", default="sigma_dxy_closure.png")
    ap.add_argument("--xlabel", default=r"$|\sigma_{d_{xy}}|\ (\mu_2)$ [cm]")
    args = ap.parse_args()

    cells = read_cells(args.infile, args.var)
    print(f"Read {len(cells)} cells for '{args.var}' from {args.infile}")

    cset = build_correction(cells, args.name, args.nsample)
    with open(args.out, "w") as fh:
        json.dump(cset, fh)
    print(f"Wrote correctionlib JSON -> {args.out}")

    closure_grid(cells, args.xlabel, args.closure)

    # Validate + smoke-test with correctionlib if present.
    try:
        import correctionlib
        cs = correctionlib.CorrectionSet.from_file(args.out)
        c = cs[args.name]
        pt, ae = 6.0, 0.6
        probes = np.linspace(0.0015, 0.0045, 7)
        vals = [c.evaluate(pt, ae, float(x)) for x in probes]
        mono = all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))
        print(f"correctionlib load OK; monotone in sigma: {mono}")
        print("  sigma_mc  -> sigma_corr  (pt=6, |eta|=0.6):")
        for x, y in zip(probes, vals):
            print(f"    {x:.5f} -> {y:.5f}")
    except ImportError:
        print("correctionlib not installed; JSON written but not smoke-tested.")


if __name__ == "__main__":
    main()