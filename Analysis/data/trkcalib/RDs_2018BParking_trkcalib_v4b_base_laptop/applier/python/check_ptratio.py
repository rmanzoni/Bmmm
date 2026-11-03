"""pT after / pT before the momentum correction, per track type, from a flat ntuple.

    python check_ptratio.py mc.root tree --track k1:hadron --track k2:hadron --track pi:hadron --track mu:muon \
        --cut hadron=1.0 --cut muon=7.0 --out ptratio

For each --track <prefix>:<kind> it reads <prefix>_pt/_eta/_phi/_charge (or _px/_py/_pz/_charge), computes the
factor f = pT'/pT with the applier (nominal and, with --variations all, every variation), and writes:
  <out>_<kind>.pdf/.png   f against pT before (2D), and f in pT bins (1D)
  a table on screen       quantiles of f per pT bin, the largest f seen, and for each --cut the safe prefilter
                          threshold cut / f_max and the tracks that cross the cut (up and down)
The same physical track in several candidates is counted once per candidate: use a one-candidate-per-event
ntuple, or read the table as per-candidate.
"""
import argparse, os, sys
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rds_trkcalib import TrackCalib, track_id_from_kinematics

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(HERE, "..", "data", "rds_trkcalib_2018BParking_v4.json")
EVENT_ID = [("run",), ("lumi", "luminosityBlock"), ("event",)]
PT_EDGES = [0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 50]


def kinematics(arr, t):
    if f"{t}_pt" in arr and f"{t}_eta" in arr:
        return arr[f"{t}_pt"].astype(float), arr[f"{t}_eta"].astype(float), arr[f"{t}_phi"].astype(float)
    px, py, pz = (arr[f"{t}_{c}"].astype(float) for c in ("px", "py", "pz"))
    pt = np.hypot(px, py)
    return pt, np.arcsinh(pz / pt), np.arctan2(py, px)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile"); ap.add_argument("tree")
    ap.add_argument("--track", action="append", required=True, help="prefix:kind, kind = hadron (kaons, pions) or muon")
    ap.add_argument("--cut", action="append", default=[], help="kind=pT cut [GeV] applied after the correction")
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--variations", default="", help="'all' to include every variation in the largest f")
    ap.add_argument("--max-entries", type=int, default=None)
    ap.add_argument("--out", default="ptratio")
    a = ap.parse_args()

    cal = TrackCalib(a.json)
    variations = ["nominal"] + ([v for v in cal.variation_names if v != "nominal"] if a.variations == "all" else [])
    tracks = [s.split(":") for s in a.track]
    cuts = {k: float(v) for k, v in (c.split("=") for c in a.cut)}
    tin = uproot.open(a.infile)[a.tree]; have = set(tin.keys())
    evt_branches = [next((b for b in alts if b in have), None) for alts in EVENT_ID]
    need = [b for b in evt_branches if b]
    for t, _ in tracks:
        cols = ("pt", "eta", "phi") if f"{t}_pt" in have and f"{t}_eta" in have else ("px", "py", "pz")
        need += [f"{t}_{c}" for c in cols + ("charge",)]

    out = {}                                           # kind -> lists of pt, f nominal, f max over variations
    n0 = 0
    for arr in tin.iterate(need, library="np", step_size="200 MB", entry_stop=a.max_entries):
        n = len(arr[need[-1]])
        run, lumi, evt = ((arr[b] if b else np.zeros(n, np.int64)) for b in evt_branches)
        if not evt_branches[2]:
            evt = np.arange(n0, n0 + n)
        n0 += n
        for t, kind in tracks:
            pt, eta, phi = kinematics(arr, t); q = arr[f"{t}_charge"]; p = pt * np.cosh(eta)
            z = cal.gauss(run, lumi, evt, track_id_from_kinematics(pt, eta, phi))
            f = [cal.factor(p, pt, eta, q, z, v, kind=kind) for v in variations]
            d = out.setdefault(kind, {"pt": [], "f": [], "fmax": [], "fmin": []})
            d["pt"].append(pt); d["f"].append(f[0]); d["fmax"].append(np.max(f, 0)); d["fmin"].append(np.min(f, 0))

    for kind, d in out.items():
        pt, f, fmax, fmin = (np.concatenate(d[k]) for k in ("pt", "f", "fmax", "fmin"))
        print(f"\n{kind}: {len(pt):,} tracks; f = pT'/pT ({', '.join(variations) if len(variations) < 4 else 'all variations'})")
        print(f"  {'pT bin [GeV]':>14} {'N':>10} {'median':>8} {'rms':>7} {'0.1%':>7} {'99.9%':>7} {'max':>7}")
        for lo, hi in zip(PT_EDGES[:-1], PT_EDGES[1:]):
            s = (pt >= lo) & (pt < hi)
            if s.sum() < 10:
                continue
            q = np.percentile(f[s], [0.1, 50, 99.9])
            print(f"  {lo:>6.1f}-{hi:<7.1f} {s.sum():>10,} {q[1]:>8.4f} {f[s].std():>7.4f} {q[0]:>7.4f} {q[2]:>7.4f} {fmax[s].max():>7.4f}")
        if kind in cuts:
            c = cuts[kind]; fm = fmax.max()
            up = ((pt < c) & (pt * fmax >= c)).sum(); down = ((pt >= c) & (pt * fmin < c)).sum()
            print(f"  cut pT' > {c} GeV: largest f seen {fm:.4f} -> prefilter pT > {c / fm:.3f} GeV loses nothing in this sample")
            print(f"  tracks crossing the cut: {up:,} up, {down:,} down ({(up + down) / max((pt >= c).sum(), 1):.2%} of those above it)")

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
        h = a1.hist2d(pt, f, bins=[np.geomspace(0.5, 50, 60), np.linspace(0.96, 1.04, 81)], cmin=1, norm=matplotlib.colors.LogNorm())
        fig.colorbar(h[3], ax=a1, label="tracks"); a1.set_xscale("log")
        a1.set_xlabel("pT before [GeV]"); a1.set_ylabel("pT after / pT before"); a1.axhline(1, color="0.4", lw=0.8)
        if kind in cuts:
            a1.axvline(cuts[kind], color="r", lw=1, ls="--", label=f"cut {cuts[kind]} GeV"); a1.legend()
        for lo, hi in [(0.5, 1), (1, 2), (2, 4), (4, 8), (8, 50)]:
            s = (pt >= lo) & (pt < hi)
            if s.sum() > 10:
                a2.hist(f[s], bins=np.linspace(0.96, 1.04, 161), histtype="step", density=True, label=f"pT {lo}-{hi} GeV")
        a2.set_yscale("log"); a2.set_xlabel("pT after / pT before"); a2.set_ylabel("normalised"); a2.legend(fontsize=9)
        fig.suptitle(f"{kind} tracks, {os.path.basename(a.json)}")
        for ext in ("pdf", "png"):
            fig.savefig(f"{a.out}_{kind}.{ext}", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  plots: {a.out}_{kind}.pdf/.png")


if __name__ == "__main__":
    main()
