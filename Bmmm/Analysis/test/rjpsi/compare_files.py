#!/usr/bin/env python3
"""
compare_productions.py

Compare two productions (A and B) of the same flat ntuple, where each row is
one candidate and an event can hold several candidates (triplets). The goal is
to find out where the two productions differ.

Three levels of comparison
--------------------------
1. Events.
   Which (run, lumi, event) appear only in A, only in B, or in both. The
   counts are broken down per lumisection and per run. This separates two
   patterns: whole lumisections missing, which points to input or bookkeeping
   (files, JSON, dataset version), and scattered events missing inside shared
   lumisections, which points to selection or reconstruction.
2. Candidates.
   For events present in both productions, the script checks whether the
   number of candidates is the same. Candidates are paired inside each event
   by rank: the stored order by default, or the order given by --cand-keys.
3. Values.
   For every candidate pair, each shared branch is checked for equality within
   rtol/atol, and separately for bit-identity.

Duplicate checks on each file alone
-----------------------------------
  - Duplicate candidates: two rows of the same event that are identical in
    every compared branch. A 64-bit fingerprint of each row is accumulated
    while the branches are read anyway, so the extra cost is small.
  - Duplicate events, which can show up in two ways:
      (a) the event's rows are stored in several separate blocks of the
          file. This is the signature of a file merged twice.
      (b) every candidate of the event has an exact copy inside the event.
          This is what a duplication stored in one contiguous block looks
          like.

Optional distribution check (branch_discrimination.csv)
--------------------------------------------------------
Within each file, candidates that have no partner in the other file are
compared to the candidates that do have one. This is an association only:
it shows which variables differ between the two groups, not which cut is
responsible.

Performance
-----------
- Event matching is done in numpy, with no per-event Python loop. The key
  (run, lumi, event) is packed into a single uint64.
- Branches are read and compared one at a time, and the value comparison is
  done in chunks. Peak memory is roughly: index arrays (~30 bytes/row/file)
  + one branch in both files + one chunk of temporaries.
- A progress bar is shown with tqdm when it is installed; otherwise a plain
  text counter is printed.

Requires uproot >= 5, awkward >= 2, numpy, pandas. Written to be compatible
with Python 3.9, so it also runs inside CMSSW.
"""

import argparse
import os
import re
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import awkward as ak
import numpy as np
import pandas as pd
import uproot
from uproot.interpretation.jagged import AsJagged
from uproot.interpretation.numerical import Numerical

try:
    from tqdm import tqdm
except ImportError:  # plain-text fallback below
    tqdm = None


KEYS = ("run", "lumi", "event")
T0 = time.time()


# ---------------------------------------------------------------------------
# Logging / progress
# ---------------------------------------------------------------------------
def log(msg):
    print("[{:8.1f}s] {}".format(time.time() - T0, msg), flush=True)


def progress(iterable, total, desc):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, unit="branch",
                    dynamic_ncols=True, file=sys.stdout)
    return _plain_progress(iterable, total, desc)


def _plain_progress(iterable, total, desc):
    start = time.time()
    last = 0.0
    for i, item in enumerate(iterable, 1):
        yield item
        now = time.time()
        if now - last > 5.0 or i == total:
            elapsed = now - start
            eta = elapsed / i * (total - i)
            print("  {}: {}/{}  elapsed {:.0f}s  ETA {:.0f}s".format(
                desc, i, total, elapsed, eta), flush=True)
            last = now


def peak_mem_gb():
    # ru_maxrss is in kB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2


# ---------------------------------------------------------------------------
# Branch classification and reading
# ---------------------------------------------------------------------------
def classify(branch):
    """
    Return one of:
      ("flat",   dtype, inner_shape)  numeric scalar or fixed-size array
      ("jagged", dtype, ())           variable-length vector of numbers
      ("skip",   None,  reason)       anything else (strings, objects, ...)
    """
    interp = branch.interpretation
    if isinstance(interp, Numerical):
        dt = np.dtype(interp.to_dtype)
        if dt.base.kind in "biuf":
            return ("flat", dt.base, dt.shape)
        return ("skip", None, "non-numeric dtype {}".format(dt))
    if isinstance(interp, AsJagged) and isinstance(interp.content, Numerical):
        dt = np.dtype(interp.content.to_dtype)
        if dt.base.kind in "biuf" and dt.shape == ():
            return ("jagged", dt.base, ())
        return ("skip", None, "jagged with inner shape {}".format(dt.shape))
    return ("skip", None, "unsupported interpretation {}".format(type(interp).__name__))


def read_branch(branch, kind, executor):
    library = "np" if kind == "flat" else "ak"
    return branch.array(library=library,
                        decompression_executor=executor,
                        interpretation_executor=executor)


def read_key_columns(tree, names, label, executor):
    available = set(tree.keys())
    missing = [n for n in names if n not in available]
    if missing:
        raise KeyError("{}: missing branches {}".format(label, missing))
    out = {}
    for n in names:
        kind, _, shape = classify(tree[n])
        if kind != "flat" or shape != ():
            raise TypeError("{}: branch {!r} must be a flat numeric scalar "
                            "to be used as a key".format(label, n))
        out[n] = read_branch(tree[n], "flat", executor)
    return out


# ---------------------------------------------------------------------------
# Event key packing: (run, lumi, event) -> uint64, order-preserving
# ---------------------------------------------------------------------------
class KeyPacker:
    def __init__(self, cols_A, cols_B):
        for label, cols in (("A", cols_A), ("B", cols_B)):
            for k in KEYS:
                arr = cols[k]
                if arr.dtype.kind not in "iu":
                    raise TypeError("{}: key branch {!r} has non-integer dtype {}"
                                    .format(label, k, arr.dtype))
                if arr.dtype.kind == "i" and arr.min() < 0:
                    raise ValueError("{}: key branch {!r} has negative values"
                                     .format(label, k))

        def mn(k):
            return int(min(cols_A[k].min(), cols_B[k].min()))

        def mx(k):
            return int(max(cols_A[k].max(), cols_B[k].max()))

        self.run_min = mn("run")
        self.b_run = max(1, (mx("run") - self.run_min).bit_length())
        self.b_lumi = max(1, mx("lumi").bit_length())
        self.b_evt = max(1, mx("event").bit_length())
        total = self.b_run + self.b_lumi + self.b_evt
        if total > 64:
            raise ValueError(
                "Cannot pack (run, lumi, event) into 64 bits: needs {} bits "
                "(run span {}, lumi {}, event {}). Split the comparison by run "
                "range.".format(total, self.b_run, self.b_lumi, self.b_evt))
        self.m_lumi = np.uint64((1 << self.b_lumi) - 1)
        self.m_evt = np.uint64((1 << self.b_evt) - 1)

    def pack(self, cols):
        r = cols["run"].astype(np.uint64) - np.uint64(self.run_min)
        key = r << np.uint64(self.b_lumi + self.b_evt)
        key |= cols["lumi"].astype(np.uint64) << np.uint64(self.b_evt)
        key |= cols["event"].astype(np.uint64)
        return key

    def unpack(self, key):
        evt = key & self.m_evt
        lumi = (key >> np.uint64(self.b_evt)) & self.m_lumi
        run = (key >> np.uint64(self.b_evt + self.b_lumi)) + np.uint64(self.run_min)
        return run, lumi, evt

    def lumi_key(self, key):
        return key >> np.uint64(self.b_evt)

    def unpack_lumi_key(self, lk):
        lumi = lk & self.m_lumi
        run = (lk >> np.uint64(self.b_lumi)) + np.uint64(self.run_min)
        return run, lumi


# ---------------------------------------------------------------------------
# Per-file event index
# ---------------------------------------------------------------------------
class FileIndex:
    """
    Sort rows by event key (then by candidate keys, or keep the stored order).

    Attributes
      uniq    sorted unique event keys
      counts  number of candidates per event
      starts  position of each event's first row in the sorted order
      order   sorted position -> original row
    """

    def __init__(self, label, evkey, cand_cols):
        n = len(evkey)
        self.label = label
        self.n_rows = n
        idx_dtype = np.int32 if n < 2 ** 31 else np.int64

        if cand_cols:
            # np.lexsort: the LAST key is the primary one; lexsort is stable
            order = np.lexsort(tuple(reversed(cand_cols)) + (evkey,))
        else:
            order = np.argsort(evkey, kind="stable")

        sk = evkey[order]
        new = np.empty(n, dtype=bool)
        new[0] = True
        np.not_equal(sk[1:], sk[:-1], out=new[1:])
        starts = np.flatnonzero(new)

        self.uniq = sk[starts]
        self.counts = np.diff(np.append(starts, n)).astype(np.int32)
        self.starts = starts.astype(idx_dtype)
        self.n_events = len(self.uniq)

        del sk, new

        self.order = order.astype(idx_dtype)
        del order

        # Events whose candidates are split into several blocks in the
        # stored order (a file merged twice looks exactly like this)
        brk = np.flatnonzero(evkey[1:] != evkey[:-1]) + 1
        n_blocks = len(brk) + 1
        self.split_keys = np.zeros(0, dtype=np.uint64)   # events stored in >1 block
        self.split_nblocks = np.zeros(0, dtype=np.int64)
        if n_blocks > self.n_events:
            block_keys = evkey[np.concatenate(([0], brk))]
            uk, bc = np.unique(block_keys, return_counts=True)
            self.split_keys = uk[bc > 1]
            self.split_nblocks = bc[bc > 1]
        self.n_split_events = len(self.split_keys)


# ---------------------------------------------------------------------------
# Row fingerprint and duplicate analysis
# ---------------------------------------------------------------------------
_P = np.uint64(0x100000001B3)          # odd multiplier (FNV-1a 64-bit prime)
_Q = np.uint64(0x9E3779B97F4A7C15)     # odd multiplier for positions in a vector


def _bits(a):
    """Raw bytes of each value as uint64 (a deterministic function of the stored value)."""
    a = np.ascontiguousarray(a)
    utype = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize]
    return a.view(utype).astype(np.uint64)


class RowFingerprint:
    """
    Per-row hash h <- h * P + value_bits, accumulated branch by branch
    (mod 2^64). Because P is odd, two rows that differ in a single branch
    always get different fingerprints; a collision needs differences in
    several branches that cancel exactly, which has negligible probability.
    """

    def __init__(self, n_rows):
        self.h = np.zeros(n_rows, dtype=np.uint64)
        self.n_branches = 0

    def _add(self, v):
        with np.errstate(over="ignore"):
            np.multiply(self.h, _P, out=self.h)
            np.add(self.h, v, out=self.h)

    def add_flat(self, a):
        if a.ndim == 1:
            self._add(_bits(a))
        else:
            flat = a.reshape(len(a), -1)
            for j in range(flat.shape[1]):
                self._add(_bits(flat[:, j]))
        self.n_branches += 1

    def add_jagged(self, arr):
        counts = ak.to_numpy(ak.num(arr, axis=1)).astype(np.int64)
        flat = ak.to_numpy(ak.flatten(arr))
        offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        local = np.arange(len(flat), dtype=np.int64) - np.repeat(offsets[:-1], counts)
        maxlen = int(counts.max()) if len(counts) else 0
        with np.errstate(over="ignore"):
            powers = np.ones(max(maxlen, 1), dtype=np.uint64)
            for k in range(1, maxlen):
                powers[k] = powers[k - 1] * _Q
            elem = (_bits(flat) + np.uint64(1)) * powers[local]
            prefix = np.zeros(len(flat) + 1, dtype=np.uint64)
            np.cumsum(elem, out=prefix[1:])
            row = prefix[offsets[1:]] - prefix[offsets[:-1]]
        self._add(counts.astype(np.uint64))
        self._add(row)
        self.n_branches += 1


def duplicate_analysis(ix, rowhash):
    """
    Within each event, group candidates with identical fingerprints.
    Returns per-file totals and a per-event table (only events with issues).
    """
    multi = ix.counts > 1
    ev_idx = np.flatnonzero(multi)
    res = {"extra_candidate_copies": 0, "events_with_duplicate_candidates": 0,
           "events_fully_duplicated": 0}
    empty = pd.DataFrame(columns=["ev", "n_cand", "extra_copies", "fully_duplicated"])
    if len(ev_idx) == 0:
        return res, empty

    rows, _, local, _ = expand_rows(ix, ev_idx, ix.counts[ev_idx])
    h = rowhash[rows]
    del rows
    o = np.lexsort((h, local))            # by event, then fingerprint
    e2, h2 = local[o], h[o]
    del o, h, local
    same_prev = np.zeros(len(e2), dtype=bool)
    same_prev[1:] = (e2[1:] == e2[:-1]) & (h2[1:] == h2[:-1])
    del h2

    extra = np.bincount(e2[same_prev], minlength=len(ev_idx))
    res["extra_candidate_copies"] = int(extra.sum())
    res["events_with_duplicate_candidates"] = int((extra > 0).sum())

    # groups of identical candidates; an event is fully duplicated if all its
    # groups have the same size >= 2 (every candidate appears k >= 2 times)
    gstart = np.flatnonzero(~same_prev)
    gsize = np.diff(np.append(gstart, len(e2)))
    gev = e2[gstart]
    del e2, same_prev
    evstart = np.flatnonzero(np.r_[True, gev[1:] != gev[:-1]])
    gmin = np.minimum.reduceat(gsize, evstart)
    gmax = np.maximum.reduceat(gsize, evstart)
    full = np.zeros(len(ev_idx), dtype=bool)
    full[gev[evstart]] = (gmin >= 2) & (gmin == gmax)
    res["events_fully_duplicated"] = int(full.sum())

    sel = extra > 0
    table = pd.DataFrame({"ev": ev_idx[sel], "n_cand": ix.counts[ev_idx[sel]],
                          "extra_copies": extra[sel], "fully_duplicated": full[sel]})
    return res, table


def duplicate_events_frame(packer, ix, table):
    """Merge fingerprint duplicates and split-block events into one list."""
    d1 = pd.DataFrame({"key": ix.uniq[table["ev"].to_numpy(dtype=np.int64)],
                       "n_cand": table["n_cand"].to_numpy(),
                       "extra_candidate_copies": table["extra_copies"].to_numpy(),
                       "all_candidates_duplicated": table["fully_duplicated"].to_numpy()})
    d2 = pd.DataFrame({"key": ix.split_keys,
                       "stored_in_n_blocks": ix.split_nblocks})
    df = d1.merge(d2, on="key", how="outer")
    if len(df) == 0:
        return pd.DataFrame(columns=["run", "lumi", "event", "n_cand",
                                     "extra_candidate_copies",
                                     "all_candidates_duplicated", "stored_in_n_blocks"])
    keys = df["key"].to_numpy(dtype=np.uint64)
    pos = np.searchsorted(ix.uniq, keys)
    df["n_cand"] = ix.counts[pos]
    df["extra_candidate_copies"] = df["extra_candidate_copies"].fillna(0).astype(np.int64)
    df["all_candidates_duplicated"] = df["all_candidates_duplicated"].fillna(False).astype(bool)
    df["stored_in_n_blocks"] = df["stored_in_n_blocks"].fillna(1).astype(np.int64)
    run, lumi, evt = packer.unpack(keys)
    df.insert(0, "run", run.astype(np.int64))
    df.insert(1, "lumi", lumi.astype(np.int64))
    df.insert(2, "event", evt)
    return df.drop(columns="key").sort_values(["run", "lumi", "event"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Matching events and pairing candidates
# ---------------------------------------------------------------------------
def expand_rows(ix, ci_sel, m):
    """
    For the events ix.uniq[ci_sel], take the first m[i] candidates of each
    (in ix's sorted order). Returns original row numbers, the start of each
    event in the output, the output->event index, and the rank in the event.
    """
    m = m.astype(np.int64)
    total = int(m.sum())
    first = np.cumsum(m) - m
    local = np.repeat(np.arange(len(m), dtype=np.int64), m)
    rank = np.arange(total, dtype=np.int64) - first[local]
    rows = ix.order[ix.starts[ci_sel][local] + rank]
    return rows, first, local, rank


class Pairing:
    """
    Candidate pairs between A and B, for events present in both files.

    - Same number of candidates in A and B: pair the k-th with the k-th
      (stored order, or order by --cand-keys).
    - Different number of candidates: pair only candidates whose --cand-keys
      values are exactly equal; without --cand-keys, leave them unpaired
      (pairing by rank would shift pairs and create fake value differences).
    """

    def __init__(self, ixA, ixB, candA, candB):
        uA, uB = ixA.uniq, ixB.uniq
        pos = np.searchsorted(uB, uA)
        pos_c = np.minimum(pos, len(uB) - 1)
        in_B = uB[pos_c] == uA
        del pos

        self.ciA = np.flatnonzero(in_B)          # common events, index in A
        self.ciB = pos_c[self.ciA]               # same events, index in B
        del pos_c
        self.only_A_mask = ~in_B                 # over A's unique events
        in_A = np.zeros(len(uB), dtype=bool)
        in_A[self.ciB] = True
        self.only_B_mask = ~in_A                 # over B's unique events

        self.common = uA[self.ciA]
        self.cA = ixA.counts[self.ciA]
        self.cB = ixB.counts[self.ciB]
        eq = self.cA == self.cB

        # (1) equal multiplicity: pair by rank
        self.eq_ev = np.flatnonzero(eq)
        m = self.cA[eq]
        rowsA1, first1, local1, rank1 = expand_rows(ixA, self.ciA[eq], m)
        rowsB1, _, _, _ = expand_rows(ixB, self.ciB[eq], m)
        pair_ev1 = self.eq_ev[local1]
        self.eq_first = first1                   # start of each eq event in pair list
        self.n_rank_pairs = len(rowsA1)
        del local1

        # (2) changed multiplicity: pair by exact cand-key match
        self.ch_ev = np.flatnonzero(~eq)
        rowsA2 = rowsB2 = pair_ev2 = np.zeros(0, dtype=np.int64)
        self.key_matching = bool(candA)
        if candA and len(self.ch_ev):
            ra, _, la, _ = expand_rows(ixA, self.ciA[~eq], self.cA[~eq])
            rb, _, lb, _ = expand_rows(ixB, self.ciB[~eq], self.cB[~eq])
            kcols = ["k{}".format(i) for i in range(len(candA))]
            dA = pd.DataFrame(dict({"ev": la, "rowA": ra},
                                   **{k: c[ra] for k, c in zip(kcols, candA)}))
            dB = pd.DataFrame(dict({"ev": lb, "rowB": rb},
                                   **{k: c[rb] for k, c in zip(kcols, candB)}))
            # identical candidates inside an event: pair the n-th copy with the n-th
            dA["dup"] = dA.groupby(["ev"] + kcols, dropna=False).cumcount()
            dB["dup"] = dB.groupby(["ev"] + kcols, dropna=False).cumcount()
            mg = dA.merge(dB, on=["ev"] + kcols + ["dup"], how="inner")
            rowsA2 = mg["rowA"].to_numpy(dtype=np.int64)
            rowsB2 = mg["rowB"].to_numpy(dtype=np.int64)
            pair_ev2 = self.ch_ev[mg["ev"].to_numpy(dtype=np.int64)]
        self.n_key_pairs = len(rowsA2)

        self.rowsA = np.concatenate([rowsA1, rowsA2]).astype(ixA.order.dtype)
        self.rowsB = np.concatenate([rowsB1, rowsB2]).astype(ixB.order.dtype)
        self.pair_ev = np.concatenate([pair_ev1, pair_ev2])
        # rank in event for rank-paired candidates; -1 = paired by cand keys
        self.rank = np.concatenate([rank1, np.full(len(rowsA2), -1)]).astype(np.int32)
        self.n_pairs = len(self.rowsA)
        del rowsA1, rowsB1, rowsA2, rowsB2, pair_ev1, pair_ev2, rank1

        # Pair belongs to an event with >1 candidate in at least one file
        multi_ev = (self.cA > 1) | (self.cB > 1)
        self.pair_multi = multi_ev[self.pair_ev]

        self.matched_A = np.zeros(ixA.n_rows, dtype=bool)
        self.matched_A[self.rowsA] = True
        self.matched_B = np.zeros(ixB.n_rows, dtype=bool)
        self.matched_B[self.rowsB] = True

    def per_event_max(self, per_pair):
        """Max of a per-pair quantity over the candidates of each common event."""
        out = np.zeros(len(self.common), dtype=per_pair.dtype)
        if self.n_rank_pairs:
            out[self.eq_ev] = np.maximum.reduceat(per_pair[:self.n_rank_pairs], self.eq_first)
        if self.n_key_pairs:
            np.maximum.at(out, self.pair_ev[self.n_rank_pairs:], per_pair[self.n_rank_pairs:])
        return out


# ---------------------------------------------------------------------------
# Value comparison
# ---------------------------------------------------------------------------
def _new_stats():
    return {"n_nan_mismatch": 0, "max_abs_diff": 0.0, "max_rel_diff": 0.0,
            "sum_signed_diff": 0.0, "n_finite_diff": 0}


def _update_stats(st, a, b, elem_diff):
    """Stats on the differing elements only (a, b arrays of equal shape)."""
    if not elem_diff.any():
        return
    ad = a[elem_diff].astype(np.float64)
    bd = b[elem_diff].astype(np.float64)
    nan_a, nan_b = np.isnan(ad), np.isnan(bd)
    st["n_nan_mismatch"] += int((nan_a != nan_b).sum())
    fin = np.isfinite(ad) & np.isfinite(bd)
    if fin.any():
        d = bd[fin] - ad[fin]
        absd = np.abs(d)
        denom = np.maximum(np.abs(ad[fin]), np.abs(bd[fin]))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(denom > 0, absd / denom, 0.0)
        st["max_abs_diff"] = max(st["max_abs_diff"], float(absd.max()))
        st["max_rel_diff"] = max(st["max_rel_diff"], float(rel.max()))
        st["sum_signed_diff"] += float(d.sum())
        st["n_finite_diff"] += int(fin.sum())


def _elem_masks(a, b, is_float, rtol, atol):
    """Elementwise (not_close, not_identical)."""
    if is_float:
        a = a.astype(np.float64, copy=False)
        b = b.astype(np.float64, copy=False)
        both_nan = np.isnan(a) & np.isnan(b)
        not_ident = ~((a == b) | both_nan)
        not_close = ~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    else:
        not_ident = a != b
        not_close = not_ident
    return not_close, not_ident


def compare_flat(a_all, b_all, pairing, rtol, atol, chunk):
    n = pairing.n_pairs
    is_float = a_all.dtype.kind == "f" or b_all.dtype.kind == "f"
    diff = np.zeros(n, dtype=bool)
    notid = np.zeros(n, dtype=bool)
    st = _new_stats()
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        a = a_all[pairing.rowsA[s:e]]
        b = b_all[pairing.rowsB[s:e]]
        nc, ni = _elem_masks(a, b, is_float, rtol, atol)
        _update_stats(st, a, b, nc)
        if nc.ndim > 1:  # fixed-size array branch: a row differs if any element does
            axes = tuple(range(1, nc.ndim))
            nc = nc.any(axis=axes)
            ni = ni.any(axis=axes)
        diff[s:e] = nc
        notid[s:e] = ni
    return diff, notid, st


def compare_jagged(a_all, b_all, pairing, rtol, atol, chunk):
    n = pairing.n_pairs
    diff = np.zeros(n, dtype=bool)
    notid = np.zeros(n, dtype=bool)
    st = _new_stats()
    st["n_length_mismatch"] = 0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        a = a_all[pairing.rowsA[s:e]]
        b = b_all[pairing.rowsB[s:e]]
        na = ak.to_numpy(ak.num(a, axis=1))
        nb = ak.to_numpy(ak.num(b, axis=1))
        same_len = na == nb
        st["n_length_mismatch"] += int((~same_len).sum())
        row_nc = ~same_len
        row_ni = ~same_len
        if same_len.any():
            fa = ak.to_numpy(ak.flatten(a[same_len]))
            fb = ak.to_numpy(ak.flatten(b[same_len]))
            is_float = fa.dtype.kind == "f" or fb.dtype.kind == "f"
            nc, ni = _elem_masks(fa, fb, is_float, rtol, atol)
            _update_stats(st, fa, fb, nc)
            counts = na[same_len]
            row_nc_same = ak.to_numpy(ak.any(ak.unflatten(nc, counts), axis=1))
            row_ni_same = ak.to_numpy(ak.any(ak.unflatten(ni, counts), axis=1))
            row_nc[same_len] |= row_nc_same
            row_ni[same_len] |= row_ni_same
        diff[s:e] = row_nc
        notid[s:e] = row_ni
    return diff, notid, st


def value_repr(arr, row):
    v = arr[row]
    if isinstance(v, np.ndarray):
        return repr(v.tolist())
    if isinstance(v, (np.generic,)):
        return repr(v.item())
    return repr(ak.to_list(v))


# ---------------------------------------------------------------------------
# Distribution comparison: unpaired vs paired candidates within one file
# ---------------------------------------------------------------------------
def ks_statistic(x, y):
    """Two-sample KS statistic (max distance between the two CDFs)."""
    x = np.sort(x[np.isfinite(x)])
    y = np.sort(y[np.isfinite(y)])
    if len(x) == 0 or len(y) == 0:
        return np.nan
    allv = np.concatenate([x, y])
    cx = np.searchsorted(x, allv, side="right") / len(x)
    cy = np.searchsorted(y, allv, side="right") / len(y)
    return float(np.max(np.abs(cx - cy)))


def distribution_stats(values, matched_mask, sub_unpaired, sub_paired):
    v = values.astype(np.float64, copy=False)
    x = v[~matched_mask]
    y = v[matched_mask]
    fx, fy = np.isfinite(x), np.isfinite(y)
    xs, ys = x[fx], y[fy]
    res = {
        "unpaired_n": len(x),
        "paired_n": len(y),
        "unpaired_nonfinite_fraction": float(1 - fx.mean()) if len(x) else np.nan,
        "paired_nonfinite_fraction": float(1 - fy.mean()) if len(y) else np.nan,
        "unpaired_mean": float(xs.mean()) if len(xs) else np.nan,
        "paired_mean": float(ys.mean()) if len(ys) else np.nan,
        "unpaired_std": float(xs.std()) if len(xs) else np.nan,
        "paired_std": float(ys.std()) if len(ys) else np.nan,
    }
    del x, y, xs, ys
    su = v[sub_unpaired]
    sp = v[sub_paired]
    res["unpaired_median_subsample"] = float(np.nanmedian(su)) if np.isfinite(su).any() else np.nan
    res["paired_median_subsample"] = float(np.nanmedian(sp)) if np.isfinite(sp).any() else np.nan
    res["ks_statistic_subsample"] = ks_statistic(su, sp)

    pooled = np.sqrt((res["unpaired_std"] ** 2 + res["paired_std"] ** 2) / 2.0)
    if np.isfinite(pooled) and pooled > 0:
        res["standardized_mean_difference"] = (
            res["unpaired_mean"] - res["paired_mean"]) / pooled
    else:
        res["standardized_mean_difference"] = np.nan
    res["nonfinite_fraction_difference"] = (
        res["unpaired_nonfinite_fraction"] - res["paired_nonfinite_fraction"])
    return res


# ---------------------------------------------------------------------------
# Per-lumisection breakdown
# ---------------------------------------------------------------------------
def per_lumi_counts(packer, uniq, only_mask):
    lk = packer.lumi_key(uniq)  # sorted, because uniq is sorted
    new = np.empty(len(lk), dtype=bool)
    new[0] = True
    np.not_equal(lk[1:], lk[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    n_ev = np.diff(np.append(starts, len(lk)))
    n_only = np.add.reduceat(only_mask.astype(np.int64), starts)
    return lk[starts], n_ev, n_only


def lumi_breakdown(packer, ixA, ixB, pairing):
    lkA, nA, oA = per_lumi_counts(packer, ixA.uniq, pairing.only_A_mask)
    lkB, nB, oB = per_lumi_counts(packer, ixB.uniq, pairing.only_B_mask)
    dfA = pd.DataFrame({"lk": lkA, "events_A": nA, "only_A": oA})
    dfB = pd.DataFrame({"lk": lkB, "events_B": nB, "only_B": oB})
    df = dfA.merge(dfB, on="lk", how="outer").fillna(0)
    for c in ("events_A", "only_A", "events_B", "only_B"):
        df[c] = df[c].astype(np.int64)
    lk = df["lk"].to_numpy(dtype=np.uint64)
    run, lumi = packer.unpack_lumi_key(lk)
    df.insert(0, "run", run.astype(np.int64))
    df.insert(1, "lumi", lumi.astype(np.int64))
    df = df.drop(columns="lk")
    df["common"] = df["events_A"] - df["only_A"]

    status = np.full(len(df), "partial", dtype=object)
    status[(df["only_A"] == 0) & (df["only_B"] == 0)] = "identical"
    status[df["events_B"] == 0] = "only_in_A"
    status[df["events_A"] == 0] = "only_in_B"
    df["status"] = status
    return df.sort_values(["run", "lumi"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def write_capped(df, path, max_rows):
    truncated = len(df) > max_rows
    df.head(max_rows).to_csv(path, index=False)
    return truncated


def events_frame(packer, keys, extra=None):
    run, lumi, evt = packer.unpack(keys)
    df = pd.DataFrame({"run": run.astype(np.int64),
                       "lumi": lumi.astype(np.int64),
                       "event": evt})
    if extra:
        for k, v in extra.items():
            df[k] = v
    return df


def fmt_pct(num, den):
    return "{:.4f}%".format(100.0 * num / den) if den else "n/a"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Compare two productions of the same ntuple "
                    "(events, candidates, branch values).")
    p.add_argument("file_A")
    p.add_argument("file_B")
    p.add_argument("--tree", default="tree")
    p.add_argument("--outdir", default="production_comparison")
    p.add_argument("--rtol", type=float, default=1e-6,
                   help="relative tolerance (default 1e-6; float32 precision is ~1.2e-7)")
    p.add_argument("--atol", type=float, default=1e-12)
    p.add_argument("--cand-keys", nargs="*", default=[],
                   help="branches used to order candidates inside an event before "
                        "pairing them (e.g. the three muon pT). Default: stored order.")
    p.add_argument("--include", default=None,
                   help="regex: only compare branches matching it")
    p.add_argument("--exclude", default=None,
                   help="regex: skip branches matching it")
    p.add_argument("--max-examples", type=int, default=20,
                   help="example differing candidates stored per branch")
    p.add_argument("--max-rows", type=int, default=1_000_000,
                   help="cap on rows of the per-event CSV lists")
    p.add_argument("--chunk", type=int, default=5_000_000,
                   help="candidate pairs compared per chunk (memory vs speed)")
    p.add_argument("--no-dup-check", action="store_true",
                   help="skip the duplicate-candidate fingerprint (saves ~16 bytes/row)")
    p.add_argument("--no-discrimination", action="store_true",
                   help="skip the unpaired-vs-paired distribution comparison")
    p.add_argument("--ks-sample", type=int, default=1_000_000,
                   help="subsample size per group for KS statistic and medians")
    p.add_argument("--threads", type=int, default=4,
                   help="threads for basket decompression (1 = none)")
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    executor = ThreadPoolExecutor(max_workers=args.threads) if args.threads > 1 else None
    rng = np.random.default_rng(args.seed)

    fA = uproot.open(args.file_A)
    fB = uproot.open(args.file_B)
    for label, f, fn in (("A", fA, args.file_A), ("B", fB, args.file_B)):
        if args.tree not in f:
            raise KeyError("Tree {!r} not found in {} ({}). Keys: {}".format(
                args.tree, label, fn, f.keys()))
    tA, tB = fA[args.tree], fB[args.tree]
    if tA.num_entries == 0 or tB.num_entries == 0:
        raise ValueError("Empty tree: A has {}, B has {} entries".format(
            tA.num_entries, tB.num_entries))
    log("A: {:,} rows   B: {:,} rows".format(tA.num_entries, tB.num_entries))

    # ---------------- keys ----------------
    key_names = list(KEYS) + [c for c in args.cand_keys if c not in KEYS]
    log("reading key branches {}".format(key_names))
    colsA = read_key_columns(tA, key_names, "A", executor)
    colsB = read_key_columns(tB, key_names, "B", executor)

    packer = KeyPacker(colsA, colsB)
    log("packing keys: run {} + lumi {} + event {} bits".format(
        packer.b_run, packer.b_lumi, packer.b_evt))
    evA = packer.pack(colsA)
    evB = packer.pack(colsB)
    candA = [colsA[c] for c in args.cand_keys]
    candB = [colsB[c] for c in args.cand_keys]
    del colsA, colsB

    log("indexing A")
    ixA = FileIndex("A", evA, candA)
    del evA
    log("indexing B")
    ixB = FileIndex("B", evB, candB)
    del evB

    log("matching events and pairing candidates")
    pr = Pairing(ixA, ixB, candA, candB)
    del candA, candB
    n_common = len(pr.common)
    n_only_A = int(pr.only_A_mask.sum())
    n_only_B = int(pr.only_B_mask.sum())
    mult_changed = pr.cA != pr.cB
    n_mult_changed = int(mult_changed.sum())
    unpaired_A = ixA.n_rows - pr.n_pairs
    unpaired_B = ixB.n_rows - pr.n_pairs
    log("common events {:,}, only A {:,}, only B {:,}, paired candidates {:,}".format(
        n_common, n_only_A, n_only_B, pr.n_pairs))

    # ---------------- per-lumi breakdown ----------------
    log("per-lumisection breakdown")
    lumi_df = lumi_breakdown(packer, ixA, ixB, pr)
    lumi_df[lumi_df["status"] != "identical"].to_csv(
        os.path.join(args.outdir, "lumi_breakdown.csv"), index=False)
    run_df = lumi_df.groupby("run").agg(
        lumis=("lumi", "size"),
        lumis_only_in_A=("status", lambda s: int((s == "only_in_A").sum())),
        lumis_only_in_B=("status", lambda s: int((s == "only_in_B").sum())),
        lumis_partial=("status", lambda s: int((s == "partial").sum())),
        events_A=("events_A", "sum"), events_B=("events_B", "sum"),
        only_A=("only_A", "sum"), only_B=("only_B", "sum"),
    ).reset_index()
    run_df.to_csv(os.path.join(args.outdir, "run_breakdown.csv"), index=False)

    st_counts = lumi_df["status"].value_counts()
    only_A_in_missing_lumis = int(lumi_df.loc[lumi_df.status == "only_in_A", "only_A"].sum())
    only_B_in_missing_lumis = int(lumi_df.loc[lumi_df.status == "only_in_B", "only_B"].sum())

    # ---------------- event lists ----------------
    log("writing event lists")
    trunc = {}
    trunc["events_only_A.csv"] = write_capped(
        events_frame(packer, ixA.uniq[pr.only_A_mask],
                     {"n_cand_A": ixA.counts[pr.only_A_mask]}),
        os.path.join(args.outdir, "events_only_A.csv"), args.max_rows)
    trunc["events_only_B.csv"] = write_capped(
        events_frame(packer, ixB.uniq[pr.only_B_mask],
                     {"n_cand_B": ixB.counts[pr.only_B_mask]}),
        os.path.join(args.outdir, "events_only_B.csv"), args.max_rows)
    trunc["events_multiplicity_changed.csv"] = write_capped(
        events_frame(packer, pr.common[mult_changed],
                     {"n_cand_A": pr.cA[mult_changed], "n_cand_B": pr.cB[mult_changed]}),
        os.path.join(args.outdir, "events_multiplicity_changed.csv"), args.max_rows)

    mult_hist = pd.DataFrame({"n_cand_A": pr.cA, "n_cand_B": pr.cB}).value_counts() \
        .rename("common_events").reset_index().sort_values(["n_cand_A", "n_cand_B"])
    mult_hist.to_csv(os.path.join(args.outdir, "multiplicity_table.csv"), index=False)

    # ---------------- branches ----------------
    names_A = set(tA.keys())
    names_B = set(tB.keys())
    shared = sorted((names_A & names_B) - set(KEYS))
    if args.include:
        rx = re.compile(args.include)
        shared = [b for b in shared if rx.search(b)]
    if args.exclude:
        rx = re.compile(args.exclude)
        shared = [b for b in shared if not rx.search(b)]

    do_discr = not args.no_discrimination
    subs = {}
    if do_discr:
        for label, matched in (("A", pr.matched_A), ("B", pr.matched_B)):
            unp = np.flatnonzero(~matched)
            if len(unp) == 0:
                subs[label] = None
                continue
            par = np.flatnonzero(matched)
            su = np.sort(rng.choice(unp, size=min(args.ks_sample, len(unp)), replace=False))
            sp = np.sort(rng.choice(par, size=min(args.ks_sample, len(par)), replace=False))
            subs[label] = (su, sp)
            del unp, par

    do_dup = not args.no_dup_check
    if do_dup:
        fpA = RowFingerprint(ixA.n_rows)
        fpB = RowFingerprint(ixB.n_rows)

    n_branch_diff = np.zeros(pr.n_pairs, dtype=np.uint16)
    branch_rows, example_rows, discr_rows, not_compared = [], [], [], []

    log("comparing {} shared branches".format(len(shared)))
    for name in progress(shared, len(shared), "branches"):
        kA, dA, shA = classify(tA[name])
        kB, dB, shB = classify(tB[name])
        if kA == "skip" or kB == "skip":
            not_compared.append((name, shA if kA == "skip" else shB))
            continue
        if kA != kB or shA != shB:
            not_compared.append((name, "structure differs: A {} {} / B {} {}".format(
                kA, shA, kB, shB)))
            continue

        a_all = read_branch(tA[name], kA, executor)
        b_all = read_branch(tB[name], kB, executor)

        if do_dup:
            if kA == "flat":
                fpA.add_flat(a_all)
                fpB.add_flat(b_all)
            else:
                fpA.add_jagged(a_all)
                fpB.add_jagged(b_all)

        if kA == "flat":
            diff, notid, st = compare_flat(a_all, b_all, pr, args.rtol, args.atol, args.chunk)
        else:
            diff, notid, st = compare_jagged(a_all, b_all, pr, args.rtol, args.atol, args.chunk)

        nd = int(diff.sum())
        n_branch_diff += diff
        row = {
            "branch": name, "kind": kA,
            "dtype_A": str(dA), "dtype_B": str(dB),
            "pairs_compared": pr.n_pairs,
            "pairs_differing": nd,
            "fraction_differing": nd / pr.n_pairs if pr.n_pairs else np.nan,
            "pairs_not_bit_identical": int(notid.sum()),
            "differing_in_multi_cand_events": int((diff & pr.pair_multi).sum()),
            "nan_mismatches": st["n_nan_mismatch"],
            "max_abs_diff": st["max_abs_diff"],
            "max_rel_diff": st["max_rel_diff"],
            "mean_diff_B_minus_A": (st["sum_signed_diff"] / st["n_finite_diff"]
                                    if st["n_finite_diff"] else np.nan),
        }
        if kA == "jagged":
            row["length_mismatches"] = st["n_length_mismatch"]
        branch_rows.append(row)

        if nd and args.max_examples > 0:
            idx = np.flatnonzero(diff)[:args.max_examples]
            run, lumi, evt = packer.unpack(pr.common[pr.pair_ev[idx]])
            for j, i in enumerate(idx):
                example_rows.append({
                    "branch": name, "run": int(run[j]), "lumi": int(lumi[j]),
                    "event": int(evt[j]), "cand_rank": int(pr.rank[i]),  # -1: paired by cand keys
                    "value_A": value_repr(a_all, pr.rowsA[i]),
                    "value_B": value_repr(b_all, pr.rowsB[i]),
                })

        if do_discr and kA == "flat" and shA == ():
            for label, vals, matched in (("A", a_all, pr.matched_A),
                                         ("B", b_all, pr.matched_B)):
                if subs.get(label) is None:
                    continue
                su, sp = subs[label]
                ds = distribution_stats(vals, matched, su, sp)
                discr_rows.append(dict({"file": label, "branch": name}, **ds))

        del a_all, b_all, diff, notid

    # ---------------- duplicates within each file ----------------
    dup = {}
    if do_dup:
        for label, ix, fp in (("A", ixA, fpA), ("B", ixB, fpB)):
            log("duplicate check in {} ({} branches in the fingerprint)".format(
                label, fp.n_branches))
            res, table = duplicate_analysis(ix, fp.h)
            dfd = duplicate_events_frame(packer, ix, table)
            trunc["duplicates_{}.csv".format(label)] = write_capped(
                dfd, os.path.join(args.outdir, "duplicates_{}.csv".format(label)),
                args.max_rows)
            res["events_any_duplication"] = len(dfd)
            dup[label] = res
            del table, dfd
        n_fp_branches = fpA.n_branches
        del fpA, fpB

    # ---------------- write branch outputs ----------------
    br_df = pd.DataFrame(branch_rows)
    if len(br_df):
        br_df = br_df.sort_values(["pairs_differing", "branch"], ascending=[False, True])
    br_df.to_csv(os.path.join(args.outdir, "branch_differences.csv"), index=False)
    pd.DataFrame(example_rows, columns=["branch", "run", "lumi", "event", "cand_rank",
                                        "value_A", "value_B"]).to_csv(
        os.path.join(args.outdir, "branch_difference_examples.csv"), index=False)

    # events with at least one differing branch (max over candidates)
    per_event = pr.per_event_max(n_branch_diff)
    has_diff = per_event > 0
    trunc["events_with_value_differences.csv"] = write_capped(
        events_frame(packer, pr.common[has_diff],
                     {"max_branches_differing_per_candidate": per_event[has_diff]}),
        os.path.join(args.outdir, "events_with_value_differences.csv"), args.max_rows)

    if do_discr:
        dd = pd.DataFrame(discr_rows)
        if len(dd):
            dd["separation"] = np.fmax(dd["ks_statistic_subsample"],
                                       dd["nonfinite_fraction_difference"].abs())
            dd = dd.sort_values(["file", "separation"], ascending=[True, False])
        dd.to_csv(os.path.join(args.outdir, "branch_discrimination.csv"), index=False)

    # ---------------- summary ----------------
    n_diff_branches = int((br_df["pairs_differing"] > 0).sum()) if len(br_df) else 0
    n_notid_branches = int((br_df["pairs_not_bit_identical"] > 0).sum()) if len(br_df) else 0
    pairs_any = int((n_branch_diff > 0).sum())
    pairs_any_multi = int(((n_branch_diff > 0) & pr.pair_multi).sum())
    n_pairs_multi = int(pr.pair_multi.sum())

    S = []
    S += ["FILES",
          "  A: {}".format(args.file_A),
          "  B: {}".format(args.file_B),
          "  tree: {}".format(args.tree),
          "",
          "1. DUPLICATES WITHIN EACH FILE (expected: all zero)",
          "  rows (candidates) : A {:,}   B {:,}".format(ixA.n_rows, ixB.n_rows),
          "  distinct events   : A {:,}   B {:,}".format(ixA.n_events, ixB.n_events),
          "  events stored in more than one separate block of rows",
          "  (signature of a file merged twice):",
          "      A {:,}   B {:,}".format(ixA.n_split_events, ixB.n_split_events)]
    if do_dup:
        S += ["  duplicate candidate = another row of the same event with identical",
              "  values in all {} compared branches (fingerprint of the row)".format(n_fp_branches),
              "  extra copies of candidates (rows that duplicate another row):",
              "      A {:,}   B {:,}".format(dup["A"]["extra_candidate_copies"],
                                          dup["B"]["extra_candidate_copies"]),
              "  events containing at least one duplicate candidate:",
              "      A {:,}   B {:,}".format(dup["A"]["events_with_duplicate_candidates"],
                                          dup["B"]["events_with_duplicate_candidates"]),
              "  events where EVERY candidate has an identical copy (the whole event",
              "  was written more than once, in one block or in separate blocks):",
              "      A {:,}   B {:,}".format(dup["A"]["events_fully_duplicated"],
                                          dup["B"]["events_fully_duplicated"]),
              "  events with any of the issues above (listed in duplicates_A/B.csv):",
              "      A {:,}   B {:,}".format(dup["A"]["events_any_duplication"],
                                          dup["B"]["events_any_duplication"])]
        if args.include or args.exclude:
            S.append("  NOTE: --include/--exclude reduce the branches in the fingerprint, "
                     "so rows differing only in skipped branches count as duplicates.")
    else:
        S.append("  duplicate-candidate check skipped (--no-dup-check)")
    S += ["",
          "2. EVENTS  (event = unique run, lumi, event)",
          "  in both files      : {:,}".format(n_common),
          "  only in A          : {:,}  ({} of A's events)".format(n_only_A, fmt_pct(n_only_A, ixA.n_events)),
          "  only in B          : {:,}  ({} of B's events)".format(n_only_B, fmt_pct(n_only_B, ixB.n_events)),
          "",
          "  Per lumisection (a lumisection counts as 'present' in a file if at",
          "  least one of its events is in that file):",
          "    identical event content : {:,}".format(int(st_counts.get("identical", 0))),
          "    present only in A       : {:,}".format(int(st_counts.get("only_in_A", 0))),
          "    present only in B       : {:,}".format(int(st_counts.get("only_in_B", 0))),
          "    in both, events differ  : {:,}".format(int(st_counts.get("partial", 0))),
          "  Of the events only in A, {:,} ({}) sit in lumisections absent from B;".format(
              only_A_in_missing_lumis, fmt_pct(only_A_in_missing_lumis, n_only_A)),
          "  the rest ({:,}) are missing from B inside lumisections B does have.".format(
              n_only_A - only_A_in_missing_lumis),
          "  Of the events only in B, {:,} ({}) sit in lumisections absent from A;".format(
              only_B_in_missing_lumis, fmt_pct(only_B_in_missing_lumis, n_only_B)),
          "  the rest ({:,}) are missing from A inside lumisections A does have.".format(
              n_only_B - only_B_in_missing_lumis),
          "",
          "3. CANDIDATES  (only events present in both files)",
          "  events with a different number of candidates in A and B: {:,} ({})".format(
              n_mult_changed, fmt_pct(n_mult_changed, n_common)),
          "  candidates paired between A and B : {:,}".format(pr.n_pairs),
          "  pairing rule:",
          "    events with the same number of candidates: the k-th candidate of A",
          "    is paired with the k-th of B, ordered by {}  ({:,} pairs)".format(
              args.cand_keys if args.cand_keys else "their stored order in the file",
              pr.n_rank_pairs),
          ("    events with a different number of candidates: paired only when all of "
           "{} are exactly equal  ({:,} pairs)".format(args.cand_keys, pr.n_key_pairs)
           if args.cand_keys else
           "    events with a different number of candidates: NOT paired "
           "(give --cand-keys to pair them)"),
          "  candidates left without a partner : A {:,}   B {:,}".format(unpaired_A, unpaired_B),
          "  (these include all candidates of events present in one file only)",
          "",
          "4. BRANCH VALUES  (paired candidates)",
          "  shared branches compared : {}   (not compared: {})".format(
              len(branch_rows), len(not_compared)),
          "  branches only in A : {}".format(sorted(names_A - names_B)),
          "  branches only in B : {}".format(sorted(names_B - names_A)),
          "  'differ' means |B - A| > atol + rtol*|B| (rtol={}, atol={}), NaN == NaN".format(
              args.rtol, args.atol),
          "  'not bit-identical' means any difference at all",
          "  branches with differing pairs        : {}".format(n_diff_branches),
          "  branches not bit-identical           : {}".format(n_notid_branches),
          "  paired candidates with >=1 differing branch: {:,} ({})".format(
              pairs_any, fmt_pct(pairs_any, pr.n_pairs)),
          "    of which in events with >1 candidate: {:,}   (such pairs: {:,} in total)".format(
              pairs_any_multi, n_pairs_multi)]
    dtype_mism = [r["branch"] for r in branch_rows if r["dtype_A"] != r["dtype_B"]]
    if dtype_mism:
        S.append("  branches whose type changed: {}".format(dtype_mism))
    if not_compared:
        S.append("  not compared:")
        for n_, why in not_compared:
            S.append("    {}: {}".format(n_, why))
    if len(br_df) and n_diff_branches:
        S += ["", "  Top differing branches (pairs differing / pairs compared):"]
        for _, r in br_df[br_df.pairs_differing > 0].head(30).iterrows():
            S.append("    {:40s} {:>12,}  {:.3e}  max|diff| {:.3g}".format(
                r.branch, int(r.pairs_differing), r.fraction_differing, r.max_abs_diff))

    S += ["", "READING HINTS (patterns, not conclusions)"]
    if ixA.n_split_events or ixB.n_split_events:
        S.append("  - Some events are stored in several separate blocks: check the "
                 "merge for files included twice.")
    if do_dup and (dup["A"]["extra_candidate_copies"] or dup["B"]["extra_candidate_copies"]):
        S.append("  - Duplicate candidates inflate the candidate counts and distort the "
                 "A/B pairing; fix the duplication before reading sections 3-4.")
    if n_only_A or n_only_B:
        S.append("  - Events missing in whole lumisections point to input differences "
                 "(files, lumi mask, dataset). Events missing inside shared "
                 "lumisections point to selection or reconstruction differences.")
    if pairs_any and not args.cand_keys and pairs_any_multi == pairs_any:
        S.append("  - All value differences are in multi-candidate events: this may be "
                 "only a change in candidate ORDER. Rerun with --cand-keys.")
    if do_discr:
        S.append("  - branch_discrimination.csv: within each file, candidates without a "
                 "partner vs candidates with one. 'separation' = max(KS statistic, "
                 "|difference of non-finite fractions|). Association only.")

    S += ["", "OUTPUT FILES"]
    for fn in ("duplicates_A.csv", "duplicates_B.csv",
               "lumi_breakdown.csv (lumisections that are not identical)",
               "run_breakdown.csv", "multiplicity_table.csv",
               "events_only_A.csv", "events_only_B.csv",
               "events_multiplicity_changed.csv", "events_with_value_differences.csv",
               "branch_differences.csv", "branch_difference_examples.csv",
               "branch_discrimination.csv"):
        base = fn.split(" ")[0]
        if (base.startswith("duplicates_") and not do_dup) or \
                (base == "branch_discrimination.csv" and not do_discr):
            continue
        mark = "  (TRUNCATED to --max-rows {:,})".format(args.max_rows) if trunc.get(base) else ""
        S.append("  {}{}".format(fn, mark))
    S += ["", "run time {:.0f} s, peak memory {:.1f} GB".format(time.time() - T0, peak_mem_gb())]

    text = "\n".join(S) + "\n"
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(text)
    print()
    print(text)
    if executor is not None:
        executor.shutdown()


if __name__ == "__main__":
    main()