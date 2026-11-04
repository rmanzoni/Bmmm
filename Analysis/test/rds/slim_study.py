#!/usr/bin/env python3
'''
Where the R(Ds) ntuples can be slimmed: how many rows each candidate
selection would remove, and how many bytes each group of branches takes.

    python3 slim_study.py /pnfs/.../rds_mc_nov26.root [more files, a dir or a glob] -o slim_mc
    python3 slim_study.py /pnfs/.../rds_data_part1/ -o slim_data --max-rows 3000000

No ROOT, no CMSSW: numpy + uproot.

Rows. A sample of the file(s) is read in blocks taken in random order (an
unbiased sample of the whole) until --max-rows rows. Each selection in CUTS is
applied alone and cumulatively in the listed order; the table gives the
fraction of rows kept
    MC   per truth class (gen_cand): the four signals, Hb (10-15),
         mu from elsewhere (20-21), KKpi not from one Ds (30), unmatched (-1)
    data per charge class (right sign, mu-pi wrong, KK wrong) and m(KKpi)
         region (signal region 1.91-2.028 GeV / outside)
and, for "one candidate per event", the rows left after keeping the first
candidate (cand_idx order, i.e. leading pT(KKpi mu)) of each event among those
surviving the previous cuts.

Bytes. Compressed bytes per row of every branch of the first file, summed in
groups (refitted tracks, vertex covariances, isolation, gen, ...), so that the
rows-times-branches budget can be read off directly.

Writes <outdir>/slim_rows.csv, slim_bytes.csv and summary.md, and prints the
summary.
'''

import os
import re
import sys
import glob
import json
import time
import argparse
from collections import OrderedDict

import numpy as np

M_BS = 5.36692

# (name, branches needed, mask function, where it comes from)
SCHEME = os.environ.get('RDS_SLIM_SCHEME', 'free')


def _s(q):
    return '%s_%s' % (SCHEME, q)


def _hlt_or(d):
    paths = [k for k in d if re.match(r'^HLT_Mu[0-9p]+_IP[0-9p]+$', k)]
    if not paths:
        return np.ones(len(d['run']), dtype=bool)
    return np.any(np.stack([d[p] > 0.5 for p in paths]), axis=0)


CUTS = [
    ('trigger: HLT OR and muon matched (dR < 0.05)',
     ['mu_trig_match'], lambda d: _hlt_or(d) & (d['mu_trig_match'] > 0.5), 'AN 5.1 (applied at production in the AN)'),
    ('KK opposite sign (drop KK-wrong)',
     ['charge_class'], lambda d: d['charge_class'] != 2, 'AN 6.2: only mu-pi wrong used for comb. + fakes'),
    ('m(KKpi) in 1.84-2.10 GeV (D+ peak and a longer upper sideband)',
     ['kkpi_mass'], lambda d: (d['kkpi_mass'] > 1.84) & (d['kkpi_mass'] < 2.10), 'proposal'),
    ('m(KKpi) in 1.908-2.028 GeV (AN: m(Ds) +- 60 MeV)',
     ['kkpi_mass'], lambda d: (d['kkpi_mass'] > 1.908) & (d['kkpi_mass'] < 2.028), 'AN 5.3'),
    ('Bs vertex fit usable',
     [_s('bs_good')], lambda d: d[_s('bs_good')] > 0.5, 'AN 5.3 (valid vertices)'),
    ('muon medium ID',
     ['mu_id_medium'], lambda d: d['mu_id_medium'] > 0.5, 'AN 5.8'),
    ('muon global',
     ['mu_id_global'], lambda d: d['mu_id_global'] > 0.5, 'AN 5.8'),
    ('muon rel. isolation (0.3 cone) < 0.3',
     ['mu_reliso_03'], lambda d: d['mu_reliso_03'] < 0.3, 'AN 5.8'),
    ('Lxy(Ds) from V_Bs < 1 cm',
     [_s('ds_lxy_bs')], lambda d: d[_s('ds_lxy_bs')] < 1.0, 'AN 5.8'),
    ('cos(alpha) > 0.8, Ds momentum vs PV -> V_Ds',
     [_s('ds_cos3d')], lambda d: d[_s('ds_cos3d')] > 0.8, 'AN 5.8 (alpha > 0.8)'),
    ('m(KKpi mu) < M(Bs)',
     ['mass'], lambda d: d['mass'] < M_BS, 'AN 5.7 (ranking, not a cut)'),
]
ONE_PER_EVENT = 'one candidate per event (first in cand_idx)'

MC_CLASSES = OrderedDict([
    ('Ds mu',  lambda g: g == 0), ('Ds* mu', lambda g: g == 1),
    ('Ds tau', lambda g: g == 2), ('Ds* tau', lambda g: g == 3),
    ('Hb',     lambda g: (g >= 10) & (g <= 15)),
    ('mu other', lambda g: (g == 20) | (g == 21)),
    ('KKpi not one Ds', lambda g: g == 30),
    ('unmatched', lambda g: g == -1),
])

BYTE_GROUPS = [   # first match wins
    ('gen / hammer',             r'^(gen_|hammer_|.*_gen_)'),
    ('trigger / L1 / prescales', r'(^HLT_|^L1_|^l1|_tag$|_ps$|trig_)'),
    ('track covariances',        r'^(k1|k2|pi|mu)_cov_'),
    ('track hit content',        r'^(k1|k2|pi|mu)_.*(hit|layer|nlost|npix|nstrip|valid)'),
    ('tracks: other',            r'^(k1|k2|pi|mu)_'),
    ('vertex covariances',       r'_cov_(xx|xy|xz|yy|yz|zz)$'),
    ('refitted momenta',         r'_rf_(pt|eta|phi)$'),
    ('Bs hypotheses (coll, xyz, math)', r'_(coll|xyz_bs|math1_bs|math2_bs)_'),
    ('soft photon',              r'_gamma_'),
    ('isolation (mu, Ds, B)',    r'^(mu|ds|b)_(iso|reliso)'),
    ('vertices: other',          r'_(phi|ds|bs)_'),
    ('event',                    r'^(run|lumi|event|npu|nti|nvtx|era|bph_part|pv_|bs_)'),
]


def expand(inputs):
    files = []
    for x in inputs:
        if os.path.isdir(x):
            files += sorted(glob.glob(os.path.join(x, '*.root')))
        elif any(c in x for c in '*?['):
            files += sorted(glob.glob(x))
        else:
            files.append(x)
    return files


def read_sample(files, tree, branches, max_rows, block, seed, log):
    import uproot
    blocks = []
    for f in files:
        with uproot.open(f) as h:
            n = h[tree].num_entries
        blocks += [(f, a, min(a + block, n)) for a in range(0, n, block)]
    n_tot = sum(b[2] - b[1] for b in blocks)
    order = np.random.default_rng(seed).permutation(len(blocks))
    parts, n_read, t0 = [], 0, time.time()
    cur, cur_tree = None, None
    for i in order:
        f, a, b = blocks[i]
        if f != cur:
            cur, cur_tree = f, uproot.open(f)[tree]
        keys = set(cur_tree.keys())
        parts.append(cur_tree.arrays([x for x in branches if x in keys], entry_start=a, entry_stop=b, library='np'))
        n_read += b - a
        log('  %d / %d rows (%.1f%%), %.0f rows/s' % (n_read, n_tot, 100. * n_read / n_tot, n_read / (time.time() - t0)))
        if max_rows and n_read >= max_rows:
            break
    d = dict((k, np.concatenate([p[k] for p in parts if k in p])) for k in parts[0])
    return d, n_read, n_tot


def byte_budget(path, tree):
    import uproot
    t = uproot.open(path)[tree]
    n = max(t.num_entries, 1)
    rows = []
    for k in t.keys():
        br = t[k]
        grp = next((g for g, rx in BYTE_GROUPS if re.search(rx, k)), 'other')
        rows.append((k, grp, br.compressed_bytes / n, br.uncompressed_bytes / n))
    return rows, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('-o', '--outdir', required=True)
    ap.add_argument('--tree', default='tree')
    ap.add_argument('--max-rows', type=int, default=2000000)
    ap.add_argument('--block', type=int, default=50000)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    log = lambda m: print(m, flush=True)

    files = expand(args.inputs)
    if not files:
        sys.exit('no input files')
    import uproot
    keys = set(uproot.open(files[0])[args.tree].keys())
    mc = 'gen_cand' in keys
    need = ['run', 'lumi', 'event', 'cand_idx', 'charge_class', 'kkpi_mass'] + (['gen_cand'] if mc else [])
    need += [k for k in keys if re.match(r'^HLT_Mu[0-9p]+_IP[0-9p]+$', k)]
    cuts = []
    for name, brs, fn, src in CUTS:
        if all(b in keys for b in brs):
            cuts.append((name, fn, src))
            need += brs
        else:
            log('skipped (branches missing): %s %s' % (name, [b for b in brs if b not in keys]))
    need = sorted(set(need))
    log('%d file(s), %s, reading %d branches' % (len(files), 'MC' if mc else 'data', len(need)))
    d, n_read, n_tot = read_sample(files, args.tree, need, args.max_rows, args.block, args.seed, log)
    n = len(d['run'])

    # the populations the fractions are given for
    pops = OrderedDict([('all rows', np.ones(n, dtype=bool))])
    if mc:
        g = d['gen_cand'].astype(int)
        for c, f in MC_CLASSES.items():
            pops[c] = f(g)
    else:
        cc, m3 = d['charge_class'].astype(int), d['kkpi_mass']
        sr = (m3 > 1.908) & (m3 < 2.028)
        for cname, cv in (('right sign', 0), ('mu-pi wrong', 1), ('KK wrong', 2)):
            pops['%s, SR' % cname] = (cc == cv) & sr
            pops['%s, outside SR' % cname] = (cc == cv) & ~sr

    rows = []
    cum = np.ones(n, dtype=bool)
    for name, fn, src in cuts:
        with np.errstate(invalid='ignore'):
            m = np.asarray(fn(d), dtype=bool)
        cum = cum & m
        rows.append((name, src, m.copy(), cum.copy()))
    # one candidate per event among the survivors of all the cuts above
    idx = np.nonzero(cum)[0]
    key = np.stack([d['run'][idx], d['lumi'][idx], d['event'][idx]], axis=1).astype(np.int64)
    order = np.lexsort((d['cand_idx'][idx], key[:, 2], key[:, 1], key[:, 0]))
    ks = key[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(ks[1:] != ks[:-1], axis=1)
    one = np.zeros(n, dtype=bool)
    one[idx[order[first]]] = True
    rows.append((ONE_PER_EVENT, 'AN 5.7 (AN ranking: charges, m < M(Bs), pT)', one, one))

    # ---- tables ----------------------------------------------------------------
    def frac(m, p):
        k = int(p.sum())
        return float(np.sum(m & p)) / k if k else float('nan')

    out_csv = [['cut', 'source', 'population', 'kept_alone', 'kept_cumulative', 'n_population']]
    md = ['# Slimming study: rows', '',
          'Input: %s (%d file(s)); %d of %d rows read (%.1f%%, random blocks). Scheme for the vertex '
          'quantities: `%s`.' % (files[0] if len(files) == 1 else os.path.dirname(files[0]) + '/...',
                                  len(files), n_read, n_tot, 100. * n_read / n_tot, SCHEME), '',
          'Fraction of rows kept: **alone / cumulative** (cuts applied in the listed order).', '']
    names = list(pops)
    md.append('| cut | source | ' + ' | '.join(names) + ' |')
    md.append('|---|---|' + '---|' * len(names))
    for name, src, alone, cumm in rows:
        cells = []
        for p in names:
            a, c = frac(alone, pops[p]), frac(cumm, pops[p])
            cells.append('%.3f / %.3f' % (a, c))
            out_csv.append([name, src, p, '%.5f' % a, '%.5f' % c, int(pops[p].sum())])
        md.append('| %s | %s | %s |' % (name, src, ' | '.join(cells)))
    md.append('| population (rows read) | | ' + ' | '.join('%d' % pops[p].sum() for p in names) + ' |')
    if mc:
        ev = len(np.unique(np.stack([d['run'], d['lumi'], d['event']], axis=1), axis=0))
        md += ['', 'Candidates per event with at least one: %.2f (rows / distinct events).' % (n / max(ev, 1))]

    # ---- bytes --------------------------------------------------------------------
    brows, nent = byte_budget(files[0], args.tree)
    grp = OrderedDict()
    for k, gname, cb, ub in brows:
        x = grp.setdefault(gname, [0, 0., 0.])
        x[0] += 1; x[1] += cb; x[2] += ub
    tot_c = sum(v[1] for v in grp.values())
    md += ['', '# Slimming study: bytes per row (%s, %d rows)' % (os.path.basename(files[0]), nent), '',
           '| branch group | branches | compressed B/row | share | uncompressed B/row |', '|---|---|---|---|---|']
    for gname, (nb, cb, ub) in sorted(grp.items(), key=lambda kv: -kv[1][1]):
        md.append('| %s | %d | %.1f | %.1f%% | %.1f |' % (gname, nb, cb, 100 * cb / tot_c, ub))
    md.append('| total | %d | %.1f | 100%% | %.1f |' % (len(brows), tot_c, sum(v[2] for v in grp.values())))

    # the other trees of the file (MC: tree_gen, one row per processed event)
    h = uproot.open(files[0])
    trees = [(k.split(';')[0], h[k]) for k in h.keys(cycle=False) if h.classname_of(k) == 'TTree']
    md += ['', '| tree | entries | branches | compressed MB |', '|---|---|---|---|']
    for name, t in trees:
        md.append('| %s | %d | %d | %.1f |' % (name, t.num_entries, len(t.keys()),
                                              sum(t[k].compressed_bytes for k in t.keys()) / 1e6))

    with open(os.path.join(args.outdir, 'summary.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    with open(os.path.join(args.outdir, 'slim_rows.csv'), 'w') as f:
        f.write('\n'.join(','.join('"%s"' % c if isinstance(c, str) else str(c) for c in r) for r in out_csv) + '\n')
    with open(os.path.join(args.outdir, 'slim_bytes.csv'), 'w') as f:
        f.write('branch,group,compressed_bytes_per_row,uncompressed_bytes_per_row\n')
        for k, gname, cb, ub in sorted(brows, key=lambda r: -r[2]):
            f.write('%s,"%s",%.3f,%.3f\n' % (k, gname, cb, ub))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
