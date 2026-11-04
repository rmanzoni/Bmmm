#!/usr/bin/env python3
'''
Merge the per-input-file parts of one CRAB job into a single ntuple, with
uproot instead of hadd.

    python3 merge_ntuples.py [--compression zstd:5] [--step 100000] out.root part0.root part1.root ...

Why not hadd: hadd keeps the input baskets' layout (the inspector flushes every
5000 rows, so ~20 kB baskets for a float32 branch) and needs ROOT. Here every
tree is streamed through a buffer of --step entries and written back with one
basket per branch per buffer (~100 kB float32 baskets at the default 25000).
Memory is bounded by the buffer, independent of the number or size of the
parts.

Benchmark (Nov 2026, 5 signal outputs, 102k entries, CMSSW uproot 5.6.6):
  hadd -f505        147.8 MB, full read 4.1 s
  this, zstd:5      145.9 MB, full read 1.6 s, 17 s to write, peak RSS 0.5 GB at
                    --step 20000 (1.3 GB at 1e5, same size and read speed)
  zstd:1 / 9        +4% / -0.4% size; lz4 +29% size; zlib:4 slower to write and
                    read; lzma:1 -5% size but 6x slower to read. Hence zstd:5.

The ntuples are flat (float32 + int64 run/lumi/event, NtupleWriter.py): the
schema of every tree is taken from the first part and every other part must
match it exactly (names and dtypes), otherwise the merge fails. Objects that
are not TTrees are refused: hadd would sum histograms, this does not.

After writing, the output is reopened and checked tree by tree: number of
entries and the sums of the int64 branches must equal those of the parts. The
output is written under a temporary name and renamed only if the check passes.
'''

import os
import sys
import argparse
import numpy as np
import uproot

CODECS = {'zstd': uproot.ZSTD, 'lz4': uproot.LZ4, 'zlib': uproot.ZLIB, 'lzma': uproot.LZMA}


def parse_compression(spec):
    algo, _, level = spec.partition(':')
    if algo.lower() not in CODECS:
        raise argparse.ArgumentTypeError('unknown compression %r (one of %s)' % (algo, ', '.join(CODECS)))
    return CODECS[algo.lower()](int(level) if level else 5)


def tree_names(path):
    with uproot.open(path) as f:
        cls = {k.split(';')[0]: c for k, c in f.classnames().items()}
    other = sorted(k for k, c in cls.items() if c != 'TTree')
    if other:
        raise RuntimeError('%s has objects that are not TTrees: %s (merge_ntuples.py does not sum '
                           'histograms)' % (path, ', '.join('%s (%s)' % (k, cls[k]) for k in other)))
    return sorted(cls)


def schema(tree):
    return {b.name: np.dtype(b.interpretation.to_dtype) for b in tree.branches}


def merge(out, parts, compression, step):
    trees = tree_names(parts[0])
    for p in parts[1:]:
        if tree_names(p) != trees:
            raise RuntimeError('%s has trees %s, %s has %s' % (parts[0], trees, p, tree_names(p)))

    expected = {}
    tmp = out + '.tmp'
    with uproot.recreate(tmp, compression=compression) as fout:
        for name in trees:
            with uproot.open(parts[0]) as f:
                types = schema(f[name])
            ints = [b for b, t in types.items() if t.kind in 'iu']
            fout.mktree(name, types)
            n_tot, sums = 0, dict.fromkeys(ints, 0)
            buf, n_buf = [], 0

            def write():
                fout[name].extend({b: np.concatenate([c[b] for c in buf]) for b in types})
                buf.clear()

            for p in parts:
                with uproot.open(p) as f:
                    t = f[name]
                    if schema(t) != types:
                        raise RuntimeError('%s:%s schema differs from %s:%s' % (p, name, parts[0], name))
                    for chunk in t.iterate(library='np', step_size=step):
                        n = len(chunk[next(iter(types))])
                        if n == 0:
                            continue
                        buf.append(chunk)
                        n_buf += n
                        n_tot += n
                        for b in ints:
                            sums[b] += int(chunk[b].sum(dtype=np.int64))
                        if n_buf >= step:
                            write()
                            n_buf = 0
            if buf:
                write()
            expected[name] = (n_tot, sums)

    # reopen and check what was written
    with uproot.open(tmp) as f:
        for name, (n_tot, sums) in expected.items():
            t = f[name]
            got_sums = {b: int(t[b].array(library='np').sum(dtype=np.int64)) for b in sums}
            ok = t.num_entries == n_tot and got_sums == sums
            print('>>> merged %-10s %9d entries (parts %9d) %s' % (name, t.num_entries, n_tot, 'OK' if ok else 'MISMATCH'))
            if not ok:
                raise RuntimeError('%s: merged %d entries, sums %s; parts %d entries, sums %s'
                                   % (name, t.num_entries, got_sums, n_tot, sums))
    os.replace(tmp, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--compression', type=parse_compression, default='zstd:5',
                    help='algo:level, algo in zstd, lz4, zlib, lzma (default zstd:5)')
    ap.add_argument('--step', type=int, default=25000, help='entries per written basket (default 25000)')
    ap.add_argument('out')
    ap.add_argument('parts', nargs='+')
    a = ap.parse_args()
    if os.path.abspath(a.out) in map(os.path.abspath, a.parts):
        sys.exit('output %s is also an input' % a.out)
    merge(a.out, a.parts, a.compression, a.step)


if __name__ == '__main__':
    main()
