#!/usr/bin/env python3
"""
make_filelists_cscs.py
======================
Query DAS (DBS instance prod/phys03) for the two published RJpsi skim
campaigns at T2_CH_CSCS and write per-year flat file lists -- one bare LFN
per line, e.g. /store/user/manzoni/... . Those lists feed
submitter_data_diff_pnfs.py, which turns each LFN into a CSCS xrootd URL at
submit time (so the lists stay portable if the door ever changes).

USAGE
-----
Run from a shell with dasgoclient on $PATH and a valid grid proxy:

    source /cvmfs/cms.cern.ch/cmsset_default.sh   # or cmsenv in your release
    voms-proxy-init -voms cms -valid 192:00
    python3 make_filelists_cscs.py

Options:
    --outdir DIR        where to write the lists (default: .)
    --tag TAG           filename suffix (default: today, e.g. 03jul26)
    --check-site        query 'site dataset=...' and warn if T2_CH_CSCS absent
    --dump-first N      print the first N LFNs of each dataset (schema/sanity)
    --dry-run           print the DAS queries and exit (no dasgoclient calls)

FAIL-LOUD (ground-truth over reconstruction)
--------------------------------------------
Every dataset name comes from a LIVE DAS query -- never rebuilt from
part x era x version. The script asserts each pattern resolves to >=1
dataset and each dataset to >=1 file, refuses to write an empty list, and
prints per-dataset + total counts. Fill EXPECTED below to hard-assert exact
dataset counts on re-runs.

NOTE on instance: these are /USER (CRAB-published) datasets, so BOTH the
dataset-level and file-level queries carry `instance=prod/phys03`. The
default prod/global returns nothing for them.
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

DBS_INSTANCE = 'prod/phys03'   # USER datasets live in phys03, NOT prod/global

# dataset patterns, verbatim from DAS; one flat list written per key
CAMPAIGNS = {
    '2024': '/ParkingDoubleMuonLowMass*/manzoni-rjpsi_run3_23jun26_v2_ParkingDoubleMuonLowMass*_Run2024*_MINIv6NANOv15_*-2344730b5c341a1d09de0777eeb1fe94/USER',
    '2025': '/ParkingDoubleMuonLowMass*/manzoni-rjpsi_run3_23jun26_v2_ParkingDoubleMuonLowMass*_Run2025*_PromptReco_*-cb31c552909115217b0520f2761c9df4/USER',
}

# optional exact-count guards for re-runs, e.g. {'2024': 8, '2025': 6}.
# Empty dict = off (first run, when the count is not yet known).
EXPECTED = {}


def das(query, limit=0):
    """Run dasgoclient and return stripped, non-empty output lines."""
    cmd = ['dasgoclient', '-query=%s' % query, '-limit=%d' % limit]
    try:
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=300,
        )
    except FileNotFoundError:
        raise RuntimeError(
            'dasgoclient not found on $PATH -- source '
            '/cvmfs/cms.cern.ch/cmsset_default.sh (or cmsenv) first.')
    except subprocess.TimeoutExpired:
        raise RuntimeError('dasgoclient timed out (300s) on query: %s' % query)
    if res.returncode != 0:
        raise RuntimeError(
            'dasgoclient failed (rc=%d) on query: %s\n%s'
            % (res.returncode, query, res.stderr.strip()))
    return [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]


def datasets_for(pattern):
    return sorted(set(das('dataset=%s instance=%s' % (pattern, DBS_INSTANCE))))


def files_for(dataset):
    return sorted(set(das('file dataset=%s instance=%s' % (dataset, DBS_INSTANCE))))


def sites_for(dataset):
    return sorted(set(das('site dataset=%s instance=%s' % (dataset, DBS_INSTANCE))))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outdir', default='.')
    ap.add_argument('--tag', default=datetime.now().strftime('%d%b%y').lower())
    ap.add_argument('--check-site', action='store_true')
    ap.add_argument('--dump-first', type=int, default=0, metavar='N')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if args.dry_run:
        for year, pat in CAMPAIGNS.items():
            print('>>>> [%s] dataset=%s instance=%s' % (year, pat, DBS_INSTANCE))
        return 0

    os.makedirs(args.outdir, exist_ok=True)
    grand_total = 0

    for year, pattern in CAMPAIGNS.items():
        print('=' * 78)
        print('>>>> campaign %s' % year)
        print('>>>>   pattern: %s' % pattern)

        datasets = datasets_for(pattern)
        if not datasets:
            raise RuntimeError(
                'no datasets matched pattern for %s in %s -- check the '
                'pattern/instance, or that publication has finished.'
                % (year, DBS_INSTANCE))
        print('>>>>   %d dataset(s) matched' % len(datasets))

        if year in EXPECTED and len(datasets) != EXPECTED[year]:
            raise RuntimeError(
                'expected %d datasets for %s, got %d (fail-loud guard).'
                % (EXPECTED[year], year, len(datasets)))

        all_files = []
        for ds in datasets:
            flist = files_for(ds)
            if not flist:
                raise RuntimeError('dataset has 0 files: %s' % ds)
            print('>>>>     %6d files  %s' % (len(flist), ds))
            if args.check_site:
                st = sites_for(ds)
                if 'T2_CH_CSCS' not in st:
                    print('>>>>     WARNING: T2_CH_CSCS not in sites %s for %s'
                          % (st, ds))
            if args.dump_first:
                for f in flist[:args.dump_first]:
                    print('>>>>         %s' % f)
            all_files += flist

        # dedup across datasets (should not overlap, but be safe) + sort
        all_files = sorted(set(all_files))
        if not all_files:
            raise RuntimeError(
                'no files collected for %s -- refusing to write an empty list.'
                % year)

        outpath = os.path.join(
            args.outdir, 'files_data%s_cscs_%s.txt' % (year, args.tag))
        with open(outpath, 'w') as fh:
            fh.write('\n'.join(all_files) + '\n')
        print('>>>>   wrote %d files -> %s' % (len(all_files), outpath))
        grand_total += len(all_files)

    print('=' * 78)
    print('>>>> DONE. %d file(s) across %d campaign(s).'
          % (grand_total, len(CAMPAIGNS)))
    return 0


if __name__ == '__main__':
    sys.exit(main())