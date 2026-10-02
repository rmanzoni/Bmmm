'''
Self-test for Bmmm.Analysis.LumiMask, the certification mask the inspectors
apply event by event (--lumi-mask) and the CRAB data configs use for
Data.lumiMask.

Pure python: no ROOT, no input file. Builds synthetic JSONs in a tmpdir, then
checks the real data/golden_jsons directory of this release if one is found.

    cmsenv
    python3 test_lumi_mask.py

What it checks:
  1. membership: range edges, gaps, unknown runs, overlapping/adjacent ranges
     merged, the union of several files;
  2. the one-JSON-per-year rule: two files for one year, or a name without
     'Collisions20YY', are hard errors; a missing year is a hard error;
  3. the --lumi-mask contract: data without a spec refuses to run, MC with a
     mask refuses to run, 'none' means no mask, a directory means every JSON;
  4. the pass/fail counters behind the job summary, cache included;
  5. if $CMSSW_BASE/src/Bmmm/Analysis/data/golden_jsons exists: it parses,
     holds one JSON per year 2022-2026, and the per-year run ranges do not
     overlap (they would if a file were mislabelled).
'''

from __future__ import print_function

import os
import json
import shutil
import tempfile

from Bmmm.Analysis.LumiMask import (
    LumiMask, golden_jsons_by_year, golden_json_for_year, make_lumi_mask,
)

FAILURES = []


def check(name, condition, extra=''):
    print(('  OK   ' if condition else '  FAIL ') + name + (('   ' + extra) if extra else ''))
    if not condition:
        FAILURES.append(name)


def raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    except Exception:
        return False
    return False


def write(path, payload):
    with open(path, 'w') as fout:
        json.dump(payload, fout)
    return path


def main():
    tmp = tempfile.mkdtemp(prefix='lumimask_test_')
    try:
        a = write(os.path.join(tmp, 'Cert_Collisions2024_378981_386951_Golden.json'),
                  {'380000': [[1, 10], [20, 30], [25, 40], [41, 45]],
                   '380001': [[5, 5]]})
        b = write(os.path.join(tmp, 'Cert_Collisions2025_391658_398903_Golden.json'),
                  {'392000': [[100, 200]]})

        print('\n== 1. membership ==')
        m = LumiMask([a, b])
        check('first lumi of a range', m.contains(380000, 1))
        check('last lumi of a range', m.contains(380000, 10))
        check('gap between ranges rejected', not m.contains(380000, 15))
        check('overlapping ranges merged', m.contains(380000, 35))
        check('adjacent ranges merged (41 follows 40)', m.contains(380000, 41))
        check('beyond the last range rejected', not m.contains(380000, 46))
        check('single-lumi range', m.contains(380001, 5) and not m.contains(380001, 6))
        check('lumi 0 rejected', not m.contains(380000, 0))
        check('second file in the union', m.contains(392000, 150))
        check('unknown run rejected', not m.contains(999999, 1))
        check('certified lumi count', m.n_lumis == 10 + 26 + 1 + 101, 'n_lumis=%d' % m.n_lumis)

        print('\n== 2. one JSON per year ==')
        by_year = golden_jsons_by_year(tmp)
        check('years read from the names', sorted(by_year) == ['2024', '2025'])
        check('golden_json_for_year', golden_json_for_year(2025, tmp) == b)
        check('missing year is an error', raises(KeyError, golden_json_for_year, 2026, tmp))
        dup = write(os.path.join(tmp, 'Cert_Collisions2024_378981_386951_Muon.json'),
                    {'380000': [[1, 50]]})
        check('two JSONs for one year is an error', raises(ValueError, golden_jsons_by_year, tmp))
        os.remove(dup)
        bad = write(os.path.join(tmp, 'my_mask.json'), {'1': [[1, 1]]})
        check('name without Collisions20YY is an error', raises(ValueError, golden_jsons_by_year, tmp))
        os.remove(bad)
        check('empty directory is an error',
              raises(IOError, golden_jsons_by_year, tempfile.mkdtemp(prefix='empty_')))

        print('\n== 3. the --lumi-mask contract ==')
        check('data without --lumi-mask refuses to run', raises(ValueError, make_lumi_mask, '', False))
        check('MC with a mask refuses to run', raises(ValueError, make_lumi_mask, a, True))
        check('MC without a mask -> no mask', make_lumi_mask('', True) is None)
        check("data with 'none' -> no mask", make_lumi_mask('none', False) is None)
        from_dir = make_lumi_mask(tmp, False)
        check('a directory -> every JSON in it', sorted(from_dir.paths) == sorted([a, b]))
        from_list = make_lumi_mask('%s,%s' % (a, b), False)
        check('comma-separated files', from_list.n_lumis == m.n_lumis)
        check('missing file is an error',
              raises(IOError, make_lumi_mask, os.path.join(tmp, 'nope.json'), False))
        os.environ['BMMM_DATADIR'] = tmp + '_absent'
        check("'golden' with no golden dir is an error", raises(IOError, make_lumi_mask, 'golden', False))
        os.environ.pop('BMMM_DATADIR')

        print('\n== 4. counters ==')
        c = LumiMask([a])
        for run, lumi in [(380000, 1), (380000, 1), (380000, 15), (380001, 5), (999, 1)]:
            c.contains(run, lumi)
        check('pass / fail counted per event, cache included',
              (c.n_pass, c.n_fail) == (3, 2), 'pass=%d fail=%d' % (c.n_pass, c.n_fail))
        check('rejected runs recorded', c.failed_runs == set([380000, 999]))
        check('summary mentions the rejection', '2 rejected' in c.summary())

    finally:
        shutil.rmtree(tmp)

    print('\n== 5. the golden_jsons directory of this release ==')
    cmssw_base = os.environ.get('CMSSW_BASE', '')
    real = os.path.join(cmssw_base, 'src', 'Bmmm', 'Analysis', 'data', 'golden_jsons')
    if not cmssw_base or not os.path.isdir(real):
        print('  SKIP %s not found' % real)
    else:
        by_year = golden_jsons_by_year(real)
        check('one JSON for each of 2022-2026', sorted(by_year) == ['2022', '2023', '2024', '2025', '2026'],
              str(sorted(by_year)))
        spans = []
        for year, path in sorted(by_year.items()):
            mask = LumiMask([path])
            spans.append((mask.first_run, mask.last_run, year))
            print('       %s  runs %d-%d  %7d lumis  %s'
                  % (year, mask.first_run, mask.last_run, mask.n_lumis, os.path.basename(path)))
        spans.sort()
        overlap = [(spans[k][2], spans[k + 1][2]) for k in range(len(spans) - 1)
                   if spans[k][1] >= spans[k + 1][0]]
        check('per-year run ranges do not overlap', not overlap, str(overlap) if overlap else '')
        order = [s[2] for s in spans]
        check('run ranges ordered like the years', order == sorted(order), str(order))

    print('\n%s' % ('ALL CHECKS PASSED' if not FAILURES
                    else '%d FAILURE(S): %s' % (len(FAILURES), FAILURES)))
    return 1 if FAILURES else 0


if __name__ == '__main__':
    raise SystemExit(main())
