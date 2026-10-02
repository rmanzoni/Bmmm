'''
Certified-luminosity (golden JSON) mask, applied EVENT BY EVENT in the ntuplizers.

Why here and not only in CRAB
-----------------------------
CRAB's Data.lumiMask works with FileBased splitting, but in scriptExe mode it
only decides which FILES a job gets: CRAB writes the allowed lumis into the
PSet, the job scripts read nothing but PSet.process.source.fileNames, and the
FWLite inspectors then loop over every event in those files. A file holding a
single certified lumi would be ntuplized whole, uncertified lumis included.

So the mask is applied in the inspector itself, on (run, lumi) of every event.
The CRAB lumiMask is still set, for what it is good at: dropping files -- and
whole jobs -- that contain no certified lumi at all.

One definition, three users
---------------------------
  * the inspectors (mm and J/psi + charged), through make_lumi_mask(), from
    their --lumi-mask argument;
  * the CRAB data configs, through golden_jsons_by_year(), for Data.lumiMask;
  * the SLURM data submitters, through golden_json_dir(), for the path they
    hand to --lumi-mask.

Pure python on purpose (json, bisect, os, re): the CRAB client environment
imports it too, and must not pay for -- or depend on -- ROOT.

The golden directory
--------------------
Bmmm/Analysis/data/golden_jsons holds exactly ONE certification JSON per year.
The year is read from 'Collisions20YY' in the file name (the official
Cert_Collisions20YY_<firstrun>_<lastrun>_Golden.json convention). Two files for
one year -- a Golden and a Muon JSON, say -- is a hard error rather than a
silent union of the two: which one applies is a decision, not a default.

Spec accepted by --lumi-mask
----------------------------
  'golden'          every JSON in golden_json_dir() (one per year, see above)
  <directory>       every JSON in that directory, same one-per-year rule
  <file>[,<file>]   these JSON files, merged
  'none'            explicitly no mask (data: prints a loud warning)
  ''                MC: no mask. DATA: hard error -- forgetting the mask must
                    not be possible, so data has to say 'none' to go without.

Run numbers are globally unique, so the union of the per-year JSONs is the
right mask for a file of any year: the 2024 JSON says nothing about 2025 runs.
Runs absent from every JSON -- e.g. 2026 runs after the last certified one --
are rejected, which is exactly what a golden selection means.
'''

from __future__ import print_function

import os
import re
import json
import hashlib
from bisect import bisect_right
from glob import glob

GOLDEN_SUBDIR = 'golden_jsons'
_YEAR_IN_NAME = re.compile(r'Collisions(20\d\d)', re.IGNORECASE)
_EXTENSIONS   = ('.json', '.txt')


##########################################################################################
#####      LOCATING THE JSON FILES
##########################################################################################
def golden_json_dir():
    '''The directory holding the golden JSONs.

    Same convention as the L1 menus of inspector_mm_analysis: BMMM_DATADIR wins
    when set (a CRAB job ships data/ pieces into its working directory and
    points BMMM_DATADIR there), otherwise $CMSSW_BASE/src/Bmmm/Analysis/data.
    '''
    datadir = os.environ.get('BMMM_DATADIR', '')
    if not datadir:
        cmssw_base = os.environ.get('CMSSW_BASE', '')
        if not cmssw_base:
            raise RuntimeError('neither BMMM_DATADIR nor CMSSW_BASE is set: cannot '
                               'locate the golden JSONs. Run cmsenv, or pass the '
                               'directory explicitly to --lumi-mask.')
        datadir = os.path.join(cmssw_base, 'src', 'Bmmm', 'Analysis', 'data')
    return os.path.join(datadir, GOLDEN_SUBDIR)


def golden_jsons_by_year(directory=None):
    '''{year: path} for the certification JSONs in `directory`.

    Fails loudly on: a missing or empty directory, a file whose name carries no
    'Collisions20YY', and two files for the same year.
    '''
    directory = directory or golden_json_dir()
    if not os.path.isdir(directory):
        raise IOError('golden JSON directory %s does not exist' % directory)
    files = sorted(f for f in glob(os.path.join(directory, '*'))
                   if f.endswith(_EXTENSIONS) and os.path.isfile(f))
    if not files:
        raise IOError('no .json / .txt file in %s' % directory)

    by_year = {}
    for path in files:
        match = _YEAR_IN_NAME.search(os.path.basename(path))
        if not match:
            raise ValueError('cannot tell the year of %s: expected "Collisions20YY" '
                             'in the file name' % path)
        year = match.group(1)
        if year in by_year:
            raise ValueError('two certification JSONs for %s in %s:\n  %s\n  %s\n'
                             'keep exactly one per year (which one applies is a '
                             'decision, not a default)' % (year, directory, by_year[year], path))
        by_year[year] = path
    return by_year


def golden_json_for_year(year, directory=None):
    '''The certification JSON of one year; hard error if there is none.'''
    by_year = golden_jsons_by_year(directory)
    year = str(year)
    if year not in by_year:
        raise KeyError('no golden JSON for %s in %s (have: %s)'
                       % (year, directory or golden_json_dir(), ', '.join(sorted(by_year))))
    return by_year[year]


##########################################################################################
#####      THE MASK
##########################################################################################
class LumiMask(object):
    '''Union of certification JSONs, {"run": [[first, last], ...]}, queried per
    event. Per run the ranges are merged and sorted once, so a lookup is one
    dict access plus one bisect; the last (run, lumi) answer is cached, since
    consecutive events in a file mostly share their lumi.'''

    def __init__(self, paths):
        self.paths = list(paths)
        if not self.paths:
            raise ValueError('LumiMask needs at least one JSON file')

        raw = {}
        for path in self.paths:
            with open(path) as fin:
                payload = json.load(fin)
            if not isinstance(payload, dict) or not payload:
                raise ValueError('%s is not a non-empty {run: [[first, last], ...]} JSON' % path)
            for run, ranges in payload.items():
                for first, last in ranges:
                    if int(first) > int(last):
                        raise ValueError('%s: run %s has the inverted range [%s, %s]'
                                         % (path, run, first, last))
                    raw.setdefault(int(run), []).append((int(first), int(last)))

        self._starts = {}
        self._ends   = {}
        self.n_lumis = 0
        for run, ranges in raw.items():
            merged = []
            for first, last in sorted(ranges):
                if merged and first <= merged[-1][1] + 1:
                    merged[-1][1] = max(merged[-1][1], last)
                else:
                    merged.append([first, last])
            self._starts[run] = [r[0] for r in merged]
            self._ends[run]   = [r[1] for r in merged]
            self.n_lumis     += sum(r[1] - r[0] + 1 for r in merged)

        self.first_run = min(self._starts)
        self.last_run  = max(self._starts)
        self.sha256    = dict((os.path.basename(p), _sha256(p)) for p in self.paths)

        self.n_pass = 0
        self.n_fail = 0
        self.failed_runs = set()
        self._last_key    = None
        self._last_answer = False

    def contains(self, run, lumi):
        '''True if (run, lumi) is certified. Counts the answer.'''
        key = (run, lumi)
        if key != self._last_key:
            self._last_key    = key
            starts = self._starts.get(run)
            if starts is None:
                self._last_answer = False
            else:
                k = bisect_right(starts, lumi) - 1
                self._last_answer = k >= 0 and lumi <= self._ends[run][k]
        if self._last_answer:
            self.n_pass += 1
        else:
            self.n_fail += 1
            self.failed_runs.add(run)
        return self._last_answer

    def describe(self):
        '''Provenance, for the job log and the --lumi-json payload.'''
        return {'files'    : sorted(self.sha256),
                'sha256'   : self.sha256,
                'runs'     : [self.first_run, self.last_run],
                'n_runs'   : len(self._starts),
                'n_lumis'  : self.n_lumis}

    def summary(self):
        total = self.n_pass + self.n_fail
        text = ('[lumi mask] %d event(s) checked, %d certified, %d rejected (%.2f%%)'
                % (total, self.n_pass, self.n_fail,
                   100. * self.n_fail / total if total else 0.))
        if self.failed_runs:
            runs = sorted(self.failed_runs)
            text += '\n[lumi mask] runs with rejected events (%d): %s%s' % (
                len(runs), ' '.join(str(r) for r in runs[:30]),
                ' ...' if len(runs) > 30 else '')
        if total and not self.n_pass:
            text += ('\n[lumi mask] WARNING: no event of this job is certified. '
                     'Legitimate for a file of uncertified runs; if every job says '
                     'this, the mask does not match the data.')
        return text


def _sha256(path):
    with open(path, 'rb') as fin:
        return hashlib.sha256(fin.read()).hexdigest()


##########################################################################################
#####      FROM THE COMMAND LINE
##########################################################################################
def make_lumi_mask(spec, is_mc):
    '''--lumi-mask spec -> LumiMask or None (see module docstring).'''
    spec = (spec or '').strip()

    if is_mc:
        if spec and spec != 'none':
            raise ValueError('--lumi-mask %r with --mc: a certification mask is a '
                             'data-taking quantity and does not apply to simulation' % spec)
        return None

    if not spec:
        raise ValueError("data requires --lumi-mask: 'golden' (every JSON in %s), a "
                         "directory, a JSON file, or 'none' to run without a mask on "
                         "purpose" % GOLDEN_SUBDIR)
    if spec == 'none':
        print('#### WARNING: --lumi-mask none -- NO certification mask on data, every '
              'lumi is kept')
        return None

    if spec == 'golden':
        paths = sorted(golden_jsons_by_year().values())
    elif os.path.isdir(spec):
        paths = sorted(golden_jsons_by_year(spec).values())
    else:
        paths = [p.strip() for p in spec.split(',') if p.strip()]
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            raise IOError('--lumi-mask: no such file %s' % ', '.join(missing))

    mask = LumiMask(paths)
    info = mask.describe()
    print('#### lumi mask: %d file(s), runs %d-%d, %d certified lumis'
          % (len(info['files']), info['runs'][0], info['runs'][1], info['n_lumis']))
    for name in info['files']:
        print('####   %s  sha256 %s' % (name, info['sha256'][name][:16]))
    return mask
