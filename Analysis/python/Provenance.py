'''
Code provenance of an ntuple job: git commit, branch, uncommitted changes of
the Bmmm repository the job imports its code from.

The SLURM jobs import the python code from the live working tree
($CMSSW_BASE/src/Bmmm), so every job prints this block at start-up: a job
that ran on different code than the one recorded at submission
(submitter_rds_slurm.py writes provenance.json and the full diff) shows a
different commit or diff hash.

    from Bmmm.Analysis.Provenance import provenance, print_provenance
'''

import os
import hashlib
import subprocess


def repo_dir():
    here = os.path.dirname(os.path.realpath(__file__))     # .../Bmmm/Analysis/python
    base = os.environ.get('CMSSW_BASE', '')
    cand = [os.path.join(base, 'src', 'Bmmm')] if base else []
    cand.append(os.path.normpath(os.path.join(here, '..', '..')))
    for c in cand:
        if os.path.isdir(os.path.join(c, '.git')) or os.path.isfile(os.path.join(c, '.git')):
            return c
    return cand[-1]


def _git(args, cwd):
    try:
        return subprocess.check_output(['git'] + args, cwd=cwd, stderr=subprocess.DEVNULL).decode()
    except Exception:
        return None


def provenance(with_diff=False):
    '''dict: repo, commit, branch, describe, dirty (list of 'XY path' from git
    status --porcelain, untracked files left out), diff_sha1 (of `git diff
    HEAD`, tracked files), and the diff itself if with_diff.'''
    d = repo_dir()
    out = {'repo': d}
    out['commit'] = (_git(['rev-parse', 'HEAD'], d) or '?').strip()
    out['branch'] = (_git(['rev-parse', '--abbrev-ref', 'HEAD'], d) or '?').strip()
    out['remote'] = (_git(['config', '--get', 'remote.origin.url'], d) or '?').strip()
    status = _git(['status', '--porcelain', '--untracked-files=no'], d) or ''
    out['dirty'] = [l for l in status.splitlines() if l.strip()]
    diff = _git(['diff', 'HEAD'], d) or ''
    out['diff_sha1'] = hashlib.sha1(diff.encode()).hexdigest() if diff else ''
    out['pushed'] = bool(_git(['branch', '-r', '--contains', out['commit']], d))
    if with_diff:
        out['diff'] = diff
    return out


def print_provenance():
    p = provenance()
    print('#### code: %s  commit %s (%s)%s, pushed: %s'
          % (p['repo'], p['commit'], p['branch'],
             '' if not p['dirty'] else ', UNCOMMITTED changes in %d file(s), diff sha1 %s'
             % (len(p['dirty']), p['diff_sha1']), p['pushed']))
    for l in p['dirty']:
        print('####   %s' % l)
    return p
