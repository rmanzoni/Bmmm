#!/usr/bin/env python3
"""
Aggregate `crab status` across every task in a CRAB workArea.

Usage:
    source /cvmfs/cms.cern.ch/common/crab-setup.sh   # (or .csh) + valid grid proxy
    python3 crab_status_summary.py <workArea> [options]

    ipython -i -- crab_status_summary.py crab_skims_02jul26_data2024_v2 --workers 16 --exitcodes

    <workArea> is the directory that holds the crab_<request> project subdirs
    (i.e. General.workArea from the submitter). A single project dir also works.

Options:
    --workers N        parallel status calls (default 8; keep modest, be nice to cmsweb)
    --fail-threshold F failed+held fraction to flag a task           (default 0.10)
    --idle-threshold F idle fraction to flag a task                  (default 0.30)
    --exitcodes        also summarise per-job exit codes. NB this forces `--long`
                       status (heavier per-task schedd download) -> slower. Opt-in.
    --json PATH        also dump the full structured result as JSON
    --dump-first       print the raw (long) status dict of the first task and exit
                       -- use this once to eyeball the schema on your client version
    --sequential       disable the process pool (debugging)

Uses the CRAB python API (crabCommand), not stdout parsing. Schema relied upon
(confirmed against dmwm/CRABClient source):
    res['status']        -> combined/scheduler task status string
    res['jobsPerStatus'] -> {state: count}     (clean for FileBased; probes pollute
                                                 this under Automatic splitting)
    res['publication']   -> {published, publication_failed, not_published, publishing}
                            The 'PUBLICATION vs DONE' section compares each task's
                            #finished jobs against its 'published' count and lists
                            every task where they differ (see pub_mismatch()).
    per-job exit codes   -> live in a {jobid: {'State':.., 'Error':[ec,msg]}} structure
                            that is ONLY populated with long=True. The KEY holding it
                            varies by client version, so extract_exitcodes() scans for
                            it generically rather than assuming a name; if absent it
                            reports 'unavailable' instead of guessing.
Everything is read defensively with .get(); missing keys degrade, they don't crash.
"""
from __future__ import division
import os
import sys
import json
import argparse
import contextlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import CRABClient  # noqa: F401  -- adds WMCore/DBS to PYTHONPATH; must precede the import below
from CRABAPI.RawCommand import crabCommand

try:
    from http.client import HTTPException
except ImportError:                      # py2 fallback, harmless
    from httplib import HTTPException     # noqa


# short labels for the exit codes you actually hit; everything else prints as the bare
# number. Full list: https://twiki.cern.ch/twiki/bin/view/CMSPublic/JobExitCodes
EXITCODE_LABELS = {
    90000: 'postprocessing (stage-out/publish)',
    50660: 'RAM kill (too much memory)',
    50664: 'wall-clock kill',
    50513: 'wrapper/script failure',
    60302: 'input file not found',
    60307: 'stage-out to SE failed',
    60317: 'stage-out timeout',
    60318: 'stage-out / FJR error',
    10040: 'cmsRun cfg generation failed',
    8021:  'cmsRun fatal exception',
    8028:  'cmsRun file-read exception',
    139:   'segfault',
    -1:    'invalid/missing framework job report',
}


# ----------------------------------------------------------------------------------------
# low-level helpers
# ----------------------------------------------------------------------------------------
@contextlib.contextmanager
def _silence():
    """Silence crab's noisy console output at the fd level (keeps the table clean)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_out, old_err = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(devnull)
        os.close(old_out)
        os.close(old_err)


def _extract_pub(pub):
    """Pull the four publication counters out of whatever shape 'publication' has."""
    if not isinstance(pub, dict):
        return {}
    inner = pub.get('status') if isinstance(pub.get('status'), dict) else pub
    keys = ('published', 'publication_failed', 'not_published', 'publishing')
    return {k: int(inner[k]) for k in keys
            if isinstance(inner.get(k), (int, float))}


def _find_perjob(res):
    """
    Locate the per-job structure {jobid: {'State':.., 'Error':[ec,msg], ...}} in the
    status return dict, without assuming its key name (varies across client versions).
    Returns the dict or None.
    """
    for key in ('jobs', 'jobList', 'jobsInfo'):          # 'jobs' is the usual one
        v = res.get(key)
        if isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values()):
            return v
    for v in res.values():                               # last resort: sniff for it
        if isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values()):
            sample = next(iter(v.values()))
            if any(k in sample for k in ('State', 'Error', 'exitcode')):
                return v
    return None


def extract_exitcodes(res):
    """
    Best-effort per-job exit-code tally -> {code: count}, or None if the per-job
    structure isn't exposed (i.e. status wasn't run long, or the key is unknown).
    Skips finished jobs: their stored code can be stale from a prior attempt.
    """
    per_job = _find_perjob(res)
    if per_job is None:
        return None
    tally = defaultdict(int)
    for info in per_job.values():
        if str(info.get('State', '')).lower() == 'finished':
            continue
        ec = None
        err = info.get('Error')
        if isinstance(err, (list, tuple)) and err:
            ec = err[0]
        elif 'exitcode' in info:
            ec = info.get('exitcode')
        if ec in (None, 0, '0', ''):
            continue
        try:
            ec = int(ec)
        except (TypeError, ValueError):
            ec = str(ec)
        tally[ec] += 1
    return dict(tally)


def status_one(projdir, silence=True, keep_raw=False, long_=False):
    """Run crab status on a single project dir, return a parsed record (never raises)."""
    rec = {
        'name': os.path.basename(projdir.rstrip('/')),
        'projdir': projdir,
        'error': None,
        'sched_status': None,
        'db_status': None,
        'jobs': {},
        'total': 0,
        'publication': {},
        'exitcodes': None,          # None = not requested / not exposed; {} = none found
        'failure_msg': None,
        'raw': None,
    }
    try:
        if silence:
            with _silence():
                res = crabCommand('status', dir=projdir, long=long_)
        else:
            res = crabCommand('status', dir=projdir, long=long_)
    except HTTPException as exc:
        rec['error'] = 'HTTPException: %s' % exc
        return rec
    except Exception as exc:                       # proxy expired, corrupt cache, etc.
        rec['error'] = '%s: %s' % (type(exc).__name__, exc)
        return rec

    if keep_raw:
        rec['raw'] = res
    rec['sched_status'] = res.get('status') or res.get('commandStatus')
    rec['db_status'] = res.get('dbStatus')
    rec['failure_msg'] = res.get('statusFailureMsg') or res.get('taskFailureMsg') or None

    jps = res.get('jobsPerStatus') or {}
    if not jps and isinstance(res.get('jobList'), list):   # reconstruct from [state, id]
        counter = defaultdict(int)
        for item in res['jobList']:
            try:
                counter[item[0]] += 1
            except (TypeError, IndexError):
                pass
        jps = dict(counter)
    rec['jobs'] = {k: int(v) for k, v in jps.items()}
    rec['total'] = sum(rec['jobs'].values())
    rec['publication'] = _extract_pub(res.get('publication'))
    if long_:
        rec['exitcodes'] = extract_exitcodes(res)
    return rec


# ----------------------------------------------------------------------------------------
# interpretation
# ----------------------------------------------------------------------------------------
def buckets(jobs):
    """Collapse the many job states into display buckets; 'other' catches the rest."""
    done  = jobs.get('finished', 0)
    run   = jobs.get('running', 0) + jobs.get('transferring', 0)
    idle  = jobs.get('idle', 0)
    fail  = jobs.get('failed', 0) + jobs.get('held', 0)
    known = done + jobs.get('running', 0) + jobs.get('transferring', 0) + idle \
            + jobs.get('failed', 0) + jobs.get('held', 0)
    other = sum(jobs.values()) - known
    return done, run, idle, fail, other


def pub_mismatch(rec):
    """
    Compare 'done' jobs against 'published' jobs for one task.

    Returns (finished, published, present, is_mismatch):
      finished    -> jobs in the finished/'done' state
      published   -> jobs whose output publication is 'done'
      present     -> the task actually reports publication counters. False means
                     publication is disabled or hasn't been reported yet; such a
                     task is NEVER counted as a mismatch (avoids flagging every
                     non-publishing task).
      is_mismatch -> present and finished != published

    Caveat: 'published' is really a count of published output *files*. It equals the
    number of finished jobs only when each job produces exactly one publishable
    output -- true for the single-TTree ntuple tasks here. A task with N output
    datasets would publish ~N files per job, so divide before comparing there.
    """
    finished  = rec['jobs'].get('finished', 0)
    pub       = rec['publication']
    published = pub.get('published', 0)
    present   = bool(pub)
    return finished, published, present, (present and finished != published)


def flags_for(rec, fail_thr, idle_thr):
    """Classify a task into action flags. Empty list == nothing to do."""
    if rec['error']:
        return ['ERROR']
    fl = []
    st = (rec['sched_status'] or '').upper()
    if 'FAIL' in st:
        fl.append('RECOVER')                       # dead DAG: resubmit won't work
    total = rec['total']
    jobs = rec['jobs']
    if total:
        if (jobs.get('failed', 0) + jobs.get('held', 0)) / total > fail_thr:
            fl.append('FAILS>%d%%' % int(fail_thr * 100))
        if jobs.get('idle', 0) / total > idle_thr:
            fl.append('IDLE>%d%%' % int(idle_thr * 100))
    pub = rec['publication']
    if pub.get('publication_failed', 0) > 0:
        fl.append('PUB-FAIL')
    elif 'COMPLET' in st and pub.get('not_published', 0) > 0:
        fl.append('PUB-LAG')
    # general 'done != published' check, independent of the reason above. Only
    # added when the more specific flags didn't already fire, to keep the column
    # readable; the dedicated report section below lists every mismatch regardless.
    _, _, _, mm = pub_mismatch(rec)
    if mm and 'PUB-FAIL' not in fl and 'PUB-LAG' not in fl:
        fl.append('PUB!=DONE')
    return fl


def severity(rec):
    """Sort key for the attention list: hard problems first, then by failed fraction."""
    if rec['error']:
        return (0, 0.0)
    st = (rec['sched_status'] or '').upper()
    if 'FAIL' in st:
        return (1, 0.0)
    total = rec['total'] or 1
    frac = (rec['jobs'].get('failed', 0) + rec['jobs'].get('held', 0)) / total
    return (2, -frac)


def ec_label(ec):
    return EXITCODE_LABELS.get(ec, '') if isinstance(ec, int) else ''


# ----------------------------------------------------------------------------------------
# discovery / io
# ----------------------------------------------------------------------------------------
def discover(workdir):
    """Return the list of crab project dirs under workdir (or workdir itself)."""
    if os.path.exists(os.path.join(workdir, '.requestcache')):
        return [os.path.abspath(workdir)]
    out = []
    for name in sorted(os.listdir(workdir)):
        p = os.path.join(workdir, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, '.requestcache')):
            out.append(os.path.abspath(p))
    return out


def common_prefix(names):
    """Longest common prefix, trimmed at the last underscore, for compact display."""
    if not names:
        return ''
    pre = os.path.commonprefix(names)
    return pre[:pre.rfind('_') + 1] if '_' in pre else ''


def _short_name(rec, prefix):
    """Task name with the 'crab_' and the common campaign prefix stripped for display."""
    short = rec['name'][len('crab_'):] if rec['name'].startswith('crab_') else rec['name']
    if prefix and short.startswith(prefix):
        short = short[len(prefix):]
    return short


def print_table(records, prefix):
    hdr = ('task', 'sched', 'tot', 'done', 'publ', 'run', 'idle', 'fail', 'oth', '%done', 'flags')
    widths = (46, 11, 5, 5, 5, 4, 5, 5, 4, 6, 24)
    line = '  '.join('%-*s' % (w, h) for w, h in zip(widths, hdr))
    print(line)
    print('-' * len(line))
    for r in records:
        short = r['name'][len('crab_'):] if r['name'].startswith('crab_') else r['name']
        if prefix and short.startswith(prefix):
            short = short[len(prefix):]
        short = short[:widths[0]]
        if r['error']:
            print('%-*s  %-11s  %s' % (widths[0], short, 'ERROR', r['error'][:60]))
            continue
        done, run, idle, fail, other = buckets(r['jobs'])
        publ = r['publication'].get('published', 0) if r['publication'] else '-'
        pct = ('%.0f' % (100.0 * done / r['total'])) if r['total'] else '-'
        flags = ','.join(r['flags'])
        vals = (short, (r['sched_status'] or '-')[:11], r['total'],
                done, publ, run, idle, fail, other, pct, flags)
        print('  '.join('%-*s' % (w, v) for w, v in zip(widths, vals)))


def main():
    ap = argparse.ArgumentParser(description='Aggregate crab status over a workArea.')
    ap.add_argument('workdir', help='CRAB workArea (holds crab_<request> subdirs)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--fail-threshold', type=float, default=0.10)
    ap.add_argument('--idle-threshold', type=float, default=0.30)
    ap.add_argument('--exitcodes', action='store_true',
                    help='summarise per-job exit codes (forces --long status; slower)')
    ap.add_argument('--json', dest='json_path', default=None)
    ap.add_argument('--dump-first', action='store_true')
    ap.add_argument('--sequential', action='store_true')
    args = ap.parse_args()

    projdirs = discover(args.workdir)
    if not projdirs:
        sys.exit('No CRAB project dirs (with .requestcache) found under %s' % args.workdir)

    if args.dump_first:
        import pprint
        rec = status_one(projdirs[0], silence=False, keep_raw=True, long_=True)
        print('\n=== raw (long) status dict for %s ===' % rec['name'])
        print('top-level keys:', sorted(rec['raw'].keys()) if rec['raw'] else None)
        pprint.pprint(rec['raw'])
        if rec['error']:
            print('error:', rec['error'])
        return

    long_ = args.exitcodes
    print('Querying %d tasks (%s%s)...\n'
          % (len(projdirs),
             'sequential' if args.sequential else '%d workers' % args.workers,
             ', long/exitcodes' if long_ else ''))

    records = []
    if args.sequential:
        for p in projdirs:
            records.append(status_one(p, long_=long_))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(status_one, p, True, False, long_): p for p in projdirs}
            for fut in as_completed(futs):
                records.append(fut.result())

    for r in records:
        r['flags'] = flags_for(r, args.fail_threshold, args.idle_threshold)
    records.sort(key=lambda r: r['name'])

    prefix = common_prefix([r['name'][len('crab_'):] if r['name'].startswith('crab_')
                            else r['name'] for r in records])
    print_table(records, prefix)

    # ---- campaign totals -----------------------------------------------------------
    tot = defaultdict(int)
    pub = defaultdict(int)
    for r in records:
        for k, v in r['jobs'].items():
            tot[k] += v
        for k, v in r['publication'].items():
            pub[k] += v
    grand = sum(tot.values())
    print('\nCAMPAIGN TOTALS  (%d tasks, %d jobs)' % (len(records), grand))
    if grand:
        for state in sorted(tot, key=lambda k: -tot[k]):
            print('  %-14s %7d  (%4.1f%%)' % (state, tot[state], 100.0 * tot[state] / grand))
    if pub:
        print('  publication: ' + ', '.join('%s=%d' % (k, pub[k]) for k in sorted(pub)))

    # ---- publication vs done mismatch ----------------------------------------------
    # tasks where #jobs in 'done'/finished != #jobs whose publication is 'done'.
    mism, no_pub = [], []
    for r in records:
        if r['error']:
            continue
        finished, published, present, is_mm = pub_mismatch(r)
        if not present:
            if finished:
                no_pub.append(r)
            continue
        if is_mm:
            mism.append((_short_name(r, prefix)[:46], finished, published, r))

    print('\nPUBLICATION vs DONE  (tasks where #done jobs != #published jobs)')
    if not mism:
        print('  none: every task reporting publication has published == done.')
    else:
        mism.sort(key=lambda t: (t[1] - t[2]), reverse=True)   # biggest deficit first
        namew = max(len(nm) for nm, _, _, _ in mism)
        print('  %-*s  %6s  %6s  %6s   %-11s %s'
              % (namew, 'task', 'done', 'publ', 'diff', 'sched', 'where the rest sit'))
        for nm, finished, published, r in mism:
            rest = ['%s=%d' % (k, r['publication'][k])
                    for k in ('publishing', 'not_published', 'publication_failed')
                    if r['publication'].get(k, 0)]
            print('  %-*s  %6d  %6d  %+6d   %-11s %s'
                  % (namew, nm, finished, published, published - finished,
                     (r['sched_status'] or '-')[:11], ', '.join(rest) or '-'))
        print('  %d task(s) with a done/published mismatch.' % len(mism))
        print('  note: tasks still running are expected to show a deficit until the')
        print('        async publisher catches up -- read the sched column to tell')
        print('        those apart from COMPLETED tasks that genuinely need attention.')
    if no_pub:
        print('  (%d task(s) have finished jobs but report no publication counters '
              '-- publication off or not yet started)' % len(no_pub))

    # ---- exit-code summary ---------------------------------------------------------
    if long_:
        codes = defaultdict(int)
        n_unavailable = sum(1 for r in records if r['exitcodes'] is None and not r['error'])
        for r in records:
            if r['exitcodes']:
                for ec, n in r['exitcodes'].items():
                    codes[ec] += n
        print('\nEXIT CODE SUMMARY  (failed/held jobs; finished-job codes skipped as stale)')
        if codes:
            total_ec = sum(codes.values())
            for ec in sorted(codes, key=lambda c: -codes[c]):
                lbl = ec_label(ec)
                lbl = '  %s' % lbl if lbl else ''
                print('  %-8s %6d  (%4.1f%%)%s'
                      % (ec, codes[ec], 100.0 * codes[ec] / total_ec, lbl))
        else:
            print('  no non-zero exit codes found among failed jobs.')
        if n_unavailable:
            print('  NOTE: per-job exit-code structure was not exposed for %d task(s).'
                  % n_unavailable)
            print('        Run with --dump-first to inspect the dict and I can pin the key.')
        print('  codes ref: https://twiki.cern.ch/twiki/bin/view/CMSPublic/JobExitCodes')

    # ---- needs attention -----------------------------------------------------------
    flagged = [r for r in records if r['flags']]
    if not flagged:
        print('\nAll tasks nominal.')
    else:
        flagged.sort(key=severity)
        print('\nNEEDS ATTENTION  (%d tasks)' % len(flagged))
        for r in flagged:
            short = r['name'][len('crab_'):] if r['name'].startswith('crab_') else r['name']
            extra = ''
            if r['error']:
                extra = '  <- %s' % r['error'][:70]
            elif r['failure_msg']:
                extra = '  <- %s' % str(r['failure_msg'])[:70]
            # append the task's dominant exit code, if we have it
            if r['exitcodes']:
                top = max(r['exitcodes'], key=r['exitcodes'].get)
                extra += '  [ec %s x%d]' % (top, r['exitcodes'][top])
            print('  [%-18s] %s%s' % (','.join(r['flags']), short, extra))
        print('\n  legend: RECOVER=dead DAG, needs recovery task (resubmit is a no-op)')
        print('          FAILS/IDLE=fraction over threshold; PUB-FAIL/LAG/!=DONE=publish')

    if args.json_path:
        for r in records:                          # raw is not JSON-friendly / not kept
            r.pop('raw', None)
        with open(args.json_path, 'w') as fout:
            json.dump({'tasks': records,
                       'totals': dict(tot),
                       'publication': dict(pub),
                       'pub_mismatch': [
                           {'name': r['name'],
                            'done': r['jobs'].get('finished', 0),
                            'published': r['publication'].get('published', 0),
                            'diff': (r['publication'].get('published', 0)
                                     - r['jobs'].get('finished', 0)),
                            'sched_status': r['sched_status']}
                           for r in records if pub_mismatch(r)[3]]},
                      fout, indent=2)
        print('\nwrote %s' % args.json_path)


if __name__ == '__main__':
    main()