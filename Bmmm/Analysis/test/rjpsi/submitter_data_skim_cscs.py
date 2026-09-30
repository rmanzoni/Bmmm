'''
Submitter for the SLURM system (PSI Tier-3).

Direct submission: the FWLite ntuplizer runs natively under the current
CMSSW release. No el7 apptainer/singularity wrapper anymore -- the recent
CMSSW (el9) matches the worker-node OS, so jobs run directly on the node.

Run this submitter from a shell where you've already done `cmsenv` in the
CMSSW_X_Y_Z/src you want the jobs to use: the release path is read from
$CMSSW_BASE and baked into each job script, and `scram runtime` is sourced
from there (NOT from the output dir, which need not live inside the release).

----------------------------------------------------------------------------
CSCS INPUT (REVISED)
----------------------------------------------------------------------------
The skim files now live at T2_CH_CSCS (published on DAS) and are read
REMOTELY via xrootd -- there is no PSI-local copy of the inputs. Only the
*input* source changed:

  * file lists hold bare LFNs (/store/user/manzoni/...), produced by
    make_filelists_cscs.py (one list per year);
  * at submit time each LFN becomes  root://<input_redirector>//store/...  ;
  * OUTPUT is unchanged -- merged chunks are still xrdcp'd to PSI dCache
    (se_host / pnfs_base below), and auto_chunk_offset still lists the PSI
    output dir to avoid overwriting existing chunks.

Run this submitter once per year (set `year` below); each year lands in its
own out_dir.

----------------------------------------------------------------------------
DIFF SUBMISSION (kept for reuse; OFF by default here)
----------------------------------------------------------------------------
Set `diff_mode = True` to submit ONLY the files that are new in
`addendum_file` relative to `base_file` (i.e. the set difference
addendum \\ base). `addendum_file` is assumed to be a superset of
`base_file` (it contains base + the new files), so the diff is exactly the
files that still need processing.

When diff_mode = False the submitter behaves as before and runs over
`input_file`.

NO OVERWRITE -- chunk numbering: by default (`auto_chunk_offset = True`) the
submitter lists the chunk files already present in the pnfs out_dir, finds the
highest existing index, and starts the new submission at last + 1. So the diff
jobs land in the SAME pnfs directory as the base run without clobbering it.
The listing is done with `xrdfs <se_host> ls`, falling back to the local pnfs
mount. Set `auto_chunk_offset = False` to use the fixed `chunk_offset` instead.
The offset also takes into account the job scripts already in the LOCAL
out_dir (submitter_chunkN.sh), so a chunk whose jobs all failed (no file on
pnfs) can never have its script -- i.e. its list of input files -- overwritten
by a later submission. If the pnfs dir cannot be listed the submitter aborts
instead of silently starting again from chunk 0.

----------------------------------------------------------------------------
RESUBMISSION OF FAILED CHUNKS:  --resubmit
----------------------------------------------------------------------------
    python3 submitter_data_skim_cscs.py --year 2022 --resubmit --dry-run
    python3 submitter_data_skim_cscs.py --year 2022 --resubmit

The job scripts already written in the local out_dir (submitter_chunkN.sh)
are the record of what was submitted: each one holds the exact input files
of chunk N, whatever offset or diff round produced it. Resubmission never
rebuilds chunks from the file lists; it re-sends the existing scripts.

A chunk is resubmitted only if ALL of these hold:
  * a job script submitter_chunkN.sh exists in the local out_dir;
  * no merged output  rjpsi_chunkN.root  exists on pnfs;
  * no SLURM job named  N_<out_dir>  is still pending or running.
Chunks whose output exists on pnfs with size 0 are NOT resubmitted (xrdcp
would refuse to overwrite them); they are listed with the command to remove
them, after which a second --resubmit pass picks them up.

Before submitting, the submitter
  * aborts if the pnfs dir or the SLURM queue cannot be read (never guesses);
  * aborts if the grid proxy is missing, in /tmp, or too short-lived;
  * prints how many chunks are finished / queued / missing, and why each
    missing chunk failed (read from logs/ and errs/); a full per-chunk report,
    including the input files that failed, goes to
    <out_dir>/resubmit_report_<timestamp>.txt;
  * asks for confirmation (skip with --yes).
Old logs are kept: logs/chunkN.log -> logs/chunkN.log.attemptK (same for errs).
The ntuplizer config is NOT re-copied: resubmitted jobs run the
inspector_rjpsi.py already in the out_dir, like the original jobs.
--redirector HOST switches the input xrootd door in the scripts of the chunks
being resubmitted (original kept as submitter_chunkN.sh.orig); the input
files themselves are checked to be unchanged.
--out-dir DIR resubmits any existing production dir, not only this year's.
'''

import os
import re
import sys
import shutil
import random
import argparse
import datetime
import subprocess
from glob import glob

# ---------------------------------------------------------------------------
# INPUT SOURCE (CSCS, read via xrootd)  --  NEW
# ---------------------------------------------------------------------------
# Pick ONE redirector. Each bare LFN in the file list is turned into
#     root://<input_redirector>//store/user/manzoni/...
# at submit time. Default is the CSCS site door (most direct: these are
# single-replica USER datasets that live only at CSCS, so the EU redirector
# would just delegate to CSCS anyway). If you hit open timeouts / door
# flakiness at scale, switch to the EU redirector (AAA retry/backoff, still
# routes to CSCS). Switching = uncommenting one line.
#
#   VERIFY the CSCS door host before a big submission, e.g.:
#     xrdfs cms03.lcg.cscs.ch:1094 ls /store/user/manzoni
#     dasgoclient -query="site dataset=<one ds> instance=prod/phys03"
#
# input_redirector = 'xrootd-cms.infn.it'        # EU regional redirector (routes to CSCS)
input_redirector = 'cms03.lcg.cscs.ch:1094'      # CSCS site door (direct)  <-- default
# input_redirector = 'cms-xrd-global.cern.ch'    # global redirector (last resort)

# ---------------------------------------------------------------------------
# per-year selection: run this submitter once per year (separate out_dirs)
# ---------------------------------------------------------------------------
year = '2022' # '2024' or '2025'

input_files_by_year = {
    '2022': '../files/files_data2022_cscs_24sep26.txt',
    '2023': '../files/files_data2023_cscs_24sep26.txt',
    '2024': '../files/files_data2024_cscs_24sep26.txt',
    '2025': '../files/files_data2025_cscs_24sep26.txt',
    '2026': '../files/files_data2026_cscs_24sep26.txt',
}

addendum_files_by_year = {
    '2022': '',
    '2023': '',
    '2024': '',
    '2025': '',
    '2026': '',
}

out_dir_by_year = {
    '2022': 'RJpsi_23Jun2026_data2022_cscs_24sep26_v1',
    '2023': 'RJpsi_23Jun2026_data2023_cscs_24sep26_v1',
    '2024': 'RJpsi_23Jun2026_data2024_cscs_24sep26_v1',
    '2025': 'RJpsi_23Jun2026_data2025_cscs_24sep26_v1',
    '2026': 'RJpsi_23Jun2026_data2026_cscs_24sep26_v1',
}

# ---------------------------------------------------------------------------
# command line (all optional: without arguments the submitter behaves as before)
# ---------------------------------------------------------------------------
queues = {'short': 60, 'standard': 720, 'long': 10080}  # SLURM partition -> --time [min]

parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip())
parser.add_argument('--year', default=year, choices=sorted(out_dir_by_year),
                    help='year to process (default: %(default)s, set in the file)')
parser.add_argument('--queue', default='standard', choices=sorted(queues),
                    help='SLURM partition (default: %(default)s)')
parser.add_argument('--min-proxy-hours', type=float, default=12.,
                    help='abort if the grid proxy has less than this many hours left (default: %(default)s)')
parser.add_argument('--resubmit', action='store_true',
                    help='resubmit the chunks of an existing production whose output is missing on pnfs')
parser.add_argument('--dry-run', action='store_true',
                    help='with --resubmit: run all checks and write the report, submit nothing')
parser.add_argument('--yes', action='store_true',
                    help='with --resubmit: do not ask for confirmation')
parser.add_argument('--out-dir', default=None,
                    help='with --resubmit: production dir to resubmit (default: the one of --year)')
parser.add_argument('--redirector', default=None,
                    help='with --resubmit: rewrite the input xrootd door (e.g. xrootd-cms.infn.it) '
                         'in the scripts of the resubmitted chunks')
args = parser.parse_args()

if not args.resubmit:
    for opt in ('dry_run', 'yes', 'out_dir', 'redirector'):
        if getattr(args, opt):
            parser.error('--%s only makes sense together with --resubmit' % opt.replace('_', '-'))

year  = args.year
queue = args.queue; time = queues[queue]

out_dir = args.out_dir.rstrip('/') if args.out_dir else out_dir_by_year[year]
# out_dir = 'RJpsi_10Jun2026_notrig_Hb_inclusive_v1'

out_file_name = 'rjpsi'

cfg = 'inspector_rjpsi.py'

# ---------------------------------------------------------------------------
# diff submission config (kept for reuse; OFF for a fresh CSCS submission)
# ---------------------------------------------------------------------------
# When True: submit only (addendum \ base). When False: submit input_file.
diff_mode = False

input_file    = input_files_by_year[year]         # used when diff_mode = False
# For a real diff run, point base_file / addendum_file at the proper lists
# (base = already-processed, addendum = base + new). They default to the
# per-year list here only so the names are always defined.
base_file     = input_files_by_year[year]         # already-processed list
addendum_file = addendum_files_by_year[year]         # superset: base + new files

# Chunk numbering / no-overwrite behaviour.
#   auto_chunk_offset = True : list the pnfs out_dir, start at (highest existing
#                              chunk index) + 1, so a diff/resubmission lands in
#                              the SAME pnfs dir without overwriting anything.
#   auto_chunk_offset = False: use the fixed `chunk_offset` below.
auto_chunk_offset = True
chunk_offset      = 0          # manual offset, used only when auto_chunk_offset = False

# storage element for the OUTPUT (PSI dCache) -- UNCHANGED.
# NOTE: se_host / pnfs_base are the *output* side (where merged chunks land and
# where auto_chunk_offset lists existing chunks). They are NOT the input door;
# the input door is `input_redirector` above.
se_host   = 't3dcachedb03.psi.ch:1094'
pnfs_base = '/pnfs/psi.ch/cms/trivcat/store/user/manzoni'

old_files = []
files = []


def _read_list(path):
    '''Read a file list, stripping whitespace and dropping blank lines.'''
    with open(path) as f:
        return [line.strip() for line in f.read().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# helpers shared by the fresh submission and --resubmit
# ---------------------------------------------------------------------------
def die(msg):
    '''Fail loudly: print and exit non-zero. Never continue on a guess.'''
    print('>>>> FATAL: %s' % msg)
    sys.exit(1)


def compact(indices):
    '''[1,2,3,7,9,10] -> "1-3, 7, 9-10" (for readable printouts).'''
    indices = sorted(indices)
    out, i = [], 0
    while i < len(indices):
        j = i
        while j + 1 < len(indices) and indices[j + 1] == indices[j] + 1:
            j += 1
        out.append(str(indices[i]) if i == j else '%d-%d' % (indices[i], indices[j]))
        i = j + 1
    return ', '.join(out)


def list_pnfs_chunks(se_host, se_path, out_file_name):
    '''
    Strict listing of the merged chunk files '<out_file_name>_chunk<N>.root'
    (not the part files) in the pnfs output dir.

    Returns ({chunk index: size in bytes}, how-it-was-listed).
    Returns an empty dict ONLY if the directory genuinely does not exist.
    Any other failure aborts: an empty listing caused by an error would look
    like "nothing done yet" and trigger a mass resubmission, or make a new
    submission restart at chunk 0.
    Uses `xrdfs <se_host> ls -l`, falling back to the local pnfs mount
    (available on the T3 user interfaces, not on the worker nodes).
    '''
    pat = re.compile(r'%s_chunk(\d+)\.root$' % re.escape(out_file_name))
    chunks = {}
    try:
        res = subprocess.run(
            ['xrdfs', se_host, 'ls', '-l', se_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=600,
        )
        if res.returncode == 0:
            # line format: <flags> <date> <time> <size> <path>
            for line in res.stdout.splitlines():
                fields = line.split()
                if len(fields) < 2:
                    continue
                m = pat.search(os.path.basename(fields[-1]))
                if not m:
                    continue
                try:
                    chunks[int(m.group(1))] = int(fields[-2])
                except ValueError:
                    die('cannot read the file size in this `xrdfs ls -l` line: %r' % line)
            return chunks, 'xrdfs'
        print('>>>> xrdfs ls returned %d (%s); trying the local pnfs mount'
              % (res.returncode, res.stderr.strip()))
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print('>>>> xrdfs unavailable (%s); trying the local pnfs mount' % exc)

    if os.path.isdir(se_path):
        with os.scandir(se_path) as entries:
            for entry in entries:
                m = pat.search(entry.name)
                if m:
                    chunks[int(m.group(1))] = entry.stat().st_size
        return chunks, 'local pnfs mount'
    if os.path.isdir(os.path.dirname(se_path)):
        return {}, 'local pnfs mount (directory does not exist yet)'
    die('cannot list %s: xrdfs failed and the pnfs mount is not visible from '
        'this machine. Run from a T3 user interface.' % se_path)


def local_chunk_scripts(out_dir):
    '''{chunk index: path} of the job scripts submitter_chunkN.sh in out_dir.'''
    pat = re.compile(r'submitter_chunk(\d+)\.sh$')
    scripts = {}
    for path in glob(os.path.join(out_dir, 'submitter_chunk*.sh')):
        m = pat.search(os.path.basename(path))
        if m:
            scripts[int(m.group(1))] = path
    return scripts


def queued_chunks(out_dir):
    '''Chunk indices with a SLURM job (any state) named "<N>_<out_dir>".'''
    user = os.environ.get('USER', '')
    if not user:
        die('$USER is not set, cannot query the SLURM queue')
    try:
        res = subprocess.run(
            ['squeue', '-h', '-u', user, '-o', '%.500j'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=120,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        die('cannot query the SLURM queue (%s)' % exc)
    if res.returncode != 0:
        die('squeue returned %d (%s)' % (res.returncode, res.stderr.strip()))
    pat = re.compile(r'^(\d+)_%s$' % re.escape(out_dir))
    return {int(m.group(1)) for m in (pat.match(l.strip()) for l in res.stdout.splitlines()) if m}


def check_proxy(min_hours):
    '''
    The job scripts copy $X509_USER_PROXY on the worker node at run time, so
    the variable must be set here (sbatch exports the environment), point to
    a file the worker nodes can see, and stay valid while jobs wait and run.
    '''
    proxy = os.environ.get('X509_USER_PROXY', '')
    if not proxy:
        die('X509_USER_PROXY is not set. Create the proxy in a shared location, e.g.\n'
            '       voms-proxy-init -voms cms -valid 192:00 -out ~/.x509up_u$(id -u)\n'
            '       export X509_USER_PROXY=~/.x509up_u$(id -u)')
    if not os.path.isfile(proxy):
        die('X509_USER_PROXY=%s does not exist' % proxy)
    if os.path.realpath(proxy).startswith('/tmp/'):
        die('X509_USER_PROXY=%s is in /tmp, which is local to this machine: '
            'the worker nodes cannot copy it. Put the proxy in your home or /work.' % proxy)
    left = []
    for opt in ('-timeleft', '-actimeleft'):   # proxy itself, and its VOMS extension
        try:
            res = subprocess.run(['voms-proxy-info', '-file', proxy, opt],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 universal_newlines=True, timeout=60)
            left.append(int(res.stdout.strip()))
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError) as exc:
            die('cannot read the proxy lifetime with `voms-proxy-info %s` (%s)' % (opt, exc))
    hours = min(left) / 3600.
    if hours < min_hours:
        die('grid proxy has %.1f h left (< %.1f h, --min-proxy-hours). Renew it with\n'
            '       voms-proxy-init -voms cms -valid 192:00 -out %s' % (hours, min_hours, proxy))
    print('>>>> grid proxy OK: %s, %.1f h left' % (proxy, hours))


def sbatch_command(out_dir, jobid):
    '''The sbatch command line of one chunk (identical for submission and resubmission).'''
    return [
        'sbatch',
        '-p', queue,
        '--account=t3',
        '-o', '%s/logs/chunk%d.log' % (out_dir, jobid),
        '-e', '%s/errs/chunk%d.err' % (out_dir, jobid),
        '--job-name=%d_%s' % (jobid, out_dir),
        '--time=%d' % time,
        '--nodes=1', '--ntasks=1', '--nodelist=t3wn[80-91]',
        # '-w', 't3wn70,t3wn71,t3wn72,t3wn73', # only the best nodes
        '%s/submitter_chunk%d.sh' % (out_dir, jobid),
    ]


def submit_chunk(out_dir, jobid):
    '''sbatch one chunk; return True on success. No shell involved.'''
    cmd = sbatch_command(out_dir, jobid)
    print(' '.join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    if res.returncode == 0:
        print('       %s' % res.stdout.strip())
        return True
    print('       sbatch FAILED (%d): %s' % (res.returncode, res.stderr.strip()))
    return False


def archive_logs(out_dir, jobid):
    '''logs/chunkN.log -> logs/chunkN.log.attemptK (first free K); same for errs.'''
    for sub, ext in (('logs', 'log'), ('errs', 'err')):
        src = os.path.join(out_dir, sub, 'chunk%d.%s' % (jobid, ext))
        if not os.path.exists(src):
            continue
        k = 1
        while os.path.exists('%s.attempt%d' % (src, k)):
            k += 1
        os.rename(src, '%s.attempt%d' % (src, k))


def _read_text(path):
    try:
        with open(path, errors='replace') as f:
            return f.read()
    except (FileNotFoundError, IsADirectoryError):
        return None


def diagnose_chunk(out_dir, jobid):
    '''
    Why did chunk N not produce its output? Read from the messages the job
    script itself prints (logs/) and from SLURM's messages (errs/).
    Returns (reason, [input files the ntuplizer failed on]).
    '''
    log = _read_text(os.path.join(out_dir, 'logs', 'chunk%d.log' % jobid))
    err = _read_text(os.path.join(out_dir, 'errs', 'chunk%d.err' % jobid))
    if log is None and err is None:
        return 'no log at all (job never started, or its sbatch failed)', []
    log, err = log or '', err or ''
    failed_inputs = re.findall(r'>>>> FAILED: part\d+ of chunk\d+ \((\S+)\)', log)
    if 'FATAL: cmsenv' in log:
        return 'CMSSW environment did not load on the worker node', failed_inputs
    if 'DUE TO TIME LIMIT' in err:
        return 'killed by SLURM: time limit (%d min)' % time, failed_inputs
    if re.search(r'oom[-_ ]kill|out of memory', err, re.IGNORECASE):
        return 'killed by SLURM: out of memory', failed_inputs
    if 'CANCELLED' in err:
        return 'cancelled (scancel, or node failure)', failed_inputs
    if failed_inputs:
        return 'ntuplizer failed on at least one input file', failed_inputs
    if 'xrdcp FAILED' in log:
        return 'copy of the merged file to pnfs (xrdcp) failed', failed_inputs
    if 'xrdcp succeeded' in log:
        return 'log says the copy succeeded, but the file is NOT on pnfs -- investigate', failed_inputs
    return 'no error message in the logs (killed? node problem?)', failed_inputs


def rewrite_redirector(script, redirector, dry_run):
    '''
    Point the --inputFiles URLs of one job script to another xrootd door.
    Only the host changes: the list of input LFNs is checked to be identical
    before and after. The original script is kept as <script>.orig (first
    rewrite only). Returns True if the script changes.
    '''
    lfns = lambda t: re.findall(r'--inputFiles=root://[^/\s]+/(/store/\S+)', t)
    hosts = lambda t: set(re.findall(r'--inputFiles=root://([^/\s]+)//store/', t))
    txt = _read_text(script)
    new = re.sub(r'root://[^/\s]+//store/', 'root://%s//store/' % redirector, txt)
    if not lfns(txt) or lfns(new) != lfns(txt) or hosts(new) != {redirector}:
        die('could not safely rewrite the redirector in %s' % script)
    if new == txt:
        return False
    if not dry_run:
        if not os.path.exists(script + '.orig'):
            shutil.copy2(script, script + '.orig')
        with open(script, 'w') as f:
            f.write(new)
    return True


def resubmit_missing(out_dir, args):
    '''
    Resubmit the chunks of an existing production whose merged output is
    missing on pnfs. See the RESUBMISSION section of the module docstring.
    Returns the process exit code.
    '''
    se_path = '/'.join([pnfs_base, out_dir])
    if not os.path.isdir(out_dir):
        die('local production dir %s not found in %s -- run from the directory '
            'the original submission was launched from' % (out_dir, os.getcwd()))

    scripts = local_chunk_scripts(out_dir)
    if not scripts:
        die('no job scripts (submitter_chunkN.sh) in %s: nothing was submitted from here' % out_dir)

    print('>>>> reading the pnfs output dir ...')
    on_pnfs, how = list_pnfs_chunks(se_host, se_path, out_file_name)
    print('>>>> reading the SLURM queue ...')
    in_queue = queued_chunks(out_dir)

    # a chunk still in SLURM counts as in flight even if a file is already on
    # pnfs: it may be the copy in progress (a file being written can show size 0)
    submitted = set(scripts)
    queued    = in_queue & submitted
    done      = {i for i, size in on_pnfs.items() if size > 0} - queued
    empty     = ({i for i, size in on_pnfs.items() if size == 0} & submitted) - queued
    orphans   = set(on_pnfs) - submitted
    missing   = sorted(submitted - done - empty - queued)

    print('>>>> diagnosing %d missing chunk(s) from their logs ...' % len(missing))
    reasons = {}
    for jobid in missing:
        reasons[jobid] = diagnose_chunk(out_dir, jobid)

    releases = set()
    for jobid in missing:
        releases |= set(re.findall(r'^cd (\S+)/src$', _read_text(scripts[jobid]), re.MULTILINE))

    # ------------------------------------------------------------- summary
    line = '>>>> ' + '-' * 74
    print(line)
    print('>>>> RESUBMISSION CHECK  %s' % out_dir)
    print('>>>>   local job dir  : %s' % os.path.abspath(out_dir))
    print('>>>>   pnfs output dir: %s  (listed via %s)' % (se_path, how))
    if releases:
        print('>>>>   CMSSW release written in the job scripts: %s' % ', '.join(sorted(releases)))
    if len(releases) > 1:
        print('>>>>   WARNING: the missing chunks use DIFFERENT CMSSW releases')
    print(line)
    print('>>>>   chunks submitted  (job script in local dir)       : %6d' % len(submitted))
    print('>>>>   finished          (merged file on pnfs, size > 0) : %6d' % len(done & submitted))
    print('>>>>   still in SLURM    (pending or running)            : %6d' % len(queued))
    print('>>>>   empty file on pnfs (size 0) -- NOT resubmitted    : %6d' % len(empty))
    print('>>>>   missing on pnfs   -> TO RESUBMIT                  : %6d' % len(missing))
    if orphans:
        print('>>>>   WARNING: %d file(s) on pnfs have no job script here: chunks %s'
              % (len(orphans), compact(orphans)))
    if missing:
        print(line)
        print('>>>>   why the missing chunks failed:')
        by_reason = {}
        for jobid, (reason, _) in reasons.items():
            by_reason.setdefault(reason, []).append(jobid)
        for reason, ids in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            print('>>>>   %6d  %s' % (len(ids), reason))
            print('>>>>           chunks: %s' % compact(ids))
    if empty:
        print(line)
        print('>>>>   size-0 files: remove them, then run --resubmit again:')
        for jobid in sorted(empty):
            print('       xrdfs %s rm %s/%s_chunk%d.root' % (se_host, se_path, out_file_name, jobid))
    print(line)

    # ------------------------------------------------------------- report file
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    report = os.path.join(out_dir, 'resubmit_report_%s%s.txt' % (stamp, '_dryrun' if args.dry_run else ''))
    with open(report, 'w') as f:
        f.write('# resubmission check of %s, %s\n' % (out_dir, stamp))
        f.write('# submitted %d, finished %d, in SLURM %d, size-0 %d, missing %d\n'
                % (len(submitted), len(done & submitted), len(queued), len(empty), len(missing)))
        for jobid in missing:
            reason, failed_inputs = reasons[jobid]
            f.write('chunk %d: %s\n' % (jobid, reason))
            for infile in failed_inputs:
                f.write('    failed input: %s\n' % infile)
        for jobid in sorted(empty):
            f.write('chunk %d: size-0 file on pnfs, not resubmitted\n' % jobid)
        for jobid in sorted(queued):
            f.write('chunk %d: still in SLURM, not resubmitted\n' % jobid)
    print('>>>> per-chunk report: %s' % report)

    if not missing:
        print('>>>> nothing to resubmit')
        return 1 if empty else 0

    # ------------------------------------------------------------- redirector
    if args.redirector:
        n = sum(rewrite_redirector(scripts[j], args.redirector, args.dry_run) for j in missing)
        print('>>>> input door -> %s: %d of %d script(s) %s'
              % (args.redirector, n, len(missing), 'would change' if args.dry_run else 'rewritten'))

    if args.dry_run:
        print('>>>> dry run: nothing submitted')
        return 0

    check_proxy(args.min_proxy_hours)
    print('>>>> queue: %s (%d min)' % (queue, time))

    if not args.yes:
        try:
            answer = input('>>>> resubmit %d chunk(s)? type "yes" to proceed: ' % len(missing))
        except EOFError:
            answer = ''
        if answer.strip() != 'yes':
            print('>>>> aborted, nothing submitted')
            return 1

    failures = []
    for i, jobid in enumerate(missing, 1):
        archive_logs(out_dir, jobid)
        print('[%d/%d] ' % (i, len(missing)), end='')
        if not submit_chunk(out_dir, jobid):
            failures.append(jobid)

    print(line)
    print('>>>> resubmitted %d of %d chunk(s)' % (len(missing) - len(failures), len(missing)))
    if failures:
        print('>>>> sbatch FAILED for chunks %s -- run --resubmit again' % compact(failures))
        return 1
    return 1 if empty else 0


# ---------------------------------------------------------------------------
# --resubmit: works only from what is on disk; the file lists are not used.
# ---------------------------------------------------------------------------
if args.resubmit:
    sys.exit(resubmit_missing(out_dir, args))


if diff_mode:
    base     = set(_read_list(base_file))
    addendum = _read_list(addendum_file)

    # preserve addendum order; keep only files not already in base (or old_files)
    raw_files = [ifile for ifile in addendum
                 if ifile not in base and ifile not in old_files]

    print('>>>> diff mode')
    print('>>>>   base     (%s): %d files' % (base_file, len(base)))
    print('>>>>   addendum (%s): %d files' % (addendum_file, len(addendum)))
    print('>>>>   new to submit: %d files' % len(raw_files))
    if auto_chunk_offset:
        print('>>>>   auto_chunk_offset ON: new chunks continue after the '
              'highest index already on pnfs (no overwrite).')
    elif chunk_offset == 0:
        print('>>>>   WARNING: auto_chunk_offset OFF and chunk_offset = 0 -- if '
              'this out_dir already holds base output, those chunkN.root files '
              'WILL be overwritten.')
else:
    raw_files = [ifile for ifile in _read_list(input_file)
                 if ifile not in old_files]

# turn each bare LFN (/store/user/manzoni/...) into a CSCS xrootd URL:
#   root://<input_redirector>//store/user/manzoni/...
files += ['root://%s//%s' % (input_redirector, ifile.lstrip('/'))
          for ifile in raw_files]

# random.shuffle(files)

files_per_job = 5 # for base submission
chunks = list(map(list, list(zip(*[iter(files)]*files_per_job))))

if len(files)%files_per_job!=0:
    last_idx = len(files)%files_per_job
    chunks += [files[-last_idx:]]

# CMSSW release to set up inside the job, captured from the current shell.
cmssw_base = os.environ.get('CMSSW_BASE', '')
scram_arch = os.environ.get('SCRAM_ARCH', '')
if not cmssw_base:
    raise RuntimeError('CMSSW_BASE is not set -- run `cmsenv` in your CMSSW_X_Y_Z/src before launching this submitter.')

check_proxy(args.min_proxy_hours)

##########################################################################################
##########################################################################################

# make output dir
if not os.path.exists(out_dir):
    try:
        os.makedirs('/'.join([pnfs_base, out_dir]))
    except:
        print('pnfs directory exists')
    os.makedirs(out_dir)
    os.makedirs(out_dir + '/logs')
    os.makedirs(out_dir + '/errs')

os.system('cp %s %s' %(cfg, out_dir))

# --- continue chunk numbering after whatever is already on pnfs (no overwrite) ---
if auto_chunk_offset:
    se_path  = '/'.join([pnfs_base, out_dir])
    on_pnfs, how = list_pnfs_chunks(se_host, se_path, out_file_name)
    existing     = sorted(on_pnfs)
    scripts      = sorted(local_chunk_scripts(out_dir))
    used         = sorted(set(existing) | set(scripts))
    chunk_offset = (max(used) + 1) if used else 0
    print('>>>> auto chunk_offset (no overwrite)')
    print('>>>>   pnfs dir: %s  (listed via %s)' % (se_path, how))
    if existing:
        print('>>>>   %d existing chunk file(s) on pnfs; highest index = %d'
              % (len(existing), existing[-1]))
    else:
        print('>>>>   no existing chunk files found on pnfs')
    if scripts:
        print('>>>>   %d job script(s) already in %s; highest index = %d'
              % (len(scripts), out_dir, scripts[-1]))
    print('>>>>   new submissions start at chunk index %d' % chunk_offset)

# never overwrite the job script of an already-submitted chunk: it is the only
# record of which input files belong to that chunk.
clash = sorted(set(local_chunk_scripts(out_dir)) &
               set(range(chunk_offset, chunk_offset + len(chunks))))
if clash:
    die('job scripts for chunk(s) %s already exist in %s; refusing to overwrite them. '
        'Use --resubmit to resubmit failed chunks, or pick a chunk_offset beyond %d.'
        % (compact(clash), out_dir, max(local_chunk_scripts(out_dir))))


sbatch_failures = []
for ijob, ichunk in enumerate(chunks):

    # actual chunk id on disk / in batch system (offset avoids output collisions)
    jobid = ijob + chunk_offset

#     if ijob>2: break

    to_write = '\n'.join([
        '#!/bin/bash',
        '',
        '# --- scratch dir ---',
        'mkdir -p /scratch/manzoni/{scratch_dir}',
        'ls /scratch/manzoni/',
        'FAILED_PARTS=0',
        '',
        '# --- CMSSW runtime (native, no container) ---',
        'export SCRAM_ARCH={scram_arch}',
        'source /cvmfs/cms.cern.ch/cmsset_default.sh',
        'echo ">>>> moving to {dir}"',
        'cd {cmssw_base}/src',
        'echo ">>>> now in $PWD"',
        'eval `scramv1 runtime -sh`',
        'echo ">>>> CMSSW_BASE=$CMSSW_BASE"',
        'echo ">>>> using python: $(which python3)"',
        '',
        '# --- fail loudly if cmsenv did not take effect ---',
        'python3 -c "import sys; print(\'>>>> python startup OK\', sys.version.split()[0])"',
        'if [ $? -ne 0 ]; then',
        '    echo ">>>> FATAL: cmsenv did not take effect (CMSSW python cannot find its standard library). Aborting chunk {ijob}."',
        '    exit 1',
        'fi',
        'which hadd',
        '',
        '# --- grid proxy: stage a private copy that survives the whole job ---',
        'cp $X509_USER_PROXY /scratch/manzoni/{scratch_dir}/x509proxy_{ijob}',
        'chmod 600 /scratch/manzoni/{scratch_dir}/x509proxy_{ijob}',
        'export X509_USER_PROXY=/scratch/manzoni/{scratch_dir}/x509proxy_{ijob}',
        '',
        '# --- run from the output dir so per-job loggers land there ---',
        'cd {dir}',
        'echo ">>>> now running in $PWD"',
        '',
    ]).format(
        dir         = '/'.join([os.getcwd(), out_dir]),
        scratch_dir = out_dir,
        cmssw_base  = cmssw_base,
        scram_arch  = scram_arch,
        ijob        = jobid,
    )

    for idx, ifile in enumerate(ichunk):
        to_write += (
            'python3 {dir}/{cfg} '
            '--inputFiles={infiles} '
            '--logfreq=5000 '
            '--destination=/scratch/manzoni/{scratch_dir} '
            '--skim '
            #'--savenontrig '
            '--filename={outfile}_chunk{ijob}_part{idx} \n'
            'if [ $? -ne 0 ]; then\n'
            '    echo ">>>> FAILED: part{idx} of chunk{ijob} ({infiles})"\n'
            '    FAILED_PARTS=$((FAILED_PARTS+1))\n'
            'fi\n'
        ).format(
            dir         = '/'.join([os.getcwd(), out_dir]),
            scratch_dir = out_dir,
            cfg         = cfg,
            outfile     = out_file_name,
            ijob        = jobid,
            infiles     = ifile,
            idx         = idx,
        )

    to_write += '\n'.join([
        '',
        'ls -latrh /scratch/manzoni/{scratch_dir}',
        'echo ">>>> $FAILED_PARTS part(s) failed for chunk {ijob}"',
        '',
        'if [ $FAILED_PARTS -gt 0 ]; then',
        '    echo ">>>> ABORTING merge and transfer for chunk {ijob}: not all parts succeeded"',
        '    exit 1',
        'fi',
        '',
        'hadd -f -k '
        '/scratch/manzoni/{scratch_dir}/{outfile}_chunk{ijob}.root '
        '/scratch/manzoni/{scratch_dir}/{outfile}_chunk{ijob}_part*.root',
        '',
        'xrdcp '
        '/scratch/manzoni/{scratch_dir}/{outfile}_chunk{ijob}.root '
        'root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/manzoni/{se_dir}/{outfile}_chunk{ijob}.root',
        '',
        'if [ $? -eq 0 ]; then',
        '    echo ">>>> xrdcp succeeded, cleaning scratch"',
        '    rm -f /scratch/manzoni/{scratch_dir}/{outfile}_chunk{ijob}.root',
        '    rm -f /scratch/manzoni/{scratch_dir}/{outfile}_chunk{ijob}_part*.root',
        'else',
        '    echo ">>>> xrdcp FAILED for chunk {ijob}, scratch files kept for inspection"',
        '    exit 1',
        'fi',
        '',
    ]).format(
        scratch_dir = out_dir,
        outfile     = out_file_name,
        ijob        = jobid,
        se_dir      = out_dir,
    )

    with open("%s/submitter_chunk%d.sh" %(out_dir, jobid), "wt") as flauncher:
        flauncher.write(to_write)

    print('[%d/%d] ' % (ijob + 1, len(chunks)), end='')
    if not submit_chunk(out_dir, jobid):
        sbatch_failures.append(jobid)

if sbatch_failures:
    die('sbatch failed for %d chunk(s): %s -- rerun with --resubmit to send them'
        % (len(sbatch_failures), compact(sbatch_failures)))