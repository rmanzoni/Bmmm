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
DIFF SUBMISSION (adding neglected files to an existing production)
----------------------------------------------------------------------------
Set `diff_mode = True` to submit ONLY the files that are in `addendum_file`
(the NEW, complete list) and not in `base_file` (the list used for the base
submission). Usage, once per year:

    cmsenv   # in the SAME release used for the base submission
    python3 submitter_data_skim_cscs.py 2024          # dry run (default)
    # check the printout, then set dry_run = False and rerun

Safety checks in diff mode (all abort unless explicitly overridden):
  * base-minus-addendum must be empty: a file processed in the base run but
    absent from the new list means a dataset version switched, and adding
    the new version would double-count events;
  * the local out_dir of the base submission must exist;
  * the inspector (`cfg`) must be identical to the copy in out_dir, and
    $CMSSW_BASE must match the release baked into the base job scripts;
  * files already present in any existing submitter_chunk*.sh are excluded
    whatever base_file says (and reported).

NO OVERWRITE -- chunk numbering: with `auto_chunk_offset = True` the new
submission starts at 1 + the highest chunk index found EITHER on pnfs
(`<out_file_name>_chunk<N>.root`, via `xrdfs <se_host> ls`, falling back to
the local mount) OR among the local submitter_chunk<N>.sh scripts. The second
source matters when the last base chunks failed and never reached pnfs.
Set `auto_chunk_offset = False` to use the fixed `chunk_offset` instead.

DRY RUN: with `dry_run = True` nothing is written to disk and nothing is
submitted; the plan (files, chunk range, sbatch commands) is printed.

Each real submission appends a line to <out_dir>/submissions.log and saves
the submitted file list to <out_dir>/submitted_files_chunk<first>-<last>.txt.
'''

import os
import re
import sys
import random
import filecmp
import difflib
import datetime
import subprocess
from glob import glob
from collections import Counter

resubmit = False

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
if len(sys.argv) > 1:
    year = sys.argv[1]  # e.g. `python3 submitter_data_skim_cscs.py 2024`

input_files_by_year = {
    '2022': '../files/files_data2022_cscs_24sep26.txt',
    '2023': '../files/files_data2023_cscs_24sep26.txt',
    '2024': '../files/files_data2024_cscs_24sep26.txt',
    '2025': '../files/files_data2025_cscs_24sep26.txt',
    '2026': '../files/files_data2026_cscs_24sep26.txt',
}

# NEW (complete) lists from the fixed make_cscs_file_list.py.
# Set new_list_date to the --date used when generating them.
new_list_date = '29sep26'
addendum_files_by_year = {
    y: '../files/files_data%s_cscs_%s.txt' % (y, new_list_date)
    for y in ['2022', '2023', '2024', '2025', '2026']
}


addendum_files_by_year = {
    '2022': 'files_data2022_cscs_24sep26_v2.txt',
    '2023': 'files_data2023_cscs_24sep26_v2.txt',
    '2024': 'files_data2024_cscs_24sep26_v2.txt',
    '2025': 'files_data2025_cscs_24sep26_v2.txt',
    '2026': 'files_data2026_cscs_24sep26_v2.txt',
}


out_dir_by_year = {
    '2022': 'RJpsi_23Jun2026_data2022_cscs_24sep26_v1',
    '2023': 'RJpsi_23Jun2026_data2023_cscs_24sep26_v1',
    '2024': 'RJpsi_23Jun2026_data2024_cscs_24sep26_v1',
    '2025': 'RJpsi_23Jun2026_data2025_cscs_24sep26_v1',
    '2026': 'RJpsi_23Jun2026_data2026_cscs_24sep26_v1',
}

# ---------------------------------------------------------------------------
# diff submission config (kept for reuse; OFF for a fresh CSCS submission)
# ---------------------------------------------------------------------------
# When True: submit only (addendum \ base). When False: submit input_file.
diff_mode = True

# DRY RUN: print the plan, write nothing, submit nothing.
dry_run = False

# Overrides for the diff-mode safety checks (leave False unless you know why)
allow_dropped_base_files = False  # base files missing from the new list
allow_cfg_change         = False  # inspector differs from the base copy
allow_release_change     = False  # $CMSSW_BASE differs from the base jobs

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


def existing_chunk_indices(se_host, se_path, out_file_name):
    '''
    Return the sorted list of chunk indices already present in the pnfs dir.
    Matches merged files named '<out_file_name>_chunk<N>.root' (not part files).
    Uses `xrdfs <se_host> ls <se_path>`, falling back to the local pnfs mount.
    '''
    pat = re.compile(r'%s_chunk(\d+)\.root$' % re.escape(out_file_name))
    listing = []
    try:
        res = subprocess.run(
            ['xrdfs', se_host, 'ls', se_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=120,
        )
        if res.returncode == 0:
            listing = res.stdout.splitlines()
        else:
            print('>>>> xrdfs ls returned %d (%s); falling back to local mount'
                  % (res.returncode, res.stderr.strip()))
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print('>>>> xrdfs unavailable (%s); falling back to local mount' % exc)

    if not listing:
        try:
            listing = glob(os.path.join(se_path, '*'))
        except Exception:
            listing = []

    indices = []
    for entry in listing:
        m = pat.search(os.path.basename(entry.strip()))
        if m:
            indices.append(int(m.group(1)))
    return sorted(set(indices))


def local_submitter_info(out_dir):
    '''
    Scan <out_dir>/submitter_chunk<N>.sh.
    Returns (indices, submitted input LFNs, set of CMSSW src dirs used).
    '''
    pat_sh  = re.compile(r'submitter_chunk(\d+)\.sh$')
    pat_in  = re.compile(r'--inputFiles=(\S+)')
    pat_rel = re.compile(r'^cd (\S+/src)\s*$', re.M)
    indices, submitted, releases = [], set(), set()
    for path in glob(os.path.join(out_dir, 'submitter_chunk*.sh')):
        m = pat_sh.search(os.path.basename(path))
        if not m:
            continue
        indices.append(int(m.group(1)))
        with open(path) as f:
            content = f.read()
        for url in pat_in.findall(content):
            # root://<door>//store/... -> /store/...
            submitted.add('/' + url.split('//', 2)[-1].lstrip('/'))
        releases.update(pat_rel.findall(content))
    return sorted(indices), submitted, releases


def dataset_dir(lfn):
    '''/store/user/manzoni/<PD>/<processed>/... -> <PD>/<processed>'''
    parts = lfn.split('/')
    return '/'.join(parts[4:6]) if len(parts) > 6 else lfn


def abort(msg):
    print('>>>> ABORT: ' + msg)
    sys.exit(1)


out_dir = out_dir_by_year[year]
cmssw_base = os.environ.get('CMSSW_BASE', '')
cfg = 'inspector_rjpsi.py'

if diff_mode:
    if not addendum_file or not os.path.exists(addendum_file):
        abort('addendum_file (new list) not found: %r' % addendum_file)
    if os.path.abspath(addendum_file) == os.path.abspath(base_file):
        abort('addendum_file and base_file are the same file (%s): set '
              'new_list_date to the date of the NEW lists.' % base_file)
    if not os.path.isdir(out_dir):
        abort('local out_dir %s of the base submission not found. Run from the '
              'directory where the base submission was made.' % out_dir)

    base          = set(_read_list(base_file))
    addendum      = _read_list(addendum_file)
    addendum_set  = set(addendum)

    print('>>>> diff mode (year %s)' % year)
    print('>>>>   base     (%s): %d files' % (base_file, len(base)))
    print('>>>>   addendum (%s): %d files' % (addendum_file, len(addendum)))

    # 1) nothing processed in the base run may be missing from the new list
    dropped = sorted(base - addendum_set)
    if dropped:
        print('>>>>   %d base files are NOT in the new list, by dataset:' % len(dropped))
        for ds, n in sorted(Counter(dataset_dir(f) for f in dropped).items()):
            print('>>>>     %6d  %s' % (n, ds))
        if not allow_dropped_base_files:
            abort('base files missing from the new list: a dataset version may '
                  'have switched, submitting would double-count events. '
                  'Set allow_dropped_base_files = True to override.')

    # 2) same code and release as the base submission
    base_cfg = os.path.join(out_dir, cfg)
    if not os.path.exists(base_cfg):
        print('>>>>   WARNING: no base copy of %s in %s, cannot check the code'
              % (cfg, out_dir))
    elif not filecmp.cmp(cfg, base_cfg, shallow=False):
        print('>>>>   %s differs from the base copy %s:' % (cfg, base_cfg))
        with open(base_cfg) as fa, open(cfg) as fb:
            for line in list(difflib.unified_diff(
                    fa.readlines(), fb.readlines(), base_cfg, cfg, n=1))[:40]:
                print('      ' + line.rstrip())
        if not allow_cfg_change:
            abort('inspector changed since the base submission. '
                  'Set allow_cfg_change = True to override.')

    local_idx, already_submitted, base_releases = local_submitter_info(out_dir)
    cur_release = os.path.join(cmssw_base, 'src') if cmssw_base else ''
    if base_releases and base_releases != {cur_release}:
        print('>>>>   base jobs used %s, current is %s'
              % (sorted(base_releases), cur_release or '(no CMSSW_BASE)'))
        if not allow_release_change:
            abort('CMSSW release differs from the base submission. '
                  'Set allow_release_change = True to override.')

    # 3) the diff, excluding anything already in a submitter script
    raw_files = list(dict.fromkeys(
        ifile for ifile in addendum
        if ifile not in base and ifile not in old_files))
    in_scripts = [f for f in raw_files if f in already_submitted]
    if in_scripts:
        print('>>>>   %d new files already appear in existing submitter scripts '
              '(excluded)' % len(in_scripts))
        raw_files = [f for f in raw_files if f not in already_submitted]
    not_in_base = already_submitted - base
    if not_in_base:
        print('>>>>   NOTE: %d files in existing submitter scripts are not in '
              'base_file (earlier diff submissions?)' % len(not_in_base))

    print('>>>>   new to submit: %d files, by dataset:' % len(raw_files))
    for ds, n in sorted(Counter(dataset_dir(f) for f in raw_files).items()):
        print('>>>>     %6d  %s' % (n, ds))

    if not raw_files:
        print('>>>> nothing to submit for %s' % year)
        sys.exit(0)

    if auto_chunk_offset:
        print('>>>>   auto_chunk_offset ON: new chunks continue after the '
              'highest index on pnfs or in the local submitter scripts.')
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

queue = 'standard'; time = 720
# queue = 'short'   ; time = 60
# queue = 'long'    ; time = 10080

# out_dir = 'RJpsi_10Jun2026_notrig_Hb_inclusive_v1'   (out_dir is set above from year)

out_file_name = 'rjpsi'

# cfg and cmssw_base (CMSSW release captured from the current shell) are set
# above, before the diff-mode checks that need them.
scram_arch = os.environ.get('SCRAM_ARCH', '')
if not cmssw_base:
    raise RuntimeError('CMSSW_BASE is not set -- run `cmsenv` in your CMSSW_X_Y_Z/src before launching this submitter.')

##########################################################################################
##########################################################################################

# make output dir
if dry_run:
    print('>>>> DRY RUN: nothing will be written or submitted')
elif not os.path.exists(out_dir):
    try:
        os.makedirs('/'.join([pnfs_base, out_dir]))
    except:
        print('pnfs directory exists')
    os.makedirs(out_dir)
    os.makedirs(out_dir + '/logs')
    os.makedirs(out_dir + '/errs')

# in diff mode the base copy is kept (checked identical above)
if not dry_run and not diff_mode:
    os.system('cp %s %s' %(cfg, out_dir))

# --- continue chunk numbering after whatever is already on pnfs (no overwrite) ---
if auto_chunk_offset:
    se_path  = '/'.join([pnfs_base, out_dir])
    existing = existing_chunk_indices(se_host, se_path, out_file_name)
    local_existing = local_submitter_info(out_dir)[0] if os.path.isdir(out_dir) else []
    highest = max(existing + local_existing) if (existing or local_existing) else -1
    chunk_offset = highest + 1
    print('>>>> auto chunk_offset (no overwrite)')
    print('>>>>   pnfs dir: %s' % se_path)
    print('>>>>   pnfs chunk files    : %d (highest index %s)'
          % (len(existing), existing[-1] if existing else '-'))
    print('>>>>   local submitter .sh : %d (highest index %s)'
          % (len(local_existing), local_existing[-1] if local_existing else '-'))
    missing_on_pnfs = sorted(set(local_existing) - set(existing))
    if missing_on_pnfs:
        print('>>>>   NOTE: %d submitted chunks have no file on pnfs (failed or '
              'still running?), e.g. %s' % (len(missing_on_pnfs), missing_on_pnfs[:10]))
    print('>>>>   new submissions start at chunk index %d' % chunk_offset)

first_chunk = chunk_offset
last_chunk  = chunk_offset + len(chunks) - 1
print('>>>> %d files in %d chunks: chunk%d ... chunk%d'
      % (len(files), len(chunks), first_chunk, last_chunk))

if not dry_run:
    tag = 'chunk%d-%d' % (first_chunk, last_chunk)
    with open(os.path.join(out_dir, 'submitted_files_%s.txt' % tag), 'w') as f:
        f.write('\n'.join(raw_files) + '\n')
    with open(os.path.join(out_dir, 'submissions.log'), 'a') as f:
        f.write('%s  year=%s  diff_mode=%s  files=%d  chunks=%s  release=%s  '
                'base=%s  addendum=%s\n' % (
                    datetime.datetime.now().isoformat(timespec='seconds'), year,
                    diff_mode, len(files), tag, cmssw_base, base_file,
                    addendum_file if diff_mode else input_file))


for ijob, ichunk in enumerate(chunks):

    # actual chunk id on disk / in batch system (offset avoids output collisions)
    jobid = ijob + chunk_offset

    if resubmit:
        if jobid not in toresubmit: continue

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

    if not dry_run:
        with open("%s/submitter_chunk%d.sh" %(out_dir, jobid), "wt") as flauncher:
            flauncher.write(to_write)

    command_sh_batch = ' '.join([
        'sbatch',
        '-p %s'%queue,
        '--account=t3',
        '-o %s/logs/chunk%d.log' %(out_dir, jobid),
        '-e %s/errs/chunk%d.err' %(out_dir, jobid),
        '--job-name=%d_%s' %(jobid, out_dir),
        '--time=%d'%time,
        '--nodes=1 --ntasks=1 --nodelist=t3wn[80-91]',
        # '-w t3wn70,t3wn71,t3wn72,t3wn73', # only the best nodes
        '%s/submitter_chunk%d.sh' %(out_dir, jobid),
    ])

    print(command_sh_batch)
    if not dry_run:
        os.system(command_sh_batch)

if dry_run:
    print('>>>> DRY RUN done: %d jobs would be submitted (chunk%d ... chunk%d). '
          'Set dry_run = False to submit.' % (len(chunks), first_chunk, last_chunk))