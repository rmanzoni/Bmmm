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
DIFF SUBMISSION
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
'''

import os
import re
import random
import subprocess
from glob import glob

resubmit = False

# ---------------------------------------------------------------------------
# diff submission config
# ---------------------------------------------------------------------------
# When True: submit only (addendum \ base). When False: submit input_file.
diff_mode = True

input_file    = 'files_data2024_partial_skim_23jun26.txt'                       # used when diff_mode = False
base_file     = 'files_data2024_partial_skim_23jun26.txt'                        # already-processed list
addendum_file = 'files_data2024_partial_skim_23jun26_addendum_23jun26.txt'       # superset: base + new files

# Chunk numbering / no-overwrite behaviour.
#   auto_chunk_offset = True : list the pnfs out_dir, start at (highest existing
#                              chunk index) + 1, so a diff/resubmission lands in
#                              the SAME pnfs dir without overwriting anything.
#   auto_chunk_offset = False: use the fixed `chunk_offset` below.
auto_chunk_offset = True
chunk_offset      = 0          # manual offset, used only when auto_chunk_offset = False

# storage element (must match the xrdcp destination used below)
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

files += ['root://t3dcachedb03.psi.ch:1094//' + ifile for ifile in raw_files]

# random.shuffle(files)

files_per_job = 1
chunks = list(map(list, list(zip(*[iter(files)]*files_per_job))))

if len(files)%files_per_job!=0:
    last_idx = len(files)%files_per_job
    chunks += [files[-last_idx:]]

queue = 'standard'; time = 720
# queue = 'short'   ; time = 60
# queue = 'long'    ; time = 10080

out_dir = 'RJpsi_23Jun2026_notrig_data2024_partial_v1'
# out_dir = 'RJpsi_10Jun2026_notrig_Hb_inclusive_v1'

out_file_name = 'rjpsi'

cfg = 'inspector_rjpsi.py'

# CMSSW release to set up inside the job, captured from the current shell.
cmssw_base = os.environ.get('CMSSW_BASE', '')
scram_arch = os.environ.get('SCRAM_ARCH', '')
if not cmssw_base:
    raise RuntimeError('CMSSW_BASE is not set -- run `cmsenv` in your CMSSW_X_Y_Z/src before launching this submitter.')

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
    existing = existing_chunk_indices(se_host, se_path, out_file_name)
    chunk_offset = (max(existing) + 1) if existing else 0
    print('>>>> auto chunk_offset (no overwrite)')
    print('>>>>   pnfs dir: %s' % se_path)
    if existing:
        print('>>>>   %d existing chunk file(s); highest index = %d'
              % (len(existing), existing[-1]))
    else:
        print('>>>>   no existing chunk files found')
    print('>>>>   new submissions start at chunk index %d' % chunk_offset)


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
            '--savenontrig '
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
    os.system(command_sh_batch)
