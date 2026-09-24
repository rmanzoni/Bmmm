'''
Submitter for the SLURM system (PSI Tier-3) -- Hb inclusive MC, production of 21 Sep 2026.

Derived from submitter_hb_mc_pnfs.py. Companion of submitter_bc_mc_pnfs_21sep26.py.

  * track covariance fix      --covflow cf_block_ref.
                              NOTE: this is switched ON here. The previous Hb
                              production (RJpsi_15Sep2026_..._v1) ran WITHOUT it,
                              while the Bc one had it; Bc and Hb enter the same fit,
                              so they should share the track covariance treatment.
  * Hammer FF weights         NOT run. Hammer is validated against EvtGen BC_VMN 1
                              as configured in OUR Bc DEC file. A Bc inside the Hb
                              sample decays through EvtGen's generic table, whose
                              model has not been checked, so its weights would use
                              an unvalidated denominator. The hammer_* branches are
                              in the schema anyway, all NaN (status NaN = not run).
  * Bc lifetime weights       gen_bc_ctau_weight* are filled automatically for any
                              Bc, but they assume tau_MC = 0.507 ps, the lifetime of
                              the dedicated Bc sample. A Bc in the Hb sample gets its
                              lifetime from the generic evt.pdl: DO NOT use these
                              branches on this sample without checking that value.

Run from Bmmm/Analysis/test/rjpsi, in a shell where you have done `cmsenv`:

    python3 covariance/submitter_hb_mc_pnfs_21sep26.py --test 2
    python3 covariance/submitter_hb_mc_pnfs_21sep26.py --dry-run
    python3 covariance/submitter_hb_mc_pnfs_21sep26.py
'''

import argparse
import datetime
import json
import os
import shutil
import subprocess
from glob import glob

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--test', type=int, default=0, metavar='N',
                help='submit only the first N chunks, into <out_dir>_test')
ap.add_argument('--dry-run', action='store_true',
                help='write the job scripts and the provenance, submit nothing')
args = ap.parse_args()

resubmit = False
toresubmit = []          # chunk indices, used only when resubmit = True

old_files = []
files = []

with open('files_hb_skim_15jun26.txt') as f:
    ifiles = f.read().splitlines()
    ifiles = ['root://t3dcachedb03.psi.ch:1094//'+ifile for ifile in ifiles if ifile not in old_files]
    files += ifiles

files_per_job = 1
chunks = list(map(list, list(zip(*[iter(files)]*files_per_job))))

if len(files)%files_per_job!=0:
    last_idx = len(files)%files_per_job
    chunks += [files[-last_idx:]]

queue = 'standard'; time = 720
# queue = 'short'   ; time = 60
# queue = 'long'    ; time = 10080

# out_dir = 'RJpsi_15Sep2026_notrig_HbToJPsiMuMu_inclusive_v1'
out_dir = 'RJpsi_21Sep2026_notrig_HbToJPsiMuMu_inclusive_covflow_v1'
if args.test:
    out_dir += '_test'

out_file_name = 'rjpsi'

cfg = 'inspector_rjpsi.py'

COVFLOW_REF   = os.path.join(os.getcwd(), 'cf_block_ref')

# CMSSW release to set up inside the job, captured from the current shell.
cmssw_base = os.environ.get('CMSSW_BASE', '')
scram_arch = os.environ.get('SCRAM_ARCH', '')
if not cmssw_base:
    raise RuntimeError('CMSSW_BASE is not set -- run `cmsenv` in your CMSSW_X_Y_Z/src before launching this submitter.')

##########################################################################################
# PREFLIGHT: anything that would fail every job is caught here, once
##########################################################################################

def die(msg):
    raise SystemExit('[PREFLIGHT] ' + msg)

if not os.path.isfile(cfg):
    die('%s not found in %s: run the submitter from Bmmm/Analysis/test/rjpsi.' % (cfg, os.getcwd()))
if not os.path.exists(COVFLOW_REF):
    die('covflow reference %s not found.' % COVFLOW_REF)
##########################################################################################
##########################################################################################

# make output dir
if not os.path.exists(out_dir):
    try:
        os.makedirs('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/'+out_dir)
    except:
        print('pnfs directory exists')
    os.makedirs(out_dir)
    os.makedirs(out_dir + '/logs')
    os.makedirs(out_dir + '/errs')

os.system('cp %s %s' %(cfg, out_dir))

# NEW: provenance, written at submission time
def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT,
                                       universal_newlines=True).strip()
    except subprocess.CalledProcessError as exc:
        return '<failed: %s>' % exc.output.strip()

repo = os.path.join(cmssw_base, 'src', 'Bmmm')
dirty = sh('git -C %s status --porcelain --untracked-files=no' % repo)
provenance = {
    'submitted'        : datetime.datetime.now().isoformat(timespec='seconds'),
    'out_dir'          : out_dir,
    'cmssw_base'       : cmssw_base,
    'scram_arch'       : scram_arch,
    'bmmm_commit'      : sh('git -C %s rev-parse HEAD' % repo),
    'bmmm_dirty_files' : dirty.splitlines() if dirty else [],
    'n_chunks'         : len(chunks),
    'test'             : args.test,
    'input_list'       : 'files_hb_skim_15jun26.txt',
    'covflow_ref'      : COVFLOW_REF,
    'hammer'           : 'not run on this sample (see docstring)',
    'ctau_weights'     : 'filled, but assume tau_MC = 0.507 ps: NOT valid for Bc in '
                         'this sample without checking the generic evt.pdl lifetime',
}
with open(os.path.join(out_dir, 'PROVENANCE.json'), 'w') as fprov:
    json.dump(provenance, fprov, indent=1)
if dirty:
    print('[WARN] the Bmmm checkout has uncommitted changes; they are listed in '
          '%s/PROVENANCE.json' % out_dir)
print('#### provenance: Bmmm %s' % provenance['bmmm_commit'][:10])

n_submitted = 0
for ijob, ichunk in enumerate(chunks):

    if resubmit:
        if ijob not in toresubmit: continue

    if args.test and ijob >= args.test:
        break

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
        ijob        = ijob,
    )

    for idx, ifile in enumerate(ichunk):
        to_write += (
            'python3 {dir}/{cfg} '
            '--inputFiles={infiles} '
            '--logfreq=5000 '
            '--destination=/scratch/manzoni/{scratch_dir} '
            '--savenontrig '
            '--covflow {covflow} '
            '--mc '
            '--skim '
            '--filename={outfile}_chunk{ijob}_part{idx} \n'
            'if [ $? -ne 0 ]; then\n'
            '    echo ">>>> FAILED: part{idx} of chunk{ijob} ({infiles})"\n'
            '    FAILED_PARTS=$((FAILED_PARTS+1))\n'
            'fi\n'
        ).format(
            dir         = '/'.join([os.getcwd(), out_dir]),
            covflow     = COVFLOW_REF,
            scratch_dir = out_dir,
            cfg         = cfg,
            outfile     = out_file_name,
            ijob        = ijob,
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
        ijob        = ijob,
        se_dir      = out_dir,
    )

    with open("%s/submitter_chunk%d.sh" %(out_dir, ijob), "wt") as flauncher:
        flauncher.write(to_write)

    command_sh_batch = ' '.join([
        'sbatch',
        '-p %s'%queue,
        '--account=t3',
        '-o %s/logs/chunk%d.log' %(out_dir, ijob),
        '-e %s/errs/chunk%d.err' %(out_dir, ijob),
        '--job-name=%d_%s' %(ijob, out_dir),
        '--time=%d'%time,
        '--nodes=1 --ntasks=1 --nodelist=t3wn[80-91]',
        '--mem=4000',
        '%s/submitter_chunk%d.sh' %(out_dir, ijob),
    ])

    print(command_sh_batch)
    if not args.dry_run:
        os.system(command_sh_batch)
    n_submitted += 1

print('#### %s %d chunk(s) of %d into %s'
      % ('wrote (dry run)' if args.dry_run else 'submitted', n_submitted, len(chunks), out_dir))
