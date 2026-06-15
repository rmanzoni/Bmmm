'''
Submitter for the SLURM system (PSI Tier-3).

Direct submission: the FWLite ntuplizer runs natively under the current
CMSSW release. No el7 apptainer/singularity wrapper anymore -- the recent
CMSSW (el9) matches the worker-node OS, so jobs run directly on the node.

Run this submitter from a shell where you've already done `cmsenv` in the
CMSSW_X_Y_Z/src you want the jobs to use: the release path is read from
$CMSSW_BASE and baked into each job script, and `scram runtime` is sourced
from there (NOT from the output dir, which need not live inside the release).
'''

import os
import random
from glob import glob

resubmit = False

old_files = []
files = []

with open('files_bc_skim_15jun26.txt') as f:
# with open('files_BcToJPsiMuMu_inclusive_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1.txt') as f:
# with open('files_HbToJPsiMuMu_3MuFilter_TuneCP5_13TeV-pythia8-evtgen_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1_ext1-v3.txt') as f:
    ifiles = f.read().splitlines()
    ifiles = ['root://t3dcachedb03.psi.ch:1094//'+ifile for ifile in ifiles if ifile not in old_files]
    files += ifiles

# random.shuffle(files)

files_per_job = 1
chunks = list(map(list, list(zip(*[iter(files)]*files_per_job))))

if len(files)%files_per_job!=0:
    last_idx = len(files)%files_per_job
    chunks += [files[-last_idx:]]

queue = 'standard'; time = 720
# queue = 'short'   ; time = 60
# queue = 'long'    ; time = 10080

out_dir = 'RJpsi_15Jun2026_notrig_BcToJPsiMuMu_inclusive_v3'
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
        os.makedirs('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/'+out_dir)
    except:
        print('pnfs directory exists')
    os.makedirs(out_dir)
    os.makedirs(out_dir + '/logs')
    os.makedirs(out_dir + '/errs')

os.system('cp %s %s' %(cfg, out_dir))


for ijob, ichunk in enumerate(chunks):

    if resubmit:
        if ijob not in toresubmit: continue

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
        ijob        = ijob,
    )

    for idx, ifile in enumerate(ichunk):
        to_write += (
            'python3 {dir}/{cfg} '
            '--inputFiles={infiles} '
            '--logfreq=5000 '
            '--destination=/scratch/manzoni/{scratch_dir} '
            '--savenontrig '
            '--mc '
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
        # '-w t3wn70,t3wn71,t3wn72,t3wn73', # only the best nodes
        '%s/submitter_chunk%d.sh' %(out_dir, ijob),
    ])

    print(command_sh_batch)
    os.system(command_sh_batch)
