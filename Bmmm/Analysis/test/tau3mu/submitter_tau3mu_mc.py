'''
Submitter for the SLURM system (PSI Tier-3) -- tau3mu signal MC.

Runs inspector_tau3mu.py over the displaced tau -> a(mu mu) mu signal MC
(MiniAODv6) produced on the 21-point parameter grid:
    mass = (0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5) GeV
    ctau = (10, 30, 100) mm

The samples live on the PSI pnfs area, so files are discovered directly by
globbing the local /pnfs mount. The per-sample directory encodes the grid
point (e.g. a_mass_0p3gev_ctau_10mm); the CRAB timestamp and the 0000/0001/...
counter directories are wildcarded. Files are read via the dCache redirector
root://t3dcachedb03.psi.ch:1094// .

Direct submission: the FWLite ntuplizer runs natively under the current
CMSSW release (el9 worker nodes, no apptainer wrapper).

Run this submitter from a shell where you've already done `cmsenv` in the
CMSSW_X_Y_Z/src you want the jobs to use: the release path is read from
$CMSSW_BASE and baked into each job script.
'''

import os
from glob import glob

resubmit   = False
toresubmit = []

# signal MC parameter grid -- same values as the GEN production
masses = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]  # GeV
ctaus  = [10, 30, 100]                          # mm

files_per_job = 100  # tau3mu inspector takes a comma-separated --inputFiles list, one output per job

queue = 'standard'; time = 720
# queue = 'short'   ; time = 60
# queue = 'long'    ; time = 10080

out_dir       = 'Tau3Mu_26Jun2026_signalMC_v2'
out_file_name = 'tau3mu'
cfg           = 'inspector_tau3mu.py'

# pnfs layout: <pnfs_base>/<sample_dir>/<mc_prod_tag>/<timestamp>/<counter>/*.root
# the timestamp and counter dirs vary per CRAB task -> wildcard them.
pnfs_base   = '/pnfs/psi.ch/cms/trivcat/store/user/manzoni'
mc_prod_tag = 'tau3mu_displaced_run3_MiniAOD_25jun26'
redirector  = 'root://t3dcachedb03.psi.ch:1094//'

cmssw_base = os.environ.get('CMSSW_BASE', '')
scram_arch = os.environ.get('SCRAM_ARCH', '')
if not cmssw_base:
    raise RuntimeError(
        'CMSSW_BASE is not set -- run `cmsenv` in your CMSSW_X_Y_Z/src '
        'before launching this submitter.'
    )

##########################################################################################
##########################################################################################

# make output dir
if not os.path.exists(out_dir):
    try:
        os.makedirs('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/' + out_dir)
    except:
        print('pnfs directory exists')
    os.makedirs(out_dir)
    os.makedirs(out_dir + '/logs')
    os.makedirs(out_dir + '/errs')

os.system('cp %s %s' % (cfg, out_dir))

global_job_idx = 0

for mass in masses:
    mass_str = f'{mass:.1f}'.replace('.', 'p')  # 0.3 -> '0p3', 1.1 -> '1p1'

    for ctau in ctaus:

        sample_dir = f'a_mass_{mass_str}gev_ctau_{ctau}mm'
        pattern    = '/'.join([pnfs_base, sample_dir, mc_prod_tag, '*', '*', '*.root'])
        raw_files  = sorted(glob(pattern))

        if not raw_files:
            print(f'WARNING: no files found for {sample_dir} (pattern: {pattern}), skipping')
            continue

        files = [redirector + f for f in raw_files]

        # human-readable label used in filenames, e.g. mass0p3gev_ctau10mm
        point_label = f'mass{mass_str}gev_ctau{ctau}mm'

        # split into fixed-size chunks
        chunks = [files[i:i + files_per_job] for i in range(0, len(files), files_per_job)]

        print(f'{sample_dir}  ->  {len(files)} files, {len(chunks)} jobs')

        for ichunk_idx, ichunk in enumerate(chunks):

#             if global_job_idx > 2: break

            ijob = global_job_idx
            global_job_idx += 1

            if resubmit:
                if ijob not in toresubmit:
                    continue

            input_files_str = ','.join(ichunk)
            # output filename passed to --filename (base name, no .root suffix)
            out_filename = f'{out_file_name}_{point_label}_chunk{ichunk_idx}'

            to_write = '\n'.join([
                '#!/bin/bash',
                '',
                '# --- scratch dir ---',
                'mkdir -p /scratch/manzoni/{scratch_dir}',
                'ls /scratch/manzoni/',
                '',
                '# --- CMSSW runtime (native, no container) ---',
                'export SCRAM_ARCH={scram_arch}',
                'source /cvmfs/cms.cern.ch/cmsset_default.sh',
                'echo ">>>> moving to {cmssw_base}/src"',
                'cd {cmssw_base}/src',
                'echo ">>>> now in $PWD"',
                'eval `scramv1 runtime -sh`',
                'echo ">>>> CMSSW_BASE=$CMSSW_BASE"',
                'echo ">>>> using python: $(which python3)"',
                '',
                '# --- fail loudly if cmsenv did not take effect ---',
                'python3 -c "import sys; print(\'>>>> python startup OK\', sys.version.split()[0])"',
                'if [ $? -ne 0 ]; then',
                '    echo ">>>> FATAL: cmsenv did not take effect. Aborting job {ijob}."',
                '    exit 1',
                'fi',
                'which hadd',
                '',
                '# --- grid proxy ---',
                'cp $X509_USER_PROXY /scratch/manzoni/{scratch_dir}/x509proxy_{ijob}',
                'chmod 600 /scratch/manzoni/{scratch_dir}/x509proxy_{ijob}',
                'export X509_USER_PROXY=/scratch/manzoni/{scratch_dir}/x509proxy_{ijob}',
                '',
                '# --- run from the output dir ---',
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

            to_write += (
                'ipython -i -- {dir}/{cfg} '
                '--inputFiles={infiles} '
                '--logfreq=5000 '
                '--mc '
                '--destination=/scratch/manzoni/{scratch_dir} '
                '--filename={outfile} '
                '--maxevents=-1\n'
                'if [ $? -ne 0 ]; then\n'
                '    echo ">>>> FAILED: job {ijob} ({point_label} chunk {ichunk_idx})"\n'
                '    exit 1\n'
                'fi\n'
            ).format(
                dir         = '/'.join([os.getcwd(), out_dir]),
                scratch_dir = out_dir,
                cfg         = cfg,
                infiles     = input_files_str,
                outfile     = out_filename,
                ijob        = ijob,
                point_label = point_label,
                ichunk_idx  = ichunk_idx,
            )

            to_write += '\n'.join([
                '',
                'ls -latrh /scratch/manzoni/{scratch_dir}',
                '',
                'xrdcp '
                '/scratch/manzoni/{scratch_dir}/{outfile}.root '
                'root://t3dcachedb03.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/manzoni/{se_dir}/{outfile}.root',
                '',
                'if [ $? -eq 0 ]; then',
                '    echo ">>>> xrdcp succeeded, cleaning scratch"',
                '    rm -f /scratch/manzoni/{scratch_dir}/{outfile}.root',
                'else',
                '    echo ">>>> xrdcp FAILED for job {ijob}, scratch file kept"',
                '    exit 1',
                'fi',
                '',
            ]).format(
                scratch_dir = out_dir,
                outfile     = out_filename,
                ijob        = ijob,
                se_dir      = out_dir,
            )

            script_path = '%s/submitter_chunk%d.sh' % (out_dir, ijob)
            with open(script_path, 'wt') as flauncher:
                flauncher.write(to_write)

            command_sh_batch = ' '.join([
                'sbatch',
                '-p %s' % queue,
                '--account=t3',
                '-o %s/logs/chunk%d.log' % (out_dir, ijob),
                '-e %s/errs/chunk%d.err' % (out_dir, ijob),
                '--job-name=%d_%s' % (ijob, out_dir),
                '--time=%d' % time,
                '--nodes=1 --ntasks=1 --nodelist=t3wn[80-91]',
#                 '--mem=4000',
                script_path,
            ])

            print(command_sh_batch)
            os.system(command_sh_batch)