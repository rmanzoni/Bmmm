'''
Submitter for the SLURM system (PSI Tier-3).

Runs inspector_tau3mu.py over the 2024 ParkingDoubleMuonLowMass datasets.
Input file lists must be pre-generated with dasgoclient (one .txt per dataset).

Run from a shell where you've already done `cmsenv` in the CMSSW_X_Y_Z/src
you want the jobs to use: the release path is read from $CMSSW_BASE and baked
into each job script.
'''

import os

resubmit   = False
toresubmit = []

eras = [
    ('C', '-v1'),
    ('D', '-v1'),
    ('E', '-v1'),
    ('F', '-v3'),
    ('G', '-v3'),
    ('H', '-v3'),
    ('I', '-v3'),
    ('I', '_v2-v2'),
]

files_per_job = 35

queue = 'standard'; time = 720
# queue = 'short'   ; time = 60
# queue = 'long'    ; time = 10080

out_dir       = 'Tau3Mu_26Jun2026_data2024_v2'
out_file_name = 'tau3mu'
cfg           = 'inspector_tau3mu.py'

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

for (iera, iversion) in eras:
    for part in range(8):

        dataset   = f'/ParkingDoubleMuonLowMass{part}/Run2024{iera}-MINIv6NANOv15{iversion}/MINIAOD'
        txt_file  = '/'.join([
            os.environ['CMSSW_BASE'],
            'src',
            'Bmmm/Analysis/test/files/' + dataset.lstrip('/').replace('/', '-') + '.txt'
            ])
        
        if not os.path.exists(txt_file):
            print(f'WARNING: {txt_file} not found, skipping {dataset}')
            continue

        with open(txt_file) as f:
            files = [
#                 'root://t3dcachedb03.psi.ch:1094//' + line.strip()
                'root://cms-xrd-global.cern.ch//' + line.strip()
                for line in f.read().splitlines()
                if line.strip()
            ]

        if not files:
            print(f'WARNING: {txt_file} is empty, skipping {dataset}')
            continue

        # human-readable label used in filenames, e.g. LowMass0_Run2024C_MINIv6NANOv15_v1
        ver_clean     = iversion.lstrip('-_').replace('-', '')
        dataset_label = f'LowMass{part}_Run2024{iera}_MINIv6NANOv15_{ver_clean}'

        # split into fixed-size chunks
        chunks = [files[i:i + files_per_job] for i in range(0, len(files), files_per_job)]

        print(f'{dataset}  ->  {len(files)} files, {len(chunks)} jobs')

        for ichunk_idx, ichunk in enumerate(chunks):
            
#             if global_job_idx>2: break
            
            ijob = global_job_idx
            global_job_idx += 1

            if resubmit:
                if ijob not in toresubmit:
                    continue

            input_files_str = ','.join(ichunk)
            # output filename passed to --filename (base name, no .root suffix)
            out_filename = f'{out_file_name}_{dataset_label}_chunk{ichunk_idx}'

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
                '--destination=/scratch/manzoni/{scratch_dir} '
                '--filename={outfile} '
                '--maxevents=-1\n'
                'if [ $? -ne 0 ]; then\n'
                '    echo ">>>> FAILED: job {ijob} ({dataset_label} chunk {ichunk_idx})"\n'
                '    exit 1\n'
                'fi\n'
            ).format(
                dir           = '/'.join([os.getcwd(), out_dir]),
                scratch_dir   = out_dir,
                cfg           = cfg,
                infiles       = input_files_str,
                outfile       = out_filename,
                ijob          = ijob,
                dataset_label = dataset_label,
                ichunk_idx    = ichunk_idx,
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
                script_path,
            ])

            print(command_sh_batch)
            os.system(command_sh_batch)
