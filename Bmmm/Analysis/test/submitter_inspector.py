'''
Submitter for the SLURM system
'''

import os
import random
from glob import glob

with open('files_2018D_part1.txt') as f:
    files = f.read().splitlines()
    files = ['root://cms-xrd-global.cern.ch//'+ifile for ifile in files]

# random.shuffle(files)

files_per_job = 20
chunks = map(list, list(zip(*[iter(files)]*files_per_job)))

if len(files)%files_per_job!=0:
    last_idx = len(files)%files_per_job
    chunks += files[-last_idx:]

queue = 'standard'; time = 720
# queue = 'short'   ; time = 60
# queue = 'long'    ; time = 10080

# out_dir = 'ParkingBPH1_Run2018D_UL2018_MiniAODv2_v1_14apr22_v2'
out_dir = 'Bmmmm_ParkingBPH1_Run2018D_UL2018_MiniAODv2_v1_05jun22_v1'

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

os.system('cp inspector_bmmm_analysis.py %s' %out_dir)
# os.system('cp decays.py %s' %out_dir)
  
# for ijob, ichunk in enumerate(chunks[:50]):
for ijob, ichunk in enumerate(chunks):

#     if ijob>=50: break
        
    to_write = '\n'.join([
        '#!/bin/bash',
        'cd {dir}',
        'scramv1 runtime -sh',
        'mkdir -p /scratch/manzoni/{scratch_dir}',
        'ls /scratch/manzoni/',
        '',
    ]).format(
        dir         = '/'.join([os.getcwd(), out_dir]), 
        scratch_dir = out_dir, 
        cfg         = 'inspector_bmmm_analysis.py', 
        ijob        = ijob, 
        infiles     = ','.join(['/scratch/manzoni/{scratch_dir}/{ifile}'.format(scratch_dir=out_dir, ifile=ifile) for ifile in ichunk]),
        se_dir      = out_dir,
        )
        
#     for idx, ifile in enumerate(ichunk):
#         to_write += 'python {cfg} --inputFiles={infiles} --logfreq=5000 --destination=/scratch/manzoni/{scratch_dir} --filename=bmmm_data_chunk{ijob} --filemode={filemode} \n'.format(
#             dir         = '/'.join([os.getcwd(), out_dir]), 
#             scratch_dir = out_dir, 
#             cfg         = 'inspector_bmmm_analysis.py', 
#             ijob        = ijob, 
#             infiles     = ifile,
#             se_dir      = out_dir, 
#             filemode    = 'recreate' if idx==0 else 'update',     
#         )
    for idx, ifile in enumerate(ichunk):
        to_write += 'python {cfg} --inputFiles={infiles} --logfreq=5000 --destination=/scratch/manzoni/{scratch_dir} --filename=bmmm_data_chunk{ijob}_part{idx} \n'.format(
            dir         = '/'.join([os.getcwd(), out_dir]), 
            scratch_dir = out_dir, 
            cfg         = 'inspector_bmmm_analysis.py', 
            ijob        = ijob, 
            infiles     = ifile,
            se_dir      = out_dir, 
            idx         = idx,
        )
        
    to_write += '\n'.join([
        '',
        'hadd -f -k /scratch/manzoni/{scratch_dir}/bmmm_data_chunk{ijob}.root /scratch/manzoni/{scratch_dir}/bmmm_data_chunk{ijob}_part*.root',
        'xrdcp /scratch/manzoni/{scratch_dir}/bmmm_data_chunk{ijob}.root root://t3dcachedb.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/manzoni/{se_dir}/bmmm_data_chunk{ijob}.root',
        'rm /scratch/manzoni/{scratch_dir}/bmmm_data_chunk{ijob}*.root',
        '',
    ]).format(
        dir         = '/'.join([os.getcwd(), out_dir]), 
        scratch_dir = out_dir, 
        cfg         = 'inspector_bmmm_analysis.py', 
        ijob        = ijob, 
        infiles     = ','.join(['/scratch/manzoni/{scratch_dir}/{ifile}'.format(scratch_dir=out_dir, ifile=ifile.split('/')[-1]) for ifile in ichunk]),
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
#         '-w t3wn70,t3wn71,t3wn72,t3wn73', # only the best nodes
#         '-w t3wn70', # only the best nodes
        '%s/submitter_chunk%d.sh' %(out_dir, ijob), 
    ])

    print(command_sh_batch)
    os.system(command_sh_batch)
    
    
    
