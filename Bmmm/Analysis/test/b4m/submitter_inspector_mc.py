'''
Submitter for the SLURM system

remember python3

is you use plain python, it won't find ROOT

'''

import os
import random
from glob import glob



#files_jpsi_phi_4mu_2022.txt
#files_jpsi_phi_4mu_2022EE.txt

with open('files_jpsi_phi_4mu_2022.txt') as f:
#with open('files_jpsi_phi_4mu_2022EE.txt') as f:
    files = f.read().splitlines()
    files = ['root://cms-xrd-global.cern.ch//'+ifile for ifile in files]

# random.shuffle(files)

files_per_job = 1
chunks = map(list, list(zip(*[iter(files)]*files_per_job)))

if len(files)%files_per_job!=0:
    last_idx = len(files)%files_per_job
    chunks += files[-last_idx:]

# queue = 'standard'; time = 720
queue = 'short'   ; time = 60
# queue = 'long'    ; time = 10080

out_dir = 'BsToJPsiPhiTo4Mu_Run2022_26jan24_v1'
#out_dir = 'BsToJPsiPhiTo4Mu_Run2022EE_26jan24_v1'

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
    os.makedirs(out_dir + '/cutflow')

os.system('cp inspector_b4m_analysis.py %s' %out_dir)
# os.system('cp decays.py %s' %out_dir)
  
# for ijob, ichunk in enumerate(chunks[:50]):
for ijob, ichunk in enumerate(chunks):

    #if ijob>=5: break
        
    to_write = '\n'.join([
        '#!/bin/bash',
        'cd {dir}',
        'echo "doing CMSENV"',
        'scramv1 runtime -sh',
        'echo $CMSSW_BASE',
        'mkdir -p /scratch/manzoni/{scratch_dir}',
        'mkdir -p /scratch/manzoni/{scratch_dir}/logs',
        'ls /scratch/manzoni/',
        '',
    ]).format(
        dir         = '/'.join([os.getcwd(), out_dir]), 
        scratch_dir = out_dir, 
        cfg         = 'inspector_b4m_analysis.py', 
        ijob        = ijob, 
        infiles     = ','.join(['/scratch/manzoni/{scratch_dir}/{ifile}'.format(scratch_dir=out_dir, ifile=ifile) for ifile in ichunk]),
        se_dir      = out_dir,
        )
        
#     for idx, ifile in enumerate(ichunk):
#         to_write += 'python {cfg} --inputFiles={infiles} --logfreq=5000 --destination=/scratch/manzoni/{scratch_dir} --filename=bmmm_data_chunk{ijob} --filemode={filemode} \n'.format(
#             dir         = '/'.join([os.getcwd(), out_dir]), 
#             scratch_dir = out_dir, 
#             cfg         = 'inspector_b4m_analysis.py', 
#             ijob        = ijob, 
#             infiles     = ifile,
#             se_dir      = out_dir, 
#             filemode    = 'recreate' if idx==0 else 'update',     
#         )
    for idx, ifile in enumerate(ichunk):
        to_write += 'python3 {cfg} --inputFiles={infiles} --logfreq=5000 --destination=/scratch/manzoni/{scratch_dir} --filename=b4m_chunk{ijob}_part{idx} --logfile="b4m_logger{ijob}_part{idx}"\n'.format(
            dir         = '/'.join([os.getcwd(), out_dir]), 
            scratch_dir = out_dir, 
            cfg         = 'inspector_b4m_analysis.py', 
            ijob        = ijob, 
            infiles     = ifile,
            se_dir      = out_dir, 
            idx         = idx,
        )
        
    to_write += '\n'.join([
        '',
        'hadd -f -k /scratch/manzoni/{scratch_dir}/b4m_chunk{ijob}.root /scratch/manzoni/{scratch_dir}/b4m_chunk{ijob}_part*.root',
        'xrdcp /scratch/manzoni/{scratch_dir}/b4m_chunk{ijob}.root root://t3dcachedb.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/manzoni/{se_dir}/b4m_chunk{ijob}.root',
        #'xrdcp /scratch/manzoni/{scratch_dir}/b4m_logger{ijob}.txt root://t3dcachedb.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/manzoni/{se_dir}/logs/',
        'cp /scratch/manzoni/{scratch_dir}/b4m_logger{ijob}*.txt {dir}/cutflow/',
        'rm /scratch/manzoni/{scratch_dir}/b4m_chunk{ijob}*.root',
        'rm /scratch/manzoni/{scratch_dir}/b4m_logger{ijob}*.txt',
        '',
    ]).format(
        dir         = '/'.join([os.getcwd(), out_dir]), 
        scratch_dir = out_dir, 
        cfg         = 'inspector_b4m_analysis.py', 
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
    
    
    
