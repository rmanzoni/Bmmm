'''
Submitter for the SLURM system
'''

import os
import random
from glob import glob

resubmit = True

old_files = []

for ipart in range(6):
    with open('files_2022E_prompt_v1_part%d_07oct22v2.txt' %ipart) as f:
        ifiles = f.read().splitlines()
        #ifiles = ['root://cms-xrd-global.cern.ch//'+ifile for ifile in ifiles]
        old_files += ifiles

files = []

for ipart in range(6):
    with open('files_2022E_prompt_v1_part%d_09oct22.txt' %ipart) as f:
        ifiles = f.read().splitlines()
        ifiles = ['root://cms-xrd-global.cern.ch//'+ifile for ifile in ifiles if ifile not in old_files]
        files += ifiles

# for ipart in range(6):
#     with open('files_2022C_prompt_v1_part%d_12sept22.txt' %ipart) as f:
#         ifiles = f.read().splitlines()
#         ifiles = ['root://cms-xrd-global.cern.ch//'+ifile for ifile in ifiles if ifile not in old_files]
#         files += ifiles
#     with open('files_2022D_prompt_v1_part%d_12sept22.txt' %ipart) as f:
#         ifiles = f.read().splitlines()
#         ifiles = ['root://cms-xrd-global.cern.ch//'+ifile for ifile in ifiles if ifile not in old_files]
#         files += ifiles
#     with open('files_2022D_prompt_v2_part%d_12sept22.txt' %ipart) as f:
#         ifiles = f.read().splitlines()
#         ifiles = ['root://cms-xrd-global.cern.ch//'+ifile for ifile in ifiles if ifile not in old_files]
#         files += ifiles

# random.shuffle(files)

files_per_job = 1
chunks = list(map(list, list(zip(*[iter(files)]*files_per_job))))

if len(files)%files_per_job!=0:
    last_idx = len(files)%files_per_job
    chunks += [files[-last_idx:]]

queue = 'standard'; time = 720
# queue = 'short'   ; time = 60
# queue = 'long'    ; time = 10080

#out_dir = 'Kee_DoubleEleLow_12sep22_v1'
#out_dir = 'Kee_DoubleEleLow_06oct22_v1'
#out_dir = 'Kee_DoubleEleLow_07oct22_v1'
#out_dir = 'Kee_DoubleEleLow_07oct22_v2' # file extra apparsi nel pomeriggio
out_dir = 'Kee_DoubleEleLow_09oct22_v1' # file extra apparsi nel pomeriggio

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

os.system('cp inspector_kee_analysis.py %s' %out_dir)
# os.system('cp decays.py %s' %out_dir)

if resubmit:
    import subprocess
    import re
    result = subprocess.check_output('ls /pnfs/psi.ch/cms/trivcat/store/user/manzoni/%s' %out_dir, shell=True)
    rootfiles = [int(ii) for ii in re.findall(r'\d+', str(result))]
    
    failed = []
    for ifile in rootfiles:
        myfile = '/pnfs/psi.ch/cms/trivcat/store/user/manzoni/%s/kee_data_chunk%d.root' %(out_dir, ifile)
        if os.path.getsize(myfile)<1200:
            failed.append(ifile)
    
    missing = list(set(range(len(files))) - set(rootfiles))
    toresubmit = list(set(failed + missing))
    
    for jj in toresubmit:
        myfile = '/pnfs/psi.ch/cms/trivcat/store/user/manzoni/%s/kee_data_chunk%d.root' %(out_dir, jj)
        print('removing %s' %myfile)    
        os.system('rm -f %s' %myfile)

# for ijob, ichunk in enumerate(chunks[:50]):
for ijob, ichunk in enumerate(chunks):
    
    if resubmit:
        if ijob not in toresubmit: continue
       #if ijob not in failed: continue
    
#    if ijob<=4: continue
#    if ijob>4: break
        
    to_write = '\n'.join([
        '#!/bin/bash',
        'cd {dir}',
        'eval `scramv1 runtime -sh`',
        'mkdir -p /scratch/manzoni/{scratch_dir}',
        'ls /scratch/manzoni/',
        '',
    ]).format(
        dir         = '/'.join([os.getcwd(), out_dir]), 
        scratch_dir = out_dir, 
        cfg         = 'inspector_kee_analysis.py', 
        ijob        = ijob, 
        infiles     = ','.join(['/scratch/manzoni/{scratch_dir}/{ifile}'.format(scratch_dir=out_dir, ifile=ifile) for ifile in ichunk]),
        se_dir      = out_dir,
        )
        
    for idx, ifile in enumerate(ichunk):
        to_write += 'ipython -- {dir}/{cfg} --inputFiles={infiles} --logfreq=5000 --destination=/scratch/manzoni/{scratch_dir} --filename=kee_data_chunk{ijob}_part{idx} \n'.format(
            dir         = '/'.join([os.getcwd(), out_dir]), 
            scratch_dir = out_dir, 
            cfg         = 'inspector_kee_analysis.py', 
            ijob        = ijob, 
            infiles     = ifile,
            se_dir      = out_dir, 
            idx         = idx,
        )
        
    to_write += '\n'.join([
        '',
        'ls -latrh /scratch/manzoni/{scratch_dir}',
        'hadd -f -k /scratch/manzoni/{scratch_dir}/kee_data_chunk{ijob}.root /scratch/manzoni/{scratch_dir}/kee_data_chunk{ijob}_part*.root',
        'xrdcp /scratch/manzoni/{scratch_dir}/kee_data_chunk{ijob}.root root://t3dcachedb.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/manzoni/{se_dir}/kee_data_chunk{ijob}.root',
        'rm /scratch/manzoni/{scratch_dir}/kee_data_chunk{ijob}*.root',
        '',
    ]).format(
        dir         = '/'.join([os.getcwd(), out_dir]), 
        scratch_dir = out_dir, 
        cfg         = 'inspector_kee_analysis.py', 
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
        # '-w t3wn70,t3wn71,t3wn72,t3wn73', # only the best nodes
        '%s/submitter_chunk%d.sh' %(out_dir, ijob), 
    ])

    print(command_sh_batch)
    os.system(command_sh_batch)