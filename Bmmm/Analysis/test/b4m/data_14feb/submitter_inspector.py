'''
Submitter for the SLURM system
'''
import os
import random
from glob import glob

resubmit = False

allfiles = []

for ipart in range(8):
    #if ipart>1: break # FIXME!
    #with open('../../files/files_ParkingDoubleMuonLowMass%d-Run2022C-PromptReco-v1.txt' %ipart) as f:
    #with open('../../files/files_ParkingDoubleMuonLowMass%d-Run2022D-PromptReco-v1.txt' %ipart) as f:
    with open('../../files/files_ParkingDoubleMuonLowMass%d-Run2022D-PromptReco-v2.txt' %ipart) as f:
    #with open('../../files/files_ParkingDoubleMuonLowMass%d-Run2022E-PromptReco-v1.txt' %ipart) as f:
    #with open('../../files/files_ParkingDoubleMuonLowMass%d-Run2022F-PromptReco-v1.txt' %ipart) as f:
    #with open('../../files/files_ParkingDoubleMuonLowMass%d-Run2022G-PromptReco-v1.txt' %ipart) as f:
        files = f.read().splitlines()
        files = ['root://cms-xrd-global.cern.ch//'+ifile for ifile in files]
        allfiles += files

files = allfiles
# random.shuffle(files)

files_per_job = 1
chunks = map(list, list(zip(*[iter(files)]*files_per_job)))


if len(files)%files_per_job!=0:
    last_idx = len(files)%files_per_job
    chunks += files[-last_idx:]

queue = 'standard'; time = 720
# queue = 'short'   ; time = 60
# queue = 'long'    ; time = 10080

time_tag = '15feb24'
version = 0
ntuplizer = 'inspector_b4m_analysis.py'

#out_dir = 'B4Mu_ntuples_ParkingDoubleMuonLowMass_Run2022C_PromptReco_v1_%s_v%d' %(time_tag, version)
#out_dir = 'B4Mu_ntuples_ParkingDoubleMuonLowMass_Run2022D_PromptReco_v1_%s_v%d' %(time_tag, version)
out_dir = 'B4Mu_ntuples_ParkingDoubleMuonLowMass_Run2022D_PromptReco_v2_%s_v%d' %(time_tag, version)
#out_dir = 'B4Mu_ntuples_ParkingDoubleMuonLowMass_Run2022E_PromptReco_v1_%s_v%d' %(time_tag, version)
#out_dir = 'B4Mu_ntuples_ParkingDoubleMuonLowMass_Run2022F_PromptReco_v1_%s_v%d' %(time_tag, version)
#out_dir = 'B4Mu_ntuples_ParkingDoubleMuonLowMass_Run2022G_PromptReco_v1_%s_v%d' %(time_tag, version)

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

os.system('cp %s %s' %(ntuplizer, out_dir))

import re
good_files = glob('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/%s/*root' %out_dir)
good_idx = []
for ifile in good_files:
    idx = int(re.findall(r'chunk\d+.root', ifile)[0].replace('chunk', '').replace('.root',''))
    good_idx.append(idx)
  
for ijob, ichunk in enumerate(chunks):
    
    #if ijob>3: break

    if resubmit and (ijob in good_idx): continue

    #if ijob>2: break
#     files = ['root://cms-xrd-global.cern.ch//'+ifile for ifile in files]
#     import pdb ; pdb.set_trace() 
    #cp_files = '\n'.join(['xrdcp root://cms-xrd-global.cern.ch//{ifile} /scratch/manzoni/{scratch_dir}/'.format(ifile=ifile               , scratch_dir=out_dir) for ifile in ichunk])
    #rm_files = '\n'.join(['rm /scratch/manzoni/{scratch_dir}/{ifile}'                                   .format(ifile=ifile.split('/')[-1], scratch_dir=out_dir) for ifile in ichunk])
    
    if not resubmit:
        to_write = '\n'.join([
            '#!/bin/bash',
            'cd {dir}',
            'echo "doing CMSENV"',
            'scramv1 runtime -sh',
            'echo $CMSSW_BASE',
            'echo "should have printed CMSENV"',
            'mkdir -p /scratch/manzoni/{scratch_dir}',
            'ls /scratch/manzoni/',
            'python3 {cfg} --inputFiles={infiles} --logfreq=5000 --destination=/scratch/manzoni/{scratch_dir} --filename=b4m_data_chunk{ijob} --logger=logger_b4m_data_chunk{ijob}',
            'xrdcp /scratch/manzoni/{scratch_dir}/b4m_data_chunk{ijob}.root root://t3dcachedb.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/manzoni/{se_dir}/b4m_data_chunk{ijob}.root',
            'cp /scratch/manzoni/{scratch_dir}/logger_b4m_data_chunk{ijob}.txt {dir}/cutflow/',
            'rm /scratch/manzoni/{scratch_dir}/b4m_data_chunk{ijob}*.root',
            'rm /scratch/manzoni/{scratch_dir}/b4m_data_chunk{ijob}*.txt',
            '',
        ]).format(
            dir         = '/'.join([os.getcwd(), out_dir]), 
            scratch_dir = out_dir, 
            cfg         = ntuplizer, 
            ijob        = ijob, 
            infiles     = ','.join(ichunk),
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
        '%s/submitter_chunk%d.sh' %(out_dir, ijob), 
    ])
    
    print(command_sh_batch)
    os.system(command_sh_batch)
    
    
    
