'''
Submitter for the SLURM system
'''
import os
import random
from glob import glob
from copy import copy

#######################################################################
#######################################################################
#######################################################################
####                      _               _         _              ####
####                     | |             (_)       (_)             ####
####  _ __ ___  ___ _   _| |__  _ __ ___  _ ___ ___ _  ___  _ __   ####
#### | '__/ _ \/ __| | | | '_ \| '_ ` _ \| / __/ __| |/ _ \| '_ \  ####
#### | | |  __/\__ \ |_| | |_) | | | | | | \__ \__ \ | (_) | | | | ####
#### |_|  \___||___/\__,_|_.__/|_| |_| |_|_|___/___/_|\___/|_| |_| ####
####                                                               ####
#######################################################################
#######################################################################
#######################################################################

resubmit = True

#######################################################################
#######################################################################
#######################################################################

testing = False

if testing:
    print('TESTING - NO EXECUTION!')
else:
    print('WILL EXECUTE THE SUBMISSION!')

#######################################################################
#######################################################################
#######################################################################

pnfs = '/pnfs/psi.ch/cms/trivcat/store/user/manzoni'

periods = [
    'Run2023B-PromptReco-v1',
    'Run2023C-PromptReco-v1',
    'Run2023C-PromptReco-v2',
    'Run2023C-PromptReco-v3',
    'Run2023C-PromptReco-v4',
    'Run2023D-PromptReco-v1',
    'Run2023D-PromptReco-v2',
]

# obtain these with a different script
to_resubmit = {}
to_resubmit['Run2023B-PromptReco-v1'] = [45, 47, 48, 49]
to_resubmit['Run2023C-PromptReco-v1'] = [700]
to_resubmit['Run2023C-PromptReco-v2'] = [202, 203, 204]
to_resubmit['Run2023C-PromptReco-v3'] = []
to_resubmit['Run2023C-PromptReco-v4'] = []
to_resubmit['Run2023D-PromptReco-v1'] = [5, 6, 7, 13, 14, 15, 16, 91, 96, 98, 99, 102, 106, 107, 108, 722]
to_resubmit['Run2023D-PromptReco-v2'] = [88, 91, 92, 99, 108, 237]

queue = 'standard'; time = 720
# queue = 'short'   ; time = 60
# queue = 'long'    ; time = 10080


time_tag = '15feb24'
version = 0
ntuplizer = 'inspector_b4m_analysis.py'
files_per_job = 10


for iperiod in periods:
    
    print('#'*80)
    print('\n', iperiod, '\n')

    out_dir = 'B4Mu_ntuples_ParkingDoubleMuonLowMass_%s_%s_v%d' %(iperiod, time_tag, version)

    if not resubmit:

        allfiles = []
    
        for ipart in range(8):
            with open('../../files/files_ParkingDoubleMuonLowMass%d-%s.txt' %(ipart, iperiod)) as f:
                files = f.read().splitlines()
                files = ['root://cms-xrd-global.cern.ch//'+ifile for ifile in files]
                allfiles += files
        
        files = allfiles
                
        chunks = []
        ichunk = []
        for i, ifile in enumerate(files):
            if len(ichunk) < files_per_job and (i+1)<len(files):
                ichunk.append(ifile)
            elif i%files_per_job==0:
                chunks.append(copy(ichunk))
                ichunk = []
                ichunk.append(ifile)
            elif (i+1)==len(files):
                ichunk.append(ifile)
                chunks.append(copy(ichunk))
                ichunk = []
            else:
                print('SOMETHING WRONG!')
            
        # validation
        check_files = []
        for ichunk in chunks:
            for ifile in ichunk:
                check_files.append(ifile)
        
        if check_files == files:    
            print('CHUNKS VALIDATED')
        else:
            print('CHUNKS ARE BROKEN!')
                
        ##########################################################################################
        ##########################################################################################
        
        # make output dir
        if not os.path.exists(out_dir):
            try:
                os.makedirs('/'.join([pnfs, out_dir]))
            except:
                print('pnfs directory exists')
            os.makedirs(out_dir)
            os.makedirs(out_dir + '/logs')
            os.makedirs(out_dir + '/errs')
            os.makedirs(out_dir + '/cutflow')
        
            os.system('cp %s %s' %(ntuplizer, out_dir))
        
    if resubmit:
        print('\nresubmitting %d chunks' %(len(to_resubmit[iperiod])))
        for idx in to_resubmit[iperiod]:

            # jobs are still runnning
            if idx in [1359, 85, 159, 488, 292, 91]: continue

            print('rm %s/%s/b4m_data_chunk%d.root' %(pnfs, out_dir, idx))
            if not testing: os.system('rm %s/%s/b4m_data_chunk%d.root' %(pnfs, out_dir, idx))
            command_sh_batch = ' '.join([
                'sbatch', 
                '-p %s'%queue, 
                '--account=t3', 
                '-o %s/logs/chunk%d.log' %(out_dir, idx),
                '-e %s/errs/chunk%d.err' %(out_dir, idx), 
                '--job-name=%d_%s' %(idx, out_dir), 
                '--time=%d'%time,
                #'-w t3wn70,t3wn71,t3wn72,t3wn73', # only the best nodes
                '%s/submitter_chunk%d.sh' %(out_dir, idx), 
            ])
            print(command_sh_batch)
            if not testing: os.system(command_sh_batch)
    else:
        for ijob, ichunk in enumerate(chunks):
                
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
                '-w t3wn70,t3wn71,t3wn72,t3wn73', # only the best nodes
                '%s/submitter_chunk%d.sh' %(out_dir, ijob), 
            ])
            
            print(command_sh_batch)
            if not testing: os.system(command_sh_batch)
        
        
        
    