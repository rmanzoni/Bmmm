'''
Submitter for the SLURM system
'''
import os
import re
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

resubmit = False

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
#    'Run2022C-PromptReco-v1',
#    'Run2022D-PromptReco-v1',
#    'Run2022D-PromptReco-v2',
#    'Run2022E-PromptReco-v1',
    'Run2022F-PromptReco-v1',
#    'Run2022G-PromptReco-v1',
]



# obtain these with a different script
to_resubmit = {}
to_resubmit['Run2022C-PromptReco-v1'] = []
to_resubmit['Run2022D-PromptReco-v1'] = []
to_resubmit['Run2022D-PromptReco-v2'] = []
to_resubmit['Run2022E-PromptReco-v1'] = []
to_resubmit['Run2022F-PromptReco-v1'] = []
to_resubmit['Run2022G-PromptReco-v1'] = []

# queue = 'standard'; time = 720
queue = 'short'   ; time = 60
# queue = 'long'    ; time = 10080


time_tag = '26feb24'
version = 0
ntuplizer = 'inspector_b4m_analysis.py'
files_per_job = 1


for iperiod in periods:
    
    print('#'*80)
    print('\n', iperiod, '\n')

    out_dir_original = 'B4Mu_ntuples_ParkingDoubleMuonLowMass_%s_%s_v%d' %(iperiod, time_tag, version)
    out_dir_mini = out_dir_original
    out_dir = out_dir_original.replace('-', '_')    
    
    files = glob('/'.join([pnfs, 'B4Mu_ntuples_ParkingDoubleMuonLowMass_%s_MINIAOD'%iperiod, '*root']))
    files = ['root://t3dcachedb.psi.ch:1094//' + file for file in files]
            
    chunks = []
    ichunk = []
    for i, ifile in enumerate(files):
        if len(ichunk) < files_per_job and (i+1)<len(files):
            ichunk.append(ifile)
        elif i%files_per_job==0 and (i+1)!=len(files):
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
    #if not os.path.exists(out_dir):
    #    try:
    #        os.makedirs('/'.join([pnfs, out_dir]))
    #    except:
    #        print('pnfs directory exists')
    #    os.makedirs(out_dir)
    #    os.makedirs(out_dir + '/logs')
    #    os.makedirs(out_dir + '/errs')
    #    os.makedirs(out_dir + '/cutflow')
    #
    #    os.system('cp %s %s' %(ntuplizer, out_dir))
    
        
    #offset = glob('/'.join([pnfs, out_dir.replace('-', '_'), '*root']))
    offset = glob('/'.join([pnfs, out_dir, '*root']))
    offset = [re.findall(r'chunk\d+.root', ii)[0].replace('chunk', '').replace('.root', '') for ii in offset]
    offset = list(map(int, offset))
    offset.sort()
    offset = max(offset) + 1
        
    for ijob, ichunk in enumerate(chunks):
        
        ijob += offset
        
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
            #'-w t3wn70,t3wn71,t3wn72,t3wn73', # only the best nodes
            '%s/submitter_chunk%d.sh' %(out_dir, ijob), 
        ])
        
        print(command_sh_batch)
        if not testing: os.system(command_sh_batch)
        
        
        
    