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
    'BsToJPsiPhi_JPsiToMuMu_PhiToKK_2022'  ,
    'BsToJPsiPhi_JPsiToMuMu_PhiToKK_2022EE',
]


# obtain these with a different script
to_resubmit = {}
to_resubmit['BsToJPsiPhi_JPsiToMuMu_PhiToKK_2022'  ] = []
to_resubmit['BsToJPsiPhi_JPsiToMuMu_PhiToKK_2022EE'] = []


sample_files = {}
sample_files['BsToJPsiPhi_JPsiToMuMu_PhiToKK_2022'  ] = 'files_BsToJPsiPhi_JPsiToMuMu_PhiToKK_SoftQCDnonD_TuneCP5_13p6TeV_pythia8-evtgen__Run3Summer22MiniAODv4-130X_mcRun3_2022_realistic_v5-v2.txt'
sample_files['BsToJPsiPhi_JPsiToMuMu_PhiToKK_2022EE'] = 'files_BsToJPsiPhi_JPsiToMuMu_PhiToKK_SoftQCDnonD_TuneCP5_13p6TeV_pythia8-evtgen__Run3Summer22EEMiniAODv4-130X_mcRun3_2022_realistic_postEE_v6-v2.txt'

queue = 'standard'; time = 720
# queue = 'short'   ; time = 60
# queue = 'long'    ; time = 10080

time_tag = '08mar24'
version = 0
ntuplizer = 'inspector_b2m2k_analysis.py'
output = 'b2m2k'
files_per_job = 2


for iperiod in periods:
    
    print('#'*80)
    print('\n', iperiod, '\n')

    #out_dir = 'B4Mu_ntuples_%s_%s_v%d' %(iperiod.replace('-', '_'), time_tag, version)
    out_dir = 'B2Mu2K_ntuples_%s_%s_v%d' %(iperiod.replace('-', '_'), time_tag, version)

    if not resubmit:

        allfiles = []
        
        base_cms_dir = os.environ['CMSSW_BASE']
        files_dir = '/'.join([base_cms_dir, 'src', 'Bmmm', 'Analysis', 'test', 'files'])
    
        print('looking for files here', files_dir)
            
        with open('%s/%s' %(files_dir, sample_files[iperiod])) as f:
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
            #if idx in [1359, 85, 159, 488, 292, 91]: continue
            
            print('rm %s/%s/%s_chunk%d.root' %(pnfs, out_dir, output, idx))
            if not testing: os.system('rm %s/%s/%s_chunk%d.root' %(pnfs, out_dir, output, idx))
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
                'python3 {cfg} --inputFiles={infiles} --logfreq=5000 --destination=/scratch/manzoni/{scratch_dir} --filename={out_file}_chunk{ijob} --logger=logger_{out_file}_chunk{ijob} --mc',
                'xrdcp /scratch/manzoni/{scratch_dir}/{out_file}_chunk{ijob}.root root://t3dcachedb.psi.ch:1094///pnfs/psi.ch/cms/trivcat/store/user/manzoni/{se_dir}/{out_file}_chunk{ijob}.root',
                'cp /scratch/manzoni/{scratch_dir}/logger_{out_file}_chunk{ijob}.txt {dir}/cutflow/',
                'rm /scratch/manzoni/{scratch_dir}/{out_file}_chunk{ijob}*.root',
                'rm /scratch/manzoni/{scratch_dir}/{out_file}_chunk{ijob}*.txt',
                '',
            ]).format(
                dir         = '/'.join([os.getcwd(), out_dir]), 
                scratch_dir = out_dir, 
                cfg         = ntuplizer, 
                ijob        = ijob, 
                infiles     = ','.join(ichunk),
                se_dir      = out_dir,
                out_file    = output,
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
        
    