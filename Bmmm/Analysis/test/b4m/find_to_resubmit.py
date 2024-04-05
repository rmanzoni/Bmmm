import os
import re
import glob
import uproot
import ROOT

pnfs = '/pnfs/psi.ch/cms/trivcat/store/user/manzoni/'

# 2022
directories = [
    'B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_2022EE_25mar24_v0',
    'B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_2022_25mar24_v0',
]

for directory in directories:
    
    print('#\n')
    print('#'+'>>>'*20)
    #print(directory)

    all_submitters = glob.glob('%s/submitter*.sh' %directory)
    all_submitters.sort()
    njobs = len(all_submitters)
    #print(njobs, 'chunks to test')
    
    bad_chunks = []
                              
    all_files_on_pnfs = glob.glob('/'.join([pnfs, directory, 'b2m2k_chunk*.root']))
    all_files_on_pnfs.sort()
    
    done_files = []
    
    for j, ifile in enumerate(all_files_on_pnfs):
        idx = re.findall(r'chunk\d+', ifile)[0].replace('chunk', '')
        idx = int(idx)
        done_files.append(idx)
    
    for i in range(njobs):
        if i not in done_files:
            continue
        with open('/'.join([directory, 'logs', 'chunk%d.log' %i])) as f:
            if 'ntuple saved, processed all desired events? True' not in f.read():
                #print('bad chunk', i)
                bad_chunks.append(i)
            else:
                pass
                #ff = uproot.open('/'.join([pnfs, directory, 'b4m_data_chunk%d.root' %i]))
                #events_in_tree = ff['tree'].num_entries
                #for iline in f.readlines():
                #    if 'number of selected events' in iline:
                #        events_in_log = re.findall(r'\d+', iline)[0]
                #    if events_in_tree != events_in_log:
                #        print('bad chunk, event number mismatch', i)
                #        bad_chunks.append(i)
                #    break
    
    missing_jobs = list(set(range(njobs)) - set(bad_chunks) - set(done_files))
    missing_jobs.sort()
    
    for ijob in missing_jobs:
        pass
        #print('missing chunk', ijob)
        
    to_resubmit = bad_chunks + missing_jobs
    to_resubmit.sort()
    
    print('#')
    print('#',directory)
    print('#%d jobs to resubmit' %(len(to_resubmit)))
    print('#to_resubmit = ', to_resubmit)


