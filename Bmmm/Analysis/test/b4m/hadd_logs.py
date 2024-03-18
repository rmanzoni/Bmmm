import os
import re
import ROOT
from os import listdir
from os.path import isfile, join
from collections import defaultdict

'''
all processed events 7361
pass HLT 3761
at least four muons 1562
	candidates after HLT and 4mu 3660
	pass mutual dz 3660
	pass mass cut 1387
	pass trigger match 1385
	pass secondary vertex 1379
at least one cand pass presel 1116
'''

def zero():
    return 0

#directory = 'BsToJPsiPhiTo4Mu_Run2022_26jan24_v1'
#directory = 'BsToJPsiPhiTo4Mu_Run2022EE_26jan24_v1'
#directory = 'B4Mu_ntuples_Bs0To4Mu_FourMuonFilter_2022_06mar24_v0'

directories =[
    'B4Mu_ntuples_BdTo4Mu_FourMuonFilter_2022EE_06mar24_v0'        ,
    'B4Mu_ntuples_BdTo4Mu_FourMuonFilter_2022_06mar24_v0'          ,
    'B4Mu_ntuples_Bs0To4Mu_FourMuonFilter_2022EE_06mar24_v0'       ,
    'B4Mu_ntuples_Bs0To4Mu_FourMuonFilter_2022_06mar24_v0'         ,
    'B4Mu_ntuples_BsToJpsiPhi_JMM_PhiMM_MuFilter_2022EE_06mar24_v0',
    'B4Mu_ntuples_BsToJpsiPhi_JMM_PhiMM_MuFilter_2022_06mar24_v0'  ,
]

for directory in directories:

    cutflow = defaultdict(zero)
    
    print('\n'*2)
    print('#'*80)
    print(directory)

    mypath = '/'.join([directory, 'cutflow'])
    
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    files.sort()
    
    pattern = r'\d+$'
    
    # che schifo di codice
    for ifile in files:
        #filename = 'b4m_logger7_part0.txt'
        #idx = int(ifile.replace('_part0.txt', '').replace('b4m_logger', ''))
        idx = int(ifile.replace('logger_b4m_chunk', '').replace('.txt', ''))
        tree = ROOT.TChain('tree')
        tree.Add('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/%s/b4m_chunk%d.root' %(directory, idx))
        entries = tree.GetEntries()
        #print('chunk %d - entries %.0f' %(idx, entries))
    
        with open('/'.join([mypath,ifile])) as f:
            content = f.readlines()
            #import pdb ; pdb.set_trace()
            
            # check if final numer of entries in log and root file is identical
            final_entries = int(re.findall(pattern, content[-1].rstrip(), re.MULTILINE)[-1])
            
            if entries != final_entries:
                print('WARNING! Mismatch in chunk %d :'%idx, '\troot file entries', entries, '\tlog entries', final_entries, 'NOT ADDING!')
                continue
            else:
                for i in content: 
                    i = i.rstrip()
                    matches = re.findall(pattern, i, re.MULTILINE)    
                    cutflow[i.split(matches[-1])[0]] += float(matches[-1])
     
    logger_name = '/'.join([directory, 'full_logger_%s.txt' %directory])
    with open(logger_name, 'w') as logger:
        for k, v in cutflow.items():
            print(k, v, file=logger)
            
        ini = list(cutflow.values())[0]
        fin = list(cutflow.values())[-1]
        print('\n')
        print('reco efficiency %.4f' %(fin/ini), file=logger)    

















