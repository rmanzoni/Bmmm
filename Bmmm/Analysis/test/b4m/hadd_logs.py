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

cutflow = defaultdict(zero)

#directory = 'BsToJPsiPhiTo4Mu_Run2022_26jan24_v1'
directory = 'BsToJPsiPhiTo4Mu_Run2022EE_26jan24_v1'

mypath = '/'.join([directory, 'cutflow'])

files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
files.sort()

pattern = r'\d+$'

# che schifo di codice
for ifile in files:
    #filename = 'b4m_logger7_part0.txt'
    idx = int(ifile.replace('_part0.txt', '').replace('b4m_logger', ''))
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


















