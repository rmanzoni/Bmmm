import os
import json
import subprocess
from collections import defaultdict

def zero():
    return []
    
filelist = 'files_jpsi_phi_4mu_2022EE'

with open('%s.txt' % filelist) as ff:
    files = [ifile.rstrip() for ifile in ff.readlines()]

file_lumi_dict = defaultdict(zero)

command = 'dasgoclient -query="lumi file=MYFILE"'

tofind = [122, 161, 9043]

for ifile in files:

    lumis = subprocess.check_output(command.replace('MYFILE', ifile), shell=True)
    # not sure there isn't a better way...
    lumis = [ilumi for ilumi in str(lumis).replace("b'","").replace("'","").rstrip().split("\\n")]
    lumis = [int(ilumi) for ilumi in lumis if len(ilumi)>0]
    
    file_lumi_dict[ifile] = lumis

    if set(tofind) & set(lumis):
        print(ifile, lumis)

with open('%s.json' %filelist, 'w', encoding='utf-8') as f:
    json.dump(file_lumi_dict, f, ensure_ascii=False, indent=4)



