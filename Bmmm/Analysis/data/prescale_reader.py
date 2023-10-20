'''
https://cms-service-lumi.web.cern.ch/cms-service-lumi/brilwsdoc.html


python -m pip install brilws --user

ssh tunneling session
    ssh -N -L 10121:itrac5117-v.cern.ch:10121 <cernusername>@lxplus.cern.ch
Provide you cern account password and the tunnel will kept open without getting back the shell prompt.
open an execution session on the same host that has the ssh tunnel
    brilcalc lumi -c offsite -r 281636
Run bril tools with connection string -c offsite.


brilcalc trg --hltpath HLT_Mu8_v12 --prescale -o HLT_Mu8_v12.csv


brilcalc trg --hltpath HLT_Mu8_v\* --prescale --output-style=csv -o HLT_Mu8.csv
'''

import csv
import pickle
import numpy as np

class PrescalePeriod():
    def __init__(self, run, ini_ls, ps_column, l1_seed, l1_ps, end_ls=1e9):
        self.run       = long(run)      
        self.ini_ls    = long(ini_ls)   
        self.end_ls    = long(end_ls)   
        self.ps_column = int(ps_column)
        self.l1_seed   = l1_seed  
        self.l1_ps     = float(l1_ps) 
    def __str__(self):
        return 'run %d, initial LS %d, final LS %d, PS column %d, L1 seed %s, L1 PS %d' %(self.run, self.ini_ls, self.end_ls, self.ps_column, self.l1_seed, self.l1_ps)
        

if __name__ == '__main__':

    # add your own triggers
    triggers = [
        'HLT_Dimuon0_Jpsi',
        'HLT_Dimuon0_Jpsi_L1_4R_0er1p5R',
        'HLT_Dimuon0_Jpsi_L1_NoOS',
        'HLT_Dimuon0_Jpsi_NoVertexing',
        'HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R',
        'HLT_Dimuon0_Jpsi_NoVertexing_L1_NoOS',
        'HLT_DoubleMu4_3_Jpsi',
        'HLT_DoubleMu4_Jpsi_Displaced',
        'HLT_DoubleMu4_Jpsi_NoVertexing',
        'HLT_IsoMu24',
        'HLT_Mu17',
        'HLT_Mu19',
        'HLT_Mu7p5_L2Mu2_Jpsi',
        'HLT_Mu7p5_Track2_Jpsi',
        'HLT_Mu7p5_Track3p5_Jpsi',
        'HLT_Mu7p5_Track7_Jpsi',
        'HLT_Mu8',
    ]

    for itrig in triggers: 
        
        prescales = []
        
        with open('%s.csv' %itrig) as csvfile:
            for i, line in enumerate(csvfile):
                if line.startswith('#'): continue
                row = line.split(',')
                run       = row[0]
                ini_ls    = row[1]
                ps_column = row[2]
                if run=='None' or ini_ls=='None': continue
                # restrict to 2018
                if long(run) < 315193 or long(run) > 325175: continue
                for iseed in row[6].rstrip().split(' '):
                    l1_seed = iseed.split('/')[0]
                    l1_ps   = iseed.split('/')[1]
                    prescales.append(PrescalePeriod(run, ini_ls, ps_column, l1_seed, l1_ps))
                
        prescales.sort(key = lambda x : (x.l1_seed, x.run, x.ini_ls))  
                        
        for j, ips in enumerate(prescales):
            if j<len(prescales)-1 and ips.run == prescales[j+1].run and ips.l1_seed == prescales[j+1].l1_seed:
                ips.end_ls = prescales[j+1].ini_ls - 1

        prescales_dict = dict()

        for ips in prescales:
            if ips.l1_seed not in prescales_dict.keys():
                prescales_dict[ips.l1_seed] = dict()
            if ips.run not in prescales_dict[ips.l1_seed].keys():
                prescales_dict[ips.l1_seed][ips.run] = dict()
            if ips.end_ls<1e8: 
                end_ls = ips.end_ls + 1
            else:
                end_ls = ips.ini_ls + 1
            for ils in range(ips.ini_ls, end_ls):
                prescales_dict[ips.l1_seed][ips.run][ils] = ips.l1_ps
                    
        with open('%s.pickle' %itrig, 'wb') as handle:
            pickle.dump(prescales_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        #with open('filename.pickle', 'rb') as handle:
        #    b = pickle.load(handle)











