'''
brilcalc trg --hltpath HLT_Mu8_v12 --prescale -o HLT_Mu8_v12.csv
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
        'HLT_Mu8_v12',
    ]

    for itrig in triggers: 
        
        prescales = []
        
        with open('%s.csv' %itrig) as csvfile:
            import pdb
            for i, line in enumerate(csvfile):
                if line.startswith('#'): continue
                row = line.split(',')
                run       = row[0]
                ini_ls    = row[1]
                ps_column = row[2]
                if run=='None' or ini_ls=='None': continue
                for iseed in row[6].rstrip().split(' '):
                    l1_seed = iseed.split('/')[0]
                    l1_ps   = iseed.split('/')[1]
                    prescales.append(PrescalePeriod(run, ini_ls, ps_column, l1_seed, l1_ps))
                
        prescales.sort(key = lambda x : (x.l1_seed, x.run, x.ini_ls))  
                        
        for j, ips in enumerate(prescales):
            if j<len(prescales)-1 and ips.run == prescales[j+1].run and ips.l1_seed == prescales[j+1].l1_seed:
                ips.end_ls = prescales[j+1].ini_ls - 1
            
        with open('%s.pickle' %itrig, 'wb') as handle:
            pickle.dump(prescales, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        #with open('filename.pickle', 'rb') as handle:
        #    b = pickle.load(handle)
