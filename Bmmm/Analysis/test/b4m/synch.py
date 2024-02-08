'''
1. create event lists from root ntuples

Attaching file b4mu_jpsi_phi_2022EE.root as _file0...
(TFile *) 0x2d61a90
root [1] tree->SetScanField(0)
root [2] .> scan_jpsi_phi_2022EE.txt
tree->Scan("run:lumi:event", "", "colsize=30")
.>
root [5] .q

2. clean the txt file from spurious gibberish at the top and at the bottom

3. run this script. Season to taste
'''

import pandas as pd
import numpy as np

bari = pd.read_csv('scan_jpsi_phi_2022EE_bari.txt', sep="*")
bari.columns = ['null1', 'idx', 'run', 'lumi', 'event', 'null2']
bari = bari.drop(['null1', 'idx', 'null2'], axis=1)
bari = bari.sort_values(by=['run', 'lumi', 'event'])

zuri = pd.read_csv('scan_jpsi_phi_2022EE_zuri.txt', sep="*")
zuri.columns = ['null1', 'idx', 'run', 'lumi', 'event', 'null2']
zuri = zuri.drop(['null1', 'idx', 'null2'], axis=1)
zuri = zuri.sort_values(by=['run', 'lumi', 'event'])

# https://stackoverflow.com/questions/19618912/finding-common-rows-intersection-in-two-pandas-dataframes
#intersection = pd.merge(bari, zuri, how='inner', on=['run', 'lumi', 'event'])

# https://stackoverflow.com/questions/28901683/pandas-get-rows-which-are-not-in-other-dataframe
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html
intersection = pd.merge(bari, zuri, on=['run', 'lumi', 'event'], how='outer', indicator=True)
intersection = intersection.sort_values(by=['run', 'lumi', 'event'])


# only in BARI
only_bari = (intersection['_merge'] == 'left_only')

# only in ZURI
only_zuri = (intersection['_merge'] == 'right_only')

# in BOTH
both = (intersection['_merge'] == 'both')


print('total events in BARI', len(bari))
print('total events in ZURI', len(zuri))

print('only in BARI', sum(only_bari))
print('only in ZURI', sum(only_zuri))
print('in both', sum(both))



