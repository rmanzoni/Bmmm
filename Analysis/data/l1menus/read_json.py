import pickle
import json

with open("goodRuns2013to2022ByYear.json") as f:
   #data = json.read(f)
   data = json.load(f)

menus = {}
for run in data["2018"]:
    menus.setdefault(run["l1_menu"],[]).append(run["run_number"])
    



# here the correspondence 
# https://twiki.cern.ch/twiki/bin/view/CMS/GlobalTriggerAvailableMenus

menu_names = {}
#menu_names['L1Menu_Collisions2017_v4_r2_m6' ] = 'L1Menu_Collisions2017_v4_r2'
menu_names['L1Menu_Collisions2018_v0_0_1-d1'] = 'L1Menu_Collisions2018_v0_0_1'
menu_names['L1Menu_Collisions2018_v1_0_0-d1'] = 'L1Menu_Collisions2018_v1_0_0'
#menu_names['L1Menu_Collisions2018_v2_0_0-d1'] = 'L1Menu_Collisions2018_v2_0_0'
menu_names['L1Menu_Collisions2018_v2_0_0-d2'] = 'L1Menu_Collisions2018_v2_0_0'
menu_names['L1Menu_Collisions2018_v2_1_0-d1'] = 'L1Menu_Collisions2018_v2_1_0'

for imenu in menu_names.keys():
    
    my_menu = {}
    with open('%s.xml' %imenu) as f:
        seeds = []
        indices = []
        
        for line in f:
            if '<name>' in line and '</name>' in line:
                seed = line.rstrip().replace('<name>', '').replace('</name>', '').replace(' ','')
                if seed.startswith('L1_'):
                    seeds.append(seed)
            if '<index>' in line and '</index>' in line:
                idx = line.rstrip().replace('<index>', '').replace('</index>', '').replace(' ','')
                indices.append(int(idx))
        
        for i in range(len(seeds)):
            my_menu[seeds[i]] = indices[i]
            
        with open('%s.pickle' %menu_names[imenu], 'wb') as handle:
            pickle.dump(my_menu, handle, protocol=pickle.HIGHEST_PROTOCOL)

    



