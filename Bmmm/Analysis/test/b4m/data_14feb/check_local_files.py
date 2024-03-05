import os
from glob import glob


files = {}

#files['Run2023B-PromptReco-v1'] = glob('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/B4Mu_ntuples_ParkingDoubleMuonLowMass_Run2023B-PromptReco-v1_MINIAOD/*root')
#files['Run2023C-PromptReco-v1'] = glob('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/B4Mu_ntuples_ParkingDoubleMuonLowMass_Run2023C-PromptReco-v1_MINIAOD/*root')
#files['Run2023C-PromptReco-v2'] = glob('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/B4Mu_ntuples_ParkingDoubleMuonLowMass_Run2023C-PromptReco-v2_MINIAOD/*root')
#files['Run2023D-PromptReco-v1'] = glob('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/B4Mu_ntuples_ParkingDoubleMuonLowMass_Run2023D-PromptReco-v1_MINIAOD/*root')
#files['Run2023D-PromptReco-v2'] = glob('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/B4Mu_ntuples_ParkingDoubleMuonLowMass_Run2023D-PromptReco-v2_MINIAOD/*root')



#files['Run2022C_PromptReco_v1'] = glob('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/B4Mu_ntuples_ParkingDoubleMuonLowMass_Run2022C-PromptReco-v1_MINIAOD/*root')
#files['Run2022D_PromptReco_v1'] = glob('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/B4Mu_ntuples_ParkingDoubleMuonLowMass_Run2022D-PromptReco-v1_MINIAOD/*root')
#files['Run2022D_PromptReco_v2'] = glob('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/B4Mu_ntuples_ParkingDoubleMuonLowMass_Run2022D-PromptReco-v2_MINIAOD/*root')
files['Run2022E_PromptReco_v1'] = glob('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/B4Mu_ntuples_ParkingDoubleMuonLowMass_Run2022E-PromptReco-v1_MINIAOD/*root')
#files['Run2022F_PromptReco_v1'] = glob('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/B4Mu_ntuples_ParkingDoubleMuonLowMass_Run2022F-PromptReco-v1_MINIAOD/*root')
#files['Run2022G_PromptReco_v1'] = glob('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/B4Mu_ntuples_ParkingDoubleMuonLowMass_Run2022G-PromptReco-v1_MINIAOD/*root')


for iperiod, ifiles in files.items():
        
    print('#'*80 + '\n' + iperiod + '\n')
    counter = 1
    
    for ifile in ifiles:
        print('\t%d' %counter)
        #print('\t' + ifile)
        #os.system('edmFileUtil file:%s | grep "runs" | grep "lumis" | grep "events"' %ifile)
        os.system('edmFileUtil file:%s | grep -i error' %ifile)
        counter +=1


