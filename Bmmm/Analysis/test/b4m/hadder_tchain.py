import ROOT
from glob import glob

directories = [

    'B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_2022EE_15mar24_v0',
    'B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_2022_15mar24_v0',

#    'B4Mu_ntuples_Bs0To4Mu_FourMuonFilter_2022EE_12mar24_v0',
#    'B4Mu_ntuples_Bs0To4Mu_FourMuonFilter_2022_12mar24_v0'  ,
#    'B4Mu_ntuples_BdTo4Mu_FourMuonFilter_2022EE_06mar24_v0'        ,
#    'B4Mu_ntuples_BdTo4Mu_FourMuonFilter_2022_06mar24_v0'          ,
#    'B4Mu_ntuples_Bs0To4Mu_FourMuonFilter_2022EE_06mar24_v0'       ,
#    'B4Mu_ntuples_Bs0To4Mu_FourMuonFilter_2022_06mar24_v0'         ,
#    'B4Mu_ntuples_BsToJpsiPhi_JMM_PhiMM_MuFilter_2022EE_06mar24_v0',
#    'B4Mu_ntuples_BsToJpsiPhi_JMM_PhiMM_MuFilter_2022_06mar24_v0'  ,
]

fouts = {}
#fouts['B4Mu_ntuples_BdTo4Mu_FourMuonFilter_2022EE_06mar24_v0'        ] = 'bd4mu_2022EE.root'
#fouts['B4Mu_ntuples_BdTo4Mu_FourMuonFilter_2022_06mar24_v0'          ] = 'bd4mu_2022.root'
#fouts['B4Mu_ntuples_Bs0To4Mu_FourMuonFilter_2022EE_06mar24_v0'       ] = 'bs4mu_2022EE.root'
#fouts['B4Mu_ntuples_Bs0To4Mu_FourMuonFilter_2022_06mar24_v0'         ] = 'bs4mu_2022.root'
#fouts['B4Mu_ntuples_BsToJpsiPhi_JMM_PhiMM_MuFilter_2022EE_06mar24_v0'] = 'bs_jpsi_phi_4mu_2022EE.root'
#fouts['B4Mu_ntuples_BsToJpsiPhi_JMM_PhiMM_MuFilter_2022_06mar24_v0'  ] = 'bs_jpsi_phi_4mu_2022.root'

#fouts['B4Mu_ntuples_Bs0To4Mu_FourMuonFilter_2022EE_12mar24_v0'] = 'bs4mu_2022EE_lifetime.root'
#fouts['B4Mu_ntuples_Bs0To4Mu_FourMuonFilter_2022_12mar24_v0'  ] = 'bs4mu_2022_lifetime.root'

fouts['B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_2022EE_15mar24_v0'] = 'bs_jpsi_phi_2mu2k_2022EE.root'
fouts['B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_2022_15mar24_v0'  ] = 'bs_jpsi_phi_2mu2k_2022.root'

for idir in directories:

    files = glob('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/' + idir + '/*root')

    tree = ROOT.TChain('tree')
    for ifile in files:
        tree.Add(ifile)
    
    print()
    print(fouts[idir], tree.GetEntries())
     
    tree.Merge(fouts[idir])   
    #fout = ROOT.TFile.Open(fouts[idir], 'recreate')
    #fout.cd()
    #tree.GetTree().Write()
    #fout.Close()
    

