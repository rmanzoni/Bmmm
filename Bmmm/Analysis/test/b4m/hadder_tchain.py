import ROOT
from glob import glob

directories = [
    'B2Mu2K_ntuples_BdToJpsiKstar_BMuonFilter_2022EE_10apr24_v0',
    'B2Mu2K_ntuples_BdToJpsiKstar_BMuonFilter_2022_10apr24_v0',
    'B2Mu2K_ntuples_BdToJpsiKstar_BMuonFilter_2023BPix_10apr24_v0',
    'B2Mu2K_ntuples_BdToJpsiKstar_BMuonFilter_2023_10apr24_v0',
    'B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_2022EE_10apr24_v0',
    'B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_2022_10apr24_v0',
    'B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_2023BPix_10apr24_v0',
    'B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_2023_10apr24_v0',
    'B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_PtEtaFilter_2022EE_10apr24_v0',
    'B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_PtEtaFilter_2022EE_ext_10apr24_v0',
    'B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_PtEtaFilter_2022_10apr24_v0',
    'B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_PtEtaFilter_2022_ext_10apr24_v0',
    'B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_PtEtaFilter_2023BPix_10apr24_v0',
    'B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_PtEtaFilter_2023_10apr24_v0',
]

fouts = {}

fouts['B2Mu2K_ntuples_BdToJpsiKstar_BMuonFilter_2022EE_10apr24_v0'                     ] = 'BdToJpsiKstar_BMuonFilter_2022EE.root'
fouts['B2Mu2K_ntuples_BdToJpsiKstar_BMuonFilter_2022_10apr24_v0'                       ] = 'BdToJpsiKstar_BMuonFilter_2022.root'
fouts['B2Mu2K_ntuples_BdToJpsiKstar_BMuonFilter_2023BPix_10apr24_v0'                   ] = 'BdToJpsiKstar_BMuonFilter_2023BPix.root'
fouts['B2Mu2K_ntuples_BdToJpsiKstar_BMuonFilter_2023_10apr24_v0'                       ] = 'BdToJpsiKstar_BMuonFilter_2023.root'
fouts['B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_2022EE_10apr24_v0'                ] = 'BsToJPsiPhi_JPsiToMuMu_PhiToKK_2022EE.root'
fouts['B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_2022_10apr24_v0'                  ] = 'BsToJPsiPhi_JPsiToMuMu_PhiToKK_2022.root'
fouts['B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_2023BPix_10apr24_v0'              ] = 'BsToJPsiPhi_JPsiToMuMu_PhiToKK_2023BPix.root'
fouts['B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_2023_10apr24_v0'                  ] = 'BsToJPsiPhi_JPsiToMuMu_PhiToKK_2023.root'
fouts['B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_PtEtaFilter_2022EE_10apr24_v0'    ] = 'BsToJPsiPhi_JPsiToMuMu_PhiToKK_PtEtaFilter_2022EE.root'
fouts['B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_PtEtaFilter_2022EE_ext_10apr24_v0'] = 'BsToJPsiPhi_JPsiToMuMu_PhiToKK_PtEtaFilter_2022EE_ext.root'
fouts['B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_PtEtaFilter_2022_10apr24_v0'      ] = 'BsToJPsiPhi_JPsiToMuMu_PhiToKK_PtEtaFilter_2022.root'
fouts['B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_PtEtaFilter_2022_ext_10apr24_v0'  ] = 'BsToJPsiPhi_JPsiToMuMu_PhiToKK_PtEtaFilter_2022_ext.root'
fouts['B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_PtEtaFilter_2023BPix_10apr24_v0'  ] = 'BsToJPsiPhi_JPsiToMuMu_PhiToKK_PtEtaFilter_2023BPix.root'
fouts['B2Mu2K_ntuples_BsToJPsiPhi_JPsiToMuMu_PhiToKK_PtEtaFilter_2023_10apr24_v0'      ] = 'BsToJPsiPhi_JPsiToMuMu_PhiToKK_PtEtaFilter_2023.root'


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
    

