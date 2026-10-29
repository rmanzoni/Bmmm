'''
Example:

ipython -i -- inspector_rjpsi.py --inputFiles="root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/HbToPsiX_JMM_TuneCP5_13TeV-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2540000/D388E03F-0214-E842-9905-26008B393E50.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/HbToPsiX_JMM_TuneCP5_13TeV-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2540000/261075CA-133E-7A4E-B619-8846615BBD44.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/HbToPsiX_JMM_TuneCP5_13TeV-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/50000/C82BE32E-7DC6-394D-A213-04F8E087875A.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/HbToPsiX_JMM_TuneCP5_13TeV-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/50000/11B8EB1F-C40C-1441-AD14-B9ABBF52B97F.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/HbToPsiX_JMM_TuneCP5_13TeV-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2520000/1EC3014E-488E-AB44-9F1A-20821D6C0189.root" --filename=rjpsi_hb --mc --maxevents=-1

ipython -i -- inspector_rjpsi.py --inputFiles="root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/BcToJPsiMuMu_inclusive_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/230000/0443354B-2D3F-CF41-A1F0-0FC4F92E718E.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/BcToJPsiMuMu_inclusive_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/230000/05C316B4-BD3D-CC4E-8BDB-C603259F1016.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/BcToJPsiMuMu_inclusive_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/230000/0D7329CE-CC25-1A4C-848D-CDF63DA314B5.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/BcToJPsiMuMu_inclusive_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/230000/127F4AE1-CE44-6E4D-95A5-C25986F28A1B.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/BcToJPsiMuMu_inclusive_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/230000/17FDD58D-FEAD-5B49-83DD-3A8C1C1A3960.root,root://cms-xrd-global.cern.ch///store/mc/RunIISummer20UL18MiniAODv2/BcToJPsiMuMu_inclusive_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/230000/189B54A2-6B0D-ED49-A7FA-A94DF21D41A0.root" --filename=rjpsi_bc_signal --mc --maxevents=-1 --savenontrig

ipython -i -- inspector_rjpsi.py --inputFiles=0443354B-2D3F-CF41-A1F0-0FC4F92E718E.root --filename=rjpsi_bc_signal_small --mc --maxevents=-1 --savenontrig




    

ipython -i -- inspector_rjpsi.py --inputFiles=root://cms-xrd-global.cern.ch///store/data/Run2022D/ParkingDoubleMuonLowMass0/MINIAOD/PromptReco-v1/000/357/539/00000/8abcc7e1-c6f0-4fcd-9be9-e07fb6878777.root,root://cms-xrd-global.cern.ch///store/data/Run2022D/ParkingDoubleMuonLowMass0/MINIAOD/PromptReco-v1/000/357/542/00000/c5941f20-7f8f-40ae-976c-cb354a3f1a06.root,root://cms-xrd-global.cern.ch///store/data/Run2022D/ParkingDoubleMuonLowMass0/MINIAOD/PromptReco-v1/000/357/542/00000/fc709d38-043c-4529-85fb-567ca09214a2.root,root://cms-xrd-global.cern.ch///store/data/Run2022D/ParkingDoubleMuonLowMass0/MINIAOD/PromptReco-v1/000/357/542/00000/6af139f3-2242-42ce-b2a5-09387387c1db.root,root://cms-xrd-global.cern.ch///store/data/Run2022D/ParkingDoubleMuonLowMass0/MINIAOD/PromptReco-v1/000/357/550/00000/9f95cf6b-dfb7-466a-9ab4-6213ffd5b080.root,root://cms-xrd-global.cern.ch///store/data/Run2022D/ParkingDoubleMuonLowMass0/MINIAOD/PromptReco-v1/000/357/542/00000/87bf23a8-e77d-44a6-8623-961db9319eda.root,root://cms-xrd-global.cern.ch///store/data/Run2022D/ParkingDoubleMuonLowMass0/MINIAOD/PromptReco-v1/000/357/542/00000/38d1ee7b-83c0-4e86-8510-204b7f737e7f.root,root://cms-xrd-global.cern.ch///store/data/Run2022D/ParkingDoubleMuonLowMass0/MINIAOD/PromptReco-v1/000/357/542/00000/ecdd5c5f-a1cd-418e-a4a1-e77c53db4d3a.root,root://cms-xrd-global.cern.ch///store/data/Run2022D/ParkingDoubleMuonLowMass0/MINIAOD/PromptReco-v1/000/357/542/00000/66783f3e-8d0d-43ec-97fe-c10dfa92e095.root --filename=data_2022d_partial --maxevents=-1 


'''

# ---------------------------------------------------------------------------
# RETIRED IMPLEMENTATION -> thin J/psi mu entry point.
#
# The RJpsi ntuplizer has been refactored into the package: the machinery now
# lives in Bmmm.Analysis.JpsiChargedInspector (shared) + Bmmm.Analysis.
# JpsiMuInspector (this channel), and the candidate/branches/cuts in the
# Jpsi{Charged,Mu}{Candidate,Branches,Cuts} modules. This file is kept as a
# working entry point (the CRAB/pnfs submitters ship it by name); it produces
# the SAME ntuple as before. New code should use inspector_jpsi_mu.py.
# ---------------------------------------------------------------------------

from Bmmm.Analysis.JpsiMuInspector import JpsiMuInspector

if __name__ == '__main__':
    JpsiMuInspector().main()
