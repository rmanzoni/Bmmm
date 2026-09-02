'''
J/psi mu (RJpsi) ntuplizer entry point: Bc -> J/psi(-> mu mu) mu.

All the logic lives in the package (Bmmm.Analysis.JpsiMuInspector, built on the
shared Bmmm.Analysis.JpsiChargedInspector), so this file only wires it up. The
produced ntuple is identical to the original inspector_rjpsi.py.

Examples:

ipython -i -- inspector_jpsi_mu.py --inputFiles=0443354B-2D3F-CF41-A1F0-0FC4F92E718E.root --filename=rjpsi_bc_signal_small --mc --maxevents=-1 --savenontrig

ipython -i -- inspector_jpsi_mu.py --inputFiles="root://cms-xrd-global.cern.ch///store/data/Run2022D/ParkingDoubleMuonLowMass0/MINIAOD/PromptReco-v1/000/357/539/00000/8abcc7e1-c6f0-4fcd-9be9-e07fb6878777.root" --filename=data_2022d_partial --maxevents=-1
'''

from Bmmm.Analysis.JpsiMuInspector import JpsiMuInspector

if __name__ == '__main__':
    JpsiMuInspector().main()
