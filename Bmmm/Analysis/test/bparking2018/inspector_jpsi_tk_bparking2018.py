'''
J/psi + track ntuplizer entry point for the 2018 B-parking data (ParkingBPH1-6,
UL2018 MiniAODv2): B+ -> J/psi K+ and Bc+ -> J/psi pi+ built as ONE candidate
carrying both mass hypotheses (kaon unprefixed, pion pi_).

Copy of test/rjpsi/inspector_jpsi_tk.py with the 2018 trigger: the event must
fire at least one HLT_Mu*_IP* parking path. All the logic lives in the package
(Bmmm.Analysis.JpsiTkBParking2018Inspector, a thin subclass of JpsiTkInspector),
so this file only wires it up. See that module for the trigger branches.

Data needs a certification mask (the inspector refuses to start without one):

python3 inspector_jpsi_tk_bparking2018.py --inputFiles=<file.root or LFN> \
    --filename=jpsi_tk_bph2018 --lumi-mask=golden --maxevents=2000 --logfreq=100
'''

from Bmmm.Analysis.JpsiTkBParking2018Inspector import JpsiTkBParking2018Inspector

if __name__ == '__main__':
    JpsiTkBParking2018Inspector().main()
