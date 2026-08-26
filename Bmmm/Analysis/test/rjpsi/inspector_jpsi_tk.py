'''
J/psi + track ntuplizer entry point: B+ -> J/psi K+  and  Bc+ -> J/psi pi+ built
as ONE candidate carrying both mass hypotheses (kaon unprefixed, pion pi_).

All the logic lives in the package (Bmmm.Analysis.JpsiTkInspector, built on the
shared Bmmm.Analysis.JpsiChargedInspector), so this file only wires it up.

Examples:

ipython -i -- inspector_jpsi_tk.py --inputFiles=<file.root> --filename=jpsi_k_data --maxevents=-1

ipython -i -- inspector_jpsi_tk.py --inputFiles=<signal.root> --filename=jpsi_k_signal --mc --maxevents=-1 --savenontrig
'''

from Bmmm.Analysis.JpsiTkInspector import JpsiTkInspector

if __name__ == '__main__':
    JpsiTkInspector().main()
