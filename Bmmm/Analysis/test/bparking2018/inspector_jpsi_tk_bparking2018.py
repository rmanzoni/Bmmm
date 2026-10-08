'''
J/psi + track ntuplizer entry point for the 2018 B-parking data (ParkingBPH1-6,
UL2018 MiniAODv2) and the matching UL2018 MC: B+ -> J/psi K+ and Bc+ -> J/psi pi+
built as ONE candidate carrying both mass hypotheses (kaon unprefixed, pion pi_).

Copy of test/rjpsi/inspector_jpsi_tk.py with the 2018 trigger: the event must
fire at least one HLT_Mu*_IP* parking path. The logic lives in the package
(Bmmm.Analysis.JpsiTkBParking2018Inspector, a thin subclass of JpsiTkInspector);
see that module for the trigger branches.

Selection override, local to this script:
    max_mass = 7.0 GeV  (jpsi_tk: 7.5, baseline: 10)
A candidate is kept if EITHER the kaon- or the pion-hypothesis mass lies in
(min_mass, max_mass) = (4.5, 7.0) GeV (JpsiTkInspector.build_candidates, the
only place max_mass is used). It goes in through a copy of the cuts under its
own key, so the shared cuts dict -- and the Run 3 jpsi_tk selection, and the
package class -- are untouched.

Data needs a certification mask (the inspector refuses to start without one):

python3 inspector_jpsi_tk_bparking2018.py --inputFiles=<file.root or LFN> \
    --filename=jpsi_tk_bph2018 --lumi-mask=golden --maxevents=2000 --logfreq=100

MC:

python3 inspector_jpsi_tk_bparking2018.py --inputFiles=<file.root or LFN> \
    --filename=jpsi_tk_mc --mc --savenontrig --maxevents=2000 --logfreq=100
'''

from copy import copy

from Bmmm.Analysis.JpsiTkCuts import cuts as _jpsi_tk_cuts
from Bmmm.Analysis.JpsiTkBParking2018Inspector import JpsiTkBParking2018Inspector

CHANNEL = 'jpsi_tk_bparking2018'
cuts = dict(_jpsi_tk_cuts)
cuts[CHANNEL] = copy(_jpsi_tk_cuts['jpsi_tk'])
cuts[CHANNEL]['max_mass'] = 7.0   # [GeV]


class Inspector(JpsiTkBParking2018Inspector):
    CHANNEL = CHANNEL
    CUTS    = cuts


if __name__ == '__main__':
    print('#### %s cuts: min_mass %.2f GeV, max_mass %.2f GeV'
          % (CHANNEL, cuts[CHANNEL]['min_mass'], cuts[CHANNEL]['max_mass']))
    Inspector().main()
