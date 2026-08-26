from copy import copy
from Bmmm.Analysis.JpsiChargedCuts import cuts

# J/psi mu (RJpsi) selection: an alias of the shared 'rjpsi' cuts under the
# channel key 'jpsi_mu'. Identical selection to the original RJpsi analysis.
cuts['jpsi_mu'] = copy(cuts['rjpsi'])
