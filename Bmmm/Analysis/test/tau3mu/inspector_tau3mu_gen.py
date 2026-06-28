'''
GEN-level inspector for the displaced tau -> 3mu sample:

    D_s -> tau nu ; tau -> mu + a ; a (-> mu mu) long-lived, m = 1 GeV, ctau = 1 mm

Runs on the genParticles collection of a GEN-SIM file and saves, in a ROOT tree
called `tree`, one row per gen candidate with:
  * the p4 of each of the three muons        (mu_tau_*, mu_disp1_*, mu_disp2_*)
  * the p4 of the displaced OS muon pair      (pair_*)
  * the decay length of the displaced pair    (decay_length, lxy, lz, ctau)
  * the p4 of the whole three-muon system     (tau3mu_*)
  * the angular separation between the scalar and the prompt muon (dr_scalar_mu)

All GEN vertices / lengths are in cm.

Example:

ipython -i -- inspector_tau3mu_gen.py \
--inputFiles="file:/work/manzoni/darien/CMSSW_15_1_1/src/tau3mu_displaced_1GeV_ctau1mm.root" \
--filename=tau3mu_gen \
--maxevents=-1
'''

from __future__ import print_function
import sys
import argparse
import numpy as np
import pandas as pd
import uproot
from time import time
from datetime import datetime, timedelta
from glob import glob
from collections import OrderedDict, namedtuple

from DataFormats.FWLite import Events, Handle

from Bmmm.Analysis.Tau3MuGenCandidate import Tau3MuGenCandidate, is_last_copy
from Bmmm.Analysis.Tau3MuGenBranches import (branches, event_branches,
                                             muon_branches, cand_branches)

# remove stdout delay
sys.stdout.flush()


##########################################################################################
#####      HANDLES   (GEN-SIM: genParticles, not pruned/packed)
##########################################################################################
handles = OrderedDict()
handles['genp'] = ('genParticles', Handle('std::vector<reco::GenParticle>'))


##########################################################################################
#####      MAIN
##########################################################################################
def main():

    parser = argparse.ArgumentParser(description='GEN-level tau -> 3mu inspector')
    parser.add_argument('--inputFiles', dest='inputFiles', required=True, type=str)
    parser.add_argument('--destination', dest='destination', default='./', type=str)
    parser.add_argument('--filename',   dest='filename',   required=True, type=str)
    parser.add_argument('--maxevents',  dest='maxevents',  default=-1,    type=int)
    parser.add_argument('--logfreq',    dest='logfreq',    default=1000,  type=int)
    parser.add_argument('--maxfiles',   dest='maxfiles',   default=-1,    type=int)

    options = namedtuple('options', parser.parse_args().__dict__.keys())(*parser.parse_args().__dict__.values())

    # build the file list (glob, comma-separated, or a .txt list)
    if 'txt' in options.inputFiles:
        with open(options.inputFiles) as f:
            files = f.read().splitlines()
    elif ',' in options.inputFiles or 'file:' in options.inputFiles or 'root://' in options.inputFiles:
        files = options.inputFiles.split(',')
    else:
        files = glob(options.inputFiles)

    if options.maxfiles > 0:
        files = files[:options.maxfiles]

    print('files:', files)

    events = Events(files)
    maxevents = options.maxevents if options.maxevents >= 0 else events.size()

    fout = uproot.recreate(options.destination + '/' + options.filename + '.root')
    row_list = []

    start = time()
    mytimestamp = datetime.now().strftime('%Y-%m-%d__%Hh%Mm%Ss')
    print('#### STARTING NOW', mytimestamp)

    ######################################################################################
    #####      EVENT LOOP
    ######################################################################################
    for i, event in enumerate(events, 1):

        if i > maxevents:
            break

        if i % options.logfreq == 0:
            percentage = float(i) / maxevents * 100.
            speed = float(i) / (time() - start)
            eta = datetime.now() + timedelta(seconds=(maxevents - i) / max(0.1, speed))
            print('\t===> processing %d / %d \t %.1f%s \t %.1f ev/s \t ETA %s' % (
                i, maxevents, percentage, '%', speed, eta.strftime('%Y-%m-%d %H:%M:%S')))

        # access the handles
        for k, v in handles.items():
            event.getByLabel(v[0], v[1])
            setattr(event, k, v[1].product())

        # build one candidate per last-copy displaced scalar
        scalars = [gp for gp in event.genp
                   if abs(gp.pdgId()) == Tau3MuGenCandidate.SCALAR_PDGID and is_last_copy(gp)]

        cands = []
        for sc in scalars:
            try:
                cands.append(Tau3MuGenCandidate(sc))
            except ValueError:
                # scalar that doesn't match the signal topology, skip it
                continue

        if len(cands) == 0:
            continue

        event.ncands = len(cands)

        # one row per candidate
        for cand in cands:

            tofill = dict(zip(branches, [np.nan] * len(branches)))

            for branch, getter in event_branches.items():
                tofill[branch] = getter(event)

            for label, mu in [('mu_tau',   cand.mu_tau),
                              ('mu_disp1', cand.mu_disp1),
                              ('mu_disp2', cand.mu_disp2)]:
                for branch, getter in muon_branches.items():
                    tofill['%s_%s' % (label, branch)] = getter(mu)

            for branch, getter in cand_branches.items():
                tofill[branch] = getter(cand)

            row_list.append(tofill)

    ######################################################################################
    #####      WRITE TO DISK
    ######################################################################################
    ntuple = pd.DataFrame(row_list, columns=branches)
    print('\nnumber of selected candidates', len(ntuple))
    if len(ntuple) > 0:
        fout['tree'] = ntuple

    finish = time()
    print('done in %.1f min' % ((finish - start) / 60.))


if __name__ == '__main__':
    main()
