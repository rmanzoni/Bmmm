'''
https://link.springer.com/content/pdf/10.1134/S1063778818030092.pdf
https://arxiv.org/pdf/1812.06004.pdf
https://link.springer.com/content/pdf/10.1140/epjc/s10052-019-7112-x.pdf

Example:

ipython -i -- inspector_b4m_analysis.py \
--inputFiles="/pnfs/psi.ch/cms/trivcat/store/user/manzoni/Bs4Mu_MINIAOD_05sep23_v1/*.root" \
--filename=bs4mu \
--mc \
--maxevents=100



ipython -i -- inspector_b4m_analysis.py \
--inputFiles="files_ParkingDoubleMuonLowMass0-PromptReco-v2.txt" \
--filename=data \
--maxevents=100000



TO DO:
- save refitted momenta  ==> DONE, but don't look good
- which other variables?
- rerunning with tighter GEN level cuts and more stats  ==> ALMOST DONE
- add mass uncertainty
- filter by JSON
- fix PU
- skim by vtx prob
'''

from __future__ import print_function
import os
import re
import sys
import ROOT
import argparse
import numpy as np
import pandas as pd
import uproot
from time import time
from copy import deepcopy as dc
from copy import copy as sc
from datetime import datetime, timedelta
from array import array
from glob import glob
from collections import OrderedDict, defaultdict
from DataFormats.FWLite import Events, Handle
from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi, bestMatch
from itertools import product, combinations
from Bmmm.Analysis.B4MuBranches import branches, paths, muon_branches, cand_branches, event_branches, bs_branches
from Bmmm.Analysis.B4MuCandidate import B4MuCandidate as Candidate
from Bmmm.Analysis.utils import drop_hlt_version, cutflow

######################################################################################
#####      PARSER
######################################################################################
parser = argparse.ArgumentParser(description='')
parser.add_argument('--inputFiles'   , dest='inputFiles' , required=True, type=str)
parser.add_argument('--verbose'      , dest='verbose'    , action='store_true' )
parser.add_argument('--destination'  , dest='destination', default='./' , type=str)
parser.add_argument('--filename'     , dest='filename'   , required=True, type=str)
parser.add_argument('--maxevents'    , dest='maxevents'  , default=-1   , type=int)
parser.add_argument('--mc'           , dest='mc'         , action='store_true')
parser.add_argument('--logfreq'      , dest='logfreq'    , default=100   , type=int)
parser.add_argument('--logger'       , dest='logger'     , default=''    , type=str)
parser.add_argument('--filemode'     , dest='filemode'   , default='recreate', type=str)
parser.add_argument('--savenontrig'  , dest='savenontrig', action='store_true' )
parser.add_argument('--maxfiles'     , dest='maxfiles'   , default=-1   , type=int)
args = parser.parse_args()

inputFiles  = args.inputFiles
destination = args.destination
fileName    = args.filename
maxevents   = args.maxevents
verbose     = args.verbose
logfreq     = args.logfreq
logger      = args.logger
filemode    = args.filemode
savenontrig = args.savenontrig
maxfiles    = args.maxfiles
mc = False; mc = args.mc

handles_mc = OrderedDict()
handles_mc['genpr'  ] = ('prunedGenParticles'  , Handle('std::vector<reco::GenParticle>')     )
#handles_mc['genpk'  ] = ('packedGenParticles'  , Handle('std::vector<pat::PackedGenParticle>'))
#handles_mc['genInfo'] = ('generator'           , Handle('GenEventInfoProduct')                )
handles_mc['pu'     ] = ('slimmedAddPileupInfo', Handle('std::vector<PileupSummaryInfo>')     )

handles = OrderedDict()
handles['muons'  ] = ('slimmedMuons'                       , Handle('std::vector<pat::Muon>')                   )
handles['trk'    ] = ('packedPFCandidates'                 , Handle('std::vector<pat::PackedCandidate>')        )
handles['ltrk'   ] = ('lostTracks'                         , Handle('std::vector<pat::PackedCandidate>')        )
handles['vtx'    ] = ('offlineSlimmedPrimaryVerticesWithBS', Handle('std::vector<reco::Vertex>')                )
#handles['vtx'    ] = ('offlineSlimmedPrimaryVertices'      , Handle('std::vector<reco::Vertex>')                )
handles['trg_res'] = (('TriggerResults', '', 'HLT' )       , Handle('edm::TriggerResults'        )              )
handles['trg_ps' ] = (('patTrigger'    , '')               , Handle('pat::PackedTriggerPrescales')              )
handles['bs'     ] = ('offlineBeamSpot'                    , Handle('reco::BeamSpot')                           )
handles['tobjs'  ] = ('slimmedPatTrigger'                  , Handle('std::vector<pat::TriggerObjectStandAlone>'))
handles['jets'   ] = ('slimmedJets'                        , Handle('std::vector<pat::Jet>')                    )

if ('txt' in inputFiles):
    with open(inputFiles) as f:
        files = f.read().splitlines()
        files = ['root://cms-xrd-global.cern.ch//' + file for file in files]
elif ',' in inputFiles or 'cms-xrd-global' in inputFiles:
    files = inputFiles.split(',')
else:
    files = glob(inputFiles)

if maxfiles>0:
    files = files[:maxfiles]

print("files:", files)

events = Events(files)
maxevents = maxevents if maxevents>=0 else events.size() # total number of events in the files

#fout = ROOT.TFile(destination + '/' + fileName + '.root', filemode)
fout = uproot.recreate(destination + '/' + fileName + '.root')

#if filemode=='update':
#    ntuple = fout.Get('tree')
#else:
#    ntuple = ROOT.TNtuple('tree', 'tree', ':'.join(branches))

#ntuple = pd.DataFrame(columns=branches)
row_list = []

tofill = OrderedDict(zip(branches, [np.nan]*len(branches)))
if sys.version_info[0]>=3:
    if sys.version_info[0]>=7:
        tofill = dict(zip(branches, [np.nan]*len(branches)))


tofill = dict(zip(branches, [np.nan]*len(branches)))


# start the stopwatch
start = time()
mytimestamp = datetime.now().strftime('%Y-%m-%d__%Hh%Mm%Ss')
print('#### STARTING NOW', mytimestamp)

for i, event in enumerate(events):
        
    if (i+1) > maxevents:
        break
            
    if i%logfreq == 0:
        percentage = float(i) / maxevents * 100.
        speed = float(i) / (time() - start)
        eta = datetime.now() + timedelta(seconds=(maxevents-i) / max(0.1, speed))
        print('\t===> processing %d / %d event \t completed %.1f%s \t %.1f ev/s \t ETA %s s' %(i, maxevents, percentage, '%', speed, eta.strftime('%Y-%m-%d %H:%M:%S')))

    # reset trees
    for k, v in tofill.items():
        tofill[k] = np.nan

    # access the handles
    for k, v in handles.items():
        event.getByLabel(v[0], v[1])
        setattr(event, k, v[1].product())
    
    event.mc = False
    
    if mc:
        event.mc = True
        for k, v in handles_mc.items():
            event.getByLabel(v[0], v[1])
            setattr(event, k, v[1].product())

        event.pu_at_bx0 = [ipu for ipu in event.pu if ipu.getBunchCrossing()==0][0]

    cutflow['all processed events'] += 1

    lumi = event.eventAuxiliary().luminosityBlock()
    iev  = event.eventAuxiliary().event()
    ######################################################################################
    #####      RECO PART HERE
    ######################################################################################
    trg_names = event.object().triggerNames(event.trg_res)

    hlt_passed = False

    for iname in trg_names.triggerNames():
        iname = str(iname)
        if not iname.startswith('HLT_'):
            continue
        for ipath in paths.keys():
            idx = len(trg_names)               
            if drop_hlt_version(iname)==ipath:
                idx = trg_names.triggerIndex(iname)
                tofill[ipath        ] = ( idx < len(trg_names)) * (event.trg_res.accept(idx))
                tofill[ipath + '_ps'] = event.trg_ps.getPrescaleForIndex(idx)
    
    triggers = {key:tofill[key] for key in paths.keys()}

    hlt_passed = any([vv for vv in triggers.values()])
    # skip events if no trigger fired, unless savenotrig option is specified
    if not(savenontrig or hlt_passed):
        continue            
    
    cutflow['pass HLT'] += 1

    # trigger matching
    good_tobjs = {key:[] for key in paths.keys()}    
    for to in [to for to in event.tobjs if to.pt()>3. and abs(to.eta())<2.6]:
        #to.unpackFilterLabels(event.object(), event.trg_res)
        to.unpackNamesAndLabels(event.object(), event.trg_res)
        for k, v in paths.items():
            if triggers[k]!=1: continue
            for ilabel in v: 
                if to.hasFilterLabel(ilabel) and to not in good_tobjs[k]:
                    good_tobjs[k].append(to)

            
    muons = [mu for mu in event.muons if mu.pt()>1. and abs(mu.eta())<2.5 and mu.isPFMuon() and mu.isGlobalMuon()]
    muons.sort(key = lambda x : x.pt(), reverse = True)

    if len(muons)<4:
        continue

    cutflow['at least four muons'] += 1

    ######################################################################################
    #####      BUILD AND SELECT 4MU CANDIDATES
    ######################################################################################
    cands = []

    for iquadruplet in combinations(muons, 4): 
    
        cutflow['\tcandidates after HLT and 4mu'] += 1

        # 4 muon candidate
        cand = Candidate(iquadruplet, event.vtx, event.bs)
        
        # 4 muons somewhat close in dz, max distance 1 cm
        if max([abs( imu.bestTrack().dz(cand.pv.position()) - jmu.bestTrack().dz(cand.pv.position()) ) for imu, jmu in combinations(cand.muons, 2)])>1: 
            #continue
            pass
        cutflow['\tpass mutual dz'] += 1
        
        # filter by mass, first
        if cand.mass()<4. or cand.mass()>7.:
            continue
        cutflow['\tpass mass cut'] += 1
                   
        # trigger matching, at least one muon matched. 
        # Later one can save the best matched trigger object to each muon, but let me keep it simple for now
        # FIXME! trigger name is hardcoded!
        cand.trig_match = False
        if sum([deltaR(ipair[0], ipair[1])<0.1 for ipair in product(iquadruplet, good_tobjs['HLT_DoubleMu4_3_LowMass'])])==0:
            if savenontrig:
                pass
            else:
                continue
        cand.trig_match = True
        cutflow['\tpass trigger match'] += 1
        
        # valid vertex
        if not cand.good_vtx:
            continue
        cutflow['\tpass secondary vertex'] += 1
        
        # if you made it this far, then save the candidate
        cands.append(cand)

    # if no cands at this point, you might as well move on to the next event
    if len(cands)==0:
        continue
    
    event.ncands = len(cands) # useful for ntuple filling
    
    cutflow['at least one cand pass presel'] += 1

    # sort candidates by charge combination and best pointing angle, i.e. cosine closer to 1
    # can implement and use other criteria later
    cands.sort(key = lambda x : (abs(x.charge())==0, x.vtx.cos2d), reverse = True)
    final_cand = cands[0]
          
    ######################################################################################
    #####      FILL
    ######################################################################################
    for branch, getter in event_branches.items():
        tofill[branch] = getter(event)    
               
    if mc:
        gen_muons = [ip for ip in event.genpr if abs(ip.pdgId())==13 and (abs(ip.mother(0).pdgId())==531 or abs(ip.mother(0).pdgId())==511)]
        #bss= [ip for ip in event.genpr if abs(ip.pdgId())==531 and abs(abs(ip.mother().pdgId())!=531)]
        #print('\n')
        #for jj, ibs in enumerate(bss):
        #    print('%d Bs PDG ID %d' %(jj, ibs.pdgId()))

    for idx in range(1, 5):
        imu = getattr(final_cand, 'mu%d' %idx)
        imu.pv = final_cand.pv
        imu.bs = final_cand.bs
        imu.iso03 = imu.pfIsolationR03()
        imu.iso04 = imu.pfIsolationR04()
        
        # jet matching
        jet, dr2 = bestMatch(imu, event.jets)        
        if dr2<0.2**2: imu.jet = jet        
        # gen matching
        if mc:
            genp, dr2 = bestMatch(imu, [ip for ip in gen_muons if ip.charge()==imu.charge()])
            if dr2<0.02**2: imu.genp = genp
        
        for branch, getter in muon_branches.items():
            tofill['mu%d_%s' %(idx, branch)] = getter(imu) 

    for branch, getter in cand_branches.items():
        tofill[branch] = getter(final_cand)    

    if getattr(final_cand.mu1, 'genp', False) and \
       getattr(final_cand.mu2, 'genp', False) and \
       getattr(final_cand.mu3, 'genp', False) and \
       getattr(final_cand.mu4, 'genp', False):
                
        if ( abs(final_cand.mu1.genp.mother(0).pdgId()) in [511, 531] ) and \
           final_cand.mu1.genp.mother(0) == final_cand.mu2.genp.mother(0) == final_cand.mu3.genp.mother(0) == final_cand.mu4.genp.mother(0):
            mother_b = final_cand.mu1.genp.mother(0)

            for branch, getter in bs_branches.items():
                tofill[branch] = getter(mother_b)    
    
    # append selected event
    row_list.append(sc(tofill))
    
##########################################################################################
#####      WRITE TO DISK
##########################################################################################
ntuple = pd.DataFrame(row_list, columns=branches)
print('\nnumber of selected events', len(ntuple))
fout['tree'] = ntuple
print('\nntuple saved, processed all desired events?', (i+1==maxevents), 'processed', i+1, 'maxevents', maxevents)

##########################################################################################
#####      SAVE LOGGER 
##########################################################################################

logger_name = logger if len(logger)>0 else 'logger_'+mytimestamp

with open('%s.txt'%logger_name, 'w') as logger:
    for k, v in cutflow.items():
        print(k, v, file=logger)


