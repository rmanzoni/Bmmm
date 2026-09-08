# -*- coding: utf-8 -*-
'''
https://link.springer.com/content/pdf/10.1134/S1063778818030092.pdf
https://arxiv.org/pdf/1812.06004.pdf
https://link.springer.com/content/pdf/10.1140/epjc/s10052-019-7112-x.pdf

Example:
MC
ipython -i -- inspector_mm_analysis.py --inputFiles="C1ACDC94-EBC6-1745-A410-359FFEAB28BC.root" --filename=signal --mc
DATI
ipython -i -- inspector_mm_analysis.py --inputFiles="5EBF575A-A990-CB41-8EC8-28A3F2035C1B.root" --filename=data_2026 --maxevents=-1


DEBUG
ipython -i -- inspector_mm_analysis.py --inputFiles="root://cms-xrd-global.cern.ch///store/data/Run2018D/Charmonium/MINIAOD/UL2018_MiniAODv2_GT36-v1/2820000/CD88CAFB-B897-3F43-AC78-7DFCA16973D8.root" --filename=debug --skip=55000


FIXME!
- HLT_Mu17 and HLT_Mu19 broken, can't match <=== fixed!
- add trigger object p4


profiler
https://stackoverflow.com/questions/582336/how-do-i-profile-a-python-script
https://www.youtube.com/watch?v=QJwVYlDzAXs

then sue pstat to analyse

https://jiffyclub.github.io/snakeviz/

https://docs.python.org/3/library/profile.html



propagate L1 muons
https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideMagneticField
https://github.com/cms-sw/cmssw/blob/eec2351f29c3f14f7c06cf612a8eb9ae7544a0c5/MagneticField/Engine/test/queryField.cc
https://github.com/rmanzoni/WTau3Mu/blob/92X/plugins/L1MuonRecoPropagator.h
https://github.com/cms-l1-dpg/Legacy-L1Ntuples/blob/6b1d8fce0bd2058d4309af71b913e608fced4b17/src/L1MuonRecoTreeProducer.cc

'''

from __future__ import print_function
import os 
import re
import ROOT
import argparse
import pickle
import json
import numpy as np
import uproot
from time import time
from datetime import datetime, timedelta
from glob import glob
from collections import OrderedDict
from DataFormats.FWLite import Events, Handle
from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi, bestMatch
from itertools import product, combinations
from Bmmm.Analysis.MuMuBranches import (
    branches, paths, event_branches, cand_branches, muon_branches,
)
from Bmmm.Analysis.CommonBranches import safe_get
from Bmmm.Analysis.NtupleWriter import WRITE_EVERY, build_branch_types, flush
from Bmmm.Analysis.MuMuCandidate import Candidate
from Bmmm.Analysis.utils import (
    COV_ELEMENT_NAMES, COV_INDEX_PAIRS,
    VTX_COV_ELEMENT_NAMES, VTX_COV_INDEX_PAIRS, vertex_cov_element,
    drop_hlt_version, resolve_input_files,
)
        
parser = argparse.ArgumentParser(description='')
parser.add_argument('--inputFiles'   , dest='inputFiles' , required=True, type=str)
parser.add_argument('--verbose'      , dest='verbose'    , action='store_true' )
parser.add_argument('--destination'  , dest='destination', default='./' , type=str)
parser.add_argument('--filename'     , dest='filename'   , required=True, type=str)
parser.add_argument('--maxevents'    , dest='maxevents'  , default=-1   , type=int)
parser.add_argument('--mc'           , dest='mc'         , action='store_true')
parser.add_argument('--logfreq'      , dest='logfreq'    , default=100   , type=int)
parser.add_argument('--filemode'     , dest='filemode'   , default='recreate', type=str)
parser.add_argument('--skip'         , dest='skip'       , default=-1    , type=int)
parser.add_argument('--savenontrig'  , dest='savenontrig', action='store_true' )
parser.add_argument('--redirector'   , dest='redirector' , default='root://cms-xrd-global.cern.ch//', type=str)
args = parser.parse_args()

inputFiles  = args.inputFiles
destination = args.destination
fileName    = args.filename
maxevents   = args.maxevents
verbose     = args.verbose
logfreq     = args.logfreq
filemode    = args.filemode
skip        = args.skip
savenontrig = args.savenontrig
redirector  = args.redirector
mc = False; mc = args.mc

handles_mc = OrderedDict()
handles_mc['genpr'  ] = ('prunedGenParticles'  , Handle('std::vector<reco::GenParticle>')     )
handles_mc['genpk'  ] = ('packedGenParticles'  , Handle('std::vector<pat::PackedGenParticle>'))
handles_mc['genInfo'] = ('generator'           , Handle('GenEventInfoProduct')                )
handles_mc['pu'     ] = ('slimmedAddPileupInfo', Handle('std::vector<PileupSummaryInfo>')     )

handles = OrderedDict()
handles['muons'  ] = ('slimmedMuons'                 , Handle('std::vector<pat::Muon>')                   )
handles['trk'    ] = ('packedPFCandidates'           , Handle('std::vector<pat::PackedCandidate>')        )
handles['ltrk'   ] = ('lostTracks'                   , Handle('std::vector<pat::PackedCandidate>')        )
handles['vtx'    ] = ('offlineSlimmedPrimaryVertices', Handle('std::vector<reco::Vertex>')                )
handles['trg_res'] = (('TriggerResults', '', 'HLT' ) , Handle('edm::TriggerResults'        )              )
handles['trg_ps' ] = (('patTrigger'    , '')         , Handle('pat::PackedTriggerPrescales')              )
handles['bs'     ] = ('offlineBeamSpot'              , Handle('reco::BeamSpot')                           )
handles['tobjs'  ] = ('slimmedPatTrigger'            , Handle('std::vector<pat::TriggerObjectStandAlone>'))
handles['jets'   ] = ('slimmedJets'                  , Handle('std::vector<pat::Jet>')                    )

#handles['gtdigis'] = (("gtDigis"      , ""     ), Handle('L1GlobalTriggerReadoutRecord')             )
#handles['l1max'  ] = (("patTrigger"   , "l1max"), Handle('pat::PackedTriggerPrescales ')             )
#handles['l1min'  ] = (("patTrigger"   , "l1min"), Handle('pat::PackedTriggerPrescales ')             )   
handles['glb_alg'] = (("gtStage2Digis", ""     ), Handle('BXVector<GlobalAlgBlk>')                   )
#handles['glb_ext'] = (("gtStage2Digis", ""     ), Handle('BXVector<GlobalExtBlk>')                   )

# # CANNOT access L1 seed name, only its bit in the menu...
# for i in range(event.glb_alg.at(0,0).getAlgoDecisionFinal().size()): print event.glb_alg.at(0,0).getAlgoDecisionFinal(i)
# https://gitlab.cern.ch/sharper/HLTAnalyserPy
# L1 menus
# https://twiki.cern.ch/twiki/bin/view/CMS/GlobalTriggerAvailableMenus
# https://twiki.cern.ch/twiki/bin/view/CMS/L1KnownIssues#Menu_AN2
# https://github.com/cms-l1-dpg
# #include "tmEventSetup/tmEventSetup.hh"
# https://github.com/cms-l1-dpg/L1Menu2018/tree/master/official/PrescaleTables
# tmeventsetup::getMmHashN("324ed470-bdf0-4315-a64f-da3b4bc3343c");
# // returns 571217662

# get prescale column
# event.glb_alg.at(0,0).getPreScColumn()
# get minimum and maximum L1 prescale. Why on earth is this info useful in this form, god knows...
# event.l1max.setTriggerNames(event.object().triggerNames(event.trg_res))
# event.l1min.setTriggerNames(event.object().triggerNames(event.trg_res))
# event.l1max.getPrescaleForName('HLT_DoubleMu4_3_Jpsi', True)
# event.l1min.getPrescaleForName('HLT_DoubleMu4_3_Jpsi', True)

# get prescales
# https://github.com/cms-sw/cmssw/blob/4b3cfa5cead4e8497f808954dc4281b885a0008c/L1Trigger/L1TGlobal/plugins/GtRecordDump.cc#L190

# BXVector<GlobalAlgBlk>                "gtStage2Digis"             ""                "RECO"
# BXVector<GlobalExtBlk>                "gtStage2Digis"             ""                "RECO"
# L1GlobalTriggerReadoutRecord          "gtDigis"                   ""                "RECO"
# pat::PackedTriggerPrescales           "patTrigger"                ""                "PAT"
# pat::PackedTriggerPrescales           "patTrigger"                "l1max"           "PAT"
# pat::PackedTriggerPrescales           "patTrigger"                "l1min"           "PAT"
# vector<pat::TriggerObjectStandAlone>    "slimmedPatTrigger"         ""                "PAT"
# vector<string>                        "slimmedPatTrigger"         "filterLabels"    "PAT"

files = resolve_input_files(inputFiles, redirector)

print("files:", files)

events = Events(files)
maxevents = maxevents if maxevents>=0 else events.size() # total number of events in the files

##########################################################################################
##########################################################################################
#  _                    _       __ 
# | |                  | |     /_ |
# | |     _____   _____| |______| |
# | |    / _ \ \ / / _ \ |______| |
# | |___|  __/\ V /  __/ |      | |
# |______\___| \_/ \___|_|      |_|
#                                  
##########################################################################################
##########################################################################################

# load L1 prescale files and add them to the branches
l1_prescales = {}

datadir = '/'.join([
    os.environ['CMSSW_BASE'],
    'src',
    'Bmmm',
    'Analysis',
    'data',
])

# brilcalc L1 prescale tables, one pickle per path. Only the 2018 paths have
# one: a Run 3 path (HLT_DoubleMu4_3_LowMass) has no table here, so skip it
# rather than dying at import. The HLT side does not depend on these -- the
# decision and the HLT prescale come from TriggerResults and
# PackedTriggerPrescales in the MINIAOD itself, which are era-independent. It
# is only the L1 seed branches that need the tables.
for ipath in paths.keys():
    ipickle = '%s/%s.pickle' %(datadir, ipath)
    if not os.path.isfile(ipickle):
        print('WARNING: no L1 prescale table for %s (%s); its L1 seed branches '
              'will not be produced. The HLT decision and prescale are '
              'unaffected.' % (ipath, os.path.basename(ipickle)))
        continue
    with open(ipickle, 'rb') as handle:
        l1_prescales.update(pickle.load(handle))

# create run:L1 menu dictionary. Skipped entirely when no L1 table was loaded
# (a Run 3 job), so a Run 3 run is never looked up in a 2018 map.
menus, run_menu_dict = {}, {}
if l1_prescales:
 with open('%s/l1menus/goodRuns2013to2022ByYear.json' %datadir) as f:
   data = json.load(f)

 menus = {}
 for run in data["2018"]:
     menus.setdefault(run["l1_menu"],[]).append(run["run_number"])

 run_menu_dict = {}
 for k, v in menus.items():
     for irun in v:
         run_menu_dict[irun] = k

 menus = {}

 for imenu in ['L1Menu_Collisions2018_v2_1_0',
               'L1Menu_Collisions2018_v2_0_0',
               'L1Menu_Collisions2018_v1_0_0',
               'L1Menu_Collisions2018_0_0_1']:
 #               'L1Menu_Collisions2018_v0_0_1']:
     with open('%s/l1menus/%s.pickle' %(datadir, imenu), 'rb') as f:
        menus[imenu] = pickle.load(f)

for l1 in l1_prescales.keys():
    branches.append(l1)
    branches.append(l1 + '_ps')
#import pdb ; pdb.set_trace()

##########################################################################################
##########################################################################################


# Written through uproot rather than as a TNtuple: a TNtuple stores every
# column as float32, which rounds any event number above 2^24 -- 2018 event
# numbers are well past that, and the tag-and-probe sample is de-duplicated on
# run/lumi/event. build_branch_types keeps those three as int64.
# Rows are buffered and flushed every WRITE_EVERY, so memory stays bounded.
outfile = destination + '/' + fileName + '.root'
if filemode == 'update':
    fout = uproot.update(outfile)          # the file must already exist
else:
    fout = uproot.recreate(outfile, compression=uproot.ZSTD(5))
    fout.mktree('tree', build_branch_types(branches))
row_list = []
tofill = OrderedDict(zip(branches, [np.nan]*len(branches)))

# start the stopwatch
start = time()



# keep track of already-reported missing runs
missing_run_warnings = set()

# skip the first N events
events.to(skip)

for i, event in enumerate(events):

#     if i < skip:
#         continue

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
    
    event.mc = mc

    if mc:
        for k, v in handles_mc.items():
            event.getByLabel(v[0], v[1])
            setattr(event, k, v[1].product())
        
        event.pu_at_bx0 = [ipu for ipu in event.pu if ipu.getBunchCrossing()==0][0]
            
    lumi = event.eventAuxiliary().luminosityBlock()
    iev  = event.eventAuxiliary().event()
        
    ######################################################################################
    #####      RECO PART HERE (GEN PART REMOVED FOR NOW)
    ######################################################################################
    
    trg_names = event.object().triggerNames(event.trg_res)
    _trg_len = len(trg_names)
    
    # pre-filter: only keep trigger names that match a known path, strip version once
    idx_to_path = {}
    for n in trg_names.triggerNames():
        stripped = drop_hlt_version(n)
        if stripped in paths:
            idx_to_path[trg_names.triggerIndex(n)] = stripped
    
    for idx, ipath in idx_to_path.items():
        accept = int(idx < _trg_len and event.trg_res.accept(idx))
        ps     = event.trg_ps.getPrescaleForIndex(idx)

#         if ipath == "HLT_Mu9_IP6":
#         if ipath == "HLT_IsoMu24":
#             print('already stored', iev, tofill[ipath], tofill[ipath + '_ps'])
#             import ipdb ; ipdb.set_trace()

        # OR across all parts/versions: a single firing part is sufficient
        tofill[ipath]         = np.nanmax([tofill[ipath        ], accept]) 
        tofill[ipath + '_ps'] = np.nanmax([tofill[ipath + '_ps'], ps    ])
    
    hlt_passed = any(tofill[p] > 0 for p in paths)
    
#     if iev==2139319168: 
#         import ipdb ; ipdb.set_trace()
    
#     if event.eventAuxiliary().event()==2139648128:
#         import ipdb ; ipdb.set_trace()
# 
#     # FIX: build stripped-name -->index map once per event, then O(1) lookups
#     _trg_len = len(trg_names)
#     idx_to_name = {trg_names.triggerIndex(n): n for n in trg_names.triggerNames() if drop_hlt_version(n) in paths.keys()}
# 
#     for idx, iname in idx_to_name.items():
#         
#         accept = ( idx < _trg_len) * (event.trg_res.accept(idx))
#         ps     = event.trg_ps.getPrescaleForIndex(idx)
#         
#         ipath = drop_hlt_version(iname)
#         
#         accept = np.nanmax([tofill[ipath        ], accept])
#         ps     = np.nanmax([tofill[ipath + '_ps'], ps    ])
#         
# 
# 
#     for ipath in paths.keys():
#         if ipath in name_to_idx:
#             idx = name_to_idx[ipath]
#             tofill[ipath]         = int(idx < _trg_len and event.trg_res.accept(idx))
#             tofill[ipath + '_ps'] = event.trg_ps.getPrescaleForIndex(idx)
#         # else: tofill already reset to nan above
#     
#     triggers = {key: tofill[key] for key in paths.keys()}
#     hlt_passed = any([vv for vv in triggers.values() if vv>0.5])
#     hlt_passed = any(v == 1 for v in triggers.values())


    
#     # yeah, fire some trigger at least! For now, I've hard coded HLT_Mu7_IP4_part0
#     trg_names = event.object().triggerNames(event.trg_res)
#     _trg_len = len(trg_names)
# 
#     hlt_passed = False
# 
#     for iname in trg_names.triggerNames():
#         for ipath in paths.keys():
#             idx = _trg_len               
#             if drop_hlt_version(iname)==ipath:
#                 idx = trg_names.triggerIndex(iname)
# #                 if "_IP" in ipath:
# #                     print(ipath)
# #                     import ipdb ; ipdb.set_trace()
#                 
#                 accept = ( idx < _trg_len) * (event.trg_res.accept(idx))
#                 ps     = event.trg_ps.getPrescaleForIndex(idx)
#                 
# #                 if "HLT_Mu9_IP6" in ipath:
# #                     if accept: print(iname, accept, ps)
# #                     import ipdb ; ipdb.set_trace()
#                                                 
#                 if ipath in tofill.keys():
#                     accept = np.nanmax([tofill[ipath        ], accept])
#                     ps     = np.nanmax([tofill[ipath + '_ps'], ps    ])
# 
#                 tofill[ipath        ] = accept
#                 tofill[ipath + '_ps'] = ps    
#                 #if ipath=='HLT_Mu7_IP4' and event.trg_ps.getPrescaleForIndex(idx)>0 and ( idx < len(trg_names)) * (event.trg_res.accept(idx)):
#                 #    hlt_passed = True
#     
#     triggers = {key:tofill[key] for key in paths.keys()}
# 
#     hlt_passed = any([vv for vv in triggers.values() if vv>0.5])
        
    # skip events if no trigger fired, unless savenotrig option is specified
    if not(savenontrig or hlt_passed):
        continue            
        
    # trigger matching
    # these are the filters, MAYBE!! too lazy to check confDB. Or, more appropriately: confDB sucks
    # https://github.com/cms-sw/cmssw/blob/6d2f66057131baacc2fcbdd203588c41c885b42c/Configuration/Skimming/python/pwdgSkimBPark_cfi.py#L11-L18 
#     good_tobjs = {key:[] for key in paths.keys()}    


    good_tobjs      = {key: []      for key in paths.keys()}
    good_tobjs_seen = {key: set()   for key in paths.keys()}


    for to in event.tobjs:
        if to.pt() < 3. or abs(to.eta()) >= 2.6:
            continue
        to.unpackNamesAndLabels(event.object(), event.trg_res)
        for k, v in paths.items():
            if tofill[k]!=1: continue
            for ilabel in v: 
#                 if to.hasFilterLabel(ilabel) and to not in good_tobjs[k]:
#                     good_tobjs[k].append(to)
                if to.hasFilterLabel(ilabel) and id(to) not in good_tobjs_seen[k]:
                    good_tobjs[k].append(to)
                    good_tobjs_seen[k].add(id(to))

    # muons = [mu for mu in event.muons if mu.pt()>4. and abs(mu.eta())<2.5 and mu.isPFMuon() and mu.isGlobalMuon()]
    muons = [mu for mu in event.muons if mu.pt()>4. and abs(mu.eta())<2.5]
    muons.sort(key = lambda x : x.pt(), reverse = True)

    if len(muons)<2:
        continue

    # build analysis candidates
    cands = []
    
    for itriplet in combinations(muons, 2): 

        # 4 muon candidate
        cand = Candidate(itriplet, event.vtx, event.bs)
        
        # 4 muons somewhat close in dz, max distance 1 cm
        if max([abs( imu.bestTrack().dz(cand.pv.position()) - jmu.bestTrack().dz(cand.pv.position()) ) for imu, jmu in combinations(cand.muons, 2)])>1: 
            continue
        
        # filter by mass, first. Select only jpsi and z events
        if not (np.abs(cand.mass()-3.0969)<1. or np.abs(cand.mass()-91.19)<15.):
            continue
        
        # FIXME!           
        # trigger matching, at least one muon matched. 
        # Later one can save the best matched trigger object to each muon, but let me keep it simple for now
        #if sum([deltaR(ipair[0], ipair[1])<0.15 for ipair in product(itriplet, good_tobjs)])==0:
        #    continue
        
        # valid vertex
        if not cand.vtx.isValid():
            continue
        
        # if you made it this far, then save the candidate
        cands.append(cand)

    # if no cands at this point, you might as well move on to the next event
    if len(cands)==0:
        continue

    # computationally expensive and maybe not entirely needed
#    if mc:
#        # merge gen particles
#        event.all_genp = [ip for ip in event.genpr] + [ip for ip in event.genpk if bestMatch(ip, event.genpr)[1]>0.01*0.01]

    # sort candidates by charge combination and best pointing angle, i.e. cosine closer to 1
    # can implement and use other criteria later
    cands.sort(key = lambda x : (abs(x.charge())==0, x.mu1.pt(), x.mu2.pt()), reverse = True)
    #final_cand = cands[0]

    for final_cand in cands[:1]:     
        # fill the tree    
        # can make it smarter with lambda functions associated to the def of branches             
        # Fill from the branch getters in MuMuBranches -- event-level off the
        # event, candidate-level off the candidate, per-muon off each muon once
        # it carries the per-candidate context the shared getters expect.
        # safe_get turns a getter that raises (a vertex fit that did not
        # converge, a gen match that is not there) into a NaN instead of killing
        # the job thousands of events in; run once with --verbose after changing
        # the schema, since that safety is otherwise silent.
        event.ncands = len(cands)

        for ibranch, igetter in event_branches.items():
            tofill[ibranch] = safe_get(igetter, event, verbose=verbose, name=ibranch)

        for ibranch, igetter in cand_branches.items():
            tofill[ibranch] = safe_get(igetter, final_cand, verbose=verbose, name=ibranch)

        for idx in [1, 2]:
            imu = getattr(final_cand, 'mu%d' %idx)

            # per-candidate context: the shared per-muon getters read the PV and
            # the beamspot off the muon, as they do in the RJpsi channel
            imu.pv    = final_cand.pv
            imu.bs    = final_cand.bs
            imu.iso03 = imu.pfIsolationR03()
            imu.iso04 = imu.pfIsolationR04()

            jet, dr2 = bestMatch(imu, event.jets)
            if dr2 < 0.3**2:
                imu.jet = jet

            if mc:
                genp, dr2 = bestMatch(imu, event.genpr)
                if dr2 < 0.1**2:
                    imu.gen_match = genp

            for ibranch, igetter in muon_branches.items():
                tofill['mu%d_%s' %(idx, ibranch)] = safe_get(
                    igetter, imu, verbose=verbose, name=ibranch)
                          
        #if final_cand.dr12()<0.2:
        #    import pdb ; pdb.set_trace()

        # depends on trigger matching, which depends on the order by which filter labels are defined
        # the same muon can be both tag & probe
        for k, v in paths.items():
            if tofill[k]!=1: continue
            for idx in [1,2]:
                to, dr2 = bestMatch(getattr(final_cand, 'mu%d' %idx), good_tobjs[k])
                # if "HLT_Mu9_IP6" in k: 
                #     for iname in trg_names.triggerNames(): 
                #         if "HLT_Mu9_IP6" not in iname: continue
                #         print(iname, trg_names.triggerIndex(iname), event.trg_res.accept(trg_names.triggerIndex(iname)), event.trg_ps.getPrescaleForIndex(trg_names.triggerIndex(iname)))
                #     import ipdb ; ipdb.set_trace()
                tofill['mu%d_%s_tag'   %(idx, k)] = (dr2 < 0.15*0.15 and to.hasFilterLabel(v[0])) 
                tofill['mu%d_%s_probe' %(idx, k)] = (dr2 < 0.15*0.15 and to.hasFilterLabel(v[1])) if len(v)>1 else True                 
                
        #import pdb ; pdb.set_trace() 
        # add L1 seed prescales:
        RUN  = event.eventAuxiliary().run()
        LS   = event.eventAuxiliary().luminosityBlock()
        
        # for some reasons that escape my understanding, some legit good runs may be missing from goodRuns2013to2022ByYear.json
        # this is a workaround, it reverts to the run that is closest and exist in the key dictionary
        
        #if mc:
        #    MENU_DICT = menus['L1Menu_Collisions2018_v1_0_0']        
        #else:
        #    try:
        #        MENU = run_menu_dict[RUN]
        #        MENU_DICT = menus[MENU]
        #    except:
        #        import ipdb ; ipdb.set_trace()
        
        if not run_menu_dict:
            # no L1 tables were loaded (a Run 3 job): nothing to look up, and the
            # L1 loop below is keyed on MENU_DICT, so its branches stay at NaN
            MENU_DICT = {}

        elif mc:
            MENU_DICT = menus['L1Menu_Collisions2018_v1_0_0']

        elif not (min(run_menu_dict) <= RUN <= max(run_menu_dict)):
            # The run lies outside the period the menu map covers at all -- a
            # Run 3 run against a 2018 map, say. The closest-run fallback below
            # is for a good run MISSING FROM the covered period; extrapolating
            # past its edges would hand back 2018 prescales for Run 3 data,
            # which is worse than no number, because it looks like one.
            if RUN not in missing_run_warnings:
                print('WARNING: RUN %d is outside the range the L1 menu map covers '
                      '(%d-%d). L1 prescales left empty for this run; the HLT '
                      'decision and prescale are unaffected.'
                      % (RUN, min(run_menu_dict), max(run_menu_dict)))
                missing_run_warnings.add(RUN)
            MENU_DICT = {}

        else:
            try:
                MENU = run_menu_dict[RUN]
                MENU_DICT = menus[MENU]
        
            except KeyError:
        
                # a good run missing from the json, inside the covered period
                closest_run = min(run_menu_dict.keys(), key=lambda x: abs(x - RUN))
        
                # print warning only once per missing RUN
                if RUN not in missing_run_warnings:
                    print('WARNING: RUN %d not found in run_menu_dict. Using closest available run %d instead.' % (RUN, closest_run))
                    missing_run_warnings.add(RUN)
        
                MENU = run_menu_dict[closest_run]
                MENU_DICT = menus[MENU]
        
            except Exception as exc:

                # anything other than a missing run is unexpected. Warn once and
                # carry on with an empty menu, which leaves this event's L1
                # branches at NaN -- the L1 loop below is keyed on MENU_DICT.
                # (This used to drop into ipdb, which in a batch job either hangs
                # waiting on stdin or dies on the import.)
                if RUN not in missing_run_warnings:
                    print('WARNING: could not resolve the L1 menu for RUN %d: %s: %s. '
                          'L1 prescales left empty for this run.'
                          % (RUN, type(exc).__name__, exc))
                    missing_run_warnings.add(RUN)

                MENU_DICT = {}

                        
        ## L1Menu_Collisions2018_v1_0_0-d1_xml
        ## process HLT (release CMSSW_10_2_16_UL)
        ##   HLT menu:   '/frozen/2018/2e34/v3.2/HLT/V1'
        ##   global tag: '102X_upgrade2018_realistic_v15'
        ## menu_names['L1Menu_Collisions2018_v1_0_0-d1'] = 'L1Menu_Collisions2018_v1_0_0'

        max_ls_cache = {
            l1: {run: max(ls_dict.keys()) for run, ls_dict in run_dict.items()}
            for l1, run_dict in l1_prescales.items()
        }


        for l1 in l1_prescales.keys():        
            if mc:
                tofill['%s_ps' %l1] = 1        
            else:
                # check max LS in the range
                if RUN in l1_prescales[l1].keys():
                    #max_ls = np.max(l1_prescales[l1][RUN].keys())
                    max_ls = max_ls_cache[l1][RUN]
                    if LS in l1_prescales[l1][RUN].keys():
                        my_ls = LS
                    elif LS > max_ls:
                        my_ls = max_ls
                    else:
                        # SHOULD NEVER END UP HERE, ADD SOME DEBUGGING LOGGING
                        #import pdb ; pdb.set_trace()
                        continue
                    tofill['%s_ps' %l1] = l1_prescales[l1][RUN][my_ls]           
                else:
                    tofill['%s_ps' %l1] = 0           
        
            # check id specific L1 was fired
            #import pdb ; pdb.set_trace()
            if l1 in MENU_DICT.keys():
                idx = MENU_DICT[l1]
                tofill['%s' %l1] = event.glb_alg.at(0,0).getAlgoDecisionFinal(idx)
            else:
                tofill['%s' %l1] = 0
 
            #import pdb ; pdb.set_trace()

        #import pdb ; pdb.set_trace() 
        
        row_list.append(dict(tofill))
        if len(row_list) >= WRITE_EVERY:
            flush(fout, row_list, branches)

flush(fout, row_list, branches)
print('\nnumber of selected candidates', fout['tree'].num_entries)
fout.close()

