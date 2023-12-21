from __future__ import print_function
import os
import re
import ROOT
import argparse
import numpy as np
import pickle
from time import time
from datetime import datetime, timedelta
from array import array
from glob import glob
from collections import OrderedDict, defaultdict
from scipy.constants import c as speed_of_light
from scipy import stats
from DataFormats.FWLite import Events, Handle
from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi, bestMatch
from itertools import product, combinations
from cuts import cuts_tight, cuts_loose, cuts_gen
from Bmmm.Analysis.RDsBranches import branches, paths, event_branches, cand_branches, muon_branches, track_branches
from Bmmm.Analysis.RDsCandidate import RDsCandidate
from Bmmm.Analysis.utils import drop_hlt_version, diquarks, excitedBs, isAncestor, masses, p4_with_mass, cutflow, fillRecoTree, isMyDs, convert_cov, is_pos_def, fix_track

import particle
from particle import Particle
ROOT.gSystem.Load('libVtxFitFitter')
from ROOT import KVFitter # VertexDistance3D is contained here, dirt trick!!
from ROOT import RDsKinVtxFitter

kinfit = RDsKinVtxFitter()    
vtxfit = KVFitter()
tofit = ROOT.std.vector('reco::Track')()

'''
ipython -i -- inspector_rds.py --inputFiles="/pnfs/psi.ch/cms/trivcat/store/user/manzoni/all_signals_HbToDsPhiKKPiMuNu_MT_MINI_21jan23_v1/all_signals_HbToDsPhiKKPiMuNu_MT_99.root" --filename="test_signals.root"
'''

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
parser.add_argument('--filemode'     , dest='filemode'   , default='recreate', type=str)
parser.add_argument('--savenontrig'  , dest='savenontrig', action='store_true' )
parser.add_argument('--maxfiles'     , dest='maxfiles'   , default=-1   , type=int)
parser.add_argument('--loosecuts'    , dest='loosecuts'  , action='store_true')
args = parser.parse_args()

inputFiles  = args.inputFiles
destination = args.destination
fileName    = args.filename
maxevents   = args.maxevents
verbose     = args.verbose
logfreq     = args.logfreq
filemode    = args.filemode
savenontrig = args.savenontrig
maxfiles    = args.maxfiles
loosecuts   = args.loosecuts

cuts = cuts_loose if loosecuts else cuts_tight
mc = False; mc = args.mc
######################################################################################
    
class candidate():
    def __init__(self, ds, muon):
        self.ds = ds
        self.muon = muon
    def p4(self):
        return self.ds.p4() + self.muon.p4()
    def charge(self):
        return self.ds.charge() + self.muon.charge()

handles_mc = OrderedDict()
handles_mc['genpr'  ] = ('prunedGenParticles'  , Handle('std::vector<reco::GenParticle>')     )
handles_mc['genpk'  ] = ('packedGenParticles'  , Handle('std::vector<pat::PackedGenParticle>'))
handles_mc['genInfo'] = ('generator'           , Handle('GenEventInfoProduct')                )
handles_mc['genInfo'] = ('generator'           , Handle('GenEventInfoProduct')                )
handles_mc['pu'     ] = ('slimmedAddPileupInfo', Handle('std::vector<PileupSummaryInfo>')     )

handles = OrderedDict()
handles['muons'  ] = ('slimmedMuons'                 , Handle('std::vector<pat::Muon>')              )
handles['trk'    ] = ('packedPFCandidates'           , Handle('std::vector<pat::PackedCandidate>')   )
handles['ltrk'   ] = ('lostTracks'                   , Handle('std::vector<pat::PackedCandidate>')   )
handles['vtx'    ] = ('offlineSlimmedPrimaryVertices', Handle('std::vector<reco::Vertex>')           )
handles['trg_res'] = (('TriggerResults', '', 'HLT' ) , Handle('edm::TriggerResults'        )         )
handles['trg_ps' ] = (('patTrigger'    , '')         , Handle('pat::PackedTriggerPrescales')         )
handles['bs'     ] = ('offlineBeamSpot'              , Handle('reco::BeamSpot')                      )
handles['tobjs'  ] = ('slimmedPatTrigger'            , Handle('std::vector<pat::TriggerObjectStandAlone>'))
handles['jets'   ] = ('slimmedJets'                  , Handle('std::vector<pat::Jet>')                    )

if ('txt' in inputFiles):
    with open(inputFiles) as f:
        files = f.read().splitlines()
elif ',' in inputFiles:
    files = inputFiles.split(',')
else:
    files = glob(inputFiles)

print("files:", files)

events = Events(files)
maxevents = maxevents if maxevents>=0 else events.size() # total number of events in the files

start = time()
mytimestamp = datetime.now().strftime('%Y-%m-%d__%Hh%Mm%Ss')
print('#### STARTING NOW', mytimestamp)

fout = ROOT.TFile(destination + '/' + fileName + '.root', 'recreate')

ntuple = ROOT.TNtuple('tree_gen', 'tree_gen', ':'.join(branches))
tofill = OrderedDict(zip(branches, [np.nan]*len(branches)))

ntuple_reco = ROOT.TNtuple('tree', 'tree', ':'.join(branches))
tofill_reco = OrderedDict(zip(branches, [np.nan]*len(branches)))

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
    
    cutflow['all processed events'] += 1
    
    if verbose: print('=========>')
    
    if mc:

        for k, v in handles_mc.items():
            event.getByLabel(v[0], v[1])
            setattr(event, k, v[1].product())

        # qscale
        event.qscale = event.genInfo.qScale()

        # gen level pileup
        pu_at_bx0 = [ipu for ipu in event.pu if ipu.getBunchCrossing()==0][0]
        event.npu = pu_at_bx0.getPU_NumInteractions()
        event.nti = pu_at_bx0.getTrueNumInteractions()

        event.genp = [ip for ip in event.genpr] + [ip for ip in event.genpk]
    
        dss   = [ip for ip in event.genp if abs(ip.pdgId())==431 and isMyDs(ip)]
        muons = [ip for ip in event.genpr if abs(ip.pdgId())==13 and ip.status()==1 and ip.pt()>7. and abs(ip.eta())<1.5]
    
        candidates = []
        for ids, imuon in product(dss, muons):
            icand = candidate(ids, imuon)
            if icand.charge()==0 and icand.p4().mass()<8.:
                ancestors = []
                printAncestors(icand.ds, ancestors, verbose=False)
                ancestors = []
                printAncestors(icand.muon, ancestors, verbose=False)
                candidates.append(icand)    
        
        if len(candidates)==0:
            # how is this possible?!
            if verbose: print('no candidates, WEIRD!')
            continue
                    
        if len(candidates)>1:
            print('more than one GEN candidate! Total %d candidates' %len(candidates))
#             import pdb ; pdb.set_trace()
#             continue

        candidates.sort(key = lambda x : (x.charge()==0, x.p4().pt()), reverse=True)
        cand = candidates[0]
    
        the_bs    = cand.ds.ancestors[-1] if len(cand.ds.ancestors) else None
        the_ds_st = None
        the_ds    = cand.ds
        the_phi   = cand.ds.phi_meson
        the_kp    = None
        the_km    = None
        the_pi    = cand.ds.pion
        the_mu    = cand.muon
        the_tau   = None
        which_signal = np.nan
    
        # check if signal
        if len(the_ds.ancestors)>0 and \
           len(the_mu.ancestors)>0 and \
           the_ds.ancestors[-1]==the_mu.ancestors[-1] and \
           abs(the_bs.pdgId())==531:

            daughters = []
            for idx_dau in range(the_bs.numberOfDaughters()):
                idau = the_bs.daughter(idx_dau)
                if idau.pdgId()==22:
                    continue
                daughters.append(idau.pdgId())
            daughters.sort(key = lambda x : abs(x))
               
            # save which signal is this
            # 0 Ds  mu nu
            # 1 Ds* mu nu
            # 2 Ds  tau nu
            # 3 Ds* tau nu
            if daughters==[13,-14,431] or daughters==[-13,14,-431]:
                which_signal = 0
            if daughters==[13,-14,433] or daughters==[-13,14,-433]:
                which_signal = 1
            if daughters==[15,-16,431] or daughters==[-15,16,-431]:
                which_signal = 2
            if daughters==[15,-16,433] or daughters==[-15,16,-433]:
                which_signal = 3

            if which_signal in [1, 3]:
                for idx_dau in range(the_bs.numberOfDaughters()):
                    idau = the_bs.daughter(idx_dau)
                    if abs(idau.pdgId())==433:
                        the_ds_st = idau

        if the_phi:
            for idx_dau in range(the_phi.numberOfDaughters()):
                idau = the_phi.daughter(idx_dau)
                if idau.pdgId()==321:
                    the_kp = idau
                    continue
                elif idau.pdgId()==-321:
                    the_km = idau
                    continue

        if abs(the_mu.mother(0).pdgId())==15:
            the_tau = the_mu.mother(0)

        if the_ds is None or \
           the_mu is None or \
           the_phi is None or \
           the_km is None or \
           the_kp is None or \
           the_pi is None:
            continue

        b_lab_p4 = the_mu.p4() + the_ds.p4()
        b_scaled_p4 = b_lab_p4 * ((particle.literals.B_s_0.mass/1000.)/b_lab_p4.mass())
    
        b_scaled_p4_tlv = ROOT.TLorentzVector() ; b_scaled_p4_tlv.SetPtEtaPhiE(b_scaled_p4.pt(), b_scaled_p4.eta(), b_scaled_p4.phi(), b_scaled_p4.energy())
        the_mu_p4_tlv = ROOT.TLorentzVector() ; the_mu_p4_tlv.SetPtEtaPhiE(the_mu.pt(), the_mu.eta(), the_mu.phi(), the_mu.energy())
    
        b_scaled_p4_boost = b_scaled_p4_tlv.BoostVector()
    
        the_mu_p4_in_b_rf = the_mu_p4_tlv.Clone(); the_mu_p4_in_b_rf.Boost(-b_scaled_p4_boost)
         
        event.which_signal = which_signal
        
    
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
            if drop_hlt_version(iname, pattern=r"_part\d_v\d+")==ipath:
                idx = trg_names.triggerIndex(iname)
                tofill_reco[ipath        ] = ( idx < len(trg_names)) * (event.trg_res.accept(idx))
                tofill_reco[ipath + '_ps'] = event.trg_ps.getPrescaleForIndex(idx)
    
    triggers = {key:tofill_reco[key] for key in paths.keys()}
    #import pdb ; pdb.set_trace()

    hlt_passed = any([vv for vv in triggers.values()])
    # skip events if no trigger fired, unless savenotrig option is specified
    if not(savenontrig or hlt_passed):
        continue            
    
    cutflow['pass HLT'] += 1

    # trigger matching
    good_tobjs = {key:[] for key in paths.keys()}    
    for to in [to for to in event.tobjs if to.pt()>6. and abs(to.eta())<2.0]: # hard coded, know BPark
        to.unpackNamesAndLabels(event.object(), event.trg_res)
        for k, v in paths.items():
            if triggers[k]!=1: continue
            for ilabel in v: 
                if to.hasFilterLabel(ilabel) and to not in good_tobjs[k]:
                    good_tobjs[k].append(to)

    ######################################################################################
    muons = [mu for mu in event.muons if mu.pt()>cuts['mu_pt'] and abs(mu.eta())<cuts['mu_eta'] and cuts['mu_basic_id'](mu)]

    if len(muons)<1:
        if mc: fillRecoTree(ntuple_reco, tofill_reco)
        continue
    cutflow['>=1 muons'] += 1

    # FIXME! trigger matching done for a single HLT at the moment
    muons = [mu for mu in muons if bestMatch(mu, good_tobjs[cuts['HLT']])[1] < cuts['mu_trig_match']**2]

    if len(muons)<1:
        if mc: fillRecoTree(ntuple_reco, tofill_reco)
        continue    
    cutflow['>=1 trg match muon'] += 1

    tracks = [tk for tk in event.trk if tk.charge()!=0] + [tk for tk in event.ltrk if tk.charge()!=0]
    tracks = [tk for tk in tracks if tk.pt()>cuts['tk_pt'] and abs(tk.eta())<cuts['tk_eta'] and tk.hasTrackDetails()]
    tracks = [tk for tk in tracks if abs(tk.pdgId()) not in [11, 13]]

    if len(tracks)<3:
        if mc: fillRecoTree(ntuple_reco, tofill_reco)
        continue

    cutflow['>=3 tracks'] += 1
        
    reco_candidates = []
        
    for ii, imu in enumerate(muons):

        itracks = [tk for tk in tracks if deltaR(imu, tk)<cuts['max_dr_m_tk'] and \
                                          deltaR(imu, tk)>cuts['min_dr_m_tk'] and \
                                          abs(tk.bestTrack().dxy(event.vtx[0].position()) < cuts_loose['tk_dxy']) and \
                                          abs(imu.bestTrack().dz(event.vtx[0].position()) - tk.dz(event.vtx[0].position()))<cuts['max_dz_m_tk']]

        if len(itracks)<3: 
            continue

        for ikaons in combinations(itracks, 2):
            
            # sort by pt
            ikaons = sorted(list(ikaons), key = lambda tk : tk.pt(), reverse=True)
            tk1 = ikaons[0]
            tk2 = ikaons[1]
            
            if deltaR(tk1, tk2)>cuts['max_dr_k1_k1']:
                continue
            
            # loop over pions
            pis = [tk for tk in itracks if tk!=tk1 and tk!=tk2]
            
            for ipi in pis:
                
                cutflow['\tncands'] += 1
                
                # create candidate
                icand = RDsCandidate(imu, ikaons, ipi, event.vtx, event.bs)
                
                # loose cut on phi and Ds masses
                if abs(icand.phi1020.mass()-masses['phi'])>cuts['phi_mass_window'] or \
                   abs(icand.ds     .mass()-masses['ds' ])>cuts['ds_mass_window' ]:
                    continue

                cutflow['\tncand pass phi & Ds mass'] += 1

                if icand.mass()>cuts['max_bs_mass']:
                    continue

                cutflow['\tncand pass Bs mass < %.1f' %cuts['max_bs_mass']] += 1
                
                # filter on vertex quantities
                icand.phi1020.compute_vtx()
                if not(icand.phi1020.vtx.isValid() and icand.phi1020.vtx.prob>cuts['phi_vtx_prob']):
                    continue

                cutflow['\tncand pass phi vtx'] += 1

                # filter on vertex quantities
                icand.ds.compute_vtx()
                if not(icand.ds.vtx.isValid() and icand.ds.vtx.prob>cuts['ds_vtx_prob']):
                    continue

                cutflow['\tncand pass ds vtx'] += 1
                                    
                reco_candidates.append(icand)

    # sort candidates by charge, mass and pt
    sorter = lambda cand : (cand.charge()==0 and abs(cand.ds.charge())==1 and cand.phi1020.charge()==0 and cand.mu.charge()*cand.pi.charge()<0, cand.mass()<masses['bs'], cand.pt())
    reco_candidates.sort(key = sorter, reverse = True)
    
    event.reco_candidates = reco_candidates
    
    if len(reco_candidates)==0:
        if mc: fillRecoTree(ntuple_reco, tofill_reco)
        continue

    cutflow['>=1 cand left'] += 1
    
    mycand = reco_candidates[0]

    mycand.compute_vtx(full=True)
    mycand.compute_kinematics()

    if not(mycand.vtx.isValid()):
        if mc: fillRecoTree(ntuple_reco, tofill_reco)
        continue
    
    cutflow['chosen cand has good vtx'] += 1

    if (mycand.vtx.cos2d<0 and mycand.vtx.cos3d<0):
        if mc: fillRecoTree(ntuple_reco, tofill_reco)
        continue

    cutflow['chosen cand cos>0'] += 1
            
    ######################################################################################
    for branch, getter in event_branches.items():
        tofill_reco[branch] = getter(event) 

    for branch, getter in cand_branches.items():
        tofill_reco[branch] = getter(mycand) 

    for branch, getter in muon_branches.items():
        tofill_reco['mu_%s' %(branch)] = getter(mycand.mu) 

    for branch, getter in track_branches.items():
        tofill_reco['k1_%s' %(branch)] = getter(mycand.k1) 

    for branch, getter in track_branches.items():
        tofill_reco['k2_%s' %(branch)] = getter(mycand.k2) 

    for branch, getter in track_branches.items():
        tofill_reco['pi_%s' %(branch)] = getter(mycand.pi) 

    fillRecoTree(ntuple_reco, tofill_reco)

    #print('>>>>>>>>>>>>>>>>>')
    #print('k1 pos def', is_pos_def(convert_cov(mycand.k1.bestTrack().covariance())))
    #print('k2 pos def', is_pos_def(convert_cov(mycand.k2.bestTrack().covariance())))
    #print('pi pos def', is_pos_def(convert_cov(mycand.pi.bestTrack().covariance())))
    #print('mu pos def', is_pos_def(convert_cov(mycand.mu.bestTrack().covariance())))
    #print('=================')
    mycand.check_covariances()
    #print('k1 pos def', is_pos_def(convert_cov(mycand.k1.bestTrack().covariance())))
    #print('k2 pos def', is_pos_def(convert_cov(mycand.k2.bestTrack().covariance())))
    #print('pi pos def', is_pos_def(convert_cov(mycand.pi.bestTrack().covariance())))
    #print('mu pos def', is_pos_def(convert_cov(mycand.mu.bestTrack().covariance())))
    #print('<<<<<<<<<<<<<<<<<')
    fit_results = kinfit.Fit(mycand.k1.bestTrack(), mycand.k2.bestTrack(), mycand.pi.bestTrack(), mycand.mu.bestTrack(), masses['k'], masses['pi'], masses['mu'], masses['phi'], masses['ds'])
    phi_tree = fit_results._0
    ds_tree = fit_results._1
    bs_tree = fit_results._2
    
    if not(hasattr(phi_tree, 'isValid') and phi_tree.get().__nonzero__() and phi_tree.isValid()):
        if mc: fillRecoTree(ntuple_reco, tofill_reco)
        continue
    
    cutflow['pass constrained phi vtx fit>0'] += 1

    if not(hasattr(ds_tree, 'isValid') and ds_tree.get().__nonzero__() and ds_tree.isValid()):
        if mc: fillRecoTree(ntuple_reco, tofill_reco)
        continue
    
    cutflow['pass constrained Ds vtx fit>0'] += 1

    if not(hasattr(bs_tree, 'isValid') and bs_tree.get().__nonzero__() and bs_tree.isValid()):
        if mc: fillRecoTree(ntuple_reco, tofill_reco)
        continue
    
    cutflow['pass Bs vtx fit>0'] += 1

            
fout.cd()
if mc:
    ntuple_reco.AddFriend(ntuple)
    ntuple.Write()
ntuple_reco.Write()
fout.Close()

# save logger file
with open('logger_%s.txt'%mytimestamp, 'w') as logger:
    for k, v in cutflow.items():
        print(k, v, file=logger)

