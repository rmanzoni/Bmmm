'''
Example:
ipython -i -- inspector_kee_analysis.py --inputFiles="37c50324-b780-4536-b137-11ab9fafdbd8.root" --filename=data


ipython -i -- inspector_kee_analysis.py --inputFiles="912a511d-832e-4d14-8b0c-e6e49e7d8b17.root" --filename=data


/store/data/Run2022F/ParkingDoubleElectronLowMass0/MINIAOD/PromptReco-v1/000/361/971/00000/5043b499-ae56-403b-8243-5e9a8a0e9d8e.root

xrdcp /scratch/manzoni/Kee_DoubleEleLowMass0to5_Run2022C_11aug22_v9/bd957a0a-bb1c-45ff-9f70-0896457511c8.root .

/ParkingDoubleElectronLowMass0/Run2022C-PromptReco-v1/MINIAOD
/ParkingDoubleElectronLowMass1/Run2022C-PromptReco-v1/MINIAOD
/ParkingDoubleElectronLowMass2/Run2022C-PromptReco-v1/MINIAOD
/ParkingDoubleElectronLowMass3/Run2022C-PromptReco-v1/MINIAOD
/ParkingDoubleElectronLowMass4/Run2022C-PromptReco-v1/MINIAOD
/ParkingDoubleElectronLowMass5/Run2022C-PromptReco-v1/MINIAOD

add adnti D0 cut DONE


MC
https://github.com/DiElectronX/BParkingNANO/blob/main/BParkingNano/production/samples_Run3.yml#L40-L51

/BuTOKEE20220826bettersplitting/jodedra-SUMMER22_MINIAOD-d5db235e2a58bcae594a314d29cbde75/USER
/BuTOjpsiKEE20220831fiftyMbettersplitting/jodedra-SUMMER22_MINIAOD-d5db235e2a58bcae594a314d29cbde75/USER
/BuTOpsi2sKEE20220831fiftyMbettersplitting/jodedra-SUMMER22_MINIAOD-d5db235e2a58bcae594a314d29cbde75/USER


dasgoclient -query="file dataset=/BuTOKEE20220826bettersplitting/jodedra-SUMMER22_MINIAOD-d5db235e2a58bcae594a314d29cbde75/USER            instance=prod/phys03" > files_kee_lowq2.txt
dasgoclient -query="file dataset=/BuTOjpsiKEE20220831fiftyMbettersplitting/jodedra-SUMMER22_MINIAOD-d5db235e2a58bcae594a314d29cbde75/USER  instance=prod/phys03" > files_kee_jpsi.txt
dasgoclient -query="file dataset=/BuTOpsi2sKEE20220831fiftyMbettersplitting/jodedra-SUMMER22_MINIAOD-d5db235e2a58bcae594a314d29cbde75/USER instance=prod/phys03" > files_kee_psi2s.txt


https://stackoverflow.com/questions/582336/how-do-i-profile-a-python-script


ipython -i -- inspector_kee_analysis_v2.py --inputFiles="361971_5043b499-ae56-403b-8243-5e9a8a0e9d8e.root" --filename=ruttone

'''

from __future__ import print_function
import ROOT
import argparse
import numpy as np
from time import time
from datetime import datetime, timedelta
from array import array
from glob import glob
from collections import OrderedDict
from DataFormats.FWLite import Events, Handle
from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi, bestMatch
from itertools import product, combinations
from Bmmm.Analysis.DiEleBranches import branches, paths
from Bmmm.Analysis.KeeCandidate import DiEleCandidate, KeeCandidate, e_mass

parser = argparse.ArgumentParser(description='')
parser.add_argument('--inputFiles'   , dest='inputFiles' , required=True, type=str)
parser.add_argument('--verbose'      , dest='verbose'    , action='store_true' )
parser.add_argument('--destination'  , dest='destination', default='./' , type=str)
parser.add_argument('--filename'     , dest='filename'   , required=True, type=str)
parser.add_argument('--maxevents'    , dest='maxevents'  , default=-1   , type=int)
parser.add_argument('--mc'           , dest='mc'         , action='store_true')
parser.add_argument('--logfreq'      , dest='logfreq'    , default=100   , type=int)
parser.add_argument('--filemode'     , dest='filemode'   , default='recreate', type=str)
args = parser.parse_args()

inputFiles  = args.inputFiles
destination = args.destination
fileName    = args.filename
maxevents   = args.maxevents
verbose     = args.verbose
logfreq     = args.logfreq
filemode    = args.filemode
mc = False; mc = args.mc

handles_mc = OrderedDict()
handles_mc['genpr'  ] = ('prunedGenParticles', Handle('std::vector<reco::GenParticle>')     )
handles_mc['genpk'  ] = ('packedGenParticles', Handle('std::vector<pat::PackedGenParticle>'))
handles_mc['genInfo'] = ('generator'         , Handle('GenEventInfoProduct')                )

handles = OrderedDict()
handles['muons'  ] = ('slimmedMuons'                 , Handle('std::vector<pat::Muon>')                   )
handles['eles'   ] = ('slimmedElectrons'             , Handle('std::vector<pat::Electron>')               )
handles['trk'    ] = ('packedPFCandidates'           , Handle('std::vector<pat::PackedCandidate>')        )
handles['ltrk'   ] = ('lostTracks'                   , Handle('std::vector<pat::PackedCandidate>')        )
handles['vtx'    ] = ('offlineSlimmedPrimaryVertices', Handle('std::vector<reco::Vertex>')                )
handles['trg_res'] = (('TriggerResults', '', 'HLT' ) , Handle('edm::TriggerResults'        )              )
handles['trg_ps' ] = (('patTrigger'    , '')         , Handle('pat::PackedTriggerPrescales')              )
handles['bs'     ] = ('offlineBeamSpot'              , Handle('reco::BeamSpot')                           )
handles['tobjs'  ] = ('slimmedPatTrigger'            , Handle('std::vector<pat::TriggerObjectStandAlone>'))

if ('txt' in inputFiles):
    with open(inputFiles) as f:
        files = f.read().splitlines()
elif ',' in inputFiles or 'cms-xrd-global' in inputFiles:
    files = inputFiles.split(',')
else:
    files = glob(inputFiles)

print("files:", files)

events = Events(files)
maxevents = maxevents if maxevents>=0 else events.size() # total number of events in the files

fout = ROOT.TFile(destination + '/' + fileName + '.root', filemode)
if filemode=='update':
    ntuple = fout.Get('tree')
else:
    ntuple = ROOT.TNtuple('tree', 'tree', ':'.join(branches))
tofill = OrderedDict(zip(branches, [np.nan]*len(branches)))

# start the stopwatch
start = time()

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
    
    if mc:
        for k, v in handles_mc.items():
            event.getByLabel(v[0], v[1])
            setattr(event, k, v[1].product())
    
    lumi = event.eventAuxiliary().luminosityBlock()
    iev  = event.eventAuxiliary().event()

    ######################################################################################
    #####      RECO PART HERE (GEN PART REMOVED FOR NOW)
    ######################################################################################
    
    # yeah, fire some trigger at least! For now, I've hard coded HLT_Mu7_IP4_part0
    trg_names = event.object().triggerNames(event.trg_res)

    hlt_passed = True # FIXME! passthrough for now!!

    for iname in trg_names.triggerNames():
        iname = str(iname) # why no auto conversion from CPP string to python string?!
        if not iname.startswith('HLT_DoubleEle') or 'mMax6' not in iname : continue 
        #print(iname)
        for ipath in paths:
            idx = len(trg_names)
            # remove version
            reduced_iname = '_'.join(iname.split('_')[:-1])
            if ipath==reduced_iname:
                idx = trg_names.triggerIndex(iname)
                tofill[ipath        ] = ( idx < len(trg_names)) * (event.trg_res.accept(idx))
                tofill[ipath + '_ps'] = event.trg_ps.getPrescaleForIndex(idx)
                #if ipath=='HLT_Mu7_IP4' and event.trg_ps.getPrescaleForIndex(idx)>0 and ( idx < len(trg_names)) * (event.trg_res.accept(idx)):
                #    hlt_passed = True
    
    if not hlt_passed:
        continue            
            
    # save trigger objects for trigger matching
    # https://github.com/cms-sw/cmssw/blob/8b101cb0f00c4a961bc4a6d49512ef0335486f40/DataFormats/HLTReco/interface/TriggerTypeDefs.h
    good_tobjs = []
    for to in [to for to in event.tobjs if to.pt()>3.9 and abs(to.eta())<1.4 and (to.type(81) or to.type(82))]:
        to.unpackFilterLabels(event.object(), event.trg_res)        
        #for aa in to.filterLabels(): print(aa)
        if to.hasFilterLabel("hltDoubleEle*eta1p22mMax6ValidHitsFilter") or to.hasFilterLabel("hltDoubleEle*eta1p22mMax6NLayerITFilter"): # add the other new filters!
            good_tobjs.append(to)
                     
    eles = [ele for ele in event.eles if ele.pt()>3. and abs(ele.eta())<1.4 and ele.electronID('mvaEleID-Fall17-noIso-V1-wpLoose') and ROOT.reco.Track(ele.gsfTrack().get()).numberOfValidHits()>0 and ele.isPF()]
    eles.sort(key = lambda x : x.pt(), reverse = True)
                    
    if len(eles)<2:
        continue

    # build diele candidates
    diele_cands = []

    for ipair in combinations(eles, 2): 

        # 2 ele candidate
        cand = DiEleCandidate(ipair, event.vtx, event.bs)
        
        # 2 eles somewhat close in dz, max distance 1 cm
        if max([abs( iele.gsfTrack().get().dz(cand.pv.position()) - jele.gsfTrack().get().dz(cand.pv.position()) ) for iele, jele in combinations(cand.eles, 2)])>0.8: 
            continue
        
        # filter by mass, first
        if cand.mass()<0.1 or cand.mass()>20:
            continue
        
        # trigger matching
        to1, dr1 = bestMatch(cand.ele1, good_tobjs)
        to2, dr2 = bestMatch(cand.ele2, good_tobjs)
        
        cand.ele1.to = None
        cand.ele2.to = None

        #check = False
        #if to1==to2:
        #    check=True
        #    import pdb ; pdb.set_trace()
    
        if to1==to2 and dr1 < dr2:
            to2, dr2 = bestMatch(cand.ele2, [obj for obj in good_tobjs if obj!=to1])
            cand.ele1.to = to1 if dr1<0.2*0.2 else None
            cand.ele2.to = to2 if dr2<0.2*0.2 else None
        elif to1==to2 and dr2 < dr1:
            to1, dr1 = bestMatch(cand.ele1, [obj for obj in good_tobjs if obj!=to2])
        elif to1 != to2:
            cand.ele1.to = to1 if dr1<0.2*0.2 else None
            cand.ele2.to = to2 if dr2<0.2*0.2 else None

        #if check:
        #    import pdb ; pdb.set_trace()
        
        # valid vertex
        if not cand.vtx.isValid():
            continue
        
        # if you made it this far, then save the candidate
        diele_cands.append(cand)

    # if no cands at this point, you might as well move on to the next event
    if len(diele_cands)==0:
        continue
    
    diele_cands.sort(key = lambda x : ((x.ele1.to is not None and x.ele2.to is not None), abs(x.charge())!=1, x.pt()), reverse = True)
    diele = diele_cands[0]

    #import pdb ; pdb.set_trace()
      
    # fill the tree    
    # can make it smarter with lambda functions associated to the def of branches             
    tofill['run'   ] = event.eventAuxiliary().run()
    tofill['lumi'  ] = event.eventAuxiliary().luminosityBlock()
    tofill['event' ] = event.eventAuxiliary().event()
    tofill['npv'   ] = len(event.vtx)
    tofill['neles' ] = len(eles)
    tofill['neecands'] = len(diele_cands)
    tofill['ntrigobj'] = len(good_tobjs)
    
    tofill['mass'    ] = diele.mass()
    tofill['mcorr'   ] = diele.mass_corrected()
    tofill['pt'      ] = diele.pt()
    tofill['eta'     ] = diele.eta()
    tofill['phi'     ] = diele.phi()
    tofill['tk_mass' ] = diele.mass(3)
    tofill['tk_mcorr'] = diele.mass_corrected(3)
    tofill['tk_pt'   ] = diele.pt(3)
    tofill['tk_eta'  ] = diele.eta(3)
    tofill['tk_phi'  ] = diele.phi(3)
    tofill['sc_mass' ] = diele.mass(0)
    tofill['sc_mcorr'] = diele.mass_corrected(0)
    tofill['sc_pt'   ] = diele.pt(0)
    tofill['sc_eta'  ] = diele.eta(0)
    tofill['sc_phi'  ] = diele.phi(0)
    tofill['charge'  ] = diele.charge()

    tofill['to_mass'   ] = diele.charge()
    tofill['to_pt'     ] = diele.charge()
    tofill['to_eta'    ] = diele.charge()
    tofill['to_phi'    ] = diele.charge()
    tofill['to_charge' ] = diele.charge()
 
    tofill['dr'    ] = diele.dr()
    tofill['dr_max'] = diele.max_dr()

    tofill['pv_x' ] = diele.pv.position().x()
    tofill['pv_y' ] = diele.pv.position().y()
    tofill['pv_z' ] = diele.pv.position().z()

    tofill['bs_x0'] = event.bs.x0()
    tofill['bs_y0'] = event.bs.y0()
    tofill['bs_z0'] = event.bs.z0()

    tofill['bs_x'] = diele.bs.position().x()
    tofill['bs_y'] = diele.bs.position().y()

    tofill['vx'] = diele.vtx.position().x()
    tofill['vy'] = diele.vtx.position().y()
    tofill['vz'] = diele.vtx.position().z()
    tofill['vtx_chi2'] = diele.vtx.chi2
    tofill['vtx_prob'] = diele.vtx.prob

    tofill['cos2d'  ] = diele.vtx.cos
    tofill['lxy'    ] = diele.lxy.value()
    tofill['lxy_err'] = diele.lxy.error()
    tofill['lxy_sig'] = diele.lxy.significance()    
    
    tofill['e1_pt'         ] = diele.ele1.pt()
    tofill['e1_eta'        ] = diele.ele1.eta()
    tofill['e1_phi'        ] = diele.ele1.phi()
    tofill['e1_e'          ] = diele.ele1.energy()
    if diele.ele1.to is not None:
        tofill['e1_to_pt'      ] = diele.ele1.to.pt()
        tofill['e1_to_eta'     ] = diele.ele1.to.eta()
        tofill['e1_to_phi'     ] = diele.ele1.to.phi()
    tofill['e1_mass'       ] = diele.ele1.mass()
    tofill['e1_tk_pt'      ] = diele.ele1.gsfTrack().pt()
    tofill['e1_tk_eta'     ] = diele.ele1.gsfTrack().eta()
    tofill['e1_tk_phi'     ] = diele.ele1.gsfTrack().phi()
    tofill['e1_tk_e'       ] = np.sqrt(diele.ele1.gsfTrack().p()**2 + e_mass**2)
    tofill['e1_sc_pt'      ] = diele.ele1.p4(0).pt()
    tofill['e1_sc_eta'     ] = diele.ele1.p4(0).eta()
    tofill['e1_sc_phi'     ] = diele.ele1.p4(0).phi()
    tofill['e1_sc_e'       ] = diele.ele1.p4(0).energy()
    tofill['e1_charge'     ] = diele.ele1.charge()
    tofill['e1_id_loose'   ] = diele.ele1.electronID('mvaEleID-Fall17-noIso-V1-wpLoose')
    tofill['e1_id_wp90'    ] = diele.ele1.electronID('mvaEleID-Fall17-noIso-V1-wp90')
    tofill['e1_id_wp80'    ] = diele.ele1.electronID('mvaEleID-Fall17-noIso-V1-wp80')
    tofill['e1_dxy'        ] = diele.ele1.gsfTrack().get().dxy(diele.pv.position())
    tofill['e1_dxy_e'      ] = diele.ele1.gsfTrack().get().dxyError(diele.pv.position(), diele.pv.error())
    tofill['e1_dxy_sig'    ] = diele.ele1.gsfTrack().get().dxy(diele.pv.position()) / diele.ele1.gsfTrack().get().dxyError(diele.pv.position(), diele.pv.error())
    tofill['e1_dz'         ] = diele.ele1.gsfTrack().get().dz(diele.pv.position())
    tofill['e1_dz_e'       ] = diele.ele1.gsfTrack().get().dzError()
    tofill['e1_dz_sig'     ] = diele.ele1.gsfTrack().get().dz(diele.pv.position()) / diele.ele1.gsfTrack().get().dzError()
    tofill['e1_bs_dxy'     ] = diele.ele1.gsfTrack().get().dxy(diele.bs.position())
    tofill['e1_bs_dxy_e'   ] = diele.ele1.gsfTrack().get().dxyError(diele.bs.position(), diele.bs.error())
    tofill['e1_bs_dxy_sig' ] = diele.ele1.gsfTrack().get().dxy(diele.bs.position()) / diele.ele1.gsfTrack().get().dxyError(diele.bs.position(), diele.bs.error())
    tofill['e1_cov_pos_def'] = diele.ele1.is_cov_pos_def
    tofill['e1_det_cov'    ] = np.linalg.det(diele.ele1.cov)

    #import pdb ; pdb.set_trace()
    tofill['e1_fbrem'              ] = diele.ele1.fbrem()	 
    tofill['e1_deltaetain'         ] = abs(diele.ele1.deltaEtaSuperClusterTrackAtVtx())	 
    tofill['e1_deltaphiin'         ] = abs(diele.ele1.deltaPhiSuperClusterTrackAtVtx())	 
    tofill['e1_oldsigmaietaieta'   ] = diele.ele1.full5x5_sigmaIetaIeta()
    tofill['e1_oldhe'              ] = diele.ele1.full5x5_hcalOverEcal()
    tofill['e1_ep'                 ] = diele.ele1.eSuperClusterOverP()	 
    tofill['e1_olde15'             ] = diele.ele1.full5x5_e1x5()	 
    tofill['e1_eelepout'           ] = diele.ele1.eEleClusterOverPout()	 
    tofill['e1_kfchi2'             ] = diele.ele1.closestCtfTrackNormChi2()	 
    tofill['e1_kfhits'             ] = diele.ele1.closestCtfTrackNLayers()	 
    tofill['e1_expected_inner_hits'] = diele.ele1.gsfTrack().hitPattern().numberOfLostHits(1)	 # https://cmssdt.cern.ch/lxr/source/DataFormats/TrackReco/interface/HitPattern.h
    tofill['e1_convDist'           ] = diele.ele1.convDist()	 
    tofill['e1_convDcot'           ] = diele.ele1.convDcot()	  
    tofill['e1_r9'                 ] = diele.ele1.r9()
    tofill['e1_r9_5x5'             ] = diele.ele1.full5x5_r9()
    tofill['e1_scl_eta'            ] = diele.ele1.superCluster().eta() 
    tofill['e1_dr03TkSumPt'        ] = diele.ele1.dr03TkSumPt() 
    tofill['e1_dr03EcalRecHitSumEt'] = diele.ele1.dr03EcalRecHitSumEt() 
    tofill['e1_dr03HcalTowerSumEt' ] = diele.ele1.dr03HcalTowerSumEt()
    tofill['e1_pixhits'            ] = diele.ele1.gsfTrack().hitPattern().numberOfValidPixelHits()

    tofill['e2_pt'         ] = diele.ele2.pt()
    tofill['e2_eta'        ] = diele.ele2.eta()
    tofill['e2_phi'        ] = diele.ele2.phi()
    tofill['e2_e'          ] = diele.ele2.energy()
    if diele.ele2.to is not None:
        tofill['e2_to_pt'      ] = diele.ele2.to.pt()
        tofill['e2_to_eta'     ] = diele.ele2.to.eta()
        tofill['e2_to_phi'     ] = diele.ele2.to.phi()
    tofill['e2_mass'       ] = diele.ele2.mass()
    tofill['e2_tk_pt'      ] = diele.ele2.gsfTrack().pt()
    tofill['e2_tk_eta'     ] = diele.ele2.gsfTrack().eta()
    tofill['e2_tk_phi'     ] = diele.ele2.gsfTrack().phi()
    tofill['e2_tk_e'       ] = np.sqrt(diele.ele2.gsfTrack().p()**2 + e_mass**2)
    tofill['e2_sc_pt'      ] = diele.ele2.p4(0).pt()
    tofill['e2_sc_eta'     ] = diele.ele2.p4(0).eta()
    tofill['e2_sc_phi'     ] = diele.ele2.p4(0).phi()
    tofill['e2_sc_e'       ] = diele.ele2.p4(0).energy()
    tofill['e2_charge'     ] = diele.ele2.charge()
    tofill['e2_id_loose'   ] = diele.ele2.electronID('mvaEleID-Fall17-noIso-V1-wpLoose')
    tofill['e2_id_wp90'    ] = diele.ele2.electronID('mvaEleID-Fall17-noIso-V1-wp90')
    tofill['e2_id_wp80'    ] = diele.ele2.electronID('mvaEleID-Fall17-noIso-V1-wp80')
    tofill['e2_dxy'        ] = diele.ele2.gsfTrack().get().dxy(diele.pv.position())
    tofill['e2_dxy_e'      ] = diele.ele2.gsfTrack().get().dxyError(diele.pv.position(), diele.pv.error())
    tofill['e2_dxy_sig'    ] = diele.ele2.gsfTrack().get().dxy(diele.pv.position()) / diele.ele2.gsfTrack().get().dxyError(diele.pv.position(), diele.pv.error())
    tofill['e2_dz'         ] = diele.ele2.gsfTrack().get().dz(diele.pv.position())
    tofill['e2_dz_e'       ] = diele.ele2.gsfTrack().get().dzError()
    tofill['e2_dz_sig'     ] = diele.ele2.gsfTrack().get().dz(diele.pv.position()) / diele.ele2.gsfTrack().get().dzError()
    tofill['e2_bs_dxy'     ] = diele.ele2.gsfTrack().get().dxy(diele.bs.position())
    tofill['e2_bs_dxy_e'   ] = diele.ele2.gsfTrack().get().dxyError(diele.bs.position(), diele.bs.error())
    tofill['e2_bs_dxy_sig' ] = diele.ele2.gsfTrack().get().dxy(diele.bs.position()) / diele.ele2.gsfTrack().get().dxyError(diele.bs.position(), diele.bs.error())
    tofill['e2_cov_pos_def'] = diele.ele2.is_cov_pos_def
    tofill['e2_det_cov'    ] = np.linalg.det(diele.ele2.cov)

    tofill['e2_fbrem'              ] = diele.ele2.fbrem()	 
    tofill['e2_deltaetain'         ] = abs(diele.ele2.deltaEtaSuperClusterTrackAtVtx())	 
    tofill['e2_deltaphiin'         ] = abs(diele.ele2.deltaPhiSuperClusterTrackAtVtx())	 
    tofill['e2_oldsigmaietaieta'   ] = diele.ele2.full5x5_sigmaIetaIeta()
    tofill['e2_oldhe'              ] = diele.ele2.full5x5_hcalOverEcal()
    tofill['e2_ep'                 ] = diele.ele2.eSuperClusterOverP()	 
    tofill['e2_olde15'             ] = diele.ele2.full5x5_e1x5()	 
    tofill['e2_eelepout'           ] = diele.ele2.eEleClusterOverPout()	 
    tofill['e2_kfchi2'             ] = diele.ele2.closestCtfTrackNormChi2()	 
    tofill['e2_kfhits'             ] = diele.ele2.closestCtfTrackNLayers()	 
    tofill['e2_expected_inner_hits'] = diele.ele2.gsfTrack().hitPattern().numberOfLostHits(1)	 # https://cmssdt.cern.ch/lxr/source/DataFormats/TrackReco/interface/HitPattern.h
    tofill['e2_convDist'           ] = diele.ele2.convDist()	 
    tofill['e2_convDcot'           ] = diele.ele2.convDcot()	  
    tofill['e2_r9'                 ] = diele.ele2.r9()
    tofill['e2_r9_5x5'             ] = diele.ele2.full5x5_r9()
    tofill['e2_scl_eta'            ] = diele.ele2.superCluster().eta() 
    tofill['e2_dr03TkSumPt'        ] = diele.ele2.dr03TkSumPt() 
    tofill['e2_dr03EcalRecHitSumEt'] = diele.ele2.dr03EcalRecHitSumEt() 
    tofill['e2_dr03HcalTowerSumEt' ] = diele.ele2.dr03HcalTowerSumEt()
    tofill['e2_pixhits'            ] = diele.ele2.gsfTrack().hitPattern().numberOfValidPixelHits()

    if (diele.ele1.to is not None) and (diele.ele2.to is not None):
        top4 = diele.ele1.to.p4() + diele.ele2.to.p4()        
        tofill['to_mass'   ] = top4.mass()
        tofill['to_pt'     ] = top4.pt()
        tofill['to_eta'    ] = top4.eta()
        tofill['to_phi'    ] = top4.phi()

    tofill['trg_match'] = (diele.ele1.to is not None and diele.ele2.to is not None)    
    
    ntuple.Fill(array('f', tofill.values()))
            
fout.cd()
ntuple.Write()
fout.Close()


# dasgoclient -query="dataset file=0289b1eb-eb76-43b6-8a7d-d42783bcc0d9.root"
# 361971_5043b499-ae56-403b-8243-5e9a8a0e9d8e.root
# 37c50324-b780-4536-b137-11ab9fafdbd8.root
# 912a511d-832e-4d14-8b0c-e6e49e7d8b17.root
# 

