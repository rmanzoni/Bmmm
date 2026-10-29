'''
Example:
ipython -i -- inspector_kee_analysis.py --inputFiles="37c50324-b780-4536-b137-11ab9fafdbd8.root" --filename=data


ipython -i -- inspector_kee_analysis.py --inputFiles="912a511d-832e-4d14-8b0c-e6e49e7d8b17.root" --filename=data



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
from Bmmm.Analysis.KeeBranches import branches, paths, branches_mc
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

if mc:
    branches += branches_mc

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
        #import pdb ; pdb.set_trace()
        for ipath in paths:
            idx = len(trg_names)
            if iname.startswith(ipath):
                idx = trg_names.triggerIndex(iname)
                tofill[ipath        ] = ( idx < len(trg_names)) * (event.trg_res.accept(idx))
                tofill[ipath + '_ps'] = event.trg_ps.getPrescaleForIndex(idx)
                if ipath=='HLT_Mu7_IP4' and event.trg_ps.getPrescaleForIndex(idx)>0 and ( idx < len(trg_names)) * (event.trg_res.accept(idx)):
                    hlt_passed = True

    if not hlt_passed:
        continue            
        
#     # trigger matching
#     # these are the filters, MAYBE!! too lazy to check confDB. Or, more appropriately: confDB sucks
#     # https://github.com/cms-sw/cmssw/blob/6d2f66057131baacc2fcbdd203588c41c885b42c/Configuration/Skimming/python/pwdgSkimBPark_cfi.py#L11-L18 
#     good_tobjs = []
#     for to in [to for to in event.tobjs if to.pt()>6.5 and abs(to.eta())<2.]:
#         to.unpackFilterLabels(event.object(), event.trg_res)
#         if to.hasFilterLabel('hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered7IP4Q'):
#             good_tobjs.append(to)
#             
    eles = [ele for ele in event.eles if ele.pt()>4. and abs(ele.eta())<1.4 and ele.electronID('mvaEleID-Fall17-noIso-V1-wpLoose') and ROOT.reco.Track(ele.gsfTrack().get()).numberOfValidHits()>0 and ele.isPF()]
    eles.sort(key = lambda x : x.pt(), reverse = True)
        
    # merge packed PF candidates and lost tracks
    all_trks = [tk for tk in event.trk] + [tk for tk in event.ltrk]
    all_trks = [tk for tk in all_trks if tk.pt()>0.5 and abs(tk.eta())<2.0 and tk.charge!=0 and abs(tk.pdgId()) not in [11, 13] and tk.hasTrackDetails()]
    trks = [tk for tk in all_trks if tk.pt()>0.8 and abs(tk.eta())<1.7 and tk.charge!=0 and abs(tk.pdgId()) not in [11, 13] and tk.hasTrackDetails()]
    trks.sort(key = lambda x : x.pt(), reverse = True)

    if mc:
        bs = [pp for pp in event.genpr if abs(pp.pdgId())==521]
        for ib in bs:
            ib.daus = [abs(ib.daughter(idau).pdgId()) for idau in range(ib.numberOfDaughters())]
            ib.daus = sorted([dau for dau in ib.daus if dau!=22])
            if ib.daus == [11, 11, 321]: ib.q2bin = 0
            elif ib.daus == [11, 11, 321]: ib.q2bin = 0 # non resonant
            elif ib.daus == [321, 443]: ib.q2bin = 1 # jpsi 
            elif ib.daus == [321, 100443]: ib.q2bin = 2 # psi(2S)
            else: ib.q2bin = -1
        
        bs.sort(key = lambda x : (x.q2bin>=0, x.pt()), reverse = True) 
        myb = bs[0]
        
        myb.eles  = []
        myb.k     = None 
        myb.e1    = None 
        myb.e2    = None
        myb.jpsi  = None
        myb.psi2s = None
        
        if myb.q2bin==0:
            myb.eles = [myb.daughter(idau) for idau in range(myb.numberOfDaughters()) if abs(myb.daughter(idau).pdgId())==11]

        elif myb.q2bin==1:
            myb.jpsi = [myb.daughter(idau) for idau in range(myb.numberOfDaughters()) if abs(myb.daughter(idau).pdgId())==443][0]
            myb.eles = [myb.jpsi.daughter(idau) for idau in range(myb.jpsi.numberOfDaughters()) if abs(myb.jpsi.daughter(idau).pdgId())==11]
            
        elif myb.q2bin==2:
            myb.psi2s = [myb.daughter(idau) for idau in range(myb.numberOfDaughters()) if abs(myb.daughter(idau).pdgId())==100443][0]
            myb.eles = [myb.psi2s.daughter(idau) for idau in range(myb.psi2s.numberOfDaughters()) if abs(myb.psi2s.daughter(idau).pdgId())==11]
                
        myb.k    = [myb.daughter(idau) for idau in range(myb.numberOfDaughters()) if abs(myb.daughter(idau).pdgId())==321][0]
        myb.eles = sorted(myb.eles, key = lambda x : x.pt())
        myb.e1 = myb.eles[1]
        myb.e2 = myb.eles[0]

        if any([myb.k, myb.e1, myb.e2])==None: 
            good_gen_matching = False
            #import pdb ; pdb.set_trace()
        else:   
            good_gen_matching = True
            
    if len(eles)<2:
        continue

    if len(trks)<1:
        continue

    # build diele candidates
    diele_cands = []

    for ipair in combinations(eles, 2): 

        # 2 ele candidate
        cand = DiEleCandidate(ipair, event.vtx, event.bs)
        
        # 2 eles somewhat close in dz, max distance 1 cm
        if max([abs( iele.gsfTrack().get().dz(cand.pv.position()) - jele.gsfTrack().get().dz(cand.pv.position()) ) for iele, jele in combinations(cand.eles, 2)])>1: 
            continue
        
        # filter by mass, first
        if cand.mass()<0.1 or cand.mass()>5.3:
            continue
        
        # FIXME! disable trigger matching for now          
        # trigger matching, at least one muon matched. 
        # Later one can save the best matched trigger object to each muon, but let me keep it simple for now
#         if sum([deltaR(ipair[0], ipair[1])<0.15 for ipair in product(itriplet, good_tobjs)])==0:
#             continue
        
        # valid vertex
        if not cand.vtx.isValid():
            continue
        
        # if you made it this far, then save the candidate
        diele_cands.append(cand)

    # if no cands at this point, you might as well move on to the next event
    if len(diele_cands)==0:
        continue

    # build diele candidates
    b_cands = []

    for iee, itk in product(diele_cands, trks): 

        # 2 ele candidate
        cand = KeeCandidate(iee, itk, all_trks, event.vtx, event.bs, mass=0.493677)
        
        # dR
        if cand.r()>1.4:
            continue
        
        # eles - track somewhat close in dz, max distance 1 cm
        if max([abs( iele.gsfTrack().get().dz(cand.pv.position()) - itk.dz(iee.pv.position()) ) for iele in cand.eles])>1: 
            continue

        # no double counting
        if min([deltaR(iele, itk) for iele in cand.eles])<0.005: 
            continue
        
        # filter by mass, first
        if cand.mass()<4. or cand.mass()>6.:
            continue
                
        # valid vertex
        if not cand.vtx.isValid():
            continue
        
        # if you made it this far, then save the candidate
        b_cands.append(cand)

    # if no cands at this point, you might as well move on to the next event
    if len(b_cands)==0:
        continue

    # sort candidates by charge combination and best pointing angle, i.e. cosine closer to 1
    # can implement and use other criteria later
    b_cands.sort(key = lambda x : (abs(x.diele.charge())==1, x.vtx.cos), reverse = True)
    final_cand = b_cands[0]
      
    # fill the tree    
    # can make it smarter with lambda functions associated to the def of branches             
    tofill['run'   ] = event.eventAuxiliary().run()
    tofill['lumi'  ] = event.eventAuxiliary().luminosityBlock()
    tofill['event' ] = event.eventAuxiliary().event()
    tofill['npv'   ] = len(event.vtx)
    tofill['neles' ] = len(eles)
    tofill['nbcands'] = len(b_cands)
    tofill['neecands'] = len(diele_cands)

    tofill['b_mass'    ] = final_cand.mass()
    tofill['b_mcorr'   ] = final_cand.mass_corrected()
    tofill['b_pt'      ] = final_cand.pt()
    tofill['b_eta'     ] = final_cand.eta()
    tofill['b_phi'     ] = final_cand.phi()
    tofill['b_tk_mass' ] = final_cand.mass(3)
    tofill['b_tk_mcorr'] = final_cand.mass_corrected(3)
    tofill['b_tk_pt'   ] = final_cand.pt(3)
    tofill['b_tk_eta'  ] = final_cand.eta(3)
    tofill['b_tk_phi'  ] = final_cand.phi(3)
    tofill['b_sc_mass' ] = final_cand.mass(0)
    tofill['b_sc_mcorr'] = final_cand.mass_corrected(0)
    tofill['b_sc_pt'   ] = final_cand.pt(0)
    tofill['b_sc_eta'  ] = final_cand.eta(0)
    tofill['b_sc_phi'  ] = final_cand.phi(0)
    tofill['b_charge'  ] = final_cand.charge()

    tofill['b_dr'    ] = final_cand.r()
    tofill['b_dr_max'] = final_cand.max_dr()
    tofill['b_dr_eek'] = final_cand.dr_ee_tk()

    tofill['b_abs_tk_iso'], tofill['b_rel_tk_iso'] = final_cand.trk_iso()

    tofill['ee_mass'    ] = final_cand.diele.mass()
    tofill['ee_mcorr'   ] = final_cand.diele.mass_corrected()
    tofill['ee_pt'      ] = final_cand.diele.pt()
    tofill['ee_eta'     ] = final_cand.diele.eta()
    tofill['ee_phi'     ] = final_cand.diele.phi()
    tofill['ee_tk_mass' ] = final_cand.diele.mass(3)
    tofill['ee_tk_mcorr'] = final_cand.diele.mass_corrected(3)
    tofill['ee_tk_pt'   ] = final_cand.diele.pt(3)
    tofill['ee_tk_eta'  ] = final_cand.diele.eta(3)
    tofill['ee_tk_phi'  ] = final_cand.diele.phi(3)
    tofill['ee_sc_mass' ] = final_cand.diele.mass(0)
    tofill['ee_sc_mcorr'] = final_cand.diele.mass_corrected(0)
    tofill['ee_sc_pt'   ] = final_cand.diele.pt(0)
    tofill['ee_sc_eta'  ] = final_cand.diele.eta(0)
    tofill['ee_sc_phi'  ] = final_cand.diele.phi(0)
    tofill['ee_charge'  ] = final_cand.diele.charge()

    tofill['ee_dr'    ] = final_cand.diele.dr()
    tofill['ee_dr_max'] = final_cand.diele.max_dr()

    tofill['e1k_charge'] = final_cand.charge_e1k()
    tofill['e2k_charge'] = final_cand.charge_e2k()
    tofill['e1k_mass'  ] = final_cand.mass_e1k()
    tofill['e2k_mass'  ] = final_cand.mass_e2k()
    tofill['p1k_mass'  ] = final_cand.mass_p1k()
    tofill['p2k_mass'  ] = final_cand.mass_p2k()
    tofill['e1k_dr'    ] = final_cand.dr_e1k()
    tofill['e2k_dr'    ] = final_cand.dr_e2k()

    tofill['pv_x' ] = final_cand.pv.position().x()
    tofill['pv_y' ] = final_cand.pv.position().y()
    tofill['pv_z' ] = final_cand.pv.position().z()

    tofill['bs_x0'] = event.bs.x0()
    tofill['bs_y0'] = event.bs.y0()
    tofill['bs_z0'] = event.bs.z0()

    tofill['bs_x'] = final_cand.bs.position().x()
    tofill['bs_y'] = final_cand.bs.position().y()

    tofill['ee_vx'] = final_cand.diele.vtx.position().x()
    tofill['ee_vy'] = final_cand.diele.vtx.position().y()
    tofill['ee_vz'] = final_cand.diele.vtx.position().z()
    tofill['ee_vtx_chi2'] = final_cand.diele.vtx.chi2
    tofill['ee_vtx_prob'] = final_cand.diele.vtx.prob

    tofill['ee_cos2d'  ] = final_cand.diele.vtx.cos
    tofill['ee_lxy'    ] = final_cand.diele.lxy.value()
    tofill['ee_lxy_err'] = final_cand.diele.lxy.error()
    tofill['ee_lxy_sig'] = final_cand.diele.lxy.significance()

    tofill['b_vx'] = final_cand.vtx.position().x()
    tofill['b_vy'] = final_cand.vtx.position().y()
    tofill['b_vz'] = final_cand.vtx.position().z()
    tofill['b_vtx_chi2'] = final_cand.vtx.chi2
    tofill['b_vtx_prob'] = final_cand.vtx.prob

    tofill['b_cos2d'  ] = final_cand.vtx.cos
    tofill['b_lxy'    ] = final_cand.lxy.value()
    tofill['b_lxy_err'] = final_cand.lxy.error()
    tofill['b_lxy_sig'] = final_cand.lxy.significance()
    
    
    tofill['ele1_pt'         ] = final_cand.ele1.pt()
    tofill['ele1_eta'        ] = final_cand.ele1.eta()
    tofill['ele1_phi'        ] = final_cand.ele1.phi()
    tofill['ele1_e'          ] = final_cand.ele1.energy()
    tofill['ele1_mass'       ] = final_cand.ele1.mass()
    tofill['ele1_tk_pt'      ] = final_cand.ele1.gsfTrack().pt()
    tofill['ele1_tk_eta'     ] = final_cand.ele1.gsfTrack().eta()
    tofill['ele1_tk_phi'     ] = final_cand.ele1.gsfTrack().phi()
    tofill['ele1_tk_e'       ] = np.sqrt(final_cand.ele1.gsfTrack().p()**2 + e_mass**2)
    tofill['ele1_sc_pt'      ] = final_cand.ele1.p4(0).pt()
    tofill['ele1_sc_eta'     ] = final_cand.ele1.p4(0).eta()
    tofill['ele1_sc_phi'     ] = final_cand.ele1.p4(0).phi()
    tofill['ele1_sc_e'       ] = final_cand.ele1.p4(0).energy()
    tofill['ele1_charge'     ] = final_cand.ele1.charge()
    tofill['ele1_id_loose'   ] = final_cand.ele1.electronID('mvaEleID-Fall17-noIso-V1-wpLoose')
    tofill['ele1_id_wp90'    ] = final_cand.ele1.electronID('mvaEleID-Fall17-noIso-V1-wp90')
    tofill['ele1_id_wp80'    ] = final_cand.ele1.electronID('mvaEleID-Fall17-noIso-V1-wp80')
    tofill['ele1_dxy'        ] = final_cand.ele1.gsfTrack().get().dxy(final_cand.pv.position())
    tofill['ele1_dxy_e'      ] = final_cand.ele1.gsfTrack().get().dxyError(final_cand.pv.position(), final_cand.pv.error())
    tofill['ele1_dxy_sig'    ] = final_cand.ele1.gsfTrack().get().dxy(final_cand.pv.position()) / final_cand.ele1.gsfTrack().get().dxyError(final_cand.pv.position(), final_cand.pv.error())
    tofill['ele1_dz'         ] = final_cand.ele1.gsfTrack().get().dz(final_cand.pv.position())
    tofill['ele1_dz_e'       ] = final_cand.ele1.gsfTrack().get().dzError()
    tofill['ele1_dz_sig'     ] = final_cand.ele1.gsfTrack().get().dz(final_cand.pv.position()) / final_cand.ele1.gsfTrack().get().dzError()
    tofill['ele1_bs_dxy'     ] = final_cand.ele1.gsfTrack().get().dxy(final_cand.bs.position())
    tofill['ele1_bs_dxy_e'   ] = final_cand.ele1.gsfTrack().get().dxyError(final_cand.bs.position(), final_cand.bs.error())
    tofill['ele1_bs_dxy_sig' ] = final_cand.ele1.gsfTrack().get().dxy(final_cand.bs.position()) / final_cand.ele1.gsfTrack().get().dxyError(final_cand.bs.position(), final_cand.bs.error())
    tofill['ele1_cov_pos_def'] = final_cand.ele1.is_cov_pos_def
    tofill['ele1_det_cov'    ] = np.linalg.det(final_cand.ele1.cov)

    #import pdb ; pdb.set_trace()
    tofill['ele1_fbrem'              ] = final_cand.ele1.fbrem()	 
    tofill['ele1_deltaetain'         ] = abs(final_cand.ele1.deltaEtaSuperClusterTrackAtVtx())	 
    tofill['ele1_deltaphiin'         ] = abs(final_cand.ele1.deltaPhiSuperClusterTrackAtVtx())	 
    tofill['ele1_oldsigmaietaieta'   ] = final_cand.ele1.full5x5_sigmaIetaIeta()
    tofill['ele1_oldhe'              ] = final_cand.ele1.full5x5_hcalOverEcal()
    tofill['ele1_ep'                 ] = final_cand.ele1.eSuperClusterOverP()	 
    tofill['ele1_olde15'             ] = final_cand.ele1.full5x5_e1x5()	 
    tofill['ele1_eelepout'           ] = final_cand.ele1.eEleClusterOverPout()	 
    tofill['ele1_kfchi2'             ] = final_cand.ele1.closestCtfTrackNormChi2()	 
    tofill['ele1_kfhits'             ] = final_cand.ele1.closestCtfTrackNLayers()	 
    tofill['ele1_expected_inner_hits'] = final_cand.ele1.gsfTrack().hitPattern().numberOfLostHits(1)	 # https://cmssdt.cern.ch/lxr/source/DataFormats/TrackReco/interface/HitPattern.h
    tofill['ele1_convDist'           ] = final_cand.ele1.convDist()	 
    tofill['ele1_convDcot'           ] = final_cand.ele1.convDcot()	  
    tofill['ele1_r9'                 ] = final_cand.ele1.r9()
    tofill['ele1_r9_5x5'             ] = final_cand.ele1.full5x5_r9()
    tofill['ele1_scl_eta'            ] = final_cand.ele1.superCluster().eta() 
    tofill['ele1_dr03TkSumPt'        ] = final_cand.ele1.dr03TkSumPt() 
    tofill['ele1_dr03EcalRecHitSumEt'] = final_cand.ele1.dr03EcalRecHitSumEt() 
    tofill['ele1_dr03HcalTowerSumEt' ] = final_cand.ele1.dr03HcalTowerSumEt()

    tofill['ele2_pt'         ] = final_cand.ele2.pt()
    tofill['ele2_eta'        ] = final_cand.ele2.eta()
    tofill['ele2_phi'        ] = final_cand.ele2.phi()
    tofill['ele2_e'          ] = final_cand.ele2.energy()
    tofill['ele2_mass'       ] = final_cand.ele2.mass()
    tofill['ele2_tk_pt'      ] = final_cand.ele2.gsfTrack().pt()
    tofill['ele2_tk_eta'     ] = final_cand.ele2.gsfTrack().eta()
    tofill['ele2_tk_phi'     ] = final_cand.ele2.gsfTrack().phi()
    tofill['ele2_tk_e'       ] = np.sqrt(final_cand.ele2.gsfTrack().p()**2 + e_mass**2)
    tofill['ele2_sc_pt'      ] = final_cand.ele2.p4(0).pt()
    tofill['ele2_sc_eta'     ] = final_cand.ele2.p4(0).eta()
    tofill['ele2_sc_phi'     ] = final_cand.ele2.p4(0).phi()
    tofill['ele2_sc_e'       ] = final_cand.ele2.p4(0).energy()
    tofill['ele2_charge'     ] = final_cand.ele2.charge()
    tofill['ele2_id_loose'   ] = final_cand.ele2.electronID('mvaEleID-Fall17-noIso-V1-wpLoose')
    tofill['ele2_id_wp90'    ] = final_cand.ele2.electronID('mvaEleID-Fall17-noIso-V1-wp90')
    tofill['ele2_id_wp80'    ] = final_cand.ele2.electronID('mvaEleID-Fall17-noIso-V1-wp80')
    tofill['ele2_dxy'        ] = final_cand.ele2.gsfTrack().get().dxy(final_cand.pv.position())
    tofill['ele2_dxy_e'      ] = final_cand.ele2.gsfTrack().get().dxyError(final_cand.pv.position(), final_cand.pv.error())
    tofill['ele2_dxy_sig'    ] = final_cand.ele2.gsfTrack().get().dxy(final_cand.pv.position()) / final_cand.ele2.gsfTrack().get().dxyError(final_cand.pv.position(), final_cand.pv.error())
    tofill['ele2_dz'         ] = final_cand.ele2.gsfTrack().get().dz(final_cand.pv.position())
    tofill['ele2_dz_e'       ] = final_cand.ele2.gsfTrack().get().dzError()
    tofill['ele2_dz_sig'     ] = final_cand.ele2.gsfTrack().get().dz(final_cand.pv.position()) / final_cand.ele2.gsfTrack().get().dzError()
    tofill['ele2_bs_dxy'     ] = final_cand.ele2.gsfTrack().get().dxy(final_cand.bs.position())
    tofill['ele2_bs_dxy_e'   ] = final_cand.ele2.gsfTrack().get().dxyError(final_cand.bs.position(), final_cand.bs.error())
    tofill['ele2_bs_dxy_sig' ] = final_cand.ele2.gsfTrack().get().dxy(final_cand.bs.position()) / final_cand.ele2.gsfTrack().get().dxyError(final_cand.bs.position(), final_cand.bs.error())
    tofill['ele2_cov_pos_def'] = final_cand.ele2.is_cov_pos_def
    tofill['ele2_det_cov'    ] = np.linalg.det(final_cand.ele2.cov)

    tofill['ele2_fbrem'              ] = final_cand.ele2.fbrem()	 
    tofill['ele2_deltaetain'         ] = abs(final_cand.ele2.deltaEtaSuperClusterTrackAtVtx())	 
    tofill['ele2_deltaphiin'         ] = abs(final_cand.ele2.deltaPhiSuperClusterTrackAtVtx())	 
    tofill['ele2_oldsigmaietaieta'   ] = final_cand.ele2.full5x5_sigmaIetaIeta()
    tofill['ele2_oldhe'              ] = final_cand.ele2.full5x5_hcalOverEcal()
    tofill['ele2_ep'                 ] = final_cand.ele2.eSuperClusterOverP()	 
    tofill['ele2_olde15'             ] = final_cand.ele2.full5x5_e1x5()	 
    tofill['ele2_eelepout'           ] = final_cand.ele2.eEleClusterOverPout()	 
    tofill['ele2_kfchi2'             ] = final_cand.ele2.closestCtfTrackNormChi2()	 
    tofill['ele2_kfhits'             ] = final_cand.ele2.closestCtfTrackNLayers()	 
    tofill['ele2_expected_inner_hits'] = final_cand.ele2.gsfTrack().hitPattern().numberOfLostHits(1)	 # https://cmssdt.cern.ch/lxr/source/DataFormats/TrackReco/interface/HitPattern.h
    tofill['ele2_convDist'           ] = final_cand.ele2.convDist()	 
    tofill['ele2_convDcot'           ] = final_cand.ele2.convDcot()	  
    tofill['ele2_r9'                 ] = final_cand.ele2.r9()
    tofill['ele2_r9_5x5'             ] = final_cand.ele2.full5x5_r9()
    tofill['ele2_scl_eta'            ] = final_cand.ele2.superCluster().eta() 
    tofill['ele2_dr03TkSumPt'        ] = final_cand.ele2.dr03TkSumPt() 
    tofill['ele2_dr03EcalRecHitSumEt'] = final_cand.ele2.dr03EcalRecHitSumEt() 
    tofill['ele2_dr03HcalTowerSumEt' ] = final_cand.ele2.dr03HcalTowerSumEt()

    tofill['k_pt'            ] = final_cand.trk.pt()
    tofill['k_eta'           ] = final_cand.trk.eta()
    tofill['k_phi'           ] = final_cand.trk.phi()
    tofill['k_e'             ] = final_cand.trk.energy()
    tofill['k_mass'          ] = final_cand.trk.mass()
    tofill['k_charge'        ] = final_cand.trk.charge()
    tofill['k_dxy'           ] = final_cand.trk.bestTrack().dxy(final_cand.pv.position())
    tofill['k_dxy_e'         ] = final_cand.trk.bestTrack().dxyError(final_cand.pv.position(), final_cand.pv.error())
    tofill['k_dxy_sig'       ] = final_cand.trk.bestTrack().dxy(final_cand.pv.position()) / final_cand.trk.bestTrack().dxyError(final_cand.pv.position(), final_cand.pv.error())
    tofill['k_dz'            ] = final_cand.trk.bestTrack().dz(final_cand.pv.position())
    tofill['k_dz_e'          ] = final_cand.trk.bestTrack().dzError()
    tofill['k_dz_sig'        ] = final_cand.trk.bestTrack().dz(final_cand.pv.position()) / final_cand.trk.bestTrack().dzError()
    tofill['k_bs_dxy'        ] = final_cand.trk.bestTrack().dxy(final_cand.bs.position())
    tofill['k_bs_dxy_e'      ] = final_cand.trk.bestTrack().dxyError(final_cand.bs.position(), final_cand.bs.error())
    tofill['k_bs_dxy_sig'    ] = final_cand.trk.bestTrack().dxy(final_cand.bs.position()) / final_cand.trk.bestTrack().dxyError(final_cand.bs.position(), final_cand.bs.error())
    tofill['k_cov_pos_def'   ] = final_cand.trk.is_cov_pos_def
    tofill['k_det_cov'       ] = np.linalg.det(final_cand.trk.cov)

    if mc and good_gen_matching:
        tofill['ele1_gen_pt'   ] = myb.e1.pt()
        tofill['ele1_gen_eta'  ] = myb.e1.eta()
        tofill['ele1_gen_phi'  ] = myb.e1.phi()
        tofill['ele1_gen_e'    ] = myb.e1.energy()
        tofill['ele1_gen_match'] = deltaR(myb.e1, final_cand.ele1) < 0.2

        tofill['ele2_gen_pt'   ] = myb.e2.pt()
        tofill['ele2_gen_eta'  ] = myb.e2.eta()
        tofill['ele2_gen_phi'  ] = myb.e2.phi()
        tofill['ele2_gen_e'    ] = myb.e2.energy()
        tofill['ele2_gen_match'] = deltaR(myb.e2, final_cand.ele2) < 0.2

        tofill['k_gen_pt'      ] = myb.k.pt()
        tofill['k_gen_eta'     ] = myb.k.eta()
        tofill['k_gen_phi'     ] = myb.k.phi()
        tofill['k_gen_e'       ] = myb.k.energy()
        tofill['k_gen_match'   ] = deltaR(myb.k, final_cand.trk) < 0.2

        tofill['ee_gen_pt'     ] = (myb.e1.p4() + myb.e2.p4()).pt()
        tofill['ee_gen_eta'    ] = (myb.e1.p4() + myb.e2.p4()).eta()
        tofill['ee_gen_phi'    ] = (myb.e1.p4() + myb.e2.p4()).phi()
        tofill['ee_gen_mass'   ] = (myb.e1.p4() + myb.e2.p4()).mass()

        tofill['b_gen_pt'      ] = myb.pt()
        tofill['b_gen_eta'     ] = myb.eta()
        tofill['b_gen_phi'     ] = myb.phi()
        tofill['b_gen_mass'    ] = myb.mass()
        tofill['b_gen_q2bin'   ] = myb.q2bin
        tofill['b_gen_match'   ] = (tofill['k_gen_match'] and tofill['ele1_gen_match'] and tofill['ele2_gen_match'])

    ntuple.Fill(array('f', tofill.values()))
            
fout.cd()
ntuple.Write()
fout.Close()

