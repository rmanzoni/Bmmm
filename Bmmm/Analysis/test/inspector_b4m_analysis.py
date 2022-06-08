'''
https://link.springer.com/content/pdf/10.1134/S1063778818030092.pdf
https://arxiv.org/pdf/1812.06004.pdf
https://link.springer.com/content/pdf/10.1140/epjc/s10052-019-7112-x.pdf

Example:
ipython -i -- inspector_bmmm_analysis.py --inputFiles="../../../../../rds/CMSSW_10_6_28/src/Bmmm/MINI/Bmmm_signal_MINI.root" --filename=signal --mc
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
from Bmmm.Analysis.B4MuBranches import branches, paths
from Bmmm.Analysis.B4MuCandidate import B4MuCandidate as Candidate

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
    for k, v in tofill.items():
        tofill[k] = np.nan

    # access the handles
    for k, v in handles.iteritems():
        event.getByLabel(v[0], v[1])
        setattr(event, k, v[1].product())
    
    if mc:
        for k, v in handles_mc.iteritems():
            event.getByLabel(v[0], v[1])
            setattr(event, k, v[1].product())
    
    lumi = event.eventAuxiliary().luminosityBlock()
    iev  = event.eventAuxiliary().event()

    ######################################################################################
    #####      RECO PART HERE (GEN PART REMOVED FOR NOW)
    ######################################################################################
    
    # yeah, fire some trigger at least! For now, I've hard coded HLT_Mu7_IP4_part0
    trg_names = event.object().triggerNames(event.trg_res)

    hlt_passed = False

    for iname in trg_names.triggerNames():
        if 'part0' not in iname: continue
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
    
    # trigger matching
    # these are the filters, MAYBE!! too lazy to check confDB. Or, more appropriately: confDB sucks
    # https://github.com/cms-sw/cmssw/blob/6d2f66057131baacc2fcbdd203588c41c885b42c/Configuration/Skimming/python/pwdgSkimBPark_cfi.py#L11-L18 
    good_tobjs = []
    for to in [to for to in event.tobjs if to.pt()>6.5 and abs(to.eta())<2.]:
        to.unpackFilterLabels(event.object(), event.trg_res)
        if to.hasFilterLabel('hltL3fL1sMu22OrParkL1f0L2f10QL3Filtered7IP4Q'):
            good_tobjs.append(to)
            
    muons = [mu for mu in event.muons if mu.pt()>1. and abs(mu.eta())<2.5 and mu.isPFMuon() and mu.isGlobalMuon()]
    muons.sort(key = lambda x : x.pt(), reverse = True)

    if len(muons)<4:
        continue

    # build analysis candidates

    cands = []

    for itriplet in combinations(muons, 4): 

        # 4 muon candidate
        cand = Candidate(itriplet, event.vtx, event.bs)
        
        # 4 muons somewhat close in dz, max distance 1 cm
        if max([abs( imu.bestTrack().dz(cand.pv.position()) - jmu.bestTrack().dz(cand.pv.position()) ) for imu, jmu in combinations(cand.muons, 2)])>1: 
            continue
        
        # filter by mass, first
        if cand.mass()<4. or cand.mass()>7.:
            continue
                   
        # trigger matching, at least one muon matched. 
        # Later one can save the best matched trigger object to each muon, but let me keep it simple for now
        if sum([deltaR(ipair[0], ipair[1])<0.15 for ipair in product(itriplet, good_tobjs)])==0:
            continue
        
        # valid vertex
        if not cand.vtx.isValid():
            continue
        
        # if you made it this far, then save the candidate
        cands.append(cand)

    # if no cands at this point, you might as well move on to the next event
    if len(cands)==0:
        continue

    # sort candidates by charge combination and best pointing angle, i.e. cosine closer to 1
    # can implement and use other criteria later
    cands.sort(key = lambda x : (abs(x.charge())==1, x.vtx.cos), reverse = True)
    final_cand = cands[0]
      
    # fill the tree    
    # can make it smarter with lambda functions associated to the def of branches             
    tofill['run'   ] = event.eventAuxiliary().run()
    tofill['lumi'  ] = event.eventAuxiliary().luminosityBlock()
    tofill['event' ] = event.eventAuxiliary().event()
    tofill['npv'   ] = len(event.vtx)
    tofill['ncands'] = len(cands)

    tofill['mass'  ] = final_cand.mass()
    tofill['mcorr' ] = final_cand.mass_corrected()
    tofill['pt'    ] = final_cand.pt()
    tofill['eta'   ] = final_cand.eta()
    tofill['phi'   ] = final_cand.phi()
    tofill['charge'] = final_cand.charge()

    tofill['dr'    ] = final_cand.r()
    tofill['dr_max'] = final_cand.max_dr()
    tofill['dr_12' ] = final_cand.dr12()
    tofill['dr_13' ] = final_cand.dr13()
    tofill['dr_14' ] = final_cand.dr14()
    tofill['dr_23' ] = final_cand.dr23()
    tofill['dr_24' ] = final_cand.dr24()
    tofill['dr_34' ] = final_cand.dr34()

    tofill['charge_12' ] = final_cand.charge12()
    tofill['charge_13' ] = final_cand.charge13()
    tofill['charge_14' ] = final_cand.charge14()
    tofill['charge_23' ] = final_cand.charge23()
    tofill['charge_24' ] = final_cand.charge24()
    tofill['charge_34' ] = final_cand.charge34()

    tofill['mass_12' ] = final_cand.mass12()
    tofill['mass_13' ] = final_cand.mass13()
    tofill['mass_14' ] = final_cand.mass14()
    tofill['mass_23' ] = final_cand.mass23()
    tofill['mass_24' ] = final_cand.mass24()
    tofill['mass_34' ] = final_cand.mass34()
    tofill['min_mass'] = min([final_cand.mass12(), 
                              final_cand.mass13(), 
                              final_cand.mass14(),
                              final_cand.mass23(),
                              final_cand.mass24(),
                              final_cand.mass34(),])

    tofill['pv_x' ] = final_cand.pv.position().x()
    tofill['pv_y' ] = final_cand.pv.position().y()
    tofill['pv_z' ] = final_cand.pv.position().z()

    tofill['bs_x0'] = event.bs.x0()
    tofill['bs_y0'] = event.bs.y0()
    tofill['bs_z0'] = event.bs.z0()

    tofill['bs_x'] = final_cand.bs.position().x()
    tofill['bs_y'] = final_cand.bs.position().y()

    tofill['vx'] = final_cand.vtx.position().x()
    tofill['vy'] = final_cand.vtx.position().y()
    tofill['vz'] = final_cand.vtx.position().z()
    tofill['vtx_chi2'] = final_cand.vtx.chi2
    tofill['vtx_prob'] = final_cand.vtx.prob

    tofill['cos2d'  ] = final_cand.vtx.cos
    tofill['lxy'    ] = final_cand.lxy.value()
    tofill['lxy_err'] = final_cand.lxy.error()
    tofill['lxy_sig'] = final_cand.lxy.significance()
   
    tofill['mu1_pt'         ] = final_cand.mu1.pt()
    tofill['mu1_eta'        ] = final_cand.mu1.eta()
    tofill['mu1_phi'        ] = final_cand.mu1.phi()
    tofill['mu1_e'          ] = final_cand.mu1.energy()
    tofill['mu1_mass'       ] = final_cand.mu1.mass()
    tofill['mu1_charge'     ] = final_cand.mu1.charge()
    tofill['mu1_id_loose'   ] = final_cand.mu1.isLooseMuon()
    tofill['mu1_id_soft'    ] = final_cand.mu1.isMediumMuon()
    tofill['mu1_id_medium'  ] = final_cand.mu1.isSoftMuon(final_cand.pv)
    tofill['mu1_id_tight'   ] = final_cand.mu1.isTightMuon(final_cand.pv)
    tofill['mu1_id_soft_mva'] = final_cand.mu1.softMvaValue()
    tofill['mu1_dxy'        ] = final_cand.mu1.bestTrack().dxy(final_cand.pv.position())
    tofill['mu1_dxy_e'      ] = final_cand.mu1.bestTrack().dxyError(final_cand.pv.position(), final_cand.pv.error())
    tofill['mu1_dxy_sig'    ] = final_cand.mu1.bestTrack().dxy(final_cand.pv.position()) / final_cand.mu1.bestTrack().dxyError(final_cand.pv.position(), final_cand.pv.error())
    tofill['mu1_dz'         ] = final_cand.mu1.bestTrack().dz(final_cand.pv.position())
    tofill['mu1_dz_e'       ] = final_cand.mu1.bestTrack().dzError()
    tofill['mu1_dz_sig'     ] = final_cand.mu1.bestTrack().dz(final_cand.pv.position()) / final_cand.mu1.bestTrack().dzError()
    tofill['mu1_bs_dxy'     ] = final_cand.mu1.bestTrack().dxy(final_cand.bs.position())
    tofill['mu1_bs_dxy_e'   ] = final_cand.mu1.bestTrack().dxyError(final_cand.bs.position(), final_cand.bs.error())
    tofill['mu1_bs_dxy_sig' ] = final_cand.mu1.bestTrack().dxy(final_cand.bs.position()) / final_cand.mu1.bestTrack().dxyError(final_cand.bs.position(), final_cand.bs.error())
    tofill['mu1_cov_pos_def'] = final_cand.mu1.is_cov_pos_def

    tofill['mu2_pt'         ] = final_cand.mu2.pt()
    tofill['mu2_eta'        ] = final_cand.mu2.eta()
    tofill['mu2_phi'        ] = final_cand.mu2.phi()
    tofill['mu2_e'          ] = final_cand.mu2.energy()
    tofill['mu2_mass'       ] = final_cand.mu2.mass()
    tofill['mu2_charge'     ] = final_cand.mu2.charge()
    tofill['mu2_id_loose'   ] = final_cand.mu2.isLooseMuon()
    tofill['mu2_id_soft'    ] = final_cand.mu2.isMediumMuon()
    tofill['mu2_id_medium'  ] = final_cand.mu2.isSoftMuon(final_cand.pv)
    tofill['mu2_id_tight'   ] = final_cand.mu2.isTightMuon(final_cand.pv)
    tofill['mu2_id_soft_mva'] = final_cand.mu2.softMvaValue()
    tofill['mu2_dxy'        ] = final_cand.mu2.bestTrack().dxy(final_cand.pv.position())
    tofill['mu2_dxy_e'      ] = final_cand.mu2.bestTrack().dxyError(final_cand.pv.position(), final_cand.pv.error())
    tofill['mu2_dxy_sig'    ] = final_cand.mu2.bestTrack().dxy(final_cand.pv.position()) / final_cand.mu2.bestTrack().dxyError(final_cand.pv.position(), final_cand.pv.error())
    tofill['mu2_dz'         ] = final_cand.mu2.bestTrack().dz(final_cand.pv.position())
    tofill['mu2_dz_e'       ] = final_cand.mu2.bestTrack().dzError()
    tofill['mu2_dz_sig'     ] = final_cand.mu2.bestTrack().dz(final_cand.pv.position()) / final_cand.mu2.bestTrack().dzError()
    tofill['mu2_bs_dxy'     ] = final_cand.mu2.bestTrack().dxy(final_cand.bs.position())
    tofill['mu2_bs_dxy_e'   ] = final_cand.mu2.bestTrack().dxyError(final_cand.bs.position(), final_cand.bs.error())
    tofill['mu2_bs_dxy_sig' ] = final_cand.mu2.bestTrack().dxy(final_cand.bs.position()) / final_cand.mu2.bestTrack().dxyError(final_cand.bs.position(), final_cand.bs.error())
    tofill['mu2_cov_pos_def'] = final_cand.mu2.is_cov_pos_def

    tofill['mu3_pt'         ] = final_cand.mu3.pt()
    tofill['mu3_eta'        ] = final_cand.mu3.eta()
    tofill['mu3_phi'        ] = final_cand.mu3.phi()
    tofill['mu3_e'          ] = final_cand.mu3.energy()
    tofill['mu3_mass'       ] = final_cand.mu3.mass()
    tofill['mu3_charge'     ] = final_cand.mu3.charge()
    tofill['mu3_id_loose'   ] = final_cand.mu3.isLooseMuon()
    tofill['mu3_id_soft'    ] = final_cand.mu3.isMediumMuon()
    tofill['mu3_id_medium'  ] = final_cand.mu3.isSoftMuon(final_cand.pv)
    tofill['mu3_id_tight'   ] = final_cand.mu3.isTightMuon(final_cand.pv)
    tofill['mu3_id_soft_mva'] = final_cand.mu3.softMvaValue()
    tofill['mu3_dxy'        ] = final_cand.mu3.bestTrack().dxy(final_cand.pv.position())
    tofill['mu3_dxy_e'      ] = final_cand.mu3.bestTrack().dxyError(final_cand.pv.position(), final_cand.pv.error())
    tofill['mu3_dxy_sig'    ] = final_cand.mu3.bestTrack().dxy(final_cand.pv.position()) / final_cand.mu3.bestTrack().dxyError(final_cand.pv.position(), final_cand.pv.error())
    tofill['mu3_dz'         ] = final_cand.mu3.bestTrack().dz(final_cand.pv.position())
    tofill['mu3_dz_e'       ] = final_cand.mu3.bestTrack().dzError()
    tofill['mu3_dz_sig'     ] = final_cand.mu3.bestTrack().dz(final_cand.pv.position()) / final_cand.mu3.bestTrack().dzError()
    tofill['mu3_bs_dxy'     ] = final_cand.mu3.bestTrack().dxy(final_cand.bs.position())
    tofill['mu3_bs_dxy_e'   ] = final_cand.mu3.bestTrack().dxyError(final_cand.bs.position(), final_cand.bs.error())
    tofill['mu3_bs_dxy_sig' ] = final_cand.mu3.bestTrack().dxy(final_cand.bs.position()) / final_cand.mu3.bestTrack().dxyError(final_cand.bs.position(), final_cand.bs.error())
    tofill['mu3_cov_pos_def'] = final_cand.mu3.is_cov_pos_def
    
    tofill['mu4_pt'         ] = final_cand.mu4.pt()
    tofill['mu4_eta'        ] = final_cand.mu4.eta()
    tofill['mu4_phi'        ] = final_cand.mu4.phi()
    tofill['mu4_e'          ] = final_cand.mu4.energy()
    tofill['mu4_mass'       ] = final_cand.mu4.mass()
    tofill['mu4_charge'     ] = final_cand.mu4.charge()
    tofill['mu4_id_loose'   ] = final_cand.mu4.isLooseMuon()
    tofill['mu4_id_soft'    ] = final_cand.mu4.isMediumMuon()
    tofill['mu4_id_medium'  ] = final_cand.mu4.isSoftMuon(final_cand.pv)
    tofill['mu4_id_tight'   ] = final_cand.mu4.isTightMuon(final_cand.pv)
    tofill['mu4_id_soft_mva'] = final_cand.mu4.softMvaValue()
    tofill['mu4_dxy'        ] = final_cand.mu4.bestTrack().dxy(final_cand.pv.position())
    tofill['mu4_dxy_e'      ] = final_cand.mu4.bestTrack().dxyError(final_cand.pv.position(), final_cand.pv.error())
    tofill['mu4_dxy_sig'    ] = final_cand.mu4.bestTrack().dxy(final_cand.pv.position()) / final_cand.mu4.bestTrack().dxyError(final_cand.pv.position(), final_cand.pv.error())
    tofill['mu4_dz'         ] = final_cand.mu4.bestTrack().dz(final_cand.pv.position())
    tofill['mu4_dz_e'       ] = final_cand.mu4.bestTrack().dzError()
    tofill['mu4_dz_sig'     ] = final_cand.mu4.bestTrack().dz(final_cand.pv.position()) / final_cand.mu4.bestTrack().dzError()
    tofill['mu4_bs_dxy'     ] = final_cand.mu4.bestTrack().dxy(final_cand.bs.position())
    tofill['mu4_bs_dxy_e'   ] = final_cand.mu4.bestTrack().dxyError(final_cand.bs.position(), final_cand.bs.error())
    tofill['mu4_bs_dxy_sig' ] = final_cand.mu4.bestTrack().dxy(final_cand.bs.position()) / final_cand.mu4.bestTrack().dxyError(final_cand.bs.position(), final_cand.bs.error())
    tofill['mu4_cov_pos_def'] = final_cand.mu4.is_cov_pos_def

    # yeah, need to to work a bit on the definition of isolation    
#     mu_ch_isolation = max(0., mu.pfIsolationR03().sumChargedHadronPt - sum([ip.pt() for ip in [kp, km, pi] if deltaR(ip, mu)<0.3 and abs(mu.bestTrack().dz(the_pv.position()) - ip.dz(the_pv.position()))<0.2]))
#     mu_dbeta_neutral_isolation = max(mu.pfIsolationR03().sumNeutralHadronEt + mu.pfIsolationR03().sumPhotonEt - mu.pfIsolationR03().sumPUPt/2,0.0)
#     
#     tofill['mu_ch_iso'    ] = mu_ch_isolation
#     tofill['mu_db_n_iso'  ] = mu_dbeta_neutral_isolation
#     tofill['mu_abs_iso'   ] = (mu_ch_isolation + mu_dbeta_neutral_isolation)
#     tofill['mu_rel_iso'   ] = (mu_ch_isolation + mu_dbeta_neutral_isolation) / mu.pt()
        
    ntuple.Fill(array('f', tofill.values()))
            
fout.cd()
ntuple.Write()
fout.Close()

