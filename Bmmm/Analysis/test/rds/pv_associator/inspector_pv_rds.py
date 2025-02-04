from __future__ import print_function
import os
import re
import ROOT
import argparse
import numpy as np
import pickle
import uproot
import pandas as pd
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
#from cuts import cuts_tight, cuts_loose, cuts_gen
from Bmmm.Analysis.RDsBranches import branches, paths, event_branches, cand_branches, muon_branches, track_branches
from Bmmm.Analysis.RDsCandidate import RDsCandidate
from Bmmm.Analysis.utils import drop_hlt_version, diquarks, excitedBs, isAncestor, masses, p4_with_mass, cutflow, fillRecoTree, isMyDs, convert_cov, is_pos_def, fix_track, printAncestors, compute_IP3D

to_check = [
(1523,  2199),
(1523,  8440),
(1523,  1772),
(1523,  3670),
(1523,  9273),
(1523,  8708),
(1523,  3139),
(1523, 17743),
(1523,  7760),
(1523, 17175),
(1523, 19333),
(1523, 16520),
(1523, 26436),
(1523, 24227),
(1523, 25882),
(1523, 29609),
(1523, 34677),
(1523, 27007),
(1523, 29777),
(1523, 31445),
(1523, 31375),
(1523, 34050),
(1523, 32851),
(1523, 33719),
(1523, 37712),
]

def find_first_b(bs):
    if particle.PDGID(bs.mother(0).pdgId()).has_bottom:
        #print('beauty mother', bs.mother(0).pdgId())
        return find_first_b(bs.mother(0))
    else:
        #print('non beauty mother', bs.mother(0).pdgId())
        return bs
          
branches = [   
    'run'         ,
    'lumi'        ,
    'event'       ,

    'npv'         ,
    
    'pt'          ,
    'eta'         ,
    'phi'         ,
    'mass'        ,
    'charge'      ,
    'sig'         ,
    
    'pv_x'        ,
    'pv_y'        ,
    'pv_z'        ,

    'bs_x0'       ,
    'bs_y0'       ,
    'bs_z0'       ,

    'sv_x'        , 
    'sv_y'        , 
    'sv_z'        , 

    'tv_x'        , 
    'tv_y'        , 
    'tv_z'        , 

    'qv_x'        , 
    'qv_y'        , 
    'qv_z'        , 
    
    'ds_pt'       ,
    'ds_eta'      ,
    'ds_phi'      ,
    'ds_mass'     ,
    'ds_charge'   ,
    'ds_is_phikk' ,
    
    'ds_st_pt'    ,
    'ds_st_eta'   ,
    'ds_st_phi'   ,
    'ds_st_mass'  ,
    'ds_st_charge',

    'ph_pt'       ,
    'ph_eta'      ,
    'ph_phi'      ,
    'ph_mass'     ,
    'ph_charge'   ,

    'piz_pt'      ,
    'piz_eta'     ,
    'piz_phi'     ,
    'piz_mass'    ,
    'piz_charge'  ,

    'pi_pt'       ,
    'pi_eta'      ,
    'pi_phi'      ,
    'pi_mass'     ,
    'pi_charge'   ,

    'kp_pt'       ,
    'kp_eta'      ,
    'kp_phi'      ,
    'kp_mass'     ,
    'kp_charge'   ,

    'km_pt'       ,
    'km_eta'      ,
    'km_phi'      ,
    'km_mass'     ,
    'km_charge'   ,

    'phi_pt'      ,
    'phi_eta'     ,
    'phi_phi'     ,
    'phi_mass'    ,
    'phi_charge'  ,
    
    'mu_pt'       ,
    'mu_eta'      ,
    'mu_phi'      ,
    'mu_mass'     ,
    'mu_charge'   ,
    
    'tau_pt'      ,
    'tau_eta'     ,
    'tau_phi'     ,
    'tau_mass'    ,
    'tau_charge'  ,

    'reco_first_pv_x',
    'reco_first_pv_y',
    'reco_first_pv_z',

    'closest_dz_mu_pv_x',
    'closest_dz_mu_pv_y',
    'closest_dz_mu_pv_z',

    'min_ip3d_pv_x',
    'min_ip3d_pv_y',
    'min_ip3d_pv_z',

    'min_ip3d_pv_x_err',
    'min_ip3d_pv_y_err',
    'min_ip3d_pv_z_err',

    'min_ip3d_pv_ntracks',
    'min_ip3d_pv_tracksize',

    'max_cos2D_pv_x',       
    'max_cos2D_pv_y',       
    'max_cos2D_pv_z',       
    
    'max_cos2D_pv_ntracks'  ,
    'max_cos2D_pv_tracksize',
    'max_cos2D_pv_x_err'    ,
    'max_cos2D_pv_y_err'    ,
    'max_cos2D_pv_z_err'    ,
    
    'max_cos3D_pv_x',      
    'max_cos3D_pv_y',      
    'max_cos3D_pv_z',      
    
    'max_cos3D_pv_ntracks'  ,
    'max_cos3D_pv_tracksize',
    'max_cos3D_pv_x_err'    ,
    'max_cos3D_pv_y_err'    ,
    'max_cos3D_pv_z_err'    ,

    'reco_mu_pt'    ,
    'reco_mu_eta'   ,
    'reco_mu_phi'   ,
    'reco_mu_mass'  ,
    'reco_mu_charge',

    'reco_mu_iso04_sumChargedHadronPt'             ,
    'reco_mu_iso04_sumChargedParticlePt'           ,
    'reco_mu_iso04_sumNeutralHadronEt'             ,
    'reco_mu_iso04_sumNeutralHadronEtHighThreshold',
    'reco_mu_iso04_sumPUPt'                        ,
    'reco_mu_iso04_sumPhotonEt'                    ,
    'reco_mu_iso04_sumPhotonEtHighThreshold'       ,
    'reco_mu_iso04'                                ,
    'reco_mu_rel_iso04'                            ,


    'reco_mu_clean_new_iso04_sumChargedHadronPt'       ,
    'reco_mu_clean_new_iso04_sumPUPt'                  ,
    'reco_mu_new_iso04_sumChargedHadronPt'             ,
    'reco_mu_new_iso04_sumChargedParticlePt'           ,
    'reco_mu_new_iso04_sumNeutralHadronEt'             ,
    'reco_mu_new_iso04_sumNeutralHadronEtHighThreshold',
    'reco_mu_new_iso04_sumPUPt'                        ,
    'reco_mu_new_iso04_sumPhotonEt'                    ,
    'reco_mu_new_iso04_sumPhotonEtHighThreshold'       ,
    'reco_mu_new_iso04'                                ,
    'reco_mu_clean_new_rel_iso04'                      ,
    'reco_mu_new_rel_iso04'                            ,

    'reco_kp_pt'    ,
    'reco_kp_eta'   ,
    'reco_kp_phi'   ,
    'reco_kp_mass'  ,
    'reco_kp_charge',

    'reco_km_pt'    ,
    'reco_km_eta'   ,
    'reco_km_phi'   ,
    'reco_km_mass'  ,
    'reco_km_charge',

    'reco_pi_pt'    ,
    'reco_pi_eta'   ,
    'reco_pi_phi'   ,
    'reco_pi_mass'  ,
    'reco_pi_charge',

    'reco_ds_pt'    ,
    'reco_ds_eta'   ,
    'reco_ds_phi'   ,
    'reco_ds_mass'  ,
    'reco_ds_charge',

    'reco_dsmu_pt'    ,
    'reco_dsmu_eta'   ,
    'reco_dsmu_phi'   ,
    'reco_dsmu_mass'  ,
    'reco_dsmu_charge',

    'reco_sv_x',
    'reco_sv_y',
    'reco_sv_z',

    'reco_tv_x',
    'reco_tv_y',
    'reco_tv_z',

    'reco_qv_x',
    'reco_qv_y',
    'reco_qv_z',     

    'min_ip3d_pv_w_bs_no_sig_tk_x',
    'min_ip3d_pv_w_bs_no_sig_tk_y',
    'min_ip3d_pv_w_bs_no_sig_tk_z',

    'bs_at_min_ip3d_pv_w_bs_no_sig_tk_x',
    'bs_at_min_ip3d_pv_w_bs_no_sig_tk_y',

    'min_ip3d_pv_w_bs_x',
    'min_ip3d_pv_w_bs_y',
    'min_ip3d_pv_w_bs_z',
    
    'matched_pv_x',
    'matched_pv_y',
    'matched_pv_z',

    'miniaod_min_ip3d_pv_ntracks',
    'miniaod_min_ip3d_pv_tracksize',
    
    'miniaod_min_ip3d_pv_x',
    'miniaod_min_ip3d_pv_y',
    'miniaod_min_ip3d_pv_z',

    'miniaod_min_ip3d_pv_x_err',
    'miniaod_min_ip3d_pv_y_err',
    'miniaod_min_ip3d_pv_z_err',
    
    'distance',

    'reco_mu_clean_custom_iso04_sumChargedHadronPt',
    'reco_mu_clean_custom_iso04_sumPUPt'           ,
    'reco_mu_custom_iso04_sumChargedHadronPt'      ,
    'reco_mu_custom_iso04_sumPUPt'                 ,
    'reco_mu_custom_iso04_sumNeutralHadronEt'      ,
    'reco_mu_custom_iso04_sumPhotonEt'             ,
    'reco_mu_clean_custom_iso04'                   ,
    'reco_mu_custom_iso04'                         ,
    'reco_mu_clean_custom_rel_iso04'               ,
    'reco_mu_custom_rel_iso04'                     ,

    'lxyz_min_ip3d_pv_vtx'                    ,
    'lxyz_min_ip3d_pv_vtx_sig'                ,
    'lxyz_min_ip3d_pv_vtx_err'                ,
        
    'lxy_min_ip3d_pv_vtx'                     ,
    'lxy_min_ip3d_pv_vtx_sig'                 ,
    'lxy_min_ip3d_pv_vtx_err'                 ,
        
    'lxyz_min_ip3d_pv_w_bs_vtx'               ,
    'lxyz_min_ip3d_pv_w_bs_vtx_sig'           ,
    'lxyz_min_ip3d_pv_w_bs_vtx_err'           ,
        
    'lxy_min_ip3d_pv_w_bs_vtx'                ,
    'lxy_min_ip3d_pv_w_bs_vtx_sig'            ,
    'lxy_min_ip3d_pv_w_bs_vtx_err'            ,
        
    'lxyz_min_ip3d_pv_clean_w_bs_vtx'         ,
    'lxyz_min_ip3d_pv_clean_w_bs_vtx_sig'     ,
    'lxyz_min_ip3d_pv_clean_w_bs_vtx_err'     ,
        
    'lxy_min_ip3d_pv_clean_w_bs_vtx'          ,
    'lxy_min_ip3d_pv_clean_w_bs_vtx_sig'      ,
    'lxy_min_ip3d_pv_clean_w_bs_vtx_err'      ,
    
    'reco_mu_ip3d_sv'     ,
    'reco_mu_ip3d_sv_err' ,
    'reco_mu_ip3d_sv_sig' ,

    'reco_mu_btv_ip3d_sv'     ,
    'reco_mu_btv_ip3d_sv_err' ,
    'reco_mu_btv_ip3d_sv_sig' ,

    'ds_ch_iso04' ,
    'ds_nh_iso04' ,
    'ds_ph_iso04' ,
    'ds_pu_iso04' ,
    'ds_iso04'    ,
    'ds_iso04_rel',

    'mu_track_used',
    'kp_track_used',
    'km_track_used',
    'pi_track_used',

    'reco_mu_ip3d_ds_sv'    ,
    'reco_mu_ip3d_ds_sv_err',
    'reco_mu_ip3d_ds_sv_sig',
    
    'reco_mu_btv_ip3d_ds_sv'    ,
    'reco_mu_btv_ip3d_ds_sv_err',
    'reco_mu_btv_ip3d_ds_sv_sig',

    'reco_mu_dl3d_ds_sv'    ,
    'reco_mu_dl3d_ds_sv_err',
    'reco_mu_dl3d_ds_sv_sig',

    'reco_mu_dl3d_ds_tv'    ,
    'reco_mu_dl3d_ds_tv_err',
    'reco_mu_dl3d_ds_tv_sig',

    'reco_mu_dl3d_tv_sv_sv'    ,
    'reco_mu_dl3d_tv_sv_sv_err',
    'reco_mu_dl3d_tv_sv_sv_sig',

    'reco_mu_dl3d_tv_sv_tv'    ,
    'reco_mu_dl3d_tv_sv_tv_err',
    'reco_mu_dl3d_tv_sv_tv_sig',

    'mu_ch_sv_iso04' ,
    'mu_nh_sv_iso04' ,
    'mu_ph_sv_iso04' ,
    'mu_pu_sv_iso04' ,
    'mu_sv_iso04'    ,
    'mu_sv_iso04_rel',
    
    'acoplanarity',
    'sphericity',
    'aplanarity',
    'planarity' ,

]

import particle
from particle import Particle
ROOT.gSystem.Load('libVtxFitFitter')
from ROOT import KVFitter # VertexDistance3D is contained here, dirt trick!!
from ROOT import RDsKinVtxFitter
from ROOT import PVRefitter

kinfit = RDsKinVtxFitter()    
vtxfit = KVFitter()
pvrefitter = PVRefitter()
tofit = ROOT.std.vector('reco::Track')()

'''
ipython -i -- inspector_pv_rds.py --inputFiles="/pnfs/psi.ch/cms/trivcat/store/user/manzoni/all_signals_HbToDsPhiKKPiMuNu_MT_MINI_21jan23_v1/all_signals_HbToDsPhiKKPiMuNu_MT_158.root" --filename="pv_resolution" --mc


ipython -i -- inspector_pv_rds.py --inputFiles="/pnfs/psi.ch/cms/trivcat/store/user/manzoni/all_signals_HbToDsPhiKKPiMuNu_MT_MINI_21jan23_v1_PV_REFITTED/refittedMiniAOD.root.*" --filename="pv_resolution_refit" --mc
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

#cuts = cuts_loose if loosecuts else cuts_tight
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
#handles_mc['genp'   ] = ('genParticles'  , Handle('std::vector<reco::GenParticle>'))
handles_mc['genp'   ] = ('prunedGenParticles', Handle('std::vector<reco::GenParticle>'))
handles_mc['genInfo'] = ('generator'         , Handle('GenEventInfoProduct')           )

handles = OrderedDict()
handles['muons'  ] = ('slimmedMuons'                  , Handle('std::vector<pat::Muon>')            )
handles['trk'    ] = ('packedPFCandidates'            , Handle('std::vector<pat::PackedCandidate>') )
handles['vtx'    ] = ('offlineSlimmedPrimaryVertices' , Handle('std::vector<reco::Vertex>')         )
handles['bs'     ] = ('offlineBeamSpot'               , Handle('reco::BeamSpot')                    )

handles['refit_trk'     ] = (('unpackedTracksAndVertices', ''         , 'VertexRefit'), Handle('vector<reco::Track> ') )
handles['refit_pv'      ] = (('primaryVertexRefit'       , ''         , 'VertexRefit'), Handle('vector<reco::Vertex>') )
handles['refit_pv_w_bs' ] = (('primaryVertexRefit'       , 'WithBS'   , 'VertexRefit'), Handle('vector<reco::Vertex>') )
handles['refit_vtx'     ] = (('unpackedTracksAndVertices', ''         , 'VertexRefit'), Handle('vector<reco::Vertex>') )
handles['refit_vtx_sec' ] = (('unpackedTracksAndVertices', 'secondary', 'VertexRefit'), Handle('vector<reco::Vertex>') )

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
row_list = []

for i, event in enumerate(events):
    
    if (i+1) > maxevents:
        break
            
    if i%logfreq == 0:
        percentage = float(i) / maxevents * 100.
        speed = float(i) / (time() - start)
        eta = datetime.now() + timedelta(seconds=(maxevents-i) / max(0.1, speed))
        print('\t===> processing %d / %d event \t completed %.1f%s \t %.1f ev/s \t ETA %s s' %(i, maxevents, percentage, '%', speed, eta.strftime('%Y-%m-%d %H:%M:%S')))

    tofill = OrderedDict(zip(branches, [np.nan]*len(branches)))

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

        event.genp = [ip for ip in event.genp]
    
        bs_cands = [ip for ip in event.genp if abs(ip.pdgId())==531 and sum([abs(ip.daughter(idau).pdgId()) in [433, 431, 13, 15] for idau in range(ip.numberOfDaughters())])==2 ]        

#         all_bs_cands = [ip for ip in event.genp if abs(ip.pdgId())==531]
#         for ii in all_bs_cands: 
#             print(ii.pdgId(), ii)
#             print( [ii.daughter(idau).pdgId() for idau in range(ii.numberOfDaughters())] )
#         import ipdb ; ipdb.set_trace()

        for ib in bs_cands:
            ib.ds     = None
            ib.ds_st  = None
            ib.ph     = None
            ib.piz    = None
            ib.mu     = None
            ib.nu_mu  = None
            ib.tau    = None
            ib.nu_tau = None
            ib.vtx    = None
        
            for idau in range(ib.numberOfDaughters()):
                
                dau = ib.daughter(idau)

                if abs(dau.pdgId()) == 431:
                    ib.ds = dau

                if abs(dau.pdgId()) == 13:
                    ib.mu = dau

                if abs(dau.pdgId()) == 14:
                    ib.nu_mu = dau 

                if abs(dau.pdgId()) == 433:
                    ib.ds_st = dau

                    for jdau in range(ib.ds_st.numberOfDaughters()):
                    
                        ddau = ib.ds_st.daughter(jdau)
                    
                        if abs(ddau.pdgId()) == 431:
                            ib.ds = ddau

                        if abs(ddau.pdgId()) == 22:
                            ib.ph = ddau
    
                        if abs(ddau.pdgId()) == 111:
                            ib.piz = ddau


                if abs(dau.pdgId()) == 16:
                    ib.nu_tau = dau

                if abs(dau.pdgId()) == 15:
                    ib.tau = dau

                    for jdau in range(ib.tau.numberOfDaughters()):
                    
                        ddau = ib.tau.daughter(jdau)
                    
                        if abs(ddau.pdgId()) == 13:
                            ib.mu = ddau
    
                        if abs(ddau.pdgId()) == 14:
                            ib.nu_mu = ddau

        bs_cands = [ip for ip in bs_cands if ip.mu is not None and ip.mu.pt()>7 and abs(ip.mu.eta())<1.5 and isMyDs(ip.ds)]
        bs_cands = [ip for ip in bs_cands if ip.ds is not None and getattr(ip.ds, 'kp', None) is not None and getattr(ip.ds, 'km', None) is not None]

#         if len(bs_cands)!=len(all_bs_cands):
#             print('sth fishy')
#             import ipdb ; ipdb.set_trace()

        if len(bs_cands)==0:
            if verbose: 
                print('no candidates, WEIRD!')
                import pdb ; pdb.set_trace()
            continue

        if len(bs_cands)>1:
            if verbose: 
                print('more than one GEN candidate! Total %d candidates' %len(candidates))
                import pdb ; pdb.set_trace()
            pass
        

        bs_cands.sort(key = lambda x : (x.charge()==0, x.p4().pt()), reverse=True)
        bs = bs_cands[0]
              
        daughters = []
        for idx_dau in range(bs.numberOfDaughters()):
            idau = bs.daughter(idx_dau)
            if idau.pdgId()==22:
                continue
            daughters.append(idau.pdgId())
        daughters.sort(key = lambda x : abs(x))
           
        which_signal = -1
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
   
        if which_signal<0:
            import ipdb ; ipdb.set_trace()
   
        tofill['run'         ] = event.eventAuxiliary().run()            
        tofill['lumi'        ] = event.eventAuxiliary().luminosityBlock()
        tofill['event'       ] = event.eventAuxiliary().event()          
 
        tofill['pt'          ] = bs.p4().pt()
        tofill['eta'         ] = bs.p4().eta()
        tofill['phi'         ] = bs.p4().phi()
        tofill['mass'        ] = bs.p4().mass()
        tofill['charge'      ] = bs.charge()
        tofill['sig'         ] = which_signal

        # find the first B, exclude oscillations
        first_b = find_first_b(bs)

        tofill['pv_x'        ] = first_b.vertex().x()     
        tofill['pv_y'        ] = first_b.vertex().y()     
        tofill['pv_z'        ] = first_b.vertex().z()     

        tofill['sv_x'        ] = bs.ds.vertex().x()     
        tofill['sv_y'        ] = bs.ds.vertex().y()     
        tofill['sv_z'        ] = bs.ds.vertex().z()     

        tofill['tv_x'        ] = bs.ds.phi_meson.vertex().x()     
        tofill['tv_y'        ] = bs.ds.phi_meson.vertex().y()     
        tofill['tv_z'        ] = bs.ds.phi_meson.vertex().z()     

        tofill['qv_x'        ] = bs.ds.kp.vertex().x()     
        tofill['qv_y'        ] = bs.ds.kp.vertex().y()     
        tofill['qv_z'        ] = bs.ds.kp.vertex().z()     

        tofill['ds_pt'       ] = bs.ds.pt()
        tofill['ds_eta'      ] = bs.ds.eta()
        tofill['ds_phi'      ] = bs.ds.phi()
        tofill['ds_mass'     ] = bs.ds.mass()
        tofill['ds_charge'   ] = bs.ds.charge()
        tofill['ds_is_phikk' ] = isMyDs(bs.ds)

        tofill['pi_pt'       ] = bs.ds.pion.pt()
        tofill['pi_eta'      ] = bs.ds.pion.eta()
        tofill['pi_phi'      ] = bs.ds.pion.phi()
        tofill['pi_mass'     ] = bs.ds.pion.mass()
        tofill['pi_charge'   ] = bs.ds.pion.charge()

        tofill['kp_pt'       ] = bs.ds.kp.pt()
        tofill['kp_eta'      ] = bs.ds.kp.eta()
        tofill['kp_phi'      ] = bs.ds.kp.phi()
        tofill['kp_mass'     ] = bs.ds.kp.mass()
        tofill['kp_charge'   ] = bs.ds.kp.charge()

        tofill['km_pt'       ] = bs.ds.km.pt()
        tofill['km_eta'      ] = bs.ds.km.eta()
        tofill['km_phi'      ] = bs.ds.km.phi()
        tofill['km_mass'     ] = bs.ds.km.mass()
        tofill['km_charge'   ] = bs.ds.km.charge()

        tofill['phi_pt'      ] = bs.ds.phi_meson.pt()
        tofill['phi_eta'     ] = bs.ds.phi_meson.eta()
        tofill['phi_phi'     ] = bs.ds.phi_meson.phi()
        tofill['phi_mass'    ] = bs.ds.phi_meson.mass()
        tofill['phi_charge'  ] = bs.ds.phi_meson.charge()

        if bs.ds_st:
            tofill['ds_st_pt'    ] = bs.ds_st.pt()
            tofill['ds_st_eta'   ] = bs.ds_st.eta()
            tofill['ds_st_phi'   ] = bs.ds_st.phi()
            tofill['ds_st_mass'  ] = bs.ds_st.mass()
            tofill['ds_st_charge'] = bs.ds_st.charge()

            if bs.ph:
                tofill['ph_pt'       ] = bs.ph.pt()
                tofill['ph_eta'      ] = bs.ph.eta()
                tofill['ph_phi'      ] = bs.ph.phi()
                tofill['ph_mass'     ] = bs.ph.mass()
                tofill['ph_charge'   ] = bs.ph.charge()
    
            if bs.piz:
                tofill['piz_pt'      ] = bs.piz.pt()
                tofill['piz_eta'     ] = bs.piz.eta()
                tofill['piz_phi'     ] = bs.piz.phi()
                tofill['piz_mass'    ] = bs.piz.mass()
                tofill['piz_charge'  ] = bs.piz.charge()

        tofill['mu_pt'       ] = bs.mu.pt()
        tofill['mu_eta'      ] = bs.mu.eta()
        tofill['mu_phi'      ] = bs.mu.phi()
        tofill['mu_mass'     ] = bs.mu.mass()
        tofill['mu_charge'   ] = bs.mu.charge()

        if bs.tau:
            tofill['tau_pt'      ] = bs.tau.pt()
            tofill['tau_eta'     ] = bs.tau.eta()
            tofill['tau_phi'     ] = bs.tau.phi()
            tofill['tau_mass'    ] = bs.tau.mass()
            tofill['tau_charge'  ] = bs.tau.charge()

##########################################################################################
##########################################################################################
#     ____                             __                  __  _           
#    / __ \___  _________  ____  _____/ /________  _______/ /_(_)___  ____ 
#   / /_/ / _ \/ ___/ __ \/ __ \/ ___/ __/ ___/ / / / ___/ __/ / __ \/ __ \
#  / _, _/  __/ /__/ /_/ / / / (__  ) /_/ /  / /_/ / /__/ /_/ / /_/ / / / /
# /_/ |_|\___/\___/\____/_/ /_/____/\__/_/   \__,_/\___/\__/_/\____/_/ /_/ 
#                                                                          
##########################################################################################
##########################################################################################

        tofill['reco_first_pv_x'] = event.vtx.at(0).position().x()
        tofill['reco_first_pv_y'] = event.vtx.at(0).position().y()
        tofill['reco_first_pv_z'] = event.vtx.at(0).position().z()

        tofill['bs_x0'] = event.bs.x0()
        tofill['bs_y0'] = event.bs.y0()
        tofill['bs_z0'] = event.bs.z0()

        tofill['npv'] = len(event.vtx)          
        
##########################################################################################
##########################################################################################
        # muon

        dr2 = np.inf
        
        reco_mu = None
        
        imu, dr2 = bestMatch(bs.mu, event.muons)

        if dr2 < 0.2**2 and imu.charge()==bs.mu.charge():
            reco_mu = imu
            tofill['reco_mu_pt'       ] = reco_mu.pt()
            tofill['reco_mu_eta'      ] = reco_mu.eta()
            tofill['reco_mu_phi'      ] = reco_mu.phi()
            tofill['reco_mu_mass'     ] = reco_mu.mass()
            tofill['reco_mu_charge'   ] = reco_mu.charge()

            ch_pf_iso = reco_mu.pfIsolationR04().sumChargedHadronPt
            nh_pf_iso = reco_mu.pfIsolationR04().sumNeutralHadronEt
            pu_pf_iso = reco_mu.pfIsolationR04().sumPUPt
            ph_pf_iso = reco_mu.pfIsolationR04().sumPhotonEt

            nhhig_pf_iso = reco_mu.pfIsolationR04().sumNeutralHadronEtHighThreshold
            chpar_pf_iso = reco_mu.pfIsolationR04().sumChargedParticlePt
            phhig_pf_iso = reco_mu.pfIsolationR04().sumPhotonEtHighThreshold

            pf_iso = ch_pf_iso + max(0., nh_pf_iso + ph_pf_iso - 0.5*pu_pf_iso)

            tofill['reco_mu_iso04_sumChargedHadronPt'             ] = ch_pf_iso
            tofill['reco_mu_iso04_sumNeutralHadronEt'             ] = nh_pf_iso
            tofill['reco_mu_iso04_sumPUPt'                        ] = pu_pf_iso
            tofill['reco_mu_iso04_sumPhotonEt'                    ] = ph_pf_iso

            tofill['reco_mu_iso04_sumNeutralHadronEtHighThreshold'] = nhhig_pf_iso
            tofill['reco_mu_iso04_sumChargedParticlePt'           ] = chpar_pf_iso
            tofill['reco_mu_iso04_sumPhotonEtHighThreshold'       ] = phhig_pf_iso

            tofill['reco_mu_iso04'                                ] = pf_iso
            tofill['reco_mu_rel_iso04'                            ] = pf_iso/reco_mu.pt()

                                    
        if reco_mu:
            closest_vtx_dz_mu = None
            closest_dz = np.inf
            
            for ivtx in event.vtx:
                dz = abs(reco_mu.bestTrack().dz(ivtx.position())) 
                if  dz < closest_dz:
                    closest_vtx_dz_mu = ivtx
                    closest_dz = dz
                
            tofill['closest_dz_mu_pv_x'] = closest_vtx_dz_mu.position().x()
            tofill['closest_dz_mu_pv_y'] = closest_vtx_dz_mu.position().y()
            tofill['closest_dz_mu_pv_z'] = closest_vtx_dz_mu.position().z()

##########################################################################################
##########################################################################################
        # tracks

        dr2 = np.inf
        
        reco_kp = None
        
        ikp, dr2 = bestMatch(bs.ds.kp, [itk for itk in event.trk if itk.charge()>0 and itk.hasTrackDetails()])

        if dr2 < 0.1**2:
            reco_kp = ikp
            tofill['reco_kp_pt'       ] = reco_kp.pt()
            tofill['reco_kp_eta'      ] = reco_kp.eta()
            tofill['reco_kp_phi'      ] = reco_kp.phi()
            tofill['reco_kp_mass'     ] = reco_kp.mass()
            tofill['reco_kp_charge'   ] = reco_kp.charge()

            reco_kp.new_p4 = p4_with_mass(reco_kp, 0.493677, 1)
        ##################################################################################

        dr2 = np.inf
        
        reco_km = None
        
        ikm, dr2 = bestMatch(bs.ds.km, [itk for itk in event.trk if itk.charge()<0 and itk.hasTrackDetails()])

        if dr2 < 0.1**2:
            reco_km = ikm
            tofill['reco_km_pt'       ] = reco_km.pt()
            tofill['reco_km_eta'      ] = reco_km.eta()
            tofill['reco_km_phi'      ] = reco_km.phi()
            tofill['reco_km_mass'     ] = reco_km.mass()
            tofill['reco_km_charge'   ] = reco_km.charge()

            reco_km.new_p4 = p4_with_mass(reco_km, 0.493677, 1)
        ##################################################################################

        dr2 = np.inf
        
        reco_pi = None
        
        ipi, dr2 = bestMatch(bs.ds.pion, [itk for itk in event.trk if itk.charge()!=0 and itk.hasTrackDetails()])

        if dr2 < 0.1**2:
            reco_pi = ipi
            tofill['reco_pi_pt'       ] = reco_pi.pt()
            tofill['reco_pi_eta'      ] = reco_pi.eta()
            tofill['reco_pi_phi'      ] = reco_pi.phi()
            tofill['reco_pi_mass'     ] = reco_pi.mass()
            tofill['reco_pi_charge'   ] = reco_pi.charge()

        ##################################################################################        
        # reco Ds
        if reco_pi and reco_kp and reco_km:
            reco_ds = reco_km.new_p4 + reco_kp.new_p4 + reco_pi.p4()
            tofill['reco_ds_pt'       ] = reco_ds.pt()
            tofill['reco_ds_eta'      ] = reco_ds.eta()
            tofill['reco_ds_phi'      ] = reco_ds.phi()
            tofill['reco_ds_mass'     ] = reco_ds.mass()
            tofill['reco_ds_charge'   ] = reco_kp.charge() + reco_km.charge() + reco_pi.charge()

        ##################################################################################        
        # reco Ds mu
        if reco_pi and reco_kp and reco_km and reco_mu:
            reco_dsmu = reco_km.new_p4 + reco_kp.new_p4 + reco_pi.p4() + reco_mu.p4()
            tofill['reco_dsmu_pt'       ] = reco_dsmu.pt()
            tofill['reco_dsmu_eta'      ] = reco_dsmu.eta()
            tofill['reco_dsmu_phi'      ] = reco_dsmu.phi()
            tofill['reco_dsmu_mass'     ] = reco_dsmu.mass()
            tofill['reco_dsmu_charge'   ] = reco_kp.charge() + reco_km.charge() + reco_pi.charge() + reco_mu.charge()
        
            
            cand = RDsCandidate(reco_mu, [reco_km, reco_kp], reco_pi, event.vtx, event.bs)
            cand.compute_vtx(full=True)
            cand.compute_kinematics()
            good = cand.run_kin_fit()
            if good:
                cand.get_refitted_quantities()
    
                tofill['reco_sv_x'] = cand.bs_tree.vtx.position().x()
                tofill['reco_sv_y'] = cand.bs_tree.vtx.position().y()
                tofill['reco_sv_z'] = cand.bs_tree.vtx.position().z()
    
                tofill['reco_tv_x'] = cand.ds_tree.vtx.position().x()
                tofill['reco_tv_y'] = cand.ds_tree.vtx.position().y()
                tofill['reco_tv_z'] = cand.ds_tree.vtx.position().z()
    
                tofill['reco_qv_x'] = cand.phi_tree.vtx.position().x()
                tofill['reco_qv_y'] = cand.phi_tree.vtx.position().y()
                tofill['reco_qv_z'] = cand.phi_tree.vtx.position().z()
            
                # find PV as the closest to the Bs flight direction
                # https://www.nagwa.com/en/explainers/939127418581/#:~:text=The%20perpendicular%20distance%20between%20a%20point%20and%20a%20line%20is,any%20point%20on%20the%20line.
                pv_idx = -1
                ip3d_min = np.inf
                for idx, ivtx in enumerate(event.vtx):
                    ip3d = compute_IP3D(ivtx, cand.bs_tree.vtx.position(), cand.p4().Vect())
                    if ip3d<ip3d_min:
                        pv_idx = idx
                        ip3d_min = ip3d
                    
                min_ip3d_pv = event.vtx[pv_idx]

                tofill['min_ip3d_pv_x'] = min_ip3d_pv.x()
                tofill['min_ip3d_pv_y'] = min_ip3d_pv.y()
                tofill['min_ip3d_pv_z'] = min_ip3d_pv.z()

                tofill['min_ip3d_pv_ntracks'  ] = min_ip3d_pv.nTracks()
                tofill['min_ip3d_pv_tracksize'] = min_ip3d_pv.tracksSize()
                tofill['min_ip3d_pv_x_err'    ] = min_ip3d_pv.xError()
                tofill['min_ip3d_pv_y_err'    ] = min_ip3d_pv.yError()
                tofill['min_ip3d_pv_z_err'    ] = min_ip3d_pv.zError()



                # find PV max cosine of pointing angle
                pv_cos2d_idx = -1
                pv_cos3d_idx = -1
                cos2d_max = -np.inf
                cos3d_max = -np.inf
                
                for idx, ivtx in enumerate(event.vtx):
                    
                    # transverse plane
                    momentum2d  = np.array([cand.px(), cand.py(), 0.])
                    direction2d = np.array([cand.bs_tree.vtx.x() - ivtx.x(), 
                                            cand.bs_tree.vtx.y() - ivtx.y(), 
                                            0.])
                        
                    cos2d = np.dot(momentum2d, direction2d) / np.linalg.norm(momentum2d) / np.linalg.norm(direction2d)

                    # 3-dimensional
                    
                    momentum3d  = np.array([cand.px(), cand.py(), cand.pz()])
                    direction3d = np.array([cand.bs_tree.vtx.x() - ivtx.x(), 
                                            cand.bs_tree.vtx.y() - ivtx.y(), 
                                            cand.bs_tree.vtx.z() - ivtx.z()])

                    cos3d = np.dot(momentum3d, direction3d) / np.linalg.norm(momentum3d) / np.linalg.norm(direction3d)

                    if cos2d > cos2d_max:
                        pv_cos2d_idx = idx
                        cos2d_max = cos2d

                    if cos3d > cos3d_max:
                        pv_cos3d_idx = idx
                        cos3d_max = cos3d
                    
                max_cos2D_pv = event.vtx[pv_cos2d_idx]
                max_cos3D_pv = event.vtx[pv_cos3d_idx]

                tofill['max_cos2D_pv_x'        ] = max_cos2D_pv.x()
                tofill['max_cos2D_pv_y'        ] = max_cos2D_pv.y()
                tofill['max_cos2D_pv_z'        ] = max_cos2D_pv.z()

                tofill['max_cos2D_pv_ntracks'  ] = max_cos2D_pv.nTracks()
                tofill['max_cos2D_pv_tracksize'] = max_cos2D_pv.tracksSize()
                tofill['max_cos2D_pv_x_err'    ] = max_cos2D_pv.xError()
                tofill['max_cos2D_pv_y_err'    ] = max_cos2D_pv.yError()
                tofill['max_cos2D_pv_z_err'    ] = max_cos2D_pv.zError()

                tofill['max_cos3D_pv_x'        ] = max_cos3D_pv.x()
                tofill['max_cos3D_pv_y'        ] = max_cos3D_pv.y()
                tofill['max_cos3D_pv_z'        ] = max_cos3D_pv.z()

                tofill['max_cos3D_pv_ntracks'  ] = max_cos3D_pv.nTracks()
                tofill['max_cos3D_pv_tracksize'] = max_cos3D_pv.tracksSize()
                tofill['max_cos3D_pv_x_err'    ] = max_cos3D_pv.xError()
                tofill['max_cos3D_pv_y_err'    ] = max_cos3D_pv.yError()
                tofill['max_cos3D_pv_z_err'    ] = max_cos3D_pv.zError()


                                
                # match offline primary vertex to miniAOD refitted pv
                matched_refitted_vtx = None
                dist2 = np.inf
                for ivtx in event.refit_pv:
                    my_dist2 = np.power(min_ip3d_pv.x() - ivtx.x(), 2) + np.power(min_ip3d_pv.y() - ivtx.y(), 2) + np.power(min_ip3d_pv.z() - ivtx.z(), 2)
                    if my_dist2 < dist2:
                        dist2 = my_dist2
                        matched_refitted_vtx = ivtx
                        
                if matched_refitted_vtx is not None:
                    tofill['miniaod_min_ip3d_pv_x'] = matched_refitted_vtx.x()
                    tofill['miniaod_min_ip3d_pv_y'] = matched_refitted_vtx.y()
                    tofill['miniaod_min_ip3d_pv_z'] = matched_refitted_vtx.z()

                    tofill['miniaod_min_ip3d_pv_ntracks'  ] = matched_refitted_vtx.nTracks()
                    tofill['miniaod_min_ip3d_pv_tracksize'] = matched_refitted_vtx.tracksSize()
                    tofill['miniaod_min_ip3d_pv_x_err'    ] = matched_refitted_vtx.xError()
                    tofill['miniaod_min_ip3d_pv_y_err'    ] = matched_refitted_vtx.yError()
                    tofill['miniaod_min_ip3d_pv_z_err'    ] = matched_refitted_vtx.zError()

                    tofill['distance'] = np.sqrt(dist2)
                
                ##########################################################################
                ## REFIT PV WITH BS CONSTRAINT AND REMOVE SIGNAL TRACKS
                ##########################################################################
                
                unpacked_tracks = sorted([itk for itk in event.refit_trk], key = lambda x : x.pt(), reverse = True)
                pos_tracks = [itk for itk in unpacked_tracks if itk.charge()>0]
                neg_tracks = [itk for itk in unpacked_tracks if itk.charge()<0]
                                
                # need geometrical match, because miniAOD tracks are different from generalTracks, cannot match by pointer
                mu_track = bestMatch(reco_mu, pos_tracks)[0] if reco_mu.charge()>0 else bestMatch(reco_mu, neg_tracks)[0]
                pi_track = bestMatch(reco_pi, pos_tracks)[0] if reco_pi.charge()>0 else bestMatch(reco_pi, neg_tracks)[0]
                kp_track = bestMatch(reco_kp, pos_tracks)[0]
                km_track = bestMatch(reco_km, neg_tracks)[0]
                
                # remove signal tracks from vertices and refit.
                # need to use miniAOD refitted vertices, otherwise cannot know which track is which
                
                vertices_w_bs = []
                cleaned_vertices_w_bs = []
                
                signal_tracks_matched_to_any_vtx = False
                
                # DEBUGGING save all tracks used for primary vertexing
                all_vtx_trks = []
                
                for ivtx in event.refit_pv_w_bs:
                    # must be vertex refitted from unpacked tracks from miniAOD
                    vtx_tracks = [ivtx.trackRefAt(itk).get() for itk in range(ivtx.tracksSize())]
                    
                    # DEBUGGING
                    all_vtx_trks += vtx_tracks
                    
                    tofit.clear()
                    for itk in vtx_tracks:
                        tofit.push_back(itk)
                    
                    new_vtx = pvrefitter.fit(tofit, event.bs)
                    
                    vertices_w_bs.append(new_vtx)

                    # remove signal tracks from those that belong to the PV
                    vtx_tracks = [itk for itk in vtx_tracks if itk not in [mu_track, pi_track, kp_track, km_track]] 

#                     if (event.eventAuxiliary().luminosityBlock(),event.eventAuxiliary().event()) in to_check:
#                         if abs(first_b.vertex().z()-ivtx.z())<0.1:
#                             print('lumi, event', event.eventAuxiliary().luminosityBlock(), event.eventAuxiliary().event(), ivtx.tracksSize()==len(vtx_tracks))
#                             print('===> ivtx.z()', ivtx.z(), '  gen vtx z', first_b.vertex().z(), '  delta =', abs(ivtx.z()-first_b.vertex().z()) )            
#                             print('===> common tracks', ivtx.tracksSize()-len(vtx_tracks) )
#                             if event.eventAuxiliary().event()==24227: import ipdb ; ipdb.set_trace()
   
                    tofit.clear()
                    for itk in vtx_tracks:
                        tofit.push_back(itk)
                    

                    new_vtx = pvrefitter.fit(tofit, event.bs)
                    
                    cleaned_vertices_w_bs.append(new_vtx)
                                
                tofill['mu_track_used'] = np.int(mu_track in all_vtx_trks)
                tofill['kp_track_used'] = np.int(kp_track in all_vtx_trks)
                tofill['km_track_used'] = np.int(km_track in all_vtx_trks)
                tofill['pi_track_used'] = np.int(pi_track in all_vtx_trks)

                pv_idx = -1
                ip3d_min = np.inf
                for idx, ivtx in enumerate(cleaned_vertices_w_bs):
                    ip3d = compute_IP3D(ivtx, cand.bs_tree.vtx.position(), cand.p4().Vect())
                    if ip3d<ip3d_min:
                        pv_idx = idx
                        ip3d_min = ip3d
                    
                min_ip3d_pv_clean_w_bs = cleaned_vertices_w_bs[pv_idx]

                tofill['min_ip3d_pv_w_bs_no_sig_tk_x'] = min_ip3d_pv_clean_w_bs.x()
                tofill['min_ip3d_pv_w_bs_no_sig_tk_y'] = min_ip3d_pv_clean_w_bs.y()
                tofill['min_ip3d_pv_w_bs_no_sig_tk_z'] = min_ip3d_pv_clean_w_bs.z()

                tofill['bs_at_min_ip3d_pv_w_bs_no_sig_tk_x'] = event.bs.x(min_ip3d_pv_clean_w_bs.z())
                tofill['bs_at_min_ip3d_pv_w_bs_no_sig_tk_y'] = event.bs.y(min_ip3d_pv_clean_w_bs.z())

                pv_idx = -1
                ip3d_min = np.inf
                for idx, ivtx in enumerate(vertices_w_bs):
                    ip3d = compute_IP3D(ivtx, cand.bs_tree.vtx.position(), cand.p4().Vect())
                    if ip3d<ip3d_min:
                        pv_idx = idx
                        ip3d_min = ip3d
                    
                min_ip3d_pv_w_bs = vertices_w_bs[pv_idx]

                tofill['min_ip3d_pv_w_bs_x'] = min_ip3d_pv_w_bs.x()
                tofill['min_ip3d_pv_w_bs_y'] = min_ip3d_pv_w_bs.y()
                tofill['min_ip3d_pv_w_bs_z'] = min_ip3d_pv_w_bs.z()
                
                # best match PV, by minimum distance
                best_vtx = sorted(cleaned_vertices_w_bs, key = lambda ivtx : np.power(ivtx.x()-bs.vertex().x(), 2) + np.power(ivtx.y()-bs.vertex().y(), 2) + np.power(ivtx.z()-bs.vertex().z(), 2))[0]
                tofill['matched_pv_x'] = best_vtx.x()
                tofill['matched_pv_y'] = best_vtx.y()
                tofill['matched_pv_z'] = best_vtx.z()

                ##########################################
                ## RECOMPUTE STANDARD MUON PF ISOLATION ##
                ##########################################
    
                all_charged_particles_pf_iso_cands_04 = [itk for itk in event.trk if abs(itk.pdgId()) in [211, 321, 11, 13, 2212, 999211] and deltaR(reco_mu, itk)<0.4 and deltaR(reco_mu, itk)>1e-04]
                all_neutral_particles_pf_iso_cands_04 = [itk for itk in event.trk if abs(itk.pdgId()) in [22, 111, 130, 310, 2112]        and deltaR(reco_mu, itk)<0.4 and deltaR(reco_mu, itk)>1e-02 and itk.pt()>0.5]

                # clean PF candidates from signal tracks
                cleaned_charged_particles_pf_iso_cands_04 = [ipf for ipf in all_charged_particles_pf_iso_cands_04 if ipf not in [reco_pi, reco_kp, reco_km]]
    
                for ich_pf in all_charged_particles_pf_iso_cands_04:
                    
                    # missing association by vertex weight, but unfortunately 
                    # it does not work in miniAOD, boh!        
                    vertices = [ivtx for ivtx in event.vtx]
                    vertices.sort(key = lambda ivtx : abs(ivtx.z() - ich_pf.vz()))
                    ich_pf.matched_vtx = vertices[0]
    
                clean_ch_pf_iso_cands_04 = [itk for itk in cleaned_charged_particles_pf_iso_cands_04 if itk.matched_vtx == event.vtx[0] and abs(itk.pdgId()) in [211, 321, 2212, 99912]]
                clean_pu_pf_iso_cands_04 = [itk for itk in cleaned_charged_particles_pf_iso_cands_04 if itk.matched_vtx != event.vtx[0] and deltaR(itk, reco_mu)>0.01 and itk.pt()>0.5 ]
                ch_pf_iso_cands_04       = [itk for itk in all_charged_particles_pf_iso_cands_04     if itk.matched_vtx == event.vtx[0] and abs(itk.pdgId()) in [211, 321, 2212, 99912]]
                pu_pf_iso_cands_04       = [itk for itk in all_charged_particles_pf_iso_cands_04     if itk.matched_vtx != event.vtx[0] and deltaR(itk, reco_mu)>0.01 and itk.pt()>0.5 ]
                nh_pf_iso_cands_04       = [itk for itk in all_neutral_particles_pf_iso_cands_04     if abs(itk.pdgId()) in [111, 130, 310, 2112]]
                ph_pf_iso_cands_04       = [itk for itk in all_neutral_particles_pf_iso_cands_04     if abs(itk.pdgId()) in [22]                 ]
    
                clean_new_ch_pf_iso      = np.sum([ipf.pt() for ipf in clean_ch_pf_iso_cands_04])
                clean_new_pu_pf_iso      = np.sum([ipf.pt() for ipf in clean_pu_pf_iso_cands_04])
                new_ch_pf_iso            = np.sum([ipf.pt() for ipf in ch_pf_iso_cands_04])
                new_pu_pf_iso            = np.sum([ipf.pt() for ipf in pu_pf_iso_cands_04])
                new_nh_pf_iso            = np.sum([ipf.pt() for ipf in nh_pf_iso_cands_04])
                new_ph_pf_iso            = np.sum([ipf.pt() for ipf in ph_pf_iso_cands_04])
                
                clean_new_pf_iso = clean_new_ch_pf_iso + max(0., new_nh_pf_iso + new_ph_pf_iso - 0.5*clean_new_pu_pf_iso)
                new_pf_iso       = new_ch_pf_iso       + max(0., new_nh_pf_iso + new_ph_pf_iso - 0.5*new_pu_pf_iso      )
                                
                tofill['reco_mu_clean_new_iso04_sumChargedHadronPt'] = clean_new_ch_pf_iso
                tofill['reco_mu_clean_new_iso04_sumPUPt'           ] = clean_new_pu_pf_iso
                tofill['reco_mu_new_iso04_sumChargedHadronPt'      ] = new_ch_pf_iso
                tofill['reco_mu_new_iso04_sumPUPt'                 ] = new_pu_pf_iso
                tofill['reco_mu_new_iso04_sumNeutralHadronEt'      ] = new_nh_pf_iso
                tofill['reco_mu_new_iso04_sumPhotonEt'             ] = new_ph_pf_iso
                tofill['reco_mu_clean_new_iso04'                   ] = clean_new_pf_iso
                tofill['reco_mu_new_iso04'                         ] = new_pf_iso
                tofill['reco_mu_clean_new_rel_iso04'               ] = clean_new_pf_iso/reco_mu.pt()
                tofill['reco_mu_new_rel_iso04'                     ] = new_pf_iso      /reco_mu.pt()
    
                ###################################################
                ## COMPUTE MUON PF ISOLATION WITH UPDATED VERTEX ##
                ###################################################
    
                for ich_pf in all_charged_particles_pf_iso_cands_04:
                    
                    # missing association by vertex weight, but unfortunately 
                    # it does not work in miniAOD, boh!        
                    vertices = [ivtx for ivtx in cleaned_vertices_w_bs]
                    vertices.sort(key = lambda ivtx : abs(ivtx.z() - ich_pf.vz()))
                    ich_pf.matched_custom_vtx = vertices[0]
                        
                clean_custom_ch_pf_iso_cands_04 = [itk for itk in cleaned_charged_particles_pf_iso_cands_04 if itk.matched_custom_vtx == min_ip3d_pv_clean_w_bs and abs(itk.pdgId()) in [211, 321, 2212, 99912]]
                clean_custom_pu_pf_iso_cands_04 = [itk for itk in cleaned_charged_particles_pf_iso_cands_04 if itk.matched_custom_vtx != min_ip3d_pv_clean_w_bs and deltaR(itk, reco_mu)>0.01 and itk.pt()>0.5 ]
                custom_ch_pf_iso_cands_04       = [itk for itk in all_charged_particles_pf_iso_cands_04     if itk.matched_custom_vtx == min_ip3d_pv_clean_w_bs and abs(itk.pdgId()) in [211, 321, 2212, 99912]]
                custom_pu_pf_iso_cands_04       = [itk for itk in all_charged_particles_pf_iso_cands_04     if itk.matched_custom_vtx != min_ip3d_pv_clean_w_bs and deltaR(itk, reco_mu)>0.01 and itk.pt()>0.5 ]
                custom_nh_pf_iso_cands_04       = [itk for itk in all_neutral_particles_pf_iso_cands_04     if abs(itk.pdgId()) in [111, 130, 310, 2112]]
                custom_ph_pf_iso_cands_04       = [itk for itk in all_neutral_particles_pf_iso_cands_04     if abs(itk.pdgId()) in [22]                 ]

                clean_custom_ch_pf_iso = np.sum([ipf.pt() for ipf in clean_custom_ch_pf_iso_cands_04])
                clean_custom_pu_pf_iso = np.sum([ipf.pt() for ipf in clean_custom_pu_pf_iso_cands_04])
                custom_ch_pf_iso       = np.sum([ipf.pt() for ipf in custom_ch_pf_iso_cands_04])
                custom_pu_pf_iso       = np.sum([ipf.pt() for ipf in custom_pu_pf_iso_cands_04])
                custom_nh_pf_iso       = np.sum([ipf.pt() for ipf in custom_nh_pf_iso_cands_04])
                custom_ph_pf_iso       = np.sum([ipf.pt() for ipf in custom_ph_pf_iso_cands_04])
                
                clean_custom_pf_iso = clean_custom_ch_pf_iso + max(0., custom_nh_pf_iso + custom_ph_pf_iso - 0.5*clean_custom_pu_pf_iso)
                custom_pf_iso       = custom_ch_pf_iso       + max(0., custom_nh_pf_iso + custom_ph_pf_iso - 0.5*custom_pu_pf_iso)
                                
                tofill['reco_mu_clean_custom_iso04_sumChargedHadronPt'] = clean_custom_ch_pf_iso
                tofill['reco_mu_clean_custom_iso04_sumPUPt'           ] = clean_custom_pu_pf_iso
                tofill['reco_mu_custom_iso04_sumChargedHadronPt'      ] = custom_ch_pf_iso
                tofill['reco_mu_custom_iso04_sumPUPt'                 ] = custom_pu_pf_iso
                tofill['reco_mu_custom_iso04_sumNeutralHadronEt'      ] = custom_nh_pf_iso
                tofill['reco_mu_custom_iso04_sumPhotonEt'             ] = custom_ph_pf_iso
                tofill['reco_mu_clean_custom_iso04'                   ] = clean_custom_pf_iso
                tofill['reco_mu_custom_iso04'                         ] = custom_pf_iso
                tofill['reco_mu_clean_custom_rel_iso04'               ] = clean_custom_pf_iso/reco_mu.pt()
                tofill['reco_mu_custom_rel_iso04'                     ] = custom_pf_iso      /reco_mu.pt()
                
                ###################################################
                ###################################################
                ###################################################

                # offline vertices, no BS constraint, min IP3D
                lxyz_min_ip3d_pv_vtx = ROOT.VertexDistance3D().distance(min_ip3d_pv, cand.bs_tree.vtx)
                lxy_min_ip3d_pv_vtx  = ROOT.VertexDistanceXY().distance(min_ip3d_pv, cand.bs_tree.vtx)

                tofill['lxyz_min_ip3d_pv_vtx'    ] = lxyz_min_ip3d_pv_vtx.value()       
                tofill['lxyz_min_ip3d_pv_vtx_err'] = lxyz_min_ip3d_pv_vtx.error()       
                tofill['lxyz_min_ip3d_pv_vtx_sig'] = lxyz_min_ip3d_pv_vtx.significance()

                tofill['lxy_min_ip3d_pv_vtx'     ] = lxy_min_ip3d_pv_vtx.value()       
                tofill['lxy_min_ip3d_pv_vtx_err' ] = lxy_min_ip3d_pv_vtx.error()       
                tofill['lxy_min_ip3d_pv_vtx_sig' ] = lxy_min_ip3d_pv_vtx.significance()

                # refit with BS constraint, min IP3D
                lxyz_min_ip3d_pv_w_bs_vtx = ROOT.VertexDistance3D().distance(min_ip3d_pv_w_bs, cand.bs_tree.vtx)
                lxy_min_ip3d_pv_w_bs_vtx  = ROOT.VertexDistanceXY().distance(min_ip3d_pv_w_bs, cand.bs_tree.vtx)

                tofill['lxyz_min_ip3d_pv_w_bs_vtx'    ] = lxyz_min_ip3d_pv_w_bs_vtx.value()       
                tofill['lxyz_min_ip3d_pv_w_bs_vtx_err'] = lxyz_min_ip3d_pv_w_bs_vtx.error()       
                tofill['lxyz_min_ip3d_pv_w_bs_vtx_sig'] = lxyz_min_ip3d_pv_w_bs_vtx.significance()

                tofill['lxy_min_ip3d_pv_w_bs_vtx'     ] = lxy_min_ip3d_pv_w_bs_vtx.value()       
                tofill['lxy_min_ip3d_pv_w_bs_vtx_err' ] = lxy_min_ip3d_pv_w_bs_vtx.error()       
                tofill['lxy_min_ip3d_pv_w_bs_vtx_sig' ] = lxy_min_ip3d_pv_w_bs_vtx.significance()

                # refit with BS constraint, remove signal tracks, min IP3D
                lxyz_min_ip3d_pv_clean_w_bs_vtx = ROOT.VertexDistance3D().distance(min_ip3d_pv_clean_w_bs, cand.bs_tree.vtx)
                lxy_min_ip3d_pv_clean_w_bs_vtx  = ROOT.VertexDistanceXY().distance(min_ip3d_pv_clean_w_bs, cand.bs_tree.vtx)

                tofill['lxyz_min_ip3d_pv_clean_w_bs_vtx'    ] = lxyz_min_ip3d_pv_clean_w_bs_vtx.value()       
                tofill['lxyz_min_ip3d_pv_clean_w_bs_vtx_err'] = lxyz_min_ip3d_pv_clean_w_bs_vtx.error()       
                tofill['lxyz_min_ip3d_pv_clean_w_bs_vtx_sig'] = lxyz_min_ip3d_pv_clean_w_bs_vtx.significance()

                tofill['lxy_min_ip3d_pv_clean_w_bs_vtx'     ] = lxy_min_ip3d_pv_clean_w_bs_vtx.value()       
                tofill['lxy_min_ip3d_pv_clean_w_bs_vtx_err' ] = lxy_min_ip3d_pv_clean_w_bs_vtx.error()       
                tofill['lxy_min_ip3d_pv_clean_w_bs_vtx_sig' ] = lxy_min_ip3d_pv_clean_w_bs_vtx.significance()
                                                
                ##########################################################################
                ## COMPUTE MUON 3D IMPACT PARAMETER WRT SECONDARY VERTEX and PV to SV direction
                ##########################################################################
                                
                pv = min_ip3d_pv_clean_w_bs
                sv = cand.bs_tree.vtx
                
                direction = ROOT.GlobalVector(sv.x()-pv.x(), sv.y()-pv.y(), sv.z()-pv.z()).unit()
                
                mu_ip3d_sv     = ROOT.SignedIP3D(reco_mu.bestTrack(), sv, direction).get()
                mu_btv_ip3d_sv = ROOT.BTVSignedIP3D(reco_mu.bestTrack(), sv, direction).get()
                
                # why is the error (and hence significance) Nan?!
                #if (event.eventAuxiliary().luminosityBlock(),event.eventAuxiliary().event()) == (1523, 2199):
                #    import ipdb ; ipdb.set_trace()
                
                if mu_ip3d_sv.first:            
                    tofill['reco_mu_ip3d_sv'     ] = mu_ip3d_sv.second.value()      
                    tofill['reco_mu_ip3d_sv_err' ] = mu_ip3d_sv.second.error()      
                    tofill['reco_mu_ip3d_sv_sig' ] = np.nan_to_num(np.divide (mu_ip3d_sv.second.value() , mu_ip3d_sv.second.error()))   

                if mu_btv_ip3d_sv.first:            
                    tofill['reco_mu_btv_ip3d_sv'     ] = mu_btv_ip3d_sv.second.value()      
                    tofill['reco_mu_btv_ip3d_sv_err' ] = mu_btv_ip3d_sv.second.error()      
                    tofill['reco_mu_btv_ip3d_sv_sig' ] = np.nan_to_num(np.divide (mu_btv_ip3d_sv.second.value() , mu_btv_ip3d_sv.second.error()))     

                ##########################################################################
                ## COMPUTE MUON TRACK DISTANCE FROM DS DIRECTION (from 3-momentum)
                ##########################################################################
                                
                sv = cand.bs_tree.vtx
                tv = cand.ds_tree.vtx

                direction_ds    = ROOT.GlobalVector(cand.ds.p4().px(), cand.ds.p4().py(), cand.ds.p4().pz()).unit()
                direction_tv_sv = ROOT.GlobalVector(tv.x()-sv.x(), tv.y()-sv.y(), tv.z()-sv.z()).unit()
                
                mu_dl3d_sv_ds = ROOT.SignedDecayLength3D(reco_mu.bestTrack(), sv, direction_ds).get()
                mu_dl3d_tv_ds = ROOT.SignedDecayLength3D(reco_mu.bestTrack(), tv, direction_ds).get()

                mu_dl3d_sv_tv_sv = ROOT.SignedDecayLength3D(reco_mu.bestTrack(), sv, direction_tv_sv).get()
                mu_dl3d_tv_tv_sv = ROOT.SignedDecayLength3D(reco_mu.bestTrack(), tv, direction_tv_sv).get()
                
                #import ipdb ; ipdb.set_trace()
                
                # why is the error (and hence significance) Nan?!
                #if (event.eventAuxiliary().luminosityBlock(),event.eventAuxiliary().event()) == (1523, 2199):
                #    import ipdb ; ipdb.set_trace()
                
                if mu_dl3d_sv_ds.first:            
                    tofill['reco_mu_dl3d_ds_sv'    ] = mu_dl3d_sv_ds.second.value()      
                    tofill['reco_mu_dl3d_ds_sv_err'] = mu_dl3d_sv_ds.second.error()      
                    tofill['reco_mu_dl3d_ds_sv_sig'] = np.nan_to_num(np.divide (mu_dl3d_sv_ds.second.value() , mu_dl3d_sv_ds.second.error()))   

                if mu_dl3d_tv_ds.first:            
                    tofill['reco_mu_dl3d_ds_tv'    ] = mu_dl3d_tv_ds.second.value()      
                    tofill['reco_mu_dl3d_ds_tv_err'] = mu_dl3d_tv_ds.second.error()      
                    tofill['reco_mu_dl3d_ds_tv_sig'] = np.nan_to_num(np.divide (mu_dl3d_tv_ds.second.value() , mu_dl3d_tv_ds.second.error()))   


                if mu_dl3d_sv_tv_sv.first:            
                    tofill['reco_mu_dl3d_tv_sv_sv'    ] = mu_dl3d_sv_tv_sv.second.value()      
                    tofill['reco_mu_dl3d_tv_sv_sv_err'] = mu_dl3d_sv_tv_sv.second.error()      
                    tofill['reco_mu_dl3d_tv_sv_sv_sig'] = np.nan_to_num(np.divide (mu_dl3d_sv_tv_sv.second.value() , mu_dl3d_sv_tv_sv.second.error()))   

                if mu_dl3d_tv_tv_sv.first:            
                    tofill['reco_mu_dl3d_tv_sv_tv'    ] = mu_dl3d_tv_tv_sv.second.value()      
                    tofill['reco_mu_dl3d_tv_sv_tv_err'] = mu_dl3d_tv_tv_sv.second.error()      
                    tofill['reco_mu_dl3d_tv_sv_tv_sig'] = np.nan_to_num(np.divide (mu_dl3d_tv_tv_sv.second.value() , mu_dl3d_tv_tv_sv.second.error()))   

                ##########################################################################
                ## SELECT TRACKS WITH SMALL IMPACT PARAMETER WRT Ds (TERTIARY) VERTEX
                ## TO COMPUTE Ds ISOLATION
                ##########################################################################
                
                sv = cand.bs_tree.vtx
                tv = cand.ds_tree.vtx
                
                direction = ROOT.GlobalVector(tv.x()-sv.x(), tv.y()-sv.y(), tv.z()-sv.z()).unit()
                
                ds_isolation_tracks = []
                ds_pu_tracks = []
                
                for itrk in event.trk:
                    if itrk.charge()==0: continue
                    if itrk in [reco_kp, reco_km, reco_pi]: continue
                    if deltaR(reco_mu, itrk)<0.001: continue
                    if deltaR(cand.ds, itrk)>0.4: continue
                    if not itrk.hasTrackDetails(): continue
                    
                    itrk_ip3d_sv = ROOT.SignedIP3D(itrk.bestTrack(), tv, direction).get()
                    # 100 micron, average distance between reco and gen SV is less than 30 micron
                    if itrk_ip3d_sv.first and abs(itrk_ip3d_sv.second.value())<0.01:
                        ds_isolation_tracks.append(itrk) 
                    if itrk_ip3d_sv.first and abs(itrk_ip3d_sv.second.value())>=0.01:
                        if itrk.pt()>0.5: # as in normal isolation
                            ds_pu_tracks.append(itrk) 

                #import ipdb ; ipdb.set_trace()

                ds_isolation_neutrals = [ineu for ineu in event.trk if ineu.charge()==0 and deltaR(cand.ds, ineu)<0.4]
                ds_isolation_ph       = [ineu for ineu in ds_isolation_neutrals if ineu.pdgId()==22]
                ds_isolation_nh       = [ineu for ineu in ds_isolation_neutrals if ineu.pdgId()!=22]

                ds_ch_iso04 = np.sum([itrk.pt() for itrk in ds_isolation_tracks])     
                ds_nh_iso04 = np.sum([ineu.pt() for ineu in ds_isolation_nh])     
                ds_ph_iso04 = np.sum([ineu.pt() for ineu in ds_isolation_ph])     
                ds_pu_iso04 = np.sum([itrk.pt() for itrk in ds_pu_tracks])     
                ds_iso04    = ds_ch_iso04 + max(0., ds_nh_iso04 + ds_ph_iso04 - 0.5*ds_pu_iso04)

                tofill['ds_ch_iso04' ] = ds_ch_iso04
                tofill['ds_nh_iso04' ] = ds_nh_iso04
                tofill['ds_ph_iso04' ] = ds_ph_iso04
                tofill['ds_pu_iso04' ] = ds_pu_iso04
                tofill['ds_iso04'    ] = ds_iso04   
                tofill['ds_iso04_rel'] = ds_iso04 / cand.ds.pt()
                
#                 if event.eventAuxiliary().luminosityBlock() == 4309 and \
#                    event.eventAuxiliary().event()           == 1195641216:
#                     import ipdb ; ipdb.set_trace()

                ##########################################################################
                ## SELECT TRACKS WITH SMALL IMPACT PARAMETER WRT SECONDARY VERTEX
                ## TO COMPUTE AN ALTERNATIVE MUON ISOLATION
                ##########################################################################
                
                pv = min_ip3d_pv_clean_w_bs
                sv = cand.bs_tree.vtx
                
                direction = ROOT.GlobalVector(sv.x()-pv.x(), sv.y()-pv.y(), sv.z()-pv.z()).unit()
                
                mu_isolation_tracks = []
                mu_pu_tracks = []
                
                for itrk in event.trk:
                    if itrk.charge()==0: continue
                    if itrk in [reco_kp, reco_km, reco_pi]: continue
                    if deltaR(reco_mu, itrk)<0.001: continue
                    if deltaR(reco_mu, itrk)>0.4: continue
                    if not itrk.hasTrackDetails(): continue
                    
                    itrk_ip3d_sv = ROOT.SignedIP3D(itrk.bestTrack(), sv, direction).get()
                    # 100 micron, average distance between reco and gen SV is less than 30 micron
                    if itrk_ip3d_sv.first and abs(itrk_ip3d_sv.second.value())<0.01:
                        mu_isolation_tracks.append(itrk) 
                    if itrk_ip3d_sv.first and abs(itrk_ip3d_sv.second.value())>=0.01:
                        if itrk.pt()>0.5: # as in normal isolation
                            mu_pu_tracks.append(itrk) 

                #import ipdb ; ipdb.set_trace()

                mu_isolation_neutrals = [ineu for ineu in event.trk if ineu.charge()==0 and deltaR(reco_mu, ineu)<0.4]
                mu_isolation_ph       = [ineu for ineu in mu_isolation_neutrals if ineu.pdgId()==22]
                mu_isolation_nh       = [ineu for ineu in mu_isolation_neutrals if ineu.pdgId()!=22]

                mu_ch_iso04 = np.sum([itrk.pt() for itrk in mu_isolation_tracks])     
                mu_nh_iso04 = np.sum([ineu.pt() for ineu in mu_isolation_nh])     
                mu_ph_iso04 = np.sum([ineu.pt() for ineu in mu_isolation_ph])     
                mu_pu_iso04 = np.sum([itrk.pt() for itrk in mu_pu_tracks])     
                mu_iso04    = mu_ch_iso04 + max(0., mu_nh_iso04 + mu_ph_iso04 - 0.5*mu_pu_iso04)

                tofill['mu_ch_sv_iso04' ] = mu_ch_iso04
                tofill['mu_nh_sv_iso04' ] = mu_nh_iso04
                tofill['mu_ph_sv_iso04' ] = mu_ph_iso04
                tofill['mu_pu_sv_iso04' ] = mu_pu_iso04
                tofill['mu_sv_iso04'    ] = mu_iso04   
                tofill['mu_sv_iso04_rel'] = mu_iso04 / reco_mu.pt()


                # compute 4-body system acoplanarity
                # https://chatgpt.com/share/679a9997-8124-8005-9223-09e35efcee6a
                tofill['acoplanarity'] = abs(deltaPhi(reco_mu.phi(), cand.phi())) + \
                                         abs(deltaPhi(reco_kp.phi(), cand.phi())) + \
                                         abs(deltaPhi(reco_km.phi(), cand.phi())) + \
                                         abs(deltaPhi(reco_pi.phi(), cand.phi()))


                # Create the transverse momentum tensor (in the transverse plane)
                # For each particle, we only use Px and Py components for the transverse momentum
                T = np.zeros((3, 3))  # 2x2 matrix, since we are only considering the transverse plane
                
                for p in [reco_mu, reco_kp, reco_km, reco_pi]:
                    Px = p.px()
                    Py = p.py()
                    Pz = p.pz()
                    E  = p.energy()
                               
                    # Fill the 3x3 momentum tensor (this is the normalized sum of the tensor components)
                    T[0, 0] += (Px * Px) / E
                    T[0, 1] += (Px * Py) / E
                    T[0, 2] += (Px * Pz) / E
                    T[1, 0] += (Py * Px) / E
                    T[1, 1] += (Py * Py) / E
                    T[1, 2] += (Py * Pz) / E
                    T[2, 0] += (Pz * Px) / E
                    T[2, 1] += (Pz * Py) / E
                    T[2, 2] += (Pz * Pz) / E
                
                #https://rivet.hepforge.org/code/1.2.1/a00160.html
                
                # Calculate the eigenvalues of the momentum tensor
                eigenvalues = np.linalg.eigvals(T)
                
                # Sort the eigenvalues (largest to smallest)
                eigenvalues = np.flip(np.sort(eigenvalues))
                
                # normalize eigenvalues
                eigenvalues = eigenvalues/np.sum(eigenvalues)
                
                # Compute the sphericity from the eigenvalues
                #sphericity = 1.5 * (1 - (eigenvalues[1] + eigenvalues[0]) / eigenvalues[2])
                sphericity = 3./2. * (eigenvalues[1] + eigenvalues[2])
                aplanarity = 3./2. * (eigenvalues[2])
                planarity  = 2./3. * (sphericity - 2*aplanarity)

                tofill['sphericity'] = sphericity
                tofill['aplanarity'] = aplanarity
                tofill['planarity' ] = planarity   
                
                #import ipdb ; ipdb.set_trace()          
                               
        row_list.append(tofill)

    
fout = uproot.recreate(destination + '/' + fileName + '.root')
ntuple = pd.DataFrame(row_list, columns=branches)
fout['tree'] = ntuple
          
          
          