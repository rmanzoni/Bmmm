'''
Example:

ipython -i -- inspector_b2m2k_analysis.py \
--inputFiles="c4b91801-de87-48bf-b7a6-1d510088135f.root" \
--filename=jpsi_phi_2mu2k \
--mc \
--maxevents=100

FIXME!
make it asynchronous, it _should_ help with slow xrootd i/o
https://www.youtube.com/watch?v=dEZKySL3M9c

https://lukasa.co.uk/2016/07/The_Function_Colour_Myth/

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
import asyncio
import aiostream
from time import time
from datetime import datetime, timedelta
from array import array
from glob import glob
from collections import OrderedDict, defaultdict, namedtuple
from DataFormats.FWLite import Events, Handle
from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi, bestMatch
from itertools import product, combinations
from Bmmm.Analysis.B2Mu2TkBranches import branches, paths, muon_branches, cand_branches, event_branches, bs_branches, jpsi_branches, phi_branches, track_branches
from Bmmm.Analysis.B2Mu2TkCandidate import B2Mu2TkCandidate as Candidate
from Bmmm.Analysis.utils import drop_hlt_version, cutflow, p4_with_mass, masses, compute_mass, AsyncIter
from Bmmm.Analysis.B4Mucuts import cuts
from Bmmm.Analysis.Handles import handles, handles_mc


######################################################################################
#####      LOOPER
######################################################################################
async def looper(events, options, handles, handles_mc, row_list, start):
    
    async for i, event in aiostream.stream.enumerate(AsyncIter(events)):

        if (i+1) > options.maxevents:
            break
                
        if i%options.logfreq == 0:
            percentage = float(i) / options.maxevents * 100.
            speed = float(i) / (time() - start)
            eta = datetime.now() + timedelta(seconds=(options.maxevents-i) / max(0.1, speed))
            print('\t===> processing %d / %d event \t completed %.1f%s \t %.1f ev/s \t ETA %s s' %(i, options.maxevents, percentage, '%', speed, eta.strftime('%Y-%m-%d %H:%M:%S')))
    
        # reset trees
        tofill = dict(zip(branches, [np.nan]*len(branches)))
        
        # access the handles
        for k, v in handles.items():
            event.getByLabel(v[0], v[1])
            setattr(event, k, v[1].product())
        
        event.mc = False
        
        if options.mc:
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
                if not iname.startswith(ipath): continue
                idx = len(trg_names)               
                if drop_hlt_version(iname)==ipath:
                    idx = trg_names.triggerIndex(iname)
                    tofill[ipath        ] = ( idx < len(trg_names)) * (event.trg_res.accept(idx))
                    tofill[ipath + '_ps'] = event.trg_ps.getPrescaleForIndex(idx)
        
        triggers = {key:tofill[key] for key in paths.keys()}
    
        hlt_passed = any([vv for vv in triggers.values()])
        # skip events if no trigger fired, unless savenotrig option is specified
        if not(options.savenontrig or hlt_passed):
            continue            
        
        cutflow['pass HLT'] += 1
    
        # trigger matching
        good_tobjs = {key:[] for key in paths.keys()}    
        for to in [to for to in event.tobjs if to.pt()>cuts['2mu2k']['to_pt'] and abs(to.eta())<cuts['2mu2k']['to_eta']]:
            #to.unpackFilterLabels(event.object(), event.trg_res)
            to.unpackNamesAndLabels(event.object(), event.trg_res)
            for k, v in paths.items():
                if triggers[k]!=1: continue
                for ilabel in v: 
                    if to.hasFilterLabel(ilabel) and to not in good_tobjs[k]:
                        good_tobjs[k].append(to)
    
                
        muons = [mu for mu in event.muons if mu.pt()>cuts['2mu2k']['mu_pt'] and \
                                             abs(mu.eta())<cuts['2mu2k']['mu_eta'] and \
                                             cuts['2mu2k']['mu_id'](mu)]
        muons.sort(key = lambda x : x.pt(), reverse = True)
        event.nmuons = len(muons)
        
        if len(muons)<2:
            continue
    
        cutflow['at least two muons'] += 1
        
        track_selection = lambda tk : tk.pt()>cuts['2mu2k']['tk_pt'] and cuts['2mu2k']['tk_id'](tk) and abs(tk.eta())<cuts['2mu2k']['tk_eta'] and abs(tk.dxy())<cuts['2mu2k']['tk_dxy'] and abs(tk.dz())<cuts['2mu2k']['tk_dz'] and tk.charge()!=0 and abs(tk.pdgId()) not in [22, 11, 13, 130]
        
        tks  = [*filter(track_selection, event.trk )]
        ltks = [*filter(track_selection, event.ltrk)]
            
        tks = tks + ltks
        tks.sort(key = lambda x : x.pt(), reverse = True)
        event.ntracks = len(tks)
        
        if len(tks)<2:
            continue
    
        cutflow['at least two tracks'] += 1
    
        ######################################################################################
        #####      BUILD AND SELECT 2MU2K CANDIDATES
        ######################################################################################
        cands = []
    
        for imupair in combinations(muons, 2): 
        
            cutflow['\tcandidates after HLT and 2mu2tk'] += 1
    
            dimuon = imupair[0].p4() + imupair[1].p4()
    
            # impose charge, reduce combinatorics
            if imupair[0].charge() * imupair[1].charge() >= 0:
                continue
            cutflow['\topposite sign dimuon'] += 1
                    
            if dimuon.mass()<cuts['2mu2k']['min_dimuon_mass'] or \
               dimuon.mass()>cuts['2mu2k']['max_dimuon_mass']:
                continue
            cutflow['\tpass dimuon mass'] += 1
    
            # trigger matching, at least one muon matched. 
            # Later one can save the best matched trigger object to each muon, but let me keep it simple for now
            trig_match = sum([deltaR(ipair[0], ipair[1])<cuts['2mu2k']['hlt_dr'] for ipair in product(imupair, good_tobjs[cuts['2mu2k']['hlt']])])>=2
            if not trig_match:
                if options.savenontrig:
                    pass
                else:
                    continue
            cutflow['\tpass trigger match'] += 1
    
            # loosely preselect vertex to avoid too much track combinatorics 
            temp_pv = sorted( [vtx for vtx in event.vtx], key = lambda vtx : abs( imupair[0].bestTrack().dz(vtx.position() ) ) )[0]
            
            # select tracks near to the muons in deltaR and deltaZ
            selected_tks = [tk for tk in tks if deltaR(tk, dimuon)<cuts['2mu2k']['max_dr_k_mm'] and abs(tk.bestTrack().dz(temp_pv.position()))<cuts['2mu2k']['max_dz_presel']]
    
            for itkpair in combinations(selected_tks, 2): 
                
                # impose charge, reduce combinatorics
                if itkpair[0].charge() * itkpair[1].charge() >= 0:
                    continue
                cutflow['\t\topposite sign ditrack'] += 1
                
                dikaon = itkpair[0].p4() + itkpair[1].p4()            
                dikaon_mass = compute_mass(itkpair[0].p(), itkpair[1].p(), masses['k'], masses['k'], itkpair[0].p4().Vect().Dot(itkpair[1].p4().Vect()))
                
                # dikaon mass
                if dikaon_mass<cuts['2mu2k']['min_dikaon_mass'] or \
                   dikaon_mass>cuts['2mu2k']['max_dikaon_mass']:
                    continue
                cutflow['\t\tpass dikaon mass'] += 1
    
                # filter by mass, first
                mmkk_mass = compute_mass(dimuon.P(), dikaon.P(), dimuon.mass(), dikaon_mass, dimuon.Vect().Dot(dikaon.Vect()))
    
                if mmkk_mass<cuts['2mu2k']['min_mass'] or mmkk_mass>cuts['2mu2k']['max_mass']:
                    continue
                cutflow['\t\tpass mass cut'] += 1
    
                # clean trk-mu and trk-trk
                if (itkpair[0].charge()==imupair[0].charge() and deltaR(itkpair[0], imupair[0]) < cuts['2mu2k']['dr_cleaning'] and abs(1. - itkpair[0].pt()/imupair[0].pt()) < cuts['2mu2k']['dpt_cleaning']) or \
                   (itkpair[1].charge()==imupair[0].charge() and deltaR(itkpair[1], imupair[0]) < cuts['2mu2k']['dr_cleaning'] and abs(1. - itkpair[1].pt()/imupair[0].pt()) < cuts['2mu2k']['dpt_cleaning']) or \
                   (itkpair[0].charge()==imupair[1].charge() and deltaR(itkpair[0], imupair[1]) < cuts['2mu2k']['dr_cleaning'] and abs(1. - itkpair[0].pt()/imupair[1].pt()) < cuts['2mu2k']['dpt_cleaning']) or \
                   (itkpair[1].charge()==imupair[1].charge() and deltaR(itkpair[1], imupair[1]) < cuts['2mu2k']['dr_cleaning'] and abs(1. - itkpair[1].pt()/imupair[1].pt()) < cuts['2mu2k']['dpt_cleaning']) or \
                   (deltaR(itkpair[0], itkpair[1]) < cuts['2mu2k']['dr_cleaning'] and abs(1. - itkpair[0].pt()/itkpair[1].pt()) < cuts['2mu2k']['dpt_cleaning']):
                    continue
                cutflow['\t\tpass anti overlap'] += 1
    
                # 2m2k candidate
                cand = Candidate(imupair, {itkpair[0]:masses['k'], itkpair[1]:masses['k']}, event.vtx, event.bs) 
    
                # valid vertex
                if not cand.good_vtx:
                    continue
                cutflow['\t\tpass secondary vertex'] += 1
    
                # 4 muons somewhat close in dz, max distance 1 cm
                if max([abs( idau.bestTrack().dz(cand.pv.position()) - jdau.bestTrack().dz(cand.pv.position()) ) for idau, jdau in combinations(cand.muons + cand.tracks, 2)])>cuts['2mu2k']['max_dz']: 
                    continue
                cutflow['\t\tpass mutual dz'] += 1
                                         
                # save trig match info
                cand.trig_match = trig_match 
                                                 
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
                   
        if options.mc:
            gen_muons = [ip for ip in event.genpr if abs(ip.pdgId())==13  and abs(ip.mother(0).pdgId())==443 and abs(ip.mother(0).mother(0).pdgId())==531]
            gen_kaons = [ip for ip in event.genpr if abs(ip.pdgId())==321 and abs(ip.mother(0).pdgId())==333 and abs(ip.mother(0).mother(0).pdgId())==531]
            #bss= [ip for ip in event.genpr if abs(ip.pdgId())==531 and abs(abs(ip.mother().pdgId())!=531)]
            #print('\n')
            #for jj, ibs in enumerate(bss):
            #    print('%d Bs PDG ID %d' %(jj, ibs.pdgId()))
    
        for idx in range(1, 3):
            imu = getattr(final_cand, 'mu%d' %idx)
            imu.pv = final_cand.pv
            imu.bs = final_cand.bs
            imu.iso03 = imu.pfIsolationR03()
            imu.iso04 = imu.pfIsolationR04()
            
            # jet matching
            jet, dr2 = bestMatch(imu, event.jets)        
            if dr2<0.2**2: imu.jet = jet        
            # gen matching
            if options.mc:
                genp, dr2 = bestMatch(imu, [ip for ip in gen_muons if ip.charge()==imu.charge()])
                if dr2<0.02**2: imu.genp = genp
            
            for branch, getter in muon_branches.items():
                tofill['mu%d_%s' %(idx, branch)] = getter(imu) 
    
        for idx in range(1, 3):
            itk = getattr(final_cand, 'tk%d' %idx)
            itk.pv = final_cand.pv
            itk.bs = final_cand.bs
            
            # jet matching
            jet, dr2 = bestMatch(itk, event.jets)        
            if dr2<0.2**2: itk.jet = jet        
            # gen matching
            if options.mc:
                genp, dr2 = bestMatch(itk, [ip for ip in gen_kaons if ip.charge()==itk.charge()])
                if dr2<0.02**2: itk.genp = genp
            
            for branch, getter in track_branches.items():
                tofill['tk%d_%s' %(idx, branch)] = getter(itk) 
    
        for branch, getter in cand_branches.items():
            tofill[branch] = getter(final_cand)    
    
        if getattr(final_cand.mu1, 'genp', False) and \
           getattr(final_cand.mu2, 'genp', False) and \
           getattr(final_cand.tk1, 'genp', False) and \
           getattr(final_cand.tk2, 'genp', False):
                    
            mum = lambda x : x.genp.mother(0)
            nana = lambda x : x.genp.mother(0).mother(0)
    
            mothers = [mum(imu) for imu in final_cand.muons] + [mum(itk) for itk in final_cand.tracks] 
            grandmothers = [nana(imu) for imu in final_cand.muons] + [nana(itk) for itk in final_cand.tracks] 
            
            ## Bs Jpsi Phi
            if len(set(mothers))==2 and len(set(grandmothers))==1 and grandmothers[0].pdgId() in [531]:       
                the_b = grandmothers[0]
    
                for branch, getter in bs_branches.items():
                    tofill[branch] = getter(the_b)
                    
                the_jpsi = [ip for ip in set(mothers) if abs(ip.pdgId())==443][0]
                the_phi = [ip for ip in set(mothers) if abs(ip.pdgId())==333][0] 
    
                for branch, getter in jpsi_branches.items():
                    tofill[branch] = getter(the_jpsi)
    
                for branch, getter in phi_branches.items():
                    tofill[branch] = getter(the_phi)
        
        # append selected event
        row_list.append(tofill)
 
    # return number of processed events and cutflow
    return i, cutflow
    
    
######################################################################################
#####      MAIN
######################################################################################
async def main():

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
    
    options = namedtuple('options', parser.parse_args().__dict__.keys())(*parser.parse_args().__dict__.values())
    
    if ('txt' in options.inputFiles):
        with open(options.inputFiles) as f:
            files = f.read().splitlines()
            files = ['root://cms-xrd-global.cern.ch//' + file for file in files]
    elif ',' in options.inputFiles or 'cms-xrd-global' in options.inputFiles or 't3dcachedb' in options.inputFiles:
        files = options.inputFiles.split(',')
    else:
        files = glob(options.inputFiles)
    
    if options.maxfiles>0:
        files = files[:options.maxfiles]
    
    print("files:", files)
    
    events = Events(files)
    
    options = options._replace(maxevents = options.maxevents if options.maxevents>=0 else events.size()) # total number of events in the files
    
    fout = uproot.recreate(options.destination + '/' + options.filename + '.root')
    
    row_list = []
    
    # start the stopwatch
    start = time()
    mytimestamp = datetime.now().strftime('%Y-%m-%d__%Hh%Mm%Ss')
    print('#### STARTING NOW', mytimestamp)
    
    ##########################################################################################
    #####      PROCESS EVENTS
    ##########################################################################################    
    results = await asyncio.gather(looper(events, options, handles, handles_mc, row_list, start))
    
    n_proc_events, cutflow = results[0] # don't know why it needs to be this hard
    
    ##########################################################################################
    #####      WRITE TO DISK
    ##########################################################################################
    ntuple = pd.DataFrame(row_list, columns=branches)
    print('\nnumber of selected events', len(ntuple))
    fout['tree'] = ntuple
    print('\nntuple saved, processed all desired events?', (n_proc_events+1==options.maxevents), 'processed', n_proc_events+1, 'maxevents', options.maxevents)
    
    ##########################################################################################
    #####      SAVE LOGGER 
    ##########################################################################################
    
    logger_name = options.logger if len(options.logger)>0 else 'logger_'+mytimestamp
    
    with open('%s.txt'%logger_name, 'w') as logger_file:
        for k, v in cutflow.items():
            print(k, v, file=logger_file)
    
    finish = time()
    print('done in %.1f hours' %( (finish-start)/3600. ))
    

######################################################################################
#####      MAIN
######################################################################################
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()

