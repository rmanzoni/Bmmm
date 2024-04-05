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
--inputFiles="7d3aac57-14ea-40d8-b898-0f14ee0d45c3.root" \
--filename=lifetime_check \
--mc \
--maxevents=1000


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
from Bmmm.Analysis.B4MuBranches import branches, paths, muon_branches, cand_branches, event_branches, bs_branches, jpsi_branches, phi_branches
from Bmmm.Analysis.B4MuCandidate import B4MuCandidate as Candidate
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
        for to in [to for to in event.tobjs if to.pt()>cuts['4mu']['to_pt'] and abs(to.eta())<cuts['4mu']['to_eta']]:
            #to.unpackFilterLabels(event.object(), event.trg_res)
            to.unpackNamesAndLabels(event.object(), event.trg_res)
            for k, v in paths.items():
                if triggers[k]!=1: continue
                for ilabel in v: 
                    if to.hasFilterLabel(ilabel) and to not in good_tobjs[k]:
                        good_tobjs[k].append(to)
    
                
        muons = [mu for mu in event.muons if mu.pt()>cuts['4mu']['mu_pt'] and \
                                             abs(mu.eta())<cuts['4mu']['mu_eta'] and \
                                             cuts['4mu']['mu_id'](mu) and\
                                             abs(mu.bestTrack().dxy())<cuts['4mu']['mu_dxy']]
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
            
            p4 = iquadruplet[0].p4() + iquadruplet[1].p4() + iquadruplet[2].p4() + iquadruplet[3].p4() 
            
            # filter by mass, first
            if p4.mass()<cuts['4mu']['min_mass'] or p4.mass()>cuts['4mu']['max_mass']:
                continue
            cutflow['\tpass mass cut'] += 1
    
            # 4 muon candidate
            cand = Candidate(iquadruplet, event.vtx, event.bs)
            
            # 4 muons somewhat close in dz
            if max([abs( imu.bestTrack().dz(cand.pv.position()) - jmu.bestTrack().dz(cand.pv.position()) ) for imu, jmu in combinations(cand.muons, 2)])>cuts['4mu']['max_dz']: 
                continue
            cutflow['\tpass mutual dz'] += 1
                               
            # trigger matching, at least one muon matched. 
            # Later one can save the best matched trigger object to each muon, but let me keep it simple for now
            # FIXME! trigger name is hardcoded!
            cand.trig_match = False
            if sum([deltaR(ipair[0], ipair[1])<cuts['4mu']['hlt_dr'] for ipair in product(iquadruplet, good_tobjs[cuts['4mu']['hlt']])])<2:
                if options.savenontrig:
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
                   
        if options.mc:
            # B4mu
            gen_muons = [ip for ip in event.genpr if abs(ip.pdgId())==13 and abs(ip.mother(0).pdgId()) in [531, 511]]
            # BJpsiPhi
            if len(gen_muons)<4:
                gen_muons = [ip for ip in event.genpr if abs(ip.pdgId())==13 and (abs(ip.mother(0).pdgId())==443 or abs(ip.mother(0).pdgId())==333) and abs(ip.mother(0).mother(0).pdgId()) in [531]]
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
            if options.mc:
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
    
            mum = lambda x : x.genp.mother(0)
            nana = lambda x : x.genp.mother(0).mother(0)
    
            mothers = [mum(imu) for imu in final_cand.muons]
            grandmothers = [nana(imu) for imu in final_cand.muons]
            
            ## B4mu
            if len(set(mothers))==1 and abs(mothers[0].pdgId()) in [511, 531]:           
                the_b = mothers[0]
    
                for branch, getter in bs_branches.items():
                    tofill[branch] = getter(the_b)    
            
            ## Bs Jpsi Phi
            elif len(set(mothers))==2 and len(set(grandmothers))==1 and abs(grandmothers[0].pdgId())==531:       
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
    args = parser.parse_args()
    
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

