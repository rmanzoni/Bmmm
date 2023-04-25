'''
run this with conda activate hammer3p8 
**BEFORE** initialising cmsenv
'''

import os
import glob
import ROOT
# from Bmmm.Analysis.MuMuBranches import muon_branches
import sys ; sys.path.append('../python')
from MuMuBranches import muon_branches

ROOT.EnableImplicitMT()

#tree_data = ROOT.TChain('tree')
#files_data = [ifile for ifile in glob.glob('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/HbToMuMuX_2018UL_03Apr2023_v1/*root') if (os.path.getsize(ifile) >> 10) > 500]
#rdf_data = ROOT.RDataFrame('tree', files_data)
#
## map(tree_data.Add, files_data[:1])
## rdf_data = ROOT.RDataFrame(tree_data)
## 
## rdf_data = ROOT.RDataFrame('tree', '/pnfs/psi.ch/cms/trivcat/store/user/manzoni/HbToMuMuX_2018UL_03Apr2023_v1/hbmmx_chunk435.root')
#
##print('initial events TChain', tree_data.GetEntries())
##print('initial events RDF'   , rdf_data.Count().GetValue())
#
## che merda
#def snapshot_rdf(tag, probe, rdf):
#    # che palle
#    skimmed_rdf = ROOT.RDataFrame(rdf)
#    skimmed_rdf = skimmed_rdf.Filter('HLT_Mu8>0.5 && %s_HLT_Mu8_tag>0.5 && mass<10' %tag)
#    
#    print (tag, probe)
#    print ([icol for icol in rdf_data.GetColumnNames() if icol=='tag_mu_pt'])
#    print ([icol for icol in skimmed_rdf.GetColumnNames() if icol=='tag_mu_pt'])
#    
#    # che merda
#    for ibranch in muon_branches:
#        print ('\t', ibranch, [icol for icol in skimmed_rdf.GetColumnNames() if icol=='tag_mu_pt'])
#        skimmed_rdf = skimmed_rdf.Define('tag_mu_%s'   %ibranch, '%s_%s' %(tag  , ibranch))
#        skimmed_rdf = skimmed_rdf.Define('probe_mu_%s' %ibranch, '%s_%s' %(probe, ibranch))
#        
#    skimmed_rdf.Snapshot('tree', '/pnfs/psi.ch/cms/trivcat/store/user/manzoni/rjpsi_sf/hb_tag_%s_hlt_mu8.root' %tag)
#    del skimmed_rdf
#
#snapshot_rdf('mu2', 'mu1', rdf_data)
#snapshot_rdf('mu1', 'mu2', rdf_data)
#






tree_data = ROOT.TChain('tree')
files_data = [ifile for ifile in glob.glob('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/DoubleMuon_Run2018*-UL2018_MiniAODv2_GT36-v1_03Apr2023_v1/*root') if (os.path.getsize(ifile) >> 10) > 500]
rdf_data = ROOT.RDataFrame('tree', files_data)

# map(tree_data.Add, files_data[:1])
# rdf_data = ROOT.RDataFrame(tree_data)
# 
# rdf_data = ROOT.RDataFrame('tree', '/pnfs/psi.ch/cms/trivcat/store/user/manzoni/HbToMuMuX_2018UL_03Apr2023_v1/hbmmx_chunk435.root')

#print('initial events TChain', tree_data.GetEntries())
#print('initial events RDF'   , rdf_data.Count().GetValue())

# che merda
def snapshot_rdf(tag, probe, rdf):
    # che palle
    skimmed_rdf = ROOT.RDataFrame(rdf)
    skimmed_rdf = skimmed_rdf.Filter('HLT_Mu8>0.5 && %s_HLT_Mu8_tag>0.5 && mass<10' %tag)
    
    print (tag, probe)
    print ([icol for icol in rdf_data.GetColumnNames() if icol=='tag_mu_pt'])
    print ([icol for icol in skimmed_rdf.GetColumnNames() if icol=='tag_mu_pt'])
    
    # che merda
    for ibranch in muon_branches:
        print ('\t', ibranch, [icol for icol in skimmed_rdf.GetColumnNames() if icol=='tag_mu_pt'])
        skimmed_rdf = skimmed_rdf.Define('tag_mu_%s'   %ibranch, '%s_%s' %(tag  , ibranch))
        skimmed_rdf = skimmed_rdf.Define('probe_mu_%s' %ibranch, '%s_%s' %(probe, ibranch))
        
    skimmed_rdf.Snapshot('tree', '/pnfs/psi.ch/cms/trivcat/store/user/manzoni/rjpsi_sf/doublemu_ul2018_tag_%s_hlt_mu8.root' %tag)
    del skimmed_rdf

snapshot_rdf('mu2', 'mu1', rdf_data)
snapshot_rdf('mu1', 'mu2', rdf_data)

