import os
import glob
import ROOT
import numpy as np
from itertools import product
from collections import OrderedDict
from cmsstyle import CMS_lumi
from officialStyle import officialStyle

newdir = 'trigger_tnp'
if not os.path.exists(newdir):
    os.makedirs(newdir)

officialStyle(ROOT.gStyle, ROOT.TGaxis)

ROOT.TH1.SetDefaultSumw2(True)
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

file_data = ROOT.TFile.Open('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/rjpsi_sf/doublemu_ul2018_hlt_mu8.root', 'read')
file_data.cd()
tree_data = file_data.Get('tree')

file_mc = ROOT.TFile.Open('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/rjpsi_sf/hb_hlt_mu8.root', 'read')
file_mc.cd()
tree_mc = file_mc.Get('tree')

tag_trigger = 'HLT_Mu8'
#probe_trigger = 'HLT_Mu8'
probe_trigger = 'HLT_DoubleMu4_3_Jpsi'
#probe_trigger = 'HLT_Dimuon0_Jpsi_NoVertexing'
#probe_trigger = 'HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R'

# replace tagmu and probemu
selection = ' && '.join([
    'tag_mu_pt>8.'             ,
    'probe_mu_pt>4.'           ,
    #'abs(mu1_eta)<2.4'         ,
    #'abs(mu2_eta)<2.4'         ,
    'abs(mu1_eta)<1.4'         , # match most restrictive L1 requirement
    'abs(mu2_eta)<1.4'         , # match most restrictive L1 requirement
    'tagtrigger>0.5'           ,
    #'charge==0'                ,
    'abs(mass-3.1)<0.12'       ,
    'tag_mu_id_medium>0.5 '    ,
    'probe_mu_id_soft_mva>0.5' ,
    'cos2d>0.91'               , # slightly tighter than at online
    'vtx_prob>0.12'            , # slightly tighter than at online
    #'tag_mu_tagtrigger_tag>0.5', # shit add trigger info in flattened ntuple
    #'abs(mu1_dz)<0.2'          ,
    #'abs(mu2_dz)<0.2'          ,
    #'abs(mu1_dxy)<0.05'        ,
    #'abs(mu2_dxy)<0.05'        ,
    #'lxy_sig<3 '               ,
]).replace('tagtrigger', tag_trigger)

rdf_data = ROOT.RDataFrame(tree_data).Filter(selection)
rdf_mc   = ROOT.RDataFrame(tree_mc  ).Filter(selection)

# save Least Common Multiplier between tag and probe trigger prescales
rdf_data = rdf_data.Define('tag_L1_ps', '3*670') # 670 it the _hopefully_ constant L1 prescale of L1_SingleMu7 https://cmsoms.cern.ch/cms/triggers/l1_rates?cms_run=319579&props.11273_11270.selectedCells=L1A%20physics:2
rdf_data = rdf_data.Define('lcm'      , 'std::lcm(670*int(%s), int(%s))' %(tag_trigger+'_ps', probe_trigger+'_ps'))
rdf_mc   = rdf_mc  .Define('lcm'      , 'std::lcm(int(%s), int(%s))' %(tag_trigger+'_ps', probe_trigger+'_ps'))

bins = np.array([3., 4., 6., 9., 12., 20.])

histos = OrderedDict()
histos['probe_mu_pt'] = ROOT.TH1D('probe_mu_pt', '', len(bins)-1, bins)

labels = {}
labels['probe_mu_pt'] = 'probe #mu p_{T} (GeV)'

# Canvas and Pad gymnastics
c1 = ROOT.TCanvas('c1', '', 700, 700)
c1.Draw()
c1.cd()
main_pad = ROOT.TPad('main_pad', '', 0., 0.25, 1. , 1.  )
main_pad.Draw()
c1.cd()
ratio_pad = ROOT.TPad('ratio_pad', '', 0., 0., 1., 0.25)
ratio_pad.Draw()
main_pad.SetTicks(True)
main_pad.SetBottomMargin(0.)
ratio_pad.SetTopMargin(0.)   
ratio_pad.SetGridy()
ratio_pad.SetBottomMargin(0.45)
     
hmodel = ROOT.RDF.TH1DModel(histos['probe_mu_pt'])

hdata_den = rdf_data.Histo1D(hmodel, 'probe_mu_pt', 'tag_L1_ps') 
hmc_den   = rdf_mc  .Histo1D(hmodel, 'probe_mu_pt')

# shit, add trigger information in flattened ntuples
#hdata_num = rdf_data.Filter('mu2_%s_tag>0.5' %probe_trigger).Histo1D(hmodel, 'mu2_pt', 'lcm')
#hmc_num   = rdf_mc  .Filter('mu2_%s_tag>0.5' %probe_trigger).Histo1D(hmodel, 'mu2_pt', 'lcm')

hdata_num = rdf_data.Filter('%s>0.5' %probe_trigger).Histo1D(hmodel, 'probe_mu_pt', 'lcm')
hmc_num   = rdf_mc  .Filter('%s>0.5' %probe_trigger).Histo1D(hmodel, 'probe_mu_pt', 'lcm')

# not the best way to use RDFs ... FIXME!
hdata_den = hdata_den.GetValue()
hmc_den   = hmc_den  .GetValue()
hdata_num = hdata_num.GetValue()
hmc_num   = hmc_num  .GetValue()

hdata_num.Divide(hdata_den)
hmc_num  .Divide(hmc_den  )

hdata_num.GetYaxis().SetTitle('efficiency')
hdata_num.GetXaxis().SetTitle(labels['probe_mu_pt'])
hdata_num.SetLineColor(ROOT.kBlack)
hdata_num.SetMarkerStyle(8)

hmc_num.GetYaxis().SetTitle('efficiency')
hmc_num.GetXaxis().SetTitle(labels['probe_mu_pt'])
hmc_num.SetLineColor(ROOT.kRed)
hmc_num.SetMarkerStyle(21)
hmc_num.SetMarkerColor(ROOT.kRed)

c1.cd()
main_pad.cd()

hdata_num.GetYaxis().SetRangeUser(0.001, 1.4)

hdata_num.Draw('ep')
hmc_num  .Draw('ep same')

leg = ROOT.TLegend(0.6,.7,.88,.88)
leg.SetBorderSize(0)
leg.SetFillColor(0)
leg.SetFillStyle(0)
leg.AddEntry(hdata_num, 'observed'                            , 'ep')
leg.AddEntry(hmc_num  , 'nonprompt J/#psi#rightarrow#mu#mu MC', 'ep' )
leg.Draw('same')

#main_pad.SetLogy(True)

c1.cd()
ratio_pad.cd()
ratio = hdata_num.Clone()
ratio.Divide(hmc_num)
ratio.Draw('ep')
ratio.GetYaxis().SetRangeUser(1e-3, 2-1e-3)
line = ROOT.TLine(ratio.GetXaxis().GetXmin(), 1., ratio.GetXaxis().GetXmax(), 1.)
line.Draw('same')
ratio_pad.SetLogy(False)

ratio.GetYaxis().SetTitle('SF')
ratio.GetYaxis().SetTitleOffset(0.5)
ratio.GetYaxis().SetNdivisions(405)
ratio.GetXaxis().SetLabelSize(3.* ratio.GetXaxis().GetLabelSize())
ratio.GetYaxis().SetLabelSize(3.* ratio.GetYaxis().GetLabelSize())
ratio.GetXaxis().SetTitleSize(3.* ratio.GetXaxis().GetTitleSize())
ratio.GetYaxis().SetTitleSize(3.* ratio.GetYaxis().GetTitleSize())

CMS_lumi(main_pad, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = '')

c1.SaveAs('%s/%s_over_%s_vs_%s.pdf' %(newdir, probe_trigger, tag_trigger, 'probe_mu_pt'))

