import os
import ROOT
from itertools import product
from collections import OrderedDict
from cmsstyle import CMS_lumi
from officialStyle import officialStyle

newdir = 'nonprompt'
#newdir = 'prompt'
if not os.path.exists(newdir):
    os.makedirs(newdir)

officialStyle(ROOT.gStyle, ROOT.TGaxis)

ROOT.TH1.SetDefaultSumw2(True)
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

#file = ROOT.TFile.Open('jpsimm.root', 'read')
#file_data = ROOT.TFile.Open('data_doublemu_03apr23.root', 'read')
file_data = ROOT.TFile.Open('data_doublemu_03apr23_jpsi.root', 'read')
#file_data = ROOT.TFile.Open('data_singlemu_03apr23.root', 'read')
file_data.cd()
tree_data = file_data.Get('tree')

#file_mc = ROOT.TFile.Open('jpsimm.root', 'read')
file_mc = ROOT.TFile.Open('hbtommx.root', 'read')
file_mc.cd()
tree_mc = file_mc.Get('tree')

if newdir=='prompt': nbins=60
if newdir=='nonprompt': nbins=30

h_template_rel = ROOT.TH1F('template', '', nbins, 0, 2)
h_template_abs = ROOT.TH1F('template', '', nbins, 0, 5)

#tag_trigger = 'HLT_IsoMu24'
#probe_trigger = 'HLT_IsoMu24'

tag_trigger = 'HLT_Mu8'
probe_trigger = 'HLT_Mu8'
#probe_trigger = 'HLT_DoubleMu4_3_Jpsi'
#probe_trigger = 'HLT_Dimuon0_Jpsi_NoVertexing'
#probe_trigger = 'HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R'

# replace tagmu and probemu
selection = ' & '.join([
    'mu1_pt>4.'               ,
    'mu2_pt>4.'               ,
    #'abs(mu1_dz)<0.2'         ,
    #'abs(mu2_dz)<0.2'         ,
    #'abs(mu1_dxy)<0.05'       ,
    #'abs(mu2_dxy)<0.05'       ,
    'abs(mu1_eta)<2.4'        ,
    'abs(mu2_eta)<2.4'        ,
    'tagtrigger>0.5'          ,
    #'charge==0'               ,
    'abs(mass-3.1)<0.12'      ,
    'mu1_id_medium>0.5 '      ,
    'mu2_id_medium>0.5'       ,
    'tagmu_tagtrigger_tag>0.5',
]).replace('tagtrigger', tag_trigger)

selection += ' & tagmu_%s_tag>0.5' %tag_trigger

selection_prompt =  ' & '.join([
    'abs(mu1_dz)<0.03'  ,
    'abs(mu2_dz)<0.03'  ,
    'abs(mu1_dxy)<0.01' ,
    'abs(mu2_dxy)<0.01' ,
    'lxy<0.05'          ,
    'lxy_sig<3'         ,
])

selection_nonprompt =  ' & '.join([
    'abs(mu1_dz)>0.03'  ,
    'abs(mu2_dz)>0.03'  ,
    'abs(mu1_dxy)>0.01' ,
    'abs(mu2_dxy)>0.01' ,
    'lxy>0.05'          ,
    'lxy_sig>3'         ,
])

#selection += ' & ' + selection_prompt
selection += ' & ' + selection_nonprompt

rel_isolations = [
    'pfreliso03',
    'pfreliso04',
]

abs_isolations = [
    'pfiso03'   ,
    'pfiso04'   ,
    'pfiso03_ch',
    'pfiso03_cp',
    'pfiso03_nh',
    'pfiso03_ph',
    'pfiso03_pu',
    'pfiso04_ch',
    'pfiso04_cp',
    'pfiso04_nh',
    'pfiso04_ph',
    'pfiso04_pu',
]


labels = {}
labels['pfreliso03'] = 'PF #Delta#beta-corr Iso^{rel} R=0.3'
labels['pfreliso04'] = 'PF #Delta#beta-corr Iso^{rel} R=0.4'
labels['pfiso03'   ] = 'PF #Delta#beta-corr Iso^{abs} R=0.3'
labels['pfiso04'   ] = 'PF #Delta#beta-corr Iso^{abs} R=0.4' 
labels['pfiso03_ch'] = 'PF charged hadron Iso^{abs} R=0.3'
labels['pfiso03_cp'] = 'PF charged particle Iso^{abs} R=0.3'
labels['pfiso03_nh'] = 'PF neutral hadron Iso^{abs} R=0.3'
labels['pfiso03_ph'] = 'PF photon Iso^{abs} R=0.3'
labels['pfiso03_pu'] = 'PF pileup Iso^{abs} R=0.3'
labels['pfiso04_ch'] = 'PF charged hadron Iso^{abs} R=0.4'
labels['pfiso04_cp'] = 'PF charged particle Iso^{abs} R=0.4'
labels['pfiso04_nh'] = 'PF neutral hadron Iso^{abs} R=0.4'
labels['pfiso04_ph'] = 'PF photon Iso^{abs} R=0.4'
labels['pfiso04_pu'] = 'PF pileup Iso^{abs} R=0.4'

histos = OrderedDict()
for imu, iso in product(['mu1', 'mu2'], rel_isolations):
    histos['%s-%s' %(imu, iso)] = h_template_rel.Clone()
    histos['%s-%s' %(imu, iso)].SetName('%s_%s' %(imu, iso))

for imu, iso in product(['mu1', 'mu2'], abs_isolations):
    histos['%s-%s' %(imu, iso)] = h_template_abs.Clone()
    histos['%s-%s' %(imu, iso)].SetName('%s_%s' %(imu, iso))

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
 
for k, v in histos.items():
    iso_to_plot     = k.split('-')[1]
    muon_to_plot    = k.split('-')[0]
    muon_to_trigger = 'mu1' if muon_to_plot=='mu2' else 'mu2'
   
    xaxis_label = '%s %s' %('#mu_{1}' if muon_to_plot=='mu1' else '#mu_{2}', labels[iso_to_plot])
   
    hdata = v.Clone() 
    hdata.SetName('data_' + v.GetName())
    hdata.GetYaxis().SetTitle('a.u.')
    hdata.GetXaxis().SetTitle(xaxis_label)

    hmc = v.Clone() 
    hmc.SetName('mc_' + v.GetName())
    hmc.GetYaxis().SetTitle('a.u.')
    hmc.GetXaxis().SetTitle(xaxis_label)
    
    my_selection = selection.replace('tagmu', muon_to_trigger)
    
    c1.cd()
    main_pad.cd()

    tree_data.Draw('%s_%s>>%s' %(muon_to_plot, iso_to_plot, hdata.GetName()), '(%s) * (charge==0 - charge!=0)' %my_selection)
    tree_mc  .Draw('%s_%s>>%s' %(muon_to_plot, iso_to_plot, hmc  .GetName()), '(%s) * (charge==0 - charge!=0)' %my_selection)
    
    if newdir == 'nonprompt':
        colour = ROOT.kBlue-7
    else:
        colour = ROOT.kOrange
        
    hmc.SetFillColor(colour)
    hdata.SetLineColor(ROOT.kBlack)
    hdata.SetMarkerStyle(8)
    
    hmc  .Scale(1./hmc  .Integral())
    hdata.Scale(1./hdata.Integral())

    hmc.SetMaximum(1.2 * max(hmc.GetMaximum(), hmc.GetMaximum()))    
    #hmc.SetMinimum(1e-4)
    
    hmc.Draw('HIST')
    hdata.Draw('ep same')

    leg = ROOT.TLegend(0.6,.7,.88,.88)
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
#    leg.SetTextFont(42)
#    leg.SetTextSize(0.035)
#    leg.SetNColumns(3)
    leg.AddEntry(hdata, 'observed', 'ep')
    leg.AddEntry(hmc, '%s J/#psi#rightarrow#mu#mu MC'  %newdir, 'f')
    leg.Draw('same')
    
    main_pad.SetLogy(True)

    c1.cd()
    ratio_pad.cd()
    ratio = hdata.Clone()
    ratio.Divide(hmc)
    ratio.Draw('ep')
    line = ROOT.TLine(ratio.GetXaxis().GetXmin(), 1., ratio.GetXaxis().GetXmax(), 1.)
    line.Draw('same')
    ratio_pad.SetLogy(False)

    ratio.GetYaxis().SetTitle('obs/exp')
    ratio.GetYaxis().SetTitleOffset(0.5)
    ratio.GetYaxis().SetNdivisions(405)
    ratio.GetXaxis().SetLabelSize(3.* ratio.GetXaxis().GetLabelSize())
    ratio.GetYaxis().SetLabelSize(3.* ratio.GetYaxis().GetLabelSize())
    ratio.GetXaxis().SetTitleSize(3.* ratio.GetXaxis().GetTitleSize())
    ratio.GetYaxis().SetTitleSize(3.* ratio.GetYaxis().GetTitleSize())

    CMS_lumi(main_pad, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = '')

    c1.SaveAs('%s/jpsi_%s.pdf' %(newdir, k))

    main_pad.SetLogy(False)
    main_pad.Modified()
    main_pad.Update()
    
    c1.SaveAs('%s/jpsi_%s_lin.pdf' %(newdir, k))
    


