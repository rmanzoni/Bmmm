'''
RUN THIS WITH CONDA ROOT!!
'''

import os
import ROOT
import glob
from itertools import product
from collections import OrderedDict
from cmsstyle import CMS_lumi
from officialStyle import officialStyle

ROOT.EnableImplicitMT()

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
#file_data = ROOT.TFile.Open('data_doublemu_03apr23_jpsi.root', 'read')
#file_data = ROOT.TFile.Open('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/rjpsi_sf/charmonium_ul2018_HLT_DoubleMu4_3_Jpsi.root', 'read')
#file_data = ROOT.TFile.Open('data_singlemu_03apr23.root', 'read')
#file_data.cd()
#tree_data = file_data.Get('tree')

#file_mc = ROOT.TFile.Open('jpsimm.root', 'read')
#file_mc = ROOT.TFile.Open('hbtommx.root', 'read')
#file_mc = ROOT.TFile.Open('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/rjpsi_sf/hb_HLT_DoubleMu4_3_Jpsi.root', 'read')
#file_mc.cd()
#tree_mc = file_mc.Get('tree')



tree_data = ROOT.TChain('tree')
files_data = [ifile for ifile in glob.glob('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/Charmonium_Run2018*-UL2018_MiniAODv2_GT36-v1_26Apr2023_v1/*root') if (os.path.getsize(ifile) >> 10) > 900]
#for ifile in files_data: tree_data.Add(ifile)
rdf_data = ROOT.RDataFrame('tree', files_data)

tree_mc = ROOT.TChain('tree')
files_mc = [ifile for ifile in glob.glob('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/HbToMuMuX_2018UL_03Apr2023_v1/*root') if (os.path.getsize(ifile) >> 10) > 500]
#for ifile in files_mc: tree_mc.Add(ifile)
rdf_mc = ROOT.RDataFrame('tree', files_mc)



print('got my trees...')

#tag_trigger = 'HLT_IsoMu24'
#probe_trigger = 'HLT_IsoMu24'

#tag_trigger = 'HLT_Mu8'
tag_trigger = 'HLT_DoubleMu4_3_Jpsi'
#probe_trigger = 'HLT_Mu8'
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
    #'abs(mass-3.1)<0.12'      ,
    'mu1_id_medium>0.5 '      ,
    'mu2_id_medium>0.5'       ,
    '(mu1_tagtrigger_tag>0.5 || mu2_tagtrigger_tag>0.5)',
    #'lxy_sig<3 '             ,
]).replace('tagtrigger', tag_trigger)

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

rdf_data = rdf_data.Filter(selection)
rdf_mc   = rdf_mc  .Filter(selection)

rdf_data = rdf_data.Define('weight', '(charge==0 - charge!=0)')
rdf_mc   = rdf_mc  .Define('weight', '(charge==0 - charge!=0)')

if newdir=='prompt'   : nbins=100
if newdir=='nonprompt': nbins=40

histos = OrderedDict()

histos['npv'                ] = ROOT.TH1D('npv'        , '',   100,  0  , 100  )
histos['mu1_pt'             ] = ROOT.TH1D('mu1_pt'     , '', nbins,  0  ,  30  )
histos['mu2_pt'             ] = ROOT.TH1D('mu2_pt'     , '', nbins,  0  ,  20  )
histos['mu1_eta'            ] = ROOT.TH1D('mu1_eta'    , '', nbins, -2.5,   2.5)
histos['mu2_eta'            ] = ROOT.TH1D('mu2_eta'    , '', nbins, -2.5,   2.5)
histos['mass'               ] = ROOT.TH1D('mass'       , '', nbins,  2.5,   4.5)
histos['mass_alt'           ] = ROOT.TH1D('mass_alt'   , '', nbins,  3.1-0.12,   3.1+0.12)
histos['lxy'                ] = ROOT.TH1D('lxy'        , '', nbins,  0  ,   1  )
histos['lxy_sig'            ] = ROOT.TH1D('lxy_sig'    , '', nbins,  0  ,  20  )
# histos['log10(lxy)'         ] = ROOT.TH1D('log10lxy'   , '', nbins, -4  ,   1. )
# histos['log10(lxy_sig)'     ] = ROOT.TH1D('log10lxysig', '', nbins, -2  ,   2. )
histos['vtx_prob'           ] = ROOT.TH1D('vtx_prob'   , '', nbins,  0  ,   1  )
histos['dr_12'              ] = ROOT.TH1D('dr_12'      , '', nbins,  0  ,   0.8)
# histos['abs(mu1_dz)'        ] = ROOT.TH1D('mu1_dz_alt' , '', nbins,  0  ,   0.25)
# histos['abs(mu2_dz)'        ] = ROOT.TH1D('mu2_dz_alt' , '', nbins,  0  ,   0.25)
# histos['abs(mu1_dxy)'       ] = ROOT.TH1D('mu1_dxy_alt', '', nbins,  0  ,   0.06)
# histos['abs(mu2_dxy)'       ] = ROOT.TH1D('mu2_dxy_alt', '', nbins,  0  ,   0.06)
# histos['log10(abs(mu1_dz))' ] = ROOT.TH1D('mu1_dz'     , '', nbins, -5  ,   -0.5)
# histos['log10(abs(mu2_dz))' ] = ROOT.TH1D('mu2_dz'     , '', nbins, -5  ,   -0.5)
# histos['log10(abs(mu1_dxy))'] = ROOT.TH1D('mu1_dxy'    , '', nbins, -5  ,   -1)
# histos['log10(abs(mu2_dxy))'] = ROOT.TH1D('mu2_dxy'    , '', nbins, -5  ,   -1)

labels = {}
labels['mass'          ] = 'mass(#mu, #mu) (GeV)'
labels['mass_alt'      ] = 'mass(#mu, #mu) (GeV)'
labels['npv'           ] = 'number of PV'
labels['lxy'           ] = 'L_{xy} (cm)'
labels['lxy_sig'       ] = 'L_{xy} / #sigma(L_{xy})' 
labels['log10(lxy)'    ] = 'log_{10}(L_{xy}) (cm)' 
labels['log10(lxy_sig)'] = 'log_{10}(L_{xy} / #sigma(L_{xy})'
labels['mu1_pt'        ] = '#mu_{1} p_{T} (GeV)'
labels['mu2_pt'        ] = '#mu_{2} p_{T} (GeV)'
labels['mu1_eta'       ] = '#mu_{1} #eta'
labels['mu2_eta'       ] = '#mu_{2} #eta'
labels['vtx_prob'      ] = 'vertex(#mu, #mu) probability'
labels['dr_12'         ] = '#DeltaR(#mu_{1}, #mu_{2})'
labels['log10(abs(mu1_dz))' ] = '#mu_{1} |dz| (cm)'
labels['log10(abs(mu2_dz))' ] = '#mu_{2} |dz| (cm)'
labels['log10(abs(mu1_dxy))'] = '#mu_{1} |dxy| (cm)'
labels['log10(abs(mu2_dxy))'] = '#mu_{2} |dxy| (cm)'
labels['abs(mu1_dz)'   ] = '#mu_{1} |dz| (cm)'
labels['abs(mu2_dz)'   ] = '#mu_{2} |dz| (cm)'
labels['abs(mu1_dxy)'  ] = '#mu_{1} |dxy| (cm)'
labels['abs(mu2_dxy)'  ] = '#mu_{2} |dxy| (cm)'

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

histos_data = histos.copy()
histos_mc   = histos.copy()

for k, v in histos.items():

    print('\t...processing', k)
    
    my_selection = selection
    if k!='mass':
        my_selection += ' & abs(mass-3.1)<0.12' 
    
    histo_model = ROOT.RDF.TH1DModel(v)
    
    #import pdb ; pdb.set_trace()
    histos_data[k] = rdf_data.Filter(my_selection).Histo1D(histo_model, k.replace('_alt',''), 'weight')
    histos_mc  [k] = rdf_mc  .Filter(my_selection).Histo1D(histo_model, k.replace('_alt',''), 'weight')
    
    #tree_data.Draw('%s>>%s' %(k.replace('_alt',''), hdata.GetName()), '(%s) * (charge==0 - charge!=0)' %my_selection)
    #tree_mc  .Draw('%s>>%s' %(k.replace('_alt',''), hmc  .GetName()), '(%s) * (charge==0 - charge!=0)' %my_selection)
    

for k, v in histos.items():

    hdata = histos_data[k].GetValue().Clone() 
    hdata.SetName('data_' + v.GetName())
    hdata.GetYaxis().SetTitle('a.u.')
    hdata.GetXaxis().SetTitle(labels[k])

    hmc = histos_mc[k].GetValue().Clone() 
    hmc.SetName('mc_' + v.GetName())
    hmc.GetYaxis().SetTitle('a.u.')
    hmc.GetXaxis().SetTitle(labels[k])
           
    c1.cd()
    main_pad.cd()
    
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
    hmc.SetMinimum(1e-6)
    
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
    leg.AddEntry(hmc, '%s J/#psi#rightarrow#mu#mu MC' %newdir, 'f')
    leg.Draw('same')
    
    #main_pad.SetLogy(True)

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

    if k == 'npv':
        print ('PILEUP REWEIGHING')
        for ibin in range(ratio.GetNbinsX()):
            print(ibin+1, ratio.GetBinCenter(ibin+1)-0.5, ratio.GetBinCenter(ibin+1)+0.5, ratio.GetBinContent(ibin+1), ratio.GetBinError(ibin+1))


    CMS_lumi(main_pad, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = '')

    c1.SaveAs('%s/jpsi_%s.pdf' %(newdir, k))



