import os
import glob
import ROOT
from itertools import product
from collections import OrderedDict
from cmsstyle import CMS_lumi
from officialStyle import officialStyle

newdir = 'dymm'
if not os.path.exists(newdir):
    os.makedirs(newdir)

officialStyle(ROOT.gStyle, ROOT.TGaxis)

ROOT.TH1.SetDefaultSumw2(True)
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

#file = ROOT.TFile.Open('jpsimm.root', 'read')
#file_data = ROOT.TFile.Open('data_doublemu_03apr23.root', 'read')
#file_data = ROOT.TFile.Open('data_doublemu_03apr23_jpsi.root', 'read')
file_data = ROOT.TFile.Open('data_singlemu_03apr23.root', 'read')
file_data.cd()

#tree_data = ROOT.TChain('tree')
#files_data = [ifile for ifile in glob.glob('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/HbToMuMuX_2018UL_03Apr2023_v1/*root') if (os.path.getsize(ifile) >> 10) > 500]
#map(tree_data.Add, files_data)
#rdf_data = ROOT.RDataFrame(tree_data)

tree_data = file_data.Get('tree')

#file_mc = ROOT.TFile.Open('jpsimm.root', 'read')
file_mc = ROOT.TFile.Open('dymm.root', 'read')
file_mc.cd()
tree_mc = file_mc.Get('tree')

tag_trigger = 'HLT_IsoMu24'
probe_trigger = 'HLT_IsoMu24'

#tag_trigger = 'HLT_Mu8'
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

selection += ' & ' + selection_prompt
#selection += ' & ' + selection_nonprompt

if newdir=='dymm'   : nbins=100

histos = OrderedDict()

histos['abs(mu1_dz)'        ] = ROOT.TH1F('mu1_dz_alt' , '', nbins,  0  ,   0.25)
histos['abs(mu2_dz)'        ] = ROOT.TH1F('mu2_dz_alt' , '', nbins,  0  ,   0.25)
histos['abs(mu1_dxy)'       ] = ROOT.TH1F('mu1_dxy_alt', '', nbins,  0  ,   0.06)
histos['abs(mu2_dxy)'       ] = ROOT.TH1F('mu2_dxy_alt', '', nbins,  0  ,   0.06)
histos['log10(abs(mu1_dz))' ] = ROOT.TH1F('mu1_dz'     , '', nbins, -5  ,   -0.5)
histos['log10(abs(mu2_dz))' ] = ROOT.TH1F('mu2_dz'     , '', nbins, -5  ,   -0.5)
histos['log10(abs(mu1_dxy))'] = ROOT.TH1F('mu1_dxy'    , '', nbins, -5  ,   -1)
histos['log10(abs(mu2_dxy))'] = ROOT.TH1F('mu2_dxy'    , '', nbins, -5  ,   -1)
histos['mu1_pt'             ] = ROOT.TH1F('mu1_pt'     , '', nbins,  0  ,  70  )
histos['mu2_pt'             ] = ROOT.TH1F('mu2_pt'     , '', nbins,  0  ,  70  )
histos['mu1_eta'            ] = ROOT.TH1F('mu1_eta'    , '', nbins, -2.5,   2.5)
histos['mu2_eta'            ] = ROOT.TH1F('mu2_eta'    , '', nbins, -2.5,   2.5)
histos['mass'               ] = ROOT.TH1F('mass'       , '', nbins,  2.5,   4.5)
histos['mass_alt'           ] = ROOT.TH1F('mass_alt'   , '', nbins, 91.2-5,   91.2+5)
histos['npv'                ] = ROOT.TH1F('npv'        , '',   100,  0  , 100  )
histos['lxy'                ] = ROOT.TH1F('lxy'        , '', nbins,  0  ,   1  )
histos['lxy_sig'            ] = ROOT.TH1F('lxy_sig'    , '', nbins,  0  ,  20  )
histos['log10(lxy)'         ] = ROOT.TH1F('log10lxy'   , '', nbins, -4  ,   1. )
histos['log10(lxy_sig)'     ] = ROOT.TH1F('log10lxysig', '', nbins, -2  ,   2. )
histos['vtx_prob'           ] = ROOT.TH1F('vtx_prob'   , '', nbins,  0  ,   1  )
histos['dr_12'              ] = ROOT.TH1F('dr_12'      , '', nbins,  0  ,   7  )

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
 
for k, v in histos.items():

    hdata = v.Clone() 
    hdata.SetName('data_' + v.GetName())
    hdata.GetYaxis().SetTitle('a.u.')
    hdata.GetXaxis().SetTitle(labels[k])

    hmc = v.Clone() 
    hmc.SetName('mc_' + v.GetName())
    hmc.GetYaxis().SetTitle('a.u.')
    hmc.GetXaxis().SetTitle(labels[k])
           
    c1.cd()
    main_pad.cd()
    
    my_selection = selection
    if k!='mass':
        my_selection += ' & abs(mass-91.2)<5' 
    
    tree_data.Draw('%s>>%s' %(k.replace('_alt',''), hdata.GetName()), '(%s) * (charge==0 - charge!=0)' %my_selection)
    tree_mc  .Draw('%s>>%s' %(k.replace('_alt',''), hmc  .GetName()), '(%s) * (charge==0 - charge!=0)' %my_selection)
    
    colour = ROOT.kGreen - 6
        
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
    leg.AddEntry(hmc, 'DY#rightarrow#mu#mu MC', 'f')
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

    CMS_lumi(main_pad, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = '')

    c1.SaveAs('%s/dy_%s.pdf' %(newdir, k))



