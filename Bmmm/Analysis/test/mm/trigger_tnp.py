import ROOT
from itertools import product
from collections import OrderedDict

ROOT.TH1.SetDefaultSumw2(True)
ROOT.gROOT.SetBatch(True)


#file = ROOT.TFile.Open('jpsimm.root', 'read')
file = ROOT.TFile.Open('data_03apr23.root', 'read')
file.cd()
tree = file.Get('tree')

h_template = ROOT.TH1F('template', '', 50, 0, 15)

tag_trigger = 'HLT_Mu8'
probe_trigger = 'HLT_Mu8'
#probe_trigger = 'HLT_DoubleMu4_3_Jpsi'
#probe_trigger = 'HLT_Dimuon0_Jpsi_NoVertexing'
#probe_trigger = 'HLT_Dimuon0_Jpsi_NoVertexing_L1_4R_0er1p5R'

# replace tagmu and probemu
selection = ' & '.join([
#    'dr_12<1.4'              ,
    'abs(mu1_dz)<0.2'        ,
    'abs(mu2_dz)<0.2'        ,
    'abs(mu1_eta)<1.4'       ,
    'abs(mu2_eta)<1.4'       ,
#    'cos2d>0.9'              ,                                         
#    'vtx_prob>0.1'           ,
#    'vtx_chi2<999999'        ,
    'abs(tagmu_dz)<0.2'      ,
    'abs(probemu_dz)<0.2'    ,
    'tagtrigger>0.5'         ,
#    'pt>6.9'                 ,
    'abs(charge)==0 '        ,
    'abs(mass-3.0969)<0.12'  ,
    'tagmu_id_medium>0.5 '   ,
    'tagmu_HLT_Mu8_tag>0.5'  ,
    'probemu_id_soft_mva>0.5',
    'charge==0'              ,
    #'lxy_sig<3 '             ,
]).replace('tagtrigger', tag_trigger)

selection += ' & tagmu_%s_tag>0.5' %tag_trigger
selection_pass = selection + ' & probemu_%s_tag>0.5' %probe_trigger # FIXME! tag or probe?

histos = OrderedDict()
for imu, ihist in product(['mu1', 'mu2'], ['all', 'pass', 'fail']):
    histos['all_%s'  %imu] = h_template.Clone() ; histos['all_%s'  %imu].SetName('all_%s'  %imu)
    histos['pass_%s' %imu] = h_template.Clone() ; histos['pass_%s' %imu].SetName('pass_%s' %imu)

selections = OrderedDict()
for itnp in [['mu1', 'mu2'], ['mu2', 'mu1']]:
    selections['all_%s'  %itnp[0]] = selection     .replace('tagmu', itnp[1]).replace('probemu', itnp[0])
    selections['pass_%s' %itnp[0]] = selection_pass.replace('tagmu', itnp[1]).replace('probemu', itnp[0])

for itnp in [['mu1', 'mu2'], ['mu2', 'mu1']]:
    tree.Draw("%s_pt>>%s" %(itnp[1], 'all_%s'  %itnp[1]), selections['all_%s'  %itnp[1]])
    tree.Draw("%s_pt>>%s" %(itnp[1], 'pass_%s' %itnp[1]), selections['pass_%s' %itnp[1]])

histos['all_mu1' ].Add(histos['all_mu2' ])
histos['pass_mu1'].Add(histos['pass_mu2'])

eff = ROOT.TEfficiency(histos['pass_mu1'], histos['all_mu1'])

eff.Draw()

ROOT.gPad.SaveAs('efficiency_%s.pdf' %probe_trigger)


