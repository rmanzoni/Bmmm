import glob


# https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun3Analysis

# allfiles = glob.glob('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/data_2024_10sept24_4m_loose_id_MINIAOD/*root')

allfiles = [
    'file:/work/manzoni/rjpsi_run3/CMSSW_15_1_1/src/Bmmm/Analysis/test/rjpsi/0443354B-2D3F-CF41-A1F0-0FC4F92E718E.root',
]


##############################################
hardmu_pt_cut = 3. # GeV
##############################################


import FWCore.ParameterSet.Config as cms

process = cms.Process('SKIM')

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = "auto:run2_mc"
process.GlobalTag.globaltag = '150X_dataRun3_v2'

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000)

process.maxEvents = cms.untracked.PSet(
    #input = cms.untracked.int32(100000),
    input = cms.untracked.int32(-1),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# this inputs the input files
process.source = cms.Source (
    'PoolSource',
    fileNames=cms.untracked.vstring(allfiles),
)

# this is for filtering on HLT path
process.hltFilter = cms.EDFilter('HLTHighLevel',
     TriggerResultsTag = cms.InputTag('TriggerResults','','HLT'),
     HLTPaths = cms.vstring('HLT_DoubleMu4_3_LowMass_v*'),           # provide list of HLT paths (or patterns) you want
     eventSetupPathsKey = cms.string(''), # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
     andOr = cms.bool(True),              # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
     throw = cms.bool(True),              # throw exception on unknown path names
)

## Muon selection
process.selectedMuons = cms.EDFilter(
    "PATMuonSelector",
    src = cms.InputTag("slimmedMuons"),
    cut = cms.string(' && '.join([
            'isPFMuon'                ,
            'isMediumMuon'            ,
            'abs(eta)<2.5'            ,
            'pt>2.'                   ,
#             'abs(bestTrack().dxy)<1.8',
        ])
    ),
    filter = cms.bool(True),
)

# module to filter on the number of Leptons
process.muonFilter = cms.EDFilter(
    "PATLeptonCountFilter",
    electronSource = cms.InputTag("slimmedElectrons"),
    muonSource     = cms.InputTag("selectedMuons"),
    tauSource      = cms.InputTag("slimmedTaus"),
    countElectrons = cms.bool(False),
    countMuons     = cms.bool(True),
    countTaus      = cms.bool(False),
    minNumber      = cms.uint32(3),
    maxNumber      = cms.uint32(999999),
)


process.hardMuons = cms.EDFilter(
    "PATMuonSelector",
    src = cms.InputTag('selectedMuons'),
    cut = cms.string('pt > %f' %hardmu_pt_cut),
    filter = cms.bool(True),
)

process.hardMuonFilter = cms.EDFilter(
    "PATLeptonCountFilter",
    electronSource = cms.InputTag("slimmedElectrons"),
    muonSource     = cms.InputTag("hardMuons"),
    tauSource      = cms.InputTag("slimmedTaus"),
    countElectrons = cms.bool(False),
    countMuons     = cms.bool(True),
    countTaus      = cms.bool(False),
    minNumber      = cms.uint32(2),
    maxNumber      = cms.uint32(999999),
)

# build two muon candidates first, the module cannot handle four in one go
process.twoMuonCandidates = cms.EDProducer(
    'CandViewCombiner',
    decay        = cms.string('hardMuons@+ hardMuons@-'),
    checkCharge  = cms.bool(True),
    checkOverlap = cms.bool(True),
    cut          = cms.string('mass > 2.6 & mass < 3.6'),
)

process.twoMuonCandidateFilter = cms.EDFilter(
    'CandViewCountFilter',
    src = cms.InputTag('twoMuonCandidates'),
    minNumber = cms.uint32(1), # see comment above
)

# create a collection of tracks 
process.load('PhysicsTools.PatAlgos.slimming.unpackedTracksAndVertices_cfi')

# Load vertex reconstruction
process.load("RecoVertex.Configuration.RecoVertex_cff")

process.primaryVertexRefit = process.unsortedOfflinePrimaryVertices.clone()
process.primaryVertexRefit.TrackLabel = cms.InputTag("unpackedTracksAndVertices")

process.skim = cms.Path(
    process.hltFilter                 *
    process.selectedMuons             * 
    process.muonFilter                *
    process.hardMuons                 *
    process.hardMuonFilter            *
    process.twoMuonCandidates         *
#     process.fourMuonCandidates        *
    process.twoMuonCandidateFilter    *
    process.unpackedTracksAndVertices *
    process.primaryVertexRefit
)

# talk to output module
process.out = cms.OutputModule('PoolOutputModule',
    fileName = cms.untracked.string('skimmed_bc_tomm_inclusive.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
#         'keep *_prunedGenParticles_*_*',
#         'keep *_packedGenParticles_*_*',
#         'keep *_generator_*_*',
        'keep *_slimmedAddPileupInfo_*_*',
#         'keep patMuons_slimmedMuons_*_*',
#         'keep *_packedPFCandidates_*_*',
#         'keep *_lostTracks_*_*',
#         'keep recoVertexs_offlineSlimmedPrimaryVerticesWithBS_*_*',
#         'keep recoVertexs_offlineSlimmedPrimaryVertices_*_*',
#         'keep TriggerResults_TriggerResults_*_HLT',
        'keep edmTriggerResults_TriggerResults_*_HLT',
        'keep patPackedTriggerPrescales_patTrigger__*',
        'keep *_offlineBeamSpot_*_*',
        'keep *_slimmedPatTrigger_*_*',
        'keep *_prunedTriggerObjects_*_*',
        
        'keep patJets_slimmedJets_*_*',
        
        # this can be removed at the end
        #'keep *_twoMuonCandidates_*_*',
        #'keep *_twoHardMuonCandidates_*_*',
#         'keep *_fourMuonCandidates_*_*',
        #'keep *_cleanedfourMuonCandidates_*_*',
#
        'keep *_selectedMuons_*_*',
        #'keep *_cleanedSelectedMuons_*_*',
        #'keep *_hardMuons_*_*',

        'keep recoTracks_unpackedTracksAndVertices_*_*',
        'keep *_primaryVertexRefit_WithBS_*',
    ),
	SelectEvents = cms.untracked.PSet(
		SelectEvents = cms.vstring('skim')
	),

)

# A list of analyzers or output modules to be run after all paths have been run.
process.outpath = cms.EndPath(process.out)

