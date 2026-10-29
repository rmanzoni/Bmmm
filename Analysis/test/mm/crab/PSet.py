'''
Dummy PSet for a CRAB scriptExe job.

The dimuon ntuplizer is a FWLite script, not a cmsRun configuration, so it
cannot be handed to CRAB as a psetName directly. The standard way round that
(the same one NanoAOD-tools uses) is to give CRAB a minimal PSet it can parse
for splitting, and do the real work in scriptExe.

CRAB rewrites process.source.fileNames in the sandbox copy of this file, once
per job, with that job's share of the dataset. crab_script.sh reads them back
out of here -- that is the only reason this file exists at runtime.
'''

import FWCore.ParameterSet.Config as cms

process = cms.Process('DIMUON')

process.source = cms.Source(
    'PoolSource',
    fileNames = cms.untracked.vstring(
        # placeholder, overwritten per job by CRAB. Kept non-empty so the
        # configuration is parseable on its own.
        '/store/data/Run2022C/ParkingDoubleMuonLowMass1/MINIAOD/PromptReco-v1/000/356/615/00000/placeholder.root',
    ),
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

# CRAB wants an output module in the PSet even though nothing is written
# through it: the ntuple is produced by the script and declared in
# JobType.outputFiles instead.
process.output = cms.OutputModule(
    'PoolOutputModule',
    fileName = cms.untracked.string('dummy.root'),
)
process.out = cms.EndPath(process.output)
