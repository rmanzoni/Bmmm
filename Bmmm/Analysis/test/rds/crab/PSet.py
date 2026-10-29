'''
Dummy PSet for a CRAB scriptExe job (same as test/mm/crab/PSet.py).

The R(Ds) ntuplizer is a FWLite script, not a cmsRun configuration, so
CRAB gets this minimal PSet to do the job splitting. CRAB rewrites
process.source.fileNames in the sandbox copy once per job, and crab_script.sh
reads them back from here -- that is the only reason this file exists at runtime.
'''

import FWCore.ParameterSet.Config as cms

process = cms.Process("RDS")

process.source = cms.Source(
    'PoolSource',
    fileNames = cms.untracked.vstring(
        # placeholder, overwritten per job by CRAB; non-empty so the
        # configuration is parseable on its own
        '/store/data/Run2018A/ParkingBPH1/MINIAOD/UL2018_MiniAODv2-v1/placeholder.root',
    ),
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

# CRAB wants an output module even though nothing is written through it: the
# ntuple is produced by the script and declared in JobType.outputFiles.
process.output = cms.OutputModule(
    'PoolOutputModule',
    fileName = cms.untracked.string('dummy.root'),
)
process.out = cms.EndPath(process.output)
