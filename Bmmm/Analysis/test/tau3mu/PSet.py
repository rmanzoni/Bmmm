'''
Minimal "dummy" parameter-set for CRAB.

We do NOT run cmsRun on the grid -- the real work is done by
inspector_tau3mu.py, launched from crab_script.sh. CRAB still requires a
parseable PSet so that, for each job, it can inject that job's slice of the
input dataset into process.source.fileNames (and process.maxEvents). The
scriptExe then reads those file names back via `import PSet`.

Do not add an OutputModule here: there is no cmsRun output to collect. The
analysis ROOT file is declared in the CRAB config via JobType.outputFiles and
JobType.disableAutomaticOutputCollection = True.
'''

import FWCore.ParameterSet.Config as cms

process = cms.Process('Tau3Mu')

# CRAB overwrites fileNames per job. Leave empty here.
process.source = cms.Source(
    'PoolSource',
    fileNames = cms.untracked.vstring(),
)

# CRAB sets this per job (will stay -1 with FileBased splitting).
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1),
)

process.options = cms.untracked.PSet()
