'''
CRAB3 submitter to run inspector_tau3mu.py over the 2024
ParkingDoubleMuonLowMass MINIAOD datasets.

Why CRAB instead of the SLURM submitter:
  - The SLURM jobs read remote files through the global xrootd redirector over
    the WAN, which is slow and flaky once a few thousand jobs hammer it.
  - CRAB leaves locality scheduling ON, so each job runs at a site that hosts
    the data and reads it over the LAN; failed jobs are auto-resubmitted.
  - You no longer manage the job count by hand: CRAB splits each dataset for
    you (FileBased, unitsPerJob files per job).

How it works (scriptExe mode):
  - The real processing is still inspector_tau3mu.py, launched by crab_script.sh.
  - PSet.py is a dummy parameter-set; CRAB injects each job's input files into
    it, and crab_script.sh reads them back via `import PSet`.
  - FrameworkJobReport.xml is a static minimal report shipped with each job so
    CRAB's bookkeeping is happy without running cmsRun.

The inspector lives in $CMSSW_BASE/src/Bmmm/Analysis/test/tau3mu/ together with
its helper modules (Tau3MuCandidate.py, Tau3MuBranches.py, ...). That directory
is NOT on the worker-node python path, so we ship *all* of its .py/.h files via
JobType.inputFiles -- shipping only the inspector makes every job die at import
time (exit code 5, 0% CPU).

Required files in this directory (alongside this submitter):
    crab_script.sh
    PSet.py
    FrameworkJobReport.xml
inspector_tau3mu.py is shipped from $CMSSW_BASE/src/Bmmm/Analysis/test/tau3mu/.

Setup before running (order matters):
    cmsenv          # in the CMSSW src you built Bmmm in
    source /cvmfs/cms.cern.ch/crab3/crab.sh
    voms-proxy-init -rfc -voms cms -valid 192:00
    python3 crab_submitter_tau3mu_data.py

Useful afterwards:
    crab status   -d crab_tau3mu_data2024/crab_<requestName>
    crab getlog --short -d crab_tau3mu_data2024/crab_<requestName>   # job stdout
    crab resubmit -d crab_tau3mu_data2024/crab_<requestName>
    crab report   -d crab_tau3mu_data2024/crab_<requestName>
'''

import os
import glob
from multiprocessing import Process

from CRABClient.UserUtilities import config
from CRABAPI.RawCommand import crabCommand
from CRABClient.ClientExceptions import ClientException
from http.client import HTTPException


# ----------------------------------------------------------------------------
# user knobs
# ----------------------------------------------------------------------------
work_area     = 'crab_tau3mu_data2024_v2'
out_dir       = 'Tau3Mu_26Jun2026_data2024_crab_v3'      # under /store/user/manzoni/
files_per_job = 10
storage_site  = 'T3_CH_PSI'

# requestNames listed here are skipped (already submitted / done)
already_submitted = [
]

# era -> processing-version suffix, as in the SLURM submitter
eras = [
    ('C', '-v1'),
    ('D', '-v1'),
    ('E', '-v1'),
    ('F', '-v3'),
    ('G', '-v3'),
    ('H', '-v3'),
    ('I', '-v3'),
    ('I', '_v2-v2'),
]

# build the list of datasets to process (8 ParkingDoubleMuonLowMass parts each)
productions = []
for (iera, iversion) in eras:
    for part in range(8):
        dataset = '/ParkingDoubleMuonLowMass%d/Run2024%s-MINIv6NANOv15%s/MINIAOD' % (
            part, iera, iversion)
        productions.append(dataset)


# ----------------------------------------------------------------------------
# config builder
# ----------------------------------------------------------------------------
def create_config(dataset):
    cmssw_base = os.environ.get('CMSSW_BASE', '')
    if not cmssw_base:
        raise RuntimeError('CMSSW_BASE is not set -- run `cmsenv` first.')

    # ship the inspector AND all its sibling helper modules from test/tau3mu/.
    # That directory is not importable as a package on the WN, so the helpers
    # must travel in the sandbox alongside the inspector (CRAB flattens them
    # into the job's working dir, where `import Tau3MuCandidate` etc. resolve).
    tau3mu_dir = os.path.join(cmssw_base, 'src', 'Bmmm', 'Analysis', 'test', 'tau3mu')
    helpers = sorted(set(
        glob.glob(os.path.join(tau3mu_dir, '*.py')) +
        glob.glob(os.path.join(tau3mu_dir, '*.h'))
    ))
    if not any(f.endswith('inspector_tau3mu.py') for f in helpers):
        raise RuntimeError('inspector_tau3mu.py not found under %s' % tau3mu_dir)

    # third-party python packages that are NOT in CMSSW (e.g. `particle`) and
    # are normally picked up from ~/.local -- which the WN does not have. They
    # were installed into test/tau3mu/pylibs/ via:
    #   PYTHONNOUSERSITE=1 pip install --no-cache-dir --target=pylibs particle
    # Ship the whole tree (CRAB recurses into directories given in inputFiles)
    # and crab_script.sh prepends ./pylibs to PYTHONPATH.
    pylibs_dir = os.path.join(tau3mu_dir, 'pylibs')
    if not os.path.isdir(pylibs_dir):
        raise RuntimeError(
            'pylibs/ not found under %s -- install the non-CMSSW packages first:\n'
            '  cd %s\n'
            '  PYTHONNOUSERSITE=1 pip install --no-cache-dir --target=pylibs particle'
            % (tau3mu_dir, tau3mu_dir))

    # human-readable, unique request name, e.g. tau3mu_LowMass0_Run2024C_v1
    #   dataset = /ParkingDoubleMuonLowMass0/Run2024C-MINIv6NANOv15-v1/MINIAOD
    primary    = dataset.split('/')[1]                       # ParkingDoubleMuonLowMass0
    part       = primary.replace('ParkingDoubleMuonLowMass', '')
    processed  = dataset.split('/')[2]                        # Run2024C-MINIv6NANOv15-v1
    era        = processed.split('-')[0].replace('Run2024', '')   # C
    ver_clean  = processed.split('MINIv6NANOv15')[-1].lstrip('-_').replace('-', '')  # v1 / v2v2
    request    = 'tau3mu_LowMass%s_Run2024%s_%s' % (part, era, ver_clean)

    cfg = config()

    cfg.General.requestName     = request
    cfg.General.workArea        = work_area
    cfg.General.transferOutputs = True
    cfg.General.transferLogs    = True

    cfg.JobType.pluginName = 'Analysis'
    cfg.JobType.psetName   = 'PSet.py'
    cfg.JobType.scriptExe  = 'crab_script.sh'
    # inspector + all tau3mu helper modules + the pylibs tree + dummy report
    cfg.JobType.inputFiles = helpers + [pylibs_dir, 'FrameworkJobReport.xml']
    cfg.JobType.outputFiles = ['tau3mu.root']
    # the inspector produces a plain ROOT file, not an EDM output: tell CRAB
    # not to try to harvest outputs from the (non-existent) cmsRun report
    cfg.JobType.disableAutomaticOutputCollection = True
    # ship $CMSSW_BASE/python so `import Bmmm.Analysis...` works on the WN even
    # if you haven't re-scram-built since editing the package python
#     cfg.JobType.sendPythonFolder = True
    cfg.JobType.maxMemoryMB      = 2500
    # cfg.JobType.maxJobRuntimeMin = 1440   # uncomment/raise if jobs time out

    cfg.Data.inputDataset   = dataset
    cfg.Data.inputDBS       = 'global'
    cfg.Data.splitting      = 'FileBased'
    cfg.Data.unitsPerJob    = files_per_job
    cfg.Data.outLFNDirBase  = '/store/user/manzoni/' + out_dir + '/Run2024' + era
    cfg.Data.publication    = False
    cfg.Data.outputDatasetTag = out_dir + '_' + ver_clean
    # To restrict to certified lumis, drop in the 2024 golden JSON:
    # cfg.Data.lumiMask = '/eos/user/c/cmsdqm/www/CAF/certification/Collisions24/...'

    cfg.Site.storageSite = storage_site
    # If some input blocks sit only at T2_CH_CSCS and trip the global
    # blacklist (as seen for the private MC), uncomment:
    # cfg.Site.ignoreGlobalBlacklist = True

    return cfg


# ----------------------------------------------------------------------------
# submit (each in its own process to dodge the FWCore pset cache conflict)
# ----------------------------------------------------------------------------
def submit(cfg):
    try:
        crabCommand('submit', config=cfg)
    except HTTPException as hte:
        print('failed submitting %s: %s' % (cfg.General.requestName, hte.headers))
    except ClientException as cle:
        print('failed submitting %s: %s' % (cfg.General.requestName, cle))


if __name__ == '__main__':
    for ii, dataset in enumerate(productions):
        if ii==0: continue
        cfg = create_config(dataset)

        if cfg.General.requestName in already_submitted:
            print('skipping (already submitted): %s' % cfg.General.requestName)
            continue

        print('%s  ->  %s' % (dataset, cfg.General.requestName))

        p = Process(target=submit, args=(cfg,))
        p.start()
        p.join()
