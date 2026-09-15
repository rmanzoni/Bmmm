'''
CRAB3 submitter to run inspector_mm_analysis.py over the Run 3
ParkingDoubleMuonLowMass1 MINIAOD datasets (Part 1 only).

A transliteration of test/tau3mu/crab/crab_submitter_tau3mu_data.py, which is a
WORKING setup for the same shape of job. Where this differs, the difference is
forced by the dimuon inspector and is flagged DIFFERENT below. Nothing else is
invented.

Why CRAB instead of the SLURM submitter:
  - The SLURM jobs read remote files through the global xrootd redirector over
    the WAN, which is slow and flaky once a few thousand jobs hammer it.
  - CRAB leaves locality scheduling ON, so each job runs at a site that hosts
    the data and reads it over the LAN; failed jobs are auto-resubmitted.
  - You no longer manage the job count by hand.

How it works (scriptExe mode):
  - The real processing is still inspector_mm_analysis.py, launched by
    crab_script.sh.
  - PSet.py is a dummy parameter-set; CRAB injects each job's input files into
    it, and crab_script.sh reads them back via `import PSet`.
  - FrameworkJobReport.xml is a static minimal report shipped with each job so
    CRAB's bookkeeping is happy without running cmsRun.

The inspector lives in $CMSSW_BASE/src/Bmmm/Analysis/test/mm/. That directory is
NOT on the worker-node python path, so we ship *all* of its .py/.h files via
JobType.inputFiles -- shipping only the inspector makes every job die at import
time (exit code 5, 0% CPU).

DIFFERENT from tau3mu, and only these two things:

  1. The dimuon inspector imports the PACKAGE, not siblings:
         from Bmmm.Analysis.MuMuBranches import ...
     The tau3mu one imports Tau3MuCandidate etc. by bare name, which the
     flattened sandbox satisfies on its own. So we additionally ship
     src/Bmmm/Analysis/python as a directory, and crab_script.sh rebuilds a
     Bmmm/Analysis package from it on PYTHONPATH. sendPythonFolder is switched
     on as well; either route alone would do, and the bootstrap has the
     advantage of not depending on CRAB internals.

  2. The inspector reads L1 menus from $CMSSW_BASE/src/Bmmm/Analysis/data at
     import. That directory is 247 MB and only l1menus/ is needed (2.2 MB
     packed), so l1menus/ alone is shipped and BMMM_DATADIR points the
     inspector at the job directory.

Required files in this directory (alongside this submitter):
    crab_script.sh
    PSet.py
    FrameworkJobReport.xml
    pylibs/
inspector_mm_analysis.py is shipped from $CMSSW_BASE/src/Bmmm/Analysis/test/mm/.

Setup before running (order matters):
    cmsenv          # in the CMSSW src you built Bmmm in
    source /cvmfs/cms.cern.ch/crab3/crab.sh
    voms-proxy-init -rfc -voms cms -valid 192:00
    python3 crab_dimuon_data_run3.py

Useful afterwards:
    crab status   -d crab_<work_area>/crab_<requestName>
    crab getlog --short -d crab_<work_area>/crab_<requestName>   # job stdout
    crab resubmit -d crab_<work_area>/crab_<requestName>
    crab report   -d crab_<work_area>/crab_<requestName>
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
work_area     = 'crab_dimuon_run3_relax_mu2pt'
# out_dir       = 'dimuon_ntuples_run3_09sep2026'      # under /store/user/manzoni/
out_dir       = 'dimuon_ntuples_run3_15sep2026'      # under /store/user/manzoni/
files_per_job = 2
storage_site  = 'T3_CH_PSI'

# requestNames listed here are skipped (already submitted / done)
already_submitted = [
]

# Part 1 only, as submitted for the RJpsi Run 3 skims -- exactly the
# ParkingDoubleMuonLowMass1 entries of skims/rjpsi/crab_data_*.py, so the two
# campaigns cover the same runs. Comment out entries to submit a subset;
# submitting ONE first is how you find out what a job costs.
productions = [
    '/ParkingDoubleMuonLowMass1/Run2022C-PromptReco-v1/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2022D-PromptReco-v1/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2022D-PromptReco-v2/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2022E-PromptReco-v1/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2022F-22Sep2023-v1/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2022G-22Sep2023-v1/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2023C-22Sep2023_v1-v2/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2023C-22Sep2023_v2-v1/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2023C-22Sep2023_v3-v1/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2023C-22Sep2023_v4-v1/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2023D-22Sep2023_v1-v1/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2023D-22Sep2023_v2-v1/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2026B-PromptReco-v1/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2026D-PromptReco-v1/MINIAOD',
]


# ----------------------------------------------------------------------------
# config builder
# ----------------------------------------------------------------------------
def create_config(dataset):
    cmssw_base = os.environ.get('CMSSW_BASE', '')
    if not cmssw_base:
        raise RuntimeError('CMSSW_BASE is not set -- run `cmsenv` first.')

    # ship the inspector AND all its sibling modules from test/mm/. That
    # directory is not importable as a package on the WN, so anything it might
    # reach for must travel in the sandbox alongside it (CRAB flattens them into
    # the job's working dir).
    mm_dir = os.path.join(cmssw_base, 'src', 'Bmmm', 'Analysis', 'test', 'mm')
    helpers = sorted(set(
        glob.glob(os.path.join(mm_dir, '*.py')) +
        glob.glob(os.path.join(mm_dir, '*.h'))
    ))
    if not any(f.endswith('inspector_mm_analysis.py') for f in helpers):
        raise RuntimeError('inspector_mm_analysis.py not found under %s' % mm_dir)

    # DIFFERENT (1): the package itself. inspector_mm_analysis.py does
    #     from Bmmm.Analysis.MuMuBranches import ...
    # so the flattened siblings are not enough. Ship the package python
    # directory; crab_script.sh rebuilds Bmmm/Analysis from it on PYTHONPATH.
    package_dir = os.path.join(cmssw_base, 'src', 'Bmmm', 'Analysis', 'python')
    if not os.path.isdir(package_dir):
        raise RuntimeError('%s not found' % package_dir)

    # DIFFERENT (2): L1 menus, read at import from
    # $CMSSW_BASE/src/Bmmm/Analysis/data. Only l1menus/ is needed; the rest of
    # data/ is 247 MB and would not be worth the sandbox.
    l1menus_dir = os.path.join(cmssw_base, 'src', 'Bmmm', 'Analysis', 'data', 'l1menus')
    if not os.path.isdir(l1menus_dir):
        raise RuntimeError('%s not found' % l1menus_dir)

    # third-party python packages that are NOT in CMSSW (particle, uproot) and
    # are normally picked up from ~/.local -- which the WN does not have. Install
    # them into this directory's pylibs/ with:
    #   PYTHONNOUSERSITE=1 pip3 install --no-cache-dir --target=pylibs particle uproot
    # Ship the whole tree (CRAB recurses into directories given in inputFiles)
    # and crab_script.sh prepends ./pylibs to PYTHONPATH.
    # NB: do NOT put numpy or scipy in there -- CMSSW has them, and a second
    # numpy is how the NumPy 1.x/2.x ImportError arises.
    here       = os.path.dirname(os.path.abspath(__file__))
    pylibs_dir = os.path.join(here, 'pylibs')
    if not os.path.isdir(pylibs_dir):
        raise RuntimeError(
            'pylibs/ not found under %s -- install the non-CMSSW packages first:\n'
            '  cd %s\n'
            '  PYTHONNOUSERSITE=1 pip3 install --no-cache-dir --target=pylibs '
            'particle uproot' % (here, here))

    # human-readable, unique request name, e.g. dimuon_LowMass1_Run2022C_PromptReco_v1
    #   dataset = /ParkingDoubleMuonLowMass1/Run2022C-PromptReco-v1/MINIAOD
    primary   = dataset.split('/')[1]                   # ParkingDoubleMuonLowMass1
    part      = primary.replace('ParkingDoubleMuonLowMass', '')
    processed = dataset.split('/')[2]                   # Run2022C-PromptReco-v1
    era       = processed.split('-')[0].replace('Run', '')           # 2022C
    ver_clean = '_'.join(processed.split('-')[1:]).replace('-', '')  # PromptReco_v1
    request   = 'dimuon_LowMass%s_Run%s_%s' % (part, era, ver_clean)

    cfg = config()

    cfg.General.requestName     = request[:100]
    cfg.General.workArea        = work_area
    cfg.General.transferOutputs = True
    cfg.General.transferLogs    = True

    cfg.JobType.pluginName = 'Analysis'
    cfg.JobType.psetName   = 'PSet.py'
    cfg.JobType.scriptExe  = 'crab_script.sh'
    # inspector + all mm siblings + the package python + l1menus + pylibs +
    # the static report
    cfg.JobType.inputFiles = helpers + [package_dir, l1menus_dir, pylibs_dir,
                                        'FrameworkJobReport.xml']
    cfg.JobType.outputFiles = ['dimuon_ntuple.root']
    # the inspector produces a plain ROOT file, not an EDM output: tell CRAB
    # not to try to harvest outputs from the (non-existent) cmsRun report
    cfg.JobType.disableAutomaticOutputCollection = True
    # ship $CMSSW_BASE/python so `import Bmmm.Analysis...` works on the WN even
    # if you haven't re-scram-built since editing the package python.
    # ON here (tau3mu has it off): see DIFFERENT (1).
#     cfg.JobType.sendPythonFolder = True
    cfg.JobType.maxMemoryMB      = 3000
    # cfg.JobType.maxJobRuntimeMin = 1440   # uncomment/raise if jobs time out

    cfg.Data.inputDataset   = dataset
    cfg.Data.inputDBS       = 'global'
    cfg.Data.splitting      = 'FileBased'
    cfg.Data.unitsPerJob    = files_per_job
    cfg.Data.outLFNDirBase  = '/store/user/manzoni/' + out_dir + '/Run' + era
    cfg.Data.publication    = False
    cfg.Data.outputDatasetTag = out_dir + '_' + ver_clean

    cfg.Site.storageSite = storage_site
    # If some input blocks sit only at T2_CH_CSCS and trip the global
    # blacklist, uncomment:
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
    for dataset in productions:
        cfg = create_config(dataset)

        if cfg.General.requestName in already_submitted:
            print('skipping (already submitted): %s' % cfg.General.requestName)
            continue

        print('%s  ->  %s' % (dataset, cfg.General.requestName))

        p = Process(target=submit, args=(cfg,))
        p.start()
        p.join()
