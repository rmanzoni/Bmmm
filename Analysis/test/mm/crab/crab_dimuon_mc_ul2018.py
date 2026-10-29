'''
CRAB3 submitter to run inspector_mm_analysis.py over MINIAODSIM.

The MC twin of crab_dimuon_data_run3.py. Everything about the sandbox, the
scriptExe mechanism and the package bootstrap is identical -- see that file for
why any of it is shaped the way it is. Only three things differ, and all three
are forced by the input being simulation:

  1. --mc --savenontrig on the inspector, passed through JobType.scriptArgs as
     MC=1 SAVENONTRIG=1 and picked up by the KEY=value loop already in
     crab_script.sh. Without --mc there is no gen matching and the gen_* branches
     are all NaN; without --savenontrig the job keeps only events that fired the
     HLT path, which for an MC efficiency study is exactly what you do not want.

  2. requestName is derived from an MC dataset name, which has a different shape
     from a data one (no Run<era>, a Tune in the primary, a campaign in the
     processed name).

  3. inputDBS stays 'global' -- true for a central production. If you ever point
     this at a privately produced sample, that has to become 'phys03'.

Locality scheduling is left ON here as well, so each job runs where the data is.

Setup before running (order matters):
    cmsenv          # in the CMSSW src you built Bmmm in
    source /cvmfs/cms.cern.ch/crab3/crab.sh
    voms-proxy-init -rfc -voms cms -valid 192:00
    python3 crab_dimuon_mc_ul2018.py

Submit ONE task with max_files set before launching the campaign: that is how
you find out what a job costs, and MINIAODSIM files are typically larger than
the parked data ones this setup was tuned on.

Useful afterwards:
    crab status   -d crab_<work_area>/crab_<requestName>
    crab getlog --short -d crab_<work_area>/crab_<requestName>
    crab resubmit -d crab_<work_area>/crab_<requestName>
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
work_area     = 'crab_dimuon_mc_ul2018'
out_dir       = 'dimuon_ntuples_mc_ul2018_15sep2026'   # under /store/user/manzoni/
files_per_job = 2
storage_site  = 'T3_CH_PSI'

# Cap the number of input files, for a costing run. None submits the whole
# dataset. Set this to something small (5, say) for the first task and look at
# the runtime and output size before doing the rest.
max_files = None

# requestNames listed here are skipped (already submitted / done)
already_submitted = [
]

productions = [
    '/HbToJPsiMuMu_3MuFilter_TuneCP5_13TeV-pythia8-evtgen/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1_ext1-v3/MINIAODSIM',
]


# ----------------------------------------------------------------------------
# request name from an MC dataset name
# ----------------------------------------------------------------------------
def mc_request_name(dataset):
    '''/HbToJPsiMuMu_3MuFilter_TuneCP5_13TeV-pythia8-evtgen
       /RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1_ext1-v3
       /MINIAODSIM
       -> dimuon_mc_HbToJPsiMuMu_3MuFilter_UL18_ext1_v3

    Kept short deliberately: CRAB caps requestName at 100 characters, and the
    full conditions string would blow past that on its own while telling you
    nothing you cannot read off the dataset.
    '''
    primary, processed = dataset.split('/')[1], dataset.split('/')[2]

    # drop the tune and everything after it: TuneCP5_13TeV-pythia8-evtgen is the
    # generator setup, not something that distinguishes one task from another
    short = primary.split('_Tune')[0]

    campaign = processed.split('-')[0]                    # RunIISummer20UL18MiniAODv2
    tag = (campaign.replace('RunIISummer20', '')
                   .replace('MiniAODv2', '')
                   .replace('MiniAOD', ''))               # UL18

    # the bits that actually separate one production of the same sample from
    # another: an ext<N> extension and the final -v<N>
    rest = processed[len(campaign) + 1:]                  # 106X_..._ext1-v3
    ext = [t for t in rest.replace('-', '_').split('_') if t.startswith('ext')]
    ver = [t for t in rest.split('-') if t.startswith('v') and t[1:].isdigit()]

    parts = ['dimuon_mc', short, tag] + ext + ver
    return '_'.join(p for p in parts if p)[:100]


# ----------------------------------------------------------------------------
# config builder
# ----------------------------------------------------------------------------
def create_config(dataset):
    cmssw_base = os.environ.get('CMSSW_BASE', '')
    if not cmssw_base:
        raise RuntimeError('CMSSW_BASE is not set -- run `cmsenv` first.')

    mm_dir = os.path.join(cmssw_base, 'src', 'Bmmm', 'Analysis', 'test', 'mm')
    helpers = sorted(set(
        glob.glob(os.path.join(mm_dir, '*.py')) +
        glob.glob(os.path.join(mm_dir, '*.h'))
    ))
    if not any(f.endswith('inspector_mm_analysis.py') for f in helpers):
        raise RuntimeError('inspector_mm_analysis.py not found under %s' % mm_dir)

    package_dir = os.path.join(cmssw_base, 'src', 'Bmmm', 'Analysis', 'python')
    if not os.path.isdir(package_dir):
        raise RuntimeError('%s not found' % package_dir)

    l1menus_dir = os.path.join(cmssw_base, 'src', 'Bmmm', 'Analysis', 'data', 'l1menus')
    if not os.path.isdir(l1menus_dir):
        raise RuntimeError('%s not found' % l1menus_dir)

    here       = os.path.dirname(os.path.abspath(__file__))
    pylibs_dir = os.path.join(here, 'pylibs')
    if not os.path.isdir(pylibs_dir):
        raise RuntimeError(
            'pylibs/ not found under %s -- install the non-CMSSW packages first:\n'
            '  cd %s\n'
            '  PYTHONNOUSERSITE=1 pip3 install --no-cache-dir --target=pylibs '
            'particle uproot' % (here, here))

    request = mc_request_name(dataset)

    cfg = config()

    cfg.General.requestName     = request
    cfg.General.workArea        = work_area
    cfg.General.transferOutputs = True
    cfg.General.transferLogs    = True

    cfg.JobType.pluginName = 'Analysis'
    cfg.JobType.psetName   = 'PSet.py'
    cfg.JobType.scriptExe  = 'crab_script.sh'
    # DIFFERENT from the data submitter: the inspector needs --mc (gen matching)
    # and --savenontrig (keep events that did not fire the HLT path). These are
    # appended after the job id and parsed by the KEY=value loop in
    # crab_script.sh; a data job passes neither and is unaffected.
    cfg.JobType.scriptArgs = ['MC=1', 'SAVENONTRIG=1']
    cfg.JobType.inputFiles = helpers + [package_dir, l1menus_dir, pylibs_dir,
                                        'FrameworkJobReport.xml']
    cfg.JobType.outputFiles = ['dimuon_ntuple.root']
    cfg.JobType.disableAutomaticOutputCollection = True
    cfg.JobType.maxMemoryMB = 3000
    # MINIAODSIM files are generally bigger than the parked data ones and
    # --savenontrig keeps every event, so a job does more work per file here.
    # Raise this if the costing task times out.
    # cfg.JobType.maxJobRuntimeMin = 1440

    cfg.Data.inputDataset = dataset
    cfg.Data.inputDBS     = 'global'     # central production; 'phys03' if private
    cfg.Data.splitting    = 'FileBased'
    cfg.Data.unitsPerJob  = files_per_job
    if max_files is not None:
        cfg.Data.totalUnits = max_files
    cfg.Data.outLFNDirBase    = '/store/user/manzoni/' + out_dir
    cfg.Data.publication      = False
    cfg.Data.outputDatasetTag = out_dir

    cfg.Site.storageSite = storage_site
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
    # Guard against running the wrong submitter: this file is for Run 2 2018 UL MC ONLY.
    # Every dataset must be a RunIISummer20UL18 MINIAODSIM, or nothing is submitted.
    not_ul18 = [d for d in productions
                if not (d.endswith('/MINIAODSIM') and
                        d.split('/')[2].startswith('RunIISummer20UL18'))]
    if not_ul18:
        raise RuntimeError('crab_dimuon_mc_ul2018.py is for Run 2 2018 UL MC '
                           '(RunIISummer20UL18 MINIAODSIM) only; these are not: %s' % not_ul18)

    for dataset in productions:
        cfg = create_config(dataset)

        if cfg.General.requestName in already_submitted:
            print('skipping (already submitted): %s' % cfg.General.requestName)
            continue

        print('%s  ->  %s' % (dataset, cfg.General.requestName))
        if max_files is not None:
            print('   COSTING RUN: capped at %d input file(s)' % max_files)

        p = Process(target=submit, args=(cfg,))
        p.start()
        p.join()
