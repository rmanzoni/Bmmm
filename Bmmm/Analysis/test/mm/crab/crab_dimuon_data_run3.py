'''
CRAB3 submitter to run inspector_mm_analysis.py over Run 3 MINIAODSIM.

The MC twin of crab_dimuon_data_run3.py. Everything about the sandbox, the
scriptExe mechanism and the package bootstrap is identical -- see that file for
why any of it is shaped the way it is. What differs is forced by the input being
simulation:

  1. --mc --savenontrig on the inspector, passed through JobType.scriptArgs as
     MC=1 SAVENONTRIG=1 and picked up by the KEY=value loop in crab_script.sh.
     Without --mc there is no gen matching and the gen_* branches are all NaN;
     without --savenontrig the job keeps only events that fired the HLT path.

  2. No certification mask: no Data.lumiMask and no LUMIMASK script argument.
     The inspector refuses --lumi-mask together with --mc.

  3. requestName and the output tag are built from the MC campaign
     (Summer22, Summer22EE, Summer23, Summer23BPix, Summer24 -- the same keys
     as test/rjpsi/pileup/pu_config_run3.py), because one physics sample comes
     in several campaigns that are only usable separately (pileup profile,
     conditions, era matching to data).

  4. inputDBS stays 'global' -- true for a central production. If you ever point
     this at a privately produced sample, that has to become 'phys03'.

Output layout. CRAB writes to
    /store/user/manzoni/<out_dir>/<primary dataset>/<outputDatasetTag>/<timestamp>/<NNNN>/
Four of the samples share the primary dataset name, so outputDatasetTag carries
the campaign -- otherwise the campaigns would be told apart only by the CRAB
timestamp directory. One campaign is then one glob, e.g.
    /pnfs/psi.ch/cms/trivcat/store/user/manzoni/<out_dir>/*/<out_dir>_Summer22EE/*/*/*.root

Locality scheduling is left ON here as well, so each job runs where the data is.

Setup before running (order matters):
    cmsenv          # in the CMSSW src you built Bmmm in
    source /cvmfs/cms.cern.ch/crab3/crab.sh
    voms-proxy-init -rfc -voms cms -valid 192:00
    python3 crab_dimuon_mc_run3.py

Submit ONE task with max_files set before launching the campaign: that is how
you find out what a job costs, and MINIAODSIM files are typically larger than
the parked data ones this setup was tuned on.

Useful afterwards:
    crab status   -d crab_<work_area>/crab_<requestName>
    crab getlog --short -d crab_<work_area>/crab_<requestName>
    crab resubmit -d crab_<work_area>/crab_<requestName>
'''

import os
import re
import glob
from collections import Counter
from multiprocessing import Process

from CRABClient.UserUtilities import config
from CRABAPI.RawCommand import crabCommand
from CRABClient.ClientExceptions import ClientException
from http.client import HTTPException


# ----------------------------------------------------------------------------
# user knobs
# ----------------------------------------------------------------------------
work_area     = 'crab_dimuon_mc_run3'
out_dir       = 'dimuon_ntuples_mc_run3_02oct2026'   # under /store/user/manzoni/
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
    '/HbToJPsiMuMu_Fil-3Mu_TuneCP5_13p6TeV_pythia8-evtgen/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM',
    '/HbToJPsiMuMu_3MuFilter_TuneCP5_13p6TeV_pythia8-evtgen/Run3Summer22EEMiniAODv4-130X_mcRun3_2022_realistic_postEE_v6-v2/MINIAODSIM',
    '/HbToJPsiMuMu_3MuFilter_TuneCP5_13p6TeV_pythia8-evtgen/Run3Summer22MiniAODv4-130X_mcRun3_2022_realistic_v5-v2/MINIAODSIM',
    '/HbToJPsiMuMu_3MuFilter_TuneCP5_13p6TeV_pythia8-evtgen/Run3Summer23BPixMiniAODv4-130X_mcRun3_2023_realistic_postBPix_v6-v2/MINIAODSIM',
    '/HbToJPsiMuMu_3MuFilter_TuneCP5_13p6TeV_pythia8-evtgen/Run3Summer23MiniAODv4-130X_mcRun3_2023_realistic_v15-v2/MINIAODSIM',
    '/BcToJPsiMuMu_Fil-JPsi_TuneCP5_13p6TeV_bcvegpy2-pythia8-evtgen/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v6/MINIAODSIM',
    '/BcToJPsiMuMu_inclusive_TuneCP5_13p6TeV_bcvegpy2-pythia8-evtgen/Run3Summer22EEMiniAODv4-130X_mcRun3_2022_realistic_postEE_v6-v2/MINIAODSIM',
    '/BcToJPsiMuMu_inclusive_TuneCP5_13p6TeV_bcvegpy2-pythia8-evtgen/Run3Summer22MiniAODv4-130X_mcRun3_2022_realistic_v5-v2/MINIAODSIM',
    '/BcToJPsiMuMu_inclusive_TuneCP5_13p6TeV_bcvegpy2-pythia8-evtgen/Run3Summer23BPixMiniAODv4-130X_mcRun3_2023_realistic_postBPix_v6-v4/MINIAODSIM',
    '/BcToJPsiMuMu_inclusive_TuneCP5_13p6TeV_bcvegpy2-pythia8-evtgen/Run3Summer23MiniAODv4-Fixed_130X_mcRun3_2023_realistic_v15-v3/MINIAODSIM',
]


# ----------------------------------------------------------------------------
# campaign and request name from an MC dataset name
# ----------------------------------------------------------------------------
def mc_campaign(dataset):
    '''The MC campaign key, in the pu_config_run3.py spelling:

       Run3Summer22EEMiniAODv4-...      -> Summer22EE
       Run3Summer23BPixMiniAODv4-...    -> Summer23BPix
       RunIII2024Summer24MiniAODv6-...  -> Summer24
       RunIISummer20UL18MiniAODv2-...   -> UL18
    '''
    campaign = dataset.split('/')[2].split('-')[0]
    campaign = re.sub(r'MiniAOD(v\d+)?$', '', campaign)
    campaign = re.sub(r'^(RunIISummer20|RunIII\d{4}|Run3)', '', campaign)
    if not campaign:
        raise ValueError('cannot read the MC campaign of %s' % dataset)
    return campaign


def mc_request_name(dataset):
    '''/HbToJPsiMuMu_3MuFilter_TuneCP5_13p6TeV_pythia8-evtgen
       /Run3Summer22EEMiniAODv4-130X_mcRun3_2022_realistic_postEE_v6-v2
       /MINIAODSIM
       -> dimuon_mc_HbToJPsiMuMu_3MuFilter_Summer22EE_v2

    Kept short deliberately: CRAB caps requestName at 100 characters, and the
    full conditions string would blow past that on its own while telling you
    nothing you cannot read off the dataset.
    '''
    primary, processed = dataset.split('/')[1], dataset.split('/')[2]

    # drop the tune and everything after it: TuneCP5_13p6TeV_pythia8-evtgen is
    # the generator setup, not something that distinguishes one task from another
    short = primary.split('_Tune')[0]

    # the bits that separate one production of the same sample and campaign from
    # another: an ext<N> extension and the final -v<N>
    rest = processed.split('-', 1)[1]                     # 130X_..._v6-v2
    ext = [t for t in rest.replace('-', '_').split('_') if t.startswith('ext')]
    ver = [t for t in rest.split('-') if t.startswith('v') and t[1:].isdigit()]

    parts = ['dimuon_mc', short, mc_campaign(dataset)] + ext + ver
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
            'pylibs/ not found under %s -- build it first:\n'
            '  cd %s && ./make_pylibs.sh' % (here, here))

    campaign = mc_campaign(dataset)
    request  = mc_request_name(dataset)

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
    # crab_script.sh. No LUMIMASK: certification does not apply to simulation.
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
    # the campaign in the tag: four samples share the primary dataset name, and
    # without it their outputs would differ only by the CRAB timestamp directory
    cfg.Data.outputDatasetTag = '%s_%s' % (out_dir, campaign)

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
    # two productions of the same sample and campaign would get the same request
    # name and the same output tag: stop before submitting anything
    for what, keys in (('requestName', [mc_request_name(d) for d in productions]),
                       ('output tag',  ['%s/%s' % (d.split('/')[1], mc_campaign(d))
                                        for d in productions])):
        dup = [k for k, n in Counter(keys).items() if n > 1]
        if dup:
            raise RuntimeError('duplicate %s across productions: %s' % (what, dup))

    for dataset in productions:
        cfg = create_config(dataset)

        if cfg.General.requestName in already_submitted:
            print('skipping (already submitted): %s' % cfg.General.requestName)
            continue

        print('%s  ->  %s   [%s]' % (dataset, cfg.General.requestName,
                                     cfg.Data.outputDatasetTag))
        if max_files is not None:
            print('   COSTING RUN: capped at %d input file(s)' % max_files)

        p = Process(target=submit, args=(cfg,))
        p.start()
        p.join()