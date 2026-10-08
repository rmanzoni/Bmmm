'''
CRAB3 submitter: J/psi + track ntuples from the Run 2 2018 UL MINIAODSIM that
goes with the 2018 B-parking data, starting with the B+ -> J/psi K+ sample

    /BuToJpsiK_BMuonFilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/
     RunIISummer20UL18MiniAODv2-BPH_106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM

The MC twin of crab_jpsi_tk_bparking2018.py: same inspector
(inspector_jpsi_tk_bparking2018.py, OR of the HLT_Mu*_IP* parking paths), same
crab_script.sh, same sandbox. What differs is forced by the input being
simulation, as in test/mm/crab/crab_dimuon_mc_ul2018.py:

  1. MC=1 SAVENONTRIG=1 through JobType.scriptArgs -> --mc --savenontrig.
     --mc turns on the gen matching (the JpsiTkInspector gen match assumes the
     B+ -> J/psi K+ topology, i.e. exactly this sample); --savenontrig keeps
     events that fired NO parking path, so the trigger efficiency can be
     measured. The path decisions, prescales and mu{1,2}_<path>_tag are written
     either way: select on them offline to reproduce the data selection.

  2. No certification: no Data.lumiMask, no golden_jsons in the sandbox, no
     LUMIMASK argument (crab_script.sh stops the job if one is given with MC=1).

  3. No pileup weights: the card (data/pu_weights_run3.json) has Run 3
     campaigns only, so --pu is not passed and the pu_weight_* branches stay
     NaN. The true number of interactions is in the `nti` branch, so the
     weights can be added offline once a UL2018 profile is decided.

  4. inputDBS 'global' (central production); 'phys03' if ever pointed at a
     private sample.

Setup (tcsh, order matters):
    cmsenv ; scram b -j 8
    source /cvmfs/cms.cern.ch/crab3/crab.csh
    voms-proxy-init -rfc -voms cms -valid 192:00
    ./make_pylibs.sh                        # once, shared with the data submitter
    python3 crab_jpsi_tk_bparking2018_mc.py

Do a costing run first (max_files = 5): MINIAODSIM files are larger than the
data ones and --savenontrig keeps every event.
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
work_area     = 'crab_jpsi_tk_bparking2018_mc_08oct2026'
out_dir       = 'jpsi_tk_ntuples_mc_ul2018_08oct2026'     # under /store/user/manzoni/
files_per_job = 1
storage_site  = 'T3_CH_PSI'

# Cap the number of input files, for a costing run. None submits the whole dataset.
max_files = None

# requestNames listed here are skipped (already submitted / done)
already_submitted = [
]

productions = [
    '/BuToJpsiK_BMuonFilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/RunIISummer20UL18MiniAODv2-BPH_106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM',
]


# ----------------------------------------------------------------------------
# request name from an MC dataset name (same scheme as crab_dimuon_mc_ul2018.py)
# ----------------------------------------------------------------------------
def mc_request_name(dataset):
    '''/BuToJpsiK_BMuonFilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen
       /RunIISummer20UL18MiniAODv2-BPH_106X_upgrade2018_realistic_v16_L1v1-v2
       /MINIAODSIM
       -> jpsitk_mc_BuToJpsiK_BMuonFilter_SoftQCDnonD_UL18_v2
    '''
    primary, processed = dataset.split('/')[1], dataset.split('/')[2]
    short    = primary.split('_Tune')[0]
    campaign = processed.split('-')[0]                     # RunIISummer20UL18MiniAODv2
    tag      = (campaign.replace('RunIISummer20', '')
                        .replace('MiniAODv2', '')
                        .replace('MiniAOD', ''))           # UL18
    rest = processed[len(campaign) + 1:]                   # BPH_106X_..._L1v1-v2
    ext  = [t for t in rest.replace('-', '_').split('_') if t.startswith('ext')]
    ver  = [t for t in rest.split('-') if t.startswith('v') and t[1:].isdigit()]
    parts = ['jpsitk_mc', short, tag] + ext + ver
    return '_'.join(p for p in parts if p)[:100]


# ----------------------------------------------------------------------------
# config builder
# ----------------------------------------------------------------------------
def create_config(dataset):
    cmssw_base = os.environ.get('CMSSW_BASE', '')
    if not cmssw_base:
        raise RuntimeError('CMSSW_BASE is not set -- run `cmsenv` first.')

    test_dir = os.path.join(cmssw_base, 'src', 'Bmmm', 'Analysis', 'test', 'bparking2018')
    helpers  = sorted(glob.glob(os.path.join(test_dir, '*.py')))
    if not any(f.endswith('inspector_jpsi_tk_bparking2018.py') for f in helpers):
        raise RuntimeError('inspector_jpsi_tk_bparking2018.py not found under %s' % test_dir)

    package_dir = os.path.join(cmssw_base, 'src', 'Bmmm', 'Analysis', 'python')
    if not os.path.isfile(os.path.join(package_dir, 'JpsiTkBParking2018Inspector.py')):
        raise RuntimeError('%s has no JpsiTkBParking2018Inspector.py -- is this '
                           'CMSSW area on the bparking2018_jpsi_tk branch?' % package_dir)

    here       = os.path.dirname(os.path.abspath(__file__))
    pylibs_dir = os.path.join(here, 'pylibs')
    if not os.path.isdir(pylibs_dir):
        raise RuntimeError('pylibs/ not found under %s -- run ./make_pylibs.sh there first' % here)

    cfg = config()

    cfg.General.requestName     = mc_request_name(dataset)
    cfg.General.workArea        = work_area
    cfg.General.transferOutputs = True
    cfg.General.transferLogs    = True

    cfg.JobType.pluginName = 'Analysis'
    cfg.JobType.psetName   = 'PSet.py'
    cfg.JobType.scriptExe  = 'crab_script.sh'
    cfg.JobType.scriptArgs = ['MC=1', 'SAVENONTRIG=1']
    cfg.JobType.inputFiles = helpers + [package_dir, pylibs_dir, 'FrameworkJobReport.xml']
    cfg.JobType.outputFiles = ['jpsi_tk_ntuple.root']
    cfg.JobType.disableAutomaticOutputCollection = True
    cfg.JobType.maxMemoryMB = 3000
    # cfg.JobType.maxJobRuntimeMin = 1440   # raise if the costing run times out

    cfg.Data.inputDataset     = dataset
    cfg.Data.inputDBS         = 'global'      # central production; 'phys03' if private
    cfg.Data.splitting        = 'FileBased'
    cfg.Data.unitsPerJob      = files_per_job
    if max_files is not None:
        cfg.Data.totalUnits   = max_files
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
    # guard: Run 2 2018 UL MINIAODSIM only, or nothing is submitted
    not_ul18 = [d for d in productions
                if not (d.endswith('/MINIAODSIM') and
                        d.split('/')[2].startswith('RunIISummer20UL18'))]
    if not_ul18:
        raise RuntimeError('crab_jpsi_tk_bparking2018_mc.py is for 2018 UL MC '
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
