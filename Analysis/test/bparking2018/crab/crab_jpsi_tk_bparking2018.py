'''
CRAB3 submitter: J/psi + track ntuples (B+ -> J/psi K+, Bc+ -> J/psi pi+) from
the 2018 B-parking UL MiniAODv2 data, ParkingBPH1-6, eras A-D.

Same sandbox / scriptExe / package-bootstrap mechanism as
test/mm/crab/crab_dimuon_data_run3.py -- see that file for why it is shaped the
way it is. What is specific to this one:

  * the job runs inspector_jpsi_tk_bparking2018.py (test/bparking2018), which
    requires an OR of the 2018 parking paths HLT_Mu*_IP* (see
    Bmmm.Analysis.JpsiTkBParking2018Inspector);
  * every dataset must be /ParkingBPH<N>/Run2018<era>-UL2018_MiniAODv2-v1/MINIAOD,
    or nothing is submitted;
  * the 2018 certification JSON must be in data/golden_jsons; the official
    name Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt is fine
    (LumiMask reads 'Collisions18' as 2018). The submitter stops with
    instructions if it is not there.

Certification, two layers (as in the Run 3 data submitter):
  - Data.lumiMask = the 2018 JSON: CRAB drops the files -- and jobs -- that hold
    no certified lumi. It does NOT filter events, because the job reads the
    PSet fileNames only.
  - the golden_jsons/ directory travels in the sandbox and LUMIMASK=golden_jsons
    makes crab_script.sh pass --lumi-mask to the inspector, which rejects every
    uncertified event. Run numbers are unique across years, so the union of the
    per-year JSONs in that directory is the right mask for 2018 runs.

Required files in this directory:
    crab_script.sh   PSet.py   FrameworkJobReport.xml   pylibs/  (./make_pylibs.sh)

Setup (tcsh, order matters):
    cmsenv                                  # CMSSW area where Bmmm is built
    scram b -j 8                            # picks up the new package module
    source /cvmfs/cms.cern.ch/crab3/crab.csh
    voms-proxy-init -rfc -voms cms -valid 192:00
    ./make_pylibs.sh                        # once
    python3 crab_jpsi_tk_bparking2018.py

Submit ONE dataset with max_files set first: that is how you find out what a
job costs (runtime, memory, output size) before launching 22 tasks.

Afterwards:
    crab status   -d <work_area>/crab_<requestName>
    crab getlog --short -d <work_area>/crab_<requestName>
    crab resubmit -d <work_area>/crab_<requestName>
    crab report   -d <work_area>/crab_<requestName>
'''

import os
import re
import glob
from multiprocessing import Process

from CRABClient.UserUtilities import config
from CRABAPI.RawCommand import crabCommand
from CRABClient.ClientExceptions import ClientException
from http.client import HTTPException

# the SAME golden-JSON lookup the inspector uses (pure python, no ROOT)
from Bmmm.Analysis.LumiMask import golden_jsons_by_year


# ----------------------------------------------------------------------------
# user knobs
# ----------------------------------------------------------------------------
work_area     = 'crab_jpsi_tk_bparking2018_08oct2026'
out_dir       = 'jpsi_tk_ntuples_bparking2018_08oct2026'   # under /store/user/manzoni/
files_per_job = 1
storage_site  = 'T3_CH_PSI'

# Cap the number of input files per task, for a costing run. None submits the
# whole dataset. Set e.g. 5 for the first task, check runtime and output size.
max_files = None

# requestNames listed here are skipped (already submitted / done)
already_submitted = [
]

productions = [
    '/ParkingBPH1/Run2018A-UL2018_MiniAODv2-v1/MINIAOD',
    '/ParkingBPH1/Run2018B-UL2018_MiniAODv2-v1/MINIAOD',
    '/ParkingBPH1/Run2018C-UL2018_MiniAODv2-v1/MINIAOD',
    '/ParkingBPH1/Run2018D-UL2018_MiniAODv2-v1/MINIAOD',

    '/ParkingBPH2/Run2018A-UL2018_MiniAODv2-v1/MINIAOD',
    '/ParkingBPH2/Run2018B-UL2018_MiniAODv2-v1/MINIAOD',
    '/ParkingBPH2/Run2018C-UL2018_MiniAODv2-v1/MINIAOD',
    '/ParkingBPH2/Run2018D-UL2018_MiniAODv2-v1/MINIAOD',

    '/ParkingBPH3/Run2018A-UL2018_MiniAODv2-v1/MINIAOD',
    '/ParkingBPH3/Run2018B-UL2018_MiniAODv2-v1/MINIAOD',
    '/ParkingBPH3/Run2018C-UL2018_MiniAODv2-v1/MINIAOD',
    '/ParkingBPH3/Run2018D-UL2018_MiniAODv2-v1/MINIAOD',

    '/ParkingBPH4/Run2018A-UL2018_MiniAODv2-v1/MINIAOD',
    '/ParkingBPH4/Run2018B-UL2018_MiniAODv2-v1/MINIAOD',
    '/ParkingBPH4/Run2018C-UL2018_MiniAODv2-v1/MINIAOD',
    '/ParkingBPH4/Run2018D-UL2018_MiniAODv2-v1/MINIAOD',

    '/ParkingBPH5/Run2018A-UL2018_MiniAODv2-v1/MINIAOD',
    '/ParkingBPH5/Run2018B-UL2018_MiniAODv2-v1/MINIAOD',
    '/ParkingBPH5/Run2018C-UL2018_MiniAODv2-v1/MINIAOD',
    '/ParkingBPH5/Run2018D-UL2018_MiniAODv2-v1/MINIAOD',

    '/ParkingBPH6/Run2018A-UL2018_MiniAODv2-v1/MINIAOD',
    '/ParkingBPH6/Run2018B-UL2018_MiniAODv2-v1/MINIAOD',
]

_DATASET_RE = re.compile(r'^/ParkingBPH([1-6])/Run2018([A-D])-(UL2018_MiniAODv2-v\d+)/MINIAOD$')


# ----------------------------------------------------------------------------
# request name
# ----------------------------------------------------------------------------
def request_name(dataset):
    '''/ParkingBPH1/Run2018A-UL2018_MiniAODv2-v1/MINIAOD
       -> jpsitk_BPH1_Run2018A_UL2018_MiniAODv2_v1'''
    part, era, proc = _DATASET_RE.match(dataset).groups()
    return ('jpsitk_BPH%s_Run2018%s_%s' % (part, era, proc.replace('-', '_')))[:100]


# ----------------------------------------------------------------------------
# config builder
# ----------------------------------------------------------------------------
def create_config(dataset):
    cmssw_base = os.environ.get('CMSSW_BASE', '')
    if not cmssw_base:
        raise RuntimeError('CMSSW_BASE is not set -- run `cmsenv` first.')

    # the entry point (and any sibling module) from test/bparking2018, shipped
    # flat into the job directory
    test_dir = os.path.join(cmssw_base, 'src', 'Bmmm', 'Analysis', 'test', 'bparking2018')
    helpers  = sorted(glob.glob(os.path.join(test_dir, '*.py')))
    if not any(f.endswith('inspector_jpsi_tk_bparking2018.py') for f in helpers):
        raise RuntimeError('inspector_jpsi_tk_bparking2018.py not found under %s' % test_dir)

    # the package itself: crab_script.sh rebuilds Bmmm/Analysis from ./python
    package_dir = os.path.join(cmssw_base, 'src', 'Bmmm', 'Analysis', 'python')
    if not os.path.isfile(os.path.join(package_dir, 'JpsiTkBParking2018Inspector.py')):
        raise RuntimeError('%s has no JpsiTkBParking2018Inspector.py -- is this '
                           'CMSSW area on the bparking2018_jpsi_tk branch?' % package_dir)

    # certification: the whole directory travels (inspector mask), the 2018
    # file alone goes to Data.lumiMask (file selection)
    golden_dir = os.path.join(cmssw_base, 'src', 'Bmmm', 'Analysis', 'data', 'golden_jsons')
    golden     = golden_jsons_by_year(golden_dir)
    if '2018' not in golden:
        raise RuntimeError(
            'no 2018 certification JSON in %s (have: %s).\n'
            'Copy the legacy 2018 golden JSON there (official name is fine):\n'
            '  cp Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt %s/'
            % (golden_dir, ', '.join(sorted(golden)), golden_dir))

    # non-CMSSW python packages (particle), built by make_pylibs.sh
    here       = os.path.dirname(os.path.abspath(__file__))
    pylibs_dir = os.path.join(here, 'pylibs')
    if not os.path.isdir(pylibs_dir):
        raise RuntimeError('pylibs/ not found under %s -- run ./make_pylibs.sh there first' % here)

    part, era, proc = _DATASET_RE.match(dataset).groups()
    ver_clean = proc.replace('-', '_')                     # UL2018_MiniAODv2_v1

    cfg = config()

    cfg.General.requestName     = request_name(dataset)
    cfg.General.workArea        = work_area
    cfg.General.transferOutputs = True
    cfg.General.transferLogs    = True

    cfg.JobType.pluginName = 'Analysis'
    cfg.JobType.psetName   = 'PSet.py'
    cfg.JobType.scriptExe  = 'crab_script.sh'
    cfg.JobType.inputFiles = helpers + [package_dir, golden_dir, pylibs_dir,
                                        'FrameworkJobReport.xml']
    cfg.JobType.scriptArgs = ['LUMIMASK=golden_jsons']
    cfg.JobType.outputFiles = ['jpsi_tk_ntuple.root']
    # plain ROOT output, not EDM: no harvesting from a (non-existent) cmsRun report
    cfg.JobType.disableAutomaticOutputCollection = True
    cfg.JobType.maxMemoryMB = 3000
    # cfg.JobType.maxJobRuntimeMin = 1440   # raise if the costing run times out

    cfg.Data.inputDataset     = dataset
    cfg.Data.inputDBS         = 'global'
    cfg.Data.lumiMask         = golden['2018']
    cfg.Data.splitting        = 'FileBased'
    cfg.Data.unitsPerJob      = files_per_job
    if max_files is not None:
        cfg.Data.totalUnits   = max_files
    # CRAB appends /<primary dataset>/<outputDatasetTag>/<timestamp>/, so the
    # ParkingBPH<N> parts land in separate directories under each era
    cfg.Data.outLFNDirBase    = '/store/user/manzoni/%s/Run2018%s' % (out_dir, era)
    cfg.Data.publication      = False
    cfg.Data.outputDatasetTag = out_dir + '_' + ver_clean

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
    # guard: 2018 B-parking UL data ONLY, or nothing is submitted
    wrong = [d for d in productions if not _DATASET_RE.match(d)]
    if wrong:
        raise RuntimeError('crab_jpsi_tk_bparking2018.py is for '
                           '/ParkingBPH<1-6>/Run2018<A-D>-UL2018_MiniAODv2-vN/MINIAOD '
                           'only; these are not: %s' % wrong)
    names = [request_name(d) for d in productions]
    if len(names) != len(set(names)):
        raise RuntimeError('two datasets map to the same requestName: %s' % names)

    for dataset in productions:
        cfg = create_config(dataset)

        if cfg.General.requestName in already_submitted:
            print('skipping (already submitted): %s' % cfg.General.requestName)
            continue

        print('%s  ->  %s   (lumiMask %s)' % (dataset, cfg.General.requestName,
                                              os.path.basename(cfg.Data.lumiMask)))
        if max_files is not None:
            print('   COSTING RUN: capped at %d input file(s)' % max_files)

        p = Process(target=submit, args=(cfg,))
        p.start()
        p.join()
