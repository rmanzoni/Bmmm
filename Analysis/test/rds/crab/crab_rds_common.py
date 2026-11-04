'''
Shared pieces of the R(Ds) CRAB submitters (crab_rds_data.py, crab_rds_mc.py).

Same scriptExe mechanism as test/bparking2018/crab (Vcb parking ntuples): CRAB
runs crab_script.sh instead of cmsRun, the package is shipped as ./python and
rebuilt in the job, non-CMSSW python packages come from ./pylibs.

What travels in the sandbox (JobType.inputFiles):
  inspector_rds.py                 the entry point
  python/                          Bmmm/Analysis/python
  pylibs/                          ./make_pylibs.sh (particle)
  l1menus/                         run -> L1 menu map and the four 2018 menus
  l1_seed_prescales_bparking2018.pickle.gz
                                   the ten brilcalc L1 tables merged, written
                                   here at submission (0.8 MB instead of 120 MB)
  golden_jsons/                    data only: the certification JSONs
  trkcalib/                        MC with --trk-calib: momcal applier + JSON
  <cov-scale table>                MC with --cov-scale
  <hammer card>                    MC with --hammer
crab_script.sh sets BMMM_DATADIR to the job directory, where L1Seeds2018 and
TrackMomentumCalib look for these.

The C++ library (libBmmmAnalysis, RDsKinVtxFitter) comes with the CMSSW area
CRAB ships, so build (scram b) before submitting.
'''

import os
import sys
import glob
import json
import time
import hashlib

from CRABClient.UserUtilities import config

# pure python, importable in the CRAB client environment (no ROOT, no particle)
from Bmmm.Analysis.L1Seeds2018 import write_merged, MERGED_NAME

# the ten parking paths, as in Bmmm.Analysis.BParking2018Trigger (not imported:
# that module pulls in ROOT and the particle package, absent from the client)
BPARKING2018_PATH_NAMES = (
    'HLT_Mu7_IP4', 'HLT_Mu8_IP3', 'HLT_Mu8_IP5', 'HLT_Mu8_IP6', 'HLT_Mu8p5_IP3p5',
    'HLT_Mu9_IP4', 'HLT_Mu9_IP5', 'HLT_Mu9_IP6', 'HLT_Mu10p5_IP3p5', 'HLT_Mu12_IP6',
)

HERE = os.path.dirname(os.path.abspath(__file__))


def cmssw_paths():
    base = os.environ.get('CMSSW_BASE', '')
    if not base:
        raise RuntimeError('CMSSW_BASE is not set -- run `cmsenv` first.')
    ana = os.path.join(base, 'src', 'Bmmm', 'Analysis')
    return base, ana


def common_inputs():
    '''Files every R(Ds) job needs, checked here so a broken sandbox is
    caught at submission rather than on the grid.'''
    _, ana = cmssw_paths()
    entry = os.path.join(ana, 'test', 'rds', 'inspector_rds.py')
    if not os.path.isfile(entry):
        raise RuntimeError('%s not found' % entry)
    package_dir = os.path.join(ana, 'python')
    if not os.path.isfile(os.path.join(package_dir, 'RDsInspector.py')):
        raise RuntimeError('%s has no RDsInspector.py -- is this CMSSW area on the '
                           'rds_ntuplizer branch?' % package_dir)
    pylibs_dir = os.path.join(HERE, 'pylibs')
    if not os.path.isdir(pylibs_dir):
        raise RuntimeError('pylibs/ not found under %s -- run ./make_pylibs.sh there first' % HERE)
    l1menus_dir = os.path.join(ana, 'data', 'l1menus')
    if not os.path.isdir(l1menus_dir):
        raise RuntimeError('%s not found' % l1menus_dir)
    merged = os.path.join(HERE, MERGED_NAME)
    if not os.path.isfile(merged):
        print('writing the merged L1 prescale table %s' % merged)
        write_merged(list(BPARKING2018_PATH_NAMES), os.path.join(ana, 'data'), merged)
    return [entry, package_dir, pylibs_dir, l1menus_dir, merged,
            os.path.join(HERE, 'FrameworkJobReport.xml'),
            os.path.join(HERE, 'provenance.json')]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_provenance(allow_unpushed, extra):
    '''Code provenance at submission (git commit, pushed, uncommitted changes,
    diff sha1) + `extra` (dataset, options, sha256 of the correction files).
    Refuses dirty or unpushed code unless allow_unpushed. Writes
    ./provenance.json (shipped in the sandbox, printed by every job) and
    returns the dict; the diff, if any, goes to ./uncommitted.diff.'''
    _, ana = cmssw_paths()
    sys.path.insert(0, os.path.join(ana, 'python'))
    from Provenance import provenance
    prov = provenance(with_diff=True)
    if (prov['dirty'] or not prov['pushed']) and not allow_unpushed:
        raise RuntimeError('uncommitted changes %s or commit %s not pushed: commit and push '
                           'first (or set allow_unpushed)' % (prov['dirty'], prov['commit']))
    diff = prov.pop('diff')
    prov.update(extra)
    prov['submitted'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    prov['cmssw_base'] = os.environ.get('CMSSW_BASE')
    prov['scram_arch'] = os.environ.get('SCRAM_ARCH')
    with open(os.path.join(HERE, 'provenance.json'), 'w') as fout:
        json.dump(prov, fout, indent=1)
    if diff:
        with open(os.path.join(HERE, 'uncommitted.diff'), 'w') as fout:
            fout.write(diff)
    print('provenance: commit %s (%s), pushed %s, dirty %s'
          % (prov['commit'][:10], prov['branch'], prov['pushed'], bool(prov['dirty'])))
    return prov


def base_config(request_name, work_area, dataset, files_per_job, max_files,
                out_lfn_dir, output_tag, storage_site, inputs, script_args):
    cfg = config()
    cfg.General.requestName     = request_name[:100]
    cfg.General.workArea        = work_area
    cfg.General.transferOutputs = True
    cfg.General.transferLogs    = True

    cfg.JobType.pluginName  = 'Analysis'
    cfg.JobType.psetName    = 'PSet.py'
    cfg.JobType.scriptExe   = 'crab_script.sh'
    cfg.JobType.inputFiles  = inputs
    cfg.JobType.scriptArgs  = script_args
    cfg.JobType.outputFiles = ['rds_ntuple.root']
    cfg.JobType.disableAutomaticOutputCollection = True
    # SLURM signal jobs (4 schemes): 3.1 GB median, 3.9 GB max RSS
    cfg.JobType.maxMemoryMB = 4000
    cfg.JobType.maxJobRuntimeMin = 1315     # CRAB default, ~22 h; jobs sized to ~6 h

    cfg.Data.inputDataset     = dataset
    cfg.Data.inputDBS         = 'global'
    cfg.Data.splitting        = 'FileBased'
    cfg.Data.unitsPerJob      = files_per_job
    if max_files is not None:
        cfg.Data.totalUnits   = max_files
    cfg.Data.outLFNDirBase    = out_lfn_dir
    cfg.Data.publication      = False
    cfg.Data.outputDatasetTag = output_tag

    cfg.Site.storageSite = storage_site
    return cfg
