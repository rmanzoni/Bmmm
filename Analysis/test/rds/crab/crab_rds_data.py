'''
CRAB3 submitter: R(Ds) ntuples from the 2018 B-parking UL MiniAODv2 data,
ParkingBPH1-6, eras A-D (22 datasets, as the Vcb parking ntuples).

Certification in two layers, as in test/bparking2018/crab:
  - Data.lumiMask = the 2018 golden JSON: CRAB drops the files without a
    certified lumi;
  - golden_jsons/ travels in the sandbox and the inspector rejects every
    uncertified event (--lumi-mask).

Setup (tcsh, order matters):
    cmsenv ; scram b -j 8
    source /cvmfs/cms.cern.ch/crab3/crab.csh
    voms-proxy-init -rfc -voms cms -valid 192:00
    ./make_pylibs.sh                       # once
    python3 crab_rds_data.py

Costing run first: keep one dataset in `productions`, set max_files = 5, check
runtime / memory / output size in the job logs (the inspector prints ev/s),
then set files_per_job and submit the rest.
'''

import os
import re
from multiprocessing import Process

from CRABAPI.RawCommand import crabCommand
from CRABClient.ClientExceptions import ClientException
from http.client import HTTPException

from Bmmm.Analysis.LumiMask import golden_jsons_by_year
from crab_rds_common import cmssw_paths, common_inputs, base_config

# ----------------------------------------------------------------------------
# user knobs
# ----------------------------------------------------------------------------
work_area     = 'crab_rds_bparking2018_29oct2026'
out_dir       = 'rds_ntuples_bparking2018_29oct2026'     # under /store/user/manzoni/
files_per_job = 1
storage_site  = 'T3_CH_PSI'
max_files     = 5        # COSTING RUN; None for the whole dataset

already_submitted = [
]

productions = [
    '/ParkingBPH1/Run2018D-UL2018_MiniAODv2-v1/MINIAOD',      # costing run
#     '/ParkingBPH1/Run2018A-UL2018_MiniAODv2-v1/MINIAOD',
#     '/ParkingBPH1/Run2018B-UL2018_MiniAODv2-v1/MINIAOD',
#     '/ParkingBPH1/Run2018C-UL2018_MiniAODv2-v1/MINIAOD',
#     '/ParkingBPH2/Run2018A-UL2018_MiniAODv2-v1/MINIAOD',
#     '/ParkingBPH2/Run2018B-UL2018_MiniAODv2-v1/MINIAOD',
#     '/ParkingBPH2/Run2018C-UL2018_MiniAODv2-v1/MINIAOD',
#     '/ParkingBPH2/Run2018D-UL2018_MiniAODv2-v1/MINIAOD',
#     '/ParkingBPH3/Run2018A-UL2018_MiniAODv2-v1/MINIAOD',
#     '/ParkingBPH3/Run2018B-UL2018_MiniAODv2-v1/MINIAOD',
#     '/ParkingBPH3/Run2018C-UL2018_MiniAODv2-v1/MINIAOD',
#     '/ParkingBPH3/Run2018D-UL2018_MiniAODv2-v1/MINIAOD',
#     '/ParkingBPH4/Run2018A-UL2018_MiniAODv2-v1/MINIAOD',
#     '/ParkingBPH4/Run2018B-UL2018_MiniAODv2-v1/MINIAOD',
#     '/ParkingBPH4/Run2018C-UL2018_MiniAODv2-v1/MINIAOD',
#     '/ParkingBPH4/Run2018D-UL2018_MiniAODv2-v1/MINIAOD',
#     '/ParkingBPH5/Run2018A-UL2018_MiniAODv2-v1/MINIAOD',
#     '/ParkingBPH5/Run2018B-UL2018_MiniAODv2-v1/MINIAOD',
#     '/ParkingBPH5/Run2018C-UL2018_MiniAODv2-v1/MINIAOD',
#     '/ParkingBPH5/Run2018D-UL2018_MiniAODv2-v1/MINIAOD',
#     '/ParkingBPH6/Run2018A-UL2018_MiniAODv2-v1/MINIAOD',
#     '/ParkingBPH6/Run2018B-UL2018_MiniAODv2-v1/MINIAOD',
]

_DATASET_RE = re.compile(r'^/ParkingBPH([1-6])/Run2018([A-D])-(UL2018_MiniAODv2-v\d+)/MINIAOD$')


def request_name(dataset):
    part, era, proc = _DATASET_RE.match(dataset).groups()
    return 'rds_BPH%s_Run2018%s_%s' % (part, era, proc.replace('-', '_'))


def create_config(dataset):
    _, ana = cmssw_paths()
    golden_dir = os.path.join(ana, 'data', 'golden_jsons')
    golden = golden_jsons_by_year(golden_dir)
    if '2018' not in golden:
        raise RuntimeError('no 2018 certification JSON in %s' % golden_dir)

    part, era, proc = _DATASET_RE.match(dataset).groups()
    cfg = base_config(
        request_name  = request_name(dataset),
        work_area     = work_area,
        dataset       = dataset,
        files_per_job = files_per_job,
        max_files     = max_files,
        out_lfn_dir   = '/store/user/manzoni/%s/Run2018%s' % (out_dir, era),
        output_tag    = out_dir + '_' + proc.replace('-', '_'),
        storage_site  = storage_site,
        inputs        = common_inputs() + [golden_dir],
        script_args   = ['LUMIMASK=golden_jsons'],
    )
    cfg.Data.lumiMask = golden['2018']
    return cfg


def submit(cfg):
    try:
        crabCommand('submit', config=cfg)
    except HTTPException as hte:
        print('failed submitting %s: %s' % (cfg.General.requestName, hte.headers))
    except ClientException as cle:
        print('failed submitting %s: %s' % (cfg.General.requestName, cle))


if __name__ == '__main__':
    wrong = [d for d in productions if not _DATASET_RE.match(d)]
    if wrong:
        raise RuntimeError('only /ParkingBPH<1-6>/Run2018<A-D>-UL2018_MiniAODv2-vN/MINIAOD; '
                           'not: %s' % wrong)
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
