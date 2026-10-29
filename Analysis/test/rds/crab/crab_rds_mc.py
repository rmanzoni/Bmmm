'''
CRAB3 submitter: R(Ds) ntuples from the UL2018 MINIAODSIM (signal, Hb and the
unfiltered gen samples).

THE DATASET NAMES BELOW ARE PLACEHOLDERS: replace them with the DAS names of
the central requests (AN-24-173 refs. [31], [32]) before submitting. The
submitter refuses to run while a placeholder is left.

Per job: --mc --savenontrig (every event kept; trigger decisions, L1 seeds and
the muon matches stored), plus the production-time corrections set below:
  trk_calib   momcal correction (v4); copy rds_trkcalib.py and the JSON into
              Bmmm/Analysis/data/trkcalib/ (see the README there)
  cov_scale   sigma_dxy (or other) covariance scaling table, applied to the
              tracks and the muon; '' for none
  hammer      R(Ds) Hammer card; '' for none (weights can be added later with
              test/rds/hammer/add_hammer_weights_rds.py)
'''

import os
from multiprocessing import Process

from CRABAPI.RawCommand import crabCommand
from CRABClient.ClientExceptions import ClientException
from http.client import HTTPException

from crab_rds_common import cmssw_paths, common_inputs, base_config

# ----------------------------------------------------------------------------
# user knobs
# ----------------------------------------------------------------------------
work_area     = 'crab_rds_mc_ul2018_29oct2026'
out_dir       = 'rds_ntuples_mc_ul2018_29oct2026'
files_per_job = 1
storage_site  = 'T3_CH_PSI'
max_files     = 5        # COSTING RUN; None for the whole dataset

trk_calib = 'rds_trkcalib_2018BParking_v4.json'   # in data/trkcalib/; '' for none
cov_scale = ''                                    # path to the table; '' for none
hammer    = ''                                    # path to the card;  '' for none

already_submitted = [
]

# (short name, DAS name)
productions = [
    ('signal', 'PLACEHOLDER_SIGNAL_DATASET'),   # Bs -> Ds(*) mu/tau nu, AN ref. [31]
    ('hb',     'PLACEHOLDER_HB_DATASET'),       # inclusive Hb -> Ds + mu, AN ref. [32]
]


def create_config(short, dataset):
    _, ana = cmssw_paths()
    inputs = common_inputs()
    args = ['MC=1', 'SAVENONTRIG=1']

    if trk_calib:
        tdir = os.path.join(ana, 'data', 'trkcalib')
        for f in ('rds_trkcalib.py', trk_calib):
            if not os.path.isfile(os.path.join(tdir, f)):
                raise RuntimeError('%s not in %s (see the README there)' % (f, tdir))
        inputs.append(tdir)
        args.append('TRKCALIB=trkcalib/%s' % trk_calib)
    if cov_scale:
        if not os.path.isfile(cov_scale):
            raise RuntimeError('cov_scale table %s not found' % cov_scale)
        inputs.append(cov_scale)
        args.append('COVSCALE=%s' % os.path.basename(cov_scale))
    if hammer:
        if not os.path.isfile(hammer):
            raise RuntimeError('Hammer card %s not found' % hammer)
        inputs.append(hammer)
        args.append('HAMMER=%s' % os.path.basename(hammer))

    return base_config(
        request_name  = 'rds_mc_%s_%s' % (short, out_dir.split('_')[-1]),
        work_area     = work_area,
        dataset       = dataset,
        files_per_job = files_per_job,
        max_files     = max_files,
        out_lfn_dir   = '/store/user/manzoni/' + out_dir,
        output_tag    = '%s_%s' % (out_dir, short),
        storage_site  = storage_site,
        inputs        = inputs,
        script_args   = args,
    )


def submit(cfg):
    try:
        crabCommand('submit', config=cfg)
    except HTTPException as hte:
        print('failed submitting %s: %s' % (cfg.General.requestName, hte.headers))
    except ClientException as cle:
        print('failed submitting %s: %s' % (cfg.General.requestName, cle))


if __name__ == '__main__':
    placeholders = [d for _, d in productions if d.startswith('PLACEHOLDER')]
    if placeholders:
        raise RuntimeError('replace the placeholder dataset names first: %s' % placeholders)
    for short, dataset in productions:
        cfg = create_config(short, dataset)
        if cfg.General.requestName in already_submitted:
            print('skipping (already submitted): %s' % cfg.General.requestName)
            continue
        print('%s  ->  %s   (%s)' % (dataset, cfg.General.requestName, ' '.join(cfg.JobType.scriptArgs)))
        p = Process(target=submit, args=(cfg,))
        p.start()
        p.join()
