'''
CRAB3 submitter: R(Ds) ntuples from the UL2018 MINIAODSIM (signal; Hb when
its dataset is known).

Per job: --mc --savenontrig (every event kept; trigger decisions, L1 seeds and
the muon matches stored), plus the production-time corrections set below,
all under Bmmm/Analysis/data (index: data/README_corrections.md):
  trk_calib    momcal track+muon momentum scale/smearing; the directory is
               shipped whole (JSON + applier/python/rds_trkcalib.py)
  cov_scale    sigma_dxy quantile morph (correctionlib JSON)
  cov_muon     True: the morph is applied to the muon only
  trig_sf      trigger scale factors (map pt_vs_dxy_sig)
  hammer       R(Ds) Hammer card; '' for none
  fit_schemes  vertex-fit schemes computed and written (RDS_FIT_SCHEMES)
Pileup: offline, from nti.

Setup (bash):
    cmsenv ; scram b -j 8
    source /cvmfs/cms.cern.ch/common/crab-setup.sh
    ./make_pylibs.sh                       # once
    python3 crab_rds_mc.py

Submit from a clean CMSSW area (a fresh clone at a pushed commit): CRAB tars
every src/**/data directory, and the trkcalib caches of a working area are
far above the sandbox limit.
'''

import os
from multiprocessing import Process

from CRABAPI.RawCommand import crabCommand
from CRABClient.ClientExceptions import ClientException
from http.client import HTTPException

from crab_rds_common import cmssw_paths, common_inputs, base_config, write_provenance, sha256

# ----------------------------------------------------------------------------
# user knobs
# ----------------------------------------------------------------------------
tag           = 'free_04nov26'
work_area     = 'crab_rds_mc_' + tag
out_dir       = 'rds_mc_ul18_' + tag              # under /store/user/manzoni/
files_per_job = 8        # signal ~41 min/file median (p90 79) on the PSI WNs -> ~6 h
storage_site  = 'T3_CH_PSI'
max_files     = None     # None for the whole dataset
allow_unpushed = False

trk_calib   = 'trkcalib/RDs_2018BParking_trkcalib_v4b_base_laptop/rds_trkcalib_RDs_2018BParking_trkcalib_v4b_base_laptop.json'
cov_scale   = 'impact_parameter_quantile_morph/dxy_morphing/sigma_dxy_morph_splot_fixed_muonid_fixed_decay_length.json'
cov_muon    = True
trig_sf     = 'bparking_trigger_sf/scale_factors.json'
hammer      = ''
fit_schemes = 'free'

already_submitted = [
]

# (short name, DAS name). Main samples only, no _ext.
productions = [
    ('signal', '/BsToDsL_DsToPhiKKPi_TauToMuNuNu_TuneCP5_13TeV_pythia8-evtgen/'
               'RunIISummer20UL18MiniAODv2-Custom_RDStarPU_BParking_106X_upgrade2018_realistic_v16_L1v1-v3/MINIAODSIM'),
#   ('hb',     'PLACEHOLDER_HB_DATASET'),
]


def corrections():
    '''{name: (absolute path, path in the sandbox)}'''
    _, ana = cmssw_paths()
    data = os.path.join(ana, 'data')
    out = {}
    for name, rel in (('trk-calib', trk_calib), ('cov-scale', cov_scale),
                      ('trig-sf', trig_sf), ('hammer', hammer)):
        if not rel:
            continue
        path = rel if os.path.isabs(rel) else os.path.join(data, rel)
        if not os.path.isfile(path):
            raise RuntimeError('%s file %s not found' % (name, path))
        out[name] = path
    if trk_calib:
        app = os.path.join(os.path.dirname(out['trk-calib']), 'applier', 'python', 'rds_trkcalib.py')
        if not os.path.isfile(app):
            raise RuntimeError('momcal applier %s not found' % app)
        out['trk-calib-applier'] = app
    return out


def create_config(short, dataset, corr):
    inputs = common_inputs()
    args = ['MC=1', 'SAVENONTRIG=1', 'FITSCHEMES=' + fit_schemes]
    if 'trk-calib' in corr:
        tdir = os.path.dirname(corr['trk-calib'])           # shipped whole, keeps its name
        inputs.append(tdir)
        args.append('TRKCALIB=%s/%s' % (os.path.basename(tdir), os.path.basename(corr['trk-calib'])))
    for name, key in (('cov-scale', 'COVSCALE'), ('trig-sf', 'TRIGSF'), ('hammer', 'HAMMER')):
        if name in corr:
            inputs.append(corr[name])
            args.append('%s=%s' % (key, os.path.basename(corr[name])))
    if cov_muon and 'cov-scale' in corr:
        args.append('COVMUONLY=1')

    return base_config(
        request_name  = 'rds_mc_%s_%s' % (short, tag),
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
    corr = corrections()
    for short, dataset in productions:
        write_provenance(allow_unpushed, {
            'tool': 'crab_rds_mc.py', 'dataset': dataset, 'out_dir': out_dir,
            'files_per_job': files_per_job, 'max_files': max_files, 'fit_schemes': fit_schemes,
            'cov_scale_muon_only': cov_muon,
            'corrections': {k: {'path': v, 'sha256': sha256(v)} for k, v in corr.items()},
        })
        cfg = create_config(short, dataset, corr)
        if cfg.General.requestName in already_submitted:
            print('skipping (already submitted): %s' % cfg.General.requestName)
            continue
        print('%s  ->  %s   (%s)' % (dataset, cfg.General.requestName, ' '.join(cfg.JobType.scriptArgs)))
        p = Process(target=submit, args=(cfg,))
        p.start()
        p.join()
