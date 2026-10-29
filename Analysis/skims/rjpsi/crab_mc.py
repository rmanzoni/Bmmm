from __future__ import division
import re
import sys
import argparse
import subprocess
import multiprocessing
from http.client import HTTPException
from CRABClient.UserUtilities import config as Configuration
from CRABAPI.RawCommand import crabCommand


'''
remember to do
source /cvmfs/cms.cern.ch/common/crab-setup.csh
before using crab

usage:
    python3 crab_mc.py --dry-run     # DAS + GT checks, print configs, submit nothing
    python3 crab_mc.py               # submit
'''


# ------------------------------------------------------------------------------------
# Samples: (dataset, global tag) -- one GT per sample, passed to the pset at submission.
# ------------------------------------------------------------------------------------
# Each sample is skimmed with the most recent GT of its series: the production GT (the
# one in the processed dataset name) or a newer version of the same series (e.g. Summer24
# is produced with 150X_mcRun3_2024_realistic_v2, skimmed with _v4). The PdmV analysis
# recommendations are older than these productions. check_gt enforces same series and
# version >= production, and that the GT belongs to the campaign's era.
SAMPLES = [
    # Hb -> J/psi + X, 3-muon filter
    ('/HbToJPsiMuMu_3MuFilter_TuneCP5_13p6TeV_pythia8-evtgen/Run3Summer22EEMiniAODv4-130X_mcRun3_2022_realistic_postEE_v6-v2/MINIAODSIM'             , '130X_mcRun3_2022_realistic_postEE_v6'  ),
    ('/HbToJPsiMuMu_3MuFilter_TuneCP5_13p6TeV_pythia8-evtgen/Run3Summer22MiniAODv4-130X_mcRun3_2022_realistic_v5-v2/MINIAODSIM'                      , '130X_mcRun3_2022_realistic_v5'         ),
    ('/HbToJPsiMuMu_3MuFilter_TuneCP5_13p6TeV_pythia8-evtgen/Run3Summer23BPixMiniAODv4-130X_mcRun3_2023_realistic_postBPix_v6-v2/MINIAODSIM'         , '130X_mcRun3_2023_realistic_postBPix_v6'),
    ('/HbToJPsiMuMu_3MuFilter_TuneCP5_13p6TeV_pythia8-evtgen/Run3Summer23MiniAODv4-130X_mcRun3_2023_realistic_v15-v2/MINIAODSIM'                     , '130X_mcRun3_2023_realistic_v15'        ),
    # Bc -> J/psi + X, inclusive cocktail
    ('/BcToJPsiMuMu_inclusive_TuneCP5_13p6TeV_bcvegpy2-pythia8-evtgen/Run3Summer22EEMiniAODv4-130X_mcRun3_2022_realistic_postEE_v6-v2/MINIAODSIM'    , '130X_mcRun3_2022_realistic_postEE_v6'  ),
    ('/BcToJPsiMuMu_inclusive_TuneCP5_13p6TeV_bcvegpy2-pythia8-evtgen/Run3Summer22MiniAODv4-130X_mcRun3_2022_realistic_v5-v2/MINIAODSIM'             , '130X_mcRun3_2022_realistic_v5'         ),
    ('/BcToJPsiMuMu_inclusive_TuneCP5_13p6TeV_bcvegpy2-pythia8-evtgen/Run3Summer23BPixMiniAODv4-130X_mcRun3_2023_realistic_postBPix_v6-v4/MINIAODSIM', '130X_mcRun3_2023_realistic_postBPix_v6'),
    ('/BcToJPsiMuMu_inclusive_TuneCP5_13p6TeV_bcvegpy2-pythia8-evtgen/Run3Summer23MiniAODv4-Fixed_130X_mcRun3_2023_realistic_v15-v3/MINIAODSIM'      , '130X_mcRun3_2023_realistic_v15'        ),
    ('/BcToJPsiMuMu_Fil-JPsi_TuneCP5_13p6TeV_bcvegpy2-pythia8-evtgen/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v6/MINIAODSIM'        , '150X_mcRun3_2024_realistic_v4'         ),
    # Run 2 UL18, for reference (was the GT hardcoded in the pset before):
    # ('/BcToJPsiMuMu_inclusive_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM', '106X_upgrade2018_realistic_v16_L1v1'),
]

# campaign (first token of the processed dataset name) -> regex the GT must contain.
# An unknown campaign is a hard error: add the rule when adding a new MC campaign.
ERA_GT_RULES = [
    (r'^Run3Summer22EEMiniAODv\d+$'      , r'_mcRun3_2022_realistic_postEE_'   ),
    (r'^Run3Summer22MiniAODv\d+$'        , r'_mcRun3_2022_realistic_v\d+$'     ),
    (r'^Run3Summer23BPixMiniAODv\d+$'    , r'_mcRun3_2023_realistic_postBPix_' ),
    (r'^Run3Summer23MiniAODv\d+$'        , r'_mcRun3_2023_realistic_v\d+$'     ),
    (r'^RunIII2024Summer24MiniAODv\d+$'  , r'_mcRun3_2024_realistic_v\d+$'     ),
    (r'^RunIISummer20UL18MiniAODv\d+$'   , r'_upgrade2018_realistic_v\d+_L1v1$'),
]


def campaign(dataset):
    '''e.g. Run3Summer23BPixMiniAODv4'''
    return dataset.split('/')[2].split('-')[0]


def production_gt(dataset):
    '''GT from the processed name, e.g. Run3Summer23MiniAODv4-Fixed_130X_..._v15-v3 -> 130X_..._v15'''
    processed = dataset.split('/')[2]
    gt = processed.split('-', 1)[1].rsplit('-', 1)[0]     # drop campaign and -vN
    return re.sub(r'^.*?(?=\d{3}X_)', '', gt)             # drop prefixes such as 'Fixed_'


def split_gt(gt):
    '''130X_mcRun3_2023_realistic_postBPix_v6 -> ('130X_mcRun3_2023_realistic_postBPix', 6)'''
    m = re.match(r'^(.+)_v(\d+)$', gt)
    if not m:
        raise ValueError('cannot parse GT version: %s' % gt)
    return m.group(1), int(m.group(2))


def check_gt(dataset, gt):
    prod = production_gt(dataset)
    (series, ver), (prod_series, prod_ver) = split_gt(gt), split_gt(prod)
    if series != prod_series or ver < prod_ver:
        raise ValueError('GT %s is not the production GT %s or a newer version of it, for\n    %s' % (gt, prod, dataset))
    camp = campaign(dataset)
    for camp_re, gt_re in ERA_GT_RULES:
        if re.match(camp_re, camp):
            if not re.search(gt_re, gt):
                raise ValueError('GT %s does not match campaign %s (expected /%s/) for\n    %s' % (gt, camp, gt_re, dataset))
            return
    raise ValueError('no ERA_GT_RULES entry for campaign %s (%s) -- add one' % (camp, dataset))


def check_das(dataset):
    '''Fail loud if DAS does not know the dataset (typo, or not VALID).'''
    out = subprocess.check_output(['dasgoclient', '-query', 'dataset dataset=%s' % dataset], text=True)
    found = [line.strip() for line in out.splitlines() if line.strip()]
    if found != [dataset]:
        raise ValueError('DAS lookup failed for %s, got: %s' % (dataset, found))


def dataset_suffix(dataset):
    '''
    Verbatim primary (up to _Tune) + processed name, '-' -> '_' -- same scheme as the
    data configs. Used for the outputDatasetTag.

    The old (short, ext, ver) scheme is NOT usable for Run 3: it would give
    BcToJPsiMuMu_inclusive_ext0_v2 for both Summer22 and Summer22EE (same for Hb).
    '''
    primary   = dataset.split('/')[1]
    processed = dataset.split('/')[2]
    short     = primary.split('_Tune')[0]
    return ('%s_%s' % (short, processed)).replace('-', '_')


def request_suffix(dataset):
    '''
    Compact form for requestName (100 char cap): short primary + campaign + version,
    e.g. BcToJPsiMuMu_inclusive_Run3Summer23BPix_v4. Uniqueness is asserted in main.
    '''
    primary   = dataset.split('/')[1]
    processed = dataset.split('/')[2]
    short     = primary.split('_Tune')[0]
    camp      = re.sub(r'MiniAODv\d+$', '', campaign(dataset))
    ver       = processed.split('-')[-1]
    return ('%s_%s_%s' % (short, camp, ver)).replace('-', '_')


# CRAB publishes as <username>-<outputDatasetTag>-<32 char hash>, max 199 chars (WMCore Lexicon)
MAX_TAG_LENGTH = 199 - len('manzoni-') - 33


def create_config(dataset, global_tag, outdir, dataset_tag, request_name, pset, workarea, site):

    config = Configuration()

    ##########################################################################################
    config.section_("General")
    config.General.instance                = 'prod'
    config.General.workArea                = workarea
    config.General.requestName             = request_name
    config.General.transferOutputs         = True
    config.General.transferLogs            = True

    ##########################################################################################
    config.section_("JobType")
    config.JobType.pluginName              = 'Analysis'
    config.JobType.psetName                = pset
    config.JobType.pyCfgParams             = ['globalTag=%s' % global_tag]
    config.JobType.allowUndistributedCMSSW = True
    config.JobType.maxMemoryMB             = 3000   # as the data skims (old data run peaked at 3838 MB)
    #config.JobType.maxJobRuntimeMin       = 1440

    ##########################################################################################
    config.section_("Data")
    config.Data.inputDataset               = dataset
    config.Data.outLFNDirBase              = outdir

    config.Data.splitting                  = 'FileBased'
    config.Data.unitsPerJob                = 8
    config.Data.totalUnits                 = -1
    config.Data.publication                = True
    config.Data.outputDatasetTag           = dataset_tag

    ##########################################################################################
    config.section_("User")

    ##########################################################################################
    config.section_("Site")
    config.Site.storageSite                = site

    ##########################################################################################
    config.section_("Debug")

    return config


# ------------------------------------------------------------------------------------
# One submission per process
# ------------------------------------------------------------------------------------
# CRABClient caches the imported pset at module level and refuses (FATAL ERROR) to load
# the same pset a second time with different pyCfgParams in the same Python process
# (CRABClient/JobType/CMSSWConfig.py, configurationCache). The data configs never hit
# this because every era has its own pset with a hardcoded GT; here the GT changes from
# sample to sample, so each submission runs in a forked child that starts with an empty
# cache. The parent never calls crabCommand itself.
def _submit_worker(config):
    try:
        crabCommand('submit', config=config)
    except HTTPException as hte:
        print('HTTPException occurred: %s' % str(hte))
        sys.exit(1)
    except Exception as e:
        print('Failed to submit job: %s' % str(e))
        sys.exit(1)


def submit_isolated(config):
    ctx  = multiprocessing.get_context('fork')
    proc = ctx.Process(target=_submit_worker, args=(config,))
    proc.start()
    proc.join()
    return proc.exitcode


##########################################################################################
##########################################################################################
##########################################################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run'       , action='store_true', help='run all checks and print the configs, submit nothing')
    parser.add_argument('--skip-das-check', action='store_true', help='do not query DAS for the input datasets')
    args = parser.parse_args()

    pset     = 'vertex_refitter_cfg.py'
    outdir   = '/store/user/manzoni/skims'
    tag      = 'rjpsi_run3_24sep26_v2'
    workarea = 'crab_skims_24sep26_v2_MC_run3'
    site     = 'T3_CH_PSI'

    already_submitted = [
    ]

    todo = [(ds, gt) for ds, gt in SAMPLES if ds not in already_submitted]
    for ds, gt in SAMPLES:
        if ds in already_submitted:
            print('Already submitted', ds, 'SKIPPING')

    # --- preflight: everything is checked before the first submission --------------
    datasets = [ds for ds, _ in todo]
    assert len(set(datasets)) == len(datasets), 'duplicated dataset in SAMPLES'

    for ds, gt in todo:
        check_gt(ds, gt)

    full_tags = ['%s_%s' % (tag, dataset_suffix(ds)) for ds, _ in todo]
    req_names = ['%s_%s' % (tag, request_suffix(ds)) for ds, _ in todo]
    assert len(set(full_tags)) == len(full_tags), 'outputDatasetTag collision: %s' % full_tags
    assert len(set(req_names)) == len(req_names), 'requestName collision: %s' % req_names
    for ft, rn in zip(full_tags, req_names):
        assert len(ft) <= MAX_TAG_LENGTH, 'outputDatasetTag too long (%d > %d): %s' % (len(ft), MAX_TAG_LENGTH, ft)
        assert len(rn) <= 100           , 'requestName too long (%d > 100): %s' % (len(rn), rn)

    if not args.skip_das_check:
        for i, ds in enumerate(datasets, 1):
            print('[DAS %d/%d] %s' % (i, len(datasets), ds), flush=True)
            check_das(ds)

    # --- submission ------------------------------------------------------------------
    failed = []
    for i, ((ds, gt), full_tag, req_name) in enumerate(zip(todo, full_tags, req_names), 1):

        iconfig = create_config(
            dataset      = ds       ,
            global_tag   = gt       ,
            outdir       = outdir   ,
            dataset_tag  = full_tag ,
            request_name = req_name ,
            pset         = pset     ,
            workarea     = workarea ,
            site         = site     ,
        )

        print('\n\n[%d/%d] %s\n      GT %s' % (i, len(todo), ds, gt), flush=True)
        print(iconfig, flush=True)

        if args.dry_run:
            continue

        if submit_isolated(iconfig) != 0:
            failed.append(ds)

    print('\n\n%s: %d/%d submitted' % ('DRY RUN' if args.dry_run else 'DONE',
                                       0 if args.dry_run else len(todo) - len(failed), len(todo)))
    if failed:
        print('FAILED:')
        for ds in failed:
            print('   ', ds)
        sys.exit(1)
