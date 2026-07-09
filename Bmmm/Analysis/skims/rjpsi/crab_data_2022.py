from __future__ import division
import subprocess
from http.client import HTTPException
from CRABClient.UserUtilities import config as Configuration
from CRABAPI.RawCommand import crabCommand


'''
remember to do
source /cvmfs/cms.cern.ch/common/crab-setup.csh
before using crab
'''


# ------------------------------------------------------------------------------------
# Dataset discovery
# ------------------------------------------------------------------------------------
# The dataset list is queried live from DAS instead of being reconstructed as a
# (part x fixed-version) product. Reconstructing was wrong: the -vN reprocessing tag
# varies per (part, era) (e.g. LowMass7 Run2024G is -v4, LowMass2 Run2024F is -v2, ...),
# so the fixed-version product built names that either did not exist (task never ran)
# or pointed at the wrong reprocessing. DAS is the ground truth.
#
# NOTE the wildcard grabs *every* version that exists. Right now that is exactly the 64
# datasets we want (Run2024I legitimately has two: '-vN' and '_v2-v2'). If a NEW
# reprocessing (e.g. a future -v4) appears, it will show up here as an extra line and
# trip the EXPECTED_DATASETS guard below -- which is the point: fail loud, decide by hand
# whether the new version should replace or add to the campaign.

DAS_QUERY = 'dataset dataset=/ParkingDoubleMuonLowMass*/Run2024*MINIv6NANOv15*/MINIAOD'
EXPECTED_DATASETS = 64   # 8 parts x 8 (C,D,E,F,G,H,I,I_v2) -- update if the campaign changes


def get_datasets(query=DAS_QUERY):
    '''Return the sorted list of datasets DAS knows about for this campaign.'''
    out = subprocess.check_output(['dasgoclient', '-query', query], text=True)
    datasets = sorted(line.strip() for line in out.splitlines() if line.strip())
    return datasets


def dataset_suffix(dataset):
    '''
    Unique, filesystem-safe suffix built directly from the real dataset name -- NOT
    parsed-and-reconstructed. Taking the primary + processed strings verbatim (with '-'
    -> '_') guarantees two distinct datasets can never collapse to the same tag.

    Example:
      /ParkingDoubleMuonLowMass5/Run2024I-MINIv6NANOv15-v2/MINIAOD
        -> ParkingDoubleMuonLowMass5_Run2024I_MINIv6NANOv15_v2
      /ParkingDoubleMuonLowMass5/Run2024I-MINIv6NANOv15_v2-v2/MINIAOD
        -> ParkingDoubleMuonLowMass5_Run2024I_MINIv6NANOv15_v2_v2
    (The old (short, era, ext, ver) tuple collapsed both of these to 'Run2024I_v2'.)
    '''
    primary   = dataset.split('/')[1]
    processed = dataset.split('/')[2]
    short = primary.split('_Tune')[0].split('_13TeV')[0]   # harmless no-op for data
    proc  = processed.replace('-', '_')
    return '%s_%s' % (short, proc)


# ------------------------------------------------------------------------------------
# Config factory
# ------------------------------------------------------------------------------------
def create_config(dataset, outdir, dataset_tag, request_name, pset, workarea='crab_skims_02jul26_data2024_v2', site='T2_CH_CSCS'):

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
    config.JobType.allowUndistributedCMSSW = True
    config.JobType.maxMemoryMB             = 3000   # NB: old run peaked at 3838 MB; see note
    #config.JobType.maxJobRuntimeMin       = 1440

    ##########################################################################################
    config.section_("Data")
    config.Data.inputDataset               = dataset
    config.Data.outLFNDirBase              = outdir
    #config.Data.lumiMask                   = 'Cert_Collisions2024_..._Golden.json'  # see note below

    config.Data.splitting                  = 'FileBased'
    config.Data.unitsPerJob                = 15
#     config.Data.splitting                  = 'Automatic'
    config.Data.totalUnits                 = -1

    config.Data.publication                = True
    config.Data.outputDatasetTag           = dataset_tag
    #config.Data.inputDBS                   = 'phys03'

    ##########################################################################################
    config.section_("User")

    ##########################################################################################
    config.section_("Site")
    config.Site.storageSite                = site
    #config.Site.whitelist                  = ['T1_*','T2_US_*','T2_IT_*','T2_DE_*','T2_ES_*','T2_FR_*','T2_UK_*']
    #config.Site.blacklist                  = ['T2_ES_IFCA']

    ##########################################################################################
    config.section_("Debug")
    #config.Debug.scheddName                = 'crab3@vocms059.cern.ch'

    return config


##########################################################################################
##########################################################################################
##########################################################################################


if __name__ == '__main__':

    pset   = 'vertex_refitter_data_2024_cfg.py'   # <-- DATA pset, Run3 data GT (see note)
    outdir = '/store/user/manzoni/skims'
    tag    = 'rjpsi_run3_23jun26_v2'

#     datasets = get_datasets()
    
    datasets = [
     '/ParkingDoubleMuonLowMass0/Run2022C-10Dec2022-v2/MINIAOD',
     '/ParkingDoubleMuonLowMass0/Run2022D-10Dec2022-v2/MINIAOD',
     '/ParkingDoubleMuonLowMass0/Run2022E-10Dec2022-v2/MINIAOD',
     '/ParkingDoubleMuonLowMass0/Run2022F-22Sep2023-v1/MINIAOD',
     '/ParkingDoubleMuonLowMass0/Run2022G-22Sep2023-v1/MINIAOD',
     '/ParkingDoubleMuonLowMass1/Run2022C-10Dec2022-v3/MINIAOD',
     '/ParkingDoubleMuonLowMass1/Run2022D-10Dec2022-v3/MINIAOD',
     '/ParkingDoubleMuonLowMass1/Run2022E-10Dec2022-v2/MINIAOD',
     '/ParkingDoubleMuonLowMass1/Run2022F-22Sep2023-v1/MINIAOD',
     '/ParkingDoubleMuonLowMass1/Run2022G-22Sep2023-v1/MINIAOD',
     '/ParkingDoubleMuonLowMass2/Run2022C-10Dec2022-v2/MINIAOD',
     '/ParkingDoubleMuonLowMass2/Run2022D-10Dec2022-v2/MINIAOD',
     '/ParkingDoubleMuonLowMass2/Run2022E-10Dec2022-v2/MINIAOD',
     '/ParkingDoubleMuonLowMass2/Run2022F-22Sep2023-v1/MINIAOD',
     '/ParkingDoubleMuonLowMass2/Run2022G-22Sep2023-v1/MINIAOD',
     '/ParkingDoubleMuonLowMass3/Run2022C-10Dec2022-v2/MINIAOD',
     '/ParkingDoubleMuonLowMass3/Run2022D-10Dec2022-v2/MINIAOD',
     '/ParkingDoubleMuonLowMass3/Run2022E-10Dec2022-v2/MINIAOD',
     '/ParkingDoubleMuonLowMass3/Run2022F-22Sep2023-v1/MINIAOD',
     '/ParkingDoubleMuonLowMass3/Run2022G-22Sep2023-v1/MINIAOD',
     '/ParkingDoubleMuonLowMass4/Run2022C-10Dec2022-v2/MINIAOD',
     '/ParkingDoubleMuonLowMass4/Run2022D-10Dec2022-v2/MINIAOD',
     '/ParkingDoubleMuonLowMass4/Run2022E-10Dec2022-v2/MINIAOD',
     '/ParkingDoubleMuonLowMass4/Run2022F-22Sep2023-v1/MINIAOD',
     '/ParkingDoubleMuonLowMass4/Run2022G-22Sep2023-v2/MINIAOD',
     '/ParkingDoubleMuonLowMass5/Run2022C-10Dec2022-v2/MINIAOD',
     '/ParkingDoubleMuonLowMass5/Run2022D-10Dec2022-v2/MINIAOD',
     '/ParkingDoubleMuonLowMass5/Run2022E-10Dec2022-v2/MINIAOD',
     '/ParkingDoubleMuonLowMass5/Run2022F-22Sep2023-v1/MINIAOD',
     '/ParkingDoubleMuonLowMass5/Run2022G-22Sep2023-v1/MINIAOD',
     '/ParkingDoubleMuonLowMass6/Run2022C-10Dec2022-v2/MINIAOD',
     '/ParkingDoubleMuonLowMass6/Run2022D-10Dec2022-v2/MINIAOD',
     '/ParkingDoubleMuonLowMass6/Run2022E-10Dec2022-v2/MINIAOD',
     '/ParkingDoubleMuonLowMass6/Run2022F-22Sep2023-v1/MINIAOD',
     '/ParkingDoubleMuonLowMass6/Run2022G-22Sep2023-v1/MINIAOD',
     '/ParkingDoubleMuonLowMass7/Run2022C-10Dec2022-v2/MINIAOD',
     '/ParkingDoubleMuonLowMass7/Run2022D-10Dec2022-v2/MINIAOD',
     '/ParkingDoubleMuonLowMass7/Run2022E-10Dec2022-v2/MINIAOD',
     '/ParkingDoubleMuonLowMass7/Run2022F-22Sep2023-v1/MINIAOD',
     '/ParkingDoubleMuonLowMass7/Run2022G-22Sep2023-v1/MINIAOD',
    ]
    
    
    
    
    
    

    print('\nfound %d datasets from DAS:' % len(datasets))
    for d in datasets:
        print('   ', d)

    # Fail loud if DAS returns something other than the expected campaign (extra/missing
    # reprocessing). Better a hard stop than silently skimming the wrong set.
    assert len(datasets) == EXPECTED_DATASETS, (
        'DAS returned %d datasets, expected %d -- inspect the list above before submitting '
        '(a new reprocessing may have appeared, or one went missing).'
        % (len(datasets), EXPECTED_DATASETS)
    )

    # Provenance: freeze exactly what this campaign submitted.
    try:
        with open('campaign_datasets.txt', 'w') as fout:
            fout.write('\n'.join(datasets) + '\n')
    except OSError as err:
        print('WARNING: could not write campaign_datasets.txt: %s' % err)

    already_submitted = [
    ]

    for ids in datasets:
        if ids in already_submitted:
            print('\n\nAlready submitted', ids, 'SKIPPING')
            continue

        full_tag = '%s_%s' % (tag, dataset_suffix(ids))

        iconfig = create_config(
            dataset      = ids           ,
            outdir       = outdir        ,
            dataset_tag  = full_tag      ,
            request_name = full_tag[:100],  # CRAB caps requestName at 100 chars
            pset         = pset          ,
            site         = 'T2_CH_CSCS'  ,
        )

        print('\n\nsubmitting config:')
        print(iconfig)

        try:
            crabCommand('submit', config=iconfig)
        except HTTPException as hte:
            print("HTTPException occurred: %s" % str(hte))
        except Exception as e:
            print("Failed to submit job: %s" % str(e))