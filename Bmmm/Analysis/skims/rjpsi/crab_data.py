from __future__ import division
import re
from http.client import HTTPException
from CRABClient.UserUtilities import config as Configuration
from CRABAPI.RawCommand import crabCommand


'''
remember to do 
source /cvmfs/cms.cern.ch/common/crab-setup.csh
before using crab
'''


def dataset_suffix(dataset):
    '''
    Unique, filesystem-safe suffix from the dataset name.
      short : primary dataset, stripped of tune/energy cruft (carries the Parking index for data)
      era   : leading token of the processed dataset (Run2024C..., RunIISummer20UL18...)
      ext   : _extN if present (distinguishes MC extensions)
      ver   : trailing -vN (distinguishes reprocessings, e.g. Run2024I -v3 vs _v2-v2)
    The (short, era, ext, ver) tuple uniquely identifies every dataset in these lists.
    '''
    primary   = dataset.split('/')[1]
    processed = dataset.split('/')[2]
    short = primary.split('_Tune')[0].split('_13TeV')[0]
    era   = processed.split('-')[0]
    ver   = processed.split('-')[-1]
    m     = re.search(r'_(ext\d+)', processed)
    ext   = '_' + m.group(1) if m else ''
    return '%s_%s%s_%s' % (short, era, ext, ver)


def create_config(dataset, outdir, dataset_tag, request_name, pset, workarea='crab_skims_18jun26_data2024_v1', site='T3_CH_PSI'):
    
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
    config.JobType.maxMemoryMB             = 4000
    #config.JobType.maxJobRuntimeMin       = 1440
    
    ##########################################################################################
    config.section_("Data")
    config.Data.inputDataset               = dataset
    config.Data.outLFNDirBase              = outdir    
    #config.Data.lumiMask                   = 'Cert_Collisions2024_..._Golden.json'  # see note below

    config.Data.splitting                  = 'FileBased'
    config.Data.unitsPerJob                = 5
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
    tag    = 'rjpsi_run3_18jun26'

    eras = [
        "Run2024C-MINIv6NANOv15-v1",
        "Run2024D-MINIv6NANOv15-v1",
        "Run2024E-MINIv6NANOv15-v1",
        "Run2024F-MINIv6NANOv15-v3",
        "Run2024G-MINIv6NANOv15-v3",
        "Run2024H-MINIv6NANOv15-v3",
        "Run2024I-MINIv6NANOv15-v3",
        "Run2024I-MINIv6NANOv15_v2-v2",
    ]
    
    datasets = [
        f"/ParkingDoubleMuonLowMass{i}/{era}/MINIAOD"
        for i in range(8)
        for era in eras
    ]
    

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