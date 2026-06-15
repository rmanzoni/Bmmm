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
    Build a unique, filesystem-safe suffix from the dataset name so that
    samples sharing the same primary dataset (e.g. base / ext1 / ext2)
    do not collide in requestName or in the published dataset name.
    '''
    primary   = dataset.split('/')[1]
    processed = dataset.split('/')[2]
    short     = primary.split('_Tune')[0]              # e.g. BcToJPsiMuMu_inclusive
    m         = re.search(r'_(ext\d+)', processed)
    ext       = m.group(1) if m else 'ext0'
    ver       = processed.split('-')[-1]               # e.g. v1 / v3
    return '%s_%s_%s' % (short, ext, ver)


def create_config(dataset, outdir, dataset_tag, request_name, pset, workarea='crab_skims_14jun26_MC_v1', site='T3_CH_PSI'):
    
    config = Configuration()
    
    ##########################################################################################
    config.section_("General")
    config.General.instance                = 'prod'
    config.General.workArea                = workarea
    config.General.requestName             = request_name
    config.General.transferOutputs         = True
    config.General.transferLogs            = True
    #config.General.restHost               = ''   
    #config.General.dbInstance             = ''   
    
    ##########################################################################################
    config.section_("JobType")
    config.JobType.pluginName              = 'Analysis'
    config.JobType.psetName                = pset
    config.JobType.allowUndistributedCMSSW = True
    #config.JobType.maxMemoryMB            = 2500
    #config.JobType.maxJobRuntimeMin       = 1440
    
    ##########################################################################################
    config.section_("Data")
    config.Data.inputDataset               = dataset
    config.Data.outLFNDirBase              = outdir    
    #config.Data.splitting                  = 'Automatic'
    #config.Data.unitsPerJob                = 360
    #config.Data.totalUnits                 = -1 #config.Data.unitsPerJob * NJOBS

    config.Data.splitting                  = 'FileBased'
    config.Data.unitsPerJob                = 10
    config.Data.totalUnits                 = -1 #config.Data.unitsPerJob * NJOBS

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
    
    pset   = 'vertex_refitter_cfg.py'
    outdir = '/store/user/manzoni/skims'    
    tag    = 'rjpsi_run3_12jun26'

    datasets = [
        '/BcToJPsiMuMu_inclusive_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM'     ,
        '/BcToJPsiMuMu_inclusive_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1_ext1-v1/MINIAODSIM',
        '/BcToJPsiMuMu_inclusive_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1_ext2-v1/MINIAODSIM',
        '/HbToJPsiMuMu_3MuFilter_TuneCP5_13TeV-pythia8-evtgen/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1_ext1-v3/MINIAODSIM'         ,
    ]

    already_submitted = [
    ]

    
    for ids in datasets:
        if ids in already_submitted:
            print('\n\nAlready submitted', ids, 'SKIPPING')
            continue
        
        full_tag = '%s_%s' % (tag, dataset_suffix(ids))
        
        iconfig = create_config(
            dataset      = ids                , 
            outdir       = outdir             , 
            dataset_tag  = full_tag           , 
            request_name = full_tag[:100]     ,  # CRAB caps requestName at 100 chars
            pset         = pset               ,
        )
    
        print('\n\nsubmitting config:')
        print(iconfig)
        #import pdb ; pdb.set_trace()

        try:
            crabCommand('submit', config=iconfig)
        except HTTPException as hte:
            print("HTTPException occurred: %s" % str(hte))
        except Exception as e:
            print("Failed to submit job: %s" % str(e))