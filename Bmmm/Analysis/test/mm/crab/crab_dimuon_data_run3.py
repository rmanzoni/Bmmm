'''
CRAB submission: dimuon ntuples on Run 3 ParkingDoubleMuonLowMass, Part 1.

Same dataset family and same trigger as the RJpsi Run 3 skims
(HLT_DoubleMu4_3_LowMass), but run on the ORIGINAL MINIAOD rather than on the
skims -- the tag-and-probe measurement needs events the skim selection would
have thrown away.

Structured like skims/rjpsi/crab_data_*.py so the two campaigns are managed the
same way, with two differences that follow from what is being run:

  * the dimuon ntuplizer is a FWLite script, not a cmsRun configuration, so the
    job goes through scriptExe with a dummy PSet (see PSet.py, crab_script.sh)
    instead of psetName pointing at a real config;
  * the output is a flat ntuple, not EDM, so publication is off. The files land
    under outLFNDirBase and are collected with hadd_uproot.py, not by DBS.

remember to do
    source /cvmfs/cms.cern.ch/common/crab-setup.sh
before using crab.

    python crab_dimuon_data_run3.py            # submit
    python crab_dimuon_data_run3.py --dry-run  # print the configs, submit nothing
'''

from __future__ import division

import argparse
import os
import sys

from http.client import HTTPException
from CRABClient.UserUtilities import config as Configuration
from CRABAPI.RawCommand import crabCommand


# ------------------------------------------------------------------------------------
# Campaign
# ------------------------------------------------------------------------------------
TAG    = 'dimuon_run3_part1_08sep26'
OUTDIR = '/store/user/manzoni/dimuon_ntuples_run3_08sep2026'
# SITE   = 'T2_CH_CSCS'
SITE   = 'T3_CH_PSI'
WORKAREA = 'crab_%s' % TAG

# Part 1 only, as submitted for the RJpsi Run 3 skims -- this list is exactly the
# ParkingDoubleMuonLowMass1 entries of skims/rjpsi/crab_data_*.py, so the two
# campaigns cover the same runs. Comment out eras to submit a subset; starting
# with a single era is the sensible way to find out what the job costs before
# committing to the lot.
DATASETS = [
    '/ParkingDoubleMuonLowMass1/Run2022C-PromptReco-v1/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2022D-PromptReco-v1/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2022D-PromptReco-v2/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2022E-PromptReco-v1/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2022F-22Sep2023-v1/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2022G-22Sep2023-v1/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2023C-22Sep2023_v1-v2/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2023C-22Sep2023_v2-v1/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2023C-22Sep2023_v3-v1/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2023C-22Sep2023_v4-v1/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2023D-22Sep2023_v1-v1/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2023D-22Sep2023_v2-v1/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2026B-PromptReco-v1/MINIAOD',
    '/ParkingDoubleMuonLowMass1/Run2026D-PromptReco-v1/MINIAOD',
]

# The ntuplizer, shipped into the job alongside this directory's two wrappers.
# CRAB flattens inputFiles into the job working directory, which is why
# crab_script.sh calls it by bare name.
INSPECTOR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'inspector_mm_analysis.py')

# Writes FrameworkJobReport.xml at the end of the job. Without it every job
# fails with exit code 50115 (BadFWJRXML): CRAB's post-job parses that file
# whatever the job ran, cmsRun writes one and a scriptExe does not.
MAKE_FJR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_fjr.py')

# Opens every input file before the event loop starts, falling through to the
# next xrootd door if one does not answer.
RESOLVE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resolve_pfns.py')


def dataset_suffix(dataset):
    '''Filesystem-safe tag taken verbatim from the dataset name, so two
    datasets can never collapse onto the same one. Same rule as the RJpsi
    skim configs.'''
    primary   = dataset.split('/')[1]
    processed = dataset.split('/')[2].replace('-', '_')
    return '%s_%s' % (primary, processed)


def create_config(dataset, request_name, dataset_tag):

    config = Configuration()

    ##########################################################################################
    config.section_("General")
    config.General.instance                = 'prod'
    config.General.workArea                = WORKAREA
    config.General.requestName             = request_name
    config.General.transferOutputs         = True
    config.General.transferLogs            = True

    ##########################################################################################
    config.section_("JobType")
    config.JobType.pluginName              = 'Analysis'
    # dummy PSet: CRAB parses it for splitting and rewrites its fileNames per
    # job; the work is done by scriptExe
    config.JobType.psetName                = 'PSet.py'
    config.JobType.scriptExe               = 'crab_script.sh'
    config.JobType.inputFiles              = ['PSet.py', 'crab_script.sh',
                                              INSPECTOR, MAKE_FJR, RESOLVE]
    config.JobType.outputFiles             = ['dimuon_ntuple.root']
    config.JobType.allowUndistributedCMSSW = True
    # measured: 619 MB peak over 257 jobs. Asking for 2500 only made the jobs
    # queue behind smaller ones.
    config.JobType.maxMemoryMB             = 1500
    #config.JobType.maxJobRuntimeMin       = 1440

    ##########################################################################################
    config.section_("Data")
    config.Data.inputDataset               = dataset
    config.Data.outLFNDirBase              = OUTDIR

    # FileBased rather than Automatic: the ntuplizer is a python event loop, so
    # per-event cost is high and fairly flat, and file-based splitting keeps a
    # failed job cheap to resubmit. Start low and raise once a first campaign
    # has shown the runtime per file.
    config.Data.splitting                  = 'FileBased'
    config.Data.unitsPerJob                = 5
    config.Data.totalUnits                 = -1

    # flat ntuple, not EDM -- nothing to publish to DBS
    config.Data.publication                = False
    config.Data.outputDatasetTag           = dataset_tag

    ##########################################################################################
    config.section_("User")

    ##########################################################################################
    config.section_("Site")
    config.Site.storageSite                = SITE

    ##########################################################################################
    config.section_("Debug")

    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true',
                        help='print the configs and the dataset list, submit nothing')
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    for needed in ('PSet.py', 'crab_script.sh'):
        if not os.path.isfile(os.path.join(here, needed)):
            sys.exit('missing %s -- run this from %s' % (needed, here))
    if not os.path.isfile(INSPECTOR):
        sys.exit('cannot find the ntuplizer at %s' % INSPECTOR)
    if not os.path.isfile(MAKE_FJR):
        sys.exit('cannot find the job-report writer at %s' % MAKE_FJR)
    if not os.path.isfile(RESOLVE):
        sys.exit('cannot find the PFN resolver at %s' % RESOLVE)

    print('\nsubmitting %d dataset(s):' % len(DATASETS))
    for d in DATASETS:
        print('   ', d)

    # Provenance: freeze exactly what this campaign submitted, as the RJpsi
    # skim configs do.
    try:
        with open(os.path.join(here, 'campaign_datasets.txt'), 'w') as fout:
            fout.write('\n'.join(DATASETS) + '\n')
    except OSError as err:
        print('WARNING: could not write campaign_datasets.txt: %s' % err)

    already_submitted = [
    ]

    for ids in DATASETS:
        if ids in already_submitted:
            print('\n\nAlready submitted', ids, 'SKIPPING')
            continue

        full_tag = '%s_%s' % (TAG, dataset_suffix(ids))

        iconfig = create_config(
            dataset      = ids           ,
            request_name = full_tag[:100],   # CRAB caps requestName at 100 chars
            dataset_tag  = full_tag      ,
        )

        print('\n\nsubmitting config:')
        print(iconfig)

        if args.dry_run:
            continue

        try:
            crabCommand('submit', config=iconfig)
        except HTTPException as hte:
            print("HTTPException occurred: %s" % str(hte))
        except Exception as e:
            print("Failed to submit job: %s" % str(e))


if __name__ == '__main__':
    main()
