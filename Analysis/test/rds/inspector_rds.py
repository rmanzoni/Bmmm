'''
R(Ds) ntuplizer entry point: Bs -> Ds(-> phi(-> K K) pi) mu (nu), 2018
B-parking (ParkingBPH1-6, UL2018 MiniAODv2) and the UL2018 MC.

The logic lives in Bmmm.Analysis.RDsInspector; see test/rds/README.md.

Data (the certification mask is mandatory):

    python3 inspector_rds.py --inputFiles=<file or LFN> --filename=rds_data \
        --lumi-mask=golden --maxevents=2000 --logfreq=100

MC (all events kept, trigger decisions and matches stored; momentum
calibration and covariance scaling at production):

    python3 inspector_rds.py --inputFiles=<file or LFN> --filename=rds_mc --mc \
        --savenontrig --trk-calib=$CMSSW_BASE/src/Bmmm/Analysis/data/trkcalib/rds_trkcalib_2018BParking_v4.json \
        --cov-scale=<sigma_dxy table>.json --maxevents=2000 --logfreq=100
'''

from Bmmm.Analysis.RDsInspector import RDsInspector


if __name__ == '__main__':
    RDsInspector().main()
