# Track momentum calibration inputs

`--trk-calib` (R(Ds) ntuplizer, MC only) reads the momcal correction file and
its applier from here (or from the directory of the JSON you pass):

    rds_trkcalib.py                          # momcal applier/python/rds_trkcalib.py
    rds_trkcalib_2018BParking_v4.json        # momcal applier/data/

Copy both from the SAME momcal version: v4 has muon-specific terms
(`per_bin_muon`) that only the v4 applier (`kind` argument) applies, and the
job stops if the applier cannot. They are not committed here so that the
correction is versioned in one place (rmanzoni/momcal).
