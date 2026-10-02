'''
Inputs of build_pu_card.py: which data pileup profile each period uses, and
which MC generation profile each campaign was produced with.

Sources: TWiki PileupJSONFileforData (r53, 2026-04-08), and
SimGeneral/MixingModule/python/ in CMSSW for the MC profiles.

Every entry marked  # CONFIRM  is a choice to check before production.
'''

import os

from Bmmm.Analysis.LumiMask import golden_json_for_year

EOS_CERT = '/eos/user/c/cmsdqm/www/CAF/certification'

# Golden JSONs for pileupCalc: the SAME files the ntuple production applies as
# its lumi mask (Bmmm/Analysis/data/golden_jsons, one per year), so the data
# pileup profile is computed on exactly the lumisections the ntuples keep.
GOLDEN_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           '..', '..', '..', 'data', 'golden_jsons'))

# Minimum-bias cross sections [ub]. TWiki: 69.2 mb recommended for Run 3,
# uncertainty 4.6%; the central histograms are made at exactly these values.
XSEC_UB = {'nominal': 69200, 'up': 72400, 'down': 66000}

# Binning shared by the MC profiles and the data histograms: 100 unit bins on
# [0, 100). Every Run 3 profile in SimGeneral/MixingModule has exactly this.
NBINS = 100

##########################################################################################
#####      DATA PERIODS
##########################################################################################
# kind = 'central'    : the centrally produced ROOT histograms in `dir`. The file
#                       for each cross section is found by the "<xsec>ub" in its
#                       name; if a directory holds more than one candidate the
#                       builder stops and asks for `files` to be set explicitly:
#                       files = {'nominal': ..., 'up': ..., 'down': ...}
# kind = 'pileupCalc' : run pileupCalc.py --calcMode true on the golden JSON of
#                       the year against the central pileup JSON. Used where the
#                       TWiki lists no central histogram (2024) or nothing at all
#                       (2025, 2026).
#
# Golden JSONs: taken from GOLDEN_DIR (see above). The pileup JSON only needs to
# CONTAIN the golden lumisections, so the widest one of the year is the right
# choice (e.g. 2024 BCDEFGHI).
PERIODS = {
    '2022BCD': dict(year='2022', kind='central',
                    dir=EOS_CERT + '/Collisions22/PileUp/BCD/'),
    '2022EFG': dict(year='2022', kind='central',
                    dir=EOS_CERT + '/Collisions22/PileUp/EFG/'),
    '2023BC' : dict(year='2023', kind='central',
                    dir=EOS_CERT + '/Collisions23/PileUp/BC/'),
    '2023D'  : dict(year='2023', kind='central',
                    dir=EOS_CERT + '/Collisions23/PileUp/D/'),
    '2024'   : dict(year='2024', kind='pileupCalc',
                    pileup_json=EOS_CERT + '/Collisions24/PileUp/pileup_JSON-2024BCDEFGHI.txt',
                    golden_json=golden_json_for_year('2024', GOLDEN_DIR)),
    '2025'   : dict(year='2025', kind='pileupCalc',
                    pileup_json=None,                                        # CONFIRM: not on the TWiki
                    golden_json=golden_json_for_year('2025', GOLDEN_DIR)),
    '2026'   : dict(year='2026', kind='pileupCalc',
                    pileup_json=None,                                        # CONFIRM: not on the TWiki
                    golden_json=golden_json_for_year('2026', GOLDEN_DIR)),
}

##########################################################################################
#####      MC CAMPAIGNS
##########################################################################################
# mc_cfi   : SimGeneral.MixingModule module holding the generation profile.
# periods  : data year -> data period; one pu_weight_<year> block per entry.
# confirmed: the profile has been checked against the nti distribution of the
#            unskimmed MINIAODSIM (mc_pu_profile_from_miniaod.py). Until then
#            the inspector refuses the campaign unless ':allow-unconfirmed'.
#
# The profile assignments below are NOT verified yet: the premix library of
# each campaign was produced with some --pileup scenario, which is recorded in
# McM, not in the dataset. The 2023 choice in particular is between three
# similar-looking hybrid profiles (means 57, 62, 67).
CAMPAIGNS = {
    'Summer22'    : dict(mc_cfi='Run3_2022_LHC_Simulation_10h_2h_cfi',              # CONFIRM
                         periods={'2022': '2022BCD'}, confirmed=False,
                         gt='130X_mcRun3_2022_realistic_v5'),
    'Summer22EE'  : dict(mc_cfi='Run3_2022_LHC_Simulation_10h_2h_cfi',              # CONFIRM
                         periods={'2022': '2022EFG'}, confirmed=False,
                         gt='130X_mcRun3_2022_realistic_postEE_v6'),
    'Summer23'    : dict(mc_cfi='Run3_2023_LHC_Simulation_12p5h_9h_hybrid2p23_cfi', # CONFIRM
                         periods={'2023': '2023BC'}, confirmed=False,
                         gt='130X_mcRun3_2023_realistic_v15'),
    'Summer23BPix': dict(mc_cfi='Run3_2023_LHC_Simulation_12p5h_9h_hybrid2p23_cfi', # CONFIRM
                         periods={'2023': '2023D'}, confirmed=False,
                         gt='130X_mcRun3_2023_realistic_postBPix_v6'),
    # no 2025 / 2026 MC exists or will exist: Summer24 stands in for all three
    'Summer24'    : dict(mc_cfi='mix_2024_25ns_RunIII2024Summer24_PoissonOOTPU_cfi', # CONFIRM
                         periods={'2024': '2024', '2025': '2025', '2026': '2026'},
                         confirmed=False,
                         gt='150X_mcRun3_2024_realistic_v4'),
}
