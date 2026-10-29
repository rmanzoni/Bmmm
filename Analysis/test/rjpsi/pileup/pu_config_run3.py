'''
Inputs of build_pu_card.py: which data pileup profile each period uses, and
which MC generation profile each campaign was produced with.

Sources: TWiki PileupJSONFileforData (r53, 2026-04-08), the certification area
listed below, and SimGeneral/MixingModule/python/ in CMSSW for the MC profiles.

Every entry marked  # TODO  is still to be filled before the card is complete.
'''

import os

from Bmmm.Analysis.LumiMask import golden_json_for_year

EOS_CERT = '/eos/user/c/cmsdqm/www/CAF/certification'
# Web mirror of the same tree, used by build_pu_card.py wherever /eos is not
# mounted (e.g. the PSI T3): EOS_CERT + '/X' is fetched from CERT_URL + '/X'.
CERT_URL = 'https://cms-service-dqmdc.web.cern.ch/CAF/certification'

# Golden JSONs for pileupCalc: the SAME files the ntuple production applies as
# its lumi mask (Bmmm/Analysis/data/golden_jsons, one per year), so a profile
# computed here uses exactly the lumisections the ntuples keep.
GOLDEN_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           '..', '..', '..', 'data', 'golden_jsons'))

# Minimum-bias cross sections [ub]. TWiki: 69.2 mb recommended for Run 3,
# uncertainty 4.6%; the central histograms are made at exactly these values.
XSEC_UB = {'nominal': 69200, 'up': 72400, 'down': 66000}

# Binning shared by the MC profiles and the data histograms: 100 unit bins on
# [0, 100). Every Run 3 profile in SimGeneral/MixingModule has exactly this.
NBINS = 100


def central_files(subdir, pattern):
    '''{variation: EOS path} of a central histogram set, built from ONE file-name
    pattern with a %d for the cross section. Building the three names from
    XSEC_UB means nominal/up/down cannot be paired with the wrong file, and it
    avoids the name matching of `dir`, which is ambiguous in a directory that
    holds one histogram set per era (Collisions24/PileUp has ten of them).'''
    return dict((var, '%s/%s/%s' % (EOS_CERT, subdir, pattern % xsec))
                for var, xsec in XSEC_UB.items())


##########################################################################################
#####      DATA PERIODS
##########################################################################################
# kind = 'central'    : centrally produced pileup histograms. Either `files`
#                       ({variation: path}, preferred) or `dir`, in which the file
#                       of each cross section is found by the "<xsec>ub" in its
#                       name; the builder stops if that match is ambiguous.
# kind = 'pileupCalc' : run pileupCalc.py --calcMode true on the golden JSON of
#                       the year against the central pileup JSON. Only where no
#                       central histogram exists.
#
# Central histograms are computed by the Lumi POG on THEIR golden JSON of the
# year; the card records which file was used (path, URL, sha256).
PERIODS = {
    '2022BCD': dict(year='2022', kind='central',
                    dir=EOS_CERT + '/Collisions22/PileUp/BCD/'),
    '2022EFG': dict(year='2022', kind='central',
                    dir=EOS_CERT + '/Collisions22/PileUp/EFG/'),
    '2023BC' : dict(year='2023', kind='central',
                    dir=EOS_CERT + '/Collisions23/PileUp/BC/'),
    '2023D'  : dict(year='2023', kind='central',
                    dir=EOS_CERT + '/Collisions23/PileUp/D/'),
    # the full-year (BCDEFGHI) Golden set, not the per-era ones in the same dir
    '2024'   : dict(year='2024', kind='central',
                    dir=EOS_CERT + '/Collisions24/PileUp/',
                    files=central_files('Collisions24/PileUp',
                                        'dataPileupHistogram-2024BCDEFGHI_Golden-%dub.root')),
    '2025'   : dict(year='2025', kind='central',
                    dir=EOS_CERT + '/Collisions25/PileUp/',
                    files=central_files('Collisions25/PileUp',
                                        'dataPileupHistogram-2025pp_Golden-%dub.root')),
    # FIXME! Temporary solution until 2026 pileup will become available
    '2026'   : dict(year='2026', kind='central',
                    dir=EOS_CERT + '/Collisions25/PileUp/',
                    files=central_files('Collisions25/PileUp',
                                        'dataPileupHistogram-2025pp_Golden-%dub.root')),



    # TODO: no central 2026 histogram known yet. Either set kind='central' with
    # files=central_files('Collisions26/PileUp', '<pattern with %d>ub.root'), or
    # keep pileupCalc and set pileup_json to the 2026 pileup JSON. Then add
    # '2026' to SUMMER24_YEARS below.
#     '2026'   : dict(year='2026', kind='pileupCalc',
#                     pileup_json=None,
#                     golden_json=golden_json_for_year('2026', GOLDEN_DIR)),
}

# Reference only, not read by the builder in 'central' mode: the pileup JSONs
# behind the 2024 / 2025 central histograms, for a pileupCalc cross-check.
PILEUP_JSONS = {
    '2024': EOS_CERT + '/Collisions24/PileUp/pileup_JSON-2024BCDEFGHI_Golden.txt',
    '2025': EOS_CERT + '/Collisions25/PileUp/pileup_JSON-2025_Golden.txt',
}

##########################################################################################
#####      MC CAMPAIGNS
##########################################################################################
# mc_cfi   : SimGeneral.MixingModule module holding the generation profile.
# periods  : data year -> data period; one pu_weight_<year> block per entry.
# confirmed: the profile has been checked against the nti distribution of the
#            MINIAODSIM (mc_pu_profile_from_miniaod.py, chi2/ndf in the comment).
#            Unconfirmed campaigns are refused unless ':allow-unconfirmed'.
#
# 2023: the earlier guess (Run3_2023_LHC_Simulation_12p5h_9h_hybrid2p23) was
# WRONG for both campaigns; the per-era PoissonOOTPU profiles fit.

# Summer24 stands in for all of 2024, 2025 and 2026 (no 2025/2026 MC exists or
# will exist): one pu_weight_<year> block per year listed here. 2026 is left out
# until its data period above is complete -- pu_weight_2026 is then NaN, never 1.
SUMMER24_YEARS = ('2024', '2025', '2026')          # TODO: add '2026'

CAMPAIGNS = {
    'Summer22'    : dict(mc_cfi='Run3_2022_LHC_Simulation_10h_2h_cfi',
                         periods={'2022': '2022BCD'}, confirmed=True,   # Bc nti: chi2/ndf 0.8
                         gt='130X_mcRun3_2022_realistic_v5'),
    'Summer22EE'  : dict(mc_cfi='Run3_2022_LHC_Simulation_10h_2h_cfi',
                         periods={'2022': '2022EFG'}, confirmed=True,   # Bc nti: chi2/ndf 1.1
                         gt='130X_mcRun3_2022_realistic_postEE_v6'),
    'Summer23'    : dict(mc_cfi='mix_2023_25ns_EraC_PoissonOOTPU_cfi',
                         periods={'2023': '2023BC'}, confirmed=True,    # Bc nti: chi2/ndf 0.9
                         gt='130X_mcRun3_2023_realistic_v15'),
    'Summer23BPix': dict(mc_cfi='mix_2023_25ns_EraD_PoissonOOTPU_cfi',
                         periods={'2023': '2023D'}, confirmed=True,     # Bc nti: chi2/ndf 1.0
                         gt='130X_mcRun3_2023_realistic_postBPix_v6'),
    'Summer24'    : dict(mc_cfi='mix_2024_25ns_RunIII2024Summer24_PoissonOOTPU_cfi',
                         periods=dict((y, y) for y in SUMMER24_YEARS),
                         confirmed=True,                                # Bc nti, unskimmed MINIAOD: chi2/ndf 1.0
                         gt='150X_mcRun3_2024_realistic_v4'),
}