'''
Hit content of a track: ONE definition for every place that needs it.

Two consumers must agree on these numbers exactly:

  * the ntuple branches <obj>_<name> (CommonBranches.muon_branches), which is
    what a covflow run is TRAINED on, and
  * utils.COVFLOW_CONTEXT_GETTERS, which is what the same flow is CONDITIONED on
    when it is APPLIED at ntuplization time.

Before this module they were two hand-written copies of the same getters, kept
in step by a comment. A flow conditioned on a quantity that means one thing in
training and another at application gives a well-formed, wrong correction that
no downstream check would catch -- so the definition now lives here only, and
both sides are built from it.

What is measured, and on which track
------------------------------------
Everything is read from obj.bestTrack(), the track whose covariance is written
out as <obj>_cov_* and corrected by covflow. For the J/psi-momentum muons of
these analyses that is the tracker-only track: muon::sigmaSwitch/tevOptimized
(DataFormats/MuonReco/src/MuonCocktails.cc) return the tracker fit below
ptThresholdToFillCandidateP4WithGlobalFit = 200 GeV. The <obj>_best_trk_type
branch records the choice per muon so that statement is checkable in every file.

It is NOT read from the rebuilt track fit_track() hands to the fitters under
--covflow / --cov-scale: utils.track_with_cov builds a new reco::Track without a
hit pattern, so its hitPattern() is empty. Hit content always comes from
bestTrack(), whatever the fits used.

The quantities (Phase-1 pixel detector: 4 BPix layers, 3 FPix disks per side)

  n_pix_hit, n_pix_b_hit, n_pix_e_hit
        valid pixel HITS (overlaps count twice), unchanged from before
  n_pix_layer        pixel LAYERS with a measurement (overlaps count once)
  n_trk_layer        tracker (pixel + strip) layers with a measurement
  pix_first_b_layer  innermost BPix layer with a valid hit, 1-4; 0 if none
  pix_first_e_disk   innermost FPix disk with a valid hit, 1-3; 0 if none
  pix_first_layer    the innermost pixel measurement as ONE code:
                        1-4  BPix layer 1-4
                        5-7  FPix disk 1-3, for a track with no BPix hit
                        0    no valid pixel hit
                     A track with any BPix hit is coded by its barrel layer: the
                     barrel ends before the first disk in |z|, so along a track
                     from the luminous region the barrel hits come first.
  n_pix_miss_inner   pixel layers expected BEFORE the first valid hit, on an
                     ACTIVE module, with no hit found (MISSING_INNER_HITS,
                     type MISSING): inefficiency
  n_pix_inact_inner  same, but the module was flagged inactive or bad in the
                     conditions (MISSING_INNER_HITS, type INACTIVE): dead module
        The two are set by TrackProducerBase::setSecondHitPattern from
        MeasurementDet::isActive() / hasBadComponents(). Their split is what
        separates "the track starts at L2 because L1 is dead there" from "the
        track starts at L2 because L1 missed it". A MC produced with conditions
        that do not know a data-taking dead module can only ever say MISSING or
        nothing -- which is exactly the kind of mismatch they are here to show.

Which of these may condition a flow is decided in COVFLOW_HIT_CONTEXT below, not
by their presence in the ntuple: the missing/inactive counts describe WHY the
first hit is where it is, the covariance depends on WHERE it is, so they are
saved as diagnostics and deliberately kept out of the context set.

pat::PackedCandidate caveat
---------------------------
For a packed candidate bestTrack() is a pseudo-track whose hit pattern is
SYNTHESISED (DataFormats/PatCandidates/src/PackedCandidate.cc): the first hit is
exact, the remaining layers are filled in sequence from the stored counts, and
the inner category holds only lostInnerHits() as MISSING, never INACTIVE. The
getters run on it, but only pix_first_* keep their meaning there. This module is
therefore wired into the MUON block only.
'''

import ROOT

# PixelSubdetector::PixelBarrel / PixelEndcap
# (DataFormats/SiPixelDetId/interface/PixelSubdetector.h). Spelled as literals
# because PixelSubdetector is a bare enum holder without a ROOT dictionary.
PXB = 1
PXF = 2

N_BPIX_LAYERS = 4
N_FPIX_DISKS  = 3
FPIX_CODE_OFFSET = N_BPIX_LAYERS   # disk d -> code 4 + d in pix_first_layer

# The order is the branch order in the ntuple. The first three are the branches
# that existed before this module and keep their position.
HIT_CONTENT_NAMES = (
    'n_pix_hit'        ,
    'n_pix_b_hit'      ,
    'n_pix_e_hit'      ,
    'n_pix_layer'      ,
    'n_trk_layer'      ,
    'pix_first_b_layer',
    'pix_first_e_disk' ,
    'pix_first_layer'  ,
    'n_pix_miss_inner' ,
    'n_pix_inact_inner',
)

# The subset a covflow context may use. Geometric and upstream of the hit
# errors: they fix the lever arm and the extrapolation distance to the beamline
# without being computed from the (possibly mismodelled) uncertainties.
COVFLOW_HIT_CONTEXT = (
    'n_pix_hit'        ,
    'n_pix_b_hit'      ,
    'n_pix_e_hit'      ,
    'n_pix_layer'      ,
    'n_trk_layer'      ,
    'pix_first_b_layer',
    'pix_first_e_disk' ,
    'pix_first_layer'  ,
)


def pix_first_layer_code(first_b_layer, first_e_disk):
    '''The pix_first_layer code from its two ingredients (see module doc).'''
    if first_b_layer:
        return first_b_layer
    if first_e_disk:
        return FPIX_CODE_OFFSET + first_e_disk
    return 0


def compute_hit_content(trk):
    '''All HIT_CONTENT_NAMES for one reco::Track, in one pass over its hits.

    Raises ValueError on a pixel layer outside the Phase-1 geometry rather than
    writing a code no one can decode: through safe_get that becomes a NaN branch
    (visible), through the covflow corrector a counted, printed failure.
    '''
    HP = ROOT.reco.HitPattern
    hp = trk.hitPattern()

    # innermost valid BPix layer / FPix disk. The minimum is taken rather than
    # the first hit met: nothing below relies on the storage order of the hits.
    first_b = 0
    first_e = 0
    for i in range(hp.numberOfAllHits(HP.TRACK_HITS)):
        pattern = hp.getHitPattern(HP.TRACK_HITS, i)
        if not (HP.validHitFilter(pattern) and HP.pixelHitFilter(pattern)):
            continue
        layer = int(HP.getLayer(pattern))
        if HP.pixelBarrelHitFilter(pattern):
            if not 1 <= layer <= N_BPIX_LAYERS:
                raise ValueError('valid BPix hit on layer %d, outside 1-%d'
                                 % (layer, N_BPIX_LAYERS))
            first_b = layer if first_b == 0 else min(first_b, layer)
        else:
            if not 1 <= layer <= N_FPIX_DISKS:
                raise ValueError('valid FPix hit on disk %d, outside 1-%d'
                                 % (layer, N_FPIX_DISKS))
            first_e = layer if first_e == 0 else min(first_e, layer)

    # inactive pixel modules the track was expected to cross before its first
    # valid hit. HitPattern has a counter for the MISSING ones in this category
    # (numberOfLostPixelHits) but none for the INACTIVE ones, hence the loop.
    n_inact_inner = 0
    for i in range(hp.numberOfAllHits(HP.MISSING_INNER_HITS)):
        pattern = hp.getHitPattern(HP.MISSING_INNER_HITS, i)
        if HP.pixelHitFilter(pattern) and HP.inactiveHitFilter(pattern):
            n_inact_inner += 1

    return {
        'n_pix_hit'         : hp.numberOfValidPixelHits()                         ,
        'n_pix_b_hit'       : hp.numberOfValidPixelBarrelHits()                   ,
        'n_pix_e_hit'       : hp.numberOfValidPixelEndcapHits()                   ,
        'n_pix_layer'       : hp.pixelLayersWithMeasurement()                     ,
        'n_trk_layer'       : hp.trackerLayersWithMeasurement()                   ,
        'pix_first_b_layer' : first_b                                             ,
        'pix_first_e_disk'  : first_e                                             ,
        'pix_first_layer'   : pix_first_layer_code(first_b, first_e)              ,
        'n_pix_miss_inner'  : hp.numberOfLostPixelHits(HP.MISSING_INNER_HITS)     ,
        'n_pix_inact_inner' : n_inact_inner                                       ,
    }


def hit_content(obj):
    '''compute_hit_content(obj.bestTrack()), memoized on the object.

    Same convention as obj.cov: the muons are shared across the candidates of an
    event and several branches (and the covflow context) read the same pass, so
    it runs once per object per event.
    '''
    cached = getattr(obj, '_hit_content', None)
    if cached is None:
        cached = compute_hit_content(obj.bestTrack())
        obj._hit_content = cached
    return cached


# <obj>_<name> getters, in HIT_CONTENT_NAMES order, taking the pat object
hit_content_branches = {}
for _name in HIT_CONTENT_NAMES:
    hit_content_branches[_name] = (lambda obj, name=_name : hit_content(obj)[name])
