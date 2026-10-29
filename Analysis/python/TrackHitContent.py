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

Per-layer hit masks (for the hit-loss emulation of covflow, HITEMU.md)
  The quantities above say where the FIRST pixel hit is. The emulation that
  gives the MC the data hit pattern needs every layer: whether L2 is there
  after L1 is removed, and one kill map per layer and disk, which needs to know
  for every track and every layer whether it has a valid hit. Bit b of each
  mask is one pixel layer or disk:

        bit 0-3   BPix layer 1-4          (value 1, 2, 4, 8)
        bit 4-6   FPix disk 1-3           (value 16, 32, 64)
                  (either side: a track crosses the disks of one side only;
                   the side is the sign of its eta)

  pix_valid_mask   layers with at least one VALID hit
  pix_miss_mask    layers the track was expected to cross (MISSING hit in
                   TRACK_HITS, MISSING_INNER_HITS or MISSING_OUTER_HITS) on an
                   ACTIVE module, with no valid hit on that layer: inefficiency
  pix_inact_mask   same, module INACTIVE or BAD in the conditions, no valid hit
                   on that layer: dead module
  pix_hit_count    valid hits per layer, 2 bits per layer (0-3, overlaps make
                   2), packed: count of layer bit b = (pix_hit_count >> 2b) & 3

  A layer can be in pix_miss_mask and pix_inact_mask at once (two overlapping
  modules, one missing and one inactive), never in either and pix_valid_mask.
  A layer in none of the three was not crossed, or crossed outside what the
  pattern recorder considered (the emulation then uses the helix crossing).
  Masks are stored in the float32 ntuple branches: exact up to 2^24, i.e.
  every value here. Decode in numpy with  (mask.astype(int) >> b) & 1.
  Python helpers: pix_layer_bit(), PIX_MASK_LAYERS.

Which of these may condition a flow is decided in COVFLOW_HIT_CONTEXT below, not
by their presence in the ntuple: the missing/inactive counts describe WHY the
first hit is where it is, the covariance depends on WHERE it is, so they are
saved as diagnostics and deliberately kept out of the context set. The same
holds for the per-layer masks below: they are inputs to the hit-loss emulation,
not covflow context.

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

# bit of each pixel layer / disk in the pix_*_mask branches, in bit order
PIX_MASK_LAYERS = tuple(['L%d' % l for l in range(1, N_BPIX_LAYERS + 1)] +
                        ['D%d' % d for d in range(1, N_FPIX_DISKS + 1)])
PIX_COUNT_BITS  = 2   # bits per layer in pix_hit_count
PIX_COUNT_MAX   = (1 << PIX_COUNT_BITS) - 1


def pix_layer_bit(barrel, layer):
    '''Bit of a pixel layer in the pix_*_mask branches: BPix layer l -> l-1,
    FPix disk d -> N_BPIX_LAYERS + d - 1. Raises ValueError outside Phase-1.'''
    if barrel:
        if not 1 <= layer <= N_BPIX_LAYERS:
            raise ValueError('BPix layer %d outside 1-%d' % (layer, N_BPIX_LAYERS))
        return layer - 1
    if not 1 <= layer <= N_FPIX_DISKS:
        raise ValueError('FPix disk %d outside 1-%d' % (layer, N_FPIX_DISKS))
    return N_BPIX_LAYERS + layer - 1

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
    'pix_valid_mask'   ,
    'pix_miss_mask'    ,
    'pix_inact_mask'   ,
    'pix_hit_count'    ,
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

    # innermost valid BPix layer / FPix disk, and the per-layer masks. The
    # minimum is taken rather than the first hit met: nothing below relies on
    # the storage order of the hits.
    first_b = 0
    first_e = 0
    valid_mask = 0
    miss_mask  = 0
    inact_mask = 0
    counts = [0] * len(PIX_MASK_LAYERS)
    for category in (HP.TRACK_HITS, HP.MISSING_INNER_HITS, HP.MISSING_OUTER_HITS):
        for i in range(hp.numberOfAllHits(category)):
            pattern = hp.getHitPattern(category, i)
            if not HP.pixelHitFilter(pattern):
                continue
            barrel = bool(HP.pixelBarrelHitFilter(pattern))
            layer = int(HP.getLayer(pattern))
            if HP.validHitFilter(pattern):
                if category != HP.TRACK_HITS:
                    continue          # cannot happen; never trust it if it does
                if barrel:
                    if not 1 <= layer <= N_BPIX_LAYERS:
                        raise ValueError('valid BPix hit on layer %d, outside 1-%d'
                                         % (layer, N_BPIX_LAYERS))
                    first_b = layer if first_b == 0 else min(first_b, layer)
                else:
                    if not 1 <= layer <= N_FPIX_DISKS:
                        raise ValueError('valid FPix hit on disk %d, outside 1-%d'
                                         % (layer, N_FPIX_DISKS))
                    first_e = layer if first_e == 0 else min(first_e, layer)
                bit = pix_layer_bit(barrel, layer)
                valid_mask |= 1 << bit
                counts[bit] = min(counts[bit] + 1, PIX_COUNT_MAX)
            elif HP.inactiveHitFilter(pattern) or HP.badHitFilter(pattern):
                inact_mask |= 1 << pix_layer_bit(barrel, layer)
            elif HP.missingHitFilter(pattern):
                miss_mask |= 1 << pix_layer_bit(barrel, layer)
    # a layer with a valid hit is not "missing" because an overlapping module
    # on it was missed
    miss_mask  &= ~valid_mask
    inact_mask &= ~valid_mask
    hit_count = 0
    for bit, n in enumerate(counts):
        hit_count |= n << (PIX_COUNT_BITS * bit)

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
        'pix_valid_mask'    : valid_mask                                          ,
        'pix_miss_mask'     : miss_mask                                           ,
        'pix_inact_mask'    : inact_mask                                          ,
        'pix_hit_count'     : hit_count                                           ,
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


##########################################################################################
#####      BEST-TRACK KINEMATICS (covflow context pt / eta)
##########################################################################################
# The covflow context pt and eta must be the SAME quantity in training and in
# application. Application has always read them from bestTrack() (the track
# whose covariance is corrected); training used <obj>_pt / <obj>_eta, i.e. the
# pat::Muon p4 -- equal for tracker-fit muons, but two definitions of one
# variable. These branches give training the bestTrack() values, and
# utils.COVFLOW_CONTEXT_GETTERS builds 'pt', 'log_pt', 'eta' and 'abs_eta' from
# these very getters. Train on <obj>_best_trk_pt / <obj>_best_trk_eta.
best_track_kinematics_branches = {
    'best_trk_pt'  : lambda obj : obj.bestTrack().pt() ,
    'best_trk_eta' : lambda obj : obj.bestTrack().eta(),
}
