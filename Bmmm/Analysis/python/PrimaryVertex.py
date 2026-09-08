'''
Per-candidate primary vertex, shared by every channel.

The reference point for all wrt-PV quantities is a beamspot-constrained
AdaptiveVertexFitter refit of the chosen PV with the signal muons removed
(RJpsiKinVtxFitter.refitPVRemovingTracks, which reproduces the BPH-slides
PVRefitter). That replaces the Run2 hybrid PV (beamspot x,y + PV z) with a
properly formed reco::Vertex carrying its own 3D covariance and the beamspot
information.

This lived inside JpsiChargedCandidate. It is here so the dimuon channel uses
the SAME code rather than a second implementation that drifts: a dxy or
covariance correction measured on dimuon events only transfers to RJpsi if the
PV it is measured against is built the same way.

Usage -- mix into a candidate class that provides self.muons, self.pv and
self.pv_idx:

    class Candidate(PrimaryVertexRefitMixin):
        ...
        self.bs = build_beamspot_vertex(beamspot, self.pv.z())
        self.refit_primary_vertex(beamspot, pf, lost, vertices)
        self.pv_bs = self.pv_refit if self.pv_refit_valid \
                     else build_hybrid_pv(self.bs, self.pv)
'''

import ROOT
from PhysicsTools.HeppyCore.utils.deltar import deltaR

ROOT.gSystem.Load('libBmmmAnalysis')
from ROOT import RJpsiKinVtxFitter

# single fitter instance, shared by every candidate of every channel
pvfit = RJpsiKinVtxFitter()

# ----- signal-muon <-> track/candidate matching ----------------------------
# the muon best-track and the unpacked / packed candidate live in different
# collections, so they are matched by proximity, not by reference. Keep in sync
# with the C++ refitPVRemovingTracks defaults (drMatch / relPtMatch).
MU_TRK_DR_MATCH    = 0.01    # dR(cand, muon) match window
MU_TRK_RELPT_MATCH = 0.05    # |pt_cand - pt_mu| / pt_mu match window

# ----- PV-finder track filter (offlinePrimaryVertices TkFilterParameters) -----
# The closest-z PV association in refit_primary_vertex must be fed the same track
# set the offline PV reconstruction used; otherwise the refit input is broader
# (softer / lower-quality tracks, and all of lostTracks) than the real PV, which
# inflates pv_ntrk and degrades the refit resolution. These mirror the
# TrackFilterForPVFinding cuts of unsortedOfflinePrimaryVertices (the producer the
# old primaryVertexRefit cloned). VERIFY against your release, e.g.
#   print(process.primaryVertexRefit.TkFilterParameters.dumpPython())
PV_TRK_MAX_NORM_CHI2  = 10.0   # maxNormalizedChi2
PV_TRK_MIN_PIX_LAYERS = 2      # minPixelLayersWithHits
PV_TRK_MIN_TRK_LAYERS = 5      # minSiliconLayersWithHits (pixel + strip)
PV_TRK_MAX_D0_SIG     = 4.0    # maxD0Significance (transverse IP wrt the beamline)
PV_TRK_MAX_D0_ERR     = 1.0    # maxD0Error [cm]
PV_TRK_MAX_DZ_ERR     = 1.0    # maxDzError [cm]
PV_TRK_MIN_PT         = 0.0    # minPt [GeV]
PV_TRK_MAX_ETA        = 2.4

# ----- PV refit fitter config + vertex acceptance (WithBS vertexCollections) ---
# Match the offline PrimaryVertexProducer WithBS collection so the in-loop refit
# reproduces offlinePrimaryVerticesWithBS. VERIFY against your release dump.
PV_AVF_CHI2CUTOFF     = 2.5    # AdaptiveVertexFitter annealing cutoff (default is 3.0!)
PV_MIN_NDOF           = 2.0    # minNdof (WithBS); ndof = 2*sum(weights) - 3
PV_MAX_DIST_TO_BEAM   = 1.0    # maxDistanceToBeam [cm]

# ----- PV refit track-set selection -------------------------------------------
PV_USED_IN_FIT        = 3      # pat::PackedCandidate::PVAssociationQuality::PVUsedInFit
PV_REFIT_MIN_TRK      = 2      # minimum tracks to attempt a refit (AVF needs >= 2)
PV_REFIT_MAX_DZ_TO_PV = 0.1    # [cm] max |z(refit) - z(chosen PV)|; else fromPV
                               #      picked the wrong vertex -> closest-z fallback


def build_beamspot_vertex(beamspot, z):
    '''
    Build a reco::Vertex from the beamspot, evaluated at the given z, so it can
    be passed to the VertexDistance / IPTools tools alongside a vertex.
    '''
    bs_point = ROOT.reco.Vertex.Point(
        beamspot.x(z),
        beamspot.y(z),
        beamspot.z0(),
    )
    bs_error = beamspot.covariance3D()
    chi2 = 0.
    ndof = 0.
    return ROOT.reco.Vertex(bs_point, bs_error, chi2, ndof, 3)


def build_hybrid_pv(bs, pv):
    '''
    Hybrid primary vertex: beamspot transverse position + PV longitudinal
    position, with the PV covariance. The pre-refit convention, kept as the
    fallback when the refit is not available.
    '''
    return ROOT.reco.Vertex(
        ROOT.reco.Vertex.Point(bs.position().x(),   # beamspot x
                               bs.position().y(),   # beamspot y
                               pv.position().z()),  # PV z
        pv.error(), pv.chi2(), pv.ndof(), pv.tracksSize()
    )


def passes_pv_track_filter(cand, trk, beamspot):
    '''
    Approximate the offlinePrimaryVertices TkFilterParameters
    (TrackFilterForPVFinding) on a packed/lost candidate, so the PV refit is fed
    the track set the offline PV reconstruction would have used rather than every
    nearby candidate.

    cand : the pat::PackedCandidate (kinematics, IP, IP errors)
    trk  : cand.pseudoTrack(), passed in to avoid rebuilding it (chi2, layers)
    The caller must have checked cand.hasTrackDetails().

    Packed quantities are reduced-precision, so this is a close but not bit-exact
    replica; the layer accessors go through the pseudo-track hit pattern -- verify
    they are populated in your PackedCandidate version (pv_refit_valid collapsing
    to ~0 is the canary for a bad accessor).
    '''
    if abs(trk.eta()) > PV_TRK_MAX_ETA:
        return False

    if trk.normalizedChi2() > PV_TRK_MAX_NORM_CHI2:
        return False

    hp = trk.hitPattern()
    if hp.pixelLayersWithMeasurement()   < PV_TRK_MIN_PIX_LAYERS:
        return False
    if hp.trackerLayersWithMeasurement() < PV_TRK_MIN_TRK_LAYERS:
        return False

    if cand.pt() <= PV_TRK_MIN_PT:
        return False

    d0_err = cand.dxyError()
    dz_err = cand.dzError()
    if d0_err > PV_TRK_MAX_D0_ERR or dz_err > PV_TRK_MAX_DZ_ERR:
        return False

    # transverse IP significance wrt the beamline evaluated at the candidate z
    # (value recomputed against the beamspot; error is the stored packed dxyError)
    d0 = cand.dxy(beamspot.position(cand.vz()))
    if d0_err > 0. and abs(d0) / d0_err > PV_TRK_MAX_D0_SIG:
        return False

    return True


class PrimaryVertexRefitMixin(object):
    '''
    refit_primary_vertex + the signal-muon proximity match, for any candidate
    class exposing self.muons (the signal muons), self.pv (the chosen
    reco::Vertex) and self.pv_idx (its index in the vertex collection).
    '''

    def is_signal_muon_cand(self, cand,
                            dr_max=MU_TRK_DR_MATCH,
                            rel_pt_max=MU_TRK_RELPT_MATCH):
        '''
        True if the candidate `cand` (a PF candidate or a track) is one of the
        signal muons, matched by charge, dR and relative pt. The muon and the PF
        candidate are in different collections, so this is a proximity match,
        identical in spirit to the C++ PV-refit removal.
        '''
        for imu in self.muons:
            if cand.charge() != imu.charge():
                continue
            if (deltaR(imu.eta(), imu.phi(), cand.eta(), cand.phi()) < dr_max and
                    abs(cand.pt() - imu.pt()) < rel_pt_max * imu.pt()):
                return True
        return False

    def refit_primary_vertex(self, beamspot, pf, lost, vertices):
        '''
        Per-candidate primary vertex: AdaptiveVertexFitter refit of the chosen PV
        with the transverse beamspot constraint and the signal muons removed.

        The PV track set is rebuilt IN THE LOOP from the packed (pf) + lost (lost)
        candidates. Primary selection: the offline Deterministic-Annealing
        fit-track assignment, read directly from the packed-candidate PV
        association (fromPV(pv_idx) == PVUsedInFit) -- the tracks the offline PV
        fit actually used for the chosen PV, with no re-clustering and no
        closest-z approximation, O(1) per candidate, and already offline-quality
        (no re-filter applied). Fallback (if that yields too few tracks, or the
        refit lands at the wrong z): the closest-z PV association with the
        offline track-quality filter applied.

        Sets:
          self.pv_refit       : the refitted reco::Vertex (None on failure)
          self.pv_refit_valid : bool, True iff a valid refit was obtained
          self.pv_refit_ntrk  : number of tracks fed to the successful refit (0 else)

        On any failure -- pf not loaded, too few surviving tracks, fit failure --
        pv_refit_valid is False and the caller falls back to the hybrid PV, so
        behaviour without packed candidates is unchanged.
        '''
        self.pv_refit       = None
        self.pv_refit_valid = False
        self.pv_refit_ntrk  = 0

        if pf is None:
            return  # no packed candidates -> cannot rebuild the PV track set

        try:
            vtxs = list(vertices)
            if not vtxs:
                return

            # signal muons to remove from the refit. Same for either track
            # selection below, so build them once.
            mu_tracks = ROOT.std.vector('reco::Track')()
            for imu in self.muons:
                mu_tracks.push_back(imu.bestTrack())

            def _refit(selector, apply_filter):
                # collect the PV track set (selector, with optional quality filter),
                # run the beamspot-constrained AVF refit with the signal muons
                # removed, and return (VALID reco::Vertex, ntrk) or (None, ntrk).
                pv_tracks = ROOT.std.vector('reco::Track')()
                for coll in (pf, lost):
                    if coll is None:
                        continue
                    for cand in coll:
                        if not cand.hasTrackDetails():
                            continue
                        trk = cand.pseudoTrack()  # built once; filter + refit input
                        if apply_filter and not passes_pv_track_filter(cand, trk, beamspot):
                            continue
                        if selector(cand):
                            pv_tracks.push_back(trk)
                ntrk = pv_tracks.size()
                if ntrk < PV_REFIT_MIN_TRK:
                    return None, ntrk
                v = pvfit.refitPVRemovingTracks(
                    pv_tracks, mu_tracks, beamspot,
                    MU_TRK_DR_MATCH,      # drMatch
                    MU_TRK_RELPT_MATCH,   # relPtMatch
                    PV_AVF_CHI2CUTOFF,    # AVF annealing cutoff (offline WithBS)
                    PV_MIN_NDOF,          # minNdof (WithBS)
                    PV_MAX_DIST_TO_BEAM,  # maxDistanceToBeam [cm]
                )
                return (v if v.isValid() else None), ntrk

            # ---- primary: offline DA fit-track set, straight from fromPV --------
            # PVUsedInFit means the offline PV fit used this track for vertex
            # pv_idx; trust it as-is -- it already passed the offline
            # TkFilterParameters, so no closest-z and no re-filter.
            refit, ntrk = _refit(
                lambda cand: cand.fromPV(self.pv_idx) == PV_USED_IN_FIT,
                apply_filter=False)

            # ---- z-consistency guard ---------------------------------------------
            # the refit MUST sit on the chosen PV. A large |z(refit) - z(pv)| means
            # fromPV gathered a different vertex's tracks (PV-index misalignment
            # between the collection read here and the one the associations
            # reference). The track-count check inside _refit cannot catch this --
            # it returns plenty of (wrong) tracks -- so guard on z here and fall
            # back to the closest-z association + offline track-quality filter.
            if refit is None or abs(refit.z() - self.pv.z()) > PV_REFIT_MAX_DZ_TO_PV:
                vtx_z = [vv.position().z() for vv in vtxs]  # hoist PyROOT calls out
                def _closest_is_pv(cand):
                    vz = cand.vz()
                    return min(range(len(vtx_z)),
                               key=lambda i: abs(vtx_z[i] - vz)) == self.pv_idx
                # closest-z picks tracks nearest pv_idx by construction, so it is
                # not subject to the misalignment above; accept its result as-is.
                refit, ntrk = _refit(_closest_is_pv, apply_filter=True)

            if refit is not None:
                self.pv_refit       = refit
                self.pv_refit_valid = True
                self.pv_refit_ntrk  = ntrk
        except Exception:
            # any PyROOT / collection issue -> silent fall back to hybrid PV
            self.pv_refit_valid = False
