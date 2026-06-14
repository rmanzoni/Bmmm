#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
//#include "MagneticField/ParametrizedEngine/src/OAEParametrizedMagneticField.h"
#include "OAEParametrizedMagneticField.h"

#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticleFactoryFromTransientTrack.h"
#include "RecoVertex/KinematicFit/interface/TwoTrackMassKinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/KinematicConstrainedVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleFitter.h"
#include "RecoVertex/KinematicFit/interface/MassKinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicTree.h"
#include "RecoVertex/KinematicFitPrimitives/interface/Matrices.h"


// RM: dirrrrrrty
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"

// impact parameter tools (signed IP, decay length, ...)
// NOTE: requires <use name="TrackingTools/IPTools"/> in the package BuildFile.xml
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"

// beamspot-constrained primary-vertex refit, AdaptiveVertexFitter
// (RecoVertex/AdaptiveVertexFit + DataFormats/BeamSpot in the package BuildFile.xml).
// This mirrors the PVRefitter from the BPH vertexing+isolation slides (backup):
// AdaptiveVertexFitter + per-track setBeamSpot(bs) + fitter.vertex(tracks, bs).
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include <cmath>

#include <vector>
#include <cstddef>
#include <limits>
#include <utility>


class RJpsiKinVtxFitter {

  public:
    RJpsiKinVtxFitter() {};
    virtual ~RJpsiKinVtxFitter() {};

    // ------------------------------------------------------------------------
    // transient track helpers
    // ------------------------------------------------------------------------

    // constructed from reco::TrackRef
    reco::TransientTrack getTransientTrack(const reco::TrackRef& trackRef) {
      reco::TransientTrack transientTrack(trackRef, paramField);
      return transientTrack;
    }

    // constructed from reco::Track
    reco::TransientTrack getTransientTrack(const reco::Track& track) {
      reco::TransientTrack transientTrack(track, paramField);
      return transientTrack;
    }

    // ------------------------------------------------------------------------
    // general N-body vertex fit
    //
    // Fit an arbitrary number of tracks to a common vertex. The caller passes
    // the tracks together with their mass hypotheses, one mass per track:
    // tracks[i] is fitted with mass masses[i].
    // ------------------------------------------------------------------------
    RefCountedKinematicTree Fit(const std::vector<reco::Track>& tracks,
                                const std::vector<double>&      masses)
    {
        // one mass hypothesis per track, and at least two tracks for a vertex
        if (tracks.size() != masses.size()) return RefCountedKinematicTree();
        if (tracks.size() < 2)              return RefCountedKinematicTree();

        // factory that turns transient tracks into kinematic particles
        KinematicParticleFactoryFromTransientTrack pFactory;

        // the particles to be fitted to a common vertex
        std::vector<RefCountedKinematicParticle> toFit;
        toFit.reserve(tracks.size());

        // per-particle bookkeeping (same convention as the original code)
        float chi   = 0.0;
        float ndf   = 0.0;
        float sigma = 1e-6;

        for (std::size_t i = 0; i < tracks.size(); ++i) {
            ParticleMass mass = masses[i];
            toFit.push_back(pFactory.particle(getTransientTrack(tracks[i]), mass, chi, ndf, sigma));
        }

        // run the constrained vertex fit
        KinematicConstrainedVertexFitter kcvFitter;
        RefCountedKinematicTree tree = kcvFitter.fit(toFit);

        // return an empty tree if the fit failed
        if (tree == 0)        return RefCountedKinematicTree();
        if (!tree->isValid()) return RefCountedKinematicTree();

        return tree;
    };

    // ------------------------------------------------------------------------
    // convenience overloads (kept for backward compatibility)
    // they simply forward to the general Fit(...) above, so existing call
    // sites keep working unchanged
    // ------------------------------------------------------------------------

    // three-body fit, e.g. mu mu mu
    RefCountedKinematicTree Fit(const reco::Track & mu1,
                                const reco::Track & mu2,
                                const reco::Track & mu3,
                                const double      & mmu1,
                                const double      & mmu2,
                                const double      & mmu3)
    {
        return Fit(std::vector<reco::Track>{mu1, mu2, mu3},
                   std::vector<double>     {mmu1, mmu2, mmu3});
    };

    // two-body fit, e.g. mu mu
    RefCountedKinematicTree Fit2Body(const reco::Track & mu1,
                                     const reco::Track & mu2,
                                     const double      & mmu1,
                                     const double      & mmu2)
    {
        return Fit(std::vector<reco::Track>{mu1, mu2},
                   std::vector<double>     {mmu1, mmu2});
    };

    // ------------------------------------------------------------------------
    // two-body fit WITH an invariant-mass constraint on the pair
    //
    // The two tracks are fitted to a common vertex while their invariant mass
    // is constrained to massConstraint (e.g. the J/psi mass on mu mu). Uses the
    // simultaneous KinematicConstrainedVertexFitter + TwoTrackMassKinematicConstraint.
    //
    // The returned tree is structurally identical to the one from Fit(...):
    //   top particle        -> the mass-constrained pair (e.g. the J/psi)
    //   currentDecayVertex  -> the dimuon vertex
    //   the two children    -> the refitted muons, with momenta consistent with
    //                          BOTH the common vertex and the mass constraint
    // so it is a drop-in replacement for Fit2Body and all downstream handling
    // (is_good_vtx, displacement, ...) is unchanged. Returns an empty tree on
    // failure.
    // ------------------------------------------------------------------------
    RefCountedKinematicTree Fit2BodyMassConstraint(const reco::Track & mu1,
                                                   const reco::Track & mu2,
                                                   const double      & mmu1,
                                                   const double      & mmu2,
                                                   const double      & massConstraint)
    {
        // build the two kinematic particles from the transient tracks
        KinematicParticleFactoryFromTransientTrack pFactory;

        float chi   = 0.0;
        float ndf   = 0.0;
        float sigma = 1e-6;

        std::vector<RefCountedKinematicParticle> toFit;
        toFit.reserve(2);

        ParticleMass m1 = mmu1;
        ParticleMass m2 = mmu2;
        toFit.push_back(pFactory.particle(getTransientTrack(mu1), m1, chi, ndf, sigma));
        toFit.push_back(pFactory.particle(getTransientTrack(mu2), m2, chi, ndf, sigma));

        // invariant-mass constraint on the two-track system (e.g. J/psi)
        ParticleMass mc = massConstraint;
        TwoTrackMassKinematicConstraint constraint(mc);

        // simultaneous common-vertex + mass-constrained fit
        KinematicConstrainedVertexFitter kcvFitter;
        RefCountedKinematicTree tree = kcvFitter.fit(toFit, &constraint);

        if (tree == 0)        return RefCountedKinematicTree();
        if (!tree->isValid()) return RefCountedKinematicTree();

        return tree;
    };

    // line-to-line distance: the impact parameter of the track w.r.t. the axis
    std::pair<double, Measurement1D> jetTrackDistance(const reco::Track & track,
                                                      const double & dirx, const double & diry, const double & dirz,
                                                      const reco::Vertex& vertex)
    {
        GlobalVector direction(dirx, diry, dirz);
        std::pair<double, Measurement1D> res =
            IPTools::jetTrackDistance(getTransientTrack(track), direction, vertex);
        return res;   // raw IPTools result (.first along-axis, .second line-to-line); sign handled Python-side
    }
    
    // ------------------------------------------------------------------------
    // impact parameters (thin wrappers around IPTools)
    //
    //   track            : the track whose IP we want (e.g. the bachelor muon)
    //   dirx/diry/dirz   : reference direction used to lifetime-sign the IP
    //                      (e.g. the B flight direction, PV -> SV). Passed as
    //                      three doubles so nothing special is needed Python-side.
    //   vertex           : reference vertex (e.g. the PV or the J/psi vertex)
    //
    // each returns a Measurement1D, with .value() / .error() / .significance();
    // a NaN-valued Measurement1D is returned if the track extrapolation fails.
    // ------------------------------------------------------------------------

    // lifetime-signed 3D impact parameter (helical extrapolation to the vertex)
    Measurement1D signedIP3D(const reco::Track & track,
                             const double      & dirx,
                             const double      & diry,
                             const double      & dirz,
                             const reco::Vertex& vertex)
    {
        GlobalVector direction(dirx, diry, dirz);
        std::pair<bool, Measurement1D> res =
            IPTools::signedImpactParameter3D(getTransientTrack(track), direction, vertex);
        return res.first ? res.second : nanMeasurement();
    };

    // lifetime-signed transverse (2D) impact parameter
    Measurement1D signedIP2D(const reco::Track & track,
                             const double      & dirx,
                             const double      & diry,
                             const double      & dirz,
                             const reco::Vertex& vertex)
    {
        GlobalVector direction(dirx, diry, dirz);
        std::pair<bool, Measurement1D> res =
            IPTools::signedTransverseImpactParameter(getTransientTrack(track), direction, vertex);
        return res.first ? res.second : nanMeasurement();
    };

    // signed 3D decay length: projection of the flight onto the direction
    // (the longitudinal complement of the impact parameter)
    Measurement1D signedDecayLength3D(const reco::Track & track,
                                      const double      & dirx,
                                      const double      & diry,
                                      const double      & dirz,
                                      const reco::Vertex& vertex)
    {
        GlobalVector direction(dirx, diry, dirz);
        std::pair<bool, Measurement1D> res =
            IPTools::signedDecayLength3D(getTransientTrack(track), direction, vertex);
        return res.first ? res.second : nanMeasurement();
    };

    // ------------------------------------------------------------------------
    // beamspot-constrained primary-vertex refit, with a set of tracks removed
    //
    // Reproduces the PVRefitter from the BPH vertexing+isolation slides (backup
    // slide): AdaptiveVertexFitter, the transverse beamspot constraint applied
    // BOTH per track (setBeamSpot) and to the fit (vertex(tracks, beamspot)),
    // returning a reco::Vertex via the TransientVertex conversion (its 3D
    // position + covariance come from the fit).
    //
    //   pv              : the primary vertex to refit. Its OWN constituent
    //                     tracks (weight >= minWeight) are the refit input, so
    //                     pv must still carry its track references -- true for
    //                     the freshly-run primaryVertexRefit:WithBS, false for
    //                     the slimmed offlinePrimaryVertices, which drops them.
    //   tracksToRemove  : tracks to drop from the refit (the signal muons). A PV
    //                     track is removed if it matches one of these in charge,
    //                     dR < drMatch and |dpt| < relPtMatch * pt -- a proximity
    //                     match, since the muon best-track and the unpacked PV
    //                     track live in different collections.
    //   beamspot        : the transverse beamspot constraint.
    //
    // Returns an INVALID reco::Vertex on failure (fewer than two surviving
    // tracks or an invalid fit); check .isValid() on the Python side and fall
    // back to the hybrid/bare-beamspot PV there.
    // ------------------------------------------------------------------------
    reco::Vertex refitPVRemovingTracks(const reco::Vertex&             pv,
                                       const std::vector<reco::Track>& tracksToRemove,
                                       const reco::BeamSpot&           beamspot,
                                       const double                    minWeight  = 0.5,
                                       const double                    drMatch    = 0.01,
                                       const double                    relPtMatch = 0.05)
    {
        std::vector<reco::TransientTrack> ttks;

        for (reco::Vertex::trackRef_iterator it = pv.tracks_begin(); it != pv.tracks_end(); ++it) {
            if (pv.trackWeight(*it) < minWeight) continue;

            const reco::Track& trk = **it;

            bool is_removed = false;
            for (std::size_t i = 0; i < tracksToRemove.size(); ++i) {
                const reco::Track& m = tracksToRemove[i];
                if (trk.charge() != m.charge()) continue;
                const double deta = trk.eta() - m.eta();
                const double dphi = reco::deltaPhi(trk.phi(), m.phi());
                const double dr   = std::sqrt(deta * deta + dphi * dphi);
                if (dr < drMatch && std::fabs(trk.pt() - m.pt()) < relPtMatch * m.pt()) {
                    is_removed = true;
                    break;
                }
            }
            if (is_removed) continue;

            reco::TransientTrack tt = getTransientTrack(trk);
            tt.setBeamSpot(beamspot);                        // per-track BS (slides)
            ttks.push_back(tt);
        }

        if (ttks.size() < 2) return reco::Vertex();          // invalid

        AdaptiveVertexFitter fitter;
        TransientVertex tv = fitter.vertex(ttks, beamspot);  // beamspot constraint
        if (!tv.isValid()) return reco::Vertex();            // invalid

        return reco::Vertex(tv);                             // TransientVertex -> reco::Vertex
    };

  private:
    OAEParametrizedMagneticField *paramField = new OAEParametrizedMagneticField("3_8T");

    // Measurement1D filled with NaNs, returned when an IP extrapolation fails
    static Measurement1D nanMeasurement() {
        double nan = std::numeric_limits<double>::quiet_NaN();
        return Measurement1D(nan, nan);
    }

};
