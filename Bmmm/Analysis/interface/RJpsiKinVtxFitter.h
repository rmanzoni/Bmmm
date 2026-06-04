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

  private:
    OAEParametrizedMagneticField *paramField = new OAEParametrizedMagneticField("3_8T");

    // Measurement1D filled with NaNs, returned when an IP extrapolation fails
    static Measurement1D nanMeasurement() {
        double nan = std::numeric_limits<double>::quiet_NaN();
        return Measurement1D(nan, nan);
    }

};
