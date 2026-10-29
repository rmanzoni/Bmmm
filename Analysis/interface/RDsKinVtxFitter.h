#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "OAEParametrizedMagneticField.h"

#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticleFactoryFromTransientTrack.h"
#include "RecoVertex/KinematicFit/interface/MassKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleFitter.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleVertexFitter.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicTree.h"

#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"

#include <vector>
#include <exception>

// ----------------------------------------------------------------------------
// Kinematic fits of Bs -> Ds(-> phi(-> K K) pi) mu (nu), one step per call:
//
//   FitTwoTracks(k1, k2, mk, mconstr)   K K, with the mass constraint mconstr
//                                       (mconstr <= 0: no constraint)
//   AddTrack(tree, trk, m, mconstr)     the top particle of `tree` + trk to a
//                                       common vertex, with the mass constraint
//                                       mconstr on the pair (<= 0: none)
//
// Each step is the sequential recipe of the KinematicFit package (Prokofiev and
// Speer): a Kalman vertex fit of the inputs (KinematicParticleVertexFitter),
// then, for a mass constraint, KinematicParticleFitter with
// MassKinematicConstraint on the resulting particle. The constrained particle
// is what enters the next step. The constraint updates the top particle and its
// decay vertex, whose chi2 and ndof become vertex + mass (2 ndof for two
// inputs); the daughters keep their vertex-fit momenta.
//
// It replaces KinematicConstrainedVertexFitter (one Lagrange-multiplier fit of
// vertex and constraint), whose unconstrained fits broke on nearly collinear
// inputs: it starts the vertex from a 100 cm^2 variance and gets the ~1e-6 cm^2
// covariance as a difference of terms of that size, so on the full signal MC
// ~20% of the Bs vertices and ~15% of the free KK and KK pi ones came back
// valid with a negative chi2 or a covariance that was not positive definite.
// With the Kalman fit there are none.
//
// RDsCandidate chains them into three schemes (phic: phi constraint only; cons:
// phi and Ds constraints; free: none), each KK -> KK pi -> KK pi mu.
//
// It reads every tree BEFORE the tree enters the next fit: that fit updates,
// in place, the top particle of the tree it is built from (the Ds of the Ds
// tree comes back refitted at the Bs vertex, i.e. with the muon information
// in it), so a tree read afterwards would not show its own fit. The vertex
// position is not changed.
//
// For the same reason a composite must enter only ONE further fit (feeding one
// phi to two Ds fits rewired the first chain): cons runs its own phi fit.
//
// A failed fit, a fit that throws (KinematicFit raises on degenerate input)
// or a call on an empty input tree returns an empty tree (test with
// tree.get() / isValid() on the Python side), never a dead job.
// ----------------------------------------------------------------------------

class RDsKinVtxFitter {

  public:
    RDsKinVtxFitter() {};
    virtual ~RDsKinVtxFitter() {};

    reco::TransientTrack getTransientTrack(const reco::Track& track) {
      return reco::TransientTrack(track, paramField);
    }

    RefCountedKinematicTree FitTwoTracks(const reco::Track & k1,
                                         const reco::Track & k2,
                                         const double      & mk,
                                         const double      & mconstr)
    {
        std::vector<RefCountedKinematicParticle> parts = {particle(k1, mk), particle(k2, mk)};
        return fitWithMass(parts, mconstr);
    }

    RefCountedKinematicTree AddTrack(RefCountedKinematicTree & tree,
                                     const reco::Track       & trk,
                                     const double            & m,
                                     const double            & mconstr)
    {
        if (!good(tree)) return RefCountedKinematicTree();
        std::vector<RefCountedKinematicParticle> parts = {particle(trk, m), top(tree)};
        return fitWithMass(parts, mconstr);
    }

  private:
    OAEParametrizedMagneticField *paramField = new OAEParametrizedMagneticField("3_8T");

    RefCountedKinematicParticle particle(const reco::Track & trk, const double & mass) {
        KinematicParticleFactoryFromTransientTrack pFactory;
        float chi = 0.0, ndf = 0.0, sigma = 1e-6;
        ParticleMass m = mass;
        return pFactory.particle(getTransientTrack(trk), m, chi, ndf, sigma);
    }

    // width of the mass constraints: effectively exact, far below any resolution
    static constexpr float MASS_CONSTRAINT_SIGMA = 1e-5;   // GeV

    // vertex fit, then the mass constraint on the top particle if mconstr > 0
    RefCountedKinematicTree fitWithMass(std::vector<RefCountedKinematicParticle> & parts,
                                        const double & mconstr) {
        try {
            KinematicParticleVertexFitter vtxFitter;
            RefCountedKinematicTree tree = vtxFitter.fit(parts);
            if (!good(tree)) return RefCountedKinematicTree();
            if (mconstr <= 0.) return tree;
            ParticleMass mass = mconstr;
            MassKinematicConstraint constr(mass, MASS_CONSTRAINT_SIGMA);
            KinematicParticleFitter constrFitter;
            tree = constrFitter.fit(&constr, tree);
            return good(tree) ? tree : RefCountedKinematicTree();
        } catch (std::exception & e) {
            return RefCountedKinematicTree();
        } catch (...) {
            return RefCountedKinematicTree();
        }
    }

    static bool good(const RefCountedKinematicTree & tree) {
        return tree.get() != nullptr && !tree->isEmpty() && tree->isValid();
    }

    static RefCountedKinematicParticle top(RefCountedKinematicTree & tree) {
        tree->movePointerToTheTop();
        return tree->currentParticle();
    }
};
