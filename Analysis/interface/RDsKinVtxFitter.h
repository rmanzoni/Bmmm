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
#include "RecoVertex/KinematicFit/interface/TwoTrackMassKinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/KinematicConstrainedVertexFitter.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicTree.h"

#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"

#include <vector>
#include <exception>

// ----------------------------------------------------------------------------
// Sequential kinematic fits of Bs -> Ds(-> phi(-> K K) pi) mu (nu).
//
// Three schemes, run side by side on the same tracks so they can be compared
// on MC (RDsCandidate):
//
//   FitPhiConstrained   [phi, ds_phic, bs_phic, ds_cons, bs_cons]
//     phi     : K K with the phi mass constraint, shared by the two chains
//     -- phic: phi constraint only
//     ds_phic : phi + pi, no constraint     -> m(KKpi) with the phi constraint
//                                              (the AN v4 / previous ntuples)
//     bs_phic : ds_phic + mu, no constraint (neutrinos)
//     -- cons: phi and Ds constraints
//     ds_cons : phi + pi with the Ds mass constraint
//     bs_cons : ds_cons + mu, no constraint (neutrinos)
//
//   FitFree             [kk, kkpi, bs]
//     kk      : K K, no constraint
//     kkpi    : KK + pi, no constraint
//     bs      : kkpi + mu, no constraint
//
// Every returned vector has the full length; a failed step and every step
// after it in the same chain are empty trees (test with tree.get() /
// isValid() on the Python side). A fit that throws (KinematicFit raises on degenerate input) is a
// failed step, not a dead job.
// ----------------------------------------------------------------------------

class RDsKinVtxFitter {

  public:
    RDsKinVtxFitter() {};
    virtual ~RDsKinVtxFitter() {};

    reco::TransientTrack getTransientTrack(const reco::Track& track) {
      return reco::TransientTrack(track, paramField);
    }

    std::vector<RefCountedKinematicTree> FitPhiConstrained(const reco::Track & k1,
                                                           const reco::Track & k2,
                                                           const reco::Track & pi,
                                                           const reco::Track & mu,
                                                           const double      & mk,
                                                           const double      & mpi,
                                                           const double      & mmu,
                                                           const double      & mphi,
                                                           const double      & mds)
    {
        std::vector<RefCountedKinematicTree> out(5, RefCountedKinematicTree());

        // phi: K K, phi mass constraint
        std::vector<RefCountedKinematicParticle> kk = {particle(k1, mk), particle(k2, mk)};
        ParticleMass phiMass = mphi;  // the constraint ctor takes a non-const reference
        TwoTrackMassKinematicConstraint phiConstr(phiMass);
        out[0] = fit(kk, &phiConstr);
        if (!good(out[0])) return out;
        RefCountedKinematicParticle phi = top(out[0]);

        // phic: phi + pi without the Ds constraint, then + mu
        std::vector<RefCountedKinematicParticle> dsPhicIn = {particle(pi, mpi), phi};
        out[1] = fit(dsPhicIn, nullptr);
        if (good(out[1])) {
            std::vector<RefCountedKinematicParticle> bsPhicIn = {particle(mu, mmu), top(out[1])};
            out[2] = fit(bsPhicIn, nullptr);
        }

        // cons: phi + pi with the Ds constraint, then + mu
        std::vector<RefCountedKinematicParticle> dsConsIn = {particle(pi, mpi), phi};
        ParticleMass dsMass = mds;
        TwoTrackMassKinematicConstraint dsConstr(dsMass);
        out[3] = fit(dsConsIn, &dsConstr);
        if (!good(out[3])) return out;
        std::vector<RefCountedKinematicParticle> bsConsIn = {particle(mu, mmu), top(out[3])};
        out[4] = fit(bsConsIn, nullptr);
        return out;
    }

    std::vector<RefCountedKinematicTree> FitFree(const reco::Track & k1,
                                                 const reco::Track & k2,
                                                 const reco::Track & pi,
                                                 const reco::Track & mu,
                                                 const double      & mk,
                                                 const double      & mpi,
                                                 const double      & mmu)
    {
        std::vector<RefCountedKinematicTree> out(3, RefCountedKinematicTree());

        std::vector<RefCountedKinematicParticle> kk = {particle(k1, mk), particle(k2, mk)};
        out[0] = fit(kk, nullptr);
        if (!good(out[0])) return out;
        RefCountedKinematicParticle kkp = top(out[0]);

        std::vector<RefCountedKinematicParticle> kkpiIn = {particle(pi, mpi), kkp};
        out[1] = fit(kkpiIn, nullptr);
        if (!good(out[1])) return out;
        RefCountedKinematicParticle kkpi = top(out[1]);

        std::vector<RefCountedKinematicParticle> bsIn = {particle(mu, mmu), kkpi};
        out[2] = fit(bsIn, nullptr);
        return out;
    }

  private:
    OAEParametrizedMagneticField *paramField = new OAEParametrizedMagneticField("3_8T");

    RefCountedKinematicParticle particle(const reco::Track & trk, const double & mass) {
        KinematicParticleFactoryFromTransientTrack pFactory;
        float chi = 0.0, ndf = 0.0, sigma = 1e-6;
        ParticleMass m = mass;
        return pFactory.particle(getTransientTrack(trk), m, chi, ndf, sigma);
    }

    // nullptr constraint -> plain common-vertex fit
    RefCountedKinematicTree fit(std::vector<RefCountedKinematicParticle> & parts,
                                MultiTrackKinematicConstraint * constraint) {
        try {
            KinematicConstrainedVertexFitter kcvFitter;
            RefCountedKinematicTree tree = constraint ? kcvFitter.fit(parts, constraint)
                                                      : kcvFitter.fit(parts);
            if (tree == 0 || tree->isEmpty() || !tree->isValid()) return RefCountedKinematicTree();
            return tree;
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
