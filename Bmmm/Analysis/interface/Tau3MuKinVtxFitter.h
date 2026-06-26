#ifndef Bmmm_Analysis_Tau3MuKinVtxFitter_h
#define Bmmm_Analysis_Tau3MuKinVtxFitter_h

// ----------------------------------------------------------------------------
// Tau3MuKinVtxFitter
//
// Vertex fitter for the displaced tau -> 3mu topology
//     Ds -> tau nu , tau -> mu a , a -> mu mu      (a long-lived, pdgId 9900015)
//
// It inherits the ENTIRE RJpsiKinVtxFitter machinery unchanged -- the transient
// track helpers, the generic N-body Fit(...), Fit2Body(...), the IP tools
// (signedIP3D / signedIP2D / signedDecayLength3D / jetTrackDistance) and the
// beamspot-constrained PV refit (refitPVRemovingTracks) -- and only ADDS the
// second stage of the sequential vertex fit:
//
//     FitMotherPlusTrack(aTree, bachelorTrack, mBachelor)
//
// The first stage is the plain two-body fit of the displaced opposite-sign muon
// pair (the "a"): aTree = Fit2Body(mu1, mu2, mMu, mMu) -- deliberately WITHOUT a
// mass constraint, since m(mu mu) = m(a) is the search variable. The second
// stage takes the mother particle of that fit (the refitted "a", carrying its
// full covariance) and fits it to a COMMON, DIFFERENT vertex with the bachelor
// muon, forming the upstream tau vertex. The "a" enters as a single
// KinematicParticle, so the tau vertex and the a vertex are properly correlated
// -- this is the hierarchical/sequential fit, NOT a flat three-track fit.
//
// Keeping this in a subclass means the tau3mu sub-package is purely additive:
// the rjpsi code and its RJpsiKinVtxFitter header are not modified.
// ----------------------------------------------------------------------------

#include "Bmmm/Analysis/interface/RJpsiKinVtxFitter.h"

class Tau3MuKinVtxFitter : public RJpsiKinVtxFitter {

  public:
    Tau3MuKinVtxFitter() {};
    virtual ~Tau3MuKinVtxFitter() {};

    // ------------------------------------------------------------------------
    // sequential vertex fit, SECOND stage
    //
    //   motherTree : the RefCountedKinematicTree from the first (OS pair) fit,
    //                e.g. aTree = Fit2Body(mu1, mu2, mMu, mMu). Its top particle
    //                (the refitted "a") is reused, covariance included.
    //   trk        : the bachelor-muon track (the mu from tau -> mu a)
    //   mTrk       : the bachelor-muon mass hypothesis
    //
    // Returns the tau decay tree (top = tau, children = a and bachelor mu), or
    // an empty tree on failure (invalid input tree or failed fit). The returned
    // tree is structurally identical to the ones from Fit(...), so all the
    // downstream handling (is_good_vtx, compute_displacement, ...) is unchanged.
    // ------------------------------------------------------------------------
    RefCountedKinematicTree FitMotherPlusTrack(RefCountedKinematicTree motherTree,
                                               const reco::Track&      trk,
                                               const double&           mTrk)
    {
        // the first-stage fit must have produced a valid, non-empty tree
        if (motherTree == 0)        return RefCountedKinematicTree();
        if (!motherTree->isValid()) return RefCountedKinematicTree();
        if (motherTree->isEmpty())  return RefCountedKinematicTree();

        // the mother ("a") is the top particle of the first tree; it carries the
        // momentum AND covariance from the OS-pair vertex fit
        motherTree->movePointerToTheTop();
        RefCountedKinematicParticle mother = motherTree->currentParticle();
        if (!mother) return RefCountedKinematicTree();

        // build the bachelor muon as a kinematic particle from its transient track
        KinematicParticleFactoryFromTransientTrack pFactory;
        float chi   = 0.0;
        float ndf   = 0.0;
        float sigma = 1e-6;
        ParticleMass m = mTrk;
        RefCountedKinematicParticle bachelor =
            pFactory.particle(getTransientTrack(trk), m, chi, ndf, sigma);

        // fit { a , bachelor mu } to a common vertex (the tau vertex)
        std::vector<RefCountedKinematicParticle> toFit;
        toFit.reserve(2);
        toFit.push_back(mother);
        toFit.push_back(bachelor);

        KinematicParticleVertexFitter kpvFitter;
        RefCountedKinematicTree tree = kpvFitter.fit(toFit);

        if (tree == 0)        return RefCountedKinematicTree();
        if (!tree->isValid()) return RefCountedKinematicTree();

        return tree;
    };

};

#endif
