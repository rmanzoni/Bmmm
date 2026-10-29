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
#include <limits>

// ----------------------------------------------------------------------------
// Sequential kinematic fits of Bs -> Ds(-> phi(-> K K) pi) mu (nu).
//
// Three schemes, run side by side on the same tracks so they can be compared
// on MC (RDsCandidate):
//
//   phic  phi mass constraint only
//     phi     : K K with the phi mass constraint
//     ds_phic : phi + pi, no constraint     -> m(KKpi) with the phi constraint
//                                              (the AN v4 / previous ntuples)
//     bs_phic : ds_phic + mu, no constraint (neutrinos)
//   cons  phi and Ds mass constraints
//     phi     : refitted identically to the phic one (see below)
//     ds_cons : phi + pi with the Ds mass constraint
//     bs_cons : ds_cons + mu, no constraint (neutrinos)
//   free  no mass constraint
//     kk      : K K
//     kkpi    : KK + pi
//     bs_free : kkpi + mu
//
// Two stages, so that the candidates failing the vertex cuts (most of them)
// only pay for the fits the cuts need:
//
//   FitVertexCut   [phi, ds_phic, kk, kkpi]             the cut fits
//   FitRest        [bs_phic, ds_cons, bs_cons, bs_free] continues the chains
//                  from the FitVertexCut trees
//
// The Bs fit also updates, in place, the Ds of the tree it is built from (the
// Ds refitted at the Bs vertex, so it carries the muon information). FitRest
// therefore records each Ds momentum just BEFORE its Bs fit, the muon-free Ds
// trajectory: DsMomentaBeforeBs() = [phic, cons, free] x (px, py, pz), NaN
// where that Ds fit failed. The Ds vertex position is not changed.
//
// A composite KinematicParticle must enter only ONE further fit: the fitter
// links its sub-tree into the new tree, so feeding the same phi to the phic
// and cons Ds fits rewires the first chain (its Bs tree top came back as the
// phi). Each chain therefore runs its own phi fit.
//
// Every returned vector has the full length; a failed step and every step
// after it in the same chain are empty trees (test with tree.get() /
// isValid() on the Python side). A fit that throws (KinematicFit raises on
// degenerate input) is a failed step, not a dead job.
// ----------------------------------------------------------------------------

class RDsKinVtxFitter {

  public:
    RDsKinVtxFitter() {};
    virtual ~RDsKinVtxFitter() {};

    reco::TransientTrack getTransientTrack(const reco::Track& track) {
      return reco::TransientTrack(track, paramField);
    }

    std::vector<RefCountedKinematicTree> FitVertexCut(const reco::Track & k1,
                                                      const reco::Track & k2,
                                                      const reco::Track & pi,
                                                      const double      & mk,
                                                      const double      & mpi,
                                                      const double      & mphi)
    {
        std::vector<RefCountedKinematicTree> out(4, RefCountedKinematicTree());

        // phic: phi with the mass constraint, + pi without the Ds constraint
        out[0] = fitPhi(k1, k2, mk, mphi);
        if (good(out[0])) {
            std::vector<RefCountedKinematicParticle> dsPhicIn = {particle(pi, mpi), top(out[0])};
            out[1] = fit(dsPhicIn, nullptr);
        }

        // free: K K, + pi
        std::vector<RefCountedKinematicParticle> kk = {particle(k1, mk), particle(k2, mk)};
        out[2] = fit(kk, nullptr);
        if (good(out[2])) {
            std::vector<RefCountedKinematicParticle> kkpiIn = {particle(pi, mpi), top(out[2])};
            out[3] = fit(kkpiIn, nullptr);
        }
        return out;
    }

    std::vector<RefCountedKinematicTree> FitRest(std::vector<RefCountedKinematicTree> & cut,
                                                 const reco::Track & k1,
                                                 const reco::Track & k2,
                                                 const reco::Track & pi,
                                                 const reco::Track & mu,
                                                 const double      & mk,
                                                 const double      & mpi,
                                                 const double      & mmu,
                                                 const double      & mphi,
                                                 const double      & mds)
    {
        std::vector<RefCountedKinematicTree> out(4, RefCountedKinematicTree());
        dsMomenta_.assign(9, std::numeric_limits<double>::quiet_NaN());

        // phic: ds_phic + mu
        if (good(cut[1])) {
            keepMomentum(top(cut[1]), 0);
            std::vector<RefCountedKinematicParticle> bsPhicIn = {particle(mu, mmu), top(cut[1])};
            out[0] = fit(bsPhicIn, nullptr);
        }

        // cons: its own phi fit (see the header), + pi with the Ds constraint, + mu
        if (good(cut[0])) {
            RefCountedKinematicTree phiCons = fitPhi(k1, k2, mk, mphi);
            if (good(phiCons)) {
                std::vector<RefCountedKinematicParticle> dsConsIn = {particle(pi, mpi), top(phiCons)};
                ParticleMass dsMass = mds;  // the constraint ctor takes a non-const reference
                TwoTrackMassKinematicConstraint dsConstr(dsMass);
                out[1] = fit(dsConsIn, &dsConstr);
                if (good(out[1])) {
                    keepMomentum(top(out[1]), 1);
                    std::vector<RefCountedKinematicParticle> bsConsIn = {particle(mu, mmu), top(out[1])};
                    out[2] = fit(bsConsIn, nullptr);
                }
            }
        }

        // free: kkpi + mu
        if (good(cut[3])) {
            keepMomentum(top(cut[3]), 2);
            std::vector<RefCountedKinematicParticle> bsFreeIn = {particle(mu, mmu), top(cut[3])};
            out[3] = fit(bsFreeIn, nullptr);
        }
        return out;
    }

    // the Ds momenta of the last FitRest call, before the Bs fits (see the header)
    const std::vector<double> & DsMomentaBeforeBs() const { return dsMomenta_; }

  private:
    std::vector<double> dsMomenta_;

    void keepMomentum(const RefCountedKinematicParticle & p, const unsigned int i) {
        GlobalVector mom = p->currentState().globalMomentum();
        dsMomenta_[3 * i]     = mom.x();
        dsMomenta_[3 * i + 1] = mom.y();
        dsMomenta_[3 * i + 2] = mom.z();
    }

    OAEParametrizedMagneticField *paramField = new OAEParametrizedMagneticField("3_8T");

    RefCountedKinematicParticle particle(const reco::Track & trk, const double & mass) {
        KinematicParticleFactoryFromTransientTrack pFactory;
        float chi = 0.0, ndf = 0.0, sigma = 1e-6;
        ParticleMass m = mass;
        return pFactory.particle(getTransientTrack(trk), m, chi, ndf, sigma);
    }

    RefCountedKinematicTree fitPhi(const reco::Track & k1, const reco::Track & k2,
                                   const double & mk, const double & mphi) {
        std::vector<RefCountedKinematicParticle> kk = {particle(k1, mk), particle(k2, mk)};
        ParticleMass phiMass = mphi;  // the constraint ctor takes a non-const reference
        TwoTrackMassKinematicConstraint phiConstr(phiMass);
        return fit(kk, &phiConstr);
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
