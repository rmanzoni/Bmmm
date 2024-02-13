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
#include "MagneticField/ParametrizedEngine/src/OAEParametrizedMagneticField.h"

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


class B4MuKinVtxFitter {

  public:
    B4MuKinVtxFitter() {};
    virtual ~B4MuKinVtxFitter() {};

//    struct FitResults {
//        RefCountedKinematicTree PhiTree = 0;
//        RefCountedKinematicTree DsTree  = 0;
//        RefCountedKinematicTree BsTree  = 0;
//    };

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

    // constructed from reco::Track
    //RefCountedKinematicTree Fit(const reco::Track & k1, 
    //FitResults Fit(const reco::Track & k1  , 
    //std::tuple<RefCountedKinematicTree, RefCountedKinematicTree, RefCountedKinematicTree> Fit(const reco::Track & k1  , 
    auto Fit(const reco::Track & mu1, 
             const reco::Track & mu2,
             const reco::Track & mu3,
             const reco::Track & mu4,
             const double      & mmu)
    {
        //define a factory
        KinematicParticleFactoryFromTransientTrack pFactory;
        
        //define the vector for the particles to be fitted
        std::vector<RefCountedKinematicParticle> bToFit;

        // add the final states
        ParticleMass muMass = mmu ;
    
        float chi   = 0.0;
        float ndf   = 0.0;
        float sigma = 1e-6;
        //float phiMassSigma = 1e-6;
        //float dsMassSigma  = 1e-6;
        //float muMassSigma  = 1e-6;

        //--------------------------------------------------------------------------------
        // create empty results container
        //FitResults results;
        
        //auto results = std::make_tuple(RefCountedKinematicTree(), RefCountedKinematicTree(), RefCountedKinematicTree());

        //--------------------------------------------------------------------------------
        // create fitter object
        KinematicConstrainedVertexFitter kcvFitter;

        //--------------------------------------------------------------------------------
        // fit phi1020
        bToFit.push_back(pFactory.particle(getTransientTrack(mu1), muMass, chi, ndf, sigma));
        bToFit.push_back(pFactory.particle(getTransientTrack(mu2), muMass, chi, ndf, sigma));
        bToFit.push_back(pFactory.particle(getTransientTrack(mu3), muMass, chi, ndf, sigma));
        bToFit.push_back(pFactory.particle(getTransientTrack(mu4), muMass, chi, ndf, sigma));
        
        // fit
        RefCountedKinematicTree bTree = kcvFitter.fit(bToFit);
        //return phiTree;
        // return void results if failed
        if (bTree==0) return RefCountedKinematicTree();
        if (!bTree->isValid()) return RefCountedKinematicTree();

        return bTree;

    };

  private:
    OAEParametrizedMagneticField *paramField = new OAEParametrizedMagneticField("3_8T");

};