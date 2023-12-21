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


class RDsKinVtxFitter {

  public:
    RDsKinVtxFitter() {};
    virtual ~RDsKinVtxFitter() {};

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
    auto Fit(const reco::Track & k1  , 
             const reco::Track & k2  ,
             const reco::Track & pi  ,
             const reco::Track & mu  ,
             const double      & mk  ,
             const double      & mpi ,
             const double      & mmu ,
             const double      & mphi,
             const double      & mds )
    {
    
        //define a factory
        KinematicParticleFactoryFromTransientTrack pFactory;
        
        //define the vector for the particles to be fitted
        std::vector<RefCountedKinematicParticle> phiToFit;
        std::vector<RefCountedKinematicParticle> dsToFit ;
        std::vector<RefCountedKinematicParticle> bsToFit ;

        // add the final states
        ParticleMass kMass   = mk  ;
        ParticleMass piMass  = mpi ;
        ParticleMass muMass  = mmu ;
        ParticleMass phiMass = mphi;
        ParticleMass dsMass  = mds ;
    
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
        phiToFit.push_back(pFactory.particle(getTransientTrack(k1), kMass , chi, ndf, sigma));
        phiToFit.push_back(pFactory.particle(getTransientTrack(k2), kMass , chi, ndf, sigma));
        
        // constraints
        MultiTrackKinematicConstraint* phiConstr = new TwoTrackMassKinematicConstraint(phiMass); //, phiMassSigma);
        
        // fit
        RefCountedKinematicTree phiTree = kcvFitter.fit(phiToFit, phiConstr);
        //return phiTree;
        // return void results if failed
        if (!phiTree->isValid()) return std::make_tuple(RefCountedKinematicTree(), RefCountedKinematicTree(), RefCountedKinematicTree());

        phiTree -> movePointerToTheTop();
        RefCountedKinematicParticle              phiParticle = phiTree->currentParticle()    ;
        RefCountedKinematicVertex                phiVertex   = phiTree->currentDecayVertex() ;
        std::vector<RefCountedKinematicParticle> phiDaus     = phiTree->finalStateParticles();

        //std::cout << "we should have 2 of them: " << phiDaus.size() << std::endl;

        //--------------------------------------------------------------------------------
        // fit Ds
        dsToFit.push_back(pFactory.particle(getTransientTrack(pi), piMass, chi, ndf, sigma));
        dsToFit.push_back(phiParticle);

        // constraints
        MultiTrackKinematicConstraint* dsConstr = new TwoTrackMassKinematicConstraint(dsMass); // , dsMassSigma);

        // fit
        RefCountedKinematicTree dsTree = kcvFitter.fit(dsToFit, dsConstr);
        if (!dsTree->isValid()) return std::make_tuple(phiTree, RefCountedKinematicTree(), RefCountedKinematicTree());

        dsTree -> movePointerToTheTop();
        RefCountedKinematicParticle              dsParticle = dsTree->currentParticle()    ;
        RefCountedKinematicVertex                dsVertex   = dsTree->currentDecayVertex() ;
        std::vector<RefCountedKinematicParticle> dsDaus     = dsTree->finalStateParticles();

        //--------------------------------------------------------------------------------
        // fit Ds
        bsToFit .push_back(pFactory.particle(getTransientTrack(mu), muMass, chi, ndf, sigma));
        bsToFit.push_back(dsParticle);

        // fit
        RefCountedKinematicTree bsTree = kcvFitter.fit(bsToFit); // no constraint for bs bc missing momentum
        if (!bsTree->isValid()) return std::make_tuple(phiTree, dsTree, RefCountedKinematicTree()); 

        bsTree -> movePointerToTheTop();
        RefCountedKinematicParticle              bsParticle = bsTree->currentParticle();
        RefCountedKinematicVertex                bsVertex   = bsTree->currentDecayVertex();
        std::vector<RefCountedKinematicParticle> bsDaus     = bsTree->finalStateParticles();

        //--------------------------------------------------------------------------------
        // return results
        
        return std::make_tuple(phiTree, dsTree, bsTree);

    };

  private:
    OAEParametrizedMagneticField *paramField = new OAEParametrizedMagneticField("3_8T");

};