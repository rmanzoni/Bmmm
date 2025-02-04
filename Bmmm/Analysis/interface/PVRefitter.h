#include "DataFormats/VertexReco/interface/Vertex.h"

#include "MagneticField/ParametrizedEngine/src/OAEParametrizedMagneticField.h"

#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"


class PVRefitter {

  public:
    PVRefitter() {};
    virtual ~PVRefitter() {};

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


    reco::Vertex fit(const std::vector<reco::Track>& tracks, const reco::BeamSpot& bs){
    //TransientVertex fit(const std::vector<reco::Track>& tracks, const reco::BeamSpot& bs){
    //CachingVertex<5> fit(const std::vector<reco::Track>& tracks, const reco::BeamSpot& bs){
    
      std::vector<reco::TransientTrack> transient_tracks;
      
      for (reco::TrackCollection::const_iterator itrk=tracks.begin(); itrk!=tracks.end(); itrk++){
        reco::TransientTrack tmpTransientTrack = getTransientTrack(*itrk); 
        tmpTransientTrack.setBeamSpot(bs);
        transient_tracks.push_back(tmpTransientTrack);
      }
      
      AdaptiveVertexFitter fitter;
      TransientVertex vertex = fitter.vertex(transient_tracks, bs);
      
//       return vertex; 
      return reco::Vertex(vertex); 
    }

  private:
    OAEParametrizedMagneticField *paramField = new OAEParametrizedMagneticField("3_8T");

};