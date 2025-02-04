#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "MagneticField/ParametrizedEngine/src/OAEParametrizedMagneticField.h"

#include "TrackingTools/IPTools/interface/IPTools.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

// Assume you have the following:
// - a `reco::Track` object called `track`
// - a `reco::Vertex` object called `primaryVertex`
// - a `GlobalVector` called `direction` (e.g., the flight direction or jet direction)
// - a `TransientTrackBuilder` called `transientTrackBuilder`

class JetTrackDistance{

    public:
        JetTrackDistance(const reco::Track& track, const reco::Vertex& vertex, const GlobalVector& direction) {
        
            // Create a TransientTrack
            //reco::TransientTrack transientTrack = transientTrackBuilder->build(track);
            reco::TransientTrack transientTrack(track, paramField);
           
            // Get the signed 3D decay length (is it the distance between the track and the direction/jet axis?)
            jetTrackDistance = IPTools::jetTrackDistance(transientTrack, direction, vertex);
                    
        };
        
        virtual ~JetTrackDistance() {};


        std::pair<bool, Measurement1D> get() {return jetTrackDistance;};


    private:
    
        std::pair<bool, Measurement1D> jetTrackDistance;
        
        OAEParametrizedMagneticField *paramField = new OAEParametrizedMagneticField("3_8T");

};