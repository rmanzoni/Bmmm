#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "MagneticField/ParametrizedEngine/src/OAEParametrizedMagneticField.h"

#include "RecoBTag/BTagTools/interface/SignedImpactParameter3D.h"

#include "TrackingTools/IPTools/interface/IPTools.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

// Assume you have the following:
// - a `reco::Track` object called `track`
// - a `reco::Vertex` object called `primaryVertex`
// - a `GlobalVector` called `direction` (e.g., the flight direction or jet direction)
// - a `TransientTrackBuilder` called `transientTrackBuilder`

class BTVSignedIP3D{


    public:
        BTVSignedIP3D(const reco::Track& track, const reco::Vertex& vertex, const GlobalVector& direction) {
        
            // Create a TransientTrack
            reco::TransientTrack transientTrack(track, paramField);
           
            ip3D = signedIP3D.apply(transientTrack, direction, vertex);
                    
        };
        
        virtual ~BTVSignedIP3D() {};

        std::pair<bool, Measurement1D> get() {return ip3D;};


    private:
        
        SignedImpactParameter3D signedIP3D;
        std::pair<bool, Measurement1D> ip3D;
        
        OAEParametrizedMagneticField *paramField = new OAEParametrizedMagneticField("3_8T");

};