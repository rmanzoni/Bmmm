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

class SignedIP3D{


    public:
        SignedIP3D(const reco::Track& track, const reco::Vertex& vertex, const GlobalVector& direction) {
        
            // Create a TransientTrack
            //reco::TransientTrack transientTrack = transientTrackBuilder->build(track);
            reco::TransientTrack transientTrack(track, paramField);
           
            // Get the signed 3D impact parameter
            signedIP3D = IPTools::signedImpactParameter3D(transientTrack, direction, vertex);
                    
        };
        
        virtual ~SignedIP3D() {};


        std::pair<bool, Measurement1D> get() {return signedIP3D;};



//         std::pair<bool, Measurement1D> 
// 
//         // Create a TransientTrack
//         reco::TransientTrack transientTrack = transientTrackBuilder->build(track);
//         
//         // Get the signed 3D impact parameter
//         std::pair<bool, Measurement1D> signedIP3D = IPTools::signedImpactParameter3D(transientTrack, direction, primaryVertex);
//         
//         if (signedIP3D.first) { // Check if the computation is valid
//             double value = signedIP3D.second.value();     // Signed impact parameter value
//             double error = signedIP3D.second.error();     // Uncertainty
//             double significance = value / error;          // Significance
//         }


    private:
    
        std::pair<bool, Measurement1D> signedIP3D;
        
        OAEParametrizedMagneticField *paramField = new OAEParametrizedMagneticField("3_8T");

};