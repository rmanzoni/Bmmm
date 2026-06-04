from DataFormats.FWLite import Handle
from collections import OrderedDict

handles_mc = OrderedDict()
handles_mc['genpr'  ] = ('prunedGenParticles'  , Handle('std::vector<reco::GenParticle>')     )
handles_mc['genpk'  ] = ('packedGenParticles'  , Handle('std::vector<pat::PackedGenParticle>'))
#handles_mc['genInfo'] = ('generator'           , Handle('GenEventInfoProduct')                )
handles_mc['pu'     ] = ('slimmedAddPileupInfo', Handle('std::vector<PileupSummaryInfo>')     )

handles = OrderedDict()
handles['muons'  ] = ('slimmedMuons'                       , Handle('std::vector<pat::Muon>')                   )
handles['trk'    ] = ('packedPFCandidates'                 , Handle('std::vector<pat::PackedCandidate>')        )
handles['ltrk'   ] = ('lostTracks'                         , Handle('std::vector<pat::PackedCandidate>')        )
# handles['vtx'    ] = ('offlineSlimmedPrimaryVerticesWithBS', Handle('std::vector<reco::Vertex>')                )
handles['vtx'    ] = ('offlineSlimmedPrimaryVertices'      , Handle('std::vector<reco::Vertex>')                )
handles['trg_res'] = (('TriggerResults', '', 'HLT' )       , Handle('edm::TriggerResults'        )              )
handles['trg_ps' ] = (('patTrigger'    , '')               , Handle('pat::PackedTriggerPrescales')              )
handles['bs'     ] = ('offlineBeamSpot'                    , Handle('reco::BeamSpot')                           )
handles['tobjs'  ] = ('slimmedPatTrigger'                  , Handle('std::vector<pat::TriggerObjectStandAlone>'))
handles['jets'   ] = ('slimmedJets'                        , Handle('std::vector<pat::Jet>')                    )
