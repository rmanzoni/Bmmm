#define G__DICTIONARY

#include <atomic>

#include "DataFormats/Common/interface/Wrapper.h"
#include "Bmmm/Analysis/interface/KVFitter.h"
#include "Bmmm/Analysis/interface/RDsKinVtxFitter.h"
#include "Bmmm/Analysis/interface/PVRefitter.h"
#include "Bmmm/Analysis/interface/IP3D.h"
#include "Bmmm/Analysis/interface/IP3D_BTV.h"
#include "Bmmm/Analysis/interface/SignedDecayLength3D.h"
#include "Bmmm/Analysis/interface/JetTrackDistance.h"

namespace {
  struct RDs {
    KVFitter KalVtx_;
    RDsKinVtxFitter KinVtx_;
    PVRefitter PVRefit_;
    SignedIP3D IP3D_;
    BTVSignedIP3D BTVIP3D_;
    SignedDecayLength3D DL3D_;
    JetTrackDistance JTD_;
  };
}
