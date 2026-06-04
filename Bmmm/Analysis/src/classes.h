#define G__DICTIONARY

#include <atomic>

#include "DataFormats/Common/interface/Wrapper.h"
#include "Bmmm/Analysis/interface/KVFitter.h"
#include "Bmmm/Analysis/interface/RDsKinVtxFitter.h"
#include "Bmmm/Analysis/interface/B4MuKinVtxFitter.h"
#include "Bmmm/Analysis/interface/RJpsiKinVtxFitter.h"

namespace {
  struct RDs {
    KVFitter KalVtx_;
    RDsKinVtxFitter KinVtx_;
    B4MuKinVtxFitter BKinVtx_;
    RJpsiKinVtxFitter RJpsiKinVtx_;
  };
}
