#define G__DICTIONARY

#include <atomic>

#include "DataFormats/Common/interface/Wrapper.h"
#include "Bmmm/Analysis/interface/KVFitter.h"
#include "Bmmm/Analysis/interface/RDsKinVtxFitter.h"

namespace {
  struct RDs {
    KVFitter KalVtx_;
    RDsKinVtxFitter KinVtx_;
  };
}
