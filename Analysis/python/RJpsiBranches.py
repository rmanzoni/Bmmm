'''
RETIRED -> thin re-export shim (single source of truth).

The RJpsi branch schema is now assembled in Bmmm.Analysis.JpsiMuBranches (from the
shared Bmmm.Analysis.JpsiChargedBranches). This module re-exports the same names
so existing imports keep working; 'branches' is byte-for-byte the RJpsi schema.
'''
from Bmmm.Analysis.JpsiMuBranches import (  # noqa: F401
    branches, paths, muon_branches, cand_branches, event_branches,
    bc_branches, jpsi_branches, safe_get,
)
