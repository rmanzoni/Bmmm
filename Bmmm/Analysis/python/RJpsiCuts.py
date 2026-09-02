'''
RETIRED -> thin re-export shim (single source of truth).

The cuts are now defined in Bmmm.Analysis.JpsiChargedCuts (baseline + rjpsi). This
module re-exports the shared 'cuts' dict so existing imports keep working; the
'rjpsi' key is unchanged.
'''
from Bmmm.Analysis.JpsiChargedCuts import cuts  # noqa: F401
