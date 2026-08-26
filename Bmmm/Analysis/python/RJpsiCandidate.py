'''
RETIRED -> thin re-export shim (single source of truth).

The RJpsi candidate has been refactored into a generalized base
(Bmmm.Analysis.JpsiChargedCandidate) specialized by Bmmm.Analysis.JpsiMuCandidate.
RJpsiCandidate is kept as an ALIAS so existing imports keep working; there is no
duplicated implementation. New code should import JpsiMuCandidate directly.
'''
from Bmmm.Analysis.JpsiChargedCandidate import kinfit, BachelorTrack  # noqa: F401
from Bmmm.Analysis.JpsiMuCandidate import JpsiMuCandidate as RJpsiCandidate  # noqa: F401
