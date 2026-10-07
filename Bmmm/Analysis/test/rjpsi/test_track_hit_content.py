'''
Self-test for Bmmm.Analysis.TrackHitContent: the hit-content branches
(<obj>_n_pix_*, <obj>_pix_first_*, <obj>_n_pix_*_inner) and the covflow context
built from the same getters.

Needs a CMSSW environment: the tracks are REAL reco::Track objects whose hit
pattern is filled here through TrackBase::appendTrackerHitPattern, so every
HitPattern call goes through the same PyROOT bindings the ntuplizer uses. No
input file, no event loop. Runs in a few seconds.

    cmsenv
    python3 test_track_hit_content.py

What it checks:

  1. each quantity on hand-built patterns with a known answer: full barrel,
     L1 dead (inactive_inner), L1 missed (missing_inner), FPix only, barrel +
     disk, overlap hits, no pixel at all, strip-only inactive inner hits;
  2. the innermost layer does not depend on the order the hits are stored in;
  3. hit_content() runs one pass per object and every branch getter reads it;
  4. utils.COVFLOW_CONTEXT_GETTERS returns, for every admissible hit-content
     name, exactly the value the branch getter writes -- the property that the
     single-definition refactor exists to guarantee;
  5. the missing/inactive-inner counts are NOT accepted as covflow context;
  6. the per-layer masks (pix_valid_mask, pix_miss_mask, pix_inact_mask,
     pix_hit_count): values on the same patterns plus intermediate missing /
     inactive / bad layers and missing outer hits, consistency with
     pix_first_*, n_pix_*, and not accepted as covflow context.
'''

import ROOT

from Bmmm.Analysis.TrackHitContent import (
    HIT_CONTENT_NAMES, COVFLOW_HIT_CONTEXT, PXB, PXF,
    compute_hit_content, hit_content, hit_content_branches,
    PIX_MASK_LAYERS, PIX_COUNT_BITS, pix_layer_bit,
)
from Bmmm.Analysis.utils import COVFLOW_CONTEXT_GETTERS

FAILURES = []

TIB = 3  # StripSubdetector::TIB (DataFormats/SiStripDetId/interface/StripSubdetector.h)
T   = ROOT.TrackingRecHit


def check(name, condition, extra=''):
    print(('  OK   ' if condition else '  FAIL ') + name + (('   ' + extra) if extra else ''))
    if not condition:
        FAILURES.append(name)


def make_track(track_hits, inner_hits=()):
    '''A reco::Track with the given hits. Each hit is (subdet, layer, type).
    HitPattern requires all hits of one category to be appended in a row:
    TRACK_HITS first, then MISSING_INNER_HITS.'''
    trk = ROOT.reco.Track()
    for subdet, layer, htype in list(track_hits) + list(inner_hits):
        if not trk.appendTrackerHitPattern(subdet, layer, 0, htype):
            raise RuntimeError('appendTrackerHitPattern refused %s' % ((subdet, layer, htype),))
    return trk


class FakeObj(object):
    '''Just enough of a pat::Muon: bestTrack(), counting the calls.'''
    def __init__(self, trk):
        self._trk = trk
        self.n_calls = 0

    def bestTrack(self):
        self.n_calls += 1
        return self._trk


def bits(*layers):
    '''Mask value from layer names, e.g. bits('L1', 'D2').'''
    return sum(1 << PIX_MASK_LAYERS.index(l) for l in layers)


def counts(**per_layer):
    '''pix_hit_count value from per-layer counts, e.g. counts(L1=2, L2=1).'''
    return sum(n << (PIX_COUNT_BITS * PIX_MASK_LAYERS.index(l)) for l, n in per_layer.items())


def main():
    V, MI, II = T.valid, T.missing_inner, T.inactive_inner
    M, I, B, MO = T.missing, T.inactive, T.bad, T.missing_outer
    strips = [(TIB, 1, V), (TIB, 2, V)]

    cases = [
        ('full barrel',
         make_track([(PXB, 1, V), (PXB, 2, V), (PXB, 3, V), (PXB, 4, V)] + strips),
         dict(n_pix_hit=4, n_pix_b_hit=4, n_pix_e_hit=0, n_pix_layer=4, n_trk_layer=6,
              pix_first_b_layer=1, pix_first_e_disk=0, pix_first_layer=1,
              n_pix_miss_inner=0, n_pix_inact_inner=0,
              pix_valid_mask=bits('L1', 'L2', 'L3', 'L4'), pix_miss_mask=0, pix_inact_mask=0,
              pix_hit_count=counts(L1=1, L2=1, L3=1, L4=1))),
        ('L1 dead (inactive_inner)',
         make_track([(PXB, 2, V), (PXB, 3, V), (PXB, 4, V)] + strips, [(PXB, 1, II)]),
         dict(pix_first_layer=2, n_pix_miss_inner=0, n_pix_inact_inner=1,
              pix_valid_mask=bits('L2', 'L3', 'L4'), pix_miss_mask=0,
              pix_inact_mask=bits('L1'))),
        ('L1 missed (missing_inner)',
         make_track([(PXB, 2, V), (PXB, 3, V), (PXB, 4, V)] + strips, [(PXB, 1, MI)]),
         dict(pix_first_layer=2, n_pix_miss_inner=1, n_pix_inact_inner=0,
              pix_valid_mask=bits('L2', 'L3', 'L4'), pix_miss_mask=bits('L1'),
              pix_inact_mask=0)),
        ('FPix only',
         make_track([(PXF, 1, V), (PXF, 2, V), (PXF, 3, V)] + strips),
         dict(n_pix_b_hit=0, n_pix_e_hit=3, pix_first_b_layer=0, pix_first_e_disk=1,
              pix_first_layer=5, pix_valid_mask=bits('D1', 'D2', 'D3'),
              pix_hit_count=counts(D1=1, D2=1, D3=1))),
        ('FPix from disk 2',
         make_track([(PXF, 2, V), (PXF, 3, V)] + strips),
         dict(pix_first_layer=6)),
        ('barrel + disk',
         make_track([(PXB, 1, V), (PXB, 2, V), (PXF, 1, V)] + strips),
         dict(pix_first_b_layer=1, pix_first_e_disk=1, pix_first_layer=1,
              pix_valid_mask=bits('L1', 'L2', 'D1'))),
        ('overlap: two hits on L1',
         make_track([(PXB, 1, V), (PXB, 1, V), (PXB, 2, V)]),
         dict(n_pix_hit=3, n_pix_layer=2, pix_first_layer=1,
              pix_valid_mask=bits('L1', 'L2'), pix_hit_count=counts(L1=2, L2=1))),
        ('no pixel hit',
         make_track(strips),
         dict(n_pix_hit=0, n_pix_layer=0, pix_first_layer=0,
              pix_valid_mask=0, pix_miss_mask=0, pix_inact_mask=0, pix_hit_count=0)),
        ('strip-only inactive inner hit is not counted',
         make_track(strips, [(TIB, 1, II)]),
         dict(n_pix_inact_inner=0, n_pix_miss_inner=0, pix_inact_mask=0)),
        ('stored outside-in: still the innermost',
         make_track([(PXB, 4, V), (PXB, 3, V), (PXB, 2, V)]),
         dict(pix_first_layer=2)),
        # --- per-layer masks: the cases pix_first_* cannot see
        ('L2 missed in the middle of the track',
         make_track([(PXB, 1, V), (PXB, 2, M), (PXB, 3, V), (PXB, 4, V)] + strips),
         dict(pix_first_layer=1, n_pix_hit=3, pix_valid_mask=bits('L1', 'L3', 'L4'),
              pix_miss_mask=bits('L2'), pix_inact_mask=0)),
        ('L2 dead in the middle of the track',
         make_track([(PXB, 1, V), (PXB, 2, I), (PXB, 3, V), (PXB, 4, V)] + strips),
         dict(pix_valid_mask=bits('L1', 'L3', 'L4'), pix_miss_mask=0,
              pix_inact_mask=bits('L2'))),
        ('L3 bad counts as dead',
         make_track([(PXB, 1, V), (PXB, 2, V), (PXB, 3, B), (PXB, 4, V)] + strips),
         dict(pix_valid_mask=bits('L1', 'L2', 'L4'), pix_inact_mask=bits('L3'))),
        ('L1 and L2 dead before the first hit',
         make_track([(PXB, 3, V), (PXB, 4, V)] + strips, [(PXB, 1, II), (PXB, 2, II)]),
         dict(pix_first_layer=3, n_pix_inact_inner=2, pix_valid_mask=bits('L3', 'L4'),
              pix_inact_mask=bits('L1', 'L2'), pix_miss_mask=0)),
        ('overlap: one module of L1 missed, the other valid -> L1 is valid',
         make_track([(PXB, 1, V), (PXB, 1, M), (PXB, 2, V)]),
         dict(pix_valid_mask=bits('L1', 'L2'), pix_miss_mask=0,
              pix_hit_count=counts(L1=1, L2=1))),
        ('D2 missed between D1 and D3',
         make_track([(PXB, 1, V), (PXF, 1, V), (PXF, 2, M), (PXF, 3, V)] + strips),
         dict(pix_valid_mask=bits('L1', 'D1', 'D3'), pix_miss_mask=bits('D2'))),
        ('D3 missed after the last hit (missing_outer)',
         make_track([(PXB, 1, V), (PXF, 1, V), (PXF, 2, V)], [(PXF, 3, MO)]),
         dict(pix_valid_mask=bits('L1', 'D1', 'D2'), pix_miss_mask=bits('D3'))),
    ]

    print('\n== 1-2. values on hand-built hit patterns ==')
    for name, trk, expected in cases:
        got = compute_hit_content(trk)
        check('%s: all names present' % name, list(sorted(got)) == sorted(HIT_CONTENT_NAMES))
        wrong = dict((k, (got[k], v)) for k, v in expected.items() if got[k] != v)
        check(name, not wrong, '' if not wrong else 'got/expected %s' % wrong)

    print('\n== 3. one pass per object ==')
    obj = FakeObj(cases[1][1])
    values = dict((k, g(obj)) for k, g in hit_content_branches.items())
    check('bestTrack() read once for all branches', obj.n_calls == 1, 'calls = %d' % obj.n_calls)
    check('branches equal hit_content()', values == hit_content(obj))

    print('\n== 4. covflow context == branch, for every admissible name ==')
    for name, trk, _ in cases:
        for ctx in COVFLOW_HIT_CONTEXT:
            a, b = FakeObj(trk), FakeObj(trk)
            from_ctx    = COVFLOW_CONTEXT_GETTERS[ctx](a, a.bestTrack(), None)
            from_branch = hit_content_branches[ctx](b)
            if from_ctx != from_branch:
                check('%s / %s' % (name, ctx), False, '%s != %s' % (from_ctx, from_branch))
    check('context and branch agree on every case', not [f for f in FAILURES if ' / ' in f])

    print('\n== 5. diagnostics are not context ==')
    for name in ('n_pix_miss_inner', 'n_pix_inact_inner'):
        check('%s not a covflow context variable' % name, name not in COVFLOW_CONTEXT_GETTERS)

    print('\n== 6. per-layer masks: consistency and not context ==')
    for name, trk, _ in cases:
        h = compute_hit_content(trk)
        vm, cnt = int(h['pix_valid_mask']), int(h['pix_hit_count'])
        per = [(cnt >> (PIX_COUNT_BITS * b)) & ((1 << PIX_COUNT_BITS) - 1)
               for b in range(len(PIX_MASK_LAYERS))]
        fb = next((l for l in range(1, 5) if vm >> pix_layer_bit(True, l) & 1), 0)
        fe = next((d for d in range(1, 4) if vm >> pix_layer_bit(False, d) & 1), 0)
        ok = (fb == h['pix_first_b_layer'] and fe == h['pix_first_e_disk']
              and sum(per[:4]) == h['n_pix_b_hit'] and sum(per[4:]) == h['n_pix_e_hit']
              and all((c > 0) == bool(vm >> b & 1) for b, c in enumerate(per))
              and bin(vm).count('1') == h['n_pix_layer']
              and not (h['pix_miss_mask'] | h['pix_inact_mask']) & vm)
        check('%s: masks agree with pix_first_*, n_pix_*, n_pix_layer' % name, ok,
              '' if ok else 'mask %s count %s first %s/%s' % (bin(vm), per, fb, fe))
    for name in ('pix_valid_mask', 'pix_miss_mask', 'pix_inact_mask', 'pix_hit_count'):
        check('%s not a covflow context variable' % name, name not in COVFLOW_CONTEXT_GETTERS)

    print('\n%s' % ('ALL CHECKS PASSED' if not FAILURES
                    else '%d FAILURE(S): %s' % (len(FAILURES), FAILURES)))
    return 1 if FAILURES else 0


if __name__ == '__main__':
    raise SystemExit(main())
