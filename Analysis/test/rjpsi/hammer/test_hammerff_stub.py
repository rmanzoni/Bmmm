#!/usr/bin/env python3
'''
Exercise the HammerFF plumbing WITHOUT a Hammer install.

A fake `hammer` module records every call and returns a deterministic weight, so
this checks the parts that are ours and can silently go wrong -- the option
strings, the sticky 15-vector, the decay-chain registration, the process/vertex
construction, the branch schema, the status codes -- without touching physics.
It is not a substitute for the real closure tests; it is what tells you a
refactor broke the wiring before you spend a night on 10 M events.

    python3 test_hammerff_stub.py
'''
import math
import sys
import types

import numpy as np


# --------------------------------------------------------------------------
# fake Hammer
# --------------------------------------------------------------------------
class _FourMomentum(object):
    def __init__(self, e, px, py, pz):
        self.v = (e, px, py, pz)


class _Particle(object):
    def __init__(self, p4, pid):
        self.p4, self.pid = p4, pid


class _Process(object):
    def __init__(self):
        self.particles, self.vertices = [], []

    def add_particle(self, part):
        self.particles.append(part)
        return len(self.particles)          # Hammer indices are 1-based-ish; only identity matters here

    def add_vertex(self, parent, children):
        self.vertices.append((parent, list(children)))


class _Hammer(object):
    def __init__(self):
        self.calls = []
        self.options = []
        self.decays = []
        self.coords = None
        self.proc = None

    def include_decay(self, chain):
        self.decays.append(list(chain))

    def add_ff_scheme(self, name, mapping):
        self.calls.append(('add_ff_scheme', name, dict(mapping)))

    def set_ff_input_scheme(self, mapping):
        self.calls.append(('set_ff_input_scheme', dict(mapping)))

    def set_units(self, units):
        self.calls.append(('set_units', units))

    def init_run(self):
        self.calls.append(('init_run',))

    def set_options(self, string):
        self.options.append(string)

    def init_event(self):
        self.proc = None

    def add_process(self, proc):
        self.proc = proc
        return 1

    def process_event(self):
        self.calls.append(('process_event',))

    def set_ff_eigenvectors(self, xtoy, target, coords):
        assert len(coords) == 15, 'the eigenvector map is sticky: pass all 15 coords'
        self.coords = list(coords)
        self.ev_keys = (xtoy, target)

    def retrieve_ff_eigenvectors(self, process, group):
        # mimics the real Dict[str, float]; a wrong process/group is a no-op
        # in the real library, so model that too
        if (process, group) != getattr(self, 'ev_keys', None):
            return {}
        return dict(('delta_e%d' % j, c) for j, c in enumerate(self.coords))

    def get_weight(self, scheme):
        # deterministic and coordinate-dependent, so a mixed-up coordinate order
        # shows up as a mismatched weight
        return 1.0 + sum((j + 1) * c for j, c in enumerate(self.coords)) / 100.


fake = types.ModuleType('hammer')
fake.Hammer = _Hammer
fake.Process = _Process
fake.Particle = _Particle
fake.FourMomentum = _FourMomentum
sys.modules['hammer'] = fake

from Bmmm.Analysis.HammerFF import (            # noqa: E402
    BRANCH_NAMES, BRANCH_NOMINAL, BRANCH_STATUS, HammerSession, N_COORDS,
    STATUS_NOT_SIGNAL, STATUS_NO_LEAVES, STATUS_OK,
)


def _leaves(code):
    lv = {'gen_bc_decay': code}
    kin = {
        'gen_b':          (10.0, 0.5,  0.1, 6.2745, 541),
        'gen_ham_mup':    (4.0, 0.4,  0.05, 0.1057, -13),
        'gen_ham_mum':    (3.0, 0.6,  0.15, 0.1057,  13),
        'gen_ham_lep':    (2.5, 0.3, -0.10, 0.1057, -13),
        'gen_ham_nu':     (1.5, 0.7,  0.30, 0.0000,  14),
        'gen_ham_taumu':  (1.2, 0.35, 0.02, 0.1057, -13),
        'gen_ham_taunut': (0.8, 0.25, 0.20, 0.0000, -16),
        'gen_ham_taunum': (0.6, 0.45, 0.40, 0.0000,  14),
    }
    for pfx, (pt, eta, phi, mass, pid) in kin.items():
        lv['%s_pt' % pfx], lv['%s_eta' % pfx] = pt, eta
        lv['%s_phi' % pfx], lv['%s_mass' % pfx] = phi, mass
        lv['%s_pdgid' % pfx] = pid
    if code == 1:                       # mu channel: tau legs absent
        for pfx in ('gen_ham_taumu', 'gen_ham_taunut', 'gen_ham_taunum'):
            for comp in ('pt', 'eta', 'phi', 'mass', 'pdgid'):
                lv['%s_%s' % (pfx, comp)] = float('nan')
    return lv


def main():
    card = sys.argv[1] if len(sys.argv) > 1 else None
    # wiring only: the shipped card is the legacy-P1 one until it is refitted
    ses = HammerSession(card_path=card, allow_stale=True, allow_legacy_p1=True)
    ham = ses.ham

    assert ['BcJpsiMuNu', 'JpsiMuMu'] in ham.decays, ham.decays
    assert ['BcJpsiTauNu', 'TauMuNuNu', 'JpsiMuMu'] in ham.decays, ham.decays
    assert len(ham.decays) == 2, 'one include_decay per chain, no merged call'
    central = [o for o in ham.options if o.startswith('BctoJpsiBGLVar: {')
               and 'abcdmatrix' not in o]
    assert central, ham.options
    for key in ('avec', 'bvec', 'cvec', 'dvec'):
        assert '%s: [' % key in central[0], key
    assert any('abcdmatrix' in o for o in ham.options), 'abcdmatrix never set'
    # Hammer REFUSES GeneralRates: false without a WC specialization ("The hive
    # is out of balance") and integrates anyway, so the default must NOT send it
    assert not any('GeneralRates' in o for o in ham.options), \
        'GeneralRates must not be set by default: Hammer rejects it'
    print('[ok] setup: 2 decay chains, central options, abcdmatrix')
    print('[ok] eigenvector set/retrieve verification ran at init')

    assert len(BRANCH_NAMES) == 2 * N_COORDS + 2, len(BRANCH_NAMES)
    assert BRANCH_NAMES[0] == BRANCH_NOMINAL and BRANCH_NAMES[-1] == BRANCH_STATUS
    print('[ok] schema: %d branches (%s ... %s)'
          % (len(BRANCH_NAMES), BRANCH_NAMES[1], BRANCH_NAMES[-2]))

    # mu channel
    out = ses.weights(_leaves(1))
    assert out[BRANCH_STATUS] == STATUS_OK
    assert len(ham.proc.particles) == 6 and len(ham.proc.vertices) == 2
    assert abs(out[BRANCH_NOMINAL] - 1.0) < 1e-12, out[BRANCH_NOMINAL]
    for j in range(N_COORDS):
        up = out['hammer_ff_ev%02d_up' % j]
        dn = out['hammer_ff_ev%02d_dn' % j]
        assert abs(up - (1.0 + (j + 1) / 100.)) < 1e-12, (j, up)
        assert abs((up + dn) / 2. - 1.0) < 1e-12, (j, up, dn)
    print('[ok] mu: 6 particles, 2 vertices, nominal = 1, up/dn hit the right coordinate')

    # tau channel
    out = ses.weights(_leaves(7))
    assert out[BRANCH_STATUS] == STATUS_OK
    assert len(ham.proc.particles) == 9 and len(ham.proc.vertices) == 3
    print('[ok] tau: 9 particles, 3 vertices')

    # non-signal and broken rows
    out = ses.weights(_leaves(3))
    assert out[BRANCH_STATUS] == STATUS_NOT_SIGNAL
    assert math.isnan(out[BRANCH_NOMINAL])
    broken = _leaves(7)
    broken['gen_ham_taumu_pt'] = float('nan')
    out = ses.weights(broken)
    assert out[BRANCH_STATUS] == STATUS_NO_LEAVES
    assert math.isnan(out[BRANCH_NOMINAL])
    print('[ok] status codes: non-signal and missing-leaf rows are NaN, not 1')

    # the mu channel must not silently pick up the tau legs
    lv = _leaves(1)
    assert all(math.isnan(lv['gen_ham_taumu_%s' % c]) for c in ('pt', 'eta', 'phi'))
    out = ses.weights(lv)
    assert out[BRANCH_STATUS] == STATUS_OK
    print('[ok] mu rows with NaN tau legs are still reweightable')

    print(ses.summary())
    print('\nALL STUB TESTS PASSED -- the wiring is intact; physics closure still '
          'needs the real Hammer (add_hammer_weights.py --closure and the <w> checks).')


if __name__ == '__main__':
    main()
