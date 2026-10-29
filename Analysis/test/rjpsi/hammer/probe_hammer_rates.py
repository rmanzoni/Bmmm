#!/usr/bin/env python3
'''
Ask Hammer itself for the Bc -> J/psi rates, and get R(J/psi) with no MC at all.

Why this matters: everything validated so far -- the Harrison-2024 refit, the q2
spectra, R = 0.2602 against the lattice 0.2597(27) -- went through our Python
port of Hammer's BGL evaluator. Hammer's own C++ was never in the loop. If
get_rate works, the C++ evaluates the same coefficients independently, and

    R = Gamma_tau / Gamma_mu

drops out in seconds. Agreement closes the last link: the option string parsed,
BctoJpsiBGLVar took the coefficients in the convention we think it did, and the
port matches the library. Disagreement localises the problem before a single
event is reweighted.

get_rate is declared `get_rate(*args)` in the Cython bindings -- no introspectable
signature -- so this script DISCOVERS the calling convention by trying the
plausible ones and reporting what each returns, rather than guessing.

    python3 probe_hammer_rates.py                  # discover + report
    python3 probe_hammer_rates.py --card <path>    # a specific FF card

Beware one trap: Hammer's built-in default for BctoJpsiBGLVar is Harrison-2020,
whose R is about 0.258 -- close enough to the 2024 value that "the numbers look
right" does NOT prove set_options took. That is why the script also dumps
show_available_ff_params so you can read the loaded coefficients back.
'''
import argparse
import itertools

from Bmmm.Analysis.HammerFF import (
    FF_INPUT, FF_TARGET, PROCESS, SCHEME, XTOY, make_hammer_session,
)

# process labels worth trying; the include_decay names are the ones the session
# registered, the others are how Hammer tends to label rate processes
MU_NAMES = ('BcJpsiMuNu', 'BcToJpsiMuNu', 'BcJpsiLepNu', PROCESS)
TAU_NAMES = ('BcJpsiTauNu', 'BcToJpsiTauNu', PROCESS)


def try_call(ham, name, args):
    try:
        val = getattr(ham, name)(*args)
    except Exception as exc:                                  # noqa: BLE001
        return None, '%s: %s' % (type(exc).__name__, exc)
    return val, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--card', default='default:allow-stale')
    ap.add_argument('--quiet-params', action='store_true')
    args = ap.parse_args()

    # this script is precisely about the rates, so ask for them explicitly
    ses = make_hammer_session(args.card, general_rates=True)
    ham = ses.ham

    if not args.quiet_params:
        print('\n==== show_available_ff_params(%r) ====' % XTOY)
        print('(prints to stdout and returns None; read the coefficients back '
              'here to confirm set_options actually took)')
        ham.show_available_ff_params(XTOY)
        print('==== end ====\n')

    # --- discover the calling convention ------------------------------------
    print('probing get_rate and get_denominator_rate ...')
    hits = {}
    for method, procs in (('get_rate', MU_NAMES + TAU_NAMES),
                          ('get_denominator_rate', MU_NAMES + TAU_NAMES)):
        for proc in procs:
            candidates = [(SCHEME, proc), (proc, SCHEME), (proc,),
                          (SCHEME, proc, FF_TARGET)]
            if method == 'get_denominator_rate':
                candidates += [(FF_INPUT, proc), (proc, FF_INPUT)]
            for call_args in candidates:
                val, err = try_call(ham, method, call_args)
                if val is None:
                    continue
                print('  %-22s%-46s -> %.6g' % (method, repr(call_args), val))
                hits[(method, proc, call_args)] = val

    if not hits:
        print('\nNo call pattern worked. Widen the process names above, or dump '
              'what Hammer knows about:\n'
              "  python3 -c \"import hammer; h=hammer.Hammer(); "
              'h.show_available_ff_params()"')
        return

    # --- R(J/psi), for every (mu, tau) pair that came back ------------------
    print('\nR(J/psi) from Hammer rates:')
    seen = set()
    for (m1, p1, a1), v1 in hits.items():
        for (m2, p2, a2), v2 in hits.items():
            if m1 != m2 or 'Tau' not in p2 or 'Tau' in p1 or v1 <= 0:
                continue
            key = (p1, p2, m1)
            if key in seen:
                continue
            seen.add(key)
            print('  %-22s %s / %s = %.4f   (lattice 0.2597 +/- 0.0027)'
                  % (m1, p2, p1, v2 / v1))
    print('\nIf one of these lands on 0.2597, Hammer\'s C++ reproduces the '
          'refitted form factors\nand the last unvalidated link is closed.')


if __name__ == '__main__':
    main()
