#!/usr/bin/env python3
r'''
TEST 2 of the FF validation: the observables that dGamma/dq^2 cannot see.

Everything validated so far -- the q2 spectra, the tau/mu ratio, R(J/psi) --
depends on the helicity amplitudes only through

    H+^2 + H-^2  =  2 f^2 + 2 M^2 |p|^2 g^2 ,

which is EVEN in g. A sign flip of g, or any g <-> f confusion preserving that
sum, is invisible in every test done up to now. The observables below are not:
A_FB carries H-^2 - H+^2 and the H0 H_t interference, A_lambda carries H_t
against the rest, F_L separates the longitudinal piece.

Harrison arXiv:2503.15090 eq. (35), for the tau channel:

    A_lambda_tau = 0.5093(42)    F_L^{J/psi} = 0.4421(55)    A_FB = -0.0567(61)

    python3 check_angular_observables.py
    python3 check_angular_observables.py --card /path/to/card.json

Nothing here involves Hammer, the fit, or any Monte Carlo: it reads the card,
evaluates the helicity amplitudes, and integrates. It is therefore an
independent probe of the same coefficients the rest of the chain uses.

On the A_FB sign: Harrison's eq. (34) defines A_FB with an explicit minus sign
and integrates cos(theta_W) d(theta_W) rather than d(cos theta_W). The
magnitude is the physics; the sign and a few-percent normalisation difference
follow his angular convention, so compare |A_FB| unless you have reproduced his
measure exactly.
'''
import argparse

from Bmmm.Analysis.HammerFF import DEFAULT_CARD, load_card

import ff_rate as FR

# Harrison arXiv:2503.15090 eq. (35), tau channel
PUBLISHED = {
    'A_lambda': (0.5093, 0.0042),
    'F_L': (0.4421, 0.0055),
    'A_FB': (-0.0567, 0.0061),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--card', default=DEFAULT_CARD)
    ap.add_argument('--abs-afb', action='store_true', default=True,
                    help='compare |A_FB| (default): see the note above')
    args = ap.parse_args()

    card = load_card(args.card)
    vec = FR.coeffs_from_card(card)
    print('card %r (fit %s)' % (card.get('name'), card.get('fit_date')))

    for lep, mlep in (('mu', FR.M_MU), ('tau', FR.M_TAU)):
        obs = FR.angular_observables(vec, mlep)
        print('\n%s channel' % lep)
        print('  internal closure of the two lepton-helicity pieces against '
              'dgamma_dq2: %.2e' % obs['rate_closure'])
        for key in ('A_lambda', 'F_L', 'A_FB'):
            val = obs[key]
            line = '  %-9s = %+.4f' % (key, val)
            if lep == 'tau':
                ref, err = PUBLISHED[key]
                a, b = (abs(val), abs(ref)) if (key == 'A_FB' and args.abs_afb) \
                    else (val, ref)
                pull = (a - b) / err
                line += '   published %+.4f +/- %.4f   pull %+.1f sigma' % (ref, err, pull)
            print(line)

    print('\nThe tau-channel numbers are the test. A_lambda and F_L are '
          'convention-free;\nA_FB should match in magnitude. Agreement here '
          'constrains the relative sign\nof g and f and the size of H_t, none '
          'of which the q2 spectrum can determine.')


if __name__ == '__main__':
    main()
