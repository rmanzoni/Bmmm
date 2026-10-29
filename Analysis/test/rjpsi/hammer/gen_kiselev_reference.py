#!/usr/bin/env python3
r'''
TEST 3, step 1: the EvtGen truth reference for the Kiselev denominator.

Why this test exists
--------------------
Every validation so far concerns the NUMERATOR -- Harrison-2024 through BGL.
The weights actually used in the analysis are

    w = |M_Harrison|^2 / |M_Kiselev|^2 ,

and the denominator has never been checked. `KISELEV_INPUT` is empty, i.e.
Hammer is using its own built-in Kiselev defaults, which have never been
compared with the BC_VMN model that generated the sample:

    0.32853885 MyJ/psi mu-  anti-nu_mu  PHOTOS BC_VMN 1;
    0.32853885 MyJ/psi tau- anti-nu_tau PHOTOS BC_VMN 1;

(BcToJpsiMuMuInclusive.dec). If Hammer's Kiselev differs from EvtGen's BC_VMN
whichfit=1, every weight is wrong by the ratio of the two, with no symptom in
any q2 plot of the numerator.

Reading EvtGen's EvtBCVFF source and porting it would just move the risk: the
mapping of (F_V, F_A+, F_A0, F_A-) onto (V, A1, A2, A0) is exactly the sort of
convention that produced the 20x bug the first time round. So instead we ask
EvtGen itself.

The test
--------
  step 1 (this file): generate a GEN-only, UNFILTERED sample with the analysis
         DEC file, and write out the generated q2 per channel. No reco, no gen
         filter -- which is what makes it usable, since the production sample
         is gen-filtered and its q2 spectrum is biased.

  step 2: produce Hammer's own Kiselev spectrum and compare:

     python3 hammer_q2_from_scratch.py -o plots_kiselev -j 16 \
         --ff-target Kiselev \
         --reference evtgen_q2.root:tree:q2:code

     Agreement means Hammer's denominator is the model that generated the MC.
     Disagreement localises the problem to the denominator, and the size of the
     disagreement is the size of the bias in every weight.

  step 3 (cross-check on the parameters themselves):

     python3 -c "import hammer; h=hammer.Hammer(); h.show_available_ff_params('BctoJpsi')"

     against the EvtGen BC_VMN whichfit=1 values. AN-20-223 quotes a single-pole
     form, F(q2) = F(0)/(1 - q2/M_pole^2) with M_pole = 4.5 GeV and
     F_V = 0.11, F_A+ = -0.074, F_A0 = 5.9, F_A- = 0.12. Treat that as a
     cross-check of step 2, not as a substitute: the basis mapping is the part
     that is easy to get wrong.

Generating the reference
------------------------
This file is a template, not a runnable cmsRun config: the GEN fragment depends
on your CMSSW release and generator setup. What matters for the test is only
that the sample is (a) the same DEC file, (b) UNFILTERED, (c) gen level.

A minimal Pythia8+EvtGen GEN fragment looks like the block below; drop it into
a cmsRun config with no filter and no HLT, run a few 100k events, and then run
this script on the EDM output to extract q2.

    from Configuration.Generator.Pythia8CommonSettings_cfi import *
    generator = cms.EDFilter("Pythia8GeneratorFilter",
        comEnergy = cms.double(13600.),
        ExternalDecays = cms.PSet(
            EvtGen130 = cms.untracked.PSet(
                decay_table = cms.string('GeneratorInterface/EvtGenInterface/'
                                         'data/DECAY_2014_NOLONGLIFE.DEC'),
                particle_property_file = cms.FileInPath(
                    'GeneratorInterface/EvtGenInterface/data/evt_2014.pdl'),
                user_decay_file = cms.vstring(
                    'path/to/BcToJpsiMuMuInclusive.dec'),
                list_forced_decays = cms.vstring('MyBc+', 'MyBc-'),
                operates_on_particles = cms.vint32(541, -541),
            ),
            parameterSets = cms.vstring('EvtGen130'),
        ),
        # NOTE: no gen filter of any kind. That is the whole point.
    )

Then:

    python3 gen_kiselev_reference.py GEN.root -o evtgen_q2.root

which walks the gen particles, finds Bc -> J/psi l nu, computes
q2 = (p_Bc - p_Jpsi)^2, the W helicity angle cos(theta_W), and the channel code
(1 = mu, 7 = tau) with the same convention as gen_bc_decay, and writes a flat
tree. The angle is computed with ff_rate.cos_theta_w -- literally the function
used on the Hammer side -- so the comparison assumes no angular convention.
'''
import argparse
import math
import os
import sys

import numpy as np

# the angle must be measured with the SAME function as on the Hammer side, so
# that no convention is assumed anywhere: import it rather than re-derive it
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ff_rate import cos_theta_w        # noqa: E402

MU_PDG, TAU_PDG, JPSI_PDG, BC_PDG = 13, 15, 443, 541
NUMU_PDG, NUTAU_PDG, GAMMA_PDG = 14, 16, 22

# BcToJpsiMuMuInclusive.dec gives the two signal channels IDENTICAL forced
# branching fractions (0.32853885 each), so an unfiltered sample must contain
# them in a 1:1 ratio, within statistics. A generator-level muon filter would
# favour the mu channel (harder muon), so a significant departure from 1 is a
# direct symptom of filtering.
DEC_TAU_OVER_MU = 1.0


def p4(p):
    return np.array([p.energy(), p.px(), p.py(), p.pz()], dtype=float)


def m2(v):
    return v[0] ** 2 - v[1] ** 2 - v[2] ** 2 - v[3] ** 2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', help='GEN EDM files (FWLite-readable)')
    ap.add_argument('-o', '--output', default='evtgen_q2.root')
    ap.add_argument('-n', '--max-events', type=int, default=None)
    ap.add_argument('--label', default='genParticles',
                    help="gen-particle collection: 'genParticles' in GEN / "
                         "GEN-SIM / AODSIM, 'prunedGenParticles' in MINIAOD")
    args = ap.parse_args()

    from DataFormats.FWLite import Events, Handle

    handle = Handle('std::vector<reco::GenParticle>')

    rows = dict((k, []) for k in ('q2', 'q2_lnu', 'cos_theta_w',
                                  'cos_theta_w_lg', 'n_gamma', 'code'))
    n_events = n_bc = 0
    skipped = {'no_jpsi_daughter': 0, 'no_lepton': 0, 'no_neutrino': 0}

    events = Events(args.inputs)
    for i, event in enumerate(events):
        if args.max_events is not None and i >= args.max_events:
            break
        n_events += 1
        event.getByLabel(args.label, handle)
        if not handle.isValid():
            sys.exit('[FATAL] no %r collection in event %d. Try --label '
                     'prunedGenParticles for MINIAOD.' % (args.label, i))

        for gp in handle.product():
            # only the DECAYING copy of the Bc: earlier copies in the chain
            # have a single daughter (the next copy of themselves)
            if abs(gp.pdgId()) != BC_PDG or gp.numberOfDaughters() < 3:
                continue
            n_bc += 1
            daus = [gp.daughter(j) for j in range(gp.numberOfDaughters())]
            ids = [abs(d.pdgId()) for d in daus]

            # DIRECT J/psi daughter only: Bc -> psi(2S) l nu, chi_c l nu and the
            # hadronic modes are other entries of the DEC and must not enter
            if JPSI_PDG not in ids:
                skipped['no_jpsi_daughter'] += 1
                continue
            lep = [d for d in daus if abs(d.pdgId()) in (MU_PDG, TAU_PDG)]
            if not lep:
                skipped['no_lepton'] += 1
                continue
            nu = [d for d in daus if abs(d.pdgId()) in (NUMU_PDG, NUTAU_PDG)]
            if not nu:
                skipped['no_neutrino'] += 1
                continue

            jpsi = daus[ids.index(JPSI_PDG)]
            gammas = [d for d in daus if abs(d.pdgId()) == GAMMA_PDG]
            code = 1 if abs(lep[0].pdgId()) == MU_PDG else 7

            # two definitions of q2. They differ only through PHOTOS:
            #   q2      = (p_Bc - p_Jpsi)^2      includes FSR photons with the W*
            #   q2_lnu  = (p_l  + p_nu)^2        the lepton pair alone
            # The first is the one to compare with Hammer (the W* virtuality);
            # the spread between the two is printed as a PHOTOS diagnostic.
            rows['q2'].append(m2(p4(gp) - p4(jpsi)))
            rows['q2_lnu'].append(m2(p4(lep[0]) + p4(nu[0])))
            # the W helicity angle, from the charged lepton as generated...
            rows['cos_theta_w'].append(cos_theta_w(p4(gp), p4(jpsi), p4(lep[0])))
            # ...and with the FSR photons added back to it, a pre-PHOTOS proxy.
            # Their difference is the size of the PHOTOS effect on the angle.
            lg = p4(lep[0]) + sum((p4(g) for g in gammas), np.zeros(4))
            rows['cos_theta_w_lg'].append(cos_theta_w(p4(gp), p4(jpsi), lg))
            rows['n_gamma'].append(float(len(gammas)))
            rows['code'].append(float(code))

    import uproot
    data = dict((k, np.asarray(v, dtype=np.float64)) for k, v in rows.items())
    with uproot.recreate(args.output) as fout:
        fout.mktree('tree', dict((k, 'float64') for k in data))
        fout['tree'].extend(data)

    # ---------------------------------------------------------------- report
    print('events read: %d   decaying Bc: %d' % (n_events, n_bc))
    for why, n in skipped.items():
        print('  skipped (%s): %d' % (why, n))
    n_mu = int((data['code'] == 1).sum())
    n_tau = int((data['code'] == 7).sum())
    print('kept: mu %d   tau %d' % (n_mu, n_tau))

    if n_mu and n_tau:
        ratio = n_tau / float(n_mu)
        err = ratio * math.sqrt(1. / n_tau + 1. / n_mu)
        pull = (ratio - DEC_TAU_OVER_MU) / err
        print('\nUNFILTERED CHECK: n_tau/n_mu = %.4f +/- %.4f   (DEC: %.1f)   '
              'pull %+.1f sigma' % (ratio, err, DEC_TAU_OVER_MU, pull))
        if abs(pull) > 3:
            print('  [WARN] inconsistent with the DEC branching fractions: the '
                  'sample looks FILTERED.\n         Its q2 spectrum is biased and '
                  'it cannot serve as the reference.')

    if data['q2'].size:
        d = data['q2'] - data['q2_lnu']
        withg = data['n_gamma'] > 0
        print('\nPHOTOS DIAGNOSTIC: %.1f%% of decays have FSR photons at the Bc '
              'vertex;' % (100. * withg.mean()))
        # without photons the two definitions coincide exactly, so look only
        # at the events that radiated: that is where PHOTOS could matter
        if withg.any():
            dg = d[withg]
            print('  among those, q2 - q2_lnu: mean %.4f, std %.4f GeV^2 '
                  '(zero by construction without photons)' % (dg.mean(), dg.std()))
            print('  -> compare with a bin width of the q2 plots (~0.25 GeV^2): '
                  'if much smaller, FSR is irrelevant to this test')
        if withg.any():
            dc = (data['cos_theta_w'] - data['cos_theta_w_lg'])[withg]
            print('  cos(theta_W), lepton vs lepton+photons: mean %.4f, std %.4f '
                  '(compare with a cos bin width of 0.1)' % (dc.mean(), dc.std()))
        nophot = d[~withg]
        if nophot.size and np.max(np.abs(nophot)) > 1e-6:
            print('  [WARN] q2 != q2_lnu in events WITHOUT photons (max %.2e): '
                  'the Bc daughters do not balance -- check the topology'
                  % np.max(np.abs(nophot)))

    for name, code in (('mu', 1), ('tau', 7)):
        sel = data['code'] == code
        if sel.any():
            c = data['cos_theta_w'][sel]
            afb = ((c > 0).sum() - (c < 0).sum()) / float(c.size)
            print('A_FB (EvtGen, %s) = %+.4f +/- %.4f'
                  % (name, afb, math.sqrt(max(1. - afb * afb, 0.) / c.size)))

    print('\nwrote %s  (tree: q2, q2_lnu, cos_theta_w, cos_theta_w_lg, n_gamma, '
          'code)' % args.output)
    print('now:  python3 hammer_q2_from_scratch.py -o plots_kiselev -j 16 '
          '--ff-target Kiselev --reference %s:tree:q2:code' % args.output)


if __name__ == '__main__':
    main()
