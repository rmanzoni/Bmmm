#!/usr/bin/env python3
'''Unit tests of the statistics in cfcore (numpy only):

    python3 -m unittest test_compare_fits -v
'''

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cfcore import (wrap_angle, robust_stats, weighted_quantile, rank_correlation, auc, auc_oriented,
                    roc_curve, equal_pop_edges, bin_index, fisher_yields, ratio_rel_error, fisher_ratios)
from cfio import residual


def auc_brute(s, b, ws, wb):
    '''P(s > b) + P(s = b) / 2 by explicit pairs.'''
    gt = (s[:, None] > b[None, :]).astype(float)
    eq = (s[:, None] == b[None, :]).astype(float)
    W = ws[:, None] * wb[None, :]
    return np.sum(W * (gt + 0.5 * eq)) / np.sum(W)


class TestRobust(unittest.TestCase):

    def test_gaussian(self):
        r = np.random.default_rng(1).normal(0.3, 2., 200000)
        st = robust_stats(r)
        self.assertAlmostEqual(st['median'], 0.3, delta=0.02)
        self.assertAlmostEqual(st['s68'], 2., delta=0.02)
        self.assertAlmostEqual(st['s95'], 2., delta=0.03)
        self.assertAlmostEqual(st['tail3'], 0.0027, delta=0.0006)

    def test_nan_counted(self):
        r = np.array([0., 1., np.nan, -1., np.inf])
        st = robust_stats(r)
        self.assertEqual(st['n'], 3)                # finite entries
        self.assertAlmostEqual(st['frac_ok'], 0.6)

    def test_weighted_quantile(self):
        x = np.array([1., 2., 3.])
        self.assertAlmostEqual(weighted_quantile(x, [0.5], np.array([1., 1., 10.]))[0], 3., delta=0.5)
        self.assertAlmostEqual(weighted_quantile(x, [0.5])[0], 2., delta=1e-9)

    def test_wrap(self):
        self.assertAlmostEqual(float(wrap_angle(np.array([3 * np.pi / 2]))[0]), -np.pi / 2)
        self.assertAlmostEqual(float(residual(np.array([3.]), np.array([-3.]), 'angle')[0]), 6 - 2 * np.pi)
        self.assertAlmostEqual(float(residual(np.array([1.1]), np.array([1.]), 'rel')[0]), 0.1)

    def test_rank_correlation(self):
        x = np.random.default_rng(2).normal(size=1000)
        self.assertAlmostEqual(rank_correlation(x, np.exp(x)), 1., places=9)
        self.assertAlmostEqual(rank_correlation(x, -x ** 3), -1., places=9)


class TestAUC(unittest.TestCase):

    def test_against_pairs(self):
        g = np.random.default_rng(3)
        s = np.round(g.normal(0.5, 1, 300), 1)     # rounded: ties
        b = np.round(g.normal(0., 1, 400), 1)
        ws, wb = g.uniform(0.2, 2, 300), g.uniform(0.2, 2, 400)
        self.assertAlmostEqual(auc(s, b, ws, wb), auc_brute(s, b, ws, wb), places=12)
        self.assertAlmostEqual(auc(s, b), auc_brute(s, b, np.ones(300), np.ones(400)), places=12)

    def test_orientation(self):
        g = np.random.default_rng(4)
        s, b = g.normal(-1, 1, 1000), g.normal(0, 1, 1000)
        a, sign = auc_oriented(s, b)
        self.assertEqual(sign, -1)
        self.assertAlmostEqual(a, 1 - auc(s, b), places=12)

    def test_limits(self):
        self.assertEqual(auc(np.array([2., 3.]), np.array([0., 1.])), 1.)
        self.assertEqual(auc(np.array([1., 1.]), np.array([1., 1.])), 0.5)
        self.assertTrue(np.isnan(auc(np.array([np.nan]), np.array([1.]))))

    def test_roc_area(self):
        g = np.random.default_rng(5)
        s, b = g.normal(1, 1, 20000), g.normal(0, 1, 20000)
        eff_b, eff_s = roc_curve(s, b, n_points=400)[:2]
        o = np.argsort(eff_b)
        trapz = getattr(np, 'trapezoid', None) or np.trapz
        area = trapz(np.asarray(eff_s)[o], np.asarray(eff_b)[o])
        self.assertAlmostEqual(area, auc(s, b), delta=0.01)


class TestBinning(unittest.TestCase):

    def test_equal_pop(self):
        x = np.random.default_rng(6).exponential(size=10000)
        e = equal_pop_edges(x, 10)
        self.assertEqual(e[0], -np.inf)
        self.assertEqual(e[-1], np.inf)
        n = np.histogram(x, e)[0]
        self.assertTrue(np.all(np.abs(n - 1000) <= 2))

    def test_bin_index(self):
        e1, e2 = np.array([-np.inf, 0., np.inf]), np.array([-np.inf, 1., 2., np.inf])
        idx, nb = bin_index([np.array([-1., 1., 1., np.nan]), np.array([0., 0., 5., 0.])], [e1, e2])
        self.assertEqual(nb, 7)
        self.assertEqual(list(idx), [0, 1, 5, 6])


class TestFisher(unittest.TestCase):

    def test_one_template(self):
        cov, _ = fisher_yields(np.array([[0.2, 0.3, 0.5]]), np.array([1000.]))
        self.assertAlmostEqual(cov[0, 0], 1000., places=6)        # Poisson

    def test_disjoint_templates(self):
        t = np.array([[0.5, 0.5, 0., 0.], [0., 0., 0.4, 0.6]])
        cov, _ = fisher_yields(t, np.array([300., 700.]))
        np.testing.assert_allclose(np.diag(cov), [300., 700.])
        self.assertAlmostEqual(cov[0, 1], 0., places=8)
        self.assertAlmostEqual(ratio_rel_error(cov, np.array([300., 700.]), 0, 1),
                               np.sqrt(1 / 300. + 1 / 700.), places=10)

    def test_two_overlapping(self):
        # two templates over two bins: invert the 2x2 by hand
        t = np.array([[0.8, 0.2], [0.3, 0.7]])
        N = np.array([500., 800.])
        nu = N @ t
        F = np.array([[np.sum(t[a] * t[b] / nu) for b in range(2)] for a in range(2)])
        cov, _ = fisher_yields(t, N)
        np.testing.assert_allclose(cov, np.linalg.inv(F), rtol=1e-10)

    def _classes(self, shift, n=20000, seed=7):
        '''Four classes in two variables: x separates tau from mu by `shift`,
        y separates Ds from Ds* perfectly.'''
        g = np.random.default_rng(seed)
        cols = []
        for c in range(4):
            tau, st = c >= 2, c in (1, 3)
            cols.append([g.normal(shift * tau, 1, n), g.uniform(0, 1, n) + 2. * st])
        return cols

    def test_perfect_separation(self):
        cols = self._classes(shift=100.)
        ws = [np.ones(len(c[0])) for c in cols]
        edges = [np.array([-np.inf, 50., np.inf]), np.array([-np.inf, 1.5, np.inf])]
        f = fisher_ratios(cols, ws, edges)
        self.assertAlmostEqual(f['dil_ds'], 1., places=6)
        self.assertAlmostEqual(f['dil_dsst'], 1., places=6)
        self.assertAlmostEqual(f['dil_taumu'], 1., places=6)
        self.assertAlmostEqual(f['r_ds'], np.sqrt(2. / 20000), places=8)

    def test_scaling_and_ordering(self):
        ws = [np.ones(20000)] * 4
        edges = [equal_pop_edges(np.concatenate([c[0] for c in self._classes(1.)]), 20),
                 np.array([-np.inf, 1.5, np.inf])]
        weak = fisher_ratios(self._classes(0.5), ws, edges, n_ref=1e5)
        strong = fisher_ratios(self._classes(2.), ws, edges, n_ref=1e5)
        self.assertGreater(weak['dil_taumu'], strong['dil_taumu'])
        self.assertGreater(weak['r_ds'], strong['r_ds'])
        self.assertAlmostEqual(strong['r_ds_count'], np.sqrt(2. / 1e5), places=10)   # scaled to n_ref

    def test_taumu_ignores_ds_dsst(self):
        # without the y variable Ds and Ds* are indistinguishable: the
        # four-yield fit is singular, the tau/mu one is not
        cols = [[c[0]] for c in self._classes(2.)]
        cols[1], cols[3] = cols[0], cols[2]       # Ds* templates = Ds templates, exactly
        ws = [np.ones(20000)] * 4
        edges = [equal_pop_edges(np.concatenate([c[0] for c in cols]), 20)]
        f = fisher_ratios(cols, ws, edges)
        self.assertTrue(not np.isfinite(f['r_ds']) or f['dil_ds'] > 100)
        self.assertTrue(1. < f['dil_taumu'] < 3.)


if __name__ == '__main__':
    unittest.main()
