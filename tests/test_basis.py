#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: test_basis.py
Author: Nicholas Chisholm
Email: nchishol@andrew.cmu.edu
Description: Tests for shape functions.
"""

import numpy as np
import time
import unittest
import sem.basis


def scalar_func1d(x):
    return np.sin(np.pi*x)

def scalar_func1d_deriv(x):
    return np.pi*np.cos(np.pi*x)

def vector_valued_func1d(x):
    out = np.zeros((2, x.size))
    out[0] = scalar_func1d(x)
    out[1] = scalar_func1d(x - 1)
    return out

def vector_valued_func1d_deriv(x):
    out = np.zeros((2, x.size))
    out[0] = scalar_func1d_deriv(x)
    out[1] = scalar_func1d_deriv(x - 1)
    return out

def scalar_func2d(x, on_grid=False):
    if on_grid:
        x = np.meshgrid(*x, indexing='ij', sparse=True)
    return x[0] * x[1]

def vector_func2d(x, on_grid=False):
    if on_grid:
        shape = (2,) + tuple(x_c.size for x_c in x)
        x = np.meshgrid(*x, indexing='ij', sparse=True)
        out = np.zeros(shape)
    else:
        out = np.zeros((2, x.shape[1]))
    out[0] = x[0] * x[1]
    out[1] = x[0] + x[1]
    return out


class TestLagrangeAtGaussLobatto(unittest.TestCase):

    basis = sem.basis.LagrangeAtGaussLobatto(9)

    def setUp(self):
        self.xx = np.linspace(-1, 1)
        self.quad_grid = self.basis.get_quadrature_rule().get_abscissa()

    def test_kronecker_delta_property(self):
        basis = self.basis
        quad_grid = self.quad_grid
        restricted_fn_space = basis(quad_grid)
        expected_result = np.zeros([basis.n_coeffs]*2)
        expected_result[np.diag_indices_from(expected_result)] = 1.
        self.assertTrue(np.allclose(restricted_fn_space, expected_result))

    def test_interpolation(self):
        xx = np.linspace(-1, 1)
        y_support = scalar_func1d(xx)
        quad_grid = self.quad_grid
        coeffs = scalar_func1d(quad_grid)
        yy_interp = self.basis.interpolate(coeffs, xx)
        self.assertTrue(np.allclose(yy_interp, y_support, rtol=1e-2, atol=1e-4),
                        "scalar function interpolation failed.")

    def test_interpolation_vector(self):
        xx = np.linspace(-1, 1)
        y_support = vector_valued_func1d(xx)
        quad_grid = self.quad_grid
        coeffs = vector_valued_func1d(quad_grid)
        yy_interp = self.basis.interpolate(coeffs, xx)
        self.assertTrue(yy_interp.shape == y_support.shape)
        self.assertTrue(np.allclose(yy_interp, y_support, rtol=1e-2, atol=1e-4))

    def test_differentiation(self):
        quad_grid = self.quad_grid
        coeffs = scalar_func1d(quad_grid)
        dy = scalar_func1d_deriv(quad_grid)
        dy_result = self.basis.deriv(coeffs)
        self.assertTrue(np.allclose(dy, dy_result, rtol=1e-2, atol=1e-4))

    def test_differentiation_vector(self):
        quad_grid = self.quad_grid
        coeffs = vector_valued_func1d(quad_grid)
        dy = vector_valued_func1d_deriv(quad_grid)
        dy_result = self.basis.deriv(coeffs)
        self.assertTrue(np.allclose(dy, dy_result, rtol=1e-2, atol=1e-4))

    def test_integration(self):
        # TODO: could test if a degree 2N-1 polynomial is integrted exactly
        quad_grid = self.quad_grid
        coeffs = quad_grid + 1
        integral = self.basis.integrate(coeffs)
        self.assertTrue(np.isclose(integral, 2.0))


class TestTensorProductSupported(unittest.TestCase):

    basis = sem.basis.TensorProductSupported(
        sem.basis.LagrangeAtGaussLobatto(5),
        sem.basis.LagrangeAtGaussLobatto(6))

    def setUp(self):
        # quadrature points in each direction
        quadrature = self.basis.get_quadrature_rule()
        self.quad_grid = quadrature.abscissa
        # equispaced grid of support points
        self.support_grid = [np.linspace(-1, 1, len(quad_pts))
                             for quad_pts in self.quad_grid]

    def test_kronecker_delta_property(self):
        basis = self.basis
        quad_grid = np.meshgrid(*self.quad_grid, indexing='ij', sparse=True)
        restricted_fn_space = basis.vandermonde_matrix(quad_grid)
        expected_result = np.zeros([basis.n_coeffs]*2)
        expected_result[np.diag_indices_from(expected_result)] = 1.
        self.assertTrue(np.allclose(restricted_fn_space, expected_result))

    def test_interpolation(self):
        # Pick some random points in [-1, 1]^(ndim)
        x = 2*np.random.random((2, 50)) - 1
        coeffs = vector_func2d(self.quad_grid, True)
        y = vector_func2d(x)
        y_interp = self.basis.interpolate(coeffs, x)
        self.assertTrue(np.allclose(y_interp, y), "interpolation failed")

    def test_interpolation_on_grid(self):
        fine_grid = [np.linspace(-1, 1, 50), np.linspace(-1, 1, 49)]
        for func in [scalar_func2d, vector_func2d]:
            coeffs = func(self.quad_grid, on_grid=True)
            y = func(fine_grid, on_grid=True)
            y_interp = self.basis.interpolate_on_grid(coeffs, fine_grid)
            self.assertTrue(np.allclose(y_interp, y),
                            "interpolation failed.")

    def test_coeff_computation(self):
        tpb = self.basis
        support_grid = self.support_grid
        for func in [scalar_func2d, vector_func2d]:
            coeffs = func(self.quad_grid, on_grid=True)
            support_values = func(support_grid, on_grid=True)
            computed_coeffs = tpb.compute_coeffs_grid(support_values,
                                                      support_grid)
            self.assertTrue(np.allclose(computed_coeffs, coeffs))

    def test_gradient(self):
        coeffs = vector_func2d(self.quad_grid, on_grid=True)
        grad = self.basis.gradient(coeffs)

    def test_integration(self):
        coeffs = vector_func2d(self.quad_grid, on_grid=True)
        integ = self.basis.integrate(coeffs)


if __name__ == '__main__':
    unittest.main()
else:
    suite = unittest.TestLoader().loadTestsFromTestCase(
        TestTensorProductSupported)
    suite.debug()
