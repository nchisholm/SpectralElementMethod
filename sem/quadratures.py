# -*- coding: utf-8 -*-
"""
Module for defining and using numerical quadratures.

Created on Mon Aug 18 10:48:26 2014
"""



import numpy as np
from numpy.polynomial.legendre import Legendre


class Quadrature1D(object):
    # Abstract base class for quadrature rules
    """ An `n`-point quadrature rule.

    Attributes
    ----------
    abscissa : numpy.ndarray
        The `n` quadrature points on the interval [-1, 1].
    weights : numpy.ndarray
        The `n` quadrature weights.

    If called, apply the integration rule to a "function" of a single
    variable over the interval [-1, 1].

    Parameters
    ----------
    func : callable or array-like
        * If `func` is callable, try to evaluate at the quadrature
        points and then sum over the weights.
        * If `func` is array-like, then the values of the array are
        assumed to be the values of the function at the integration
        points.

    Returns
    -------
    integral : int
        The value of the definite integral.

    Notes
    -----
    Integration is always assumed to be over the interval [-1, 1].
    """

    @property
    def ndim(self):
        return 1

    @property
    def n_points(self):
        return len(self._abscissa)

    def __init__(self, abscissa, weights):
        self._abscissa = abscissa
        self._weights = weights

    def __call__(self, f):
        """
        Apply the integration rule to a "function" of a single
        variable over the interval [-1, 1].

        Parameters
        ----------
        f : callable or array-like
            If `func` is callable, try to evaluate at the quadrature
            points and then sum over the weights.
            If `func` is array-like, then the values of the array are
            assumed to be the values of the function at the integration
            points.

        Returns
        -------
        integral : int
            The value of the definite integral.
        """

        try:
            return np.dot(self._weights, f)
        except TypeError:
            return np.dot(self._weights, f(self._abscissa))

    @property
    def abscissa(self):
        return self._abscissa

    @property
    def weights(self):
        return self._weights

    def get_abscissa(self):
        return self._abscissa

    def get_weights(self):
        return self._weights

    def integrate(self, values):
        """
        Apply the integration rule given a set of values at the quadrature
        points.
        """
        weights = self._weights
        assert values.shape[0] == weights.size
        rank_shape = values.shape[1:]
        values = values.reshape(weights.size, -1)
        result = np.dot(weights, values)
        result.shape = rank_shape
        return result

    def xweight(self, f_vals):
        """Return values at quadrature points multiplied by the quadrature
        weights (but not yet summed).
        """
        return f_vals * self._weights

    def __repr__(self):
        return "{}(n={})".format(self.__class__.__name__, self.n_points)


class GaussLobatto(Quadrature1D):
    r"""
    The Gauss-Lobatto quadrature is similar to Gaussian quadrature. It
    integrates exactly polynomials of degree 2`n`-3 or less. Integration
    abscissa include end points of the integration interval, so there are n-2
    free abscissa.

    Notes:
    ------
    The quadrature points are given as

    .. math::
        \xi_i =
        \begin{cases}
            -1 & i = 0 \\
            \text{roots of}\ \frac{dL_P}{d\xi} & i = 1, 2, ..., P-1 \\
            1 & i = P
        \end{cases}

    Where :Math:`L_P` is the P\ :sup:`th` Legendre polynomial. The weights
    are evaluated as [Can98]_

        .. math::
            w_i = \frac{2}{P(P+1)} \frac{1}{L_P(\xi_i)^2}
            \quad i = 0, ..., P.
    """

    def __init__(self, n):
        r"""
        Defines an `n`-point Gauss-Legendre-Lobatto quadrature rule.

        Parameters
        ----------
        n : int
            The number of points and weights in the quadrature rule.

        Notes
        -----
        The algorithm used is very similar to the
        ``numpy.polynomial.legendre.leggauss`` function.
        """
        int_n = int(n)
        if int_n != n or int_n < 1:
            raise ValueError("n must be a positive integer")

        # The n-2 *interior* quadrature abscissa are roots of the derivative
        # of the degree-(n-1) Legendre polynomial.
        leg_pn = Legendre.basis(n-1)
        dleg_pn = leg_pn.deriv()

        # first approximation of roots using the companion matrix
        x = np.zeros(n)
        x[0] = -1.  # Integration end points are abscissa
        x[-1] = 1.
        x[1:-1] = dleg_pn.roots()

        # improve the interior roots by applying one Newton-Raphson iteration
        d2leg_pn = dleg_pn.deriv()
        x[1:-1] -= dleg_pn(x[1:-1]) / d2leg_pn(x[1:-1])

        # compute the weights up to a scale factor
        wt = np.ones(n)     # weights
        wt[1:-1] /= leg_pn(x[1:-1])**2

        # symmetrize the interior quadrature points and weights
        x[1:-1] = (x[1:-1] - x[-2:0:-1])/2.
        wt[1:-1] = (wt[1:-1] + wt[-2:0:-1])/2.

        # rescale the weights (they must sum to 2)
        wt *= 2. / wt.sum()

        self._abscissa = x
        self._weights = wt

    @property
    def deg(self):
        """
        Degree of the polynomial integrated exactly by the quadrature rule.
        """
        return 2*len(self._abscissa) - 3


class TensorQuadratureRule(object):

    @property
    def ndim(self):
        return self._ndim

    @property
    def n_points(self):
        return self._n_points

    @property
    def n_subquads(self):
        return len(self.weights)

    @property
    def shape(self):
        return tuple(len(abscissa) for abscissa in self._abscissa)

    @property
    def abscissa(self):
        return self._abscissa[:]

    @property
    def weights(self):
        return self._weights[:]

    def __init__(self, *quad_rules):
        self._ndim = 0
        self._n_points = 1
        self._abscissa = []
        self._weights = []
        for rule in quad_rules:
            self._ndim += rule.ndim
            self._n_points *= rule.abscissa.size
            self._abscissa.append(rule.abscissa)
            self._weights.append(rule.weights)

    def get_abscissa(self, sparse=False):
        """Return the abscissa of the quadrature rule in a sparse or dense
        format."""
        # TODO: May not work for quadrature rules > 1D
        return np.meshgrid(*self.abscissa, indexing='ij', sparse=sparse)

    def get_weights(self, sparse=False):
        """Return the quadrature weights in a sparse or dense format."""
        wt_grid = np.meshgrid(*self._weights, indexing='ij', sparse=sparse)
        if sparse:
            return wt_grid
        else:
            return np.prod(wt_grid)

    def __call__(self, f):
        try:
            # Assume input is an array of values at the abscissa
            return self.integrate(f)
        except TypeError:
            # Perhaps we are integrating a function (callable) instead
            return self.integrate(f(self._abscissa))

    def integrate(self, f_vals):
        result = f_vals
        for wt in reversed(self.weights):
            result = np.inner(result, wt)
        return result

    def xweight(self, f_vals):
        """Return values at quadrature points multiplied by the quadrature
        weights (but not yet summed).
        """
        out = f_vals.copy()
        for wt1d in self.get_weights(sparse=True):
            out *= wt1d
        return out

    # TODO: __repr__() method
