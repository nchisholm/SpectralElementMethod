#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module produces the nodes and barycentric weights for a Lagrange
interpolating basis through the roots of orthoganal polynomials.  Thus, these
bases have associated Gaussian quadrature rules, and the quadrature weights are
also computed.  The data generated can be pre-computed and stored or generated
on-the-fly (although this may be slow for >10 nodes).
"""

import sympy as sym
from sympy import pi, cos, Rational as Rat, Eq
from mpmath import mp
import numpy as np
import h5py


def gauss_legendre_lobatto(n):
    r"""
    Compute the nodes and barycentric weights of a Lagrange interpolation
    basis through the Gauss-Legendre-Lobatto quadrature points; the quadrature
    weights are also computed.

    The endpoints of the interval [-1, 1] in x are included as nodes.

    Parameters
    ----------
    n : int
        Number of nodes in the interpolation basis / quadrature rule.

    Returns
    -------
    nodes : mpmath.matrix
        The non-negative `n`-point Gauss-Legendre-Lobatto quadrature nodes,
        which are symmetric about x=0.
    bary_wts : mpmath.matrix
        The `n` barycentric Lagrange interpolation weights associated with each
        node.
    quad_wts : mpmath.matrix
        The `n` quadrature weights associated with each node, which are
        symmetric about x=0.

    Notes
    -----
    Nodes at the interval endpoints (-1 and 1) are included by definition.  The
    interior quadrature nodes are given by the roots of :math:`P'_{n-1}(x)`,
    where :math:`P_{k}` is the :math:`k`th-degree Legendre polynomial.
    Equivalently, the nodes are given by the roots of the Jacobi polynomial
    :math:`P_{n-2}^{(1,1)}(x)`.  The nodes are symmetric about zero; thus, only
    values for the *non-negative* nodes are computed by this function.

    The nodes are first guessed as :math:`x_i \sim cos(\pi i / (n-1))` (the
    extrema of the :math:`n-1` degree Chebyshev polynomials) for :math:`i = 1,
    2, \dots, n-2`, and are then resolved using Newton iteration.  From this,
    the barycentric weights :math:`b_i = 1/P_{n-1}(x_i)` and quadrature weights
    :math:`w_i = 1/[n(n-1)P_{n-1}(x_i)^2]` are computed for each node
    :math:`x_i` where :math:`P_{k}` is the k-th order Legendre polynomial.

    Note that :math:`\sum_{i=0}^{n-1} w_i = 2` (a fact which is realized upon
    integrating the interpolant of unity).  Also, the barycentric weights are
    symmetric about zero for odd `n` and anti-symmetric for even `n`.  Hence,
    their sum vanishes.

    TODO: add references
    """

    if n < 2:
        raise ValueError("At least two quadrature points are required")
    deg = n - 1

    # the quadrature nodes are the extrema of a Legendre polynomial
    x = sym.symbols('x', real=True)
    legp_expr = sym.legendre_poly(deg, x)
    dlegp_expr = legp_expr.diff(x)
    legp = sym.lambdify(x, legp_expr, modules='mpmath', dummify=False)

    nodes = mp.matrix(1, deg//2 + 1)

    k = 0
    if deg % 2 == 0:
        nodes[0] = 0.
        k += 1
    # machine epsilon for double precision floats
    macheps = np.finfo(np.float64).epsneg
    while k < deg // 2:
        # make a rough approximation of the root before computing it via
        # Newton's method to machine precision.
        root_guess = cos(pi * Rat(deg//2 - k, deg)).n()
        root = sym.nsolve(Eq(dlegp_expr, 0), x, root_guess, tol=macheps,
                          solver='newton')
        nodes[k] = root
        k += 1
    nodes[k] = 1.

    # evaluate the barycentric weights and quadrature weights
    bary_wts = mp.matrix(1, deg//2 + 1)
    quad_wts = mp.matrix(1, deg//2 + 1)
    for i in range(deg//2 + 1):
        bary_wts[i] = 1 / legp(nodes[i])
        quad_wts[i] = bary_wts[i]**2        # ... up to constant factor
    # quadrature weights sum to 2
    if deg % 2 == 0:
        quad_wt_sum = quad_wts[0] + 2 * sum(quad_wts[1:])
    else:
        quad_wt_sum = 2 * sum(quad_wts)
    quad_wts *= 2 / quad_wt_sum

    return nodes, bary_wts, quad_wts


def write_data(fpath, max_order=10):
    """
    Save pre-computed Lagrange basis data to an HDF5 file.

    Parameters
    ----------
    fpath : str
        Path to file in which to store the data.
    max_order : int
        Compute data for Lagrange bases up to this order.
    """
    with h5py.File(fpath, 'w') as f:
        grp = f.require_group('GaussLegendreLobatto')
        grp.attrs['max_order'] = max_order
        for n_pts in range(2, max_order + 2):
            basis_data = np.array(gauss_legendre_lobatto(n_pts), float)
            order = n_pts - 1
            grp.create_dataset(str(order), data=basis_data)
