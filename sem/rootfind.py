#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: rootfind.py
Author: Nicholas Chisholm
Email: nchishol@andrew.cmu.edu
Description: Routines for root finding.
"""

import numpy as np
import scipy.linalg as la


class SolverFailure(Exception):
    """Base exception to raise upon the failure of a non-linear solver to find
    a 'good' solution.
    """
    pass


def newton(f, x0, jac, it_max, tol):
    """Use Netwon-Raphson iteration to find the roots of a vector-valued
    function.

    f : callable
        A vector-valued function whose roots to compute
    x0 : ndarray
        Initial guess
    jac : callable
        Accepts the same arguments as `f` and returns the Jacobian
        matrix of `f`.
    it_max : int
        Maximum number of iterations to perform
    tol : float
        tolerance for termination
    returns : TODO
    """
    # number of times the L2 norm of the solution fails to be reduced.
    # n_diverge = 0
    # normL2_last = np.inf

    x = x0[:]

    for itn in range(it_max):
        f_x = f(x)
        jac_x = jac(x)
        dx = la.solve(jac_x, -f_x)
        x += dx
        if np.isclose(la.norm(dx), 0., atol=tol):
            return x
    raise SolverFailure("Maximum number of iterations exceeded before"
                        "tolerence could be met.")
