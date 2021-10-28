#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from . import mesh as meshplt2d


def new_mpl_fig():
    fig = plt.figure()
    ax = fig.gca()
    return ax


def triangulate_data(dof_mngr, coeffs):
    tri = meshplt2d.triangulate(dof_mngr.mesh)
    values = dof_mngr.values_at_nodes(coeffs)
    return tri, values


def tricontour(dof_mngr, soln_vec, ax=None, **kwargs):
    if ax is None:
        ax = new_mpl_fig()
    tri, u_eq = triangulate_data(dof_mngr, soln_vec)
    return ax.tricontour(tri, u_eq, **kwargs)


def tricontourf(dof_mngr, soln_vec, ax=None, **kwargs):
    if ax is None:
        ax = new_mpl_fig()
    tri, u_eq = triangulate_data(dof_mngr, soln_vec)
    return ax.tricontourf(tri, u_eq, **kwargs)


def surface(dof_mngr, soln_vec, ax=None, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    tri, u_eq = triangulate_data(dof_mngr, soln_vec)
    return ax.plot_trisurf(tri, u_eq, **kwargs)
