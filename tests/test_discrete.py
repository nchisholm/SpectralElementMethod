#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import unittest
from sem.discrete import *
import sem.geometry
import sem.basis_functions
import sem.grid_importers


class TestDiscrete(object):

    def setUp(self):
        mesh = sem.grid_importers.load_msh("./mesh/square.msh", ndim=2)
        dof_mngr = DOFManager(mesh)


class TestFiniteElements(unittest.TestCase):

    def setUp(self):
        # Produce a 2D test mesh with a single quadrilateral cell
        # TODO: transform nodes to something other than a unit cell on [-1, 1]
        # Set up the mesh
        geometry = sem.geometry.Quadrilateral(9, 9)
        mesh = Mesh(geometry.ndim)
        nodes_slc = tuple(slice(-1, 1, s*1j) for s in geometry.shape)
        nodes = np.mgrid[nodes_slc].reshape(geometry.ndim, -1)
        mesh.set_nodes(nodes)
        mesh.add_geometry(geometry)
        mesh.new_region("*")
        node_ind = np.arange(geometry.n_nodes).reshape(geometry.shape)
        mesh.add_cell(node_ind, 0, 0)
        # Set up the basis functions for the finite elements
        basis1d = sem.basis_functions.LagrangeAtGaussLobatto(8)
        fe_basis = sem.basis_functions.TensorProductSupported(basis1d, basis1d)
        # Set up degrees of freedom
        self.dof_mngr = DOFManager(mesh, basis=fe_basis)

    def test_finite_element(self):
        fe = next(self.dof_mngr.finite_elements(x_phys=True, Jacobian=True))


if __name__ == '__main__':
    unittest.main()
