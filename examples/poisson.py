#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
Solve Poisson's equation on a square.  The left and bottom bondaries have
Diriclet boundary conditions while the top and right boundaries have Neumann
boundary conditions.  The right hand side is equal to unity.

..math:: \nabla u = 1
with
..math:: u_bl = 0; \quad \nabla \vec{n} \cdot u_tr = 0

Example
-------
>>> # Load a mesh file
>>> poisp = PoissonPlate("meshes/square.msh")
>>> # Compute the solution
>>> poisp.run()
>>> # Plot the solution
>>> plot_solution(poisp)
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

import py_hp_fem
from py_hp_fem import grid_importers
from py_hp_fem import basis_sets as bs
from py_hp_fem.linalg import det_inv_2x2, sp_schur_solve
# from py_hp_fem.sp_array import KroneckerArray
import py_hp_fem.plot2d.contours as pltsoln


class SolverFailure(Exception):
    """Exception indicating failure to converge to a solution.
    """
    pass


def grid_in(mesh_file):
    """Load a Gmsh mesh from a ".msh" file.

    Parameters
    ----------
    mesh_file : str
        Location of the mesh to be loaded.  This should be *.msh file generated
        by Gmsh.
    """
    # load mesh from file
    print("Importing mesh...")
    mesh = grid_importers.Gmsh.load_mesh(mesh_file, ndim=2)
    print("Substructuring mesh...")
    mesh.substructure()
    print("Finding adjacent elements...")
    mesh.compute_adjacencies()
    return mesh


class PoissonPlate(object):
    """
    Poisson's equation on a 2D plate.
    """

    def __init__(self, mesh):
        """
        Parameters
        ----------
        phys_params : dict
            Dictionary of physical parameters used in the problem.
        solver_params : dict
            Dictionary of solver parameters
        """
        self.mesh = mesh
        self.phys_params = dict()
        self.setup()

    def run(self):
        self.set_boundary_conditions()
        self.compute_operators()
        self.solve()

    def setup(self):
        # Do some general initialization

        mesh = self.mesh

        self.pde_rank = 1
        # indicates the number of unknowns/equations globally in
        # the PDE system (not the algebraic one, which will presumably have
        # many more equations/unknowns...)

        # number of global DOFs
        self.n_gdof = mesh.n_nodes * self.pde_rank
        # number of global boundary DOFs
        self.n_gbdof = mesh.n_nodes_elem_exterior * self.pde_rank
        # number of globally scattered boundary DOFs
        self.n_mat_nnz = sum(
            (self.pde_rank*elem.n_nodes_bnd)**2 for elem in
            mesh.bulk_elements())

        # Store contribution from BCs
        self.sc_cint = np.zeros(self.n_gbdof, dtype=np.float64)

        # initialize global solution vectors (both of which are scalar
        # quantities)
        self.soln_vec = np.zeros(self.n_gdof)

        # Track which degrees of freedom are unknown
        self.gdof_mask = np.ones(self.n_gbdof, dtype=bool)
        # DOFs at nodes subject to essential boundary conditions
        # will automatically be known.

        # Select appropriate shape functions and automatically apply them to
        # all mesh elements
        mesh.set_shape_functions(bs.LagrangeGaussLobatto)

    def set_initial_guess(self):
        # Can set non-homogeneous initial guess here.
        pass

    def set_boundary_conditions(self):
        mesh = self.mesh

        # Loop through 1D boundary elements
        for elem in mesh.boundary_elements():
            # Get the global node indicies in orthographic order
            # (the basis has orthographically arranged DOFs)
            local = elem.get_node_ix(ortho=True)
            # Map real coordinates to computational coordinates
            x, y = elem.get_mapping()

            # elements with essential boundary condition (EBC)
            if elem in mesh.regions["ebc"]:
                # Homogeneous EBC
                self.soln_vec[local] = 0.2*((x+1) + (y+1))
                self.gdof_mask[local] = False
            elif elem in mesh.regions["nbc"]:
                # Don't need to do anything here if the NBC is homogeneous.
                pass

    def compute_operators(self):
        """
        Compute local operators for each element and store them for later.  All
        of these operators are *independent* of the solution vector.
        """
        self.operators = dict()     # store operators for each element

        mesh = self.mesh

        for elem in mesh.bulk_elements():

            # Mapping from computational -> physical coordinates
            x = elem.get_mapping()
            # Compute the Jacobian of the element
            J = x.jacobian()

            # Compute the Jacobian determinant and inverse using
            # closed-form expressions.
            detJ, invJ = det_inv_2x2(J)
            assert np.all(detJ > 0)

            # gradient of the shape functions
            # Component in first computational coordinate (xi0)
            gradh_xi0 = np.einsum(
                'mp,imn->imnp', elem.basis.diff_mat[0], invJ[0, ...])
            # Component in first computational coordinate (xi1)
            gradh_xi1 = np.einsum(
                'nq,imn->imnq', elem.basis.diff_mat[1], invJ[1, ...])

            # Jacobian determinant times the integration weights
            quad_wt = elem.basis.quadrature.weights
            JxW = np.einsum('m,n,mn->mn', quad_wt[0], quad_wt[1], detJ)

            # Compute operators for each element

            # Scalar Laplacian operator
            Lse = np.zeros(elem.shape*2)
            # Make some indexing variables
            p, q, r = np.ogrid[[slice(N) for N in (elem.shape[0],)*3]]
            Lse[p, q, r, q] += np.einsum(
                'mn,imnp,imnr->pnr', JxW, gradh_xi0, gradh_xi0)
            Lse += np.einsum(
                'mn,imnp,imns->pnms', JxW, gradh_xi0, gradh_xi1)
            # TODO: next entry is just the last entry transposed ...
            # don't compute it twice
            Lse += np.einsum(
                'mn,imnq,imnr->mqrn', JxW, gradh_xi1, gradh_xi0)
            Lse[p, q, p, r] += np.einsum(
                'mn,imnq,imns->mqs', JxW, gradh_xi1, gradh_xi1)

            # Mass operator
            # Me = KroneckerArray(shape=elem.shape*2)
            # Me.add_diag(JxW, [0, 1, 0, 1])

            # "forcing" term (RHS)
            fe = JxW

            # Save the computed operators per element:
            self.operators[elem] = (Lse, fe)

    def solve(self):
        # Compute solution using the Schur-complement matrix

        # Initialize global matrix
        row = np.zeros(self.n_mat_nnz, dtype=np.uint32)
        col = np.zeros(self.n_mat_nnz, dtype=np.uint32)
        entries = np.zeros(self.n_mat_nnz, dtype=np.float64)
        sc_mat = sparse.coo_matrix((entries, (row, col)),
                                   (self.n_gbdof, self.n_gbdof))
        # right-hand-side (RHS) vector
        sc_rhs = np.zeros(self.n_gbdof, dtype=np.float64)
        # solution vector

        # Contributions to RHS vector from boundary conditions
        sc_rhs[:] = self.sc_cint

        local_systems = dict()  # Storage for local system

        for elem, (Lse, fe) in self.operators.iteritems():

            # Number of local degrees of freedom per element
            n_ldof = elem.n_nodes * self.pde_rank
            # local = elem.get_node_ix(ortho=True)

            # Operators on element
            Lse, fe = self.operators[elem]

            # local algebraic system
            lmat = np.zeros((n_ldof, n_ldof))
            lrhs = np.zeros(n_ldof)
            lmat[0::1, 0::1] = Lse.reshape(elem.n_nodes, elem.n_nodes)
            lrhs[0::1] = fe.reshape(elem.n_nodes)

            # Apply boundary conditions locally insteady of globally
            # ldof_mask = self.gdof_mask[local]
            # u_loc = self.soln_vec[local]
            # if np.any(ldof_mask):
            #     unk = ldof_mask
            #     on_ebc = ~unk
            #     lmat_unk = lmat[unk, unk]       # Unknown part
            #     lmat_ebc = lmat[unk, on_ebc]    # Known part
            #     lrhs_unk = lrhs[unk]
            #     lrhs_ebc = lrhs[on_ebc]
            #     lrhs_unk -= lmat_ebc.dot(u_loc[on_ebc])

            # Rearrange degrees of freedom so boundary nodes are first
            ldof_map = np.zeros(n_ldof, dtype=np.int32)
            ldof_map[0::1] = elem.ixmap.ravel()
            lmat[np.ix_(ldof_map, ldof_map)] = lmat.copy()
            lrhs[ldof_map] = lrhs.copy()

            local_systems[elem] = (lmat, lrhs)

        sp_schur_solve((sc_mat, sc_rhs, self.soln_vec, self.gdof_mask),
                       local_systems)


def plot_solution(problem):
    # Plot the solution vector on a contour plot
    mesh = problem.mesh
    soln = problem.soln_vec
    pltsoln.surface(mesh, soln, cmap=plt.cm.cool)
