#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import solve
from scipy.sparse.linalg import spsolve


def sp_schur_solve(global_system, local_systems):

    gbmat, gbrhs, gsoln, gbmask = global_system

    # start/end indicies for inserting data into sparse matrix storage
    data_start = 0
    data_end = 0

    for elem, (lmat, lrhs) in local_systems.items():

        # Split local matrix RHS vector into global and boundary parts.

        assert lrhs.size % elem.n_nodes == 0
        n_ldof = lrhs.size                  # number of local DOFs
        dpn = n_ldof // elem.n_nodes        # DOFs per node
        n_lbdof = elem.n_nodes_bnd * dpn    # local DOFs on element boundary
        assert n_lbdof < n_ldof             # Must have some interior nodes
        # Might not have any interior modes if elements are 1st order
        # Named slices for convenience
        bnd = slice(None, n_lbdof)  # boundary elements
        itr = slice(n_lbdof, None)  # interior elements

        # Reorder local degrees of freedom to separate DOFs associated with
        # element boundaries from DOFs on element interiors.
        ixmap = elem.ixmap
        ldof_map = np.zeros(n_ldof, dtype=np.int32)
        ldof_map[0::2] = 2*ixmap.ravel()
        ldof_map[1::2] = 2*ixmap.ravel() + 1
        lmat[np.ix_(ldof_map, ldof_map)] = lmat.copy()
        lrhs[ldof_map] = lrhs.copy()

        # Compute the local Schur complement system
        sc_tmp = solve(lmat[itr, itr].T, lmat[bnd, itr].T,
                       check_finite=False).T
        # Note we allow for non-finite values in the above, as they might occur
        # for nodes on, e.g., axisymmetry axes.
        loc_sc_mat = lmat[bnd, bnd] - sc_tmp.dot(lmat[itr, bnd])
        loc_sc_rhs = lrhs[bnd] - sc_tmp.dot(lrhs[itr])

        # Assemble the global Schur-complement boundary system

        # indicies of global degrees of freedom
        gdof_ix = np.zeros(n_ldof, dtype=np.uint32)
        for i in range(dpn):
            gdof_ix[i::dpn] = dpn*elem.node_ix + i
        # gdof_ix[0::2] = 2*elem.node_ix,
        # gdof_ix[1::2] = gdof_ix[0::2] + 1 ...

        # row/column numbers
        row_nums, col_nums = np.meshgrid(
            gdof_ix[bnd], gdof_ix[bnd], indexing='ij')
        # Insert to row/col/data vectors of sparse matrix
        data_end += n_lbdof**2
        gbmat.row[data_start:data_end] = row_nums.ravel()
        gbmat.col[data_start:data_end] = col_nums.ravel()
        gbmat.data[data_start:data_end] = loc_sc_mat.ravel()

        # RHS vector
        gbrhs[gdof_ix[bnd]] += loc_sc_rhs
        data_start = data_end

    # Solve the global system on the element boundaries
    gbmat1 = gbmat.tocsr()[gbmask]  # Delete rows on essential BCs
    gbsoln = gsoln[:gbrhs.size]
    gbrhs1 = gbrhs[gbmask] - gbmat1[:, ~gbmask].dot(gbsoln[~gbmask])
    gbmat1 = gbmat1[:, gbmask]      # Delete columns on essential BCs
    gbsoln[gbmask] = spsolve(gbmat1, gbrhs1)

    # Compute change in solution vector on interior systems of each
    # element.

    for elem, (lmat, lrhs) in local_systems.items():

        # Split local matrix RHS vector into global and boundary parts.

        n_ldof = lrhs.size                  # number of local DOFs
        dpn = n_ldof // elem.n_nodes        # DOFs per node
        n_lbdof = elem.n_nodes_bnd * dpn    # local DOFs on element boundary
        # Might not have any interior modes if elements are 1st order
        # Named slices for convenience
        bnd = slice(None, n_lbdof)  # boundary elements
        itr = slice(n_lbdof, None)  # interior elements

        # get indicies of global DOF
        gdof_ix = np.zeros(n_ldof, dtype=np.uint32)
        for i in range(dpn):
            gdof_ix[i::dpn] = dpn*elem.node_ix + i

        # compute solution on element interior from the solution on the
        # element boundary
        gsoln[gdof_ix[itr]] = solve(
            lmat[itr, itr],
            lrhs[itr] - lmat[itr, bnd].dot(gsoln[gdof_ix[bnd]])
        )


def det_inv_2x2(mat):
    """Compute the determinant and inverse of a 2x2 matrix (or matricies)."""
    # Compute the determinant and inverse using the closed-form expression.
    det = mat[0, 0]*mat[1, 1] - mat[0, 1]*mat[1, 0]
    inv = np.empty_like(mat)
    inv[0, 0] = mat[1, 1]
    inv[0, 1] = -mat[0, 1]
    inv[1, 0] = -mat[1, 0]
    inv[1, 1] = mat[0, 0]
    inv *= 1/det
    return det, inv
