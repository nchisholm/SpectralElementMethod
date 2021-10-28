#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import itertools as itt
import numpy as np
from scipy import sparse, linalg
import sem.grid_importers
import sem.discrete
import sem.basis_functions as basis_funcs
from sem.sp_array import KroneckerArray
import h5py


def squirmer_vslip_profile(beta):
    def vslip(sin_th, cos_th):
        return 3./2 * sin_th * (1. + beta*cos_th)
    return vslip


def zero_slip_vel(sin_th, cos_th):
    return np.zeros_like(sin_th)


def sfn_potential(rho, z):
    """
    Compute stream function (in cylindrical coordinates) for irrotational
    potential flow past a sphere.  Unit radii and velocity are assumed (i.e.
    the stream function returned is dimensionless).  Flow is from the +z
    direction.
    """
    r = np.sqrt(rho**2 + z**2)  # radial distance from origin
    sin_th = rho / r            # angle measured from z-axis
    return -(r**2 - 1/r)/2. * sin_th**2


def sfn_free_stream(rho, z):
    r = np.sqrt(rho**2 + z**2)  # radial distance from origin
    sin_th = rho / r            # angle measured from z-axis
    return 0.5 * (r * sin_th)**2


def grid_in(mesh_file):
    """Load a Gmsh mesh from a ".msh" file.

    Parameters
    ----------
    mesh_file : str
        Location of the mesh to be loaded.  This should be *.msh file generated
        by Gmsh.
    """
    # load mesh from file
    return sem.grid_importers.load_msh(mesh_file, ndim=2)


class SolverFailure(Exception):
    """Indicates something went wrong in the numerical solution procedure.
    """
    pass


class SphereWithSlipVel(object):
    """
    Problem for a spherical body with a fixed surface velocity in
    axisymmetric flow.

    This simply serves as a base class of common methods for the `FixedSphere`
    and `Squirmer` classes.
    """

    def __init__(self, mesh):
        """
        Initializes the problem and all necessary global variables

        Parameters
        ----------
        mesh : Mesh
            Mesh to use for computations.
        """
        # Dictionary to store physical parameters
        self.phys_params = dict()

        # Number of discrete degrees of freedom per node on the computational
        # mesh (0: stream function, 1: vorticity)
        dpn = 2

        # Select shape functions and create an object that handles the degrees
        # of freedom on the computational mesh
        basis1d = basis_funcs.LagrangeAtGaussLobatto(8)
        basis2d = basis_funcs.TensorProductSupported(basis1d, basis1d)
        self.dof_mngr = sem.discrete.DOFManagerSC(mesh, dpn, basis2d)

        # initialize the global solution vector
        self.soln_vec = np.zeros(self.dof_mngr.ndof)
        self.sfn = self.soln_vec[0::dpn]   # stream function
        self.vort = self.soln_vec[1::dpn]  # vorticity

        # track which degrees of freedom are unknown for applying essential
        # (Direchlet) boundadry conditions (BCs).
        self.dof_mask = np.ones(self.dof_mngr.ndof_exterior, dtype=bool)
        self.dof_mask_sfn = self.dof_mask[0::dpn]
        self.dof_mask_vort = self.dof_mask[1::dpn]

        # store contour integral contributions from natural (Neumann) BCs
        self.cint = np.zeros(self.dof_mngr.ndof_exterior, dtype=np.float64)
        self.wte_cint = self.cint[0::dpn]
        self.wdef_cint = self.cint[1::dpn]
        # "wte" = vorticity transport equation
        # "wdef" = vorticity definition (in terms of stream function)

    def set_initial_guess(self):
        """Set an initial guess for the flow field (i.e. the stream function
        and vorticity fields).  This guess is that of potential (irrotational)
        flow past a sphere."""
        self.vort[:] = 0  # potential flow is irrotational
        for fe in self.dof_mngr.finite_elements(x_phys=True):
            # stream function -> potential flow around sphere
            rho, z = fe.x_phys
            loc = fe.node_ind
            self.sfn[loc] = sfn_potential(rho, z)

    def _apply_bcs_to_fe(self, fe, speed, slip_vel):
        # Apply BCs to any boundary elements on the current finite element
        for bnd_fe in fe.boundary_elements("sphere"):
            loc = bnd_fe.node_ind
            # Stream function = 0 on sphere boundary
            self.sfn[loc] = 0
            self.dof_mask_sfn[loc] = False
            # NBC on stream function to enforce the slip velocity
            rho, z = bnd_fe.x_phys
            # compute the normal vector
            n_rho, n_z = bnd_fe.normal()  # non-normalized unit vector
            r = np.sqrt(rho**2 + z**2)
            sin_th = rho / r
            cos_th = z / r
            v_slip_th = slip_vel(sin_th, cos_th)
            v_rho = v_slip_th * cos_th
            v_z = -v_slip_th * sin_th
            # TODO: double check the math on this
            n_grad_sfn = rho * (n_rho * v_z - n_z * v_rho)
            self.wdef_cint[loc] += -bnd_fe.quadrature.xweight(rho * n_grad_sfn)
            # Note minus sign to account for orientation of the normal
            # vector.
        for bnd_fe in fe.boundary_elements("symaxis"):
            loc = bnd_fe.node_ind
            # Zero stream line on symmetry axis
            self.sfn[loc] = 0
            self.dof_mask_sfn[loc] = False
            # Vorticity is also zero
            self.vort[loc] = 0
            self.dof_mask_vort[loc] = False
        for bnd_fe in fe.boundary_elements("shell"):
            loc = bnd_fe.node_ind
            # Free-stream far from the sphere
            rho, z = bnd_fe.x_phys
            self.sfn[loc] = -sfn_free_stream(rho, z) * speed
            self.dof_mask_sfn[loc] = False
            self.vort[loc] = 0
            self.dof_mask_vort[loc] = False

    def pre_assembly(self, speed, slip_vel, n_rey):

        # Store physical parameters
        self.phys_params['speed'] = speed
        self.phys_params['slip_profile'] = slip_vel
        self.phys_params['N_Re'] = n_rey

        # Re-initialize contour integral contributions
        self.wte_cint[:] = 0.
        self.wdef_cint[:] = 0.

        # Store operators:
        self.operators = []

        for fe in self.dof_mngr.finite_elements(x_phys=True, Jacobian=True):

            # --------------------------
            # Handle boundary conditions
            # --------------------------
            self._apply_bcs_to_fe(fe, speed, slip_vel)

            # -----------------------
            # Compute local operators
            # -----------------------

            x = fe.x_phys
            JxW = fe.detJxW
            invJ = fe.invJ
            dmats = fe.basis.get_diff_matrices()

            # "contribution" from first computational coordinate (xi0)
            gradh_xi0 = np.einsum('imn,mp->imnp', invJ[0, :], dmats[0])
            # "contribution" from second computational coordinate (xi1)
            gradh_xi1 = np.einsum('imn,nq->imnq', invJ[1, :], dmats[1])

            # "E^2" and vector laplacian operators
            E2e = np.zeros(fe.basis.coeff_shape*2)
            rho_JxW = x[0] * JxW
            # Make some indexing variables
            p, q, r = np.ogrid[
                [slice(N) for N in (fe.basis.coeff_shape[0],)*3]]
            E2e[p, q, r, q] += np.einsum(
                'mn,imnp,imnr->pnr', rho_JxW, gradh_xi0, gradh_xi0)
            E2e += np.einsum(
                'mn,imnp,imns->pnms', rho_JxW, gradh_xi0, gradh_xi1)
            # TODO: next entry is just the last entry transposed ...
            # don't compute it twice
            E2e += np.einsum(
                'mn,imnq,imnr->mqrn', rho_JxW, gradh_xi1, gradh_xi0)
            E2e[p, q, p, r] += np.einsum(
                'mn,imnq,imns->mqs', rho_JxW, gradh_xi1, gradh_xi1)

            p, q = np.ogrid[[slice(N) for N in fe.basis.coeff_shape]]
            Lve = E2e.copy()  # vector Laplacian
            Lve[p, q, p, q] += JxW/x[0]
            # Note: some division by zero warnings may be emitted.  They
            # can be safely ignored as they should only occur on the axis
            # of symmetry and their contributions will be eliminated by the
            # boundary conditions.

            p, q, r = np.ogrid[
                [slice(N) for N in (fe.basis.coeff_shape +
                                    fe.basis.coeff_shape[:1])]]
            E2e[p, q, r, q] += 2*np.einsum('mn,mnr->mnr', JxW, gradh_xi0[0])
            E2e[p, q, p, r] += 2*np.einsum('mn,mns->mns', JxW, gradh_xi1[0])

            # Advection operator (this one is sparse)
            Ae = KroneckerArray(shape=fe.basis.coeff_shape*3)
            Ae.add_diag(
                n_rey * (np.einsum('mn,mnr,mnu->mnru',
                                   JxW, gradh_xi0[0], gradh_xi1[1]) -
                         np.einsum('mn,mnr,mnu->mnru',
                                   JxW, gradh_xi0[1], gradh_xi1[0])),
                [0, 1, 2, 1, 0, 3])
            Ae.add_diag(
                n_rey * (np.einsum('mn,mns,mnt->mnst',
                                   JxW, gradh_xi1[0], gradh_xi0[1]) -
                         np.einsum('mn,mns,mnt->mnst',
                                   JxW, gradh_xi1[1], gradh_xi0[0])),
                [0, 1, 0, 2, 3, 1])
            Ae.add_diag(
                n_rey * np.einsum('mn,mnr->mnr', JxW/x[0], gradh_xi0[1]),
                [0, 1, 2, 1, 0, 1]
            )
            Ae.add_diag(
                n_rey * np.einsum('mn,mns->mns', JxW/x[0], gradh_xi1[1]),
                [0, 1, 0, 2, 0, 1]
            )

            # Mass operator
            Me = KroneckerArray(shape=fe.basis.coeff_shape*2)
            Me.add_diag(rho_JxW * x[0], [0, 1, 0, 1])

            # Save the computed operators for each finite element:
            self.operators.append((E2e, Lve, Ae, Me))

    def compute_local_system(self, fe, ops):
        """
        Compute the local (elemental) residual vector and its Jacobian matrix.
        """

        # unpack operators
        E2e, Lve, Ae, Me = ops

        # get local stream function and vorticity values
        loc = fe.node_ind
        sfn = self.sfn[loc]
        vort = self.vort[loc]

        # initialize residual vector and Jacobian matrix
        res_l = np.zeros(fe.ndof)
        jac_l = np.zeros((fe.ndof, fe.ndof))

        # compontents associated with the vorticity transport equation
        Ae_w_term = Ae.dot_dense(vort, [4, 5])
        jac_l[0::2, 0::2] = (
            Ae_w_term.to_array().reshape(fe.n_nodes, fe.n_nodes)
        )
        jac_l[0::2, 1::2] = (
            Ae.dot_dense(sfn, [2, 3]).to_array() + Lve
        ).reshape(fe.n_nodes, fe.n_nodes)
        res_l[0::2] = (
            Ae_w_term.dot_dense(sfn, [2, 3]).to_array() +
            np.einsum('pqrs,rs', Lve, vort)
        ).reshape(fe.n_nodes)

        # components associated with the definition of vorticity
        jac_l[1::2, 0::2] = E2e.reshape(fe.n_nodes, fe.n_nodes)
        jac_l[1::2, 1::2] = -Me.to_array().reshape(fe.n_nodes, fe.n_nodes)
        res_l[1::2] = (
            np.einsum('pqrs,rs', E2e, sfn) -
            Me.dot_dense(vort, [2, 3]).to_array()
        ).reshape(fe.n_nodes)

        return (jac_l, -res_l)

    def reorder_loc_sys_ext_itr(self, fe, local_system):
        # Reorder lexicographic DOFs on the local finite element system to a
        # hierarchical order where exterior DOFs follow interior DOFs.
        lmat, lrhs = local_system
        hier_dof_ord = fe.loc_dof_ind_hier
        lmat_h = lmat[np.ix_(hier_dof_ord, hier_dof_ord)]
        lrhs_h = lrhs[hier_dof_ord]
        return lmat_h, lrhs_h

    def compute_loc_sc_sys(self, fe, local_system_h):
        # Local system must be have hierarchically ordered DOFs.
        # Computes the local Schur complement matrix and corresponding rhs
        # vector.

        lmat_h, lrhs_h = local_system_h

        # Slices taking exterior/interior DOF components of the local system
        ext = slice(None, fe.ndof_exterior)
        itr = slice(fe.ndof_exterior, None)

        # Compute the local Schur complement system
        sc_tmp = linalg.solve(
            lmat_h[itr, itr].T, lmat_h[ext, itr].T, check_finite=False).T
        # Note we allow for non-finite values in the above, as they might occur
        # for nodes on, e.g., axisymmetry axes.
        loc_sc_mat = lmat_h[ext, ext] - sc_tmp.dot(lmat_h[itr, ext])
        loc_sc_rhs = lrhs_h[ext] - sc_tmp.dot(lrhs_h[itr])

        # Interior part of the Schur complement must be finite
        assert np.isfinite(loc_sc_mat[itr, itr]).all()
        assert np.isfinite(loc_sc_rhs[itr]).all()

        return loc_sc_mat, loc_sc_rhs

    def assemble_global_sc_sys(self, global_sc_system, local_systems_h):
        """Assemble the global Schur complement system from the local systems
        on each finite element.
        """
        gmat, grhs = global_sc_system

        # Reset RHS vector to only the NBC contributions
        grhs[:] = self.cint

        # start/end indicies for inserting data into sparse matrix storage
        gind0 = 0
        gind1 = 0

        iter_fes = self.dof_mngr.finite_elements()
        for fe, loc_sys in itt.izip(iter_fes, local_systems_h):
            # compute the local Schur complement system
            loc_sc_mat, loc_sc_rhs = self.compute_loc_sc_sys(fe, loc_sys)
            # row/column numbers
            ndof_ext = fe.ndof_exterior
            inds_ext = fe.global_dof_ind_hier[:ndof_ext]
            row, col = np.meshgrid(inds_ext, inds_ext, indexing='ij')
            # assemble global matrix
            gind1 += ndof_ext**2
            gmat.row[gind0:gind1] = row.ravel()
            gmat.col[gind0:gind1] = col.ravel()
            gmat.data[gind0:gind1] = loc_sc_mat.ravel()
            # assemble RHS vector
            grhs[inds_ext] += loc_sc_rhs
            gind0 = gind1

    def _solve_boundary_unks(self, global_sc_system, unk_vec):
        # Solve global Schur complement system for boundary DOFs
        sc_mat, sc_rhs = global_sc_system
        isunk = self.dof_mask
        unkv_ext = unk_vec[:self.dof_mngr.ndof_exterior]
        sc_mat1 = sc_mat.tocsr()[isunk]  # Delete rows on essential BCs
        sc_rhs1 = sc_rhs[isunk] - sc_mat1[:, ~isunk].dot(unkv_ext[~isunk])
        sc_mat1 = sc_mat1[:, isunk]      # Delete columns on essential BCs
        unkv_ext[isunk] = sparse.linalg.spsolve(sc_mat1, sc_rhs1)

    def _solve_interior_unks(self, local_systems_h, unk_vec):
        iter_fes = self.dof_mngr.finite_elements()
        for fe, (lmat, lrhs) in itt.izip(iter_fes, local_systems_h):
            # exterior/interior DOF slices of the local system
            ext = slice(None, fe.ndof_exterior)
            itr = slice(fe.ndof_exterior, None)
            inds = fe.global_dof_ind_hier
            # Ensure the interior system doesn't contain NaNs
            assert np.isfinite(lmat[itr, itr]).any()
            assert np.isfinite(lrhs[itr]).any()
            # compute DOFs on element interiors
            unk_vec[inds[itr]] = linalg.solve(
                lmat[itr, itr],
                lrhs[itr] - lmat[itr, ext].dot(unk_vec[inds[ext]])
            )

    def solve(self, it_max=10, tol=1e-6, max_n_diverge=3):
        """Solve the squirmer problem via Newton-Raphson iteration,
        applied to the Schur complement of the Jacobian matrix of
        the discretized NSE.

        Parameters
        ----------
        it_max : int
            max number of Newton-Raphson iterations
        tol : int
            Desired tolerance for the flow field in order for the solution to
            be considered converged.
        divergences_allowed : int
            number of Newton iterations that the solution is allowed to diverge
            before a `SolverFailure` exception is raised.
        """

        n_diverge = 0  # Number of iterations where the solution diverged
        du_norm_last = np.inf

        # Initialize global matrix system for DOFs on element exteriors.
        # We will use a Schur complement method so the interior DOFs will be
        # separately solved for explicitly.
        sc_mat, sc_rhs = self.dof_mngr.init_global_linear_system()
        dsoln = np.zeros_like(self.soln_vec)

        # do Newton iteration
        for itn in xrange(it_max):

            local_systems_h = []
            iter_fes = self.dof_mngr.finite_elements()
            for fe, ops in itt.izip(iter_fes, self.operators):
                loc_sys = self.compute_local_system(fe, ops)
                # must reorder DOFs for Schur complement method
                loc_sys_h = self.reorder_loc_sys_ext_itr(fe, loc_sys)
                local_systems_h.append(loc_sys_h)
            self.assemble_global_sc_sys((sc_mat, sc_rhs), local_systems_h)
            self._solve_boundary_unks((sc_mat, sc_rhs), dsoln)
            self._solve_interior_unks(local_systems_h, dsoln)

            # Update the solution vector
            self.soln_vec += dsoln

            # Test convergence
            du_norm = linalg.norm(dsoln[1::2])
            if du_norm > du_norm_last:
                # Solution appears to be diverging!
                n_diverge += 1
                if n_diverge >= max_n_diverge:
                    # FAIL: solution diverged too many times.
                    raise SolverFailure(
                        "Solution diverged {} times"
                        "(||du|| = {})".format(n_diverge, du_norm))
            # else:
            #    n_diverge = 0  # Reset the counter
            if np.allclose(du_norm, 0., atol=tol):
                # Successful convergence of solution!
                print(" => Calculation converged in {} Newton iterations\n"
                      "    ||du|| = {}".format(itn, du_norm))
                return
            du_norm_last = du_norm
            print("[Iteration {}]: ||du|| = {}".format(itn, du_norm))
        # FAIL: couldn't meet desired tolerance in specified maximum number
        # of iterations.
        raise SolverFailure(
            "Calculation failed to reach specified tolerance "
            "after {} Newton iterations.\n => Diff = {}"
            .format(it_max, du_norm)
        )

    def calc_force(self):
        """Compute the total hydrodynamic force on the sphere.

        Note: This code is specific for computing the force on a
        spherical surface of radius 1, and will not work for other
        shapes."""

        total_force = 0.

        # Loop through elements on the squirmer surface and compute the
        # hydrodynamic stresses on each one
        for elem_S in self.mesh.elems_in_region("sphere"):
            # get the "bulk" element adjacent to the surface element.
            _S, elem_V = elem_S.adj_map['*']
            # get the element mapping
            x_cyl = elem_V.get_mapping()
            jac = x_cyl.jacobian()
            detJ, invJ = det_inv_2x2(jac)

            # coordinates in cylindrical and polar form
            x_cyl_S = elem_S.get_mapping()
            # let *_S denote quantities defined at the element surface only
            # theta = np.arctan2(x_cyl_S[0], x_cyl_S[1])  # polar angle
            sin_th = x_cyl_S[0]  # here, r = 1
            sin2_th = sin_th**2
            cos_th = x_cyl_S[1]

            # surface slip velocity
            slip_profile = self.phys_params["slip_profile"]
            vslip = slip_profile(sin_th, cos_th)

            # solution for vorticity field
            vort_gl = self.soln_vec[1::2]
            vort = elem_V.get_coeffs(vort_gl)

            invJ_S = invJ.get_boundary(_S)
            # compute d{vorticity}/d(xi, eta, ...)
            dw_du_S = vort.jacobian().get_boundary(_S)
            # d(rho, z)/d(xi, eta, ...)
            drhoz_dr_S = x_cyl.get_boundary(_S)
            # d{vorticity}/dr at squirmer surface
            dw_dr_S = np.einsum('im,ijm,jm->m',
                                dw_du_S, invJ_S, drhoz_dr_S)

            # compute stresses
            vort_S = vort.get_boundary(_S)
            n_rey = self.phys_params["N_Re"]
            bernouli_stress = np.pi * n_rey * vslip**2 * sin_th * cos_th
            w_asym_stress = np.pi * (dw_dr_S + vort_S) * sin2_th
            pressure_stress = bernouli_stress + w_asym_stress
            viscous_stress = -2*np.pi * vort_S * sin2_th
            total_stress = pressure_stress + viscous_stress

            # differential arc length
            t_vec = x_cyl_S.jacobian()  # tangent vector
            d_arc = np.sqrt(t_vec[0]**2 + t_vec[1]**2)
            # compute integrands
            total_force += bs.CoeffArray.integrate(total_stress * d_arc)

        return total_force


class FixedSphere(SphereWithSlipVel):
    """Describes the problem for flow past a fixed sphere in uniform flow."""

    def pre_assembly(self, n_rey):
        SphereWithSlipVel.pre_assembly(self, 1., zero_slip_vel, n_rey)

    def run(self, n_rey, **flow_solver_opts):
        """Compute the flow past a fixed sphere.

        Parameters
        ----------
        n_rey : float
            The Reynolds number of the flow.
        **flow_solver_opts
            Keyword arguments passed to the `solve()` method.  See the
            documentation for `SphereWithSlipVel.solve` for more detail.
        """
        self.set_initial_guess()
        self.pre_assembly(n_rey)
        self.solve(**flow_solver_opts)


class Squirmer(SphereWithSlipVel):

    def pre_assembly(self, n_rey, speed=None, beta=None):
        r"""Sets the boundary condition vectors.

        Parameters
        ----------
        speed : float
            The translational speed of the squirmer (or equivalently the
            far-field uniform speed of the oncoming flow).
        beta : float
            Parameter controlling the slip velocity.

        Notes
        -----
        The prescribed slip velocity for the at the squirmer surface takes the
        form

        .. math::
            v_s = v_\theta|_{r=1} = \frac{3}{2} \sin\theta (1 + \beta
            \cos\theta),

        where :math:`\theta` is the polar angle.
        """
        if beta is None:
            slip_profile = self.phys_params['slip_profile']
        else:
            slip_profile = squirmer_vslip_profile(beta)
            self.phys_params['beta'] = beta
        if speed is None:
            speed = self.phys_params['speed']
        SphereWithSlipVel.pre_assembly(self, speed, slip_profile, n_rey)

    def run(self, n_rey, beta=None, speed=None, **flow_solver_opts):
        """Compute the flow field around a squirmer.

        Parameters
        ----------
        n_rey : float
            The Reynolds number
        beta : float
            Parameter determining the swimming "stroke".
        speed : float
            The translational speed of the squirmer (default = 1).
        **flow_solver_opts
            Keyword arguments passed to the `solve()` method.  See the
            documentation for `SphereWithSlipVel.solve` for more detail.
        """
        self.set_boundary_conditions(speed, beta)
        self.compute_operators(n_rey)
        self.solve(**flow_solver_opts)

    def save_data(self, f):
        """Save result into a specified h5py file.

        Parameters
        ----------
        f : h5py.File
            H5PY file.
        """
        # store the solution vector
        label = "Re={:.2e},beta={:.2e}".format(self.phys_params["N_Re"],
                                               self.phys_params["beta"])
        dset = f.create_dataset(label, data=self.soln_vec)
        # store parameters of the computation that we'll need to know later
        dset.attrs["speed"] = self.phys_params["speed"]
        dset.attrs["N_Re"] = self.phys_params["N_Re"]
        dset.attrs["beta"] = self.phys_params["beta"]

    def load_data(self, dset):
        self.soln_vec[:] = dset[:]
        self.phys_params.update(dset.attrs)

    def guess_from(self, other):
        """Get initial guess from another squirmer instace"""
        for fe, elem_o in itt.izip(self.mesh._elems, other.mesh._elems):
            if fe.ndim != fe.mesh.ndim:
                continue
            sfn_cf_o = elem_o.get_coeffs(other.soln_vec[0::2])
            vort_cf_o = elem_o.get_coeffs(other.soln_vec[1::2])
            qpts = fe.basis.quadrature.get_abscissa()
            gix = fe.get_node_ix()
            self.soln_vec[0::2][gix] = sfn_cf_o.eval(qpts)
            self.soln_vec[1::2][gix] = vort_cf_o.eval(qpts)
            self.phys_params.update(other.phys_params)

    def calc_speed(self, speed_guess, n_rey=None, beta=None,
                   flow_solver_opts=None, speed_solver_opts=None):
        """Compute the swimming speed of a squirmer.

        Parameters
        ----------
        speed_guess : float or list thereof
            Initial guess(es) for the speed.  Two must be supplied.  If only
            one is supplied, the other guess is taken from `phys_params`.  The
            initial guesses must be distinct.
        speed0 : float
            First initial guess at the swimming velocity.
        speed1 : float
            Another initial guess at the swimming velocity.  This must be a
            different value than `speed0`.
        n_rey : float
            The Reynolds number
        beta : float
            Parameter determining the swimming "stroke".
        **flow_solver_opts
            Keyword arguments passed to the `solve()` method.  See the
            documentation for `SphereWithSlipVel.solve` for more detail.

        Notes
        -----
        The secant method is used to compute the swimming speed at which the
        component of the force in the swimming direction vanishes.

        Two different initial guesses must be supplied for the speed.  If the
        supplied guesses are too close or too far apart, convergence may be
        slow or might not be reached.  This method does not guarantee
        convergence, but usually works given reasonable guesses.

        Example
        -------
        Here, we compute the swimming speed of a squirmer at Re=1 and beta=1 on
        a mesh defined in the file "meshes/donut.msh".

        >>> mesh = grid_in("meshes/donut.msh")
        >>> sqrm = Squirmer(mesh)
        >>> swim_speed = sqrm.calc_speed([0.99, 1.01], n_rey=1, beta=1)
        >>> print swim_speed
        0.92571156681483957
        """

        # if possible, pull parameters from `phys_params`
        if beta is None:
            beta = self.phys_params['beta']
        try:
            if len(speed_guess) == 2:
                speed0, speed1 = speed_guess
            elif len(speed_guess) == 1:
                speed0 = self.phys_params['speed']
                speed1 = float(speed_guess[0])
        except TypeError:
            speed0 = self.phys_params['speed']
            speed1 = float(speed_guess)

        # Default solver parameters
        if flow_solver_opts is None:
            flow_solver_opts = dict()
        flow_solver_opts.setdefault("it_max", 10)
        flow_solver_opts.setdefault("tol", 1e-6)
        if speed_solver_opts is None:
            speed_solver_opts = dict()
        it_max = speed_solver_opts.setdefault("it_max", 10)
        tol = speed_solver_opts.setdefault("tol", 1e-5)

        # Initial guesses for speed can't be the same value
        if speed0 == speed1:
            raise ValueError("Two distinct guesses for the speed must be "
                             "supplied.")

        print("COMPUTING SPEED... [initializing from guesses]")

        # Solve for flow and compute the force at the initial guesses
        print("finding force at speed = {}".format(speed0))
        self.set_boundary_conditions(speed0, beta)
        if n_rey is not None:
            self.compute_operators(n_rey)
        elif 'n_rey' not in self.phys_params:
            raise ValueError("Initial Reynolds number must be supplied to"
                             "calculation.")
        self.solve(**flow_solver_opts)
        force0 = self.calc_force()
        print("finding force at speed = {}".format(speed1))
        self.set_boundary_conditions(speed1, beta)
        self.solve(**flow_solver_opts)
        force1 = self.calc_force()

        for itn in xrange(1, it_max+1):
            # Secant method to estimate speed when the force is zero
            speed2 = (speed1*force0 - speed0*force1)/(force0 - force1)
            print("speed0 = {}  speed1 = {}  =>  speed2 = {}\n"
                  "force0 = {}  force1 = {}"
                  .format(speed0, speed1, speed2, force0, force1))
            # Compute flow field at swimming speed estimate
            print("COMPUTING SPEED... [Iteration {}]:".format(itn))
            print("@ speed = {}".format(speed2))
            self.set_boundary_conditions(speed2, beta)
            self.solve(**flow_solver_opts)
            force2 = self.calc_force()
            # Check if we have the speed to a satisfactory tolerance
            if abs(speed2 - speed1) < tol:
                print(" => Swimming speed converged after {} iterations."
                      .format(itn))
                return speed2
            # If not, update and start the next iteration
            speed0 = speed1
            speed1 = speed2
            force0 = force1
            force1 = force2
        raise SolverFailure("Swimming speed could not be found within the"
                            " desired tolerance in the max number of"
                            " iterations ({}).".format(it_max))


def main(squirmer, n_rey_list, beta_list, speed_guess=[0.99, 1.01],
         filename=None, step_reduction_factor=0.5, min_step=0.,
         flow_solver_opts=None, speed_solver_opts=None):
    """Compute the swimming speed of a squirmer in axisymmetric flow across a
    range of Reynolds numbers (and beta values).

    Parameters
    ----------
    squirmer : Squirmer
        A squirmer instance to use for computations.
    n_rey_list : list of floats
        A list of Reynolds numbers (Re) to perform computation.  These will be
        visited in ascending order.
    beta_list : list of floats
        Values of beta to perform computations (at all Re given in
        `n_rey_list`).
    speed_guess : list of two floats
        Initial guesses for the squirmer speed (default is [0.99, 1.01]).
    filename : str
        Name of a .h5py file which to save results.
    step_reduction_factor : float (default=0.5)
        Amount to reduce the increment in Re should any individual computation
        at a specific Re and beta fail to converge.
    min_step : float (default=0.)
        The minimum increment in Re allowed before raising a `SolverFailure`.

    Notes:
    ------
    This routine implements a continuation strategy if a solution fails to
    converge by increasing Re in smaller increments.  This will often cause a
    previously ill-behaved computation to converge because better initial
    guesses for the flow field are "continually" procured at each step in Re.
    Of course, convergence is not guaranteed with this method, so it will raise
    an exception if the Re increment decreases below `min_step`.
    """

    if flow_solver_opts is None:
        flow_solver_opts = dict()
    if speed_solver_opts is None:
        speed_solver_opts = dict()

    if not 0. < step_reduction_factor < 1.:
        raise ValueError("reduction factor must be between 0 and 1")
    n_rey_list.sort()  # put Re values in ascending order

    # Set up file to save results
    if filename is None:
        results_file = None
    else:
        results_file = h5py.File(filename)

    for beta in beta_list:
        # Keep track of computed swimming speeds two iterations back.
        # Also, reset to initial guesses when the value of beta changes.

        speed = speed_guess[:]
        speed.append(0.)

        # Do initial calculation at first value of Re and save the result
        n_rey = n_rey_list[0]
        print("\n### beta = {:.2g}, Re = {:.2g} ###".format(beta, n_rey))

        label = "Re={:.2e},beta={:.2e}".format(n_rey, beta)
        if results_file and label in results_file:
            print("Data exists in file \"{}\" ... loading it"
                  .format(results_file.filename))
            squirmer.load_data(results_file[label])
            speed[2] = squirmer.phys_params["speed"]
        else:
            squirmer.set_initial_guess()
            speed[2] = squirmer.calc_speed(speed[:2], n_rey, beta,
                                           flow_solver_opts, speed_solver_opts)
            squirmer.save_data(results_file)
        last_converged_soln = squirmer.soln_vec.copy()

        speed[:2] = speed[1:]

        # initial increment in Re value
        delta = (n_rey_list[1] - n_rey_list[0])
        i = 1
        while True:
            # Do calculations at subsequent values of Re, using the last
            # computed flow field and swimming speed as initial guesses.
            n_rey += delta
            # Do we have a "requested" value for Re?
            if 0.99*n_rey_list[i] < n_rey:
                # account for possible rounding error
                n_rey = n_rey_list[i]
                do_increment = True
            else:
                do_increment = False
            lbl_str = "beta = {}, Re = {}".format(beta, n_rey)
            try:
                if do_increment is True:
                    print("\n### {} ###".format(lbl_str))
                    label = "Re={:.2e},beta={:.2e}".format(n_rey, beta)
                    if results_file and label in results_file:
                        print("Data exists in file \"{}\" ... loading it"
                              .format(results_file.filename))
                        squirmer.load_data(results_file[label])
                        speed[2] = squirmer.phys_params["speed"]
                    else:
                        speed[2] = squirmer.calc_speed(
                            speed[:2], n_rey, beta,
                            flow_solver_opts, speed_solver_opts)
                        if results_file is not None:
                            squirmer.save_data(results_file)
                    i += 1
                    if i >= len(n_rey_list):
                        break
                    # Reset the increment in Re
                    delta = n_rey_list[i] - n_rey_list[i-1]
                else:
                    # We're on an intermediate value of Re
                    print("\n### {} (continuing) ###".format(lbl_str))
                    speed[2] = squirmer.calc_speed(
                        speed[:2], n_rey, beta,
                        flow_solver_opts, speed_solver_opts)
                speed[:2] = speed[1:]
                last_converged_soln = squirmer.soln_vec.copy()
            except SolverFailure as exc:
                # Solution failed to converge ...
                print("NOTICE: Solver failed with message:\n"
                      "{}\nAttempting to continue...".format(exc.message))
                # Go back and try at an intermediate Re value
                n_rey -= delta
                delta *= step_reduction_factor
                squirmer.soln_vec[:] = last_converged_soln
                if delta < min_step:  # Step is too small to continue
                    raise SolverFailure(
                        "Continuation step reduced below minimum size.")
