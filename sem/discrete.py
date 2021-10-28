#!/usr/bin/env python
# encoding: utf-8

"""
A module which handles elements and meshes, and their degrees of freedom.
"""


from collections import namedtuple
import itertools as itt
import numpy as np
from scipy import linalg
from scipy import sparse
from scipy.sparse import csgraph
import sem.mapping
from sem.mapping import _subface_slice


class OutsideDomain(Exception):
    """Exception to be raised if a given physical point is outside the domain
    of a finite element mesh.
    """
    pass


class Static_COO_Matrix(object):

    """Represents a sparse matrix in "COOrdinate" format.

    The only purpose of this class is to eventually export the contents into
    scipy's built in COOrdinate matrix class."""

    def __init__(self, data, row_col, shape):
        self.data = data
        self.row_col = row_col
        self.row = row_col[0]
        self.col = row_col[1]
        self.shape = shape

    def tocoo(self):
        return sparse.coo_matrix((self.data, self.row_col))


class DOFManager(object):
    """Manages the degrees of freedom on a finite element mesh.
    """

    _compute_flag_keys = {'x_phys', 'Jacobian'}
    default_compute_flags = dict.fromkeys(_compute_flag_keys, False)

    @property
    def ndof_per_node(self):
        """Number of degrees of freedom for each node on the mesh.

        Returns
        -------
        int
        """
        return self._dpn

    @property
    def ndof(self):
        """Total number of degrees of freedom.

        Returns
        -------
        int
        """
        return self._dpn * self._mesh.n_nodes

    @property
    def mesh(self):
        """The mesh object associated with a DOFManager instance.

        Returns
        -------
        Mesh
        """
        return self._mesh

    def __init__(self, mesh, dofs_per_node=1, basis=None, mapping_basis=None,
                 rcm_order=True):
        """Create a new DOFManager instance.

        Parameters
        ----------
        mesh : Mesh
            A mesh over which to assign degrees of freedom.
        dofs_per_node : int
            Number of degrees of freedom to be assigned to each node.
        basis : Basis, optional
            Basis to be used for finite elements.
        mapping_basis : basis, optional
            Basis to be used in mapping the physical coordinates of an element.
        rcm_order : bool, optional
            Whether to re-order the nodes (and hence degrees of freedom) of a
            mesh using the Reverse Cuthill-McKee (RCM) algorithm.

        See Also
        --------
        DOFManagerSC

        Notes
        -----
        Using RCM will tend to reduce the bandwidth of the resulting global
        finite element matrix equation, and hence speed up its solution if the
        nodes are not arranged in an otherwise optimal fashion.
        """
        self._mesh = mesh
        self._dpn = dofs_per_node
        self._mesh._compute_cell_centroids()
        # if basis is None:
        #     raise ValueError("Basis must be initialized")
        self._basis = basis
        if mapping_basis is None:
            self._map_basis = basis
        else:
            self._map_basis = mapping_basis
        # FIXME: creating multiple DOFManager objects with the same Mesh object
        # can apparently cause problems, most likely because the nodes of the
        # Mesh object are reordered differently every time a new DOFManager
        # object is initialized.
        if rcm_order:
            self._reorder_nodes_rcm()

    def _resolve_cmpflag_dependencies(self, compute_flags):
        """Resolve any dependencies between the `compute_flags`.
        (e.g. the `Jacobian` flag requires the `x_phys` flag)
        """

        # Check for unrecognized flag names
        bad_flags = set(compute_flags) - self._compute_flag_keys
        if bad_flags:
            raise ValueError('Unrecognized flags {}.'.format(bad_flags))
        # Set unspecified flags to default values
        for (flag, default_value) in self.default_compute_flags.items():
            compute_flags.setdefault(flag, default_value)
        # Work out dependent flags
        if compute_flags['Jacobian']:
            compute_flags['x_phys'] = True

    def _get_connectivity_graph(self):
        """Constructs and returns (in CSR format) the connectivity graph of
        each node on the mesh.
        """

        mesh = self._mesh
        # initialize graph of nodes as a sparse matrix
        n_entries = sum(cell.n_nodes**2 for cell in mesh.cells)
        row_col = np.zeros((2, n_entries), dtype=np.uint32)
        data = np.zeros(n_entries, dtype=np.bool)
        graph = sparse.coo_matrix((data, row_col), (mesh.n_nodes,)*2)

        # assemble the graph of the pattern of node connectivity within and
        # between cells
        ix0 = ix1 = 0
        for cell in mesh.cells:
            row, col = np.meshgrid(*(cell.node_ind_lexicographic,)*2,
                                   indexing='ij')
            ix1 += cell.n_nodes**2
            slc = slice(ix0, ix1)
            graph.row[slc] = row.ravel()
            graph.col[slc] = col.ravel()
            graph.data[slc] = True
            ix0 = ix1

        return graph.tocsr()

    def _reorder_nodes_rcm(self):
        """Reorder nodes using the Reverse Cuthill-McKee algorithm, which
        tends to reduce the bandwidth of the connectivity matrix/graph.
        """
        # compute the permutation of nodes given by the Reverse Cuthill McKee
        # algorithm and apply it to the mesh
        mesh = self._mesh
        graph = self._get_connectivity_graph()
        perm = csgraph.reverse_cuthill_mckee(graph, True)
        mesh._permute_nodes(perm)

    # TODO: methods for getting and setting the finite element basis and
    # mapping basis

    # TODO: Method for changing the flags that control what is computed (?)
    def get_finite_element(self, i, **compute_flags):
        self._resolve_cmpflag_dependencies(compute_flags)
        cell = self._mesh.get_cell(i)
        return FiniteElement(self, cell, compute_flags)

    def finite_elements(self, **compute_flags):
        """Iterator through all finite elements on the mesh, computing the
        requested values over each element.

        Parameters
        ----------
        x_phys : bool, optional
            Compute the mapping of the finite elements from parametric to real
            space.
        Jacobian : bool, optional
            Compute the finite element Jacobians (along with their determinants
            and inverses).

        Yields
        ------
        FiniteElement
        """
        self._resolve_cmpflag_dependencies(compute_flags)

        for cell in self._mesh.cells:
            yield FiniteElement(self, cell, compute_flags)

    def boundary_elements(self, name, **compute_flags):
        """Iterate through boundary elements on the mesh.
        """
        self._resolve_cmpflag_dependencies(compute_flags)
        # TODO: this is kind of indirect
        for cell in self.mesh.cells_on_boundary(name):
            fe_parent = FiniteElement(self, cell, compute_flags)
            for bnd_fe in fe_parent.boundary_elements(name):
                yield fe_parent, bnd_fe

    def interpolate(self, coeffs, x_phys):
        """Interpolate from a global set of coefficients.

        Parameters
        ----------
        coeffs : numpy.ndarray
            Global set of coefficients representing some data field.
        x_phys : numpy.ndarray
            Point at which to interpolate the data.
        """
        fe, x_param = self.find_elem_containing_point(x_phys)
        coeffs_local = coeffs[..., fe.node_ind]
        return fe.interpolate(coeffs_local, x_param)

    def values_at_nodes(self, coeffs):
        """Compute values of an interpolated function at equispaced nodes
        within the finite element given the values of the coefficients of the
        basis functions.

        Parameters
        ----------
        coeffs : ndarray
            array of the basis coefficients representing some function over the
            parametric space of the finite element.

        Returns
        -------
        ndarray
            Array of values of the function interpolated at equispaced points
            in parametric space.
        """
        values = np.empty_like(coeffs)
        for fe in self.finite_elements():
            loc = fe.node_ind
            # TODO: this will only work if all elements share the same basis
            values[..., loc] = self._basis.interpolate_on_grid_eq(
                coeffs[..., loc])
        return values

    def get_global_matrix_equation(self):
        raise NotImplementedError()

    def find_elem_containing_point(self, point):
        """Find the element containing the given point.  Once it is found,
        returns a finite element object and the parametric coordinates of the
        point within the finite element.
        """
        point = np.asarray(point, float)
        # get distance from point to centers of mesh cells
        dist = np.sqrt(np.sum((point - self._mesh._centroids)**2, axis=1))
        for i in np.argsort(dist):
            cell = self._mesh.get_cell(i)
            fe = FiniteElement(self, cell, dict(x_phys=True, Jacobian=True))
            try:
                x_param = fe.mapping.inv(point)
                return fe, x_param
            except sem.mapping.OutsideDomain:
                pass
        raise OutsideDomain("Point {} appears outside the domain of the mesh.".
                            format(point))


class DOFManagerSC(DOFManager):
    """Manages the degrees of freedom (DOFs) on a finite element mesh, and
    orders them in such a way that a Schur complement may be formed by
    'eliminating' nodes interior to finite elements. Thus, this is more
    suitable for use with meshes having high order finite elements.
    """

    @property
    def ndof_exterior(self):
        """Gives the number of DOFs that are on element exteriors or boundaries.
        """
        return self._mesh.n_nodes_cell_exterior * self._dpn

    @property
    def ndof_interior(self):
        """Gives the number of DOFs that are on element interiors
        """
        return self._mesh.n_nodes_cell_interior * self._dpn

    def __init__(self, mesh, dofs_per_node=1, basis=None, mapping_basis=None,
                 rcm_order=True):
        super(DOFManagerSC, self).__init__(mesh, dofs_per_node, basis,
                                           mapping_basis, rcm_order=False)
        # FIXME: creating multiple DOFManager objects with the same Mesh object
        # can apparently cause problems, most likely because the nodes of the
        # Mesh object are reordered differently every time a new DOFManager
        # object is initialized.
        self._do_static_condensation()
        if rcm_order:
            self._reorder_nodes_rcm()

    def _do_static_condensation(self):
        """Reorder mesh nodes so that those on cell boundaries are ordered
        first followed by those on cell interiors.
        """

        mesh = self._mesh

        # count the (scattered) exterior nodes and interior nodes on the mesh
        n_ext_nodes_scat = 0        # nodes on cell boundaries
        n_int_nodes = 0             # nodes within cell interiors
        for cell in mesh.cells:
            n_ext_nodes_scat += cell.n_exterior_nodes
            n_int_nodes += cell.n_interior_nodes

        # initialize maps of new node indices -> old node indices
        scat_ext_ix_map = np.full(n_ext_nodes_scat, -1, dtype=int)
        int_ix_map = np.full(n_int_nodes, -1, dtype=int)

        # start/end indices for exterior nodes
        ix0_ext = 0
        ix1_ext = 0
        # start/end indices for interior nodes
        ix0_int = 0
        ix1_int = 0
        # loop through each cell and re-compute the global node indices
        for cell in mesh.cells:
            ix1_ext += cell.n_exterior_nodes
            ix1_int += cell.n_interior_nodes
            scat_ext_ix_map[ix0_ext:ix1_ext] = cell.exterior_node_ind
            int_ix_map[ix0_int:ix1_int] = cell.interior_node_ind
            ix0_ext = ix1_ext
            ix0_int = ix1_int

        # remove duplicates from the scattered exterior node indices
        ext_ix_map = np.unique(scat_ext_ix_map)
        n_ext_nodes = ext_ix_map.size
        int_ix_map.sort()
        ix_map = np.concatenate((ext_ix_map, int_ix_map))
        assert ix_map.size == mesh.n_nodes
        assert n_ext_nodes + n_int_nodes == mesh.n_nodes

        mesh._permute_nodes(ix_map)

        mesh.n_nodes_cell_exterior = n_ext_nodes
        mesh.n_nodes_cell_interior = n_int_nodes
        mesh.condensed = True

    def _get_connectivity_graph(self):
        """Constructs and returns (in CSR format) the connectivity graph of
        each node on the mesh.  Only nodes on the boundaries of finite elements
        are considered.
        """
        mesh = self._mesh

        # initialize graph of nodes as a sparse matrix
        n_entries = sum(cell.n_exterior_nodes**2 for cell in mesh.cells)
        row_col = np.zeros((2, n_entries), dtype=np.uint32)
        data = np.zeros(n_entries, dtype=np.bool)
        graph = sparse.coo_matrix((data, row_col),
                                  (mesh.n_nodes_cell_exterior,)*2)

        # assemble the graph of the pattern of node connectivity within and
        # between cells
        ix0 = ix1 = 0
        for cell in mesh.cells:
            row, col = np.meshgrid(*(cell.exterior_node_ind,)*2, indexing='ij')
            ix1 += cell.n_exterior_nodes**2
            slc = slice(ix0, ix1)
            graph.row[slc] = row.ravel()
            graph.col[slc] = col.ravel()
            graph.data[slc] = True
            ix0 = ix1

        return graph.tocsr()

    def _reorder_nodes_rcm(self):
        """Reorder nodes using the Reverse Cuthill-McKee algorithm, thus
        reducing the bandwidth of the connectivity matrix/graph.
        """
        # compute the permutation of nodes given by the Reverse Cuthill McKee
        # algorithm and apply it to the mesh
        mesh = self._mesh
        graph = self._get_connectivity_graph()
        perm = np.empty(mesh.n_nodes, np.uint32)
        n_nodes = mesh.n_nodes
        n_ext_nodes = mesh.n_nodes_cell_exterior
        perm[:n_ext_nodes] = sparse.csgraph.reverse_cuthill_mckee(graph, True)
        perm[n_ext_nodes:] = np.arange(n_ext_nodes, n_nodes)
        mesh._permute_nodes(perm)

    def init_global_linear_system(self):
        """Initializes a global matrix equation.

        Returns
        -------
        matrix : coo_matrix
            NxN sparse matrix in COO format where N is the number of nodes on
            finite element boundaries. The row, column, and data vectors are
            also initialized to zero-vectors appropriate lengths.
        rhs_vec : ndarray
            The right-hand-side vector, which is a initialized as a
            zero-vector of length N.
        """
        # Construct the global system
        n_mat_entries = sum(fe.ndof_exterior**2 for fe in
                            self.finite_elements())
        row_col = np.zeros((2, n_mat_entries), dtype=np.uint32)
        entries = np.zeros(n_mat_entries, dtype=np.float64)
        # Global matrix for degrees of freedom on element exteriors
        ndof_ext = self.ndof_exterior
        matrix = Static_COO_Matrix(entries, row_col, (ndof_ext, ndof_ext))
        rhs_vec = np.zeros(ndof_ext, dtype=np.float64)
        return matrix, rhs_vec

    @staticmethod
    def reorder_local_system_hier(fe, local_system):
        """Reorder lexicographic DOFs on a local finite element system to a
        hierarchical order where exterior DOFs follow interior DOFs."""
        lmat, lrhs = local_system
        hier_dof_ord = fe.loc_dof_ind_hier
        lmat_h = lmat[np.ix_(hier_dof_ord, hier_dof_ord)]
        lrhs_h = lrhs[hier_dof_ord]
        return lmat_h, lrhs_h

    @staticmethod
    def compute_local_sc_system(fe, local_system):
        """Compute the Schur Complement matrix and corresponding RHS vector
        from an appropriately ordered matrix/RHS system for the local DOFs of a
        finite element.

        Parameters
        ----------
        fe : FiniteElement
            Finite element to which the local system belongs.
        local_system : tuple of NxN and N dimensional ndarray objects
            The local matrix (NxN array) and RHS vector (array of length N)
            that correspond to the finite element and where N is the number of
            DOFs of the element.
        is_hier : bool, optional
            Whether the DOFs in the local system are in hierarchical order. If
            `false`, the DOFs are assumed to be in lexicographic order and are
            reordered appropriately.
        """

        lmat_h, lrhs_h = local_system

        # Slices taking exterior/interior components of the local system
        ext = slice(None, fe.ndof_exterior)  # on element exterior (faces)
        itr = slice(fe.ndof_exterior, None)  # in element interior

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

    def assemble_global_sc_system(self, global_sc_system, local_systems):

        gmat, grhs = global_sc_system

        # start/end indices for COO sparse matrix entries (row, col, data)
        gind0 = 0
        gind1 = 0

        for fe, loc_sys in zip(self.finite_elements(), local_systems):
            # compute the local Schur complement system
            loc_sc_mat, loc_sc_rhs = self.compute_local_sc_system(fe, loc_sys)
            # row/column numbers
            ndof_ext = fe.ndof_exterior
            inds_ext = fe.global_dof_ind_hier[:ndof_ext]
            row, col = np.meshgrid(inds_ext, inds_ext, indexing='ij')
            # assemble global matrix
            gind1 += ndof_ext**2
            gmat.row[gind0:gind1] = row.ravel()
            gmat.col[gind0:gind1] = col.ravel()
            gmat.data[gind0:gind1] = loc_sc_mat.ravel()
            # assemble global RHS vector
            grhs[inds_ext] += loc_sc_rhs
            gind0 = gind1

    def _solve_boundary_dofs(self, global_sc_system, dof_vec, on_ebc):
        # Solve global Schur complement system for boundary DOFs
        sc_mat, sc_rhs = global_sc_system
        is_unk = ~on_ebc
        ext_dofs = dof_vec[:self.ndof_exterior]
        sc_mat1 = sc_mat.tocoo().tocsr()
        sc_mat1 = sc_mat1[is_unk]  # Delete rows on essential BCs
        sc_rhs1 = sc_rhs[is_unk] - sc_mat1[:, on_ebc].dot(ext_dofs[on_ebc])
        sc_mat1 = sc_mat1[:, is_unk]      # Delete columns on essential BCs
        ext_dofs[is_unk] = sparse.linalg.spsolve(sc_mat1, sc_rhs1)

    def _solve_interior_dofs(self, local_systems, dof_vec):
        for fe, loc_sys in zip(self.finite_elements(), local_systems):
            lmat, lrhs = loc_sys
            # exterior/interior DOF slices of the local element system
            ext = slice(None, fe.ndof_exterior)
            itr = slice(fe.ndof_exterior, None)
            # compute DOFs on element interiors
            inds = fe.global_dof_ind_hier
            dof_vec[inds[itr]] = linalg.solve(
                lmat[itr, itr],
                lrhs[itr] - lmat[itr, ext].dot(dof_vec[inds[ext]])
            )

    def solve(self, global_sc_system, local_systems, dof_vec, on_ebc):
        self._solve_boundary_dofs(global_sc_system, dof_vec, on_ebc)
        self._solve_interior_dofs(local_systems, dof_vec)


class FiniteElement(object):

    """Represents a finite element and holds relevant data."""

    def __init__(self, dof_mngr, cell, compute_flags):
        """Represents a finite element that is mainly defined by a single cell
        on a mesh and its local degrees of freedom, which map to global degrees
        of freedom defined on a mesh.

        Parameters
        ----------
        dof_mngr : DOFManager
            Defines the global degrees of freedom to which the element's local
            degrees of freedom map.
        cell : Cell
            The cell on a mesh on which the finite element is defined.
        compute_flags : dict
            Dictionary of flags specifying what quantities should be computed
            and available after initialization of the `FiniteElement` object.
        """
        self._cell = cell
        self._dpn = dof_mngr._dpn
        self._basis = dof_mngr._basis
        self._quad_rule = self._basis._quad_rule
        self._mapping = sem.mapping.Mapping(dof_mngr._map_basis, cell,
                                            compute_flags)
        self._cmpflags = compute_flags

        self._l_dof_ind_hier, self._g_dof_ind_hier = self._compute_hier_dofs()

    def _compute_hier_dofs(self):

        # hierarchically number the local indices of the degrees of freedom
        dpn = self._dpn
        l_node_ind_hier = self._cell.geometry.hierarchical_node_order
        l_dof_ind_hier = np.zeros(self.ndof, dtype=np.uint32)
        for i in range(dpn):
            l_dof_ind_hier[i::dpn] = dpn * l_node_ind_hier + i

        # hierarchically number the global indices of the degrees of freedom
        g_node_ind_hier = self._cell.node_ind_hierarchical
        g_dof_ind_hier = np.zeros(self.ndof, dtype=np.uint32)
        for i in range(dpn):
            g_dof_ind_hier[i::dpn] = dpn * g_node_ind_hier + i

        return l_dof_ind_hier, g_dof_ind_hier

    @property
    def ndim(self):
        return self._basis.ndim

    @property
    def x_phys(self):
        return self._mapping.x_phys

    @property
    def J(self):
        return self._mapping.J

    @property
    def invJ(self):
        return self._mapping.invJ

    @property
    def detJxW(self):
        detJ = self.mapping.detJ
        return self._quad_rule.xweight(detJ)

    @property
    def ndof(self):
        return self.n_nodes * self._dpn

    @property
    def ndof_exterior(self):
        return self.n_exterior_nodes * self._dpn

    @property
    def ndof_interior(self):
        return self.n_interior_nodes * self._dpn

    @property
    def loc_dof_ind_hier(self):
        return self._l_dof_ind_hier

    @property
    def global_dof_ind_hier(self):
        return self._g_dof_ind_hier

    @property
    def exterior_dof_ind(self):
        return self._g_dof_ind_hier[:self.ndof_exterior]

    @property
    def interior_dof_ind(self):
        return self._g_dof_ind_hier[self.ndof_exterior:]

    @property
    def n_nodes(self):
        return self._cell.n_nodes

    @property
    def n_exterior_nodes(self):
        return self._cell.n_exterior_nodes

    @property
    def n_interior_nodes(self):
        return self._cell.n_interior_nodes

    @property
    def basis(self):
        """Returns the basis used to approximate the continuous unknown
        functions (the degrees of freedom) over the parametric coordinates of
        the finite element.
        """
        return self._basis

    @property
    def mapping(self):
        """Returns the basis used for mapping the parametric coordinates of the
        finite element to the real coordinates of the mesh.
        """
        return self._mapping

    @property
    def quadrature(self):
        return self._quad_rule

    @property
    def node_ind(self):
        """Returns the lexicographically arranged global node indices
        corresponding to the cell describing the finite element.
        """
        return self._cell.node_ind_lexicographic

    def local(self, arr):
        """Return the local part of a global array
        """
        return arr[self.node_ind]

    def interpolate(self, coeffs, x_param):
        assert (x_param >= -1).all() and (x_param <= 1).all()
        return self._basis.interpolate(coeffs, x_param)

    def deriv(self, coeffs, dim):
        # TODO: differentiate coefficents with respect to physical coordinates
        # on the finite element
        return np.einsum('i...,i...',
                         self.invJ[:, dim], self._basis.gradient(coeffs))

    def gradient(self, coeffs):
        # TODO: gradient of coefficents with respect to physical coordinates
        # on the finite element
        return np.einsum('ij...,i...->j...',
                         self.invJ, self._basis.gradient(coeffs))

    def integrate(self, coeffs):
        # TODO: integrate coefficients on the finite element
        return (coeffs * self.detJxW).sum()
        # return self.basis.integrate(coeffs)

    def values_at_nodes(self, coeffs):
        """
        Return the values of a function approximated by `coeffs` at the nodes
        of the finite element, which are equispaced in the parametric
        coordinates of the element.
        """
        return self._basis.interpolate_on_grid_eq(coeffs)

    def sub_fe(self, face):
        return SubFiniteElement(self, face)

    def boundary_elements(self, name):
        bnd_id = self._cell._mesh._boundary_id_lookup[name]
        for ndim, face in self._cell._boundary_data.get(bnd_id, []):
            yield SubFiniteElement(self, face)


class SubFiniteElement(FiniteElement):
    """Finite element that lies on another higher-dimensional finite element.
    """

    def __init__(self, parent_fe, face):

        self._parent_fe = parent_fe
        self._cell = parent_fe._cell.sub_cell(face)
        self._dpn = parent_fe._dpn
        self._cmpflags = parent_fe._cmpflags
        # (parametric) normal axis of the sub-element
        # get the proper sub-basis from the parent element
        naxis = face // 2
        self._basis = parent_fe.basis.get_subbasis(naxis)
        self._mapping = parent_fe.mapping.get_submapping(face)
        # TODO: probably should have a get_subquadrature() method to this.
        self._quad_rule = self._basis._quad_rule

        self._l_dof_ind_hier, self._g_dof_ind_hier = self._compute_hier_dofs()

    @property
    def parent_fe(self):
        return self._parent_fe

    @property
    def n_dS(self):
        return self._mapping.n_dS

    @property
    def dS(self):
        return self._mapping.dS

    @property
    def dSxW(self):
        return self._quad_rule.xweight(self.dS)

    @property
    def unit_normal(self):
        return self._mapping.unit_normal

    @property
    def n_dSxW(self):
        return self._quad_rule.xweight(self.n_dS)

    def slice_from_parent(self, arr):
        par_fe = self._parent_fe
        face = self._mapping._face
        return _subface_slice(face, arr, par_fe.ndim)

    def parent_dofs(self):
        par_fe = self._parent_fe
        arr = np.arange(par_fe.n_nodes).reshape(par_fe.basis.coeff_shape)
        face_id = self._mapping._face
        node_idx = _subface_slice(face_id, arr, par_fe.ndim)
        dof_idx = np.empty(self.ndof, dtype=np.uint32)
        dpn = self._dpn
        for i in range(dpn):
            dof_idx[i::dpn] = node_idx * dpn + i
        return dof_idx

    def integrate(self, coeffs):
        return (coeffs * self.dSxW).sum(axis=-1)

    def gradient(self, coeffs):
        grad = self._parent_fe.gradient(coeffs)
        face_id = self._mapping._face
        return _subface_slice(face_id, grad, self._parent_fe.ndim)


class CellBase(object):
    """Part of a mesh whose nodes define the geometry of a cell."""

    @property
    def ndim(self):
        return self.geometry.ndim

    @property
    def n_nodes(self):
        return self.geometry.n_nodes

    @property
    def n_exterior_nodes(self):
        return self.geometry.n_exterior_nodes

    @property
    def n_interior_nodes(self):
        return self.geometry.n_interior_nodes

    @property
    def geometry(self):
        return self._geometry

    def __init__(self, mesh, geometry, node_map):
        """TODO: to be defined1.

        :geometry: TODO

        """
        self._mesh = mesh
        self._geometry = geometry
        self._node_map = node_map

    @property
    def node_ind_lexicographic(self):
        return self._node_map

    @property
    def nodes_lexicographic(self):
        return self._mesh.nodes[:, self.node_ind_lexicographic]

    @property
    def node_ind_hierarchical(self):
        return self._node_map.flat[self.geometry._hier_node_order]

    @property
    def nodes_hierarchical(self):
        return self._mesh.nodes[:, self.node_ind_hierarchical]

    @property
    def vertex_node_ind(self):
        corner_ind_loc = self.geometry.vertex_node_ind
        return self._node_map.flat[corner_ind_loc]

    @property
    def vertex_nodes(self):
        return self._mesh.nodes[:, self.vertex_node_ind]

    @property
    def exterior_node_ind(self):
        bnd_ind_loc = self._geometry.exterior_node_ind
        return self._node_map.flat[bnd_ind_loc]

    @property
    def exterior_nodes(self):
        return self._mesh.nodes[:, self.exterior_node_ind]

    @property
    def interior_node_ind(self):
        int_ind_loc = self._geometry.interior_node_ind
        return self._node_map.flat[int_ind_loc]

    @property
    def interior_nodes(self):
        return self._mesh.nodes[:, self.interior_node_ind]

    def sub_cell(self, face):
        return SubCell(self, face)


class Cell(CellBase):

    def __init__(self, mesh, geometry, node_map, region_id, adj_map,
                 boundary_data):
        CellBase.__init__(self, mesh, geometry, node_map)
        self._region_id = region_id
        self._adj_map = adj_map
        self._boundary_data = boundary_data

    @property
    def region_id(self):
        return self._region_id

    @property
    def region_name(self):
        return self._mesh._region_names[self._region_id]

    def neighbor(self, face):
        neighbor_cell_face = self._adj_map[face]
        if neighbor_cell_face is not None:
            return self._mesh.get_cell(self._adj_map[face])

    def boundary_cells(self, name):
        bnd_id = self._mesh._boundary_id_lookup[name]
        for ndim, face in self._boundary_data.get(bnd_id, []):
            yield self.sub_cell(face, ndim)


class SubCell(CellBase):

    """A cell on a face/edge of another cell.
    """

    def __init__(self, parent_cell, face):
        """Initialize a SubCell

        Parameters
        ----------
        parent_cell : Cell
            Cell on which the SubCell lies
        face : int
            A number specifying the face/edge of `parent_cell` defining the
            SubCell

        """
        pc = parent_cell
        axis = face // 2
        # axis_pos = bool(face % 2)
        mesh = pc._mesh
        geometry = pc.geometry.sub_geometry(axis)
        node_map = _subface_slice(face, pc._node_map, pc.ndim)

        # tr = range(axis, pc.ndim) + range(0, axis)
        # par_node_map_T = pc._node_map.transpose(tr)
        # if axis_pos is True:
        #     node_map = par_node_map_T[pc.geometry.shape[axis] - 1]
        # else:
        #     node_map = par_node_map_T[0].transpose()

        CellBase.__init__(self, mesh, geometry, node_map)
        self._parent_cell = pc


class Mesh(object):
    """
    Class representing a finite element mesh.

    Attributes
    ----------
    ndim : int
        Number of dimensions in the mesh.
    n_nodes : int
        Number of nodes on the mesh
    n_cells : int
        Number of cells on the mesh
    """

    CellData = namedtuple("CellData",
                          ['geometry_id', 'region_id', 'node_map'])
    BoundaryData = namedtuple("BoundaryData", ['ndim', 'index'])

    @property
    def ndim(self):
        """Number of spacial dimensions spanned by the mesh.
        """
        return self._ndim

    @property
    def n_nodes(self):
        """Number of nodes on the mesh.
        """
        return self.nodes.shape[1]

    @property
    def n_cells(self):
        """Number of cells on the mesh.
        """
        return len(self._cell_data)

    @property
    def n_boundary_cells(self):
        """Number of boundary cells.
        """
        return len(self._boundary_map)

    def __init__(self, ndim):
        """
        Create an empty mesh with the specified number of dimensions.

        Parameters
        ----------
        ndim : int
            number of spacial dimensions occupied by the mesh.
        """
        self._ndim = ndim
        self._geometries = []
        self._cell_data = []
        self._adj_map = []

        # Map region id # to a string identifier
        self._region_names = []
        self._region_id_lookup = {}
        # Map boundary id # to a string identifier
        self._boundary_names = []
        self._boundary_id_lookup = {}
        # Numbers of cells on each boundary
        # (# dims, index of sub-cell)
        self._boundary_map = {}
        self._boundary_cells = []

        self._finalized = False
        self.condensed = False

    def add_geometry(self, geometry):
        """Add a new cell geometry to the mesh.
        """
        if geometry.ndim <= self.ndim:
            geometry_id = len(self._geometries)
            self._geometries.append(geometry)
            return geometry_id
        else:
            raise ValueError(
                "Cell geometry has more dimensions than the mesh.")

    def new_region(self, name):
        """Add a new named region to the mesh.
        """
        region_id = len(self._region_names)
        self._region_names.append(name)
        self._region_id_lookup[name] = region_id
        return region_id

    def new_boundary(self, name):
        """Add a new named boundary to the mesh.
        """
        boundary_id = len(self._boundary_names)
        self._boundary_names.append(name)
        self._boundary_id_lookup[name] = boundary_id
        self._boundary_cells.append(set())
        return boundary_id

    def set_nodes(self, nodes):
        """Set coordinates of nodes on the mesh

        Parameters
        ----------
        nodes : array-like, `ndim`-by-N
            The (global) set of nodes on the mesh.
        """
        # TODO: Raise exception if mesh is finalized
        self.nodes = np.asarray(nodes)
        if self.nodes.shape[0] != self.ndim:
            raise ValueError("Points have the wrong number of dimensions.")

    def add_cell(self, node_ind, geometry_id, region_id):
        """
        Add cells to the mesh.

        Parameters
        ----------
        geometry : geometries.geometry
            geometry to be associated with the new cell.
        region_name : str
            Name of the mesh region that the cell belongs to.
        node_ind : array-like[Q]
            Global indices of nodes associated with the cell.
        """
        node_ind = np.asarray(node_ind, dtype=np.uint32)
        entry = Mesh.CellData(geometry_id, region_id, node_ind)
        self._cell_data.append(entry)
        geometry = self._geometries[geometry_id]
        self._adj_map.append([None] * geometry.n_sub_geometries())

    def add_boundary_cell(self, cell_number, bnd_id, ndim, index):
        """Identify boundary of a cell as being a mesh boundary

        Parameters
        ----------
        cell_number : int
            Identifying number of the cell
        bnd_id : int
            Identifying number of the boundary
        ndim : int
            Dimensionality of the boundary cell
        index : int
            Number identifying which face/edge of the cell the boundary is on
        """
        bnd_data_cell = self._boundary_map.setdefault(cell_number, {})
        bnd_data_bnd = bnd_data_cell.setdefault(bnd_id, [])
        bnd = Mesh.BoundaryData(ndim, index)
        bnd_data_bnd.append(bnd)
        self._boundary_cells[bnd_id].add(cell_number)

    def get_geometries(self):
        return self._geometries

    def get_cell(self, i):
        geometry_id, region_id, node_map = self._cell_data[i]
        geometry = self._geometries[geometry_id]
        adj_map = self._adj_map[i]
        boundary_map = self._boundary_map.get(i, {})
        return Cell(self, geometry, node_map, region_id, adj_map, boundary_map)

    @property
    def cells(self):
        """Iterator through all cells on the mesh
        """
        for i in range(self.n_cells):
            yield self.get_cell(i)

    def cells_on_boundary(self, name):
        """Iterator through all cells on a boundary named `name`.
        """
        bnd_id = self._boundary_id_lookup[name]
        bnd_cells = sorted(self._boundary_cells[bnd_id])
        for cell_num in bnd_cells:
            yield self.get_cell(cell_num)

    def cells_are_neighbors(self, cell1, cell2):
        """ Check if two cells border one another.
        """
        node_is_common = np.in1d(cell1.vertex_node_ind,
                                 cell2.vertex_node_ind,
                                 True)
        # if np.all(node_is_common == False):
        #     return False
        for side, vertex_map in enumerate(cell1.geometry.corner_verts):
            if np.all(node_is_common == vertex_map):
                return side
        return -1

    def _compute_cell_centroids(self):
        """Compute the approximate centers of each cell."""
        centroids = np.zeros((self.n_cells, 2))
        for i, cell in enumerate(self.cells):
            centroids[i] = np.mean(cell.vertex_nodes.reshape(2, -1), axis=1)
        self._centroids = centroids

    def _permute_nodes(self, perm):
        """Permute the indices of the nodes on the mesh.
        """
        # reorder the global node data
        self.nodes[:, :perm.size] = self.nodes[:, perm]

        # find the inverse permutation array
        inv_perm = np.zeros_like(perm)
        inv_perm[perm] = np.arange(perm.size)

        # Update the cells with the new permutation of the nodes
        for cell_data in self._cell_data:
            cell_data.node_map[:] = inv_perm[cell_data.node_map]
