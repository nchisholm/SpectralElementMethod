# -*- coding: utf-8 -*-

"""Module for describing sets of finite element basis functions.
"""


import os.path
import itertools as it
import numpy as np
import scipy.linalg as spla
import h5py

import sem
from . import quadratures


class _Basis(object):
    """Common base class to be inherited by more specific classes describing
    sets of finite element basis functions.
    """

    def get_coeff_rank(self, coeffs):
        return coeffs.ndim - self.ndim

    def interpolate(self, coeffs, x):
        # Subclasses might implement their own faster interpolation methods
        assert x.ndim == 1
        assert coeffs.shape[-1] == self.n_coeffs
        values = np.einsum('mr,...r->...m', self(x), coeffs)
        return values


class _Nodal(object):
    """Nodal bases should inherit from this class.
    """

    @property
    def nodes(self):
        """Nodes of the basis, which lie on the interval [-1, 1]"""
        return self._nodes

    @property
    def n_nodes(self):
        """Number of nodes in the basis."""
        return self._nodes.size

    @property
    def n_coeffs(self):
        """Number of coefficients in the basis."""
        return self._nodes.size


class _QuadSupported(object):
    """This class should be inherited by *nodal* bases whose nodes may be
    prescribed quadrature weights.
    """

    @property
    def quad_rule(self):
        """Quadrature rule whose abscissa are the nodes of the basis."""
        return self._quad_rule

    def __init__(self, quad_wts):
        """
        Parameters
        ----------
        quad_rule : quadrature.QuadratureRule
            Quadrature rule.
        """
        self._quad_rule = quadratures.Quadrature1D(self._nodes, quad_wts)

    def integrate(self, coeffs):
        """Compute the definite integral using numerical quadrature."""
        return self._quad_rule.integrate(coeffs)


class _Basis1D(_Basis):
    """All one-dimensional basis classes should inherit from this class.
    """

    @property
    def ndim(self):
        """Number of the dimensions in the basis, which is one."""
        return 1

    @property
    def coeff_shape(self):
        return (self.n_coeffs,)

    @property
    def D1(self):
        """First-order differentiation matrix, :math:`D^{(1)}`.

        Given the coefficients :math:`f_i` that interpolate :math:`f(x)`,
        :math:`I[f(x)] = f_i b_i(x)`, left-multiplying by this matrix produces
        takes the derivative of :math:`I[f(x)]`. :math:`I[f(x)]' =
        f'_i b_i(x)`. Thus, if the original interpolating function is
        .. math::
           c'_i = D^{(1)} c_i"""
        return self._D1

    def get_D1_matrix(self, dim=0):
        """Return the differentiation matrix for a specified dimension.

        Parameters
        ----------
        dim : int
            Dimension to which the differentiation matrix should correspond.
        """
        return self._diff_matrix

    def get_D1_matrices(self):
        """Returns a list of the first-order differentiation matrices of the
        basis. The first element corresponds to the differentiation matrix for
        the first dimension, the second corresponds to the second dimension,
        and so on.
        """
        return [self._D1]

    def deriv(self, coeffs):
        assert coeffs.shape[-1] == self.n_coeffs
        dcoeffs = np.einsum('mr,...r->...m', self._D1, coeffs)
        return dcoeffs

    def gradient(self, coeffs):
        return self.deriv(coeffs)


class _BasisND(_Basis):
    """Multi-dimensional basis classes should inherit from this class.
    """

    @property
    def ndim(self):
        """Number of dimensions spanned by the basis"""
        return self._ndim

    @property
    def D1(self):
        """A list of the first-order differentiation matrices of the basis. The
        first element corresponds to the differentiation matrix for the first
        dimension, the second corresponds to the second dimension, and so on.
        """
        return self._D1

    def get_D1_matrix(self, dim):
        """Return the differentiation matrix for a specified dimension.

        Parameters
        ----------
        dim : int
            Dimension to which the differentiation matrix should correspond.
        """
        return self._D1_mats[dim]

    def get_D1_matrices(self):
        """Returns a list of the first-order differentiation matrices of the
        basis. The first element corresponds to the differentiation matrix for
        the first dimension, the second corresponds to the second dimension,
        and so on.
        """
        return self._D1_mats[:]

    def deriv(self, dim, coeffs):
        """Differentiate in the specified dimension

        Parameters
        ----------
        dim : int
           Dimension along which to differentiate
        coeffs : numpy.ndarray
           Coefficients to differentiate

        Returns
        -------
        numpy.ndarray
            Differentiated coefficients
        """
        return np.einsum('mr,...r->...m', self._D1_mats[dim], coeffs)

    def gradient(self):
        raise NotImplementedError()


class BarycentricLagrange(_Basis1D, _Nodal):
    """Represents a Lagrange polynomial basis where the Lagrange nodes
    correspond to the nodes of a quadrature rule.
    """

    @property
    def deg(self):
        """Polynomial degree of the basis functions"""
        return self._nodes.size - 1

    @property
    def bary_wts(self):
        """Barycentric interpolation weights"""
        return self._bary_wts

    def __init__(self, nodes, bary_wts):
        """
        Parameters
        ----------
        nodes : ndarray
            Nodes of the Lagrange polynomial basis
        bary_wts : ndarray
            Barycentric Lagrange interpolation weights
        """

        self._nodes = nodes
        self._bary_wts = bary_wts

        # differentiation matrix (1st derivative)
        D1 = bary_wts[None, :] / bary_wts[:, None]
        D1 /= nodes[:, None] - nodes[None, :]
        np.fill_diagonal(D1, 0.)
        np.fill_diagonal(D1, -D1.sum(axis=1))

        self._D1 = D1

        # interpolation matrix to equally spaced points
        x_eq = np.linspace(-1, 1, self.n_nodes)
        self._interp_eq_mat = self(x_eq)
        self._interp_eq_mat_lu = spla.lu_factor(self._interp_eq_mat)

    def __call__(self, x):
        r"""
        Evaluate each Lagrange basis function at the given set of points.

        Parameters
        ----------
        x : numpy.ndarray
            Points at which to evaluate each of the Lagrange basis functions.

        Returns
        -------
        result : numpy.ndarray
            Values of each of the Lagrange basis polynomials at the points `x`
            such that :math:`B_{ij} = p_j(x_i)` where :math:`p_j` is the j-th
            Lagrange basis polynomial and :math:`B` is the array output.
        """
        # TODO: *test* identity matrix property

        # multiple sets of points `x` may be passed through
        nodes = self._nodes
        weights = self._bary_wts

        kern = weights / (x[..., None] - nodes)
        kern_sum = kern.sum(axis=-1)
        kern_sum.shape += (1,)

        result = kern / kern_sum
        result[np.isnan(result)] = 1.

        return result

    def __repr__(self):
        return "{}(deg={})".format(self.__class__.__name__, self.deg)

    def interpolate(self, f, x, broadcast=False):
        r"""Given values of a function at the nodes, evaluate the Lagrange
        interpolating function at the specified points.

        Parameters
        ----------
        f : numpy.ndarray
            Values of the function f at the nodes of the Lagrange basis.  The
            size of the last axis of this array should match the number of
            nodes in the basis.  These are effectively the coefficients of the
            basis functions.
        x : numpy.ndarray
            Points at which to evaluate the interpolant of f.
        broadcast : bool
            If False (default), `f` is assumed to be the same for all `x`.  If
            `True`, broadcast `f` against `x` such that the leading dimensions
            of `f` correspond to the dimensions of `x`.  Thus, the leading
            dimensions of `f` must either match dimensions of `x` or be
            singleton dimensions.

        Returns
        -------
        result : numpy.ndarray
            The values of the interpolating Lagrange polynomial at `x` with
            values equal to `f` on the nodes.

        Notes
        -----
        Multiple coefficient sets :math:`f_{pqr \dots i}` (corresponding to
        distinct Lagrange interpolants) and multidimensional arrays of points
        :math:`x_{lmn \dots}` can be passed such that the output is like
        :math:`P_{lmn \dots pqr \dots} = I[f_{pqr\dots}](x_{ijk\dots})` where
        :math:`I[f]` is the Lagrange interpolant of :math:`f = f(x)`.

        With `broadcast=True`, the behavior of `broadcast=False` can be
        replicated by making `f` contain as many leading singleton dimensions
        as there are dimensions in `x`.
        """

        nodes = self._nodes
        weights = self._bary_wts

        # interpolate using the barycentric Lagrange formula (for all
        # values in `x` that aren't on a node exactly).  This formula
        # is stable even if `x` is very close (but not equal) to a node.

        # NOTE: `[...]` after numpy expressions disables automatic conversion
        #   of the array to a scalar
        x = np.asarray(x)
        kern = weights / (x[..., None] - nodes)
        kern_sum = kern.sum(axis=-1)[...]
        if broadcast:
            # number of "extra" axes in the coefficient array (that do not
            # correspond to interpolation points)
            n_free_axes = f.ndim - 1 - x.ndim
            f_ax = [Ellipsis] + list(range(n_free_axes)) + [n_free_axes]
            kern_ax = [Ellipsis] + [n_free_axes]
            result_ax = [Ellipsis] + list(range(n_free_axes))
            result = np.einsum(kern, kern_ax, f, f_ax, result_ax)[...]
        else:
            n_free_axes = f.ndim - 1
            result = np.inner(kern, f)[...]
        kern_sum.shape += (1,) * n_free_axes
        result /= kern_sum

        # handle case where an element of `x` is exactly a on a node, replacing
        # all `nan`s with the specified values of f at the node.
        inf_ix = np.nonzero(np.isinf(kern))
        node_pts = inf_ix[:-1]
        node_val = inf_ix[-1]
        if node_val.size > 0:
            if x.ndim == 0:
                result[:] = f[..., node_val[0]]
            elif broadcast:
                f_ix = node_pts + (Ellipsis, node_val)
                result[node_pts] = f[f_ix]
            else:
                result[node_pts] = np.rollaxis(f[..., node_val], -1)

        # NOTE: the [()] indexing below will trigger automatic conversion of 0D
        # numpy arrays (scalars) to a python scalar type unless [...] is added.
        return result[()]


class LagrangeGaussLobatto(BarycentricLagrange, _QuadSupported):
    """A Lagrange interpolation basis through the Gauss-Legendre-Lobotto
    quadrature nodes.
    """

    def __init__(self, order):
        """
        Parameters
        ----------
        order : int
            Order of the Lagrange interpolating polynomials (>= 1).
        """

        if order < 1:
            raise ValueError("Must specify an order of 1 or greater.")

        # Load the pre-computed nodes, barycentric weights, and associated
        # quadrature weights.
        module_path = os.path.dirname(sem.__file__)
        basis_data_file = os.path.join(module_path, 'data', 'basis-data.hdf5')
        with h5py.File(basis_data_file, 'r') as dataf:
            group = dataf["GaussLegendreLobatto"]
            max_order = group.attrs["max_order"]
            if order > max_order:
                raise NotImplementedError(
                    "Basis only available up to order {}.".format(max_order))
            basis_data = group[str(order)][:]

        nodes = np.zeros(order + 1)
        bary_wts = np.zeros_like(nodes)
        quad_wts = np.zeros_like(nodes)

        # now assign these values for the interval over x [-1, 1]
        m = nodes.size // 2
        nodes[m:] = basis_data[0, :]
        bary_wts[m:] = basis_data[1, :]
        quad_wts[m:] = basis_data[2, :]
        if nodes.size % 2 == 1:
            nodes[:m] = -basis_data[0, -1:0:-1]
            bary_wts[:m] = basis_data[1, -1:0:-1]
            quad_wts[:m] = basis_data[2, -1:0:-1]
        else:
            nodes[:m] = -basis_data[0, -1::-1]
            bary_wts[:m] = -basis_data[1, -1::-1]
            quad_wts[:m] = basis_data[2, -1::-1]

        self._n_coeffs = order + 1

        BarycentricLagrange.__init__(self, nodes, bary_wts)
        _QuadSupported.__init__(self, quad_wts)


class TensorProduct(_BasisND):
    """A basis that is a tensor product of lower dimensional sub-bases.
    """

    @property
    def coeff_shape(self):
        return self._coeff_shape

    @property
    def n_subbases(self):
        return len(self._subbases)

    def __init__(self, *subbases):
        """
        Construct a basis of shape functions formed by two or more sets of
        lower-dimensional basis functions.

        Parameters
        ----------
        subbases : Subbasis
            The individual sets of functions used to form the basis.

        Notes
        -----
        The number of dimensions spanned by the TensorProduct basis is equal to
        the sum of the number of dimensions spanned by each sub-basis supplied.
        """
        if len(subbases) < 1:
            raise ValueError("Tensor product basis must comprise at "
                             "least two lower dimensional bases.")
        self._subbases = subbases
        self._ndim = sum(basis.ndim for basis in subbases)
        # Specify the shape of an array specifying the coefficient of each
        # function in the basis set.
        self._coeff_shape = tuple(it.chain.from_iterable(
            basis.coeff_shape for basis in self._subbases))
        # Compute the total number of degrees of freedom (individual functions)
        # in the basis set and assign spacial axes to each sub-basis.
        self._subbasis_dims = []
        self._D1_mats = []
        self._n_coeffs = 1
        dim_first = 0
        for basis in self._subbases:
            self._n_coeffs *= basis.n_coeffs
            if isinstance(basis, _Basis1D):
                self._subbasis_dims.append(dim_first)
                dim_first += 1
                self._D1_mats.append(basis.D1)
            else:
                dim_last = dim_first + basis.ndim
                self._subbasis_dims.append(slice(dim_first, dim_last))
                dim_first = dim_last
                self._D1_mats.extend(basis._D1_mats)

    def get_subbasis(self, dim):
        """
        Get the sub-basis lying on the given face of a given basis.

        Parameters
        ----------
        basis : BasisND
            The parent basis of dimension N.
        dim : int
            Dimension to which the sub-basis is normal.

        Returns
        -------
        Basis
            Sub-basis of dimension N-1.

        """
        if self.ndim == 2:
            return self._subbases[dim]
        # "roll" subbases into the right order
        subbases = self._subbases[dim+1:] + self._subbases[:dim]
        cls = type(self)
        return cls(subbases)

    def iter_subbases(self, reverse=False):
        if not reverse:
            return zip(self._subbasis_dims, self._subbases)
        return zip(reversed(self._subbasis_dims),
                       reversed(self._subbases))

    def __call__(self, x):
        """
        Evaluate the basis at coordinate array `x`.

        Parameters
        ----------
        x : sequence of numpy.ndarray
            Points at which to evaluate the basis.  Each element of the
            sequence corresponds to each dimension.
        """
        # This function evaluates each of the basis functions at `x`.
        # B[M, I] = B[I](x[M]) = b1[i](x1[m]) b2[j](x2[n]) ...

        if len(x) != self.ndim:
            raise ValueError("Cannot evaluate {}-dimensional basis at "
                             "a {}-dimensional set of points"
                             .format(self.ndim, len(x)))

        # result_outer_shape = tuple(sb.n_coeffs for sb in self._subbases)
        # result_inner_shape = x.shape
        # result_ndim = len(result_outer_shape) + len(result_inner_shape)

        einsum_args = []
        for i, (dim, basis) in enumerate(self.iter_subbases()):
            values = basis(x[dim])
            einsum_args.append(values)
            einsum_args.append([Ellipsis, i])
        einsum_args.append([Ellipsis] + list(range(self.n_subbases)))
        return np.einsum(*einsum_args)

    def interpolate(self, coeffs, x):
        """Interpolation to arbitrary points `x`
        """
        coeff_shape = coeffs.shape[-self.n_subbases:]
        assert coeff_shape == self.coeff_shape

        out = coeffs
        for dim, basis in self.iter_subbases(reverse=True):
            if dim < self.ndim - 1:
                out = basis.interpolate(out, x[dim], broadcast=True)
            else:
                out = basis.interpolate(out, x[dim])
        return out

    def interpolate_on_grid(self, coeffs, x):
        """Interpolation to a grid of points `x` in the parametric space of the
        basis.
        """
        # more efficient interpolation on a grid of points
        assert len(x) == self.ndim
        coeff_shape = coeffs.shape[-self.n_subbases:]
        assert coeff_shape == self.coeff_shape

        out = coeffs
        for dim, basis in self.iter_subbases(reverse=True):
            out = basis.interpolate(out, x[dim])

        return out

    def interpolate_on_grid_eq(self, coeffs):
        """Interpolation to an equispaced grid of points `x` in the parametric
        space of the basis. The number of nodes in each direction is the same
        as the number of coefficients in each 1D sub-basis.
        """
        # more efficient interpolation on a grid of points
        coeff_shape = coeffs.shape[-self.n_subbases:]
        grid_shape = coeff_shape
        rank_shape = coeffs.shape[:-self.n_subbases]
        assert coeff_shape == self.coeff_shape

        out = coeffs.reshape((-1,) + coeff_shape)
        out = np.rollaxis(out, 0, self.n_subbases + 1)
        for dim, basis in self.iter_subbases(reverse=True):
            # convert from multidimensional (index) form to a matrix of an
            # appropriate shape.
            out = np.rollaxis(out, self.n_subbases - 1)
            out = out.reshape(coeff_shape[dim], -1)
            # compute intermediate values
            out = np.dot(basis._interp_eq_mat, out)
            # convert back to indexed multi-dimensional form
            nd_shape0 = [grid_shape[dim]]
            nd_shape1 = [grid_shape[d] for d in range(dim + 1, self.ndim)]
            nd_shape2 = [coeff_shape[i] for i in range(0, dim)]
            nd_shape3 = [-1]
            nd_shape = nd_shape0 + nd_shape1 + nd_shape2 + nd_shape3
            out.shape = nd_shape
            # values = np.einsum('js,...rs->...jr', basis(x[dim]), values)
        out = np.rollaxis(out, -1)
        out = out.reshape(rank_shape + grid_shape)
        return out

    def compute_coeffs_grid(self, values, x):
        """Compute coefficients
        """
        assert len(x) == self.ndim
        coeff_shape = self.coeff_shape
        grid_shape = tuple(len(xd) for xd in x)
        rank_shape = values.shape[:-self.ndim]
        assert values.shape[-self.ndim:] == grid_shape
        assert grid_shape == coeff_shape

        out = values.reshape((-1,) + grid_shape)
        out = np.rollaxis(out, 0, self.ndim + 1)
        for dim, basis in self.iter_subbases():
            # convert from multidimensional (index) form to a matrix of an
            # appropriate shape.
            out = out.reshape(grid_shape[dim], -1)
            out = spla.solve(basis(x[dim]), out)
            # convert back to multidimensional form and "transpose"
            nd_shape0 = [basis.n_coeffs]
            nd_shape1 = [grid_shape[d] for d in range(dim+1, self.ndim)]
            nd_shape2 = [coeff_shape[i] for i in range(0, dim)]
            nd_shape3 = [-1]
            out.shape = nd_shape0 + nd_shape1 + nd_shape2 + nd_shape3
            out = np.rollaxis(out, 0, self.n_subbases)
        out = np.rollaxis(out, -1)
        out = out.reshape(rank_shape + coeff_shape)
        return out

    def compute_coeffs_grid_eq(self, values):
        """Compute the basis coefficients from the values of scalar field(s) on
        an equispaced grid of appropriate size.
        """
        coeff_shape = self.coeff_shape
        grid_shape = coeff_shape
        rank_shape = values.shape[:-self.ndim]
        assert values.shape[-self.ndim:] == grid_shape

        out = values.reshape((-1,) + grid_shape)
        out = np.rollaxis(out, 0, self.ndim + 1)
        for dim, basis in self.iter_subbases():
            # convert from multidimensional (index) form to a matrix of an
            # appropriate shape.
            out = out.reshape(grid_shape[dim], -1)
            out = spla.lu_solve(basis._interp_eq_mat_lu, out)
            # convert back to multidimensional form and "transpose"
            nd_shape0 = [basis.n_coeffs]
            nd_shape1 = [grid_shape[d] for d in range(dim+1, self.ndim)]
            nd_shape2 = [coeff_shape[i] for i in range(0, dim)]
            nd_shape3 = [-1]
            out.shape = nd_shape0 + nd_shape1 + nd_shape2 + nd_shape3
            out = np.rollaxis(out, 0, self.n_subbases)
        out = np.rollaxis(out, -1)
        out = out.reshape(rank_shape + coeff_shape)
        return out

    def deriv(self, coeffs, dim):
        """
        Differentiate with respect to the specified dimension.
        """
        coeff_shape = coeffs.shape[-self.n_subbases:]
        # rank_shape = coeffs.shape[:-self.n_subbases]
        assert coeff_shape == self.coeff_shape

        diff_mat = self._D1_mats[dim]
        dummy_sub = self.n_subbases    # identify summation subscript
        out_subs = [Ellipsis] + list(range(self.n_subbases))
        coeff_subs = [dummy_sub if d == dim else d for d in out_subs]
        return np.einsum(diff_mat, [dim, dummy_sub], coeffs,
                         coeff_subs, out_subs)

    def gradient(self, coeffs):
        """Differentiate with respect to all directions."""
        coeff_shape = coeffs.shape[-self.n_subbases:]
        rank_shape = coeffs.shape[:-self.n_subbases]
        assert coeff_shape == self.coeff_shape

        grad = np.empty((self.ndim,) + rank_shape + coeff_shape)
        for i in range(self.ndim):
            grad[i] = self.deriv(coeffs, i)
        return grad

    def __repr__(self):
        arglist = [basis.__repr__() for basis in self._subbases]
        return "{}({})".format(self.__class__.__name__, ", ".join(arglist))

    def __str__(self):
        return ("<{}D Basis> with basis functions:\n".format(self.ndim) +
                "\n".join("[dim {}]: {!s}".format(i, basis)
                          for i, basis in enumerate(self._subbases)))


class NodalTensorProduct(TensorProduct):

    @property
    def nodes(self):
        return tuple(sb.nodes for sb in self._subbases)

    def __init__(self, *subbases):
        for sb in subbases:
            if not isinstance(sb, _Nodal):
                raise ValueError("All subbases must be nodal.")
        TensorProduct.__init__(self, *subbases)

    def nodegrid(self, sparse=False):
        return np.meshgrid(*self.nodes, indexing='ij', sparse=sparse)

    def check_subbases(self, subbases):
        for sb in subbases:
            if not isinstance(sb, _Nodal):
                raise ValueError("All subbases must be nodal.")


class TensorProductQS(NodalTensorProduct, _QuadSupported):
    """Nodal tensor product basis with a quadrature rule defined on its nodes.
    """

    def __init__(self, *subbases):
        # All subbases must be supported
        if not all(isinstance(basis, _QuadSupported)
                   for basis in subbases):
            raise ValueError("All subbases must be supported by a quadrature "
                             "rule.")
        TensorProduct.__init__(self, *subbases)
        quad_rule = quadratures.TensorQuadratureRule(
            *(basis._quad_rule for basis in self._subbases)
        )
        self._quad_rule = quad_rule
