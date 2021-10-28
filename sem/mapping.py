#! /usr/bin/env python

'''
Classes mapping concrete elements to master elements.
'''

import numpy as np
import sem.rootfind
import sem.linalg


class OutsideDomain(Exception):
    """Exception to be raised if a given physical point is outside the domain
    of a finite element mapping object.
    """
    pass


def _subface_slice(face, arr, ndim):
    """Slice the coefficents on the boundary (face) of an N-D (N > 1) basis
    and return the coefficients of the corresponding (N-1)-D basis.

    Parameters
    ----------
    face : int
        Face # on the parent element.
    arr : ndarray
        Set of coefficient values on the parent element.
    ndim :
        Number of dimensions of the parent element.

    Returns
    -------
    ndarray
        Appropriately sliced and re-strided array that is a view on the
        original `arr` passed to the function.
    """
    assert ndim > 1
    assert face < 2*ndim
    # 'Roll' axes of the 'parent' array into the appropriate positions
    rank = arr.ndim - ndim
    ax = face // 2 + rank
    ax_pos = bool(face % 2)
    arrT = arr.transpose(
        list(range(rank)) + list(range(ax, arr.ndim)) + list(range(rank, ax)))

    if ax_pos is True:
        if ndim > 2:
            # Parametric coordinates of the element give the face a positive
            # orientation relative to the parametric coordinates of the
            # face-element.
            slc = (slice(None),)*rank + (-1,)
        else:
            # Orient 1-D boundary in counter-clockwise direction
            if face == 3:
                slc = (slice(None),)*rank + (-1, slice(None, None, -1))
            else:
                slc = (slice(None),)*rank + (-1, slice(None))
        return arrT[slc]
    else:
        if ndim > 2:
            # Parametric coordinates of the element give the face a negative
            # orientation relative to the parametric coordinates of the
            # face-element, and hence we need to flip the directions.
            slc = (slice(None),)*rank + (0,)
            # Transpose axes to reverse orientation to be outward to the
            # unit (hyper)cube.
            return arrT[slc].transpose(
                list(range(rank)) + list(range(arr.ndim-2, rank-1, -1)))
        else:
            # Orient 1-D boundary in counter-clockwise direction
            if face == 0:
                slc = (slice(None),)*rank + (0, slice(None, None, -1))
            else:
                slc = (slice(None),)*rank + (0, slice(None))
            return arrT[slc]


class Mapping(object):
    """Maps the parametric coordinate system of an element to (and from) the
    physical coordinate system.
    Represents the mapping M: R^n -> R^n from physical space to parametric
    space of a finite element.
    """

    def __init__(self, basis, cell, compute_flags):
        self._basis = basis
        self._cell = cell
        if compute_flags['x_phys']:
            self._x_phys = self._compute_x_phys()
        if compute_flags['Jacobian']:
            self._J, self._invJ, self._detJ = self._compute_jacobian()
        self._cmpflags = compute_flags
        # TODO: the above quantities should be read-only
        # IDEA: could lazily evaluate the above quantites depending on what
        # methods of this class are called.

    def _compute_x_phys(self):
        """Compute the basis coefficients that map the parametric coordinates
        of the finite element to the physical coordinates of the cell.
        """
        nodes = self._cell.nodes_lexicographic
        return self._basis.compute_coeffs_grid_eq(nodes)

    def _compute_jacobian(self):
        """Compute the basis coefficients of the Jacobian and inverse Jacobian
        matrix along with the Jacobian determinant multiplied by the quadrature
        integration weights at the quadrature points.
        """
        if self.ndim != 2:
            raise NotImplementedError("Only supporting 2D elements right now")
        # Jacobian is equal to the gradient (WRT physical coordinates)
        jacobian = self._basis.gradient(self.x_phys).swapaxes(0, 1)
        # TODO: only computing determinant and inverse for the 2D case and we
        # need to implement 1D and 3D cases.
        det_jacobian, inv_jacobian = sem.linalg.det_inv_2x2(jacobian)
        assert np.all(det_jacobian > 0)
        # detJxW = self._quad_rule.xweight(det_jacobian)
        return jacobian, inv_jacobian, det_jacobian

    @property
    def ndim(self):
        return self._basis.ndim

    @property
    def x_phys(self):
        return self._x_phys

    @property
    def J(self):
        return self._J

    @property
    def invJ(self):
        return self._invJ

    @property
    def detJ(self):
        return self._detJ

    def __call__(self, x_param):
        """Map parametric coordinates to physical coordinates.
        """
        return self._basis.interpolate(self.x_phys, x_param).swapaxes(-1, 0)

    def inv(self, x_phys, x_param_guess=None):
        """Map physical coordinates to parametric coordinates.

        Parameters
        ----------
        x_phys : array_like
            Physical coordinates
        x_param_guess : array_like
            Guess for local elemental computational coordinate.
        """
        x_phys = np.array(x_phys)
        x_phys = x_phys.reshape(self.ndim)

        if x_param_guess is None:
            x_param_guess = np.zeros_like(x_phys)

        # function that returns the relative distance (in physical units) to
        # the desired point
        def delta_x_phys(x_param):
            # Ensure we stay within the element domain
            return self(x_param) - x_phys

        def eval_Jacobian(x_param):
            return self._basis.interpolate(self.J, x_param)

        x_param = sem.rootfind.newton(
            delta_x_phys, x_param_guess, eval_Jacobian, it_max=8, tol=1e-8)

        # Make sure the point is within the element
        if (x_param >= -1.).all() and (x_param <= 1.).all():
            return x_param
        raise OutsideDomain("Given physical point is not in the parametric "
                            "domain of the finite element.")

    def get_submapping(self, face):
        return SubMapping(self, face)


class SubMapping(Mapping):

    def __init__(self, parent_mapping, face):
        self._face = face
        self._parent_mapping = pmap = parent_mapping
        self._basis = pmap._basis.get_subbasis(face // 2)
        self._cell = pmap._cell.sub_cell(face)
        self._cmpflags = pmap._cmpflags.copy()
        if pmap._cmpflags['Jacobian']:
            self._normal_vec = self._compute_normal_vec()
            self._cmpflags['normal'] = True

    def _compute_normal_vec(self):
        # NOTE: this computes the surface Jacobian
        tangent_vecs = self._tangents()
        if self.ndim == 1:
            # simply compute the (outward) facing perpendicular vector, which
            # is trivial in 2-D
            dS = np.roll(tangent_vecs, 1, axis=0)
            dS[1] *= -1
        elif self.ndim == 2:
            np.cross(tangent_vecs[0], tangent_vecs[1], axis=0)
        else:
            # TODO: could generalize to higher dimensions using the
            # wedge/exterior product.
            raise NotImplementedError("only 1D and 2D sub-elements are "
                                      "supported.")
        return dS

    @property
    def x_phys(self):
        pmap = self._parent_mapping
        return _subface_slice(self._face, pmap.x_phys, pmap.ndim)

    @property
    def J(self):
        pmap = self._parent_mapping
        return _subface_slice(self._face, pmap.J, pmap.ndim)

    @property
    def invJ(self):
        pmap = self._parent_mapping
        return _subface_slice(self._face, pmap.invJ, pmap.ndim)

    @property
    def detJ(self):
        pmap = self._parent_mapping
        return _subface_slice(self._face, pmap.detJ, pmap.ndim)

    def _tangents(self):
        """Computes the orthogonal tangent vectors of the SubElement (not
        normalized)."""
        pmap = self._parent_mapping
        face = self._face
        # get tangent vectors from the parent element
        par_J = _subface_slice(face, pmap.J, pmap.ndim)

        # take only the tangent components of the Jacobian
        ax = face // 2
        ax_pos = bool(face % 2)
        if ax_pos:
            cols = list(range(ax + 1, pmap.ndim)) + list(range(0, ax))
        else:
            cols = list(range(ax - 1, -1, -1)) + list(range(pmap.ndim - 1, ax, -1))

        if self.ndim == 1:
            # orient tangent vector to 2-D element counter-clockwise
            tangent_vecs = par_J[:, cols[0]].copy()
            if face in (0, 3):
                tangent_vecs *= -1
        else:
            tangent_vecs = par_J[:, cols]
        return tangent_vecs

    @property
    def n_dS(self):
        return self._normal_vec

    @property
    def dS(self):
        return np.linalg.norm(self.n_dS, axis=0)

    @property
    def unit_normal(self):
        return self.n_dS / self.dS

    def inv(self, x_phys):
        raise NotImplementedError("Cannot compute the parametric coordinates"
                                  "of a SubMapping from physical coordinates.")
