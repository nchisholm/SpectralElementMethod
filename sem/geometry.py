#!/usr/bin/env python
# encoding: utf-8

"""
Classes describing the support geometries of elements.
"""

import itertools as itt
import numpy as np
import scipy.special as sf


class Geometry(object):
    """
    Describes the geometrical properties of elements important to FEM methods.
    """
    pass


class Simplex(Geometry):
    """
    Describes geometrical properties of a simplex-shaped element (e.g.
    line-segments, triangles, tetrahedra).
    """
    # This class may be implemented in the future to work with
    # triangular/tetrahedral elements.

    def __init__(self):
        raise NotImplementedError()


class NCube(Geometry):
    """
    Describes geometrical properties of an orthotope-shaped element (e.g.
    line-segments, rectangles, hexahedra).  Specific element geometries are
    defined in subclasses.
    """
    # The following properties must be defined by the sub-classes:
    #
    # (int) ndim: number of dimensions
    #
    # (dict) corner_verts: maps sides to verticies.
    # Keys label each side  with a number for each direction.  The sign of the
    # number designates whether the side is oriented in the positive or
    # negative direction.  The values are a list of numbered corner verticies
    # lying on each side.  The special key 0 is assigned to all corner
    # verticies in ascending order.  Diagrams are given in the sub-classes.

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def shape(self):
        return self._shape

    @property
    def n_nodes(self):
        return self._n_nodes

    @property
    def n_exterior_nodes(self):
        return self._n_exterior_nodes

    @property
    def n_interior_nodes(self):
        return self._n_interior_nodes

    @property
    def vertex_node_ind(self):
        return self._hier_node_order[:2**self.ndim]

    @property
    def hierarchical_node_order(self):
        return self._hier_node_order

    @property
    def exterior_node_ind(self):
        return self._hier_node_order[:self._n_exterior_nodes]

    @property
    def interior_node_ind(self):
        return self._hier_node_order[self._n_exterior_nodes:]

    @property
    def nodes(self):
        return self._node_locations

    def __init__(self, *shape):
        assert all(isinstance(i, int) for i in shape)
        assert all(i > 0 for i in shape)
        self._shape = shape
        self._n_nodes = self._compute_n_nodes()
        self._n_interior_nodes = self._compute_n_interior_nodes()
        self._n_exterior_nodes = self._n_nodes - self._n_interior_nodes
        self._node_locations = np.meshgrid(
            *(np.linspace(-1, 1, s) for s in self.shape),
            indexing='ij', sparse=True)
        # Hierarchically grouped/ordered indices
        self._hier_node_order = self._compute_hierarchical_node_ordering()
        self._sub_geo_data = [self.sub_geometry_ix_exps(d) for d in
                              range(self.ndim+1)]
        self._sub_geo_class = NCube

    def _compute_n_nodes(self):
        """
        Return the total number of nodes within the element.
        """
        node_count = 1
        for s in self.shape:
            node_count *= s
        return node_count

    def _compute_n_interior_nodes(self):
        """
        Return the number of nodes on the element interior.
        """
        node_count = 1
        for s in self.shape:
            node_count *= s-2
        assert node_count >= 0
        return node_count

    def _compute_n_exterior_nodes(self):
        """
        Return the number of nodes on the element exterior.
        """
        return self.n_nodes - self.n_interior_nodes

    def n_sub_geometries(self, dim=-1):
        """
        Compute the number of `dim`-dimensional sub-geometries on the
        boundaries of a geometry object.

        Parameters
        ----------
        dim : int
            Number of dimensions of the sub-geometry.
        """
        if dim < 0:
            dim = self.ndim + dim
        if dim > self.ndim:
            raise ValueError("No {}D sub-geometry in a {}D parent geometry"
                             .format(dim, self.ndim))
        elif dim < 0:
            raise ValueError("Dimension of sub-elements must be > 0")

        n = self.ndim
        return 2**(n-dim) * sf.comb(n, dim, True)

    def sub_geometry_ix_exps(self, dim=None, inclusive=True):
        """
        Retrieve the local indices of the nodes on each ``dim``-dimensional
        sub-element of the geometry.  In this case, `element` refers to, e.g.
        the vertices, edges, and faces of a 3D cube.  Only interior nodes of
        each element are considered.  Each ``dim``-D element is ordered
        lexicographically.
        """
        # TODO: docstring needs an example to better explain things
        if dim is None:
            dim = self.ndim - 1
        if dim > self.ndim:
            raise ValueError("No {}D sub-geometry on a {}D parent geometry"
                             .format(dim, self.ndim))
        elif dim < 0:
            raise ValueError("Dimension of sub-elements must be > 0")

        n_free_axes = dim
        n_fixed_axes = self.ndim - dim

        sub_geo_data = []

        fixed_ax_combs = itt.combinations(range(self.ndim), n_fixed_axes)
        for fixed_axes in fixed_ax_combs:
            # get first and last indices of fixed axes
            iter_const_ax_ind = itt.product(
                *[(0, self.shape[ax] - 1) for ax in fixed_axes])
            for const_ax_ind in iter_const_ax_ind:
                indices = []
                shape = []
                n = 0
                for d in range(self.ndim):
                    if n < n_fixed_axes and d == fixed_axes[n]:
                        indices.append(const_ax_ind[n])
                        n += 1
                    else:
                        if inclusive is True:
                            indices.append(slice(0, self.shape[d]))
                            shape.append(self.shape[d])
                        else:
                            indices.append(slice(1, self.shape[d] - 1))
                            shape.append(self.shape[d] - 2)
                sub_geo_data.append((tuple(shape), tuple(indices)))

        return sub_geo_data

    def _compute_hierarchical_node_ordering(self):
        node_order = np.zeros(self.n_nodes, dtype=np.uint32)
        linear_ind_map = np.arange(self.n_nodes).reshape(self.shape)
        linear_indices = [linear_ind_map[ix_exp] for shape, ix_exp in
                          self.sub_geometry_ix_exps(0, False)]
        i0 = 0
        i1 = len(linear_indices)
        node_order[i0:i1] = linear_indices
        for d in range(1, self.ndim+1):
            linear_indices = [linear_ind_map[ix_exp] for shape, ix_exp in
                              self.sub_geometry_ix_exps(d, False)]
            for ind in linear_indices:
                i0 = i1
                i1 += ind.size
                node_order[i0:i1] = ind.ravel()
        return node_order

    def sub_geometry(self, axis):
        geo_shape = self.shape[axis+1:self.ndim] + self.shape[0:axis]
        return self._sub_geo_class(*geo_shape)


class Line(NCube):
    # Enumeration of sub-geometries
    # +-->u0  (0)--*--(1)

    @property
    def ndim(self):
        return 1
    
    corner_verts = [np.array([True, False]),
                    np.array([False, True])]

    def __init__(self, shape_u):
        NCube.__init__(self, shape_u)
        self._sub_geo_class = None

    def sub_geometry(self):
        raise NotImplementedError("The sub-geometry of a line is a single "
                                  "point, which is all not useful.")


class Quadrilateral(NCube):

    @property
    def ndim(self):
        return 2

    # Enumeration of vertex and edge sub-geometries
    #        1--(3)--3
    #        |       |
    # u1    (0)  *  (1)
    # |      |       |
    # +--u0  0--(2)--2

    corner_verts = [np.array([1, 1, 0, 0], dtype=bool),
                    np.array([0, 0, 1, 1], dtype=bool),
                    np.array([1, 0, 1, 0], dtype=bool),
                    np.array([0, 1, 0, 1], dtype=bool)]

    def __init__(self, shape_u, shape_v):
        NCube.__init__(self, shape_u, shape_v)
        self._sub_geo_class = Line

    # def sub_geometry(self, ind, dim=1):
    #     if 0 <= ind < 2:
    #         return Line(self.shape[0])
    #     elif 2 <= ind < 4:
    #         return Line(self.shape[1])
    #     else:
    #         raise ValueError("No sub-geometry with that index")
