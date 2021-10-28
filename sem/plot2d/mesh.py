# -*- coding: utf-8 -*-

"""
Tools for visualizing 2-dimensional meshes
"""

import itertools as it
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


class PlottingError(Exception):
    pass


def triangulate(mesh):
    """Convert a mesh to a matplotlib Triangulation object useful for making 2D
    plots.

    Parameters
    ----------
    mesh : discrete.Mesh
        Mesh to triangulate
    """

    def local_triangles(geo):
        # Compute local triangles on quadrilateral element
        n_loc_tri = 2*(geo.shape[0]-1)*(geo.shape[1]-1)
        loc_tri = np.zeros((n_loc_tri, 3), dtype=np.uint32)
        n = 0
        for i, j in it.product(range(geo.shape[0]-1),
                               range(geo.shape[1]-1)):
            # Make two counter clockwise triangles each "square" formed by four
            # adjacent nodes within a master element.
            loc_tri[n] = np.ravel_multi_index(
                [[i, i+1, i], [j, j+1, j+1]], geo.shape)
            n += 1
            loc_tri[n] = np.ravel_multi_index(
                [[i, i+1, i+1], [j, j, j+1]], geo.shape)
            n += 1
        return loc_tri

    # Count the total number of triangles in the triangulation.
    n_tri = 0
    for cell in mesh.cells:
        geo = cell.geometry
        n_tri += 2 * (geo.shape[0] - 1) * (geo.shape[1] - 1)
    # Allocate an array to store the global list of triangles
    tri = np.zeros((n_tri, 3), dtype=np.uint32)

    # Generate a "mesh" of trinagles from the mesh of high-order elements.
    i0 = i1 = 0
    local_tris = {geo: local_triangles(geo) for geo in mesh.get_geometries()}
    for cell in mesh.cells:
        node_ind = cell.node_ind_lexicographic.ravel()
        loc_tri = local_tris[cell.geometry]
        i1 = i0 + len(loc_tri)
        tri[i0:i1] = node_ind[loc_tri]
        i0 = i1

    x, y = mesh.nodes
    return mpl.tri.Triangulation(x, y, tri)


def draw_nodes(mesh, marker='.', show_indices=False, ax=None):
    """Plots the nodes on a mesh.
    """

    if mesh.ndim != 2:
        raise PlottingError("A 2D mesh is required")

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    x, y = mesh.nodes
    ax.plot(x, y, marker)
    # Label the nodes by their index
    if show_indices:
        for i in range(mesh.n_nodes):
            ax.text(x[i], y[i], str(i))

    ax.axis('scaled')


def draw_cell(cell, draw_param_axes=False, ax=None):
    """Plots the nodes in a cell.
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    # draw the polygon tracing the edge of the cell

    poly_coords = np.zeros((cell.n_exterior_nodes, 2))
    vtx_indices = [(slice(None),) + ix_exp for (shape, ix_exp) in
                   cell.geometry.sub_geometry_ix_exps(0, False)]
    edge_indices = [(slice(None),) + ix_exp for (shape, ix_exp) in
                    cell.geometry.sub_geometry_ix_exps(1, False)]

    i0 = 0
    i1 = 0
    vtx_order = [0, 1, 3, 2]
    edge_order = [0, 3, 1, 2]
    rev_order = [False, False, True, True]
    for vtx_num, edge_num, rev in zip(vtx_order, edge_order, rev_order):
        # add vertex point
        i1 += 1
        poly_coords[i0:i1] = cell.nodes_lexicographic[vtx_indices[vtx_num]]
        i0 = i1
        # add edge points
        edge_pts = cell.nodes_lexicographic[edge_indices[edge_num]]
        i1 += edge_pts.shape[1]
        if rev:
            poly_coords[i0:i1] = edge_pts.T[::-1]
        else:
            poly_coords[i0:i1] = edge_pts.T
        i0 = i1

    ax.add_patch(plt.Polygon(poly_coords, fill=False))

    if draw_param_axes:
        # Plot arrows showing local coordinate directions
        vtx = cell.vertex_nodes
        dxi = vtx[:, 2] - vtx[:, 0]
        deta = vtx[:, 1] - vtx[:, 0]
        # draw axes offset from bottom-left corner of cell
        offset_x = (dxi[0] + deta[0])*0.1
        offset_y = (dxi[1] + deta[1])*0.1
        # relative length of axes
        axlen = 0.2
        x, y = vtx
        ax.arrow(x[0] + offset_x, y[0] + offset_y,
                 dxi[0]*axlen, dxi[1]*axlen,
                 fc='b', ec='b')
        ax.arrow(x[0] + offset_x, y[0] + offset_y,
                 deta[0]*axlen, deta[1]*axlen,
                 fc='g', ec='g')
        # TODO: scale arrowhead size with element size

        # arrow0_start = (x[0, 0] + (dxi[0] + deta[0])*0.1,
        #                 x[1, 0] + (dxi[1] + deta[1])*0.1)
        # arrow0_end = (arrow0_start[0] + dxi[0]*0.2,
        #               arrow0_start[1] + dxi[1]*0.2)

        # arrow1_start = (x[0, 0] + (dxi[0] + deta[0])*0.1,
        #                 x[1, 0] + (dxi[1] + deta[1])*0.1)
        # arrow1_end = (arrow1_start[0] + deta[0]*0.2,
        #               arrow1_start[1] + deta[1]*0.2)

        # ax.annotate("", xy=arrow0_end, xytext=arrow0_start,
        #             arrowprops=dict(fc='b', ec='b'))
        # ax.annotate("", xy=arrow1_end, xytext=arrow1_start,
        #             arrowprops=dict(fc='g', ec='g'))


def draw_cell_nodes(cell, global_indices=False, local_indices=False,
                    hierarchcal_order=False, ax=None):
    """Draw the nodes in a cell and optionally the global and/or local indices
    of each node.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    if hierarchcal_order:
        node_ind = cell.node_ind_hierarchical.ravel()
        x, y = cell.nodes_hierarchical.reshape(2, -1)
    else:
        node_ind = cell.node_ind_lexicographic.ravel()
        x, y = cell.nodes_lexicographic.reshape(2, -1)
    ax.plot(x, y, '.')
    if not (global_indices or local_indices):
        return
    for i in range(node_ind.size):
        if local_indices and global_indices:
            ax.text(x[i], y[i], "{0}|{1}".format(i, node_ind[i]))
        elif local_indices:
            ax.text(x[i], y[i], "{}".format(i))
        else:   # only global indices
            ax.text(x[i], y[i], "{}".format(node_ind[i]))


def draw_cells(mesh, draw_nums=False, draw_param_axes=False, ax=None):
    """Plots the cells (elements) in a mesh.
    """

    if mesh.ndim != 2:
        raise PlottingError("A 2D mesh is required")

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    for num, cell in enumerate(mesh.cells):
        draw_cell(cell, draw_param_axes=draw_param_axes, ax=ax)
        if draw_nums:
            x_lbl, y_lbl = np.mean(cell.vertex_nodes, axis=1)
            ax.text(x_lbl, y_lbl, str(num), ha='center', va='center')

    ax.axis('scaled')


def add_arrow_to_line(line, position=None, reverse=False, size=15, color=None):
    """Add an arrow to a line.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()

    start_ix = np.argmin(np.abs(xdata - position))
    if reverse:
        end_ix = start_ix + 1
    else:
        end_ix = start_ix - 1

    xy_start = (xdata[start_ix], ydata[start_ix])
    xy_end = (xdata[end_ix], ydata[end_ix])
    line.axes.annotate('', xytext=xy_start, xy=xy_end,
                       arrowprops=dict(arrowstyle='->', color=color),
                       size=size)
