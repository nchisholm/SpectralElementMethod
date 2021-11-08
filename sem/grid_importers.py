#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from . import discrete
from . import geometry as geo


class FileFormatError(Exception):
    """Exception to be raised if an error occurs while reading a mesh file.
    """
    pass


# Map of Gmsh element type (integer) specifiers to their correspnoding
# geometries.  The keys are integers specifing the element type and the values
# are functions that construct appropriate geometry objects when called.  This
# way, we may construct only the geometries actually present on a given mesh.
construct_geometry = {
    # Line cells
    1:  lambda: geo.Line(2),
    8:  lambda: geo.Line(3),
    26: lambda: geo.Line(4),
    27: lambda: geo.Line(5),
    28: lambda: geo.Line(6),
    62: lambda: geo.Line(7),
    63: lambda: geo.Line(8),
    64: lambda: geo.Line(9),
    65: lambda: geo.Line(10),
    66: lambda: geo.Line(11),
    # Quadrilateral cells
    3:  lambda: geo.Quadrilateral(2, 2),
    10: lambda: geo.Quadrilateral(3, 3),
    36: lambda: geo.Quadrilateral(4, 4),
    37: lambda: geo.Quadrilateral(5, 5),
    38: lambda: geo.Quadrilateral(6, 6),
    47: lambda: geo.Quadrilateral(7, 7),
    48: lambda: geo.Quadrilateral(8, 8),
    49: lambda: geo.Quadrilateral(9, 9),
    50: lambda: geo.Quadrilateral(10, 10),
    51: lambda: geo.Quadrilateral(11, 11)
}


def load_msh(file_path, ndim):
    # Handle the metadata in the file header
    with open(file_path, 'rb') as f:
        # Determine if the remainder of the file is binary or ASCII
        # and the position just past the header data of the file.
        is_binary, fpos = parse_format(f)

    mesh = discrete.Mesh(ndim)
    # make a (temporary) mesh to store boundary cells
    bnd_mesh = discrete.Mesh(ndim)
    # Re-open the file in the correct mode and read in the mesh data
    with open(file_path, 'rbU' if is_binary else 'rU') as f:
        f.seek(fpos)
        # mesh.set_phys_ids(parse_physical_names(f))
        region_id_map, boundary_id_map = parse_physical_names(f, mesh, bnd_mesh)
        if is_binary:
            parse_nodes_bin(f, mesh, bnd_mesh)
            parse_elements_bin(f, mesh, bnd_mesh, region_id_map, boundary_id_map)
            find_cell_neighbors(mesh, bnd_mesh)
        else:
            raise NotImplementedError(
                "Reading ASCII *.msh files is not yet supported. Save the "
                "mesh in binary format and try again.")
    return mesh


def parse_format(f):
    """
    Determine the format of a mesh file.
    """
    # Check for beginning of "$MeshFormat" section in the file
    if not f.readline().startswith(b"$MeshFormat"):
        raise FileFormatError("Expected 'MeshFormat' data")

    # Read metadata and make sure it makes sense
    version, is_binary, data_size = f.readline().split()
    if version != b'2.2':
        raise FileFormatError("Expected Gmsh file format version 2.2, but"
                              "got {} instead".format(version.decode('utf-8')))
    if is_binary not in [b'0', b'1']:
        raise FileFormatError("Unable to recognize file format")
    else:
        is_binary = bool(int(is_binary))
    if data_size != b'8':
        raise FileFormatError("Expected a data size of 8, but got"
                              "{} instead".format(data_size.decode('utf-8')))

    if is_binary:
        binary_one = f.readline().rstrip()
        # TODO: `binary_one` may be used to determine file endianness
        # although in practice its always little as far as I can tell.

    # Check for the end of the mesh format section
    if not f.readline().startswith(b"$EndMeshFormat"):
        raise FileFormatError("Malformed mesh format specification")

    return is_binary, f.tell()


def parse_physical_names(f, mesh, bnd_mesh):
    # Parse the "physical names" in the mesh, which are associated with
    # regions (with a certain number of dimensions) within the mesh.

    if not f.readline().startswith(b"$PhysicalNames"):
        raise FileFormatError("Expected 'PhysicalNames' data")

    n_phys_names = int(f.readline().rstrip())
    region_id_map = {}
    boundary_id_map = {}
    for i in range(n_phys_names):
        line = f.readline()
        # Parse data from string
        str_data = line.split()
        ndim = int(str_data[0])
        phys_id = int(str_data[1]) - 1  # (-1) for 0-based index
        assert phys_id == i     # Make sure the numbering is consecutive
        phys_name = str_data[2].strip(b'"').decode('utf-8')
        if ndim == mesh.ndim:
            region_id = mesh.new_region(phys_name)
            region_id_map[phys_id] = region_id
        elif ndim < mesh.ndim:
            boundary_id = bnd_mesh.new_region(phys_name)
            boundary_id_map[phys_id] = boundary_id
            mesh.new_boundary(phys_name)

    if not f.readline().startswith(b"$EndPhysicalNames"):
        raise FileFormatError("Wrong number of physical names specifed")

    return region_id_map, boundary_id_map


def parse_nodes_bin(f, mesh, bnd_mesh):
    # Load the node coordinates into a numpy array.

    if not f.readline().startswith(b"$Nodes"):
        raise FileFormatError("Expected 'Nodes' data")

    n_nodes = int(f.readline().rstrip())
    dt = np.dtype([('index', 'i4'), ("coord", '3f8')])
    nodes_in = np.fromfile(f, dt, count=n_nodes)
    # self.mesh.set_nodes(nodes_in['coord'].T[:self.mesh.ndim].copy())

    f.readline()  # advance past extra newline

    if not f.readline().startswith(b"$EndNodes"):
        raise FileFormatError("Expected end of 'Nodes' data")
    # Should have consecutively indexed nodes
    assert np.all(nodes_in['index'] == np.arange(1, n_nodes+1))

    nodes = nodes_in["coord"][:, :mesh.ndim].T
    mesh.set_nodes(nodes)
    bnd_mesh.set_nodes(nodes)


def parse_elements_bin(f, mesh, bnd_mesh, region_id_map, boundary_id_map):
    # Load global node indices associated with each element

    if not f.readline().startswith(b"$Elements"):
        raise FileFormatError("Expected 'Elements' data")

    n_elems = int(f.readline().rstrip())
    n_elem_read = 0  # Track number of elements read from file

    # Map (gmsh elem_type #) -> (geometry object)
    geo_tbl = dict()

    while n_elem_read < n_elems:

        # Read element header data (a list of 3 integers)
        header = np.fromfile(f, dtype='i4', count=3)
        elem_type, n_elem_follow, n_tags = header

        # Fetch (or construct) the element geometry
        try:
            geometry = geo_tbl[elem_type]
            print("The try section worked out")
            print(geo_tbl[elem_type])
        except KeyError:
            print ("I am in KeyError section of grid_importer.py")
            print(elem_type)
            geometry = construct_geometry[elem_type]()
            geo_tbl[elem_type] = geometry
            if geometry.ndim == mesh.ndim:
                geometry_id = mesh.add_geometry(geometry)
            elif geometry.ndim < mesh.ndim:
                geometry_id = bnd_mesh.add_geometry(geometry)

        n_nodes = geometry.n_nodes

        # Read the index, tags, and global node indicies for elements (the
        # number of which is given by `n_elem_follow`)
        dt = np.dtype([('index', 'u4'),
                       ('tags', 'u4', n_tags),
                       ('node_ix', 'u4', n_nodes)])
        elem_data = np.fromfile(f, dt, n_elem_follow)
        assert np.all(elem_data['index'] == np.arange(
            n_elem_read + 1, n_elem_read + n_elem_follow + 1))

        # convert indexing from 1-based to 0-based
        elem_data['node_ix'] -= 1

        for i, tags, node_ix in elem_data:
            _convert_ix_order_to_lexicographic(geometry.shape, node_ix)
            phys_id = tags[0] - 1
            if geometry.ndim == mesh.ndim:
                region_id = region_id_map[phys_id]
                mesh.add_cell(node_ix, geometry_id, region_id)
            elif geometry.ndim < mesh.ndim:
                # got a boundary element
                boundary_id = boundary_id_map[phys_id]
                bnd_mesh.add_cell(node_ix, geometry_id, boundary_id)

        n_elem_read += n_elem_follow

    f.readline()  # Advance past extra newline

    if not f.readline().startswith(b"$EndElements"):
        raise FileFormatError("Expected 'Elements' data")


def find_cell_neighbors(mesh, bnd_mesh):
    # Get centroids of all cells
    cell_centroids = np.empty((mesh.ndim, mesh.n_cells))
    bnd_cell_centroids = np.empty((mesh.ndim, bnd_mesh.n_cells))
    for i, cell in enumerate(mesh.cells):
        nodes = cell.nodes_lexicographic.reshape(mesh.ndim, -1)
        cell_centroids[:, i] = np.mean(nodes, axis=1)
    for i, cell in enumerate(bnd_mesh.cells):
        nodes = cell.nodes_lexicographic.reshape(mesh.ndim, -1)
        bnd_cell_centroids[:, i] = np.mean(nodes, axis=1)

    all_centroids = np.hstack((bnd_cell_centroids, cell_centroids))
    # track which centroids belong to boundary cells
    cell_is_bnd0 = np.zeros(
        bnd_cell_centroids.shape[1] + cell_centroids.shape[1], dtype=bool)
    cell_is_bnd0[:bnd_mesh.n_cells] = True
    for i, cell in enumerate(mesh.cells):
        # Compute distances to all other (boundary) cells
        this_centroid = cell_centroids[:, i][:, None]
        dist = np.linalg.norm(cell_centroids - this_centroid, axis=0)
        dist_b = np.linalg.norm(bnd_cell_centroids - this_centroid, axis=0)
        closest = np.argsort(dist)[1:]
        closest_b = np.argsort(dist_b)
        cell_is_bnd = cell_is_bnd0[closest]
        # Starting with the closest distinct cell, find all adjacent
        # cells.
        n_adj_cells = cell.geometry.n_sub_geometries()
        n_adj_cells_found = 0
        j1 = 0
        j2 = 0
        while n_adj_cells_found < n_adj_cells:
            dist_to_next_closest_bnd_cell = dist_b[closest_b[j1]]
            dist_to_next_closest_cell = dist[closest[j2]]
            if dist_to_next_closest_bnd_cell < dist_to_next_closest_cell:
                bnd_cell_num = closest_b[j1]
                nearby_cell = bnd_mesh.get_cell(bnd_cell_num)
                side = mesh.cells_are_neighbors(cell, nearby_cell)
                if side >= 0:
                    mesh.add_boundary_cell(i, nearby_cell.region_id,
                                              nearby_cell.ndim, side)
                    n_adj_cells_found += 1
                j1 += 1
            else:
                cell_num = closest[j2]
                nearby_cell = mesh.get_cell(cell_num)
                side = mesh.cells_are_neighbors(cell, nearby_cell)
                if side >= 0:
                    mesh._adj_map[i][side] = cell_num
                    n_adj_cells_found += 1
                j2 += 1


def _convert_ix_order_to_lexicographic(shape, global_indicies):
    '''
    Converts global node indicies of a cell into a lexicographic order.
    '''

    # For now, this only works on 1D or 2D cells
    if len(shape) == 0:
        return np.array(0)
    elif len(shape) == 1:
        M, N = shape[0], 1
    elif len(shape) == 2:
        M, N = shape
    else:
        raise NotImplementedError("Can only take 2 arguments for now...")
    idxmap = np.zeros((M, N), dtype=int)

    # Nodes are labeled with verticies first, followed by edges in a
    # counterclockwise fashion.  Interior nodes are labeled recursively
    # in the same manner.
    k = 0                       # Count number of nodes labeled
    l = 0                       # Keep track of the recursion number
    while l < min(M, N)//2:
        # Label verticies (corners)
        corners = ([l, -l-1, -l-1, l], [l, l, -l-1, -l-1])
        idxmap[corners] = np.arange(k, k+4)
        k += 4
        # south edge (increasing indicies)
        p_ns = M-2*(l+1)
        idxmap[l+1:-l-1, l] = np.arange(k, k+p_ns)
        k += p_ns
        # east edge (increasing indicies)
        p_ew = N-2*(l+1)
        idxmap[-l-1, l+1:-l-1] = np.arange(k, k+p_ew)
        k += p_ew
        # north edge (decreasing indicies)
        idxmap[l+1:-l-1, -l-1] = np.arange(k+p_ns-1, k-1, -1)
        k += p_ns
        # west edge (decreasing indicies)
        idxmap[l, l+1:-l-1] = np.arange(k+p_ew-1, k-1, -1)
        k += p_ew
        l += 1

    # We are now left with some funky logic to deal with numbering nodes
    # that lie in the center of the cell when either M or N are odd.
    if (M % 2 or N % 2) and (min(M, N) != 2):
        if M > N:  # line of nodes on horizontal centerline
            idxmap[[l, -l-1], [l, l]] = np.arange(k, k+2)
            k += 2
            idxmap[l+1:-l-1, l] = np.arange(k, M*N)
        elif M < N:  # line of nodes on vertical centerline
            idxmap[[l, l], [l, -l-1]] = np.arange(k, k+2)
            k += 2
            idxmap[l, l+1:-l-1] = np.arange(k, M*N)
        else:  # M = N ... single node at center of cell
            idxmap[l, l] = M*N - 1

    idxmap = idxmap.squeeze()
    global_indicies_old = global_indicies.copy()
    global_indicies.shape = idxmap.shape
    global_indicies[:] = global_indicies_old[idxmap]
