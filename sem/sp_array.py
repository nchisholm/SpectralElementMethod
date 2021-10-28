# -*- coding: utf-8 -*-

"""
Module for sparse arrays.
"""

import numpy as np
import itertools as it


class KroneckerArray(object):
    """
    Holds sparse arrays defined by "diagonal" arrays (e.g. with Kronecker
    deltas).
    """

    def __init__(self, shape, *args, **kwargs):
        """
        KroneckerArray(shape, subarr0, kdmap0, subarr1, kdmap1, ...)
        Initialize a new KroneckerArray

        Parameters
        ----------
        shape : tuple of ints
            Shape of the array
        subarrN : array-like
            Sub-array forming a diagonal as specified with kdmapN
        kdmapN : tuple of ints
            Sequence of integers specifying how the axes of the KroneckerArray
            map to axes of the input argument
        dtype : type, optional
            Desired data type of the array. Sub-arrays will be 'upcasted' to
            match the given type, but not downcasted.

        Example
        -------
        TODO
        """
        self.dtype = kwargs.get("dtype", np.float)
        self.shape = shape
        # numpy arrays for storing the KroneckerArray "diagonals"
        self.data = []
        self.kdmap = []
        for subarr, axes_map in zip(args[::2], args[1::2]):
            self.add_diag(subarr, axes_map)

    @property
    def ndim(self):
        return len(self.shape)

    def add_diag(self, subarr, axes_map):
        """
        Add a diagonal to the array.

        Parameters
        ----------
        subarr : array-like
            Array giving the data on the diagonal.
        axes_map : list of ints
            Mapping of the axes in the KroneckerArray to the axes in `subarr`.
        """
        # Are all axes of the KroneckerArray are properly mapped to axes
        # of the input sub-array?
        assert len(axes_map) == self.ndim  # correct number of axes mapped?
        assert sorted(set(axes_map)) == list(range(subarr.ndim))
        # Is the shape of each axis mapped in correct?
        for i in range(self.ndim):
            assert self.shape[i] == subarr.shape[axes_map[i]]
        # Add the array as a new sub-array
        self.data.append(np.asarray(subarr, dtype=self.dtype))
        self.kdmap.append(axes_map)

    def dot_dense(self, array, axes):
        '''
        Dot product with a dense numpy array along the specified axes.
        '''
        assert len(array.shape) == len(axes)
        # Initialize the output KroneckerArray
        shape_out = tuple(self.shape[i] for i in range(self.ndim)
                          if i not in axes)
        out = KroneckerArray(shape=shape_out)

        for data, kdmap in zip(self.data, self.kdmap):
            # Number the axes of the [dense] data array
            data_axes = list(range(data.ndim))
            # Determine which axes to sum over according to Kronecker deltas in
            # the array
            mapped_axes = [kdmap[ax] for ax in axes]
            # Determine the "Kronecker deltas" in the output
            kdmap_out = [kdmap[ax] for ax in range(self.ndim)
                         if ax not in axes]
            # Collapse the axes number in the output array
            axes_out = sorted(set(kdmap_out))

            # Remove any skipped axes numbers in the output kdmap
            map_axes_out = dict(zip(axes_out, range(len(kdmap_out))))
            for i, ax in enumerate(kdmap_out):
                kdmap_out[i] = map_axes_out[ax]

            data_out = np.einsum(data, data_axes, array, mapped_axes, axes_out)
            out.add_diag(data_out, kdmap_out)
        return out

    def to_array(self):
        '''
        Convert to a dense numpy array.
        '''
        out = np.zeros(self.shape, dtype=self.dtype)
        for data, kdmap in zip(self.data, self.kdmap):
            ix_ogrid = np.ogrid[[slice(N) for N in data.shape]]
            dense_ix = tuple(ix_ogrid[i] for i in kdmap)
            out[dense_ix] += data
        return out
