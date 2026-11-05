#!/usr/bin/env python3
"""
Provide functionality to easily export scalar, vector or tensor dara to a VTU file.
"""

import numpy as np
import meshio

def save_points_to_vtu_file(points, filename, scalar_data=None, vector_data=None, tensor_data=None):
    """
    Saves point coordinates and associated data (scalars, vectors, tensors) to a VTU file.
    Designed for simplicity and compatibility with ParaView.
    """
    scalar_data = {} if scalar_data is None else scalar_data
    vector_data = {} if vector_data is None else vector_data
    tensor_data = {} if tensor_data is None else tensor_data
    if len(scalar_data) < 1 and len(vector_data) < 1 and len(tensor_data) < 1:
        raise ValueError("At least one of scalar_data, vector_data, or tensor_data needs to be provided")

    points = np.asarray(points, dtype=float)
    cells = [("vertex", np.arange(len(points), dtype=np.int64).reshape(-1, 1))]
    point_data = {}

    for name, scalar_array in scalar_data.items():
        scalar_array = np.asarray(scalar_array, dtype=float)
        assert scalar_array.shape == (len(points),)
        point_data[name] = scalar_array

    for name, tensor_array in tensor_data.items():
        tensor_array = np.asarray(tensor_array, dtype=float)
        assert tensor_array.shape == (len(points), 3, 3)
        # Store as 9-component point data, which ParaView can interpret as tensors.
        point_data[name] = tensor_array.reshape(len(points), 9)

    for name, vector_array in vector_data.items():
        vector_array = np.asarray(vector_array, dtype=float)
        assert vector_array.shape == (len(points), 3)
        point_data[name] = vector_array

    mesh = meshio.Mesh(points=points, cells=cells, point_data=point_data)
    meshio.write(filename, mesh, file_format="vtu")

def save_surface_to_vtu_file(nodes, triangles, filename):
    """Saves a triangle surface to a VTU file, e.g. the head model for ParaView."""
    mesh = meshio.Mesh(points=np.asarray(nodes, dtype=float), cells=[("triangle", np.asarray(triangles, dtype=np.int64))])
    meshio.write(filename, mesh, file_format="vtu")
