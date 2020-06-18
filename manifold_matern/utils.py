""" Utility functions. """


import autograd.numpy as np
from autograd.numpy.linalg import cholesky
import warnings
from firedrake import File, Function
from scipy.sparse import coo_matrix
import networkx as nx


def jitchol(mat, jitter=0):
    """Run Cholesky decomposition with an increasing jitter,
    until the jitter becomes too large.

    Arguments
    ---------
    mat : (m, m) np.ndarray
        Positive-definite matrix
    jitter : float
        Initial jitter
    """
    try:
        chol = cholesky(mat)
        return chol
    except np.linalg.LinAlgError:
        new_jitter = jitter*10.0 if jitter > 0.0 else 1e-15
        if new_jitter > 1.0:
            raise RuntimeError('Matrix not positive definite even with jitter')
        warnings.warn(
            'Matrix not positive-definite, adding jitter {:e}'
            .format(new_jitter),
            RuntimeWarning)
        return jitchol(mat + new_jitter * np.eye(mat.shape[-1]), new_jitter)


def export_fun(fname, *funs):
    """
    Export a list of functions to a VTK file (.pvd).

    Arguments
    ---------
    fname : str
        Filename to export to
    *funs : firedrake.Function
        Any number of functions to export
    """
    outfile = File(fname)
    outfile.write(*funs)


def convert_to_firedrake_function(function_space, raw_data):
    """
    Convert raw function values to `firedrake.Function` object.

    Arguments
    ---------
    function_space : firedrake.FunctionSpace
    raw_data : np.ndarray

    Returns:
    fun : firedrake.Function
    """
    fun = Function(function_space)
    fun.vector()[:] = raw_data
    return fun


def mesh_triangulation(mesh):
    """
    Return vertex coordinates and triangles that constitute a mesh.

    A triangle is represented as a 3-tuple with vertex indices that lie in
    the given triangle.

    Arguments
    ---------
    mesh : firedrake.Mesh

    Returns
    -------
    coordinates : np.ndarray
    triangles : np.ndarray
    """
    assert mesh.ufl_cell().cellname() == 'triangle', \
        "Only triangular meshes are supported"

    coordinates = mesh.coordinates.dat.data_ro
    cell_node_map = mesh.coordinates.cell_node_map().values

    idx = (0, 1, 2)

    triangles = cell_node_map[:, idx]

    return coordinates, triangles


def rescale_eigenfunctions(eigenfunctions, scale_factor=1.0):
    """ Scale eigenfunctions by a `scale_factor` """
    return scale_factor * eigenfunctions


def construct_mesh_graph(mesh):
    """
    Construct a graph given a mesh.
    Vertices of the graphs are mesh vertices.
    Two vertices are connected by an edge if there is a triangle in the mesh
    that these vertices lie in.

    Arguments
    ---------
    mesh : firedrake.Mesh

    Returns
    -------
    G : networkx.Graph
    """
    import numpy
    coordinates = mesh.coordinates.dat.data_ro
    cell_node_map = mesh.coordinates.cell_node_map().values

    data = numpy.zeros(6 * len(cell_node_map))
    rows = numpy.zeros_like(data)
    cols = numpy.zeros_like(data)

    for i, tr in enumerate(cell_node_map):
        # TODO: refactor that
        node0 = coordinates[tr[0]]
        node1 = coordinates[tr[1]]
        node2 = coordinates[tr[2]]
        data[6*i] = data[6*i+3] = numpy.sqrt(numpy.sum((node0 - node1)**2))
        data[6*i+1] = data[6*i+4] = numpy.sqrt(numpy.sum((node0 - node2)**2))
        data[6*i+2] = data[6*i+5] = numpy.sqrt(numpy.sum((node1 - node2)**2))

        rows[6*i] = cols[6*i+3] = tr[0]
        cols[6*i] = rows[6*i+3] = tr[1]

        rows[6*i+1] = cols[6*i+4] = tr[0]
        cols[6*i+1] = rows[6*i+4] = tr[2]

        rows[6*i+2] = cols[6*i+5] = tr[1]
        cols[6*i+2] = rows[6*i+5] = tr[2]

    dst = coo_matrix((data, (rows.astype(numpy.int), cols.astype(numpy.int))))
    G = nx.Graph(dst)

    return G


def check_mesh_connected(mesh, graph=None):
    """ Check if the mesh represents a connected manifold. """
    if graph is None:
        graph = construct_mesh_graph(mesh)

    return nx.is_connected(graph)
