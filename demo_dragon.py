from firedrake import Mesh, FunctionSpace

from manifold_matern.cli import argument_parser
from manifold_matern.manifold_matern_gp import ManifoldMaternGP
from manifold_matern.laplace_eigenvalues import get_eigenpairs

from manifold_matern.utils import export_fun, construct_mesh_graph, \
    convert_to_firedrake_function

import numpy as np
import os


def construct_ground_truth(mesh):
    import networkx as nx

    mesh_graph = construct_mesh_graph(mesh)

    geodesics = nx.shortest_path_length(mesh_graph, source=0, weight='weight')

    N = mesh.num_vertices()

    ground_truth = np.zeros((N))
    period = 2*np.pi / 0.3 * 2
    for i in range(N):
        ground_truth[i] = 2 * np.sin(geodesics.get(i) * period + 0.3)

    return ground_truth


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()

    mesh = Mesh('meshes/dragon_connected.msh', dim=3)
    num_eigenpairs = args.num_eigenpairs

    print('Constructing ground truth. It may take a while')
    ground_truth = construct_ground_truth(mesh)

    mayavi_installed = False
    if args.mayavi:
        try:
            import manifold_matern.plotting
            mayavi_installed = True
        except ImportError:
            import warnings
            warnings.warn('Mayavi does not seem to be installed.\n'
                          'No mayavi-based plotting will occur.',
                          RuntimeWarning)

    V = FunctionSpace(mesh, "Lagrange", 1)

    if args.eigenpairs_file is None:
        print('Getting eigenpairs. It may take a while')
        eigenpairs = get_eigenpairs(mesh, V, num_eigvals=num_eigenpairs,
                                    eps_target=1e-10)
    else:
        print('Reading eigenpairs from a file')
        eigenpairs_ = np.load(args.eigenpairs_file)
        eigenpairs = (eigenpairs_[:, 0],
                      eigenpairs_[:, 1:])

    X = [3316, 10960, 24947, 33498, 33513, 34844, 35740, 35830, 36026,
         36654, 36727, 37597, 39066, 39886, 40050, 41165, 41387, 41430,
         42420, 43012, 43383, 45296, 45796, 47431, 48346, 48779, 49033,
         49127, 50374, 50878, 51132, 51132, 51168, 51175, 55643, 55863,
         56414, 56568, 56933, 57130, 57168, 57536, 60028, 61432, 61915,
         62936, 63657, 64280, 64542, 75551, 87643, 95423]
    Y = ground_truth[X]

    gp = ManifoldMaternGP(mesh, V, X, Y, eigenpairs)
    gp.kappa.constrain_bounded(0.01, 5.0)
    gp.sigma_n.fix(1e-15)
    gp.sigma_f.constrain_bounded(1., 100000.0)

    if args.seed is not None:
        print('Setting seed %d' % args.seed)
        np.random.seed(args.seed)

    print('Optimizing MLL. It may take a while')
    gp.optimize_restarts(num_restarts=5)

    print('Optimization finished. Proceeding to output')
    mean, variance = gp.predict()
    std = np.nan_to_num(np.sqrt(variance))
    samples = gp.posterior_samples(args.num_samples)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    export_fun(os.path.join(output_dir, '0_groud_truth.pvd'),
               convert_to_firedrake_function(V, ground_truth))
    export_fun(os.path.join(output_dir, '0_mean.pvd'),
               convert_to_firedrake_function(V, mean))
    export_fun(os.path.join(output_dir, '0_std.pvd'),
               convert_to_firedrake_function(V, std))

    for s in range(len(samples)):
        export_fun(os.path.join(output_dir, '0_sample_%d.pvd' % s),
                   convert_to_firedrake_function(V, samples[s]))

    if args.mayavi and mayavi_installed:
        from manifold_matern.plotting import plot_example
        vmin, vmax = -4., 4.
        plot_example(os.path.join(output_dir, '1_ground_truth.png'),
                     mesh, V, ground_truth, X, vmin, vmax)
        plot_example(os.path.join(output_dir, '1_mean.png'),
                     mesh, V, mean, X, vmin, vmax)
        plot_example(os.path.join(output_dir, '1_std.png'),
                     mesh, V, std, X, std.min(), std.max(), colormap='viridis')
        for s in range(len(samples)):
            plot_example(os.path.join(output_dir, '1_sample_%d.png' % s),
                         mesh, V, samples[s], X, vmin, vmax)
