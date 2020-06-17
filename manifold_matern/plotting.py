from mayavi import mlab
from firedrake import Function
from .utils import mesh_triangulation


def plot_function_3d(mesh, function=None, **kwargs):
    assert mesh.topological_dimension() == 2, \
        "Only 2d meshes are supported"

    coordinates, triangles = mesh_triangulation(mesh)

    x, y, z = (coordinates[:, i] for i in range(3))

    scalars = None
    if function is not None:
        scalars = function.vector()[:]

    mlab.triangular_mesh(x, y, z, triangles, scalars=scalars, **kwargs)


def plot_function_3d_with_data(mesh, function, vertices,
                               v_options={}, **kwargs):

    plot_function_3d(mesh, function, **kwargs)
    coordinates = mesh.coordinates.dat.data_ro

    mlab.points3d(coordinates[vertices, 0],
                  coordinates[vertices, 1],
                  coordinates[vertices, 2],
                  **v_options)


def set_camera():
    mlab.gcf().scene.z_minus_view()
    mlab.gcf().scene.camera.position = [-0.0058909000000000045, 0.12453400410783291, -0.35696672385642464]
    mlab.gcf().scene.camera.focal_point = [-0.0058909000000000045, 0.12453400410783291, -0.00460435]
    mlab.gcf().scene.camera.view_angle = 30.0
    mlab.gcf().scene.camera.view_up = [0.0, 1.0, 0.0]
    mlab.gcf().scene.camera.clipping_range = [0.2578140606178604, 0.472229139714271]
    mlab.gcf().scene.camera.compute_view_plane_normal()
    mlab.gcf().scene.render()


def plot_example(filename,
                 mesh, V, function_values, X, vmin, vmax, colormap='plasma'):
    fn = Function(V)
    fn.vector()[:] = function_values
    v_options = {'mode': 'sphere',
                 'scale_factor': 3e-3, }

    fig = mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    plot_function_3d_with_data(mesh, fn, X,
                               v_options=v_options,
                               vmin=vmin, vmax=vmax,
                               colormap=colormap,
                               opacity=1)
    set_camera()
    mlab.savefig(filename, magnification=5)
