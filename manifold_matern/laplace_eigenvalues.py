import numpy as np
from firedrake import *
from firedrake.petsc import PETSc
try:
    from slepc4py import SLEPc
except ImportError:
    import sys
    warning("Unable to import SLEPc, eigenvalue computation" +
            "not possible (try firedrake-update --slepc)")
    sys.exit(0)


def get_eigenpairs(mesh, V, num_eigvals=100, eps_target=1.0):
    """
    Get eigenvalues and corresponding eigenfunctions of the Laplace-Beltrami

    `mesh` is a firedrake's Mesh
    `V` is firedrake's FunctionSpace

    Return two ndarrays:
    `eigenvalues` of shape (num_eigvals,) 
    `eigenfunctions` of shape (num_eigvals, num_mesh_nodes)
    """
    
    ### Define Laplace-Beltrami eigenproblem
    psi, phi = TestFunction(V), TrialFunction(V)
    L = inner(grad(phi), grad(psi)) * dx
    M = phi * psi * dx

    petsc_L = assemble(L).M.handle
    petsc_M = assemble(M).M.handle

    ### Run eigensolver
    opts = PETSc.Options()
    opts.setValue("eps_type", "arnoldi")
    opts.setValue('eps_gen_hermitian', None)
    opts.setValue("eps_tol", 1e-10)
    opts.setValue('eps_st_type', 'sinvert')
    opts.setValue('eps_target', eps_target)

    st = SLEPc.ST().create(comm=COMM_WORLD)
    st.setType(SLEPc.ST.Type.SINVERT)

    es = SLEPc.EPS().create(comm=COMM_WORLD)
    es.setST(st)
    es.setDimensions(num_eigvals)
    es.setOperators(petsc_L, petsc_M)
    es.setWhichEigenpairs(es.Which.SMALLEST_MAGNITUDE)
    es.setFromOptions()
    es.solve()

    nconv = es.getConverged()

    if nconv == 0:
        raise RuntimeError("Did not converge any eigenvalues")

    eigenvalues = np.zeros(nconv)
    eigenfunctions_list = []

    vr, vi = petsc_L.getVecs()
    for i in range(nconv):
        eigv = es.getEigenpair(i, vr, vi)

        eigenvalues[i] = eigv.real
        eigenfunctions_list.append(np.copy(vr.getArray()))

    eigenfunctions = np.array(eigenfunctions_list).T
    norm = np.sqrt(np.sum(eigenfunctions**2, axis=0))
    eigenfunctions_orthonormal = (eigenfunctions / norm).T

    return eigenvalues, eigenfunctions_orthonormal


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    from .cli import argument_parser

    parser = argument_parser()
    args = parser.parse_args()

    print('Reading mesh')
    mesh = Mesh('meshes/dragon_connected.msh', dim=3)
    print('Ncells', mesh.num_vertices())

    V = FunctionSpace(mesh, "Lagrange", 1)

    print('Solving the eigenproblem, it may take a while')
    eigvals, eigfuns = get_eigenpairs(mesh, V, args.num_eigenpairs)

    plt.title('Eigenvalues')
    plt.plot(eigvals)
    plt.scatter(np.arange(len(eigvals)), eigvals, marker='+')
    plt.show()

    # (M, N)
    eigenpairs = np.zeros((len(eigvals),
                           mesh.num_vertices() + 1))
    eigenpairs[:, 0] = eigvals
    eigenpairs[:, 1:] = eigfuns

    if args.eigenpairs_file is not None:
        np.save(args.eigenpairs_file, eigenpairs)
