""" Riemannian Manifold Gaussian Processes. """

import paramz
import autograd.numpy as np
from autograd.scipy.linalg import solve_triangular
from autograd import elementwise_grad as egrad
from .laplace_eigenvalues import get_eigenpairs
from .utils import jitchol


class ManifoldMaternGP(paramz.Model):
    """
    Interface for GPs on Riemannian Manifolds.

    The class inherits `paramz.Model` (as does GPy) so the usual methods
    such as `optimize` and `optimize_restarts` will work.
    The trainable parameters are `kappa`, `sigma_f` and `sigma_n` (see below).

    Attributes
    ----------
    mesh : firedrake.Mesh
        Triangular mesh.
    V : firedrake.FunctionSpace
        Function space for the Galerkin method.
    X : (n,) np.ndarray
        Indices of mesh vertices with the data.
    Y : (n,) np.ndarray
        Data observations.

    eigenpairs : tuple
        A tuple returned by laplace_eigenvalues.get_eigenpairs.
        If None, `num_eigenpairs` eigenpairs will be retrieved automatically.
        This argument is useful when one has Laplace eigenpairs precomputed.

    num_eigenpairs : int
        Number of eigenpairs.
        This will be the number of eigenpairs if `eigenpairs` is None.

    nu : float
        Non-trainable smoothness hyperparameter.
    kappa : float
        Trainable lengthscale hyperparameter.
    sigma_f : float
        Trainable scaling kernel hyperparameter.
    sigma_n : float
        Trainable noise likelihood variance.
    """

    def __init__(self, mesh, V, X, Y,
                 eigenpairs=None, num_features=500,
                 nu=3/2, kappa=1.0, sigma_f=1.0, sigma_n=1e-15):
        # s --- number of samples
        # l --- number of Fourier features
        # n --- number of data
        # t --- number of test points
        super().__init__(name='manifold_matern_gp')
        self.mesh = mesh
        self.V = V
        if eigenpairs is None:
            self.eigenvalues, self.eigenfunctions = \
                get_eigenpairs(mesh,
                               self.V,
                               num_features)
        else:
            self.eigenvalues, self.eigenfunctions = eigenpairs
        self.eigenfunctions = self.eigenfunctions.T
        self.nu = nu
        self.kappa = paramz.Param('kappa', kappa,
                                  default_constraint=paramz.transformations.Logexp())
        self.sigma_f = paramz.Param('sigma_f', sigma_f,
                                    default_constraint=paramz.transformations.Logexp())
        self.sigma_n = paramz.Param('sigma_n', sigma_n,
                                    default_constraint=paramz.transformations.Logexp())
        self.link_parameters(self.kappa, self.sigma_f, self.sigma_n)

        self.X = X
        self.Y = Y
        self.dim = self.mesh.topological_dimension()

        self._dL_dsigma_f = egrad(self._neg_log_likelihood_alt, 0)
        self._dL_dkappa = egrad(self._neg_log_likelihood_alt, 1)
        self._dL_dsigma_n = egrad(self._neg_log_likelihood_alt, 2)

    def objective_function(self):
        return self._neg_log_likelihood_alt(self.sigma_f,
                                            self.kappa,
                                            self.sigma_n)

    def parameters_changed(self):
        grad = self._neg_log_likelihood_alt_grad(self.sigma_f.values,
                                                 self.kappa.values,
                                                 self.sigma_n.values)
        self.sigma_f.gradient[:] = grad[0]
        self.kappa.gradient[:] = grad[1]
        self.sigma_n.gradient[:] = grad[2]

    def eval_K(self, S):
        """ Compute the matrix K(X, X). """
        K = (self.eigenfunctions[self.X] * S[None, :]) @ \
            self.eigenfunctions[self.X].T  # shape (n,n)
        return K

    def eval_S(self, kappa, sigma_f):
        """ Compute spectral density. """
        d = self.nu + 0.5 * self.dim
        S = np.power(kappa, 2*d) * \
            np.power(1. + np.power(kappa, 2)*self.eigenvalues, -d)
        S /= np.sum(S)
        S *= sigma_f
        return S

    def eval_K_chol(self, S, sigma_n, sigma_f):
        """ Compute Cholesky decomposition of the matrix
        K(X,X) + sigma_n*I. """
        K = self.eval_K(S)
        K += sigma_n * np.eye(K.shape[0])
        K_chol = jitchol(K)
        return K_chol

    def prior_samples(self, nsamples, coords=None):
        """ Draw `nsamples` samples from the prior. """
        S = self.eval_S(self.kappa, self.sigma_f)
        if coords is None:
            coords = slice(self.mesh.num_vertices())  # take all coords
        weights = np.random.normal(scale=np.sqrt(S),
                                   size=(nsamples,) + S.shape)  # shape (s, l)
        prior = np.einsum('sl,nl->sn', weights, self.eigenfunctions[coords])

        return prior

    def prior_variance(self):
        """ Return marginal variance k(x,x) for all x. """
        S = self.eval_S(self.kappa, self.sigma_f)
        variance = np.sum((self.eigenfunctions * S[None, :]).T *
                          self.eigenfunctions.T, axis=0)
        return variance

    def posterior_samples(self, nsamples):
        """ Compute `nsamples` posterior samples with the Materon rule """
        prior = self.prior_samples(nsamples)

        S = self.eval_S(self.kappa, self.sigma_f)
        K_chol = self.eval_K_chol(S, self.sigma_n, self.sigma_f)

        residue = (self.Y - prior[..., self.X]).T  # shape (n, s)

        K_chol_inv_R = solve_triangular(K_chol,
                                        residue, lower=True)  # shape (n, s)

        phi_S_phistar = (self.eigenfunctions[self.X] * S[None, :]) @ \
            self.eigenfunctions.T  # shape (n, t)

        K_chol_inv_phi_S_phistar = solve_triangular(K_chol,
                                                    phi_S_phistar,
                                                    lower=True)  # shape (n, t)
        update_term = np.einsum('nt,ns->st',
                                K_chol_inv_phi_S_phistar,
                                K_chol_inv_R)

        return prior + update_term  # shape (s, t)

    def predict(self, coords=None):
        """Return posterior mean and variance"""
        S = self.eval_S(self.kappa, self.sigma_f)
        K_chol = self.eval_K_chol(S, self.sigma_n, self.sigma_f)

        woodbury_vector = solve_triangular(K_chol, self.Y, lower=True)
        phi_S_phistar = (self.eigenfunctions[self.X] * S[None, :]) @ \
            self.eigenfunctions.T
        W = solve_triangular(K_chol, phi_S_phistar, lower=True)

        mean = np.einsum('nt,n->t', W, woodbury_vector)

        B = solve_triangular(K_chol, phi_S_phistar, lower=True)
        variance = np.sum((self.eigenfunctions * S[None, :]).T *
                          self.eigenfunctions.T, axis=0) - np.sum(B**2, axis=0)

        return mean, variance

    def _neg_log_likelihood_alt(self, sigma_f, kappa, sigma_n):
        if self.Y is None:
            return 0.0

        S = self.eval_S(kappa, sigma_f)

        K_chol = self.eval_K_chol(S, sigma_n, sigma_f)
        K_chol_inv_Y = solve_triangular(K_chol, self.Y, lower=True)

        data_fit = 0.5 * np.sum(K_chol_inv_Y * K_chol_inv_Y)
        penalty = np.sum(np.log(np.diag(K_chol)))
        objective = data_fit + penalty + 0.5 * len(self.Y) * np.log(2*np.pi)

        return objective

    def _neg_log_likelihood_alt_grad(self, sigma_f, kappa, sigma_n):
        if self.Y is None:
            return [0.0, 0.0, 0.0]
        grad = [self._dL_dsigma_f(sigma_f, kappa, sigma_n),
                self._dL_dkappa(sigma_f, kappa, sigma_n),
                self._dL_dsigma_n(sigma_f, kappa, sigma_n)]
        return grad
