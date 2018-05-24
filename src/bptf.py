import numpy as np
import scipy.special as sp
import tensorly as tl
import logger
from datetime import datetime

from tensorly.tenalg import khatri_rao

# The logger is going to log all output to the given file
log_file = '/tmp/bptf_' + datetime.now().isoformat() + '.log'
log_name = 'bptf'
log = logger.create_logger(log_name, log_file)


class BPTF:

    def __init__(self, shape, n_components, a=0.1, b=1):
        tl.set_backend('numpy')
        self.shape = shape
        self.n_components = n_components
        self.n_modes = len(shape)
        self.Z = []     # factors
        self.X = None   # tensor
        # Gamma shape(A) and mean(B) objects. Gamma scale is B/A
        self.A = self._create_gamma_prior(a)
        self.B = self._create_gamma_prior(b)
        # Inference variables:
        self.C = []   # variational shape parameters for factors
        self.D = []   # variational scale parameters for factors
        self.E = []   # arithmetic expectations of factors
        self.L = []   # geometric expectations of factors
        # Randomly initialize itself
        self.rand_init()

    # -------------------------------------
    # Create a list of gamma prior matrices
    # -------------------------------------
    #   - prior:    prior object
    #   - returns:  list of matrices in the shape of factors
    def _create_gamma_prior(self, prior=1.0):
        # Case 1: prior is a shared scalar, then a[m][i,j] = prior
        if isinstance(prior, (int, float)):
            return [np.ones((s, self.n_components)) * prior for s in self.shape]
        # Case 2: prior is a list
        if isinstance(prior, list):
            assert len(prior) == self.n_modes
            # Case 2a: prior is a list of scalars, then a[m][i,j] = prior[m]
            if isinstance(prior[0], (int, float)):
                return [np.ones((s, self.n_components)) * prior[i] for i, s in enumerate(self.shape)]
            # Case 2b: prior is a list of matrices, then a[m][i,j] = prior[m][i,j]
            elif isinstance(prior[0], np.ndarray):
                assert [p.shape for p in prior] == [(s, self.n_components) for s in self.shape]
                return prior
        # For all other cases:
        raise ValueError('Illegal argument for Gamma prior')

    # ---------------------------------------------------
    # Randomly generate factors from the generative model
    # ---------------------------------------------------
    def rand_init(self):
        assert [a.shape for a in self.A] == [(s, self.n_components) for s in self.shape]
        assert [b.shape for b in self.B] == [(s, self.n_components) for s in self.shape]
        self.Z = [np.random.gamma(a, b/a) for a, b in zip(self.A, self.B)]
        self.X = np.sum(np.random.poisson(khatri_rao(self.Z)), -1).reshape(self.shape)

    # ---------------------------------------------------
    # Pretty print factors and X
    # ---------------------------------------------------
    def print(self):
        for i, z in enumerate(self.Z):
            print('Factor {}:\n-----------'.format(i + 1))
            print(z)
        print('Data:\n-----')
        print(self.X)

    # ---------------------------------------------------
    # Estimates BPTF factors with Variational-Bayes
    # ---------------------------------------------------
    #   - X:        tensor to be factorized
    #   - M:        mask
    #   - max_iter: maximum number of iterations
    #   - min_iter: minimum number of iterations
    #   - tol:      tolerance for early stopping.
    def fit(self, X, M=None, max_iter=2000, min_iter=10, tol=1e-6):
        # sanity check
        assert X.shape == self.shape
        if M is None:
            M = np.ones(X.shape)

        # initialize
        self._init_factors()
        elbo = [self._calculate_elbo(X, M), ]
        log.debug('Initial elbo: {}'.format(elbo[-1]))

        # Main update iteration:
        for i in range(max_iter):
            # Update each component wrt. other components
            elbo.append(self._update_factors(X, M))
            log.debug('Elbo at iteration {}: {}'.format(i, elbo[-1]))

            # Check that bound is non-decreasing
            delta = (elbo[-1] - elbo[-2]) / abs(elbo[-2])
            if delta < 0:
                log.error('Elbo has decreased by a factor of {} at iteration {}'.format(np.abs(delta), i))
                break

            # Check for early stop
            if i >= min_iter and delta < tol:
                log.debug('Relative increase in elbo is less than tolerance {} < {}'.format(delta, tol))
                log.debug('Stopping early at iteration {}'.format(i))
                break

        self._point_estimate()
        return elbo

    # --------------------------------------------------------
    # Initialize all components before variational bayes
    # --------------------------------------------------------
    #   - smoothness: smoothness of Gamma variables, which has mean = 1
    def _init_factors(self, smoothness=100):
        # Check that parameters are not initialized before
        assert len(self.C) == len(self.D) == len(self.E) == len(self.L) == 0
        # Initialize each component
        for m in range(self.n_modes):
            # Randomly initialize Gamma parameters
            c = np.random.gamma(smoothness, 1.0 / smoothness, size=(self.shape[m], self.n_components))
            d = np.random.gamma(smoothness, 1.0 / smoothness, size=(self.shape[m], self.n_components))
            self.C.append(c)
            self.D.append(d)
            # Calculate arithmetic and logarithmic expectations
            self.E.append(c * d)
            self.L.append(np.exp(sp.psi(c) + np.log(d)))
            # Check newly created component
            self._check_component(m)
        self._point_estimate()        

    # -------------------------------------------------------------
    # Point estimate of factors and X from expecatations of factors
    # -------------------------------------------------------------
    def _point_estimate(self):
        self.Z = self.E
        self.X = tl.kruskal_to_tensor(self.Z)

    # ----------------------------------------------------------
    # Update the given component conditioned on other components
    # ----------------------------------------------------------
    #   - X:    tensor to factorizex
    #   - mask: mask matrix
    #   - n:    component index
    def _update_factors(self, X, M):
        # Update components
        for n in range(self.n_modes):
            self.C[n] = self.A[n] + self.L[n] * self._uttkrp(M * (X / tl.kruskal_to_tensor(self.L)), self.L, n)
            self.D[n] = 1 / (self.A[n] / self.B[n] + self._uttkrp(M, self.E, n))
            self.E[n] = self.C[n] * self.D[n]
            self._check_component(n)
        # Calculate lower bound
        elbo = self._calculate_elbo(X, M)
        # Update exp(log(<Z>) values:
        for n in range(self.n_modes):
            self.L[n] = np.exp(sp.psi(self.C[n])) * self.D[n]
        return elbo

    # ----------------------------------------------------------
    # Calculate evidence lower bound
    # ----------------------------------------------------------
    #   - m: component index
    def _calculate_elbo(self, X, M):
        Lx = tl.kruskal_to_tensor(self.L)
        krL = khatri_rao(self.L)
        temp = (krL * np.log(krL)).sum(axis=-1).reshape(X.shape)
        bound = - np.sum(M * tl.kruskal_to_tensor(self.E)) \
                - np.sum(M * sp.gammaln(X + 1, dtype='float')) \
                - np.sum(M * X * ((temp / Lx) - np.log(Lx)))
        for n in range(self.n_modes):
            bound += - np.sum((self.A[n] / self.B[n]) * self.E[n]) \
                     - np.sum(sp.gammaln(self.A[n], dtype='float')) \
                     - np.sum(self.A[n] * np.log(self.B[n] / self.A[n])) \
                     + np.sum(self.C[n] * (np.log(self.D[n]) + 1) + sp.gammaln(self.C[n], dtype='float'))
        return bound
    
    # --------------------------------------------------------
    # Check that the given component is in good numeric limits
    # --------------------------------------------------------
    #   - n: component index
    def _check_component(self, n):
        # Assert that all values are finite and there's no NaN
        for obj in [self.C, self.D, self.E, self.L]:
            assert np.isfinite(obj[n]).all()
            assert ~np.isnan(obj[n]).any()

    # ---------------------------------------------------------
    # Unfolded Tensor Times Khatri-Rao Product of given factors
    # ---------------------------------------------------------
    #   - tensor :  the tensor (yeah, what did you expect?)
    #   - factors:  list of factors (seriously, dude!)
    #   - n:        unfolding mode
    @staticmethod
    def _uttkrp(tensor, factors, n):
        return np.dot(tl.unfold(tensor, n), khatri_rao(factors, skip_matrix=n))
