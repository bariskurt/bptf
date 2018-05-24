import numpy as np
import scipy.special as sp
import tensorly as tl
import logger
from datetime import datetime

# The logger is going to log all output to the given file
log_file = '/tmp/bptf_' + datetime.now().isoformat() + '.log'
log_name = 'bptf'
log = logger.create_logger(log_name, log_file)


class NMF():

    def __init__(self, shape, n_components, alpha=0.1, beta=1):
        tl.set_backend('numpy')
        self.shape = shape
        self.n_components = n_components
        self.n_modes = len(shape)
        self.T = None  # factor 1
        self.V = None  # factor 2
        self.X = None  # tensor
        # Gamma shape(A) and mean(B) objects. Gamma scale is B/A
        self.At = self.Av = None
        self.Bt = self.Bv = None
        # Inference variables:
        self.Ct = self.Cv = None
        self.Dt = self.Dv = None
        self.Et = self.Ev = None
        self.Lt = self.Lv = None
        # Randomly initialize itself
        self.rand_init(alpha, beta)
        self.rand_init(alpha, beta)

    # -------------------------------------
    # Create a list of gamma prior matrices
    # -------------------------------------
    #   - prior:    prior object
    #   - shape:    shape of the gamma prior
    #   - returns:  the gamma prior
    @staticmethod
    def _create_gamma_prior(prior, shape):
        # Case 1: prior is a shared scalar, then a[i,j] = prior
        if isinstance(prior, (int, float)):
            return np.ones(shape) * prior
        # Case 2: prior is a matrix, then a[i,j] = prior[i,j]
        elif isinstance(prior, np.ndarray):
            assert prior.shape == shape
            return prior
        raise ValueError('Illegal argument for Gamma prior')

    # ---------------------------------------------------
    # Randomly generate factors from the generative model
    # ---------------------------------------------------
    def rand_init(self, alpha, beta):
        if not isinstance(alpha, list):
            alpha = [alpha, alpha]
        if not isinstance(beta, list):
            beta = [beta, beta]
        # Rand init At, Bt
        self.At = self._create_gamma_prior(alpha[0], (self.shape[0], self.n_components))
        self.Bt = self._create_gamma_prior(beta[0], (self.shape[0], self.n_components))
        self.T = np.random.gamma(self.At, self.Bt / self.At)

        self.Av = self._create_gamma_prior(alpha[1], (self.n_components, self.shape[1]))
        self.Bv = self._create_gamma_prior(beta[1], (self.n_components, self.shape[1]))
        self.V = np.random.gamma(self.Av, self.Bv / self.Av)

        self.X = np.random.poisson(np.dot(self.T, self.V))

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
    #   - max_iter: maximum number of iterations
    #   - min_iter: minimum number of iterations
    #   - tol:      tolerance for early stopping.
    def fit(self, X, M=None, max_iter=500, min_iter=10, tol=1e-6):
        # sanity check
        assert X.shape == self.shape
        if M is None:
            M = np.ones(X.shape)

        # initialize
        self._init_components()
        elbo = [self._calculate_elbo(X, M), ]
        log.debug('Initial elbo: {}'.format(elbo[-1]))

        # Main update iteration:
        for i in range(max_iter):
            # Update each component wrt. other components
            elbo.append(self._update_components(X, M))
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
    def _init_components(self, smoothness=100):
        # Initialize T
        self.Ct = np.random.gamma(smoothness, 1.0 / smoothness, size=(self.shape[0], self.n_components))
        self.Dt = np.random.gamma(smoothness, 1.0 / smoothness, size=(self.shape[0], self.n_components))
        self.Et = self.Ct * self.Dt
        self.Lt = np.exp(sp.psi(self.Ct)) * self.Dt
        self._check_component('T')
        # Initialize V
        self.Cv = np.random.gamma(smoothness, 1.0 / smoothness, size=(self.n_components, self.shape[1]))
        self.Dv = np.random.gamma(smoothness, 1.0 / smoothness, size=(self.n_components, self.shape[1]))
        self.Ev = self.Cv * self.Dv
        self.Lv = np.exp(sp.psi(self.Cv)) * self.Dv
        self._check_component('V')
        # Make point estimate
        self._point_estimate()

    # -------------------------------------------------------------
    # Point estimate of factors and X from expectations of factors
    # -------------------------------------------------------------
    def _point_estimate(self):
        self.T = self.Et
        self.V = self.Ev
        self.X = np.dot(self.T, self.V)

    # -------------------------------------------------------------
    # Update T and V, and return new lower bound
    # -------------------------------------------------------------
    def _update_components(self, X, M):
        # Update T
        self.Ct = self.At + self.Lt * np.dot((X * M) / np.dot(self.Lt, self.Lv), self.Lv.T)
        self.Dt = 1 / (self.At / self.Bt + np.dot(M, self.Ev.T))
        self.Et = self.Ct * self.Dt
        # Update V
        self.Cv = self.Av + self.Lv * np.dot(self.Lt.T, ((X * M) / np.dot(self.Lt, self.Lv)))
        self.Dv = 1 / (self.Av / self.Bv + np.dot(self.Et.T, M))
        self.Ev = self.Cv * self.Dv
        # Calculate bound
        elbo = self._calculate_elbo(X, M)
        # Update exp(<log(T)>) and exp(<log(V)>)
        self.Lt = np.exp(sp.psi(self.Ct)) * self.Dt
        self.Lv = np.exp(sp.psi(self.Cv)) * self.Dv
        # Check components for numerical errors
        self._check_component('T')
        self._check_component('V')
        return elbo

    # ----------------------------------------------------------
    # Calculate evidence lower bound
    # ----------------------------------------------------------
    #   - m: component index
    def _calculate_elbo(self, X, M):
        temp = np.dot(self.Lt * np.log(self.Lt), self.Lv) + np.dot(self.Lt, self.Lv * np.log(self.Lv))
        Lx = np.dot(self.Lt, self.Lv)
        return -np.sum(M * np.dot(self.Et, self.Ev)) \
               - np.sum(M * sp.gammaln(X+1, dtype='float')) \
               - np.sum(M * X * ((temp / Lx) - np.log(Lx))) \
               + self._elbo_part(self.At, self.Bt, self.Ct, self.Dt, self.Et) \
               + self._elbo_part(self.Av, self.Bv, self.Cv, self.Dv, self.Ev)

    @staticmethod
    def _elbo_part(A, B, C, D, E):
        return -np.sum((A / B) * E) - np.sum(sp.gammaln(A, dtype='float')) + np.sum(A * np.log(A / B)) \
                + np.sum(C * (np.log(D) + 1) + sp.gammaln(C, dtype='float'))

    # --------------------------------------------------------
    # Check that the given component is in good numeric limits
    # --------------------------------------------------------
    #   - n: component index
    def _check_component(self, component):
        if component == 'T':
            check_list = [self.Ct, self.Dt, self.Et, self.Lt]
        elif component == 'V':
            check_list = [self.Cv, self.Dv, self.Ev, self.Lv]
        else:
            raise ValueError('What are you trying to do ?')
        # Assert that all values are finite and there's no NaN
        for obj in check_list:
            assert np.isfinite(obj).all()
            assert ~np.isnan(obj).any()