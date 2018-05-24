import numpy as np
import tensorly as tl
import logger

from tensorly.tenalg import khatri_rao

log = logger.create_logger('parafac', '/tmp/parafac.log')


class Parafac:

    def __init__(self, shape, n_components):
        # Set backend as numpy
        tl.set_backend('numpy')
        self.shape = shape
        self.n_components = n_components
        self.Z = []
        self.X = None
        self.rand_init()

    # Randomly generate a Parafac object
    #   - alpha:          Gamma shape parameter
    #   - beta:           Gamma rate parameter
    def rand_init(self, alpha=1, beta=1):
        self.Z = [np.random.gamma(alpha, 1. / beta, size=(s, self.n_components)) for s in self.shape]
        self.X = tl.kruskal_to_tensor(self.Z)

    # Estimates Parafac factors with alternating least squares
    #   - data:           data tensor to be factorized
    #   - max_iter:       maximum number of iterations
    #   - min_iter:       minimum numner of iterations
    #   - tol:            tolerance for early stopping.
    #   - start:          'random' or 'svd'
    def fit(self, X, max_iter=500, min_iter=10, tol=1e-6, start='random'):
        # sanity check
        assert X.shape == self.shape
        # Initialize
        if start == 'random':
            log.debug('Initializing randomly.')
            self.rand_init()
        elif start == 'svd':
            # TODO: Start with SVD
            raise ValueError('Not implemented yet.')
        else:
            raise ValueError('Wrong argument: start')

        dist = [self._frobenius_norm(X), ]
        log.debug('Initial distance: {}'.format(dist[-1]))
        for i in range(max_iter):
            # Update all factors once
            dist.append(self._update_factors(X))
            log.debug('Distance at iteration {}: {}'.format(i, dist[-1]))

            # Check that distance is non-increasing
            if dist[-1] > dist[-2]:
                log.error('Distance should be non-increasing')
                break

            # Stop if the decrease in distance is less than the tolerance
            if i >= min_iter:
                dec = dist[i - 1] - dist[i]
                if dec < tol:
                    log.debug('Decrease in distance is less than tolerance {} < {}'.format(dec, tol))
                    log.debug('Stopping early at iteration {}'.format(i))
                    break
        return dist

    # Run alternating least squares and update factors
    #   - data: tensor to be factorized.
    def _update_factors(self, X):
        for n in range(X.ndim):
            self.Z[n] = np.dot(tl.unfold(X, n), np.linalg.pinv(khatri_rao(self.Z, n)).T)
        self.X = tl.kruskal_to_tensor(self.Z)
        return self._frobenius_norm(X)

    def _frobenius_norm(self, X):
        return np.sqrt(np.sum(np.power(X-self.X, 2)))

    # Pretty prints the Parafac object
    def print(self):
        for i, z in enumerate(self.Z):
            print('Factor {}:\n-----------'.format(i+1))
            print(z)
        print('Data:\n-----')
        print(self.X)