import math
import jax.numpy as jnp
import jax.random
from pbnn.data import DataSet
from pbnn.typings import JArray, JKey


def _logistic(x):
    return 1. / (1. + jnp.exp(-x))


class OneDimGaussBern(DataSet):
    r"""
    p(\theta) = N(m, v)
    p(Y | \theta) = Bernoulli(logistic(\theta ** 2))
    """
    n: int
    m: float
    v: float
    ys: JArray

    def __init__(self, key: JArray, n: int, m: float, v: float):
        thetas = m + math.sqrt(v) * jax.random.normal(key) * jnp.ones((n, 1))

        key, _ = jax.random.split(key)
        ys = jax.random.bernoulli(key, self.emission(thetas))

        self.n = n
        self.m = m
        self.v = v
        self.ys = self.reshape(ys)

    @staticmethod
    def emission(theta):
        return _logistic(theta ** 2)

    def draw_subset(self, key, batch_size: int):
        inds = jax.random.choice(key, jnp.arange(self.n), (batch_size,), replace=False)
        return self.reshape(self.ys[inds, :])

    def enumerate_subset(self, i: int):
        inds = self.rnd_inds[i]
        return self.reshape(self.ys[inds, :])

    def unnormalised_log_posterior(self, theta):
        r"""log p(\theta | Y_{1:n}) = \sum^n_i \log p(Y_i | \theta) + \log p(\theta)
        """
        return jnp.sum(jax.scipy.stats.bernoulli.logpmf(self.ys, self.emission(theta)), axis=0) \
            + jax.scipy.stats.norm.logpdf(theta, self.m, jnp.sqrt(self.v))

    def posterior(self, theta_grids):
        r"""p(\theta | Y_{1:n})
        """
        us = jnp.exp(jax.vmap(lambda x: self.unnormalised_log_posterior(x), in_axes=[0])(theta_grids))
        z = jnp.trapz(us[:, 0], theta_grids)
        return us / z


class Crescent(DataSet):
    r"""
    \phi ~ N(m, cov),
    Y | X ~ N(\phi_1 / \psi_0 + \psi_1 * (\phi_0 ** 2 + \psi_0 ** 2), 1.)
    """

    def __init__(self, n: int, key: JKey, psi: float = 1.):
        self.n = n
        self.psi = psi
        self.m = jnp.array([0., 0.])
        self.cov = jnp.array([[2., 0.],
                              [0., 1.]])
        self.cov_is_diag = True

        self.xs = jnp.ones((n, 1))

        phi = jnp.array([0., 0.])
        self.ys = self.reshape(self.emission(phi, psi) + 1. * jax.random.normal(key, (n,)))

    @staticmethod
    def emission(phi, psi):
        return phi[1] / psi + 0.5 * (phi[0] ** 2 + psi ** 2)

    def log_prior_pdf(self, phi):
        if self.cov_is_diag:
            return jnp.sum(jax.scipy.stats.norm.logpdf(phi, self.m, jnp.diag(self.cov)))
        else:
            return jax.scipy.stats.multivariate_normal.logpdf(phi, self.m, self.cov)

    def log_cond_pdf_likelihood(self, ys, phi, psi):
        return jnp.sum(jax.scipy.stats.norm.logpdf(ys, self.emission(phi, psi), math.sqrt(1.)))

    def posterior(self, phi_mesh):
        def energy(phi): return jnp.exp(self.log_prior_pdf(phi)
                                        + jnp.sum(self.log_cond_pdf_likelihood(self.ys, phi, self.psi)))

        evals = jax.vmap(jax.vmap(energy, in_axes=[0]), in_axes=[0])(phi_mesh)
        z = jax.scipy.integrate.trapezoid(jax.scipy.integrate.trapezoid(evals, phi_mesh[0, :, 0], axis=0),
                                          phi_mesh[0, :, 0])
        return evals / z
