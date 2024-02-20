import jax
import jax.numpy as jnp
from pbnn.typings import JArray, JFloat, Array
from typing import NamedTuple, Callable


class GaussianSum1D(NamedTuple):
    """Unidimensional Gaussian-sum distribution.
    """
    N: int
    means: JArray
    variances: JArray
    weights: JArray

    def pdf(self, xs):
        pdfs = jax.scipy.stats.norm.pdf(jnp.atleast_1d(xs)[:, None], self.means, jnp.sqrt(self.variances))
        return jnp.sum(pdfs * self.weights[None, :], axis=1)

    def sampler(self, key, n):
        cs = jax.random.choice(key, self.means.shape[0], (n,), p=self.weights)
        key, _ = jax.random.split(key)
        return self.means[cs] + jnp.sqrt(self.variances[cs]) * jax.random.normal(key, (n,))

    @classmethod
    def new(cls, means: JArray, variances: JArray, weights: JArray):
        return cls(N=means.shape[0], means=means, variances=variances, weights=weights)


class GaussianSumND(NamedTuple):
    """Multidimensional Gaussian-sum distribution.
    """
    d: int
    N: int
    means: JArray  # (n, d)
    covs: JArray  # (n, d, d)
    weights: JArray  # (n, )

    def pdf(self, x):
        pdfs = [jax.scipy.stats.multivariate_normal.pdf(x, mean, cov) * weight for mean, cov, weight in
                zip(self.means, self.covs, self.weights)]
        return jnp.sum(jnp.array(pdfs))

    def logpdf(self, x):
        """Log pdf using logsumexp
        """
        log_pdfs = jnp.array(
            [jax.scipy.stats.multivariate_normal.logpdf(x, mean, cov) for mean, cov in zip(self.means, self.covs)])
        return jax.scipy.special.logsumexp(log_pdfs, b=self.weights)

    def sampler(self, key, nsamples):
        cs = jax.random.choice(key, self.means.shape[0], (nsamples,), p=self.weights)
        key, _ = jax.random.split(key)
        return self.means[cs, :] + jnp.einsum('...ij,...j->...i', jnp.linalg.cholesky(self.covs[cs, :, :]),
                                              jax.random.normal(key, (nsamples, self.d)))

    @classmethod
    def new(cls, means: JArray, covs: JArray, weights: JArray):
        N, d = means.shape
        return cls(N=N, d=d, means=means, covs=covs, weights=weights)


def nlpd_mc(cond_pdf: Callable, posterior_samples: JArray, param: JArray, ys: JArray, xs: JArray) -> JFloat:
    r"""Compute negative log predictive density based on Monte Carlo samples.

    .. math::
        \frac{1}{n} \sum_{i=1}^n -\log p(y_i) =
        \frac{1}{n} \sum_{i=1}^n -\log \int p(y_i | \phi; \psi) p^*(\phi) d\phi

    Parameters
    ----------
    cond_pdf : Callable (n, dy), (dw, ), (n, dx), (dp, ) -> (n, )
    posterior_samples : (j, dw)
    param : (dp, )
    ys : (n, dy)
    xs : (n, dx)

    Returns
    -------

    """

    def single_nlpd(y, x) -> JFloat:
        return -jnp.log(jnp.mean(jax.vmap(cond_pdf, in_axes=[None, 0, None, None])(y[None, :], posterior_samples,
                                                                                   x[None, :], param)))

    return jnp.mean(jax.vmap(single_nlpd, in_axes=[0, 0])(ys, xs))


def nlpd_mc_seq(cond_pdf: Callable, posterior_samples: JArray, param: JArray,
                ys: JArray, xs: JArray,
                nchunks: int) -> float:
    """If xs and ys are large, it is not possible to compute the NLPD due to memory constrain. Do it sequentially.
    """

    def scan_body(carry, elem):
        s = carry
        y, x = elem

        s += nlpd_mc(cond_pdf, posterior_samples, param, y, x)
        return s, None

    ys_chunks = jnp.stack(jnp.array_split(ys, nchunks))
    xs_chunks = jnp.stack(jnp.array_split(xs, nchunks))
    return jax.lax.scan(scan_body, 0., (ys_chunks, xs_chunks))[0] / nchunks


def accuracy(predicted_logits: Array, true_labels: Array) -> JFloat:
    """

    Parameters
    ----------
    predicted_logits : Array-like (n, d)
        An array of prediction logits, where `n` and `d` stands for the numbers of data points and classes, resp. The
        logits must sum to one.
    true_labels : Array-like (n, d)
        An array of true labels in one-hot encoding.

    Returns
    -------
    JFloat
        The accuracy.
    """
    predicted_labels = jnp.argmax(predicted_logits, axis=-1)
    n_corrected_preds = jnp.sum(jnp.diag(true_labels[:, predicted_labels]))
    return n_corrected_preds / predicted_labels.shape[0]
