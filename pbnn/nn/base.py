import jax
import jax.numpy as jnp
import math
from jax.flatten_util import ravel_pytree
from flax import linen
from pbnn.typings import JArray, JFloat, JKey
from functools import partial
from typing import Sequence, List, Tuple, Callable, Union, Iterable





def _categorical_logpmf(ys: JArray, ps: JArray) -> JFloat:
    """ys: one-hot vector (d, ), ps: probability bins (d, )
    """
    return jnp.dot(jax.nn.log_softmax(ps), ys)


def make_nn(nn: linen.Module,
            dict_params: dict) -> Tuple[JArray, Callable[[JArray], dict], Callable[[JArray, JArray], JArray],
Callable[[JArray, JArray], JArray]]:
    """Make a neural network for experiments.

    Parameters
    ----------
    nn : linen.Module
        A `linen.Module` class instance.
    dict_params : dict
        A dictionary of the neural network parameters.

    Returns
    -------
    JArray, Callable[(dw, ), (n, dx) -> (n, dy)], Callable[(s, dw), (n, dx) -> (s, n, dy)]
        The (dictionary) parameters ravelled in the array format, a function that ravels the array parameters back to
        the dictionary format, a function that evaluates forward pass of the neural network, and the vectorised version
        of the previous function.
    """
    array_params, array_to_dict = ravel_pytree(dict_params)

    def forward_pass(_array_params: JArray, _xs: JArray) -> JArray:
        return nn.apply(array_to_dict(_array_params), _xs)

    return array_params, array_to_dict, forward_pass, jax.vmap(jax.vmap(forward_pass, in_axes=[None, 0]),
                                                               in_axes=[0, None])


def concat_nns(nns: Sequence[linen.Module], dict_params: Sequence[dict]) -> Tuple[
    List[JArray], List[Callable], Callable]:
    """Sequentially concatenate a bunch of neural networks.

    Parameters
    ----------
    nns : Sequence[linen.Module]
        A sequence (e.g., list and tuple) of neural network instances.
    dict_params : Sequence[dict]
        A sequence of dictionaries, usually created by the `linen.Module.init` method.

    Returns
    -------
    List[JArray], List[Callable], Callable[Sequence[(dw, )], (n, dx) -> (n, dy)]
        A list of initial array parameters, a list of dictionary-making functions, a function that evaluates the
        forward pass of the neural network, and the vmap of the forward pass function.

    Notes
    -----
    This is a naive concatenation implementation, may not be efficient.
    """
    array_params_and_dict_fns = [ravel_pytree(dict_param) for dict_param in dict_params]

    def forward_pass(_array_params: Sequence[JArray], _xs: JArray):
        out = _xs
        for nn, _array_param, (_, array_to_dict) in zip(nns, _array_params, array_params_and_dict_fns):
            out = nn.apply(array_to_dict(_array_param), out)
        return out

    return ([item[0] for item in array_params_and_dict_fns], [item[1] for item in array_params_and_dict_fns],
            forward_pass)


def make_nn_likelihood(forward_pass: Callable[[JArray, JArray], JArray],
                       likelihood: str = 'bernoulli') -> Tuple[
    Callable[[JArray, JArray, JArray], JFloat], Callable[[JArray, JArray, JArray], JFloat]]:
    r"""Make the measurement conditional PDF and score defined by a neural network.

    Parameters
    ----------
    forward_pass : Callable[(dw, ), (dx, ) -> (dy, )]
        The forward function.
    likelihood : str, default='bernoulli'
        Likelihood model. 'bernoulli' or 'categorical'.

    Returns
    -------

    """

    def likelihood_cond_log_pdf(ys, sample, xs):
        evals = jax.vmap(forward_pass, in_axes=[None, 0])(sample, xs)  # (n, dy)
        if likelihood == 'bernoulli':
            return jnp.sum((1 - ys) * jax.nn.log_sigmoid(-evals) + ys * jax.nn.log_sigmoid(evals))
        elif likelihood == 'categorical':
            return jnp.sum(jax.vmap(_categorical_logpmf, in_axes=[0, 0])(ys, evals))
        else:
            raise NotImplementedError(f'Likelihood {likelihood} not implemented')

    def score(ys, sample, xs):
        return jax.grad(likelihood_cond_log_pdf, argnums=1)(ys, sample, xs)

    return likelihood_cond_log_pdf, score


def make_pbnn(nns: Sequence[linen.Module],
              random_argnums: Sequence[int],
              in_dims: Sequence[int],
              batch_size: int,
              keys: JKey) -> Tuple[Tuple[JArray, Callable], Tuple[JArray, Callable], Callable, Callable]:
    """
    Parameters
    ----------
    nns : Sequence[linen.Module]
        A sequence (e.g., list and tuple) of neural network instances.
    random_argnums : Sequence[int]
        A sequence of positional integers that specifies the stochastic neural network components. For example, [0, 3]
        means that the first the fourth components in `nns` are the stochastic.
    in_dims : Sequence[int]
        A sequence of integers that define the input dimension of each neural network in `nns`. This in principle
        should be done under the hood of this function, but I don't want to bother to implement it, so please manually
        specify the dimensions.
    batch_size : int
        The data batch size.
    keys : JKey
        AX random keys. Must be the same length as `nns`.

    Returns
    -------
    Tuple[JArray, Callable], Tuple[JArray, Callable], Callable (dw, ), (dp, ), (n, dx) -> (n, dy),
    Callable (s, dw), (dp, ), (n, dx) -> (s, n, dy)
        Two tuples of initial arrays and pytree functions, a function that evaluates the forward pass of the pBNN, and
        a vmap function of the forward pass.

    Notes
    -----
    The integers in `random_argnums` must be in the ascending order.
    """
    deterministic_argnums = [argnum for argnum in range(len(nns)) if argnum not in random_argnums]

    init_dicts = []
    for nn, in_dim, key in zip(nns, in_dims, keys):
        if isinstance(in_dim, Iterable):
            shape = (batch_size, *in_dim)
        else:
            shape = (batch_size, in_dim)
        init_dicts.append(nn.init(key, jnp.ones(shape)))

    ls_of_random_dicts = [init_dicts[random_argnum] for random_argnum in random_argnums]
    ls_of_deterministic_dicts = [init_dicts[deterministic_argnum] for deterministic_argnum in deterministic_argnums]

    random_array, random_array_to_pytree = ravel_pytree(ls_of_random_dicts)
    deterministic_array, deterministic_array_to_pytree = ravel_pytree(ls_of_deterministic_dicts)

    def forward_pass(random_param: JArray, deterministic_param: JArray, xs: JArray) -> JArray:
        """
        random_param : (dw, )
        deterministic_param : (dp, )
        xs : (n, dx) or (dx, )
        return : (n, dy) or (dy, )
        """
        random_param_pytree = random_array_to_pytree(random_param)
        deterministic_param_pytree = deterministic_array_to_pytree(deterministic_param)

        j, k = 0, 0
        out = xs
        for i in range(len(nns)):
            if i in random_argnums:
                out = nns[i].apply(random_param_pytree[j], out)
                j += 1
            else:
                out = nns[i].apply(deterministic_param_pytree[k], out)
                k += 1
        return out

    forward_pass_vmap = jax.vmap(jax.vmap(forward_pass, in_axes=[None, None, 0]), in_axes=[0, None, None])

    return ((random_array, random_array_to_pytree), (deterministic_array, deterministic_array_to_pytree), forward_pass,
            forward_pass_vmap)


def make_pbnn_likelihood(forward_pass: Callable[[JArray, JArray, JArray], JArray],
                         likelihood_type: Union[str, float, JArray],
                         ignore_nan: bool = False) -> Tuple[Callable[
    [JArray, JArray, JArray, JArray], JFloat], Callable, Callable]:
    r"""Make likelihood models with pBNN.

    Parameters
    ----------
    forward_pass : Callable (dw, ), (dp, ), (n, dx) -> (n, dy)
        A function that evaluates the pBNN with three arguments: random_params, deterministic_params, and batched input,
        and returns the batched output.
    likelihood_type : Union[str, float, JArray]
        This argument defines the likelihood model.

            - `bernoulli`: The Bernoulli distribution.
            - 'categorical': The categorical distribution.
            - float: The Normal distribution with this number as the variance.
            - JArray (dy, dy): The multivariate Normal distribution with this as the covariance.
    ignore_nan : bool, default=False
        Whether use jax.numpy.nansum and jax.numpy.nanprod. This is used for inflated data.

    Returns
    -------
    Callable, Callable, Callable
        The log conditional PDF, conditional PDF, and score.
    """
    sum_fn = jnp.nansum if ignore_nan else jnp.sum
    prod_fn = jnp.nanprod if ignore_nan else jnp.prod

    def log_cond_pdf(ys: JArray, sample: JArray, xs: JArray, param: JArray) -> JFloat:
        """Log conditional PDF.

        Parameters
        ----------
        ys : JArray (n, dy)
        sample : JArray (dw, )
        xs : JArray (n, dx)
        param : JArray (dp, )

        Returns
        -------
        JFloat
        """
        evals = forward_pass(sample, param, xs)  # (n, dy)
        if likelihood_type == 'bernoulli':
            return sum_fn((1 - ys) * jax.nn.log_sigmoid(-evals) + ys * jax.nn.log_sigmoid(evals))
        elif likelihood_type == 'categorical':
            return sum_fn(jax.vmap(_categorical_logpmf, in_axes=[0, 0])(ys, evals))
        elif isinstance(likelihood_type, float):
            return sum_fn(jax.scipy.stats.norm.logpdf(ys, evals, math.sqrt(likelihood_type)))
        elif isinstance(likelihood_type, JArray):
            return sum_fn(jax.scipy.stats.multivariate_normal.logpdf(ys, evals, likelihood_type))
        else:
            raise NotImplementedError(f'Likelihood type {likelihood_type} is not implemented')

    def cond_pdf(ys: JArray, sample: JArray, xs: JArray, param: JArray) -> JFloat:
        pass

    def score(ys: JArray, sample: JArray, xs: JArray, param: JArray) -> JArray:
        return jax.grad(log_cond_pdf, argnums=1)(ys, sample, xs, param)

    return log_cond_pdf, cond_pdf, score
