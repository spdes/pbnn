import jax
import jax.numpy as jnp
from pbnn.typings import JArray
from typing import Callable


def _linear_update(m, v, H, pred, xi, y):
    S = H @ v @ H.T + xi
    if H.shape[0] == 1:
        K = v @ H.T / S
    else:
        chol = jax.lax.linalg.cholesky(S)
        K = jax.lax.linalg.triangular_solve(chol, H @ v).T
    return m + K @ (y - pred), v - K @ S @ K.T


def _ekf_update(measurement_cond_m_cov, mp, vp, xs, ys, fwd_jacobian):
    # measurement_cond_m_cov: (dz, ), (dx, ) -> (dy, )
    if fwd_jacobian:
        jac = jax.jacfwd
    else:
        jac = jax.jacrev
    Hs = jax.vmap(jac(lambda u, v: measurement_cond_m_cov(u, v)[0]), in_axes=[None, 0])(mp, xs)  # (batch, dy, dz)
    pred_ys, pred_xis = jax.vmap(measurement_cond_m_cov, in_axes=[None, 0])(mp, xs)  # (batch, dy), (batch, dy, dy)
    ms, vs = jax.vmap(_linear_update, in_axes=[None, None, 0, 0, 0, 0])(mp, vp, Hs, pred_ys, pred_xis, ys)
    return jnp.mean(ms, axis=0), jnp.mean(vs, axis=0), jnp.mean((ys - pred_ys) ** 2)


def ekf_kernel(mf: JArray, vf: JArray, xs: JArray, ys: JArray,
               state_cond_m_cov: Callable, dt: float,
               measurement_cond_m_cov: Callable,
               fwd_jacobian: bool = False):
    """The step kernel of extended Kalman filter.

    Parameters
    ----------
    mf
    vf
    xs
    ys
    state_cond_m_cov
    dt
    measurement_cond_m_cov
    fwd_jacobian

    Returns
    -------

    """
    # Prediction
    jac_F = jax.jacfwd(lambda u: state_cond_m_cov(u, dt)[0])(mf)
    mp, Sigma = state_cond_m_cov(mf, dt)
    vp = jac_F @ vf @ jac_F.T + Sigma

    # Update
    mf, vf, loss = _ekf_update(measurement_cond_m_cov, mp, vp, xs, ys, fwd_jacobian)
    return mf, vf, loss


def sgp_kernel():
    pass
