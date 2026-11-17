# Wasserstein gradient and evaluation of the Blahut-Arimoto functional.
import jax.numpy as jnp
import jax
from jax.scipy.special import logsumexp
from functools import partial
import common.jax_utils


@partial(jax.jit, static_argnames=['distort_type'])
def wgrad_baf(mu_x, mu_w, nu_x, nu_w, distort_type: str, rd_lambda):
  """
  Compute the Wasserstein gradient of the Blahut-Arimoto functional (BAF).
  :param mu_x:
  :param mu_w:
  :param nu_x:
  :param nu_w:
  :return:
  """

  pairwise_distort_fn = common.jax_utils.get_pairwise_distort_fn(distort_type)

  def compute_psi_sum(nu_x):
    """
    Here we compute a surrogate loss based on \psi, in order for jax autodiff to compute \nabla \psi, i.e., the desired
    Wasserstein gradient.
    :param nu_x:
    :return: psi_sum = \sum_i \psi(nu_x[i]), loss = BAF(nu_x).
    """
    C = pairwise_distort_fn(mu_x, nu_x)  # [m, n]
    scaled_C = rd_lambda * C
    log_nu_w = jnp.log(nu_w)  # [1, n]

    # Solve BA inner problem with a fixed nu. This only takes one step.
    phi = - logsumexp(-scaled_C + log_nu_w, axis=1, keepdims=True)  # Similar to SE1. [m, 1]
    loss = jnp.sum(mu_w * phi)

    # Now evaluate \psi on the atoms of \nu. Here \psi(y) is the first variation of BAF w.r.t. \nu;
    # this is analogous to the Schrodinger potential \psi which is the first variation of EOT functional w.r.t. \nu.
    phi = jax.lax.stop_gradient(phi)  # Important. phi is treated as a const.
    psi = - jnp.sum(jnp.exp(phi - scaled_C) * mu_w, axis=0)
    psi_sum = jnp.sum(psi)  # For computing gradient w.r.t. each nu_x atom.
    metrics = dict(loss=loss, psi_min=jnp.min(psi), psi_max=jnp.max(psi), psi_mean=jnp.mean(psi))
    return psi_sum, metrics

  # Evaluate the Wasserstein gradient of BAF, i.e., \nabla \psi, on nu_x.
  psi_prime, metrics = jax.grad(compute_psi_sum, has_aux=True)(nu_x)
  n = jnp.size(psi_prime, 0)
  psi_prime_norm = jnp.mean(jnp.linalg.norm(jnp.reshape(psi_prime, [n, -1]), axis=1))
  scalar_metrics = dict(**metrics, psi_prime_norm=psi_prime_norm)
  return psi_prime, scalar_metrics


@partial(jax.jit, static_argnames=['distort_type'])
def estimate_rd(mu_x, mu_w, nu_x, nu_w, distort_type: str, rd_lambda):
  """
  Given mu and nu, compute the (rate, distortion) associated with the optimal kernel K (or, transition distribution
  Q_{Y|X}) as defined by the inner minimization of the Blahut-Arimoto variational problem.
  :param mu_x:
  :param mu_w:
  :param nu_x:
  :param nu_w:
  :param dist_fun:
  :param rd_lambda:
  :return:
  """
  pairwise_distort_fn = common.jax_utils.get_pairwise_distort_fn(distort_type)
  C = pairwise_distort_fn(mu_x, nu_x)  # [m, n]
  scaled_C = rd_lambda * C
  log_nu_w = jnp.log(nu_w)  # [1, n]

  # Solve BA inner problem with a fixed nu. This only takes one step.
  phi = - logsumexp(-scaled_C + log_nu_w, axis=1, keepdims=True)  # Similar to SE1. [m, 1]
  loss = jnp.sum(mu_w * phi)
  # Find \pi^* via \phi
  pi = jnp.exp(phi - scaled_C) * jnp.outer(mu_w, nu_w)  # [m, n]; this is the optimized joint distribution P_X Q*_{Y|X}.
  distortion = jnp.sum(pi * C)
  rate = loss - rd_lambda * distortion

  scalar_metrics = dict(loss=loss, rate=rate, distortion=distortion)
  return scalar_metrics, pi
