# Wasserstein GD on the rate / Blahut-Arimoto functional ("BAF"). The main method studied in the paper.
import numpy as np
import jax.numpy as jnp
import jax
from jax._src.random import PRNGKey
from jax.scipy.special import logsumexp
from functools import partial
import common.jax_utils
from common.custom_train_state import TrainState


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


from common.jax_experiment import BaseExperiment


class Experiment(BaseExperiment):
  """Perform Wasserstein GD with particles"""

  @property
  def rd_lambda(self):
    return self.config.model_config.rd_lambda

  @property
  def distort_type(self):
    return self.config.model_config.distort_type

  def init_state(self, rng: PRNGKey):
    config = self.config
    # self.rd_lambda = config.model_config.rd_lambda
    self.lr_schedule = self.get_lr_schedule()

    train_iter = self.train_iter
    train_m = config.train_data_config.batchsize
    n = config.model_config.nu_support_size

    # Initialize \nu atoms using random training samples.
    X = jnp.concatenate([jax.tree_map(jnp.asarray, next(train_iter)) for _ in range(n // train_m + 1)], axis=0)
    rand_idx = jax.random.permutation(rng, jnp.size(X, 0))[:n]
    nu_x = X[rand_idx]
    # Make sure there are no duplicate points in representation of nu
    nu_x += 0.1 * jnp.std(nu_x, axis=0) * jax.random.normal(key=rng, shape=nu_x.shape)
    nu_w = 1 / n * jnp.ones((1, n))
    params = dict(nu_x=nu_x, nu_w=nu_w)
    # Create train state, which encapsulates model params (`params`) and optimizer state.
    state = TrainState.create(
      apply_fn=None,  # We don't use this for BA.
      params=params,
      tx_fn=self.get_optimizer)
    return state

  def train_step(self, base_rng, state: TrainState, batch, eval=False):
    # rng = jax.random.fold_in(base_rng, jax.lax.axis_index('batch'))
    # rng = jax.random.fold_in(base_rng, state.step)
    del base_rng
    rd_lambda = self.rd_lambda
    distort_type = self.distort_type

    params = state.params
    nu_x = params['nu_x']
    nu_w = params['nu_w']
    mu_x = batch
    m = jnp.size(batch, 0)
    mu_w = 1 / m * jnp.ones((m, 1))
    psi_prime, scalar_metrics = wgrad_baf(mu_x, mu_w, nu_x, nu_w, distort_type, rd_lambda)
    if eval:  # This repeats some computation from wgrad_baf but it's alright if done infrequently.
      eval_metrics = self.eval_step(-1, params, batch)
      scalar_metrics.update(eval_metrics['scalars'])
    grads = {'nu_x': psi_prime, 'nu_w': jnp.zeros_like(nu_w)}

    learning_rate = self.lr_schedule(state.step)
    new_state = state.apply_gradients(grads=grads, lr=learning_rate)

    scalar_metrics['lr'] = learning_rate
    metrics = dict(scalars=scalar_metrics)
    return new_state, metrics

  def eval_step(self, base_rng, params, batch, eval_step=0):
    del base_rng, eval_step

    nu_x = params['nu_x']
    nu_w = params['nu_w']
    mu_x = batch
    m = jnp.size(batch, 0)
    mu_w = 1 / m * jnp.ones((m, 1))
    scalar_metrics, _ = estimate_rd(mu_x, mu_w, nu_x, nu_w, self.distort_type, self.rd_lambda)

    metrics = dict(scalars=scalar_metrics)
    return metrics
