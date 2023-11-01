# Blahut-Arimoto.
import jax.numpy as jnp
import jax
from jax._src.random import PRNGKey
from jax.scipy.special import logsumexp
import common.jax_utils
from common.custom_train_state import TrainState
from functools import partial


@jax.jit
def ba_loss_and_update_given_scaled_distort(mu_w, nu_w, scaled_C):
  # scaled_C = rd_lambda * C
  log_nu_w = jnp.log(nu_w)  # [1, n]

  # To evaluate the BA functional at the current nu, we need to solve the inner optimization problem w.r.t. \pi.
  # Solve BA inner problem with a fixed nu. This only takes one step.
  phi = - logsumexp(-scaled_C + log_nu_w, axis=1, keepdims=True)  # Similar to SE1. [m, 1]
  loss = jnp.sum(mu_w * phi)  # BAF(nu)
  # Find \pi^* via \phi. Here we compute it in two steps, as we can reuse the computation in step 1 to find an R-D lb.
  pi = jnp.exp(phi - scaled_C) * jnp.outer(mu_w, nu_w)  # [m, n]; this is the optimized joint distribution P_X Q*_{Y|X}.

  # Marginalize.
  new_nu_w = jnp.sum(pi, axis=0, keepdims=True)  # [1, n]

  return new_nu_w, loss


@partial(jax.jit, static_argnames=['distort_type'])
def ba_loss_and_update(mu_x, mu_w, nu_x, nu_w, distort_type: str, rd_lambda):
  pairwise_distort_fn = common.jax_utils.get_pairwise_distort_fn(distort_type)
  C = pairwise_distort_fn(mu_x, nu_x)  # [m, n]
  scaled_C = rd_lambda * C
  return ba_loss_and_update_given_scaled_distort(mu_w, nu_w, scaled_C)


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
  # # Find \pi^* via \phi
  # pi = jnp.exp(phi - scaled_C) * jnp.outer(mu_w, nu_w)  # [m, n]; this is the optimized joint distribution P_X Q*_{Y|X}.
  # distortion = jnp.sum(pi * C)
  # rate = loss - rd_lambda * distortion

  # Find \pi^* via \phi. Here we compute it in two steps, as we can reuse the computation in step 1 to find an R-D lb.
  # Note that this is an lb on the R(D) of the empirical distribution, not necessarily the true distribution.
  # pi = jnp.exp(phi - scaled_C) * jnp.outer(mu_w, nu_w)  # [m, n]; this is the optimized joint distribution P_X Q*_{Y|X}.
  pi_factor = jnp.exp(phi - scaled_C) * mu_w
  pi = pi_factor * nu_w
  scaled_distortion = jnp.sum(pi * scaled_C)
  distortion = scaled_distortion / rd_lambda
  rate = loss - scaled_distortion

  rdlb_part_fun = jnp.sum(pi_factor, axis=0)  # for each y, compute \int e^{-lamb * \rho(x, y) + phi(x)} d\mu(x)
  ublb_gap = jnp.log(jnp.max(rdlb_part_fun))
  rdlb = loss - ublb_gap

  scalar_metrics = dict(loss=loss, rate=rate, distortion=distortion, rdlb=rdlb, ublb_gap=ublb_gap)
  return scalar_metrics, pi


from common.jax_experiment import BaseExperiment


class Experiment(BaseExperiment):
  """Main class responsible for train/eval."""

  @property
  def rd_lambda(self):
    return self.config.model_config.rd_lambda

  @property
  def distort_type(self):
    return self.config.model_config.distort_type

  def init_state(self, rng: PRNGKey):
    config = self.config
    # self.rd_lambda = config.model_config.rd_lambda
    # self.lr_schedule = self.get_lr_schedule()   # Unused for BA.

    train_iter = self.train_iter
    train_m = config.train_data_config.batchsize
    n = config.model_config.nu_support_size

    # Initialize \nu atoms using random training samples.
    X = jnp.concatenate([jax.tree_map(jnp.asarray, next(train_iter)) for _ in range(n // train_m + 1)], axis=0)
    rand_idx = jax.random.permutation(rng, jnp.size(X, 0))[:n]
    nu_x = X[rand_idx]
    nu_w = 1 / n * jnp.ones((1, n))
    params = dict(nu_x=nu_x, nu_w=nu_w)
    # Create train state, which encapsulates model params (`params`) and optimizer state.
    state = TrainState.create(
      apply_fn=None,  # We don't use this for BA.
      params=params,
      tx_fn=self.get_optimizer)

    # Cache the pairwise distortion matrix if the data source is a fixed set of particles.
    if config.train_data_config.get('fixed_batch'):
      mu_x = next(train_iter)
      pairwise_distort_fn = common.jax_utils.get_pairwise_distort_fn(self.distort_type)
      C = pairwise_distort_fn(mu_x, nu_x)  # [m, n]
      scaled_C = self.rd_lambda * C
      self.scaled_C = scaled_C
    else:
      self.scaled_C = None

    return state

  def train_step(self, base_rng, state: TrainState, batch, eval=False):
    # rng = jax.random.fold_in(base_rng, jax.lax.axis_index('batch'))
    # rng = jax.random.fold_in(base_rng, state.step)
    del base_rng
    params = state.params
    if self.scaled_C is not None:
      # nu_x = params['nu_x']
      nu_w = params['nu_w']
      m = jnp.size(self.scaled_C, 0)
      mu_w = 1 / m * jnp.ones((m, 1))
      new_nu_w, loss = ba_loss_and_update_given_scaled_distort(mu_w, nu_w, self.scaled_C)
    else:
      nu_x = params['nu_x']
      nu_w = params['nu_w']
      mu_x = batch
      m = jnp.size(batch, 0)
      mu_w = 1 / m * jnp.ones((m, 1))
      new_nu_w, loss = ba_loss_and_update(mu_x, mu_w, nu_x, nu_w, self.distort_type, self.rd_lambda)

    scalar_metrics = dict(loss=loss)
    if eval:  # Wasteful but done rarely.
      eval_metrics = self.eval_step(-1, params, batch)
      scalar_metrics.update(eval_metrics['scalars'])
    new_params = dict(nu_x=params['nu_x'], nu_w=new_nu_w)
    new_state = state.replace(step=state.step + 1, params=new_params)
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
