# Wasserstein GD on the rate / Blahut-Arimoto functional ("BAF"). The main method studied in the paper.
import numpy as np
import jax.numpy as jnp
import jax
from jax._src.random import PRNGKey
from jax.scipy.special import logsumexp
from functools import partial
import common.jax_utils
from common.custom_train_state import TrainState


from bagd.core import wgrad_baf, estimate_rd


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
