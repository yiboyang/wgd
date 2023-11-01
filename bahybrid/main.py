# Wasserstein GD + BA step on the Blahut-Arimoto functional (BAF).
import numpy as np
import jax.numpy as jnp
import jax
import optax
from jax._src.random import PRNGKey
from jax.scipy.special import logsumexp
import common.jax_utils
from common.custom_train_state import TrainState
from functools import partial
from bagd.main import wgrad_baf
import bagd.main
from ba.main import ba_loss_and_update


class Experiment(bagd.main.Experiment):

  def train_step(self, base_rng, state: TrainState, batch, eval=False):
    # rng = jax.random.fold_in(base_rng, jax.lax.axis_index('batch'))
    # rng = jax.random.fold_in(base_rng, state.step)
    del base_rng
    rd_lambda = self.rd_lambda
    distort_type = self.distort_type

    # First run Wasserstein gradient step to get updated nu_x.
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

    # Then perform Blahut-Arimoto step to get udpated nu_w.
    nu_x = new_state.params['nu_x']  # Retrieve updated \nu atom locations.
    new_nu_w, loss = ba_loss_and_update(mu_x, mu_w, nu_x, nu_w, self.distort_type, self.rd_lambda)
    new_params = dict(nu_x=nu_x, nu_w=new_nu_w)
    new_state = state.replace(step=state.step + 1, params=new_params)

    # Note the loss we return is the BAF prior to any updates to \nu.
    return new_state, metrics
