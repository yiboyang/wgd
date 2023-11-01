# A self-contained, minimal implementation of Wasserstein GD on the rate functional L_{BA}.
import jax.numpy as jnp
import jax
from jax.scipy.special import logsumexp

# Define the distortion function \rho.
squared_diff = lambda x, y: jnp.sum((x - y) ** 2)
pairwise_distortion_fn = jax.vmap(jax.vmap(squared_diff, (None, 0)), (0, None))


def wgrad(mu_x, mu_w, nu_x, nu_w, rd_lambda):
  """
  Compute the Wasserstein gradient of the rate functional, which we will use
  to move the \nu particles.
  :param mu_x: locations of \mu atoms.
  :param mu_w: weights  of \mu atoms.
  :param nu_x: locations of \nu atoms.
  :param nu_w: weights  of \nu atoms.
  :param rd_lambda: R-D tradeoff hyperparameter.
  :return:
  """

  def compute_psi_sum(nu_x):
    """
    Here we compute a surrogate loss based on the first variation \psi, which
    allows jax autodiff to compute the desired Wasserstein gradient.
    :param nu_x:
    :return: psi_sum = \sum_i \psi(nu_x[i])
    """
    C = pairwise_distortion_fn(mu_x, nu_x)
    scaled_C = rd_lambda * C  # [m, n]
    log_nu_w = jnp.log(nu_w)  # [1, n]

    # Solve BA inner problem with a fixed nu.
    phi = - logsumexp(-scaled_C + log_nu_w, axis=1, keepdims=True)  # [m, 1]
    loss = jnp.sum(mu_w * phi)  # Evaluate the rate functional. Eq (6) in paper.

    # Let's also report rate and distortion estimates (discussed in Sec. 4.4 of the paper).
    # Find \pi^* via \phi
    pi = jnp.exp(phi - scaled_C) * jnp.outer(mu_w, nu_w)  # [m, n]
    distortion = jnp.sum(pi * C)
    rate = loss - rd_lambda * distortion

    # Now evaluate \psi on the atoms of \nu.
    phi = jax.lax.stop_gradient(phi)
    psi = - jnp.sum(jnp.exp(jax.lax.stop_gradient(phi) - scaled_C) * mu_w, axis=0)
    psi_sum = jnp.sum(psi)  # For computing gradient w.r.t. each nu_x atom.
    return psi_sum, (loss, rate, distortion)

  # Evaluate the Wasserstein gradient, i.e., \nabla \psi, on nu_x.
  psi_prime, loss = jax.grad(compute_psi_sum, has_aux=True)(nu_x)
  return psi_prime, loss


def wgd(X, n, rd_lambda, num_steps, lr, rng):
  """
  A basic demo of Wasserstein gradient descent on a discrete distribution.
  :param X: a 2D array [N, d] of data points defining the source \mu.
  :param n: the number of particles to use for \nu.
  :param rd_lambda: R-D tradeoff hyperparameter.
  :param num_steps: total number of gradient updates.
  :param lr:  step size.
  :param rng: jax random key.
  :return: (nu_x, nu_w), the locations and weights of the final \nu.
  """
  # Set up the source measure \mu.
  m = jnp.size(X, 0)
  mu_x = X
  mu_w = 1 / m * jnp.ones((m, 1))
  # Initialize \nu atoms using random training samples.
  rand_idx = jax.random.permutation(rng, m)[:n]
  nu_x = X[rand_idx]  # Locations of \nu atoms.
  nu_w = 1 / n * jnp.ones((1, n))  # Uniform weights.
  for step in range(num_steps):
    psi_prime, (loss, rate, distortion) = wgrad(mu_x, mu_w, nu_x, nu_w, rd_lambda)
    nu_x -= lr * psi_prime
    print(f'step={step}, loss={loss:.4g}, rate={rate:.4g}, distortion={distortion:.4g}')

  return nu_x, nu_w


if __name__ == '__main__':
  # Run a toy example on 2D Gaussian samples.
  rng = jax.random.PRNGKey(0)
  X = jax.random.normal(rng, [10, 2])
  nu_x, nu_w = wgd(X, n=4, rd_lambda=2., num_steps=100, lr=0.1, rng=rng)
