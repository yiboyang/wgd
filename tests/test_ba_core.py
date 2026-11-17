"""Numerical comparison of the BA and EOT functionals."""
import jax
import jax.numpy as jnp
import numpy as np

from bagd.core import wgrad_baf, estimate_rd
from eotgd.main import wgrad_eot, estimate_eot


def test_ba_coordinate_gradient_includes_particle_weights():
  x = jnp.array([[-1., 0.2], [0.1, 0.5], [1.4, -0.3]])
  y = jnp.array([[-0.5, 0.3], [0.8, -0.1]])
  a, b = jnp.array([[0.2], [0.3], [0.5]]), jnp.array([[0.4, 0.6]])
  gradient, metrics = wgrad_baf(x, a, y, b, 'half_sse', 2.)
  coordinate_gradient = jax.grad(
      lambda points: estimate_rd(x, a, points, b, 'half_sse', 2.)[0]['loss'])(y)
  np.testing.assert_allclose(coordinate_gradient, b.T * gradient, atol=2e-6, rtol=2e-6)
  evaluation, pi = estimate_rd(x, a, y, b, 'half_sse', 2.)
  np.testing.assert_allclose(metrics['loss'], evaluation['loss'], atol=1e-6)
  np.testing.assert_allclose(pi.sum(1), a[:, 0], atol=1e-6)
  assert np.sum(np.abs(pi.sum(0) - b[0])) > 1e-3


def test_symmetric_ba_and_eot_agree():
  x = jnp.array([[-1.], [1.]])
  a, b = jnp.ones((2, 1)) / 2, jnp.ones((1, 2)) / 2
  for radius in (0., 0.15, 0.7, 0.957504, 1.3):
    y = jnp.array([[-radius], [radius]])
    gb, mb = wgrad_baf(x, a, y, b, 'half_sse', 2.)
    ge, me = wgrad_eot(x, a, y, b, 'half_sse', 2.)
    assert me['sinkhorn_converged']
    closed_form = 1. + radius**2 - np.logaddexp(2*radius, -2*radius) + np.log(2)
    np.testing.assert_allclose(mb['loss'], closed_form, atol=1e-6)
    np.testing.assert_allclose(me['loss'], closed_form, atol=1e-6)
    np.testing.assert_allclose(gb, ge, atol=2e-6)


def test_eot_exceeds_ba_when_output_marginal_differs():
  x, y = jnp.array([[-1.], [1.]]), jnp.array([[-0.4], [1.2]])
  a, b = jnp.array([[0.3], [0.7]]), jnp.array([[0.6, 0.4]])
  ba, pi = estimate_rd(x, a, y, b, 'half_sse', 2.)
  eot, _ = estimate_eot(x, a, y, b, 'half_sse', 2.)
  assert eot['sinkhorn_converged']
  assert np.sum(np.abs(pi.sum(0) - b[0])) > 0.1
  assert eot['loss'] > ba['loss'] + 1e-3
