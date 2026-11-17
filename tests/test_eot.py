"""Numerical checks for fixed-marginal EOT and its Wasserstein gradient."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from common.jax_utils import get_pairwise_distort_fn
from eotgd.main import estimate_eot, solve_sinkhorn, validate_inputs, wgrad_eot
from eotgd.example import run_demo

jax.config.update('jax_enable_x64', True)


@pytest.fixture
def problem():
  x = jnp.array([[-1.2, 0.4], [0.3, -0.7], [1.4, 0.8], [2., -0.3]])
  y = jnp.array([[-0.6, 0.2], [0.9, -0.4], [1.7, 0.6]])
  a = jnp.array([[0.1], [0.2], [0.3], [0.4]])
  b = jnp.array([[0.2, 0.5, 0.3]])
  return x, a, y, b


def test_marginals_objective_and_independent_two_by_two_solution():
  a, b = jnp.array([0.3, 0.7]), jnp.array([0.6, 0.4])
  C = jnp.array([[0.2, 2.1], [1.3, 0.4]])
  phi, psi, loss, metrics = solve_sinkhorn(a, b, C, tol=1e-12)
  pi = np.exp(np.log(a[:, None]) + np.log(b[None, :]) + phi + psi - C)
  np.testing.assert_allclose(pi.sum(1), a, atol=1e-12, rtol=0)
  np.testing.assert_allclose(pi.sum(0), b, atol=1e-12, rtol=0)
  assert metrics['sinkhorn_converged']
  # A 2x2 feasible coupling has only one free variable. This scalar optimization
  # is independent of Sinkhorn, checking both the objective sign and KL convention.
  def objective(t):
    p = np.array([[t, 0.3-t], [0.6-t, 0.1+t]])
    return np.sum(p * C) + np.sum(p * np.log(p / np.outer(a, b)))
  independent = minimize_scalar(objective, bounds=(1e-12, 0.3-1e-12),
                                 method='bounded', options={'xatol': 1e-14})
  np.testing.assert_allclose(loss, independent.fun, atol=1e-11, rtol=0)
  np.testing.assert_allclose(pi[0, 0], independent.x, atol=1e-8, rtol=0)


@pytest.mark.parametrize('distort_type,factor', [('half_sse', 1.), ('sse', 2.), ('mse', 1.)])
def test_unequal_weight_gradients_and_descent(problem, distort_type, factor):
  x, a, y, b = problem
  rd_lambda = 1.7
  gradient, metrics = wgrad_eot(x, a, y, b, distort_type, rd_lambda, tol=1e-12)
  evaluation, pi = estimate_eot(x, a, y, b, distort_type, rd_lambda, tol=1e-12)
  assert metrics['sinkhorn_converged'] and evaluation['sinkhorn_converged']
  np.testing.assert_allclose(metrics['loss'], evaluation['primal_loss'], atol=1e-11)
  np.testing.assert_allclose(evaluation['eot_loss'], metrics['loss'] / rd_lambda)
  np.testing.assert_allclose(pi.sum(1), a[:, 0], atol=1e-12)
  np.testing.assert_allclose(pi.sum(0), b[0], atol=1e-12)
  # For quadratic costs, the conditional barycenter gives an independent formula.
  coupling_gradient = rd_lambda * factor * (y - pi.T @ x / b.T)
  np.testing.assert_allclose(gradient, coupling_gradient, atol=1e-10, rtol=1e-10)
  eps = 1e-5
  finite_diff = np.zeros_like(y)
  for index in np.ndindex(y.shape):
    delta = np.zeros_like(y)
    delta[index] = eps
    plus, _ = estimate_eot(x, a, y + delta, b, distort_type, rd_lambda, tol=1e-12)
    minus, _ = estimate_eot(x, a, y - delta, b, distort_type, rd_lambda, tol=1e-12)
    assert plus['sinkhorn_converged'] and minus['sinkhorn_converged']
    finite_diff[index] = (plus['loss'] - minus['loss']) / (2 * eps)
  np.testing.assert_allclose(finite_diff, b.T * gradient, atol=2e-8, rtol=2e-7)
  after, _ = estimate_eot(x, a, y - 0.02 * gradient, b, distort_type, rd_lambda, tol=1e-12)
  assert after['sinkhorn_converged'] and after['loss'] < metrics['loss']


def test_uniform_measures_on_collinear_points():
  x, y = jnp.arange(10.).reshape(5, 2), jnp.arange(6.).reshape(3, 2)
  a, b = jnp.ones((5, 1)) / 5, jnp.ones((1, 3)) / 3
  gradient, metrics = wgrad_eot(x, a, y, b, 'half_sse', 1., tol=1e-11)
  assert metrics['sinkhorn_converged']
  np.testing.assert_allclose(metrics['loss'], 6.9770048727, atol=1e-9)
  np.testing.assert_allclose(gradient.mean(0), [-2., -2.], atol=1e-9)


def test_iteration_limit_and_warm_start(problem):
  x, a, y, b = problem
  C = 15 * get_pairwise_distort_fn('half_sse')(x, y)
  _, _, loss, metrics = solve_sinkhorn(a, b, C, tol=1e-12, max_sink_its=1)
  assert np.isfinite(loss) and metrics['valid']
  assert metrics['sinkhorn_its'] == 1 and not metrics['sinkhorn_converged']
  phi, psi, loss, metrics = solve_sinkhorn(a, b, C, tol=1e-12, max_sink_its=2000)
  assert metrics['sinkhorn_converged']
  _, _, warm_loss, warm = solve_sinkhorn(a, b, C, phi, psi, tol=1e-12, max_sink_its=2000)
  assert warm['sinkhorn_converged'] and warm['sinkhorn_its'] == 1
  np.testing.assert_allclose(warm_loss, loss, atol=1e-11)


@pytest.mark.parametrize('bad', [jnp.array([0., 1.]), jnp.array([-0.1, 1.1]),
                                jnp.array([0.2, 0.2]), jnp.array([jnp.nan, 0.5]),
                                jnp.array([jnp.inf, 0.5])])
def test_invalid_weights_are_reported_under_jit(bad):
  x = jnp.array([[0.], [1.]])
  good = jnp.array([0.4, 0.6])
  with pytest.raises(ValueError):
    validate_inputs(x, bad, x, good, 1.)
  _, _, loss, metrics = jax.jit(solve_sinkhorn)(bad, good, jnp.ones((2, 2)))
  assert not metrics['valid'] and not metrics['sinkhorn_converged']
  assert metrics['sinkhorn_its'] == 0 and np.isnan(loss)


@pytest.mark.parametrize('rd_lambda', [0., -1., jnp.inf, jnp.nan])
def test_invalid_lambda(problem, rd_lambda):
  x, a, y, b = problem
  with pytest.raises(ValueError):
    validate_inputs(x, a, y, b, rd_lambda)
  _, metrics = wgrad_eot(x, a, y, b, 'half_sse', rd_lambda)
  assert not metrics['valid'] and not metrics['sinkhorn_converged']
  assert np.isnan(metrics['loss'])


def test_shapes_nonfinite_values_and_options(problem):
  x, a, y, b = problem
  with pytest.raises(ValueError):
    wgrad_eot(x, a.T, y, b, 'half_sse', 1.)
  with pytest.raises(ValueError):
    wgrad_eot(x[:, :1], a, y, b, 'half_sse', 1.)
  with pytest.raises(ValueError):
    solve_sinkhorn(a, b, jnp.zeros((4, 3)), max_sink_its=0)
  with pytest.raises(ValueError):
    solve_sinkhorn(a, b, jnp.zeros((4, 3)), phi=jnp.zeros(4))
  with pytest.raises(ValueError):
    validate_inputs(x.at[0, 0].set(jnp.inf), a, y, b, 1.)
  for cost in (jnp.full((4, 3), jnp.nan), jnp.full((4, 3), jnp.inf)):
    _, _, loss, metrics = solve_sinkhorn(a, b, cost)
    assert np.isnan(loss) and not metrics['valid']
  _, _, loss, metrics = solve_sinkhorn(a, b, jnp.zeros((4, 3)), tol=-1.)
  assert np.isnan(loss) and not metrics['valid']


@pytest.mark.parametrize('dtype,tol', [(jnp.float32, 1e-6), (jnp.float64, 1e-12)])
def test_jit_multidimensional_particles_and_precision(problem, dtype, tol):
  x, a, y, b = [v.astype(dtype) for v in problem]
  x, y = x.reshape(4, 1, 2), y.reshape(3, 1, 2)
  lam = jnp.asarray(1.7, dtype)
  fn = lambda y: wgrad_eot(x, a.ravel(), y, b.ravel(), 'mse', lam, tol=tol)
  compiled, metrics = jax.jit(fn)(y)
  with jax.disable_jit():
    eager, eager_metrics = fn(y)
  assert compiled.dtype == dtype and compiled.shape == y.shape
  assert metrics['sinkhorn_converged'] and eager_metrics['sinkhorn_converged']
  np.testing.assert_allclose(compiled, eager, atol=10*tol, rtol=10*tol)


def test_single_particle_and_zero_cost():
  x = jnp.array([[0.], [2.]])
  y = jnp.array([[3.]])
  a, b = jnp.array([0.25, 0.75]), jnp.array([1.])
  gradient, metrics = wgrad_eot(x, a, y, b, 'half_sse', 2., tol=1e-12)
  np.testing.assert_allclose(gradient, [[3.]])
  np.testing.assert_allclose(metrics['loss'], 3.)
  _, _, loss, metrics = solve_sinkhorn(a, a, jnp.zeros((2, 2)), tol=1e-12)
  np.testing.assert_allclose(loss, 0., atol=1e-14)
  assert metrics['sinkhorn_converged'] and metrics['sinkhorn_its'] == 1


def test_log_domain_solver_with_underflowing_kernel():
  a, b = jnp.array([0.4, 0.6]), jnp.array([0.7, 0.3])
  C = jnp.array([[0., 1000.], [1000., 0.]])
  phi, psi, loss, metrics = solve_sinkhorn(a, b, C, tol=1e-10, max_sink_its=5000)
  assert metrics['sinkhorn_converged'] and np.isfinite(loss)
  pi = jnp.exp(jnp.log(a[:, None]) + jnp.log(b[None, :]) + phi + psi - C)
  np.testing.assert_allclose(pi.sum(1), a, atol=1e-10, rtol=0)
  np.testing.assert_allclose(pi.sum(0), b, atol=1e-10, rtol=0)
  np.testing.assert_allclose(pi, [[0.4, 0.], [0.3, 0.3]], atol=1e-10, rtol=0)


def test_compiled_end_to_end_demo_and_failure():
  summary, arrays = run_demo(steps=10)
  assert summary['final_loss'] < summary['initial_loss']
  assert summary['max_marginal_error'] <= summary['tol']
  assert np.all(np.diff(arrays['loss']) < 0)
  np.testing.assert_array_equal(arrays['nu_w'], np.ones((1, 16)) / 16)
  with pytest.raises(RuntimeError, match='rejected'):
    run_demo(steps=2, tol=1e-12, max_sink_its=1)
